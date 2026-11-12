# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from typing_extensions import Self
from ultralytics.models import YOLOWorld as ultralytics_YOLO_WORLD
from ultralytics.nn.modules.block import MaxSigmoidAttnBlock
from ultralytics.nn.modules.head import BNContrastiveHead
from ultralytics.nn.tasks import WorldModel
from ultralytics.nn.text_model import build_text_model as build_clip_text_model

from qai_hub_models import Precision, SampleInputsType
from qai_hub_models.datasets.coco import CocoDataset
from qai_hub_models.models.openai_clip.model import patched_in_projection_packed
from qai_hub_models.models.templates.common import replace_module_recursively
from qai_hub_models.models.templates.corevision.labels import get_class_names
from qai_hub_models.models.templates.detection.detection_evaluator import (
    DetectionEvaluator,
)
from qai_hub_models.models.templates.ultralytics.detect_patches import (
    patch_ultralytics_detection_head,
)
from qai_hub_models.models.templates.ultralytics.patches import (
    BNContrastiveHeadInf,
    MaxSigmoidAttnBlockInf,
)
from qai_hub_models.models.templates.yolo.model import Yolo, yolo_detect_postprocess
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import (
    BaseModel,
)
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

SUPPORTED_WEIGHTS = [
    "yolov8m-worldv2.pt",
    "yolov8l-worldv2.pt",
    "yolov8s-worldv2.pt",
]
DEFAULT_WEIGHTS = "yolov8s-worldv2.pt"

CLIP_CONTEXT_LENGTH = 77


class YoloWorldTextEncoder(BaseModel):
    """Exportable CLIP text encoder for YoloWorld.

    Takes pre-tokenized CLIP token IDs and returns text embeddings. Tokenization
    (string → token IDs) must be performed on the host before calling this component.
    """

    def __init__(self, clip_model: torch.nn.Module) -> None:
        super().__init__()
        self.clip_model = clip_model

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS) -> Self:
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )
        world_model = ultralytics_YOLO_WORLD(ckpt_name)
        wm = cast(WorldModel, world_model.model)
        device = next(wm.parameters()).device
        clip_model = cast(
            torch.nn.Module,
            build_clip_text_model("clip:ViT-B/32", device=device).model,
        )
        return cls(clip_model)

    @staticmethod
    def get_input_spec(
        num_classes: int = 80,
        context_length: int = CLIP_CONTEXT_LENGTH,
    ) -> InputSpec:
        return {
            "tokens": TensorSpec(
                shape=(num_classes, context_length),
                dtype="int32",
                io_type=IoType.TENSOR,
            )
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if input_spec is not None:
            num_classes, context_length = input_spec["tokens"][0]
        else:
            num_classes = 80
            context_length = CLIP_CONTEXT_LENGTH
        coco_tokens = YoloWorldTextEncoder.tokenize_classes(get_class_names("coco"))
        # Tile/slice to match num_classes.
        repeats = (num_classes + coco_tokens.shape[0] - 1) // coco_tokens.shape[0]
        tokens = coco_tokens.repeat(repeats, 1)[:num_classes, :context_length]
        return {"tokens": [tokens.numpy()]}

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode pre-tokenized CLIP token IDs into text embeddings.

        Parameters
        ----------
        tokens
            Pre-tokenized CLIP token IDs.
            Shape: [num_classes, context_length] dtype int32.

        Returns
        -------
        txt_feats : torch.Tensor
            Normalized text embeddings.
            Shape: [1, num_classes, 512] dtype float32.
        """
        with patched_in_projection_packed():
            txt_feats = self.clip_model.encode_text(tokens).to(torch.float32)  # type: ignore[operator]
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats.unsqueeze(0)

    def encode_classes(self, class_names: list[str]) -> torch.Tensor:
        """
        Encode a list of class name strings into text embeddings.

        Convenience wrapper that tokenizes ``class_names`` and calls ``forward``.

        Parameters
        ----------
        class_names
            Human-readable class labels, e.g. ``["bed", "vase", "clock"]``.

        Returns
        -------
        txt_feats : torch.Tensor
            Normalized text embeddings.
            Shape: [1, num_classes, 512] dtype float32.
        """
        tokens = YoloWorldTextEncoder.tokenize_classes(class_names)
        with torch.no_grad():
            return self(tokens)

    @staticmethod
    def tokenize_classes(class_names: list[str]) -> torch.Tensor:
        """
        Tokenize class name strings to CLIP token IDs.

        Parameters
        ----------
        class_names
            Human-readable class labels.

        Returns
        -------
        tokens : torch.Tensor
            CLIP token IDs. Shape: [num_classes, context_length] dtype int32.
        """
        clip_wrapper = build_clip_text_model(
            "clip:ViT-B/32", device=torch.device("cpu")
        )
        return clip_wrapper.tokenize(class_names).to(torch.int32)

    def get_output_names(self) -> list[str]:
        return ["txt_feats"]

    def get_output_spec(self) -> OutputSpec:
        return {"txt_feats": TensorSpec()}


class YoloWorldDetector(Yolo):
    """Exportable YoloWorld image detector that takes pre-computed text embeddings."""

    def __init__(
        self,
        model: WorldModel,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> None:
        super().__init__(model=model)
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output

        patch_ultralytics_detection_head(model)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> Self:
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )

        world_model = ultralytics_YOLO_WORLD(ckpt_name)
        model = cast(WorldModel, world_model.model)

        replace_module_recursively(model, MaxSigmoidAttnBlock, MaxSigmoidAttnBlockInf)
        replace_module_recursively(model, BNContrastiveHead, BNContrastiveHeadInf)

        return cls(
            model,
            include_postprocessing,
            split_output,
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 640,
        width: int = 640,
        num_classes: int = 80,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
            "txt_feats": TensorSpec(
                shape=(batch_size, num_classes, 512),
                dtype="float32",
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        sample_inputs = super()._sample_inputs_impl(input_spec)
        if input_spec is not None:
            txt_feats_shape = input_spec["txt_feats"][0]
        else:
            txt_feats_shape = (1, 80, 512)
        sample_inputs["txt_feats"] = [np.zeros(txt_feats_shape, dtype=np.float32)]
        return sample_inputs

    def forward(
        self, image: torch.Tensor, txt_feats: torch.Tensor
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor]
        | torch.Tensor
    ):
        """
        Run YoloWorld on `image` and `txt_feats`, and produce a predicted set of bounding boxes from expected class prompts.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB
        txt_feats
            Text embeddings for the prompt.
            Shape: [1, num_classes, 512]

        Returns
        -------
        result : tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor
            If self.include_postprocessing is True, returns:
            boxes
                Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
            scores
                Class scores multiplied by confidence. Shape is [batch, num_preds].
            classes
                Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.

            If self.include_postprocessing is False and self.split_output is True, returns:
            boxes
                Bounding box predictions in xywh format. Shape [batch, 4, num_preds].
            scores
                Full score distribution over all classes for each box. Shape [batch, num_classes, num_preds].

            If self.include_postprocessing is False and self.split_output is False, returns:
            detector_output
                Boxes and scores concatenated into a single tensor. Shape [batch, 4 + num_classes, num_preds].
        """
        boxes, scores = self.model(image, txt_feats=txt_feats)

        if not self.include_postprocessing:
            if self.split_output:
                return boxes, scores
            return torch.cat([boxes, scores], dim=1)

        boxes, scores, classes = yolo_detect_postprocess(boxes, scores)

        return boxes, scores, classes

    @staticmethod
    def get_hub_litemp_percentage(_: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10

    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(
            image_height, image_width, score_threshold=0.001, nms_iou_threshold=0.7
        )


class YoloWorld(WorkbenchModelCollection):
    """YoloWorld collection model: image detector + CLIP text encoder.

    Component order puts the detector first so that component[0].get_input_spec()
    returns an image spec, which is what the evaluate pipeline and the export
    pipeline expect (they use [0] to configure the dataset/inference).
        [0] detector      — input_spec: {"image": ..., "txt_feats": ...}
        [1] text_encoder  — input_spec: {"tokens": ...}
    """

    def __init__(
        self,
        detector: YoloWorldDetector,
        text_encoder: YoloWorldTextEncoder,
    ) -> None:
        super().__init__({"detector": detector, "text_encoder": text_encoder})
        self.detector = detector
        self.text_encoder = text_encoder

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> Self:
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )

        world_model = ultralytics_YOLO_WORLD(ckpt_name)
        wm = cast(WorldModel, world_model.model)

        replace_module_recursively(wm, MaxSigmoidAttnBlock, MaxSigmoidAttnBlockInf)
        replace_module_recursively(wm, BNContrastiveHead, BNContrastiveHeadInf)

        # Build the detector component
        detector = YoloWorldDetector(wm, include_postprocessing, split_output)

        # Extract the CLIP model that WorldModel already loaded during __init__
        # (ultralytics calls set_classes with COCO defaults, which populates wm.clip_model).
        clip_wrapper = getattr(wm, "clip_model", None)
        if clip_wrapper is None:
            clip_wrapper = build_clip_text_model(
                "clip:ViT-B/32", device=next(wm.parameters()).device
            )
        text_encoder = YoloWorldTextEncoder(cast(torch.nn.Module, clip_wrapper.model))

        return cls(detector, text_encoder)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CocoDataset]  # type: ignore[type-abstract]

    def get_mixed_precisions(self, precision: Precision) -> dict[str, Precision]:
        # The CLIP text encoder is sensitive to int8 quantization (embeddings
        # degrade significantly at w8a8/w8a16). Always compile it at float
        # precision (FP16 on-device) regardless of the requested precision.
        base = super().get_mixed_precisions(precision)
        base["text_encoder"] = Precision.float
        return base

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoDataset

    def get_evaluator(self) -> BaseEvaluator:
        return self.detector.get_evaluator()

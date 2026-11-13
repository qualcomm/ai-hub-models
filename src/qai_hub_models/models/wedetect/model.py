# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models import Precision, SampleInputsType, TargetRuntime
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.detection.detection_evaluator import (
    DetectionEvaluator,
)
from qai_hub_models.models.templates.wedetect.constants import (
    DEFAULT_NUM_CLASSES,
    TEXT_EMBEDDING_DIM,
)
from qai_hub_models.models.templates.wedetect.model import (
    BaseDetector,
    BaseTextEncoder,
    load_wedetect_model,
)
from qai_hub_models.models.wedetect.dataset import WeDetectMainCocoDataset
from qai_hub_models.models.wedetect.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

WEDETECT_WEIGHT_TO_CONFIG: dict[str, str] = {
    "wedetect_tiny.pth": "config/wedetect_tiny.py",
    "wedetect_base.pth": "config/wedetect_base.py",
    "wedetect_large.pth": "config/wedetect_large.py",
}

DEFAULT_WEIGHTS = "wedetect_tiny.pth"

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

INPUT_IMAGE_SIZE = 640


def _load_base_model(ckpt_name: str) -> torch.nn.Module:
    """Validate ckpt_name, fetch weights, and return the loaded WeDetect base model."""
    if ckpt_name not in WEDETECT_WEIGHT_TO_CONFIG:
        raise ValueError(
            f"Unsupported checkpoint name provided {ckpt_name}.\n"
            f"Supported checkpoints are {list(WEDETECT_WEIGHT_TO_CONFIG.keys())}."
        )
    wedetect_path = EXTERNAL_REPO_PATHS["wedetect"]
    weight_path = CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, ckpt_name
    ).fetch()
    config_path = wedetect_path / WEDETECT_WEIGHT_TO_CONFIG[ckpt_name]
    return load_wedetect_model(
        config_path=str(config_path),
        weight_path=weight_path,
        repo_root=str(wedetect_path),
    )


def _sample_txt_feats_asset(ckpt_name: str) -> CachedWebModelAsset:
    """Cached S3 asset holding pre-computed COCO txt_feats for the given ckpt."""
    stem = ckpt_name.rsplit(".", 1)[0]
    return CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, f"{stem}_sample_txt_feats.pt"
    )


class WeDetectTextEncoder(BaseTextEncoder):
    """WeDetect text encoder loaded from the mmdet checkpoint."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS) -> Self:
        base_model = _load_base_model(ckpt_name)
        return cls(base_model)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if (
            target_runtime in (TargetRuntime.TFLITE, TargetRuntime.QNN_DLC)
            and "--truncate_64bit_tensors" not in other_compile_options
        ):
            other_compile_options += " --truncate_64bit_tensors"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )


class WeDetectDetector(BaseDetector):
    """
    Exportable WeDetect text-conditioned object detector.

    Used standalone or as the detector component of ``WeDetectModel``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ) -> None:
        super().__init__(model=model, include_postprocessing=include_postprocessing)
        self.ckpt_name = ckpt_name

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ) -> Self:
        base_model = _load_base_model(ckpt_name)
        return cls(
            base_model,
            ckpt_name=ckpt_name,
            include_postprocessing=include_postprocessing,
        )

    def component_precision(self) -> Precision:
        return Precision.w8a16

    def get_output_spec(self) -> dict[str, TensorSpec]:
        return {
            "boxes": TensorSpec(),
            "scores": TensorSpec(),
            "class_idx": TensorSpec(),
        }

    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(
            image_height, image_width, score_threshold=0.001, nms_iou_threshold=0.7
        )

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "room.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))

        sample_inputs = {"image": [app_to_net_image_inputs(image)[1].numpy()]}

        if input_spec is not None:
            txt_feats_shape = input_spec["txt_feats"][0]
        else:
            txt_feats_shape = (1, DEFAULT_NUM_CLASSES, TEXT_EMBEDDING_DIM)
        feats = torch.load(_sample_txt_feats_asset(self.ckpt_name).fetch())
        if feats.ndim == 3:
            feats = feats.squeeze(0)
        batch_size = txt_feats_shape[0]
        sample_inputs["txt_feats"] = [
            feats.unsqueeze(0).expand(batch_size, -1, -1).contiguous().numpy()
        ]
        return sample_inputs

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if (
            target_runtime in (TargetRuntime.TFLITE, TargetRuntime.QNN_DLC)
            and "--truncate_64bit_io" not in other_compile_options
        ):
            other_compile_options += " --truncate_64bit_io"

        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )


class WeDetectModel(WorkbenchModelCollection):
    """Collection model for WeDetect: detector + text encoder.

    Component order (detector first) ensures component[0].get_input_spec()
    returns an image spec, matching the auto-generated template expectation.
        [0] detector      — input_spec: {image, txt_feats}
        [1] text_encoder  — input_spec: {input_ids, attention_mask}
    """

    def __init__(
        self,
        detector: WeDetectDetector,
        text_encoder: BaseTextEncoder,
    ) -> None:
        super().__init__({"detector": detector, "text_encoder": text_encoder})
        self.detector = detector
        self.text_encoder = text_encoder

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ) -> Self:
        base_model = _load_base_model(ckpt_name)
        detector = WeDetectDetector(
            base_model,
            ckpt_name=ckpt_name,
            include_postprocessing=include_postprocessing,
        )
        text_encoder = WeDetectTextEncoder(base_model)
        return cls(detector, text_encoder)

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [WeDetectMainCocoDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return WeDetectMainCocoDataset

    def get_evaluator(self) -> BaseEvaluator:
        return self.detector.get_evaluator()

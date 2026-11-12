# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Callable, Generator
from typing import cast

import numpy as np
import torch
from PIL import Image
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
    transform_boxes_to_original_size,
)
from qai_hub_models.models.templates.corevision.draw import draw_box_from_xyxy
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.models.templates.corevision.labels import get_class_names
from qai_hub_models.models.templates.wedetect.constants import (
    DEFAULT_MAX_SEQ_LEN,
)
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    AsyncOnDeviceResult,
)
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


def resolve_class_labels(class_labels: list[str] | str) -> list[str]:
    """Normalise *class_labels* to a plain ``list[str]`` of class names.

    Parameters
    ----------
    class_labels
        Accepted formats:

        * ``list[str]`` — returned as-is::

            ["cat", "dog", "person"]

        * Comma-separated string — split on ``,`` and stripped::

            "cat, dog, person"

        * Path to a ``.txt`` file — one class name per line::

            # classes.txt
            cat
            dog
            person

        * Path to a ``.json`` file — either a flat list of strings or a list
          of single-element lists (COCO-style)::

            ["cat", "dog", "person"]
            [["cat"], ["dog"], ["person"]]

    Returns
    -------
    list[str]
        Flat list of class name strings with whitespace stripped.
    """
    if isinstance(class_labels, list):
        return class_labels

    if class_labels.endswith(".txt"):
        with open(class_labels) as f:
            lines = f.readlines()
        return [t.rstrip("\r\n").strip() for t in lines if t.strip()]

    if class_labels.endswith(".json"):
        with open(class_labels) as f:
            data = json.load(f)
        return [x[0] if isinstance(x, list) else x for x in data]

    return [t.strip() for t in class_labels.split(",") if t.strip()]


def tokenize_class_names(
    tokenizer: PreTrainedTokenizerBase,
    class_names: list[str],
    num_classes: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize *class_names* to fixed shape ``(num_classes, max_seq_len)``.

    Shorter prompts are right-padded with empty strings whose attention mask is
    zero — the detector's compiled txt_feats slot expects ``num_classes`` rows,
    and rows corresponding to zero-mask inputs produce embeddings that do not
    contribute to any real class score.
    """
    padded = list(class_names) + [""] * max(0, num_classes - len(class_names))
    padded = padded[:num_classes]
    enc = tokenizer(
        text=padded,
        return_tensors="pt",
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
    )
    return (
        enc["input_ids"].to(torch.int32),
        enc["attention_mask"].to(torch.int32),
    )


class WeDetectApp(
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
):
    """
    End-to-end inference app for WeDetect open-vocabulary object detection.

    The two-stage pipeline:
      1. ``tokenize(class_names) → input_ids, attention_mask``
      2. ``text_encoder(input_ids, attention_mask) → txt_feats``
      3. ``detector(image, txt_feats) → (boxes, scores, class_idx)``

    Both ``text_encoder`` and ``detector`` share a ``(*tensors,) → tensor(s)``
    interface across torch and on-device paths — no torch-only fallbacks.
    """

    def __init__(
        self,
        detector: Callable[
            [torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        text_encoder: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor | AsyncOnDeviceResult
        ],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        nms_score_threshold: float = 0.05,
        nms_iou_threshold: float = 0.45,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        Parameters
        ----------
        detector
            Callable ``(image, txt_feats) → (boxes, scores, class_idx)``.
        text_encoder
            Callable ``(input_ids, attention_mask) → txt_feats[N, 768]``.
        tokenizer
            HuggingFace-style tokenizer used to convert class name strings
            into ``(input_ids, attention_mask)`` at fixed shape.
        max_seq_len
            Compile-time sequence length used when tokenizing class names.
        nms_score_threshold
            Score threshold for NMS.
        nms_iou_threshold
            IoU threshold for NMS.
        input_spec
            Input spec from the detector; H/W are extracted for resize-pad.
        """
        self.detector = detector
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        if input_spec is not None:
            _, _, self.input_height, self.input_width = input_spec["image"][0]
            # txt_feats shape is (batch, num_classes, embed_dim); record num_classes
            # so class-name tokenization can pad to the compiled row count.
            self.compiled_num_classes: int | None = input_spec["txt_feats"][0][1]
        else:
            self.input_height = 640
            self.input_width = 640
            self.compiled_num_classes = None
        self.default_class_labels: list[str] = get_class_names("coco")

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        invalid_dims = [s for s in pixel_values.shape[-2:] if s % 32 != 0]
        if invalid_dims:
            raise ValueError(
                f"Invalid image size: dimensions {invalid_dims} are not divisible by 32."
            )

    def _txt_feats_for_image(
        self, pixel_values: torch.Tensor, txt_feats: torch.Tensor
    ) -> torch.Tensor:
        """Return text embeddings broadcast to the current image batch size."""
        txt_feats = txt_feats.to(device=pixel_values.device)
        batch_size = pixel_values.shape[0]
        if txt_feats.shape[0] == batch_size:
            return txt_feats
        if txt_feats.shape[0] != 1:
            raise ValueError(
                f"Text embedding batch size {txt_feats.shape[0]} does not match "
                f"image batch size {batch_size}."
            )
        return txt_feats.expand(batch_size, -1, -1)

    def _encode_classes(self, class_names: list[str]) -> torch.Tensor:
        """Tokenize *class_names* and run the text encoder.

        Handles both torch (returns Tensor) and on-device (returns
        AsyncOnDeviceResult) text_encoder invocations. Returns a batched
        embedding of shape ``(1, N, 768)`` where ``N`` matches the compiled
        row count (padded from class_names via empty-string tokens when
        needed).
        """
        n_rows = self.compiled_num_classes or len(class_names)
        input_ids, attention_mask = tokenize_class_names(
            self.tokenizer, class_names, n_rows, self.max_seq_len
        )
        out = self.text_encoder(input_ids, attention_mask)
        if isinstance(out, AsyncOnDeviceResult):
            resolved = out.wait()
            out = resolved[0] if isinstance(resolved, tuple) else resolved
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim == 2:
            out = out.unsqueeze(0)
        return out

    def predict_boxes_from_image(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
        raw_output: bool = False,
        class_labels: list[str] | str | None = None,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | list[np.ndarray]
    ):
        """
        Detect objects in *pixel_values_or_image*.

        Parameters
        ----------
        pixel_values_or_image
            PIL image, numpy ``(H W C uint8)`` / ``(N H W C uint8)``, torch
            ``(N C H W float32)`` in [0, 1], or list of PIL images.
        raw_output
            Return raw tensors instead of annotated images.
        class_labels
            Class names to detect.  Falls back to ``self.default_class_labels``.

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | list[np.ndarray]
            Raw tensors ``(boxes, scores, class_idx)`` when ``raw_output=True`` or the
            input is a torch tensor; otherwise a list of annotated ``uint8`` images.
        """
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )

        NCHW_fp32_resized, scale, pad = resize_pad(
            NCHW_fp32_torch_frames,
            (self.input_height, self.input_width),
        )

        if class_labels is not None:
            classes: list[str] = (
                [t.strip() for t in class_labels.split(",") if t.strip()]
                if isinstance(class_labels, str)
                else list(class_labels)
            )
        else:
            classes = self.default_class_labels
        if not classes:
            raise ValueError("class_labels must contain at least one class name.")

        txt_feats = self._encode_classes(classes)

        model_raw_output = self.detector(
            NCHW_fp32_resized,
            self._txt_feats_for_image(NCHW_fp32_resized, txt_feats),
        )
        if isinstance(model_raw_output, AsyncOnDeviceResult):
            model_output: tuple[torch.Tensor, ...] = model_raw_output.wait()
        else:
            model_output = model_raw_output
        pred_boxes, pred_scores, pred_class_idx = model_output

        pred_post_nms_boxes, pred_post_nms_scores, pred_post_nms_class_idx = (
            batched_nms(
                self.nms_iou_threshold,
                self.nms_score_threshold,
                pred_boxes,
                pred_scores,
                pred_class_idx,
            )
        )

        if raw_output or isinstance(pixel_values_or_image, torch.Tensor):
            return (pred_post_nms_boxes, pred_post_nms_scores, pred_post_nms_class_idx)

        orig_h = NCHW_fp32_torch_frames.shape[2]
        orig_w = NCHW_fp32_torch_frames.shape[3]

        for batch_idx in range(len(pred_post_nms_boxes)):
            boxes_b = pred_post_nms_boxes[batch_idx]
            scores_b = pred_post_nms_scores[batch_idx]
            class_idx_b = pred_post_nms_class_idx[batch_idx]

            if len(boxes_b) == 0:
                continue

            boxes_b = transform_boxes_to_original_size(
                boxes_b, pad, scale, orig_h, orig_w
            )

            for i, box in enumerate(boxes_b):
                class_idx = int(class_idx_b[i].item())
                score = float(scores_b[i].item())
                label = (
                    f"{classes[class_idx]} {score:.2f}"
                    if class_idx < len(classes)
                    else f"cls{class_idx} {score:.2f}"
                )
                draw_box_from_xyxy(
                    NHWC_int_numpy_frames[batch_idx],
                    box[0:2].int(),
                    box[2:4].int(),
                    color=(0, 255, 0),
                    size=2,
                    text=label,
                )

        return NHWC_int_numpy_frames

    @classmethod
    @abstractmethod
    def _load_tokenizer(cls) -> PreTrainedTokenizerBase:
        """Load the XLM-RoBERTa tokenizer for this model variant.

        Concrete subclasses override this to resolve the tokenizer bundle
        path from their external-repo layout.
        """

    @classmethod
    def from_components(
        cls,
        models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel],
    ) -> WeDetectApp:
        """Create app from collection components ``[detector, text_encoder]``.

        The tokenizer is loaded via ``cls._load_tokenizer()`` — no runtime
        attribute is smuggled onto ``AsyncOnDeviceModel``.
        """
        detector_component = models[0]
        text_encoder_component = models[1]

        detector_input_spec = (
            detector_component.get_input_spec()
            if hasattr(detector_component, "get_input_spec")
            else None
        )
        return cls(
            detector=detector_component,  # type: ignore[arg-type]
            text_encoder=text_encoder_component,
            tokenizer=cls._load_tokenizer(),
            max_seq_len=DEFAULT_MAX_SEQ_LEN,
            input_spec=detector_input_spec,
        )

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        model = collection_model.components[component_name]
        input_spec = (
            input_specs[component_name] if input_specs else model.get_input_spec()
        )
        batch_size = get_batch_size(input_spec) or 1

        text_encoder_component = collection_model.components["text_encoder"]
        coco_labels = get_class_names("coco")
        tokenizer = cls._load_tokenizer()

        if component_name == "text_encoder":
            # Text encoder: calibrate with tokenised COCO labels repeated N times.
            n_rows, seq_len = input_spec["input_ids"][0]
            input_ids, attention_mask = tokenize_class_names(
                tokenizer, coco_labels, n_rows, seq_len
            )
            num_samples = num_samples or 128
            ids_data: list[torch.Tensor | np.ndarray] = [input_ids] * num_samples
            mask_data: list[torch.Tensor | np.ndarray] = [attention_mask] * num_samples
            return make_hub_dataset_entries(
                (ids_data, mask_data),
                ["input_ids", "attention_mask"],
            )

        # Detector calibration: (image, txt_feats) pairs — precompute real embeddings.
        te_input_spec = text_encoder_component.get_input_spec()
        te_num_classes, te_seq_len = te_input_spec["input_ids"][0]
        input_ids, attention_mask = tokenize_class_names(
            tokenizer, coco_labels, te_num_classes, te_seq_len
        )
        txt_feats = text_encoder_component(input_ids, attention_mask).unsqueeze(0)

        detector_component = collection_model.components["detector"]
        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        detector_image_spec = (input_specs or {}).get(
            "detector", detector_component.get_input_spec()
        )
        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=detector_image_spec,
        )
        num_samples = num_samples or dataset.default_num_calibration_samples()
        num_samples = (num_samples // batch_size) * batch_size
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)

        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]
        for sample_input, _ in dataloader:
            if isinstance(sample_input, (tuple, list)):
                image = sample_input[0]
            else:
                image = sample_input
            batch_txt = txt_feats.expand(image.shape[0], -1, -1)
            inputs[0].append(image)
            inputs[1].append(batch_txt)

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        if isinstance(model_input, tuple):
            image_raw, ids_raw, mask_raw = model_input
            image_chunks: tuple[torch.Tensor, ...] = (
                image_raw.split(model_batch_size, dim=0)
                if isinstance(image_raw, torch.Tensor)
                else cast(tuple[torch.Tensor, ...], image_raw)
            )
            input_ids = (
                ids_raw[0] if isinstance(ids_raw, torch.Tensor) else ids_raw[0][0]
            )
            attention_mask = (
                mask_raw[0] if isinstance(mask_raw, torch.Tensor) else mask_raw[0][0]
            )
        else:
            image_chunks = cast(tuple[torch.Tensor, ...], next(model_input))
            ids_chunks = cast(tuple[torch.Tensor, ...], next(model_input))
            mask_chunks = cast(tuple[torch.Tensor, ...], next(model_input))
            input_ids = ids_chunks[0][0]
            attention_mask = mask_chunks[0][0]

        text_output = self.text_encoder(input_ids, attention_mask)
        yield (text_output,) if isinstance(text_output, torch.Tensor) else text_output

        if isinstance(text_output, AsyncOnDeviceResult):
            enc_result = text_output.wait()
            txt_feats = enc_result[0] if isinstance(enc_result, tuple) else enc_result
        else:
            txt_feats = text_output
        if txt_feats.ndim == 2:
            txt_feats = txt_feats.unsqueeze(0)

        txt_feats_chunks = tuple(
            txt_feats.expand(chunk.shape[0], -1, -1).contiguous()
            for chunk in image_chunks
        )
        det_output: AsyncOnDeviceResult | tuple[torch.Tensor, ...]
        if isinstance(self.detector, AsyncOnDeviceModel):
            det_output = self.detector(image_chunks, txt_feats_chunks)
        else:
            # Local torch path: eval helper always slices to model_batch_size=1, so
            # there is exactly one chunk.
            assert len(image_chunks) == 1
            det_output = self.detector(image_chunks[0], txt_feats_chunks[0])
        yield det_output
        return det_output

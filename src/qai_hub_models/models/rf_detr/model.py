# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import rfdetr as _rfdetr
import torch
import torch.nn.functional as F
from rfdetr.models.ops.modules import MSDeformAttn
from typing_extensions import Self

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_xyxy,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.detection.detection_evaluator import (
    DetectionEvaluator,
)
from qai_hub_models.models.templates.detr.model import DETR
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Supported detection variants (XL and 2XL require rfdetr[plus] and are excluded).
SUPPORTED_VARIANTS = ["nano", "small", "medium", "base", "large"]
DEFAULT_VARIANT = "small"

# Default input resolution per variant (sourced from rfdetr package configs).
VARIANT_RESOLUTION: dict[str, int] = {
    "nano": 384,
    "small": 512,
    "medium": 576,
    "base": 560,
    "large": 704,
}

# Maps variant name to the rfdetr PyPI package class.
_VARIANT_CLASS: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}

DEFAULT_RESOLUTION = VARIANT_RESOLUTION[DEFAULT_VARIANT]


def _patch_bicubic_antialias(model: torch.nn.Module) -> None:
    """Replace antialias=True bicubic interpolations in DINOv2 position encoding
    with antialias=False so the model traces to a standard ONNX Resize op.
    The aten::_upsample_bicubic2d_aa operator (antialias=True) has no ONNX
    symbolic and causes export failures.
    """
    _real_interpolate = F.interpolate

    def _no_antialias(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs["antialias"] = False
        return _real_interpolate(*args, **kwargs)

    for module in model.modules():
        if not hasattr(module, "interpolate_pos_encoding"):
            continue
        original: Callable[..., torch.Tensor] = module.interpolate_pos_encoding  # type: ignore[assignment, unused-ignore]

        def _patched(
            self_mod: torch.nn.Module,
            embeddings: torch.Tensor,
            height: int,
            width: int,
            _orig: Callable[..., torch.Tensor] = original,
            _interp: Callable[..., torch.Tensor] = _no_antialias,
        ) -> torch.Tensor:
            # Shadow F.interpolate only on the calling module's class
            # by calling original with a locally-swapped reference.

            saved = torch.nn.functional.interpolate
            torch.nn.functional.interpolate = _interp
            try:
                return _orig(embeddings, height, width)
            finally:
                torch.nn.functional.interpolate = saved

        module.interpolate_pos_encoding = types.MethodType(_patched, module)  # type: ignore[assignment]


def _patch_msdeformattn_export(model: torch.nn.Module) -> None:
    """Neutralize MSDeformAttn's data-dependent length assert so the model
    traces under torch.export.

    The assert compares ``(input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()``
    against the flattened input length. Reading those values off the tensor yields an
    unbacked symbolic int under torch.export, turning the comparison into a data-dependent
    guard that aborts tracing (GuardOnDataDependentSymNode: Eq(u0, 1)). Spatial shapes are
    static for a fixed input resolution, and the deformable-attention core already uses the
    Python-int (H, W) pairs threaded in as ``input_spatial_shapes_hw``, so skipping the
    assert changes nothing numerically.
    """

    def _noop_assert(*args: Any, **kwargs: Any) -> None:
        return None

    for module in model.modules():
        if not isinstance(module, MSDeformAttn):
            continue
        original: Callable[..., torch.Tensor] = module.forward

        def _patched(
            self_mod: torch.nn.Module,
            *args: Any,
            _orig: Callable[..., torch.Tensor] = original,
            **kwargs: Any,
        ) -> torch.Tensor:
            # Make the (validation-only) assert a no-op for the duration of this
            # forward so torch.export never records a guard on the tensor-derived length.
            saved = torch._assert
            torch._assert = _noop_assert
            try:
                return _orig(*args, **kwargs)
            finally:
                torch._assert = saved

        module.forward = types.MethodType(_patched, module)


class RF_DETR(DETR):
    """Exportable RF-DETR model, end-to-end.

    Supports the following variants: nano, small, medium, base, large.
    The XLarge and 2XLarge variants require rfdetr[plus] and are excluded.
    """

    def __init__(self, model: torch.nn.Module, variant: str = DEFAULT_VARIANT) -> None:
        super().__init__(model=model)
        self.variant = variant

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run RF-DETR on `image` and produce high quality detection results.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        boxes : torch.Tensor
            Shape (batch_size, num_queries, 4) — bounding box coordinates (x1, y1, x2, y2).
        scores : torch.Tensor
            Shape (batch_size, num_queries) — confidence scores per detection.
        labels : torch.Tensor
            Shape (batch_size, num_queries) — predicted class index per detection.
        """
        image_array = normalize_image_torchvision(image)
        # boxes: (center_x, center_y, w, h)
        predictions = self.model(image_array)
        # RF-DETR has swapped output order compared to standard DETR
        # logits are at index 1, boxes are at index 0
        logits, boxes = predictions[1], predictions[0]
        boxes, scores, labels = self.detr_postprocess(logits, boxes, image_array.shape)

        return boxes, scores, labels

    def detr_postprocess(
        self, logits: torch.Tensor, boxes: torch.Tensor, image_shape: tuple
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Postprocess the output of the RF-DETR model.

        RF-DETR uses sigmoid activations (focal-loss trained) rather than
        softmax with a "no-object" class.  The official PostProcess selects
        the top-K (num_select) predictions by sigmoid score across all
        class-query slots, then converts boxes from normalized cxcywh to
        pixel-space xyxy.

        Parameters
        ----------
        logits
            Shape (B, num_queries, num_classes) -- raw (pre-sigmoid) class logits.
        boxes
            Shape (B, num_queries, 4) -- normalized (cx, cy, w, h) in [0, 1].
        image_shape
            Shape of the normalized input image tensor: (B, C, H, W).

        Returns
        -------
        boxes : torch.Tensor
            Shape (B, num_select, 4) -- pixel-space (x1, y1, x2, y2).
        scores : torch.Tensor
            Shape (B, num_select) -- sigmoid confidence scores.
        labels : torch.Tensor
            Shape (B, num_select) -- predicted class indices (int32).
        """
        _, _, h, w = image_shape

        # RF-DETR is trained with sigmoid focal loss -- use sigmoid, not softmax.
        prob = logits.sigmoid()

        # Select top-K predictions across all (query, class) slots, matching
        # the official rfdetr PostProcess behaviour (num_select = num_queries).
        batch_size, num_queries, num_classes = prob.shape
        num_select = num_queries
        topk_values, topk_indexes = torch.topk(
            prob.view(batch_size, -1), num_select, dim=1
        )
        scores = topk_values
        topk_boxes_idx = topk_indexes // num_classes
        labels = topk_indexes % num_classes

        # Convert normalized cxcywh -> xyxy, then scale to pixel space.
        boxes_xyxy = box_xywh_to_xyxy(boxes)
        boxes_xyxy = torch.gather(
            boxes_xyxy, 1, topk_boxes_idx.unsqueeze(-1).expand(-1, -1, 4)
        )
        scale = torch.tensor(
            [w, h, w, h], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device
        )
        boxes_xyxy = boxes_xyxy * scale

        boxes_xyxy = boxes_xyxy.to(torch.float32)
        scores = scores.to(torch.float32)
        labels = labels.to(torch.int32)

        return boxes_xyxy, scores, labels

    def get_evaluator(self) -> BaseEvaluator:
        """
        Returns a DetectionEvaluator configured for RF-DETR evaluation.

        RF-DETR outputs top-K predictions per image ranked by sigmoid score.
        For correct mAP computation all predictions must reach the evaluator
        so the full precision-recall curve is captured -- a hard score
        threshold would discard true positives and deflate mAP.  NMS is
        applied to suppress duplicate boxes before scoring.
        """
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(
            image_height,
            image_width,
            nms_iou_threshold=0.7,
            score_threshold=None,
        )

    @classmethod
    def from_pretrained(
        cls,
        variant: str = DEFAULT_VARIANT,
        checkpoint_path: str | None = None,
    ) -> Self:
        """
        Load a pretrained RF-DETR model.

        Parameters
        ----------
        variant
            Model variant to load. One of: nano, small, medium, base, large.
            XLarge and 2XLarge variants require rfdetr[plus] and are excluded.
        checkpoint_path
            Optional path to a custom pre-trained checkpoint (.pth file).
            When provided, the checkpoint weights are loaded instead of the
            default Roboflow-published weights for the selected variant.
            The checkpoint must be compatible with the chosen variant's
            architecture (e.g. a fine-tuned RF-DETR base checkpoint for
            ``variant="base"``).

        Returns
        -------
        Self
            An instance of ``RF_DETR`` loaded with the specified variant's
            published weights, or with ``checkpoint_path`` weights when provided.
        """
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant '{variant}'. "
                f"Supported variants: {SUPPORTED_VARIANTS}"
            )
        resolution = VARIANT_RESOLUTION[variant]

        rfdetr_cls = getattr(_rfdetr, _VARIANT_CLASS[variant])
        # Pass pretrain_weights only when a custom checkpoint is supplied;
        # otherwise let rfdetr download its default published weights.
        kwargs: dict = dict(resolution=resolution, device="cpu")
        if checkpoint_path is not None:
            p = Path(checkpoint_path).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")
            if p.suffix not in {".pth", ".pt", ".ckpt"}:
                raise ValueError(
                    f"Unexpected checkpoint extension {p.suffix!r}. "
                    "Only .pth / .pt / .ckpt files are accepted."
                )
            kwargs["pretrain_weights"] = str(p)
        # torch 2.11.0 MKL-DNN has a bug on CPUs without AVX-512 (e.g. AMD EPYC)
        # that causes SIGFPE in Conv2d with kernel >= 4x4. Disabling MKL-DNN
        # falls back to native PyTorch kernels with identical numerics.
        torch.backends.mkldnn.set_flags(_enabled=False)
        torch_model = rfdetr_cls(**kwargs)
        torch_model.optimize_for_inference(compile=False)
        inference_model = torch_model.model.inference_model
        # Disable antialias in DINOv2 pos-encoding interpolation so the model
        # exports cleanly to ONNX (aten::_upsample_bicubic2d_aa is unsupported).
        _patch_bicubic_antialias(inference_model)
        _patch_msdeformattn_export(inference_model)
        return cls(inference_model, variant=variant)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int | None = None,
        width: int | None = None,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type)). This can be
        used to submit profiling job on Qualcomm® AI Hub Workbench.

        Parameters
        ----------
        batch_size
            Number of images per batch.
        height
            Input image height in pixels. Defaults to the canonical resolution
            for the selected variant (see ``VARIANT_RESOLUTION``).
        width
            Input image width in pixels. Defaults to the canonical resolution
            for the selected variant (see ``VARIANT_RESOLUTION``).

        Returns
        -------
        InputSpec
            Mapping of input name to tensor shape and dtype.
        """
        res = VARIANT_RESOLUTION.get(self.variant, DEFAULT_RESOLUTION)
        h = height if height is not None else res
        w = width if width is not None else res
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, h, w),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from qai_hub.client import Device
from torch import nn
from torch.nn import functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerMaskPredictor,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
)
from transformers.models.swin.modeling_swin import SwinLayer, SwinSelfAttention
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    TargetRuntime,
)
from qai_hub_models.datasets.coco import CocoPanopticSegmentationDataset
from qai_hub_models.models.mask2former.evaluator import (
    PanopticSegmentationEvaluator,
)
from qai_hub_models.models.mask2former.model_patches import (
    PatchedMask2FormerMaskPredictor,
    PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
    PatchedSwinLayer,
    PatchedSwinSelfAttention,
)
from qai_hub_models.models.templates.common import replace_module_recursively
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "facebook/mask2former-swin-tiny-coco-panoptic"
NUM_CLASSES = 134


class Mask2Former(BaseModel):
    """Mask2Former segmentation"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    @classmethod
    def from_pretrained(cls, ckpt: str = DEFAULT_WEIGHTS) -> Self:
        net = Mask2FormerForUniversalSegmentation.from_pretrained(ckpt)
        replace_module_recursively(
            net, Mask2FormerMaskPredictor, PatchedMask2FormerMaskPredictor
        )
        replace_module_recursively(
            net,
            Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
            PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
        )
        replace_module_recursively(net, SwinSelfAttention, PatchedSwinSelfAttention)
        replace_module_recursively(net, SwinLayer, PatchedSwinLayer)
        return cls(net)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict panoptic segmentation an input `image`.

        Parameters
        ----------
        image
            A [1, 3, height, width] image with value range of [0, 1], RGB channel layout.

        Returns
        -------
        scores : torch.Tensor
            Score (probability) of the predicted label for each batch. Shape [num_batches, num_classes]
        labels : torch.Tensor
            Index of the predicted label for each batch.  Shape [num_batches, num_classes]
        masks : torch.Tensor
            Logits for each pixel. Logits predict which pixel is part of the predicted mask.
            [1, num_classes, modified_height, modified_width]
            where the modified height and width will be some factor smaller than the input image.
        """
        outputs = self.model(normalize_image_torchvision(image), return_dict=False)
        class_logits, mask_logits = outputs[0], outputs[1]
        (class_pred_scores, class_pred_labels) = F.softmax(class_logits, dim=-1).max(-1)
        return class_pred_scores, class_pred_labels, mask_logits

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 384,
        width: int = 384,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "scores": TensorSpec(),
            "labels": TensorSpec(),
            "masks": TensorSpec(apply_runtime_channel_reordering=True),
        }

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"

        return compile_options

    def get_evaluator(self) -> BaseEvaluator:
        return PanopticSegmentationEvaluator(NUM_CLASSES)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CocoPanopticSegmentationDataset]

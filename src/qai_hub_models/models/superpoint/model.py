# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing_extensions import Self

from qai_hub_models.models.superpoint.dataset import (
    HPatchesDataset,
    get_sample_image_pair_paths,
)
from qai_hub_models.models.superpoint.evaluator import HPatchesEvaluator
from qai_hub_models.models.superpoint.external_repos.superpoint.superpoint_pytorch import (
    SuperPoint as SourceSuperPoint,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    numpy_image_to_torch,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_torch,
)
from qai_hub_models.utils.base_dataset import BaseDataset
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
MODEL_ASSET_VERSION = 1

# Pinned to rpautrat/SuperPoint@1411bbd68c50163555d39c1b26e9e046ebd48f27
DEFAULT_WEIGHTS = CachedWebModelAsset(
    "https://raw.githubusercontent.com/rpautrat/SuperPoint/"
    "1411bbd68c50163555d39c1b26e9e046ebd48f27/weights/superpoint_v6_from_tf.pth",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "superpoint_v6_from_tf.pth",
)

# Matches the official HPatches benchmark config (configs/superpoint_hpatches.yaml).
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640


class SuperPoint(BaseModel):
    """
    SuperPoint keypoint detector + descriptor, exportable for on-device inference.

    forward() only computes the static-shape, per-pixel dense outputs (raw
    per-pixel detection scores and dense descriptors) for each of the two
    input images -- the same architecture as the upstream
    ``superpoint_pytorch.SuperPoint`` module, minus NMS, border suppression,
    and its final variable-length keypoint extraction step, none of which can
    be part of a fixed-shape traced/compiled graph (and, for quantized
    precisions, none of which should share a tensor with the raw scores --
    NMS's zeroing and the border sentinel value both distort the quantizer's
    calibrated range for what is otherwise a well-behaved softmax output).
    All of that is done on the host in ``app.py`` / ``evaluator.py`` via
    ``postprocess.py``.
    """

    @classmethod
    def from_pretrained(
        cls,
        weights: str | CachedWebModelAsset = DEFAULT_WEIGHTS,
        nms_radius: int = 4,
        remove_borders: int = 4,
    ) -> Self:
        checkpoint = load_torch(weights)
        net = SourceSuperPoint(nms_radius=nms_radius, remove_borders=remove_borders)
        net.load_state_dict(checkpoint)
        return cls(net)

    def _dense_forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net = cast(SourceSuperPoint, self.model)
        if image.shape[1] == 3:
            weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * weights).sum(1, keepdim=True)

        features = net.backbone(image)
        descriptors_dense = F.normalize(net.descriptor(features), p=2, dim=1)

        scores = net.detector(features)
        scores = F.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, net.stride, net.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * net.stride, w * net.stride
        )

        return scores, descriptors_dense

    def forward(
        self, image_a: torch.Tensor, image_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        image_a
            RGB tensor, shape ``[B, 3, H, W]``, values in ``[0, 1]``.
        image_b
            RGB tensor, shape ``[B, 3, H, W]``, values in ``[0, 1]``.

        Returns
        -------
        scores_a : torch.Tensor
            Raw (pre-NMS, pre-border-suppression) detection score map for
            ``image_a``, shape ``[B, H, W]``.
        descriptors_a : torch.Tensor
            Dense L2-normalized descriptor map for ``image_a``,
            shape ``[B, 256, H / 8, W / 8]``.
        scores_b : torch.Tensor
            Same as ``scores_a``, for ``image_b``.
        descriptors_b : torch.Tensor
            Same as ``descriptors_a``, for ``image_b``.
        """
        scores_a, descriptors_a = self._dense_forward(image_a)
        scores_b, descriptors_b = self._dense_forward(image_b)
        return scores_a, descriptors_a, scores_b, descriptors_b

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> InputSpec:
        shape = (batch_size, 3, height, width)
        image_spec = TensorSpec(
            shape=shape,
            dtype="float32",
            io_type=IoType.IMAGE,
            value_range=(0.0, 1.0),
            image_metadata=ImageMetadata(color_format=ColorFormat.RGB),
        )
        return {"image_a": image_spec, "image_b": image_spec}

    def get_output_spec(self) -> OutputSpec:
        return {
            "scores_a": TensorSpec(io_type=IoType.TENSOR),
            "descriptors_a": TensorSpec(io_type=IoType.TENSOR),
            "scores_b": TensorSpec(io_type=IoType.TENSOR),
            "descriptors_b": TensorSpec(io_type=IoType.TENSOR),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        height, width = DEFAULT_HEIGHT, DEFAULT_WIDTH
        if input_spec is not None:
            _, _, height, width = input_spec["image_a"][0]
        path_a, path_b = get_sample_image_pair_paths()
        image_a = (
            load_image(path_a).convert("RGB").resize((width, height), Image.BILINEAR)
        )
        image_b = (
            load_image(path_b).convert("RGB").resize((width, height), Image.BILINEAR)
        )
        tensor_a = numpy_image_to_torch(np.array(image_a)).numpy()
        tensor_b = numpy_image_to_torch(np.array(image_b)).numpy()
        return dict(image_a=[tensor_a], image_b=[tensor_b])

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [HPatchesDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return HPatchesDataset

    def get_evaluator(self) -> HPatchesEvaluator:
        return HPatchesEvaluator()

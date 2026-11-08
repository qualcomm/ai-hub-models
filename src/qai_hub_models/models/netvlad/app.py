# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL.Image import Image

from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec


class NetVLADApp:
    """
    End-to-end app for NetVLAD place-recognition.

    Accepts image(s) as PIL Images, numpy arrays, or [0,1] float tensors,
    resizes to the model's canonical resolution, runs the model, and returns
    L2-normalized VLAD descriptors suitable for cosine-similarity retrieval.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        input_spec: InputSpec | None = None,
    ) -> None:
        self.model = model
        if input_spec is None:
            input_spec = model.get_input_spec()  # type: ignore[attr-defined]
        shape = input_spec["image"][0]
        self.input_height: int = shape[2]
        self.input_width: int = shape[3]

    def predict_features(
        self, *images: torch.Tensor | np.ndarray | Image
    ) -> np.ndarray:
        """
        Generate place descriptors for one or more images.

        Parameters
        ----------
        *images
            One or more images: PIL Image, uint8 numpy (H W C), or float32
            torch tensor (1 C H W, range [0, 1]).

        Returns
        -------
        np.ndarray
            Array of shape (N, descriptor_dim) with one L2-normalized
            VLAD descriptor per input image.
        """
        batch_tensors = []
        for image in images:
            _, nchw = app_to_net_image_inputs(image)
            resized = F.interpolate(
                nchw,
                size=(self.input_height, self.input_width),
                mode="bilinear",
                align_corners=False,
            )
            batch_tensors.append(resized)
        batch = torch.cat(batch_tensors, dim=0)
        descriptors = self.model(batch.contiguous())
        return np.asarray(descriptors.detach().cpu())

    def __call__(self, *images: torch.Tensor | np.ndarray | Image) -> np.ndarray:
        return self.predict_features(*images)

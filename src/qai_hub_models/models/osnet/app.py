# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.input_spec import InputSpec


class OSNetApp:
    """
    Lightweight application for OSNet person re-identification.

    This app processes input images and generates L2-normalized feature embeddings
    using an OSNet model, suitable for person re-identification tasks such as
    matching individuals across different camera views.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        input_spec: InputSpec,
    ) -> None:
        self.model = model
        _, _, self.input_height, self.input_width = input_spec["image"][0]

    def predict_features(
        self, *images: torch.Tensor | np.ndarray | Image
    ) -> np.ndarray:
        """
        Generate re-identification embeddings for one or more input images.

        Parameters
        ----------
        *images
            One or more input images, each as a PIL Image, numpy array
            (H W C x uint8), or PyTorch tensor (1 C H W x fp32, range [0, 1]).

        Returns
        -------
        embeddings : np.ndarray
            Array of shape [N, feature_dim] containing one L2-normalized
            embedding per input image.
        """
        batch_tensors = []
        for image in images:
            _, nchw = app_to_net_image_inputs(image)
            resized = torch.nn.functional.interpolate(
                nchw,
                size=(self.input_height, self.input_width),
                mode="bilinear",
                align_corners=False,
            )
            batch_tensors.append(resized)
        batch = torch.cat(batch_tensors, dim=0)
        embeddings = self.model(batch.contiguous())
        return np.asarray(embeddings.detach().cpu())

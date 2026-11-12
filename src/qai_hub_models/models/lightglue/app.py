# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.lightglue.postprocess import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MAX_KEYPOINTS,
    extract_padded_features,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.superpoint.model import SuperPoint
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.draw import draw_connections, draw_points
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec


class LightGlueApp(CollectionAppEvaluateProtocol):
    """End-to-end sparse feature matching: SuperPoint extraction + LightGlue matching.

    Accepts a pair of PIL images, extracts SuperPoint keypoints/descriptors
    for each, matches them with LightGlue, and returns matched point pairs
    in each image's original pixel coordinates.
    """

    def __init__(
        self,
        model: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        extractor: Callable[
            [torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ]
        | None = None,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.model = model
        # Lazy: eval constructs this App via from_components() without ever
        # touching self.extractor, so defer SuperPoint.from_pretrained() until
        # predict() actually needs it.
        self._extractor = extractor
        # Height/width aren't part of LightGlue's input spec (its forward()
        # takes keypoints/descriptors, not an image), so these stay fixed at
        # the model's own compiled-in defaults rather than being app args.
        self.input_height = DEFAULT_HEIGHT
        self.input_width = DEFAULT_WIDTH
        if input_spec is not None:
            _, self.max_keypoints, _ = input_spec["keypoints0"][0]
        else:
            self.max_keypoints = MAX_KEYPOINTS

    @property
    def extractor(
        self,
    ) -> Callable[
        [torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if self._extractor is None:
            self._extractor = SuperPoint.from_pretrained()
        return self._extractor

    @classmethod
    def from_components(
        cls, models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel]
    ) -> LightGlueApp:
        # components are ordered (superpoint, lightglue); LightGluePairDataset
        # already runs SuperPoint host-side, so only lightglue is invoked here.
        return cls(model=models[-1])  # type: ignore[arg-type]

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        # self.model may be an on-device model: each positional arg is then a
        # tuple of per-job chunk tensors rather than a single torch.Tensor.
        if isinstance(model_input, tuple):
            output = cast(Any, self.model)(*model_input)
        else:
            chunks = tuple(next(model_input) for _ in range(6))
            output = cast(Any, self.model)(*chunks)
        yield output
        return output

    def predict(
        self,
        image0: Image.Image,
        image1: Image.Image,
        raw_output: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Match keypoints between image0 and image1.

        Parameters
        ----------
        image0
            PIL image (RGB or grayscale), any resolution.
        image1
            PIL image (RGB or grayscale), any resolution.
        raw_output
            If True, return the raw matched point pairs instead of an
            annotated visualization.

        Returns
        -------
        result : np.ndarray | tuple[np.ndarray, np.ndarray]
            If raw_output: (points0, points1), each shape (N, 2) (x, y)
            pixel coordinates in that image's original resolution, where
            row i of points0 is a mutual match with row i of points1.
            Otherwise: an RGB uint8 numpy array with image0 and image1
            placed side by side and mutual matches drawn as connecting lines.
        """
        orig_w0, orig_h0 = image0.size
        orig_w1, orig_h1 = image1.size
        resized0 = image0.convert("RGB").resize(
            (self.input_width, self.input_height), Image.BILINEAR
        )
        resized1 = image1.convert("RGB").resize(
            (self.input_width, self.input_height), Image.BILINEAR
        )
        _, tensor0 = app_to_net_image_inputs(resized0)
        _, tensor1 = app_to_net_image_inputs(resized1)

        kp0, desc0, n0, kp1, desc1, n1 = extract_padded_features(
            self.extractor, tensor0, tensor1, self.max_keypoints
        )

        m0, _m1, _mscores0, _mscores1 = self.model(kp0, desc0, n0, kp1, desc1, n1)

        n0_i = int(n0[0].item())
        m0_np = m0[0].detach().numpy()[:n0_i]
        valid = m0_np > -1
        idx0 = np.where(valid)[0]
        idx1 = m0_np[valid].astype(np.int64)

        sx0, sy0 = orig_w0 / self.input_width, orig_h0 / self.input_height
        sx1, sy1 = orig_w1 / self.input_width, orig_h1 / self.input_height
        points0 = kp0[0].detach().numpy()[idx0] * np.array([sx0, sy0])
        points1 = kp1[0].detach().numpy()[idx1] * np.array([sx1, sy1])

        if raw_output:
            return points0, points1

        rgb0 = np.array(image0.convert("RGB"))
        rgb1 = np.array(image1.convert("RGB"))
        h = max(rgb0.shape[0], rgb1.shape[0])
        canvas = np.zeros((h, rgb0.shape[1] + rgb1.shape[1], 3), dtype=np.uint8)
        canvas[: rgb0.shape[0], : rgb0.shape[1]] = rgb0
        canvas[: rgb1.shape[0], rgb0.shape[1] :] = rgb1

        offset = rgb0.shape[1]
        points1_shifted = points1 + np.array([offset, 0])
        if len(points0) > 0:
            point_pairs = np.stack([points0, points1_shifted], axis=1)
            draw_connections(canvas, point_pairs, color=(0, 255, 0), size=1)
        draw_points(canvas, points0, (255, 0, 0), 3)
        draw_points(canvas, points1_shifted, (255, 0, 0), 3)
        return canvas

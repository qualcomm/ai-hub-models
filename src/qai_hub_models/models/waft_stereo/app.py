# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import cast

import numpy as np
import torch
from PIL import Image
from qai_hub.client import DatasetEntries

from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    colorize_disparity,
    undo_resize_pad,
)
from qai_hub_models.models.waft_stereo.dataset import resize_pad_rgb
from qai_hub_models.models.waft_stereo.model import build_calibration_data
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec


class WaftStereoApp(CollectionAppEvaluateProtocol, CollectionAppQuantizeProtocol):
    """End-to-end pipeline for the WAFT-Stereo disparity model."""

    def __init__(
        self,
        encoder: Callable[
            [torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        classification: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        regression: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        input_spec: InputSpec | None = None,
    ) -> None:
        self.encoder = encoder
        self.classification = classification
        self.regression = regression
        if input_spec is None:
            # Falls back to the component's own spec when the caller (e.g.
            # validate's App(**model.components)) doesn't have one on hand.
            input_spec = encoder.get_input_spec()  # type: ignore[attr-defined]
        shape = input_spec["img1"][0]
        self.height: int = shape[2]
        self.width: int = shape[3]

    def predict(
        self,
        left: torch.Tensor | np.ndarray | Image.Image,
        right: torch.Tensor | np.ndarray | Image.Image,
        raw_output: bool = False,
    ) -> Image.Image | np.ndarray:
        return self.predict_disparity(left, right, raw_output)

    @classmethod
    def from_components(
        cls, models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel]
    ) -> WaftStereoApp:
        return cls(models[0], models[1], models[2])  # type: ignore[arg-type]

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        img1, img2 = model_input

        enc_output = self.encoder(cast(torch.Tensor, img1), cast(torch.Tensor, img2))
        yield enc_output

        if isinstance(enc_output, AsyncOnDeviceResult):
            fmap1, fmap2, net_state = enc_output.wait()
            cls_output = self.classification(
                fmap1.split(model_batch_size, dim=0),
                fmap2.split(model_batch_size, dim=0),
            )
        else:
            fmap1, fmap2, net_state = enc_output
            cls_output = (
                self.classification(
                    cast(torch.Tensor, fmap1), cast(torch.Tensor, fmap2)
                ),
            )
        yield cls_output

        if isinstance(cls_output, AsyncOnDeviceResult):
            disp = cls_output.wait()
            reg_output = self.regression(
                fmap1.split(model_batch_size, dim=0),
                fmap2.split(model_batch_size, dim=0),
                net_state.split(model_batch_size, dim=0),
                disp.split(model_batch_size, dim=0),
            )
        else:
            (disp,) = cls_output
            reg_output = self.regression(
                cast(torch.Tensor, fmap1),
                cast(torch.Tensor, fmap2),
                cast(torch.Tensor, net_state),
                cast(torch.Tensor, disp),
            )

        yield reg_output  # type: ignore[misc]
        return reg_output  # type: ignore[return-value]

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        return build_calibration_data(
            collection_model, component_name, input_specs, num_samples
        )

    def predict_disparity(
        self,
        left: torch.Tensor | np.ndarray | Image.Image,
        right: torch.Tensor | np.ndarray | Image.Image,
        raw_output: bool = False,
    ) -> np.ndarray | Image.Image:
        """
        Run the stereo disparity pipeline on a left/right image pair.

        Parameters
        ----------
        left
            Left-view RGB image.
        right
            Right-view RGB image.
        raw_output
            If True, return the raw float32 disparity array (H, W).
            If False, return a colorized (INFERNO) PIL.Image visualization.

        Returns
        -------
        np.ndarray | Image.Image
            Raw float32 disparity array of shape (H, W) when
            ``raw_output=True``, otherwise a colorized PIL.Image.
        """
        orig_img, img1_t = app_to_net_image_inputs(left, image_layout="RGB")
        orig_img_right, img2_t = app_to_net_image_inputs(right, image_layout="RGB")

        assert len(orig_img) == 1 and len(orig_img_right) == 1, (
            "predict_disparity supports a single left/right image pair, not a list."
        )
        assert orig_img[0].shape == orig_img_right[0].shape, (
            f"Left and right images must be the same size, got "
            f"{orig_img[0].shape[:2]} and {orig_img_right[0].shape[:2]}."
        )

        img1_t, scale, padding = resize_pad_rgb(img1_t, (self.height, self.width))
        img2_t, _, _ = resize_pad_rgb(img2_t, (self.height, self.width))

        with torch.no_grad():
            fmap1, fmap2, net_state = self.encoder(img1_t, img2_t)
            disp = self.classification(fmap1, fmap2)
            disp_pred = self.regression(fmap1, fmap2, net_state, disp)

        h, w = orig_img[0].shape[:2]
        disp_pred = undo_resize_pad(disp_pred, (w, h), scale, padding)
        # Disparity is a pixel offset, so unlike RGB values it must be
        # rescaled by the same factor used to resize the spatial dimensions.
        disp_pred = disp_pred / scale
        disp_np = disp_pred.squeeze().numpy().astype(np.float32)

        if raw_output:
            return disp_np
        return colorize_disparity(disp_np)

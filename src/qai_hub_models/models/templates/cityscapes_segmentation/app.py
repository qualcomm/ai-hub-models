# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
from PIL import Image as ImageModule
from PIL.Image import Image

from qai_hub_models.models.templates.cityscapes_segmentation.external_repos.ffnet.ffnet_datasets.cityscapes.dataloader.get_dataloaders import (
    return_dataloader,
)
from qai_hub_models.models.templates.cityscapes_segmentation.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    pil_resize_pad,
    pil_undo_resize_pad,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.input_spec import InputSpec


def _load_cityscapes_loader(cityscapes_path: str | None = None) -> object:
    if cityscapes_path is None:
        # Allow a loader without data. There are useful auxiliary functions.
        cityscapes_path = str(
            ASSET_CONFIG.get_local_store_model_path(
                MODEL_ID,
                MODEL_ASSET_VERSION,
                "cityscapes_dummy",
            )
        )

        os.makedirs(
            os.path.join(cityscapes_path, "leftImg8bit", "train"), exist_ok=True
        )
        os.makedirs(os.path.join(cityscapes_path, "leftImg8bit", "val"), exist_ok=True)

    cityscapes_path = os.path.abspath(cityscapes_path)

    os.environ["FFNET_CITYSCAPES_PATH"] = cityscapes_path

    return return_dataloader(num_workers=1, batch_size=1)


def preprocess_cityscapes_image(image: Image) -> torch.Tensor:
    out_tensor: torch.Tensor = standard_transforms.ToTensor()(image)
    return out_tensor.unsqueeze(0)


class CityscapesSegmentationApp:
    """
    This class consists of light-weight "app code" that is required to perform
    end to end inference for single-view (left) semantic segmentation of the
    Cityscapes (https://cityscapes-dataset.com/) dataset.

    The app uses 1 model:
        * Cityscapes segmentation model

    For a given image input, the app will:
        * Pre-process the image
        * Run model inference
        * Resize predictions to map image size
        * Visualize results by super-imposing on input image
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_spec: InputSpec,
    ) -> None:
        self.model = model
        self.color_mapping = _load_cityscapes_loader().dataset.color_mapping  # type: ignore[attr-defined]
        (_, _, self.model_height, self.model_width) = input_spec["image"][0]

    def predict(self, image: Image, raw_output: bool = False) -> Image | np.ndarray:
        """
        From the provided image or tensor, predict semantic segmentation over
        the Cityscapes classes.

        Parameters
        ----------
        image
            A PIL Image in RGB format.
        raw_output
            Whether to return raw output or annotated image.

        Returns
        -------
        output : Image | np.ndarray
            If raw_output is False it will return an annotated image of the
            same size as the input image. If True, it will return raw logit
            probabilities as an numpy array of shape [1, CLASSES, HEIGHT,
            WIDTH]. Note, that WIDTH and HEIGHT will be smaller than the input
            image.
        """
        resized_image, scale, padding = pil_resize_pad(
            image, (self.model_height, self.model_width)
        )

        input_tensor = preprocess_cityscapes_image(resized_image)
        small_res_output = self.model(input_tensor)

        output = F.interpolate(
            small_res_output,
            (resized_image.height, resized_image.width),
            mode="bilinear",
        )
        if raw_output:
            return output.detach().numpy()
        predictions = output[0].argmax(0).byte().cpu().numpy()

        color_mask = ImageModule.fromarray(predictions.astype(np.uint8)).convert("P")
        color_mask.putpalette(self.color_mapping)
        out = ImageModule.blend(resized_image, color_mask.convert("RGB"), 0.5)

        # Resize / unpad annotated image
        return pil_undo_resize_pad(out, image.size, scale, padding)

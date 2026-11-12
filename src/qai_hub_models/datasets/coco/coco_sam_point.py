# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np
import torch

from qai_hub_models.datasets.coco.coco import CocoDatasetBase
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

# Instances covering less than this fraction of the image are too small to click on.
MIN_AREA_FRACTION = 0.01
# SAM's pixel mean in [0, 1]; padding with it is zero after the encoder's normalization.
SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]) / 255.0


def mask_center_point(mask: np.ndarray) -> tuple[int, int]:
    """
    Return the (x, y) pixel farthest from the mask boundary.

    This is the first-click rule of the SAM paper (appendix D.1), taken from
    interactive segmentation. The image border counts as boundary.

    Parameters
    ----------
    mask
        Binary uint8 mask with shape [H, W] containing at least one nonzero pixel.

    Returns
    -------
    x : int
        Column of the center point.
    y : int
        Row of the center point.
    """
    padded = np.pad(mask, 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    return int(x), int(y)


class CocoSamPointDataset(CocoDatasetBase):
    """
    COCO instance masks paired with a single point prompt, for promptable segmentation.

    Each sample is one non-crowd instance that covers at least MIN_AREA_FRACTION of its image.
    The prompt is the point farthest from the mask boundary, which is the SAM
    paper's default first click, so it is deterministic. Absolute scores still
    differ from published SAM numbers, which average 23 datasets and score all masks.

    Images are resized (aspect preserved) into the top-left of the model input
    frame, matching SAM's preprocessing. The remainder is filled with SAM's pixel mean.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        super().__init__(split=split, input_spec=input_spec)
        self.samples: list[tuple[int, int]] = []
        for image_id in self.ids:
            info = self.coco.imgs[image_id]
            image_area = info["height"] * info["width"]
            for ann in self.coco.loadAnns(
                self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
            ):
                if ann["area"] / image_area >= MIN_AREA_FRACTION:
                    self.samples.append((image_id, ann["id"]))

    def __len__(self) -> int:
        return len(self.samples)

    def _to_frame(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resize [1, C, H, W] into the top-left of the [target_h, target_w] frame."""
        return resize_pad(
            tensor,
            (self.target_h, self.target_w),
            vertical_float="top",
            horizontal_float="left",
        )[0]

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Get dataset item.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        model_input : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            image
                RGB, range [0-1], shape [3, target_h, target_w].
            point_coords
                Prompt point (x, y) in pixels of the model input frame,
                float32 with shape [1, 2].
            point_labels
                Always 1 (foreground), float32 with shape [1].

        ground_truth : torch.Tensor
            Binary instance mask in the model input frame, uint8 with shape [target_h, target_w].
        """
        image_id, ann_id = self.samples[index]
        image = self._load_image(image_id)
        ann = self.coco.anns[ann_id]

        image_tensor = app_to_net_image_inputs(image)[1]
        valid = self._to_frame(torch.ones_like(image_tensor[:, :1])) > 0.5
        image_tensor = torch.where(
            valid, self._to_frame(image_tensor), SAM_PIXEL_MEAN.view(1, 3, 1, 1)
        ).squeeze(0)

        mask_tensor = torch.from_numpy(self.coco.annToMask(ann))[None, None].float()
        mask = (self._to_frame(mask_tensor) > 0.5).squeeze().to(torch.uint8)

        x, y = mask_center_point(mask.numpy())
        point_coords = torch.tensor([[x, y]], dtype=torch.float32)
        point_labels = torch.ones(1, dtype=torch.float32)
        return (image_tensor, point_coords, point_labels), mask

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100

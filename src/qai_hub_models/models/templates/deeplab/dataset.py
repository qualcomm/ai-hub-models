# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from pycocotools import mask as coco_mask
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from qai_hub_models.datasets.coco import CocoDatasetBase
from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.utils.base_dataset import DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

# VOC class index -> COCO category id.
# Matches torchvision references/segmentation/coco_utils.py CAT_LIST exactly.
# Index = VOC label (0=background, 1=aeroplane, ..., 20=tvmonitor).
# Source: https://github.com/pytorch/vision/blob/main/references/segmentation/coco_utils.py
_VOC_CAT_LIST = [
    0,
    5,
    2,
    16,
    9,
    44,
    6,
    3,
    17,
    62,
    21,
    67,
    18,
    19,
    4,
    1,
    64,
    20,
    63,
    7,
    72,
]

# COCO category id -> remapped VOC label index.
_COCO_ID_TO_VOC: dict[int, int] = {
    coco_id: voc_idx for voc_idx, coco_id in enumerate(_VOC_CAT_LIST)
}

IGNORE_INDEX = 255

# Standard input size for the COCO-with-VOC-labels evaluation protocol.
# Source: https://github.com/pytorch/vision/blob/main/references/segmentation/train.py
_DEFAULT_INPUT_SIZE = 520


class CocoVocSegDataset(CocoDatasetBase):
    """COCO 2017 images with VOC semantic segmentation labels.

    Implements the COCO-with-VOC-labels evaluation protocol:

      - Images: COCO 2017, resized to input_spec dimensions (default 520x520),
                returned as float [0, 1] RGB.
      - Labels: per-pixel semantic masks built from COCO instance annotations,
                remapped to 21 VOC classes via ConvertCocoPolysToMask logic.
                Overlapping instance pixels -> IGNORE_INDEX (255).
      - Target resized (nearest-neighbour) to match the image dimensions.

    References
    ----------
        https://github.com/pytorch/vision/blob/main/references/segmentation/coco_utils.py
        https://github.com/pytorch/vision/blob/main/references/segmentation/train.py
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        input_spec = input_spec or {
            "image": TensorSpec(shape=(1, 3, _DEFAULT_INPUT_SIZE, _DEFAULT_INPUT_SIZE))
        }
        self.input_height: int = input_spec["image"][0][2]
        self.input_width: int = input_spec["image"][0][3]

        super().__init__(split=split, input_spec=input_spec)

        # Keep only images that have at least one annotation.
        # Matches torchvision's FilterAndRemapCocoCategories behaviour.
        annotated_img_ids: set[int] = {
            ann["image_id"] for ann in self.coco.anns.values()
        }
        self.ids = [img_id for img_id in self.ids if img_id in annotated_img_ids]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single (image, segmentation mask) pair.

        Parameters
        ----------
        index
            Dataset index.

        Returns
        -------
        image : torch.Tensor
            Shape (3, input_height, input_width), dtype float32, range [0, 1], RGB.
        target : torch.Tensor
            Shape (input_height, input_width), dtype int64.
            Values in [0, 20] are VOC class labels; 255 = IGNORE_INDEX.
        """
        img, anns = CocoDetection.__getitem__(self, index)
        orig_w, orig_h = img.size

        img_tensor = preprocess_PIL_image(
            img.resize((self.input_width, self.input_height))
        ).squeeze(0)

        target = self._build_semantic_target(anns, orig_h, orig_w)
        target_tensor = (
            F.resize(
                target.unsqueeze(0),
                size=[self.input_height, self.input_width],
                interpolation=InterpolationMode.NEAREST,
            )
            .squeeze(0)
            .long()
        )

        return img_tensor, target_tensor

    @staticmethod
    def _convert_polys_to_mask(
        segmentations: list[Any], height: int, width: int
    ) -> torch.Tensor:
        """Convert COCO polygon/RLE segmentations to a stacked binary mask tensor.

        Parameters
        ----------
        segmentations
            List of COCO segmentation entries; each entry is either a polygon list
            or an RLE dict as returned by the COCO API.
        height
            Image height in pixels.
        width
            Image width in pixels.

        Returns
        -------
        masks : torch.Tensor
            Shape (N, height, width), dtype uint8. Binary mask per annotation.
            Returns shape (0, height, width) when the input list is empty.
        """
        masks = []
        for polygons in segmentations:
            if not isinstance(polygons, (list, dict)):
                continue
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        if masks:
            return torch.stack(masks, dim=0)
        return torch.zeros((0, height, width), dtype=torch.uint8)

    @staticmethod
    def _build_semantic_target(
        anns: list[dict[str, Any]], height: int, width: int
    ) -> torch.Tensor:
        """Build a per-pixel semantic label map from COCO annotations.

        Replicates torchvision's ConvertCocoPolysToMask exactly:
          - Remaps COCO category ids to VOC label indices (0-20).
          - Composites masks using (masks * cats).max(dim=0) — highest VOC label wins.
          - Pixels covered by more than one instance mask are set to IGNORE_INDEX (255).
          - Annotations not in the VOC category list are skipped.

        Parameters
        ----------
        anns
            COCO annotation dicts for a single image, as returned by CocoDetection.
        height
            Image height in pixels.
        width
            Image width in pixels.

        Returns
        -------
        target : torch.Tensor
            Shape (height, width), dtype uint8.
            Values in [0, 20] are VOC class labels; 255 = IGNORE_INDEX.
        """
        voc_anns = [a for a in anns if a["category_id"] in _COCO_ID_TO_VOC]
        if not voc_anns:
            return torch.zeros((height, width), dtype=torch.uint8)

        segmentations = [a["segmentation"] for a in voc_anns]
        cats = torch.as_tensor(
            [_COCO_ID_TO_VOC[a["category_id"]] for a in voc_anns],
            dtype=torch.uint8,
        )

        masks = CocoVocSegDataset._convert_polys_to_mask(segmentations, height, width)

        # Composite: take the maximum VOC label at each pixel.
        target, _ = (masks * cats[:, None, None]).max(dim=0)
        # Pixels covered by more than one instance are ambiguous -> ignore.
        target[masks.sum(0) > 1] = IGNORE_INDEX
        return target

    @staticmethod
    def default_samples_per_job() -> int:
        return 1000

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description="val2017 split with VOC class labels",
        )

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import torch
from torchvision.datasets.coco import CocoDetection

from qai_hub_models.datasets.coco.coco import CocoDatasetBase, CocoDatasetClass
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)


class CocoSegDataset(CocoDatasetBase):
    """
    Wrapper class around COCO dataset https://cocodataset.org/

    Contains Segmentation samples and labels spanning 80 or 91 classes.

    This wrapper supports the train and val splits of the 2017 version.
    """

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, int]]:
        """
        Get dataset item.

        Parameters
        ----------
        index
            Index of the sample to retrieve.


        Returns
        -------
        image : torch.Tensor
            RGB, range [0-1] network input image.

        ground_truth : tuple[torch.Tensor, torch.Tensor, int]
            mask_data
                mask data with shape (self.max_boxes, self.target_h, self.target_w)
            labels
                labels with shape (self.max_boxes,)
            bbox_count
                number of actual boxes present
        """
        image, target = CocoDetection.__getitem__(self, index)
        image = image.resize((self.target_w, self.target_h))
        width, height = image.size

        masks_list = []
        labels_list = []
        for annotation in target:
            mask = cv2.resize(
                self.coco.annToMask(annotation),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            masks_list.append(mask)
            label = int(annotation.get("category_id"))
            if self.num_classes == CocoDatasetClass.SUBSET_CLASSES:
                label = self.class80_label_map[label]
            labels_list.append(label)

        masks = torch.tensor(masks_list).to(torch.uint8)
        labels = torch.tensor(labels_list).to(torch.uint8)

        num_boxes = len(labels)
        if num_boxes == 0:
            masks = torch.zeros((self.max_boxes, self.target_h, self.target_w)).to(
                torch.uint8
            )
            labels = torch.zeros(self.max_boxes).to(torch.uint8)
        elif num_boxes > self.max_boxes:
            raise ValueError(
                f"Sample has more boxes than max boxes {self.max_boxes}. "
                "Re-initialize the dataset with a larger value for max_boxes."
            )
        else:
            extra_masks = torch.zeros(
                (
                    self.max_boxes - num_boxes,
                    self.target_h,
                    self.target_w,
                )
            ).to(torch.uint8)
            extra_labels = torch.zeros(self.max_boxes - num_boxes).to(torch.uint8)
            masks = torch.concat([masks, extra_masks])
            labels = torch.concat([labels, extra_labels])

        image_pt = app_to_net_image_inputs(image)[1].squeeze(0)
        return image_pt, (masks, labels, num_boxes)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100

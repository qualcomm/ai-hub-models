# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers.models.mask2former.image_processing_mask2former import (
    compute_segments,
    remove_low_and_no_objects,
)

from qai_hub_models.models.templates.corevision.draw import create_color_map
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)


class Mask2FormerApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference for Segmentation.

    For a given image input, the app will:
        * Run inference
        * Convert the output segmentation mask into a visual representation
        * Overlay the segmentation mask onto the image and return it
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ) -> None:
        self.model = model

    def predict(self, *args: Any, **kwargs: Any) -> list[Image.Image] | np.ndarray:
        # See segment_image.
        return self.segment_image(*args, **kwargs)

    def segment_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: bool = False,
    ) -> list[Image.Image] | np.ndarray:
        """
        Return the input image with the segmentation mask overlayed on it.

        Parameters
        ----------
        pixel_values_or_image
            PIL image(s)
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        raw_output
            See "returns" doc section for details.

        Returns
        -------
        result : list[Image.Image] | np.ndarray
            If raw_output is False, returns:
            segmented_images
                Images with segmentation map overlaid with an alpha of 0.5.

            If raw_output is True, returns:
            masks
                A list of predicted masks.
        """
        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )

        # Run prediction
        class_pred_scores, class_pred_labels, mask_logits = self.model(
            NCHW_fp32_torch_frames
        )

        pred_mask_img = self.post_process_panoptic_segmentation(
            class_pred_scores, class_pred_labels, mask_logits
        )[0]
        pred_mask_img = pred_mask_img["segmentation"].unsqueeze(0)

        if raw_output:
            return pred_mask_img

        # Create color map and convert segmentation mask to RGB image
        # Overlay the segmentation masks on the image.
        color_map = create_color_map(pred_mask_img.max().item() + 1)
        out = []
        for i, img_tensor in enumerate(NHWC_int_numpy_frames):
            out.append(
                Image.blend(
                    Image.fromarray(img_tensor),
                    Image.fromarray(color_map[pred_mask_img[i]]),
                    alpha=0.5,
                )
            )
        return out

    @staticmethod
    def post_process_panoptic_segmentation(
        class_pred_scores: torch.Tensor,
        class_pred_labels: torch.Tensor,
        masks_queries_logits: torch.Tensor,
        num_labels: int = 133,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: set[int] | None = None,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Converts the output of [`Mask2FormerForUniversalSegmentationOutput`] into image panoptic segmentation
        predictions. Only supports PyTorch.

        Parameters
        ----------
        class_pred_scores
            Score (probability) of the predicted label for each batch. Shape [num_batches, num_classes]
        class_pred_labels
            Index of the predicted label for each batch.  Shape [num_batches, num_classes]
        masks_queries_logits
            Lotgits for each pixel. Logits predict which pixel is part of the predicted mask.
            [1, num_classes, modified_height, modified_width]
            where the modified height and width will be some factor smaller than the input image.
        num_labels
            Number of labels (default 133).
        threshold
            The probability score threshold to keep predicted instance masks (default 0.5).
        mask_threshold
            Threshold to use when turning the predicted masks into binary values (default 0.5).
        overlap_mask_area_threshold
            The overlap mask area threshold to merge or discard small disconnected parts within each binary
            instance mask.
        label_ids_to_fuse
            The labels in this state will have all their instances be fused together. For instance we could say
            there can only be one sky in an image, but several persons, so the label ID for sky would be in that
            set, but not the one for person.
        target_sizes
            List of length (batch_size), where each list item corresponds to the requested
            final size (height, width) of each prediction in batch. If left to None, predictions will not be
            resized.

        Returns
        -------
        segmentation_results : list[dict[str, Any]]
            A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
            to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
            to the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """
        if label_ids_to_fuse is None:
            # logger.warning("`label_ids_to_fuse` unset. No instance will be fused.")
            label_ids_to_fuse = set()
        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = F.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )
        batch_size = class_pred_scores.shape[0]
        mask_probs = masks_queries_logits.sigmoid()
        # Loop over items in batch size
        results = []

        for i in range(batch_size):
            (
                mask_probs_item,
                pred_scores_item,
                pred_labels_item,
            ) = remove_low_and_no_objects(
                mask_probs[i],
                class_pred_scores[i],
                class_pred_labels[i],
                threshold,
                num_labels,
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = (
                    target_sizes[i]
                    if target_sizes is not None
                    else mask_probs_item.shape[1:]
                )
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            # Get segmentation map and segment information of batch item
            target_size = target_sizes[i] if target_sizes is not None else None
            segmentation, segments = compute_segments(
                mask_probs=mask_probs_item,
                pred_scores=pred_scores_item,
                pred_labels=pred_labels_item,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                label_ids_to_fuse=label_ids_to_fuse,
                target_size=target_size,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

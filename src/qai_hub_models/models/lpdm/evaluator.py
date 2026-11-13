# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import torch

from qai_hub_models.models.lpdm.dataset import (
    CLASS_NAMES,
    EMIRATE_CLASS_IDS,
    PLATE_CLASS_ID,
)
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import PLATE_STRING_ACCURACY, MetricMetadata

# Character classes: digits 0-9 and letters A-Z only (not plate/exp/emirate)
CHAR_CLASS_IDS = frozenset(
    i
    for i, name in enumerate(CLASS_NAMES)
    if name not in {"plate", "exp"} and i not in EMIRATE_CLASS_IDS
)

# Human-readable emirate display names
_EMIRATE_DISPLAY = {
    "new_DUBAI": "Dubai",
    "old_DUBAI": "Dubai",
    "new_abudabi": "Abu Dhabi",
    "old_abudabi": "Abu Dhabi",
    "new_RAK": "Ras Al Khaimah",
    "old_RAK": "Ras Al Khaimah",
    "new_ajman": "Ajman",
    "old_ajman": "Ajman",
    "new_am": "Umm Al Quwain",
    "old_am": "Umm Al Quwain",
    "new_fujairah": "Fujairah",
    "old_fujira": "Fujairah",
    "old_sharka": "Sharjah",
}


@dataclass
class PlateComponents:
    """Decomposed UAE licence plate reading."""

    category_number: str = ""  # digits/letters LEFT of the emirate logo
    plate_number: str = ""  # digits/letters RIGHT of the emirate logo
    emirate: str = ""  # normalised emirate name (e.g. "Abu Dhabi")


def decompose_plate(
    boxes: torch.Tensor,
    classes: torch.Tensor,
) -> PlateComponents:
    """Decompose detections for one image into UAE plate components.

    UAE plates have the structure:
        [category number]  [emirate logo]  [plate number]

    Characters to the LEFT of the emirate logo centre-x → category number.
    Characters to the RIGHT → plate number.
    If no emirate is detected, all characters are placed in plate_number.
    """
    if boxes.numel() == 0:
        return PlateComponents()

    # Bulk-convert to Python lists to avoid per-element device syncs.
    cls_list: list[int] = classes.tolist()
    cx_list: list[float] = ((boxes[:, 0] + boxes[:, 2]) / 2).tolist()

    emirate_cx: float | None = None
    emirate_name = ""
    chars: list[tuple[float, str]] = []  # (centre_x, char)

    for cls, cx in zip(cls_list, cx_list, strict=False):
        if cls == PLATE_CLASS_ID:
            continue
        if cls in EMIRATE_CLASS_IDS:
            # Keep the first emirate found; duplicates are unexpected in practice.
            if emirate_name == "":
                raw = CLASS_NAMES[cls]
                emirate_name = _EMIRATE_DISPLAY.get(raw, raw)
                emirate_cx = cx
        elif cls in CHAR_CLASS_IDS:
            chars.append((cx, CLASS_NAMES[cls]))

    # Sort all characters left-to-right
    chars.sort(key=lambda t: t[0])

    if emirate_cx is None:
        # No emirate detected — all chars go to plate_number
        return PlateComponents(
            plate_number="".join(c for _, c in chars),
            emirate=emirate_name,
        )

    category = "".join(c for x, c in chars if x < emirate_cx)
    number = "".join(c for x, c in chars if x >= emirate_cx)
    return PlateComponents(
        category_number=category,
        plate_number=number,
        emirate=emirate_name,
    )


class LicensePlateEvaluator(BaseEvaluator):
    """Evaluator for the LPDM UAE licence-plate detector.

    Primary score (``get_accuracy_score``): full plate match rate — image is
    correct when plate_number AND emirate both match ground truth exactly.

    Ground truth: (image_id, h, w, boxes [MAX_BOXES, 4], labels [MAX_BOXES], num_boxes)
    Model output: (pred_boxes [B, N, 4], pred_scores [B, N], pred_classes [B, N])
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        score_threshold: float,
        nms_iou_threshold: float,
    ) -> None:
        self.scale_x = 1.0 / image_width
        self.scale_y = 1.0 / image_height
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.reset()

    def reset(self) -> None:
        self._total = 0
        self._full_match = 0  # category + number + state all correct
        self._category_exact = 0  # plate category (left of logo) exact
        self._number_exact = 0  # plate number (right of logo) exact
        self._state_correct = 0  # emirate / plate state correct
        self._state_total = 0  # images where GT has an emirate label

    def add_batch(
        self,
        output: Collection[torch.Tensor],
        gt: Collection[torch.Tensor],
    ) -> None:
        """Accumulate one batch of predictions against ground truth.

        output
            (pred_boxes [B, N, 4], pred_scores [B, N], pred_class_idx [B, N]) —
            raw model output before NMS (pixel-scale xyxy boxes).
        gt
            (image_id, height, width, boxes [B, MAX_BOXES, 4], labels [B, MAX_BOXES],
            num_boxes [B]) — GT boxes in normalised [0, 1] xyxy coords.
        """
        pred_boxes, pred_scores, pred_class_idx = output
        _, _, _, all_gt_boxes, all_gt_labels, all_num_boxes = gt

        pred_nms_boxes, _, pred_nms_classes = batched_nms(
            self.nms_iou_threshold,
            self.score_threshold,
            pred_boxes.float(),
            pred_scores.float(),
            pred_class_idx,
        )

        batch_size = pred_boxes.shape[0]
        for i in range(batch_size):
            n_gt = int(all_num_boxes[i].item())
            gt_boxes = all_gt_boxes[i, :n_gt]
            gt_labels = all_gt_labels[i, :n_gt]

            # Normalise predicted boxes to [0, 1] to match GT coordinate space.
            p_boxes_norm = pred_nms_boxes[i].float()
            p_boxes_norm[:, [0, 2]] *= self.scale_x
            p_boxes_norm[:, [1, 3]] *= self.scale_y

            pred = decompose_plate(p_boxes_norm, pred_nms_classes[i])
            gt_plate = decompose_plate(gt_boxes, gt_labels)

            self._total += 1

            cat_ok = pred.category_number == gt_plate.category_number
            num_ok = pred.plate_number == gt_plate.plate_number
            state_ok = pred.emirate == gt_plate.emirate

            if cat_ok:
                self._category_exact += 1
            if num_ok:
                self._number_exact += 1
            if gt_plate.emirate:
                self._state_total += 1
                if state_ok:
                    self._state_correct += 1
            if cat_ok and num_ok and state_ok:
                self._full_match += 1

    def get_accuracy_score(self) -> float:
        """Primary metric: fraction of images where category, number, and state all match, 0-100."""
        if self._total == 0:
            return 0.0
        return 100.0 * self._full_match / self._total

    def formatted_accuracy(self) -> str:
        if self._total == 0:
            return "No samples evaluated."

        full_acc = 100.0 * self._full_match / self._total
        cat_acc = 100.0 * self._category_exact / self._total
        num_acc = 100.0 * self._number_exact / self._total
        state_acc = (
            100.0 * self._state_correct / self._state_total
            if self._state_total > 0
            else 0.0
        )
        return (
            f"Plate String Accuracy: {full_acc:.1f}% | "
            f"Plate Category: {cat_acc:.1f}% | "
            f"Plate Number: {num_acc:.1f}% | "
            f"Plate State: {state_acc:.1f}%"
        )

    def get_metric_metadata(self) -> MetricMetadata:
        return PLATE_STRING_ACCURACY

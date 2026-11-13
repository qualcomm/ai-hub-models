# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from collections.abc import Collection
from typing import cast

import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from qai_hub_models.datasets.coco.coco import COCO_ANNOTATIONS
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
    box_xywh_to_xyxy,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    MEAN_AVERAGE_PRECISION_IOU_5_95,
    MetricMetadata,
)
from qai_hub_models.utils.printing import suppress_stdout


class COCODetectionEvaluator(BaseEvaluator):
    """Detection evaluator using the official COCO metric (pycocotools COCOeval).

    Predictions are accumulated as COCO-format dicts and evaluated with
    COCOeval(iouType="bbox"), matching the methodology used by source repos
    such as meituan/YOLOv6 and mmdetection.

    Predictions are automatically unpadded and unscaled from letterboxed model
    input space back to original image coordinates before evaluation. Class
    indices are remapped from the contiguous class index back to COCO's
    ``category_id`` using the category ordering from the annotation file,
    mirroring the mapping built by ``CocoDataset``.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        nms_iou_threshold: float | None = None,
        score_threshold: float | None = None,
        use_multi_label_nms: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        image_height
            Model input image height in pixels.
        image_width
            Model input image width in pixels.
        nms_iou_threshold
            If set, class-aware non-maximum suppression is applied to the
            model's output boxes using this IoU threshold before accumulating
            predictions.
        score_threshold
            If set, all detections with a confidence score below this value
            are discarded before accumulating predictions.
        use_multi_label_nms
            If True, ``output`` must be a single raw detector tensor of shape
            (batch, num_preds, 5 + num_classes) in
            [x_c, y_c, w, h, obj_conf, cls_conf...] format.
            Multi-label NMS (one detection per box-class pair) is applied,
            matching the meituan/YOLOv6 eval pipeline exactly.
        """
        self.coco_gt: COCO = COCO(
            str(COCO_ANNOTATIONS.extracted_path / "instances_val2017.json")
        )
        # Build contiguous-index -> category_id mapping from the annotation file,
        # mirroring CocoDataset.class80_label_map (category_id -> index), inverted.
        categories = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        categories.sort(key=lambda x: x["id"])
        self.idx_to_category_id: list[int] = [cat["id"] for cat in categories]
        self.image_height = image_height
        self.image_width = image_width
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold
        self.use_multi_label_nms = use_multi_label_nms
        self.reset()

    def reset(self) -> None:
        """Reset accumulated predictions."""
        self.predictions: list[dict] = []

    def _letterbox_geometry(
        self, orig_h: float, orig_w: float
    ) -> tuple[float, float, float]:
        """Return (scale, pad_x, pad_y) used to unletterbox predictions back to original image coordinates.

        Override to match a model's own preprocessing convention (e.g. rounded
        resize + floor-divided padding) instead of this unrounded float default.
        """
        scale = min(self.image_height / orig_h, self.image_width / orig_w)
        pad_x = (self.image_width - orig_w * scale) / 2.0
        pad_y = (self.image_height - orig_h * scale) / 2.0
        return scale, pad_x, pad_y

    def add_batch(
        self,
        output: torch.Tensor
        | torch.NumberType
        | Collection[torch.Tensor | torch.NumberType],
        gt: torch.Tensor
        | torch.NumberType
        | Collection[torch.Tensor | torch.NumberType],
    ) -> None:
        """Add a batch of predictions for COCO evaluation.

        Parameters
        ----------
        output
            If ``use_multi_label_nms`` is False (default):
                pred_boxes
                    Shape (batch_size, num_candidates, 4), dtype float32.
                    Bounding boxes in (x1, y1, x2, y2) pixel order within the
                    letterboxed model input space.
                pred_scores
                    Shape (batch_size, num_candidates), dtype float32.
                    Confidence scores in [0, 1].
                pred_class_idx
                    Shape (batch_size, num_candidates), dtype int64.
                    Contiguous class indices (0-79); remapped to COCO
                    ``category_id`` via ``self.idx_to_category_id``.
            If ``use_multi_label_nms`` is True:
                raw_output
                    Shape (batch_size, num_preds, 5 + num_classes), dtype float32.
                    Raw detector tensor in [x_c, y_c, w, h, obj_conf, cls_conf...]
                    format. Multi-label NMS is applied internally.
        gt
            image_ids
                Shape (batch_size,), dtype int64.
                COCO image IDs used to look up original image dimensions.
            image_heights
                Shape (batch_size,). Unused; original size is fetched from
                the COCO annotation file.
            image_widths
                Shape (batch_size,). Unused; original size is fetched from
                the COCO annotation file.
            bounding_boxes
                Shape (batch_size, max_boxes, 4). Unused for accumulation.
            classes
                Shape (batch_size, max_boxes). Unused for accumulation.
            num_boxes
                Shape (batch_size,). Unused for accumulation.
        """
        output = cast(Collection[torch.Tensor], output)
        gt = cast(Collection[torch.Tensor], gt)
        image_ids, _, _, _, _, _ = gt

        if self.use_multi_label_nms:
            raw = cast(
                torch.Tensor,
                output[0] if isinstance(output, (tuple, list)) else output,
            )
            self._add_batch_multi_label(raw, image_ids)
            return

        pred_boxes, pred_scores, pred_class_idx = output

        if self.nms_iou_threshold is not None:
            pred_boxes_list, pred_scores_list, pred_class_list = batched_nms(
                self.nms_iou_threshold,
                self.score_threshold,
                pred_boxes.float(),
                pred_scores.float(),
                pred_class_idx,
            )
        else:
            pred_boxes_list = list(pred_boxes)
            pred_scores_list = list(pred_scores)
            pred_class_list = list(pred_class_idx)
            if self.score_threshold is not None:
                score_masks = [s > self.score_threshold for s in pred_scores_list]
                pred_boxes_list = [
                    b[score_masks[i]] for i, b in enumerate(pred_boxes_list)
                ]
                pred_class_list = [
                    c[score_masks[i]] for i, c in enumerate(pred_class_list)
                ]
                pred_scores_list = [
                    s[score_masks[i]] for i, s in enumerate(pred_scores_list)
                ]

        for i in range(len(image_ids)):
            image_id = int(image_ids[i])

            # Get original image size from COCO annotations (not from gt tuple,
            # which contains model input size, not original image size).
            img_info = self.coco_gt.loadImgs(image_id)[0]
            orig_h = float(img_info["height"])
            orig_w = float(img_info["width"])

            scale, pad_x, pad_y = self._letterbox_geometry(orig_h, orig_w)

            boxes = pred_boxes_list[i]
            scores = pred_scores_list[i]
            classes = pred_class_list[i]

            for j in range(len(scores)):
                box = boxes[j].tolist()
                if any(math.isnan(v) for v in box):
                    continue
                score = float(scores[j])
                cls80 = int(classes[j])
                if cls80 >= len(self.idx_to_category_id):
                    continue
                category_id = self.idx_to_category_id[cls80]

                # Unpad and unscale from model pixel space to original image coords
                x1 = max(0.0, min((box[0] - pad_x) / scale, orig_w))
                y1 = max(0.0, min((box[1] - pad_y) / scale, orig_h))
                x2 = max(0.0, min((box[2] - pad_x) / scale, orig_w))
                y2 = max(0.0, min((box[3] - pad_y) / scale, orig_h))

                self.predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )

    def _add_batch_multi_label(
        self,
        raw_output: torch.Tensor,
        image_ids: torch.Tensor,
    ) -> None:
        """Apply multi-label NMS to raw detector output and accumulate predictions.

        Replicates the meituan/YOLOv6 ``non_max_suppression(multi_label=True)``
        pipeline: pre-filter by obj_conf and max_cls_conf (threshold 0.03),
        multiply obj_conf into class scores, then expand one candidate per
        (box, class) pair and run class-offset NMS.

        Parameters
        ----------
        raw_output
            Shape (batch, num_preds, 5 + num_classes) in
            [x_c, y_c, w, h, obj_conf, cls_conf...] format.
        image_ids
            Shape (batch,). COCO image IDs.
        """
        # Thresholds match meituan/YOLOv6 non_max_suppression defaults.
        pre_filter_thres = 0.03  # applied to obj_conf and max_cls_conf before multiply
        post_filter_thres = 0.03  # applied to combined score after multiply
        iou_thres = (
            self.nms_iou_threshold if self.nms_iou_threshold is not None else 0.65
        )
        max_wh = 4096
        max_nms = 30000  # cap before NMS to avoid OOM on dense predictions

        for i in range(raw_output.shape[0]):
            x = raw_output[i].clone()  # (num_preds, 5+nc)
            image_id = int(image_ids[i])

            img_info = self.coco_gt.loadImgs(image_id)[0]
            orig_h = float(img_info["height"])
            orig_w = float(img_info["width"])

            # Pre-filter: both obj_conf and max_cls_conf must exceed threshold
            obj_conf = x[:, 4]
            cls_conf_max = x[:, 5:].max(dim=-1).values
            mask = (obj_conf > pre_filter_thres) & (cls_conf_max > pre_filter_thres)
            x = x[mask]
            if x.shape[0] == 0:
                continue

            # combined_score = obj_conf * cls_conf
            x[:, 5:] *= x[:, 4:5]

            # xywh -> xyxy in model pixel space
            boxes = box_xywh_to_xyxy(x[:, :4])

            # Multi-label expansion: one entry per (box, class) pair above threshold
            box_idx, cls_idx = (x[:, 5:] > post_filter_thres).nonzero(as_tuple=False).T
            scores_ml = x[box_idx, cls_idx + 5]
            boxes_ml = boxes[box_idx]

            # Cap before NMS
            if len(scores_ml) > max_nms:
                top_idx = scores_ml.argsort(descending=True)[:max_nms]
                scores_ml = scores_ml[top_idx]
                boxes_ml = boxes_ml[top_idx]
                cls_idx = cls_idx[top_idx]

            # Class-offset NMS (each class in its own tile)
            offsets = cls_idx.float().unsqueeze(1) * max_wh
            keep = torchvision.ops.nms(boxes_ml + offsets, scores_ml, iou_thres)
            boxes_ml = boxes_ml[keep]
            scores_ml = scores_ml[keep]
            cls_idx = cls_idx[keep]

            # Unscale from model space to original image coordinates
            scale = min(self.image_height / orig_h, self.image_width / orig_w)
            pad_x = (self.image_width - orig_w * scale) / 2.0
            pad_y = (self.image_height - orig_h * scale) / 2.0

            pad = torch.tensor([pad_x, pad_y, pad_x, pad_y], dtype=boxes_ml.dtype)
            clip = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=boxes_ml.dtype)
            boxes_orig = ((boxes_ml - pad) / scale).clamp(min=0.0)
            boxes_orig = torch.min(boxes_orig, clip)

            for j in range(len(keep)):
                score = float(scores_ml[j])
                cls80 = int(cls_idx[j])
                if cls80 >= len(self.idx_to_category_id):
                    continue
                category_id = self.idx_to_category_id[cls80]
                x1, y1, x2, y2 = boxes_orig[j].tolist()
                self.predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )

    def get_accuracy_score(self) -> float:
        return self._run_coco_eval()["AP"]

    def formatted_accuracy(self) -> str:
        results = self._run_coco_eval()
        return f"{results['AP']:.3f} mAP@0.50:0.95"

    def get_metric_metadata(self) -> MetricMetadata:
        return MEAN_AVERAGE_PRECISION_IOU_5_95

    def _run_coco_eval(self) -> dict[str, float]:
        """Run pycocotools COCOeval on accumulated predictions."""
        if not self.predictions:
            return {"AP": 0.0, "AP50": 0.0}

        pred_image_ids = list({p["image_id"] for p in self.predictions})
        with suppress_stdout():
            coco_dt = self.coco_gt.loadRes(self.predictions)
            coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
            coco_eval.params.imgIds = pred_image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        return {
            "AP": float(coco_eval.stats[0]) * 100,
            "AP50": float(coco_eval.stats[1]) * 100,
        }

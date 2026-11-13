# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import math

import pytest
from podm.metrics import BoundingBox

from qai_hub_models.models.templates.detection.detection_evaluator import mAPEvaluator


class _StoreOnlyEvaluator(mAPEvaluator):
    """mAPEvaluator is abstract; these tests exercise storage + mAP only."""

    def add_batch(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


def _box(xtl: float, ytl: float, xbr: float, ybr: float, score: float) -> BoundingBox:
    return BoundingBox.of_bbox("img0", 0, xtl, ytl, xbr, ybr, score)


def test_get_map_ignores_non_finite_predictions() -> None:
    """
    Non-finite predicted boxes must not reach podm.

    A model that overflows on device (eg. fp16) emits inf coordinates. podm's
    intersection_over_union then computes a nan IoU and trips its internal
    `assert iou >= 0`, crashing evaluation. Such a prediction should be dropped
    and scored as a miss instead.
    """
    evaluator = _StoreOnlyEvaluator()
    # Exact coordinates from the yolor float failure: the prediction has zero
    # width and infinite height, and overlaps the GT box in x -- podm's
    # is_intersecting short-circuits to 0 unless they overlap, so a
    # non-overlapping inf box would not reproduce the crash.
    gt = [
        _box(
            0.7390468716621399,
            0.7701718807220459,
            0.7918280959129333,
            0.832812488079071,
            1.0,
        )
    ]
    inf_pred = _box(0.7562500000000001, -math.inf, 0.7562500000000001, math.inf, 0.9)
    evaluator.store_bboxes_for_eval(gt, [inf_pred])

    # Would raise AssertionError from podm's intersection_over_union if the
    # non-finite prediction were not dropped.
    mAP, _, _, _, _ = evaluator.get_mAP()
    assert mAP == pytest.approx(0.0)
    assert evaluator.pred_bbox == []


def test_get_map_keeps_finite_predictions() -> None:
    """A finite prediction overlapping the ground truth still scores."""
    evaluator = _StoreOnlyEvaluator()
    gt = [_box(0.2, 0.2, 0.4, 0.4, 1.0)]
    evaluator.store_bboxes_for_eval(gt, [_box(0.2, 0.2, 0.4, 0.4, 0.9)])

    assert len(evaluator.pred_bbox) == 1
    mAP, _, _, _, _ = evaluator.get_mAP()
    assert mAP == pytest.approx(100.0)

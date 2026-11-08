# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Centralized metric definitions used by evaluators and info.yaml validation.

Each metric is registered via ``_register_metric``, which appends to
a module-level list and collects ``(name, unit)`` into VALID_METRIC_PAIRS.
After all metrics are defined, VALID_METRIC_PAIRS is frozen to prevent
accidental mutation.
"""

from __future__ import annotations

from typing import NamedTuple


class MetricMetadata(NamedTuple):
    name: str
    unit: str
    description: str
    range: tuple[float | None, float | None]
    float_vs_device_threshold: float | None = None
    # False for error/distance metrics (WER, NME, perplexity, ...), where a
    # smaller score is a better model.
    higher_is_better: bool = True

    def with_description(self, description: str) -> MetricMetadata:
        """Return a copy with a different description."""
        return self._replace(description=description)


_VALID_METRIC_PAIRS: set[tuple[str, str]] = set()


def _register_metric(metric: MetricMetadata) -> MetricMetadata:
    """Register a MetricMetadata and return it."""
    key = (metric.name, metric.unit)
    assert key not in _VALID_METRIC_PAIRS, (
        f"Duplicate metric key {key!r} — use a distinct unit to differentiate"
    )
    _VALID_METRIC_PAIRS.add(key)
    return metric


ACCURACY = _register_metric(
    MetricMetadata(
        name="Accuracy",
        unit="%",
        description="Percentage of model predictions that are correct.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

ACCURACY_TOP1 = _register_metric(
    MetricMetadata(
        name="Top-1 Accuracy",
        unit="%",
        description="Percentage of top 1 model predictions that are correct.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

AVERAGE_PRECISION = _register_metric(
    MetricMetadata(
        name="Average Precision",
        unit="AP",
        description="Area under the precision-recall curve for a single class.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

AVERAGE_RECALL_1000 = _register_metric(
    MetricMetadata(
        name="Average Recall",
        unit="AR@1000",
        description="Class-agnostic average recall over IoU thresholds 0.5 to 0.95 in 0.05 increments, with up to 1000 detections per image.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

COLORFULNESS = _register_metric(
    MetricMetadata(
        name="Colorfulness",
        unit="",
        description="A measure of how varied the colors are in the image.",
        range=(0.0, None),
        float_vs_device_threshold=10.0,
    )
)

DELTA_THRESHOLD_ACCURACY = _register_metric(
    MetricMetadata(
        name="Delta Threshold Accuracy",
        unit="δ1",
        description="The percentage of pixels where the predicted depth is within 25% of the expected.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

KL_DIVERGENCE = _register_metric(
    MetricMetadata(
        name="KL divergence",
        unit="kldiv",
        description="A distance metric between two probability distributions.",
        range=(0.0, None),
        higher_is_better=False,
    )
)

MEAN_ANGULAR_ERROR = _register_metric(
    MetricMetadata(
        name="Mean Angular Error",
        unit="MAE (Degrees)",
        description="Mean angular error between predicted and ground truth gaze directions.",
        range=(0.0, None),
        float_vs_device_threshold=5.0,
        higher_is_better=False,
    )
)

MEAN_AVERAGE_PRECISION = _register_metric(
    MetricMetadata(
        name="Mean Average Precision",
        unit="mAP",
        description="Mean Average Precision (across predicted classes).",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

MEAN_AVERAGE_PRECISION_IOU_5_95 = _register_metric(
    MetricMetadata(
        name="Mean Average Precision",
        unit="mAP@0.5:0.95",
        description="Mean Average Precision averaged over IOU thresholds 0.5 to 0.95 in 0.05 increments.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

MEAN_AVERAGE_PRECISION_IOU_50 = _register_metric(
    MetricMetadata(
        name="Mean Average Precision",
        unit="mAP@0.50",
        description="Mean Average Precision at IoU threshold 0.50.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

MEAN_IOU = _register_metric(
    MetricMetadata(
        name="Mean Intersection Over Union",
        unit="mIOU",
        description="Overlap of predicted and expected segmentation divided by the union size.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

MMLU = _register_metric(
    MetricMetadata(
        name="Massive Multitask Language Understanding",
        unit="MMLU",
        description="Set of multiple choice questions that the model answered correctly.",
        range=(0.0, 1.0),
        float_vs_device_threshold=0.1,
    )
)

MMMU = _register_metric(
    MetricMetadata(
        name="Massive Multi-discipline Multimodal Understanding",
        unit="MMMU",
        description="Multimodal multiple choice questions spanning diverse academic subjects.",
        range=(0.0, 1.0),
        float_vs_device_threshold=0.1,
    )
)

NORMALIZED_MEAN_ERROR = _register_metric(
    MetricMetadata(
        name="Normalized Mean Error",
        unit="NME",
        description="Average error between predictions and ground truth, normalized by a reference scale.",
        range=(0.0, None),
        float_vs_device_threshold=0.01,
        higher_is_better=False,
    )
)

PANOPTIC_QUALITY = _register_metric(
    MetricMetadata(
        name="Panoptic Quality",
        unit="PQ",
        description="A measure of how well all objects in the image were correctly identified and segmented.",
        range=(0.0, 1.0),
        float_vs_device_threshold=0.1,
    )
)

PERCENTAGE_CORRECT_KEYPOINTS = _register_metric(
    MetricMetadata(
        name="Percentage Correct Keypoints (head-normalized)",
        unit="PCKh",
        description="Percentage of keypoints within a certain distance of expected.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

PERPLEXITY = _register_metric(
    MetricMetadata(
        name="Perplexity",
        unit="PPL",
        description="A measure of how likely the model is to predict a given sequence of words.",
        range=(0.0, None),
        higher_is_better=False,
    )
)

PSNR = _register_metric(
    MetricMetadata(
        name="Peak Signal-to-Noise Ratio (PSNR)",
        unit="dB",
        description="A measure of how similar two images are.",
        range=(0.0, None),
        float_vs_device_threshold=10.0,
    )
)

SSIM = _register_metric(
    MetricMetadata(
        name="Structural Similarity",
        unit="SSIM",
        description="A measure of the perceived quality difference between two images.",
        range=(0.0, 1.0),
        float_vs_device_threshold=0.1,
    )
)

WORD_ERROR_RATE = _register_metric(
    MetricMetadata(
        name="Word Error Rate",
        unit="WER",
        description="The percentage of words incorrectly predicted.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
        higher_is_better=False,
    )
)

DIARIZATION_ERROR_RATE = _register_metric(
    MetricMetadata(
        name="Diarization Error Rate",
        unit="DER",
        description="Percentage of reference speech time affected by missed speech, false alarm, or speaker confusion errors.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
        higher_is_better=False,
    )
)

HOMOGRAPHY_ACCURACY = _register_metric(
    MetricMetadata(
        name="Homography Estimation @3px",
        unit="correctness@3px",
        description=(
            "Fraction of HPatches pairs where the RANSAC-estimated homography "
            "has mean corner reprojection error < 3 px."
        ),
        range=(0.0, 100.0),
        float_vs_device_threshold=5.0,
    )
)


def _grace_grade(version: int) -> MetricMetadata:
    """Register the Grace metric for one version of the prompt set + rubric.

    Grace is "Grading Response Accuracy Evaluation". Superseded versions stay
    registered: manifest baselines measured under them keep their own label
    rather than being silently reported as the current one.
    """
    return _register_metric(
        MetricMetadata(
            name=f"Grace{version}",
            unit="%",
            description=(
                f"Grading Response Accuracy Evaluation: percentage score from "
                f"grading generated responses to the Grace{version} prompt set "
                f"with a grader LLM."
            ),
            range=(0.0, 100.0),
            float_vs_device_threshold=10.0,
        )
    )


GRACE1_GRADE = _grace_grade(1)
GRACE2_GRADE = _grace_grade(2)

# The current Grace version
GRACE_GRADE = GRACE2_GRADE

MEAN_ABSOLUTE_ERROR = _register_metric(
    MetricMetadata(
        name="Mean Absolute Error",
        unit="MAE",
        description="Per-pixel mean absolute error between predicted saliency map and binary ground truth mask.",
        range=(0.0, 1.0),
        float_vs_device_threshold=0.01,
        higher_is_better=False,
    )
)

RECALL_AT_1 = _register_metric(
    MetricMetadata(
        name="Recall@1",
        unit="%",
        description="Fraction of queries where the correct result is ranked first (text-to-image direction).",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

PLATE_STRING_ACCURACY = _register_metric(
    MetricMetadata(
        name="Plate String Accuracy",
        unit="%",
        description=(
            "Fraction of images where plate category (characters left of the "
            "emirate logo), plate number (characters right of the logo), and "
            "plate state (emirate) all exactly match ground truth."
        ),
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

MEAN_AVERAGE_PRECISION_5DEG_2CM = _register_metric(
    MetricMetadata(
        name="Mean Average Precision",
        unit="mAP@5deg2cm",
        description="Mean Average Precision at 5-degree 2cm pose accuracy threshold on NOCS REAL275.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

# Freeze the valid pairs set now that all metrics are registered.
VALID_METRIC_PAIRS: frozenset[tuple[str, str]] = frozenset(_VALID_METRIC_PAIRS)

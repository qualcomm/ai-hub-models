# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import RECALL_AT_1, MetricMetadata

# Deviates from the paper's point-cloud-overlap ground truth: a revisit is "close in
# space, far apart in time" -- distance threshold is our own choice (upstream's actual
# positive label is overlap ratio > 0.3, which we don't compute); frame gap of 100
# matches upstream's own ground-truth generation (tools/utils/gen_gt_data.py: `(i-idx) > 100`).
REVISIT_DISTANCE_M = 3.0
REVISIT_FRAME_GAP = 100


class LidarPlaceRecognitionEvaluator(BaseEvaluator):
    """Recall@1 for LiDAR place recognition retrieval.

    For every query scan, the nearest database scan by descriptor distance
    (excluding frames within REVISIT_FRAME_GAP of the query) counts as a hit
    if its pose is within REVISIT_DISTANCE_M of the query's pose. Queries with
    no true revisit anywhere in the database are skipped, matching the
    standard KITTI LiDAR place-recognition evaluation protocol.
    """

    def __init__(self) -> None:
        self.descriptors: list[torch.Tensor] = []
        self.positions: list[torch.Tensor] = []
        self.frame_indices: list[torch.Tensor] = []
        self._recall: float | None = None

    def add_batch(
        self,
        output: torch.Tensor,
        gt: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    ) -> None:
        """Accumulate a batch of descriptors and ground-truth pose/frame-index labels.

        Parameters
        ----------
        output:
            Model output descriptors of shape [N, descriptor_dim].
        gt:
            Tuple of (positions, frame_indices): positions is [N, 3],
            frame_indices is [N].
        """
        assert isinstance(output, torch.Tensor)
        assert isinstance(gt, (tuple, list)) and len(gt) == 2
        positions, frame_indices = gt
        self.descriptors.append(output.detach().cpu())
        self.positions.append(torch.as_tensor(positions).reshape(-1, 3).cpu())
        self.frame_indices.append(torch.as_tensor(frame_indices).reshape(-1).cpu())
        self._recall = None

    def reset(self) -> None:
        self.descriptors = []
        self.positions = []
        self.frame_indices = []
        self._recall = None

    def _compute_recall(self) -> float | None:
        """Recall@1 over all samples added so far, or None if none has a valid revisit yet.

        None (rather than raising) lets evaluate()'s per-batch progress logging call this
        mid-accumulation -- e.g. a batch of scans all within REVISIT_FRAME_GAP of each
        other has no valid query yet, which isn't a real failure until every batch is in.
        """
        if self._recall is not None:
            return self._recall
        if not self.descriptors:
            raise ValueError("No batches were added; cannot compute Recall@1.")

        descriptors = torch.cat(self.descriptors, dim=0)
        positions = torch.cat(self.positions, dim=0)
        frame_indices = torch.cat(self.frame_indices, dim=0)

        descriptor_dists = torch.cdist(descriptors, descriptors)
        pose_dists = torch.cdist(positions, positions)
        frame_gap = (frame_indices.unsqueeze(1) - frame_indices.unsqueeze(0)).abs()
        candidate_mask = frame_gap > REVISIT_FRAME_GAP
        is_revisit = pose_dists < REVISIT_DISTANCE_M

        valid_query_mask = (candidate_mask & is_revisit).any(dim=1)
        masked_dists = descriptor_dists.masked_fill(~candidate_mask, float("inf"))
        nearest = masked_dists.argmin(dim=1)
        hit_mask = (
            is_revisit[torch.arange(descriptors.shape[0]), nearest] & valid_query_mask
        )

        hits = int(hit_mask.sum().item())
        valid_queries = int(valid_query_mask.sum().item())

        if valid_queries == 0:
            return None
        self._recall = 100.0 * hits / valid_queries
        return self._recall

    def get_accuracy_score(self) -> float:
        recall = self._compute_recall()
        if recall is None:
            raise ValueError(
                "No query scan has a true revisit in the sampled database; "
                "cannot compute Recall@1."
            )
        return recall

    def formatted_accuracy(self) -> str:
        recall = self._compute_recall()
        if recall is None:
            return "N/A (no query scan has a true revisit in the samples accumulated so far)"
        return f"{recall:.2f}% (Recall@1)"

    def get_metric_metadata(self) -> MetricMetadata:
        return RECALL_AT_1.with_description(
            "Fraction of query scans whose nearest retrieved database scan "
            f"(excluding frames within {REVISIT_FRAME_GAP} of the query) is a "
            f"true revisit (pose within {REVISIT_DISTANCE_M}m). Computed only "
            "over the frames sampled for a given run, not the full database "
            "-- not directly comparable to the paper's full-sequence 0.906."
        )

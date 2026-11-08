# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import RECALL_AT_1, MetricMetadata

# GPS distance (metres) below which a retrieved image counts as correct.
POS_DIST_THR_M = 25.0


class PlaceRecognitionEvaluator(BaseEvaluator):
    """
    Evaluator for GPS-supervised place recognition using Recall at K.

    Accumulates L2-normalized descriptors from all images (queries and
    database) along with their UTM coordinates and is_query flags.  On
    ``get_accuracy_score``, builds a distance matrix between all query and
    database descriptors, ranks database images per query by descriptor
    distance, and checks whether the nearest database image within the top-K
    ranks is within ``pos_dist_thr_m`` metres of the query's GPS position.
    """

    def __init__(self, pos_dist_thr_m: float = POS_DIST_THR_M) -> None:
        self.pos_dist_thr_m = pos_dist_thr_m
        self._descriptors: list[torch.Tensor] = []
        self._utms: list[np.ndarray] = []
        self._is_query: list[bool] = []
        self._cached: dict[int, float] | None = None

    def add_batch(
        self,
        output: torch.Tensor,
        gt: tuple[np.ndarray, bool] | list,
    ) -> None:
        """
        Parameters
        ----------
        output
            Model descriptor tensor, shape (N, descriptor_dim).
        gt
            Tuple of (utm_coords, is_query) where utm_coords is (N, 2) and
            is_query is a bool tensor or list of length N.
        """
        utm_batch, is_query_batch = gt[0], gt[1]
        if isinstance(output, torch.Tensor):
            self._descriptors.append(output.detach().cpu())
        if isinstance(utm_batch, torch.Tensor):
            utm_batch = utm_batch.numpy()
        if isinstance(is_query_batch, torch.Tensor):
            is_query_batch = is_query_batch.tolist()
        # utm_batch may be (N, 2) for a batch or (2,) for a single sample.
        utm_arr = np.atleast_2d(utm_batch)
        is_q = (
            [bool(is_query_batch)]
            if isinstance(is_query_batch, (bool, int, float))
            else [bool(v) for v in is_query_batch]
        )
        for i in range(utm_arr.shape[0]):
            self._utms.append(utm_arr[i])
            self._is_query.append(is_q[i] if i < len(is_q) else is_q[0])
        self._cached = None

    def reset(self) -> None:
        self._descriptors = []
        self._utms = []
        self._is_query = []
        self._cached = None

    def _compute_recall(self) -> dict[int, float]:
        if self._cached is not None:
            return self._cached
        if not self._descriptors:
            self._cached = {1: 0.0, 5: 0.0, 10: 0.0}
            return self._cached

        all_desc = torch.cat(self._descriptors, dim=0)  # (N, D)
        is_query = np.array(self._is_query, dtype=bool)
        utms = np.stack(self._utms)  # (N, 2)

        q_desc = all_desc[is_query]  # (Nq, D)
        db_desc = all_desc[~is_query]  # (Ndb, D)
        utm_q = utms[is_query]  # (Nq, 2)
        utm_db = utms[~is_query]  # (Ndb, 2)

        if q_desc.shape[0] == 0 or db_desc.shape[0] == 0:
            self._cached = {1: 0.0, 5: 0.0, 10: 0.0}
            return self._cached

        # Descriptor distance matrix: (Nq, Ndb)
        sim = torch.mm(q_desc, db_desc.t())  # cosine sim (descriptors are L2-normed)
        ranks = torch.argsort(sim, dim=1, descending=True).numpy()  # (Nq, Ndb)

        # GPS distance matrix: (Nq, Ndb)
        diff = utm_q[:, None, :] - utm_db[None, :, :]  # (Nq, Ndb, 2)
        gps_dist = np.linalg.norm(diff, axis=2)  # (Nq, Ndb)

        recalls: dict[int, float] = {}
        for k in (1, 5, 10):
            top_k = ranks[:, :k]  # (Nq, k)
            # A query is recalled if any of its top-k retrievals is GPS-close.
            hit = np.any(
                gps_dist[np.arange(q_desc.shape[0])[:, None], top_k]
                <= self.pos_dist_thr_m,
                axis=1,
            )
            recalls[k] = float(hit.mean()) * 100.0

        self._cached = recalls
        return self._cached

    def get_accuracy_score(self) -> float:
        return self._compute_recall()[1]

    def formatted_accuracy(self) -> str:
        r = self._compute_recall()
        return f"R@1={r[1]:.1f}%  R@5={r[5]:.1f}%  R@10={r[10]:.1f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return RECALL_AT_1.with_description(
            "Fraction of queries whose top-1 retrieved database image is within "
            f"{self.pos_dist_thr_m} m GPS distance."
        )

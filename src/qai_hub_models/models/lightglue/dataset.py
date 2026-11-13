# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models.lightglue.postprocess import (
    MAX_KEYPOINTS,
    extract_padded_features,
)
from qai_hub_models.models.superpoint.dataset import HPatchesDataset
from qai_hub_models.models.superpoint.model import SuperPoint
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec


class LightGluePairDataset(BaseDataset):
    """HPatches image pairs, pre-processed into SuperPoint keypoints/descriptors.

    LightGlue is a matcher, not a feature extractor: it consumes keypoints
    and descriptors already produced by an extractor for two images. This
    dataset reuses the already-onboarded, MIT-licensed `superpoint` model
    (never LightGlue's own bundled extractor, which carries a separate,
    non-open Magic Leap license) to produce those features from HPatches
    image pairs, so LightGlue's own forward pass only ever sees plain
    keypoints/descriptors/num_keypoints tensors -- never an image or a
    reference to the restricted extractor code.

    Each item is one HPatches (reference, target) pair, as already paired up
    by the wrapped `HPatchesDataset`.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        input_spec: InputSpec | None = None,
    ) -> None:
        self._images = HPatchesDataset(split=split)
        self._extractor = SuperPoint.from_pretrained()
        super().__init__(self._images.dataset_path, split, input_spec)

    def _download_data(self) -> None:
        # HPatchesDataset (constructed above, before super().__init__) already
        # downloaded and validated its own data.
        pass

    @classmethod
    def dataset_name(cls) -> str:
        return "hpatches_superpoint_pairs"

    @staticmethod
    def default_samples_per_job() -> int:
        return 50

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return HPatchesDataset.get_dataset_metadata()

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(
        self, item: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Extract SuperPoint features for one HPatches (ref, tgt) pair.

        Parameters
        ----------
        item
            Pair index in [0, len(self)).

        Returns
        -------
        inputs : tuple[torch.Tensor, ...]
            (keypoints0, descriptors0, num_keypoints0,
            keypoints1, descriptors1, num_keypoints1) -- matches
            LightGlue.forward's argument order exactly.
        gt : tuple[torch.Tensor, ...]
            (H_gt, keypoints0, keypoints1, num_keypoints0, num_keypoints1).
            The evaluator only ever sees LightGlue's own match
            indices/scores (its outputs) plus this ground truth, so the
            keypoint coordinates the match indices refer back into must
            travel alongside H_gt here.
        """
        (image_a, image_b), h_gt = self._images[item]
        kp0, desc0, n0, kp1, desc1, n1 = extract_padded_features(
            self._extractor, image_a[None], image_b[None], MAX_KEYPOINTS
        )
        inputs = (kp0[0], desc0[0], n0[0], kp1[0], desc1[0], n1[0])
        gt = (h_gt, kp0[0], kp1[0], n0[0], n1[0])
        return inputs, gt

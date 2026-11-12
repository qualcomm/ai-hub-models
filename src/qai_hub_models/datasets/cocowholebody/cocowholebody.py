# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import torch

from qai_hub_models.datasets.coco.cocobody import CocoBodyDatasetBase
from qai_hub_models.extern.mmengine import patch_mmengine_pkgresources
from qai_hub_models.extern.mmpose import patch_mmpose_no_build_deps
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

with patch_mmpose_no_build_deps(), patch_mmengine_pkgresources():
    from mmpose.apis import MMPoseInferencer
    from mmpose.apis.inferencers import Pose3DInferencer
    from mmpose.apis.inferencers.base_mmpose_inferencer import BaseMMPoseInferencer

DEFAULT_INFERENCER_ARCH = "rtmpose-m_8xb64-270e_coco-wholebody-256x192"


def _set_mmpose_base_inferencer_show_progress(
    inferencer: BaseMMPoseInferencer, show_progress: bool = False
) -> BaseMMPoseInferencer:
    inferencer.show_progress = show_progress
    if hasattr(inferencer, "detector") and inferencer.detector is not None:
        inferencer.detector.show_progress = show_progress
    if isinstance(inferencer, Pose3DInferencer):
        _set_mmpose_base_inferencer_show_progress(
            inferencer.pose2d_model, show_progress
        )
    return inferencer


def _set_mmpose_inferencer_show_progress(
    inferencer: MMPoseInferencer, show_progress: bool = False
) -> MMPoseInferencer:
    """
    Disables verbose logging output when the inferencer object runs the model.
    Returns the input inferencer. The object is modified in-place.
    """
    _set_mmpose_base_inferencer_show_progress(inferencer, show_progress)
    _set_mmpose_base_inferencer_show_progress(inferencer.inferencer, show_progress)
    return inferencer


class CocoWholeBodyDataset(CocoBodyDatasetBase):
    """
    Wrapper class around CocoWholeBody Human Pose dataset
    http://images.cocodataset.org/
    COCO-WholeBody keypoints::
        0-16: 17 body keypoints,
        17-22: 6 foot keypoints,
        23-90: 68 face keypoints,
        91-132: 42 hand keypoints
        In total, 133 keypoints for wholebody pose estimation.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        super().__init__(split, input_spec)
        self.inference = MMPoseInferencer(DEFAULT_INFERENCER_ARCH, device="cpu")
        _set_mmpose_inferencer_show_progress(self.inference, False)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, tuple[int, int, torch.Tensor, torch.Tensor]]:
        """
        Get an item in this dataset.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            Input image resized for the network. RGB, floating point range [0-1].

        ground_truth : tuple[int, int, torch.Tensor, torch.Tensor]
            image_id
                Image ID within the original dataset
            category_id
                Ground truth prediction category ID.
            scale
                Bounding box scaling factor, with shape(2,) -- (x, y).
            bbox
                Bounding box coordinates (xyxy format)
        """
        (
            file_name,
            image_id,
            category_id,
            _center,
            scale,
        ) = self.kpt_db[index]
        img_path = self.image_dir / file_name
        data_numpy = cv2.imread(
            str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)  # type: ignore[arg-type,unused-ignore] # This ignore is applicable only for a single model's environment
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(data_numpy)
        inputs = self.inference.preprocess(NHWC_int_numpy_frames, batch_size=1)
        proc_inputs, _ = next(iter(inputs))
        proc_inputs_ = proc_inputs["inputs"][0]
        image = proc_inputs_.to(dtype=torch.float32)
        bbox = proc_inputs["data_samples"][0].gt_instances.bboxes[0]
        scale = proc_inputs["data_samples"][0].gt_instances.bbox_scales[0]
        return image, (image_id, category_id, scale, bbox)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 500

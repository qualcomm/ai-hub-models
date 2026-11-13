# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.datasets import instantiate_dataset
from qai_hub_models.models.mediapipe_pose.model import (
    DETECT_DSCALE,
    DETECT_DXY,
    DETECT_SCORE_CLIPPING_THRESHOLD,
    DRAW_POSE_KEYPOINT_INDICES,
    FILTER_OOB_BOX,
    POSE_KEYPOINT_INDEX_END,
    POSE_KEYPOINT_INDEX_START,
    POSE_LANDMARK_CONNECTIONS,
    ROTATION_VECTOR_OFFSET_RADS,
    MediaPipePose,
)
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    compute_box_corners_with_rotation,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    compute_vector_rotation,
    torch_image_to_numpy,
)
from qai_hub_models.models.templates.mediapipe.app import MediaPipeApp
from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    WorkbenchModelCollection,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec


class MediaPipePoseApp(MediaPipeApp):
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with MediaPipe's pose landmark detector.

    The app uses 2 models:
        * MediaPipePoseDetector
        * MediaPipePoseLandmark

    See the class comment for the parent class for details.
    """

    def __init__(
        self,
        pose_detector: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor],
        ],
        pose_landmark_detector: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        pose_detector_includes_postprocessing: bool,
        anchors: torch.Tensor,
        pose_detector_input_spec: InputSpec,
        landmark_detector_input_spec: InputSpec,
        min_detector_pose_box_score: float = 0.75,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
    ) -> None:
        """
        Construct a mediapipe pose application.

        Parameters
        ----------
        pose_detector
            Pose detection model callable.
        pose_landmark_detector
            Pose landmark detection model callable.
        pose_detector_includes_postprocessing
            Whether the pose detector includes postprocessing.
        anchors
            Detector anchors.
        pose_detector_input_spec
            Input spec for pose detector.
        landmark_detector_input_spec
            Input spec for landmark detector.
        min_detector_pose_box_score
            Minimum score threshold for pose box detection.
        nms_iou_threshold
            IoU threshold for non-maximum suppression.
        min_landmark_score
            Minimum score threshold for landmark detection.
        """

        def unified_pose_detector(
            inp: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Combines the pose detector's four output tensors into two unified tensors
            and does the concat outside of the model for optimization purposes.

            Parameters
            ----------
            inp
                Input image tensor to the pose detector model.

            Returns
            -------
            box_coords : torch.Tensor
                Bounding box coordinates with shape [1, 896, 12]
                (batch_size, num_anchors, 12_coordinates_per_anchor).
            box_scores : torch.Tensor
                Confidence scores with shape [1, 896, 1]
                (batch_size, num_anchors, 1_score_per_anchor).
            """
            box_coords1, box_coords2, box_scores1, box_scores2 = pose_detector(inp)  # type: ignore[misc]
            box_coords = torch.cat([box_coords1, box_coords2], dim=1)
            box_scores = torch.cat([box_scores1, box_scores2], dim=1)
            return box_coords, box_scores

        super().__init__(
            unified_pose_detector
            if not pose_detector_includes_postprocessing
            else pose_detector,  # type: ignore[arg-type]
            anchors,
            pose_detector_includes_postprocessing,
            pose_landmark_detector,
            cast(
                tuple[int, int],
                pose_detector_input_spec["image"][0][-2:],
            ),
            cast(
                tuple[int, int],
                landmark_detector_input_spec["image"][0][-2:],
            ),
            POSE_KEYPOINT_INDEX_START,
            POSE_KEYPOINT_INDEX_END,
            ROTATION_VECTOR_OFFSET_RADS,
            DETECT_DXY,
            DETECT_DSCALE,
            min_detector_pose_box_score,
            DETECT_SCORE_CLIPPING_THRESHOLD,
            nms_iou_threshold,
            min_landmark_score,
            POSE_LANDMARK_CONNECTIONS,
            DRAW_POSE_KEYPOINT_INDICES,
            FILTER_OOB_BOX,
        )

    def _compute_object_roi(
        self,
        batched_selected_boxes: list[torch.Tensor],
        batched_selected_keypoints: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        See parent function for base functionality and parameter documentation.

        The MediaPipe pose pipeline computes the ROI not from the detector bounding box,
        but from specific detected keypoints. This override implements that behavior.
        """
        batched_selected_roi: list[torch.Tensor] = []
        for boxes, keypoints in zip(
            batched_selected_boxes, batched_selected_keypoints, strict=False
        ):
            if boxes.nelement() == 0 or keypoints.nelement() == 0:
                batched_selected_roi.append(torch.Tensor())
                continue

            # Compute bounding box center and rotation
            theta = compute_vector_rotation(
                keypoints[:, self.keypoint_rotation_vec_start_idx, ...],
                keypoints[:, self.keypoint_rotation_vec_end_idx, ...],
                self.rotation_offset_rads,
            )
            xc = keypoints[..., self.keypoint_rotation_vec_start_idx, 0]
            yc = keypoints[..., self.keypoint_rotation_vec_start_idx, 1]
            x1 = keypoints[..., self.keypoint_rotation_vec_end_idx, 0]
            y1 = keypoints[..., self.keypoint_rotation_vec_end_idx, 1]

            # Square box always
            w = ((xc - x1) ** 2 + (yc - y1) ** 2).sqrt() * 2 * self.detect_box_scale
            h = w

            # Compute box corners from box center, width, height
            batched_selected_roi.append(
                compute_box_corners_with_rotation(xc, yc, w, h, theta)
            )

        return batched_selected_roi

    @classmethod
    def from_pretrained(cls, model: CollectionModel) -> MediaPipePoseApp:
        assert isinstance(model, MediaPipePose)
        return cls(
            model.pose_detector,
            model.pose_landmark_detector,
            model.pose_detector.include_postprocessing,
            model.pose_detector.anchors,
            model.pose_detector.get_input_spec(),
            model.pose_landmark_detector.get_input_spec(),
        )

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        assert isinstance(collection_model, MediaPipePose)

        det_spec = (
            input_specs.get("pose_detector") if input_specs else None
        ) or collection_model.pose_detector.get_input_spec()
        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=det_spec,
        )
        num_samples = num_samples or dataset.default_samples_per_job()

        if component_name == "pose_detector":
            entries: dict[str, list[np.ndarray]] = {"image": []}
            for i in range(min(num_samples, len(dataset))):
                image_tensor, _ = dataset[i]
                entries["image"].append(image_tensor.unsqueeze(0).numpy())
            return entries

        if component_name == "pose_landmark_detector":
            app = cls.from_pretrained(collection_model)
            entries = {"image": []}
            collected = 0
            for i in range(len(dataset)):
                if collected >= num_samples:
                    break
                image_tensor, _ = dataset[i]
                NCHW = image_tensor.unsqueeze(0)
                raw = app.predict_landmarks_from_image(NCHW, raw_output=True)
                _, _, batched_roi_4corners = raw[:3]
                if not batched_roi_4corners or batched_roi_4corners[0].numel() == 0:
                    continue
                cropped = app.crop_landmark_inputs(
                    torch_image_to_numpy(NCHW), batched_roi_4corners[0]
                )
                for j in range(cropped.shape[0]):
                    if collected >= num_samples:
                        break
                    entries["image"].append(cropped[j : j + 1].numpy())
                    collected += 1
            return entries

        raise ValueError(f"Unknown component: {component_name}")

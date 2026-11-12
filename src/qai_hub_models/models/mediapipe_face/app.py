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
from qai_hub_models.models.mediapipe_face.model import (
    DETECT_DSCALE,
    DETECT_DXY,
    DETECT_SCORE_CLIPPING_THRESHOLD,
    FACE_LANDMARK_CONNECTIONS,
    LEFT_EYE_KEYPOINT_INDEX,
    RIGHT_EYE_KEYPOINT_INDEX,
    ROTATION_VECTOR_OFFSET_RADS,
    MediaPipeFace,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    torch_image_to_numpy,
)
from qai_hub_models.models.templates.mediapipe.app import MediaPipeApp
from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    WorkbenchModelCollection,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec


class MediaPipeFaceApp(MediaPipeApp):
    """
    MediaPipe Face application that wires model-specific behavior into the
    shared MediaPipeApp base.

    The face detector returns two sets of coordinates and scores; this class
    adapts those outputs to the base class interface and delegates all generic
    pre/post-processing, ROI computation, landmark mapping, and drawing to the base class.
    """

    def __init__(
        self,
        face_detector: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor],
        ],
        face_landmark_detector: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        face_detector_includes_postprocessing: bool,
        anchors: torch.Tensor,
        face_detector_input_spec: InputSpec,
        landmark_detector_input_spec: InputSpec,
        min_detector_face_box_score: float = 0.8,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
    ) -> None:
        """
        Construct a MediaPipe Face application.

        Parameters
        ----------
        face_detector
            Multi-head face detector returning (coords1, coords2, scores1, scores2).
        face_landmark_detector
            Landmark detector returning (scores, landmarks).
        face_detector_includes_postprocessing
            Whether the face detector includes postprocessing.
        anchors
            Detector anchors.
        face_detector_input_spec
            Input spec for detector (to derive input dims).
        landmark_detector_input_spec
            Input spec for landmark detector (to derive input dims).
        min_detector_face_box_score
            Minimum score threshold for face box detection.
        nms_iou_threshold
            IoU threshold for non-maximum suppression.
        min_landmark_score
            Minimum score threshold for landmark detection.
        """
        detector_input_dims = cast(
            tuple[int, int], face_detector_input_spec["image"][0][-2:]
        )
        landmark_input_dims = cast(
            tuple[int, int], landmark_detector_input_spec["image"][0][-2:]
        )

        def unified_face_detector(
            inp: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Combines the face detector's four output tensors into two unified tensors
            and does the concat outside of the model for optimization purposes.

            Parameters
            ----------
            inp
                Input image tensor to the face detector model.

            Returns
            -------
            box_coords : torch.Tensor
                Bounding box coordinates with shape [1, 896, 12]
                (batch_size, num_anchors, 12_coordinates_per_anchor).
            box_scores : torch.Tensor
                Confidence scores with shape [1, 896, 1]
                (batch_size, num_anchors, 1_score_per_anchor).
            """
            box_coords1, box_coords2, box_scores1, box_scores2 = face_detector(inp)  # type: ignore[misc]
            box_coords = torch.cat([box_coords1, box_coords2], dim=1)
            box_scores = torch.cat([box_scores1, box_scores2], dim=1)
            return box_coords, box_scores

        super().__init__(
            detector=unified_face_detector
            if not face_detector_includes_postprocessing
            else face_detector,  # type: ignore[arg-type]
            detector_anchors=anchors,
            detector_includes_postprocessing=face_detector_includes_postprocessing,
            landmark_detector=face_landmark_detector,
            detector_input_dims=detector_input_dims,
            landmark_input_dims=landmark_input_dims,
            keypoint_rotation_vec_start_idx=RIGHT_EYE_KEYPOINT_INDEX,
            keypoint_rotation_vec_end_idx=LEFT_EYE_KEYPOINT_INDEX,
            rotation_offset_rads=ROTATION_VECTOR_OFFSET_RADS,
            detect_box_offset_xy=DETECT_DXY,
            detect_box_scale=DETECT_DSCALE,
            min_detector_box_score=min_detector_face_box_score,
            detector_score_clipping_threshold=DETECT_SCORE_CLIPPING_THRESHOLD,
            nms_iou_threshold=nms_iou_threshold,
            min_landmark_score=min_landmark_score,
            landmark_connections=FACE_LANDMARK_CONNECTIONS,
        )

    @classmethod
    def from_pretrained(cls, model: CollectionModel) -> MediaPipeFaceApp:
        assert isinstance(model, MediaPipeFace)
        return cls(
            model.face_detector,
            model.face_landmark_detector,
            model.face_detector.include_postprocessing,
            model.face_detector.anchors,
            model.face_detector.get_input_spec(),
            model.face_landmark_detector.get_input_spec(),
        )

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        assert isinstance(collection_model, MediaPipeFace)

        det_spec = (
            input_specs.get("face_detector") if input_specs else None
        ) or collection_model.face_detector.get_input_spec()
        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=det_spec,
        )
        num_samples = num_samples or dataset.default_samples_per_job()

        if component_name == "face_detector":
            entries: dict[str, list[np.ndarray]] = {"image": []}
            for i in range(min(num_samples, len(dataset))):
                image_tensor, _ = dataset[i]
                entries["image"].append(image_tensor.unsqueeze(0).numpy())
            return entries

        if component_name == "face_landmark_detector":
            # Run the detector on each image to get face ROIs,
            # then crop and resize to produce proper landmark inputs.
            app = cls.from_pretrained(collection_model)
            entries = {"image": []}
            collected = 0
            for i in range(len(dataset)):
                if collected >= num_samples:
                    break
                image_tensor, _ = dataset[i]
                NCHW = image_tensor.unsqueeze(0)

                # Detect faces
                raw = app.predict_landmarks_from_image(NCHW, raw_output=True)
                _, _, batched_roi_4corners = raw[:3]
                if not batched_roi_4corners or batched_roi_4corners[0].numel() == 0:
                    continue

                # Crop using the shared helper
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

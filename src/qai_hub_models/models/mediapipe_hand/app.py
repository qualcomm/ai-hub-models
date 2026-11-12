# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import cv2
import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models.mediapipe_hand.model import (
    DETECT_DSCALE,
    DETECT_DXY,
    DETECT_SCORE_CLIPPING_THRESHOLD,
    HAND_LANDMARK_CONNECTIONS,
    MIDDLE_FINDER_KEYPOINT_INDEX,
    ROTATION_VECTOR_OFFSET_RADS,
    WRIST_CENTER_KEYPOINT_INDEX,
)
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    compute_box_affine_crop_resize_matrix,
)
from qai_hub_models.models.templates.corevision.draw import (
    draw_connections,
    draw_points,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    apply_affine_to_coordinates,
    apply_batched_affines_to_frame,
    numpy_image_to_torch,
)
from qai_hub_models.models.templates.mediapipe.app import MediaPipeApp
from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.input_spec import InputSpec


class MediaPipeHandApp(MediaPipeApp):
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with MediaPipe's hand landmark detector.

    The app uses 2 models:
        * MediaPipeHandDetector
        * MediaPipeHandLandmark

    See the class comment for the parent class for details.
    """

    def __init__(
        self,
        hand_detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        hand_landmark_detector: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        anchors: torch.Tensor,
        detector_includes_postprocessing: bool,
        hand_detector_input_spec: InputSpec,
        landmark_detector_input_spec: InputSpec,
        min_detector_hand_box_score: float = 0.95,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
    ) -> None:
        """
        Construct a mediapipe hand application.

        Parameters
        ----------
        hand_detector
            Hand detection model callable.
        hand_landmark_detector
            Hand landmark detection model callable.
        anchors
            Detector anchors.
        detector_includes_postprocessing
            Whether the detector includes postprocessing.
        hand_detector_input_spec
            Input spec for hand detector.
        landmark_detector_input_spec
            Input spec for landmark detector.
        min_detector_hand_box_score
            Minimum score threshold for hand box detection.
        nms_iou_threshold
            IoU threshold for non-maximum suppression.
        min_landmark_score
            Minimum score threshold for landmark detection.
        """
        super().__init__(
            hand_detector,
            anchors,
            detector_includes_postprocessing,
            hand_landmark_detector,
            cast(
                tuple[int, int],
                hand_detector_input_spec["image"][0][-2:],
            ),
            cast(
                tuple[int, int],
                landmark_detector_input_spec["image"][0][-2:],
            ),
            WRIST_CENTER_KEYPOINT_INDEX,
            MIDDLE_FINDER_KEYPOINT_INDEX,
            ROTATION_VECTOR_OFFSET_RADS,
            DETECT_DXY,
            DETECT_DSCALE,
            min_detector_hand_box_score,
            DETECT_SCORE_CLIPPING_THRESHOLD,
            nms_iou_threshold,
            min_landmark_score,
            HAND_LANDMARK_CONNECTIONS,
        )

    def predict_landmarks_from_image(  # type: ignore[override]
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        raw_output: bool = False,
    ) -> (
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[list[bool]],
        ]
        | list[np.ndarray]
    ):
        """
        From the provided image or tensor, predict the bounding boxes & classes of the hand detected within.

        Parameters
        ----------
        pixel_values_or_image
            See parent function documentation.
        raw_output
            See parent function documentation.

        Returns
        -------
        result : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[list[bool]]] | list[np.ndarray]
            See parent function documentation for generic return values.

            If raw_output is false, returns an additional output:

            batched_is_right_hand
                Whether each landmark represents a right (True) or left (False) hand.
                Organized like the following:
                [
                    # Batch 0 (for Input Image 0)
                    [
                        True (for Selected Landmark 1)
                        False (Selected Landmark 2)
                        ...
                    ]
                    # Batch 1 (for Input Image 1)
                    [] # (this image has no detected palm)
                    ...
                ]
        """
        return super().predict_landmarks_from_image(pixel_values_or_image, raw_output)  # type: ignore[return-value]

    def _draw_predictions(  # type: ignore[override]
        self,
        NHWC_int_numpy_frames: list[np.ndarray],
        batched_selected_boxes: list[torch.Tensor],
        batched_selected_keypoints: list[torch.Tensor],
        batched_roi_4corners: list[torch.Tensor],
        batched_selected_landmarks: list[torch.Tensor],
        batched_is_right_hand: list[list[bool]],
    ) -> None:
        """
        Override of mediapipe::app.py::MediaPipeApp::_draw_predictions.
        Also draws whether the detection is a right or left hand.

        Parameters
        ----------
        NHWC_int_numpy_frames
            List of numpy arrays of shape (H W C x uint8) -- RGB channel layout
            Length of list is # of batches (the number of input images)

        batched_selected_boxes
            Selected object bounding box coordinates.
            Empty tensor if batch had no bounding boxes with a score above the threshold.
            Shape of each list element is [num_selected_boxes, 2, 2].
                Layout is
                    [[box_x1, box_y1],
                        [box_x2, box_y2]]

        batched_selected_keypoints
            Selected object bounding box keypoints.
            Empty tensor if batch had no bounding boxes with a score above the threshold.
            Shape of each list element is [num_selected_boxes, # of keypoints, 2].
                Layout is
                    [[keypoint_0_x, keypoint_0_y],
                        ...,
                        [keypoint_max_x, keypoint_max_y]]

        batched_roi_4corners
            Selected object "region of interest" (region used as input to the landmark detector) corner coordinates.
            Empty tensor if batch had no bounding boxes with a score above the threshold.
            Shape of each list element is [num_selected_boxes, 4, 2], where 2 == (x, y)
            The order of points is (top left point, bottom left point, top right point, bottom right point)

        batched_selected_landmarks
            Selected landmarks. Organized like the following:
            [
                # Batch 0 (for Input Image 0)
                torch.Tensor([
                    Selected Landmark 1 w/ shape (# of landmark points, 3)
                    Selected Landmark 2 w/ shape (# of landmark points, 3)
                    ...
                ]),
                # Batch 1 (for Input Image 1)
                torch.Tensor([]) # (this image has no detected object)
                ...
            ]
            The shape of each inner list element is [# of landmark points, 3],
            where 3 == (X, Y, Conf)

        batched_is_right_hand
            True if the detection is a right hand, False if it's a left hand.
            None if no hand detected.

        Notes
        -----
        Drawing is done on input frame.
        """
        for batch_idx in range(len(NHWC_int_numpy_frames)):
            image = NHWC_int_numpy_frames[batch_idx]
            ld = batched_selected_landmarks[batch_idx]
            box = batched_selected_boxes[batch_idx]
            kp = batched_selected_keypoints[batch_idx]
            roi_4corners = batched_roi_4corners[batch_idx]
            irh = batched_is_right_hand[batch_idx]

            if (
                box.nelement() != 0
                and kp.nelement() != 0
                and roi_4corners.nelement() != 0
            ):
                self._draw_box_and_roi(image, box, kp, roi_4corners)
            if ld.nelement() != 0 and len(irh) != 0:
                self._draw_landmarks(image, ld, irh)

    def _draw_landmarks(  # type: ignore[override]
        self,
        NHWC_int_numpy_frame: np.ndarray,
        landmarks: torch.Tensor,
        is_right_hand: list[bool],
    ) -> None:
        """
        Override of mediapipe::app.py::MediaPipeApp::draw_landmarks
        Also draws whether the detection is a right or left hand.
        """
        for ldm, irh in zip(landmarks, is_right_hand, strict=False):
            # Draw landmark points
            draw_points(NHWC_int_numpy_frame, ldm[:, :2], (0, 255, 0))
            # Draw connections between landmark points
            if self.landmark_connections:
                draw_connections(
                    NHWC_int_numpy_frame,
                    ldm[:, :2],
                    self.landmark_connections,
                    (255 if irh else 0, 0, 0 if irh else 255),
                    2,
                )

    def _run_landmark_detector(  # type: ignore[override]
        self,
        NHWC_int_numpy_frames: list[np.ndarray],
        batched_roi_4corners: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[list[bool]]]:
        """
        Override of mediapipe::app.py::MediaPipeApp::run_landmark_detector
        Additionally returns whether the detection is a right or left hand.
        """
        # selected landmarks for the ROI (if any)
        # list[torch.Tensor(shape=[Num Selected Landmarks, K, 3])],
        # where K == number of landmark keypoints, 3 == (x, y, confidence)
        #
        # A list element will be None if there is no ROI.
        batched_selected_landmarks: list[torch.Tensor] = []

        # whether the selected landmarks for the ROI (if applicable) are for a left or right hand
        #
        # A list element will be None if there is no ROI.
        batched_is_right_hand: list[list[bool]] = []

        # For each input image...
        for batch_idx, roi_4corners in enumerate(batched_roi_4corners):
            if roi_4corners.nelement() == 0:
                continue
            affines = compute_box_affine_crop_resize_matrix(
                roi_4corners[:, :3], self.landmark_input_dims
            )

            # Create input images by applying the affine transforms.
            keypoint_net_inputs = numpy_image_to_torch(
                apply_batched_affines_to_frame(
                    NHWC_int_numpy_frames[batch_idx], affines, self.landmark_input_dims
                )
            )

            # Compute hand landmarks.
            ld_scores, lr, landmarks = self.landmark_detector(keypoint_net_inputs)

            # Convert [0-1] ranged values of landmarks to integer pixel space.
            landmarks[:, :, 0] *= self.landmark_input_dims[0]
            landmarks[:, :, 1] *= self.landmark_input_dims[1]

            # 1 landmark is predicted for each ROI of each input image.
            # For each region of interest & associated predicted landmarks...
            all_landmarks = []
            all_lr = []
            for ld_batch_idx in range(landmarks.shape[0]):
                # Exclude landmarks that don't meet the appropriate score threshold.
                if ld_scores[ld_batch_idx] >= self.min_detector_box_score:
                    # Apply the inverse of affine transform used above to the landmark coordinates.
                    # This will convert the coordinates to their locations in the original input image.
                    inverted_affine = torch.from_numpy(
                        cv2.invertAffineTransform(affines[ld_batch_idx])
                    ).float()
                    landmarks[ld_batch_idx][:, :2] = apply_affine_to_coordinates(
                        landmarks[ld_batch_idx][:, :2], inverted_affine
                    )

                    # Add the predicted landmarks to our list.
                    all_landmarks.append(landmarks[ld_batch_idx])
                    all_lr.append(torch.round(lr[ld_batch_idx]).item() == 1)

            # Add this batch of landmarks to the output list.
            batched_selected_landmarks.append(
                torch.stack(all_landmarks, dim=0) if all_landmarks else torch.Tensor()
            )
            batched_is_right_hand.append(all_lr)
        # Add None for these lists, since this batch has no predicted bounding boxes.
        batched_selected_landmarks.append(torch.Tensor())
        batched_is_right_hand.append([])

        return (batched_selected_landmarks, batched_is_right_hand)

    @classmethod
    def from_pretrained(cls, model: CollectionModel) -> MediaPipeHandApp:
        from qai_hub_models.models.mediapipe_hand.model import MediaPipeHand

        assert isinstance(model, MediaPipeHand)
        return cls(
            model.hand_detector,
            model.hand_landmark_detector,
            model.hand_detector.anchors,
            model.hand_detector.include_postprocessing,
            model.hand_detector.get_input_spec(),
            model.hand_landmark_detector.get_input_spec(),
        )

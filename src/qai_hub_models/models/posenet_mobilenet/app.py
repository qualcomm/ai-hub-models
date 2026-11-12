# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from qai_hub_models.models.templates.corevision.draw import draw_points
from qai_hub_models.models.templates.corevision.image_processing import (
    pil_resize_pad,
    pil_undo_resize_pad,
)
from qai_hub_models.utils.input_spec import InputSpec

# Most code here is from the source repo https://github.com/rwightman/posenet-pytorch

PART_NAMES = [
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
]

NUM_KEYPOINTS = len(PART_NAMES)

PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}
LOCAL_MAXIMUM_RADIUS = 1

POSE_CHAIN = [
    ("nose", "leftEye"),
    ("leftEye", "leftEar"),
    ("nose", "rightEye"),
    ("rightEye", "rightEar"),
    ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"),
    ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"),
    ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
]

PARENT_CHILD_TUPLES = [
    (PART_IDS[parent], PART_IDS[child]) for parent, child in POSE_CHAIN
]
CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"),
    ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"),
    ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"),
    ("leftHip", "rightHip"),
]

CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]
OUTPUT_STRIDE = 16


def traverse_to_targ_keypoint(
    edge_id: int,
    source_keypoint: np.ndarray,
    target_keypoint_id: int,
    scores: np.ndarray,
    offsets: np.ndarray,
    displacements: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Given a source keypoint and target_keypoint_id,
        predict the score and coordinates of the target keypoint.

    Parameters
    ----------
    edge_id
        Index of the edge being considered.
        Equivalent to the index in `POSE_CHAIN`.
    source_keypoint
        (y, x) coordinates of the keypoint.
    target_keypoint_id
        Which body part type of the 17 this keypoint is.
    scores
        See `decode_multiple_poses`.
    offsets
        See `decode_multiple_poses`.
    displacements
        See `decode_multiple_poses`.

    Returns
    -------
    score : float
        Target keypoint score.
    image_coord : np.ndarray
        Target keypoint coordinates.
    """
    height = scores.shape[1]
    width = scores.shape[2]

    source_keypoint_indices = np.clip(
        np.round(source_keypoint / OUTPUT_STRIDE),
        a_min=0,
        a_max=[height - 1, width - 1],
    ).astype(np.int32)

    displaced_point = (
        source_keypoint
        + displacements[edge_id, source_keypoint_indices[0], source_keypoint_indices[1]]
    )

    displaced_point_indices = np.clip(
        np.round(displaced_point / OUTPUT_STRIDE),
        a_min=0,
        a_max=[height - 1, width - 1],
    ).astype(np.int32)

    score = scores[
        target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]
    ]

    image_coord = (
        displaced_point_indices * OUTPUT_STRIDE
        + offsets[
            target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]
        ]
    )

    return score, image_coord


def decode_pose(
    root_score: float,
    root_id: int,
    root_image_coord: np.ndarray,
    scores: np.ndarray,
    offsets: np.ndarray,
    displacements_fwd: np.ndarray,
    displacements_bwd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get all keypoint predictions for a pose given a root keypoint with a high score.

    Parameters
    ----------
    root_score
        The confidence score of the root keypoint.
    root_id
        Which body part type of the 17 this keypoint is.
    root_image_coord
        (y, x) coordinates of the keypoint.
    scores
        See `decode_multiple_poses`.
    offsets
        See `decode_multiple_poses`.
    displacements_fwd
        See `decode_multiple_poses`.
    displacements_bwd
        See `decode_multiple_poses`.

    Returns
    -------
    instance_keypoint_scores : np.ndarray
        List of keypoint scores.
    instance_keypoint_coords : np.ndarray
        List of keypoint coordinates.
    """
    num_parts = scores.shape[0]
    num_edges = len(PARENT_CHILD_TUPLES)

    instance_keypoint_scores = np.zeros(num_parts)
    instance_keypoint_coords = np.zeros((num_parts, 2))
    instance_keypoint_scores[root_id] = root_score
    instance_keypoint_coords[root_id] = root_image_coord

    for edge in reversed(range(num_edges)):
        target_keypoint_id, source_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (
            instance_keypoint_scores[source_keypoint_id] > 0.0
            and instance_keypoint_scores[target_keypoint_id] == 0.0
        ):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores,
                offsets,
                displacements_bwd,
            )
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    for edge in range(num_edges):
        source_keypoint_id, target_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (
            instance_keypoint_scores[source_keypoint_id] > 0.0
            and instance_keypoint_scores[target_keypoint_id] == 0.0
        ):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores,
                offsets,
                displacements_fwd,
            )
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    return instance_keypoint_scores, instance_keypoint_coords


def within_nms_radius_fast(
    pose_coords: np.ndarray, nms_radius: float, point: np.ndarray
) -> bool:
    """
    Whether the candidate point is nearby any existing point in `pose_coords`.

    Parameters
    ----------
    pose_coords
        Numpy array of points, shape (N, 2).
    nms_radius
        The distance between two points for them to be considered nearby.
    point
        The candidate point, shape (2,).

    Returns
    -------
    within_radius : bool
        Whether the point is within the NMS radius of any existing pose coords.
    """
    if not pose_coords.shape[0]:
        return False
    return bool(np.any(np.sum((pose_coords - point) ** 2, axis=1) <= nms_radius**2))


def get_instance_score_fast(
    exist_pose_coords: np.ndarray,
    nms_radius: int,
    keypoint_scores: np.ndarray,
    keypoint_coords: np.ndarray,
) -> float:
    """
    Compute a probability that the given pose is real.
    Equal to the average confidence of each keypoint, excluding keypoints
    that are shared with existing poses.

    Parameters
    ----------
    exist_pose_coords
        Keypoint coordinates of poses that have already been found.
        Shape (N, 17, 2)
    nms_radius
        If two candidate keypoints for the same body part are within this distance,
        they are considered the same, and the lower confidence one discarded.
    keypoint_scores
        Keypoint scores for the new pose. Shape (17,)
    keypoint_coords
        Coordinates for the new pose. Shape (17, 2)

    Returns
    -------
    confidence_score : float
        Confidence score for the pose.
    """
    if exist_pose_coords.shape[0]:
        s = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2) > nms_radius**2
        not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return not_overlapped_scores / len(keypoint_scores)


def build_part_with_score_torch(
    score_threshold: float, scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get candidate keypoints to be considered the root for a pose.
    Score for the keypoint must be >= all neighboring scores.
    Score must also be above given score_threshold.

    Parameters
    ----------
    score_threshold
        Minimum score for a keypoint to be considered as a root.
    scores
        See `decode_multiple_poses`.

    Returns
    -------
    scores_vec : torch.Tensor
        Torch scores for each keypoint to be considered.
    max_loc_idx : torch.Tensor
        Indices of the considered keypoints. Shape (N, 3) where the 3 indices
        map to the dimensions of the scores tensor with shape (17, h, w).
    """
    # Compute the local maxima of `scores` via maxpool, then keep positions that
    # equal their local max (i.e. peaks) and clear the score threshold. The
    # maxpool is recomputed here from `scores` (as upstream does) rather than
    # taken as a separate model output: on device a separate output would be
    # quantized independently from `scores`, so the `==` comparison would rarely
    # hold -> zero peaks -> mAP collapse. Recomputing keeps the equality exact
    # regardless of quantization.
    # https://github.com/rwightman/posenet-pytorch/blob/6f7376d/posenet/decode_multi.py#L28
    lmd = 2 * LOCAL_MAXIMUM_RADIUS + 1
    max_pooled_vals = F.max_pool2d(scores, lmd, stride=1, padding=LOCAL_MAXIMUM_RADIUS)
    max_loc = (scores == max_pooled_vals) & (scores >= score_threshold)
    max_loc_idx = max_loc.nonzero()
    scores_vec = scores[max_loc]
    sort_idx = torch.argsort(scores_vec, descending=True)
    return scores_vec[sort_idx], max_loc_idx[sort_idx]


def decode_multiple_poses(
    scores: torch.Tensor,
    offsets: torch.Tensor,
    displacements_fwd: torch.Tensor,
    displacements_bwd: torch.Tensor,
    max_pose_detections: int = 10,
    score_threshold: float = 0.25,
    nms_radius: int = 20,
    min_pose_score: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts raw model outputs into image with keypoints drawn.
    Can detect multiple poses in the same image, up to `max_pose_detections`.
    This model has 17 candidate keypoints it predicts.
    In this docstring, (h, w) correspond to height and width of the grid
    and are roughly equal to input image size divided by 16.

    Parameters
    ----------
    scores
        Tensor of scores in range [0, 1] indicating probability
        a candidate pose is real. Shape [17, h, w].
    offsets
        Tensor of offsets for a given keypoint, relative to the grid point.
        Shape [34, h, w].
    displacements_fwd
        When tracing the points for a pose, given a source keypoint, this value
        gives the displacement to the next keypoint in the pose. There are 16
        connections from one keypoint to another (it's a minimum spanning tree).
        Shape [32, h, w].
    displacements_bwd
        Same as displacements_fwd, except when traversing keypoint connections
        in the opposite direction.
    max_pose_detections
        Maximum number of distinct poses to detect in a single image.
    score_threshold
        Minimum score for a keypoint to be considered the root for a pose.
    nms_radius
        If two candidate keypoints for the same body part are within this distance,
        they are considered the same, and the lower confidence one discarded.
    min_pose_score
        Minimum confidence that a pose exists for it to be displayed.

    Returns
    -------
    pose_scores : np.ndarray
        Numpy array of pose confidence scores.
    pose_keypoint_scores : np.ndarray
        Numpy array of keypoint confidence scores.
    pose_keypoint_coords : np.ndarray
        Numpy array of keypoint coordinates.
    """
    part_scores_pt, part_idx_pt = build_part_with_score_torch(score_threshold, scores)
    part_scores = part_scores_pt.cpu().numpy()
    part_idx = part_idx_pt.cpu().numpy()

    scores_np = scores.cpu().numpy()
    height = scores_np.shape[1]
    width = scores_np.shape[2]
    # change dimensions from (x, h, w) to (x//2, h, w, 2) to allow return of complete coord array
    offsets_np = (
        offsets.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    )
    displacements_fwd_np = (
        displacements_fwd.cpu()
        .numpy()
        .reshape(2, -1, height, width)
        .transpose((1, 2, 3, 0))
    )
    displacements_bwd_np = (
        displacements_bwd.cpu()
        .numpy()
        .reshape(2, -1, height, width)
        .transpose((1, 2, 3, 0))
    )

    pose_count = 0
    pose_scores = np.zeros(max_pose_detections)
    pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))
    pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))

    for root_score, (root_id, root_coord_y, root_coord_x) in zip(
        part_scores, part_idx, strict=False
    ):
        root_coord = np.array([root_coord_y, root_coord_x])
        root_image_coords = (
            root_coord * OUTPUT_STRIDE + offsets_np[root_id, root_coord_y, root_coord_x]
        )

        if within_nms_radius_fast(
            pose_keypoint_coords[:pose_count, root_id, :],
            nms_radius,
            root_image_coords,
        ):
            continue

        keypoint_scores, keypoint_coords = decode_pose(
            root_score,
            root_id,
            root_image_coords,
            scores_np,
            offsets_np,
            displacements_fwd_np,
            displacements_bwd=displacements_bwd_np,
        )

        pose_score = get_instance_score_fast(
            pose_keypoint_coords[:pose_count, :, :],
            nms_radius,
            keypoint_scores,
            keypoint_coords,
        )

        # NOTE this isn't in the original implementation, but it appears that by initially ordering by
        # part scores, and having a max # of detections, we can end up populating the returned poses with
        # lower scored poses than if we discard 'bad' ones and continue (higher pose scores can still come later).
        # Set min_pose_score to 0. to revert to original behaviour
        if min_pose_score == 0.0 or pose_score >= min_pose_score:
            pose_scores[pose_count] = pose_score
            pose_keypoint_scores[pose_count, :] = keypoint_scores
            pose_keypoint_coords[pose_count, :, :] = keypoint_coords
            pose_count += 1

        if pose_count >= max_pose_detections:
            break

    return pose_scores, pose_keypoint_scores, pose_keypoint_coords


def get_adjacent_keypoints(
    keypoint_scores: np.ndarray, keypoint_coords: np.ndarray, score_threshold: float
) -> list[np.ndarray]:
    """
    Compute which keypoints should be connected in the image.

    Parameters
    ----------
    keypoint_scores
        Scores for all candidate keypoints in the pose.
    keypoint_coords
        Coordinates for all candidate keypoints in the pose.
    score_threshold
        If either keypoint in a candidate edge is below this threshold, omit the edge.

    Returns
    -------
    results : list[np.ndarray]
        List of (2, 2) numpy arrays containing coordinates of edge endpoints.
    """
    results = []
    for left, right in CONNECTED_PART_INDICES:
        if (
            keypoint_scores[left] < score_threshold
            or keypoint_scores[right] < score_threshold
        ):
            continue
        results.append(
            np.array(
                [keypoint_coords[left][::-1], keypoint_coords[right][::-1]]
            ).astype(np.int32),
        )
    return results


def draw_skel_and_kp(
    img: np.ndarray,
    instance_scores: np.ndarray,
    keypoint_scores: np.ndarray,
    keypoint_coords: np.ndarray,
    min_pose_score: float = 0.5,
    min_part_score: float = 0.5,
) -> None:
    """
    Draw the keypoints and edges on the input numpy array image in-place.

    Parameters
    ----------
    img
        Numpy array of the image.
    instance_scores
        Numpy array of confidence for each pose.
    keypoint_scores
        Numpy array of confidence for each keypoint.
    keypoint_coords
        Numpy array of coordinates for each keypoint.
    min_pose_score
        Minimum score for a pose to be displayed.
    min_part_score
        Minimum score for a keypoint to be displayed.
    """
    adjacent_keypoints = []
    points = []
    sizes = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_connections = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score
        )
        adjacent_keypoints.extend(new_connections)

        for ks, kc in zip(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], strict=False
        ):
            if ks < min_part_score:
                continue
            points.append([kc[1], kc[0]])
            sizes.append(10.0 * ks)

    if points:
        points_np = np.array(points)
        draw_points(img, points_np, color=(255, 255, 0), size=sizes)
        cv2.polylines(img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))


class PosenetApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Posenet.

    The app uses 1 model:
        * Posenet

    For a given image input, the app will:
        * pre-process the image
        * Run Posenet inference
        * Convert the output into a list of keypoint coordiates
        * Return raw coordinates or an image with keypoints overlayed
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        input_spec: InputSpec | None = None,
    ) -> None:
        self.model = model
        if input_spec is not None:
            _, _, self.input_height, self.input_width = input_spec["image"][0]
        else:
            self.input_height = 513
            self.input_width = 257

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> Image.Image | tuple[np.ndarray, np.ndarray, np.ndarray]:
        # See predict_pose_keypoints.
        return self.predict_pose_keypoints(*args, **kwargs)

    def predict_pose_keypoints(
        self,
        image: Image.Image,
        raw_output: bool = False,
    ) -> Image.Image | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts up to 17 pose keypoints for up to 10 people in the image.

        Parameters
        ----------
        image
            Image on which to predict pose keypoints.
        raw_output
            See "returns" doc section for details.

        Returns
        -------
        Image.Image | tuple[np.ndarray, np.ndarray, np.ndarray]
            If raw_output is true, returns a tuple of:
                pose_scores: np.ndarray
                    Confidence score that a given pose is real for up to 10 poses. Shape (10,).
                keypoint_scores: np.ndarray
                    Confidence score that a given keypoint is real.
                    There can be up to 10 poses and up to 17 keypoints per pose. Shape (10, 17).
                keypoint_coords: np.ndarray
                    Coordinates of predicted keypoints in (y, x) format. Shape (10, 17, 2).

            Otherwise, returns:
                predicted_images: PIL.Image.Image
                    Image with keypoints drawn.
        """
        original_size = image.size
        image, scale, padding = pil_resize_pad(
            image, (self.input_height, self.input_width)
        )
        tensor = transforms.ToTensor()(image)
        tensor = tensor.reshape(1, 3, self.input_height, self.input_width)

        (
            heatmaps_result,
            offsets_result,
            displacement_fwd_result,
            displacement_bwd_result,
        ) = self.model(tensor)
        pose_scores, keypoint_scores, keypoint_coords = decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            max_pose_detections=10,
            min_pose_score=0.25,
        )
        if raw_output:
            return pose_scores, keypoint_scores, keypoint_coords
        output_arr = np.array(image)
        draw_skel_and_kp(
            output_arr,
            pose_scores,
            keypoint_scores,
            keypoint_coords,
            min_pose_score=0.25,
            min_part_score=0.1,
        )
        image_result = Image.fromarray(output_arr)
        return pil_undo_resize_pad(image_result, original_size, scale, padding)

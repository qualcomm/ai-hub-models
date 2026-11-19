# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from typing import Any, cast

import numpy as np
import torch
from PIL.Image import Image, fromarray
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.datasets import instantiate_dataset
from qai_hub_models.datasets.coco.coco_person_keypoints import (
    CocoDetectorKeypointsDataset,
)
from qai_hub_models.models.hrnet_pose.dataset import CocoHRNetPoseDataset
from qai_hub_models.models.hrnet_pose.model import HRNetPose
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.models.templates.corevision.draw import draw_points
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.pose.pose_utils import get_final_preds
from qai_hub_models.models.templates.pose.topdown import (
    PERSON_CLASS_ID,
    box_xyxy_to_cs,
    crop_to_center_scale,
    detect_and_crop,
)
from qai_hub_models.utils.base_app import CollectionModelEvalGenerator
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec

DETECTOR_SCORE_THRESHOLD = 0.3
DETECTOR_NMS_IOU_THRESHOLD = 0.65


class _HRNetPoseEvalResult(AsyncOnDeviceResult):
    """Attach host-side detector metadata to an asynchronous pose result."""

    def __init__(
        self,
        pose_result: AsyncOnDeviceResult,
        metadata: tuple[torch.Tensor, ...],
    ) -> None:
        self.pose_result = pose_result
        self.metadata = metadata

    def wait(self) -> tuple[torch.Tensor, ...]:
        output = self.pose_result.wait()
        heatmaps = output[0] if isinstance(output, tuple) else output
        return (heatmaps, *self.metadata)


class _CompletedHRNetPoseEvalResult(AsyncOnDeviceResult):
    def __init__(self, output: tuple[torch.Tensor, ...]) -> None:
        self.output = output

    def wait(self) -> tuple[torch.Tensor, ...]:
        return self.output


def get_max_preds(batch_heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])

    Parameters
    ----------
    batch_heatmaps
        Heatmaps with shape [batch, num_joints, height, width].

    Returns
    -------
    preds : np.ndarray
        Keypoint coordinates with shape [batch, num_joints, 2].
    maxvals : np.ndarray
        Peak confidence per joint with shape [batch, num_joints, 1].
    """
    assert isinstance(batch_heatmaps, np.ndarray), (
        "batch_heatmaps should be numpy.ndarray"
    )
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    # shape [batch, num joints, image h * w]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    # index of max pixel per joint
    idx = np.argmax(heatmaps_reshaped, 2)

    # value of max pixel per joint
    maxvals = np.amax(heatmaps_reshaped, 2)

    # Reshape to prep for tiling
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # Tile indices to make room for (x, y)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # Convert index [..., 0] to x
    preds[:, :, 0] = (preds[:, :, 0]) % width
    # Convert [..., 1] to y
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # Tile mask back to (x, y) plane to ignore negatives
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    # apply mask
    preds *= pred_mask
    return preds, maxvals


class HRNetPoseApp:
    """
    End-to-end two-stage top-down pose estimation.

    Stage 1 — RTMDetPersonDetector: finds the person box in the scene image.
    Stage 2 — PoseEstimator: predicts keypoint heatmaps from the person crop.

    The box -> center/scale conversion and affine crop between the two run on the
    host in repo-owned code (models/templates/pose/topdown.py), replacing the
    uncompiled detector that mmpose's MMPoseInferencer used to run here.
    """

    def __init__(
        self,
        person_detector: ExecutableModelProtocol | AsyncOnDeviceModel,
        pose_estimator: ExecutableModelProtocol | AsyncOnDeviceModel,
    ) -> None:
        self.person_detector = person_detector
        self.pose_estimator = pose_estimator
        _, _, pose_height, pose_width = pose_estimator.get_input_spec()["image"][0]  # type: ignore[union-attr]
        _, _, detector_height, detector_width = person_detector.get_input_spec()[  # type: ignore[union-attr]
            "image"
        ][0]
        self.input_hw = (pose_height, pose_width)
        self.detector_hw = (detector_height, detector_width)

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray | list[Image]:
        return self.predict_pose_keypoints(*args, **kwargs)

    def predict_pose_keypoints(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        raw_output: bool = False,
    ) -> np.ndarray | list[Image]:
        """
        Predicts pose keypoints for a person in the image.

        Parameters
        ----------
        pixel_values_or_image
            PIL image(s)
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout
        raw_output
            See "returns" doc section for details.

        Returns
        -------
        result : np.ndarray | list[Image]
            If raw_output is True, returns:
            keypoints
                Numpy array of keypoints with shape [B, N, 2].
                Each keypoint is an (x, y) pair of coordinates within the image.

            If raw_output is False, returns:
            predicted_images
                Images with keypoints drawn.
        """
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(pixel_values_or_image)

        all_keypoints = []
        for frame in NHWC_int_numpy_frames:
            crop, center, scale = detect_and_crop(
                cast(
                    Callable[
                        [torch.Tensor],
                        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    ],
                    self.person_detector,
                ),
                frame,
                self.input_hw,
                self.detector_hw,
            )
            heatmaps = cast(torch.Tensor, self.pose_estimator(crop))
            keypoints, _ = get_final_preds(
                heatmaps.detach().numpy(), np.array([center]), np.array([scale])
            )
            all_keypoints.append(np.round(keypoints[0]).astype(np.int32))

        keypoints_arr = np.stack(all_keypoints)
        if raw_output:
            return keypoints_arr

        predicted_images = []
        for i, img in enumerate(NHWC_int_numpy_frames):
            draw_points(img, keypoints_arr[i], color=(255, 0, 0), size=6)
            predicted_images.append(fromarray(img))
        return predicted_images

    @property
    def uses_ondevice_model(self) -> bool:
        """True if any component is an AsyncOnDeviceModel; False if all are local."""
        return isinstance(self.person_detector, AsyncOnDeviceModel) or isinstance(
            self.pose_estimator, AsyncOnDeviceModel
        )

    @classmethod
    def from_components(
        cls,
        models: Sequence[ExecutableModelProtocol] | Sequence[AsyncOnDeviceModel],
    ) -> HRNetPoseApp:
        """
        Create an HRNetPoseApp from a list of model components.

        Parameters
        ----------
        models
            List of two components: [person_detector, pose_estimator].

        Returns
        -------
        HRNetPoseApp
            App initialized with the provided components.
        """
        return cls(
            person_detector=models[0],
            pose_estimator=models[1],
        )

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        if isinstance(model_input, tuple):
            image_tensor = model_input[0]
            detector_input: torch.Tensor | list[torch.Tensor] = image_tensor
        else:
            image_chunks: list[torch.Tensor] = list(cast(Any, next(model_input)))
            image_tensor = torch.cat(image_chunks)
            detector_input = image_chunks

        detector_output = self.person_detector(detector_input)  # type: ignore[arg-type]
        yield detector_output
        if isinstance(detector_output, AsyncOnDeviceResult):
            detector_output = detector_output.wait()
        boxes, scores, class_idx = detector_output

        person_scores = scores.clone()
        person_scores[class_idx != PERSON_CLASS_ID] = -1
        boxes_nms, scores_nms = batched_nms(
            DETECTOR_NMS_IOU_THRESHOLD,
            DETECTOR_SCORE_THRESHOLD,
            boxes.cpu().float(),
            person_scores.cpu().float(),
        )

        crops: list[torch.Tensor] = []
        centers: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        box_scores: list[torch.Tensor] = []
        areas: list[torch.Tensor] = []
        batch_indices: list[int] = []
        aspect_ratio = self.input_hw[1] / self.input_hw[0]
        for batch_idx, (image_boxes, image_scores) in enumerate(
            zip(boxes_nms, scores_nms, strict=True)
        ):
            image = (
                image_tensor[batch_idx]
                .permute(1, 2, 0)
                .mul(255)
                .round()
                .clamp(0, 255)
                .byte()
                .numpy()
            )
            for box, score in zip(image_boxes, image_scores, strict=True):
                box_np = box.numpy()
                center, scale = box_xyxy_to_cs(box_np, aspect_ratio)
                crops.append(crop_to_center_scale(image, center, scale, self.input_hw))
                centers.append(center)
                scales.append(scale)
                box_scores.append(score)
                areas.append((box[2] - box[0]) * (box[3] - box[1]))
                batch_indices.append(batch_idx)

        if not crops:
            empty_output = (
                torch.empty((0, 17, self.input_hw[0] // 4, self.input_hw[1] // 4)),
                torch.empty((0, 2)),
                torch.empty((0, 2)),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0, dtype=torch.int64),
            )
            if self.uses_ondevice_model:
                return _CompletedHRNetPoseEvalResult(empty_output)
            return empty_output

        pose_inputs = torch.cat(crops)
        pose_output: torch.Tensor | AsyncOnDeviceResult
        if isinstance(self.pose_estimator, AsyncOnDeviceModel):
            pose_output = self.pose_estimator(list(pose_inputs.split(model_batch_size)))
            yield pose_output
        else:
            heatmap_batches: list[torch.Tensor] = []
            for crop_batch in pose_inputs.split(model_batch_size):
                heatmaps = cast(
                    torch.Tensor | tuple[torch.Tensor, ...],
                    self.pose_estimator(crop_batch),
                )
                heatmap_batches.append(
                    heatmaps[0] if isinstance(heatmaps, tuple) else heatmaps
                )
            pose_output = torch.cat(heatmap_batches)
            yield (pose_output,)

        metadata = (
            torch.from_numpy(np.stack(centers)),
            torch.from_numpy(np.stack(scales)),
            torch.stack(box_scores),
            torch.stack(areas),
            torch.tensor(batch_indices, dtype=torch.int64),
        )
        if isinstance(pose_output, AsyncOnDeviceResult):
            return _HRNetPoseEvalResult(pose_output, metadata)
        heatmaps = pose_output[0] if isinstance(pose_output, tuple) else pose_output
        return (heatmaps, *metadata)

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        assert isinstance(collection_model, HRNetPose)
        if component_name not in {"pose_estimator", "person_detector"}:
            raise ValueError(f"Unknown component: {component_name}")
        pose_spec = (
            input_specs.get("pose_estimator") if input_specs else None
        ) or collection_model.pose_estimator.get_input_spec()
        dataset_cls = (
            CocoDetectorKeypointsDataset
            if component_name == "pose_estimator"
            else CocoHRNetPoseDataset
        )
        dataset_spec = pose_spec
        if component_name == "person_detector":
            dataset_spec = (
                input_specs.get("person_detector") if input_specs else None
            ) or collection_model.person_detector.get_input_spec()
        dataset = instantiate_dataset(
            dataset_cls, DatasetSplit.TRAIN, input_spec=dataset_spec
        )
        num_samples = min(
            num_samples or dataset.default_samples_per_job(), len(dataset)
        )

        if component_name == "pose_estimator":
            entries: dict[str, list[np.ndarray]] = {"image": []}
            for i in range(num_samples):
                image_tensor, _ = dataset[i]
                entries["image"].append(image_tensor.unsqueeze(0).numpy())
            return entries

        if component_name == "person_detector":
            entries = {"image": []}
            for i in range(num_samples):
                image_tensor, _ = dataset[i]
                entries["image"].append(image_tensor.unsqueeze(0).numpy())
            return entries

        raise ValueError(f"Unknown component: {component_name}")

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import warnings
from collections.abc import Callable, Generator, Sequence
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.datasets import instantiate_dataset
from qai_hub_models.models.deepbox.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.deepbox.external_repos.boundingbox_3d.library.Math import (
    calc_location,
)
from qai_hub_models.models.deepbox.external_repos.boundingbox_3d.library.Plotting import (
    plot_3d_box,
)
from qai_hub_models.models.deepbox.external_repos.boundingbox_3d.torch_lib import (
    ClassAverages,
    Dataset,
)
from qai_hub_models.models.deepbox.model import (
    DEFAULT_YOLO_WEIGHTS,
    DeepBox,
    Yolo2DDetection,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.base_app import CollectionModelEvalGenerator
from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    WorkbenchModelCollection,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec

REPO_PATH = EXTERNAL_REPO_PATHS["boundingbox_3d"]


class DeepBoxApp:
    """
    App code to perform end-to-end DeepBox 3D object detection inference.

    The app uses 2 models:
        * Yolo2DDetection
        * VGG3DDetection

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1]).
        * Detect the object using Yolo2DDetection.
        * For Every Detected Object, Makes the 2D detection to 3D.
        * Map the 3D Bounding boxes to the original input frame.
    """

    def __init__(
        self,
        bbox2D_detector: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        bbox3D_detector: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        bbox2D_detector_input_spec: InputSpec | None = None,
        nms_score_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
    ) -> None:
        """
        Construct a DeepBox 3D object detection application.

        Parameters
        ----------
        bbox2D_detector
            The 2D boundary box detection model.
            Input is an image [N C H W], channel layout is RGB [0-1], output is [pred_boxes, pred_scores, pred_class_idx].
        bbox3D_detector
            The 3D boundary box detection model.
            Input is an image [N C H W], channel layout is RGB [0-1], output is [proj_matrix, orient, dim, location].
        bbox2D_detector_input_spec
            Input spec of bbox2D_detector. Required by detect_image; not needed for
            run_model_for_eval, so from_components leaves it unset.
        nms_score_threshold
            Score threshold for when NMS is run on the detector output boxes.
        nms_iou_threshold
            IOU threshold for when NMS is run on the detector output boxes.
        """
        self.yolo = bbox2D_detector
        self.vgg = bbox3D_detector
        self.yolo_input_spec = bbox2D_detector_input_spec
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> (
        tuple[
            list[npt.NDArray[np.float64]],
            list[np.float64],
            list[npt.NDArray[np.float32]],
            list[list[np.float64]],
        ]
        | Image.Image
    ):
        # See predict_3d_boxes_from_image.
        return self.detect_image(*args, **kwargs)

    def detect_image(
        self,
        image: Image.Image,
        raw_output: bool = False,
    ) -> (
        tuple[
            list[npt.NDArray[np.float64]],
            list[np.float64],
            list[npt.NDArray[np.float32]],
            list[list[np.float64]],
        ]
        | Image.Image
    ):
        """
        From the provided image or tensor, predict the 3D bounding boxes and classes of objects detected within.

        Parameters
        ----------
        image
            PIL image.
        raw_output
            If False, returns annotated image. If True, returns raw outputs.

        Returns
        -------
        result : tuple[list[npt.NDArray[np.float64]], list[np.float64], list[npt.NDArray[np.float32]], list[list[np.float64]]] | Image.Image
            If raw_output is False:
                PIL Image with predicted 3D Bounding Boxes applied.
            If raw_output is True:
                proj_matrixes
                    Camera to img matrix.
                orients
                    Global orientations.
                dims
                    Dimensions for the 3D bboxes.
                locations
                    Centers of 3D bboxes.
        """
        # Input Prep
        numpy_image = np.array(image)
        (H, W) = numpy_image.shape[:2]
        assert self.yolo_input_spec is not None, (
            "bbox2D_detector_input_spec is required for detect_image; "
            "construct DeepBoxApp via from_pretrained() to set it."
        )
        (H_resized, W_resized) = self.yolo_input_spec["image"][0][-2:]
        image_resized = image.resize((W_resized, H_resized))

        raw_pred_boxes, pred_scores, pred_class_idx = self.detect_2d_bboxes(
            image_resized
        )

        # Converting output floating point box coordinates to the input image's coordinate space
        height_scale = H / H_resized
        width_scale = W / W_resized
        pred_boxes = raw_pred_boxes[0]
        pred_boxes[:, (0, 2)] = pred_boxes[:, (0, 2)] * width_scale
        pred_boxes[:, (1, 3)] = pred_boxes[:, (1, 3)] * height_scale

        # Detect 3d bboxes for each objects detected
        proj_matrixes: list[npt.NDArray[np.float64]] = []
        orients: list[np.float64] = []
        dims: list[npt.NDArray[np.float32]] = []
        locations: list[list[np.float64]] = []
        for i in range(pred_scores[0].shape[0]):
            output = self.detect_3d_bboxes(
                numpy_image, pred_boxes[i], pred_class_idx[0][i]
            )
            if output is None:
                continue
            proj_matrixes.append(output[0])
            orients.append(output[1])
            dims.append(output[2])
            locations.append(output[3])

        if raw_output:
            return proj_matrixes, orients, dims, locations

        return Image.fromarray(numpy_image)

    def detect_2d_bboxes(
        self, image_resized: Image.Image
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        _, NCHW_fp32_torch_frames = app_to_net_image_inputs(image_resized)

        raw_boxes, raw_scores, raw_class_idx = self.yolo(NCHW_fp32_torch_frames)

        pred_boxes, pred_scores, pred_class_idx = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            raw_boxes,
            raw_scores,
            raw_class_idx,
        )

        return pred_boxes, pred_scores, pred_class_idx

    def detect_3d_bboxes(
        self,
        numpy_image: np.ndarray,  # H W C, int8 [0, 255] image
        pred_boxes: torch.Tensor,
        pred_class_idx: torch.Tensor,
    ) -> (
        tuple[
            npt.NDArray[np.float64],
            np.float64,
            npt.NDArray[np.float32],
            list[np.float64],
        ]
        | None
    ):
        averages = ClassAverages.ClassAverages()
        angle_bins = Dataset.generate_bins(2)

        # Gets the labels and camera calib
        labels_path = REPO_PATH / "weights" / "coco.names"
        with open(labels_path) as labels_f:
            labels = labels_f.read().split("\n")
        calib_file = REPO_PATH / "camera_cal" / "calib_cam_to_cam.txt"

        x1, y1, x2, y2 = pred_boxes

        # skip invalid bboxes
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            return None

        # change the bbox from xyxy tensor to list([xy][xy]) and
        # assign the label for the class
        box_2d = [[int(x1), int(y1)], [int(x2), int(y2)]]
        detected_class = labels[int(pred_class_idx)]
        if detected_class == "person":
            detected_class = "pedestrian"

        # detects only for car, truck, van, tram, cyclist and pedestrian
        if not averages.recognized_class(detected_class):
            return None

        detectedObject = Dataset.DetectedObject(
            numpy_image, detected_class, box_2d, str(calib_file)
        )
        theta_ray = detectedObject.theta_ray
        proj_matrix = detectedObject.proj_matrix

        # Crop to detected bounding box, reshape to input of vgg net
        pt1 = box_2d[0]
        pt2 = box_2d[1]
        cropped_image = numpy_image[pt1[1] : pt2[1] + 1, pt1[0] : pt2[0] + 1]
        # Note that this resize does not preserve aspect ratio. While odd, this is the implementation in the original paper, so we kept it.
        cropped_image = cv2.resize(
            cropped_image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC
        )
        cropped_image = cropped_image.astype(np.float32) / 255.0

        # detect the 3d bbox
        orient_t, conf_t, dim_t = self.vgg(
            torch.as_tensor(cropped_image.transpose(2, 0, 1)).unsqueeze(0)
        )
        orient_np = orient_t.numpy()[0, :, :]
        conf_np = conf_t.numpy()[0, :]
        dim_np = dim_t.numpy()[0, :]

        # add avgerage dim of the detected class
        dim_np += averages.get_item(detected_class)

        # global orientation
        argmax = np.argmax(conf_np)
        cos, sin = orient_np[argmax, :]
        alpha = np.arctan2(sin, cos) + angle_bins[argmax] - np.pi
        orient = alpha + theta_ray

        # calculate best_loc, [left_constraints, right_constraints]
        location, _X = calc_location(dim_np, proj_matrix, box_2d, alpha, theta_ray)

        # plots 3d boxes. plot_3d_box uses BGR-ordered cv2 color constants,
        # so draw on a BGR copy and convert back to keep numpy_image RGB.
        image_bgr = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        plot_3d_box(image_bgr, proj_matrix, orient, dim_np, location)
        numpy_image[:] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return (proj_matrix, orient, dim_np, location)

    @classmethod
    def from_pretrained(cls, model: CollectionModel) -> DeepBoxApp:
        assert isinstance(model, DeepBox)
        return cls(
            model.yolo_2d_det,
            model.vgg_3d_det,
            model.yolo_2d_det.get_input_spec(),
        )

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        assert isinstance(collection_model, DeepBox)

        yolo_spec = (
            input_specs.get("yolo_2d_detection") if input_specs else None
        ) or collection_model.yolo_2d_det.get_input_spec()
        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=yolo_spec,
        )
        num_samples = num_samples or dataset.default_samples_per_job()

        if component_name == "yolo_2d_detection":
            entries: dict[str, list[np.ndarray]] = {"image": []}
            for i in range(min(num_samples, len(dataset))):
                image_tensor, _ = dataset[i]
                entries["image"].append(image_tensor.unsqueeze(0).numpy())
            return entries

        if component_name == "vgg_3d_detection":
            # Run 2D detector to get bounding boxes, then crop for 3D estimator.
            #
            # Use a FRESH Yolo2DDetection, not collection_model.yolo_2d_det:
            # during a w8a16 export the YOLO component is serialized via
            # torch.export beforehand, which poisons the ultralytics
            # DetectionModel (cached anchors/grids become FakeTensors) so its
            # eager forward returns unbacked symbolic shapes that break
            # range()/int() in detect_2d_bboxes. A freshly loaded module was
            # never exported and returns concrete tensors.
            fresh_yolo = Yolo2DDetection.from_pretrained(DEFAULT_YOLO_WEIGHTS)
            app = cls(
                fresh_yolo, collection_model.vgg_3d_det, fresh_yolo.get_input_spec()
            )
            vgg_spec = (
                input_specs.get("vgg_3d_detection") if input_specs else None
            ) or collection_model.vgg_3d_det.get_input_spec()
            vgg_h, vgg_w = vgg_spec["image"][0][-2:]
            entries = {"image": []}
            collected = 0
            for i in range(len(dataset)):
                if collected >= num_samples:
                    break
                image_tensor, _ = dataset[i]
                image_pil = Image.fromarray(
                    (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                )
                raw_boxes, pred_scores, _ = app.detect_2d_bboxes(image_pil)
                if not pred_scores or pred_scores[0].shape[0] == 0:
                    continue
                numpy_image = np.array(image_pil)
                for j in range(raw_boxes[0].shape[0]):
                    if collected >= num_samples:
                        break
                    x1, y1, x2, y2 = raw_boxes[0][j].int().tolist()
                    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                        continue
                    cropped = numpy_image[y1 : y2 + 1, x1 : x2 + 1]
                    if cropped.size == 0:
                        continue
                    cropped = cv2.resize(
                        cropped, dsize=(vgg_w, vgg_h), interpolation=cv2.INTER_CUBIC
                    )
                    cropped = cropped.astype(np.float32) / 255.0
                    entries["image"].append(
                        np.expand_dims(cropped.transpose(2, 0, 1), 0)
                    )
                    collected += 1
            return entries

        raise ValueError(f"Unknown component: {component_name}")

    @property
    def uses_ondevice_model(self) -> bool:
        """True if any component is an AsyncOnDeviceModel; False if all are local."""
        return isinstance(self.yolo, AsyncOnDeviceModel) or isinstance(
            self.vgg, AsyncOnDeviceModel
        )

    @classmethod
    def from_components(
        cls,
        models: Sequence[ExecutableModelProtocol] | Sequence[AsyncOnDeviceModel],
    ) -> DeepBoxApp:
        """
        Create a DeepBoxApp from a list of model components.

        Parameters
        ----------
        models
            List of two components: [yolo_2d_detection, vgg_3d_detection].

        Returns
        -------
        DeepBoxApp
            App initialized with the provided components.
        """
        return cls(
            bbox2D_detector=models[0],  # type: ignore[arg-type]
            bbox3D_detector=models[1],  # type: ignore[arg-type]
        )

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        # Two device jobs per samples-per-job group: one batched YOLO job over all
        # images, then one batched VGG job over all detected crops. Each is
        # submitted and yielded BEFORE it is waited on, so the round-robin drain
        # (utils/evaluate/helpers.py) has every job in flight before blocking.
        # Pipeline: YOLO -> NMS + crop (CPU) -> VGG -> geometry decode (CPU).
        #
        # Crops are batched safely because a crop is a full [1, 3, 224, 224]
        # dataset entry, not a compiled tensor dimension: AI Hub runs the batch-1
        # VGG artifact once per entry with no recompile.
        on_device = isinstance(self.yolo, AsyncOnDeviceModel) or isinstance(
            self.vgg, AsyncOnDeviceModel
        )

        # On device, model_input is a generator of per-sample [1, 3, H, W] chunk
        # tuples; pass the chunks tuple to the AsyncOnDeviceModel as-is (one
        # dataset entry per sample), since collapsing to [B, 3, H, W] would fail on
        # a fixed batch-1 model. Locally it's a plain [B, 3, H, W] tuple. The
        # concatenated image_tensor is used only for CPU cropping below.
        if isinstance(model_input, tuple):
            image_tensor = model_input[0]
            yolo_input: object = image_tensor
        else:
            first = next(model_input)
            chunks = first if isinstance(first, tuple) else (first,)
            image_tensor = torch.cat(cast("list[torch.Tensor]", list(chunks)), dim=0)
            yolo_input = chunks

        # Stage 1: submit the single batched YOLO job and yield before waiting.
        yolo_output = self.yolo(yolo_input)  # type: ignore[arg-type]
        yield yolo_output

        if isinstance(yolo_output, AsyncOnDeviceResult):
            yolo_output = yolo_output.wait()
        pred_boxes_raw, pred_scores_raw, pred_class_idx_raw = yolo_output

        pred_boxes_nms, pred_scores_nms, pred_class_idx_nms = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            pred_boxes_raw,
            pred_scores_raw,
            pred_class_idx_raw,
        )

        batch_size = image_tensor.shape[0]
        _, _, yolo_h, yolo_w = image_tensor.shape

        # CPU: crop every valid detection across the group into a flat list.
        # crop_meta[k] records which image crop k came from plus the geometry
        # needed to decode VGG output k; per_image_detections[b] accumulates
        # decoded results.
        crops: list[torch.Tensor] = []
        crop_meta: list[tuple] = []
        per_image_detections: list[dict] = [
            {
                "orients": [],
                "dims": [],
                "locations": [],
                "pred_boxes_2d": [],
                "pred_scores": [],
                "yolo_h": int(yolo_h),
                "yolo_w": int(yolo_w),
            }
            for _ in range(batch_size)
        ]

        num_candidates = 0
        for b in range(batch_size):
            numpy_image = np.ascontiguousarray(
                (image_tensor[b].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            boxes = (
                pred_boxes_nms[b] if b < len(pred_boxes_nms) else torch.zeros((0, 4))
            )
            scores = pred_scores_nms[b] if b < len(pred_scores_nms) else torch.zeros(0)
            class_idx = (
                pred_class_idx_nms[b] if b < len(pred_class_idx_nms) else torch.zeros(0)
            )

            num_candidates += scores.shape[0]
            for i in range(scores.shape[0]):
                cropped = self._crop_for_detection(numpy_image, boxes[i], class_idx[i])
                if cropped is None:
                    continue
                crop_tensor, detected_class, box_2d, proj_matrix, theta_ray = cropped
                crops.append(crop_tensor)
                crop_meta.append(
                    (
                        b,
                        [float(v) for v in boxes[i]],
                        float(scores[i]),
                        detected_class,
                        box_2d,
                        proj_matrix,
                        theta_ray,
                    )
                )

        # Individual drops are routine (unrecognized class, degenerate box), but a
        # high drop rate signals a systemic bug (e.g. coordinate-space mismatch)
        # that would otherwise only surface as unexplained low AP.
        if num_candidates > 0 and len(crops) / num_candidates < 0.5:
            warnings.warn(
                f"DeepBox: dropped {num_candidates - len(crops)}/{num_candidates} "
                "detections in this group before VGG (unrecognized class, "
                "degenerate box, or failed geometry setup). If this rate holds "
                "across the whole eval run, accuracy results may be meaningless.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Stage 2: submit ONE batched VGG job over all crops, yield before waiting.
        # On device, crops go as a tuple of [1, 3, 224, 224] entries (one job).
        # Locally, self.vgg may be a fixed batch-1 QDQ-ONNX executor that can't
        # take a batched input, so run per-crop and stack the outputs afterward.
        if crops:
            if on_device:
                vgg_result = self.vgg(tuple(crops))  # type: ignore[arg-type]
            else:
                per_crop_outputs = [self.vgg(crop) for crop in crops]
                orient_out, conf_out, dim_out = zip(*per_crop_outputs, strict=True)
                vgg_result = (
                    torch.cat(orient_out, dim=0),
                    torch.cat(conf_out, dim=0),
                    torch.cat(dim_out, dim=0),
                )
        else:
            vgg_result = None
        yield vgg_result  # type: ignore[misc]

        # CPU: wait the single VGG job (if async), then decode each crop's geometry
        # and scatter it back to the image it came from.
        if crops:
            assert vgg_result is not None  # crops truthy -> VGG was submitted above
            resolved = (
                vgg_result.wait()
                if isinstance(vgg_result, AsyncOnDeviceResult)
                else vgg_result
            )
            orient_all, conf_all, dim_all = resolved
            for k, meta in enumerate(crop_meta):
                (
                    b,
                    box_2d_xyxy,
                    score,
                    detected_class,
                    box_2d,
                    proj_matrix,
                    theta_ray,
                ) = meta
                decoded = self._decode_vgg_outputs(
                    orient_all[k].numpy(),
                    conf_all[k].numpy(),
                    dim_all[k].numpy(),
                    detected_class,
                    box_2d,
                    proj_matrix,
                    theta_ray,
                )
                if decoded is None:
                    continue
                orient, dim_np, location = decoded
                det = per_image_detections[b]
                det["orients"].append(float(orient))
                det["dims"].append(dim_np)
                det["locations"].append([float(v) for v in location])
                det["pred_boxes_2d"].append(box_2d_xyxy)
                det["pred_scores"].append(score)

        # The last stage is CPU geometry decode, so the result is already final in
        # both the on-device and local paths.
        return (per_image_detections,)  # type: ignore[return-value]

    def _crop_for_detection(
        self,
        numpy_image: np.ndarray,
        box: torch.Tensor,
        class_idx: torch.Tensor,
    ) -> tuple | None:
        """
        Validate a detection and build its [1, 3, 224, 224] VGG input crop.

        Returns ``(crop_tensor, detected_class, box_2d, proj_matrix, theta_ray)``
        or None for invalid / unrecognized detections. No inference is run here so
        that all crops can be submitted as one batched VGG job (see
        run_model_for_eval); decoding is done by _decode_vgg_outputs.
        """
        averages = ClassAverages.ClassAverages()
        labels_path = REPO_PATH / "weights" / "coco.names"
        with open(labels_path) as f:
            labels = f.read().split("\n")
        calib_file = REPO_PATH / "camera_cal" / "calib_cam_to_cam.txt"

        x1, y1, x2, y2 = box
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            return None

        box_2d = [[int(x1), int(y1)], [int(x2), int(y2)]]
        detected_class = labels[int(class_idx)]
        if detected_class == "person":
            detected_class = "pedestrian"
        if not averages.recognized_class(detected_class):
            return None

        try:
            detected_obj = Dataset.DetectedObject(
                numpy_image, detected_class, box_2d, str(calib_file)
            )
        except Exception as e:
            # Geometry setup failed -- likely a malformed shared calib_file, which
            # would fail for every detection. Warn rather than swallow silently.
            warnings.warn(
                f"DeepBox: failed to construct DetectedObject for box {box_2d} "
                f"(class={detected_class}): {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        pt1, pt2 = box_2d
        cropped = numpy_image[pt1[1] : pt2[1] + 1, pt1[0] : pt2[0] + 1]
        if cropped.size == 0:
            return None
        cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_CUBIC)
        crop_tensor = torch.as_tensor(
            (cropped.astype(np.float32) / 255.0).transpose(2, 0, 1)
        ).unsqueeze(0)

        return (
            crop_tensor,
            detected_class,
            box_2d,
            detected_obj.proj_matrix,
            detected_obj.theta_ray,
        )

    def _decode_vgg_outputs(
        self,
        orient_np: np.ndarray,
        conf_np: np.ndarray,
        dim_np: np.ndarray,
        detected_class: str,
        box_2d: list[list[int]],
        proj_matrix: np.ndarray,
        theta_ray: float,
    ) -> tuple | None:
        """
        Decode one crop's VGG output into 3D geometry.

        Parameters are single-crop slices of the batched VGG result: orient_np is
        [bins, 2] (cos, sin per bin), conf_np is [bins], dim_np is [3]. Returns
        ``(orient, dim, location)``.
        """
        averages = ClassAverages.ClassAverages()
        angle_bins = Dataset.generate_bins(2)

        # New array (not +=) to avoid mutating the shared batched VGG output view.
        dim_np = dim_np + averages.get_item(detected_class)

        argmax = int(np.argmax(conf_np))
        cos_val, sin_val = orient_np[argmax, :]
        alpha = float(np.arctan2(sin_val, cos_val)) + angle_bins[argmax] - np.pi
        orient = alpha + theta_ray

        location, _ = calc_location(dim_np, proj_matrix, box_2d, alpha, theta_ray)
        return (orient, dim_np, location)

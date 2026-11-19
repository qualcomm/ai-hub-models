# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from collections.abc import Generator, Sequence
from typing import Any, cast

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.datasets import instantiate_dataset
from qai_hub_models.models.mobile_facenet.dataset import LFW_ASSET, LFWDataset
from qai_hub_models.models.mobile_facenet.external_repos.pytorch_mobile_facenet.align_faces import (
    get_reference_facial_points,
    warp_and_crop_face,
)
from qai_hub_models.models.mobile_facenet.external_repos.pytorch_mobile_facenet.facenet_utils import (
    select_significant_face,
)
from qai_hub_models.models.mobile_facenet.model import (
    SCORE_THRESHOLD,
    MobileFaceNet,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    numpy_image_to_torch,
)
from qai_hub_models.models.templates.retinaface.app import detect_faces
from qai_hub_models.utils.base_app import CollectionModelEvalGenerator
from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    WorkbenchModelCollection,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec


class MobileFaceNetApp:
    """
    Application wrapper for the two-component MobileFaceNet face verification pipeline.

    Stage 1 — RetinaFaceDetector: locates faces and 5-point landmarks in the scene image.
    Stage 2 — FaceRecognizer: embeds each aligned 112x112 face crop.

    Anchor decoding, NMS and the landmark warp run on the host between the two
    compiled components. Each crop is embedded twice (original + horizontal flip)
    and the two results are summed before normalisation (test-time augmentation).
    """

    def __init__(
        self,
        face_detector: ExecutableModelProtocol | AsyncOnDeviceModel,
        face_recognizer: ExecutableModelProtocol | AsyncOnDeviceModel,
        threshold: float = 0.0,
    ) -> None:
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        recognizer_shape = face_recognizer.get_input_spec()["img1"][0]  # type: ignore[union-attr]
        self.input_height: int = recognizer_shape[2]
        self.input_width: int = recognizer_shape[3]
        self.threshold = threshold
        self.device = "cpu"
        self.default_square = True
        self.inner_padding_factor = 0.25
        self.outer_padding = (0, 0)

    def predict(self, *args: Any, **kwargs: Any) -> tuple[bool, float, float]:
        return self.predict_features(*args, **kwargs)

    def predict_features(
        self,
        img1: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        img2: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        already_aligned: bool = False,
    ) -> tuple[bool, float, float]:
        """
        Verify whether two face images depict the same person.

        Both images are decoded, optionally detected and aligned via the
        RetinaFace component, then passed together in a single recognizer call.
        The recognizer applies TTA (original + horizontal flip) internally for
        each image and returns interleaved embeddings. The cosine angle between
        the two embeddings is compared against ``self.threshold`` to produce a
        same/different verdict.

        Parameters
        ----------
        img1
            First face image. Accepts a ``torch.Tensor``, ``np.ndarray``,
            ``PIL.Image``, or a list of ``PIL.Image`` objects (any resolution).
        img2
            Second face image. Accepts a ``torch.Tensor``, ``np.ndarray``,
            ``PIL.Image``, or a list of ``PIL.Image`` objects (any resolution).
        already_aligned
            If ``True``, skip face detection and alignment — each image is
            resized directly to ``(input_height, input_width)``. Set this when
            the inputs are pre-cropped face chips.

        Returns
        -------
        same : bool
            ``True`` if the cosine angle is below ``self.threshold`` (same person).
        angle : float
            Cosine angle in degrees between the two embeddings.
            Range ``[0°, 180°]``; lower means more similar.
        cosine : float
            Raw cosine similarity in ``[-1, 1]``; higher means more similar.
        """
        img1_bgr = app_to_net_image_inputs(img1, image_layout="BGR", to_float=False)[0][
            0
        ]
        img2_bgr = app_to_net_image_inputs(img2, image_layout="BGR", to_float=False)[0][
            0
        ]

        if already_aligned:
            img1_bgr = cv.resize(img1_bgr, (self.input_width, self.input_height))
            img2_bgr = cv.resize(img2_bgr, (self.input_width, self.input_height))
        else:
            img1_bgr = self.detect_and_align(img1_bgr)
            img2_bgr = self.detect_and_align(img2_bgr)

        img1_t = numpy_image_to_torch(img1_bgr[..., ::-1].copy())
        img2_t = numpy_image_to_torch(img2_bgr[..., ::-1].copy())
        output = self.face_recognizer(img1_t, img2_t)  # (2, 128)
        output = cast(torch.Tensor, output)

        emb1 = output[0].cpu().detach().numpy()
        emb2 = output[1].cpu().detach().numpy()
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        cosine = float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))
        angle = math.acos(cosine) * 180.0 / math.pi
        return angle < self.threshold, angle, cosine

    def align_face(self, raw: np.ndarray, facial5points: np.ndarray) -> np.ndarray:
        """
        Warp and crop a face image to a canonical ``(input_height, input_width)`` BGR crop.

        Parameters
        ----------
        raw
            Full-resolution source image as a ``uint8`` BGR NumPy array.
        facial5points
            Five 2-D facial landmark coordinates as a flat or ``(2, 5)`` array
            in the order: left-eye, right-eye, nose, left-mouth, right-mouth.

        Returns
        -------
        crop : np.ndarray
            Aligned face crop of shape ``(input_height, input_width, 3)`` in BGR.
        """
        facial5points = np.reshape(facial5points, (2, 5))

        crop_size = (self.input_height, self.input_width)
        output_size = (self.input_height, self.input_width)

        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(
            output_size,
            self.inner_padding_factor,
            self.outer_padding,
            self.default_square,
        )
        return warp_and_crop_face(
            raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size
        )

    def detect_and_align(self, image: np.ndarray) -> np.ndarray:
        """
        Detect the most prominent face in ``image`` and return an aligned BGR crop.

        Runs the compiled RetinaFace component to locate all faces, selects the
        most significant one via ``select_significant_face``, then warps the five
        detected landmarks to a canonical reference frame via ``align_face``.

        Parameters
        ----------
        image
            Input image as a ``uint8`` BGR NumPy array of any resolution.

        Returns
        -------
        crop : np.ndarray
            Aligned face crop of shape ``(input_height, input_width, 3)`` in BGR.
        """
        detections = detect_faces(
            self.face_detector,  # type: ignore[arg-type]
            image[..., ::-1].copy(),  # shared helper takes RGB
            score_threshold=SCORE_THRESHOLD,
        )
        if not detections:
            raise RuntimeError(
                "No face detected in the input image\n"
                "If the image is already a cropped face, pass --already_aligned to skip detection."
            )
        # detect_faces returns (box[4], landmarks[5,2], score) sorted by score;
        # select_significant_face wants (N, 5) boxes with a trailing score column.
        boxes = np.array(
            [np.concatenate([box, [score]]) for box, _lm, score in detections],
            dtype=np.float32,
        )
        i = select_significant_face(boxes)
        # align_face reshapes to (2, 5), i.e. x-row then y-row.
        return self.align_face(image, detections[i][1].T.reshape(-1))

    @classmethod
    def from_pretrained(cls, model: CollectionModel) -> MobileFaceNetApp:
        assert isinstance(model, MobileFaceNet)
        return cls(
            model.face_detector,
            model.face_recognizer,
        )

    @property
    def uses_ondevice_model(self) -> bool:
        """True if any component is an AsyncOnDeviceModel; False if all are local."""
        return isinstance(self.face_detector, AsyncOnDeviceModel) or isinstance(
            self.face_recognizer, AsyncOnDeviceModel
        )

    @classmethod
    def from_components(
        cls,
        models: Sequence[ExecutableModelProtocol] | Sequence[AsyncOnDeviceModel],
    ) -> MobileFaceNetApp:
        """
        Create a MobileFaceNetApp from a list of model components.

        Parameters
        ----------
        models
            List of two components: [face_detector, face_recognizer].

        Returns
        -------
        MobileFaceNetApp
            App initialized with the provided components.
        """
        return cls(
            face_detector=models[0],
            face_recognizer=models[1],
        )

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        # LFW verification is scored on the dataset's own landmark annotations
        # (the standard protocol), so the detector is not in the accuracy path --
        # only the recognizer runs here. The detector is still compiled and
        # profiled as a component, which is what makes the published perf figure
        # cover the deployable pipeline.
        if isinstance(model_input, tuple):
            img1, img2 = model_input[0], model_input[1]
            output = self.face_recognizer(img1, img2)
        else:
            first = next(model_input)
            chunks = first if isinstance(first, tuple) else (first,)
            output = self.face_recognizer(*chunks)

        yield output
        if isinstance(output, AsyncOnDeviceResult):
            output = output.wait()
        return cast("tuple[torch.Tensor, ...] | AsyncOnDeviceResult", output)

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        assert isinstance(collection_model, MobileFaceNet)
        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        dataset = instantiate_dataset(calibration_dataset_cls, DatasetSplit.TRAIN)
        assert isinstance(dataset, LFWDataset)
        num_samples = num_samples or dataset.default_samples_per_job()
        num_samples = min(num_samples, len(dataset))

        if component_name == "face_recognizer":
            # LFW yields pre-aligned 112x112 crop pairs, which is exactly the
            # recognizer's input.
            entries: dict[str, list[np.ndarray]] = {"img1": [], "img2": []}
            for i in range(num_samples):
                (img1, img2), _label = dataset[i]
                entries["img1"].append(img1.unsqueeze(0).numpy())
                entries["img2"].append(img2.unsqueeze(0).numpy())
            return entries

        if component_name == "face_detector":
            # The detector needs full scene images, which the aligned-crop
            # dataset entries do not carry -- read the raw LFW files instead.
            det_spec = (
                input_specs.get("face_detector") if input_specs else None
            ) or collection_model.face_detector.get_input_spec()
            det_h, det_w = det_spec["image"][0][-2:]
            entries = {"image": []}
            for sample in dataset._samples[:num_samples]:
                path = LFW_ASSET.extracted_path / sample["full_path"]
                img_bgr = cv.imread(str(path), cv.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
                img_rgb = cv.resize(img_rgb, (det_w, det_h))
                entries["image"].append(numpy_image_to_torch(img_rgb).numpy())
            return entries

        raise ValueError(f"Unknown component: {component_name}")

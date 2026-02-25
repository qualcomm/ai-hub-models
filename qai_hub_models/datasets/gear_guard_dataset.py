# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file

try:
    from qai_hub_models.utils._internal.download_private_datasets import (
        download_gear_guard_files,
    )
except ImportError:
    download_gear_guard_files = None  # type: ignore[assignment]
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
    transform_resize_pad_coordinates,
)
from qai_hub_models.utils.input_spec import InputSpec

GEARGUARD_DATASET_VERSION = 1
GEARGUARD_DATASET_ID = "gearguard_dataset"
GEARGUARD_DATASET_DIR_NAME = "gearguard_trainvaltest"

VALID_CLASS_IDX = [0, 1]


class GearGuardDataset(BaseDataset):
    """Wrapper class for gear_guard_net dataset

    This class can be subclassed to create variants with different:
    - Annotation formats (override _parse_annotation_data)
    - Box transformation logic (override _transform_and_normalize_boxes)
    - Valid class indices (override _get_valid_class_indices)
    - Image ID generation (override _generate_image_id)
    - Padding parameters (override _get_pad_params)
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        input_data_zip: str | None = None,
        max_boxes: int = 20,
    ) -> None:
        """Initialize the GearGuard dataset.

        Parameters
        ----------
        split
            Which split of the dataset to use (train, validation, or test).

        input_spec
            Model input spec; determines shapes for model input produced by this dataset.

        input_data_zip
            Path to the input data zip file containing the GearGuard dataset.
            If None, the dataset is expected to be already downloaded.

        max_boxes
            The maximum number of bounding boxes for a given sample.
            This parameter ensures consistent tensor dimensions when loading
            multiple samples in a batch via a dataloader.

            If a sample has fewer than this many boxes, the tensor of boxes
            will be zero-padded up to this amount.

            If a sample has more than this many boxes, a ValueError is raised.
        """
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            GEARGUARD_DATASET_ID, GEARGUARD_DATASET_VERSION, "data"
        )

        # input_spec is (h, w) and target_image_size is (w, h)
        input_spec = input_spec or {"image": ((1, 3, 320, 192), "")}
        self.target_h = input_spec["image"][0][2]
        self.target_w = input_spec["image"][0][3]
        self.max_boxes = max_boxes
        self.input_data_zip = input_data_zip

        BaseDataset.__init__(self, self.data_path, split=split, input_spec=input_spec)

    def _get_valid_class_indices(self) -> list[int]:
        """Return the list of valid class indices for this dataset.

        Subclasses can override this to provide different class indices.

        Returns
        -------
        list[int]
            List of valid class indices.
        """
        return VALID_CLASS_IDX

    def _get_pad_params(self) -> tuple[str, float]:
        """Return the padding mode and value for resize_pad.

        Subclasses can override this to provide custom padding parameters.

        Returns
        -------
        str
            Padding mode (e.g., "constant", "edge", "reflect").
        float
            Padding value (used when pad_mode is "constant").
        """
        return "constant", 0.0

    def _parse_annotation_data(
        self, gt_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parse annotation file and return boxes and labels.

        This method handles the v1 annotation format (pixel coordinates).
        Subclasses can override this to handle different annotation formats.

        Parameters
        ----------
        gt_path
            Path to the ground truth annotation file.

        Returns
        -------
        torch.Tensor
            Bounding box coordinates with shape (N, 4).
        torch.Tensor
            Ground truth category IDs with shape (N,).
        """
        try:
            annotation_data = np.genfromtxt(gt_path, dtype=np.float32)
            # Handle empty file
            if annotation_data.size == 0:
                annotation = np.zeros((0, 5), dtype=np.float32)
            # Handle single line
            elif annotation_data.ndim == 1:
                annotation = annotation_data.reshape(1, 5)
            # Handle multiple lines
            else:
                annotation = annotation_data.reshape(-1, 5)
        except Exception as e:
            print(f"Error processing ground truth file {gt_path}: {e}")
            annotation = np.zeros((0, 5), dtype=np.float32)

        labels = (
            torch.Tensor(annotation[:, 0]) if annotation.size > 0 else torch.zeros(0)
        )
        boxes = (
            torch.Tensor(annotation[:, 1:5].astype(np.float32))
            if annotation.size > 0
            else torch.zeros((0, 4))
        )
        return boxes, labels

    def _generate_image_id(self, image_path: Path) -> int:
        """Generate a unique image ID for the given image path.

        Subclasses can override this to use different ID generation methods.

        Parameters
        ----------
        image_path
            Path to the image file.

        Returns
        -------
        int
            Unique image ID.
        """
        return abs(hash(str(image_path.name[:-4]))) % (10**8)

    def _transform_and_normalize_boxes(
        self,
        boxes: torch.Tensor,
        scale: float,
        padding: tuple,
        target_width: int,
        target_height: int,
        orig_width: int | None = None,
        orig_height: int | None = None,
    ) -> torch.Tensor:
        """Transform box coordinates based on scale and padding, then normalize them.

        This method processes bounding boxes in xyxy format (x1, y1, x2, y2) through
        three steps:
        1. Scale: Apply the scale factor from resize operation
        2. Pad: Adjust coordinates for padding added during resize_pad
        3. Normalize: Convert to normalized [0, 1] coordinates relative to target dimensions

        Subclasses can override this to use different transformation logic (e.g., for
        boxes already in normalized format or in different coordinate systems like xywh).

        Parameters
        ----------
        boxes
            Original box coordinates in xyxy format (x1, y1, x2, y2) with shape (N, 4).
            These are pixel coordinates from the original image.
        scale
            Scale factor used during resize_pad
        padding
            Padding values (left, top) added during resize_pad
        target_width
            Target image width in pixel
        target_height
            Target image height in pixel
        orig_width
            Original image width (optional, used by some subclasses)
        orig_height
            Original image height (optional, used by some subclasses)

        Returns
        -------
        torch.Tensor
            Transformed and normalized box coordinates with shape (N, 4)
        """
        # Reshape to (N*2, 2) for coordinate transformation
        coords = boxes.reshape(-1, 2)

        # Apply scale and padding transformation
        transformed_coords = transform_resize_pad_coordinates(coords, scale, padding)

        # Normalize coordinates
        transformed_coords[:, 0] /= target_width
        transformed_coords[:, 1] /= target_height

        # Reshape back to (N, 4)
        return transformed_coords.reshape(-1, 4)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Get an item from the GearGuard dataset at the specified index.

        This method loads an image and its corresponding ground truth annotations,
        processes them to the required format, and returns them as tensors ready
        for model input.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            Input image resized and padded for the network.
            RGB format with floating point values in range [0-1].
            Shape [C, H, W] where C=3, H=input_height, W=input_width.
        ground_truth : tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
            image_id
                Unique identifier for the image within the dataset.
            target_height
                Height of the processed image (equal to input_height).
            target_width
                Width of the processed image (equal to input_width).
            bboxes
                Bounding box coordinates with shape (self.max_boxes, 4).
                Each box is represented as (x1, y1, x2, y2) in normalized [0, 1] coordinates.
                Coordinates are normalized to reflect their positions in the resized and
                padded input image.
            labels
                Ground truth category IDs with shape (self.max_boxes).
                Values correspond to class indices defined in VALID_CLASS_IDX.
            num_boxes
                Number of valid bounding boxes present in the ground truth data.
                Shape [1].

        Raises
        ------
        ValueError
            If the sample contains more bounding boxes than the specified max_boxes.
        """
        image_path = self.image_list[index]
        gt_path = self.gt_list[index]
        image = Image.open(image_path)
        src_image_w, src_image_h = image.size

        # Convert to torch (NCHW, range [0, 1]) tensor.
        torch_image = app_to_net_image_inputs(image)[1]

        # Get padding parameters from subclass
        pad_mode, pad_value = self._get_pad_params()

        # Scale and center-pad image to user-requested target image shape.
        scaled_padded_torch_image, scale_factor, pad = resize_pad(
            torch_image,
            (self.target_h, self.target_w),
            pad_mode=pad_mode,
            pad_value=pad_value,
        )

        # Parse annotation data (subclass-specific)
        boxes, labels = self._parse_annotation_data(gt_path)

        # Validate labels are within expected range
        valid_class_indices = self._get_valid_class_indices()
        valid_labels = np.isin(
            labels.numpy() if isinstance(labels, torch.Tensor) else labels,
            valid_class_indices,
        )
        if not np.all(valid_labels) and len(labels) > 0:
            warnings.warn(
                f"File {gt_path} contains invalid label indices",
                stacklevel=2,
            )
            # Filter out invalid labels
            if isinstance(labels, torch.Tensor):
                labels = labels[torch.from_numpy(valid_labels)]
                boxes = boxes[torch.from_numpy(valid_labels)]

        # Transform box coordinates (subclass-specific)
        if boxes.numel() > 0:
            boxes = self._transform_and_normalize_boxes(
                boxes,
                scale_factor,
                pad,
                self.target_w,
                self.target_h,
                src_image_w,
                src_image_h,
            )

        # Pad the number of boxes to a standard value
        num_boxes = len(labels)
        if num_boxes == 0:
            boxes = torch.zeros((self.max_boxes, 4))
            labels = torch.zeros(self.max_boxes)
        elif num_boxes > self.max_boxes:
            raise ValueError(
                f"Sample has more boxes than max boxes {self.max_boxes}. "
                "Re-initialize the dataset with a larger value for max_boxes."
            )
        else:
            boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - num_boxes), value=0)
            labels = F.pad(labels, (0, self.max_boxes - num_boxes), value=0)

        # Generate image ID (subclass-specific)
        image_id = self._generate_image_id(image_path)

        return scaled_padded_torch_image.squeeze(0), (
            image_id,
            self.target_h,
            self.target_w,
            boxes,
            labels,
            torch.tensor([num_boxes]),
        )

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.image_list)

    def _validate_data(self) -> bool:
        """Validate that the dataset files exist and are properly structured.

        This method checks for the existence of image and ground truth directories,
        and verifies that each image has a corresponding ground truth annotation file.
        It populates the image_list and gt_list attributes with valid file paths.

        Returns
        -------
        bool
            True if all dataset files are valid and properly structured, False otherwise.
        """
        images_path = (
            self.data_path / GEARGUARD_DATASET_DIR_NAME / "images" / self.split_str
        )
        gt_path = (
            self.data_path / GEARGUARD_DATASET_DIR_NAME / "labels" / self.split_str
        )
        if not images_path.exists() or not gt_path.exists():
            return False

        self.image_list: list[Path] = []
        self.gt_list: list[Path] = []
        for _img_path in images_path.iterdir():
            _gt_filename = _img_path.name.replace(".jpg", ".txt")
            _gt_path = gt_path / _gt_filename

            if not _gt_path.exists():
                print(f"Ground truth file not found: {_gt_path!s}")
                return False
            self.image_list.append(_img_path)
            self.gt_list.append(_gt_path)
        return True

    def _download_data(self, zip_path: str | None = None) -> None:
        """Download and extract the GearGuard dataset from a zip file.

        This method extracts the dataset from the provided zip file path
        to the local data directory. If no zip file is provided, it will
        attempt to auto-download the dataset if the internal download
        function is available.

        Parameters
        ----------
        zip_path
            Path to the zip file. If None, uses self.input_data_zip or
            attempts auto-download.

        Raises
        ------
        UnfetchableDatasetError
            If no zip file is available and auto-download is not possible,
            or if the zip file does not point to a valid GearGuard dataset.
        """
        # Use passed arg if provided, otherwise use instance attribute
        if zip_path is None:
            zip_path = self.input_data_zip

        # If no file provided/set, try auto-download
        if zip_path is None and download_gear_guard_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, f"{GEARGUARD_DATASET_DIR_NAME}.zip")
                download_gear_guard_files(zip_path)
                self._download_data(zip_path)
            return

        if zip_path is None or not zip_path.endswith(
            GEARGUARD_DATASET_DIR_NAME + ".zip"
        ):
            raise UnfetchableDatasetError(
                dataset_name=self.dataset_name(),
                installation_steps=None,
            )

        os.makedirs(self.data_path, exist_ok=True)
        extract_zip_file(zip_path, self.data_path / GEARGUARD_DATASET_DIR_NAME)

    @staticmethod
    def default_samples_per_job() -> int:
        """Return the default number of samples to run in each inference job.

        Returns
        -------
        int
            Default number of samples per inference job.
        """
        return 422

    @staticmethod
    def default_num_calibration_samples() -> int:
        """Return the default number of samples to use for calibration.

        Returns
        -------
        int
            Default number of calibration samples.
        """
        return 100

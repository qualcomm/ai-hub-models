# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import torch

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.datasets.gear_guard_dataset import GearGuardDataset
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file

try:
    from qai_hub_models.utils._internal.download_private_datasets import (
        download_gear_guard_v2_files,
    )
except ImportError:
    download_gear_guard_v2_files = None  # type: ignore[assignment]
from qai_hub_models.utils.bounding_box_processing import box_xywh_to_xyxy
from qai_hub_models.utils.image_processing import (
    transform_resize_pad_normalized_coordinates,
)
from qai_hub_models.utils.input_spec import InputSpec

GEARGUARD_V2_DATASET_VERSION = 1
GEARGUARD_V2_DATASET_ID = "gear_guard_v2_dataset"
GEARGUARD_V2_DATASET_DIR_NAME = "gearguard_val"

# Number of PPE detection classes
NUM_CLASSES = 8

# Image preprocessing constants
# These values match the preprocessing used during model training
MODEL_INPUT_PAD_MODE = "constant"
MODEL_INPUT_PAD_VALUE = 114 / 255.0  # Normalized gray value used for padding


class GearGuardV2Dataset(GearGuardDataset):
    """Wrapper class for gear_guard_net_v2 dataset

    This class inherits from GearGuardDataset and overrides specific methods
    to handle the v2 dataset format:
    - YOLO-format annotations (normalized coordinates)
    - 8 PPE classes instead of 2
    - Custom padding parameters
    - MD5-based image ID generation
    - Different validation logic (val.txt file list)
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        input_data_zip: str | None = None,
        max_boxes: int = 20,
    ) -> None:
        """Initialize the GearGuardV2 dataset.

        Parameters
        ----------
        split
            Which split of the dataset to use. Both TRAIN and VAL splits use the same
            validation data since no separate training data is available.

        input_spec
            Input specification dictionary containing image shape information.
            If None, defaults to {"image": ((1, 3, 384, 224), "")}.
            The shape tuple defines (batch_size, channels, height, width).

        input_data_zip
            Path to the input data zip file containing the GearGuardV2 dataset.
            If None, the dataset is expected to be already downloaded.

        max_boxes
            The maximum number of bounding boxes for a given sample.
            This parameter ensures consistent tensor dimensions when loading
            multiple samples in a batch via a dataloader.

            If a sample has fewer than this many boxes, the tensor of boxes
            will be zero-padded up to this amount.

            If a sample has more than this many boxes, a ValueError is raised.
        """
        # Note: Both TRAIN and VAL splits use the same validation data
        # since no separate training data is available for this dataset
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            GEARGUARD_V2_DATASET_ID, GEARGUARD_V2_DATASET_VERSION, "data"
        )
        # input_spec is (h, w) and target_image_size is (w, h)
        input_spec = input_spec or {"image": ((1, 3, 384, 224), "")}
        self.target_h = input_spec["image"][0][2]
        self.target_w = input_spec["image"][0][3]
        self.max_boxes = max_boxes
        self.input_data_zip = input_data_zip
        BaseDataset.__init__(self, self.data_path, split=split, input_spec=input_spec)

    def _get_valid_class_indices(self) -> list[int]:
        """Return the list of valid class indices for v2 dataset.

        Returns
        -------
        list[int]
            List of valid class indices from 0 to 7 (8 PPE classes total).
        """
        return list(range(NUM_CLASSES))

    def _get_pad_params(self) -> tuple[str, float]:
        """Return custom padding parameters for v2 dataset.

        Returns
        -------
        pad_mode : str
            Padding mode ('constant').
        pad_value : float
            Padding value (114/255.0).
        """
        return MODEL_INPUT_PAD_MODE, MODEL_INPUT_PAD_VALUE

    def _generate_image_id(self, image_path: Path) -> int:
        """Generate image ID using MD5 hash for better uniqueness.

        Parameters
        ----------
        image_path
            Path to the image file.

        Returns
        -------
        int
            Integer image ID generated from the first 8 characters of the MD5 hash
            of the relative image path.
        """
        return int(
            hashlib.md5(
                str(image_path.relative_to(self.data_path)).encode()
            ).hexdigest()[:8],
            16,
        )

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
        """Convert box coordinates from normalized xywh to normalized xyxy format.

        This method handles boxes that are in normalized [0, 1] xywh format (YOLO format)
        relative to the original image dimensions, and transforms them to normalized xyxy format
        relative to target image dimensions (after resize and pad operations).

        The transformation process follows these steps:

        1. Convert from normalized xywh format to normalized xyxy format
        2. Convert normalized coordinates to pixel coordinates in original image space
        3. Apply scale and padding transformations from resize_pad operation
        4. Re-normalize coordinates to [0, 1] relative to the target (resized+padded) image

        This ensures the bounding boxes correctly align with the transformed image while
        maintaining normalized coordinate format for the model input.

        Parameters
        ----------
        boxes
            Original box coordinates in xywh format with normalized [0, 1] values,
            shape (N, 4), where each box is (center_x, center_y, width, height).
            These are normalized relative to the original image dimensions (YOLO format).
        scale
            Scale factor used during resize_pad operation
        padding
            Padding values (left, top) added during resize_pad
        target_width
            Target image width in pixels (after resize and pad)
        target_height
            Target image height in pixels (after resize and pad)
        orig_width
            Original image width in pixels (required for v2)
        orig_height
            Original image height in pixels (required for v2)

        Returns
        -------
        torch.Tensor
            Transformed box coordinates with shape (N, 4) in normalized [0, 1] format,
            now relative to the target (resized+padded) image dimensions.
        """
        if orig_width is None or orig_height is None:
            raise ValueError("v2 dataset requires orig_width and orig_height")

        boxes = box_xywh_to_xyxy(boxes)

        # Reshape to (N*2, 2) for coordinate transformation
        coords = boxes.reshape(-1, 2)

        # Transform normalized coordinates to account for resize and padding
        # This utility function handles the conversion from normalized coords in original
        # image space to normalized coords in resized+padded image space
        transformed_coords = transform_resize_pad_normalized_coordinates(
            coords,
            src_image_shape=(orig_width, orig_height),
            resized_image_shape=(target_width, target_height),
            scale_factor=scale,
            pad=padding,
        )

        # Reshape back to (N, 4)
        return transformed_coords.reshape(-1, 4)

    def _validate_data(self) -> bool:
        """Validate v2 dataset structure (uses val.txt file list).

        The dataset is split into TRAIN and VAL subsets:
        - TRAIN split: First 100 samples (for calibration)
        - VAL split: Remaining 250 samples (for evaluation)

        Returns
        -------
        bool
            True if the dataset structure is valid and at least one valid image-label
            pair is found, False otherwise.
        """
        # Construct path to the image list file based on split
        list_file = self.data_path / GEARGUARD_V2_DATASET_DIR_NAME / "val.txt"
        if not list_file.exists():
            print(f"Image list file not found: {list_file!s}")
            return False

        # Read image paths from the list file
        self.image_list: list[Path] = []
        self.gt_list: list[Path] = []

        try:
            with open(list_file) as f:
                image_paths = f.read().splitlines()
        except (OSError, UnicodeDecodeError) as e:
            print(f"Error reading image list file {list_file}: {e}")
            return False

        # Determine which samples to use based on split
        num_calibration_samples = self.default_num_calibration_samples()
        if self.split == DatasetSplit.TRAIN:
            image_paths = image_paths[:num_calibration_samples]
        else:  # VAL split
            image_paths = image_paths[num_calibration_samples:]

        # Process each image path from the list
        for img_path_str in image_paths:
            # Construct full image path
            img_path = self.data_path / GEARGUARD_V2_DATASET_DIR_NAME / img_path_str

            if not img_path.exists():
                print(f"Image file not found: {img_path!s}")
                continue

            # Derive label path by replacing 'images' with 'labels' and extension with '.txt'
            label_path_str = img_path_str.replace("images/", "labels/")
            label_path_str = label_path_str.rsplit(".", 1)[0] + ".txt"
            label_path = self.data_path / GEARGUARD_V2_DATASET_DIR_NAME / label_path_str

            if not label_path.exists():
                print(f"Label file not found: {label_path!s}")
                continue

            self.image_list.append(img_path)
            self.gt_list.append(label_path)

        return len(self.image_list) > 0

    def _download_data(self, zip_path: str | None = None) -> None:
        """Download and extract the gear_guard_v2 dataset.

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
            or if the zip file does not point to a valid GearGuardV2 dataset.
        """
        # Use passed arg if provided, otherwise use instance attribute
        if zip_path is None:
            zip_path = self.input_data_zip

        # If no file provided/set, try auto-download
        if zip_path is None and download_gear_guard_v2_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, f"{GEARGUARD_V2_DATASET_DIR_NAME}.zip")
                download_gear_guard_v2_files(zip_path, GEARGUARD_V2_DATASET_VERSION)
                self._download_data(zip_path)
            return

        if zip_path is None or not zip_path.endswith(
            GEARGUARD_V2_DATASET_DIR_NAME + ".zip"
        ):
            raise UnfetchableDatasetError(
                dataset_name=self.dataset_name(),
                installation_steps=None,
            )

        os.makedirs(self.data_path, exist_ok=True)
        extract_zip_file(zip_path, self.data_path / GEARGUARD_V2_DATASET_DIR_NAME)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job.

        Returns
        -------
        int
            The default number of samples (250) to process per inference job.
        """
        return 250

    @staticmethod
    def default_num_calibration_samples() -> int:
        """The default number of samples to use for calibration (TRAIN split).

        Returns
        -------
        int
            The number of samples (100) to use for calibration.
        """
        return 100

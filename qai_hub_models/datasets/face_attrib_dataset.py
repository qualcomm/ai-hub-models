# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from PIL import Image

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.models.face_attrib_net.model import OUT_NAMES, FaceAttribNet
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file

try:
    from qai_hub_models.utils._internal.download_private_datasets import (
        download_face_attrib_files,
    )
except ImportError:
    download_face_attrib_files = None  # type: ignore[assignment]
from qai_hub_models.utils.image_processing import app_to_net_image_inputs, resize_pad
from qai_hub_models.utils.input_spec import InputSpec

FACEATTRIB_DATASET_VERSION = 3
FACEATTRIB_DATASET_ID = "faceattrib"
FACEATTRIB_DATASET_DIR_NAME = "faceattrib_trainvaltest"


def load_annotations(file_path: Path) -> dict[str, int]:
    """
    Helper function to load label data from an annotation file.

    Parameters
    ----------
    file_path
        The path to the annotation file.

    Returns
    -------
    annotations : dict[str, int]
        A dictionary containing label data:
            - Key : str
                The filename.

            - Value : int
                The attribute value:
                    - 0: Closed/Absent
                    - 1: Opened/Present
    """
    annotations = {}
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) == 2:
                filename, label = parts
                annotations[filename] = int(label)
    return annotations


class FaceAttribDataset(BaseDataset):
    """Wrapper class for face_attrib_net private dataset, for detections output"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        FaceAttribDataset constructor

        Parameters
        ----------
        split
            either DatasetSplit.TRAIN or DatasetSplit.VAL.
            DatasetSplit.TRAIN: training dataset
            DatasetSplit.VAL: validation dataset

        input_data_zip
            path to zip file containing dataset.

        input_spec
            name, shape, type of sample data.
        """
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            FACEATTRIB_DATASET_ID, FACEATTRIB_DATASET_VERSION, "data"
        )
        self.input_data_zip = input_data_zip
        self.image_list: list[Path] = []
        self.gt_list: list[int] = []
        self.attr_index_list: list[int] = []

        input_spec = input_spec or FaceAttribNet.get_input_spec()
        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]
        BaseDataset.__init__(self, self.data_path, split)

    def getitem_image(self, index: int) -> tuple[torch.Tensor, Path]:
        """
        Helper function to retrieve an image and its corresponding file path from the dataset.

        Parameters
        ----------
        index
            The index used to retrieve the image from the dataset.

        Returns
        -------
        image_tensor : torch.Tensor
            A face image with shape (C, H, W), where:
                - C: Number of channels (3 for RGB)
                - H: Height (128 pixels)
                - W: Width (128 pixels)
            The image is scaled and center-padded to match the model's expected input dimensions.
            Pixel values should be in the range [0, 1].
        image_path : Path
            The file path of the retrieved image.
        """
        assert index < len(self.image_list)
        image_path = self.image_list[index]
        image = Image.open(image_path)
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        if image.size != (self.input_width, self.input_height):
            print(f"image size for {image_path} does not fit model, we'll resize it")
            image_tensor = resize_pad(
                image=image_tensor, dst_size=(self.input_height, self.input_width)
            )[0]

        return image_tensor, image_path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple containing an input image tensor and label data.

        Parameters
        ----------
        index
            The index used to retrieve the image from the dataset.

        Returns
        -------
        image_tensor : torch.Tensor
            A face image with shape (C, H, W), where:
                - C: Number of channels (3 for RGB)
                - H: Height (128 pixels)
                - W: Width (128 pixels)
            The image is scaled and center-padded to match the model's expected input dimensions.
            Pixel values should be in the range [0, 1].
        label_data : torch.Tensor
            An integer tensor of shape (5) representing the following attributes in order:
                - left_openness
                - right_openness
                - glasses
                - mask
                - sunglasses

            Value meanings:
                - 0: Closed/Absent
                - 1: Opened/Present
                - -1: not available
        """
        image_tensor, _ = self.getitem_image(index=index)
        label_data: torch.Tensor = torch.full((5,), -1, dtype=torch.int32)

        if self.split == DatasetSplit.VAL:
            assert index < len(self.gt_list)
            assert index < len(self.attr_index_list)
            attr_val = self.gt_list[index]
            attr_index = self.attr_index_list[index]
            label_data[attr_index] = attr_val

        return image_tensor, label_data

    def __len__(self) -> int:
        """
        Return number of samples in the dataset.

        Returns
        -------
        num_samples : int
            number of samples
        """
        return len(self.image_list)

    def _validate_data(self) -> bool:
        """
        Validate and load dataset.

        Returns
        -------
        is_valid : bool
            True: data is valid and loaded successfully
            False: otherwise
        """
        image_path = (
            self.data_path / FACEATTRIB_DATASET_DIR_NAME / "images" / self.split_str
        )
        gt_path = self.data_path / FACEATTRIB_DATASET_DIR_NAME / "labels"
        if not image_path.exists():
            print(f"Missing image {image_path}")
            return False

        if not gt_path.exists() and self.split == DatasetSplit.VAL:
            print(f"Missing ground truth {gt_path}")

        if self.split == DatasetSplit.TRAIN:
            for _img_path in sorted(image_path.iterdir(), key=lambda item: item.name):
                self.image_list.append(_img_path)

        elif self.split == DatasetSplit.VAL:
            for _attr_index, _out_name in enumerate(OUT_NAMES):
                _images_dir = image_path / _out_name
                _gt_txt = gt_path / f"{_out_name}_answer.txt"
                if not _images_dir.exists():
                    print(f"Missing image path for '{_out_name}': {_images_dir}")
                    return False
                if not _gt_txt.exists():
                    print(f"Missing ground truth for '{_out_name}': {gt_path}")
                    return False

                _gt_dict = load_annotations(_gt_txt)

                for _img_file in sorted(
                    _images_dir.iterdir(), key=lambda item: item.name
                ):
                    if _img_file.name in _gt_dict:
                        self.image_list.append(_img_file)
                        self.gt_list.append(_gt_dict[_img_file.name])
                        self.attr_index_list.append(_attr_index)
                    else:
                        print(f"Warning: No annotation found for {_img_file.name}")
                        return False
        else:
            return False

        return True

    def _download_data(self, zip_path: str | None = None) -> None:
        """Helper function to unzip and setup dataset."""
        # Use passed arg if provided, otherwise use instance attribute
        if zip_path is None:
            zip_path = self.input_data_zip

        # If no file provided/set, try auto-download
        if zip_path is None and download_face_attrib_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, f"{FACEATTRIB_DATASET_DIR_NAME}.zip")
                download_face_attrib_files(zip_path, FACEATTRIB_DATASET_VERSION)
                self._download_data(zip_path)
            return

        if zip_path is None or not zip_path.endswith(
            FACEATTRIB_DATASET_DIR_NAME + ".zip"
        ):
            raise UnfetchableDatasetError(
                dataset_name=self.dataset_name(),
                installation_steps=None,
            )

        os.makedirs(self.data_path, exist_ok=True)
        extract_zip_file(zip_path, self.data_path / FACEATTRIB_DATASET_DIR_NAME)

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.

        Returns
        -------
        samples_per_job : int
            samples to run in each inference job.
        """
        return 500

    @staticmethod
    def default_num_calibration_samples() -> int:
        """
        The default value for how many samples to include in quantization job.

        Returns
        -------
        num_calibration_samples : int
            number of samples to include in quantization job.
        """
        return 1000

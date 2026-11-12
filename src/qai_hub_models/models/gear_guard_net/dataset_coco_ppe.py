# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
)
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.download import download_many_urls
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT

DEFAULT_SELECTED_LIST_FILE = (
    QAIHM_PACKAGE_ROOT.parent
    / "qai_hub_models"
    / "models"
    / "gear_guard_net"
    / "coco_2017_select.txt"
)


DATASET_ID = "coco_ppe"
DATASET_ASSET_VERSION = 2


class CocoPPEDataset(BaseDataset):
    r"""
    Subset wrapper around COCO-2017 dataset that filters samples to a provided list.

    The selection list should contain image paths within MSCOCO that include the split
    and filename, for example:
      - 'val2017/000000140556.jpg'
      - 'train2017/000000146786.jpg'
    Backslashes are supported (e.g., 'val2017\\000000140556.jpg') and will be normalized.

    Only entries that exist within the requested split (train or val of COCO-2017)
    will be kept; entries from other splits (e.g., train2014 or test2017) are ignored.

    This dataset can only be used for training.

    By default, list of selected samples is defined DEFAULT_SELECTED_LIST_FILE
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        selected_list_file: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        split
            Whether to use the train or val split of the dataset (COCO-2017).

        input_spec
            Model input spec; determines shapes for model input produced by this dataset.

        selected_list_file
            Optional path to a text file containing one path per line that identifies
            selected images. Lines beginning with '#' are ignored. If neither this nor
            selected_paths is provided, a placeholder path is used: DEFAULT_SELECTED_LIST_FILE
        """
        if split != DatasetSplit.TRAIN:
            raise ValueError("coco_ppe dataset only supports train split.")
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            DATASET_ID, DATASET_ASSET_VERSION, "images"
        )
        # If nothing provided, use the placeholder path so users know where to plug their list
        if selected_list_file is None:
            selected_list_file = str(DEFAULT_SELECTED_LIST_FILE)

        # Store selection inputs for use during filtering
        self.selected_list_file = selected_list_file

        self._resolve_chosen_samples()

        BaseDataset.__init__(self, self.data_path, split, input_spec)
        # input_spec is (h, w) and target_image_size is (w, h)
        if input_spec is not None:
            self.target_h = input_spec["image"][0][2]
            self.target_w = input_spec["image"][0][3]
        else:
            self.target_h = 320
            self.target_w = 192
        self.max_boxes = 1

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        int,
    ]:
        """
        Get a single sample from the dataset.

        This dataset does not contain any annotations, num_boxes of every item will be 0.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        scaled_padded_torch_image : torch.Tensor
            Preprocessed image tensor of shape (C, H, W), range [0, 1].
        label: int
            Hardcoded to 0 since no label is currently resolved.
        """
        filepath = self.data_path / self.chosen_samples[index]["filename"]
        image = Image.open(filepath).convert("RGB")

        # Convert to torch (NCHW, range [0, 1]) tensor.
        torch_image = app_to_net_image_inputs(image)[1]
        # Scale and center-pad image to user-requested target image shape.
        scaled_padded_torch_image, _, _ = resize_pad(
            torch_image, (self.target_h, self.target_w)
        )
        return scaled_padded_torch_image.squeeze(0), 0

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns
        -------
        num_samples : int
            Number of samples in the dataset.
        """
        return len(self.chosen_samples)

    def _validate_data(self) -> bool:
        """
        Validate that the dataset has been properly loaded.

        Returns
        -------
        is_valid : bool
            True if the dataset attribute exists, False otherwise.
        """
        return self.data_path.exists() and len(os.listdir(self.data_path)) > 20

    def _download_data(self) -> None:
        """
        Download the chosen images.

        Raises
        ------
        ValueError
            If no target images are found in the dataset after filtering.
        """
        asyncio.run(
            download_many_urls(self.chosen_samples, self.data_path, "filename", "url")
        )

    @staticmethod
    def default_samples_per_job() -> int:
        """
        Get the default number of samples to process in each inference job.

        Returns
        -------
        samples_per_job : int
            Default number of samples per job (30).
        """
        return 30

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        """
        Get metadata information about the dataset.

        Returns
        -------
        DatasetMetadata
            Metadata object containing dataset information including link and split description.
        """
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description="2017 edition",
        )

    def _resolve_chosen_samples(self) -> None:
        """
        Parse selected_list_file, normalize separators, and return
        a list of image IDs and download urls.

        Sets the field self.chosen_samples with a list of dictionaires with the format:
        [{"coco_url": "https...", "file_name": "000000000013.jpg"}, ...]
        """
        self.chosen_samples: list[dict[str, str]] = []
        split_token = "train2017"
        with open(self.selected_list_file) as f:
            for line in f:
                entry: dict[str, str] = {}
                line = line.strip()
                if not line or line.startswith("#") or split_token not in line:
                    continue
                line = line.replace("\\", "/")
                filename = Path(line).name
                entry["url"] = f"http://images.cocodataset.org/{split_token}/{filename}"
                entry["filename"] = filename
                self.chosen_samples.append(entry)

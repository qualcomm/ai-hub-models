# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
import shutil
from functools import cache
from pathlib import Path

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT


@cache
def get_class_names(dataset_name: str) -> list[str]:
    """
    Read class names from qai_hub_models/labels/<dataset_name>_labels.txt.

    Parameters
    ----------
    dataset_name
        Base name of the labels file (e.g. "coco", "imagenet", "kinetics400").
        The file at ``labels/<dataset_name>_labels.txt`` is read; each non-empty
        line is one class name, in the order the model produces logits.

    Returns
    -------
    list[str]
        Class names, one per line, with surrounding whitespace stripped.
    """
    labels_path = QAIHM_PACKAGE_ROOT / "labels" / f"{dataset_name}_labels.txt"
    with open(labels_path) as f:
        return [line.strip() for line in f if line.strip()]


def write_labels_file(
    dataset_name: str,
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
) -> None:
    """
    Copy a labels file from qai_hub_models/labels/ to output_dir and register
    it in metadata.supplementary_files.

    Parameters
    ----------
    dataset_name
        Base name of the labels file in qai_hub_models/labels/ (e.g. "coco").
        The file at ``labels/<dataset_name>_labels.txt`` is copied.
    output_dir
        Directory where the file should be written.
    metadata
        Model metadata; supplementary_files will be updated.
    """
    out_path = Path(output_dir) / "labels.txt"
    labels_path = QAIHM_PACKAGE_ROOT / "labels" / f"{dataset_name}_labels.txt"
    shutil.copyfile(labels_path, out_path)
    metadata.supplementary_files["labels.txt"] = (
        "Mapping of model prediction indices -> string labels."
    )

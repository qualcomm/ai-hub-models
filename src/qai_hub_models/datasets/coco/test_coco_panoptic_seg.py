# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from qai_hub_models.test.mock_imports import install_mock_packages

# cv2 is only needed to import the coco dataset modules, not by these tests.
install_mock_packages(["cv2"])

from qai_hub_models.datasets.coco.coco import COCO_VAL_DATASET  # noqa: E402
from qai_hub_models.datasets.coco.coco_panoptic_seg import (  # noqa: E402
    COCO_ANNOTATIONS_ASSET,
    CocoPanopticSegmentationDataset,
)
from qai_hub_models.utils.base_dataset import DatasetSplit  # noqa: E402
from qai_hub_models.utils.input_spec import TensorSpec  # noqa: E402

# One synthetic image whose dimensions force math.floor rounding: 101x33
# scaled into an 80x80 letterbox gives a non-integer scaled height/width, so a
# test that assumes integer rounding would compute the wrong crop region.
IMG_WIDTH = 101
IMG_HEIGHT = 33
# The panoptic mask is painted with a single non-zero id so the padded border
# (which must stay 0) is distinguishable from the valid image region.
PANOPTIC_ID = 7


@pytest.fixture
def fake_coco_panoptic(tmp_path: Path) -> Iterator[None]:
    """Build a one-image on-disk COCO-panoptic layout and point the module's
    asset singletons at it, so the dataset loads without a real download.
    """
    val_root = tmp_path / "val_asset"
    ann_root = tmp_path / "ann_asset"
    ann_dir = ann_root / "annotations"
    # CocoPanopticSegmentationDataset expects the doubly-nested panoptic dir.
    panoptic_dir = ann_dir / "panoptic_val2017" / "panoptic_val2017"
    val_root.mkdir(parents=True)
    panoptic_dir.mkdir(parents=True)

    Image.fromarray(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)).save(
        val_root / "img1.jpg"
    )
    Image.fromarray(
        np.full((IMG_HEIGHT, IMG_WIDTH, 3), PANOPTIC_ID, dtype=np.uint8)
    ).save(panoptic_dir / "pan1.png")

    annotations = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": IMG_WIDTH, "height": IMG_HEIGHT}
        ],
        "annotations": [{"image_id": 1, "file_name": "pan1.png", "segments_info": []}],
        "categories": [],
    }
    with open(ann_dir / "panoptic_val2017.json", "w") as f:
        json.dump(annotations, f)

    # COCO_VAL_DATASET and COCO_ANNOTATIONS_ASSET are module-level singletons;
    # repointing their extracted paths makes the dataset load from the fake
    # layout above instead of triggering a real COCO download.
    orig_val = COCO_VAL_DATASET._local_cache_extracted_path
    orig_ann = COCO_ANNOTATIONS_ASSET._local_cache_extracted_path
    COCO_VAL_DATASET._local_cache_extracted_path = val_root
    COCO_ANNOTATIONS_ASSET._local_cache_extracted_path = ann_root
    try:
        yield
    finally:
        COCO_VAL_DATASET._local_cache_extracted_path = orig_val
        COCO_ANNOTATIONS_ASSET._local_cache_extracted_path = orig_ann


def _spec(height: int, width: int) -> dict[str, TensorSpec]:
    return {"image": TensorSpec(shape=(1, 3, height, width), dtype="float32")}


def test_getitem_letterbox_shapes_and_pad_tensor(fake_coco_panoptic: None) -> None:
    """__getitem__ returns a letterboxed image, an (H, W, 3) target mask, and a
    pad *tensor* (not a plain tuple) alongside the image id and scale.
    """
    ds = CocoPanopticSegmentationDataset(input_spec=_spec(80, 80))
    assert len(ds) == 1

    image_tensor, target = ds[0]
    mask, img_id, scale, pad = target

    assert image_tensor.shape == (3, 80, 80)
    assert mask.shape == (80, 80, 3)
    assert img_id == 1

    # pad must be a tensor so DataLoader's default_collate stacks it into a
    # (B, 2) tensor instead of transposing a tuple into two length-B tensors.
    assert isinstance(pad, torch.Tensor)
    assert pad.shape == (2,)

    # 101x33 -> 80x80 letterbox: scale = 80 / 101 (width is the binding side).
    assert scale == pytest.approx(80 / IMG_WIDTH)


def test_getitem_math_floor_rounding_and_padding(fake_coco_panoptic: None) -> None:
    """The panoptic mask is letterboxed with math.floor rounding: the valid
    region matches floor(orig * scale) and the padded border stays zero.
    """
    ds = CocoPanopticSegmentationDataset(input_spec=_spec(80, 80))
    _, (mask, _, scale, pad) = ds[0]

    left_pad, top_pad = int(pad[0]), int(pad[1])
    scaled_w = math.floor(IMG_WIDTH * scale)
    scaled_h = math.floor(IMG_HEIGHT * scale)

    mask_np = mask.numpy()
    valid = mask_np[top_pad : top_pad + scaled_h, left_pad : left_pad + scaled_w]
    # The whole valid region came from the (uniformly PANOPTIC_ID) source mask.
    assert (valid == PANOPTIC_ID).all()

    # Everything outside the valid region is padding and must be zero.
    total_valid = int((mask_np == PANOPTIC_ID).sum())
    assert total_valid == scaled_h * scaled_w * 3
    assert mask_np.size - int((mask_np == 0).sum()) == total_valid


def test_input_spec_drives_height_width(fake_coco_panoptic: None) -> None:
    """H/W are derived from the input_spec image shape."""
    ds = CocoPanopticSegmentationDataset(input_spec=_spec(64, 48))
    assert ds.input_height == 64
    assert ds.input_width == 48

    image_tensor, _ = ds[0]
    assert image_tensor.shape == (3, 64, 48)


def test_non_val_split_rejected() -> None:
    """Only the VAL split is supported (images are fetched for val2017 only)."""
    with pytest.raises(AssertionError, match=r"only supports DatasetSplit\.VAL"):
        CocoPanopticSegmentationDataset(split=DatasetSplit.TRAIN)

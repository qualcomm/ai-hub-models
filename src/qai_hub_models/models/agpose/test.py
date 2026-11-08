# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.models.agpose.app import AGPoseCollectionApp
from qai_hub_models.models.agpose.dataset import get_bbox
from qai_hub_models.models.agpose.demo import DEMO_SCENE_ASSET, FRAME, load_scene
from qai_hub_models.models.agpose.demo import main as demo_main
from qai_hub_models.models.agpose.model import AG_POSE_SAMPLE_INPUTS as GOLDEN_OUTPUTS
from qai_hub_models.models.agpose.model import (
    AGPoseBackbone,
    AGPoseCollection,
    AGPoseRotationHead,
)
from qai_hub_models.utils.asset_loaders import load_numpy


def _load_golden_and_collection() -> tuple[
    np.lib.npyio.NpzFile, AGPoseBackbone, AGPoseRotationHead
]:
    """Load golden outputs and initialise backbone + rotation head in eval mode."""
    golden = load_numpy(GOLDEN_OUTPUTS.fetch())
    collection = AGPoseCollection.from_pretrained()
    backbone = collection.backbone
    rotation_head = collection.rotation_head
    backbone.eval()
    rotation_head.eval()
    return golden, backbone, rotation_head


def _assert_outputs(
    pred_r: torch.Tensor,
    pred_t: torch.Tensor,
    pred_s: torch.Tensor,
    golden: np.lib.npyio.NpzFile,
) -> None:
    """Assert numerical parity of rotation, translation and size against golden."""
    np.testing.assert_allclose(
        pred_r.numpy(), golden["pred_rotation"], atol=1e-3, rtol=1e-3
    )
    np.testing.assert_allclose(
        pred_t.numpy(), golden["pred_translation"], atol=1e-3, rtol=1e-3
    )
    np.testing.assert_allclose(
        pred_s.numpy(), golden["pred_size"], atol=1e-3, rtol=1e-3
    )


def test_numericals() -> None:
    golden, backbone, rotation_head = _load_golden_and_collection()
    with torch.no_grad():
        pre_pose_feat, c = backbone(
            torch.tensor(golden["category_label"]),
            torch.tensor(golden["choose"]),
            torch.tensor(golden["fps1"]),
            torch.tensor(golden["fps2"]),
            torch.tensor(golden["fps3"]),
            torch.tensor(golden["fps4"]),
            torch.tensor(golden["pts"]),
            torch.tensor(golden["rgb"]),
        )
        pred_r, pred_t, pred_s = rotation_head(pre_pose_feat, c)
    _assert_outputs(pred_r, pred_t, pred_s, golden)


def test_demo() -> None:
    demo_main(is_test=True)


def test_task() -> None:
    collection = AGPoseCollection.from_pretrained()
    app = AGPoseCollectionApp(collection.backbone, collection.rotation_head)

    scene_dir = str(DEMO_SCENE_ASSET.fetch(extract=True))
    scene = load_scene(scene_dir, FRAME)
    target_inst = 5
    category_id, _ = scene["meta"][target_inst]
    bool_mask = scene["mask"] == target_inst
    rows = np.any(bool_mask, axis=1)
    cols = np.any(bool_mask, axis=0)
    y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    rmin, rmax, cmin, cmax = get_bbox(np.array([y1, x1, y2, x2]))
    bbox = (rmin, rmax, cmin, cmax)
    rgb_crop = scene["color"][rmin:rmax, cmin:cmax]
    result = app.predict(rgb_crop, scene["depth"], bbox, bool_mask, category_id)
    assert isinstance(result, dict)
    assert "rotation" in result
    assert "translation" in result
    assert "size" in result

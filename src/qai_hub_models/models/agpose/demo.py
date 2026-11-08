# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from qai_hub_models.models.agpose.app import AGPoseCollectionApp
from qai_hub_models.models.agpose.dataset import (
    get_bbox,
    load_depth,
)
from qai_hub_models.models.agpose.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    AGPoseCollection,
)
from qai_hub_models.utils.args import (
    add_output_dir_arg,
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.display import display_or_save_image

DEMO_SCENE_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "agpose_demo_scene.tar.gz"
)
FRAME = "0000"


def load_scene(scene_dir: str, frame: str) -> dict:
    """Load color, depth, mask and metadata for a single NOCS scene frame."""
    color = cv2.imread(f"{scene_dir}/{frame}_color.png")
    if color is None:
        raise RuntimeError(f"Failed to load color image: {scene_dir}/{frame}_color.png")
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depth = load_depth(f"{scene_dir}/{frame}_depth.png")
    mask_raw = cv2.imread(f"{scene_dir}/{frame}_mask.png", cv2.IMREAD_UNCHANGED)
    if mask_raw is None:
        raise FileNotFoundError(f"Failed to load mask: {scene_dir}/{frame}_mask.png")
    # NOCS *_mask.png may be 3-channel — collapse to single channel using
    # the first channel (instance IDs are identical across channels).
    mask = mask_raw[:, :, 0] if mask_raw.ndim == 3 else mask_raw
    meta = {}
    with open(f"{scene_dir}/{frame}_meta.txt") as f:
        for line in f:
            parts = line.strip().split()
            inst_id = int(parts[0])
            cat_id = int(parts[1]) - 1
            name = parts[2]
            meta[inst_id] = (cat_id, name)
    return {"color": color, "depth": depth, "mask": mask, "meta": meta}


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(AGPoseCollection)
    parser = get_on_device_demo_parser(parser)
    add_output_dir_arg(parser)
    parser.add_argument(
        "--scene-dir",
        type=str,
        default=None,
        help="Path to NOCS scene directory (e.g. .../real/test/scene_1).",
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=5,
        help="Instance ID from mask (default: 5=laptop in scene_1/0000).",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    scene_dir = args.scene_dir or str(DEMO_SCENE_ASSET.fetch(extract=True))

    _, (backbone, rotation_head) = demo_model_components_from_cli_args(
        AGPoseCollection, MODEL_ID, args
    )
    app = AGPoseCollectionApp(
        backbone,  # type: ignore[arg-type]
        rotation_head,  # type: ignore[arg-type]
    )

    scene = load_scene(scene_dir, FRAME)

    target_inst = args.instance
    if target_inst not in scene["meta"]:
        raise ValueError(
            f"Instance {target_inst} not found. Available: {list(scene['meta'].keys())}"
        )

    category_id, obj_name = scene["meta"][target_inst]
    bool_mask = scene["mask"] == target_inst
    rows = np.any(bool_mask, axis=1)
    cols = np.any(bool_mask, axis=0)
    y1 = int(np.where(rows)[0][0])
    y2 = int(np.where(rows)[0][-1])
    x1 = int(np.where(cols)[0][0])
    x2 = int(np.where(cols)[0][-1])
    rmin, rmax, cmin, cmax = get_bbox(np.array([y1, x1, y2, x2]))
    bbox = (rmin, rmax, cmin, cmax)
    rgb_crop = scene["color"][rmin:rmax, cmin:cmax]
    result = app.predict(rgb_crop, scene["depth"], bbox, bool_mask, category_id)
    assert isinstance(result, dict)

    vis_image = AGPoseCollectionApp.draw_pose(
        scene["color"],
        result["rotation"],
        result["translation"],
        result["size"],
        app.intrinsics,
    )
    if is_test:
        return

    display_or_save_image(
        Image.fromarray(vis_image),
        args.output_dir,
        filename=f"{obj_name}.png",
    )


if __name__ == "__main__":
    main()

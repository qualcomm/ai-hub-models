# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.datasets.nclt_lidar_place_recognition.nclt_lidar_place_recognition import (
    read_velodyne_sync_bin,
)
from qai_hub_models.models.overlap_transformer.app import OverlapTransformerApp
from qai_hub_models.models.overlap_transformer.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OverlapTransformer,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

# NCLT scans (ODbL, commercial use permitted), picked by real pose distance
# from NCLT's public groundtruth.csv -- not gated behind the KITTI download.
QUERY_SCAN = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "query.bin"
)
REFERENCE_SCAN_REVISIT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "reference_revisit.bin"
)
REFERENCE_SCAN_DISTANT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "reference_distant.bin"
)


def load_scan(asset: CachedWebModelAsset) -> np.ndarray:
    return read_velodyne_sync_bin(asset.fetch())


def overlap_transformer_demo(
    model_cls: type[OverlapTransformer], is_test: bool = False
) -> None:
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser)
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    input_spec = model.get_input_spec()
    app = OverlapTransformerApp(model, input_spec)
    if not is_test:
        print("Model loaded.\n")

    query = load_scan(QUERY_SCAN)
    ref_revisit = load_scan(REFERENCE_SCAN_REVISIT)
    ref_distant = load_scan(REFERENCE_SCAN_DISTANT)

    des_query, des_revisit, des_distant = app.predict(query, ref_revisit, ref_distant)
    sim_revisit = float(1 / (1 + np.linalg.norm(des_query - des_revisit)))
    sim_distant = float(1 / (1 + np.linalg.norm(des_query - des_distant)))

    if not is_test:
        print("LiDAR Place Recognition Similarity Results")
        print("=" * 60)
        print(
            f"  Query vs true revisit (2.99m away, 600 frames later): {sim_revisit:.4f}"
        )
        print(
            f"  Query vs distant scan (336.2m away)                 : {sim_distant:.4f}"
        )
        print("=" * 60)
        print(
            "Ground truth from NCLT's public groundtruth.csv confirms the "
            "revisit scan is a true spatial match for the query and the "
            "distant scan is not."
        )


def main(is_test: bool = False) -> None:
    overlap_transformer_demo(OverlapTransformer, is_test=is_test)


if __name__ == "__main__":
    main()

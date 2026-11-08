# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.netvlad.app import NetVLADApp
from qai_hub_models.models.netvlad.model import MODEL_ID, NetVLAD
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

# Borrow sample images from the osnet sibling — any natural-scene images
# work for demonstrating descriptor cosine similarity.
_OSNET_ID = "osnet"
_OSNET_VER = 1
QUERY_IMAGE = CachedWebModelAsset.from_asset_store(
    _OSNET_ID, _OSNET_VER, "person_a_1.jpg"
)
REFERENCE_IMAGE = CachedWebModelAsset.from_asset_store(
    _OSNET_ID, _OSNET_VER, "person_a_2.jpg"
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(NetVLAD)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--query-image",
        type=str,
        default=QUERY_IMAGE,
        help="Query image (path or URL).",
    )
    parser.add_argument(
        "--reference-image",
        type=str,
        default=REFERENCE_IMAGE,
        help="Reference image to compare against (path or URL).",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(NetVLAD, MODEL_ID, args)
    input_spec = model.get_input_spec()
    app = NetVLADApp(model, input_spec)  # type: ignore[arg-type]

    query_img = load_image(args.query_image)
    ref_img = load_image(args.reference_image)

    descriptors = app.predict_features(query_img, ref_img)
    d_query, d_ref = descriptors[0], descriptors[1]

    sim = float(np.dot(d_query, d_ref))

    if not is_test:
        print("NetVLAD Place Descriptor Similarity")
        print("=" * 40)
        print(f"  Query image    : {args.query_image}")
        print(f"  Reference image: {args.reference_image}")
        print(
            f"  Cosine similarity: {sim:.4f}  (1.0 = identical place, 0.0 = unrelated)"
        )
        print("=" * 40)
        print(f"  Descriptor dim : {d_query.shape[0]}")


if __name__ == "__main__":
    main()

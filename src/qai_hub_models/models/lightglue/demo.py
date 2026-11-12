# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from PIL import Image

from qai_hub_models.models.lightglue.app import LightGlueApp
from qai_hub_models.models.lightglue.model import (
    IMAGE0_ADDRESS,
    IMAGE1_ADDRESS,
    MODEL_ID,
    SuperPointLightGlue,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.display import display_or_save_image


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(SuperPointLightGlue)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image0",
        type=str,
        default=None,
        help="Path or URL of the first input image. Defaults to a bundled sample.",
    )
    parser.add_argument(
        "--image1",
        type=str,
        default=None,
        help="Path or URL of the second input image. Defaults to a bundled sample.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    _, (superpoint_model, lightglue_model) = demo_model_components_from_cli_args(
        SuperPointLightGlue, MODEL_ID, args
    )
    # input_spec is omitted (not just "unavailable" on an on-device-wrapped
    # component) -- MAX_KEYPOINTS is this recipe's only supported padding
    # width, so LightGlueApp's own default already matches it.
    app = LightGlueApp(
        lightglue_model,  # type: ignore[arg-type]
        extractor=superpoint_model,  # type: ignore[arg-type]
    )

    image0 = load_image(args.image0 if args.image0 else IMAGE0_ADDRESS)
    image1 = load_image(args.image1 if args.image1 else IMAGE1_ADDRESS)
    canvas = app.predict(image0, image1)
    assert isinstance(canvas, np.ndarray)

    if not is_test:
        display_or_save_image(
            Image.fromarray(canvas), args.output_dir, "lightglue_matches.png", MODEL_ID
        )


if __name__ == "__main__":
    main()

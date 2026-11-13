# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from PIL import Image

from qai_hub_models.models.centerpoint import App
from qai_hub_models.models.centerpoint.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CenterPoint,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

BIN_PATH = "data/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"
INPUT_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, BIN_PATH
).fetch()


def centerpoint_demo(
    model_type: type[CenterPoint],
    model_id: str,
    default_input: str,
    is_test: bool = False,
) -> None:
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--bin-path",
        type=str,
        default=default_input,
        help="Path for the lidar bin file",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_type, model_id, args)
    validate_on_device_demo_args(args, model_id)
    app = App(model, model_type.load_config())
    res = app.preprocess_bin_file(args.bin_path)
    image = app.predict(res)
    if not is_test and isinstance(image, Image.Image):
        display_or_save_image(image)


def main(is_test: bool = False) -> None:
    centerpoint_demo(CenterPoint, MODEL_ID, INPUT_ASSET, is_test)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from PIL import Image

from qai_hub_models.models.statetransformer.app import StateTransformerApp as App
from qai_hub_models.models.statetransformer.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MODEL_PATH,
    StateTransformer,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MAP_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "nuplan-maps-v1.0.zip"
)

DATA_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "nuplan-v1.1_test.zip"
)


def state_transformer_demo(
    model_type: type[StateTransformer],
    model_id: str,
    model_path: str,
    data_path: str,
    map_path: str,
    is_test: bool = False,
) -> None:
    """
    Runs a demo pipeline for the StateTransformer model.

    Steps
    -----
    1. Parse CLI arguments (or use defaults if testing).
    2. Load the pretrained model.
    3. Initialize the application wrapper.
    4. Extract model-ready input samples from NuPlan scenarios.
    5. Run inference and optionally print the output.

    Parameters
    ----------
    model_type
        The class type of the model to instantiate.

    model_id
        Identifier for the model instance.

    model_path
        Path to the pretrained model weights.

    data_path
        Path to the NuPlan scenario data.


    map_path
        Path to the NuPlan map data.

    is_test
        If True, disables CLI parsing and suppresses output printing.
        Default is False.
    """
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)

    parser.add_argument(
        "--model-path",
        type=str,
        default=model_path,
        help="Path to the pretrained model",
    )
    parser.add_argument(
        "--db-file-path",
        type=str,
        default=data_path,
        help="Path to the NuPlan scenario .db files directoty",
    )
    parser.add_argument(
        "--map-path",
        type=str,
        default=map_path,
        help="Path to the NuPlan map data from dataset",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)
    # Load the model from pretrained weights
    model = demo_model_from_cli_args(model_type, model_id, args)

    # Initialize the application wrapper
    app = App(model)

    # Run inference using the model
    high_res_raster, low_res_raster, context_actions = app.extract_model_samples(
        args.model_path, args.db_file_path, args.map_path
    )
    img = app.predict(high_res_raster, low_res_raster, context_actions)
    if not is_test and isinstance(img, Image.Image):
        display_or_save_image(img)


def main(is_test: bool = False) -> None:
    """
    Entry point for running the StateTransformer demo.

    Parameters
    ----------
    is_test
        If True, runs in test mode without printing.
    """
    state_transformer_demo(
        StateTransformer, MODEL_ID, MODEL_PATH, DATA_PATH, MAP_PATH, is_test
    )


if __name__ == "__main__":
    main()

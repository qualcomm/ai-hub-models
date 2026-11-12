# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os

from PIL.Image import Image

from qai_hub_models import TargetRuntime
from qai_hub_models.models.templates.cityscapes_segmentation.app import (
    CityscapesSegmentationApp,
)
from qai_hub_models.models.templates.cityscapes_segmentation.model import (
    TEST_CITYSCAPES_LIKE_IMAGE_ASSET,
    TEST_CITYSCAPES_LIKE_IMAGE_NAME,
    CityscapesSegmentor,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_model_input_spec_parser,
    get_on_device_demo_parser,
    input_spec_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image


# Run Imagenet Classifier end-to-end on a sample image.
# The demo will print the predicted class to terminal.
def cityscapes_segmentation_demo(
    model_type: type[CityscapesSegmentor],
    model_id: str,
    is_test: bool = False,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_model_input_spec_parser(model_type, parser)
    parser = get_on_device_demo_parser(
        parser, available_target_runtimes=[TargetRuntime.TFLITE], add_output_dir=True
    )
    parser.add_argument(
        "--image",
        type=str,
        help="File path or URL to an input image to use for the demo.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    if args.image is None:
        image = TEST_CITYSCAPES_LIKE_IMAGE_ASSET.fetch()
        image_name = TEST_CITYSCAPES_LIKE_IMAGE_NAME
    else:
        image = args.image
        image_name = os.path.basename(image)

    inference_model = demo_model_from_cli_args(model_type, model_id, args)
    input_spec = input_spec_from_cli_args(inference_model, args)
    app = CityscapesSegmentationApp(inference_model, input_spec)  # type: ignore[arg-type]

    # Run app
    orig_image = load_image(image)
    image_annotated = app.predict(orig_image)
    assert isinstance(image_annotated, Image)

    if not is_test:
        display_or_save_image(
            image_annotated,
            args.output_dir,
            "annotated_" + image_name,
            "predicted image",
        )

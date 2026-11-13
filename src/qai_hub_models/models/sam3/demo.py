# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

from qai_hub_models.models.sam3.app import SAM3App
from qai_hub_models.models.sam3.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAM3,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.sam.app import SAMInputImageLayout
from qai_hub_models.utils.args import (
    add_output_dir_arg,
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "input_image.jpg"
)
DEFAULT_PROMPT = "cup"


def main(is_test: bool = False) -> None:
    """Run SAM3 text-based segmentation demo.

    Parameters
    ----------
    is_test
        If True, runs in test mode with default parameters.
    """
    # Demo parameters
    parser = get_model_cli_parser(SAM3)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="Image file path or URL",
    )
    parser.add_argument(
        "--text-prompts",
        type=str,
        default=DEFAULT_PROMPT,
        help="Comma-separated text prompts for segmentation (e.g., 'cup,person,car')",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for filtering predictions",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for class-aware NMS over per-query predictions.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        help=(
            "Threshold applied to raw mask logits before binarization. "
            "0.0 (default) is equivalent to sigmoid(logits) > 0.5; "
            "raise for tighter masks, lower for greedier masks."
        ),
    )
    add_output_dir_arg(parser)
    get_on_device_demo_parser(parser)

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    # Parse text prompts; the head is single-prompt, so the app loops
    # the head once per entry. Reject empty input.
    text_prompts = [prompt.strip() for prompt in args.text_prompts.split(",")]
    assert len(text_prompts) >= 1, "--text-prompts must contain at least one prompt"

    sam3, (sam3_vision_backbone, sam3_head) = demo_model_components_from_cli_args(
        SAM3, MODEL_ID, args
    )

    image_height, image_width = sam3.vision_backbone.get_input_spec()["image"][0][-2:]
    head: Any = sam3.head
    tokenizer = head.language_model.tokenizer
    context_length = int(head.language_model.context_length)
    app = SAM3App(
        input_image_channel_layout=SAMInputImageLayout["RGB"],
        sam3_vision_backbone=sam3_vision_backbone,  # type: ignore[arg-type]
        sam3_head=sam3_head,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        context_length=context_length,
        image_height=image_height,
        image_width=image_width,
        mask_threshold=args.mask_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        device=args.model_device,
    )

    # Load image
    image = load_image(args.image)

    # Run text-based segmentation
    print(f"\n** Performing text-based segmentation with prompts: {text_prompts} **\n")
    output_image = app.predict_mask_from_text(
        image,
        text_prompts=text_prompts,
        confidence_threshold=args.confidence_threshold,
    )

    if not is_test:
        display_or_save_image(
            output_image,
            output_dir=args.output_dir,
            filename="sam3_output.png",
            desc="SAM3 segmentation result",
        )


if __name__ == "__main__":
    main()

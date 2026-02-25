# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL.Image import Image

from qai_hub_models.models.maskrcnn.app import MaskRCNNApp
from qai_hub_models.models.maskrcnn.model import MODEL_ASSET_VERSION, MODEL_ID, MaskRCNN
from qai_hub_models.utils.args import (
    add_output_dir_arg,
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
)
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.evaluate import EvalMode

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image.jpg"
)


# Run MaskRCNN app end-to-end on a sample image.
def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(MaskRCNN)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="test image file path or URL",
    )
    parser.add_argument(
        "--proposal_iou_threshold",
        type=float,
        default=0.7,
        help="Proposal IoU threshold",
    )
    parser.add_argument(
        "--boxes_iou_threshold", type=float, default=0.5, help="Boxes IoU threshold"
    )
    parser.add_argument(
        "--boxes_score_threshold",
        type=float,
        default=0.5,
        help="Boxes score threshold (ROI heads already apply NMS)",
    )
    parser.add_argument(
        "--mask_threshold", type=float, default=0.05, help="Mask binarization threshold"
    )
    parser.add_argument(
        "--max_det_pre_nms",
        type=int,
        default=6000,
        help="Maximum Proposal detections before NMS (torchvision testing default: 1000)",
    )
    parser.add_argument(
        "--max_det_post_nms",
        type=int,
        default=200,
        help="Maximum Proposal detections after NMS (torchvision testing default: 1000)",
    )
    add_output_dir_arg(parser)
    get_on_device_demo_parser(parser)

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    wrapper = MaskRCNN.from_pretrained()

    if args.eval_mode == EvalMode.ON_DEVICE:
        proposal_generator, roi_head = demo_model_components_from_cli_args(
            MaskRCNN, MODEL_ID, args
        )
    else:
        proposal_generator, roi_head = wrapper.proposal_generator, wrapper.roi_head

    input_spec = wrapper.proposal_generator.get_input_spec()
    height, width = input_spec["image"][0][2:]

    app = MaskRCNNApp(
        proposal_generator,  # type: ignore[arg-type]
        roi_head,  # type: ignore[arg-type]
        model_image_height=height,
        model_image_width=width,
        proposal_iou_threshold=args.proposal_iou_threshold,
        boxes_iou_threshold=args.boxes_iou_threshold,
        boxes_score_threshold=args.boxes_score_threshold,
        mask_threshold=args.mask_threshold,
        max_det_pre_nms=args.max_det_pre_nms,
        max_det_post_nms=args.max_det_post_nms,
    )

    img = load_image(args.image)
    pred_images = app.predict(img)

    # Show the predicted boxes and masks on the image.
    if not is_test:
        for i, pred_image in enumerate(pred_images):
            assert isinstance(pred_image, Image)
            display_or_save_image(pred_image, args.output_dir, f"image_{i}.png")


if __name__ == "__main__":
    main()

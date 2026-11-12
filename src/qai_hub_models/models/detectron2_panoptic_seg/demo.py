# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL.Image import Image

from qai_hub_models.models.detectron2_panoptic_seg.app import (
    Detectron2PanopticSegApp,
)
from qai_hub_models.models.detectron2_panoptic_seg.evaluator import (
    DEFAULT_BOXES_IOU_THRESHOLD,
    DEFAULT_BOXES_SCORE_THRESHOLD,
    DEFAULT_MAX_DET_POST_NMS,
)
from qai_hub_models.models.detectron2_panoptic_seg.model import (
    MODEL_ID,
    Detectron2PanopticSeg,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.detectron2.model import IMAGE_ADDRESS
from qai_hub_models.utils.args import (
    add_output_dir_arg,
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(Detectron2PanopticSeg)
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
        "--boxes_iou_threshold",
        type=float,
        default=DEFAULT_BOXES_IOU_THRESHOLD,
        help="Boxes IoU threshold",
    )
    parser.add_argument(
        "--boxes_score_threshold",
        type=float,
        default=DEFAULT_BOXES_SCORE_THRESHOLD,
        help="Boxes score threshold",
    )
    parser.add_argument(
        "--max_det_pre_nms",
        type=int,
        default=1000,
        help="Maximum Proposal detections before NMS",
    )
    parser.add_argument(
        "--max_det_post_nms",
        type=int,
        default=DEFAULT_MAX_DET_POST_NMS,
        help="Maximum Proposal detections after NMS",
    )
    add_output_dir_arg(parser)
    get_on_device_demo_parser(parser)

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    wrapper, (proposal_generator, roi_head) = demo_model_components_from_cli_args(
        Detectron2PanopticSeg, MODEL_ID, args
    )

    input_spec = wrapper.proposal_generator.get_input_spec()

    app = Detectron2PanopticSegApp(
        proposal_generator=proposal_generator,
        roi_head=roi_head,
        proposal_iou_threshold=args.proposal_iou_threshold,
        boxes_iou_threshold=args.boxes_iou_threshold,
        boxes_score_threshold=args.boxes_score_threshold,
        max_det_pre_nms=args.max_det_pre_nms,
        max_det_post_nms=args.max_det_post_nms,
        input_spec=input_spec,
    )

    img = load_image(args.image)
    pred_images = app.predict(img)

    if not is_test:
        for i, pred_image in enumerate(pred_images):
            assert isinstance(pred_image, Image)
            display_or_save_image(pred_image, args.output_dir, f"image_{i}.png")


if __name__ == "__main__":
    main()

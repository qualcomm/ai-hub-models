# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from PIL import Image

from qai_hub_models import TargetRuntime
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.corevision.labels import get_class_names
from qai_hub_models.models.templates.yolo.app import (
    YoloOBBApp,
    YoloObjectDetectionApp,
    YoloPoseApp,
    YoloSegmentationApp,
    YoloWorldPromptDetectionApp,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_model_input_spec_parser,
    get_on_device_demo_parser,
    input_spec_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_model import BaseModel


# Run Yolo end-to-end on a sample image.
# The demo will display a image with the predicted bounding boxes.
def yolo_detection_demo(
    model_type: type[BaseModel],
    model_id: str,
    app_type: Callable[..., YoloObjectDetectionApp],
    default_image: str | CachedWebAsset,
    stride_multiple: int | None = None,
    is_test: bool = False,
    default_score_threshold: float = 0.45,
    default_iou_threshold: float = 0.7,
    output_fn: Callable[[YoloObjectDetectionApp, Any, Any], None] | None = None,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_model_input_spec_parser(model_type, parser)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    image_help = "image file path or URL."
    if stride_multiple:
        image_help = f"{image_help} Image spatial dimensions (x and y) must be multiples of {stride_multiple}."
    parser.add_argument("--image", type=str, default=default_image, help=image_help)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=default_score_threshold,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=default_iou_threshold,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    args = parser.parse_args([] if is_test else None)

    validate_on_device_demo_args(args, model_id)

    model = demo_model_from_cli_args(model_type, model_id, args)
    input_spec = input_spec_from_cli_args(model, args)

    app = app_type(
        model,
        args.score_threshold,
        args.iou_threshold,
        args.include_postprocessing,
        input_spec=input_spec,
    )

    print("Model Loaded")
    image = load_image(args.image)
    if output_fn is not None:
        output_fn(app, image, args)
    else:
        pred_images = app.predict_boxes_from_image(image, False)
        out = Image.fromarray(pred_images[0])
        if not is_test:
            display_or_save_image(out, args.output_dir, "yolo_demo_output.png")


# Run Yolo end-to-end on a sample image.
# The demo will display a image with the predicted bounding boxes.
def yolo_prompt_detect_demo(
    model_type: type[BaseModel | WorkbenchModelCollection],
    model_id: str,
    app_type: type[YoloWorldPromptDetectionApp],
    default_image: str | CachedWebAsset,
    stride_multiple: int | None = None,
    is_test: bool = False,
    default_score_threshold: float = 0.45,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    image_help = "image file path or URL."
    if stride_multiple:
        image_help = f"{image_help} Image spatial dimensions (x and y) must be multiples of {stride_multiple}."
    parser.add_argument("--image", type=str, default=default_image, help=image_help)
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help=(
            "Comma separated list of all classes for the model to export/demo "
            "execution. Defaults to the full COCO-80 class list."
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=default_score_threshold,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    args = parser.parse_args([] if is_test else None)

    if args.prompt_text is None:
        requested_classes = get_class_names("coco")
    elif args.prompt_text.endswith(".txt"):
        with open(args.prompt_text) as f:
            requested_classes = [t.rstrip("\r\n").strip() for t in f if t.strip()]
    else:
        requested_classes = [
            t.strip() for t in args.prompt_text.split(",") if t.strip()
        ]
    args.num_classes = len(requested_classes)

    validate_on_device_demo_args(args, model_id)

    if issubclass(model_type, WorkbenchModelCollection):
        # Collection model (e.g. YoloWorld): two components — detector + text_encoder.
        # from_components is defined on YoloWorldDetectionApp (subclass of app_type)
        _, components = demo_model_components_from_cli_args(model_type, model_id, args)
        app = cast(Any, app_type).from_components(
            list(components), prompt_text=requested_classes
        )
    else:
        # Single-model path (YoloWorldDetector used standalone).
        model = demo_model_from_cli_args(model_type, model_id, args)
        app = app_type(
            model,  # type: ignore[arg-type]
            args.score_threshold,
            args.iou_threshold,
            args.include_postprocessing,
        )

    print("Model Loaded")
    image = load_image(args.image)
    pred_images = app.predict_boxes_from_image(image, prompt_text=requested_classes)
    out = Image.fromarray(pred_images[0])
    if not is_test:
        display_or_save_image(out, args.output_dir, "yolo_demo_output.png")


def yolo_segmentation_demo(
    model_type: type[BaseModel],
    model_id: str,
    default_image: str | CachedWebAsset,
    default_score_threshold: float = 0.45,
    stride_multiple: int | None = None,
    is_test: bool = False,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(
        parser, available_target_runtimes=[TargetRuntime.TFLITE], add_output_dir=True
    )
    image_help = "image file path or URL."
    if stride_multiple:
        image_help = f"{image_help} Image spatial dimensions (x and y) must be multiples of {stride_multiple}."

    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="Test image file path or URL",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=default_score_threshold,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Load image & model
    model = demo_model_from_cli_args(model_type, model_id, args)
    app = YoloSegmentationApp(
        model,  # type: ignore[arg-type]
        args.score_threshold,
        args.iou_threshold,
    )

    print("Model Loaded")

    image = load_image(args.image)
    image_annotated = app.predict_segmentation_from_image(image)[0]
    assert isinstance(image_annotated, Image.Image)

    if not is_test:
        display_or_save_image(image_annotated, args.output_dir)


def yolo_pose_estimation_demo(
    model_type: type[BaseModel],
    model_id: str,
    app_type: Callable[..., YoloPoseApp],
    default_image: str | CachedWebAsset,
    output_filename: str = "yolo_pose_demo_output.png",
    is_test: bool = False,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    model = demo_model_from_cli_args(model_type, model_id, args)
    print("Model Loaded")
    image = load_image(args.image)

    app = app_type(model)
    keypoint_images = app.predict_pose_keypoints(image)
    if not is_test:
        display_or_save_image(
            keypoint_images[0],
            args.output_dir,
            output_filename,
            "keypoints",
        )


def yolo_obb_demo(
    model_type: type[BaseModel],
    model_id: str,
    app_type: Callable[..., YoloOBBApp],
    default_image: str | CachedWebAsset,
    stride_multiple: int | None = None,
    is_test: bool = False,
    default_score_threshold: float = 0.5,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)

    image_help = "image file path or URL."
    if stride_multiple:
        image_help = (
            f"{image_help} Image spatial dimensions (x and y) must be multiples "
            f"of {stride_multiple}."
        )

    parser.add_argument("--image", type=str, default=default_image, help=image_help)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=default_score_threshold,
        help="Score threshold for OBB NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.1,
        help="Rotated IoU threshold for OBB NonMaximumSuppression",
    )

    args = parser.parse_args([] if is_test else None)

    validate_on_device_demo_args(args, model_id)

    model = demo_model_from_cli_args(model_type, model_id, args)

    # OBB-specific app (handles rotated boxes internally)
    app = app_type(
        model,
        args.score_threshold,
        args.iou_threshold,
    )

    print("OBB Model Loaded")

    image = load_image(args.image)

    pred_images = app.predict_obb_from_image(image)

    out = pred_images[0]
    assert isinstance(out, Image.Image)

    if not is_test:
        display_or_save_image(out, args.output_dir, "yolo_obb_demo_output.png")

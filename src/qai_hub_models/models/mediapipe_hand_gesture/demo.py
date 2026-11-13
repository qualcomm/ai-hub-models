# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

import numpy as np
from PIL import Image

from qai_hub_models.models.mediapipe_hand_gesture.app import MediaPipeHandGestureApp
from qai_hub_models.models.mediapipe_hand_gesture.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediaPipeHandGesture,
)
from qai_hub_models.models.templates.corevision.camera_capture import (
    capture_and_display_processed_frames,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.evaluate.helpers import EvalMode

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "thumbs_up.jpg"
)


# Run Mediapipe Hand Gesture Classifier end-to-end on a sample image or camera stream.
# The demo will display output with the predicted landmarks,bounding boxes and label drawn.
def mediapipe_hand_gesture_demo(
    model_cls: type[MediaPipeHandGesture], is_test: bool = False
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="image file path or URL",
    )
    parser.add_argument(
        "--use-default-image",
        action="store_true",
        help="Use the default test image instead of camera",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera Input ID",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.95,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    parser = get_on_device_demo_parser(parser, add_output_dir=True)

    print(
        "Note: This demo is running through torch, and not meant to be real-time without dedicated ML hardware."
    )
    print("Use Ctrl+C in your terminal to exit.")

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    # Handle default behavior: use camera unless --use-default-image or --image is specified
    if args.use_default_image or is_test:
        args.image = INPUT_IMAGE_ADDRESS

    if args.eval_mode == EvalMode.ON_DEVICE and not args.image:
        raise ValueError(
            "On-device demo mode is not supported with camera input. "
            "Please provide an image using --image or --use-default-image."
        )
    torch_model, (palm_detector, landmark_detector, gesture_classifier) = (
        demo_model_components_from_cli_args(MediaPipeHandGesture, MODEL_ID, args)
    )

    # Load app
    app = MediaPipeHandGestureApp(
        palm_detector,  # type: ignore[arg-type]
        landmark_detector,  # type: ignore[arg-type]
        torch_model.palm_detector.anchors,
        torch_model.palm_detector.include_postprocessing,
        torch_model.palm_detector.get_input_spec(),
        torch_model.hand_landmark_detector.get_input_spec(),
        gesture_classifier,  # type: ignore[arg-type]
    )
    # Load app
    print("Model and App Loaded")

    if args.image:
        image = load_image(args.image)
        pred_image = app.predict_landmarks_from_image(image)
        assert isinstance(pred_image[0], np.ndarray)
        out_image = Image.fromarray(pred_image[0], "RGB")
        if not is_test:
            display_or_save_image(out_image, args.output_dir)
    else:

        def frame_processor(frame: np.ndarray) -> np.ndarray:
            return cast(np.ndarray, app.predict_landmarks_from_image(frame)[0])

        capture_and_display_processed_frames(
            frame_processor, "QAIHM Mediapipe Hand Demo", args.camera
        )


def main(is_test: bool = False) -> None:
    mediapipe_hand_gesture_demo(MediaPipeHandGesture, is_test)


if __name__ == "__main__":
    main()

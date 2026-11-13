# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import cv2
import numpy as np
import psutil
import torch
from mobile_sam.utils.transforms import ResizeLongestSide

from qai_hub_models.models.mobilesam.model import (
    DEFAULT_MODEL_TYPE,
    MobileSAM,
)
from qai_hub_models.models.templates.corevision.display import (
    generate_video_from_frames,
)
from qai_hub_models.models.templates.sam.app import SAMApp, SAMInputImageLayout
from qai_hub_models.models.track_anything.app import TrackAnythingApp
from qai_hub_models.models.track_anything.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    TrackAnythingWrapper,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

VIDEO_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "demo.mp4"
)


def generate_frames_from_video(video_path: str) -> list[np.ndarray]:
    """
    Generates list of frames from a video.

    Parameters
    ----------
    video_path
        The path of the input video.

    Returns
    -------
    list[np.ndarray]
        Frames generated from a video.
    """
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            current_memory_usage = psutil.virtual_memory().percent
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if current_memory_usage > 90:
                print(
                    "Memory usage is too high (>90%). Please reduce the video resolution or frame rate."
                )
                break
        else:
            break
    return frames


def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(TrackAnythingWrapper)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)

    parser.add_argument(
        "--video",
        type=str,
        help="video file path or URL.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="Number of frames to track. Minimum 5. Defaults to track all frames.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        help=f"SAM model type to load. Tested with model type `{DEFAULT_MODEL_TYPE}`.",
    )
    parser.add_argument(
        "--point-coordinates",
        type=str,
        default="280,120;",
        help="Comma separated x and y coordinate. Multiple coordinate separated by `;`."
        " e.g. `x1,y1;x2,y2`. Default: `280,120;`",
    )
    parser.add_argument(
        "--single-mask-mode",
        type=bool,
        default=True,
        help="If True, returns single mask. For multiple points multiple masks could lead to better results.",
    )
    args = parser.parse_args(["--model-type", DEFAULT_MODEL_TYPE] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    coordinates: list[str] = list(filter(None, args.point_coordinates.split(";")))

    # Load SAM Application
    sam_wrapper = MobileSAM.from_pretrained(args.model_type)
    sam_app = SAMApp(
        sam_wrapper.sam.image_encoder.img_size,
        sam_wrapper.sam.mask_threshold,
        SAMInputImageLayout[sam_wrapper.sam.image_format],
        [sam_wrapper.encoder],
        sam_wrapper.decoder,
        ResizeLongestSide,
        sam_wrapper.sam.pixel_mean,
    )

    video_input = str(VIDEO_ADDRESS.fetch()) if args.video is None else str(args.video)

    # Get first frame
    frames = generate_frames_from_video(video_input)
    first_frame = frames[0]

    # Point segmentation using sam decoder
    print("\n** Performing point segmentation **\n")

    # Input points
    input_coords = []
    input_labels = []

    for coord in coordinates:
        coord_split = coord.split(",")
        if len(coord_split) != 2:
            raise RuntimeError(
                f"Expecting comma separated x and y coordinate. Provided {coord_split}."
            )

        input_coords.append([int(coord_split[0]), int(coord_split[1])])
        # Set label to `1` to include current point for segmentation
        input_labels.append(1)

    # Generate masks with given input points
    generated_mask_pt, *_ = sam_app.predict_mask_from_points(
        first_frame, torch.Tensor(input_coords), torch.Tensor(input_labels)
    )
    generated_mask = generated_mask_pt.squeeze(0).squeeze(0).numpy()

    # load TrackAnything Application
    (
        wrapper,
        (
            encode_key_with_shrinkage,
            encode_value,
            encode_key_without_shrinkage,
            segment,
        ),
    ) = demo_model_components_from_cli_args(TrackAnythingWrapper, MODEL_ID, args)

    enc_shape = wrapper.EncodeValue.get_input_spec()["image"][0]
    input_shape = (enc_shape[-2], enc_shape[-1])
    app = TrackAnythingApp(
        encode_key_with_shrinkage,  # type: ignore[arg-type]
        encode_value,  # type: ignore[arg-type]
        encode_key_without_shrinkage,  # type: ignore[arg-type]
        segment,  # type: ignore[arg-type]
        wrapper.config,
        model_input_shape=input_shape,
    )

    # Run inference
    print("\n** Tracking object... **\n")
    num_frames = args.num_frames
    if num_frames is not None:
        frames = frames[:num_frames]
    painted_images = app.track(frames, generated_mask)

    if not is_test:
        generate_video_from_frames(
            painted_images,
            output_dir=args.output_dir,
            output_filename="out_painted_image.mp4",
            fps=30,
        )


if __name__ == "__main__":
    main()

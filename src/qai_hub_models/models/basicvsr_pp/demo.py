# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from qai_hub_models.models.basicvsr_pp.app import (
    BasicVSRPlusPlusApp,
    load_frames_from_dir,
    load_frames_from_video,
    pil_list_to_tensor,
)
from qai_hub_models.models.basicvsr_pp.model import (
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_WIDTH,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    BasicVSRPlusPlus,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.evaluate.helpers import EvalMode

# Standard Baboon test image downsampled 4x — used as default LR demo input.
SAMPLE_LR_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "baboon_x4.png",
)


def main(is_test: bool = False) -> None:
    """BasicVSR++ video super-resolution demo.

    Input options (checked in order):
      --video      path to a video file (mp4/avi/mov …); not supported with --eval-mode on-device
      --input-dir  directory of LR PNG/JPG frames sorted by filename
      (default)    downloads baboon_x4.png and replicates it as a short clip

    Output: HR frames displayed, or saved to --output-dir when -o is given.
    """
    parser = get_model_cli_parser(BasicVSRPlusPlus)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "Path to a video file (mp4/avi/mov …). Decoded at native FPS with "
            "torchvision.io.read_video (torchcodec fallback), then resized and "
            "center-cropped to the model LR input size. "
            "Not supported with --eval-mode on-device."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory of LR input frames (PNG/JPG, sorted by filename). "
        "Mutually exclusive with --video.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of frames to process (default: {DEFAULT_NUM_FRAMES}).",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="First frame index when using --input-dir (default: 0).",
    )
    parser.add_argument(
        "--filename-tmpl",
        type=str,
        default="{:08d}.png",
        help="Output filename template for saved frames (default: %(default)s).",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    is_on_device = args.eval_mode == EvalMode.ON_DEVICE

    if is_on_device and args.video is not None:
        raise ValueError(
            "--video is not supported with --eval-mode on-device. "
            "Use --input-dir or the default sample input instead."
        )

    _, components = demo_model_components_from_cli_args(
        BasicVSRPlusPlus, MODEL_ID, args
    )
    (
        flownet,
        feat_extract,
        prop_backward_1,
        prop_forward_1,
        prop_backward_2,
        prop_forward_2,
        reconstruction,
    ) = components
    app = BasicVSRPlusPlusApp(
        flownet,
        feat_extract,
        prop_backward_1,
        prop_forward_1,
        prop_backward_2,
        prop_forward_2,
        reconstruction,
    )

    if is_test:
        lq_frames = torch.rand(4, 3, DEFAULT_HEIGHT, DEFAULT_WIDTH)
        with torch.no_grad():
            app.upscale_video(lq_frames)
        return

    # ---- load input frames ------------------------------------------------
    if args.video is not None:
        lq_frames, lq_pil = load_frames_from_video(args.video, args.num_frames)
        source_desc = f"video: {args.video}"
    elif args.input_dir is not None:
        lq_frames, lq_pil = load_frames_from_dir(
            args.input_dir, args.num_frames, args.start_idx
        )
        source_desc = f"frames: {args.input_dir}"
    else:
        lq_pil_single = load_image(SAMPLE_LR_IMAGE)
        # Replicating one still produces all-zero optical flow, so temporal
        # propagation is not exercised; use --input-dir for a real sequence.
        lq_pil = [lq_pil_single] * DEFAULT_NUM_FRAMES
        lq_frames = pil_list_to_tensor(lq_pil)
        source_desc = "default sample (baboon_x4.png)"

    # Center-crop (with resize-to-fit) to match the dataset preprocessing.
    if lq_frames.shape[2] != DEFAULT_HEIGHT or lq_frames.shape[3] != DEFAULT_WIDTH:
        h, w = lq_frames.shape[2], lq_frames.shape[3]
        if h < DEFAULT_HEIGHT or w < DEFAULT_WIDTH:
            raise ValueError(
                f"Input frames ({w}x{h}) are smaller than the model input "
                f"({DEFAULT_WIDTH}x{DEFAULT_HEIGHT}). Provide larger frames."
            )
        lq_frames = TF.resize(lq_frames, min(DEFAULT_HEIGHT, DEFAULT_WIDTH))
        lq_frames = TF.center_crop(lq_frames, [DEFAULT_HEIGHT, DEFAULT_WIDTH])
        lq_pil = [
            Image.fromarray(
                (lq_frames[i].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(
                    np.uint8
                )
            )
            for i in range(lq_frames.shape[0])
        ]

    h_lr, w_lr = lq_frames.shape[2], lq_frames.shape[3]
    print(
        f"Running BasicVSR++ | {source_desc}\n"
        f"  LR input : {w_lr}x{h_lr}  →  HR output: {w_lr * 4}x{h_lr * 4}"
        f"  ({len(lq_pil)} frame(s))"
    )

    with torch.no_grad():
        hr_images = app.upscale_video(lq_frames)

    # ---- output verification -----------------------------------------------
    lr_first = lq_pil[0]
    hr_first = hr_images[0]
    print(
        f"Output verified: frame 0  "
        f"LR {lr_first.size[0]}x{lr_first.size[1]}  →  "
        f"HR {hr_first.size[0]}x{hr_first.size[1]}"
    )

    output_dir = getattr(args, "output_dir", None)
    start_idx = getattr(args, "start_idx", 0)
    tmpl = getattr(args, "filename_tmpl", "{:08d}.png")

    display_or_save_image(
        lr_first, output_dir, "input_lowres_0.png", "LR input (frame 0)"
    )
    display_or_save_image(
        hr_first, output_dir, "output_highres_0.png", "HR output (frame 0)"
    )

    if output_dir is not None and len(hr_images) > 1:
        for i, hr_img in enumerate(hr_images[1:], start=start_idx + 1):
            hr_img.save(os.path.join(output_dir, tmpl.format(i)))
        print(f"Saved {len(hr_images)} upscaled frames to {output_dir}")


if __name__ == "__main__":
    main()

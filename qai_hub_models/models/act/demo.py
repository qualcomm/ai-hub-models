# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import numpy as np

from qai_hub_models.models.act.app import ACTApp
from qai_hub_models.models.act.model import (
    ACT,
    MODEL_ID,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.display import generate_video_from_frames


def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(ACT)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)

    parser.add_argument(
        "--episode_len",
        type=int,
        default=1,
        help="Number of steps in the simulation (1 to 400)",
    )

    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(ACT, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load
    app = ACTApp(model)  # type: ignore[arg-type]
    print("Model Loaded")

    image_list = app.predict(args.episode_len)
    assert isinstance(image_list, list)
    if not is_test:
        cam_names = list(image_list[0].keys())
        frames = []
        for image_dict in image_list:
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                images.append(image)
            frames.append(np.concatenate(images, axis=1))
        generate_video_from_frames(
            frames,
            output_dir=args.output_dir,
            output_filename="output_actions.mp4",
            fps=50,
        )


if __name__ == "__main__":
    main()

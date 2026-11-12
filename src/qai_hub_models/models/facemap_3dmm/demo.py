# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from qai_hub_models.models.facemap_3dmm.app import FaceMap_3DMMApp
from qai_hub_models.models.facemap_3dmm.model import (
    INPUT_IMAGE_PATH,
    MODEL_ID,
    FaceMap_3DMM,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image


def _parse_face_box(face_box_str: str) -> list[float]:
    values = [float(x) for x in face_box_str.split(",")]
    if len(values) != 4:
        raise ValueError(
            "Face box must be 4 comma-separated float values: left,right,top,bottom (normalized to [0,1])"
        )

    return values


# Run FaceMap_3DMM end-to-end on a sample image.
# The demo will display a image with the predicted landmark displayed.
def facemap_3dmm_demo(
    model_cls: type[FaceMap_3DMM] = FaceMap_3DMM,
    model_id: str = MODEL_ID,
    is_test: bool = False,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=str(INPUT_IMAGE_PATH.fetch()),
        help="image file path or URL",
    )
    parser.add_argument(
        "--face-box",
        type=_parse_face_box,
        default="0.0,1.0,0.0,1.0",
        help=(
            "Part of image where to apply face landmark algorithm. "
            "This should be centered around the face for best landmark performance. "
            "We recommend using a face detector to retrieve the face box (not included in this demo). "
            "The values are expressed as 'left,right,top,bottom' with floating point values "
            "normalized to [0, 1]."
        ),
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_cls, model_id, args)
    validate_on_device_demo_args(args, model_id)

    # Load image
    image = load_image(args.image)

    print("Model Loaded")

    shape = model_cls.from_pretrained().get_input_spec()["image"][0]
    app = FaceMap_3DMMApp(model, model_input_shape=(shape[-2], shape[-1]))  # type: ignore[arg-type]

    # Get face bounding box info (from file or face detector)
    x0, x1, y0, y1 = (
        np.int32(round(image.width * args.face_box[0])),
        np.int32(round(image.width * args.face_box[1])),
        np.int32(round(image.height * args.face_box[2])),
        np.int32(round(image.height * args.face_box[3])),
    )

    lmk, output = app.landmark_prediction(image, x0, x1, y0, y1)

    if not is_test:
        # Annotated lmk
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            annotation_path = Path(args.output_dir) / "demo_output_lmk.txt"
            np.savetxt(
                annotation_path,
                lmk.detach().numpy(),
            )
            print("Saving annotations to", annotation_path)

        # Annotated image
        display_or_save_image(
            Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)),
            args.output_dir,
            filename="demo_output_img.png",
        )


if __name__ == "__main__":
    facemap_3dmm_demo(FaceMap_3DMM, MODEL_ID)

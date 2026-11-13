# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import json
import os
from pathlib import Path

from qai_hub_models.models.face_det_lite.app import FaceDetLiteApp
from qai_hub_models.models.face_det_lite.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceDetLite,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_640x480_Rooney.jpg"
)


# Run face_det_lite model end-to-end on a sample image.
# The demo will output the face bounding boxes in json files
# the bounding box represented by left, top, width, and height.
def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(FaceDetLite)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(FaceDetLite, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image
    orig_image = load_image(args.image)
    print("Model Loaded")

    app = FaceDetLiteApp(model)  # type: ignore[arg-type]
    res, out = app.run_inference_on_image(orig_image)
    out_dict = {}

    out_dict["bounding box"] = str(res)

    if not is_test:
        output_path = (
            args.output_dir or str(Path() / "build")
        ) + "/FaceDetLitebNet_output.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as wf:
            json.dump(out_dict, wf, ensure_ascii=False, indent=4)
        display_or_save_image(out, args.output_dir, "FaceDetLitebNet_output.png")
        print(f"Model outputs are saved at: {output_path}")


if __name__ == "__main__":
    main()

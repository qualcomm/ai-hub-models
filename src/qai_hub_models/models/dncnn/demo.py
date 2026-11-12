# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from PIL import Image

from qai_hub_models.models.dncnn.app import DnCNNApp
from qai_hub_models.models.dncnn.model import DnCNN
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import load_image


def make_noisy_image(
    height: int = 256, width: int = 256, sigma: float = 25.0, seed: int = 42
) -> tuple[Image.Image, Image.Image]:
    """
    Create a clean/noisy grayscale image pair.

    Parameters
    ----------
    height
        Image height in pixels.
    width
        Image width in pixels.
    sigma
        Gaussian noise standard deviation (on [0, 255] scale).
    seed
        Random seed for reproducible noise.

    Returns
    -------
    noisy_image : Image.Image
        Grayscale image with added Gaussian noise.
    clean_image : Image.Image
        Clean grayscale image (smooth sinusoidal pattern).
    """
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    clean_arr = np.clip(np.sin(xx * 12) * np.cos(yy * 8) * 0.3 + 0.5, 0, 1)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma / 255.0, clean_arr.shape).astype(np.float32)
    noisy_arr = np.clip(clean_arr + noise, 0, 1)

    clean_img = Image.fromarray((clean_arr * 255).astype(np.uint8))
    noisy_img = Image.fromarray((noisy_arr * 255).astype(np.uint8))
    return noisy_img, clean_img


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(DnCNN)
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="URL or path to an input image (optional).",
    )
    args = parser.parse_args([] if is_test else None)

    model = model_from_cli_args(DnCNN, args)
    app = DnCNNApp(model)

    if args.image is not None:
        noisy_img = load_image(args.image).convert("L")
    else:
        noisy_img, _ = make_noisy_image()

    result = app.denoise_image(noisy_img)

    if not is_test:
        display_or_save_image(noisy_img, "dncnn_noisy_input.png", "noisy input")
        display_or_save_image(result[0], "dncnn_denoised_output.png", "denoised output")


if __name__ == "__main__":
    main()

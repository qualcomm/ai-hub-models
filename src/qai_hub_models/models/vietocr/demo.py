# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

from qai_hub_models.models.vietocr.app import VietOCRApp
from qai_hub_models.models.vietocr.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    VIETOCR_CONFIG_NAME,
    VietOCR,
)
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

DEFAULT_SAMPLE_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_text.jpg"
)


def _load_transformer_head_and_vocab() -> tuple[Any, Any, Any]:
    """Load VietOCR's Transformer seq2seq head and output vocab.

    The AI Hub model in this directory is only the CNN backbone (the first of
    VietOCR's two recognizer components). The Transformer seq2seq head plus its
    output vocabulary -- the second component -- are loaded here from the
    installed ``vietocr`` package so the demo can decode the backbone features
    into recognized text end to end.
    """
    import torch
    from vietocr.model.transformerocr import VietOCR as _VietOCR
    from vietocr.model.vocab import Vocab
    from vietocr.tool.config import Cfg
    from vietocr.tool.utils import download_weights

    cfg = Cfg.load_config_from_name(VIETOCR_CONFIG_NAME)
    cfg["device"] = "cpu"
    vocab = Vocab(cfg["vocab"])
    full_model = _VietOCR(
        len(vocab),
        cfg["backbone"],
        cfg["cnn"],
        cfg["transformer"],
        cfg["seq_modeling"],
    ).eval()
    weights = download_weights(cfg["pretrain"])
    full_model.load_state_dict(torch.load(weights, map_location="cpu"))
    return full_model, full_model.transformer, vocab


def main(is_test: bool = False) -> None:
    import numpy as np

    # Demo parameters
    parser = get_model_cli_parser(VietOCR)
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_SAMPLE_IMAGE,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)

    # Load the exported CNN backbone (AI Hub model) and wrap it in the app.
    app = VietOCRApp(model_from_cli_args(VietOCR, args))

    # Load image and run the backbone -> per-column feature sequence.
    image = load_image(args.image)
    features = app.predict_features_from_image(image)

    # Load the second recognizer component (Transformer head + vocab) from the
    # vietocr package and decode the backbone features into text end to end.
    full_model, transformer, vocab = _load_transformer_head_and_vocab()
    text = app.recognize_text(image, transformer, vocab)

    # Sanity check: the exported backbone reproduces the package backbone's
    # features on the same input, proving the on-device CNN export is correct.
    pixel_values = app.preprocess_image(image)
    with __import__("torch").no_grad():
        reference = np.asarray(full_model.cnn(pixel_values))
    backbone_features = app.predict_features_from_image(pixel_values)
    features_match = np.allclose(backbone_features, reference, atol=1e-4)

    if not is_test:
        print(f"Backbone feature sequence shape: {features.shape}")
        print(f"Exported backbone matches package backbone: {features_match}")
        print(f"Recognized text: {text}")
    else:
        assert features_match, "Exported backbone features diverge from VietOCR's."


if __name__ == "__main__":
    main()

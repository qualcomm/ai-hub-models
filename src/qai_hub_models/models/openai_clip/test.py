# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.openai_clip.app import ClipApp
from qai_hub_models.models.openai_clip.demo import main as demo_main
from qai_hub_models.models.openai_clip.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OpenAIClip,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)

IMAGE_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image1.jpg"
)
TEXTS = ["a pyramid in the desert", "a dog on a beach", "a cat on a sofa"]
GOLDEN_OUTPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "golden_logits.npy"
)


def test_task() -> None:
    model = OpenAIClip.from_pretrained()
    app = ClipApp(
        image_encoder=model.image_encoder,
        text_encoder=model.text_encoder,
        logit_scale=model.logit_scale,
    )
    image = load_image(IMAGE_ASSET)
    logits = app.predict_similarity([image], TEXTS)
    assert logits.shape == (1, len(TEXTS))

    golden_logits = load_numpy(GOLDEN_OUTPUT.fetch())
    logits_np = logits.detach().cpu().numpy()

    np.testing.assert_allclose(
        logits_np,
        golden_logits,
        rtol=1e-2,
        atol=1e-2,
    )


def test_demo() -> None:
    demo_main(is_test=True)

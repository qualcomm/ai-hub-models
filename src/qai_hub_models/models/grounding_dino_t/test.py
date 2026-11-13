# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

from qai_hub_models.models.grounding_dino_t.app import (
    DEFAULT_BOX_THRESHOLD,
    GroundingDINOTCollectionApp,
)
from qai_hub_models.models.grounding_dino_t.dataset import IMG_HEIGHT, IMG_WIDTH
from qai_hub_models.models.grounding_dino_t.demo import DEFAULT_TEXT, IMAGE_ADDRESS
from qai_hub_models.models.grounding_dino_t.demo import main as demo_main
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.groundingdino import (
    build_groundingdino,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.util.misc import (
    NestedTensor,
)
from qai_hub_models.models.grounding_dino_t.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    GroundingDINOTCollection,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    letterbox_resize_pad_pil,
    normalize_image_torchvision,
    preprocess_PIL_image,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.inference import AsyncOnDeviceResult

# Golden detections for the demo image/text (box_threshold/text_threshold/nms_threshold
# at predict()'s defaults). Captured from a known-good run; regenerate if the model,
# preprocessing, or demo image/text intentionally changes.
EXPECTED_SCORES = [
    0.7120910286903381,
    0.6070619225502014,
    0.5334751605987549,
    0.4492429196834564,
]
EXPECTED_PHRASES = ["plants", "plants", "mirror", "books"]
EXPECTED_BOXES_XYXY = [
    [184.8665771484375, 130.14193725585938, 240.66653442382812, 227.28326416015625],
    [341.5661315917969, 214.67930603027344, 429.3395080566406, 349.38720703125],
    [85.21841430664062, 95.49128723144531, 195.90994262695312, 191.7403106689453],
    [423.125244140625, 37.273712158203125, 561.4688720703125, 351.5998229980469],
]

# Literal config from upstream's config/GroundingDINO_SwinT_OGC.py (swin_T_224_1k
# backbone branch). Passed to build_groundingdino/build_backbone/build_transformer
# as a plain namespace -- SLConfig machinery isn't needed, these are the only
# fields that branch reads.
_UPSTREAM_CONFIG = dict(
    batch_size=1,
    modelname="groundingdino",
    backbone="swin_T_224_1k",
    position_embedding="sine",
    pe_temperatureH=20,
    pe_temperatureW=20,
    return_interm_indices=[1, 2, 3],
    backbone_freeze_keywords=None,
    enc_layers=6,
    dec_layers=6,
    pre_norm=False,
    dim_feedforward=2048,
    hidden_dim=256,
    dropout=0.0,
    nheads=8,
    num_queries=900,
    query_dim=4,
    num_patterns=0,
    num_feature_levels=4,
    enc_n_points=4,
    dec_n_points=4,
    two_stage_type="standard",
    two_stage_bbox_embed_share=False,
    two_stage_class_embed_share=False,
    transformer_activation="relu",
    dec_pred_bbox_embed_share=True,
    dn_box_noise_scale=1.0,
    dn_label_noise_ratio=0.5,
    dn_label_coef=1.0,
    dn_bbox_coef=1.0,
    embed_init_tgt=True,
    dn_labelbook_size=2000,
    max_text_len=256,
    text_encoder_type="bert-base-uncased",
    use_text_enhancer=True,
    use_fusion_layer=True,
    use_checkpoint=True,
    use_transformer_ckpt=True,
    use_text_cross_attention=True,
    text_dropout=0.0,
    fusion_dropout=0.0,
    fusion_droppath=0.1,
    sub_sentence_present=True,
)

# Same two keys model_patches.py's _load_weights() excludes before its own
# strict load: label_enc.weight is training-only, bert.embeddings.position_ids
# is a buffer absent from the pinned checkpoint's BERT.
_KNOWN_UNEXPECTED = {"label_enc.weight", "bert.embeddings.position_ids"}


def _build_upstream_reference(out_path: str, image: str, text: str) -> None:
    """Build a pristine (unpatched) upstream GroundingDINO-T, run it on
    `image`/`text`, and save raw pred_logits/pred_boxes to `out_path`.

    Must run as `test.py`'s `__main__` entry point in a fresh subprocess (see
    `test_upstream_parity`), never imported: model_patches.apply_patches()
    monkeypatches upstream classes (SwinTransformer, MSDeformAttn,
    TransformerEncoder/Decoder, ...) shared with the external_repos module
    this builds from, and that patch is never undone for the rest of the
    process once GroundingDINOTCollection.from_pretrained() runs.
    """
    args = types.SimpleNamespace(**_UPSTREAM_CONFIG)
    model = build_groundingdino(args)
    model.eval()

    weights_path = str(
        CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "groundingdino_swint_ogc.pth"
        ).fetch()
    )
    raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = {
        k.removeprefix("module."): v for k, v in raw.get("model", raw).items()
    }
    state_dict = {k: v for k, v in state_dict.items() if k not in _KNOWN_UNEXPECTED}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, (
        f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}"
    )

    pil_image = load_image(image)
    resized = letterbox_resize_pad_pil(pil_image, IMG_HEIGHT, IMG_WIDTH)
    image_tensor = preprocess_PIL_image(resized)  # [1, 3, H, W], [0, 1]
    image_norm = normalize_image_torchvision(image_tensor)
    # Pristine upstream's own position-encoding/backbone code requires a bool
    # mask (does `~mask`); the patched pipeline uses a float32-zeros
    # convention that only works there because of its own P1 reimplementation.
    image_mask = torch.zeros(
        image_norm.shape[0], image_norm.shape[2], image_norm.shape[3], dtype=torch.bool
    )
    samples = NestedTensor(image_norm, image_mask)

    # Upstream's forward() tokenizes internally but does not itself apply the
    # lowercase/trailing-period normalization dataset.py's tokenize_caption does.
    caption = text.lower().strip()
    if not caption.endswith("."):
        caption += "."

    with torch.no_grad():
        out = model(samples, captions=[caption])

    np.savez_compressed(
        out_path,
        pred_logits=out["pred_logits"].detach().numpy(),  # [1, 900, 256]
        pred_boxes=out["pred_boxes"].detach().numpy(),  # [1, 900, 4]
    )


def test_task() -> None:
    """The collection app's decoded detections on the demo image/text must match
    a known-good set of (score, phrase, box) triples.
    """
    image = load_image(IMAGE_ADDRESS)

    collection = GroundingDINOTCollection.from_pretrained()
    app = GroundingDINOTCollectionApp.from_components(
        list(collection.components.values()), DEFAULT_TEXT
    )
    _, scores, phrases, boxes_xyxy = app.predict(image)

    assert phrases == EXPECTED_PHRASES
    np.testing.assert_allclose(
        scores.detach().numpy(), EXPECTED_SCORES, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        boxes_xyxy.detach().numpy(), EXPECTED_BOXES_XYXY, rtol=1e-3, atol=0.5
    )


def test_upstream_parity() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        upstream_npz_path = Path(tmpdir) / "upstream_outputs.npz"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__)),
                str(upstream_npz_path),
                str(IMAGE_ADDRESS),
                DEFAULT_TEXT,
            ],
            check=True,
        )
        with np.load(upstream_npz_path) as data:
            upstream_pred_logits = torch.from_numpy(data["pred_logits"])
            upstream_pred_boxes = data["pred_boxes"]

    image = load_image(IMAGE_ADDRESS)
    collection = GroundingDINOTCollection.from_pretrained()
    app = GroundingDINOTCollectionApp.from_components(
        list(collection.components.values()), DEFAULT_TEXT
    )

    resized_image = letterbox_resize_pad_pil(image, IMG_HEIGHT, IMG_WIDTH)
    image_tensor = preprocess_PIL_image(resized_image)

    with torch.no_grad():
        vision_out = app.vision(image_tensor)
        assert not isinstance(vision_out, AsyncOnDeviceResult)
        bert_out = app.bert(
            app._input_ids,
            app._token_type_ids,
            app._position_ids,
            app._text_self_attention_masks,
        )
        assert not isinstance(bert_out, AsyncOnDeviceResult)
        fusion_decoder_out = app.fusion_decoder(
            bert_out,
            app._attention_mask,
            app._position_ids,
            app._text_self_attention_masks,
            *vision_out,
        )
        assert not isinstance(fusion_decoder_out, AsyncOnDeviceResult)
        patched_pred_logits, patched_pred_boxes = fusion_decoder_out

    np.testing.assert_allclose(
        patched_pred_logits.sigmoid().detach().numpy(),
        upstream_pred_logits.sigmoid().numpy(),
        atol=0.03,
        rtol=1e-2,
    )

    # Below DEFAULT_BOX_THRESHOLD, queries have no stable identity: near-zero-confidence
    # "background" queries can swap/drift under tiny FP perturbations compounded across
    # decoder layers, with no bearing on any actual detection. Only compare boxes for
    # queries confident enough to actually surface in predict()'s output.
    confident = (
        upstream_pred_logits.sigmoid().amax(dim=-1).numpy() > DEFAULT_BOX_THRESHOLD
    )
    assert confident.any(), (
        "No queries cleared DEFAULT_BOX_THRESHOLD; nothing to compare."
    )
    np.testing.assert_allclose(
        patched_pred_boxes.detach().numpy()[confident],
        upstream_pred_boxes[confident],
        atol=1e-2,
        rtol=1e-2,
    )


def test_demo() -> None:
    demo_main(is_test=True)


if __name__ == "__main__":
    # test_upstream_parity's subprocess entry point -- see _build_upstream_reference.
    _build_upstream_reference(sys.argv[1], sys.argv[2], sys.argv[3])

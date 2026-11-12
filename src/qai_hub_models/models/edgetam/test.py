# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from sam2.sam2_image_predictor import SAM2ImagePredictor

from qai_hub_models.models.edgetam.app import EdgeTAMApp, EdgeTAMVideoApp
from qai_hub_models.models.edgetam.demo import VIDEO_ADDRESS, generate_frames_from_video
from qai_hub_models.models.edgetam.demo import main as demo_main
from qai_hub_models.models.edgetam.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    EdgeTAM,
    EdgeTAMLoader,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.test_helpers import (
    assert_most_close,
)

INPUT_MASK = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "input_mask.npy"
)
OUTPUT_MASK = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_mask.npy"
)


def _make_app(model: EdgeTAM) -> EdgeTAMVideoApp:
    return EdgeTAMVideoApp(
        encoder=model.encoder,
        video_decoder=model.video_decoder,
        memory_encoder=model.memory_encoder,
        sam2=model.sam2,
        maskmem_pos_enc=model.memory_encoder.maskmem_pos_enc,
        encoder_input_img_size=model.sam2.image_size,
    )


def test_e2e_numerical() -> None:
    """Verify encoder and decoder components individually against the source model."""
    model = EdgeTAM.from_pretrained()
    frames = generate_frames_from_video(str(VIDEO_ADDRESS.fetch()))[:1]
    frame_np = frames[0]

    point_coords = torch.tensor([[400.0, 150.0], [430.0, 380.0]])
    point_labels = torch.tensor([1.0, 1.0])

    img_size = model.sam2.image_size
    pil_img = PILImage.fromarray(frame_np).convert("RGB").resize((img_size, img_size))
    _, frame_tensor = app_to_net_image_inputs(np.array(pil_img))

    h, w = frame_np.shape[:2]
    scaled_coords = point_coords.clone().unsqueeze(0)
    scaled_coords[..., 0] /= w
    scaled_coords[..., 1] /= h

    # Encoder returns pix_feat (image_embeddings + no_mem_embed) as the 5th
    # output. SAM2ImagePredictor adds no_mem_embed in set_image, so compare
    # pix_feat directly.
    _, hr1, hr2, sparse, pix_feat = model.encoder(
        frame_tensor, scaled_coords, point_labels.unsqueeze(0)
    )
    predictor = SAM2ImagePredictor(EdgeTAMLoader._load_sam2())
    predictor.set_image(np.array(pil_img))
    assert_most_close(
        predictor._features["image_embed"].numpy(),
        pix_feat.detach().numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )
    assert_most_close(
        predictor._features["high_res_feats"][0].numpy(),
        hr1.detach().numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )
    assert_most_close(
        predictor._features["high_res_feats"][1].numpy(),
        hr2.detach().numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )

    # Decoder: SAM2VideoDecoder applies object-score gating that SAM2ImagePredictor
    # does not, so their low_res_masks outputs are not numerically comparable.
    # Instead, verify that the object is detected (score > 0) with valid prompts.
    _, _, _, object_score_logits = model.video_decoder(pix_feat, hr1, hr2, sparse)
    assert object_score_logits.item() > 0, (
        f"object_score_logits={object_score_logits.item():.2f} — "
        "decoder thinks object is absent with valid point prompts."
    )


def test_app() -> None:
    """Verify full pipeline output matches golden masks."""
    model = EdgeTAM.from_pretrained()
    frames = generate_frames_from_video(str(VIDEO_ADDRESS.fetch()))[:3]
    point_coords = torch.tensor(load_numpy(INPUT_MASK.fetch()), dtype=torch.float32)
    point_labels = torch.ones(point_coords.shape[0], dtype=torch.float32)

    output = np.asarray(
        _make_app(model).track(frames, point_coords, point_labels, raw_output=True)
    )
    assert_most_close(
        output, load_numpy(OUTPUT_MASK.fetch()), 0.005, rtol=0.001, atol=0.001
    )


def test_image_mode() -> None:
    """Verify EdgeTAMApp.predict (single-frame image path) produces correct output."""
    model = EdgeTAM.from_pretrained()
    frames = generate_frames_from_video(str(VIDEO_ADDRESS.fetch()))[:1]
    frame_np = frames[0]

    point_coords = torch.tensor([[400.0, 150.0], [430.0, 380.0]])
    point_labels = torch.ones(point_coords.shape[0], dtype=torch.float32)

    app = EdgeTAMApp(
        encoder=model.encoder,
        video_decoder=model.video_decoder,
        memory_encoder=model.memory_encoder,
        sam2=model.sam2,
        maskmem_pos_enc=model.memory_encoder.maskmem_pos_enc,
        encoder_input_img_size=model.sam2.image_size,
    )

    # raw_output=True: binary mask, same spatial size as input frame
    mask = app.predict(frame_np, point_coords, point_labels, raw_output=True)
    assert mask.shape == frame_np.shape[:2], (
        f"Mask shape {mask.shape} does not match frame shape {frame_np.shape[:2]}"
    )
    assert mask.dtype == np.uint8
    assert mask.max() > 0, (
        "Mask is entirely empty — object not detected with valid prompts."
    )
    assert mask.min() == 0, (
        "Mask covers the entire frame — expected partial segmentation."
    )

    # raw_output=False: painted RGB image, same spatial size as input frame
    painted = app.predict(frame_np, point_coords, point_labels, raw_output=False)
    assert painted.shape == frame_np.shape, (
        f"Painted shape {painted.shape} does not match frame shape {frame_np.shape}"
    )
    assert painted.dtype == np.uint8


def test_demo() -> None:
    demo_main(is_test=True)


def test_demo_image_mode() -> None:
    """Exercise the demo's --image code path end-to-end."""
    frames = generate_frames_from_video(str(VIDEO_ADDRESS.fetch()))[:1]
    frame_np = frames[0]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cv2.imwrite(tmp_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        demo_main(is_test=True, image_path=tmp_path)
    finally:
        os.remove(tmp_path)

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import zipfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

# Disable dynamo in lerobot forward
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import onnxruntime as ort
import pytest
import torch
from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel, PI05Policy

from qai_hub_models.models.pi05 import Model
from qai_hub_models.models.pi05.app import Pi05App, Pi05AppConfig
from qai_hub_models.models.pi05.demo import (
    DATASET_REPO_ID,
    HF_MODEL_ID,
    build_action_seq_from_first_episode,
    compute_sequence_metrics,
    load_one_batch,
    run_pi05_on_sample,
)
from qai_hub_models.models.pi05.model import NUM_CAMERAS, Pi05PaliGemmaTokenEmbed
from qai_hub_models.utils.inference import OnDeviceModel


@pytest.mark.nightly
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_one_sample() -> None:
    """
    Run a single forward pass and assert the predicted actions are close to

    1. Original pytorch impl (in element-wise comparison)
    2. Ground truth (in terms of rmse, sqnr)
    """
    torch.manual_seed(1234)
    # Load pretrained model and policy.
    model = Model.from_pretrained(HF_MODEL_ID)
    device = "cpu"
    policy = PI05Policy.from_pretrained(HF_MODEL_ID).to(device).eval()

    # Prepare one batch.
    batch, postprocessor, dataset = load_one_batch(
        dataset_repo_id=DATASET_REPO_ID,
        pi05_config=policy.config,
        batch_size=1,
        device=device,
    )

    # Prepare a deterministic noise tensor shared by both paths.
    cfg = policy.config
    bsize = 1
    tcfg = cfg.chunk_size
    dcfg = cfg.max_action_dim
    noise_shape = (bsize, tcfg, dcfg)

    noise_fixed = torch.normal(
        mean=0.0,
        std=1.0,
        size=noise_shape,
        dtype=torch.float32,
        device=device,
    )

    # Predict actions via the demo helper (app-like path) with fixed noise.
    pred_actions_app = run_pi05_on_sample(
        model=model,
        policy=policy,
        batch=batch,
        post_processor=postprocessor,
        noise=noise_fixed,
    )

    # Direct policy path with the identical noise returned by a patched
    # sampler. Count calls and validate shape and device.
    call_counter: dict[str, int] = {"n": 0}

    def _fixed_sample_noise(
        self: object, shape: tuple[int, int, int], dev: torch.device
    ) -> torch.Tensor:
        call_counter["n"] += 1
        assert tuple(shape) == noise_shape
        assert dev.type == torch.device(device).type
        return noise_fixed

    policy.model.sample_noise = _fixed_sample_noise.__get__(
        policy.model, type(policy.model)
    )

    pred_actions_policy = policy.predict_action_chunk(batch)
    pred_actions_policy = postprocessor(pred_actions_policy)

    # Ensure the sampler was called exactly once by the policy path.
    assert call_counter["n"] == 1

    # Parity checks after postprocessing.
    assert tuple(pred_actions_policy.shape) == tuple(pred_actions_app.shape)
    torch.testing.assert_close(
        pred_actions_policy, pred_actions_app, rtol=0.5, atol=0.1
    )

    # Build ground-truth sequence after prediction to avoid leakage.
    gt_actions = build_action_seq_from_first_episode(
        dataset=dataset,
        device=torch.device(device),
        seq_len=50,
    )

    # Compute metrics against ground truth using the demo prediction.
    mse, rmse, sqnr = compute_sequence_metrics(
        pred=pred_actions_app,
        gt=gt_actions,
    )

    # Sanity and threshold checks.
    assert pred_actions_app.dim() == 3
    assert gt_actions.dim() == 3
    assert pred_actions_app.shape == gt_actions.shape == (1, 50, 7)
    assert float(mse) < 0.002
    assert float(rmse) < 0.05
    assert float(sqnr) > 20


@pytest.fixture
def token_embed() -> Pi05PaliGemmaTokenEmbed:
    """Pi05PaliGemmaTokenEmbed with a tiny embed_tokens table (no checkpoint)."""
    paligemma = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    torch.nn.Module.__init__(paligemma)
    paligemma.paligemma = torch.nn.Module()
    paligemma.paligemma.language_model = torch.nn.Module()
    paligemma.paligemma.language_model.embed_tokens = torch.nn.Embedding(16, 8)
    model = Pi05PaliGemmaTokenEmbed.__new__(Pi05PaliGemmaTokenEmbed)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    model.model.paligemma_with_expert = paligemma
    model.register_buffer("inv_timescale", 10000.0 ** (-torch.arange(4) / 4))
    return model.eval()


def _inputs(batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    tokens = torch.tensor([[1, 2, 0, 0]], dtype=torch.int32).expand(batch_size, -1)
    language_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]]).expand(batch_size, -1)
    images = tuple(torch.zeros(batch_size, 256, 8) for _ in range(NUM_CAMERAS))
    return tokens, language_mask, *images


@pytest.mark.parametrize("num_cameras", [1, 2, 3])
def test_camera_padding(token_embed: Pi05PaliGemmaTokenEmbed, num_cameras: int) -> None:
    tokens, language_mask, *images = _inputs()
    camera_mask = torch.zeros(1, NUM_CAMERAS)
    camera_mask[:, :num_cameras] = 1
    with torch.no_grad():
        eager = token_embed(tokens, language_mask, *images[:num_cameras])
        fixed = token_embed(tokens, language_mask, *images, camera_mask=camera_mask)
    torch.testing.assert_close(fixed, eager, rtol=0, atol=0)
    # Active slots stay unmasked (0.0 in full_att_4d) even with zero embeddings.
    valid_prefix = num_cameras * 256 + 2
    assert (fixed[-1][0, 0, 0, :-50] == 0).sum().item() == valid_prefix
    expected_rope = token_embed._positions_to_sin_cos(
        torch.arange(valid_prefix, valid_prefix + 50)[None, :]
    )
    torch.testing.assert_close(fixed[4:6], expected_rope)


def test_export_camera_mask(
    token_embed: Pi05PaliGemmaTokenEmbed, tmp_path: Path
) -> None:
    spec = token_embed.get_input_spec(batch_size=2)
    assert list(spec) == [
        "lang_tokens",
        "lang_mask",
        "img_embed1",
        "img_embed2",
        "img_embed3",
        "camera_mask",
    ]
    # Shrink token length and embed dim to match the fixture's Embedding(16, 8).
    spec["lang_tokens"].shape = spec["lang_mask"].shape = (2, 4)
    for i in range(NUM_CAMERAS):
        spec[f"img_embed{i + 1}"].shape = (2, 256, 8)
    samples = token_embed._sample_inputs_impl(spec)
    np.testing.assert_array_equal(samples["camera_mask"][0], np.ones((2, NUM_CAMERAS)))
    exported = token_embed.serialize(tmp_path, spec)
    if zipfile.is_zipfile(exported):
        with zipfile.ZipFile(exported) as archive:
            archive.extractall(tmp_path / "export")
        exported = next((tmp_path / "export").rglob("*.onnx"))
    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    assert [inp.name for inp in session.get_inputs()] == list(spec)
    tokens, language_mask, *images = _inputs(batch_size=2)
    # Run different masks in one session to check camera_mask wasn't constant-folded.
    partial = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    for mask in (partial, torch.ones_like(partial), partial):
        inputs = (tokens, language_mask, *images, mask)
        with torch.no_grad():
            expected = token_embed(*inputs)
        actual = session.run(
            None, dict(zip(spec, [x.numpy() for x in inputs], strict=True))
        )
        for result, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(result, reference.numpy(), rtol=1e-5, atol=1e-6)
        expected_slots = (1 - mask).repeat_interleave(256, dim=1) * -1e4
        np.testing.assert_array_equal(actual[-1][:, 0, 0, :768], expected_slots.numpy())


@pytest.mark.parametrize("mode", ["eager", "device", "legacy"])
def test_app_camera_mask(mode: str) -> None:
    on_device = mode != "eager"
    token = Mock(spec=OnDeviceModel) if on_device else Mock()
    token.input_names = list(Pi05PaliGemmaTokenEmbed.get_input_spec_static())
    if mode == "legacy":
        token.input_names = token.input_names[:-1]
    token.return_value = tuple(torch.zeros(1) for _ in range(7))
    vision = Mock(return_value=torch.zeros(2 if on_device else 1, 256, 8))
    backbone = Mock(return_value=(torch.zeros(1), torch.zeros(1)))
    app = Pi05App(Pi05AppConfig(), vision, token, Mock(), backbone)
    app._on_device = on_device
    tokens, language_mask, *_ = _inputs()
    warning = (
        pytest.warns(UserWarning, match="Re-export token_emb")
        if mode == "legacy"
        else nullcontext()
    )
    with warning:
        app.populate_prefix([torch.zeros(1, 3, 224, 224)] * 2, tokens, language_mask)
    args = token.call_args.args
    assert len(args) == {"eager": 4, "device": 6, "legacy": 5}[mode]
    if mode != "legacy":
        mask = args[-1][0] if on_device else token.call_args.kwargs["camera_mask"]
        torch.testing.assert_close(mask, torch.tensor([[1.0, 1.0, 0.0]]))

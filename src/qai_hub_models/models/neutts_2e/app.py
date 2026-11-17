# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from qai_hub_models.models.neutts_2e.external_repos.neutts.neutts.neutts_2e import (
    NeuTTS2E as UpstreamNeuTTS2E,
)
from qai_hub_models.models.neutts_2e.model import (
    BACKBONE_REPO,
    CODEC_REPO,
    CONTEXT_LENGTH,
    DECODE_SEQ_LEN,
    NUM_LAYERS,
    PREFILL_SEQ_LEN,
    SAMPLE_RATE,
    NeuTTS2E,
    build_attention_mask,
    empty_kv_cache,
    load_codec,
)

MIN_NEW_TOKENS = 50
SPEECH_END_TOKEN = "<|SPEECH_GENERATION_END|>"

# Taken from upstream rather than restated, so a change to either list upstream
# does not silently diverge from the demo's accepted values.
SPEAKERS: tuple[str, ...] = UpstreamNeuTTS2E.SPEAKERS
EMOTIONS: tuple[str, ...] = UpstreamNeuTTS2E.EMOTIONS


class _KVHistory:
    """Per-layer KV history in one preallocated buffer.

    Entries sit after ``CONTEXT_LENGTH - 1`` zeros, so a right-aligned window of
    any width is a view and appending writes only the new entries.
    """

    _PAD = CONTEXT_LENGTH - 1

    def __init__(self) -> None:
        self._buffers = empty_kv_cache(self._PAD + CONTEXT_LENGTH)
        self.num_cached = 0

    def window(self, width: int) -> list[Tensor]:
        end = self._PAD + self.num_cached
        out: list[Tensor] = []
        for layer in range(NUM_LAYERS):
            out.append(self._buffers[2 * layer][..., end - width : end])
            out.append(self._buffers[2 * layer + 1][:, :, end - width : end, :])
        return out

    def append(self, new: list[Tensor], skip: int) -> None:
        """Append newly computed entries, dropping the first ``skip`` positions."""
        start = self._PAD + self.num_cached
        count = new[0].shape[-1] - skip
        for layer in range(NUM_LAYERS):
            self._buffers[2 * layer][..., start : start + count] = new[2 * layer][
                ..., skip:
            ]
            self._buffers[2 * layer + 1][:, :, start : start + count, :] = new[
                2 * layer + 1
            ][:, :, skip:, :]
        self.num_cached += count


def _sample(logits: Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    values, indices = torch.topk(logits / temperature, min(top_k, logits.shape[-1]))
    probs = torch.softmax(values, dim=-1)
    return int(indices[torch.multinomial(probs, num_samples=1)])


def generate_speech_tokens(
    model: NeuTTS2E,
    prompt_ids: list[int],
    temperature: float = 1.0,
    top_k: int = 50,
) -> list[int]:
    """Autoregressive decode at the prefill and decode graph shapes.

    On device these are two graphs of one linked binary; in torch they are the
    same module called at different sequence lengths. Each call gets a
    right-aligned fixed-width window of the cache, as the graphs expect.
    Generation stops at EOS or when the context window is full.
    """
    if len(prompt_ids) > CONTEXT_LENGTH - PREFILL_SEQ_LEN:
        raise ValueError(
            f"Prompt is {len(prompt_ids)} tokens, over the "
            f"{CONTEXT_LENGTH - PREFILL_SEQ_LEN} that fit in the context window. "
            "Use shorter text."
        )
    backbone = model.backbone
    eos_id = backbone.tokenizer.convert_tokens_to_ids(SPEECH_END_TOKEN)

    history = _KVHistory()
    last_logits: Tensor | None = None

    # Prefill: left-pad to a whole number of chunks so real tokens stay
    # right-aligned. Pad rows are masked out and their cache entries dropped.
    pad = -len(prompt_ids) % PREFILL_SEQ_LEN
    padded = [0] * pad + list(prompt_ids)
    for start in range(0, len(padded), PREFILL_SEQ_LEN):
        skip = pad if start == 0 else 0
        num_real = PREFILL_SEQ_LEN - skip
        num_cached = history.num_cached
        position_ids = torch.tensor(
            [[0] * skip + list(range(num_cached, num_cached + num_real))]
        )
        cos, sin = backbone.embedding.get_embedding(position_ids)
        with torch.no_grad():
            out = backbone(
                torch.tensor(
                    [padded[start : start + PREFILL_SEQ_LEN]], dtype=torch.int32
                ),
                build_attention_mask(PREFILL_SEQ_LEN, num_cached + num_real),
                cos,
                sin,
                *history.window(CONTEXT_LENGTH - PREFILL_SEQ_LEN),
            )
        last_logits = out[0][0, -1]
        history.append(out[1:], skip)

    assert last_logits is not None
    generated: list[int] = []
    while True:
        if len(generated) < MIN_NEW_TOKENS:
            last_logits = last_logits.clone()
            last_logits[eos_id] = float("-inf")
        token = _sample(last_logits, temperature, top_k)
        if token == eos_id:
            break
        generated.append(token)
        if history.num_cached >= CONTEXT_LENGTH - DECODE_SEQ_LEN:
            break

        num_cached = history.num_cached
        position_ids = torch.tensor([[num_cached]])
        cos, sin = backbone.embedding.get_embedding(position_ids)
        with torch.no_grad():
            out = backbone(
                torch.tensor([[token]], dtype=torch.int32),
                build_attention_mask(DECODE_SEQ_LEN, num_cached + DECODE_SEQ_LEN),
                cos,
                sin,
                *history.window(CONTEXT_LENGTH - DECODE_SEQ_LEN),
            )
        last_logits = out[0][0, -1]
        history.append(out[1:], 0)

    return generated


class NeuTTS2EApp:
    """End-to-end NeuTTS-2E synthesis app.

    Generation runs at the same two fixed graph shapes that are exported to
    device -- a prefill pass over ``PREFILL_SEQ_LEN`` tokens and a single-token
    decode pass -- with the KV cache passed in and out explicitly. Everything
    else (prompt assembly, emotion conditioning, sampling, codec decode) is CPU
    work borrowed from the upstream pipeline.

    There is no voice cloning, as the four speakers ship as pre-encoded reference
    codes, so no reference audio is read and the codec's encoder is never called.

    Inputs
    ------
    input_text: str
        The text to synthesize.
    speaker: str
        One of ``SPEAKERS``.
    emotion: str
        One of ``EMOTIONS``.

    Output
    ------
    ``np.ndarray`` of float32 audio samples at 24kHz.
    """

    def __init__(self, model: NeuTTS2E, seed: int | None = None) -> None:
        # Upstream's infer() calls torch.manual_seed(self._call_seed()) itself, so
        # seeding from outside has no effect; the seed has to go in here.
        self.model = model
        self.pipeline = _GraphPipeline(model, seed=seed)

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.synthesize(*args, **kwargs)

    __call__ = predict

    def synthesize(
        self,
        input_text: str,
        speaker: str = "emily",
        emotion: str = "neutral",
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> np.ndarray:
        return self.pipeline.infer(
            input_text,
            speaker=speaker,
            emotion=emotion,
            temperature=temperature,
            top_k=top_k,
        )

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE


class _GraphPipeline(UpstreamNeuTTS2E):
    """Upstream pipeline with weight loading and generation redirected.

    Reuses upstream speaker-reference lookup, prompt assembly and emotion
    handling, but binds the backbone and codec to the recipe's already-loaded
    modules (avoiding a second copy of each) and replaces generation with the
    exported-graph decode loop.
    """

    def __init__(self, model: NeuTTS2E, seed: int | None = None) -> None:
        self._qaihm_model = model
        # backbone_repo is only an identifier here -- _load_backbone below ignores
        # it and binds the already-loaded module, and the phonemizer branch that
        # would otherwise consult it never runs for a BPE-format model.
        super().__init__(
            backbone_repo=BACKBONE_REPO,
            codec_repo=CODEC_REPO,
            backbone_device="cpu",
            codec_device="cpu",
            seed=seed,
        )

    def _load_backbone(self, backbone_repo: str, backbone_device: str) -> None:
        backbone = self._qaihm_model.backbone
        self.tokenizer = backbone.tokenizer
        self.backbone = backbone.model
        config = backbone.llm_config.neuphonic
        self.input_format = config["input_format"]
        self._supported_emotions = config["supported_emotions"]

    def _load_codec(self, codec_repo: str, codec_device: str) -> None:
        self.codec = load_codec()

    def _infer_torch(
        self, prompt_ids: list[int], temperature: float = 1.0, top_k: int = 50
    ) -> str:
        tokens = generate_speech_tokens(
            self._qaihm_model, prompt_ids, temperature, top_k
        )
        tokenizer = self._qaihm_model.backbone.tokenizer
        return str(tokenizer.decode(tokens, add_special_tokens=False))

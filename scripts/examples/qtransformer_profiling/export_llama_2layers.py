#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .external_repos.efficient_transformers.QEfficient import QEFFAutoModelForCausalLM

# We build a tiny random Llama with vocab_size=13 and save it
# to a temporary directory that both loaders can read.
MODEL_ID = "meta-llama/Llama-3.2-3B"
NUM_LAYERS_KEEP = 2  # we only keep first 2 transformer layers
VOCAB_SIZE = 13  # changed from 128k to 13
SEED = 42
MAX_SEQ_LEN = 4096

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def build_tiny_llama_config(vocab_size: int) -> LlamaConfig:
    """
    Build a Llama config by loading the pretrained config indicated by
    MODEL_ID and overriding only `vocab_size` and `num_hidden_layers`.
    All other hyperparameters remain identical to the base config.
    """
    base_cfg = AutoConfig.from_pretrained(MODEL_ID)
    if not isinstance(base_cfg, LlamaConfig):
        raise TypeError("Pretrained config is not a LlamaConfig")

    cfg: LlamaConfig = base_cfg
    cfg.vocab_size = vocab_size
    cfg.num_hidden_layers = NUM_LAYERS_KEEP
    return cfg


def init_random_llama_model(config: LlamaConfig) -> LlamaForCausalLM:
    """
    Initialize a LlamaForCausalLM with random weights using the given
    config. We do not load any pretrained weights.
    """
    return LlamaForCausalLM(config)


def save_model_to_temp_dir(model: LlamaForCausalLM, tmp_dir: Path) -> None:
    """
    Save the given Hugging Face model and its config to the provided directory.

    Also writes a minimal tokenizer_config.json to avoid loader warnings.
    We do not need an actual tokenizer for this script.

    Parameters
    ----------
    model
        The HF model instance to save.
    tmp_dir
        The directory where the model should be saved.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Save the model weights and config
    model.save_pretrained(tmp_dir.as_posix())

    # Also save a minimal tokenizer config to avoid loader warnings.
    (tmp_dir / "tokenizer_config.json").write_text(
        '{"model_max_length": 128}',
        encoding="utf-8",
    )


def truncate_model_layers_qeff(
    model: "QEFFAutoModelForCausalLM",
    num_keep: int,
) -> "QEFFAutoModelForCausalLM":
    """
    For QEfficient wrapper, truncate the wrapped HF model and update
    the wrapper's bookkeeping (num_layers) accordingly.
    """
    # QEFFAutoModelForCausalLM wraps the HF model at `.model`
    if not hasattr(model, "model"):
        raise RuntimeError("QEff wrapper missing `.model` (wrapped HF model)")
    hf = model.model

    # Llama-style: hf.model.layers (no 'decoder' on Llama)
    if hasattr(hf, "model") and hasattr(hf.model, "layers"):
        layers = hf.model.layers
        assert len(layers) >= num_keep, "Not enough layers"
        hf.model.layers = torch.nn.ModuleList(layers[:num_keep])
    else:
        raise RuntimeError("Cannot find Llama layers at hf.model.layers for QEff model")

    # Keep configs and wrapper metadata consistent
    if hasattr(hf, "config") and hasattr(hf.config, "num_hidden_layers"):
        hf.config.num_hidden_layers = num_keep
    if hasattr(model, "num_layers"):
        model.num_layers = num_keep

    return model


class _SimpleStaticCache(Cache):
    """
    Minimal static cache that matches what LlamaModel expects. It
    implements:
      - get_seq_length() -> int
      - update(key, value, layer_idx, cache_kwargs) -> tuple[Tensor, Tensor]
      - __len__() -> int  (for len(cache))

    Storage is pre-allocated to [bsz, n_kv, max_len, hd]. The first
    `past_len` positions are filled with the provided past KV. New KV
    are appended (written) at positions given by cache_position.
    """

    def __init__(
        self,
        keys_per_layer: list[torch.Tensor],
        values_per_layer: list[torch.Tensor],
        past_len: int,
        max_len: int,
    ) -> None:
        self.past_len = int(past_len)
        self.max_len = int(max_len)

        assert len(keys_per_layer) == len(values_per_layer)
        self.num_layers = len(keys_per_layer)

        bsz, n_kv, seq_len_k, hd = keys_per_layer[0].shape
        bsz_v, n_kv_v, seq_len_v, hd_v = values_per_layer[0].shape

        assert bsz == bsz_v and n_kv == n_kv_v
        assert seq_len_k == seq_len_v == self.past_len
        assert hd == hd_v

        # Keep the full concatenated KV per layer, growing up to max_len.
        # Also keep the most recent "new segment" per layer to avoid
        # slicing when the caller needs the freshly appended window.
        self.keys: list[torch.Tensor] = [k.contiguous() for k in keys_per_layer]
        self.values: list[torch.Tensor] = [v.contiguous() for v in values_per_layer]

        # Track current filled length and the last appended KV segments.
        self.cur_len = int(self.past_len)
        self.last_new_keys: list[torch.Tensor | None] = [None] * self.num_layers
        self.last_new_values: list[torch.Tensor | None] = [None] * self.num_layers

    def __len__(self) -> int:
        """Return the number of layers cached."""
        return self.num_layers

    def get_seq_length(self) -> int:
        """Return the number of past tokens already cached."""
        return self.past_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new KV to the cache and keep only the most recent max_len tokens.

        Truncates from the front if needed. No cache_position is used.

        Parameters
        ----------
        key_states
            New key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        value_states
            New value tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        layer_idx
            Layer index to update.
        cache_kwargs
            Unused for this implementation.

        Returns
        -------
        key : torch.Tensor
            Updated key tensor up to the current end, of shape (batch, num_kv_heads, current_len, head_dim).
        value : torch.Tensor
            Updated value tensor up to the current end, of shape (batch, num_kv_heads, current_len, head_dim).
        """
        # Concatenate existing cache with the new states.
        k_cat = torch.cat([self.keys[layer_idx], key_states], dim=2)
        v_cat = torch.cat([self.values[layer_idx], value_states], dim=2)

        # Keep only the most recent max_len tokens.
        if k_cat.shape[2] > self.max_len:
            k_cat = k_cat[:, :, -self.max_len :, :]
            v_cat = v_cat[:, :, -self.max_len :, :]

        # Save back and record the most recent appended segment so the
        # caller can retrieve it without expensive slicing later.
        self.keys[layer_idx] = k_cat
        self.values[layer_idx] = v_cat
        self.last_new_keys[layer_idx] = key_states
        self.last_new_values[layer_idx] = value_states

        self.cur_len = int(k_cat.shape[2])
        return k_cat, v_cat

    def get_new_kv_cache(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Return the most recent appended KV tensors per layer. This
        avoids slicing from large buffers when only the latest window
        is required by the caller.

        Returns
        -------
        keys : list[torch.Tensor]
            List of key tensors per layer, each shaped (batch, num_kv_heads, seq_new, head_dim).
        values : list[torch.Tensor]
            List of value tensors per layer, each shaped (batch, num_kv_heads, seq_new, head_dim).
        """
        keys_out: list[torch.Tensor] = []
        values_out: list[torch.Tensor] = []
        for li in range(self.num_layers):
            k_new = self.last_new_keys[li]
            v_new = self.last_new_values[li]
            if k_new is None or v_new is None:
                # If update was never called on a layer, provide an
                # empty segment to keep shapes consistent.
                empty_k = self.keys[li][:, :, 0:0, :].contiguous()
                empty_v = self.values[li][:, :, 0:0, :].contiguous()
                keys_out.append(empty_k)
                values_out.append(empty_v)
            else:
                keys_out.append(k_new)
                values_out.append(v_new)
        return keys_out, values_out


class QEffPrefillLlama(torch.nn.Module):
    """
    A thin wrapper that adapts the QEfficient Llama model to a fixed
    prefill interface for ONNX export. It accepts a specific set of
    input tensors and returns logits and KV tensors.

    Expected inputs:
      - input_ids: int32[1, seq_len]
      - attention_mask: float32[1, 1, seq_len, MAX_SEQ_LEN]
      - past_key_0_in: float32[8, 1, head_dim, MAX_SEQ_LEN - seq_len]
      - past_value_0_in: float32[8, 1, MAX_SEQ_LEN - seq_len, head_dim]
      - past_key_1_in: float32[8, 1, head_dim, MAX_SEQ_LEN - seq_len]
      - past_value_1_in: float32[8, 1, MAX_SEQ_LEN - seq_len, head_dim]

    Inputs for K/V follow [H, B, Hd, S] for K and [H, B, S, Hd] for V.
    We convert to HF/QEff layout [B, H, S, Hd]. If K's trailing
    [S, Hd] does not match V's trailing [S, Hd], we swap K's last two
    dims so they align.

    Outputs (fixed shapes):
      - logits: float32[1, seq_len, 13]
      - past_key_0_out: float32[8, 1, head_dim, seq_len]
      - past_value_0_out: float32[8, 1, seq_len, head_dim]
      - past_key_1_out: float32[8, 1, head_dim, seq_len]
      - past_value_1_out: float32[8, 1, seq_len, head_dim]

    Notes
    -----
      - HF expects 2D attention masks; we derive one from input_ids.
      - Incoming KV are converted to HF layout for the forward. For the
        outputs, we materialize the requested dynamic seq_len and
        return K/V with shapes matching the spec above.
      - The past length is computed as MAX_SEQ_LEN - seq_len, where seq_len is
        the length of input_ids.
    """

    def __init__(self, qeff: "QEFFAutoModelForCausalLM") -> None:
        super().__init__()
        if not hasattr(qeff, "model"):
            raise RuntimeError("QEff wrapper must expose `.model`")
        self.qeff = qeff.model  # qeff doesn't have eval
        self.qeff.eval()

    def _kv_in_to_hf(
        self, k_in: torch.Tensor, v_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert incoming K/V to HF/QEff [B, H, S, Hd].

        K arrives as [H, B, Hd, S], V as [H, B, S, Hd].
        After permuting both to batch-first, if the K tail does not
        match V tail, swap the last two dims of K.
        """
        # Batch-first:
        #   K: [H, B, Hd, S] -> [B, H, Hd, S]
        #   V: [H, B, S, Hd] -> [B, H, S, Hd]
        k_bhhs = k_in.permute(1, 0, 2, 3).contiguous()
        v_bhshd = v_in.permute(1, 0, 2, 3).contiguous()

        # If needed, swap K last two dims to form [B, H, S, Hd].
        if k_bhhs.shape[-2:] != v_bhshd.shape[-2:]:
            exp = (v_bhshd.shape[-2], v_bhshd.shape[-1])  # (S, Hd)
            # k_bhhs currently [B, H, Hd, S]; after swap -> [B, H, S, Hd]
            if (k_bhhs.shape[-2], k_bhhs.shape[-1]) != (exp[1], exp[0]):
                raise ValueError(
                    "Incompatible K/V tails after batch-first: "
                    f"K {k_bhhs.shape} vs V {v_bhshd.shape}"
                )
            k_bhshd = k_bhhs.permute(0, 1, 3, 2).contiguous()
        else:
            # Already aligned as [B, H, S, Hd]
            k_bhshd = k_bhhs

        return k_bhshd, v_bhshd

    def _kv_out_dynamic(
        self, k_hf: torch.Tensor, v_hf: torch.Tensor, num_heads: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        HF K/V are [bsz, n_kv, seq_new, hd]. We must return:
          - K as [num_heads, bsz, head_dim, seq_new]
          - V as [num_heads, bsz, seq_new, head_dim]
        Copy the overlapping region if head counts differ.
        """
        bsz, n_kv, seq_new, hd = k_hf.shape
        device = k_hf.device
        dtype = torch.float32

        # Prepare outputs in requested layout.
        k_out = torch.zeros((num_heads, bsz, hd, seq_new), dtype=dtype, device=device)
        v_out = torch.zeros((num_heads, bsz, seq_new, hd), dtype=dtype, device=device)

        heads_copy = min(num_heads, n_kv)

        # Permute source to [n_kv, bsz, hd, seq_new] for K and
        # [n_kv, bsz, seq_new, hd] for V to match target indexing.
        k_src = k_hf.permute(1, 0, 3, 2).to(dtype=dtype)  # n_kv, bsz, hd, seq
        v_src = v_hf.permute(1, 0, 2, 3).to(dtype=dtype)  # n_kv, bsz, seq, hd

        # Copy into the top-left corner.
        k_out[:heads_copy, :bsz, :hd, :seq_new] = k_src[
            :heads_copy, :bsz, :hd, :seq_new
        ]
        v_out[:heads_copy, :bsz, :seq_new, :hd] = v_src[
            :heads_copy, :bsz, :seq_new, :hd
        ]

        return k_out, v_out

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_0_in: torch.Tensor,
        past_value_0_in: torch.Tensor,
        past_key_1_in: torch.Tensor,
        past_value_1_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward computes logits and dynamic-shape KV outputs for two
        layers. KV inputs are accepted to match the interface, but they
        are repacked for HF and do not constrain the output shapes.
        """
        # Standardize ids and build a simple 2D attention mask.
        input_ids = input_ids.to(dtype=torch.long)
        seq_len = int(input_ids.size(1))
        attn_mask_2d = torch.ones(
            (1, seq_len), dtype=torch.long, device=input_ids.device
        )

        # Convert incoming KV to HF/QEff layout [B, H, S, Hd].
        k0_hf, v0_hf = self._kv_in_to_hf(past_key_0_in, past_value_0_in)
        k1_hf, v1_hf = self._kv_in_to_hf(past_key_1_in, past_value_1_in)

        # Prepare a static cache that matches LlamaModel expectations.
        # Past length is MAX_SEQ_LEN - seq_len. We preallocate MAX_SEQ_LEN and the
        # model writes new tokens at positions past_len..(MAX_SEQ_LEN-1).
        max_len = MAX_SEQ_LEN
        past_len = max_len - seq_len
        keys_per_layer: list[torch.Tensor] = [k0_hf, k1_hf]
        values_per_layer: list[torch.Tensor] = [v0_hf, v1_hf]
        cache = _SimpleStaticCache(
            keys_per_layer=keys_per_layer,
            values_per_layer=values_per_layer,
            past_len=past_len,
            max_len=max_len,
        )

        # Run the wrapped model with a cache object. LlamaModel will
        # drive the cache via cache.update(...).
        out = self.qeff(
            input_ids=input_ids,
            attention_mask=attn_mask_2d,
            use_cache=True,
            past_key_values=cache,
            output_hidden_states=False,
            return_dict=True,
        )

        # Logits: shape [1, seq_len, 13].
        logits = out.logits.to(dtype=torch.float32)
        bsz, seq_len_out, vocab = logits.shape
        vocab_target = 13
        logits_fixed = torch.zeros(
            (bsz, seq_len, vocab_target),
            dtype=torch.float32,
            device=logits.device,
        )
        seq_copy = min(seq_len, seq_len_out)
        voc_copy = min(vocab_target, vocab)
        logits_fixed[:, :seq_copy, :voc_copy] = logits[:, :seq_copy, :voc_copy]

        # Build dynamic-shape KV outputs from the most recent segments.
        # get_new_kv_cache returns lists of [bsz, n_kv, seq_new, hd].
        # We expect seq_new == seq_len (the number of new tokens).
        num_heads = 8
        new_keys, new_values = cache.get_new_kv_cache()

        # Light checks to ensure the cache produced the expected dims.
        for t_k, t_v in zip(new_keys, new_values, strict=False):
            assert t_k.dim() == 4 and t_v.dim() == 4
            assert t_k.shape[0] == t_v.shape[0]  # bsz
            assert t_k.shape[1] == t_v.shape[1]  # n_kv
            assert t_k.shape[2] == seq_len or t_k.shape[2] == 0
            assert t_v.shape[2] == seq_len or t_v.shape[2] == 0
            assert t_k.shape[3] == t_v.shape[3]  # hd

        # Convert to requested output layout:
        #   K: [8, 1, head_dim, seq_len]
        #   V: [8, 1, seq_len, head_dim]
        k0_out, v0_out = self._kv_out_dynamic(
            new_keys[0], new_values[0], num_heads=num_heads
        )
        k1_out, v1_out = self._kv_out_dynamic(
            new_keys[1], new_values[1], num_heads=num_heads
        )

        return logits_fixed, k0_out, v0_out, k1_out, v1_out


def export_qeff_prefill_onnx(wrapper: QEffPrefillLlama, onnx_path: Path) -> None:
    """
    Export the QEffPrefillLlama wrapper to ONNX with static shapes
    matching the declared interface and the specified outputs.
    """
    wrapper.eval()

    # Force CPU-only export and freeze params to avoid "requires_grad"
    # constants during export.
    wrapper.to("cpu")
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Static example inputs for tracing the graph.
    seq = 128
    num_heads = 8
    dev = "cpu"

    input_ids = torch.ones((1, seq), dtype=torch.int32, device=dev)
    attention_mask = torch.ones(
        (1, 1, seq, MAX_SEQ_LEN), dtype=torch.float32, device=dev
    )
    past_key_0_in = torch.zeros(
        (num_heads, 1, seq, 3968), dtype=torch.float32, device=dev
    )
    past_value_0_in = torch.zeros(
        (num_heads, 1, 3968, seq), dtype=torch.float32, device=dev
    )
    past_key_1_in = torch.zeros(
        (num_heads, 1, seq, 3968), dtype=torch.float32, device=dev
    )
    past_value_1_in = torch.zeros(
        (num_heads, 1, 3968, seq), dtype=torch.float32, device=dev
    )

    input_names = [
        "input_ids",
        "attention_mask",
        "past_key_0_in",
        "past_value_0_in",
        "past_key_1_in",
        "past_value_1_in",
    ]
    output_names = [
        "logits",
        "past_key_0_out",
        "past_value_0_out",
        "past_key_1_out",
        "past_value_1_out",
    ]

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (
                input_ids,
                attention_mask,
                past_key_0_in,
                past_value_0_in,
                past_key_1_in,
                past_value_1_in,
            ),
            onnx_path.as_posix(),
            input_names=input_names,
            output_names=output_names,
            opset_version=13,
            # do_constant_folding=True,
            # training=torch.onnx.TrainingMode.EVAL,
        )


def export_seq_len(model: QEffPrefillLlama, seq_len: int) -> None:
    """
    Export the wrapper to ONNX for a given seq_len.

    K/V input shapes and past length are derived from seq_len as MAX_SEQ_LEN - seq_len.
    The output file is export/qeff_ar{seq_len}.onnx.

    Parameters
    ----------
    model
        QEffPrefillLlama to export (will be moved to CPU, eval).
    seq_len
        Input sequence length (1 <= seq_len <= 128).
    """
    assert 1 <= seq_len <= 128

    model.eval()
    model.to("cpu")

    for p in model.parameters():
        p.requires_grad_(False)

    num_heads = 8
    head_dim = 128  # tiny Llama head size
    dev = "cpu"
    past_len = MAX_SEQ_LEN - seq_len

    # Build example inputs for tracing with the requested seq_len.
    input_ids = torch.ones((1, seq_len), dtype=torch.int32, device=dev)
    attention_mask = torch.ones(
        (1, 1, seq_len, MAX_SEQ_LEN), dtype=torch.float32, device=dev
    )
    # K: [H, B, head_dim, past_len]; V: [H, B, past_len, head_dim]
    past_key_0_in = torch.zeros(
        (num_heads, 1, head_dim, past_len), dtype=torch.float32, device=dev
    )
    past_value_0_in = torch.zeros(
        (num_heads, 1, past_len, head_dim), dtype=torch.float32, device=dev
    )
    past_key_1_in = torch.zeros(
        (num_heads, 1, head_dim, past_len), dtype=torch.float32, device=dev
    )
    past_value_1_in = torch.zeros(
        (num_heads, 1, past_len, head_dim), dtype=torch.float32, device=dev
    )

    input_names = [
        "input_ids",
        "attention_mask",
        "past_key_0_in",
        "past_value_0_in",
        "past_key_1_in",
        "past_value_1_in",
    ]
    output_names = [
        "logits",
        "past_key_0_out",
        "past_value_0_out",
        "past_key_1_out",
        "past_value_1_out",
    ]

    # Quick CPU run to validate shapes.
    with torch.no_grad():
        (
            logits,
            past_key_0_out,
            past_value_0_out,
            past_key_1_out,
            past_value_1_out,
        ) = model(
            input_ids,
            attention_mask,
            past_key_0_in,
            past_value_0_in,
            past_key_1_in,
            past_value_1_in,
        )

    print(f"seq_len={seq_len} validation:")
    print(f"{logits.shape=}")
    print(f"{past_key_0_out.shape=}")
    print(f"{past_value_0_out.shape=}")
    print(f"{past_key_1_out.shape=}")
    print(f"{past_value_1_out.shape=}")

    assert logits.shape == (1, seq_len, 13)
    assert past_key_0_out.shape == (num_heads, 1, head_dim, seq_len)
    assert past_value_0_out.shape == (num_heads, 1, seq_len, head_dim)
    assert past_key_1_out.shape == (num_heads, 1, head_dim, seq_len)
    assert past_value_1_out.shape == (num_heads, 1, seq_len, head_dim)

    export_dir = Path("export")
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / f"qeff_ar{seq_len}.onnx"

    print(f"Exporting ONNX to {onnx_path.as_posix()} ...")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (
                input_ids,
                attention_mask,
                past_key_0_in,
                past_value_0_in,
                past_key_1_in,
                past_value_1_in,
            ),
            onnx_path.as_posix(),
            input_names=input_names,
            output_names=output_names,
            opset_version=13,
        )
    print("ONNX export complete.")
    print(f"Saved: {onnx_path.resolve().as_posix()}")


def main() -> None:
    print("Building tiny random Llama (vocab_size=13) ...")
    cfg = build_tiny_llama_config(VOCAB_SIZE)
    tiny = init_random_llama_model(cfg)

    print("Saving tiny model to a temporary directory ...")
    with tempfile.TemporaryDirectory(prefix="tiny-llama-13vocab-") as tmp:
        save_model_to_temp_dir(tiny, Path(tmp))

        print("Loading QEfficient model from temp dir (CPU only) ...")
        qeff_full = QEFFAutoModelForCausalLM.from_pretrained(
            str(tmp),
            torch_dtype=torch.float32,
            device_map=None,
        )
    print("Truncating QEfficient model to first", NUM_LAYERS_KEEP, "layers")
    qeff = truncate_model_layers_qeff(qeff_full, NUM_LAYERS_KEEP)

    print("Wrapping QEff model with QEffPrefillLlama ...")
    wrapper = QEffPrefillLlama(qeff).to("cpu")
    wrapper.eval()

    # Export for seq_len=1 and seq_len=128. All input creation and
    # validation are handled inside export_seq_len.
    export_seq_len(wrapper, 1)
    export_seq_len(wrapper, 128)


if __name__ == "__main__":
    main()

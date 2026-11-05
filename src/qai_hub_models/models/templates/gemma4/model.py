# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Gemma4 family base classes for the PreSplit/Part/Collection LLM architecture.

Supports:
- Dual attention (sliding window + global) with different head_dims
- Per-layer embedding (PLE) computed internally
- KV shared layers (only non-shared layers appear in ONNX I/O)
- MQA (1 KV head) or GQA
- Partial RoPE on global attention layers
- HuggingFace input_ids IO type (RoPE computed internally)
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import onnx
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from packaging.version import Version
from qai_hub.client import Device
from qai_hub.public_rest_api import DatasetEntries
from torch.export import Dim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.gemma4 import modeling_gemma4
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets import instantiate_dataset
from qai_hub_models.datasets.wikitext import WikiText, WikiTextChat
from qai_hub_models.datasets.wikitext.wikitext_chat import (
    chat_control_token_ids,
    count_token_ids,
)
from qai_hub_models.models.templates.gemma4._utils import (
    _MultiShardSafetensors,
    load_dequantized_state_dict,
    resolve_shard_paths,
)
from qai_hub_models.models.templates.gemma4.model_adaptations import (
    QCGemma4MLP,
    SHAGemma4Attention,
    qc_gemma4_text_model_forward,
)
from qai_hub_models.models.templates.gemma4.vision_encoder import Gemma4VisionEncoder
from qai_hub_models.models.templates.llm._utils import (
    _set_lm_head_to_8b,
    _tie_quantizers_for_kv_cache,
)
from qai_hub_models.models.templates.llm.common import LLMIOType
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EXPORT_CONTEXT_LENGTHS,
    DEFAULT_EXPORT_SEQUENCE_LENGTHS,
    DEFAULT_SEQUENCE_LENGTH,
    DynamicPreSplitOnnxMixin,
    DynamicQuantizablePreSplitMixin,
    DynamicSplitCollectionBase,
    DynamicSplitPartBase,
    Embedding,
    LLM_AIMETOnnx,
    LLMBase,
    LLMDynamic_AIMETOnnx,
    LLMDynamicBase,
    SingleSlotCacheMixin,
    get_onnx_model,
)
from qai_hub_models.models.templates.llm.model_adaptations import (
    ConvInplaceLinear,
    _apply_rope_single,
)
from qai_hub_models.models.templates.llm.sha_dynamic_kvcache import (
    SHADynamicCacheNewValueOnly,
)
from qai_hub_models.models.templates.lm_driver.generator import HubCompatibleGenerator
from qai_hub_models.models.templates.lm_driver.utils.attention_mask import (
    convert_2d_attention_mask_to_4d,
    convert_2d_attention_mask_to_4d_sliding_window,
)
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle
from qai_hub_models.utils.printing import print_with_box
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

# Chat turn markers, from the canonical Gemma4 template: every turn is bracketed
# as <|turn>{role}\n ... <turn|>\n, the assistant role is named "model", and
# bos_token is emitted once at the very start.
BOS_TOKEN = "<bos>"
START_TURN = "<|turn>"
END_TURN = "<turn|>"
SYSTEM_ID = "system"
USER_ID = "user"
ASSISTANT_ID = "model"

# Gemma4Processor.replace_image_token expands one image to
# boi_token + image_token * n + eoi_token, so these bracket the soft tokens.
VISION_START = "<|image>"
VISION_END = "<image|>"

# Chat generation stop tokens (in addition to the tokenizer's eos_token_id).
# An -it model ends a turn with <turn|>, not the bare <eos>; _fix_gemma4_genie_config
# writes the same pair into genie_config.json's eos-token.
END_TOKENS = {END_TURN, "<eos>"}
DEFAULT_USER_PROMPT = "What is gravity?"


# ---------------------------------------------------------------------------
# Dual RoPE Embedding class for Gemma4
# ---------------------------------------------------------------------------
class Gemma4RopeEmbedding(Embedding):
    """Precomputes dual RoPE embeddings for Gemma4.

    Gemma4 has two RoPE configurations:
    - SWA layers: full rotation, theta=10000, dim=head_dim (256)
    - Global layers: partial rotation (25%), theta=1000000, dim=global_head_dim*partial_factor (128)

    Both are precomputed and stored. At runtime, the model receives:
    - swa_position_ids_cos: (1, 1, seq_len, swa_embed_dim) for SWA layers
    - swa_position_ids_sin: (1, 1, seq_len, swa_embed_dim)
    - position_ids_cos: (1, 1, seq_len, global_embed_dim) for Global layers
    - position_ids_sin: (1, 1, seq_len, global_embed_dim)

    Where embed_dim = rope_dim / 2 (because _apply_rope_single splits into first/second halves).
    """

    def __init__(self, max_length: int = 2048, config: Any = None) -> None:
        self.max_length = max_length
        if config is None:
            # Defaults (E2B geometry)
            self.head_dim = 256
            self.global_head_dim = 512
            self.partial_rotary_factor = 0.25
            self.rope_theta_swa = 10000.0
            self.rope_theta_global = 1000000.0
        else:
            self.head_dim = getattr(config, "head_dim", 256)
            self.global_head_dim = getattr(config, "global_head_dim", 512)
            # Extract RoPE params from config
            rope_params = getattr(config, "rope_parameters", None)
            if rope_params and "full_attention" in rope_params:
                self.partial_rotary_factor = rope_params["full_attention"].get(
                    "partial_rotary_factor", 0.25
                )
                self.rope_theta_global = rope_params["full_attention"].get(
                    "rope_theta", 1000000.0
                )
            else:
                self.partial_rotary_factor = 0.25
                self.rope_theta_global = 1000000.0
            if rope_params and "sliding_attention" in rope_params:
                self.rope_theta_swa = rope_params["sliding_attention"].get(
                    "rope_theta", 10000.0
                )
            else:
                self.rope_theta_swa = 10000.0

        # Precompute SWA RoPE (full head_dim rotation)
        self.swa_cos, self.swa_sin = self._precompute_rope(
            dim=self.head_dim, theta=self.rope_theta_swa, max_length=max_length
        )

        # Global RoPE in Genie's "proportional" layout: a full 256-frequency
        # table whose first 64 are real and whose rest are zero-frequency. This
        # is equivalent to partial-rotating only n_rope_dims, but Genie rejects
        # the compact 64-dim table with a runtime ShapeError.
        global_rope_dim = int(self.global_head_dim * self.partial_rotary_factor)
        self.global_cos, self.global_sin = self._precompute_rope(
            dim=self.global_head_dim,
            theta=self.rope_theta_global,
            max_length=max_length,
            n_freqs=self.global_head_dim // 2,  # 256 (full); zero-pad past real
            n_real_freqs=global_rope_dim // 2,  # 64 real, rest zero-frequency
        )

        # embed_dim = dim / 2 (for _apply_rope_single which splits first/second halves)
        self.swa_embed_dim = self.head_dim // 2  # 128
        self.global_embed_dim = (
            self.global_head_dim // 2
        )  # 256 (Genie proportional-RoPE layout)

    @staticmethod
    def _precompute_rope(
        dim: int,
        theta: float,
        max_length: int,
        n_freqs: int | None = None,
        n_real_freqs: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin tables for RoPE.

        The frequency base is ALWAYS normalized by the full rotary ``dim``
        (head_dim), matching the real Gemma4 rotary: inv_freq[i] = theta^(-2i/dim).

        ``n_freqs`` sets the OUTPUT cos/sin width (number of frequencies kept,
        default dim/2 = full rotation). ``n_real_freqs`` (proportional RoPE):
        if given, only the first ``n_real_freqs`` frequencies are real; the
        remaining ``n_freqs - n_real_freqs`` are forced to ZERO frequency so
        cos=1, sin=0 (identity, no rotation). This mirrors transformers'
        _compute_proportional_rope_parameters (rope_angles real + nope_angles
        zero), giving the full-width 256-dim global table Genie expects while
        only rotating the first ``2*n_real_freqs`` head dims.

        Returns cos, sin each of shape (1, 1, max_length, n_freqs).
        """
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )  # shape: (half_dim,) — normalized by FULL dim
        if n_freqs is not None:
            inv_freq = inv_freq[:n_freqs]  # keep first n_freqs frequencies
        if n_real_freqs is not None:
            # Proportional RoPE: zero out frequencies past n_real_freqs so those
            # dims become identity (cos=1, sin=0), matching transformers'
            # _compute_proportional_rope_parameters nope-angles padding.
            inv_freq = inv_freq.clone()
            inv_freq[n_real_freqs:] = 0.0

        position_ids = torch.arange(max_length, dtype=torch.float32)
        freqs = torch.outer(position_ids, inv_freq)  # (max_length, n_freqs)

        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, max_length, n_freqs)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, max_length, n_freqs)
        return cos, sin

    def get_embedding(
        self, position_ids: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get SWA RoPE embeddings for given positions.

        position_ids: [batch_size, sequence_length]
        Returns: (cos, sin) each of shape [batch_size, 1, sequence_length, swa_embed_dim]
        """
        cos = self.swa_cos[0, 0, :, :].to(position_ids.device)
        sin = self.swa_sin[0, 0, :, :].to(position_ids.device)
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin

    def get_global_embedding(
        self, position_ids: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get Global RoPE embeddings for given positions.

        position_ids: [batch_size, sequence_length]
        Returns: (cos, sin) each of shape [batch_size, 1, sequence_length, global_embed_dim]
        """
        cos = self.global_cos[0, 0, :, :].to(position_ids.device)
        sin = self.global_sin[0, 0, :, :].to(position_ids.device)
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


# ---------------------------------------------------------------------------
# Helper to determine layer types
# ---------------------------------------------------------------------------
def get_gemma4_layer_types(
    num_layers: int, sliding_window_pattern: int = 5
) -> list[str]:
    """Generate the layer_types list for Gemma4.

    Pattern: (sliding_window_pattern - 1) sliding_attention + 1 full_attention, repeating.
    Last layer is always full_attention.
    """
    layer_types = []
    for i in range(num_layers):
        if (i + 1) % sliding_window_pattern == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("sliding_attention")
    # Last layer is always full_attention
    layer_types[-1] = "full_attention"
    return layer_types


def get_non_shared_layer_indices(
    num_layers: int, num_kv_shared_layers: int
) -> list[int]:
    """Get indices of layers that have their own KV cache (non-shared)."""
    first_shared = num_layers - num_kv_shared_layers
    return list(range(first_shared))


def kv_prefix(layer_type: str) -> str:
    """KV tensor name prefix by layer type.

    Genie groups KV cache tensors by prefix: sliding_attention layers use
    "swa_" (kv-dim/window cache-group) and full_attention layers use "past_"
    (full cache-group). This naming scheme routes the exported context binary's
    KV tensors to the correct Genie cache-group on device.
    """
    return "swa" if layer_type == "sliding_attention" else "past"


# ---------------------------------------------------------------------------
# Gemma4 host-side embedding LUT export (Genie bundle)
# ---------------------------------------------------------------------------
# Gemma4 takes inputs_embeds + per_layer_inputs as graph inputs, so both
# embedding tables live host-side; Genie loads them from ufixed8/ufixed16 LUT .bin
# files and dequantizes with the scale/offset written into genie_config.json.


def _global_uint16_encoding(fp: np.ndarray, bw: int = 16) -> dict[str, Any]:
    """Single asymmetric ufixed{bw} encoding (scale/offset) for a whole table."""
    fmin = float(fp.min())
    fmax = float(fp.max())
    qmax = (1 << bw) - 1
    scale = (fmax - fmin) / qmax
    offset = round(fmin / scale)
    return {"bw": bw, "scale": scale, "offset": offset}


def _quantize_uint16(
    fp: np.ndarray, scale: float, offset: int, bw: int = 16
) -> np.ndarray:
    shifted = fp / scale - offset
    q = np.floor(np.abs(shifted) + 0.5) * np.where(shifted < 0, -1, 1)
    return np.clip(q, 0, (1 << bw) - 1).astype(np.uint16 if bw == 16 else np.uint8)


# Rows per quantization block. The expression in _quantize_uint16 promotes to
# float64 (np.where yields int64, which upcasts the float32 operand), so it needs
# ~6x the table's float32 size in live temporaries. E4B's PLE table is
# 262144 x (42*256) = 10.5 GiB as float32, i.e. ~74 GiB peak in one shot, which
# the OOM killer takes out with no traceback. Blocking bounds that to a few GiB.
_LUT_QUANT_ROWS = 8192


def _write_uint16_lut(
    fp: np.ndarray, scale: float, offset: int, path: Path, bw: int = 16
) -> None:
    """Quantize ``fp`` to ufixed{bw} and stream it to ``path`` in row blocks.

    Bit-identical to ``_quantize_uint16(fp, ...).tofile(path)`` -- the same
    expression is evaluated per block -- but never materializes a whole-table
    float64 temporary.
    """
    with open(path, "wb") as f:
        for start in range(0, fp.shape[0], _LUT_QUANT_ROWS):
            _quantize_uint16(
                fp[start : start + _LUT_QUANT_ROWS], scale, offset, bw
            ).tofile(f)


def export_gemma4_embeddings(
    checkpoint: str | os.PathLike,
    output_dir: str | os.PathLike,
    hidden_size: int,
    num_layers: int,
    ple_dim: int,
    hf_repo_name: str | None = None,
    perlayer_bitwidth: int = 16,
) -> dict[str, dict[str, Any]]:
    """Export Gemma4 token + per-layer embedding tables as host-side LUTs.

    Loads the original float tables as FP32, folds in embed_scale (sqrt(dim)),
    computes a global asymmetric encoding, and writes the .bin LUTs + encoding
    JSONs. Returns embedding / perlayer_embedding dicts (lut-path, size, bw,
    scale, offset) for injection into genie_config.json.

    The token table is always ufixed16; ``perlayer_bitwidth`` sets the PLE width.
    """
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ple_total = num_layers * ple_dim

    # Resolve the safetensors shard(s) from a local dir (checkpoint, then
    # hf_repo_name), else an HF repo id -- a calibrated w4a16 dir ships no
    # safetensors. Never pass a local path to snapshot_download.
    def _looks_like_repo_id(s: str) -> bool:
        return bool(s) and "/" in s and not Path(s).exists() and not s.startswith("/")

    try:
        shard_paths = resolve_shard_paths(str(checkpoint))
    except FileNotFoundError:
        try:
            shard_paths = resolve_shard_paths(str(hf_repo_name)) if hf_repo_name else []
        except FileNotFoundError:
            shard_paths = []
        if not shard_paths:
            repo_id = None
            if _looks_like_repo_id(str(checkpoint)):
                repo_id = str(checkpoint)
            elif hf_repo_name and _looks_like_repo_id(str(hf_repo_name)):
                repo_id = str(hf_repo_name)
            if repo_id is None:
                raise FileNotFoundError(
                    f"Could not locate safetensors for embedding export: "
                    f"checkpoint '{checkpoint}' has no local safetensors and is "
                    f"not a valid HF repo id, and hf_repo_name='{hf_repo_name}' "
                    f"is neither a local safetensors dir nor a valid HF repo "
                    f"id. The safetensors checkpoint is required to export the "
                    f"host-side embedding LUTs."
                ) from None
            shard_paths = resolve_shard_paths(
                snapshot_download(repo_id=repo_id, allow_patterns=["*.safetensors*"])
            )

    def _dequant_embedding(f: Any, base: str) -> np.ndarray:
        """Load an original (float) embedding table as float32."""
        fp_key = f"{base}.weight"
        if fp_key not in f:
            # Name the checkpoint and expected key rather than letting
            # get_tensor raise a bare KeyError, matching the diagnostics of the
            # shard-resolution failure above.
            raise KeyError(
                f"No embedding weights for '{base}' in {checkpoint}. Expected "
                f"a float tensor ('{fp_key}')."
            )
        # .float() already yields float32; copy=False avoids a second full-table
        # copy (10.5 GiB for E4B's PLE table).
        return np.asarray(f.get_tensor(fp_key).float().cpu().numpy(), np.float32)

    result: dict[str, dict[str, Any]] = {}
    with _MultiShardSafetensors(shard_paths) as f:
        # Main token embedding: multiply by embed_scale = sqrt(hidden_size).
        emb = _dequant_embedding(f, "model.language_model.embed_tokens")
        emb *= float(hidden_size) ** 0.5
        emb_enc = _global_uint16_encoding(emb)
        emb_lut_name = "embedding_int16_lut.bin"
        emb_enc_name = "embed_encodings.json"
        _write_uint16_lut(
            emb,
            emb_enc["scale"],
            emb_enc["offset"],
            output_dir / emb_lut_name,
        )
        with open(output_dir / emb_enc_name, "w") as jf:
            json.dump({"lut_enc": emb_enc, "size": hidden_size}, jf, indent=4)
        result["embedding"] = {
            "lut-path": emb_lut_name,
            "enc-path": emb_enc_name,
            "size": hidden_size,
            "bw": emb_enc["bw"],
            "scale": emb_enc["scale"],
            "offset": emb_enc["offset"],
        }
        # Release the token table before loading the larger PLE one, so only one
        # multi-GiB table is resident at a time.
        del emb

        # Per-layer (PLE) embedding: embed_scale = sqrt(ple_dim).
        ple = _dequant_embedding(f, "model.language_model.embed_tokens_per_layer")
        ple *= float(ple_dim) ** 0.5
        ple_enc = _global_uint16_encoding(ple, perlayer_bitwidth)
        ple_lut_name = f"embed_token_int{perlayer_bitwidth}_lut.bin"
        ple_enc_name = "embed_tokens_encodings.json"
        _write_uint16_lut(
            ple,
            ple_enc["scale"],
            ple_enc["offset"],
            output_dir / ple_lut_name,
            perlayer_bitwidth,
        )
        with open(output_dir / ple_enc_name, "w") as jf:
            json.dump({"lut_enc": ple_enc, "size": ple_total}, jf, indent=4)
        result["perlayer_embedding"] = {
            "lut-path": ple_lut_name,
            "enc-path": ple_enc_name,
            "size": ple_total,
            "bw": ple_enc["bw"],
            "scale": ple_enc["scale"],
            "offset": ple_enc["offset"],
        }

    return result


# ---------------------------------------------------------------------------
# Gemma4 Base Classes
# ---------------------------------------------------------------------------


class Gemma4Base(LLMBase):
    """Base class for Gemma4 LLMs.

    External-embedding (genie_input_embeds) IO type, matching the reference
    QcGemma4 adaptation and the VLM models (qwen2_vl): the two embedding tables
    live OUTSIDE the ONNX graph. The graph takes:
    - inputs_embeds: (1, seq, hidden) — token embeddings (host-side Gather)
    - per_layer_inputs: (1, seq, num_layers, hidden_size_per_layer_input) —
      raw per-layer embeddings (host-side Gather + reshape); the PLE projection
      and gating stay IN-graph.
    Dual RoPE (SWA + Global) is passed as 4 precomputed cos/sin tensors, and the
    two attention masks (global + sliding-window) as separate inputs.
    """

    LMClass: type | None = None  # Set after monkey_patch
    EmbeddingClass = Gemma4RopeEmbedding

    # Architecture parameters (override in subclass)
    num_kv_shared_layers: int = 0
    sliding_window_pattern: int = 5  # 4 SWA + 1 global
    sliding_window: int = 512  # SWA window size (KV cache length for SWA layers)
    head_dim: int = 256  # SWA head_dim
    global_head_dim: int = 512  # Global attention head_dim
    hidden_size_per_layer_input: int = 256  # PLE dim
    # Total decoder layers. Annotation only (no value): the concrete PreSplit /
    # Part classes assign it. Declaring it here stops torch's
    # Module.__getattr__ -> Tensor | Module from masking the int type.
    num_layers: int

    # Default IO type: external embeddings (inputs_embeds + per_layer_inputs)
    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        """Genie turn tokens for Gemma4.

        Without this the base returns {} and the bundle ships a GenieChatTemplate
        with every turn token empty, so Genie concatenates raw message text with
        no role markers. think_* are set separately by the Collection.
        """
        return {
            "global_prefix": BOS_TOKEN,
            "system_prefix": f"{START_TURN}{SYSTEM_ID}\n",
            "system_suffix": f"{END_TURN}\n",
            "user_prefix": f"{START_TURN}{USER_ID}\n",
            "user_suffix": f"{END_TURN}\n",
            "assistant_prefix": f"{START_TURN}{ASSISTANT_ID}\n",
            "assistant_suffix": f"{END_TURN}\n",
            "vision_start": VISION_START,
            "vision_end": VISION_END,
            "default_system_prompt": cls.default_system_prompt,
        }

    @property
    def main_input_name(self) -> str:
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"  # HuggingFace uses 'inputs_embeds' (with 's')
        return "input_ids"

    @staticmethod
    def monkey_patch(skip_optimizations: list[str] | None = None) -> None:
        """Replace HF Gemma4 classes with QC-optimized versions.

        Imports transformers.models.gemma4.modeling_gemma4 and applies all
        patches (SHA attention, bypass RoPE, apply_rotary_pos_emb, MLP, lm_head
        prepare_conv). Called by LLMBase.__init__ before LMClass.from_pretrained.
        Also sets Gemma4Base.LMClass to the (patched) Gemma4ForCausalLM.
        """
        Gemma4Base._apply_monkey_patch_to_module(modeling_gemma4, skip_optimizations)
        # LMClass points at the patched CausalLM (now has prepare_conv)
        Gemma4Base.LMClass = modeling_gemma4.Gemma4ForCausalLM

    @staticmethod
    def _apply_monkey_patch_to_module(
        modeling_module: Any, skip_optimizations: list[str] | None = None
    ) -> None:
        """Apply monkey patches to the loaded modeling module."""
        if not (skip_optimizations and "sha_attention" in skip_optimizations):
            modeling_module.Gemma4TextAttention = SHAGemma4Attention

        # Our QC forward takes the two attention masks (global + sliding-window)
        # and the dual RoPE embeddings as explicit precomputed inputs, routing
        # each per layer_type.
        modeling_module.Gemma4TextModel.forward = qc_gemma4_text_model_forward

        def QcGemma4_apply_rotary_pos_emb(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            unsqueeze_dim: int = 1,
        ) -> torch.Tensor:
            """Apply rotary using precomputed cos/sin via _apply_rope_single."""
            return _apply_rope_single(x, (cos, sin))

        modeling_module.apply_rotary_pos_emb = QcGemma4_apply_rotary_pos_emb
        modeling_module.Gemma4TextMLP = QCGemma4MLP

        # Convert each decoder layer's per-layer-embedding (PLE) projections to
        # Conv2d so their weights become named Conv initializers in the exported
        # ONNX, which is what lets the weight encodings attach by name.
        decoder_cls = modeling_module.Gemma4TextDecoderLayer

        def _prepare_conv_decoder(self: Any) -> None:
            if getattr(self, "_ple_conv_done", False):
                return
            if hasattr(self, "per_layer_input_gate") and isinstance(
                self.per_layer_input_gate, torch.nn.Linear
            ):
                self.per_layer_input_gate = ConvInplaceLinear(self.per_layer_input_gate)
            if hasattr(self, "per_layer_projection") and isinstance(
                self.per_layer_projection, torch.nn.Linear
            ):
                self.per_layer_projection = ConvInplaceLinear(self.per_layer_projection)
            self._ple_conv_done = True

        decoder_cls.prepare_conv = _prepare_conv_decoder

        # Add prepare_conv to ForCausalLM
        orig_causal_lm = modeling_module.Gemma4ForCausalLM

        def _prepare_conv_lm_head(self: Any) -> None:
            self.lm_head = ConvInplaceLinear(self.lm_head)

        orig_causal_lm.prepare_conv = _prepare_conv_lm_head

    def _verify_ckpt(self) -> None:
        if self.llm_config.model_type not in ("gemma4", "gemma4_text"):
            raise ValueError(
                f"Model config type '{self.llm_config.model_type}' is not compatible "
                "with Gemma4 implementation."
            )

    @staticmethod
    def _get_input_spec(
        num_hidden_layers: int,
        sequence_length: int,
        context_length: int,
        hidden_size: int,
        num_key_value_heads: int,
        num_attention_heads: int,
        head_dim: int | None = None,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
        # Gemma4-specific geometry. Declared AFTER llm_io_type so every parameter
        # LLMBase._get_input_spec defines keeps its position in this signature.
        global_head_dim: int | None = None,
        num_kv_shared_layers: int = 0,
        sliding_window_pattern: int = 5,
        partial_rotary_factor: float = 0.25,
        hidden_size_per_layer_input: int = 256,
        sliding_window: int = 512,
    ) -> InputSpec:
        """Build input spec for Gemma4 with per-layer KV shapes and dual RoPE.

        Only non-shared layers get KV I/O entries.
        SWA (sliding_attention) layers use head_dim and a WINDOW-sized KV cache
        (sliding_window), global (full_attention) layers use global_head_dim and
        a full-length KV cache. The two attention masks have matching widths:
        attention_mask (global) is full context, swa_attention_mask is
        sliding_window + sequence_length.
        RoPE: two sets of cos/sin (SWA full rotation + Global partial rotation).
        Embeddings are external: inputs_embeds + per_layer_inputs are graph
        inputs (genie_input_embeds), not input_ids.
        """
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        if global_head_dim is None:
            global_head_dim = head_dim

        # Authored as (shape, dtype) tuples, normalized to TensorSpec at the
        # return below.
        input_spec: dict[str, tuple[tuple[int, ...], str]] = {}

        # External-embedding mode feeds the post-Gather token embeddings and the
        # raw per-layer embeddings directly (both computed host-side); the
        # embedding tables are not in the graph.
        if llm_io_type == LLMIOType.genie_input_embeds:
            input_spec["inputs_embeds"] = ((1, sequence_length, hidden_size), "float32")
            input_spec["per_layer_inputs"] = (
                (1, sequence_length, num_hidden_layers, hidden_size_per_layer_input),
                "float32",
            )
        else:
            input_spec["input_ids"] = ((1, sequence_length), "int32")

        # KV cache lengths: global layers cache the full past (context - seq),
        # SWA layers cache only the sliding window.
        assert sequence_length < context_length
        kv_seq_len = context_length - sequence_length
        swa_kv_len = min(sliding_window, kv_seq_len)

        # Two separate 4D masks, routed per layer type: attention_mask is
        # full-causal over context_length, swa_attention_mask is windowed-causal
        # over swa_kv_len + seq.
        input_spec["attention_mask"] = (
            (1, 1, sequence_length, context_length),
            "float32",
        )
        input_spec["swa_attention_mask"] = (
            (1, 1, sequence_length, swa_kv_len + sequence_length),
            "float32",
        )

        if llm_io_type == LLMIOType.huggingface_input_ids:
            # Integer position IDs (model computes RoPE internally)
            input_spec["position_ids"] = ((1, sequence_length), "int32")
        else:
            # Precomputed RoPE cos/sin (genie_input_ids mode)
            # SWA RoPE: full rotation, embed_dim = head_dim / 2
            swa_embed_dim = head_dim // 2
            input_spec["swa_position_ids_cos"] = (
                (1, 1, sequence_length, swa_embed_dim),
                "float32",
            )
            input_spec["swa_position_ids_sin"] = (
                (1, 1, sequence_length, swa_embed_dim),
                "float32",
            )
            # Global RoPE: proportional rotation, so cos/sin span the full
            # global_head_dim//2 width even though only the first
            # (global_head_dim*partial_factor)//2 frequencies are real.
            global_embed_dim = global_head_dim // 2
            input_spec["position_ids_cos"] = (
                (1, 1, sequence_length, global_embed_dim),
                "float32",
            )
            input_spec["position_ids_sin"] = (
                (1, 1, sequence_length, global_embed_dim),
                "float32",
            )

        # KV cache inputs for non-shared layers only. SWA layers use a
        # window-sized cache (swa_kv_len), global layers the full kv_seq_len.
        layer_types = get_gemma4_layer_types(num_hidden_layers, sliding_window_pattern)
        non_shared_indices = get_non_shared_layer_indices(
            num_hidden_layers, num_kv_shared_layers
        )

        for idx in non_shared_indices:
            is_sliding = layer_types[idx] == "sliding_attention"
            layer_hd = head_dim if is_sliding else global_head_dim
            layer_kv = swa_kv_len if is_sliding else kv_seq_len
            pfx = kv_prefix(layer_types[idx])

            for h in range(num_key_value_heads):
                # Per-head KV tensors: each is (1, 1, head_dim, kv_len), batch=1.
                # GQA (num_key_value_heads>1) uses separate tensors per head so
                # Genie's ring-buffer always sees leading batch dim = 1.
                input_spec[f"{pfx}_key_{idx}_h{h}_in"] = (
                    (1, 1, layer_hd, layer_kv),
                    "float32",
                )
                input_spec[f"{pfx}_value_{idx}_h{h}_in"] = (
                    (1, 1, layer_kv, layer_hd),
                    "float32",
                )

        # The compile/profile path expects TensorSpec objects, not tuples.
        return {
            k: TensorSpec(shape=tuple(shape), dtype=dtype)
            for k, (shape, dtype) in input_spec.items()
        }

    @staticmethod
    def _get_output_spec(
        num_hidden_layers: int,
        num_kv_shared_layers: int = 0,
        sliding_window_pattern: int = 5,
        num_key_value_heads: int = 1,
    ) -> dict[str, TensorSpec]:
        """Output spec with KV outputs only for non-shared layers.

        KV outputs use per-head naming ({pfx}_key_{idx}_h{h}_out) so each
        head is a separate tensor with batch dim = 1.  This lets Genie's
        ring-buffer manage each head independently (required for GQA
        num_key_value_heads > 1; a multi-head stacked tensor causes HTP
        Error 6004 at execute).
        """
        output_spec: dict[str, TensorSpec] = {"logits": TensorSpec()}
        layer_types = get_gemma4_layer_types(num_hidden_layers, sliding_window_pattern)
        non_shared_indices = get_non_shared_layer_indices(
            num_hidden_layers, num_kv_shared_layers
        )
        for idx in non_shared_indices:
            pfx = kv_prefix(layer_types[idx])
            for h in range(num_key_value_heads):
                output_spec[f"{pfx}_key_{idx}_h{h}_out"] = TensorSpec()
                output_spec[f"{pfx}_value_{idx}_h{h}_out"] = TensorSpec()
        return output_spec

    def forward(
        self,
        *inputs: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward pass for ONNX tracing.

        Positional inputs follow _get_input_spec order exactly.

        genie_input_embeds mode (default, external embeddings):
            inputs_embeds:       (1, seq, hidden) float32
            per_layer_inputs:    (1, seq, num_layers, ple_dim) float32
            attention_mask:      (1, 1, seq, ctx) float32  (global, full causal)
            swa_attention_mask:  (1, 1, seq, ctx) float32  (sliding-window causal)
            swa_position_ids_cos:    (1, 1, seq, swa_embed_dim)    - SWA RoPE
            swa_position_ids_sin:    (1, 1, seq, swa_embed_dim)
            position_ids_cos:        (1, 1, seq, global_embed_dim) - Global RoPE
            position_ids_sin:        (1, 1, seq, global_embed_dim)
            past_key/value pairs for non-shared layers only

        genie_input_ids mode (legacy): input_ids replaces
            (inputs_embeds, per_layer_inputs); RoPE/masks as above.

        huggingface_input_ids mode: input_ids + single position_ids (int),
            model computes RoPE/PLE internally.
        """
        inputs_embeds = None
        per_layer_inputs = None
        input_tokens = None

        if self.llm_io_type == LLMIOType.genie_input_embeds:
            inputs_embeds = inputs[0]
            per_layer_inputs = inputs[1]
            attention_mask = inputs[2]
            swa_attention_mask = inputs[3]
            rope_and_kv = inputs[4:]
        else:
            input_tokens = inputs[0]
            attention_mask = inputs[1]
            swa_attention_mask = inputs[2]
            rope_and_kv = inputs[3:]

        if self.llm_io_type == LLMIOType.huggingface_input_ids:
            position_ids = rope_and_kv[0]
            past_key_values_flat = rope_and_kv[1:]
            # RoPE computed internally by model
            position_embeddings_swa = None
            position_embeddings_global = None
        else:
            # Precomputed RoPE cos/sin
            position_ids_cos_swa = rope_and_kv[0]
            position_ids_sin_swa = rope_and_kv[1]
            position_ids_cos = rope_and_kv[2]
            position_ids_sin = rope_and_kv[3]
            past_key_values_flat = rope_and_kv[4:]
            position_ids = None  # Will be derived from attention_mask
            position_embeddings_swa = (position_ids_cos_swa, position_ids_sin_swa)
            position_embeddings_global = (
                position_ids_cos,
                position_ids_sin,
            )

        num_layers = self.llm_config.num_hidden_layers
        num_kv_shared = getattr(self.llm_config, "num_kv_shared_layers", 0)
        non_shared_indices = get_non_shared_layer_indices(num_layers, num_kv_shared)

        # Build KV cache for non-shared layers.
        # Input flat list layout: for each non-shared layer, num_kv_heads pairs
        # of (k_hN, v_hN) in head order. Total = num_non_shared * num_kv_heads * 2.
        assert isinstance(self.llm_config.num_key_value_heads, int)
        num_kv_heads = self.llm_config.num_key_value_heads
        kv_cache = SHADynamicCacheNewValueOnly()

        entries_per_layer = num_kv_heads * 2
        for cache_idx, actual_layer_idx in enumerate(non_shared_indices):
            offset = cache_idx * entries_per_layer
            layer_flat = past_key_values_flat[offset : offset + entries_per_layer]
            k_split = [layer_flat[h * 2] for h in range(num_kv_heads)]
            v_split = [layer_flat[h * 2 + 1] for h in range(num_kv_heads)]
            kv_cache.update(k_split, v_split, actual_layer_idx, {})

        # Calls the monkey-patched QC Gemma4TextModel.forward, which routes
        # attention_mask/position_ids to the full_attention layers and
        # swa_attention_mask/swa_position_ids to the sliding_attention ones.
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            token_kwargs: dict[str, Any] = {
                "inputs_embeds": inputs_embeds,
                "per_layer_inputs": per_layer_inputs,
            }
        else:
            token_kwargs = {"input_ids": input_tokens}

        if self.llm_io_type == LLMIOType.huggingface_input_ids:
            model_kwargs = {
                **token_kwargs,
                "attention_mask": attention_mask,
                "swa_attention_mask": swa_attention_mask,
                "position_ids": position_ids,
                "past_key_values": kv_cache,
                "use_cache": True,
            }
        else:
            # Precomputed dual RoPE: global as position_ids, SWA as
            # swa_position_ids (both (cos, sin) tuples).
            model_kwargs = {
                **token_kwargs,
                "attention_mask": attention_mask,
                "swa_attention_mask": swa_attention_mask,
                "position_ids": position_embeddings_global,
                "swa_position_ids": position_embeddings_swa,
                "past_key_values": kv_cache,
                "use_cache": True,
            }

        out = self.model(**model_kwargs)

        # One graph output per KV head, for non-shared layers only. Genie's
        # ring-buffer requires batch=1 per head; stacking heads into one tensor
        # with num_kv_heads>1 gives GQA models HTP Error 6004 at execute.
        out_cache = out["past_key_values"]
        flat_output_past_key_values = []
        for actual_layer_idx in non_shared_indices:
            if hasattr(out_cache, "key_cache"):
                k_heads = out_cache.key_cache[actual_layer_idx]
                v_heads = out_cache.value_cache[actual_layer_idx]
            elif hasattr(out_cache.layers[actual_layer_idx], "keys"):
                k_heads = out_cache.layers[actual_layer_idx].keys
                v_heads = out_cache.layers[actual_layer_idx].values
            else:
                k_heads = out_cache.layers[actual_layer_idx][0]
                v_heads = out_cache.layers[actual_layer_idx][1]
            # Interleave k/v per head (k_h0, v_h0, k_h1, ...) to match the spec.
            for k, v in zip(k_heads, v_heads, strict=False):
                flat_output_past_key_values += [k, v]

        return [out["logits"], *flat_output_past_key_values]

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        """Add chat-formatted WikiText to the shared LLM eval tasks.

        Gemma4 ships only ``-it`` (instruction-tuned) checkpoints, whose
        deployed input is always chat-templated. Raw-WikiText perplexity is
        therefore a poor proxy for on-device quality; ``wikitext_chat`` scores
        the same corpus in the format the model actually sees. Both are kept so
        a calibration change can be reported against each rather than whichever
        number looks better.
        """
        return [*super().get_eval_dataset_classes(), WikiTextChat]


# ---------------------------------------------------------------------------
# Dynamic-shape Gemma4 classes
# ---------------------------------------------------------------------------


class Gemma4DynamicBase(LLMDynamicBase, Gemma4Base):
    """Gemma4 FP base with dynamic-shape ONNX export."""

    def _build_dynamic_shapes(self) -> dict[str, dict[int, Any]]:
        """Build the per-input dynamic dims for Gemma4's input layout.

        Returned as ``extra_dynamic_shapes`` for ``get_onnx_model``, which merges
        it over its own defaults. Every Gemma4 input is named here because each
        one differs from the generic LLM layout: inputs_embeds +
        per_layer_inputs, two masks, dual RoPE, and per-layer KV whose length
        depends on whether the layer is sliding or global.

        Sequence dims share one named Dim, global KV dims share kv_seq_len, and
        SWA KV dims share a separate swa_kv_len. Named Dims rather than
        Dim.DYNAMIC because the derived swa mask width needs arithmetic
        (Dim.DYNAMIC raises TypeError on ``+``) and the AI Hub compiler rejects
        independent per-tensor symbols given concrete shapes.

        Returns
        -------
        dict[str, dict[int, Any]]
            Input name -> {axis index: Dim} for every graph input.
        """
        # Bounds: keep seq/kv well under the vocab-derived int32 guard that
        # torch.export infers from the lm_head/embedding (2**31 / vocab_size,
        # ~8192 for vocab 262144). 4096 matches the real max context length.
        seq_d = Dim("seq_len", min=1, max=4096)
        kv_d = Dim("kv_seq_len", min=1, max=4096)
        swa_kv_d = Dim("swa_kv_len", min=1, max=4096)
        ctx_d = Dim("ctx_len", min=2, max=8192)

        spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=self.sequence_length,
            context_length=self.context_length,
            llm_io_type=self.llm_io_type,
        )

        layer_types = get_gemma4_layer_types(
            self.num_layers, self.sliding_window_pattern
        )
        kv_re = re.compile(r"(?:past|swa)_(key|value)_(\d+)_")

        per_input: dict[str, dict[int, Any]] = {}
        for name in spec:
            kv_m = kv_re.search(name)
            if name == "inputs_embeds":
                per_input[name] = {1: seq_d}  # (1, seq, hidden)
            elif name == "per_layer_inputs":
                per_input[name] = {1: seq_d}  # (1, seq, num_layers, ple)
            elif name == "input_ids":
                per_input[name] = {1: seq_d}  # (1, seq)
            elif name == "attention_mask":
                per_input[name] = {2: seq_d, 3: ctx_d}  # global, full ctx
            elif name == "swa_attention_mask":
                # windowed: width = seq + sliding_window (derived Dim+int)
                per_input[name] = {2: seq_d, 3: seq_d + self.sliding_window}
            elif name == "position_ids":
                per_input[name] = {1: seq_d}  # (1, seq)
            elif "position_ids" in name and ("cos" in name or "sin" in name):
                per_input[name] = {2: seq_d}  # (1, 1, seq, embed_dim)
            elif kv_m:
                is_sliding = layer_types[int(kv_m.group(2))] == "sliding_attention"
                d = swa_kv_d if is_sliding else kv_d
                if kv_m.group(1) == "key":
                    per_input[name] = {3: d}  # (1, nkv, head_dim, kv)
                else:  # value
                    per_input[name] = {2: d}  # (1, nkv, kv, head_dim)

        return per_input

    def get_full_onnx_bundle(self, temp_path: Path) -> ONNXBundle:
        """Export full ONNX from PyTorch with dynamic shapes."""
        # Own directory, not default_checkpoint's: that is where
        # fetch_default_checkpoint() extracts the published w4a16 zip, which the
        # probe below would then load as if it were this FP export.
        # Concrete models always set model_id; the guard is for the bases, which
        # default it to "" (an empty id would cache into the store root).
        cache_dir = (
            ASSET_CONFIG.get_local_store_model_path(
                self.model_id, self.model_asset_version, "fp_dynamic_onnx"
            )
            if self.model_id
            else None
        )

        if cache_dir is not None:
            cached_onnx = cache_dir / "model_dynamic.onnx"
            cached_data = cache_dir / "model.data"
            if cached_onnx.exists() and cached_data.exists():
                print(f"\nLoading cached dynamic ONNX from {cache_dir}")
                bundle_dir = temp_path / "full_dynamic"
                bundle_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(cached_onnx, bundle_dir / "model.onnx")
                shutil.copy(cached_data, bundle_dir / "model.data")
                return ONNXBundle.from_bundle_path(bundle_dir, "model")

        print_with_box(
            [
                "Exporting ONNX model with dynamic shapes.",
                "This may take around 30 minutes.",
            ]
        )
        onnx_dir = temp_path / "full_dynamic"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"
        get_onnx_model(
            fp_model=self,
            context_length=self.context_length,
            sequence_length=self.sequence_length,
            path=str(onnx_path),
            return_model=False,
            llm_io_type=self.llm_io_type,
            quiet=True,
            extra_dynamic_shapes=self._build_dynamic_shapes(),
        )

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(onnx_dir / "model.onnx", cache_dir / "model_dynamic.onnx")
            shutil.copy(onnx_dir / "model.data", cache_dir / "model.data")
            print(f"\nCached dynamic ONNX to {cache_dir}")

        return ONNXBundle.from_bundle_path(onnx_dir, "model")


def build_gemma4_genie_inputs(
    input_ids: torch.Tensor,
    text_model: Any,
    llm_config: Any,
    sequence_length: int,
    context_length: int,
    num_layers: int,
    sliding_window_pattern: int,
    num_kv_shared_layers: int,
    head_dim: int,
    global_head_dim: int,
    sliding_window: int,
    rope: Gemma4RopeEmbedding,
    attn_min: float = -100.0,
    past_key_values: list[torch.Tensor] | None = None,
    num_past_tokens: int = 0,
) -> dict[str, torch.Tensor]:
    """Assemble Gemma4's genie_input_embeds inputs for a single prefill.

    Produces the exact input dict (in get_input_spec order) the Gemma4
    PreSplit/QuantSim forward expects: external embeddings + per-layer inputs,
    dual (global/SWA) attention masks, dual RoPE cos/sin, and the per-head KV
    cache. ``text_model`` is the FP text model exposing embed_tokens +
    get_per_layer_inputs.

    ``past_key_values`` carries the accumulated cache from prior chunks (flat
    list, key/value interleaved in get_output_spec order) and
    ``num_past_tokens`` how many real tokens it holds; both default to an empty
    cache for a standalone prefill. Passing them lets PPL eval score a full
    ``context_length`` sample as chunks that attend to earlier text, rather than
    scoring every chunk from a cold cache.
    """
    layer_types = get_gemma4_layer_types(num_layers, sliding_window_pattern)
    non_shared_indices = get_non_shared_layer_indices(num_layers, num_kv_shared_layers)
    num_kv_heads = llm_config.num_key_value_heads
    kv_seq_len = context_length - sequence_length
    swa_kv_len = min(sliding_window, kv_seq_len)

    input_ids = input_ids.to(torch.long)
    length = input_ids.shape[1]
    if length < sequence_length:
        pad = torch.zeros(
            (1, sequence_length - length), dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat((pad, input_ids), dim=1)
    else:
        input_ids = input_ids[:, :sequence_length]
        length = sequence_length

    # The embedding tables may live on a different device than input_ids (FP
    # text model on CPU, QuantSim generator on CUDA). Align for the lookup.
    embed_device = text_model.embed_tokens.weight.device
    input_ids = input_ids.to(embed_device)

    with torch.no_grad():
        inputs_embeds = text_model.embed_tokens(input_ids)
        per_layer_inputs = text_model.get_per_layer_inputs(input_ids, inputs_embeds)

    # Mark the right-most ``num_past_tokens`` cache slots as attendable so this
    # chunk can see the preceding text; the rest of the cache stays masked.
    kv_mask = torch.zeros((1, kv_seq_len), dtype=torch.float32)
    if num_past_tokens > 0:
        attendable = min(num_past_tokens, kv_seq_len)
        kv_mask[:, kv_seq_len - attendable :] = 1.0
    seq_mask = torch.zeros((1, sequence_length), dtype=torch.float32)
    seq_mask[:, sequence_length - length :] = 1.0
    padded_mask = torch.cat((kv_mask, seq_mask), dim=-1)

    attention_mask = convert_2d_attention_mask_to_4d(
        padded_mask, sequence_length, context_length
    ).clip(attn_min, 0)
    swa_full = convert_2d_attention_mask_to_4d_sliding_window(
        padded_mask, sequence_length, context_length, sliding_window
    ).clip(attn_min, 0)
    swa_attention_mask = swa_full[
        ..., context_length - (swa_kv_len + sequence_length) :
    ]

    position_ids = (torch.cumsum(padded_mask, dim=1, dtype=torch.int32) - 1).clip(
        0, context_length - 1
    )[:, -sequence_length:]
    swa_cos, swa_sin = rope.get_embedding(position_ids)
    global_cos, global_sin = rope.get_global_embedding(position_ids)

    sample: dict[str, torch.Tensor] = {
        "inputs_embeds": inputs_embeds.to(torch.float32),
        "per_layer_inputs": per_layer_inputs.to(torch.float32),
        "attention_mask": attention_mask.to(torch.float32),
        "swa_attention_mask": swa_attention_mask.to(torch.float32),
        "swa_position_ids_cos": swa_cos.to(torch.float32),
        "swa_position_ids_sin": swa_sin.to(torch.float32),
        "position_ids_cos": global_cos.to(torch.float32),
        "position_ids_sin": global_sin.to(torch.float32),
    }
    # KV cache inputs, in the same (layer, head, key/value) order the model's
    # get_output_spec emits them, so a prior step's outputs feed straight back in.
    kv_iter = iter(past_key_values or [])
    for idx in non_shared_indices:
        is_sliding = layer_types[idx] == "sliding_attention"
        layer_hd = head_dim if is_sliding else global_head_dim
        layer_kv = swa_kv_len if is_sliding else kv_seq_len
        pfx = kv_prefix(layer_types[idx])
        for h in range(num_kv_heads):
            key_shape = (1, 1, layer_hd, layer_kv)
            value_shape = (1, 1, layer_kv, layer_hd)
            past_key = next(kv_iter, None)
            past_value = next(kv_iter, None)
            # The cache arrives in HF layout (batch, heads, kv_len, head_dim)
            # because TransposedKVGeneratorMixin already un-permuted the graph's
            # key outputs. Keys need that swap re-applied; values do not.
            if past_key is not None:
                past_key = past_key.transpose(-1, -2)
            sample[f"{pfx}_key_{idx}_h{h}_in"] = _fit_kv(past_key, key_shape, -1)
            sample[f"{pfx}_value_{idx}_h{h}_in"] = _fit_kv(past_value, value_shape, -2)
    return sample


# HF-layout KV caches carry the token count on dim -2 (batch, heads, seq, dim).
_HF_KV_SEQ_AXIS = -2


def _fit_kv(
    tensor: torch.Tensor | None, shape: tuple[int, ...], seq_axis: int
) -> torch.Tensor:
    """Right-align ``tensor`` into a zero cache of ``shape`` along ``seq_axis``.

    The graph's KV inputs are fixed-width (``context_length - sequence_length``,
    or the SWA window). A cache carried from earlier chunks may be shorter (early
    steps) or longer (once it exceeds the window), so keep the most recent
    entries and left-pad with zeros -- matching the mask built above, which marks
    only the right-most ``num_past_tokens`` slots attendable.
    """
    out = torch.zeros(shape, dtype=torch.float32)
    if tensor is None:
        return out
    tensor = tensor.detach().to(torch.float32).cpu()
    if tensor.shape[seq_axis] == 0:
        return out
    width = shape[seq_axis]
    if tensor.shape[seq_axis] > width:
        tensor = tensor.narrow(seq_axis, tensor.shape[seq_axis] - width, width)
    n = tensor.shape[seq_axis]
    # Guard against a shape mismatch on the non-sequence dims (would otherwise
    # broadcast silently and corrupt the cache).
    expected = list(shape)
    expected[seq_axis] = n
    if list(tensor.shape) != expected:
        return out
    out.narrow(seq_axis, width - n, n).copy_(tensor)
    return out


class Gemma4Generator(HubCompatibleGenerator):
    """PPL generator for Gemma4's genie_input_embeds I/O.

    The base HubCompatibleGenerator builds generic LLM inputs; Gemma4 instead
    needs inputs_embeds + per_layer_inputs, dual (global/SWA) masks, dual RoPE,
    and per-head SWA/global KV. This override assembles those via
    build_gemma4_genie_inputs so ``evaluate.py --use-presplit`` works for both
    the FP PreSplit and the QuantSim (w4a16) model. The QuantSim (ONNX) has no
    torch embed tables, so a FP text model is lazily loaded once from the
    original checkpoint and cached.
    """

    # Keyed by checkpoint so a process touching both E2B and E4B doesn't get one
    # variant's FP embeddings back for the other (silently wrong numbers).
    _fp_text_model_cache: dict[str, Any] = {}

    @staticmethod
    def _output_names_from_descriptors(
        layer_cache_descriptors: list,
    ) -> list[str]:
        """Name one KV pair per (layer, head), matching Gemma4's forward outputs.

        The base emits one pair per layer, assuming heads are stacked into a
        single tensor. Gemma4 emits them per head (Genie's ring buffer needs
        batch=1 per head), so on E4B (num_kv_heads=2) the forward returns twice
        the tensors the base names and ``parse_model_outputs`` zips them away --
        dropping half the cache and shifting the rest into wrong layer slots.

        Names stay ``past_``-prefixed and sequentially numbered so
        TransposedKVGeneratorMixin un-permutes all of them back to HF layout.

        Parameters
        ----------
        layer_cache_descriptors
            per-layer cache descriptors for the layers that own KV.

        Returns
        -------
        list[str]
            output names aligned 1:1 with the forward's returned tensors.
        """
        names = ["logits"]
        slot = 0
        for desc in layer_cache_descriptors:
            for _ in range(desc.num_kv_heads):
                names += [f"past_key_{slot}_out", f"past_value_{slot}_out"]
                slot += 1
        return names

    @classmethod
    def _resolve_gemma4_text_model(cls, gemma: Any) -> Any:
        """Return an FP text model exposing embed_tokens / get_per_layer_inputs."""
        # FP PreSplit already owns the torch text model.
        inner = getattr(gemma, "model", None)
        if inner is not None and hasattr(getattr(inner, "model", None), "embed_tokens"):
            return inner.model
        # QuantSim path: build (and cache) the FP text model once per checkpoint
        # from the original checkpoint (embeddings must match the checkpoint
        # being evaluated).
        fp_cls = gemma.FPModel
        checkpoint = os.environ.get("GEMMA4_LOCAL_CHECKPOINT") or getattr(
            fp_cls, "hf_repo_name", None
        )
        key = str(checkpoint)
        # Cache the text model, not the wrapper: release() nulls the wrapper's
        # .model, which this class-level cache outlives -- so caching the wrapper
        # hands back a gutted object after anything releases the FP model.
        cached = cls._fp_text_model_cache.get(key)
        if cached is None:
            cached = fp_cls.from_pretrained(
                checkpoint=checkpoint,
                host_device=torch.device("cpu"),
            ).model.model
            cls._fp_text_model_cache[key] = cached
        return cached

    @classmethod
    def prepare_inputs(  # type: ignore[override]
        cls,
        model: torch.nn.Module,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        past_key_values: list[torch.Tensor],
        sequence_length: int,
        context_length: int,
        pad_token: int = 0,
        attention_mask_min: int = -100,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        layer_cache_descriptors: list | None = None,
        **kwargs: Any,
    ) -> OrderedDict[str, torch.Tensor]:
        # The geometry attrs only live on the FP class, so source them from there
        # for both PreSplit (FP) and QuantizablePreSplit (QuantSim). Any:
        # Module.__getattr__ is typed Tensor | Module, hiding those attrs.
        gemma: Any = model._model
        geom: Any = gemma if hasattr(gemma, "head_dim") else gemma.FPModel
        text_model = cls._resolve_gemma4_text_model(gemma)
        llm_config = model.config
        rope = Gemma4RopeEmbedding(max_length=context_length, config=llm_config)
        assert input_ids is not None
        # Thread the accumulated cache through so each chunk attends to the
        # preceding text. Without it every chunk is scored from a cold cache and
        # PPL reflects a 128-token context regardless of context_length.
        num_past_tokens = 0
        if past_key_values:
            kv_seq_len = context_length - sequence_length
            num_past_tokens = min(
                int(past_key_values[0].shape[_HF_KV_SEQ_AXIS]), kv_seq_len
            )
        sample = build_gemma4_genie_inputs(
            input_ids=input_ids,
            text_model=text_model,
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            num_layers=geom.num_layers,
            sliding_window_pattern=geom.sliding_window_pattern,
            num_kv_shared_layers=geom.num_kv_shared_layers,
            head_dim=geom.head_dim,
            global_head_dim=geom.global_head_dim,
            sliding_window=geom.sliding_window,
            rope=rope,
            attn_min=float(attention_mask_min),
            past_key_values=list(past_key_values) if past_key_values else None,
            num_past_tokens=num_past_tokens,
        )
        prepared: OrderedDict[str, torch.Tensor] = OrderedDict()
        # Any: Module.__getattr__ types this as Tensor | Module, but the LLM
        # wrappers all carry a real torch.device.
        device: Any = model.device
        for name, tensor in sample.items():
            prepared[name] = tensor.to(device)
        return prepared


# ---------------------------------------------------------------------------
# PreSplit / Part / Collection bases
# ---------------------------------------------------------------------------


class Gemma4PreSplitBase(
    SingleSlotCacheMixin, DynamicPreSplitOnnxMixin, Gemma4DynamicBase
):
    """FP PreSplit base for Gemma4 models.

    Manages the full torch model and ONNX splitting.
    """

    GeneratorClass = Gemma4Generator

    # --- per-model configuration (override in subclass) ---
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 256
    global_head_dim: int = 512
    num_kv_shared_layers: int = 0
    sliding_window_pattern: int = 5
    sliding_window: int = 512
    partial_rotary_factor: float = 0.25
    hidden_size_per_layer_input: int = 256
    hf_repo_name: str = ""

    # Default AR sequence-length / context-length buckets used by the shared
    # quantize flow (stage A ONNX export) when the caller does not pass explicit
    # --sequence-length / --context-length values.
    default_sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS
    default_context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS
    split_model_name: str = ""
    num_splits: int = 0
    # None when the split is driven by `splitting_points` instead of a uniform
    # layers-per-split count. Widens DynamicPreSplitOnnxMixin's `int`.
    num_layers_per_split: int | None = 0  # type: ignore[assignment]
    # Embeddings are external (genie_input_embeds), so there is no embedding part
    # to split off.
    split_embedding: bool = False
    # KV-shared layers make a later split read an earlier split's KV-cache graph
    # outputs; those must become leaf inputs, or the partition drags the whole
    # earlier layer chain into the later part.
    reuse_emitted_graph_outputs: bool = True

    # Asset / cache config
    min_memory_recommended: int = 0
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict[Precision, str] = {}
    default_precision: Precision = Precision.w4a16

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path | None = None,
        *args: Any,
        load_pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        ckpt = checkpoint or self.hf_repo_name
        # HF from_pretrained cannot load the checkpoint's plain float weights
        # onto the SHA-adapted model, so always build from config
        # (load_pretrained=False) and load the state_dict from safetensors below.
        super().__init__(*args, checkpoint=ckpt, load_pretrained=False, **kwargs)
        if load_pretrained:
            self._load_dequantized_weights(ckpt)

    def _load_dequantized_weights(self, checkpoint: str | os.PathLike | Path) -> None:
        """Load original (float) weights onto the SHA-adapted model.

        Builds an FP32 state_dict keyed for the post-prepare_sha module
        structure (q_proj_sha.N, o_proj_conv, ...) and loads it non-strictly
        (norms/scalars not all present is fine). Accepts a single-file, sharded
        (``model.safetensors.index.json``), or loose-``*.safetensors``
        checkpoint directory.
        """
        ckpt_dir = Path(checkpoint)
        try:
            resolve_shard_paths(str(ckpt_dir))
            safetensors_source = ckpt_dir
        except FileNotFoundError:
            # Not a local checkpoint dir: treat ``checkpoint`` as an HF repo id
            # and download the full snapshot (handles sharded repos too).
            try:
                safetensors_source = Path(
                    snapshot_download(
                        repo_id=str(checkpoint), allow_patterns=["*.safetensors*"]
                    )
                )
            except (HfHubHTTPError, OSError) as e:
                raise FileNotFoundError(
                    f"Expected safetensors weights under {ckpt_dir}, and could "
                    f"not download them from HF repo '{checkpoint}'. Real-weight "
                    f"export requires the safetensors checkpoint (local dir or "
                    f"accessible HF repo). Original error: {e}"
                ) from e

        layer_types = get_gemma4_layer_types(
            self.num_layers, self.sliding_window_pattern
        )
        sd = load_dequantized_state_dict(
            safetensors_path=str(safetensors_source),
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            global_head_dim=self.global_head_dim,
            layer_types=layer_types,
            num_kv_shared_layers=self.num_kv_shared_layers,
        )
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(
            f"Loaded {len(sd)} dequantized tensors "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    def _verify_ckpt(self) -> None:
        """Verify checkpoint compatibility."""
        super()._verify_ckpt()
        if not (
            self.llm_config.num_hidden_layers == self.num_layers
            and self.llm_config.hidden_size == self.hidden_size
            and self.llm_config.num_attention_heads == self.num_attention_heads
            and self.llm_config.num_key_value_heads == self.num_key_value_heads
        ):
            raise ValueError("Model config is not compatible with our implementation.")

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: str | os.PathLike | Path | None = None,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ) -> Self:
        """Load or return a cached FP PreSplit."""
        checkpoint = checkpoint or cls.hf_repo_name
        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
            return cached

        instance = cls(
            checkpoint=checkpoint,
            host_device=host_device,
            load_pretrained=True,
            _skip_optimizations=_skip_optimizations,
        )
        cls.cache_store(instance, cache_key)
        return instance

    def get_output_spec(self) -> dict[str, TensorSpec]:
        """Get output names for the full model."""
        return Gemma4Base._get_output_spec(
            self.num_layers,
            self.num_kv_shared_layers,
            self.sliding_window_pattern,
            self.llm_config.num_key_value_heads,
        )

    def _get_output_spec(  # type: ignore[override]
        self, num_hidden_layers: int
    ) -> dict[str, TensorSpec]:
        """Instance override so the generic ONNX exporter's 1-arg call
        (``fp_model._get_output_spec(num_hidden_layers)`` in templates/llm/model.py)
        injects THIS model's num_kv_shared_layers + sliding_window_pattern.

        Without this, the call resolves to the static Gemma4Base._get_output_spec
        whose defaults (num_kv_shared_layers=0, sliding_window_pattern=5) mislabel
        the KV output tensor prefixes: e.g. for a pattern-6 model the exporter
        would name past_/swa_ by pattern 5 (global at 4,9,14,19) while the forward
        sizes them by pattern 6 (global at 5,11,17,23) -> swa_key_5 named sliding
        but sized 512 -> Genie I/O ShapeError. Invisible for E2B (pattern 5 ==
        default); E4B (pattern 6) is the first to hit it.
        """
        return Gemma4Base._get_output_spec(
            num_hidden_layers,
            self.num_kv_shared_layers,
            self.sliding_window_pattern,
            self.llm_config.num_key_value_heads,
        )

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
    ) -> InputSpec:
        return self._static_input_spec(
            llm_config, sequence_length, context_length, llm_io_type
        )

    @classmethod
    def _static_input_spec(
        cls,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
    ) -> InputSpec:
        """Build input spec from class constants."""
        return Gemma4Base._get_input_spec(
            num_hidden_layers=cls.num_layers,
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=cls.hidden_size,
            num_key_value_heads=cls.num_key_value_heads,
            num_attention_heads=cls.num_attention_heads,
            head_dim=cls.head_dim,
            global_head_dim=cls.global_head_dim,
            num_kv_shared_layers=cls.num_kv_shared_layers,
            sliding_window_pattern=cls.sliding_window_pattern,
            partial_rotary_factor=cls.partial_rotary_factor,
            hidden_size_per_layer_input=cls.hidden_size_per_layer_input,
            sliding_window=cls.sliding_window,
            llm_io_type=llm_io_type,
        )


Gemma4PreSplitT = TypeVar("Gemma4PreSplitT", bound=Gemma4PreSplitBase)


class Gemma4Base_AIMETOnnx(LLM_AIMETOnnx):
    """AIMET-ONNX quantized base for Gemma4 (mirrors Qwen3Base_AIMETOnnx)."""

    EmbeddingClass = Gemma4RopeEmbedding
    FPModel = Gemma4Base

    # Matches the FP Gemma4Base default; LLM_AIMETOnnx defaults to
    # genie_input_ids, which would mis-shape the calibration / input spec.
    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    # KV-shared layer count, set by concrete subclass so _get_output_spec
    # only emits KV for non-shared layers.
    num_kv_shared_layers: int = 0
    sliding_window_pattern: int = 5

    def _get_output_spec(self, num_hidden_layers: int) -> dict[str, TensorSpec]:
        return Gemma4Base._get_output_spec(
            num_hidden_layers,
            self.num_kv_shared_layers,
            self.sliding_window_pattern,
            self.llm_config.num_key_value_heads,
        )

    @staticmethod
    def _gemma4_kv_io_map(quant_sim: Any) -> dict[str, str]:
        """Pair every KV input tensor with its output, covering BOTH the
        sliding-window (swa_) and full-attention (past_) prefixes.

        The base _get_kv_io_map only matches 'past_key'/'past_value', so it
        misses Gemma4's swa_ KV tensors — leaving the 28 sliding layers' KV
        un-tied and causing "Non-identical quantization parameters" across the
        split. We pair by name: swa_key_3_in -> swa_key_3_out, etc. AIMET may
        append _updated/_qdq to quantized output tensor names; we strip those so
        the mapped value is the ORIGINAL (unquantized) graph tensor name that
        _get_enabled_quantizer / set_quantizers expect (matches the base
        _get_kv_io_map, which likewise strips '_updated').
        """

        def strip(name: str) -> str:
            for suf in ("_updated", "_qdq"):
                if name.endswith(suf):
                    return name[: -len(suf)]
            return name

        kv_re = re.compile(r"^((?:past|swa)_(?:key|value)_\d+(?:_h\d+)?)_in$")
        inputs = {
            m.group(1): t.name
            for t in quant_sim.model.graph().input
            if (m := kv_re.match(t.name))
        }
        out_re = re.compile(r"^((?:past|swa)_(?:key|value)_\d+(?:_h\d+)?)_out$")
        outputs = {}
        for t in quant_sim.model.graph().output:
            base = strip(t.name)
            m = out_re.match(base)
            if m:
                # Store the stripped (original-graph) output tensor name.
                outputs[m.group(1)] = base
        # Map input tensor name -> output tensor name for shared base ids.
        return {inputs[k]: outputs[k] for k in inputs if k in outputs}

    @classmethod
    def _configure_quant_sim(cls, quant_sim: Any, precision: Precision) -> Any:
        """Gemma4 override: minimal KV quantizer sharing without Concat tying.

        The base w4a16 path would apply _apply_int8_kv_cache_tying_and_lm_head
        which:
        1. _tie_quantizers_for_op_types(["Concat"]) — spreads int8 through the
           Concat chain including embedding downstream → creates a mass encoding
           mismatch with the int16 activations.
        2. Forces KV to int8 — conflicts with the int16 KV values.

        We do ONLY the minimal step needed: share KV in/out quantizer objects
        so that when compute_encodings calibrates, the same object gets one
        encoding for both the KV output (part1) and KV input (part2),
        guaranteeing cross-part consistency. No Concat tying, no bw-forcing.
        """
        if precision == Precision.w4a16:
            kv_io_map = cls._gemma4_kv_io_map(quant_sim)
            _tie_quantizers_for_kv_cache(quant_sim, kv_io_map)
        elif precision == Precision.w4:
            _set_lm_head_to_8b(quant_sim)
            cls._apply_precision_activations(quant_sim, precision)
        cls._hold_params_at_int8(quant_sim)
        return quant_sim


class Gemma4DynamicBase_AIMETOnnx(LLMDynamic_AIMETOnnx, Gemma4Base_AIMETOnnx):
    """Dynamic-shape variant of Gemma4Base_AIMETOnnx."""

    FPModel = Gemma4DynamicBase  # type: ignore[assignment]


class Gemma4QuantizablePreSplitBase(  # type: ignore[misc]
    DynamicQuantizablePreSplitMixin[Gemma4PreSplitT],
    Gemma4DynamicBase_AIMETOnnx,
    Generic[Gemma4PreSplitT],
):
    """Quantizable PreSplit base for Gemma4 models.

    Loads the dynamic ONNX + self-calibrated encodings and splits for deployment.
    """

    GeneratorClass = Gemma4Generator
    FPModel: type[Gemma4PreSplitT]

    num_layers: int = 0
    num_kv_shared_layers: int = 0
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict[Precision, str] = {}
    supported_precisions: list[Precision] = []
    default_precision: Precision = Precision.w4a16

    # Default AR sequence-length / context-length buckets used by the shared
    # quantize flow (stage A ONNX export) when the caller does not pass explicit
    # --sequence-length / --context-length values.
    default_sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS
    default_context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS
    num_splits: int = 0
    # None when the split is driven by `splitting_points` instead of a uniform
    # layers-per-split count. Widens DynamicPreSplitOnnxMixin's `int`; the value
    # is only forwarded to split_onnx(), whose parameter is already int | None.
    num_layers_per_split: int | None = 0  # type: ignore[assignment]
    # See Gemma4PreSplitBase: embeddings are external, no embedding part.
    split_embedding: bool = False
    # See Gemma4PreSplitBase: KV-shared layers make later splits read the
    # earlier split's KV-cache graph outputs.
    reuse_emitted_graph_outputs: bool = True

    def get_output_spec(self) -> dict[str, TensorSpec]:
        return Gemma4Base._get_output_spec(
            self.num_layers,
            self.FPModel.num_kv_shared_layers,
            self.FPModel.sliding_window_pattern,
            self.llm_config.num_key_value_heads,
        )

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
    ) -> InputSpec:
        return self.FPModel._static_input_spec(
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=llm_io_type,
        )

    def get_calibration_data(  # type: ignore[override]
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        image_size: tuple[int, int] | None = None,
        fp_model: LLMBase | None = None,
        use_chat_template: bool = True,
    ) -> DatasetEntries | None:
        """Build AIMET-ONNX activation-calibration data for Gemma4.

        The base ``LLMDynamic_AIMETOnnx.get_calibration_data`` prepares a single
        attention_mask, a single RoPE cos/sin and no per_layer_inputs, which does
        not match Gemma4's genie_input_embeds graph, so assemble each sample
        directly in ``get_input_spec()`` order. Embeddings are host-side, so
        ``fp_model`` supplies inputs_embeds and the raw per_layer_inputs (the
        in-graph project_per_layer_inputs is not applied here); the KV cache
        starts as zeros, since one prefill pass captures representative ranges.

        The leading parameters mirror the base signature so the shared
        ``templates/llm/quantize.py`` flow can call this by keyword.

        Parameters
        ----------
        num_samples
            number of WikiText samples (0 -> auto from ctx length).
        input_spec
            unused; the spec is always derived from *sequence_length* /
            *context_length* so it matches the assembled samples.
        sequence_length
            calibration sequence length.
        context_length
            calibration context length.
        image_size
            unused; this is the text graph, whose inputs carry no image.
        fp_model
            dequantized FP PreSplit model (provides embed_tokens /
            get_per_layer_inputs / tokenizer). Defaults to the cached
            ``self.FPModel.from_pretrained()`` instance.
        use_chat_template
            calibrate on chat-templated WikiText (``WikiTextChat``) instead of
            raw prose. Gemma4 ``-it`` models only ever see chat-formatted input
            on device, so raw text fits ranges to a distribution the deployed
            model never sees. Set False for the raw-text behaviour.

        Returns
        -------
        DatasetEntries | None
            AIMET-ONNX calibration inputs keyed by input-spec name, or None
            when no calibration data is produced.
        """
        if fp_model is None:
            fp_model = self.FPModel.from_pretrained()
        if num_samples == 0:
            num_samples = math.ceil(80000 / context_length)

        tokenizer = fp_model.tokenizer
        dataset_cls: type[WikiText] = (
            WikiTextChat if use_chat_template and tokenizer.chat_template else WikiText
        )
        if use_chat_template and not tokenizer.chat_template:
            print(
                "Calibration: tokenizer has no chat_template; falling back to "
                "raw WikiText."
            )
        print(f"Calibration dataset: {dataset_cls.dataset_name()}")
        dataset = instantiate_dataset(
            dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=None,
            tokenizer=tokenizer,
            block_size=sequence_length,
            context_length=context_length,
            num_samples=num_samples,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=self.llm_io_type,
        )
        assert input_spec is not None
        input_names = list(input_spec.keys())
        # Annotated with the union make_hub_dataset_entries accepts (it takes
        # numpy arrays too), so tuple(inputs) matches its parameter type.
        inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in input_names]

        # Host-side embedding tables live on the FP text model. Typed Any because
        # torch's Module.__getattr__ returns Tensor | Module for submodules, which
        # hides embed_tokens / get_per_layer_inputs.
        text_model: Any = fp_model.model.model

        rope = Gemma4RopeEmbedding(max_length=context_length, config=self.llm_config)
        layer_types = get_gemma4_layer_types(
            self.num_layers, self.FPModel.sliding_window_pattern
        )
        non_shared_indices = get_non_shared_layer_indices(
            self.num_layers, self.FPModel.num_kv_shared_layers
        )
        num_kv_heads = self.llm_config.num_key_value_heads
        head_dim = self.FPModel.head_dim
        global_head_dim = self.FPModel.global_head_dim
        sliding_window = self.FPModel.sliding_window

        kv_seq_len = context_length - sequence_length
        swa_kv_len = min(sliding_window, kv_seq_len)
        attn_min = float(getattr(self, "attention_mask_min_clip", None) or -100.0)

        # Shared zero KV-cache buffers, keyed by shape (only 4 distinct shapes).
        # Marked read-only so an accidental in-place write downstream fails
        # loudly here rather than silently corrupting every window at once.
        zero_cache: dict[tuple[int, ...], torch.Tensor] = {}

        def _zeros(shape: tuple[int, ...]) -> torch.Tensor:
            t = zero_cache.get(shape)
            if t is None:
                t = torch.zeros(shape, dtype=torch.float32)
                t.requires_grad_(False)
                zero_cache[shape] = t
            return t

        def build_sample(input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
            # input_ids: (1, L), L <= sequence_length. Left-pad to seq_len.
            input_ids = input_ids.to(torch.long)
            length = input_ids.shape[1]
            if length < sequence_length:
                pad = torch.zeros(
                    (1, sequence_length - length),
                    dtype=torch.long,
                    device=input_ids.device,
                )
                input_ids = torch.cat((pad, input_ids), dim=1)
            else:
                input_ids = input_ids[:, :sequence_length]

            # Host-side embeddings (external genie_input_embeds inputs).
            with torch.no_grad():
                inputs_embeds = text_model.embed_tokens(input_ids)
                per_layer_inputs = text_model.get_per_layer_inputs(
                    input_ids, inputs_embeds
                )

            # 2D attention mask: pad-region (left) masked, prompt attended. The
            # 4D converters expect a (1, context_length) mask covering the KV
            # cache (zeros) + the seq window.
            kv_mask = torch.zeros((1, kv_seq_len), dtype=torch.float32)
            seq_mask = torch.zeros((1, sequence_length), dtype=torch.float32)
            seq_mask[:, sequence_length - length :] = 1.0
            padded_mask = torch.cat((kv_mask, seq_mask), dim=-1)

            attention_mask = convert_2d_attention_mask_to_4d(
                padded_mask, sequence_length, context_length
            ).clip(attn_min, 0)
            swa_full = convert_2d_attention_mask_to_4d_sliding_window(
                padded_mask, sequence_length, context_length, sliding_window
            ).clip(attn_min, 0)
            # SWA mask graph width = swa_kv_len + seq (window + new tokens), so
            # keep only the last (swa_kv_len + sequence_length) KV columns.
            swa_attention_mask = swa_full[
                ..., context_length - (swa_kv_len + sequence_length) :
            ]

            # Dual RoPE from causal position ids (cumsum of attended mask - 1).
            position_ids = (
                torch.cumsum(padded_mask, dim=1, dtype=torch.int32) - 1
            ).clip(0, context_length - 1)[:, -sequence_length:]
            swa_cos, swa_sin = rope.get_embedding(position_ids)
            global_cos, global_sin = rope.get_global_embedding(position_ids)

            sample: dict[str, torch.Tensor] = {
                "inputs_embeds": inputs_embeds.to(torch.float32),
                "per_layer_inputs": per_layer_inputs.to(torch.float32),
                "attention_mask": attention_mask.to(torch.float32),
                "swa_attention_mask": swa_attention_mask.to(torch.float32),
                "swa_position_ids_cos": swa_cos.to(torch.float32),
                "swa_position_ids_sin": swa_sin.to(torch.float32),
                "position_ids_cos": global_cos.to(torch.float32),
                "position_ids_sin": global_sin.to(torch.float32),
            }
            # Zero KV cache for non-shared layers, one tensor per head. Shared
            # per shape via _zeros: allocating fresh would hold ~78 MiB per
            # window (48.75 GiB at 640 windows) of duplicate zero buffers.
            for idx in non_shared_indices:
                is_sliding = layer_types[idx] == "sliding_attention"
                layer_hd = head_dim if is_sliding else global_head_dim
                layer_kv = swa_kv_len if is_sliding else kv_seq_len
                pfx = kv_prefix(layer_types[idx])
                for h in range(num_kv_heads):
                    sample[f"{pfx}_key_{idx}_h{h}_in"] = _zeros(
                        (1, 1, layer_hd, layer_kv)
                    )
                    sample[f"{pfx}_value_{idx}_h{h}_in"] = _zeros(
                        (1, 1, layer_kv, layer_hd)
                    )
            return sample

        # WikiText.__getitem__ slices by context_length but the graph consumes
        # sequence_length at a time, so walk each item in windows rather than
        # keeping only the first: otherwise 3% of tokens reach the quantizers.
        control_ids = chat_control_token_ids(tokenizer)
        control_counts = dict.fromkeys(control_ids, 0)
        num_calib_tokens = 0

        for sample in tqdm(
            dataloader, total=len(dataloader), desc="Building calibration data"
        ):
            input_ids, _attention_mask, _ = sample
            total = input_ids.shape[1]
            for t, c in count_token_ids(input_ids, control_ids).items():
                control_counts[t] += c
            num_calib_tokens += total
            for start in range(0, total, sequence_length):
                window = input_ids[:, start : start + sequence_length]
                if window.shape[1] == 0:
                    break
                built = build_sample(window)
                for i, name in enumerate(input_names):
                    inputs[i].append(built[name].cpu())

        # Report (and on the chat path, enforce) that the chat control tokens
        # reached the quantizers: a dataset swap or template change would
        # silently degrade the encodings without any visible failure.
        print(f"Calibration token stream: {num_calib_tokens} tokens")
        for t in control_ids:
            token = tokenizer.convert_ids_to_tokens([t])[0]
            print(f"  control id {t} ({token!r}): {control_counts[t]} occurrence(s)")
        if dataset_cls is WikiTextChat and not any(control_counts.values()):
            raise ValueError(
                f"Chat-formatted calibration produced no chat control tokens "
                f"{control_ids} in {num_calib_tokens} tokens. The chat template "
                f"did not reach the calibration stream, so calibration would "
                f"repeat the raw-text mismatch it is meant to fix."
            )

        return make_hub_dataset_entries(tuple(inputs), input_names)

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """Adapt AIMET encodings for the dynamic-shape ONNX before splitting.

        Gemma4-specific (vs Llama3): there are TWO embedding Gather nodes
        (embed_tokens + embed_tokens_per_layer), so we set encodings for all
        Gather outputs, not just the first. Three steps:
        1. Set each embedding Gather output encoding (copy from weight enc)
        2. Promote weight activation encodings to param_encodings
        3. Propagate encodings through memory ops (Concat/Transpose/Reshape/...)
           - critical for transposed-key Concat at split boundaries
        """
        with open(src_encodings_path) as f:
            encodings = json.load(f)

        model = onnx.load(onnx_model_path, load_external_data=False)

        uses_lists = Version(encodings["version"]) >= Version("1.0.0")
        if uses_lists:
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }

        # Step 1: set encoding for EVERY embedding Gather output
        # (Gemma4 has embed_tokens + embed_tokens_per_layer)
        for node in model.graph.node:
            if node.op_type != "Gather":
                continue
            gather_output = node.output[0]
            embed_weight_name = node.input[0]

            if self.precision == Precision.w4:
                encodings["activation_encodings"][gather_output] = {
                    "bw": 16,
                    "dtype": "FLOAT",
                    "enc_type": "PER_TENSOR",
                    "name": gather_output,
                }
            else:
                # Copy from weight encoding (param or activation)
                weight_enc = encodings["param_encodings"].get(
                    embed_weight_name
                ) or encodings["activation_encodings"].get(embed_weight_name)
                if weight_enc is not None:
                    embed_enc = copy.deepcopy(weight_enc)
                    embed_enc["name"] = gather_output
                    encodings["activation_encodings"][gather_output] = embed_enc

        # Step 2: promote weight entries in activation_encodings to param_encodings
        for key, value in list(encodings["activation_encodings"].items()):
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(value)

        # Step 3: propagate through memory ops (Concat for transposed KV, etc.)
        propagate_memory_encodings(encodings, model)

        # Always emit list form for encoding version >= 1.0.0 (required by the
        # AI Hub compiler). Guard against any input that was already dict-form.
        if isinstance(encodings["activation_encodings"], dict):
            encodings["activation_encodings"] = list(
                encodings["activation_encodings"].values()
            )
        if isinstance(encodings["param_encodings"], dict):
            encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as f:
            json.dump(encodings, f, indent=4, sort_keys=True)

    def _postprocess_full_onnx_bundle(self, bundle: ONNXBundle) -> ONNXBundle:
        """Adapt encodings before the ONNX is split into parts."""
        if bundle.aimet_encodings_path is not None:
            self._adapt_aimet_encodings(
                str(bundle.aimet_encodings_path),
                str(bundle.aimet_encodings_path),
                str(bundle.onnx_graph_path),
            )
        return super()._postprocess_full_onnx_bundle(bundle)


class Gemma4PartBase(DynamicSplitPartBase):
    """Gemma4 Part base.

    Overrides get_graph_input_spec to handle Gemma4's non-uniform KV shapes:
    - SWA layers use head_dim=256, global layers use global_head_dim=512
    - Only non-shared layers (0 to first_shared-1) have KV I/O
    - Dual RoPE: swa_position_ids_cos/sin (SWA, embed_dim=128) and
      position_ids_cos/sin (Global, embed_dim=64)
    """

    # Override in subclass
    head_dim: int = 256
    global_head_dim: int = 512
    num_kv_shared_layers: int = 0
    sliding_window_pattern: int = 5
    sliding_window: int = 512
    num_layers: int = 0  # total decoder layers (set by concrete model)
    partial_rotary_factor: float = 0.25
    hidden_size_per_layer_input: int = 256

    def _get_onnx_graph_inputs(self) -> tuple[list[str], dict[str, list]]:
        """Input names and per-input shapes from ONE parse of the split ONNX.

        Shapes carry dynamic dims as None, so intermediate cross-part tensors
        get their true shape: Gemma4's PLE produces cross-part tensors shaped
        (1, 1, hidden) — a concrete middle dim of 1, NOT the dynamic seq_len.
        Assuming (1, seq, hidden) for these breaks the AI Hub compile shape
        check.

        Names and shapes come from the same ``graph.input`` list, so they are
        returned together rather than via ``_get_onnx_input_names()`` plus a
        second ``onnx.load`` of the same file — ``get_graph_input_spec`` runs
        per (sequence_length, context_length) graph per Part across
        export/profile/quantize, and parsing the protobuf twice per call adds
        up over a 3-4 way split.
        """
        bundle = self._get_onnx_bundle()
        model = onnx.load(str(bundle.onnx_graph_path), load_external_data=False)
        names: list[str] = []
        shapes: dict[str, list] = {}
        for inp in model.graph.input:
            names.append(inp.name)
            shapes[inp.name] = [
                None if d.dim_param else d.dim_value
                for d in inp.type.tensor_type.shape.dim
            ]
        return names, shapes

    def get_graph_input_spec(self, graph_name: str) -> InputSpec:
        """Build input spec with per-layer head_dim and dual RoPE dims."""
        sequence_length, context_length = self._graph_names[graph_name]
        sequence_length = int(sequence_length)
        context_length = int(context_length)

        kv_seq_len = context_length - sequence_length
        swa_kv_len = min(self.sliding_window, kv_seq_len)
        onnx_input_names, onnx_input_shapes = self._get_onnx_graph_inputs()

        # Gemma4 has no pure-embedding part: the PLE is recomputed from input_ids
        # inside every decoder layer, so the spec comes from the actual ONNX
        # input names with no special case for part 1.

        layer_types = get_gemma4_layer_types(
            self.num_layers, self.sliding_window_pattern
        )
        swa_embed_dim = self.head_dim // 2  # 128
        # Global RoPE uses the full global_head_dim//2 cos/sin width
        # (Genie proportional-RoPE layout; first 64 freqs real, rest identity).
        global_embed_dim = self.global_head_dim // 2  # 256

        # Match both past_ (global) and swa_ (sliding) KV tensor names.
        kv_re = re.compile(r"(?:past|swa)_(?:key|value)_(\d+)_")

        # Authored as (shape, dtype) tuples, normalized to TensorSpec at the
        # return below.
        spec: dict[str, tuple[tuple[int, ...], str]] = {}
        for name in onnx_input_names:
            m = kv_re.search(name)
            if m:
                layer_idx = int(m.group(1))
                is_sliding = layer_types[layer_idx] == "sliding_attention"
                hd = self.head_dim if is_sliding else self.global_head_dim
                # *_in holds the past cache; *_out is this step's newly computed
                # KV, of length sequence_length. A split part can receive an
                # *_out as an input (KV sharing), so size those seq_len.
                is_out = name.endswith("_out")
                lkv = (
                    sequence_length
                    if is_out
                    else (swa_kv_len if is_sliding else kv_seq_len)
                )
                if "_key_" in name:
                    spec[name] = ((1, 1, hd, lkv), "float32")
                else:  # value
                    spec[name] = ((1, 1, lkv, hd), "float32")
            elif name == "input_ids":
                # Legacy genie_input_ids mode: input_ids feeds every part that
                # has decoder layers (it recomputes its PLE slice internally).
                spec[name] = ((1, sequence_length), "int32")
            elif name == "inputs_embeds":
                # External-embedding mode: token embeddings feed the first part.
                spec[name] = ((1, sequence_length, self.hidden_size), "float32")
            elif name == "per_layer_inputs":
                # External-embedding mode: the raw per-layer embeddings are
                # consumed by every layer, so this tensor is an input to every
                # part that contains decoder layers.
                spec[name] = (
                    (
                        1,
                        sequence_length,
                        self.num_layers,
                        self.hidden_size_per_layer_input,
                    ),
                    "float32",
                )
            elif name == "attention_mask":
                spec[name] = ((1, 1, sequence_length, context_length), "float32")
            elif name == "swa_attention_mask":
                spec[name] = (
                    (1, 1, sequence_length, swa_kv_len + sequence_length),
                    "float32",
                )
            elif name in ("swa_position_ids_cos", "swa_position_ids_sin"):
                spec[name] = ((1, 1, sequence_length, swa_embed_dim), "float32")
            elif name in ("position_ids_cos", "position_ids_sin"):
                spec[name] = ((1, 1, sequence_length, global_embed_dim), "float32")
            else:
                # Intermediate cross-part tensor (incoming hidden state, or a
                # PLE-derived (1, 1, hidden) tensor). Use the ACTUAL ONNX shape,
                # substituting any dynamic dim with sequence_length.
                actual = onnx_input_shapes.get(name)
                if actual is not None:
                    concrete = tuple(
                        sequence_length if d is None else d for d in actual
                    )
                    spec[name] = (concrete, "float32")
                else:
                    spec[name] = ((1, sequence_length, self.hidden_size), "float32")

        # Authored as (shape, dtype) tuples above; the compile/profile path
        # (get_channel_last, build_compile_options) needs TensorSpec objects.
        return {
            k: TensorSpec(shape=tuple(shape), dtype=dtype)
            for k, (shape, dtype) in spec.items()
        }

    def get_graph_hub_compile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        """Gemma4 compile options, without ``--quantize_full_type``/``--quantize_io``.

        Those flags route the job through the qairt-quantizer, which the v73 HTP
        rejects for Gemma4 (garbage decoder output + link exit 14). The scales
        already live in each split's baked AIMET encodings, so we skip the
        ``DynamicSplitPartBase`` override and call the grandparent directly.
        """
        return MultiGraphWorkbenchModel.get_graph_hub_compile_options(
            self,
            graph_name,
            target_runtime,
            precision,
            other_compile_options,
            device,
        )


def _fix_gemma4_genie_config(
    inner: dict[str, Any],
    llm_config: Any,
    fp: type[Gemma4PreSplitBase],
    bundle_dir: Path,
    runtime: TargetRuntime,
) -> None:
    """Correct the genie_config.json fields the shared generator mis-emits.

    ``create_genie_config`` models a single-RoPE decoder: one ``rope-theta`` /
    ``pos-id-dim`` pair in ``backend.QnnHtp`` and ``eos-token`` from
    ``llm_config.eos_token_id``. Neither holds for Gemma4:

    * **Dual RoPE.** The sliding-window and global layers use different rotary
      settings, which a flat backend ``rope-theta`` cannot express. The shared
      output silently gave every layer the sliding theta and left the global
      layers unrotated.
    * **cache-groups.** The two families need separate KV caches at different
      kv-dims, the SWA one bounded by the sliding window. With a single implicit
      cache the SWA layers read the wrong KV geometry.
    * **eos-token.** An ``-it`` model almost never emits the bare ``<eos>`` that
      ``text_config.eos_token_id`` names, so every prompt runs to the context
      cap.

    Everything is derived from ``llm_config`` / the model class rather than
    hardcoded, so a config change cannot silently desync this block.

    Parameters
    ----------
    inner
        The ``dialog`` sub-dict of genie_config.json, edited in place.
    llm_config
        Gemma4 ``text_config`` (nested RoPE params under ``rope_parameters``).
    fp
        FP PreSplit class, source of the head dims and sliding-window size.
    bundle_dir
        Bundle directory, loaded as a tokenizer to resolve the stop-token ids.
    runtime
        Target runtime, which selects the dual-RoPE schema.
    """
    rope_params = getattr(llm_config, "rope_parameters", None) or {}
    glb = rope_params.get("full_attention") or {}
    loc = rope_params.get("sliding_attention") or {}
    if not glb or not loc:
        raise ValueError(
            "Gemma4 genie config needs both 'full_attention' and "
            "'sliding_attention' rope_parameters; got "
            f"{sorted(rope_params)}. The checkpoint's config.json layout "
            "changed -- update _fix_gemma4_genie_config rather than letting "
            "the sliding-window layers export with no rotary encoding."
        )

    model = inner["engine"]["model"]
    qnn_htp = inner["engine"]["backend"]["QnnHtp"]

    # Global (full-attention) RoPE: half the 512-wide global head, theta 1e6,
    # partial rotary. Insert before "binary" so the file reads in Genie's order.
    global_rope: dict[str, Any] = {
        "type": "rope",
        "rope-dim": fp.global_head_dim // 2,
        "rope-theta": glb["rope_theta"],
    }
    partial = glb.get("partial_rotary_factor")
    if partial is not None:
        global_rope["rope-scaling"] = {
            "rope-type": glb.get("rope_type", "proportional"),
            "partial-rotary-factor": partial,
        }
    local_rope: dict[str, Any] = {
        "type": "rope",
        "rope-dim": fp.head_dim // 2,
        "rope-theta": loc["rope_theta"],
    }
    binary = model.pop("binary")
    # Incompatible schemas: Genie 2.48+ takes a list keyed by "prefix" and
    # rejects local-positional-encoding; geniex-qairt wants the two keys.
    if runtime == TargetRuntime.GENIEX_QAIRT:
        model["positional-encoding"] = global_rope
        model["local-positional-encoding"] = local_rope
    else:
        model["positional-encoding"] = [global_rope, {"prefix": "swa_", **local_rope}]
    model["binary"] = binary

    # Per-family KV caches. Prefixes/tensor names match the exported ONNX I/O.
    inner["engine"]["cache-groups"] = [
        {
            "version": 1,
            "prefix": "past_",
            "kv-dim": fp.global_head_dim,
            "attention-mask-tensor-name": "attention_mask",
        },
        {
            "version": 1,
            "prefix": "swa_",
            "kv-dim": fp.head_dim,
            "attention-mask-tensor-name": "swa_attention_mask",
            "longcontext": {
                "version": 1,
                "type": "sliding-window",
                "sliding-window": {"version": 1, "window-size": fp.sliding_window},
            },
        },
    ]

    # The flat backend RoPE keys are now expressed per-family above; leaving
    # them would give Genie two conflicting sources of truth.
    qnn_htp.pop("pos-id-dim", None)
    qnn_htp.pop("rope-theta", None)

    # eos-token: an -it model almost never emits the bare <eos>, it ends turns
    # with <turn|>. generation_config.json's full stop set also lists
    # tool-calling stops that would cut a plain chat turn short, so use just
    # these two.
    tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir))
    turn_end = tokenizer.convert_tokens_to_ids("<turn|>")
    eos: list[int] = []
    for token in (tokenizer.eos_token_id, turn_end):
        # convert_tokens_to_ids returns int | list[int]; a str input is a scalar.
        if isinstance(token, int) and token not in eos:
            eos.append(token)
    if eos:
        inner["context"]["eos-token"] = eos if len(eos) > 1 else eos[0]

    pad_token = getattr(llm_config, "pad_token_id", None)
    if pad_token is not None:
        inner["context"]["pad-token"] = pad_token


def _fix_gemma4_chat_template(bundle_dir: Path) -> None:
    r"""Join adjacent string literals in the bundled chat template.

    Gemma4's upstream template calls ``raise_exception`` with Python-style
    adjacent string literals::

        {{- raise_exception(
            "line one "
            "line two"
        ) -}}

    Jinja2 concatenates those, so ``apply_chat_template`` works and the bug is
    invisible on the host. minja (the C++ engine Genie uses) does not: it parses
    the whole template up front, so this is a **hard load failure on device**
    even for a benchmark that never touches tool-calling.

    Rewrites each run of adjacent literals into one literal. Idempotent, and a
    no-op for templates that don't use the pattern.

    Parameters
    ----------
    bundle_dir
        Bundle directory holding tokenizer_config.json (the template is embedded
        there rather than shipped as chat_template.jinja, see the
        ``save_jinja_files=False`` call in the shared bundle writer).
    """
    # "..." followed by only whitespace/newlines then another "..." -- i.e. two
    # literals with no operator between them, which minja rejects.
    adjacent = re.compile(r'"((?:[^"\\]|\\.)*)"\s*\n\s*"((?:[^"\\]|\\.)*)"')

    def join_all(template: str) -> str:
        # Loop: one pass merges pairs, so a 3-literal run needs a second pass.
        while True:
            merged = adjacent.sub(r'"\1\2"', template)
            if merged == template:
                return merged
            template = merged

    tok_cfg_path = bundle_dir / "tokenizer_config.json"
    if tok_cfg_path.exists():
        with open(tok_cfg_path, encoding="utf-8") as f:
            tok_cfg = json.load(f)
        template = tok_cfg.get("chat_template")
        if isinstance(template, str):
            fixed = join_all(template)
            if fixed != template:
                tok_cfg["chat_template"] = fixed
                with open(tok_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(tok_cfg, f, indent=2, ensure_ascii=False)

    # Some checkpoints also ship the standalone file; keep the two in sync.
    jinja = bundle_dir / "chat_template.jinja"
    if jinja.exists():
        text = jinja.read_text(encoding="utf-8")
        fixed = join_all(text)
        if fixed != text:
            jinja.write_text(fixed, encoding="utf-8")


# Weight files transformers' from_pretrained accepts for a local directory.
_TORCH_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _has_torch_weights(checkpoint: str | Path | None) -> bool:
    """Whether ``checkpoint`` is a local dir holding loadable torch weights.

    A "DEFAULT*" sentinel and an HF repo id are both False: neither is a local
    directory, so there is nothing to inspect and the caller should fall back to
    its own repo. A calibrated w4a16 directory is also False -- it ships
    model_dynamic.onnx + model.encodings, not safetensors.
    """
    if checkpoint is None:
        return False
    path = Path(checkpoint)
    if not path.is_dir():
        return False
    return any((path / name).exists() for name in _TORCH_WEIGHT_FILES)


class Gemma4PreSplitCollectionBase(DynamicSplitCollectionBase):
    """Gemma4 Collection base.

    Concrete subclasses set ``vision_encoder_cls`` to their
    ``Gemma4_E*B_VisionEncoder``, which makes the VEG a component of the
    Collection so it is built, compiled and profiled by the export pipeline and the
    scorecard alongside the text Parts (mirroring ``Qwen3VLCollectionBase``).

    The VEG is appended *after* the Parts rather than prepended. Everything in
    :class:`DynamicSplitCollectionBase` that reaches for "the first Part" does
    so via ``next(iter(self.components.values()))``, so keeping the Parts first
    means the inherited behaviour is untouched and no change is needed in
    ``templates/llm``.
    """

    # Component name for the VEG. Genie's ctx-bins filter below matches on it.
    vision_encoder_component_name: str = "vision_encoder"
    vision_encoder_cls: type[Gemma4VisionEncoder] | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = True,
        # Empty lists are a read-only "use cls defaults" sentinel (never mutated).
        sequence_lengths: list[int] = [],  # noqa: B006
        context_lengths: list[int] = [],  # noqa: B006
    ) -> Self:
        """Create the Collection with all Parts, plus the VEG for a VLM.

        Parameters
        ----------
        checkpoint
            Path to checkpoint with ONNX + encodings, or ``"DEFAULT"`` to create
            from HuggingFace.
        host_device
            Device for computation.
        _skip_quantsim_creation
            Skip QuantSim creation (for testing).
        sequence_lengths
            Sequence lengths to compile for. Empty means use
            ``cls.default_sequence_lengths``.
        context_lengths
            Context lengths to compile for. Empty means use
            ``cls.default_context_lengths``.

        Returns
        -------
        Self
            The Collection with every Part, and the vision encoder last when
            ``vision_encoder_cls`` is set.
        """
        instance = super().from_pretrained(
            checkpoint=checkpoint,
            host_device=host_device,
            _skip_quantsim_creation=_skip_quantsim_creation,
            sequence_lengths=sequence_lengths,
            context_lengths=context_lengths,
        )
        if cls.vision_encoder_cls is not None:
            # The VEG needs torch vision weights, which most checkpoints handed
            # to export lack (a "DEFAULT*" sentinel resolves to a published
            # *text* asset; a calibrated w4a16 dir holds only ONNX + encodings),
            # so fall back to the VEG's own HF repo. It takes none of the
            # split-graph kwargs: one graph, fixed geometry.
            instance.add_vision_encoder(
                cls.vision_encoder_cls.from_pretrained(
                    checkpoint=(checkpoint if _has_torch_weights(checkpoint) else None),
                    device=host_device,
                )
            )
        return instance

    def add_vision_encoder(self, vision_encoder: Gemma4VisionEncoder) -> None:
        """Append the VEG as the last component (after every text Part).

        Parameters
        ----------
        vision_encoder
            The loaded VEG instance.
        """
        name = self.vision_encoder_component_name
        self.components[name] = vision_encoder
        setattr(self, name, vision_encoder)

    def restrict_to_single_instantiation(self) -> None:
        """Reduce every Part to a single instantiation, skipping the VEG.

        The VEG is a plain :class:`BaseModel` with one fixed-geometry graph, so
        it has no instantiations to reduce and no such method.
        """
        for component in self.components.values():
            if isinstance(component, self.part_base_cls):
                component.restrict_to_single_instantiation()

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        """Write the Genie bundle, adding Gemma4's host-side embedding LUTs.

        Delegates the standard artifacts (genie_config.json, tokenizer/config,
        htp_backend_ext_config.json, sample_prompt.txt) to the base, then
        exports the token + per-layer (PLE) embedding tables as LUTs and injects
        the ``embedding`` / ``perlayer-embedding`` sections into
        genie_config.json (the shared create_genie_config emits neither for
        Gemma4).

        For a VLM, also drops the vision-encoder binary from the decoder's
        ``ctx-bins`` (it is loaded by the pipeline's image-encoder node, not the
        dialog engine) and marks the bundle vision-capable.

        Also rewrites the parts of genie_config.json that the shared generator
        gets wrong for Gemma4 -- dual RoPE, cache-groups and eos-token. See
        ``_fix_gemma4_genie_config`` for why each is needed.
        """
        output_path = Path(output_dir)

        # Standard bundle (base tokenizer/config/genie/htp logic).
        super().write_supplementary_files(output_dir, metadata)

        # Original checkpoint that ships the embedding tables: prefer the first
        # Part's local checkpoint, fall back to the HF repo id.
        first_part = next(iter(self.components.values()))
        _presplit = getattr(first_part, "_presplit", None)
        _ckpt = str(getattr(_presplit, "checkpoint", None) or "") or self.hf_repo_name

        # Gemma4Config nests text hyperparameters under `.text_config`.
        llm_config = getattr(_presplit, "llm_config", None)
        if llm_config is None:
            from transformers import AutoConfig

            llm_config = AutoConfig.from_pretrained(str(output_path))
        llm_config = getattr(llm_config, "text_config", llm_config)

        fp = self.fp_presplit_cls
        # Genie reads the LUT file size into a uint32 (LUT.cpp:89), so E4B's
        # 5.3 GB PLE table wraps; 8-bit halves it under the 4 GiB limit.
        perlayer_bitwidth = 8 if metadata.runtime == TargetRuntime.GENIE else 16
        embed_luts = export_gemma4_embeddings(
            checkpoint=_ckpt,
            output_dir=output_path,
            hidden_size=llm_config.hidden_size,
            num_layers=llm_config.num_hidden_layers,
            ple_dim=fp.hidden_size_per_layer_input,
            hf_repo_name=self.hf_repo_name,
            perlayer_bitwidth=perlayer_bitwidth,
        )

        # Inject embedding / perlayer-embedding sections into genie_config.json.
        genie_path = output_path / "genie_config.json"
        if genie_path.exists():
            with open(genie_path) as f:
                genie_config = json.load(f)
            top_key = next(iter(genie_config))
            inner = genie_config[top_key]

            def _emb_section(lut: dict[str, Any]) -> dict[str, Any]:
                return {
                    "version": 1,
                    "type": "lut",
                    "lut-path": lut["lut-path"],
                    "size": lut["size"],
                    "datatype": f"ufixed{lut['bw']}",
                    "quant-param": {
                        "scale": lut["scale"],
                        "offset": lut["offset"],
                    },
                }

            inner["embedding"] = _emb_section(embed_luts["embedding"])
            inner["perlayer-embedding"] = _emb_section(embed_luts["perlayer_embedding"])

            if self.vision_encoder_cls is not None:
                # The base builds ctx-bins from every downloaded component, but
                # the dialog engine must load only the text decoder graphs: the
                # VEG belongs to the image-encoder node of the Genie VLM
                # pipeline. Substring match, since the downloaded file name may
                # be prefixed with the model name.
                ctx_bins = inner["engine"]["model"]["binary"]["ctx-bins"]
                inner["engine"]["model"]["binary"]["ctx-bins"] = [
                    name
                    for name in ctx_bins
                    if self.vision_encoder_component_name not in name
                ]

            # Runs after the ctx-bins filter: it pops and re-inserts "binary" to
            # order the RoPE keys ahead of it, preserving the filtered list.
            _fix_gemma4_genie_config(
                inner, llm_config, fp, output_path, metadata.runtime
            )
            with open(genie_path, "w") as f:
                json.dump(genie_config, f, indent=4)

        _fix_gemma4_chat_template(output_path)

        # Genie needs to know the bundle carries an image encoder. The base
        # hardcodes supports_vision=False since most LLMs are text-only.
        if self.vision_encoder_cls is not None and metadata.genie is not None:
            metadata.genie.supports_vision = True

        if hasattr(metadata, "supplementary_files"):
            emb_lut = embed_luts["embedding"]
            ple_lut = embed_luts["perlayer_embedding"]
            metadata.supplementary_files[emb_lut["lut-path"]] = (
                f"Host-side token embedding table (ufixed{emb_lut['bw']} LUT) for Genie."
            )
            metadata.supplementary_files[ple_lut["lut-path"]] = (
                "Host-side per-layer (PLE) embedding table "
                f"(ufixed{ple_lut['bw']} LUT) for Genie."
            )
            metadata.supplementary_files[emb_lut["enc-path"]] = (
                f"Quantization encoding (scale/offset) for {emb_lut['lut-path']}."
            )
            metadata.supplementary_files[ple_lut["enc-path"]] = (
                f"Quantization encoding (scale/offset) for {ple_lut['lut-path']}."
            )

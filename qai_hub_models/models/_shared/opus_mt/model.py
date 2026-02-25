# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import struct
from abc import abstractmethod
from pathlib import Path
from typing import cast

import torch
from qai_hub import Device
from transformers import MarianMTModel, MarianTokenizer
from typing_extensions import Self

from qai_hub_models.models._shared.opus_mt.model_adaptation import (
    QcMarianDecoder,
    QcMarianEncoder,
    apply_model_adaptations,
)
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = "opus_mt_shared"
MODEL_ASSET_VERSION = 1

# Model configuration constants
MAX_SEQ_LEN_ENC = 256
MAX_SEQ_LEN_DEC = 256
TOKENIZER_ENCODE_NAME = "tokenizer_encode.bin"
TOKENIZER_DECODE_NAME = "tokenizer_decode.bin"


class OpusMTEncoder(BaseModel):
    """
    OpusMT Encoder optimized for export and inference.

    It takes text input (input_ids) and produces cross attention
    key-value cache for the decoder.
    """

    def __init__(self, model: QcMarianEncoder) -> None:
        super().__init__()
        self.encoder = model

    @classmethod
    def from_pretrained(
        cls, opus_mt_version: str = "Helsinki-NLP/opus-mt-en-es"
    ) -> Self:
        return cls(OpusMT.get_opus_model(opus_mt_version)[0])

    def forward(
        self, input_ids: torch.Tensor, encoder_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        input_ids
            Input token IDs of shape (batch_size, sequence_length)
        encoder_attention_mask
            Attention mask of shape (batch_size, sequence_length)

        Returns
        -------
        tuple[torch.Tensor, ...]
            Cross attention key and value cache tensors
        """
        return self.encoder(input_ids, encoder_attention_mask)

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type)). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "input_ids": ((1, MAX_SEQ_LEN_ENC), "int32"),
            "encoder_attention_mask": ((1, MAX_SEQ_LEN_ENC), "int32"),
        }

    @staticmethod
    def get_output_names(num_layers: int = 6) -> list[str]:
        """Returns the output names for the encoder."""
        output_names = []
        for layer_idx in range(num_layers):
            output_names.append(f"block_{layer_idx}_cross_key_states")
            output_names.append(f"block_{layer_idx}_cross_value_states")
        return output_names

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if (
            precision == Precision.float
            and target_runtime.qairt_version_changes_compilation
        ):
            compile_options = (
                compile_options + " --quantize_full_type float16 --quantize_io"
            )
        return compile_options


class OpusMTDecoder(BaseModel):
    """
    OpusMT Decoder optimized for export and inference.

    Wraps MarianDecoderMod to facilitate export.
    """

    def __init__(self, model: QcMarianDecoder) -> None:
        super().__init__()
        self.decoder = model
        self.num_layers = 6  # OpusMT has 6 decoder layers

    @classmethod
    def from_pretrained(
        cls, opus_mt_version: str = "Helsinki-NLP/opus-mt-en-es"
    ) -> Self:
        return cls(OpusMT.get_opus_model(opus_mt_version)[1])

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        position: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        input_ids
            Input token IDs of shape (batch_size, 1)
        encoder_attention_mask
            Encoder attention mask of shape (batch_size, encoder_seq_len)
        position
            Current position index of shape (1,)
        *past_key_values
            Past key-value states for self and cross attention

        Returns
        -------
        tuple[torch.Tensor, ...]
            Logits and updated key-value states
        """
        return self.decoder(
            input_ids, encoder_attention_mask, position, *past_key_values
        )

    @staticmethod
    def get_input_spec(
        num_layers: int = 6,
        attention_dim: int = 512,
        num_heads: int = 8,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type)). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        head_dim = attention_dim // num_heads

        specs = {
            "input_ids": ((1, 1), "int32"),
            "encoder_attention_mask": ((1, MAX_SEQ_LEN_ENC), "int32"),
            "position": ((1,), "int32"),
        }

        # Add past key-value states for each layer
        # Using transpose_key=False format (consistent with original notebook)
        for i in range(num_layers):
            specs[f"block_{i}_past_self_key_states"] = (
                (1, num_heads, MAX_SEQ_LEN_DEC - 1, head_dim),
                "float32",
            )
            specs[f"block_{i}_past_self_value_states"] = (
                (1, num_heads, MAX_SEQ_LEN_DEC - 1, head_dim),
                "float32",
            )
            specs[f"block_{i}_cross_key_states"] = (
                (1, num_heads, MAX_SEQ_LEN_ENC, head_dim),
                "float32",
            )
            specs[f"block_{i}_cross_value_states"] = (
                (1, num_heads, MAX_SEQ_LEN_ENC, head_dim),
                "float32",
            )

        return specs

    @staticmethod
    def get_output_names(num_layers: int = 6) -> list[str]:
        """Returns the output names for the decoder."""
        output_names = ["logits"]
        for layer_idx in range(num_layers):
            output_names.append(f"block_{layer_idx}_present_self_key_states")
            output_names.append(f"block_{layer_idx}_present_self_value_states")
        return output_names

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if (
            precision == Precision.float
            and target_runtime.qairt_version_changes_compilation
        ):
            compile_options = (
                compile_options + " --quantize_full_type float16 --quantize_io"
            )
        return compile_options


class OpusMT(CollectionModel):
    """
    Base OpusMT translation model.

    This model consists of an encoder and decoder that work together
    to translate text between languages.
    """

    def __init__(
        self,
        encoder: OpusMTEncoder,
        decoder: OpusMTDecoder,
        hf_source: str,
    ) -> None:
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.hf_source = hf_source

    @classmethod
    @abstractmethod
    def get_opus_mt_version(cls) -> str:
        """Return the HuggingFace model identifier for this OpusMT variant."""

    @classmethod
    def get_opus_model(
        cls, opus_mt_version: str | None = None
    ) -> tuple[QcMarianEncoder, QcMarianDecoder]:
        # Load the pretrained model - this downloads pytorch_model.bin and other files
        orig_model = cast(
            MarianMTModel,
            MarianMTModel.from_pretrained(opus_mt_version or cls.get_opus_mt_version()),
        )
        orig_model.eval()

        # Apply model adaptations to optimize for QNN inference
        return apply_model_adaptations(orig_model)

    @classmethod
    def from_pretrained(cls) -> Self:
        """
        Load OpusMT model from pretrained weights.

        This will download the model files (including pytorch_model.bin) from HuggingFace Hub
        if they are not already cached locally.
        """
        # Load the original Marian model
        encoder_qc, decoder_qc = cls.get_opus_model()
        opus_mt_version = cls.get_opus_mt_version()

        # Wrap in our model classes
        encoder = OpusMTEncoder(encoder_qc)
        decoder = OpusMTDecoder(decoder_qc)
        return cls(encoder, decoder, opus_mt_version)


def get_tokenizer(hf_model_name: str) -> MarianTokenizer:
    """Get the tokenizer for the specified OpusMT model."""
    return MarianTokenizer.from_pretrained(hf_model_name)


def generate_tokenizer_bins(
    hf_model_name: str, output_dir: str | os.PathLike
) -> tuple[Path, Path]:
    """
    Generate optimized tokenizer encode and decode binaries for an Opus-MT model.

    This function extracts tokenizer information from a HuggingFace model and generates
    two binary files:
    1. tokenizer_encode.bin - optimized lookup table for text encoding
    2. tokenizer_decode.bin - lookup table for token ID decoding

    Parameters
    ----------
    hf_model_name
        HuggingFace model ID, e.g. "Helsinki-NLP/opus-mt-zh-en"
    output_dir
        Output directory path for storing the generated binary files

    Returns
    -------
    tuple[Path, Path]
        Tuple of (encode_bin_path, decode_bin_path) containing relative paths to generated binaries
    """
    # 1. Load tokenizer and get vocabulary
    tokenizer = MarianTokenizer.from_pretrained(hf_model_name)
    opus_vocab_dict = tokenizer.get_vocab()  # Get token -> id mapping

    # 2. Create output directory and file paths
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encode_path = out_dir / TOKENIZER_ENCODE_NAME
    decode_path = out_dir / TOKENIZER_DECODE_NAME

    # ========== Generate encode binary file ==========
    # 3. Prepare data for encoding
    special_token_list = ["</s>", "<unk>", "<pad>"]  # Special token list
    spm_model = tokenizer.spm_source  # Get SentencePiece model

    # 4. Build vocabulary dictionary with score information
    vocab_dict = {}
    for opus_vocab_token, opus_vocab_id in opus_vocab_dict.items():
        if opus_vocab_token in special_token_list:
            continue  # Skip special tokens

        # Get token score from SentencePiece model
        spm_id = spm_model.piece_to_id(opus_vocab_token)
        score = float("-inf") if spm_id == 0 else spm_model.get_score(spm_id)
        vocab_dict[opus_vocab_token] = [score, opus_vocab_id, len(opus_vocab_token)]

    # 5. Build DAG (Directed Acyclic Graph) for efficient encoding
    def create_node(text: str, c_len: int = 0) -> dict:
        """Create DAG node"""
        return {
            "len": c_len,
            "score": 0.0,
            "token_id": 0,
            "entry_id": 0,
            "is_leaf": 0,
            "num_children": 0,
            "offset": 0,
            "text": text,
            "children": [],
        }

    root = create_node("", 0)  # Create root node
    dump_pos_idx = 1

    # 6. Build character-level DAG paths for each token
    for key, val in vocab_dict.items():
        if key in special_token_list:
            continue

        token_str, score, token_id, num_chars = (
            str(key),
            float(val[0]),
            int(val[1]),
            int(val[2]),
        )
        current = root
        chars = []

        # Build path character by character
        for char in token_str:
            chars.append(char)
            chars_str = "".join(chars)

            # Find or create child node
            curr_node = None
            for child in current["children"]:
                if child["text"] == chars_str:
                    curr_node = child
                    break

            if curr_node is None:
                # Create new child node
                curr_node = create_node(chars_str, len(chars_str.encode("utf-8")))
                current["children"].append(curr_node)
                current["num_children"] += 1
                dump_pos_idx += 1

            current = curr_node
            # Mark as leaf node if we've reached the end of the token
            if len(chars) == num_chars:
                current.update({"score": score, "is_leaf": 1, "token_id": token_id})

    # 7. Assign unique entry IDs to all nodes
    node_stack, entry_id = [root], 0
    while node_stack:
        curr_node = node_stack.pop(0)
        curr_node["entry_id"] = entry_id
        entry_id += 1
        node_stack.extend(curr_node["children"])

    # 8. Collect all nodes and calculate offsets
    all_nodes = []
    node_stack = [root]
    while node_stack:
        curr_node = node_stack.pop(0)
        all_nodes.append(curr_node)
        node_stack.extend(curr_node["children"])

    # 9. Calculate offset for each node in the binary file
    for i, node in enumerate(all_nodes):
        if i == 0:
            node["offset"] = 0
        else:
            prev_node = all_nodes[i - 1]
            # Calculate aligned offset (4-byte alignment)
            node["offset"] = (
                prev_node["offset"]
                + ((prev_node["len"] + 1 + 3) & ~0x03)
                + prev_node["num_children"] * 4
            )

    # 10. Write encode binary file
    with open(encode_path, "wb") as bin_file:
        num_entries = len(all_nodes)
        packed_data_fixed = b""  # Fixed-length data
        packed_data_variable = b""  # Variable-length data
        size_data_fixed = 0
        size_data_variable = 0

        # Pack data for each node
        for curr_node in all_nodes:
            # Fixed-length part: length, score, token_id, is_leaf, num_children, offset
            struct_format = "I f I B I I"
            # Variable-length part: text content + child node ID list
            variable_format = f"{curr_node['len'] + 1}s {curr_node['num_children']}I"
            children_ids = [child["entry_id"] for child in curr_node["children"]]

            packed_data_fixed += struct.pack(
                struct_format,
                curr_node["len"],
                curr_node["score"],
                curr_node["token_id"],
                curr_node["is_leaf"],
                curr_node["num_children"],
                curr_node["offset"],
            )
            packed_data_variable += struct.pack(
                variable_format, curr_node["text"].encode(), *children_ids
            )

            size_data_fixed += struct.calcsize(struct_format)
            size_data_variable += struct.calcsize(variable_format)

        # Write file header (num_entries, fixed_data_size, variable_data_size) and data
        bin_file.write(
            struct.pack("III", num_entries, size_data_fixed, size_data_variable)
        )
        bin_file.write(packed_data_fixed + packed_data_variable)

    # ========== Generate decode binary file ==========
    # 11. Build decode lookup table
    vocab_final = [""] * len(opus_vocab_dict)
    for token, token_id in opus_vocab_dict.items():
        vocab_final[token_id] = token  # Sort by token_id
    vocab_final.append("<pad>")  # Add padding token

    # 12. Write decode binary file
    with open(decode_path, "wb") as bin_file:
        lookups_bytes, offsets = b"", [0]

        # Encode all tokens as UTF-8 byte strings
        for token in vocab_final:
            token_bytes = token.encode("utf-8") + b"\0"  # Add null terminator
            lookups_bytes += token_bytes
            offsets.append(offsets[-1] + len(token_bytes))

        # Calculate lookup table parameters
        lookups_size = offsets[-1] + 1
        offsets = offsets[:-1]  # Remove last offset
        lookups_count = len(offsets)

        # Write decode file: lookup_count, total_size, offset_array, string_data
        struct_format = f"II{lookups_count}I{lookups_size}s"
        bin_file.write(
            struct.pack(
                struct_format, lookups_count, lookups_size, *offsets, lookups_bytes
            )
        )

    return encode_path, decode_path

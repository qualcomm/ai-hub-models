# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import json
import math
import os
import struct
from collections import OrderedDict
from pathlib import Path

############################################################
# Binary packing format
# ---------------------

# uint32_t numFulls
# uint32_t numCacheLineIndicesFulls
# uint32_t numPartials
# uint32_t numCacheLineIndicesPartials
# uint32_t sizeVocabFull
# uint32_t sizeVocabPartial
#
# char* pVocabFull
# uint32_t* pCacheLineIndicesFulls
#
# char* pVocabPartials
# uint32_t* pCacheLineIndicesPartials
#
# struct SingleTokenInfo* pFulls
# struct SingleTokenInfo* pPartials
#############################################################

# struct SingleTokenInfo
# uint32_t         vocabOffset;
# uint32_t         tokenIdx;
# int              numChars;
# int              numBytes;

#############################################################


def align_n(x: int, n: int = 8) -> int:
    """Align memory addresses to a desired boundary."""
    assert type(x) is int, "  input not of type int in align_n"
    assert type(n) is int, "  align not of type int in align_n"
    return x if n == 0 else math.ceil(math.ceil(float(x) / float(n)) * float(n))


def generate_vocab_rules(vocab_file: str) -> tuple[dict, dict]:
    """Generate full and partial vocab dictionaries."""
    with open(vocab_file) as vocab_json_file:
        vocab_dict = json.load(vocab_json_file, object_pairs_hook=OrderedDict)

    vocab_temp = vocab_dict["model"]["vocab"]
    discard_keys = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab_temp = dict(
        filter(lambda item: item[0] not in discard_keys, vocab_temp.items())
    )
    discard_key = "unused"
    core_vocab = {k: v for k, v in vocab_temp.items() if discard_key not in k}

    fulls = {k: v for k, v in core_vocab.items() if str(k).startswith("##") is False}
    partials = {k: v for k, v in core_vocab.items() if str(k).startswith("##") is True}

    return fulls, partials


def dump_vocab(vocab: dict) -> tuple[bytes, list, list]:
    """Create tokenizer vocabulary and offsets for accessing the tokens."""
    vocab_sorted_list = sorted(
        vocab.items(), key=lambda key: len(str(key[0]).encode("utf-8"))
    )
    vocab_sorted = {ele[0]: ele[1] for ele in vocab_sorted_list}

    cache_len_indices = []
    c_strlen_prev = 0
    run_idx = 0
    token_info = []
    vocab_bytes = b""
    cur_offset = 0
    next_offset = 0
    for token_str, token_id in vocab_sorted.items():
        c_strlen = len(str(token_str).encode("utf-8"))
        if c_strlen > c_strlen_prev:
            if c_strlen_prev == 0:
                cache_len_indices.extend([0] * c_strlen)
            c_strlen_prev = c_strlen
            cache_len_indices.append(run_idx)
        run_idx = run_idx + 1
        token_encoded = str(token_str).encode("utf-8") + b"\0"
        vocab_bytes += token_encoded
        next_offset += len(token_encoded)
        token_info.append([cur_offset, token_id, len(token_str), len(token_encoded)])
        cur_offset = next_offset

    return vocab_bytes, token_info, cache_len_indices


def generate_bert_tokenizer_binary(
    vocab_file: str, out_file: str | os.PathLike
) -> Path:
    """Generate the combined tokenizer binary."""
    align_num = 256

    fulls, partials = generate_vocab_rules(vocab_file)

    vocab_size = len(fulls) + len(partials)
    print(f"vocab_size: {vocab_size}")

    # Get entries for the binary
    fulls_vocab_bytes, fulls_token_info, fulls_cache_indices = dump_vocab(fulls)
    partials_vocab_bytes, partials_token_info, partials_cache_indices = dump_vocab(
        partials
    )

    # generate binary
    encode_binary = b""
    # uint32_t numFulls
    # uint32_t numCacheLineIndicesFulls
    # ----------------------------------
    num_fulls_token_info = len(fulls_token_info)
    num_fulls_cache_indices = len(fulls_cache_indices)
    print(
        f"Num token info fulls: {num_fulls_token_info}, Num cache indices fulls: {num_fulls_cache_indices}"
    )
    encode_binary += struct.pack("II", num_fulls_token_info, num_fulls_cache_indices)

    # uint32_t numPartials
    # uint32_t numCacheLineIndicesPartials
    # ------------------------------------
    num_partials_token_info = len(partials_token_info)
    num_partials_cache_indices = len(partials_cache_indices)
    print(
        f"Num token info partials: {num_partials_token_info}, Num cache indices partials: {num_partials_cache_indices}"
    )
    encode_binary += struct.pack(
        "II", num_partials_token_info, num_partials_cache_indices
    )

    # compute padding size for full token vocab array
    fulls_vocab_org_size = len(fulls_vocab_bytes)
    fulls_vocab_pad_size = (
        align_n(fulls_vocab_org_size, align_num) - fulls_vocab_org_size
    )
    fulls_vocab_w_pad_size = fulls_vocab_org_size + fulls_vocab_pad_size
    fulls_vocab_format = f"{fulls_vocab_org_size}s{fulls_vocab_pad_size}b"

    # compute padding size for partial token vocab array
    partials_vocab_org_size = len(partials_vocab_bytes)
    partials_vocab_pad_size = (
        align_n(partials_vocab_org_size, align_num) - partials_vocab_org_size
    )
    partials_vocab_w_pad_size = partials_vocab_org_size + partials_vocab_pad_size
    partials_vocab_format = f"{partials_vocab_org_size}s{partials_vocab_pad_size}b"

    # uint32_t sizeVocabFulls
    # uint32_t sizeVocabPartials
    # --------------------------
    print(
        f"Size of full vocab w pad: {fulls_vocab_w_pad_size}, Size of partial vocab w pad: {partials_vocab_w_pad_size}"
    )
    encode_binary += struct.pack(
        "II", fulls_vocab_w_pad_size, partials_vocab_w_pad_size
    )

    # char* pVocabFulls
    # -----------------
    encode_binary += struct.pack(
        fulls_vocab_format, fulls_vocab_bytes, *[0] * fulls_vocab_pad_size
    )

    # uint32_t* pCacheLineIndicesFulls
    # --------------------------------
    fulls_cache_format = f"{num_fulls_cache_indices}I"
    encode_binary += struct.pack(fulls_cache_format, *fulls_cache_indices)

    # char* pVocabPartials
    # --------------------
    encode_binary += struct.pack(
        partials_vocab_format, partials_vocab_bytes, *[0] * partials_vocab_pad_size
    )

    # uint32_t* pCacheLineIndicesPartials
    # -----------------------------------
    partials_cache_format = f"{num_partials_cache_indices}I"
    encode_binary += struct.pack(partials_cache_format, *partials_cache_indices)

    # struct SingleTokenInfo* pFulls
    # ------------------------------
    fulls_token_info_format = "IIII" * num_fulls_token_info
    fulls_token_info_flat = [item for info in fulls_token_info for item in info]
    encode_binary += struct.pack(fulls_token_info_format, *fulls_token_info_flat)

    # struct SingleTokenInfo* pPartials
    # ------------------------------
    partials_token_info_format = "IIII" * num_partials_token_info
    partials_token_info_flat = [item for info in partials_token_info for item in info]
    encode_binary += struct.pack(partials_token_info_format, *partials_token_info_flat)

    # write to file
    with open(out_file, "wb") as enc_bin_file:
        enc_bin_file.write(encode_binary)

    return Path(out_file)

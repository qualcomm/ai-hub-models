# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import qai_hub
import torch
from filelock import FileLock

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def get_rope_theta(llm_config: PretrainedConfig) -> float:
    """Read RoPE theta across transformers layouts.

    transformers <5 exposed ``llm_config.rope_theta`` directly; 5.x nests RoPE
    settings under a single ``rope_parameters`` dict (e.g. Qwen3-VL). Prefer the
    top-level attribute when present, else fall back to ``rope_parameters``.
    """
    theta = getattr(llm_config, "rope_theta", None)
    if theta is not None:
        return theta
    rope_parameters = getattr(llm_config, "rope_parameters", None) or {}
    if "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    # Nested rope_parameters (e.g. Gemma4): prefer the local RoPE.
    for key in ("sliding_attention", "full_attention"):
        sub = rope_parameters.get(key)
        if isinstance(sub, dict) and "rope_theta" in sub:
            return sub["rope_theta"]
    raise AttributeError(
        "Could not find rope_theta on llm_config (checked top-level attribute, "
        "rope_parameters, and nested sliding_attention/full_attention sub-dicts)."
    )


def get_rope_scaling(llm_config: PretrainedConfig) -> dict[str, Any] | None:
    """Read RoPE scaling/parameters across transformers layouts.

    Returns the top-level ``rope_scaling`` dict (transformers <5) when present,
    else the 5.x ``rope_parameters`` dict, else None. Both carry ``mrope_section``
    and ``rope_type`` so downstream lookups work on either.
    """
    rope_scaling = getattr(llm_config, "rope_scaling", None)
    if rope_scaling is not None:
        return rope_scaling
    return getattr(llm_config, "rope_parameters", None)


def log_evaluate_test_result(
    model_name: str, checkpoint: str, metric: str, value: float
) -> None:
    """
    Logs the result of a model evaluation to a CSV file.

    The function appends a row to 'test_evaluate.csv' with the following columns:
        - Model Name
        - Checkpoint
        - Metric
        - Value
    If the file does not exist, a header row is written first.
    The file is locked during writing to prevent concurrent access.

    Parameters
    ----------
    model_name
        Name of the model being evaluated.
    checkpoint
        Checkpoint identifier for the model.
    metric
        Name of the evaluation metric.
    value
        Value of the evaluation metric.
    """
    log_file = Path("test_evaluate.csv")
    lock_file = log_file.with_suffix(".lock")

    with FileLock(str(lock_file)):
        file_exists = log_file.exists()
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Model Name", "Checkpoint", "Metric", "Value"])
            writer.writerow([model_name, checkpoint, metric, value])


def log_perf_on_device_result(
    model_name: str,
    precision: str,
    device: str,
    tps: float | None,
    prefill_tps: float | None,
    ttft_ms: float | None,
) -> None:
    """
    Logs the performance results of a model running on a specific device to a CSV file.

    The results are appended to 'test_perf_on_device.csv' in the current directory.

    Parameters
    ----------
    model_name
        Name of the model being evaluated.
    precision
        Precision mode used for inference (e.g., 'fp32', 'int8').
    device
        Device on which the model was run (e.g., 'Snapdragon X Elite', 'Snapdragon 8 Elite').
    tps
        Tokens per second, measuring throughput (unit: tokens/sec).
    prefill_tps
        Prefill (prompt-processing) tokens per second (unit: tokens/sec).
    ttft_ms
        Time to first token, measuring latency (unit: milliseconds).
    """
    log_file = Path("test_perf_on_device.csv")
    lock_file = log_file.with_suffix(".lock")

    with FileLock(str(lock_file)):
        file_exists = log_file.exists()
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "Model Name",
                        "Precision",
                        "Device",
                        "Decode (t/s)",
                        "Prefill (t/s)",
                        "TTFT (ms)",
                    ]
                )
            writer.writerow([model_name, precision, device, tps, prefill_tps, ttft_ms])


def create_genie_config(
    context_length: int,
    llm_config: PretrainedConfig,
    embedding_type: str,
    model_list: list[str],
    embedding_size: int | None = None,
    top_level_key: str = "dialog",
    embedding_lut_path: str | None = None,
    vlm_rope_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create Genie configuration for LLM or VLM models.

    Parameters
    ----------
    context_length
        Maximum context length.
    llm_config
        LLM configuration from transformers.
    embedding_type
        Type of positional embedding (e.g., "rope").
    model_list
        List of model binary files.
    embedding_size
        For VLM models using inputs_embeds (-e flag), specify the embedding/hidden size.
        When provided, adds the "embedding" section required for VLM inference.
    top_level_key
        Top-level config key. "dialog" for standalone LLMs, "text-generator"
        for VLM pipeline configs.
    embedding_lut_path
        Path to embedding LUT file. When provided, the embedding section uses
        the LUT format (type, lut-path, size, datatype) for VLM pipelines.
    vlm_rope_config
        VLM-specific MRoPE positional encoding overrides. When provided,
        replaces the standard rope-scaling logic. Expected keys:
        "rope-type", "time-step", "spatial-merge-size", "mrope-section".

    Returns
    -------
    dict[str, Any]
        Genie configuration dictionary.
    """
    kv_dim = getattr(
        llm_config, "head_dim", llm_config.hidden_size // llm_config.num_attention_heads
    )
    rope_dim = kv_dim // 2

    sampler = {
        "version": 1,
        "seed": 42,
        "temp": 0.8,
        "top-k": 40,
        "top-p": 0.95,
    }

    qnn_htp: dict[str, Any] = {
        "version": 1,
        "use-mmap": True,
        "spill-fill-bufsize": 0,
        "mmap-budget": 0,
        "poll": True,
        "cpu-mask": "0xe0",
        "kv-dim": kv_dim,
        "allow-async-init": False,
    }

    inner: dict[str, Any] = {
        "version": 1,
        "type": "basic",
        "context": {
            "version": 1,
            "size": context_length,
            "n-vocab": llm_config.vocab_size,
            "bos-token": llm_config.bos_token_id,
            "eos-token": llm_config.eos_token_id,
        },
        "sampler": sampler,
        "tokenizer": {"version": 1, "path": "tokenizer.json"},
        "engine": {
            "version": 1,
            "n-threads": 3,
            "backend": {
                "version": 1,
                "type": "QnnHtp",
                "QnnHtp": qnn_htp,
                "extensions": "htp_backend_ext_config.json",
            },
            "model": {
                "version": 1,
                "type": "binary",
                "binary": {
                    "version": 1,
                    "ctx-bins": model_list,
                },
            },
        },
    }

    # Positional encoding handling
    if vlm_rope_config is not None:
        # VLM-specific MRoPE (e.g., Qwen2.5-VL uses qwen2vl-mrope)
        qnn_htp["enable-graph-switching"] = False
        inner["engine"]["model"]["positional-encoding"] = {
            "type": embedding_type,
            "rope-dim": rope_dim,
            "rope-theta": int(get_rope_theta(llm_config)),
            "rope-scaling": vlm_rope_config,
        }
    else:
        # Standard LLM: put rope-theta and pos-id-dim in QnnHtp backend
        qnn_htp["pos-id-dim"] = rope_dim
        qnn_htp["rope-theta"] = int(get_rope_theta(llm_config))

        # Add rope-scaling for models like Llama 3.x that have full scaling params
        rope_scaling = get_rope_scaling(llm_config)
        if rope_scaling is not None and all(
            k in rope_scaling
            for k in [
                "rope_type",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
            ]
        ):
            inner["engine"]["model"]["positional-encoding"] = {
                "type": embedding_type,
                "rope-dim": rope_dim,
                "rope-theta": int(get_rope_theta(llm_config)),
                "rope-scaling": {
                    "rope-type": rope_scaling["rope_type"],
                    "factor": rope_scaling.get("factor", 8.0),
                    "low-freq-factor": rope_scaling["low_freq_factor"],
                    "high-freq-factor": rope_scaling["high_freq_factor"],
                    "original-max-position-embeddings": rope_scaling[
                        "original_max_position_embeddings"
                    ],
                },
            }
            del qnn_htp["pos-id-dim"]
            del qnn_htp["rope-theta"]

    # Add embedding section
    if embedding_lut_path is not None and embedding_size is not None:
        # VLM pipeline: LUT-based embedding with explicit path
        inner["embedding"] = {
            "version": 1,
            "type": "lut",
            "lut-path": embedding_lut_path,
            "size": embedding_size,
            "datatype": "float32",
        }
    elif embedding_size is not None:
        # VLM with inputs_embeds (-e flag): simple embedding section
        inner["embedding"] = {
            "version": 1,
            "size": embedding_size,
            "datatype": "float32",
        }

    return {top_level_key: inner}


def generate_genie_app_script(
    nodes: dict[str, str],
    connections: list,
    sample_inputs: list,
) -> str:
    """Generate genie-app-script.txt from pipeline topology data.

    Uses the same node/connection/sample_input structures that populate
    metadata.genie, so both outputs stay in sync.
    """
    config_names = {name: f"{name}Config" for name in nodes}

    io_type_hints = {
        "GENIE_NODE_TEXT_ENCODER_TEXT_INPUT": "textFile",
        "GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT": "image",
    }

    lines: list[str] = []
    lines.append("version")
    lines.append("pipeline config create pipelineConfig")
    lines.append("pipeline create GeniePipeline pipelineConfig")
    lines.append("")

    for node_name, config_file in nodes.items():
        lines.append(f"node config create {config_names[node_name]} {config_file}")
        lines.append(f"node create {node_name} {config_names[node_name]}")
        if "textGenerator" in node_name:
            lines.append(
                f"node set textCallback {node_name}"
                f" GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT"
            )
        lines.append("")

    lines.append("#Pipeline add and connect calls")
    lines.extend(f"pipeline add GeniePipeline {node_name}" for node_name in nodes)
    lines.append("")

    lines.extend(
        f"pipeline connect GeniePipeline {conn.producer_node}"
        f" {conn.producer_node_io}"
        f" {conn.consumer_node} {conn.consumer_node_io}"
        for conn in connections
    )
    lines.append("")

    for si in sample_inputs:
        set_type = io_type_hints.get(si.node_io, "embedding")
        lines.append(f"node set {set_type} {si.node} {si.node_io} {si.file}")
    lines.append("")

    lines.append("pipeline execute GeniePipeline")
    lines.append("")
    lines.extend(f"node free {node_name}" for node_name in nodes)
    lines.append("pipeline free GeniePipeline")

    return "\n".join(lines) + "\n"


def export_embedding_weights_from_tensor(
    embed_weights: torch.Tensor,
    output_path: Path | str,
    filename: str = "embedding_weights.raw",
) -> Path:
    """Export embedding weights tensor as a raw float32 file."""
    import numpy as np

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / filename
    embed_weights.cpu().numpy().astype(np.float32).tofile(output_file)
    return output_file


def export_embedding_weights(
    model: torch.nn.Module,
    output_path: Path | str,
    filename: str = "embedding_weights.raw",
) -> Path:
    """Export the embedding table from a model as a raw float32 file."""
    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()  # type: ignore[operator, unused-ignore]
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_layer = model.model.embed_tokens
    else:
        raise ValueError("Could not find embedding layer in model")
    embed_weights: torch.Tensor = embed_layer.weight.data
    return export_embedding_weights_from_tensor(embed_weights, output_path, filename)


# The folder is not always the ABI name (may include toolchain as well)
ABI_TO_LIB_FOLDER: dict[str, str] = {
    "aarch64-windows": "aarch64-windows-msvc",
}


def copy_qairt_files_for_genie_bundle(
    hub_device: qai_hub.Device,
    output_path: Path,
    qairt_sdk_path: Path,
) -> None:
    """Copy the QAIRT files needed to create the genie_bundle."""
    hexagon_arch, abi_name, genie_file = None, None, None
    for attr in hub_device.attributes:
        if "hexagon" in attr:
            hexagon_arch = attr.replace(":", "-")
        if "abi" in attr:
            abi_name = attr.removeprefix("abi:")

    lib_name = (
        ABI_TO_LIB_FOLDER.get(abi_name, abi_name) if abi_name is not None else None
    )

    genie_file = (
        "genie-t2t-run.exe"
        if "os:windows" in hub_device.attributes
        else "genie-t2t-run"
    )
    files_copied = []
    if hexagon_arch is not None and lib_name is not None and genie_file is not None:
        path_libhex = os.path.join(qairt_sdk_path, "lib", hexagon_arch, "unsigned", "*")
        path_libqnn = os.path.join(qairt_sdk_path, "lib", lib_name, "*")
        path_exe = os.path.join(qairt_sdk_path, "bin", lib_name, genie_file)
        # Copy the lib files
        for file in glob.glob(path_libhex):
            shutil.copy(file, output_path)
            files_copied.append(file)
        # Copy the bin files
        for file in glob.glob(path_libqnn):
            shutil.copy(file, output_path)
            files_copied.append(file)
        # Copy the genie t2t file
        shutil.copy(path_exe, output_path)
        files_copied.append(path_exe)


def save_htp_config_for_genie_bundle(
    device_info: dict[str, str], output_path: Path
) -> bool:
    """Saves the htp_backend_ext_config.json to the genie_bundle directory.

    Returns True if the file was written, False if device info was insufficient.
    """
    hexagon_arch = device_info.get("hexagon")
    soc_model = device_info.get("soc-model")
    if hexagon_arch is None or soc_model is None:
        print(
            f"Could not add 'htp_backend_ext_config.json' to the genie_bundle ({output_path})"
        )
        return False

    try:
        soc_model_int = int(soc_model)
    except (ValueError, TypeError):
        raise ValueError(
            f"Expected numeric soc-model device attribute, got: {soc_model!r}"
        ) from None

    htp_config = {
        "devices": [
            {
                "soc_model": soc_model_int,
                "dsp_arch": hexagon_arch,
                "cores": [
                    {
                        "core_id": 0,
                        "perf_profile": "burst",
                        "rpc_control_latency": 100,
                    }
                ],
            }
        ],
        "memory": {"mem_type": "shared_buffer"},
        "context": {"weight_sharing_enabled": True},
    }

    with open(output_path / "htp_backend_ext_config.json", "w") as f:
        json.dump(htp_config, f)
    return True


def get_kv_cache_names(start: int, end: int) -> list[str]:
    return [
        f"past_{field}_{num}_out"
        for num in range(start, end)
        for field in ("key", "value")
    ]

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import onnx
import torch
from transformers import PreTrainedTokenizerBase

from qai_hub_models import Precision
from qai_hub_models.models.templates.llama3.model import (
    LlamaPartBase,
    LlamaPreSplitBase,
    LlamaPreSplitCollectionBase,
    LlamaQuantizablePreSplitBase,
)
from qai_hub_models.models.templates.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS as GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS,
)
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.models.templates.llm.model import (
    CollectionCompileO0Mixin,
    CompileO0Mixin,
    QuantizeFullTypeW8A16Mixin,
    QuantizeFullTypeW8A16PartMixin,
    SplitForwardMixin,
)
from qai_hub_models.models.templates.lm_driver.generator import HubCompatibleGenerator

# LLMIOType is re-exported from this module so the CLI input-spec parser can
# resolve the inherited get_input_spec's "llm_io_type" annotation, which it
# looks up in the concrete model's module.

logger = logging.getLogger(__name__)


def _strip_rmsnorm_unsqueeze_squeeze(model: onnx.ModelProto) -> None:
    """Remove the Unsqueeze/Squeeze pair wrapping the residual stream in each RMSNorm."""
    tensor_consumers: dict[str, list] = {}
    for node in model.graph.node:
        for inp in node.input:
            tensor_consumers.setdefault(inp, []).append(node)

    nodes_to_remove: set[str] = set()
    rewrites: dict[str, str] = {}

    for node in list(model.graph.node):
        if node.op_type != "Unsqueeze":
            continue
        unsqueeze_out = node.output[0]
        consumers = tensor_consumers.get(unsqueeze_out, [])
        consumer_types = {c.op_type for c in consumers}

        if consumer_types != {"Pow", "Mul"}:
            continue

        residual = node.input[0]

        pow_consumers = [c for c in consumers if c.op_type == "Pow"]
        if not pow_consumers:
            continue
        cur = pow_consumers[0]
        squeeze_node = None
        for _ in range(12):
            cur_out = cur.output[0] if cur.output else None
            if not cur_out:
                break
            next_nodes = tensor_consumers.get(cur_out, [])
            if not next_nodes:
                break
            cur = next_nodes[0]
            if cur.op_type == "Squeeze":
                squeeze_node = cur
                break

        if squeeze_node is None:
            continue

        nodes_to_remove.add(node.name)
        nodes_to_remove.add(squeeze_node.name)
        rewrites[unsqueeze_out] = residual
        rewrites[squeeze_node.output[0]] = squeeze_node.input[0]

    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rewrites:
                node.input[i] = rewrites[inp]

    new_nodes = [n for n in model.graph.node if n.name not in nodes_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    del model.graph.value_info[:]


DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 5

# Model architecture constants (from SmolLM2-1.7B-Instruct config.json)
NUM_LAYERS = 24
NUM_SPLITS = 3
NUM_LAYERS_PER_SPLIT = 12
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 32
NUM_ATTN_HEADS = 32
# SmolLM2-1.7B-Instruct: 2048 // 32 = 64 (no explicit head_dim in config, computed)
HEAD_DIM = 64

HF_REPO_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

MIN_MEMORY_RECOMMENDED = 40

DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT = {Precision.w4a16: "smollm2_1_7b_it_w4a16"}
SPLIT_MODEL_NAME = "SmolLM2_1_7B_Instruct"
SPINQUANT_CONFIG = {"enable_r1": True, "enable_r2": True, "enable_r3": False}
# AdaScale config: SmolLM2-1.7B-Instruct has 32 attn heads + 32 KV heads + 1 = 65 RMSNorms per block
ADA_SCALE_NUM_RMSNORM_PER_BLK = NUM_ATTN_HEADS + NUM_KEY_VALUE_HEADS + 1

END_TOKENS = {"<|im_end|>"}


class Smollm2_1_7B_Instruct_PreSplit(LlamaPreSplitBase):
    """FP PreSplit for Smollm2 1.7B Instruct."""

    default_user_prompt = "What is gravity? Keep the answer under ten words."
    default_system_prompt = "You are a helpful AI assistant."

    model_id = MODEL_ID
    GeneratorClass = HubCompatibleGenerator
    model_asset_version = MODEL_ASSET_VERSION
    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    hf_repo_name = HF_REPO_NAME
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        return {
            "global_prefix": "",
            "system_prefix": "<|im_start|>system\n",
            "system_suffix": "<|im_end|>",
            "user_prefix": "<|im_start|>user\n",
            "user_suffix": "<|im_end|>",
            "assistant_prefix": "<|im_start|>assistant\n",
            "assistant_suffix": "<|im_end|>",
            "default_system_prompt": cls.default_system_prompt,
        }

    @classmethod
    def get_input_prompt_with_tags(
        cls,
        user_input_prompt: str | None = None,
        system_context_prompt: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        include_image: bool = False,
        **kwargs: Any,
    ) -> str:
        if tokenizer is not None and getattr(tokenizer, "chat_template", None) is None:
            return user_input_prompt or cls.default_user_prompt
        return super().get_input_prompt_with_tags(
            user_input_prompt=user_input_prompt,
            system_context_prompt=system_context_prompt,
            tokenizer=tokenizer,
            include_image=include_image,
            **kwargs,
        )


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Smollm2_1_7B_Instruct_QuantizablePreSplit(  # type: ignore[misc]
    QuantizeFullTypeW8A16Mixin,
    CompileO0Mixin,
    LlamaQuantizablePreSplitBase[Smollm2_1_7B_Instruct_PreSplit],
):
    """Quantizable PreSplit for Smollm2 1.7B Instruct."""

    FPModel = Smollm2_1_7B_Instruct_PreSplit
    GeneratorClass = HubCompatibleGenerator

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    num_layers = NUM_LAYERS
    supported_precisions = SUPPORTED_PRECISIONS
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION

    ada_scale_model_type: str | None = "llama"
    ada_scale_num_rmsnorm_per_blk = ADA_SCALE_NUM_RMSNORM_PER_BLK
    spinquant_config = SPINQUANT_CONFIG

    @classmethod
    def resolve_default_checkpoint(
        cls,
        precision: Precision,
        host_device: torch.device,
        fp_model: Smollm2_1_7B_Instruct_PreSplit | None,
    ) -> tuple[str, Smollm2_1_7B_Instruct_PreSplit | None]:
        """Downloads the full checkpoint zip"""
        return cls.fetch_default_checkpoint(precision), fp_model

    @classmethod
    def apply_pre_sim_transforms(
        cls,
        output_dir: Path,
        spinquant_config: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Strip RMSNorm Unsqueeze/Squeeze pairs (required for AdaScale), then apply SpinQuant."""
        backbone_path = output_dir / "model_dynamic.onnx"
        backbone_onnx = onnx.load(str(backbone_path), load_external_data=True)

        _strip_rmsnorm_unsqueeze_squeeze(backbone_onnx)

        if spinquant_config:
            cls.apply_spinquant_to_onnx(backbone_onnx, spinquant_config, **kwargs)

        onnx.save_model(
            backbone_onnx,
            str(backbone_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )
        del backbone_onnx
        gc.collect()


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
class Smollm2_1_7B_Instruct_PartBase(QuantizeFullTypeW8A16PartMixin, LlamaPartBase):
    """Unified Part base for Smollm2 1.7B Instruct."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    fp_presplit_cls = Smollm2_1_7B_Instruct_PreSplit
    quant_presplit_cls = Smollm2_1_7B_Instruct_QuantizablePreSplit
    default_precision = DEFAULT_PRECISION

    def _get_onnx_input_names(self) -> list[str]:
        return [
            n.replace("/", "_").replace(".", "_")
            for n in super()._get_onnx_input_names()
        ]


class Smollm2_1_7B_Instruct_Part1_Of_3(Smollm2_1_7B_Instruct_PartBase):
    """Part 1: Embedding."""

    part_id = 1


class Smollm2_1_7B_Instruct_Part2_Of_3(Smollm2_1_7B_Instruct_PartBase):
    """Part 2: Middle layers."""

    part_id = 2


class Smollm2_1_7B_Instruct_Part3_Of_3(Smollm2_1_7B_Instruct_PartBase):
    """Part 3: Final layers + LM head."""

    part_id = 3


_SPLIT_PART_CLASSES: list[type] = [
    Smollm2_1_7B_Instruct_Part1_Of_3,
    Smollm2_1_7B_Instruct_Part2_Of_3,
    Smollm2_1_7B_Instruct_Part3_Of_3,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Smollm2_1_7B_Instruct_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Smollm2_1_7B_Instruct_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Smollm2_1_7B_Instruct_Collection(
    CollectionCompileO0Mixin, LlamaPreSplitCollectionBase
):
    """Unified Collection with 3 Parts for Smollm2 1.7B Instruct."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Smollm2_1_7B_Instruct_PreSplit
    part_base_cls = Smollm2_1_7B_Instruct_PartBase
    parts = {
        "part1_of_3": Smollm2_1_7B_Instruct_Part1_Of_3,
        "part2_of_3": Smollm2_1_7B_Instruct_Part2_Of_3,
        "part3_of_3": Smollm2_1_7B_Instruct_Part3_Of_3,
    }

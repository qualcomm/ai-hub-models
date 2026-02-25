# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import gc
import itertools
import math
from collections.abc import Generator
from typing import Any, cast

import torch
import transformers
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast

from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.model import (
    LLM_QNN,
    Embedding,
    LLM_AIMETOnnx,
    LLMBase,
)


def get_past_keyval_with_shift(
    past_key_vals: list[torch.Tensor],
    new_key_vals: list[torch.Tensor],
    length: int,
    device: torch.device = torch.device("cpu"),
) -> list[torch.Tensor]:
    """Clip past key value to feed next iteration"""
    ret = []

    if len(past_key_vals) == 0:
        for i in range(0, len(new_key_vals), 2):
            orig_key_shape = new_key_vals[i].shape
            key_shape = (orig_key_shape[0], orig_key_shape[1], orig_key_shape[2], 0)
            past_key_vals.append(torch.zeros(key_shape, device=device))

            orig_value_shape = new_key_vals[i + 1].shape
            value_shape = (
                orig_value_shape[0],
                orig_value_shape[1],
                0,
                orig_value_shape[3],
            )
            past_key_vals.append(torch.zeros(value_shape, device=device))

    if len(new_key_vals) == 0:
        for i in range(0, len(past_key_vals), 2):
            orig_key_shape = past_key_vals[i].shape
            key_shape = (orig_key_shape[0], orig_key_shape[1], orig_key_shape[2], 0)
            new_key_vals.append(torch.zeros(key_shape, device=device))

            orig_value_shape = past_key_vals[i + 1].shape
            value_shape = (
                orig_value_shape[0],
                orig_value_shape[1],
                0,
                orig_value_shape[3],
            )
            new_key_vals.append(torch.zeros(value_shape, device=device))

    # Key and Values are concatenated on batch dimension
    for i in range(0, len(past_key_vals), 2):
        key_cache = torch.cat(
            [past_key_vals[i].to(device), new_key_vals[i].to(device)],
            dim=3,
        )
        key_cache = key_cache[:, :, :, -length:]
        val_cache = torch.cat(
            [
                past_key_vals[i + 1].to(device),
                new_key_vals[i + 1].to(device),
            ],
            dim=2,
        )
        val_cache = val_cache[:, :, -length:, :]

        ret.append(key_cache)
        ret.append(val_cache)
    return ret


class LLM_Loader:
    def __init__(
        self,
        model_cls: type[LLMBase | LLM_AIMETOnnx | LLM_QNN],
        sequence_length: int,
        model_params: dict[str, Any],
        host_device: torch.device,
    ) -> None:
        self.model_cls = model_cls
        self.sequence_length = sequence_length
        self.model_params = model_params
        self.loaded_model: LLMBase | LLM_AIMETOnnx | LLM_QNN | None = None
        self.host_device = host_device

    def load(self) -> LLMBase | LLM_AIMETOnnx | LLM_QNN:
        if self.loaded_model is None:
            self.loaded_model = self.model_cls.from_pretrained(
                sequence_length=self.sequence_length, **self.model_params
            ).to(self.host_device)

        assert self.loaded_model is not None
        return self.loaded_model

    def release(self) -> None:
        if self.loaded_model is not None and isinstance(
            self.loaded_model, LLM_AIMETOnnx
        ):
            self.loaded_model = self.loaded_model.to("cpu")
            if hasattr(self.loaded_model, "quant_sim"):
                del self.loaded_model.quant_sim
        del self.loaded_model
        # Python can be in a weird state when __del__ gets called, so we
        # have to make sure these still exist.
        if "gc" in globals() and gc is not None:
            gc.collect()
        if "torch" in globals() and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.loaded_model = None

    def __del__(self) -> None:
        self.release()


class LLM_Generator(GenerationMixin, torch.nn.Module):
    _is_stateful = False

    def __init__(
        self,
        models: list[LLMBase | LLM_AIMETOnnx | LLM_QNN | LLM_Loader],
        tokenizer: transformers.PreTrainedTokenizerBase,
        embedding: Embedding,
        accumulate_logits_on_cpu: bool = False,
    ) -> None:
        super().__init__()

        self.models = models
        self.models.sort(key=lambda model: model.sequence_length)

        self.selected_model = (
            self.models[-1].load()
            if isinstance(self.models[-1], LLM_Loader)
            else self.models[-1]
        )

        self.tokenizer = tokenizer
        self.embedding = embedding
        self.accumulate_logits_on_cpu = accumulate_logits_on_cpu

    def cleanup(self) -> None:
        for model in self.models:
            if isinstance(model, LLM_Loader):
                model.release()
        if isinstance(self.selected_model, LLM_Loader):
            self.selected_model.release()
        if "gc" in globals() and gc is not None:
            gc.collect()
        if "torch" in globals() and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def config(self) -> PretrainedConfig:
        return self.selected_model.llm_config

    @property
    def main_input_name(self) -> str:
        return self.selected_model.main_input_name

    @property
    def llm_io_type(self) -> LLMIOType:
        assert self.selected_model is not None
        return self.selected_model.llm_io_type

    @property
    def _supports_cache_class(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        if isinstance(self.selected_model, LLM_AIMETOnnx):
            assert self.selected_model.host_device is not None
            return self.selected_model.host_device

        # Note: torch.nn.Module.device does not exist according to PyTorch
        # documentation and mypy.
        return next(iter(self.selected_model.parameters())).device

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | DynamicCache | None]:
        """
        Overridden prepare_inputs_for_generation function to enable Huggingface generate() on models with static
        graph constraints
        """
        # We need a way to ensure that all the previous tokens that have already been consumed are stripped out of the
        # input ids

        # If past_key_values is None, this indicates that this `prepare_inputs_for_generation()` is being called for
        # the first time, and nothing should be stripped out of `input_ids`. In other cases though, the number of tokens
        # already inside `past_key_values` indicates how many tokens should be stripped out of `input_ids`

        # Notes: `input_ids`, `attention_mask`, `past_key_values` should NOT have static shape requirements imposed on
        # them by the time they reach this function. That is, in order for this to work, the static shape padding and
        # truncation must happen directly in the model `forward` function

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        assert input_ids is not None or inputs_embeds is not None

        if past_key_values is None:
            num_processed_tokens = 0

        elif hasattr(past_key_values, "value_cache"):
            num_processed_tokens = (
                0
                if len(past_key_values.value_cache) == 0
                or past_key_values.value_cache[0] == []
                else past_key_values.value_cache[0].shape[-2]
            )
        elif past_key_values.layers and hasattr(past_key_values.layers[0], "values"):  # type: ignore[attr-defined, unused-ignore]
            num_processed_tokens = (
                0
                if past_key_values.layers[0].values is None  # type: ignore[attr-defined, unused-ignore]
                else past_key_values.layers[0].values.shape[-2]  # type: ignore[attr-defined, unused-ignore]
            )
        else:
            raise ValueError("Unsupported KV cache type")

        inputs: dict[str, torch.Tensor | DynamicCache | None] = {}
        if input_ids is not None:
            inputs["input_ids"] = input_ids[:, num_processed_tokens:]
        elif inputs_embeds is not None:
            inputs["input_embeds"] = inputs_embeds[:, num_processed_tokens:, :]

        return inputs | {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    def select_model(self, num_input_tokens: int) -> LLM_AIMETOnnx | LLM_QNN | LLMBase:
        # Select the model with the smallest sequence length that can fit all of num_input_tokens
        # If there is no model that can consume num_input_tokens in one inference, select the model with the largest
        # sequence length
        new_selected_model = self.models[
            -1
        ]  # start off by selecting model with largest sequence length
        for model in self.models:
            if (
                num_input_tokens <= model.sequence_length
                and model.sequence_length < new_selected_model.sequence_length
            ):
                new_selected_model = model  # if there is any model with a smaller sequence length that works, select it

        if self.selected_model.sequence_length == new_selected_model.sequence_length:
            return self.selected_model

        print(
            f"Switching from sequence_length={self.selected_model.sequence_length} to sequence_length={new_selected_model.sequence_length}"
        )
        # release the model to preserve memory
        if isinstance(self.selected_model, (LLM_Loader, LLM_AIMETOnnx, LLM_QNN)):
            self.selected_model.release()

        self.selected_model = (
            new_selected_model.load()
            if isinstance(new_selected_model, LLM_Loader)
            else new_selected_model
        )
        return self.selected_model

    @staticmethod
    def slice_inputs_for_inference(
        inputs: torch.Tensor, attention_mask: torch.Tensor, sequence_length: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        input_length = inputs.shape[1]
        for idx in range(0, input_length, sequence_length)[::-1]:
            idx = input_length - idx
            yield (
                inputs[:, max(0, idx - sequence_length) : idx],
                attention_mask[:, max(0, idx - sequence_length) : idx],
            )

    def prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        past_key_values: list[torch.Tensor],
        sequence_length: int,
        context_length: int,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        # If primary method of accepting inputs is inputs_embeds, but input_ids are provided (ie - in generation)
        # then convert tokens to embeddings
        if self.llm_io_type == LLMIOType.genie_input_embeds and input_ids is not None:
            inputs_embeds = cast(
                torch.FloatTensor,
                self.selected_model.convert_input_ids_to_embeddings(input_ids),
            )
            input_ids = None

        input_tokens = input_ids if input_ids is not None else inputs_embeds
        assert isinstance(input_tokens, torch.Tensor)
        input_tokens = input_tokens.to(
            dtype=torch.int32 if input_ids is not None else torch.float32
        )

        device = input_tokens.device
        batch_size = input_tokens.shape[0]
        input_length = input_tokens.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=input_tokens.device,
            )

        if input_ids is not None:
            input_tokens_extension = torch.full(
                (batch_size, sequence_length - input_length),
                fill_value=getattr(self.tokenizer, "eos_token_id", 0),
                dtype=input_tokens.dtype,
                device=device,
            )
        else:
            embedding_dim = input_tokens.shape[2]
            input_tokens_extension = torch.zeros(
                (batch_size, sequence_length - input_length, embedding_dim),
                dtype=input_tokens.dtype,
                device=device,
            )

        padded_input_tokens = torch.cat((input_tokens_extension, input_tokens), dim=1)
        attention_mask_extension = torch.zeros(
            (batch_size, sequence_length - input_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        padded_attention_mask = torch.cat(
            (torch.zeros_like(attention_mask_extension), attention_mask), dim=-1
        )

        input_specs = self.selected_model.get_input_spec(
            llm_config=self.selected_model.llm_config.to_dict(),
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=self.llm_io_type,
        )
        # Initialization of KV cache padding
        dummy_past_key_values = [
            torch.zeros(shape, device=device)
            for k, (shape, _) in input_specs.items()
            if k.startswith("past_")
        ]

        current_key_value_length = (
            past_key_values[1].shape[-2] if past_key_values else 0
        )
        key_value_padding_length = (
            context_length - sequence_length
        ) - current_key_value_length

        padded_past_key_values = get_past_keyval_with_shift(
            past_key_vals=dummy_past_key_values,
            new_key_vals=past_key_values,
            length=context_length - sequence_length,
            device=device,
        )

        kv_cache_attention_mask = torch.cat(
            (
                torch.zeros((batch_size, key_value_padding_length)),
                torch.ones((batch_size, current_key_value_length)),
            ),
            dim=-1,
        ).to(device=device)
        padded_attention_mask = torch.cat(
            (kv_cache_attention_mask, padded_attention_mask), dim=-1
        )

        position_ids = torch.cumsum(padded_attention_mask, dim=1, dtype=torch.int32) - 1
        position_ids = position_ids.clip(0, context_length - 1)
        position_ids = position_ids[..., -sequence_length:]

        attention_mask_converter = AttentionMaskConverter(True)
        cm_attention_mask = attention_mask_converter.to_4d(
            padded_attention_mask,
            query_length=sequence_length,
            key_value_length=context_length,
            dtype=torch.float32,
        )
        attention_mask_min_clip = getattr(
            self.selected_model, "attention_mask_min_clip", None
        )
        if attention_mask_min_clip is not None:
            cm_attention_mask = cm_attention_mask.clip(min=attention_mask_min_clip)

        if self.llm_io_type == LLMIOType.huggingface_input_ids:
            return (
                padded_input_tokens,
                cm_attention_mask,
                position_ids,
                *padded_past_key_values,
            )
        position_ids_cos, position_ids_sin = self.embedding.get_embedding(position_ids)
        return (
            padded_input_tokens,
            cm_attention_mask,
            position_ids_cos,
            position_ids_sin,
            *padded_past_key_values,
        )

    def combine_local_and_global_outputs(
        self,
        model: LLMBase | LLM_AIMETOnnx | LLM_QNN,
        num_valid_input_tokens: int,
        local_outputs: tuple[torch.Tensor, ...],
        global_outputs: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        device = local_outputs[0].device
        logits_device = "cpu" if self.accumulate_logits_on_cpu else device

        # strip logits corresponding to padding tokens
        local_logits = local_outputs[0]
        local_logits = torch.narrow(
            local_logits,
            1,
            local_logits.shape[1] - num_valid_input_tokens,
            num_valid_input_tokens,
        ).to(logits_device)

        # concatenate logits from local inference to global output
        if "logits" in global_outputs:
            assert isinstance(global_outputs["logits"], torch.Tensor)
            global_outputs["logits"] = torch.cat(
                [global_outputs["logits"], local_logits], dim=1
            )
        else:
            global_outputs["logits"] = local_logits

        # strip KV cache corresponding to padding tokens
        local_past_key_values = get_past_keyval_with_shift(
            past_key_vals=[],
            new_key_vals=list(local_outputs[1:]),
            length=num_valid_input_tokens,
            device=device,
        )

        past_key_values_list = global_outputs["past_key_values"]
        assert isinstance(past_key_values_list, list)

        # shift global KV cache, concatenate local KV cache
        current_key_value_length = (
            past_key_values_list[1].shape[-2] if past_key_values_list else 0
        )
        global_outputs["past_key_values"] = get_past_keyval_with_shift(
            past_key_vals=past_key_values_list,
            new_key_vals=local_past_key_values,
            length=min(
                current_key_value_length + num_valid_input_tokens,
                model.context_length - model.sequence_length,
            ),
            device=device,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        input_tokens = input_ids if input_ids is not None else inputs_embeds
        assert isinstance(input_tokens, torch.Tensor)

        # Select which model to use
        model = self.select_model(input_tokens.shape[1])

        # Create attention mask if one does not exist
        if attention_mask is None:
            batch_size = input_tokens.shape[0]
            input_length = input_tokens.shape[1]
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=input_tokens.device,
            )

        global_outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "past_key_values": (
                []
                if past_key_values is None or past_key_values.get_seq_length() == 0
                else list(
                    itertools.chain.from_iterable(past_key_values.to_legacy_cache())
                )
            )
        }

        for input_slice, attention_mask_slice in self.slice_inputs_for_inference(
            input_tokens, attention_mask, model.sequence_length
        ):
            past_key_values_list = global_outputs["past_key_values"]
            assert isinstance(past_key_values_list, list)

            prepared_inputs = self.prepare_inputs(
                input_ids=input_slice if input_ids is not None else None,
                attention_mask=attention_mask_slice,
                past_key_values=past_key_values_list,
                sequence_length=model.sequence_length,
                context_length=model.context_length,
                inputs_embeds=cast(torch.FloatTensor, input_slice)
                if inputs_embeds is not None
                else None,
            )

            local_outputs = model(*prepared_inputs)
            self.combine_local_and_global_outputs(
                model,
                input_slice.shape[1],
                local_outputs,
                global_outputs,
            )

        # make sure logits are on the correct device (necessary for generation)
        # the underlying mock_torch_onnx_inference function does not necessarily move outputs back to CUDA
        assert isinstance(global_outputs["logits"], torch.Tensor)
        logits = global_outputs["logits"].to(
            device="cpu" if self.accumulate_logits_on_cpu else input_tokens.device
        )

        # Convert KV Cache outputs into HF DynamicCache
        past_key_values = DynamicCache()
        for layer_idx in range(len(global_outputs["past_key_values"]) // 2):
            past_key_values.update(
                global_outputs["past_key_values"][layer_idx * 2],
                global_outputs["past_key_values"][layer_idx * 2 + 1],
                layer_idx,
            )
        return CausalLMOutputWithPast(
            logits=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,  # type: ignore[arg-type, unused-ignore]
        )

    def prefill(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Any,
    ) -> Generator[tuple[torch.Tensor, ...], None, None]:
        if len(self.models) > 1:
            raise RuntimeError("Prefill should only be invoked using a single model")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        input_tokens = input_ids if input_ids is not None else inputs_embeds
        assert isinstance(input_tokens, torch.Tensor)

        # Select which model to use
        model = self.select_model(input_tokens.shape[1])

        # Create attention mask if one does not exist
        if attention_mask is None:
            batch_size = input_tokens.shape[0]
            input_length = input_tokens.shape[1]
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=input_tokens.device,
            )

        # slice input ids and attention mask to drop last few tokens
        total_num_inferences = math.ceil(input_tokens.shape[1] / model.sequence_length)
        num_tokens_to_preconsume = (total_num_inferences - 1) * model.sequence_length

        input_tokens_to_preconsume = input_tokens[:, :num_tokens_to_preconsume]
        attention_mask_to_preconsume = attention_mask[:, :num_tokens_to_preconsume]

        preconsumed_outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "past_key_values": (
                []
                if past_key_values is None or past_key_values.get_seq_length() == 0
                else list(
                    itertools.chain.from_iterable(past_key_values.to_legacy_cache())
                )
            )
        }

        for input_slice, attention_mask_slice in self.slice_inputs_for_inference(
            input_tokens_to_preconsume,
            attention_mask_to_preconsume,
            model.sequence_length,
        ):
            past_key_values_list = preconsumed_outputs["past_key_values"]
            assert isinstance(past_key_values_list, list)

            prepared_inputs = self.prepare_inputs(
                input_ids=input_slice if input_ids is not None else None,
                attention_mask=attention_mask_slice,
                past_key_values=past_key_values_list,
                sequence_length=model.sequence_length,
                context_length=model.context_length,
                inputs_embeds=cast(torch.FloatTensor, input_slice)
                if inputs_embeds is not None
                else None,
            )

            yield tuple(tensor.cpu() for tensor in prepared_inputs)

            local_outputs = model(*prepared_inputs)
            self.combine_local_and_global_outputs(
                model,
                input_slice.shape[1],
                local_outputs,
                preconsumed_outputs,
            )

        remaining_input_tokens = input_tokens[:, num_tokens_to_preconsume:]
        remaining_attention_mask = attention_mask[:, num_tokens_to_preconsume:]
        past_key_values_list = preconsumed_outputs["past_key_values"]
        assert isinstance(past_key_values_list, list)
        prefilled_inputs = self.prepare_inputs(
            remaining_input_tokens,
            remaining_attention_mask,
            past_key_values_list,
            model.sequence_length,
            model.context_length,
        )

        yield tuple(tensor.cpu() for tensor in prefilled_inputs)

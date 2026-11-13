# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import GeneratorType
from typing import Any

import numpy as np
import qai_hub as hub
import torch

from qai_hub_models.models.grootn17.dataset import step_from_tensors
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.data.types import (
    VLAStepData,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.policy.gr00t_policy import (
    Gr00tPolicy,
)
from qai_hub_models.models.grootn17.model import GrootCollection as Model
from qai_hub_models.models.grootn17.model import (
    build_reference_metadata,
    preprocess_llm,
    preprocess_vlm_proj,
)
from qai_hub_models.models.grootn17.model_adaptations import assert_grid_thw_matches
from qai_hub_models.models.grootn17.utils import (
    ActionLayout,
    concatenate_action_modalities,
    get_dit_input_spec,
    get_llm_input_spec,
    get_vit_input_spec,
    get_vlm_proj_input_spec,
)
from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.base_app import CollectionModelEvalGenerator
from qai_hub_models.utils.evaluate.helpers import EvalMode
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    AsyncOnDeviceResult,
    OnDeviceModel,
)
from qai_hub_models.utils.input_spec import InputSpec


def _split_to_list(
    tensor: torch.Tensor, batch_size: int, batch_axis: int
) -> list[torch.Tensor]:
    """Split a tensor into samples while preserving its batch dimension."""
    if tensor.shape[batch_axis] != batch_size:
        raise ValueError(
            f"Expected batch dimension {batch_size} on axis {batch_axis}, "
            f"but received shape {tuple(tensor.shape)}."
        )
    samples = []
    for index in range(batch_size):
        slices = [slice(None)] * tensor.ndim
        slices[batch_axis] = slice(index, index + 1)
        samples.append(tensor[tuple(slices)])
    return samples


def _prepare_on_device_inputs(
    tensors: Sequence[torch.Tensor],
    batch_axes: Sequence[int | None],
) -> list[list[torch.Tensor]]:
    """Package component inputs as Hub samples without splitting structural axes."""
    if len(tensors) != len(batch_axes):
        raise ValueError(
            f"Expected one batch-axis descriptor per tensor, got {len(batch_axes)} "
            f"descriptors for {len(tensors)} tensors."
        )

    batch_sizes = [
        tensor.shape[axis]
        for tensor, axis in zip(tensors, batch_axes, strict=True)
        if axis is not None
    ]
    batch_size = batch_sizes[0] if batch_sizes else 1
    if any(size != batch_size for size in batch_sizes):
        raise ValueError(
            f"Component inputs have inconsistent batch sizes: {batch_sizes}"
        )
    if batch_size != 1 and any(axis is None for axis in batch_axes):
        raise ValueError(
            "Unbatched component inputs are supported only for batch-size-one inference."
        )

    return [
        [tensor] if axis is None else _split_to_list(tensor, batch_size, axis)
        for tensor, axis in zip(tensors, batch_axes, strict=True)
    ]


def _wait(
    result: torch.Tensor | tuple[torch.Tensor, ...] | AsyncOnDeviceResult,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Resolve a submitted Hub result before its dependent stage runs."""
    return result.wait() if isinstance(result, AsyncOnDeviceResult) else result


def _call_component(
    component: ExecutableModelProtocol | AsyncOnDeviceModel,
    *tensors: torch.Tensor,
    batch_axes: Sequence[int | None] | None = None,
    input_spec: InputSpec | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Call a component with stacked (B, ...) tensors.
    - FP nn.Module  : forward() accepts (B, ...) directly.
    - Device wrappers: package actual batch axes and wait for dependent results.
    """
    if isinstance(component, (OnDeviceModel, AsyncOnDeviceModel)):
        device = tensors[0].device
        if input_spec is not None:
            if len(tensors) != len(input_spec):
                raise ValueError(
                    f"Expected {len(input_spec)} component inputs, got {len(tensors)}."
                )
            for (name, spec), tensor in zip(input_spec.items(), tensors, strict=True):
                if (
                    tuple(tensor.shape) != tuple(spec.shape)
                    or str(tensor.dtype).removeprefix("torch.") != spec.dtype
                ):
                    raise ValueError(
                        f"Input '{name}' must have shape {spec.shape} and dtype {spec.dtype}; got {tuple(tensor.shape)} and {tensor.dtype}."
                    )
        axes = batch_axes if batch_axes is not None else [0] * len(tensors)
        result = _wait(component(*_prepare_on_device_inputs(tensors, axes)))  # type: ignore[arg-type]
        if isinstance(result, torch.Tensor):
            result = result.to(device)
        elif isinstance(result, (list, tuple)):
            result = tuple(t.to(device) for t in result)
    else:
        result = _wait(component(*tensors))
    return result


@dataclass
class GrootAppConfig:
    """
    Runtime dependencies extracted once from the checkpoint policy.

    Projection uses a [B, 1, S, S] key-padding mask. DiT derives its own
    text/image masks from the image-membership and backbone-validity inputs.
    """

    action_horizon: int
    action_dim: int
    seq_len: int
    num_deepstack: int
    image_token_index: int
    image_mask: torch.Tensor
    pad_token_id: int
    dtype: torch.dtype
    processor: Any
    collate_fn: Callable[[list[Any]], dict[str, Any]]
    modality_configs: dict[str, Any]
    language_key: str
    embodiment_tag: Any
    llm_embedding_weight: torch.Tensor
    vit_grid_thw: torch.Tensor
    action_layout: ActionLayout
    num_inference_timesteps: int = 4
    component_input_specs: dict[str, InputSpec] = field(default_factory=dict)
    component_output_names: dict[str, list[str]] = field(default_factory=dict)

    @staticmethod
    def from_policy(policy: Gr00tPolicy) -> GrootAppConfig:
        action_head_cfg = policy.model.action_head.config
        processor: Any = policy.processor
        modality_configs = policy.get_modality_config()
        tag = policy.embodiment_tag.value
        action_keys = tuple(modality_configs["action"].modality_keys)
        action_layout = ActionLayout(
            action_keys,
            tuple(
                int(
                    processor.state_action_processor.norm_params[tag]["action"][key][
                        "dim"
                    ]
                )
                for key in action_keys
            ),
            int(action_head_cfg.action_horizon),
        )
        metadata = build_reference_metadata(policy)
        seq_len, hidden_size = metadata.seq_len, metadata.hidden_size
        dit_config = policy.model.action_head.model.config
        num_kv_pairs = len(policy.model.action_head.model.transformer_blocks) // 2
        if num_kv_pairs != 16:
            raise ValueError(
                "The N1.7 deployment requires sixteen cross-attention K/V pairs."
            )
        kv_names = [
            f"vlm_proj_{kind}_{i}"
            for kind in ("keys", "values")
            for i in range(num_kv_pairs)
        ]
        return GrootAppConfig(
            action_horizon=int(action_head_cfg.action_horizon),
            action_dim=int(action_head_cfg.max_action_dim),
            num_inference_timesteps=int(action_head_cfg.num_inference_timesteps),
            seq_len=seq_len,
            num_deepstack=metadata.num_deepstack,
            image_token_index=int(policy.model.backbone.model.config.image_token_id),
            image_mask=metadata.image_mask.to(torch.float32),
            # Qwen3-VL leaves pad_token_id unset; the pad slots are attention-masked,
            # so any in-vocab id works. Use 0.
            pad_token_id=0,
            dtype=torch.float32,
            processor=policy.processor,
            collate_fn=policy.collate_fn,
            modality_configs=modality_configs,
            language_key=policy.language_key,
            embodiment_tag=policy.embodiment_tag,
            llm_embedding_weight=(
                policy.model.backbone.model.language_model.get_input_embeddings()
                .weight.detach()
                .clone()
            ),
            vit_grid_thw=metadata.grid_thw.clone(),
            action_layout=action_layout,
            component_input_specs={
                "vit": get_vit_input_spec(metadata.num_patches, metadata.patch_dim),
                "llm": get_llm_input_spec(
                    seq_len,
                    hidden_size,
                    metadata.num_deepstack,
                    metadata.num_vis_tokens,
                ),
                "vlm_proj": get_vlm_proj_input_spec(seq_len, hidden_size),
                "dit": get_dit_input_spec(
                    seq_len,
                    action_head_cfg.max_state_dim,
                    action_layout.action_horizon,
                    action_head_cfg.max_action_dim,
                    dit_config.num_attention_heads,
                    dit_config.attention_head_dim,
                    num_kv_pairs,
                    action_head_cfg.num_inference_timesteps,
                ),
            },
            component_output_names={
                "vit": ["image_embeds", "deepstack_features"],
                "llm": ["vlm_embeds"],
                "vlm_proj": kv_names,
                "dit": ["actions_out"],
            },
        )


class GrootApp:
    """
    Orchestrates GrootCollection components to reproduce Gr00T N1.7 policy
    inference: ViT -> LLM (deepstack) -> VLMProjection -> DiT (4-step Euler),
    with host-side padding of the variable-length instruction to the baked
    seq_len and host-side mask derivation (visual_pos / image / backbone).
    """

    def __init__(
        self,
        config: GrootAppConfig,
        vit: ExecutableModelProtocol | AsyncOnDeviceModel,
        llm: ExecutableModelProtocol | AsyncOnDeviceModel,
        vlm_proj: ExecutableModelProtocol | AsyncOnDeviceModel,
        dit: ExecutableModelProtocol | AsyncOnDeviceModel,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.vit = vit
        self.llm = llm
        self.vlm_proj = vlm_proj
        self.dit = dit
        self.device = device
        self._noise_generator = torch.Generator(device=device).manual_seed(42)
        self._attention_mask_cache: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None

        for name in Model.component_classes:
            component = getattr(self, name)
            if not isinstance(component, (OnDeviceModel, AsyncOnDeviceModel)):
                continue
            expected = config.component_input_specs[name]
            actual = component.get_input_spec()
            if list(actual) != list(expected) or any(
                actual[key].shape != expected[key].shape
                or actual[key].dtype != expected[key].dtype
                for key in expected
            ):
                raise ValueError(
                    f"Incompatible {name} model inputs; a compatible N1.7 export is required. Expected {expected}, got {actual}."
                )
            async_model = (
                component.async_model
                if isinstance(component, OnDeviceModel)
                else component
            )
            output_names = config.component_output_names[name]
            if async_model.output_names and async_model.output_names != output_names:
                raise ValueError(
                    f"Incompatible {name} output order: expected {output_names}, got {async_model.output_names}."
                )
            async_model.output_names = list(output_names)

    @property
    def uses_ondevice_model(self) -> bool:
        return any(
            isinstance(component, (OnDeviceModel, AsyncOnDeviceModel))
            for component in (self.vit, self.llm, self.vlm_proj, self.dit)
        )

    def _preprocess(self, step: VLAStepData) -> dict[str, Any]:
        """Process one inference step, right-pad tokens, and sample initial noise."""
        modality_configs = self.config.modality_configs
        state_keys = list(modality_configs["state"].modality_keys)
        inference_step = VLAStepData(
            images={
                key: list(
                    np.stack([np.asarray(frame) for frame in step.images[key]]).astype(
                        np.uint8
                    )
                )
                for key in modality_configs["video"].modality_keys
            },
            states={
                key: np.array(step.states[key], dtype=np.float32, copy=True)
                for key in state_keys
            },
            actions={},
            text=step.text,
            embodiment=self.config.embodiment_tag,
        )
        batched_states = {
            key: value[None].copy() for key, value in inference_step.states.items()
        }
        processed = self.config.processor(
            [{"type": "episode_step", "content": inference_step}]
        )
        inner = self.config.collate_fn([processed])["inputs"]

        if "image_grid_thw" not in inner:
            raise KeyError("Processor output is missing required 'image_grid_thw'.")
        assert_grid_thw_matches(self.config.vit_grid_thw, inner["image_grid_thw"])

        dev, dt = self.device, self.config.dtype
        pixel_values = inner["pixel_values"].to(dev, dtype=dt)
        input_ids = inner["input_ids"].to(dev)
        attention_mask = inner["attention_mask"].to(dev)
        state = inner["state"].to(dev, dtype=dt)
        if input_ids.shape[0] != 1 or state.shape[0] != 1:
            raise ValueError("GR00T N1.7 supports batch-size-one inference.")

        cur_len = input_ids.shape[1]
        if cur_len > self.config.seq_len:
            raise ValueError(
                f"Instruction token length {cur_len} exceeds the baked seq_len "
                f"{self.config.seq_len}; a fresh export is required."
            )
        # Keep the static deployment layout. The validity mask excludes these
        # padding keys in the LLM, every VLM projection layer, and the DiT.
        pad = self.config.seq_len - cur_len
        if pad > 0:
            pad_ids = torch.full(
                (1, pad), self.config.pad_token_id, dtype=input_ids.dtype, device=dev
            )
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros((1, pad), dtype=attention_mask.dtype, device=dev),
                ],
                dim=1,
            )

        actions = torch.randn(
            (1, self.config.action_horizon, self.config.action_dim),
            dtype=dt,
            device=dev,
            generator=self._noise_generator,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state": state,
            "actions": actions,
            "batched_states": batched_states,
        }

    def _postprocess(
        self, actions_out: torch.Tensor, batched_states: dict
    ) -> dict[str, Any]:
        """Decode normalized actions back to physical units (matches the policy)."""
        unnormalized = self.config.processor.decode_action(
            actions_out.detach().cpu().numpy(),
            self.config.embodiment_tag,
            batched_states,
        )
        return {key: value.astype(np.float32) for key, value in unnormalized.items()}

    def encode_vlm(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_pos_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run ViT -> LLM -> VLMProjection to produce the DiT cross-attention K/V pairs."""
        image_embeds, deepstack = _call_component(
            self.vit,
            pixel_values,
            batch_axes=(None,),
            input_spec=self.config.component_input_specs.get("vit"),
        )
        deepstack_list = [deepstack[i] for i in range(self.config.num_deepstack)]

        cache = self._attention_mask_cache
        cache_hit = (
            cache is not None
            and cache[0].device == attention_mask.device
            and cache[0].dtype == attention_mask.dtype
            and cache[1].device == input_ids.device
            and cache[1].dtype == self.config.llm_embedding_weight.dtype
            and torch.equal(cache[0], attention_mask)
        )
        input_embeds, llm_attention_mask = preprocess_llm(
            input_ids,
            image_embeds,
            attention_mask,
            self.config.llm_embedding_weight.to(input_ids.device),
            self.config.image_token_index,
            llm_attention_mask=cache[1] if cache_hit and cache is not None else None,
        )
        if visual_pos_masks is None:
            visual_pos_masks = input_ids == self.config.image_token_index
        visual_pos_masks = visual_pos_masks.to(input_embeds.dtype)
        vlm_embeds = _call_component(
            self.llm,
            input_embeds,
            llm_attention_mask,
            visual_pos_masks,
            *deepstack_list,
            batch_axes=(0, 0, 0, *([None] * len(deepstack_list))),
            input_spec=self.config.component_input_specs.get("llm"),
        )
        if not isinstance(vlm_embeds, torch.Tensor):
            raise TypeError("The LLM component must return one vlm_embeds tensor.")

        if cache_hit and cache is not None:
            vlm_attention_mask = cache[2]
        else:
            vlm_attention_mask = preprocess_vlm_proj(attention_mask)
            self._attention_mask_cache = (
                attention_mask.detach().clone(),
                llm_attention_mask,
                vlm_attention_mask,
            )
        vlm_proj_out = _call_component(
            self.vlm_proj,
            vlm_embeds,
            vlm_attention_mask,
            batch_axes=(0, 0),
            input_spec=self.config.component_input_specs.get("vlm_proj"),
        )
        assert isinstance(vlm_proj_out, (tuple, list))
        return tuple(vlm_proj_out)

    def denoise_steps(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        image_mask: torch.Tensor,
        backbone_mask: torch.Tensor,
        vlm_proj_kv: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Run the DiT denoising over the VLM cross-attention K/V pairs."""
        timestep_indices = (
            torch.arange(
                self.config.num_inference_timesteps,
                dtype=torch.int32,
                device=state.device,
            )
            .unsqueeze(0)
            .expand(state.shape[0], -1)
            .contiguous()
        )
        actions_out = _call_component(
            self.dit,
            state,
            actions,
            image_mask,
            backbone_mask,
            *vlm_proj_kv,
            timestep_indices,
            batch_axes=(0,) * (5 + len(vlm_proj_kv)),
            input_spec=self.config.component_input_specs.get("dit"),
        )
        assert isinstance(actions_out, torch.Tensor)
        return actions_out

    def predict_action_chunk(self, step: VLAStepData) -> dict[str, np.ndarray]:
        """Full pipeline: preprocess -> VLM encode -> DiT denoise -> decode."""
        inputs = self._preprocess(step)
        image_mask = self.config.image_mask.to(inputs["state"])
        vlm_proj_kv = self.encode_vlm(
            inputs["pixel_values"],
            inputs["input_ids"],
            inputs["attention_mask"],
            visual_pos_masks=image_mask,
        )
        backbone_mask = (inputs["attention_mask"] == 1).to(inputs["state"].dtype)
        actions_out = self.denoise_steps(
            inputs["state"],
            inputs["actions"],
            image_mask,
            backbone_mask,
            vlm_proj_kv,
        )
        return self._postprocess(actions_out, inputs["batched_states"])

    def predict_action_tensor(self, step: VLAStepData) -> torch.Tensor:
        """Return the decoded physical actions in configured order as [1, H, D]."""
        return torch.from_numpy(
            concatenate_action_modalities(
                self.predict_action_chunk(step),
                self.config.action_layout,
                has_batch_axis=True,
            )
        )

    @classmethod
    def from_components(
        cls,
        models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel],
    ) -> GrootApp:
        if len(models) != len(Model.component_classes):
            raise ValueError("Expected four components in vit,llm,vlm_proj,dit order.")
        context = Model.get_eval_context()
        vit, llm, vlm_proj, dit = models
        return cls(
            context.app_config, vit, llm, vlm_proj, dit, device=context.host_device
        )

    def run_model_for_eval(
        self, model_input: Any, model_batch_size: int
    ) -> CollectionModelEvalGenerator:
        """Run one observation; both generator paths return the completed tensor."""
        if model_batch_size != 1:
            raise ValueError("GR00T N1.7 evaluation requires model batch size 1.")
        if isinstance(model_input, GeneratorType):
            fields = []
            for chunks in model_input:
                samples = tuple(chunks)
                if len(samples) != 1:
                    raise ValueError(
                        "GR00T N1.7 on-device evaluation requires samples_per_job=1."
                    )
                fields.append(samples[0])
            model_input = tuple(fields)
        step = step_from_tensors(
            model_input, self.config.modality_configs, self.config.embodiment_tag
        )
        result = (self.predict_action_tensor(step),)
        yield result
        return result


def build_app(
    policy: Gr00tPolicy,
    eval_mode: EvalMode,
    device: hub.Device | None = None,
    hub_model_id: str | None = None,
    host_device: str = "cpu",
) -> GrootApp:
    """
    Build a GrootApp with either host PyTorch components (EvalMode.FP) or
    on-device hub.InferenceJob components (EvalMode.ON_DEVICE).
    """
    config = GrootAppConfig.from_policy(policy)

    if eval_mode == EvalMode.FP:
        collection = Model.from_policy(policy, app_config=config)
        return GrootApp(
            config=config,
            vit=collection.components["vit"],
            llm=collection.components["llm"],
            vlm_proj=collection.components["vlm_proj"],
            dit=collection.components["dit"],
            device=host_device,
        )

    if device is None:
        raise ValueError("device must be provided for ON_DEVICE eval mode")
    if hub_model_id is None:
        raise ValueError("hub_model_id must be provided for ON_DEVICE eval mode")
    hub_ids = hub_model_id.split(",")
    component_classes = Model.component_classes
    if len(hub_ids) != len(component_classes):
        raise ValueError(
            f"Expected {len(component_classes)} comma-separated hub model IDs "
            f"({', '.join(component_classes)}), got {len(hub_ids)}"
        )
    on_device_components = {}
    for name, hub_id in zip(component_classes, hub_ids, strict=True):
        on_device_components[name] = OnDeviceModel(
            model=hub.get_model(hub_id),
            input_names=[],
            device=device,
        )
    return GrootApp(
        config=config,
        vit=on_device_components["vit"],
        llm=on_device_components["llm"],
        vlm_proj=on_device_components["vlm_proj"],
        dit=on_device_components["dit"],
        device=host_device,
    )

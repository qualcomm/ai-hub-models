# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from qai_hub.client import Device
from torch import nn

from qai_hub_models import SampleInputsType, TargetRuntime
from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.models.grootn17.constants import (
    DEFAULT_EMBODIMENT_TAG,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NUM_DEEPSTACK,
    VLM_SEQ_LEN,
)
from qai_hub_models.models.grootn17.evaluator import LeRobotEvaluator
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.data.types import (
    MessageType,
    VLAStepData,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
    Gr00tN1d7Processor,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.policy.gr00t_policy import (
    Gr00tPolicy,
)
from qai_hub_models.models.grootn17.model_adaptations import (
    ActionHeadDiTOpt,
    Qwen3VLTextBackboneOpt,
    Qwen3VLVisionTransformerOpt,
    SelfAttentionTransformerOpt,
    build_causal_mask,
)
from qai_hub_models.models.grootn17.utils import (
    get_dit_input_spec,
    get_llm_input_spec,
    get_vit_input_spec,
    get_vlm_proj_input_spec,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.checkpoint import CheckpointSpec, FromPretrainedMixin
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, make_torch_inputs
from qai_hub_models.utils.qai_hub_helpers import (
    ensure_hexagon_version,
    export_torch_to_onnx_zip,
)

DEFAULT_CHECKPOINT_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "model_gr00t_n1d7_droid.tar.gz"
)

DEFAULT_DATASET_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "droid_sample.tar.gz"
)

DEFAULT_COSMOS_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "cosmos_reason2_2b_assets.tar.gz"
)


@lru_cache(maxsize=1)
def get_cosmos_path() -> str:
    """Fetch and extract the Cosmos assets on first use."""
    return str(DEFAULT_COSMOS_ASSET.fetch(extract=True))


@lru_cache(maxsize=1)
def load_checkpoint(
    checkpoint: str,
    embodiment_tag: str,
    device: str = "cpu",
) -> Gr00tPolicy:
    """Load and cache a Gr00tPolicy from a model path or HF hub ID."""
    if checkpoint == "DEFAULT":
        checkpoint = str(DEFAULT_CHECKPOINT_ASSET.fetch(extract=True))
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag, model_path=checkpoint, device=device
    )
    policy.model.eval()
    policy.model = policy.model.float()
    policy.__dict__["_qaihm_checkpoint"] = str(Path(checkpoint).resolve())
    return policy


def _make_groot_export_inputs(input_spec: InputSpec) -> list[torch.Tensor]:
    """Use valid timestep indices and an all-valid additive projection mask."""
    inputs = make_torch_inputs(input_spec)
    if "timestep_indices" in input_spec:
        shape = input_spec["timestep_indices"].shape
        inputs[list(input_spec).index("timestep_indices")] = (
            torch.arange(shape[1], dtype=torch.int32).unsqueeze(0).expand(shape)
        )
    if "vlm_attention_mask" in input_spec:
        index = list(input_spec).index("vlm_attention_mask")
        inputs[index] = torch.zeros_like(inputs[index])
    return inputs


class LoadGrootMixin(FromPretrainedMixin):
    """Shared checkpoint loading, sample inputs, and export for Groot components."""

    @classmethod
    def torch_from_pretrained(  # type: ignore[override, unused-ignore]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
    ) -> Gr00tPolicy:
        return load_checkpoint(
            str(checkpoint),
            embodiment_tag=embodiment_tag,
            device=str(host_device),
        )

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        return ensure_hexagon_version(
            min_version=73,
            target_runtime=target_runtime,
            device=device,
            model_name="GR00T-N1.7",
        )

    def get_input_spec(self) -> InputSpec:
        raise NotImplementedError

    def get_output_spec(self) -> OutputSpec:
        raise NotImplementedError

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, *args: Any, **kwargs: Any
    ) -> SampleInputsType:
        """Package valid export tensors without changing their order or values."""
        input_spec = input_spec or self.get_input_spec()
        return {
            name: [tensor.numpy()]
            for name, tensor in zip(
                input_spec, _make_groot_export_inputs(input_spec), strict=True
            )
        }

    def _serialize_onnx(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None,
        output_names: list[str] | None = None,
    ) -> str:
        """Export the current model and return the actual packaged artifact path."""
        path = Path(output_dir) / f"{self.__class__.__name__}.onnx"
        assert input_spec is not None
        return export_torch_to_onnx_zip(
            self.to("cpu"),  # type: ignore[attr-defined]
            str(path),
            _make_groot_export_inputs(input_spec),
            input_names=list(input_spec.keys()),
            skip_zip=False,
            output_names=output_names,
        )

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Serialize Groot components as ONNX instead of PT2.

        Keep the collection export pipeline on the recipe's ONNX path,
        including external weights and valid component sample inputs,
        rather than BaseModel's default PT2 serialization.
        """
        return Path(self._serialize_onnx(output_dir, input_spec))

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str:
        return self._serialize_onnx(output_path, input_spec, output_names)


# Reference metadata
#
# N1.7's mRoPE positions depend on the image grid and chat-template layout,
# The processor and Qwen3-VL's native get_rope_index supply the layout without
# running the vision or text model.


@dataclass(frozen=True)
class ReferenceMetadata:
    """Fixed deployment layout; no reference images or model features are retained."""

    grid_thw: torch.Tensor
    position_ids: torch.Tensor
    image_mask: torch.Tensor
    num_patches: int
    patch_dim: int
    seq_len: int
    hidden_size: int
    num_deepstack: int
    num_vis_tokens: int


def _build_reference_step(policy: Gr00tPolicy, seed: int = 0) -> VLAStepData:
    """Build the canonical synthetic input without dataset access or video decoding."""
    modality_configs = policy.get_modality_config()
    state_params = cast(
        Gr00tN1d7Processor, policy.processor
    ).state_action_processor.norm_params[policy.embodiment_tag.value]["state"]
    rng = np.random.default_rng(seed)
    return VLAStepData(
        images={
            key: list(rng.integers(0, 255, size=(1, 180, 320, 3), dtype=np.uint8))
            for key in modality_configs["video"].modality_keys
        },
        states={
            key: rng.standard_normal((1, int(state_params[key]["dim"]))).astype(
                np.float32
            )
            for key in modality_configs["state"].modality_keys
        },
        actions={},
        text="",
        embodiment=policy.embodiment_tag,
    )


def _require_reference_tensor(value: object, name: str) -> torch.Tensor:
    """Report missing or malformed processor metadata consistently as ValueError."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(  # noqa: TRY004 -- Invalid processor output, not a caller argument.
            f"Invalid reference metadata: {name} must be a tensor."
        )
    return value


@lru_cache(maxsize=1)
def build_reference_metadata(policy: Gr00tPolicy, seed: int = 0) -> ReferenceMetadata:
    """Derive and validate the export layout without model inference or hooks.

    Only successful results enter the policy/seed-keyed cache. The processor
    and native position helper retain their own exception behavior.
    """
    step = _build_reference_step(policy, seed)
    processed = policy.processor(
        [{"type": MessageType.EPISODE_STEP.value, "content": step}]
    )
    inputs = policy.collate_fn([processed])["inputs"]
    pixels = _require_reference_tensor(inputs.get("pixel_values"), "pixel_values")
    grid = _require_reference_tensor(inputs.get("image_grid_thw"), "image_grid_thw")
    input_ids = _require_reference_tensor(inputs.get("input_ids"), "input_ids")
    mask = _require_reference_tensor(inputs.get("attention_mask"), "attention_mask")
    integral_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    qwen = policy.model.backbone.model.model
    vision_model = qwen.visual
    text_config = qwen.language_model.config
    vision_config = vision_model.config

    patch_dim = (
        vision_config.in_channels
        * vision_config.temporal_patch_size
        * vision_config.patch_size**2
    )
    if (
        pixels.ndim != 2
        or pixels.shape[0] == 0
        or pixels.shape[1] != patch_dim
        or not pixels.is_floating_point()
    ):
        raise ValueError(
            f"Invalid reference metadata: pixel_values must be floating-point [P, {patch_dim}] with P > 0; got {tuple(pixels.shape)} ({pixels.dtype})."
        )
    if (
        grid.ndim != 2
        or grid.shape[0] == 0
        or grid.shape[1] != 3
        or grid.dtype not in integral_dtypes
        or torch.any(grid <= 0)
    ):
        raise ValueError(
            "Invalid reference metadata: image_grid_thw must be a positive integral [N, 3] tensor with N > 0."
        )
    merge_size = vision_model.spatial_merge_size
    if merge_size <= 0 or torch.any(grid[:, 1:] % merge_size != 0):
        raise ValueError(
            "Invalid reference metadata: image_grid_thw spatial dimensions must be divisible by the configured spatial_merge_size."
        )
    num_patches = int(grid.prod(dim=1).sum())
    if num_patches != pixels.shape[0]:
        raise ValueError(
            f"Invalid reference metadata: image_grid_thw describes {num_patches} patches, but pixel_values contains {pixels.shape[0]}."
        )
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape[1] == 0
        or input_ids.dtype not in integral_dtypes
        or torch.any(input_ids < 0)
    ):
        raise ValueError(
            "Invalid reference metadata: input_ids must be nonnegative integral [1, S] with S > 0."
        )
    if mask.shape != input_ids.shape or torch.any((mask != 0) & (mask != 1)):
        raise ValueError(
            "Invalid reference metadata: attention_mask must be binary with the same shape as input_ids."
        )
    seq_len = int(input_ids.shape[1])
    if seq_len > VLM_SEQ_LEN:
        raise ValueError(
            f"Invalid reference metadata: reference length {seq_len} exceeds the deployment sequence length {VLM_SEQ_LEN}."
        )
    if torch.any(mask != 1):
        raise ValueError("Invalid reference metadata: reference must be unpadded.")
    hidden_size = int(text_config.hidden_size)
    if hidden_size <= 0 or vision_config.out_hidden_size != hidden_size:
        raise ValueError(
            "Invalid reference metadata: vision output width must match the positive LLM hidden size."
        )
    num_deepstack = len(vision_model.deepstack_visual_indexes)
    if num_deepstack != NUM_DEEPSTACK:
        raise ValueError(
            f"Invalid reference metadata: expected {NUM_DEEPSTACK} DeepStack stages, got {num_deepstack}."
        )
    num_vis_tokens = num_patches // merge_size**2
    image_tokens = (input_ids == qwen.config.image_token_id) & (mask == 1)
    if int(image_tokens.sum()) != num_vis_tokens:
        raise ValueError(
            f"Invalid reference metadata: image_grid_thw requires {num_vis_tokens} visual tokens, but input_ids contains {int(image_tokens.sum())} valid image tokens."
        )
    positions, _ = qwen.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=grid,
        attention_mask=mask,
    )
    positions = _require_reference_tensor(positions, "position_ids")
    if (
        positions.shape != (3, 1, seq_len)
        or positions.dtype not in integral_dtypes
        or torch.any(positions < 0)
    ):
        raise ValueError(
            f"Invalid reference metadata: position_ids must be nonnegative integral mRoPE positions with shape (3, 1, {seq_len})."
        )
    # Extend the native image-first mRoPE layout by one per axis to full capacity;
    # runtime masks exclude unused text slots.
    if seq_len < VLM_SEQ_LEN:
        offsets = torch.arange(
            1, VLM_SEQ_LEN - seq_len + 1, dtype=positions.dtype, device=positions.device
        )
        positions = torch.cat([positions, positions[..., -1:] + offsets], dim=-1)
    return ReferenceMetadata(
        grid_thw=grid.detach().cpu().clone(),
        position_ids=positions.detach().cpu().clone(),
        image_mask=torch.nn.functional.pad(
            image_tokens, (0, VLM_SEQ_LEN - seq_len), value=False
        )
        .detach()
        .cpu(),
        num_patches=num_patches,
        patch_dim=patch_dim,
        seq_len=VLM_SEQ_LEN,
        hidden_size=hidden_size,
        num_deepstack=num_deepstack,
        num_vis_tokens=num_vis_tokens,
    )


class GrootViT(LoadGrootMixin, BaseModel):
    """
    Qwen3-VL vision encoder for Groot N1.7.
    Accepts patchified pixel_values [num_patches, patch_dim] and returns
    (image_embeds, deepstack_features).

    Preprocessing is handled upstream; grid_thw is fixed at construction.
    """

    model: Qwen3VLVisionTransformerOpt

    def __init__(self, policy: Gr00tPolicy, metadata: ReferenceMetadata) -> None:
        super().__init__()

        vision_model = policy.model.backbone.model.model.visual  # Qwen3VLVisionModel
        self.model = Qwen3VLVisionTransformerOpt(vision_model, metadata.grid_thw)
        self.num_patches = metadata.num_patches
        self.patch_dim = metadata.patch_dim
        self.to(policy.model.device)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pixel_values.shape[0] != self.num_patches:
            raise ValueError(
                f"pixel_values has {pixel_values.shape[0]} patches but this graph was "
                f"baked for grid_thw={self.model._grid_thw.tolist()} "
                f"({self.num_patches} patches). A different image resolution or camera "
                "count requires a fresh export -- see assert_grid_thw_matches, which the "
                "host-side caller should also apply to its own computed grid_thw before "
                "ever reaching this graph."
            )
        return self.model(pixel_values)

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return get_vit_input_spec(self.num_patches, self.patch_dim)

    def get_output_spec(self) -> OutputSpec:
        return {"image_embeds": TensorSpec(), "deepstack_features": TensorSpec()}


class GrootLLMBackbone(LoadGrootMixin, BaseModel):
    """
    Qwen3-VL LLM backbone for Groot N1.7, truncated to select_layer.
    Accepts preprocessed inputs_embeds, a 4D attention mask, and
    deepstack visual features.
    Returns decoder features [B, seq_len, hidden_size] before final RMSNorm.
    """

    def __init__(self, policy: Gr00tPolicy, metadata: ReferenceMetadata) -> None:
        super().__init__()

        qwen_model = policy.model.backbone.model  # Qwen3VLForConditionalGeneration
        text_model = (
            qwen_model.model.language_model
        )  # Qwen3VLTextModel (already select_layer-truncated)
        self.model = Qwen3VLTextBackboneOpt(
            text_model.config,
            text_model.state_dict(),
            metadata.position_ids,
            select_layer=policy.model.backbone.select_layer,
        )
        self.seq_len = metadata.seq_len
        self.hidden_size = metadata.hidden_size
        self.num_deepstack = metadata.num_deepstack
        self.num_vis_tokens = metadata.num_vis_tokens
        self.to(policy.model.device)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        llm_attention_mask: torch.Tensor,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_0: torch.Tensor | None = None,
        deepstack_1: torch.Tensor | None = None,
        deepstack_2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs_embeds
            [B, seq_len, hidden_size] merged token and visual embeddings.
        llm_attention_mask
            Already-built 4D additive causal+padding mask,
            `[B, 1, seq_len, seq_len]` -- see `preprocess_llm`.
        visual_pos_masks
            Optional [B, seq_len] float mask identifying visual token positions.
        deepstack_0
            Optional [num_vis_tokens, hidden_size] features for the first layer.
        deepstack_1
            Optional [num_vis_tokens, hidden_size] features for the second layer.
        deepstack_2
            Optional [num_vis_tokens, hidden_size] features for the third layer.

        Returns
        -------
        torch.Tensor
            [B, seq_len, hidden_size] decoder features before final RMSNorm.
        """
        return self.model(
            inputs_embeds,
            llm_attention_mask,
            visual_pos_masks,
            deepstack_0,
            deepstack_1,
            deepstack_2,
        )

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return get_llm_input_spec(
            self.seq_len,
            self.hidden_size,
            self.num_deepstack,
            self.num_vis_tokens,
            batch_size,
        )

    def get_output_spec(self) -> OutputSpec:
        return {"vlm_embeds": TensorSpec()}


class GrootVLMProjection(LoadGrootMixin, BaseModel):
    """
    VLM projection head for Groot N1.7.
    Accepts backbone_features [B, seq_len, backbone_embedding_dim] and a
    float32 vlm_attention_mask [B, 1, seq_len, seq_len] (0 valid / -100 padding).
    Applies VLLN and VL self-attention to return 16 cross-attention key/value
    pairs [B, num_heads, seq_len, head_dim] for the DiT action head.
    """

    def __init__(self, policy: Gr00tPolicy, metadata: ReferenceMetadata) -> None:
        super().__init__()

        action_head = policy.model.action_head
        self.vlln = action_head.vlln
        self.vl_self_attention = (
            action_head.vl_self_attention
            if isinstance(action_head.vl_self_attention, nn.Identity)
            else SelfAttentionTransformerOpt(action_head.vl_self_attention)
        )
        self.dit = (
            action_head.model
        )  # AlternateVLDiT; read-only here (to_k/to_v/norm_k)

        self.seq_len = metadata.seq_len
        self.hidden_size = metadata.hidden_size
        self.to(policy.model.device)

    def forward(
        self, backbone_features: torch.Tensor, vlm_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x = self.vlln(backbone_features)
        if isinstance(self.vl_self_attention, nn.Identity):
            x = self.vl_self_attention(x)
        else:
            x = self.vl_self_attention(x, attention_mask=vlm_attention_mask)

        vlm_proj_keys, vlm_proj_values = [], []
        batch_size = x.shape[0]

        for idx, block in enumerate(self.dit.transformer_blocks):
            if idx % 2 == 1:
                continue  # self-attention block; no cross-attn K/V here

            key = block.attn1.to_k(x)
            value = block.attn1.to_v(x)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // block.attn1.heads

            key = key.view(batch_size, -1, block.attn1.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, block.attn1.heads, head_dim).transpose(
                1, 2
            )

            if block.attn1.norm_k is not None:
                key = block.attn1.norm_k(key)

            vlm_proj_keys.append(key)
            vlm_proj_values.append(value)

        return (*vlm_proj_keys, *vlm_proj_values)

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return get_vlm_proj_input_spec(self.seq_len, self.hidden_size, batch_size)

    def get_output_spec(self) -> OutputSpec:
        # N1.7's 32-block DiT has sixteen cross-attention K/V pairs.
        n_blocks = len(self.dit.transformer_blocks) // 2
        return {
            **{f"vlm_proj_keys_{i}": TensorSpec() for i in range(n_blocks)},
            **{f"vlm_proj_values_{i}": TensorSpec() for i in range(n_blocks)},
        }


class GrootDiT(LoadGrootMixin, BaseModel):
    """
    Diffusion Transformer (DiT) action head for Groot N1.7.
    Denoises actions in four Euler steps using robot state and VLM keys/values.

    Inputs:
      - state [B, 1, max_state_dim]
      - actions [B, action_horizon, action_dim] (initial noise)
      - image_mask, backbone_mask [B, seq_len]
      - vlm_proj_keys/values_0..15 [B, num_heads, seq_len, head_dim]
      - timestep_indices [B, 4], int32 indices into precomputed timesteps;
        required for compiled inference, defaults to [0, 1, 2, 3] in eager mode.

    Returns actions_out [B, action_horizon, action_dim].
    """

    model: ActionHeadDiTOpt

    def __init__(self, policy: Gr00tPolicy, metadata: ReferenceMetadata) -> None:
        super().__init__()

        self.config = policy.model.action_head.config
        embodiment_id = cast(
            Gr00tN1d7Processor, policy.processor
        ).embodiment_id_mapping[policy.embodiment_tag.value]
        action_head = ActionHeadDiTOpt(config=self.config, embodiment_id=embodiment_id)
        action_head.load_state_dict(policy.model.action_head.state_dict(), strict=False)
        action_head.collapse_embodiment_weights()
        action_head.prepare_self_attention_qkv_for_onnx_export()
        action_head.prepare_shared_weights_for_onnx_export()
        self.model = action_head

        self.seq_len = metadata.seq_len
        self.to(policy.model.device)

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        image_mask: torch.Tensor,
        backbone_mask: torch.Tensor,
        vlm_proj_keys_0: torch.Tensor,
        vlm_proj_keys_1: torch.Tensor,
        vlm_proj_keys_2: torch.Tensor,
        vlm_proj_keys_3: torch.Tensor,
        vlm_proj_keys_4: torch.Tensor,
        vlm_proj_keys_5: torch.Tensor,
        vlm_proj_keys_6: torch.Tensor,
        vlm_proj_keys_7: torch.Tensor,
        vlm_proj_keys_8: torch.Tensor,
        vlm_proj_keys_9: torch.Tensor,
        vlm_proj_keys_10: torch.Tensor,
        vlm_proj_keys_11: torch.Tensor,
        vlm_proj_keys_12: torch.Tensor,
        vlm_proj_keys_13: torch.Tensor,
        vlm_proj_keys_14: torch.Tensor,
        vlm_proj_keys_15: torch.Tensor,
        vlm_proj_values_0: torch.Tensor,
        vlm_proj_values_1: torch.Tensor,
        vlm_proj_values_2: torch.Tensor,
        vlm_proj_values_3: torch.Tensor,
        vlm_proj_values_4: torch.Tensor,
        vlm_proj_values_5: torch.Tensor,
        vlm_proj_values_6: torch.Tensor,
        vlm_proj_values_7: torch.Tensor,
        vlm_proj_values_8: torch.Tensor,
        vlm_proj_values_9: torch.Tensor,
        vlm_proj_values_10: torch.Tensor,
        vlm_proj_values_11: torch.Tensor,
        vlm_proj_values_12: torch.Tensor,
        vlm_proj_values_13: torch.Tensor,
        vlm_proj_values_14: torch.Tensor,
        vlm_proj_values_15: torch.Tensor,
        timestep_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        exporting = (
            torch.onnx.is_in_onnx_export()
            or torch.jit.is_tracing()
            or torch.compiler.is_compiling()
        )
        if timestep_indices is None:
            if exporting:
                raise ValueError("DiT export requires the timestep_indices input.")
            timestep_indices = (
                torch.arange(
                    self.model.num_inference_timesteps,
                    dtype=torch.int32,
                    device=state.device,
                )
                .unsqueeze(0)
                .expand(state.shape[0], -1)
            )
        if timestep_indices.dtype != torch.int32:
            raise TypeError("timestep_indices must have dtype int32.")
        if not exporting:
            num_steps = self.model.num_inference_timesteps
            if timestep_indices.shape != (state.shape[0], num_steps):
                raise ValueError(
                    f"timestep_indices must have shape ({state.shape[0]}, {num_steps})."
                )
            if torch.any((timestep_indices < 0) | (timestep_indices >= num_steps)):
                raise ValueError(
                    f"timestep_indices must contain indices from 0 to {num_steps - 1}, "
                    "not raw timestep buckets."
                )
        vlm_proj_keys = [
            vlm_proj_keys_0,
            vlm_proj_keys_1,
            vlm_proj_keys_2,
            vlm_proj_keys_3,
            vlm_proj_keys_4,
            vlm_proj_keys_5,
            vlm_proj_keys_6,
            vlm_proj_keys_7,
            vlm_proj_keys_8,
            vlm_proj_keys_9,
            vlm_proj_keys_10,
            vlm_proj_keys_11,
            vlm_proj_keys_12,
            vlm_proj_keys_13,
            vlm_proj_keys_14,
            vlm_proj_keys_15,
        ]
        vlm_proj_values = [
            vlm_proj_values_0,
            vlm_proj_values_1,
            vlm_proj_values_2,
            vlm_proj_values_3,
            vlm_proj_values_4,
            vlm_proj_values_5,
            vlm_proj_values_6,
            vlm_proj_values_7,
            vlm_proj_values_8,
            vlm_proj_values_9,
            vlm_proj_values_10,
            vlm_proj_values_11,
            vlm_proj_values_12,
            vlm_proj_values_13,
            vlm_proj_values_14,
            vlm_proj_values_15,
        ]
        return self.model(
            state=state,
            actions=actions,
            vlm_proj_keys=vlm_proj_keys,
            vlm_proj_values=vlm_proj_values,
            image_mask=image_mask,
            backbone_mask=backbone_mask,
            timestep_indices=timestep_indices,
        )

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        num_heads = self.model.model.config["num_attention_heads"]
        head_dim = self.model.model.config["attention_head_dim"]
        n_blocks = len(self.model.model.transformer_blocks) // 2

        return get_dit_input_spec(
            self.seq_len,
            self.config.max_state_dim,
            self.config.action_horizon,
            self.config.max_action_dim,
            num_heads,
            head_dim,
            n_blocks,
            self.model.num_inference_timesteps,
            batch_size,
        )

    def get_output_spec(self) -> OutputSpec:
        return {"actions_out": TensorSpec()}

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> torch.jit.ScriptModule:
        """Trace with valid runtime selectors instead of random integer inputs."""
        input_spec = input_spec or self.get_input_spec()
        self.eval()
        device = next(self.parameters()).device
        inputs = [tensor.to(device) for tensor in _make_groot_export_inputs(input_spec)]
        return torch.jit.trace(self, inputs, check_trace=check_trace)


# Host preprocessing
def preprocess_llm(
    input_ids: torch.Tensor,
    vit_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    image_token_index: int,
    *,
    llm_attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge vit_embeds into token embeddings and build the 4D causal+padding mask.

    Mirrors grootn15's `preprocess_llm`: mask construction happens host-side,
    not inside the exported LLM graph.
    """
    input_embeds = torch.nn.functional.embedding(input_ids, embedding_weight)
    B, N, C = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)

    selected = input_ids_flat == image_token_index
    input_embeds_flat[selected] = vit_embeds.reshape(-1, C)
    input_embeds = input_embeds_flat.reshape(B, N, C)

    if llm_attention_mask is None:
        llm_attention_mask = build_causal_mask(
            dtype=input_embeds.dtype,
            device=input_embeds.device,
            batch_size=B,
            seq_len=N,
            attention_mask=attention_mask,
        )
    return input_embeds, llm_attention_mask


def preprocess_vlm_proj(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build an FP32 additive key-padding mask [B, 1, S, S] on the host.

    Every query may attend to every valid key (0); padding keys receive -100.
    The head dimension is expanded by the existing Diffusers processor.
    """
    key_mask = (1.0 - attention_mask.to(torch.float32)) * -100.0
    return (
        key_mask[:, None, None, :]
        .expand(-1, 1, attention_mask.shape[1], -1)
        .contiguous()
    )


@dataclass(frozen=True)
class GrootEvalContext:
    """One resolved configuration shared by collection evaluation callbacks."""

    checkpoint: str
    host_device: str
    app_config: Any


class GrootCollection(WorkbenchModelCollection):
    """
    AIHub collection model for GR00T N1.7.

    Components (export-friendly reimplementation of the Qwen3-VL + DiT pipeline):
      - vit      : Qwen3-VL vision tower (baked grid_thw)
      - llm      : Qwen3-VL text model (baked mRoPE; deepstack injection)
      - vlm_proj : VLLN + VL self-attention (cross-attention K/V for the DiT)
      - dit      : flow-matching DiT action head (4-step Euler loop, unrolled)
    """

    # Shared component order for local and on-device app construction.
    component_classes: dict[str, type[LoadGrootMixin]] = {
        "vit": GrootViT,
        "llm": GrootLLMBackbone,
        "vlm_proj": GrootVLMProjection,
        "dit": GrootDiT,
    }
    _active_eval_context: GrootEvalContext | None = None

    def __init__(
        self,
        vit: GrootViT,
        llm: GrootLLMBackbone,
        vlm_proj: GrootVLMProjection,
        dit: GrootDiT,
        eval_context: GrootEvalContext | None = None,
    ) -> None:
        super().__init__({"vit": vit, "llm": llm, "vlm_proj": vlm_proj, "dit": dit})
        self.eval_context = eval_context

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        host_device: torch.device | str = torch.device("cpu"),
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        **kwargs: object,
    ) -> GrootCollection:
        cls._active_eval_context = None
        policy = load_checkpoint(
            checkpoint=str(checkpoint),
            embodiment_tag=embodiment_tag,
            device=str(host_device),
        )
        return cls.from_policy(policy)

    @classmethod
    def from_policy(
        cls,
        policy: Gr00tPolicy,
        app_config: Any | None = None,
    ) -> GrootCollection:
        """Build components from cached metadata, reusing app_config when supplied."""
        from qai_hub_models.models.grootn17.app import GrootAppConfig

        cls._active_eval_context = None
        config = (
            GrootAppConfig.from_policy(policy) if app_config is None else app_config
        )
        context = GrootEvalContext(
            checkpoint=getattr(
                policy,
                "_qaihm_checkpoint",
                getattr(policy.model.config, "_name_or_path", "<in-memory-policy>"),
            ),
            host_device=str(policy.model.device),
            app_config=config,
        )
        metadata = build_reference_metadata(policy)
        collection = cls(
            GrootViT(policy, metadata),
            GrootLLMBackbone(policy, metadata),
            GrootVLMProjection(policy, metadata),
            GrootDiT(policy, metadata),
            eval_context=context,
        )
        cls._active_eval_context = context
        return collection

    @classmethod
    def get_eval_context(cls) -> GrootEvalContext:
        if cls._active_eval_context is None:
            raise RuntimeError(
                "Construct GrootCollection before creating an evaluation app or dataset."
            )
        return cls._active_eval_context

    @classmethod
    def get_eval_dataset_classes(cls) -> list:
        from qai_hub_models.models.grootn17.dataset import GrootDataset

        return [GrootDataset]

    def get_evaluator(self) -> LeRobotEvaluator:
        if self.eval_context is None:
            raise RuntimeError(
                "Dataset evaluation requires a collection built from a policy."
            )
        return LeRobotEvaluator(self.eval_context.app_config.action_layout.dof_slices)

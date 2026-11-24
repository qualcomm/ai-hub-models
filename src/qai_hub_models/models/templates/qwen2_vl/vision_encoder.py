# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Vision encoder for Qwen2.5-VL (VEG - Vision Encoder Graph).

Qwen2VLVisionEncoder - Adapted model for on-device export with:
- Conv3d -> Conv2d patch embedding
- Split Q/K/V Conv2d attention (instead of fused QKV)
- Pre-computed RoPE and static attention masks
- All Linear -> Conv2d conversion

When called without explicit position/mask arguments, the forward method
uses pre-computed buffers from __init__, making it usable for both
FP demo inference and ONNX export.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel


class Qwen2VLVisionWrapper(torch.nn.Module):
    """Adapts a Qwen2.5-VL vision model to the generator's vision interface.

    The generator always calls ``vision_model(pixel_values, image_grid_thw,
    mask)`` and expects the image-embedding tensor back. Two kinds of vision
    model can be wrapped, and this class picks the right call convention based
    on what it is handed:

    - Raw HF visual tower (calibration / demo): dynamic, computes its rotary
      positions and attention masks internally from ``grid_thw``. We forward
      ``grid_thw`` straight through and return its tensor output.
    - ``Qwen2VLVisionEncoder`` VEG (evaluate): fixed input shape, runs an
      onnxruntime session, with positions/masks precomputed as cached buffers.
      It only consumes ``pixel_values``; ``grid_thw`` and ``mask`` are implied
      by the fixed shape and ignored.
    """

    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual
        self._is_veg = isinstance(visual, Qwen2VLVisionEncoder)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if pixel_values is None:
            return None
        if self._is_veg:
            # VEG has a fixed input shape; positions/masks are cached buffers,
            # so grid_thw and mask are not passed. The quantized VEG runs via
            # onnxruntime and returns a CPU tensor, so move the embeddings back
            # to the input device before the generator merges them.
            embeddings = self.visual(pixel_values)
            return embeddings.to(pixel_values.device)
        dtype = next(self.visual.parameters()).dtype
        pixel_values = pixel_values.to(dtype=dtype)
        return self.visual(pixel_values, grid_thw=image_grid_thw)


class Qwen2VLVisionEncoder(BaseModel):
    """
    Adapted vision encoder for on-device export (VEG).

    This class wraps the Qwen2.5-VL vision encoder with all adaptations
    required for QNN/HTP export:
    - Conv3d -> Conv2d for patch embedding
    - Split Q/K/V attention with Conv2d projections
    - All Linear -> Conv2d conversion
    - Pre-computed RoPE and static attention masks (passed as inputs)

    The forward() takes 5 inputs:
    - pixel_values: preprocessed image patches
    - position_ids_cos: pre-computed RoPE cosine values
    - position_ids_sin: pre-computed RoPE sine values
    - window_attention_mask: attention mask for windowed attention layers
    - full_attention_mask: attention mask for full attention layers

    Reference: Tutorial_for_Qwen2_5_VL_7B_IoT/example1/Example1A/veg.ipynb
    """

    def __init__(
        self,
        visual: Any,  # Qwen2_5_VisionTransformerPretrainedModel (HF dynamic attrs)
        grid_thw: torch.Tensor,
        in_channels: int = 3,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
    ) -> None:
        """
        Create the VEG by wrapping and adapting the HF vision model.

        Parameters
        ----------
        visual
            The adapted Qwen2_5_VisionTransformerPretrainedModel.
            Must already have attention and linear adaptations applied.
        grid_thw
            Grid dimensions tensor for the target image size.
            Shape: (num_images, 3) containing (temporal, height, width) patch counts.
        in_channels
            Number of input channels (3 for RGB).
        patch_size
            Spatial patch size (14 for Qwen2.5-VL).
        temporal_patch_size
            Temporal patch size (2 for Qwen2.5-VL).
        """
        super().__init__()
        from qai_hub_models.models.templates.qwen2_vl.vision_encoder_adaptations import (
            Conv2dInplaceConv3d,
        )

        self.patch_embed = visual.patch_embed
        # Replace Conv3d with Conv2d in patch embedding
        self.patch_embed.proj = Conv2dInplaceConv3d(
            visual.patch_embed.proj,
            in_channels=in_channels,
            temporal_patch_size=temporal_patch_size,
            patch_size=patch_size,
        )

        self.blocks = visual.blocks
        self.fullatt_block_indexes = visual.fullatt_block_indexes
        self.projector = visual.merger

        self.spatial_merge_size = visual.spatial_merge_size
        self.spatial_merge_unit = visual.spatial_merge_unit

        device = next(visual.parameters()).device

        # Compute total sequence length (number of patches before merging)
        self.seq_len = int(
            torch.sum(grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).item()
        )

        # Get window indices and cu_seqlens for windowed attention
        window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
        self.register_buffer("window_index", window_index)

        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=device,
            dtype=torch.int32,
        )

        # Pre-compute RoPE embeddings
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(
            self.seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(self.seq_len, -1)

        # Store pre-computed cos/sin as buffers for get_sample_inputs
        self.register_buffer("_pos_emb_cos", rotary_pos_emb.cos())
        self.register_buffer("_pos_emb_sin", rotary_pos_emb.sin())

        # Compute cu_seqlens for full and window attention masks
        cu_window_seqlens_unique = torch.unique_consecutive(cu_window_seqlens)
        self.register_buffer("cu_window_seqlens", cu_window_seqlens_unique)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        self.register_buffer("cu_seqlens", cu_seqlens)

        # Pre-compute window and full attention masks
        window_mask = torch.full(
            [1, self.seq_len, self.seq_len],
            -1000.0,
            device=device,
            dtype=torch.float32,
        )
        full_mask = torch.full(
            [1, self.seq_len, self.seq_len],
            -1000.0,
            device=device,
            dtype=torch.float32,
        )

        for layer_num, _ in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                for i in range(1, cu_seqlens.shape[0]):
                    full_mask[
                        ...,
                        cu_seqlens[i - 1] : cu_seqlens[i],
                        cu_seqlens[i - 1] : cu_seqlens[i],
                    ] = 0
            else:
                for i in range(1, cu_window_seqlens_unique.shape[0]):
                    window_mask[
                        ...,
                        cu_window_seqlens_unique[i - 1] : cu_window_seqlens_unique[i],
                        cu_window_seqlens_unique[i - 1] : cu_window_seqlens_unique[i],
                    ] = 0

        self.register_buffer("_window_attention_mask", window_mask)
        self.register_buffer("_full_attention_mask", full_mask)

        # Compute reverse indices for unscrambling window ordering
        reverse_indices = torch.argsort(window_index)
        self.register_buffer("reverse_indices", reverse_indices)

        # Store dimensions for input spec and image resizing
        self._in_channels = in_channels
        self._patch_size = patch_size
        self._temporal_patch_size = temporal_patch_size
        self._image_height = int(grid_thw[0, 1].item()) * patch_size
        self._image_width = int(grid_thw[0, 2].item()) * patch_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids_cos: torch.Tensor | None = None,
        position_ids_sin: torch.Tensor | None = None,
        window_attention_mask: torch.Tensor | None = None,
        full_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the adapted vision encoder.

        When called without explicit position/mask arguments (e.g. for FP demo
        inference), the pre-computed buffers from __init__ are used.

        Parameters
        ----------
        pixel_values
            Preprocessed image patches.
            Shape: (num_patches, channels * temporal_patch_size * patch_size * patch_size)
        position_ids_cos
            Pre-computed RoPE cosine values (optional, defaults to stored buffer).
            Shape: (seq_len, head_dim)
        position_ids_sin
            Pre-computed RoPE sine values (optional, defaults to stored buffer).
            Shape: (seq_len, head_dim)
        window_attention_mask
            Attention mask for windowed attention layers (optional, defaults to stored buffer).
            Shape: (1, seq_len, seq_len)
        full_attention_mask
            Attention mask for full attention layers (optional, defaults to stored buffer).
            Shape: (1, seq_len, seq_len)

        Returns
        -------
        torch.Tensor
            Vision embeddings after projection.
            Shape: (num_output_tokens, out_hidden_size)
        """
        if position_ids_cos is None:
            position_ids_cos = self._pos_emb_cos  # type: ignore[assignment]
        if position_ids_sin is None:
            position_ids_sin = self._pos_emb_sin  # type: ignore[assignment]
        if window_attention_mask is None:
            window_attention_mask = self._window_attention_mask  # type: ignore[assignment]
        if full_attention_mask is None:
            full_attention_mask = self._full_attention_mask  # type: ignore[assignment]
        hidden_states = self.patch_embed(pixel_values)

        # Apply window reordering
        hidden_states = hidden_states.reshape(
            self.seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        # Use index_select instead of advanced indexing to export as
        # Gather (supported by QNN) instead of GatherND (unsupported).
        hidden_states = torch.index_select(
            hidden_states,
            dim=0,
            index=self.window_index,  # type: ignore[arg-type]
        )
        hidden_states = hidden_states.reshape(self.seq_len, -1)

        # Process through transformer blocks
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=self.cu_seqlens,
                    rotary_pos_emb=(position_ids_cos, position_ids_sin),
                    attention_mask=full_attention_mask,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=self.cu_window_seqlens,
                    rotary_pos_emb=(position_ids_cos, position_ids_sin),
                    attention_mask=window_attention_mask,
                )

        # Project through merger
        hidden_states = self.projector(hidden_states)

        # Reverse window ordering to get tokens in original spatial order.
        # Use index_select instead of advanced indexing to export as
        # Gather (supported by QNN) instead of GatherND (unsupported).
        return torch.index_select(hidden_states, dim=0, index=self.reverse_indices)  # type: ignore[arg-type]

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: torch.device | None = None,
        image_height: int = 336,
        image_width: int = 504,
    ) -> Qwen2VLVisionEncoder:
        """
        Load and adapt vision encoder for on-device export.

        Parameters
        ----------
        checkpoint
            HuggingFace model name or local path.
        device
            Device to load the model on.
        image_height
            Target image height (must be divisible by patch_size * spatial_merge_size).
        image_width
            Target image width (must be divisible by patch_size * spatial_merge_size).

        Returns
        -------
        Qwen2VLVisionEncoder
            Adapted vision encoder ready for ONNX export.
        """
        from transformers import AutoConfig
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

        from qai_hub_models.models.templates.qwen2_vl.vision_encoder_adaptations import (
            replace_linears_with_convs,
            replace_visual_attention_with_adaptation,
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        vision_config = config.vision_config

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        spatial_merge_size = vision_config.spatial_merge_size

        stride = patch_size * spatial_merge_size
        if image_height % stride != 0 or image_width % stride != 0:
            raise ValueError(
                f"image_height ({image_height}) and image_width ({image_width}) "
                f"must be divisible by patch_size * spatial_merge_size "
                f"({patch_size} * {spatial_merge_size} = {stride})."
            )

        # Load full VLM and extract visual model
        full_model = (
            modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            ).to(device)  # type: ignore[arg-type, unused-ignore]
        )
        visual = copy.deepcopy(full_model.visual)

        # Clean up full model
        del full_model

        # Apply adaptations: attention split + Conv2d, then Linear -> Conv2d
        visual = replace_visual_attention_with_adaptation(visual)
        visual = replace_linears_with_convs(visual)

        # Compute grid_thw for the target image size
        h_patches = image_height // patch_size
        w_patches = image_width // patch_size
        # grid_thw stores patch counts after Conv3d.
        # For a single image (temporal_patch_size frames), Conv3d produces
        # 1 temporal patch, so temporal=1.
        grid_thw = torch.tensor(
            [[1, h_patches, w_patches]],
            dtype=torch.int64,
            device=device,
        )

        # Create VEG wrapper
        veg = cls(
            visual=visual,
            grid_thw=grid_thw,
            in_channels=in_channels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )
        veg.to(device)
        veg.eval()

        return veg

    def get_input_spec(
        self,
        num_images: int = 1,
    ) -> InputSpec:
        return self.get_static_input_spec(
            self._image_height,
            self._image_width,
            num_images,
            self._patch_size,
            self._temporal_patch_size,
            self._in_channels,
        )

    @staticmethod
    def get_static_input_spec(
        image_height: int = 336,
        image_width: int = 504,
        num_images: int = 1,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
    ) -> InputSpec:
        """
        Get input specification for the VEG.

        Returns spec for 5 inputs: pixel_values, position_ids_cos,
        position_ids_sin, window_attention_mask, full_attention_mask.

        The temporal grid dimension is 1 (single image = 1 temporal patch
        after Conv3d).
        """
        h_patches = image_height // patch_size
        w_patches = image_width // patch_size
        patch_dim = in_channels * temporal_patch_size * patch_size * patch_size

        # seq_len = number of patches after Conv3d.
        # For single images: 1 temporal patch * h_patches * w_patches.
        seq_len = h_patches * w_patches * num_images

        # RoPE embedding dimension = hidden_size / num_heads / 2
        # For Qwen2.5-VL-7B: 1280 / 16 / 2 = 40
        # We can't compute this without the config, so we use a placeholder.
        # The actual shape is set by get_sample_inputs from the loaded model.
        # For the input_spec, we use the standard Qwen2.5-VL vision RoPE dim.
        # hidden_size=1280, num_heads=16 -> head_dim=80 -> rope_dim=40
        rope_dim = 40  # Default for Qwen2.5-VL-7B

        return {
            "pixel_values": TensorSpec(shape=(seq_len, patch_dim), dtype="float32"),
            "position_ids_cos": TensorSpec(shape=(seq_len, rope_dim), dtype="float32"),
            "position_ids_sin": TensorSpec(shape=(seq_len, rope_dim), dtype="float32"),
            "window_attention_mask": TensorSpec(
                shape=(1, seq_len, seq_len), dtype="float32"
            ),
            "full_attention_mask": TensorSpec(
                shape=(1, seq_len, seq_len), dtype="float32"
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "vision_embeddings": TensorSpec(),
        }

    def get_sample_inputs(self) -> dict[str, torch.Tensor]:
        """
        Get sample inputs with pre-computed RoPE and attention masks.

        Uses the values computed during __init__ for correct RoPE and masks.
        """
        patch_dim = (
            self._in_channels
            * self._temporal_patch_size
            * self._patch_size
            * self._patch_size
        )
        num_patches = self.seq_len

        return {
            "pixel_values": torch.randn(num_patches, patch_dim, dtype=torch.float32),
            "position_ids_cos": self._pos_emb_cos.cpu().float(),  # type: ignore[dict-item]
            "position_ids_sin": self._pos_emb_sin.cpu().float(),  # type: ignore[dict-item]
            "window_attention_mask": self._window_attention_mask.cpu().float(),  # type: ignore[dict-item]
            "full_attention_mask": self._full_attention_mask.cpu().float(),  # type: ignore[dict-item]
        }

    def get_num_output_tokens(
        self,
        image_height: int,
        image_width: int,
    ) -> int:
        """
        Calculate the number of output tokens for a given image size.

        For a single image: 1 temporal patch after Conv3d, then spatial merging.
        """
        h_patches = image_height // self._patch_size
        w_patches = image_width // self._patch_size
        merged_h = h_patches // self.spatial_merge_size
        merged_w = w_patches // self.spatial_merge_size
        return merged_h * merged_w

    @classmethod
    def _configure_quant_sim(cls, quant_sim: QuantizationSimModel) -> None:
        """Apply VEG-specific mixed-precision overrides.

        The ``supergroup_pass_list`` in ``default_config_llama.json``
        handles norm internal activation pass-through.  RMSNorm weights
        (gamma) are classified as activations by AIMET (elementwise Mul),
        so they inherit ``activation_type="int16"`` automatically.

        This method sets only the remaining VEG-specific overrides.
        """
        import re

        from aimet_onnx.qc_quantize_op import QcQuantizeOp

        for op_name, qc_op in quant_sim.qc_quantize_op_dict.items():
            if not isinstance(qc_op, QcQuantizeOp):
                continue

            # v_proj Conv output → 8-bit symmetric activation
            if (
                re.search(r"v_proj.*conv_Conv", op_name)
                and op_name in quant_sim.activation_names
            ):
                qc_op.bitwidth = 8
                qc_op.use_symmetric_encodings = True

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import onnx
import torch
import torch.nn.functional as F

from qai_hub_models import Precision
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec


class Qwen3VLVisionEncoder(BaseModel):
    """
    Adapted vision encoder for Qwen3-VL on-device export (VEG).

    Returns a tuple of outputs:
    - image_embeddings: main pooler output (post-merger)
    - deepstack features: per-layer merged visual embeddings

    The forward() takes 5 inputs:
    - pixel_values: preprocessed image patches
    - position_ids_cos: pre-computed RoPE cosine values
    - position_ids_sin: pre-computed RoPE sine values
    - window_attention_mask: attention mask for all layers (Qwen3-VL uses full attention)
    - full_attention_mask: attention mask for all layers
    """

    def __init__(
        self,
        visual: Any,
        grid_thw: torch.Tensor,
        in_channels: int = 3,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
    ) -> None:
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
        self.merger = visual.merger
        self.pos_embed = visual.pos_embed

        # Deepstack components
        self.deepstack_visual_indexes = visual.deepstack_visual_indexes
        self.deepstack_merger_list = visual.deepstack_merger_list

        self.spatial_merge_size = visual.spatial_merge_size
        self.spatial_merge_unit = visual.spatial_merge_unit

        device = next(visual.parameters()).device

        # Compute total sequence length (number of patches before merging)
        self.seq_len = int(
            torch.sum(grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).item()
        )

        # Pre-compute position embeddings (learned + rotary)
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
        self.register_buffer("_pos_embeds", pos_embeds)

        # Pre-compute RoPE embeddings
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(self.seq_len, -1)

        self.register_buffer("_pos_emb_cos", rotary_pos_emb.cos())
        self.register_buffer("_pos_emb_sin", rotary_pos_emb.sin())

        # Compute cu_seqlens for attention masks
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        self.register_buffer("cu_seqlens", cu_seqlens)

        # Pre-compute full attention mask (Qwen3-VL uses full attention for all blocks)
        full_mask = torch.full(
            [1, self.seq_len, self.seq_len],
            -1000.0,
            device=device,
            dtype=torch.float32,
        )
        for i in range(1, cu_seqlens.shape[0]):
            full_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0

        self.register_buffer("_full_attention_mask", full_mask)
        # Qwen3-VL doesn't use windowed attention, but keep for interface compat
        self.register_buffer("_window_attention_mask", full_mask.clone())

        # Store dimensions for input spec
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
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass; returns (image_features, deepstack_0, ..., deepstack_N-1)."""
        if position_ids_cos is None:
            position_ids_cos = self._pos_emb_cos  # type: ignore[assignment]
        if position_ids_sin is None:
            position_ids_sin = self._pos_emb_sin  # type: ignore[assignment]
        if full_attention_mask is None:
            full_attention_mask = self._full_attention_mask  # type: ignore[assignment]

        hidden_states = self.patch_embed(pixel_values)

        # Add learned position embeddings
        hidden_states = hidden_states + self._pos_embeds

        # Process through transformer blocks
        deepstack_features: list[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                rotary_pos_emb=(position_ids_cos, position_ids_sin),
                attention_mask=full_attention_mask,
            )
            # Collect deepstack features at designated layers
            if layer_num in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_num)
                ds_feature = self.deepstack_merger_list[ds_idx](hidden_states)
                deepstack_features.append(ds_feature)

        # Final merger for main embeddings
        image_embeddings = self.merger(hidden_states)

        return (image_embeddings, *deepstack_features)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = "Qwen/Qwen3-VL-4B-Instruct",
        device: torch.device | None = None,
        image_height: int = 512,
        image_width: int = 512,
        precision: Precision = Precision.float,
    ) -> Qwen3VLVisionEncoder:
        """Load the vision encoder from a Qwen3-VL checkpoint."""
        from transformers import AutoConfig
        from transformers.models.qwen3_vl import modeling_qwen3_vl

        from qai_hub_models.models.templates.qwen2_vl.vision_encoder_adaptations import (
            replace_linears_with_convs,
        )
        from qai_hub_models.models.templates.qwen3_vl.vision_encoder_adaptations import (
            replace_visual_attention_with_adaptation,
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True)
        vis_config = config.vision_config
        patch_size = vis_config.patch_size
        temporal_patch_size = vis_config.temporal_patch_size

        # Compute grid dimensions
        h_patches = image_height // patch_size
        w_patches = image_width // patch_size
        grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.int64)

        # Load full VLM and extract visual component
        full_model = modeling_qwen3_vl.Qwen3VLForConditionalGeneration.from_pretrained(
            str(checkpoint),
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device)  # type: ignore[arg-type, unused-ignore]
        visual = copy.deepcopy(full_model.model.visual)
        del full_model

        # Apply adaptations for on-device export
        replace_visual_attention_with_adaptation(visual)
        replace_linears_with_convs(visual)

        # Create VEG instance
        instance = cls(
            visual=visual,
            grid_thw=grid_thw.to(device),
            in_channels=vis_config.in_channels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )
        instance.to(device)
        instance.eval()

        return instance

    def get_input_spec(
        self,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> InputSpec:
        if image_height is None:
            image_height = self._image_height
        if image_width is None:
            image_width = self._image_width
        pos_emb_cos = cast(torch.Tensor, self._pos_emb_cos)
        rope_dim = int(pos_emb_cos.shape[-1])
        return self.get_static_input_spec(
            image_height, image_width, self._patch_size, rope_dim=rope_dim
        )

    @staticmethod
    def get_static_input_spec(
        image_height: int = 512,
        image_width: int = 512,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        rope_dim: int = 32,
    ) -> InputSpec:
        """
        Get input spec for the vision encoder.

        Parameters
        ----------
        image_height
            Height of input image in pixels.
        image_width
            Width of input image in pixels.
        patch_size
            Spatial patch size.
        temporal_patch_size
            Temporal patch size.
        in_channels
            Number of input channels.
        rope_dim
            RoPE embedding dimension (head_dim // 2).

        Returns
        -------
        InputSpec
            Input specification dictionary.
        """
        h_patches = image_height // patch_size
        w_patches = image_width // patch_size
        seq_len = h_patches * w_patches  # temporal=1
        pixel_dim = in_channels * temporal_patch_size * patch_size * patch_size

        input_spec: InputSpec = {}
        input_spec["pixel_values"] = TensorSpec(
            shape=(seq_len, pixel_dim),
            dtype="float32",
        )
        input_spec["position_ids_cos"] = TensorSpec(
            shape=(seq_len, rope_dim),
            dtype="float32",
        )
        input_spec["position_ids_sin"] = TensorSpec(
            shape=(seq_len, rope_dim),
            dtype="float32",
        )
        input_spec["window_attention_mask"] = TensorSpec(
            shape=(1, seq_len, seq_len),
            dtype="float32",
        )
        input_spec["full_attention_mask"] = TensorSpec(
            shape=(1, seq_len, seq_len),
            dtype="float32",
        )
        return input_spec

    def get_output_names(self) -> list[str]:
        """Return output names including deepstack features.

        The main embedding output is named ``image_features`` (not
        ``image_embeddings``) so Genie's nsp-graph determineGraphType recognizes
        this graph as an IMAGE_ENCODER: it only tags a graph as an image encoder
        when an output name is prefixed with ``image_features`` /
        ``vision_embedding`` / ``cross_attention_states``. ``image_embeddings``
        matches none of those, so Genie falls through to input-token detection
        and throws on the encoder. ``image_features`` matches the proven
        qwen2.5-VL VEG output name.
        """
        names = ["image_features"]
        names.extend(
            f"deepstack_visual_embeds_{i}"
            for i in range(len(self.deepstack_visual_indexes))
        )
        return names

    def get_output_spec(self) -> OutputSpec:
        """Output spec derived from :meth:`get_output_names`."""
        return {name: TensorSpec() for name in self.get_output_names()}

    # ------------------------------------------------------------------
    # VEG Quantization Lifecycle (classmethods)
    # ------------------------------------------------------------------

    @classmethod
    def get_calibration_data(
        cls,
        num_samples: int,
        image_height: int = 512,
        image_width: int = 512,
    ) -> list:
        """Load real images from imagenette for VEG calibration."""
        from PIL import Image
        from transformers import AutoProcessor

        from qai_hub_models.datasets.imagenet import IMAGENETTE_ASSET

        IMAGENETTE_ASSET.fetch(extract=True)
        img_root = IMAGENETTE_ASSET.extracted_path

        train_dir = img_root / "train"
        image_paths: list[Path] = []
        for class_dir in sorted(train_dir.iterdir()):
            if class_dir.is_dir():
                image_paths.extend(
                    img_path
                    for img_path in sorted(class_dir.iterdir())
                    if img_path.suffix.lower() in (".jpeg", ".jpg", ".png")
                )
        if len(image_paths) < num_samples:
            raise RuntimeError(
                f"Imagenette has {len(image_paths)} images but "
                f"{num_samples} calibration samples requested."
            )

        image_paths = image_paths[:num_samples]
        hf_repo = getattr(cls, "_hf_repo_name", None)
        assert hf_repo is not None
        processor = AutoProcessor.from_pretrained(hf_repo)

        calibration_data = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((image_width, image_height))
            inputs = processor.image_processor(
                images=[img],
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].squeeze(0)
            calibration_data.append(pixel_values.numpy())

        return calibration_data

    @classmethod
    def export_to_onnx(
        cls,
        veg_model: Qwen3VLVisionEncoder,
        host_device: torch.device,
    ) -> onnx.ModelProto:
        """Export VEG instance to a float ONNX ModelProto."""
        temp_dir = tempfile.mkdtemp()
        onnx_path = os.path.join(temp_dir, "vision_encoder.onnx")

        input_spec = veg_model.get_input_spec()
        sample_inputs = {}
        for name, (shape, dtype) in input_spec.items():
            if dtype == "float32":
                sample_inputs[name] = torch.randn(*shape, device=host_device)
            else:
                sample_inputs[name] = torch.zeros(*shape, device=host_device)

        veg_model.to(host_device)
        torch.onnx.export(
            veg_model,
            tuple(sample_inputs.values()),
            onnx_path,
            input_names=list(input_spec.keys()),
            output_names=veg_model.get_output_names(),
            opset_version=17,
        )

        return onnx.load(onnx_path, load_external_data=True)

    @classmethod
    def save_onnx(
        cls,
        onnx_model: onnx.ModelProto,
        output_dir: str | os.PathLike | Path,
        filename: str = "vision_encoder.onnx",
    ) -> Path:
        """Save VEG ONNX with external data to output_dir."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        onnx_path = output_path / filename
        data_location = Path(filename).stem + ".data"
        onnx.save_model(
            onnx_model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_location,
        )
        return onnx_path

    @classmethod
    def create_quantsim_from_onnx(
        cls,
        onnx_model: onnx.ModelProto,
        veg_model: Qwen3VLVisionEncoder | None,
        host_device: torch.device,
    ) -> tuple[Any, dict]:
        """Create AIMET QuantSim from an already-exported ONNX model.

        When *veg_model* is None, fixed_inputs is empty (caller does not
        intend to calibrate -- e.g. just holding the rotated graph for export).
        """
        from aimet_onnx.common.defs import QuantScheme
        from aimet_onnx.quantsim import QuantizationSimModel

        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers.insert(0, "CUDAExecutionProvider")

        quant_sim = QuantizationSimModel(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type="int8",
            activation_type="int16",
            providers=providers,
        )

        fixed_inputs: dict = {}
        if veg_model is not None:
            fixed_inputs = {
                "position_ids_cos": veg_model._pos_emb_cos,
                "position_ids_sin": veg_model._pos_emb_sin,
                "window_attention_mask": veg_model._window_attention_mask,
                "full_attention_mask": veg_model._full_attention_mask,
            }

        return quant_sim, fixed_inputs

    @classmethod
    def calibrate(
        cls,
        quant_sim: Any,
        calibration_data: list,
        fixed_inputs: dict,
    ) -> None:
        """Calibrate the VEG QuantSim with real images."""

        def forward_pass(session: Any, args: Any = None) -> None:
            for pixel_values_np in calibration_data:
                feed_dict = {
                    "pixel_values": pixel_values_np,
                    "position_ids_cos": fixed_inputs["position_ids_cos"].cpu().numpy(),
                    "position_ids_sin": fixed_inputs["position_ids_sin"].cpu().numpy(),
                    "window_attention_mask": fixed_inputs["window_attention_mask"]
                    .cpu()
                    .numpy(),
                    "full_attention_mask": fixed_inputs["full_attention_mask"]
                    .cpu()
                    .numpy(),
                }
                session.run(None, feed_dict)

        quant_sim.compute_encodings(forward_pass, None)

    @classmethod
    def save_quantized_checkpoint(
        cls,
        quant_sim: Any,
        output_dir: str | os.PathLike | Path,
    ) -> None:
        """Save the quantized VEG checkpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        quant_sim.export(
            str(output_path),
            "vision_encoder",
        )


class Qwen3VLVisionWrapper(torch.nn.Module):
    """Adapts a Qwen3-VL vision model to the generator's vision interface.

    The generator always calls ``vision_model(pixel_values, image_grid_thw,
    mask)`` and expects ``(image_embeddings, visual_pos_masks,
    *deepstack_features)`` back (matching ``get_visual_output_names``). Two
    kinds of vision model can be wrapped, and this class picks the right call
    convention based on what it is handed (mirrors Qwen2.5-VL's
    ``Qwen2VLVisionWrapper``):

    - Raw HF visual tower (calibration / demo): dynamic, computes its rotary
      positions and attention masks internally from ``grid_thw``. We forward
      ``grid_thw`` through and read ``pooler_output`` / ``deepstack_features``
      off the returned dataclass.
    - ``Qwen3VLVisionEncoder`` VEG (evaluate): fixed input shape, with
      positions/masks precomputed as cached buffers. It only consumes
      ``pixel_values`` and returns ``(image_embeddings, *deepstack_features)``;
      ``grid_thw`` and ``mask`` are implied by the fixed shape and ignored.
    """

    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual
        self._is_veg = isinstance(visual, Qwen3VLVisionEncoder)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor, list[torch.Tensor]]:
        if pixel_values is None:
            return None, mask, []
        if self._is_veg:
            # VEG has a fixed input shape; positions/masks are cached buffers,
            # so grid_thw and mask are not passed. The encoder returns
            # (image_embeddings, *deepstack_features). The quantized VEG runs
            # via onnxruntime and returns CPU tensors, so move outputs back to
            # the input device before the generator merges them.
            image_embeddings, *deepstack_features = self.visual(pixel_values)
            return (
                image_embeddings.to(pixel_values.device),
                mask,
                [d.to(pixel_values.device) for d in deepstack_features],
            )
        dtype = next(self.visual.parameters()).dtype
        pixel_values = pixel_values.to(dtype=dtype)
        vision_outputs = self.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True
        )
        return (
            vision_outputs.pooler_output,
            mask,
            vision_outputs.deepstack_features,
        )

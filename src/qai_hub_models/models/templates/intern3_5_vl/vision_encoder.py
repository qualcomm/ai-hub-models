# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModel, PreTrainedModel

from qai_hub_models import Precision
from qai_hub_models.datasets.imagenet import IMAGENETTE_ASSET
from qai_hub_models.models.templates.intern3_5_vl.vision_encoder_adaptations import (
    replace_visual_attention_with_adaptation,
)
from qai_hub_models.models.templates.qwen2_vl.vision_encoder_adaptations import (
    replace_linears_with_convs,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec


class Intern3_5VLVisionEncoder(BaseModel):
    """
    Adapted vision encoder for InternVL3.5 on-device export (VEG).

    Returns a tuple of outputs:
    - image_embeddings: main pooler output (post-merger)
    - deepstack features: per-layer merged visual embeddings

    The forward() takes 5 inputs:
    - pixel_values: preprocessed image patches
    - position_ids_cos: pre-computed RoPE cosine values
    - position_ids_sin: pre-computed RoPE sine values
    - window_attention_mask: attention mask placeholder (kept for interface compat)
    - full_attention_mask: attention mask for all layers
    """

    _pos_emb_cos: torch.Tensor
    _pos_emb_sin: torch.Tensor
    _window_attention_mask: torch.Tensor
    _full_attention_mask: torch.Tensor

    def __init__(
        self,
        visual: Any,
        projector: Any,
        grid_thw: torch.Tensor,
        in_channels: int = 3,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        downsample_ratio: float = 0.5,
        select_layer: int = -1,
        ps_version: str = "v2",
    ) -> None:
        super().__init__()
        self.visual = visual
        self.projector = projector
        self.downsample_ratio = downsample_ratio
        self.select_layer = select_layer
        self.ps_version = ps_version
        self.deepstack_visual_indexes: list[int] = []

        # Kept as inert buffers for interface compatibility.
        self.register_buffer("_pos_emb_cos", torch.zeros((1, 1), dtype=torch.float32))
        self.register_buffer("_pos_emb_sin", torch.zeros((1, 1), dtype=torch.float32))
        self.register_buffer(
            "_full_attention_mask", torch.zeros((1, 1, 1), dtype=torch.float32)
        )
        self.register_buffer(
            "_window_attention_mask", torch.zeros((1, 1, 1), dtype=torch.float32)
        )

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
        """Forward pass; returns image features only for InternVL vision path."""
        del (
            position_ids_cos,
            position_ids_sin,
            window_attention_mask,
            full_attention_mask,
        )

        if self.select_layer == -1:
            vit_embeds = self.visual(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds = self.visual(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.select_layer]

        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = Intern3_5VLVisionWrapper._pixel_shuffle(
            vit_embeds,
            scale_factor=self.downsample_ratio,
            ps_version=self.ps_version,
        )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        image_embeddings = self.projector(vit_embeds)
        return (image_embeddings,)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = "OpenGVLab/InternVL3_5-2B",
        device: torch.device | None = None,
        image_height: int = 512,
        image_width: int = 512,
        precision: Precision = Precision.float,
    ) -> Intern3_5VLVisionEncoder:
        """Load the vision encoder from an InternVL3.5 checkpoint."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True)
        if getattr(config, "model_type", "") != "internvl_chat":
            raise ValueError(
                "intern3_5_vl vision encoder expects InternVL checkpoints "
                "(model_type='internvl_chat')."
            )
        vis_config = config.vision_config
        patch_size = vis_config.patch_size
        temporal_patch_size = 1

        # Compute grid dimensions
        h_patches = image_height // patch_size
        w_patches = image_width // patch_size
        grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.int64)

        # Load full VLM and extract InternVL visual component.
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            type.__setattr__(PreTrainedModel, "all_tied_weights_keys", {})
        full_model = AutoModel.from_pretrained(
            str(checkpoint),
            trust_remote_code=True,
        ).to(device)  # type: ignore[arg-type, unused-ignore]
        if not hasattr(full_model, "vision_model"):
            raise ValueError(
                "InternVL checkpoint does not expose `vision_model`; "
                "cannot build vision encoder."
            )
        visual = full_model.vision_model
        projector = full_model.mlp1

        # Apply adaptations for on-device export
        replace_visual_attention_with_adaptation(visual)
        replace_linears_with_convs(visual)
        replace_linears_with_convs(projector)

        # Create VEG instance
        instance = cls(
            visual=visual,
            projector=projector,
            grid_thw=grid_thw.to(device),
            in_channels=getattr(vis_config, "in_channels", vis_config.num_channels),
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            downsample_ratio=float(getattr(full_model, "downsample_ratio", 0.5)),
            select_layer=int(getattr(full_model, "select_layer", -1)),
            ps_version=str(getattr(full_model, "ps_version", "v2")),
        )
        del full_model
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
        return self.get_static_input_spec(image_height, image_width, self._patch_size)

    @staticmethod
    def get_static_input_spec(
        image_height: int = 512,
        image_width: int = 512,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        rope_dim: int | None = None,
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
            Reserved for interface compatibility with shared vision helpers.

        Returns
        -------
        InputSpec
            Input specification dictionary.
        """
        del patch_size, temporal_patch_size, rope_dim
        input_spec: InputSpec = {}
        input_spec["pixel_values"] = TensorSpec(
            shape=(1, in_channels, image_height, image_width),
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
        and throws on the encoder.
        """
        return ["image_features"]

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
        processor = AutoImageProcessor.from_pretrained(hf_repo, trust_remote_code=True)

        calibration_data = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((image_width, image_height))
            inputs = processor(
                images=[img],
                return_tensors="pt",
            )
            # Keep batch dimension (N,C,H,W) for ORT input rank compatibility.
            pixel_values = inputs["pixel_values"]
            calibration_data.append(pixel_values.numpy())

        return calibration_data

    @classmethod
    def export_to_onnx(
        cls,
        veg_model: Intern3_5VLVisionEncoder,
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

        veg_model.to(host_device).float()
        torch.onnx.export(
            veg_model,
            tuple(sample_inputs.values()),
            onnx_path,
            input_names=list(input_spec.keys()),
            output_names=veg_model.get_output_names(),
            opset_version=18,
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
        onnx.save_model(
            onnx_model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=filename + ".data",
        )
        return onnx_path

    @classmethod
    def create_quantsim_from_onnx(
        cls,
        onnx_model: onnx.ModelProto,
        veg_model: Intern3_5VLVisionEncoder | None,
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

        return quant_sim, {}

    @classmethod
    def calibrate(
        cls,
        quant_sim: Any,
        calibration_data: list,
        fixed_inputs: dict,
    ) -> None:
        """Calibrate the VEG QuantSim with real images."""

        def forward_pass(session: Any, args: Any = None) -> None:
            _ = (fixed_inputs, args)
            for pixel_values_np in calibration_data:
                feed_dict = {"pixel_values": pixel_values_np}
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


class Intern3_5VLVisionWrapper(torch.nn.Module):
    """Adapts an InternVL3.5 vision model to the generator vision interface.

    The generator always calls ``vision_model(pixel_values, image_grid_thw,
    mask)`` and expects ``(image_embeddings, visual_pos_masks,
    *deepstack_features)`` back (matching ``get_visual_output_names``). Two
    Two kinds of vision model can be wrapped:
    - Raw InternVL chat model (vision tower under ``vision_model``, projector
      under ``mlp1``).
    - Fixed-shape export vision model (``Intern3_5VLVisionEncoder``).
    """

    visual: torch.nn.Module
    _wrapper_like: bool
    _is_veg: bool
    _is_internvl_chat_model: bool
    _is_internvl_components: bool
    projector: torch.nn.Module | None
    downsample_ratio: float
    select_layer: int
    ps_version: str
    _chat_vision_model: torch.nn.Module | None
    _chat_mlp1: torch.nn.Module | None
    _components_visual: torch.nn.Module | None

    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.projector = None
        self.downsample_ratio = 0.5
        self.select_layer = -1
        self.ps_version = "v2"
        self._chat_vision_model = None
        self._chat_mlp1 = None
        self._components_visual = None

        # Unwrap nested wrappers to avoid delegation chains.
        if isinstance(visual, Intern3_5VLVisionWrapper):
            visual = visual.visual

        self.visual = visual
        self._is_veg = isinstance(visual, Intern3_5VLVisionEncoder)
        self._is_internvl_chat_model = hasattr(visual, "vision_model") and hasattr(
            visual, "mlp1"
        )
        self._is_internvl_components = hasattr(visual, "embeddings") and hasattr(
            visual, "encoder"
        )
        self._wrapper_like = (
            not self._is_veg
            and not self._is_internvl_chat_model
            and not self._is_internvl_components
            and hasattr(visual, "visual")
            and callable(getattr(visual, "forward", None))
        )
        if self._wrapper_like:
            return
        if self._is_internvl_chat_model:
            chat_vision_model = getattr(visual, "vision_model", None)
            if not isinstance(chat_vision_model, torch.nn.Module):
                raise TypeError(
                    "InternVL chat model must expose `vision_model` as "
                    "a torch.nn.Module instance."
                )
            chat_mlp1 = getattr(visual, "mlp1", None)
            if not isinstance(chat_mlp1, torch.nn.Module):
                raise TypeError(
                    "InternVL chat model must expose `mlp1` as "
                    "a torch.nn.Module instance."
                )
            self._chat_vision_model = chat_vision_model
            self._chat_mlp1 = chat_mlp1
            self.downsample_ratio = float(getattr(visual, "downsample_ratio", 0.5))
            self.select_layer = int(getattr(visual, "select_layer", -1))
            self.ps_version = str(getattr(visual, "ps_version", "v2"))
            # Keep only the vision tower registered on this wrapper so
            # wrapper.to(device) does not retain/move the full chat model.
            self.visual = chat_vision_model
        if self._is_internvl_components:
            if not isinstance(visual, torch.nn.Module):
                raise TypeError(
                    "InternVL components visual must be a torch.nn.Module instance."
                )
            self._components_visual = visual
            projector = getattr(visual, "projector", None)
            if not isinstance(projector, torch.nn.Module):
                raise TypeError(
                    "InternVL components must expose `projector` as torch.nn.Module."
                )
            self.projector = projector
            self.downsample_ratio = float(getattr(visual, "downsample_ratio", 0.5))
            self.select_layer = int(getattr(visual, "select_layer", -1))
            self.ps_version = str(getattr(visual, "ps_version", "v2"))

    @staticmethod
    def _pixel_shuffle(
        x: torch.Tensor, scale_factor: float, ps_version: str
    ) -> torch.Tensor:
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        # InternVL uses ps_version=v2 by default and swaps H/W back.
        if ps_version != "v1":
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _run_internvl_visual_projector(
        self,
        visual_model: torch.nn.Module,
        projector: torch.nn.Module,
        pixel_values: torch.Tensor,
        select_layer: int,
        downsample_ratio: float,
        ps_version: str,
    ) -> torch.Tensor:
        if select_layer == -1:
            vit_embeds = visual_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds = visual_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[select_layer]

        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self._pixel_shuffle(
            vit_embeds, scale_factor=downsample_ratio, ps_version=ps_version
        )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return projector(vit_embeds)

    def _extract_internvl_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._is_internvl_components:
            if self._components_visual is None:
                raise TypeError("InternVL components visual is not initialized.")
            if self.projector is None:
                raise TypeError("InternVL components projector is not initialized.")
            return self._run_internvl_visual_projector(
                visual_model=self._components_visual,
                projector=self.projector,
                pixel_values=pixel_values,
                select_layer=self.select_layer,
                downsample_ratio=self.downsample_ratio,
                ps_version=self.ps_version,
            )

        if self._chat_vision_model is None or self._chat_mlp1 is None:
            raise TypeError("InternVL chat modules are not initialized.")
        return self._run_internvl_visual_projector(
            visual_model=self._chat_vision_model,
            projector=self._chat_mlp1,
            pixel_values=pixel_values,
            select_layer=self.select_layer,
            downsample_ratio=self.downsample_ratio,
            ps_version=self.ps_version,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor, list[torch.Tensor]]:
        scalar_mode = image_grid_thw is None and mask is None
        if self._wrapper_like:
            if scalar_mode:
                out = self.visual(pixel_values)
                if isinstance(out, tuple):
                    return out[0]
                return out
            return self.visual(pixel_values, image_grid_thw, mask)
        del image_grid_thw
        if pixel_values is None:
            if scalar_mode:
                return torch.empty(0)
            assert mask is not None
            return None, mask, []
        if self._is_veg:
            # VEG has a fixed input shape; positions/masks are cached buffers,
            # so grid_thw and mask are not passed. The encoder returns
            # (image_embeddings, *deepstack_features). The quantized VEG runs
            # via onnxruntime and returns CPU tensors, so move outputs back to
            # the input device before the generator merges them.
            if isinstance(self.visual, torch.nn.Module):
                veg_dtype = next(self.visual.parameters()).dtype
                pixel_values = pixel_values.to(dtype=veg_dtype)
            image_embeddings, *deepstack_features = self.visual(pixel_values)
            image_embeddings = image_embeddings.to(pixel_values.device)
            deepstack_features = [d.to(pixel_values.device) for d in deepstack_features]
            if scalar_mode:
                return image_embeddings
            assert mask is not None
            return (
                image_embeddings,
                mask,
                deepstack_features,
            )
        if self._is_internvl_chat_model:
            # Raw InternVL chat model path: use vision_model + mlp1 extract.
            if self._chat_vision_model is None:
                raise TypeError("InternVL chat vision model is not initialized.")
            vision_dtype = next(self._chat_vision_model.parameters()).dtype
            pixel_values = pixel_values.to(dtype=vision_dtype)
            image_embeddings = self._extract_internvl_feature(pixel_values)
            image_embeddings = image_embeddings.to(pixel_values.device)
            if scalar_mode:
                return image_embeddings
            assert mask is not None
            return image_embeddings, mask, []
        if self._is_internvl_components:
            if self._components_visual is None:
                raise TypeError("InternVL components visual is not initialized.")
            vision_dtype = next(self._components_visual.parameters()).dtype
            pixel_values = pixel_values.to(dtype=vision_dtype)
            image_embeddings = self._extract_internvl_feature(pixel_values)
            image_embeddings = image_embeddings.to(pixel_values.device)
            if scalar_mode:
                return image_embeddings
            assert mask is not None
            return image_embeddings, mask, []
        raise TypeError(
            "Unsupported vision module for Intern3_5VLVisionWrapper. "
            "Expected InternVL chat model or Intern3_5VLVisionEncoder."
        )

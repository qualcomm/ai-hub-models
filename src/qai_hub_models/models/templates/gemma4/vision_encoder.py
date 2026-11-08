# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Gemma4 Visual Embedding Generator (VEG) for on-device export.

The VEG is the Gemma4 vision path: a SigLIP-style ViT (``vision_tower`` =
patch-embedder + encoder + pooler) followed by ``embed_vision`` (a projection
into the language-model embedding space), producing the soft-token embeddings
the LLM consumes at image positions. It is wrapped as an AI Hub :class:`BaseModel`
and registered as the ``vision_encoder`` component of the Gemma4 Collection, so
the export pipeline and the scorecard serialize it to a float ONNX and compile it to a
QNN asset alongside the text Parts.

One compiled asset = one fixed square resolution (``image_size``, default 448):
:meth:`from_pretrained` traces the HF processor once on a synthetic image to
capture the ``pixel_values`` / ``image_position_ids`` shapes and stashes them as
buffers. Variable resolution / pan-and-scan tiling is out of scope for now.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from PIL import Image
from qai_hub.client import Device
from transformers import AutoProcessor

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.datasets.imagenet import IMAGENETTE_ASSET
from qai_hub_models.models.templates.gemma4.vision_encoder_adaptations import (
    replace_clippable_linears_with_scalar_bounds,
    replace_gemma4_attention_with_adaptation,
    replace_gemma4_rmsnorm_with_standard,
    replace_linears_with_convs,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

# Synthetic-image resolution used to trace the VEG (geometry depends only on
# resolution, not pixel content, so no calibration dataset is needed for fp16).
DEFAULT_IMAGE_SIZE = 448

# Genie tags a graph as an IMAGE_ENCODER only when an output name is prefixed with
# ``image_features`` / ``vision_embedding`` / ``cross_attention_states``.
VISION_EMBEDDING_OUTPUT_NAME = "vision_embedding"

# On-device precisions. Only float (FP16, HTP fp16-relaxed) works end-to-end.
# TODO: re-enable w16a16 (INT16) and w8a16 (INT8 weights + INT16 activations)
# once the Workbench compiler issue that fails their context-binary compile is
# resolved.
SUPPORTED_PRECISIONS: list[Precision] = [
    Precision.float,
    # Precision.w16a16,
    # Precision.w8a16,
]


def _synthetic_image(image_size: int) -> Image.Image:
    """Deterministic RGB gradient (content-independent VEG geometry)."""
    arr = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    ramp = np.linspace(0, 255, image_size, dtype=np.uint8)
    arr[:, :, 0] = ramp[None, :]
    arr[:, :, 1] = ramp[:, None]
    arr[:, :, 2] = 128
    return Image.fromarray(arr, "RGB")


class VEGWrapper(torch.nn.Module):
    """Thin ``nn.Module`` that runs the adapted vision_tower + embed_vision.

    Kept separate from the :class:`BaseModel` so it can be traced/exported as a
    plain module while the :class:`Gemma4VisionEncoder` provides the AI Hub
    Workbench interface.
    """

    vision_tower: torch.nn.Module
    embed_vision: torch.nn.Module

    def __init__(
        self,
        vision_tower: torch.nn.Module,
        embed_vision: torch.nn.Module,
        output_length: int,
        pooled_length: int,
    ) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision
        self.output_length = output_length
        self.pooled_length = pooled_length

    def forward(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor
    ) -> torch.Tensor:
        padding_positions = (image_position_ids == -1).all(dim=-1)
        inputs_embeds = self.vision_tower.patch_embedder(  # type: ignore[operator, unused-ignore]
            pixel_values, image_position_ids, padding_positions
        )
        enc = self.vision_tower.encoder(  # type: ignore[operator, unused-ignore]
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=image_position_ids,
        )
        hidden_states, _ = self.vision_tower.pooler(  # type: ignore[operator, unused-ignore]
            hidden_states=enc.last_hidden_state,
            pixel_position_ids=image_position_ids,
            padding_positions=padding_positions,
            output_length=self.output_length,
        )
        return self.embed_vision(  # type: ignore[operator, unused-ignore]
            inputs_embeds=hidden_states[:, : self.pooled_length, :]
        )


class Gemma4VisionEncoder(BaseModel):
    """Adapted Gemma4 vision encoder (VEG) for on-device export.

    ``forward(pixel_values, image_position_ids)`` returns the vision-token
    embeddings of shape ``(batch, num_soft_tokens, text_hidden_size)``.

    Concrete subclasses set ``_hf_repo_name`` (and may override
    ``default_image_size``).
    """

    _hf_repo_name: str = ""
    default_image_size: int = DEFAULT_IMAGE_SIZE
    supported_precisions: list[Precision] = SUPPORTED_PRECISIONS

    def __init__(
        self,
        veg: VEGWrapper,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> None:
        super().__init__()
        self.veg = veg
        # Reference inputs (from the processor); also the fixed input spec and
        # sample inputs for compile/profile.
        self.register_buffer("_ref_pixel_values", pixel_values)
        self.register_buffer("_ref_image_position_ids", image_position_ids)
        # On-device precision; the torch forward is always FP32, so precision
        # only affects compile + local QuantSim. Set by from_pretrained, along
        # with the checkpoint id + trace resolution (for get_calibration_data).
        self._precision: Precision = Precision.float
        self._checkpoint: str | None = None
        self._image_size: int = self.default_image_size

    def forward(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.veg(pixel_values, image_position_ids)

    def component_precision(self) -> Precision:
        """Precision this VEG is compiled at (float=FP16 today)."""
        return self._precision

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path | None = None,
        device: torch.device | None = None,
        image_size: int | None = None,
        precision: Precision = Precision.float,
    ) -> Gemma4VisionEncoder:
        """Load the VEG from a Gemma4 checkpoint and apply on-device adaptations.

        The exported graph is always a plain float ONNX; ``precision`` selects the
        on-device format applied at compile time (AI Hub post-training-quantizes
        the quantized precisions during compile; no AIMET encodings are baked in).
        """
        from transformers import (
            AutoConfig,
            AutoModelForImageTextToText,
            AutoProcessor,
        )

        if precision not in cls.supported_precisions:
            raise ValueError(
                f"{cls.__name__} supports precisions "
                f"{[str(p) for p in cls.supported_precisions]}; got {precision}."
            )
        if image_size is None:
            image_size = cls.default_image_size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_id = str(checkpoint) if checkpoint is not None else cls._hf_repo_name

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config._attn_implementation = "eager"
        # The VEG uses only vision_tower + embed_vision; the language_model /
        # audio_tower weights are loaded but unused.
        model = (
            AutoModelForImageTextToText.from_pretrained(
                model_id,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            .eval()
            .to(device)  # type: ignore[arg-type, unused-ignore]
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Capture reference inputs (fixes the graph geometry).
        img = _synthetic_image(image_size)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(  # type: ignore[union-attr, unused-ignore]
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        pixel_values = inputs["pixel_values"].to(device)
        image_position_ids = inputs["image_position_ids"].to(device)

        vision_tower = model.model.vision_tower
        embed_vision = model.model.embed_vision
        out_len = vision_tower.config.default_output_length
        pooling_kernel = vision_tower.config.pooling_kernel_size

        # Detach the vision path from the (large) full model, then free it.
        model.model.vision_tower = None
        model.model.embed_vision = None
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vision_tower = vision_tower.eval().to(device)
        embed_vision = embed_vision.eval().to(device)

        # Apply QNN-friendly adaptations (attn / RMSNorm / Linear->Conv).
        vt = replace_gemma4_attention_with_adaptation(
            copy.deepcopy(vision_tower),
            k=pooling_kernel,
            output_length=out_len,
            pixel_position_ids=image_position_ids,
        )
        vt = replace_gemma4_rmsnorm_with_standard(vt)
        ev = replace_gemma4_rmsnorm_with_standard(copy.deepcopy(embed_vision))
        # Freeze the clippable-linear bounds to floats before the Linear->Conv
        # swap, which replaces the inner nn.Linear the wrapper holds.
        vt = replace_clippable_linears_with_scalar_bounds(vt)
        ev = replace_clippable_linears_with_scalar_bounds(ev)
        vt = replace_linears_with_convs(vt)
        ev = replace_linears_with_convs(ev)
        pooled_length = int(vt.pooler.pooled_length)  # type: ignore[union-attr, arg-type, unused-ignore]

        veg = VEGWrapper(vt, ev, out_len, pooled_length).eval().to(device)
        instance = cls(
            veg=veg,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
        )
        instance._precision = precision
        instance._checkpoint = model_id
        instance._image_size = image_size
        instance.to(device)
        instance.eval()
        return instance

    def get_input_spec(
        self,
        image_size: int | None = None,
    ) -> InputSpec:
        # Fixed geometry: reject a mismatched size instead of ignoring it.
        if image_size is not None and image_size != self._image_size:
            raise ValueError(
                f"This VEG was traced at image_size={self._image_size}; re-run "
                f"from_pretrained with image_size={image_size} to change the "
                f"graph geometry."
            )
        assert isinstance(self._ref_pixel_values, torch.Tensor)
        assert isinstance(self._ref_image_position_ids, torch.Tensor)
        pv = self._ref_pixel_values
        pid = self._ref_image_position_ids
        return {
            "pixel_values": TensorSpec(shape=tuple(pv.shape), dtype="float32"),
            "image_position_ids": TensorSpec(shape=tuple(pid.shape), dtype="int64"),
        }

    def get_output_names(self) -> list[str]:
        return [VISION_EMBEDDING_OUTPUT_NAME]

    def get_output_spec(self) -> OutputSpec:
        return {name: TensorSpec() for name in self.get_output_names()}

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        assert isinstance(self._ref_pixel_values, torch.Tensor)
        assert isinstance(self._ref_image_position_ids, torch.Tensor)
        return {
            "pixel_values": [
                self._ref_pixel_values.detach().cpu().numpy().astype(np.float32)
            ],
            "image_position_ids": [
                self._ref_image_position_ids.detach().cpu().numpy().astype(np.int64)
            ],
        }

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        """Compile options for the VEG, on top of the base options:

        1. ``--truncate_64bit_io --truncate_64bit_tensors`` for non-ONNX runtimes
           (``image_position_ids`` is int64 and QNN HTP has no native int64, so
           the converter aborts otherwise).
        2. For a quantized precision, ``--quantize_full_type`` so AI Hub PTQs the
           float graph to that type, WITHOUT ``--quantize_io`` — quantized graph
           I/O (UFIXED_POINT_16) dies on v73 with a skel-side DMA error, so I/O
           stays FLOAT_32 while the graph body is quantized. ``Precision.float``
           adds nothing and runs FP16.

        The collection export passes the *text* precision (w4a16) down to every
        component, so an unsupported precision is narrowed to the VEG's own
        ``component_precision()`` rather than emitted verbatim.
        """
        if precision not in self.supported_precisions:
            precision = self.component_precision()
        compile_options = super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"
            if precision != Precision.float and "--quantize_full_type" not in (
                other_compile_options
            ):
                compile_options = compile_options.replace(" --quantize_io", "")
                compile_options += f" --quantize_full_type {precision}"
        return compile_options

    def get_calibration_data(
        self,
        num_samples: int = 100,
    ) -> dict[str, list[np.ndarray]]:
        """Real-image calibration inputs for post-training quantization.

        Mirrors the Qwen VL VEG recipe: run ``num_samples`` Imagenette images
        through the Gemma4 processor to produce correctly-shaped ``pixel_values``
        / ``image_position_ids`` for the fixed VEG geometry. Returns
        ``{input_name: [np.ndarray, ...]}`` (one array per image), for AI-Hub
        ``calibration_data`` or a local QuantSim's ``compute_encodings``. Falls
        back to the synthetic input if Imagenette is unavailable.
        """
        pixel_values: list[np.ndarray] = []
        image_position_ids: list[np.ndarray] = []

        assert isinstance(self._ref_pixel_values, torch.Tensor)
        assert isinstance(self._ref_image_position_ids, torch.Tensor)
        ref_pv_shape = tuple(self._ref_pixel_values.shape)
        ref_pid_shape = tuple(self._ref_image_position_ids.shape)

        try:
            IMAGENETTE_ASSET.fetch(extract=True)
            train_dir = IMAGENETTE_ASSET.extracted_path / "train"
            image_paths: list[Path] = []
            for class_dir in sorted(train_dir.iterdir()):
                if class_dir.is_dir():
                    image_paths.extend(
                        p
                        for p in sorted(class_dir.iterdir())
                        if p.suffix.lower() in (".jpeg", ".jpg", ".png")
                    )
            image_paths = image_paths[:num_samples]

            processor = AutoProcessor.from_pretrained(
                self._checkpoint or self._hf_repo_name, trust_remote_code=True
            )
            size = self._image_size
            for img_path in image_paths:
                img = Image.open(img_path).convert("RGB").resize((size, size))
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Describe this image."},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                pv = inputs["pixel_values"].cpu().numpy().astype(np.float32)
                pid = inputs["image_position_ids"].cpu().numpy().astype(np.int64)
                # Only keep images whose (fixed) geometry matches the traced
                # graph; pan-and-scan can vary tile count for odd aspect ratios,
                # but square resize to the trace resolution keeps it constant.
                if tuple(pv.shape) == ref_pv_shape and tuple(pid.shape) == (
                    ref_pid_shape
                ):
                    pixel_values.append(pv)
                    image_position_ids.append(pid)
        except Exception as e:
            print(
                f"[VEG] Imagenette calibration unavailable ({type(e).__name__}: "
                f"{e}); falling back to the reference synthetic input."
            )

        print(f"[VEG] calibration: kept {len(pixel_values)}/{num_samples} images.")
        if not pixel_values:
            print("[VEG] WARNING: no usable images; falling back to synthetic input.")
            return self._sample_inputs_impl()

        return {
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids,
        }

    def export_to_onnx(
        self,
        output_path: str | os.PathLike | Path | None = None,
        opset: int = 18,
    ) -> Path:
        """Export the VEG to a float ONNX model, returning its path.

        Opset 18: torch's dynamo exporter only implements opset >= 18, the QAIRT
        converter handles 18, and the graph validates at cos=1.0 vs the golden.
        External-data files (weights >2GB) are written alongside the .onnx.
        """
        if output_path is None:
            output_path = Path(tempfile.mkdtemp()) / "vision_encoder.onnx"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        veg_cpu = self.veg.to("cpu").eval()
        assert isinstance(self._ref_pixel_values, torch.Tensor)
        assert isinstance(self._ref_image_position_ids, torch.Tensor)
        pv = self._ref_pixel_values.detach().cpu()
        pid = self._ref_image_position_ids.detach().cpu()

        torch.onnx.export(
            veg_cpu,
            (pv, pid),
            str(output_path),
            input_names=["pixel_values", "image_position_ids"],
            output_names=self.get_output_names(),
            opset_version=opset,
            do_constant_folding=True,
        )

        model = onnx.load(str(output_path))
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, str(output_path))
        return output_path

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Serialize to an ONNX model for AI Hub Workbench compilation."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = out_dir / f"{self.name}.onnx"
        self.export_to_onnx(onnx_path)
        return onnx_path

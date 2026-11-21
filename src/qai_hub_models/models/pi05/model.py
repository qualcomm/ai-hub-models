# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime
import torch
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    PaliGemmaWithExpertModel,
    PI05Policy,
    PI05Pytorch,
)
from onnxsim import simplify
from qai_hub.client import Device
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaMLP
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.models.pi05.dataset import LiberoDataset
from qai_hub_models.models.pi05.model_adaptation import (
    GemmaMLPSplitLinear,
    SHAGemmaExpertAttention,
    apply_rope_direct,
)
from qai_hub_models.models.templates.llm._utils import (
    _set_matmul_second_input_to_8b,
    _set_tensors_to_output_8b_sym,
)
from qai_hub_models.utils.aimet.aimet_dummy_model import zip_aimet_model
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
from qai_hub_models.utils.aimet.encodings import apply_propagate_memory_encodings
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.checkpoint import (
    CheckpointSpec,
    CheckpointType,
    FromPretrainedMixin,
)
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
    make_torch_inputs,
)
from qai_hub_models.utils.onnx.helpers import ONNXBundle, mock_torch_onnx_inference
from qai_hub_models.utils.qai_hub_helpers import (
    ensure_hexagon_version,
    export_torch_to_onnx_zip,
)
from qai_hub_models.utils.quantization_aimet_onnx import (
    AIMETOnnxQuantizableMixin,
    aimet_quant_types,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
_PI05_AIMET_CONFIG = str(Path(__file__).parent / "aimet_config.json")

MAX_TOKEN_LENGTH = 200  # 48 for pi0, 200 for pi05
NUM_ACTION_STEPS = 50
# Set NUM_CAMERAS to override the value in policy (e.g., for profiling
# purpose).
NUM_CAMERAS = 3

DEFAULT_CHECKPOINT = "lerobot/pi05_libero_finetuned"


@lru_cache(maxsize=1)  # Cache only the most recent checkpoint
def load_checkpoint(checkpoint: CheckpointSpec) -> PI05Policy:
    if checkpoint == "DEFAULT" or Path(str(checkpoint)).is_dir():
        # Local directories are AIMET quantization checkpoints, not
        # LeRobot models.  Always load the base policy from HuggingFace.
        checkpoint = DEFAULT_CHECKPOINT
    # Use str to be hashable.
    print(f"Loading checkpoint: {checkpoint}")
    policy = PI05Policy.from_pretrained(checkpoint)

    # Hack: using global variable to pass this info
    global NUM_CAMERAS  # noqa: PLW0603

    if NUM_CAMERAS is None:
        NUM_CAMERAS = count_num_cameras(policy.config)
    print(f"Use {NUM_CAMERAS=}")
    return policy


def count_num_cameras(cfg: PI05Config) -> int:
    input_features = cfg.input_features
    return sum(
        1 for feature in input_features.values() if feature.type.name == "VISUAL"
    )


def kv_args_to_lists(
    **kv_cache_kwargs: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Convert either positional or keyword K/V cache inputs into two lists.

    kv_cache_kwargs expects keys "key_cache_l{idx}" and "value_cache_l{idx}".

    Parameters
    ----------
    **kv_cache_kwargs
        Keyword arguments containing key_cache_l{idx} and value_cache_l{idx}.

    Returns
    -------
    k_list : list[torch.Tensor]
        List of key cache tensors sorted by index.
    v_list : list[torch.Tensor]
        List of value cache tensors sorted by index.
    """

    def _sorted_by_idx(prefix: str) -> list[torch.Tensor]:
        items: list[tuple[int, torch.Tensor]] = []
        for k, v in kv_cache_kwargs.items():
            if k.startswith(prefix):
                suf = k[len(prefix) :]
                try:
                    idx = int(suf)
                except ValueError as err:
                    raise ValueError(f"Invalid cache key suffix in {k}") from err
                items.append((idx, v))
        if not items:
            raise ValueError(f"No entries for prefix {prefix}")
        items.sort(key=lambda t: t[0])
        return [t[1] for t in items]

    k_list = _sorted_by_idx("key_cache_l")
    v_list = _sorted_by_idx("value_cache_l")
    return k_list, v_list


class LoadPolicyMixin(FromPretrainedMixin):
    @classmethod
    def torch_from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
    ) -> PI05Policy:
        return load_checkpoint(str(checkpoint)).to(host_device)

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        class_name = self.__class__.__name__
        path = Path(output_dir) / f"{class_name}.onnx"
        assert input_spec is not None
        if path.exists():
            return path
        return Path(
            export_torch_to_onnx_zip(
                self.to("cpu"),  # type: ignore[attr-defined]
                str(path),
                make_torch_inputs(input_spec),
                input_names=list(input_spec.keys()),
                skip_zip=False,
            )
        )

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        return ensure_hexagon_version(
            min_version=73,
            target_runtime=target_runtime,
            device=device,
            model_name="Pi05",
        )


class Pi05PaliGemmaVision(LoadPolicyMixin, BaseModel):
    """
    PaliGemma vision encoder (fp). Resizing is performed in Pi05App,
    and this module expects inputs already at 224x224 and normalized
    to [-1, 1]. Keeping preprocessing minimal here simplifies the
    end-to-end deployment wiring.
    """

    def __init__(self, model: PI05Policy) -> None:
        flow_model: PI05Pytorch = model.model.to(torch.float32)
        super().__init__(flow_model)

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute vision embedding for a single image tensor
        of shape [B, C, 224, 224].

        Inputs are expected to be normalized to [-1, 1].
        """
        if image.ndim != 4:
            raise ValueError(f"[B,C,H,W] expected, got {image.shape}")
        if image.shape[-2:] != (224, 224):
            raise ValueError("Input image must be 224x224 after resizing.")

        paligemma = self.model.paligemma_with_expert
        assert isinstance(paligemma, PaliGemmaWithExpertModel)
        return paligemma.embed_image(image)

    @classmethod
    def get_input_spec_static(
        cls,
        batch_size: int = 1,
    ) -> InputSpec:
        # Expect images already resized to 224x224. Pi05App handles
        # resizing and normalization.
        return dict(
            image=TensorSpec(
                shape=(batch_size, 3, 224, 224),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(-1.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        )

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        return self.get_input_spec_static(batch_size)

    @staticmethod
    def get_output_spec_static() -> OutputSpec:
        return {"img_embed": TensorSpec()}

    def get_output_spec(self) -> OutputSpec:
        return self.get_output_spec_static()


class _Pi05CachedExportMixin:
    """Mixin that overrides convert_to_hub_source_model to route through
    BaseModel's version instead of LoadPolicyMixin's version (which tries to trace the model directly).

    Also propagates AIMET encodings through memory ops (Transpose, Reshape,
    etc.) during export, mirroring what LLM models do in _adapt_aimet_encodings.
    This ensures ops like Transpose that have is_output_quantized=False still
    carry encodings through to downstream consumers (e.g. MatMul on HTP).
    """

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        return Path(
            self.convert_to_onnx_and_aimet_encodings(
                Path(output_dir),
                self.__class__.__name__,
            )
        )

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | Path,
        model_name: str | None = None,
        return_zip: bool = True,
    ) -> str:
        result = super().convert_to_onnx_and_aimet_encodings(  # type: ignore[misc]
            output_dir, model_name, return_zip=False
        )
        export_dir = Path(result)
        bundle = ONNXBundle.from_bundle_path(export_dir)
        apply_propagate_memory_encodings(bundle)
        if return_zip:
            model_name = model_name or self.__class__.__name__
            zip_path = Path(output_dir) / f"{model_name}.aimet.zip"
            base_dir = Path(f"{model_name}.aimet")
            data_path = bundle.onnx_weights_path
            zip_aimet_model(
                str(zip_path),
                base_dir,
                str(bundle.onnx_graph_path),
                str(bundle.aimet_encodings_path),
                str(data_path) if data_path else "",
            )
            return str(zip_path)
        return result

    def save_calibrated_checkpoint(self, output_checkpoint: str) -> None:
        super().save_calibrated_checkpoint(output_checkpoint)  # type: ignore[misc]
        default_subfolder = getattr(self.__class__, "default_subfolder", "")
        export_dir = Path(output_checkpoint)
        if default_subfolder:
            export_dir = export_dir / default_subfolder
        apply_propagate_memory_encodings(ONNXBundle.from_bundle_path(export_dir))


class Pi05PaliGemmaVisionQuantizable(
    _Pi05CachedExportMixin, AIMETOnnxQuantizableMixin, Pi05PaliGemmaVision
):
    """Exportable PaliGemma Vision encoder that can be quantized by AIMET-ONNX."""

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_subfolder = "vision_encoder"

    def __init__(
        self,
        sim_model: QuantSimOnnx | None,
        host_device: torch.device = torch.device("cpu"),
        onnx_bundle: ONNXBundle | None = None,
        precision: Precision = Precision.w8a16,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model, onnx_bundle=onnx_bundle)
        BaseModel.__init__(self, None)
        self.host_device = host_device
        self._precision = precision

    def make_quant_sim(self) -> QuantSimOnnx | None:
        if self._onnx_bundle is None:
            return None
        param_type, act_type = aimet_quant_types(self._precision)

        onnx_model = self._onnx_bundle.load_onnx_model()
        onnx_model, _ = simplify(onnx_model, skipped_optimizers=["fuse_qkv"])

        return QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=param_type,
            activation_type=act_type,
            config_file=get_aimet_config_path("default_config"),
            providers=AIMETOnnxQuantizableMixin.get_ort_providers(self.host_device),
        )

    @classmethod
    def torch_from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
    ) -> torch.nn.Module:
        policy = load_checkpoint(str(checkpoint))
        return Pi05PaliGemmaVision(policy).to(host_device).eval()

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        precision: Precision = Precision.w8a16,
        torch_from_pretrained_kwargs: dict[str, Any] | None = None,
        cls_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        bundle = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )
        return cls(
            None, host_device=host_device, onnx_bundle=bundle, precision=precision
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return cast(torch.Tensor, AIMETOnnxQuantizableMixin.forward(self, image))

    def component_precision(self) -> Precision:
        return Precision.w8a16


class Pi05PaliGemmaTokenEmbed(LoadPolicyMixin, BaseModel):
    """
    Token embeding step. Separate it out because it has large embedding /
    model size.

    camera_mask ([B, NUM_CAMERAS] float in {0, 1}) marks active camera slots
    so zero-padded streams stay masked in exported graphs. Optional in eager
    mode when only real cameras are passed.

    Float mask convention used here and throughout:
      - Additive masks use 0.0 for allowed and -1e4 for blocked entries.

    Now also returns:
      - prefix_att_2d: additive attention mask for prefix tokens with
        shape [B, 1, src_len, src_len]. 0.0 means allowed and -1e4 means
        masked out. This is designed to be added to attention logits.
      - padded_state: Float tensor [B, max_state_dim] where state is
        right-padded with zeros (or truncated) to config.max_state_dim.
      - prefix_sin: Float tensor [B, src_len, 1, D/2] RoPE sin terms for
        prefix (images + language).
      - prefix_cos: Float tensor [B, src_len, 1, D/2] RoPE cos terms for
        prefix (images + language).
      - suffix_sin: Float tensor [B, n_steps, 1, D/2] RoPE sin terms for
        suffix (actions) with positions offset by prefix.
      - suffix_cos: Float tensor [B, n_steps, 1, D/2] RoPE cos terms for
        suffix (actions) with positions offset by prefix.
      - full_att_4d: additive mask [B, 1, Ls, Lp+Ls] where 0.0 means
        allowed and -1e4 means blocked. This is designed to be added to
        attention logits as:
          masked = attn + full_attn_4d
    """

    def component_precision(self) -> Precision:
        return Precision.float

    def __init__(self, model: PI05Policy) -> None:
        assert isinstance(model, PI05Policy)
        flow_model: PI05Pytorch = model.model.to(torch.float32)
        assert flow_model.config.tokenizer_max_length == MAX_TOKEN_LENGTH
        super().__init__(flow_model)

        # RoPE setup. Keep in sync with model head dim (256).
        head_dim = (
            flow_model.paligemma_with_expert.paligemma.config.text_config.head_dim
        )
        d_half = head_dim // 2
        max_wavelength = 10_000.0

        freq = (2.0 / float(head_dim)) * torch.arange(d_half, dtype=torch.float32)
        inv_timescale = (max_wavelength ** (-freq)).to(torch.float32)
        self.register_buffer("inv_timescale", inv_timescale, persistent=False)
        self._rope_head_dim = head_dim

    @torch.no_grad()
    def _positions_to_sin_cos(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert position indices to RoPE sin/cos embeddings.

        Parameters
        ----------
        positions
            Long[ B, L ] position indices.

        Returns
        -------
        sin : torch.Tensor
            Float32[ B, L, 1, D/2 ] sine embeddings.
        cos : torch.Tensor
            Float32[ B, L, 1, D/2 ] cosine embeddings.
        """
        if positions.ndim != 2:
            raise ValueError(f"positions must be [B, L], got {positions.shape}")

        device = positions.device
        assert isinstance(self.inv_timescale, torch.Tensor)
        radians = positions.to(torch.float32)[..., None] * self.inv_timescale[
            None, None, :
        ].to(device)
        radians = radians[..., None, :]  # [B, L, 1, D/2]
        sin = torch.sin(radians).to(torch.float32)
        cos = torch.cos(radians).to(torch.float32)
        return sin, cos

    def forward(
        self,
        lang_tokens: torch.Tensor,  # [B, T]
        # Require lang_mask to be float in {0,1}. No bool tensors used.
        # 1.0 means active/visible; 0.0 means masked out.
        lang_mask: torch.Tensor,  # [B, T] float in {0,1}
        *img_embeds: torch.Tensor,  # each [B, Si, D], up to 3 streams
        camera_mask: torch.Tensor | None = None,  # [B, NUM_CAMERAS] float in {0,1}
    ) -> tuple[
        torch.Tensor,  # prefix_embs
        torch.Tensor,  # prefix_att_2d (additive: 0 allowed, -1e4 blocked)
        torch.Tensor,  # prefix_sin
        torch.Tensor,  # prefix_cos
        torch.Tensor,  # suffix_sin
        torch.Tensor,  # suffix_cos
        torch.Tensor,  # full_att_4d (additive: 0 allowed, -1e4 blocked)
    ]:
        # ONNX export passes (img_embed1..N, camera_mask) positionally via *img_embeds.
        if len(img_embeds) == NUM_CAMERAS + 1:
            if camera_mask is not None:
                raise ValueError("camera_mask was provided twice.")
            img_embeds, camera_mask = img_embeds[:-1], img_embeds[-1]

        # Language embedding + normalization.
        paligemma = self.model.paligemma_with_expert
        assert isinstance(paligemma, PaliGemmaWithExpertModel)
        lang_emb = paligemma.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        # Validate lang_mask is float in {0,1} without creating bool tensors.
        if not torch.is_floating_point(lang_mask):
            raise TypeError("lang_mask must be a float tensor in {0,1}.")
        diffs = lang_mask - lang_mask.round()
        if torch.count_nonzero(diffs).item() != 0:
            raise ValueError("lang_mask values must be 0 or 1.")
        if lang_mask.min().item() < 0.0 or lang_mask.max().item() > 1.0:
            raise ValueError("lang_mask values must be in {0,1}.")
        # Keep as float32 mask: 1.0 active, 0.0 masked out.
        lang_mask_f = lang_mask.to(dtype=torch.float32)

        # Collect up to NUM_CAMERAS image streams; pad missing with zeros.
        # Ensure img_embeds are on the same device as lang_emb (quantsim
        # returns CPU tensors from ONNX Runtime).
        img_embeds = tuple(e.to(lang_emb.device) for e in img_embeds)

        if len(img_embeds) == 0:
            raise ValueError("At least one image embedding is required.")
        if len(img_embeds) > NUM_CAMERAS:
            raise ValueError(f"At most {NUM_CAMERAS} image embeddings allowed.")

        base_shape = img_embeds[0].shape
        if len(base_shape) != 3:
            raise ValueError(f"Image embedding must be [B,S,D], got {base_shape}.")
        bsize, base_s, base_d = base_shape
        base_device = img_embeds[0].device
        base_dtype = img_embeds[0].dtype

        if camera_mask is None:
            camera_mask = torch.ones(bsize, NUM_CAMERAS, device=base_device)
        if not torch.is_floating_point(camera_mask):
            raise TypeError("camera_mask must be a float tensor in {0,1}.")
        if not torch.jit.is_tracing():
            if camera_mask.shape != (bsize, NUM_CAMERAS):
                raise ValueError("camera_mask must be shaped [B, NUM_CAMERAS].")
            if not torch.all((camera_mask == 0) | (camera_mask == 1)).item():
                raise ValueError("camera_mask values must be 0 or 1.")
        camera_mask = camera_mask.to(device=base_device, dtype=torch.float32)

        img_list: list[torch.Tensor] = list(img_embeds)
        pad_list_f: list[torch.Tensor] = []

        # Pad missing cameras with zeros and 0.0 masks.
        for _ in range(len(img_list), int(NUM_CAMERAS)):
            zero_img = torch.zeros(
                (bsize, base_s, base_d),
                dtype=base_dtype,
                device=base_device,
            )
            img_list.append(zero_img)

        # Per-stream token masks from camera_mask (0.0 for streams padded above).
        for i, emb in enumerate(img_list):
            n_i = emb.shape[1]
            if i < len(img_embeds):
                pad_mask_i = camera_mask[:, i : i + 1].expand(bsize, n_i)
            else:
                pad_mask_i = torch.zeros(
                    bsize, n_i, dtype=torch.float32, device=emb.device
                )
            pad_list_f.append(pad_mask_i)

        # Concatenate image embeddings and language embeddings.
        prefix_embs = torch.cat([*img_list, lang_emb], dim=1)

        # Build additive prefix attention mask for logits:
        # 0.0 means allowed, -1e4 means blocked.
        prefix_pad_1d_f = torch.cat([*pad_list_f, lang_mask_f], dim=1)
        allowed = (prefix_pad_1d_f[:, None, :] * prefix_pad_1d_f[:, :, None]).to(
            torch.float32
        )
        big_neg_val = -1e4
        prefix_att_2d = (1.0 - allowed) * big_neg_val
        prefix_att_2d = prefix_att_2d.unsqueeze(-3)  # [B,1,L,L]

        # Position ids: cumsum over valid tokens, 0-based (ints).
        prefix_pos_ids = (
            torch.cumsum(prefix_pad_1d_f.to(torch.float32), dim=1) - 1
        ).to(torch.int32)

        # ----- RoPE for prefix (images + language) -----
        prefix_sin, prefix_cos = self._positions_to_sin_cos(prefix_pos_ids)

        # ----- RoPE for suffix (actions) -----
        L_suffix = NUM_ACTION_STEPS
        prefix_len = prefix_pad_1d_f.sum(dim=1, keepdim=True).to(torch.long)
        ar = torch.arange(L_suffix, device=prefix_embs.device, dtype=torch.long)[
            None, :
        ]
        positions_suffix = prefix_len + ar
        suffix_sin, suffix_cos = self._positions_to_sin_cos(positions_suffix)

        # Build additive full attention mask for suffix attending to
        # [prefix then suffix]. 0.0 allowed, -1e4 blocked.
        suffix_pad_1d_f = torch.ones(
            bsize, L_suffix, dtype=torch.float32, device=prefix_embs.device
        )
        prefix_pad_2d_f = suffix_pad_1d_f[:, :, None] * prefix_pad_1d_f[:, None, :]
        suffix_att_2d_f = suffix_pad_1d_f[:, :, None] * suffix_pad_1d_f[:, None, :]
        full_att_2d_f = torch.cat([prefix_pad_2d_f, suffix_att_2d_f], dim=2)
        full_att_4d = (1.0 - full_att_2d_f).to(torch.float32) * big_neg_val
        full_att_4d = full_att_4d[:, None, :, :]  # [B,1,Ls,Lp+Ls]

        # Return outputs.
        return (
            prefix_embs,
            prefix_att_2d,
            prefix_sin,
            prefix_cos,
            suffix_sin,
            suffix_cos,
            full_att_4d,
        )

    @classmethod
    def get_input_spec_static(
        cls,
        batch_size: int = 1,
    ) -> InputSpec:
        num_patches = 256
        embed_dim = 2048

        spec: InputSpec = {
            "lang_tokens": TensorSpec(
                shape=(batch_size, MAX_TOKEN_LENGTH),
                dtype="int32",
                io_type=IoType.TENSOR,
            ),
            "lang_mask": TensorSpec(
                shape=(batch_size, MAX_TOKEN_LENGTH),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

        # Add img_embed1, img_embed2, ... for NUM_CAMERAS
        for cam_id in range(NUM_CAMERAS):
            key = f"img_embed{cam_id + 1}"
            spec[key] = TensorSpec(
                shape=(batch_size, num_patches, embed_dim),
                dtype="float32",
                io_type=IoType.TENSOR,
            )

        spec["camera_mask"] = TensorSpec(
            shape=(batch_size, NUM_CAMERAS),
            dtype="float32",
            io_type=IoType.TENSOR,
        )

        return spec

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        return self.get_input_spec_static(batch_size)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec()
        inputs = make_torch_inputs(input_spec)
        result: dict[str, list[Any]] = {}
        for i, name in enumerate(input_spec.keys()):
            arr = inputs[i].numpy()
            if name in ("lang_mask", "camera_mask"):
                arr = np.ones_like(arr)
            result[name] = [arr]
        return result

    @staticmethod
    def get_output_spec_static() -> OutputSpec:
        return {
            "prefix_emb": TensorSpec(),
            "prefix_att_2d": TensorSpec(),
            "prefix_sin": TensorSpec(),
            "prefix_cos": TensorSpec(),
            "suffix_sin": TensorSpec(),
            "suffix_cos": TensorSpec(),
            "full_att_4d": TensorSpec(),
        }

    def get_output_spec(self) -> OutputSpec:
        return self.get_output_spec_static()

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        class_name = self.__class__.__name__
        path = Path(output_dir) / f"{class_name}.onnx"
        assert input_spec is not None
        sample_inputs = make_torch_inputs(input_spec)
        # Override mask inputs to 0/1 tensors for tracing.
        for i, name in enumerate(input_spec):
            if name in ("lang_mask", "camera_mask"):
                sample_inputs[i] = torch.ones(input_spec[name][0])
        if path.exists():
            cached = onnx.load(path, load_external_data=False)
            if [tensor.name for tensor in cached.graph.input] != list(input_spec):
                raise ValueError(
                    f"Existing {path.name} inputs do not match input_spec. "
                    "Re-export into a fresh output directory."
                )
            return path
        return Path(
            export_torch_to_onnx_zip(
                self.to("cpu"),
                str(path),
                sample_inputs,
                input_names=list(input_spec.keys()),
                skip_zip=False,
            )
        )


class Pi05ActionExpert(LoadPolicyMixin, BaseModel):
    """
    Wraps the PI05Pytorch model to run the denoising step only.

    Mask conventions:
      - full_att_4d is an additive mask with shape [B, 1, Ls, Lp+Ls].
        Use 0.0 for allowed positions and -1e4 for blocked positions.
        This mask is added to attention logits.

    This expert owns the adaRMS time conditioning. Callers provide a
    per-item time_step in [0, 1], and the expert maps it to the cached
    conditioning internally.
    """

    def __init__(
        self,
        model: PI05Policy | None,
        num_integration_steps: int = 10,
    ) -> None:
        """
        Initialize the action expert.

        Parameters
        ----------
        model
            Optional PI05Policy providing weights and config. When
            None, buffers are initialized empty.
        num_integration_steps
            Used to compute Euler dt as
            dt = -1.0 / num_integration_steps for the internal update.
        """
        if model is None:
            super().__init__(None)
            self.register_buffer(
                "cached_adarms_cond",
                torch.empty(0, dtype=torch.float32),
                persistent=False,
            )
        else:
            flow_model: PI05Pytorch = model.model.to(torch.float32)
            super().__init__(flow_model)
            cached = self.precompute_adarms_cond(model)
            self.register_buffer(
                "cached_adarms_cond",
                cached.to(torch.float32),
                persistent=False,
            )

            # Replace multi-head attention with split-head attention
            # in each expert layer for better on-device performance.
            pg_we = flow_model.paligemma_with_expert
            assert isinstance(pg_we, PaliGemmaWithExpertModel)
            pal_cfg = pg_we.paligemma.config.text_config
            for layer in pg_we.gemma_expert.model.layers:
                layer.self_attn = SHAGemmaExpertAttention(
                    layer.self_attn,
                    pal_cfg.num_attention_heads,
                    pal_cfg.num_key_value_heads,
                    pal_cfg.head_dim,
                )

        self.num_integration_steps = int(num_integration_steps)

    @torch.no_grad()
    def precompute_adarms_cond(self, model: PI05Policy) -> torch.Tensor:
        """
        Precompute adaRMS conditioning for all Euler timesteps.

        Parameters
        ----------
        model
            PI05Policy instance to extract config and MLPs.

        Returns
        -------
        cached_cond : torch.Tensor
            Tensor [S, D] where S is num_inference_steps + 1 and D is the
            action_in_proj output dimension.
        """
        cfg = model.model.config
        num_steps = int(getattr(cfg, "num_inference_steps", 10))

        # t: 1.0, 1 - 1/N, ..., 0.0 (N+1 points)
        t_vals = torch.linspace(
            1.0,
            0.0,
            steps=num_steps + 1,
            dtype=torch.float32,
        )

        # Create sinusoidal time embedding as in PI05. The embedding dim
        # must be even and matches action_in_proj.out_features.
        dim = int(model.model.action_in_proj.out_features)
        if dim % 2 != 0:
            raise ValueError("time embedding dimension must be even")

        frac = torch.linspace(
            0.0,
            1.0,
            steps=dim // 2,
            dtype=torch.float32,
        )

        min_p = torch.as_tensor(cfg.min_period, dtype=torch.float32)
        max_p = torch.as_tensor(cfg.max_period, dtype=torch.float32)
        period = min_p * (max_p / min_p) ** frac
        scale = (2.0 * torch.pi) / period

        sin_in = t_vals[:, None] * scale[None, :]
        time_emb = torch.cat(
            [torch.sin(sin_in), torch.cos(sin_in)],
            dim=1,
        )

        # Apply the model's time MLP layers (SiLU).
        model.model.to(time_emb.device)
        time_emb = model.model.time_mlp_in(time_emb)
        time_emb = torch.nn.functional.silu(time_emb)
        time_emb = model.model.time_mlp_out(time_emb)
        return torch.nn.functional.silu(time_emb)

    def lookup_adarms_cond(self, time_step: torch.Tensor) -> torch.Tensor:
        """
        Translate per-item time_step in [0, 1] to cached conditioning.

        Parameters
        ----------
        time_step
            Float tensor [B] with values in [0, 1].

        Returns
        -------
        conditioning : torch.Tensor
            Tensor [B, D] conditioning selected from cached entries.
        """
        if time_step.ndim != 1:
            raise ValueError("time_step must be shaped [B]")
        assert isinstance(self.cached_adarms_cond, torch.Tensor)
        if self.cached_adarms_cond.numel() == 0:
            raise RuntimeError("cached_adarms_cond is not initialized")

        s_len = int(self.cached_adarms_cond.shape[0])
        n_steps = s_len - 1
        idx = torch.round((1.0 - time_step) * float(n_steps)).to(torch.long)
        idx = torch.clamp(idx, 0, s_len - 1)
        return self.cached_adarms_cond.index_select(0, idx)

    def expert_forward(
        self,
        full_att_4d: torch.Tensor,  # [B,1,Ls,Lp+Ls], additive mask:
        # 0 for allowed, -1e4 for disallowed
        rope_emb_sin: torch.Tensor,  # [B, n_steps, 1, d/2]
        rope_emb_cos: torch.Tensor,  # [B, n_steps, 1, d/2]
        k_caches: list[torch.Tensor],  # 18 x [B, src_len, H_kv, D]
        v_caches: list[torch.Tensor],  # 18 x [B, src_len, H_kv, D]
        x_t: torch.Tensor,  # [B, n_steps, state_dim]
        adarms_cond: torch.Tensor,  # [B, 1024]
    ) -> torch.Tensor:
        """
        Compute suffix stream directly using unbundled per-layer KV caches
        from the prefix (images + language). Avoids any concat/slice of a
        bundled KV cache.

        Parameters
        ----------
        full_att_4d
            [B,1,Ls,Lp+Ls] additive mask: 0 for allowed, -1e4 for disallowed.
        rope_emb_sin
            [B, n_steps, 1, d/2] RoPE sine embeddings.
        rope_emb_cos
            [B, n_steps, 1, d/2] RoPE cosine embeddings.
        k_caches
            18 x [B, src_len, H_kv, D] key caches per layer.
        v_caches
            18 x [B, src_len, H_kv, D] value caches per layer.
        x_t
            [B, n_steps, state_dim] current noisy actions.
        adarms_cond
            [B, 1024] adaRMS conditioning tensor.

        Returns
        -------
        action_emb : torch.Tensor
            Float tensor [B, n_steps, max_action_dim] after projection.
        """
        assert self.model is not None
        flow = self.model
        assert isinstance(flow, PI05Pytorch)
        pg_we = flow.paligemma_with_expert
        assert isinstance(pg_we, PaliGemmaWithExpertModel)
        gemma_layers = pg_we.gemma_expert.model.layers

        # Build suffix embeddings. Masks are not needed because the caller
        # supplies full_att_4d that already encodes allowed attention.
        device = next(flow.parameters()).device
        x_t = x_t.to(device)
        rope_emb_sin = rope_emb_sin.to(device)
        rope_emb_cos = rope_emb_cos.to(device)
        adarms_cond = adarms_cond.to(device)
        full_att_4d = full_att_4d.to(device)

        suffix_embs = flow.action_in_proj(x_t)

        # Iterate expert layers with split-head attention.
        hidden_suffix = suffix_embs.to(torch.float32)

        for layer_idx, layer in enumerate(gemma_layers):
            normed, gate = layer.input_layernorm(hidden_suffix, adarms_cond)

            # Split-head attention via SHAGemmaExpertAttention.
            out_emb = layer.self_attn(
                normed,
                rope_emb_sin,
                rope_emb_cos,
                k_caches[layer_idx].to(device),
                v_caches[layer_idx].to(device),
                full_att_4d,
            )

            if gate is None:
                out_emb = out_emb + hidden_suffix
            else:
                out_emb = out_emb * gate + hidden_suffix

            after_first_residual = out_emb

            # Post-attention norm can also be adaptive.
            out_emb, gate2 = layer.post_attention_layernorm(out_emb, adarms_cond)
            out_emb = layer.mlp(out_emb)

            if gate2 is None:
                hidden_suffix = out_emb + after_first_residual
            else:
                hidden_suffix = out_emb * gate2 + after_first_residual

        # Final norm on expert model (adaptive with adarms_cond).
        hidden_suffix, gate = pg_we.gemma_expert.model.norm(hidden_suffix, adarms_cond)
        suffix_only = hidden_suffix[:, -NUM_ACTION_STEPS:]
        suffix_only = suffix_only.to(torch.float32)
        return flow.action_out_proj(suffix_only)

    def _compute_update(
        self,
        full_att_4d: torch.Tensor,  # [B,1,Ls,Lp+Ls]
        rope_emb_sin: torch.Tensor,  # [B, n_steps, 1, d/2]
        rope_emb_cos: torch.Tensor,  # [B, n_steps, 1, d/2]
        x_t: torch.Tensor,  # [B, n_steps, state_dim]
        time_step: torch.Tensor,  # [B] values in [0,1]
        **kv_cache_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the velocity update v_t for the current x_t and time_step.

        Parameters
        ----------
        full_att_4d
            [B,1,Ls,Lp+Ls] additive attention mask.
        rope_emb_sin
            [B, n_steps, 1, d/2] RoPE sine embeddings.
        rope_emb_cos
            [B, n_steps, 1, d/2] RoPE cosine embeddings.
        x_t
            [B, n_steps, state_dim] current noisy actions.
        time_step
            [B] values in [0,1] for each batch item.
        **kv_cache_kwargs
            Should have keys {key_cache_l{i}, value_cache_l{i}} for
            i = 0,..., 17.

        Returns
        -------
        velocity : torch.Tensor
            Float tensor [B, n_steps, max_action_dim] velocity field v_t.
        """
        # Normalize cache inputs into ordered K/V lists.
        k_list, v_list = kv_args_to_lists(**kv_cache_kwargs)

        adarms_cond = self.lookup_adarms_cond(time_step)

        return self.expert_forward(
            full_att_4d=full_att_4d,
            rope_emb_sin=rope_emb_sin,
            rope_emb_cos=rope_emb_cos,
            k_caches=k_list,
            v_caches=v_list,
            x_t=x_t,
            adarms_cond=adarms_cond,
        )

    def forward(
        self,
        full_att_4d: torch.Tensor,  # [B,1,Ls,Lp+Ls]
        rope_emb_sin: torch.Tensor,  # [B, n_steps, 1, d/2]
        rope_emb_cos: torch.Tensor,  # [B, n_steps, 1, d/2]
        x_t: torch.Tensor,  # [B, n_steps, state_dim]
        time_step: torch.Tensor,  # [B] values in [0,1]
        key_cache_l0: torch.Tensor,
        key_cache_l1: torch.Tensor,
        key_cache_l2: torch.Tensor,
        key_cache_l3: torch.Tensor,
        key_cache_l4: torch.Tensor,
        key_cache_l5: torch.Tensor,
        key_cache_l6: torch.Tensor,
        key_cache_l7: torch.Tensor,
        key_cache_l8: torch.Tensor,
        key_cache_l9: torch.Tensor,
        key_cache_l10: torch.Tensor,
        key_cache_l11: torch.Tensor,
        key_cache_l12: torch.Tensor,
        key_cache_l13: torch.Tensor,
        key_cache_l14: torch.Tensor,
        key_cache_l15: torch.Tensor,
        key_cache_l16: torch.Tensor,
        key_cache_l17: torch.Tensor,
        value_cache_l0: torch.Tensor,
        value_cache_l1: torch.Tensor,
        value_cache_l2: torch.Tensor,
        value_cache_l3: torch.Tensor,
        value_cache_l4: torch.Tensor,
        value_cache_l5: torch.Tensor,
        value_cache_l6: torch.Tensor,
        value_cache_l7: torch.Tensor,
        value_cache_l8: torch.Tensor,
        value_cache_l9: torch.Tensor,
        value_cache_l10: torch.Tensor,
        value_cache_l11: torch.Tensor,
        value_cache_l12: torch.Tensor,
        value_cache_l13: torch.Tensor,
        value_cache_l14: torch.Tensor,
        value_cache_l15: torch.Tensor,
        value_cache_l16: torch.Tensor,
        value_cache_l17: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity v_t and return the integrated x_{t+dt} using a
        fixed Euler step where dt = -1.0 / num_integration_steps.

        All input args are listed explicitly as required by torch.export.

        Parameters
        ----------
        full_att_4d
            Additive attention mask with shape [B, 1, Ls, Lp+Ls].
        rope_emb_sin
            RoPE sine terms for suffix, [B, n_steps, 1, d/2].
        rope_emb_cos
            RoPE cosine terms for suffix, [B, n_steps, 1, d/2].
        x_t
            Current state, [B, n_steps, state_dim].
        time_step
            Time in [0, 1] for each batch item, [B].
        key_cache_l0
            Key cache for layer 0, [B, src_len, 1, head_dim].
        key_cache_l1
            Key cache for layer 1, [B, src_len, 1, head_dim].
        key_cache_l2
            Key cache for layer 2, [B, src_len, 1, head_dim].
        key_cache_l3
            Key cache for layer 3, [B, src_len, 1, head_dim].
        key_cache_l4
            Key cache for layer 4, [B, src_len, 1, head_dim].
        key_cache_l5
            Key cache for layer 5, [B, src_len, 1, head_dim].
        key_cache_l6
            Key cache for layer 6, [B, src_len, 1, head_dim].
        key_cache_l7
            Key cache for layer 7, [B, src_len, 1, head_dim].
        key_cache_l8
            Key cache for layer 8, [B, src_len, 1, head_dim].
        key_cache_l9
            Key cache for layer 9, [B, src_len, 1, head_dim].
        key_cache_l10
            Key cache for layer 10, [B, src_len, 1, head_dim].
        key_cache_l11
            Key cache for layer 11, [B, src_len, 1, head_dim].
        key_cache_l12
            Key cache for layer 12, [B, src_len, 1, head_dim].
        key_cache_l13
            Key cache for layer 13, [B, src_len, 1, head_dim].
        key_cache_l14
            Key cache for layer 14, [B, src_len, 1, head_dim].
        key_cache_l15
            Key cache for layer 15, [B, src_len, 1, head_dim].
        key_cache_l16
            Key cache for layer 16, [B, src_len, 1, head_dim].
        key_cache_l17
            Key cache for layer 17, [B, src_len, 1, head_dim].
        value_cache_l0
            Value cache for layer 0, [B, src_len, 1, head_dim].
        value_cache_l1
            Value cache for layer 1, [B, src_len, 1, head_dim].
        value_cache_l2
            Value cache for layer 2, [B, src_len, 1, head_dim].
        value_cache_l3
            Value cache for layer 3, [B, src_len, 1, head_dim].
        value_cache_l4
            Value cache for layer 4, [B, src_len, 1, head_dim].
        value_cache_l5
            Value cache for layer 5, [B, src_len, 1, head_dim].
        value_cache_l6
            Value cache for layer 6, [B, src_len, 1, head_dim].
        value_cache_l7
            Value cache for layer 7, [B, src_len, 1, head_dim].
        value_cache_l8
            Value cache for layer 8, [B, src_len, 1, head_dim].
        value_cache_l9
            Value cache for layer 9, [B, src_len, 1, head_dim].
        value_cache_l10
            Value cache for layer 10, [B, src_len, 1, head_dim].
        value_cache_l11
            Value cache for layer 11, [B, src_len, 1, head_dim].
        value_cache_l12
            Value cache for layer 12, [B, src_len, 1, head_dim].
        value_cache_l13
            Value cache for layer 13, [B, src_len, 1, head_dim].
        value_cache_l14
            Value cache for layer 14, [B, src_len, 1, head_dim].
        value_cache_l15
            Value cache for layer 15, [B, src_len, 1, head_dim].
        value_cache_l16
            Value cache for layer 16, [B, src_len, 1, head_dim].
        value_cache_l17
            Value cache for layer 17, [B, src_len, 1, head_dim].

        Returns
        -------
        x_next : torch.Tensor
            Updated state x_{t+dt}, [B, n_steps, state_dim].
        """
        kv_cache_kwargs: dict[str, torch.Tensor] = {
            "key_cache_l0": key_cache_l0,
            "key_cache_l1": key_cache_l1,
            "key_cache_l2": key_cache_l2,
            "key_cache_l3": key_cache_l3,
            "key_cache_l4": key_cache_l4,
            "key_cache_l5": key_cache_l5,
            "key_cache_l6": key_cache_l6,
            "key_cache_l7": key_cache_l7,
            "key_cache_l8": key_cache_l8,
            "key_cache_l9": key_cache_l9,
            "key_cache_l10": key_cache_l10,
            "key_cache_l11": key_cache_l11,
            "key_cache_l12": key_cache_l12,
            "key_cache_l13": key_cache_l13,
            "key_cache_l14": key_cache_l14,
            "key_cache_l15": key_cache_l15,
            "key_cache_l16": key_cache_l16,
            "key_cache_l17": key_cache_l17,
            "value_cache_l0": value_cache_l0,
            "value_cache_l1": value_cache_l1,
            "value_cache_l2": value_cache_l2,
            "value_cache_l3": value_cache_l3,
            "value_cache_l4": value_cache_l4,
            "value_cache_l5": value_cache_l5,
            "value_cache_l6": value_cache_l6,
            "value_cache_l7": value_cache_l7,
            "value_cache_l8": value_cache_l8,
            "value_cache_l9": value_cache_l9,
            "value_cache_l10": value_cache_l10,
            "value_cache_l11": value_cache_l11,
            "value_cache_l12": value_cache_l12,
            "value_cache_l13": value_cache_l13,
            "value_cache_l14": value_cache_l14,
            "value_cache_l15": value_cache_l15,
            "value_cache_l16": value_cache_l16,
            "value_cache_l17": value_cache_l17,
        }

        v_t = self._compute_update(
            full_att_4d=full_att_4d,
            rope_emb_sin=rope_emb_sin,
            rope_emb_cos=rope_emb_cos,
            x_t=x_t,
            time_step=time_step,
            **kv_cache_kwargs,
        )

        dt = -1.0 / float(self.num_integration_steps)
        return (
            x_t
            + torch.as_tensor(
                dt,
                dtype=torch.float32,
                device=x_t.device,
            )
            * v_t
        )

    @classmethod
    def get_input_spec_static(
        cls,
        batch_size: int = 1,
    ) -> InputSpec:
        num_layers = 18
        assert NUM_CAMERAS is not None
        src_len = 256 * NUM_CAMERAS + MAX_TOKEN_LENGTH
        head_dim = 256
        state_dim = 32
        ls_len = NUM_ACTION_STEPS
        lt_len = src_len + ls_len
        spec: InputSpec = dict(
            full_att_4d=TensorSpec(
                shape=(batch_size, 1, ls_len, lt_len),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            rope_emb_sin=TensorSpec(
                shape=(batch_size, NUM_ACTION_STEPS, 1, head_dim // 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            rope_emb_cos=TensorSpec(
                shape=(batch_size, NUM_ACTION_STEPS, 1, head_dim // 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            x_t=TensorSpec(
                shape=(batch_size, NUM_ACTION_STEPS, state_dim),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            # Replace external adarms_cond with scalar time_step in [0, 1].
            time_step=TensorSpec(
                shape=(batch_size,),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        )
        # Add 18 K caches then 18 V caches.
        for i in range(num_layers):
            spec[f"key_cache_l{i}"] = TensorSpec(
                shape=(batch_size, src_len, 1, head_dim),
                dtype="float32",
                io_type=IoType.TENSOR,
            )
        for i in range(num_layers):
            spec[f"value_cache_l{i}"] = TensorSpec(
                shape=(batch_size, src_len, 1, head_dim),
                dtype="float32",
                io_type=IoType.TENSOR,
            )
        return spec

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return self.get_input_spec_static(batch_size)

    @staticmethod
    def get_output_spec_static() -> OutputSpec:
        return {"action_emb": TensorSpec()}

    def get_output_spec(self) -> OutputSpec:
        return self.get_output_spec_static()


class Pi05PaliGemmaBackboneBase(LoadPolicyMixin, BaseModel):
    """
    Runs the PaliGemma expert to fill the key/value cache over the specified
    layer_range.
    """

    def __init__(
        self,
        model: PI05Policy,
        layer_range: tuple[int, int],  # start (incl), end (excl)
        max_mlp_dim: int = 2048,
        return_hidden_state: bool = True,
    ) -> None:
        """
        Initialize the Pi05PaliGemmaBackboneBase.

        Parameters
        ----------
        model
            Policy wrapper containing the flow model and PaliGemma expert.
        layer_range
            Start (inclusive) and end (exclusive) indices of language
            layers to target.
        max_mlp_dim
            Maximum dimension allowed for a single Linear projection
            inside the MLP. If an MLP has a projection layer larger
            than this (e.g., 16384), it will be automatically replaced
            with a chunked version composed of smaller Linear layers,
            each up to max_mlp_dim. This makes the model more suitable
            for on-device ML where very large Linear layers are not
            practical.
        return_hidden_state
            True to return the final activation. Not
            needed for the last layer if we only need kv cache.
        """
        flow_model: PI05Pytorch = model.model.to(torch.float32)
        super().__init__(flow_model)
        self.layer_range = layer_range
        self.return_hidden_state = return_hidden_state

        # Only keep references to the targeted language layers. This
        # avoids touching gemma_expert or any other irrelevant modules.
        pg_we = flow_model.paligemma_with_expert
        text_layers = pg_we.paligemma.language_model.layers
        start, end = self.layer_range
        end = min(end, len(text_layers))
        self.target_layers = torch.nn.ModuleList(
            [text_layers[i] for i in range(start, end)]
        )

        # Replace large MLPs with chunked versions to limit per-linear
        # dims to max_mlp_dim (e.g., split 16384 into 4096 chunks).

        for lyr in self.target_layers:
            if isinstance(lyr.mlp, GemmaMLP):
                lyr.mlp = GemmaMLPSplitLinear(lyr.mlp, max_mlp_dim=max_mlp_dim)

        self.text_cfg = pg_we.paligemma.config.text_config

    def forward(
        self,
        hidden_state: torch.Tensor,  # [B, src_len, D]
        prefix_att_2d_masks: torch.Tensor,  # [B,1,src_len,src_len] or [B,L,L]
        rope_emb_sin: torch.Tensor,  # [B, src_len, 1, D/2]
        rope_emb_cos: torch.Tensor,  # [B, src_len, 1, D/2]
    ) -> tuple[torch.Tensor, ...]:
        """
        Mirror eager attention for the selected layers. We compute Q/K/V,
        apply RoPE to Q and K, apply the additive attention mask,
        softmax, and get attention output for the prefix tokens only.

        Returns (flattened tuple, no nested lists):
          (
            hidden_states_out,
            k_cache_l<start>, ..., k_cache_l<end-1>,
            v_cache_l<start>, ..., v_cache_l<end-1>,
          )
        where each K/V cache has shape [B, src_len, H_kv, D].
        """
        batch_size, src_len, _ = hidden_state.shape
        num_att_heads = self.text_cfg.num_attention_heads
        num_kv_heads = self.text_cfg.num_key_value_heads

        key_states_per_layer: list[torch.Tensor] = []
        value_states_per_layer: list[torch.Tensor] = []

        hidden_states = hidden_state.to(torch.float32)

        # Allow either [B,1,L,L] or [B,L,L] for the additive mask.
        if prefix_att_2d_masks.ndim == 4:
            att_mask_2d = prefix_att_2d_masks[:, 0]
        else:
            att_mask_2d = prefix_att_2d_masks
        # att_mask_2d: [B, L, L], additive (0 allowed, -1e4 blocked)
        for layer in self.target_layers:
            assert isinstance(layer, GemmaDecoderLayer)
            # Input norm
            normed, gate = layer.input_layernorm(hidden_states)
            assert gate is None

            # Project Q/K/V in float32 and reshape to [B, L, H, D]
            head_dim = layer.self_attn.head_dim
            assert isinstance(head_dim, int)
            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, head_dim)

            q_state = layer.self_attn.q_proj(normed).view(hidden_shape)
            k_state = layer.self_attn.k_proj(normed).view(hidden_shape)
            v_state = layer.self_attn.v_proj(normed).view(hidden_shape)

            # Apply RoPE to Q and K using precomputed sin/cos.
            q_state = apply_rope_direct(q_state, rope_emb_sin, rope_emb_cos)
            k_state = apply_rope_direct(k_state, rope_emb_sin, rope_emb_cos)

            # Cache stores K/V in their native [B, L, H_kv, D] form.
            key_states_per_layer.append(k_state.to(torch.float32))
            value_states_per_layer.append(v_state.to(torch.float32))

            # Expand K/V heads to full attention heads as in eager path.
            groups = num_att_heads // num_kv_heads
            k_exp = (
                k_state[:, :, :, None, :]
                .expand(batch_size, src_len, num_kv_heads, groups, head_dim)
                .reshape(batch_size, src_len, num_kv_heads * groups, head_dim)
            )
            v_exp = (
                v_state[:, :, :, None, :]
                .expand(batch_size, src_len, num_kv_heads, groups, head_dim)
                .reshape(batch_size, src_len, num_kv_heads * groups, head_dim)
            )

            # Compute attention in float32 like eager implementation.
            q_mat = q_state.to(torch.float32).transpose(1, 2)  # [B, H, L, D]
            k_mat = k_exp.to(torch.float32).transpose(1, 2)  # [B, H, L, D]

            att_weights = torch.matmul(q_mat, k_mat.transpose(2, 3))
            att_weights *= head_dim**-0.5

            # Additive mask: att_mask_2d is [B, L, L], broadcast over H.
            masked = att_weights + att_mask_2d.to(att_weights.dtype)[:, None, :, :]

            probs = torch.nn.functional.softmax(masked, dim=-1)
            probs = probs.to(v_exp.dtype)

            # [B, H, L, L] x [B, L, H, D] -> [B, H, L, D]
            att_output = torch.matmul(probs, v_exp.permute(0, 2, 1, 3))

            # Back to [B, L, H, D] then flatten head dim.
            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(
                batch_size, src_len, num_att_heads * head_dim
            )

            # Output projection and residuals match eager path.
            out_emb = layer.self_attn.o_proj(att_output)
            out_emb = out_emb + hidden_states
            after_first_residual = out_emb.clone()

            out_emb, gate = layer.post_attention_layernorm(out_emb)
            assert gate is None
            out_emb = layer.mlp(out_emb)
            hidden_states = out_emb + after_first_residual

        # Flattened tuple: hidden_out, then all K, then all V.
        flat_out = [hidden_states.to(torch.float32)] if self.return_hidden_state else []
        flat_out.extend(key_states_per_layer)
        flat_out.extend(value_states_per_layer)
        return tuple(flat_out)

    @classmethod
    def get_input_spec_static(
        cls,
        batch_size: int = 1,
    ) -> InputSpec:
        """
        Returns input specification for the backbone layers.

        Parameters
        ----------
        batch_size
            Batch size for the input specification.

        Returns
        -------
        spec : InputSpec
            Dictionary with input names, shapes, and dtypes.
        """
        assert NUM_CAMERAS is not None
        src_len = 256 * NUM_CAMERAS + MAX_TOKEN_LENGTH
        hidden_dim = 2048
        head_dim = 256
        return dict(
            hidden_state=TensorSpec(
                shape=(batch_size, src_len, hidden_dim),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            prefix_att_2d_masks=TensorSpec(
                shape=(batch_size, 1, src_len, src_len),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            rope_emb_sin=TensorSpec(
                shape=(batch_size, src_len, 1, head_dim // 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            rope_emb_cos=TensorSpec(
                shape=(batch_size, src_len, 1, head_dim // 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        )

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        return self.get_input_spec_static(batch_size)

    @staticmethod
    def get_output_spec_static(layer_range: tuple[int, int]) -> OutputSpec:
        # Generic placeholders for 6 layers. Concrete subclasses return
        # exactly 6 K and 6 V caches each, matching the range length.
        names: OutputSpec = {}
        if layer_range[1] != 18:
            names["hidden_state_out"] = TensorSpec()
        layer_ids = range(layer_range[0], layer_range[1])
        names.update({f"k_cache_l{i}": TensorSpec() for i in layer_ids})
        names.update({f"v_cache_l{i}": TensorSpec() for i in layer_ids})
        return names

    def get_output_spec(self) -> OutputSpec:
        return self.get_output_spec_static((0, 6))


class Pi05PaliGemmaBackbone(Pi05PaliGemmaBackboneBase):
    def __init__(self, policy: PI05Policy) -> None:
        super().__init__(policy, layer_range=(0, 18), return_hidden_state=False)


class Pi05PaliGemmaBackboneQuantizable(
    _Pi05CachedExportMixin, AIMETOnnxQuantizableMixin, Pi05PaliGemmaBackbone
):
    """Exportable PaliGemma full backbone that can be quantized by AIMET-ONNX."""

    default_subfolder = "backbone"
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION

    def component_precision(self) -> Precision:
        return Precision.w4a16

    def __init__(
        self,
        sim_model: QuantSimOnnx | None,
        host_device: torch.device = torch.device("cpu"),
        onnx_bundle: ONNXBundle | None = None,
        precision: Precision = Precision.w4a16,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model, onnx_bundle=onnx_bundle)
        BaseModel.__init__(self, None)
        self.host_device = host_device
        self._precision = precision

    def make_quant_sim(self) -> QuantSimOnnx | None:
        if self._onnx_bundle is None:
            return None
        param_type, act_type = aimet_quant_types(self._precision)

        onnx_model = self._onnx_bundle.load_onnx_model()
        quant_sim = QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=param_type,
            activation_type=act_type,
            config_file=_PI05_AIMET_CONFIG,
            providers=AIMETOnnxQuantizableMixin.get_ort_providers(self.host_device),
        )
        _set_matmul_second_input_to_8b(quant_sim)
        return quant_sim

    @classmethod
    def torch_from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
    ) -> torch.nn.Module:
        policy = load_checkpoint(str(checkpoint))
        return Pi05PaliGemmaBackbone(policy).to(host_device).eval()

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        precision: Precision = Precision.w4a16,
        torch_from_pretrained_kwargs: dict[str, Any] | None = None,
        cls_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        bundle = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )
        return cls(
            None, host_device=host_device, onnx_bundle=bundle, precision=precision
        )

    def forward(  # type: ignore[override]
        self,
        hidden_state: torch.Tensor,
        prefix_att_2d_masks: torch.Tensor,
        rope_emb_sin: torch.Tensor,
        rope_emb_cos: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return cast(
            tuple[torch.Tensor, ...],
            AIMETOnnxQuantizableMixin.forward(
                self, hidden_state, prefix_att_2d_masks, rope_emb_sin, rope_emb_cos
            ),
        )

    def get_output_spec(self) -> OutputSpec:
        return Pi05PaliGemmaBackboneBase.get_output_spec_static((0, 18))


class Pi05ActionExpertQuantizable(
    _Pi05CachedExportMixin, AIMETOnnxQuantizableMixin, Pi05ActionExpert
):
    """Exportable Pi05 action expert that can be quantized by AIMET-ONNX."""

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_subfolder = "action_expert"

    def component_precision(self) -> Precision:
        return Precision.w8a16

    def __init__(
        self,
        sim_model: QuantSimOnnx | None,
        host_device: torch.device = torch.device("cpu"),
        onnx_bundle: ONNXBundle | None = None,
        precision: Precision = Precision.w8a16,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model, onnx_bundle=onnx_bundle)
        BaseModel.__init__(self, None)
        self.host_device = host_device
        self._precision = precision

    def make_quant_sim(self) -> QuantSimOnnx | None:
        if self._onnx_bundle is None:
            return None
        param_type, act_type = aimet_quant_types(self._precision)

        onnx_model = self._onnx_bundle.load_onnx_model()
        quant_sim = QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=param_type,
            activation_type=act_type,
            config_file=_PI05_AIMET_CONFIG,
            providers=AIMETOnnxQuantizableMixin.get_ort_providers(self.host_device),
        )
        kv_inputs = [
            t.name
            for t in onnx_model.graph.input
            if "key_cache" in t.name or "value_cache" in t.name
        ]
        _set_tensors_to_output_8b_sym(quant_sim, kv_inputs)
        _set_matmul_second_input_to_8b(quant_sim)
        return quant_sim

    @classmethod
    def torch_from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
    ) -> torch.nn.Module:
        policy = load_checkpoint(str(checkpoint))
        return Pi05ActionExpert(policy).to(host_device).eval()

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        precision: Precision = Precision.w8a16,
        torch_from_pretrained_kwargs: dict[str, Any] | None = None,
        cls_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        host_device = torch.device(host_device)
        ckpt_type = CheckpointType.from_checkpoint(checkpoint, subfolder="")
        if ckpt_type == CheckpointType.AIMET_ONNX_EXPORT:
            bundle = ONNXBundle.from_bundle_path(Path(str(checkpoint)))
        else:
            subfolder = subfolder or cls.default_subfolder
            bundle = cls.onnx_from_pretrained(
                checkpoint=checkpoint,
                subfolder=subfolder,
                host_device=host_device,
                torch_to_onnx_options={"opset_version": 20},
            )
        return cls(
            None, host_device=host_device, onnx_bundle=bundle, precision=precision
        )

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        if self._quant_sim is None and self._onnx_bundle is not None:
            class_name = self.__class__.__name__
            out_dir = Path(output_dir) / f"{class_name}.aimet"
            if (out_dir / "model.onnx").exists():
                return out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            self._onnx_bundle.move(
                dst_folder=str(out_dir),
                dst_model_name=class_name,
                copy=True,
            )
            return out_dir
        return _Pi05CachedExportMixin.serialize(
            self,
            output_dir,
            input_spec,
        )

    def forward(  # type: ignore[override]
        self,
        full_att_4d: torch.Tensor,
        rope_emb_sin: torch.Tensor,
        rope_emb_cos: torch.Tensor,
        x_t: torch.Tensor,
        time_step: torch.Tensor,
        key_cache_l0: torch.Tensor,
        key_cache_l1: torch.Tensor,
        key_cache_l2: torch.Tensor,
        key_cache_l3: torch.Tensor,
        key_cache_l4: torch.Tensor,
        key_cache_l5: torch.Tensor,
        key_cache_l6: torch.Tensor,
        key_cache_l7: torch.Tensor,
        key_cache_l8: torch.Tensor,
        key_cache_l9: torch.Tensor,
        key_cache_l10: torch.Tensor,
        key_cache_l11: torch.Tensor,
        key_cache_l12: torch.Tensor,
        key_cache_l13: torch.Tensor,
        key_cache_l14: torch.Tensor,
        key_cache_l15: torch.Tensor,
        key_cache_l16: torch.Tensor,
        key_cache_l17: torch.Tensor,
        value_cache_l0: torch.Tensor,
        value_cache_l1: torch.Tensor,
        value_cache_l2: torch.Tensor,
        value_cache_l3: torch.Tensor,
        value_cache_l4: torch.Tensor,
        value_cache_l5: torch.Tensor,
        value_cache_l6: torch.Tensor,
        value_cache_l7: torch.Tensor,
        value_cache_l8: torch.Tensor,
        value_cache_l9: torch.Tensor,
        value_cache_l10: torch.Tensor,
        value_cache_l11: torch.Tensor,
        value_cache_l12: torch.Tensor,
        value_cache_l13: torch.Tensor,
        value_cache_l14: torch.Tensor,
        value_cache_l15: torch.Tensor,
        value_cache_l16: torch.Tensor,
        value_cache_l17: torch.Tensor,
    ) -> torch.Tensor:
        if self._quant_sim is None and self._onnx_bundle is not None:
            if not hasattr(self, "_ort_session"):
                self._ort_session = onnxruntime.InferenceSession(
                    str(self._onnx_bundle.onnx_graph_path),
                    providers=["CPUExecutionProvider"],
                )
            return cast(
                torch.Tensor,
                mock_torch_onnx_inference(
                    self._ort_session,
                    full_att_4d,
                    rope_emb_sin,
                    rope_emb_cos,
                    x_t,
                    time_step,
                    key_cache_l0,
                    key_cache_l1,
                    key_cache_l2,
                    key_cache_l3,
                    key_cache_l4,
                    key_cache_l5,
                    key_cache_l6,
                    key_cache_l7,
                    key_cache_l8,
                    key_cache_l9,
                    key_cache_l10,
                    key_cache_l11,
                    key_cache_l12,
                    key_cache_l13,
                    key_cache_l14,
                    key_cache_l15,
                    key_cache_l16,
                    key_cache_l17,
                    value_cache_l0,
                    value_cache_l1,
                    value_cache_l2,
                    value_cache_l3,
                    value_cache_l4,
                    value_cache_l5,
                    value_cache_l6,
                    value_cache_l7,
                    value_cache_l8,
                    value_cache_l9,
                    value_cache_l10,
                    value_cache_l11,
                    value_cache_l12,
                    value_cache_l13,
                    value_cache_l14,
                    value_cache_l15,
                    value_cache_l16,
                    value_cache_l17,
                ),
            )
        return cast(
            torch.Tensor,
            AIMETOnnxQuantizableMixin.forward(
                self,
                full_att_4d,
                rope_emb_sin,
                rope_emb_cos,
                x_t,
                time_step,
                key_cache_l0,
                key_cache_l1,
                key_cache_l2,
                key_cache_l3,
                key_cache_l4,
                key_cache_l5,
                key_cache_l6,
                key_cache_l7,
                key_cache_l8,
                key_cache_l9,
                key_cache_l10,
                key_cache_l11,
                key_cache_l12,
                key_cache_l13,
                key_cache_l14,
                key_cache_l15,
                key_cache_l16,
                key_cache_l17,
                value_cache_l0,
                value_cache_l1,
                value_cache_l2,
                value_cache_l3,
                value_cache_l4,
                value_cache_l5,
                value_cache_l6,
                value_cache_l7,
                value_cache_l8,
                value_cache_l9,
                value_cache_l10,
                value_cache_l11,
                value_cache_l12,
                value_cache_l13,
                value_cache_l14,
                value_cache_l15,
                value_cache_l16,
                value_cache_l17,
            ),
        )


class _Pi05LiberoCalibrationMixin:
    """Shared calibration-dataset hook for the Pi05 collections."""

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return LiberoDataset


class Pi05Collection(_Pi05LiberoCalibrationMixin, WorkbenchModelCollection):
    """
    Float (non-quantized) Pi05 collection. Its components are plain torch
    modules that run float forward passes, so Pi05App.get_calibration_data can
    invoke them directly to build calibration inputs for the quantizable
    components. All components share the lru_cached policy from load_checkpoint,
    so no duplicate float weights are materialized.
    """

    def __init__(
        self,
        vision_encoder: Pi05PaliGemmaVision,
        token_emb: Pi05PaliGemmaTokenEmbed,
        action_expert: Pi05ActionExpert,
        backbone: Pi05PaliGemmaBackbone,
    ) -> None:
        super().__init__(
            {
                "vision_encoder": vision_encoder,
                "token_emb": token_emb,
                "action_expert": action_expert,
                "backbone": backbone,
            }
        )

    def get_input_spec(self, batch_size: int = 1) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(batch_size=batch_size)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "DEFAULT",
        host_device: torch.device | str = torch.device("cpu"),
    ) -> Pi05Collection:
        return cls(
            Pi05PaliGemmaVision.from_pretrained(
                checkpoint=checkpoint, host_device=host_device
            ),
            Pi05PaliGemmaTokenEmbed.from_pretrained(
                checkpoint=checkpoint, host_device=host_device
            ),
            Pi05ActionExpert.from_pretrained(
                checkpoint=checkpoint, host_device=host_device
            ),
            Pi05PaliGemmaBackbone.from_pretrained(
                checkpoint=checkpoint, host_device=host_device
            ),
        )


class Pi05CollectionQuantized(_Pi05LiberoCalibrationMixin, WorkbenchModelCollection):
    def __init__(
        self,
        vision_encoder: Pi05PaliGemmaVisionQuantizable,
        token_emb: Pi05PaliGemmaTokenEmbed,
        action_expert: Pi05ActionExpertQuantizable,
        backbone: Pi05PaliGemmaBackboneQuantizable,
    ) -> None:
        super().__init__(
            {
                "vision_encoder": vision_encoder,
                "token_emb": token_emb,
                "action_expert": action_expert,
                "backbone": backbone,
            }
        )

    def get_input_spec(self, batch_size: int = 1) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(batch_size=batch_size)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "DEFAULT",
        host_device: torch.device | str = torch.device("cpu"),
    ) -> Pi05CollectionQuantized:
        return cls(
            Pi05PaliGemmaVisionQuantizable.from_pretrained(
                checkpoint=checkpoint, host_device=host_device
            ),
            Pi05PaliGemmaTokenEmbed.from_pretrained(
                checkpoint=checkpoint, host_device=host_device
            ),
            Pi05ActionExpertQuantizable.from_pretrained(
                checkpoint=checkpoint, host_device=host_device
            ),
            Pi05PaliGemmaBackboneQuantizable.from_pretrained(
                checkpoint=checkpoint, host_device=host_device
            ),
        )

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import pickle
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import aimet_onnx
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
from aimet_onnx.quantsim import load_encodings_to_sim
from einops import rearrange
from peft import PeftModel
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.waft_stereo.dataset import (
    CocoStereoCalibrationDataset,
    Kitti2015Dataset,
)
from qai_hub_models.models.waft_stereo.evaluator import WaftStereoD1Evaluator
from qai_hub_models.models.waft_stereo.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.waft_stereo.external_repos.waft_stereo.algorithms.waft import (
    WAFT,
)
from qai_hub_models.models.waft_stereo.external_repos.waft_stereo.model.utils import (
    Padder,
    disp_warp,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    qaihm_temp_dir,
    repo_in_sys_path,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.input_spec import (
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
    get_batch_size,
)
from qai_hub_models.utils.onnx.helpers import ONNXBundle, safe_torch_onnx_export
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries
from qai_hub_models.utils.quantization_aimet_onnx import (
    AIMETOnnxQuantizableMixin,
    aimet_quant_types,
    ensure_min_aimet_onnx_version,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Apache-2.0-clean "vits" DAv2 backbone checkpoint; tagged bsd-3-clause on HF
# (matches the repo license, not the backbone's own pretrained weights).
# Pinned to a commit rather than `main`, which is mutable and could later
# point at different weights.
DEFAULT_WEIGHTS = CachedWebModelAsset(
    "https://huggingface.co/MemorySlices/WAFT-Stereo/resolve/"
    "dfcf8ac0b65966aaaec86abb682030ae6ed66a8e/SynLarge/DAv2S-4.pth",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "DAv2S-4.pth",
)


# The only non-tensor globals this checkpoint's pickle stream references, per pickletools.
# torch's `weights_only=True` can't substitute: its restricted unpickler rejects the
# embedded `bridgedepth.config.config.CfgNode` even once allowlisted via `safe_globals`.
_SAFE_CHECKPOINT_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("builtins", "set"),
    ("__builtin__", "set"),  # legacy Python 2 module name for `set`, as pickled
    ("bridgedepth.config.config", "CfgNode"),
}


class _RestrictedCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _SAFE_CHECKPOINT_GLOBALS:
            raise pickle.UnpicklingError(f"Blocked unpickling of {module}.{name}")
        return super().find_class(module, name)


_RESTRICTED_PICKLE_MODULE = types.ModuleType("_waft_stereo_restricted_pickle")
_RESTRICTED_PICKLE_MODULE.Unpickler = _RestrictedCheckpointUnpickler  # type: ignore[attr-defined]
_RESTRICTED_PICKLE_MODULE.load = pickle.load  # type: ignore[attr-defined]


def _load_checkpoint_weights(weights_path: str | None) -> Mapping[str, torch.Tensor]:
    """Load the WAFT checkpoint's state dict via a GLOBAL-opcode allowlist.

    Applies to both DEFAULT_WEIGHTS (fetched from the third-party HF repo
    MemorySlices/WAFT-Stereo) and any caller-supplied weights_path - neither
    ever uses ``weights_only=False``.
    """
    ckpt_path = weights_path or str(DEFAULT_WEIGHTS.fetch())
    # The checkpoint pickles a config object defined in the vendored repo's
    # own top-level package; it must be importable under that name to unpickle.
    with repo_in_sys_path(str(EXTERNAL_REPO_PATHS["waft_stereo"])):
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            pickle_module=_RESTRICTED_PICKLE_MODULE,
            weights_only=False,
        )
    return checkpoint.get("model", checkpoint)


def _build_cfg() -> SimpleNamespace:
    """Mirrors configs/SynLarge/DAv2S-4.yaml from the upstream repo exactly."""
    delta_iter = SimpleNamespace(
        TYPE="vit", ARCH="vits", PATCH_SIZE=8, LORA_RANK=None, LORA_ALPHA=None
    )
    prop_iter = SimpleNamespace(
        TYPE="vit", ARCH="vits", PATCH_SIZE=8, LORA_RANK=None, LORA_ALPHA=None
    )
    return SimpleNamespace(
        WAFT=SimpleNamespace(
            MAX_DISP=800,
            LOSS=["mixlap_40"],
            FEATURE_ENCODER=SimpleNamespace(
                TYPE="dav2", ARCH="vits", LORA_RANK=8, LORA_ALPHA=16
            ),
            ITERATIVE_MODULE=SimpleNamespace(
                TASK=["delta", "delta", "delta"],
                DELTA_ITER=delta_iter,
                PROP_ITER=prop_iter,
            ),
        )
    )


def load_source_model(weights_path: str | None = None) -> WAFT:
    """Build the vendored WAFT model and load its checkpoint.

    LoRA adapters are left unmerged; ``from_pretrained`` merges them separately.
    """
    net = WAFT(_build_cfg())
    weights = _load_checkpoint_weights(weights_path)
    missing_keys, _ = net.load_state_dict(weights, strict=False)
    assert not missing_keys, f"Checkpoint is missing keys: {missing_keys}"
    return net


_QUANT_OPSET = 17

# Native resolution of the released SynLarge/DAv2S-4 checkpoint.
_NET_HEIGHT = 240
_NET_WIDTH = 320


def _export_component_onnx(component: BaseModel) -> onnx.ModelProto:
    """Export a float component to an in-memory ONNX graph for quantsim construction."""
    input_spec = component.get_input_spec()
    dummy_inputs = tuple(
        torch.zeros(shape, dtype=getattr(torch, dtype))
        for shape, dtype in input_spec.values()
    )
    with qaihm_temp_dir() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        safe_torch_onnx_export(
            component,
            dummy_inputs,
            str(onnx_path),
            input_names=list(input_spec.keys()),
            output_names=list(component.get_output_spec().keys()),
            opset_version=_QUANT_OPSET,
            dynamo=False,
        )
        return onnx.load(str(onnx_path))


def _load_calibrated_onnx_bundle(
    checkpoint_dir: str | None, subfolder: str
) -> ONNXBundle | None:
    """Look up a pre-calibrated `<checkpoint_dir>/<subfolder>/model.onnx` saved by
    quantize.py's `save_calibrated_checkpoint`, to skip SeqMSE recalibration.
    """
    if checkpoint_dir is None:
        return None
    bundle_path = Path(checkpoint_dir) / subfolder
    if not (bundle_path / "model.onnx").exists():
        return None
    return ONNXBundle.from_bundle_path(bundle_path)


def _make_component_quant_sim(
    net: WAFT,
    onnx_bundle: ONNXBundle | None,
    float_cls: type[BaseModel],
    precision: Precision,
    dedupe_shared_initializers: bool = False,
) -> QuantSimOnnx:
    """Shared by both quantizable components' ``make_quant_sim``: loads a
    pre-calibrated bundle if given, else exports the float component fresh.
    """
    ensure_min_aimet_onnx_version("2.40.0", model_id=MODEL_ID)
    if onnx_bundle is not None:
        onnx_model = onnx_bundle.load_onnx_model()
    else:
        onnx_model = _export_component_onnx(float_cls(net))
        if dedupe_shared_initializers:
            # The two batch-1 img1/img2 ViT passes share weight initializers,
            # which QuantizationSimModel rejects (one consumer per initializer).
            aimet_onnx.utils.duplicate_shared_initializers(onnx_model.graph)
    param_type, activation_type = aimet_quant_types(precision)
    sim = QuantSimOnnx(
        model=onnx_model,
        quant_scheme=QuantScheme.post_training_tf,
        param_type=param_type,
        activation_type=activation_type,
    )
    if onnx_bundle is not None:
        assert onnx_bundle.aimet_encodings_path is not None
        load_encodings_to_sim(sim, str(onnx_bundle.aimet_encodings_path))
    return sim


class WaftStereoEncoder(BaseModel):
    """WAFT-Stereo stage 1: feature encoder (backbone) only."""

    def __init__(self, net: WAFT) -> None:
        super().__init__()
        # Intentionally the same WAFT instance across all 3 components - each
        # wraps a different subset of its forward graph, not a copy error.
        self.net = net

    @classmethod
    def from_pretrained(
        cls, weights_path: str | None = None, precision: Precision = Precision.float
    ) -> Self:
        return cast(Self, WaftStereo.from_pretrained(weights_path, precision).encoder)

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        img1
            Float32 RGB [0, 1] tensor of shape ``[B, 3, H, W]`` (left).
        img2
            Float32 RGB [0, 1] tensor of shape ``[B, 3, H, W]`` (right).

        Returns
        -------
        fmap1 : torch.Tensor
            Left-view encoder feature, ``[B, C, H//2, W//2]``.
        fmap2 : torch.Tensor
            Right-view encoder feature, ``[B, C, H//2, W//2]``.
        net : torch.Tensor
            Decoder hidden state, ``[B, C, H//2, W//2]``.
        """
        net = self.net
        image1 = normalize_image_torchvision(img1).contiguous()
        image2 = normalize_image_torchvision(img2).contiguous()
        padder = Padder(image1.shape, factor=net.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)

        # Replicates DAv2Encoder.forward (dav2.py) as two batch-1 ViT calls
        # instead of one batch-2 call, to avoid blowing the on-device memory budget.
        dav2 = net.encoder
        _, _, h_pad, w_pad = image1.shape
        h, w = h_pad // 16, w_pad // 16
        img1_r = F.interpolate(
            image1, (h * 14, w * 14), mode="bilinear", align_corners=True
        )
        img2_r = F.interpolate(
            image2, (h * 14, w * 14), mode="bilinear", align_corners=True
        )
        feats1 = dav2.encoder.get_intermediate_layers(
            img1_r, dav2.idx, return_class_token=True
        )
        feats2 = dav2.encoder.get_intermediate_layers(
            img2_r, dav2.idx, return_class_token=True
        )

        fmap1_feats = [
            rearrange(x[0], "b (h w) c -> b c h w", h=h, w=w) for x in feats1
        ]
        fmap2_feats = [
            rearrange(x[0], "b (h w) c -> b c h w", h=h, w=w) for x in feats2
        ]
        fmap1 = dav2.fmap_upsample(dav2.fmap_proj(fmap1_feats))[0]
        fmap2 = dav2.fmap_upsample(dav2.fmap_proj(fmap2_feats))[0]

        hidden_feats = [
            torch.cat(
                [
                    rearrange(x1[0], "b (h w) c -> b c h w", h=h, w=w),
                    rearrange(x2[0], "b (h w) c -> b c h w", h=h, w=w),
                ],
                dim=1,
            )
            for x1, x2 in zip(feats1, feats2, strict=False)
        ]
        hidden = dav2.hidden_upsample(dav2.hidden_proj(hidden_feats))[0]

        return fmap1, fmap2, hidden

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = _NET_HEIGHT,
        width: int = _NET_WIDTH,
    ) -> InputSpec:
        shape = (batch_size, 3, height, width)
        return {
            "img1": TensorSpec(
                shape=shape,
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
            ),
            "img2": TensorSpec(
                shape=shape,
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "fmap1": TensorSpec(),
            "fmap2": TensorSpec(),
            "net": TensorSpec(),
        }

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoStereoCalibrationDataset

    def component_precision(self) -> Precision:
        return Precision.float


def build_calibration_data(
    collection_model: WorkbenchModelCollection,
    component_name: str,
    input_specs: dict[str, InputSpec] | None = None,
    num_samples: int | None = None,
) -> DatasetEntries:
    """Build calibration data for a WaftStereo component by running the float
    encoder over calibration samples. Shared by the quantizable components'
    eager calibration below and by ``WaftStereoApp.get_calibration_data``
    (the export pipeline's calibration seam), so it lives here rather than
    in app.py - app depends on model, not the reverse.
    """
    encoder = collection_model.components["encoder"]
    enc_spec = (input_specs or {}).get("encoder", encoder.get_input_spec())

    calibration_dataset_cls = encoder.get_calibration_dataset_cls()
    assert calibration_dataset_cls is not None
    dataset = instantiate_dataset(
        calibration_dataset_cls,
        DatasetSplit.TRAIN,
        input_spec=enc_spec,
    )

    input_spec = (
        input_specs[component_name]
        if input_specs
        else collection_model.components[component_name].get_input_spec()
    )
    batch_size = get_batch_size(input_spec) or 1
    num_samples = num_samples or dataset.default_num_calibration_samples()
    num_samples = max((num_samples // batch_size) * batch_size, batch_size)
    print(f"Loading {num_samples} calibration samples.")
    torch_dataset = sample_dataset(dataset, num_samples)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size)

    inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in range(len(input_spec))]
    with torch.no_grad():
        for sample_input, _ in dataloader:
            img1, img2 = sample_input
            if component_name == "encoder":
                sample_input = (img1, img2)
            else:
                fmap1, fmap2, _ = encoder(img1, img2)
                if component_name == "classification":
                    sample_input = (fmap1, fmap2)
                else:
                    raise ValueError(f"Invalid component name: {component_name}")
            for i, tensor in enumerate(sample_input):
                inputs[i].append(tensor)
    return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))


class WaftStereoEncoderQuantizable(  # type: ignore[misc]
    AIMETOnnxQuantizableMixin, WaftStereoEncoder
):
    """AIMET-ONNX-quantized WAFT-Stereo encoder (w8a16, SeqMSE-calibrated)."""

    default_subfolder: str = "encoder"

    def __init__(
        self,
        net: WAFT,
        quant_sim: QuantSimOnnx | None = None,
        onnx_bundle: ONNXBundle | None = None,
    ) -> None:
        WaftStereoEncoder.__init__(self, net)
        AIMETOnnxQuantizableMixin.__init__(self, quant_sim, onnx_bundle=onnx_bundle)
        # Set by WaftStereo.from_pretrained to a plain-float WaftStereo so
        # get_calibration_data can chain through unquantized components.
        self.collection: WaftStereo | None = None

    def component_precision(self) -> Precision:
        return Precision.w8a16

    def get_calibration_data(
        self,
        input_spec: InputSpec | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries | None:
        assert self.collection is not None, "collection must be set before calibration"
        return build_calibration_data(
            self.collection, "encoder", num_samples=num_samples
        )

    def make_quant_sim(self) -> QuantSimOnnx | None:
        return _make_component_quant_sim(
            self.net,
            self._onnx_bundle,
            WaftStereoEncoder,
            Precision.w8a16,
            dedupe_shared_initializers=True,
        )


class WaftStereoClassification(BaseModel):
    """WAFT-Stereo stage 2: coarse classification-derived disparity from encoder features."""

    def __init__(self, net: WAFT) -> None:
        super().__init__()
        self.net = net

    @classmethod
    def from_pretrained(
        cls, weights_path: str | None = None, precision: Precision = Precision.float
    ) -> Self:
        return cast(
            Self, WaftStereo.from_pretrained(weights_path, precision).classification
        )

    def forward(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        fmap1
            Left-view encoder feature from ``WaftStereoEncoder``.
        fmap2
            Right-view encoder feature from ``WaftStereoEncoder``.

        Returns
        -------
        torch.Tensor
            Coarse classification-derived disparity, ``[B, 1, H//2, W//2]``.
        """
        net = self.net
        idx_bins_2x = torch.linspace(
            0, net.max_disp / 2, net.n_bins, device=fmap1.device, dtype=fmap1.dtype
        ).view(1, net.n_bins, 1, 1)
        prop_hidden = net.prop_decoder(net.prop_proj(torch.cat([fmap1, fmap2], dim=1)))
        prob_bins = F.softmax(net.prop_bins_head(prop_hidden), dim=1)
        return torch.sum(prob_bins * idx_bins_2x, dim=1, keepdim=True)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = _NET_HEIGHT,
        width: int = _NET_WIDTH,
        fmap_channels: int = 48,
    ) -> InputSpec:
        # fmap is encoder output resolution, i.e. half the image size - keep
        # in sync with WaftStereoEncoder.get_input_spec's height/width so a
        # CLI --height/--width override reaches every component consistently.
        fmap_shape = (batch_size, fmap_channels, height // 2, width // 2)
        return {
            "fmap1": TensorSpec(
                shape=fmap_shape, dtype="float32", io_type=IoType.TENSOR
            ),
            "fmap2": TensorSpec(
                shape=fmap_shape, dtype="float32", io_type=IoType.TENSOR
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"disparity": TensorSpec()}

    def component_precision(self) -> Precision:
        return Precision.float


class WaftStereoClassificationQuantizable(  # type: ignore[misc]
    AIMETOnnxQuantizableMixin, WaftStereoClassification
):
    """AIMET-ONNX-quantized WAFT-Stereo classification head (w8a8)."""

    default_subfolder: str = "classification"

    def __init__(
        self,
        net: WAFT,
        quant_sim: QuantSimOnnx | None = None,
        onnx_bundle: ONNXBundle | None = None,
    ) -> None:
        WaftStereoClassification.__init__(self, net)
        AIMETOnnxQuantizableMixin.__init__(self, quant_sim, onnx_bundle=onnx_bundle)
        self.collection: WaftStereo | None = None

    def component_precision(self) -> Precision:
        return Precision.w8a8

    def get_calibration_data(
        self,
        input_spec: InputSpec | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries | None:
        assert self.collection is not None, "collection must be set before calibration"
        return build_calibration_data(
            self.collection, "classification", num_samples=num_samples
        )

    def make_quant_sim(self) -> QuantSimOnnx | None:
        return _make_component_quant_sim(
            self.net, self._onnx_bundle, WaftStereoClassification, Precision.w8a8
        )


class WaftStereoRegression(BaseModel):
    """WAFT-Stereo stage 2: iterative delta-refinement disparity regression."""

    def __init__(self, net: WAFT) -> None:
        super().__init__()
        self.net = net

    @classmethod
    def from_pretrained(
        cls, weights_path: str | None = None, precision: Precision = Precision.float
    ) -> Self:
        return cast(
            Self, WaftStereo.from_pretrained(weights_path, precision).regression
        )

    def forward(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        net_state: torch.Tensor,
        disp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        fmap1
            Left-view encoder feature from ``WaftStereoEncoder``.
        fmap2
            Right-view encoder feature from ``WaftStereoEncoder``.
        net_state
            Decoder hidden state from ``WaftStereoEncoder``.
        disp
            Coarse disparity from ``WaftStereoClassification``.

        Returns
        -------
        torch.Tensor
            Refined disparity map of shape ``[B, 1, H, W]``.
        """
        net = self.net
        # fmap1 is at half the original image resolution.
        height, width = 2 * fmap1.shape[-2], 2 * fmap1.shape[-1]
        padder = Padder((height, width), factor=net.factor)

        disp_up = disp
        for _ in range(net.iters):
            disp = disp.detach()
            warped_fmap2 = disp_warp(fmap2, disp, padding_mode="zeros")
            hidden = net.delta_proj(
                torch.cat([fmap1, warped_fmap2, net_state, disp], dim=1)
            )
            net_state = net.delta_decoder(hidden)
            delta_disp = net.delta_disp_head(net_state)
            mask = 0.25 * net.delta_mask_head(net_state)
            disp = disp + delta_disp
            disp_up = net.convex_upsample(disp * 2, mask)

        return padder.unpad(disp_up)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = _NET_HEIGHT,
        width: int = _NET_WIDTH,
        fmap_channels: int = 48,
    ) -> InputSpec:
        # fmap is encoder output resolution, i.e. half the image size - keep
        # in sync with WaftStereoEncoder.get_input_spec's height/width so a
        # CLI --height/--width override reaches every component consistently.
        fmap_shape = (batch_size, fmap_channels, height // 2, width // 2)
        disp_shape = (batch_size, 1, height // 2, width // 2)
        return {
            "fmap1": TensorSpec(
                shape=fmap_shape, dtype="float32", io_type=IoType.TENSOR
            ),
            "fmap2": TensorSpec(
                shape=fmap_shape, dtype="float32", io_type=IoType.TENSOR
            ),
            "net_state": TensorSpec(
                shape=fmap_shape, dtype="float32", io_type=IoType.TENSOR
            ),
            "disp": TensorSpec(
                shape=disp_shape, dtype="float32", io_type=IoType.TENSOR
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"disparity": TensorSpec()}

    def component_precision(self) -> Precision:
        return Precision.float


class WaftStereo(WorkbenchModelCollection):
    """WAFT-Stereo (DAv2S-4): warping-based iterative stereo disparity estimation."""

    def __init__(
        self,
        encoder: WaftStereoEncoder,
        classification: WaftStereoClassification,
        regression: WaftStereoRegression,
    ) -> None:
        super().__init__(
            {
                "encoder": encoder,
                "classification": classification,
                "regression": regression,
            }
        )
        self.encoder = encoder
        self.classification = classification
        self.regression = regression

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str | None = None,
        precision: Precision = Precision.float,
        quantized_checkpoint: str | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        weights_path
            Optional path to a custom WAFT ``.pth`` checkpoint. Defaults to the
            released SynLarge/DAv2S-4 weights.
        precision
            Target precision. ``mixed_with_float`` quantizes encoder (w8a16) and
            classification (w8a8); regression is always kept float.
        quantized_checkpoint
            Optional directory (as saved by ``quantize.py``) containing
            per-component pre-calibrated ``<component>/model.onnx`` +
            ``model.encodings``. When a component's subfolder is found there,
            its encodings are loaded directly instead of re-running SeqMSE
            calibration. Only used for ``mixed_with_float``.

        Returns
        -------
        Self
            Collection model with the requested precision.
        """
        net = load_source_model(weights_path)

        # Only the feature encoder is LoRA-wrapped in this config; merge its
        # adapter into the base weights so tracing sees plain Linears.
        for module in net.modules():
            if isinstance(module, PeftModel):
                module.merge_and_unload()

        if precision == Precision.float:
            encoder = WaftStereoEncoder(net)
            classification = WaftStereoClassification(net)
            regression = WaftStereoRegression(net)
            return cls(encoder, classification, regression)

        if precision != Precision.mixed_with_float:
            raise ValueError(f"WaftStereo does not support precision {precision}.")

        # Calibration for every component chains through this plain-float
        # collection (matches the validated quant-analyzer pipeline).
        float_collection = cls(
            WaftStereoEncoder(net),
            WaftStereoClassification(net),
            WaftStereoRegression(net),
        )

        encoder_bundle = _load_calibrated_onnx_bundle(
            quantized_checkpoint, WaftStereoEncoderQuantizable.default_subfolder
        )
        classification_bundle = _load_calibrated_onnx_bundle(
            quantized_checkpoint, WaftStereoClassificationQuantizable.default_subfolder
        )
        q_encoder = WaftStereoEncoderQuantizable(net, onnx_bundle=encoder_bundle)
        q_classification = WaftStereoClassificationQuantizable(
            net, onnx_bundle=classification_bundle
        )
        for component in (q_encoder, q_classification):
            component.collection = float_collection

        # Skip eager SeqMSE/AIMET calibration when a pre-calibrated checkpoint
        # (from quantize.py) was found on disk for that component.
        if encoder_bundle is None:
            q_encoder.quantize(use_seq_mse=True)
        if classification_bundle is None:
            q_classification.quantize(use_seq_mse=False)

        # Quantizing regression shows accuracy degradation for quantization; kept float.
        regression = WaftStereoRegression(net)

        return cls(q_encoder, q_classification, regression)

    def get_evaluator(self) -> BaseEvaluator:
        return WaftStereoD1Evaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [Kitti2015Dataset]

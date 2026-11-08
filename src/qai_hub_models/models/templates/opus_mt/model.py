# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging
import os
from abc import abstractmethod
from functools import lru_cache
from typing import TypeVar, cast

from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.cross_layer_equalization import equalize_model
from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim
from onnxsim import simplify
from qai_hub.client import Device
from qai_hub_models_cli.common import LOCAL_STORE_DEFAULT_PATH
from torch import Tensor
from transformers import MarianMTModel
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.common_voice import CommonVoiceText, TTSLanguage
from qai_hub_models.models.templates.llm._utils import _set_matmul_second_input_to_8b
from qai_hub_models.models.templates.opus_mt.mixed_precision_overrides import (
    ManualQuantsimMixedPrecisionConfig,
)
from qai_hub_models.models.templates.opus_mt.model_adaptation import (
    QcMarianDecoder,
    QcMarianEncoder,
    apply_model_adaptations,
)
from qai_hub_models.models.templates.opus_mt.utils import (
    write_opus_mt_supplementary_files,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, qaihm_temp_dir
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader
from qai_hub_models.utils.input_spec import (
    InputSpec,
    OutputSpec,
    TensorSpec,
    make_torch_inputs,
)
from qai_hub_models.utils.onnx.helpers import ONNXBundle, safe_torch_onnx_export
from qai_hub_models.utils.quantization_aimet_onnx import (
    AIMETOnnxQuantizableMixin,
    aimet_quant_types,
    set_aimet_log_level,
)

MAX_SEQ_LEN_ENC = 256
MAX_SEQ_LEN_DEC = 256
NUM_CALIBRATION_SAMPLES = 100


@lru_cache(maxsize=1)
def get_opus_model(mt_version: str) -> tuple[QcMarianEncoder, QcMarianDecoder]:
    """Download OpusMT model from HuggingFace or aihub-public-assets.s3 and apply adaptations."""
    if mt_version == "Helsinki-NLP/opus-mt-en-jap":
        mt_version = CachedWebModelAsset.from_asset_store(
            "opus_mt_en_ja",
            2,
            "en_jp_torch_model_v2.zip",
        ).fetch(extract=True)  # Download QC trained en_jp model
        print(f"pt_model_path = {mt_version}")

    orig_model = cast(
        MarianMTModel,
        MarianMTModel.from_pretrained(mt_version),
    )
    orig_model.eval()
    # Apply model adaptations to optimize for QNN inference
    return apply_model_adaptations(orig_model)


class OpusMTEncoder(BaseModel):
    """
    Float Encoder optimized for export and inference.

    It takes text input (input_ids) and produces cross attention
    key-value cache for the decoder.
    """

    def __init__(self, model: QcMarianEncoder) -> None:
        super().__init__()
        self.encoder = model

    @classmethod
    def from_pretrained(cls, mt_version: str) -> Self:
        return cls(get_opus_model(mt_version)[0])

    def forward(
        self, input_ids: Tensor, encoder_attention_mask: Tensor
    ) -> tuple[Tensor, ...]:
        """
        Parameters
        ----------
        input_ids
            Input token IDs of shape (batch_size, sequence_length)
        encoder_attention_mask
            Attention mask of shape (batch_size, sequence_length)

        Returns
        -------
        tuple[Tensor, ...]
            Cross attention key and value cache tensors
        """
        return self.encoder(input_ids, encoder_attention_mask)

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type)). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "input_ids": TensorSpec(
                shape=(1, MAX_SEQ_LEN_ENC),
                dtype="int32",
            ),
            "encoder_attention_mask": TensorSpec(
                shape=(1, MAX_SEQ_LEN_ENC),
                dtype="int32",
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        """Returns the output names for the encoder."""
        output_names = []
        for layer_idx in range(len(self.encoder.decoder_layers)):
            output_names.append(f"block_{layer_idx}_cross_key_states")
            output_names.append(f"block_{layer_idx}_cross_value_states")
        return {name: TensorSpec() for name in output_names}


class OpusMTDecoder(BaseModel):
    """
    Float Decoder optimized for export and inference.
    Wraps MarianDecoderMod to facilitate export.
    """

    def __init__(self, model: QcMarianDecoder) -> None:
        super().__init__()
        self.decoder = model
        self.num_layers = len(model.layers)
        self.num_heads = model.layers[0].self_attn.num_heads
        self.head_dim = model.layers[0].self_attn.head_dim

    @classmethod
    def from_pretrained(cls, mt_version: str) -> Self:
        return cls(get_opus_model(mt_version)[1])

    def forward(
        self,
        input_ids: Tensor,
        encoder_attention_mask: Tensor,
        position: Tensor,
        *past_key_values: Tensor,
    ) -> tuple[Tensor, ...]:
        """
        Parameters
        ----------
        input_ids
            Input token IDs of shape (batch_size, 1)
        encoder_attention_mask
            Encoder attention mask of shape (batch_size, encoder_seq_len)
        position
            Current position index of shape (1,)
        *past_key_values
            Past key-value states for self and cross attention

        Returns
        -------
        tuple[Tensor, ...]
            Logits and updated key-value states
        """
        return self.decoder(
            input_ids, encoder_attention_mask, position, *past_key_values
        )

    def get_input_spec(self) -> InputSpec:
        specs: InputSpec = {
            "input_ids": TensorSpec(shape=(1, 1), dtype="int32"),
            "encoder_attention_mask": TensorSpec(
                shape=(1, MAX_SEQ_LEN_ENC),
                dtype="int32",
            ),
            "position": TensorSpec(shape=(1,), dtype="int32"),
        }

        for i in range(self.num_layers):
            specs[f"block_{i}_past_self_key_states"] = TensorSpec(
                shape=(1, self.num_heads, MAX_SEQ_LEN_DEC - 1, self.head_dim),
                dtype="float32",
            )
            specs[f"block_{i}_past_self_value_states"] = TensorSpec(
                shape=(1, self.num_heads, MAX_SEQ_LEN_DEC - 1, self.head_dim),
                dtype="float32",
            )
            specs[f"block_{i}_cross_key_states"] = TensorSpec(
                shape=(1, self.num_heads, MAX_SEQ_LEN_ENC, self.head_dim),
                dtype="float32",
            )
            specs[f"block_{i}_cross_value_states"] = TensorSpec(
                shape=(1, self.num_heads, MAX_SEQ_LEN_ENC, self.head_dim),
                dtype="float32",
            )

        return specs

    def get_output_spec(self) -> OutputSpec:
        """Returns the output names for the decoder."""
        output_names = ["logits"]
        for layer_idx in range(self.num_layers):
            output_names.append(f"block_{layer_idx}_present_self_key_states")
            output_names.append(f"block_{layer_idx}_present_self_value_states")
        return {name: TensorSpec() for name in output_names}


QuantizableComponentT = TypeVar(
    "QuantizableComponentT", bound=AIMETOnnxQuantizableMixin
)


def quantize_component(
    quantizable_cls: type[QuantizableComponentT],
    fp_collection: OpusMT,
    precision: Precision,
    component_name: str,
) -> QuantizableComponentT:
    """
    Calibrate an OpusMT encoder/decoder from scratch.

    Parameters
    ----------
    quantizable_cls
        The concrete quantizable class to instantiate.
    fp_collection
        Float OpusMT collection for the desired language variant; supplies
        the float encoder/decoder to export to ONNX and the calibration data.
    precision
        the precision used for quantization.
    component_name
        "encoder" or "decoder".

    Returns
    -------
    QuantizableComponentT
        The calibrated quantizable component.
    """
    # Local import to avoid a circular import: app.py imports from this
    # module, so importing OpusMTApp at module scope would form a cycle.
    from qai_hub_models.models.templates.opus_mt.app import OpusMTApp

    fp_module: OpusMTEncoder | OpusMTDecoder
    qc_module: QcMarianEncoder | QcMarianDecoder
    if component_name == "encoder":
        fp_module = fp_collection.encoder
        qc_module = fp_collection.encoder.encoder
    else:
        fp_module = fp_collection.decoder
        qc_module = fp_collection.decoder.decoder

    # Export the float module to ONNX.
    specs = fp_module.get_input_spec()
    dummy_input = tuple(make_torch_inputs(specs))
    with qaihm_temp_dir() as tempdir:
        temp_model_path = os.path.join(tempdir, "model.onnx")
        safe_torch_onnx_export(
            fp_module,
            dummy_input,
            temp_model_path,
            export_params=True,
            input_names=list(specs.keys()),
            output_names=list(fp_module.get_output_spec()),
        )
        onnx_model, _ = simplify(temp_model_path)
        equalize_model(onnx_model)  # Apply cross-layer equalization to the ONNX model.

    # Build the QuantizationSimModel and apply the OpusMT mixed-precision
    # overrides. Also done identically when rebuilding a sim to
    # load cached encodings, since load_encodings_to_sim requires the
    # quantizer configuration to match what was originally exported.
    param_type, act_type = aimet_quant_types(precision)
    sim = QuantizationSimModel(
        model=onnx_model,
        param_type=param_type,
        activation_type=act_type,
        quant_scheme=QuantScheme.min_max,
        config_file="default",
    )
    _set_matmul_second_input_to_8b(sim)
    sim._tie_quantizers_for_op_types(["Concat"])  # pylint: disable=protected-access
    sim._rebuild_session()  # pylint: disable=protected-access
    exceptions_path = os.path.join(os.path.dirname(__file__), "exceptions.json")
    mixed_precision_cfg = ManualQuantsimMixedPrecisionConfig(exceptions_path)
    mixed_precision_cfg.apply_exceptions(sim)

    # Calibrate the quantizers, otherwise they are left uninitialized and will
    # produce garbage words on-device.
    component = quantizable_cls(qc_module, sim)
    calibration_data = OpusMTApp.get_calibration_data(
        fp_collection, component_name, num_samples=NUM_CALIBRATION_SAMPLES
    )
    data_loader = dataset_entries_to_dataloader(calibration_data)
    component.quantize(data_loader, num_samples=len(data_loader))
    return component


def load_or_quantize_component(
    quantizable_cls: type[QuantizableComponentT],
    fp_collection: OpusMT,
    precision: Precision,
    component_name: str,
) -> QuantizableComponentT:
    """
    Return a calibrated OpusMT component, using a local disk
    cache to avoid re-quantizing on every call.

    Parameters
    ----------
    quantizable_cls
        The concrete quantizable class to instantiate.
    fp_collection
        Float OpusMT collection for the desired language variant; supplies
        the float encoder/decoder, the calibration data.
    precision
        the precision used for quantization.
    component_name
        "encoder" or "decoder".

    Returns
    -------
    QuantizableComponentT
        The calibrated component, loaded from the disk cache if
        present, otherwise built from scratch.
    """
    model_name = fp_collection.hf_source.split("/")[-1]
    cache_dir = os.path.join(
        LOCAL_STORE_DEFAULT_PATH, model_name, f"{component_name}_{precision!s}"
    )
    onnx_path = os.path.join(cache_dir, "model.onnx")
    encodings_path = os.path.join(cache_dir, "model.encodings")
    # calibrate once, load forever
    if os.path.exists(onnx_path) and os.path.exists(encodings_path):
        print(f"Loading onnx + encodings from disk: \033[38;5;206m{cache_dir}\033[0m\n")
        bundle = ONNXBundle.from_bundle_path(cache_dir, model_name="model")
        onnx_model = bundle.load_onnx_model()
        param_type, act_type = aimet_quant_types(precision)
        sim = QuantizationSimModel(
            model=onnx_model,
            param_type=param_type,
            activation_type=act_type,
            quant_scheme=QuantScheme.min_max,
            config_file="default",
        )
        # load_encodings_to_sim func requires sim to be the same as quantizing from scratch
        _set_matmul_second_input_to_8b(sim)
        sim._tie_quantizers_for_op_types(["Concat"])  # pylint: disable=protected-access
        sim._rebuild_session()  # pylint: disable=protected-access
        exceptions_path = os.path.join(os.path.dirname(__file__), "exceptions.json")
        mixed_precision_cfg = ManualQuantsimMixedPrecisionConfig(exceptions_path)
        mixed_precision_cfg.apply_exceptions(sim)

        with set_aimet_log_level(logging.WARN):
            load_encodings_to_sim(sim, str(bundle.aimet_encodings_path), strict=False)

        qc_module: QcMarianEncoder | QcMarianDecoder
        if component_name == "encoder":
            qc_module = fp_collection.encoder.encoder
        else:
            qc_module = fp_collection.decoder.decoder
        return quantizable_cls(qc_module, sim)

    print(f"Saving onnx + encodings to disk: \033[38;5;206m{cache_dir}\033[0m\n")
    component = quantize_component(
        quantizable_cls, fp_collection, precision, component_name
    )  # time-consuming, calibrate component_name from scratch
    assert component.quant_sim is not None
    os.makedirs(cache_dir, exist_ok=True)
    # remember to save it to cache before return
    component.quant_sim.export(cache_dir, "model")
    return component


# For the purpose of using AIMETOnnxQuantizableMixin.forward and quantize methods
class OpusMTEncoderQuantizable(AIMETOnnxQuantizableMixin, OpusMTEncoder):  # type: ignore[misc]
    """A class that represents a quantizable OpusMT encoder."""

    def __init__(self, model: QcMarianEncoder, sim_model: QuantizationSimModel) -> None:
        OpusMTEncoder.__init__(self, model)
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)

    @classmethod
    def from_pretrained(cls, fp_collection: OpusMT, precision: Precision) -> Self:  # type: ignore[override]
        """
        Creates a calibrated OpusMTEncoderQuantizable instance, reusing a
        cached calibration from disk if one already exists.

        Parameters
        ----------
        fp_collection
            Float OpusMT collection (encoder + decoder) for the desired
            language variant. Supplies fp encoder and calibration data.
        precision
            the precision used for quantization.

        Returns
        -------
        Self
            The created instance.
        """
        return load_or_quantize_component(cls, fp_collection, precision, "encoder")

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        return super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
        )


# For the purpose of using AIMETOnnxQuantizableMixin.forward and quantize methods
class OpusMTDecoderQuantizable(AIMETOnnxQuantizableMixin, OpusMTDecoder):  # type: ignore[misc]
    """A class that represents a quantizable OpusMT decoder."""

    def __init__(self, model: QcMarianDecoder, sim_model: QuantizationSimModel) -> None:
        OpusMTDecoder.__init__(self, model)
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)

    @classmethod
    def from_pretrained(cls, fp_collection: OpusMT, precision: Precision) -> Self:  # type: ignore[override]
        """
        Creates a calibrated OpusMTDecoderQuantizable instance, reusing a
        cached calibration from disk if already exists.

        Parameters
        ----------
        fp_collection
            Float OpusMT collection (encoder + decoder) for the desired
            language variant. Supplies fp decoder and calibration data.
        precision
            the precision used for quantization.

        Returns
        -------
        Self
            The created instance.
        """
        return load_or_quantize_component(cls, fp_collection, precision, "decoder")

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        return super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
        )


class OpusMT(WorkbenchModelCollection):
    """Base quantized and float OpusMT translation model."""

    def __init__(
        self, encoder: OpusMTEncoder, decoder: OpusMTDecoder, hf_source: str
    ) -> None:
        super().__init__({"encoder": encoder, "decoder": decoder})
        self.encoder = encoder
        self.decoder = decoder
        self.hf_source = hf_source

    @classmethod
    def from_pretrained(cls, precision: Precision = Precision.float) -> Self:
        """Load an OpusMT collection for the requested precision."""
        if precision == Precision.float:
            encoder = OpusMTEncoder.from_pretrained(cls.get_opus_mt_version())
            decoder = OpusMTDecoder.from_pretrained(cls.get_opus_mt_version())
        else:
            # Quantization needs to base on float components
            fp_collection = cls.from_pretrained(Precision.float)
            encoder = OpusMTEncoderQuantizable.from_pretrained(fp_collection, precision)
            decoder = OpusMTDecoderQuantizable.from_pretrained(fp_collection, precision)
        return cls(encoder, decoder, cls.get_opus_mt_version())

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CommonVoiceText

    @classmethod
    @abstractmethod
    def get_opus_mt_version(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_language(cls) -> TTSLanguage:
        pass

    def write_supplementary_files(
        self, output_dir: str | os.PathLike, metadata: ModelMetadata
    ) -> None:
        write_opus_mt_supplementary_files(
            self.get_opus_mt_version(), output_dir, metadata
        )

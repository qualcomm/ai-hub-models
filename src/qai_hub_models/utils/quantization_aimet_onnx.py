# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Items defined in this file require that AIMET-ONNX be installed."""

from __future__ import annotations

from packaging import version

from qai_hub_models.utils.base_model import WorkbenchModel

try:
    import aimet_onnx
    from aimet_onnx import quantsim as aimet_quantsim
    from aimet_onnx.common.utils import AimetLogger
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx

    aimet_onnx_is_installed = True
except (ImportError, ModuleNotFoundError):
    aimet_onnx_is_installed = False
import contextlib
import gc
import importlib.metadata
import itertools
import os
import shutil
import sys
from collections.abc import Collection, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import onnx
import onnxruntime
import torch
from qai_hub.public_rest_api import DatasetEntries
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from qai_hub_models import Precision, SampleInputsType
from qai_hub_models.models.templates.lm_schema import (
    AdaScaleSpec,
    CalibrationSpec,
    DatasetSpec,
    SeqMSESpec,
    TechniqueSpec,
)
from qai_hub_models.utils.aimet.aimet_dummy_model import zip_aimet_model
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, qaihm_temp_dir
from qai_hub_models.utils.base_evaluator import _DataLoader
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle, mock_torch_onnx_inference
from qai_hub_models.utils.runtime_torch_wrapper import kwargs_to_dict

DEFAULT_SEQ_MSE_NUM_SAMPLES = 20
DEFAULT_ADA_SCALE_NUM_SAMPLES = 128
DEFAULT_ADA_SCALE_NUM_ITERATIONS = 512


def ensure_aimet_onnx_installed(
    expected_version: str | None = None, model_id: str | None = None
) -> None:
    if not aimet_onnx_is_installed:
        errstr = "AIMET-ONNX is missing but must be installed. "
        if not sys.platform.startswith("linux") and sys.platform not in [
            "win32",
            "cygwin",
        ]:
            errstr += "It is not supported on this operating system. You must use either Linux or Windows Subsystem for Linux to install AIMET-ONNX."
        else:
            if model_id is not None:
                install_cmd = f"qai-hub-models install {model_id}"
            elif expected_version is not None:
                install_cmd = f"pip install aimet-onnx=={expected_version}"
            else:
                install_cmd = "qai-hub-models install <your_target_model_id_here>"

            if sys.platform in ["win32", "cygwin"]:
                errstr += "AIMET-ONNX is not supported on Windows. We suggest using Windows Subsystem for Linux (WSL) to create a python environment compatible with AIMET-ONNX.\nIn a compatible WSL python env, run "
            else:
                errstr += "Run "
            errstr += f"`{install_cmd}` to install the correct version of AIMET-ONNX."

        if model_id is not None:
            errstr += f"\nAlternatively, run `qai-hub-models fetch {model_id}` to fetch pre-compiled assets for this model."

        raise RuntimeError(errstr)


def ensure_min_aimet_onnx_version(
    expected_version: str, model_id: str | None = None
) -> None:
    ensure_aimet_onnx_installed(expected_version, model_id)
    if version.Version(aimet_onnx.__version__) < version.Version(expected_version):
        raise RuntimeError(
            f"Installed AIMET-ONNX version not supported. Expected >= {expected_version}, got {aimet_onnx.__version__!s}\n"
            f"Please run `pip install aimet-onnx=={expected_version}`"
        )


def aimet_quant_types(precision: Precision) -> tuple[Any, Any]:
    """Return (param_quantize_type, activation_quantize_type) for the given precision."""
    import aimet_onnx

    _PARAM_MAP = {
        Precision.w8a16: aimet_onnx.int8,
        Precision.w8a8: aimet_onnx.int8,
        Precision.w4a16: aimet_onnx.int4,
    }
    _ACT_MAP = {
        Precision.w8a16: aimet_onnx.int16,
        Precision.w8a8: aimet_onnx.int8,
        Precision.w4a16: aimet_onnx.int16,
    }
    return _PARAM_MAP.get(precision, aimet_onnx.int8), _ACT_MAP.get(
        precision, aimet_onnx.int8
    )


@contextmanager
def set_aimet_log_level(log_level: int) -> Generator[None, None, None]:
    area_log_levels: dict[AimetLogger.LogAreas, int] = {}
    for area in AimetLogger.LogAreas:
        area_log_levels[area] = AimetLogger.get_area_logger(area).level

    try:
        AimetLogger.set_level_for_all_areas(log_level)
        yield
    finally:
        for area, level in area_log_levels.items():
            AimetLogger.set_area_logger_level(area, level)


@contextmanager
def untie_aimet_quantizers_for_op_types(
    op_types: Collection[str],
) -> Generator[None, None, None]:
    """
    Temporarily drop op types from aimet's quantizer-tying list.

    Tying replaces each input's quantizer object with the output's, bitwidth
    included, which corrupts bitwidths when replaying a calibrated checkpoint
    whose tied output is intentionally lower precision.
    """
    orig = aimet_quantsim.op_types_to_tie_qtzrs
    try:
        aimet_quantsim.op_types_to_tie_qtzrs = [t for t in orig if t not in op_types]
        yield
    finally:
        aimet_quantsim.op_types_to_tie_qtzrs = orig


class AIMETOnnxQuantizableMixin(WorkbenchModel):
    """
    Mixin that allows a model to be quantized & exported to disk using AIMET.
    Inheritor must implement WorkbenchModel for this mixin to function.
    """

    # AI Hub rejects --output_names for AIMET ONNX models.
    pass_output_names_to_compile: bool = False

    # For pre-calibrated asset lookup
    model_id: str = ""
    model_asset_version: int = -1

    # Which AIMET model type to use for AdaScale
    # (if None, the model cannot use AdaScale)
    ada_scale_model_type: str | None = None

    # RMSNorms per block (currently needed for AdaScale for Qwen3)
    ada_scale_num_rmsnorm_per_blk: int | None = None

    def __init__(
        self,
        quant_sim: QuantSimOnnx | None,
        onnx_bundle: ONNXBundle | None = None,
    ) -> None:
        """
        Parameters
        ----------
        quant_sim
            AIMET QuantizationSimModel used for inference and calibration.
            May be ``None`` if the subclass overrides :meth:`make_quant_sim`
            for lazy construction on first access.
        onnx_bundle
            Optional on-disk ONNX bundle (model + encodings) from
            ``onnx_from_pretrained``.  When set,
            ``convert_to_onnx_and_aimet_encodings`` reuses these files
            directly instead of calling ``quant_sim.export()``, which has
            an aimet-onnx bug that produces an ONNX graph with 0 inputs.
        """
        self._quant_sim = quant_sim
        self._onnx_bundle = onnx_bundle
        # Set by release() when the model is torn down to reclaim memory between
        # parametrized test cases. Used by the conftest from_pretrained cache to
        # treat a released instance as a cache miss and rebuild a fresh one.
        self._released = False

    @property
    def quant_sim(self) -> QuantSimOnnx | None:
        if self._quant_sim is None:
            self._quant_sim = self.make_quant_sim()
        return self._quant_sim

    @quant_sim.setter
    def quant_sim(self, value: QuantSimOnnx | None) -> None:
        self._quant_sim = value

    @quant_sim.deleter
    def quant_sim(self) -> None:
        self._quant_sim = None

    def make_quant_sim(self) -> QuantSimOnnx | None:
        """Override to enable lazy QuantSimOnnx construction.

        Called on first access to ``self.quant_sim`` when ``_quant_sim``
        is ``None``.  Default returns ``None`` (eager-only, same as
        current behavior for models that pass ``quant_sim`` directly).
        """
        return None

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        return Path(
            self.convert_to_onnx_and_aimet_encodings(
                output_dir=Path(output_dir),
                model_name=self.__class__.__name__,
            )
        )

    def get_calibration_data(
        self,
        input_spec: InputSpec | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries | None:
        """
        Get calibration data for quantization.

        Parameters
        ----------
        input_spec
            The input specification for the model.
        num_samples
            None to use all. Specify `num_samples` to use fewer. If
            `num_samples` are more than available, use all available (same
            behavior as None).

        Returns
        -------
        calibration_data : DatasetEntries | None
            The calibration dataset entries, or None if not available.
        """
        return None

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[str, str]:
        """
        Get the calibrated AIMET model paths.

        Returns
        -------
        onnx_path : str
            Path to .onnx file.
        encodings_path : str
            Path to .encodings file.
        """
        if not cls.model_id or cls.model_asset_version == -1:
            raise ValueError("model_id and model_asset_version must be defined")

        subfolder = Path(getattr(cls, "default_subfolder", ""))

        # Returns .onnx and .encodings paths
        onnx_file = CachedWebModelAsset.from_asset_store(
            cls.model_id,
            cls.model_asset_version,
            str(subfolder / "model.onnx"),
        ).fetch()
        with contextlib.suppress(Exception):
            _ = CachedWebModelAsset.from_asset_store(
                cls.model_id,
                cls.model_asset_version,
                str(subfolder / "model.data"),
            ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            cls.model_id,
            cls.model_asset_version,
            str(subfolder / "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings

    def _dataloader_to_numpy(
        self, data: _DataLoader, num_batches: int
    ) -> list[dict[str, Any]]:
        assert self.quant_sim is not None
        input_names = [inp.name for inp in self.quant_sim.session.get_inputs()]
        onnx_data = []
        n = min(len(data), num_batches)
        for batch in tqdm(itertools.islice(data, n), total=n):
            onnx_data.append(  # noqa: PERF401
                {
                    k: v.cpu().detach().numpy()
                    for k, v in kwargs_to_dict(input_names, *batch).items()
                }
            )
        return onnx_data

    def _apply_seq_mse(self, data: _DataLoader, num_batches: int) -> None:
        assert self.quant_sim is not None
        ensure_min_aimet_onnx_version("2.8.0")
        aimet_onnx.apply_seq_mse(
            self.quant_sim, self._dataloader_to_numpy(data, num_batches)
        )

    def _apply_ada_scale(
        self,
        data: _DataLoader,
        num_batches: int,
        num_iterations: int,
        model_type: str,
        num_rmsnorm_per_blk: int | None = None,
    ) -> None:
        if self.ada_scale_model_type is None:
            raise ValueError("AdaScale is not supported for this model.")
        assert self.quant_sim is not None
        ensure_min_aimet_onnx_version("2.26.0")
        from aimet_onnx.experimental.adascale.adascale_optimizer import (
            AdaScale,
            adascale_model_config_dict,
        )

        restore_value: int | None = None
        if num_rmsnorm_per_blk is not None:
            from aimet_onnx.graph_passes.passes.decoder_block import DecoderBlockQwen3

            restore_value = DecoderBlockQwen3.NUM_RMSNORM_PER_BLK
            DecoderBlockQwen3.NUM_RMSNORM_PER_BLK = num_rmsnorm_per_blk

        inputs = self._dataloader_to_numpy(data, num_batches=num_batches)

        # Pre-compute param encodings with a real input to avoid
        # make_dummy_input (inside apply_adascale) generating random values
        self.quant_sim._compute_param_encodings(dummy_input=inputs[0], overwrite=False)

        AdaScale.apply_adascale(
            self.quant_sim,
            inputs,
            adascale_model_config=adascale_model_config_dict[model_type],
            num_iterations=num_iterations,
        )

        if restore_value is not None:
            DecoderBlockQwen3.NUM_RMSNORM_PER_BLK = restore_value

    @staticmethod
    def apply_spinquant_to_onnx(
        onnx_model: onnx.ModelProto,
        spinquant_config: dict,
        visual_model: onnx.ModelProto | None = None,
        embedding: torch.Tensor | None = None,
    ) -> None:
        """Apply SpinQuant rotations in place on a float ONNX graph.

        Must run on the *float* graph BEFORE the QuantizationSimModel is built:
        aimet-onnx 2.33.0's block identifier walks RMSNorm -> weight-linear
        connectivity, which the QcQuantizeOp nodes inserted at QuantSim
        construction break (causing "No active RMSNorms found"). R3 also inserts
        plain-float Hadamard MatMuls that the subsequent QuantSim must wrap in
        activation quantizers, so it likewise has to precede sim creation.

        Parameters
        ----------
        onnx_model
            Backbone (text decoder) ONNX model. Mutated in-place.
        spinquant_config
            Dict of rotation flags (enable_r1, enable_r2, enable_r3).
        visual_model
            Optional vision encoder ONNX model (VLM). Its PatchMerger
            linear_fc2 weights are co-rotated in-place with R1.
        embedding
            Optional embedding weight tensor of shape [vocab, hidden]
            (VLM backbones exported with inputs_embeds). Rotated in-place.
        """
        ensure_min_aimet_onnx_version("2.33.0")
        # Local imports: optional 2.33.0-only dependency, gated by the version
        # check above -- same exception to imports-at-top as _apply_ada_scale's
        # AdaScale import.

        print()
        print(f"Apply SpinQuant ({spinquant_config})")
        print()

        from aimet_onnx.experimental.spinquant import apply_spinquant
        from aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx import (
            fix_node_names_pass,
        )
        from aimet_onnx.utils import duplicate_shared_initializers

        # Dynamo names ops generically (node_Conv_*), losing the /self_attn/
        # v_proj/ scope R2's name-based V matcher needs. Restore /{module}/{Op}
        # names first. See tetracode#20426.
        fix_node_names_pass(onnx_model)

        # SpinQuant builds a ConnectedGraph, which (like QuantSim) rejects an
        # initializer shared across consumers of different op types (Qwen3's tied
        # lm_head.weight feeds both the embedding Gather and the lm_head MatMul).
        # Dedup first; idempotent, so the later _build_quantsim call is a no-op.
        duplicate_shared_initializers(onnx_model.graph)
        apply_spinquant(
            onnx_model,
            visual_model=visual_model,
            embedding=embedding,
            **spinquant_config,
        )

    def _apply_calibration(self, data: DataLoader, num_batches: int) -> None:
        assert self.quant_sim is not None
        ensure_min_aimet_onnx_version("2.8.0")
        self.quant_sim.compute_encodings(self._dataloader_to_numpy(data, num_batches))

    def prefill_dataset(
        self,
        dataset_spec: DatasetSpec,
        num_samples: int = 0,
        sequence_length: int = 2048,
        context_length: int = 4096,
        image_size: tuple[int, int] | None = None,
    ) -> DatasetEntries | None:
        """Resolve a recipe dataset spec to a loader and prefill it into hub
        dataset entries. Implemented by the LLM/VLM subclasses (which own the
        generator + vision machinery); the base mixin has no dataset pipeline.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement prefill_dataset; recipe-driven "
            "quantization (quantize_from_steps) is only supported for LLM/VLM models."
        )

    def _resolve_step_dataset(self, step: TechniqueSpec) -> DatasetSpec:
        """The dataset a data-consuming step runs on. Implemented by LLM/VLM
        subclasses (each step must name its own dataset; there is no default).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _resolve_step_dataset; recipe-"
            "driven quantization is only supported for LLM/VLM models."
        )

    def quantize_from_steps(
        self,
        on_sim_steps: list[TechniqueSpec],
        seq_len: int = 2048,
        context_length: int = 4096,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        """Run on-sim steps (validated technique specs) in order, memoizing prefills
        by dataset + volume. SpinQuant is pre-sim (caller-handled) and must not appear.

        Per-step data volume follows GenAI Lab: ``num_iterations`` for
        Calibration/SeqMSE, ``num_batches`` for AdaScale (whose ``num_iterations`` is
        the optimizer step count, a separate axis). That count sizes the prefill; the
        loader is then consumed whole (applier ``num_batches = len(data)``).
        """
        prefill_cache: dict[str, DataLoader] = {}

        def _loader_for(step: TechniqueSpec, num_blocks: int) -> DataLoader:
            spec = self._resolve_step_dataset(step)
            cache_key = f"{spec.model_dump_json()}::{num_blocks}"
            if cache_key not in prefill_cache:
                entries = self.prefill_dataset(
                    spec,
                    num_samples=num_blocks,
                    sequence_length=seq_len,
                    context_length=context_length,
                    image_size=image_size,
                )
                assert entries is not None
                prefill_cache[cache_key] = dataset_entries_to_dataloader(entries)
            return prefill_cache[cache_key]

        for step in on_sim_steps:
            if isinstance(step, SeqMSESpec):
                data = _loader_for(step, num_blocks=step.num_iterations or 0)
                num_batches = len(data)
                print(f"\nApply Sequential MSE ({num_batches} batches)\n")
                self._apply_seq_mse(data=data, num_batches=num_batches)
                gc.collect()
                torch.cuda.empty_cache()
            elif isinstance(step, AdaScaleSpec):
                if self.ada_scale_model_type is None:
                    raise ValueError("AdaScale is not supported for this model.")
                data = _loader_for(step, num_blocks=step.num_batches or 0)
                num_iterations = step.num_iterations or DEFAULT_ADA_SCALE_NUM_ITERATIONS
                num_batches = len(data)
                print(
                    f"\nApply AdaScale ({num_batches} batches, {num_iterations} iterations)\n"
                )
                self._apply_ada_scale(
                    data=data,
                    num_batches=num_batches,
                    num_iterations=num_iterations,
                    model_type=self.ada_scale_model_type,
                    num_rmsnorm_per_blk=self.ada_scale_num_rmsnorm_per_blk,
                )
                gc.collect()
                torch.cuda.empty_cache()
            elif isinstance(step, CalibrationSpec):
                data = _loader_for(step, num_blocks=step.num_iterations or 0)
                num_batches = len(data)
                print(
                    f"\nStart QuantSim calibration for {self.__class__.__name__} "
                    f"({num_batches} batches)\n"
                )
                self._apply_calibration(data=data, num_batches=num_batches)
            else:
                raise TypeError(
                    f"Quantization technique {step.name!r} is not implemented in AI "
                    f"Hub Models. Implemented on-sim techniques: SeqMSE, AdaScale, "
                    f"Calibration."
                )

    def quantize(
        self,
        data: DataLoader | None = None,
        num_samples: int | None = None,
        use_seq_mse: bool = False,
        use_ada_scale: bool = False,
        seq_mse_num_samples: int | None = None,
        ada_scale_num_samples: int | None = None,
        ada_scale_num_iterations: int | None = None,
        weight_optimization_data: DataLoader | None = None,
    ) -> None:
        """
        Quantize the model using calibration data.

        The two-boolean path for non-LLM models; LLM/VLM models instead drive
        quantization from a recipe (``LLMDynamic_AIMETOnnx.quantize_from_steps``).

        Parameters
        ----------
        data
            If None, create data loader from get_calibration_data(), which
            must be implemented.
        num_samples
            Number of samples to use for calibration. If None, uses all
            available samples in the data loader.
        use_seq_mse
            Whether to apply sequential MSE optimization during quantization.
        use_ada_scale
            Whether to apply AdaScale optimization during quantization.
        seq_mse_num_samples
            Number of samples for sequential MSE. Defaults to num_samples.
        ada_scale_num_samples
            Number of samples for AdaScale.
        ada_scale_num_iterations
            Number of iterations for AdaScale.
        weight_optimization_data
            Separate data loader for seqMSE/AdaScale weight optimization.
            If None, falls back to using ``data`` for both optimization
            and calibration.

        Returns
        -------
        None
        """
        if use_ada_scale and self.ada_scale_model_type is None:
            raise ValueError("AdaScale is not supported for this model.")

        if data is None:
            calib_data = self.get_calibration_data()
            if calib_data is None:
                raise ValueError(
                    "`data` must be specified if get_calibration_data is not defined."
                )
            data = dataset_entries_to_dataloader(calib_data)

        # NOTE: when weight_optimization_data is None, optim_data IS data (same
        # loader for weight-opt and calibration); keep consumers read-only.
        optim_data = (
            data if weight_optimization_data is None else weight_optimization_data
        )

        # "samples": 4096 context length batches
        # "batches": actual iterations
        if hasattr(self, "context_length") and hasattr(self, "sequence_length"):
            batches_per_sample = self.context_length // self.sequence_length
        else:
            batches_per_sample = 1

        # NOTE: SpinQuant is NOT applied here. It rotates the *float* ONNX graph
        # and must run before the QuantizationSimModel is constructed (see
        # apply_spinquant_to_onnx / create_quantsim). Applying it to the already
        # built sim breaks aimet's RMSNorm block detection.

        if use_seq_mse:
            seq_mse_num_samples = min(
                len(optim_data) // batches_per_sample,
                seq_mse_num_samples or num_samples or DEFAULT_SEQ_MSE_NUM_SAMPLES,
            )
            seq_mse_num_batches = seq_mse_num_samples * batches_per_sample

            print()
            print(
                f"Apply Sequential MSE ({seq_mse_num_samples} samples / {seq_mse_num_batches} batches)"
            )
            print()
            self._apply_seq_mse(data=optim_data, num_batches=seq_mse_num_batches)
            gc.collect()
            torch.cuda.empty_cache()

        if use_ada_scale:
            assert self.ada_scale_model_type is not None
            ada_scale_num_samples = min(
                len(optim_data) // batches_per_sample,
                ada_scale_num_samples or DEFAULT_ADA_SCALE_NUM_SAMPLES,
            )
            ada_scale_num_iters = (
                ada_scale_num_iterations or DEFAULT_ADA_SCALE_NUM_ITERATIONS
            )
            ada_scale_num_batches = ada_scale_num_samples * batches_per_sample
            print()
            print(
                f"Apply AdaScale ({ada_scale_num_samples} samples / {ada_scale_num_batches} batches, {ada_scale_num_iters} iterations)"
            )
            print()
            self._apply_ada_scale(
                data=optim_data,
                num_batches=ada_scale_num_batches,
                num_iterations=ada_scale_num_iters,
                model_type=self.ada_scale_model_type,
                num_rmsnorm_per_blk=self.ada_scale_num_rmsnorm_per_blk,
            )
            gc.collect()
            torch.cuda.empty_cache()

        num_calib_samples = num_samples or len(data)
        num_calib_batches = num_calib_samples * batches_per_sample

        print()
        print(
            f"Start QuantSim calibration for {self.__class__.__name__} ({num_calib_samples} samples / {num_calib_batches} batches)"
        )
        print()
        self._apply_calibration(data=data, num_batches=num_calib_batches)

    @contextlib.contextmanager
    def remove_quantization(self) -> Generator[None, None, None]:
        """
        Context manager to temporarily remove quantization nodes from the model. Useful for prefilling data without
        quantization, e.g. for AdaScale or SeqMSE.
        """
        assert isinstance(self.quant_sim, QuantSimOnnx)
        with self.quant_sim._remove_quantization_nodes():
            self.quant_sim._rebuild_session()

            yield

        self.quant_sim._rebuild_session()

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, **kwargs: Any
    ) -> SampleInputsType:
        data = self.get_calibration_data()
        if data is None:
            # Fallback to WorkbenchModel's impl
            data = WorkbenchModel._sample_inputs_impl(
                cast(WorkbenchModel, self), input_spec
            )
        assert isinstance(data, dict)
        return data

    def forward(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """QuantSim forward pass with torch.Tensor"""
        assert self.quant_sim is not None
        return mock_torch_onnx_inference(self.quant_sim.session, *args, **kwargs)

    def save_calibrated_checkpoint(self, output_checkpoint: str) -> None:
        """Save AIMET-ONNX checkpoint to output_checkpoint/subfolder, if"""
        default_subfolder = getattr(self.__class__, "default_subfolder", "")
        export_dir = output_checkpoint
        if default_subfolder:
            export_dir = str(Path(output_checkpoint) / default_subfolder)

        shutil.rmtree(export_dir, ignore_errors=True)
        os.makedirs(export_dir, exist_ok=True)

        print(f"Saving quantized {self.__class__.__name__} to {export_dir}")
        assert self.quant_sim is not None
        self.quant_sim.export(str(export_dir), "model")
        print(f"{self.__class__.__name__} saved to {export_dir}")

    @staticmethod
    def get_ort_providers(
        device: torch.device,
    ) -> list[str | tuple[str, dict[str, str | int]]]:
        if device.type == "cuda":
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                msg = (
                    f"WARNING: GPU requested but CUDAExecutionProvider is not available. "
                    f"Falling back to CPU. Available providers: {available}"
                )
                ort_packages = [
                    d.name
                    for d in importlib.metadata.distributions()
                    if d.name and d.name.startswith("onnxruntime")
                ]
                if "onnxruntime" in ort_packages and any(
                    p != "onnxruntime" for p in ort_packages
                ):
                    msg += (
                        f"\nThis may be caused by the 'onnxruntime' (CPU) package "
                        f"shadowing a GPU-enabled variant. "
                        f"Installed onnxruntime packages: {ort_packages}. "
                        f"Try: pip uninstall onnxruntime && pip install onnxruntime-gpu"
                    )
                print(msg)
                return ["CPUExecutionProvider"]
            cuda_options: dict[str, str | int] = {
                "arena_extend_strategy": "kSameAsRequested",
            }
            if device.index is not None:
                cuda_options["device_id"] = device.index
            return [
                ("CUDAExecutionProvider", cuda_options),
                "CPUExecutionProvider",
            ]
        return ["CPUExecutionProvider"]

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | Path,
        model_name: str | None = None,
        return_zip: bool = True,
    ) -> str:
        """
        Converts the torch module to a zip file containing an unquantized ONNX model
        and an AIMET quantization encodings file if return_zip is True (default).

        If return_zip is False, the model is exported to a directory.
        In that case, the output directory is set to:

            Path(output_dir) / f"{model_name}.aimet"

        and the existing directory is forcefully removed.

        When an ``_onnx_bundle`` with encodings is available (set during
        ``from_pretrained`` from a local or cached checkpoint), the bundle
        files are used directly instead of calling ``quant_sim.export()``.
        """
        if model_name is None:
            model_name = self.__class__.__name__

        output_dir = Path(output_dir)

        # If we already have on-disk ONNX + encodings (local checkpoint or
        # S3 cache), use them directly instead of re-exporting via quant_sim.
        if self._onnx_bundle and self._onnx_bundle.aimet_encodings_path:
            return self._convert_from_bundle(output_dir, model_name, return_zip)

        if return_zip:
            # Ensure output_dir exists and define the zip path.
            os.makedirs(output_dir, exist_ok=True)
            zip_path = output_dir / f"{model_name}.aimet.zip"
            base_dir = Path(f"{model_name}.aimet")

            print(f"Exporting quantized {self.__class__.__name__} to {zip_path}")
            # Use a temporary directory to export the model before zipping.
            with qaihm_temp_dir() as tmpdir:
                export_dir = Path(tmpdir) / base_dir
                os.makedirs(export_dir)
                assert self.quant_sim is not None
                self.quant_sim.export(str(export_dir), "model")

                onnx_file_path = str(export_dir / "model.onnx")
                encoding_file_path = str(export_dir / "model.encodings")

                # Attempt to locate external data file.
                # aimet-onnx<=2.0.0 export external data with model.onnx.data
                # aimet-onnx>=2.3.0 export external data with model.data
                # version between 2.0 - 2.3 are broken on large models
                external_data_file_path = ""
                external_data_file_path2 = export_dir / "model.onnx.data"
                external_data_file_path1 = export_dir / "model.data"
                if external_data_file_path1.exists():
                    external_data_file_path = str(external_data_file_path1)
                elif external_data_file_path2.exists():
                    external_data_file_path = str(external_data_file_path2)

                zip_aimet_model(
                    str(zip_path),
                    base_dir,
                    onnx_file_path,
                    encoding_file_path,
                    external_data_file_path,
                )
            return str(zip_path)
        # Export directly to a directory at output_dir / f"{model_name}.aimet"
        export_dir = output_dir / f"{model_name}.aimet"
        shutil.rmtree(export_dir, ignore_errors=True)
        os.makedirs(export_dir, exist_ok=True)

        print(
            f"Exporting quantized {self.__class__.__name__} to directory {export_dir}"
        )
        assert self.quant_sim is not None
        self.quant_sim.export(str(export_dir), "model")
        return str(export_dir)

    def _convert_from_bundle(
        self,
        output_dir: Path,
        model_name: str,
        return_zip: bool,
    ) -> str:
        """Use pre-existing ONNXBundle files instead of quant_sim.export()."""
        assert self._onnx_bundle is not None
        os.makedirs(output_dir, exist_ok=True)

        if return_zip:
            zip_path = output_dir / f"{model_name}.aimet.zip"
            base_dir = Path(f"{model_name}.aimet")
            print(f"Exporting quantized {self.__class__.__name__} to {zip_path}")
            zip_aimet_model(
                str(zip_path),
                base_dir,
                str(self._onnx_bundle.onnx_graph_path),
                str(self._onnx_bundle.aimet_encodings_path),
                str(self._onnx_bundle.onnx_weights_path)
                if self._onnx_bundle.onnx_weights_path
                else "",
            )
            return str(zip_path)

        export_dir = output_dir / f"{model_name}.aimet"
        shutil.rmtree(export_dir, ignore_errors=True)
        print(
            f"Exporting quantized {self.__class__.__name__} to directory {export_dir}"
        )
        self._onnx_bundle.move(export_dir, "model", copy=True)
        return str(export_dir)

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        """AI Hub Workbench quantize options recommended for the model."""
        return other_options or ""

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Items defined in this file require that AIMET-ONNX be installed."""

from __future__ import annotations

from packaging import version

from qai_hub_models.utils.base_model import BaseModel

try:
    import aimet_onnx

    if version.Version(aimet_onnx.__version__) >= version.Version("2.20.0"):
        from aimet_onnx.common.utils import AimetLogger
    else:
        from aimet_common.utils import AimetLogger
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx

    aimet_onnx_is_installed = True
except (ImportError, ModuleNotFoundError):
    aimet_onnx_is_installed = False
import contextlib
import itertools
import os
import shutil
import sys
from collections.abc import Collection, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch
from qai_hub.client import DatasetEntries
from tqdm.autonotebook import tqdm

from qai_hub_models.evaluators.base_evaluators import _DataLoader
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.models.protocols import PretrainedHubModelProtocol
from qai_hub_models.utils.aimet.aimet_dummy_model import zip_aimet_model
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, qaihm_temp_dir
from qai_hub_models.utils.base_model import Precision
from qai_hub_models.utils.dataset_util import DataLoader, dataset_entries_to_dataloader
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx.helpers import mock_torch_onnx_inference
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
                install_target = f'"qai_hub_models[{model_id}]"'
            elif expected_version is not None:
                install_target = f"aimet-onnx=={expected_version}"
            else:
                install_target = '"qai_hub_models[<your_target_model_id_here>]"'

            if sys.platform in ["win32", "cygwin"]:
                errstr += "AIMET-ONNX is not supported on Windows. We suggest using Windows Subsystem for Linux (WSL) to create a python environment compatible with AIMET-ONNX.\nIn a compatible WSL python env, run "
            else:
                errstr += "Run "
            errstr += f"`pip install {install_target}` to install the correct version of AIMET-ONNX."

        if model_id is not None:
            errstr += f"\nAlternatively, for model export, you may run `python -m qai_hub_models.models.{model_id}.export.py --fetch-static-assets` to fetch pre-compiled assets for this model."

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


def ensure_max_aimet_onnx_version(
    expected_version: str, model_id: str | None = None
) -> None:
    ensure_aimet_onnx_installed(expected_version, model_id)
    if version.Version(aimet_onnx.__version__) < version.Version(expected_version):
        raise RuntimeError(
            f"Installed AIMET-ONNX version not supported. Expected=<{expected_version}, got {aimet_onnx.__version__!s}\n"
            f"Please run `pip install transformers=={expected_version}`"
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


class AIMETOnnxQuantizableMixin(PretrainedHubModelProtocol):
    """
    Mixin that allows a model to be quantized & exported to disk using AIMET.
    Inheritor must implement BaseModel for this mixin to function.
    """

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
    ) -> None:
        self.quant_sim = quant_sim
        if self.quant_sim is not None:
            self.input_names = [i.name for i in self.quant_sim.session.get_inputs()]
            self.output_names = [
                output.name for output in self.quant_sim.session.get_outputs()
            ]

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        # This must be defined by the PretrainedHubModelProtocol ABC
        raise ValueError(
            f"Cannot call convert_to_torchscript on {self.__class__.__name__}"
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
        assert self.quant_sim is not None
        ensure_min_aimet_onnx_version("2.25.0")
        from aimet_onnx.experimental.adascale.adascale_optimizer import (
            AdaScale,
            adascale_model_config_dict,
        )

        restore_value: int | None = None
        if model_type == "qwen3" and num_rmsnorm_per_blk is not None:
            from aimet_onnx.graph_passes.passes.decoder_block import DecoderBlockQwen3

            restore_value = DecoderBlockQwen3.NUM_RMSNORM_PER_BLK
            DecoderBlockQwen3.NUM_RMSNORM_PER_BLK = num_rmsnorm_per_blk

        AdaScale.apply_adascale(
            self.quant_sim,
            self._dataloader_to_numpy(data, num_batches=num_batches),
            adascale_model_config=adascale_model_config_dict[model_type],
            num_iterations=num_iterations,
        )

        if restore_value is not None:
            DecoderBlockQwen3.NUM_RMSNORM_PER_BLK = restore_value

    def _apply_calibration(self, data: DataLoader, num_batches: int) -> None:
        assert self.quant_sim is not None
        ensure_min_aimet_onnx_version("2.8.0")
        self.quant_sim.compute_encodings(self._dataloader_to_numpy(data, num_batches))

    def quantize(
        self,
        data: DataLoader | None = None,
        num_samples: int | None = None,
        use_seq_mse: bool = False,
        use_ada_scale: bool = False,
        seq_mse_num_samples: int | None = None,
        ada_scale_num_samples: int | None = None,
        ada_scale_num_iterations: int | None = None,
    ) -> None:
        """
        Quantize the model using calibration data.

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

        # "samples": 4096 context length batches
        # "batches": actual iterations
        if hasattr(self, "context_length") and hasattr(self, "sequence_length"):
            batches_per_sample = self.context_length // self.sequence_length
        else:
            batches_per_sample = 1

        if use_seq_mse:
            seq_mse_num_samples = min(
                len(data) // batches_per_sample,
                seq_mse_num_samples or num_samples or DEFAULT_SEQ_MSE_NUM_SAMPLES,
            )
            seq_mse_num_batches = seq_mse_num_samples * batches_per_sample

            print()
            print(
                f"Apply Sequential MSE ({seq_mse_num_samples} samples / {seq_mse_num_batches} batches)"
            )
            print()
            self._apply_seq_mse(data=data, num_batches=seq_mse_num_batches)

        if use_ada_scale:
            assert self.ada_scale_model_type is not None
            ada_scale_num_samples = min(
                len(data) // batches_per_sample,
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
                data=data,
                num_batches=ada_scale_num_batches,
                num_iterations=ada_scale_num_iters,
                model_type=self.ada_scale_model_type,
                num_rmsnorm_per_blk=self.ada_scale_num_rmsnorm_per_blk,
            )

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
            # Fallback to BaseModel's impl
            data = BaseModel._sample_inputs_impl(cast(BaseModel, self), input_spec)
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
    ) -> list[str | tuple[str, dict[str, int]]]:
        if device.type == "cuda":
            return (
                [
                    ("CUDAExecutionProvider", {"device_id": device.index}),
                    "CPUExecutionProvider",
                ]
                if device.index is not None
                else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
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
        """
        if model_name is None:
            model_name = self.__class__.__name__

        output_dir = Path(output_dir)

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

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        """AI Hub Workbench quantize options recommended for the model."""
        return other_options or ""

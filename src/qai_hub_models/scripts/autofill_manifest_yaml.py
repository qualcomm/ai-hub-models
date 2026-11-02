# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from typing import Any

import numpy as np
import qai_hub as hub
from ruamel.yaml import YAML
from tflite import Model as TFModel

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.export.context import resolve_recipe_dir
from qai_hub_models.utils.export.dispatch import select_pipeline
from qai_hub_models.utils.export.result import (
    CollectionExportResult,
    ExportResult,
    LegacyCollectionExportResult,
    MultiGraphCollectionExportResult,
)
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    QAIHM_MODELS_ROOT,
    SCORECARD_MODELS_ROOT,
)


def _display_with_sig_figs(num: float, num_sig_figs: int = 3) -> str:
    rounded_num = float(f"{num:.{num_sig_figs}g}")
    num_digits = len(str(int(rounded_num)))
    return f"{rounded_num:.{max(0, num_sig_figs - num_digits)}f}"


def _get_formatted_size(size: float, units: list[str], unit_step_size: float) -> str:
    unit_index = 0
    while size >= unit_step_size and unit_index < len(units) - 1:
        size /= unit_step_size
        unit_index += 1
    return f"{_display_with_sig_figs(size)}{units[unit_index]}"


def get_checkpoint_file_size(model_path: str) -> str:
    num_bytes = os.path.getsize(model_path)
    return _get_formatted_size(num_bytes, [" B", " KB", " MB", " GB", " TB"], 1024.0)


def get_tflite_unique_parameters(model_path: str) -> str:
    """
    TFLite parameters are defined at two levels: Tensors and Buffers.

    Only tensors can tell us how many parameters, but we do not want to over-count
    tensors that point to the same buffers. So we keep track of all buffers
    we have counted through tensors.
    """
    with open(model_path, "rb") as f:
        tflite_model = f.read()
    model = TFModel.GetRootAs(tflite_model, 0)

    parameter_cnt = 0
    buffers_counted: set[int] = set()
    for i in range(model.SubgraphsLength()):
        graph = model.Subgraphs(i)
        assert graph is not None
        for j in range(graph.TensorsLength()):
            tensor = graph.Tensors(j)
            assert tensor is not None
            buf_index = tensor.Buffer()

            buffer = model.Buffers(buf_index)
            assert buffer is not None
            if not buffer.DataIsNone() and buf_index not in buffers_counted:
                parameter_cnt += int(np.prod(tensor.ShapeAsNumpy()))
                buffers_counted.add(buf_index)

    return _get_formatted_size(parameter_cnt, ["", "K", "M", "B", "T"], 1000.0)


def get_model_size_and_parameters(
    job_or_model: hub.Job | hub.Model | None,
) -> tuple[str, str | None]:
    assert job_or_model is not None
    if isinstance(job_or_model, (hub.ProfileJob, hub.InferenceJob)):
        model = job_or_model.model
    elif isinstance(job_or_model, hub.CompileJob):
        if (target_model := job_or_model.get_target_model()) is None:
            raise ValueError("Failed compile job.")
        model = target_model
    elif isinstance(job_or_model, hub.Model):
        model = job_or_model
    else:
        raise TypeError(f"Invalid type for `job_or_model`: {type(job_or_model)}")
    with qaihm_temp_dir() as tmp_dirname:
        path = model.download(os.path.join(tmp_dirname, "model"))
        size = get_checkpoint_file_size(path)
        if model.model_type == hub.SourceModelType.TFLITE:
            parameters = str(get_tflite_unique_parameters(path))
        else:
            parameters = None
        return str(size), parameters


def add_details_to_internal_config(
    config_dict: dict[str, Any],
    details: dict[str, dict[str, tuple[str, str | None]]],
) -> dict[str, Any]:
    new_config = copy.deepcopy(config_dict)
    tech = new_config.setdefault("technical_details", {})

    # Clean the old keys
    for detail_key in list(tech):
        if "model size" in detail_key.lower() or "parameters" in detail_key.lower():
            del tech[detail_key]

    # Add more standard keys here
    for precision_str, precision_data in details.items():
        for name, (size, parameters) in precision_data.items():
            param_key = (
                "Number of parameters"
                if name == "model"
                else f"Number of parameters ({name})"
            )
            size_key = "Model size" if name == "model" else f"Model size ({name})"
            if parameters is not None:
                tech[param_key] = parameters
            tech[f"{size_key} ({precision_str})"] = size

    return new_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        type=str,
        default=None,
        help="Models for which to autofill internal-config.yaml technical details.",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="If set, generates files for all models.",
    )
    parser.add_argument(
        "--skip-compile-jobs",
        "-s",
        action="store_true",
        help="Skip running compile jobs.",
        default=False,
    )
    args = parser.parse_args()
    skip_compile = args.skip_compile_jobs

    assert args.all or args.models, "Must specify -a or -m."
    models: list[str]
    models = parser.parse_args().models if args.models else MODEL_IDS

    for model_name in models:
        model_dir = QAIHM_MODELS_ROOT / model_name
        yaml = YAML()
        yaml.width = sys.maxsize
        perf = QAIHMModelPerf.from_model(model_name, not_exists_ok=True)

        manifest = QAIHMModelManifest.from_model(model_name)
        precisions = manifest.supported_precisions
        details: dict[str, dict[str, tuple[str, str | None]]] = {
            str(precision): {} for precision in precisions
        }

        if manifest.is_precompiled:
            print(f"Skipping {model_name} since its a is_precompiled asset.")
            continue
        try:
            if not skip_compile:
                supported_paths = manifest.get_supported_paths_for_testing(
                    only_include_passing=True
                )
                for precision in precisions:
                    if not (runtimes := supported_paths.get(precision)):
                        print(
                            f"Skipping {model_name} precision {precision!s} since it "
                            "has no supported target runtime."
                        )
                        continue
                    target_runtime = runtimes[0]

                    # Install dependencies
                    requirements_file = None

                    if (model_dir / "requirements.txt").exists():
                        requirements_file = os.path.join(model_dir, "requirements.txt")
                    if requirements_file:
                        subprocess.run(
                            ["pip", "install", "-r", requirements_file], check=False
                        )

                    export_model = select_pipeline(resolve_recipe_dir(model_name))
                    results = export_model(
                        device=hub.Device("Samsung Galaxy S25 (Family)"),
                        skip_downloading=True,
                        skip_profiling=True,
                        skip_inferencing=True,
                        skip_summary=True,
                        target_runtime=target_runtime,
                        precision=precision,
                    )

                    if isinstance(results, ExportResult):
                        details[str(precision)]["model"] = (
                            get_model_size_and_parameters(results.compile_job)
                        )
                    elif isinstance(results, LegacyCollectionExportResult):
                        for component_name, er in results.components.items():
                            details[str(precision)][component_name] = (
                                get_model_size_and_parameters(er.compile_job)
                            )
                    elif isinstance(results, MultiGraphCollectionExportResult):
                        if results.compile_jobs:
                            seen_components: set[str] = set()
                            for (
                                component_name,
                                _graph_name,
                            ), compile_job in results.compile_jobs.items():
                                if component_name not in seen_components:
                                    details[str(precision)][component_name] = (
                                        get_model_size_and_parameters(compile_job)
                                    )
                                    seen_components.add(component_name)
                    elif isinstance(results, CollectionExportResult):
                        if results.compile_jobs:
                            for (
                                component_name,
                                compile_job,
                            ) in results.compile_jobs.items():
                                details[str(precision)][component_name] = (
                                    get_model_size_and_parameters(compile_job)
                                )
                    else:
                        raise NotImplementedError(  # noqa: TRY301
                            f"Unknown export script result type: {type(results)}"
                        )
            else:
                if perf.empty:
                    print(f"No perf data for model {model_name}. Skipping.")
                    continue
                for precision, precision_data in perf.precisions.items():
                    precision_str = str(precision)
                    for (
                        submodel_name,
                        submodel_data,
                    ) in precision_data.components.items():
                        details_key = (
                            submodel_name
                            if len(precision_data.components) > 1
                            else "model"
                        )
                        # Get model from first available profile job
                        found = False
                        for device_data in submodel_data.performance_metrics.values():
                            for runtime_data in device_data.values():
                                if job_id := runtime_data.job_id:
                                    try:
                                        job = hub.get_job(job_id)
                                        details[precision_str][details_key] = (
                                            get_model_size_and_parameters(job)
                                        )
                                        found = True
                                    except Exception:
                                        continue
                                if found:
                                    break
                            if found:
                                break

            config_path = SCORECARD_MODELS_ROOT / model_name / "internal-config.yaml"
            config_dict: dict[str, Any] = {}
            if config_path.exists():
                config_dict = yaml.load(config_path) or {}
            new_config = add_details_to_internal_config(
                config_dict=config_dict, details=details
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "wb") as f:
                yaml.dump(new_config, f)

        except Exception as e:
            raise ValueError(f"Failed for model {model_name}") from e


if __name__ == "__main__":
    main()

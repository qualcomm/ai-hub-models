# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import os

import qai_hub as hub
import torch

# Some torchscript modules have an implicit dependency on this to load it
import torchvision  # noqa: F401

from qai_hub_models.scorecard import ScorecardInferencePath
from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    DeploymentEnvvar,
    EnabledModelsEnvvar,
    SpecialModelSetting,
    StaticModelsDirEnvvar,
)
from qai_hub_models.scorecard.params import ScJobParams
from qai_hub_models.scorecard.results.yaml import InferenceScorecardJobYaml
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.scorecard.static.model_config import ScorecardModelConfig
from qai_hub_models.scorecard.utils.testing_async_utils import write_accuracy
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.compare import compute_psnr
from qai_hub_models.utils.onnx.torch_wrapper import OnnxModelTorchWrapper
from qai_hub_models.utils.printing import format_metric
from qai_hub_models.utils.qai_hub_helpers import (
    download_model_in_memory,
    parse_compile_options,
)
from qai_hub_models.utils.transpose_channel import transpose_channel_last_to_first


def evaluate_model_accuracy(
    model_id: str, deployment: str, inference_jobs_yaml: InferenceScorecardJobYaml
) -> None:
    config = ScorecardModelConfig.from_scorecard_model_id(model_id)
    dataset = hub.get_dataset(
        config.hub_input_dataset_ids_automated[deployment]
    ).download()
    model_type = config.type
    hub_model = hub.get_model(config.hub_model_ids_automated[deployment])
    if model_type == hub.SourceModelType.TORCHSCRIPT:
        model = download_model_in_memory(hub_model)
    else:
        with qaihm_temp_dir() as tmp_dir:
            model_file = hub_model.download(os.path.join(tmp_dir, "model.onnx"))
            model = OnnxModelTorchWrapper.OnCPU(model_file)

    for device in config.devices:
        for runtime in config.enabled_profile_runtimes:
            params = ScJobParams(
                model_id=model_id,
                precision=config.precision,
                path=runtime,
                device=device,
            )
            job = inference_jobs_yaml.get_job(
                params,
                wait_for_job=True,
            )

            if job is None:
                continue
            device_outputs = job.job.download_output_data()

            if device_outputs is None:
                job_name = f"qaihm::inference | {params.inference_job_id}"
                print(f"{job_name} | Job failed | {job.job.url}")
                continue
            if (batch_size := len(next(iter(dataset.values())))) != 1:
                job_name = f"qaihm::inference | {params.inference_job_id}"
                print(
                    f"{job_name} | Batch size must be 1, got {batch_size} | {job.job.url}"
                )
                continue

            compile_job = job.job.model.get_producer()
            assert isinstance(compile_job, hub.CompileJob)
            compile_options = parse_compile_options(compile_job)

            transposed_device_outputs = transpose_channel_last_to_first(
                compile_options.channel_last_output or [], device_outputs
            )
            model_inputs = tuple(torch.Tensor(t[0]) for t in dataset.values())
            with torch.no_grad():
                cpu_outputs = [out.numpy() for out in model(*model_inputs)]
            psnrs: list[str] = []
            for cpu_output, device_output in zip(
                cpu_outputs, transposed_device_outputs.values(), strict=False
            ):
                if len(psnrs) == 10:
                    break
                # Compute PSNR just on the first sample, even if dataset has multiple samples
                psnrs.append(format_metric(compute_psnr(cpu_output, device_output[0])))
            write_accuracy(
                model_id,
                device.chipset,
                config.precision,
                ScorecardInferencePath(runtime),
                psnrs,
            )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR for static models by comparing the inference job "
        "output with local onnxruntime inference.",
    )
    EnabledModelsEnvvar.add_arg(parser, {SpecialModelSetting.STATIC})
    StaticModelsDirEnvvar.add_arg(parser)
    ArtifactsDirEnvvar.add_arg(parser)
    DeploymentEnvvar.add_arg(parser)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    _, model_id_list = validate_and_split_enabled_models(
        args.models, args.static_models_dir
    )
    exceptions = []
    for model_id in sorted(model_id_list):
        print(model_id)
        try:
            evaluate_model_accuracy(
                model_id,
                args.deployment,
                InferenceScorecardJobYaml.from_test_artifacts(),
            )
        except Exception as e:
            exceptions.append(f"{model_id}: {e}")
    if len(exceptions) > 0:
        exceptions_str = "\n".join(exceptions)
        raise RuntimeError(
            f"Failed to evaluate numerics for some models: {exceptions_str}"
        )


if __name__ == "__main__":
    main()

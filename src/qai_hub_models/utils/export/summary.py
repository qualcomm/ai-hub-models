# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tool-version extraction and end-of-run summary printers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import qai_hub as hub

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.inference import job_failed_error
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.printing import (
    compute_inference_metrics,
    print_profile_metrics_from_job,
)


def extract_tool_versions(
    inference_job: hub.InferenceJob | None,
    compile_job: hub.CompileJob | None,
) -> tuple[ToolVersions | None, bool]:
    """Return ``(tool_versions, came_from_inference_job)``, falling back to the compile job."""
    if inference_job is not None and inference_job.wait():
        return ToolVersions.from_job(inference_job), True
    if compile_job is not None and compile_job.wait():
        return ToolVersions.from_job(compile_job), False
    return None, False


def print_profile_summary(inference_job: hub.InferenceJob) -> None:
    """Wait for the inference job and print latency / peak-memory metrics."""
    if not inference_job.wait().success:
        raise job_failed_error(inference_job)
    profile_data: dict[str, Any] = inference_job.download_profile()
    print_profile_metrics_from_job(inference_job, profile_data)


def compute_multi_graph_inference_summary(
    model: MultiGraphWorkbenchModel,
    graph_name: str,
    inference_job: hub.InferenceJob,
    input_spec: InputSpec,
    outputs_to_skip: list[int] | None = None,
    metrics: str = "psnr",
    print_result: bool = False,
) -> pd.DataFrame:
    """
    Compare one graph's on-device outputs against its torch baseline and return
    the metrics.

    Waits for ``inference_job``.
    """
    sample_inputs = model.get_graph_sample_inputs(
        graph_name, input_spec, use_channel_last_format=False
    )
    torch_out = model.local_inference(graph_name, sample_inputs)
    if not inference_job.wait().success:
        raise job_failed_error(inference_job)
    ij_output = inference_job.download_output_data()
    assert ij_output is not None
    return compute_inference_metrics(
        ij_output,
        torch_out,
        list(model.get_graph_output_spec(graph_name)),
        outputs_to_skip=outputs_to_skip or None,
        metrics=metrics,
        print_result=print_result,
        inference_job=inference_job,
    )


def compute_inference_summary(
    model: BaseModel,
    inference_job: hub.InferenceJob,
    input_spec: InputSpec,
    target_runtime: TargetRuntime,
    outputs_to_skip: list[int] | None = None,
    metrics: str = "psnr",
    print_result: bool = False,
) -> pd.DataFrame:
    """
    Compare on-device outputs against a torch baseline and return the metrics.

    Waits for ``inference_job``.
    """
    sample_inputs = model.sample_inputs(input_spec, use_channel_last_format=False)
    torch_out = torch_inference(
        model,
        sample_inputs,
        return_channel_last_output=target_runtime.channel_last_native_execution,
    )
    if not inference_job.wait().success:
        raise job_failed_error(inference_job)
    ij_output = inference_job.download_output_data()
    assert ij_output is not None
    return compute_inference_metrics(
        ij_output,
        torch_out,
        list(model.get_output_spec()),
        outputs_to_skip=outputs_to_skip or None,
        metrics=metrics,
        print_result=print_result,
        inference_job=inference_job,
    )

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Run on-device inference on sample inputs."""

from __future__ import annotations

import qai_hub as hub

from qai_hub_models import SampleInputsType
from qai_hub_models.utils.export.result import (
    ComponentGroup,
    MultiGraphComponentGroup,
    MultiGraphGroup,
)


def _running_or_profiling_msg(name: str, has_inputs: bool) -> str:
    if has_inputs:
        return f"Running inference for {name} on a hosted device with example inputs."
    return f"Profiling {name} on a hosted device."


def run_inference(
    inputs: SampleInputsType | None,
    model_name: str,
    device: hub.Device,
    options: str,
    target_model: hub.Model,
) -> hub.client.InferenceJob:
    """Submit an on-device inference job. Also profiles, since Workbench folds
    profiling into any inference job. ``inputs=None`` submits a profile-only job.
    """
    print(_running_or_profiling_msg(model_name, inputs is not None))
    return hub.submit_inference_job(
        model=target_model,
        inputs=inputs,
        device=device,
        name=model_name,
        options=options,
    )


def run_collection_inference(
    inputs_per_component: ComponentGroup[SampleInputsType] | None,
    model_name: str,
    device: hub.Device,
    options_per_component: dict[str, str],
    target_models: ComponentGroup[hub.Model],
    components: list[str] | None = None,
) -> ComponentGroup[hub.client.InferenceJob]:
    """Submit one inference job per component. ``inputs_per_component=None``
    submits profile-only jobs for every component.
    """
    components = components if components is not None else list(target_models)
    inference_jobs: ComponentGroup[hub.client.InferenceJob] = ComponentGroup()
    for name in components:
        inputs = (
            inputs_per_component[name] if inputs_per_component is not None else None
        )
        print(_running_or_profiling_msg(name, inputs is not None))
        inference_jobs[name] = hub.submit_inference_job(
            model=target_models[name],
            inputs=inputs,
            device=device,
            name=f"{model_name}_{name}",
            options=options_per_component.get(name, ""),
        )
    return inference_jobs


def run_multi_graph_inference(
    inputs: MultiGraphGroup[SampleInputsType] | None,
    model_name: str,
    device: hub.Device,
    options_per_graph: MultiGraphGroup[str],
    target_model: hub.Model,
) -> MultiGraphGroup[hub.client.InferenceJob]:
    """
    Submit one inference job per graph against the single linked context binary.

    Each ``options`` string selects which graph inside the model to run.
    ``inputs=None`` submits profile-only jobs for every graph.
    """
    inference_jobs: MultiGraphGroup[hub.client.InferenceJob] = MultiGraphGroup()
    for graph_name, opts in options_per_graph.items():
        graph_inputs = inputs[graph_name] if inputs is not None else None
        print(
            _running_or_profiling_msg(
                f"{model_name} ({graph_name})", graph_inputs is not None
            )
        )
        inference_jobs[graph_name] = hub.submit_inference_job(
            model=target_model,
            inputs=graph_inputs,
            device=device,
            name=f"{model_name}_{graph_name}",
            options=opts,
        )
    return inference_jobs


def run_multi_graph_collection_inference(
    inputs: MultiGraphComponentGroup[SampleInputsType] | None,
    model_name: str,
    device: hub.Device,
    options_per_graph: MultiGraphComponentGroup[str],
    target_models: ComponentGroup[hub.Model],
    components: list[str] | None = None,
) -> MultiGraphComponentGroup[hub.client.InferenceJob]:
    """
    Submit one inference job per ``(component, graph)``.

    The hub target model is selected per-component (link consolidates all graphs
    of a component into one model); ``options_per_graph`` selects the graph
    inside that model. ``inputs=None`` submits profile-only jobs for every
    ``(component, graph)``.
    """
    components = components if components is not None else list(target_models)
    inference_jobs: MultiGraphComponentGroup[hub.client.InferenceJob] = (
        MultiGraphComponentGroup()
    )
    for (comp_name, graph_name), opts in options_per_graph.items():
        if comp_name not in components:
            continue
        graph_inputs = inputs[(comp_name, graph_name)] if inputs is not None else None
        print(_running_or_profiling_msg(comp_name, graph_inputs is not None))
        inference_jobs[(comp_name, graph_name)] = hub.submit_inference_job(
            model=target_models[comp_name],
            inputs=graph_inputs,
            device=device,
            name=f"{model_name}_{comp_name}_{graph_name}",
            options=opts,
        )
    return inference_jobs

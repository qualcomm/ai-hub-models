# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Quantize ONNX hub models via AI Hub quantize jobs."""

from __future__ import annotations

import qai_hub as hub
from qai_hub.client import DatasetEntries
from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models import Precision
from qai_hub_models.utils import quantization as quantization_utils
from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import InputSpec


def run_quantize(
    precision: Precision,
    model: BaseModel,
    model_name: str,
    onnx_model: hub.Model,
    num_calibration_samples: int | None,
    extra_options: str = "",
    input_spec: InputSpec | None = None,
    calibration_data: hub.Dataset | None = None,
) -> hub.client.QuantizeJob:
    """Submit a quantize job for ``onnx_model``, building calibration data unless supplied."""
    input_spec = input_spec or model.get_input_spec()
    print(f"Quantizing {model_name}.")
    if not precision.activations_type or not precision.weights_type:
        raise CLIUserError(
            "Quantization is only supported if both weights and activations are quantized."
        )
    data: hub.Dataset | DatasetEntries = (
        calibration_data
        if calibration_data is not None
        else quantization_utils.get_calibration_data(
            model, input_spec, num_calibration_samples
        )
    )
    return hub.submit_quantize_job(
        model=onnx_model,
        calibration_data=data,
        activations_dtype=precision.activations_type,
        weights_dtype=precision.weights_type,
        name=model_name,
        options=model.get_hub_quantize_options(precision, extra_options),
    )


def resolve_component_precisions(
    model: CollectionModel,
    precision: Precision,
    components: list[str],
) -> dict[str, Precision]:
    """Expand mixed-precision shorthand to a per-component mapping."""
    if precision in (Precision.mixed, Precision.mixed_with_float):
        return model.get_mixed_precisions(precision)
    return dict.fromkeys(components, precision)


def run_collection_quantize(
    component_precisions: Precision | dict[str, Precision],
    model: CollectionModel,
    model_name: str,
    onnx_models: ComponentGroup[hub.Model],
    num_calibration_samples: int | None,
    extra_options: str = "",
    input_specs: dict[str, InputSpec] | None = None,
    app: type | None = None,
    calibration_data: ComponentGroup[hub.Dataset] | None = None,
) -> ComponentGroup[hub.client.QuantizeJob]:
    """
    Submit one quantize job per component (skipping float-precision components).

    ``component_precisions`` may be a single ``Precision`` (applied to every
    onnx_models key) or a per-component dict. Calibration data is built for
    each component missing from ``calibration_data``.
    """
    if isinstance(component_precisions, Precision):
        component_precisions = resolve_component_precisions(
            model, component_precisions, list(onnx_models)
        )
    input_specs = input_specs or model.get_input_spec()
    quantize_jobs: ComponentGroup[hub.client.QuantizeJob] = ComponentGroup()
    for name, precision in component_precisions.items():
        if precision == Precision.float:
            continue
        print(f"Quantizing {name}.")
        if not precision.activations_type or not precision.weights_type:
            raise CLIUserError(
                "Quantization is only supported if both weights and activations are quantized."
            )
        supplied = calibration_data.get(name) if calibration_data else None
        component_data = supplied or quantization_utils.get_calibration_data(
            model,
            input_specs,
            num_calibration_samples,
            component_name=name,
            app=app,
        )
        quantize_jobs[name] = hub.submit_quantize_job(
            model=onnx_models[name],
            calibration_data=component_data,
            activations_dtype=precision.activations_type,
            weights_dtype=precision.weights_type,
            name=f"{model_name}_{name}",
            options=model.get_component_hub_quantize_options(
                name,
                precision,
                extra_options,
            ),
        )
    return quantize_jobs

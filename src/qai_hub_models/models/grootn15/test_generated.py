# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import qai_hub as hub
import torch

import qai_hub_models.models.grootn15 as _model_module
from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.grootn15 import MODEL_ID, Model
from qai_hub_models.models.grootn15.export import (
    compile_model,
    export_model,
    inference_model,
    link_model,
    profile_model,
    quantize_model,
    upload_model,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.errors import CachedScorecardJobError
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    get_evaluation_parameterized_pytest_config,
    get_export_parameterized_pytest_config,
    get_link_parameterized_pytest_config,
    get_profile_parameterized_pytest_config,
    get_quantize_parameterized_pytest_config,
    needs_pre_quantize_compile,
    pytest_device_idfn,
)
from qai_hub_models.scorecard.utils.testing import skip_invalid_runtime_device
from qai_hub_models.scorecard.utils.testing_export_eval import (
    accuracy_on_sample_inputs_via_export,
    compile_via_export,
    export_test_e2e,
    inference_via_export,
    link_via_export,
    pre_quantize_compile_via_export,
    profile_via_export,
    quantize_via_export,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.validation import perform_runtime_model_validation

# All runtime + precision pairs that are enabled for testing and are compatibile with this model.
# NOTE:
#   Certain supported pairs may be excluded from this list if they are not enabled for testing.
#   For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
ENABLED_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.float: [
        TargetRuntime.QNN_CONTEXT_BINARY,
        TargetRuntime.PRECOMPILED_QNN_ONNX,
    ],
}


# All runtime + precision pairs that are enabled for testing and have no known failure reasons.
# NOTE:
#   Certain supported pairs may be excluded from this list if they are not enabled for testing.
#   For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
PASSING_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.float: [
        TargetRuntime.QNN_CONTEXT_BINARY,
        TargetRuntime.PRECOMPILED_QNN_ONNX,
    ],
}


EVAL_DEVICE = ScorecardDevice.get("Dragonwing IQ-9075 EVK")
HAS_EVAL_DATASET = len(Model.get_eval_dataset_classes()) > 0


@pytest.mark.compile
def test_runtime_model_validation() -> None:
    perform_runtime_model_validation(
        Model, MODEL_ID, getattr(_model_module, "App", None)
    )


@pytest.mark.pre_quantize_compile
@pytest.mark.skipif(
    not needs_pre_quantize_compile(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    reason="Model does not require pre-quantize compile step",
)
def test_pre_quantize_compile() -> None:
    pre_quantize_compile_via_export(
        compile_model,
        MODEL_ID,
        Model.from_pretrained(),
        upload_model,
    )


@pytest.mark.parametrize(
    "precision",
    get_quantize_parameterized_pytest_config(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.quantize
def test_quantize(precision: Precision) -> None:
    try:
        quantize_via_export(
            quantize_model,
            MODEL_ID,
            Model.from_pretrained(),
            precision,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_compile_parameterized_pytest_config(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.compile
def test_compile(
    precision: Precision, scorecard_path: ScorecardCompilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        compile_via_export(
            compile_model,
            MODEL_ID,
            Model.from_pretrained(),
            precision,
            scorecard_path,
            device,
            upload_model=upload_model,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_link_parameterized_pytest_config(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.link
def test_link(
    precision: Precision, scorecard_path: ScorecardCompilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        link_via_export(
            link_model,
            MODEL_ID,
            Model.from_pretrained(),
            precision,
            scorecard_path,
            device,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_profile_parameterized_pytest_config(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.profile
def test_profile(
    precision: Precision, scorecard_path: ScorecardProfilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        profile_via_export(
            profile_model,
            MODEL_ID,
            Model.from_pretrained(),
            precision,
            scorecard_path,
            device,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_evaluation_parameterized_pytest_config(
        MODEL_ID,
        EVAL_DEVICE,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.inference
def test_inference(
    precision: Precision, scorecard_path: ScorecardProfilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        inference_via_export(
            inference_model,
            MODEL_ID,
            Model.from_pretrained(),
            precision,
            scorecard_path,
            device,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def torch_val_outputs() -> list[np.ndarray]:
    """
    Because the below method downloads a dataset over the internet,
    it is called in a fixture so it can be reused.
    """
    # Collection models are not torch-accuracy validated in the scorecard.
    return []


@pytest.fixture(scope="module")
def torch_evaluate_mock_outputs(
    torch_val_outputs: list[np.ndarray],
) -> list[torch.Tensor | tuple[torch.Tensor, ...]]:
    """
    Because the below method does some memory movement,
    it is called in a fixture so its output can be reused.
    """
    # Collection models are not torch-accuracy validated in the scorecard.
    return []


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_evaluation_parameterized_pytest_config(
        MODEL_ID,
        EVAL_DEVICE,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.compute_device_accuracy
def test_val_accuracy(
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    torch_val_outputs: list[np.ndarray],
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
) -> None:
    try:
        accuracy_on_sample_inputs_via_export(
            export_model,
            MODEL_ID,
            Model.from_pretrained(),
            precision,
            scorecard_path,
            device,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_export_parameterized_pytest_config(
        MODEL_ID,
        EVAL_DEVICE,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
        requires_aot_prepare=True,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.export
def test_export(
    precision: Precision, scorecard_path: ScorecardProfilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        export_test_e2e(
            export_model, Model, MODEL_ID, precision, scorecard_path, device
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


# Cache serialize() and hub.upload_model() across the module so the same
# (component, graph, input_spec) is serialized once and the resulting bytes are
# uploaded once -- matters most for multi-GB AIMET LLM bundles.
@pytest.fixture(scope="module", autouse=True)
def cached_serialize_for_export(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[pytest.MonkeyPatch, None, None]:
    cache_dir = tmp_path_factory.mktemp("serialize_cache")
    with pytest.MonkeyPatch.context() as mp:
        model_cache: dict[str, Path] = {}
        upload_cache: dict[str, hub.Model] = {}

        real_upload_model = hub.upload_model

        def _cached_upload_model(
            model: hub.client.SourceModel | str,
            name: str | None = None,
            project: str | hub.client.Project | None = None,
        ) -> hub.Model:
            key = str(model)
            cached = upload_cache.get(key)
            if cached is None:
                cached = real_upload_model(model, name, project)
                upload_cache[key] = cached
            return cached

        mp.setattr(hub, "upload_model", _cached_upload_model)
        serialize_component = Model.serialize_component

        def _cached_serialize_component(
            self: Model,
            component_name: str,
            output_dir: str | os.PathLike,
            input_spec: InputSpec | None = None,
        ) -> Path:
            model_key = component_name + str(input_spec)
            cached = model_cache.get(model_key)
            if not cached:
                cached = serialize_component(
                    self, component_name, cache_dir, input_spec
                )
                model_cache[model_key] = cached
            return cached

        mp.setattr(Model, "serialize_component", _cached_serialize_component)
        yield mp

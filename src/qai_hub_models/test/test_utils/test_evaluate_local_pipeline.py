# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import qai_hub as hub

from qai_hub_models.utils.evaluate import collection_pipeline, pipeline


def _fake_model() -> mock.MagicMock:
    model = mock.MagicMock()
    model.get_input_spec.return_value = {"image": object()}
    model.get_eval_dataset_classes.return_value = [object]
    return model


@pytest.fixture
def patched_pipeline() -> Generator[SimpleNamespace]:
    model = _fake_model()
    model_cls = mock.MagicMock()
    model_cls.get_eval_dataset_classes.return_value = [object]
    model_cls.from_pretrained.return_value = model
    with (
        mock.patch.object(pipeline, "resolve_model_cls", return_value=model_cls),
        mock.patch.object(
            pipeline,
            "resolve_manifest",
            return_value=SimpleNamespace(
                can_use_quantize_job=False, supports_quantization=False
            ),
        ),
        mock.patch.object(pipeline, "get_model_kwargs", return_value={}),
        mock.patch.object(pipeline, "filter_kwargs", return_value={}),
        mock.patch.object(pipeline, "evaluate_on_dataset") as evaluate_on_dataset,
        mock.patch.object(pipeline, "compile_model_from_args") as compile_model,
        mock.patch.object(pipeline, "PrecompiledQnnOnnxModel") as local_model,
        mock.patch.object(
            pipeline, "find_component_onnx", return_value=Path("/m/w.onnx")
        ),
    ):
        yield SimpleNamespace(
            model=model,
            evaluate_on_dataset=evaluate_on_dataset,
            compile_model=compile_model,
            local_model=local_model,
        )


def _run(**kwargs: object) -> None:
    pipeline.evaluate_model(
        source_dir=Path("/recipe"),
        device=hub.Device("fake"),
        dataset_cls=object,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def test_local_dir_runs_model_locally_without_workbench(
    patched_pipeline: SimpleNamespace,
) -> None:
    _run(local_model_dir="/some/dir")

    patched_pipeline.compile_model.assert_not_called()
    patched_pipeline.local_model.assert_called_once_with(
        Path("/m/w.onnx"), input_names=["image"]
    )
    executors = patched_pipeline.evaluate_on_dataset.call_args.kwargs["model_executors"]
    assert set(executors) == {"torch", "on-device"}
    assert executors["on-device"] is patched_pipeline.local_model.return_value


def test_local_dir_with_skip_device_does_not_load_the_model(
    patched_pipeline: SimpleNamespace,
) -> None:
    _run(local_model_dir="/some/dir", skip_device_accuracy=True)

    patched_pipeline.local_model.assert_not_called()
    executors = patched_pipeline.evaluate_on_dataset.call_args.kwargs["model_executors"]
    assert set(executors) == {"torch"}


def test_local_dir_without_torch_still_gets_an_input_spec(
    patched_pipeline: SimpleNamespace,
) -> None:
    _run(local_model_dir="/some/dir", skip_torch_accuracy=True)

    kwargs = patched_pipeline.evaluate_on_dataset.call_args.kwargs
    assert set(kwargs["model_executors"]) == {"on-device"}
    assert list(kwargs["input_spec"]) == ["image"]


def test_local_dir_conflicts_with_quant_cpu_accuracy(
    patched_pipeline: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError, match="compute-quant-cpu-accuracy"):
        _run(local_model_dir="/some/dir", compute_quant_cpu_accuracy=True)

    patched_pipeline.evaluate_on_dataset.assert_not_called()


def test_collection_local_dir_conflicts_with_quant_cpu_accuracy() -> None:
    with (
        mock.patch.object(collection_pipeline, "resolve_model_cls"),
        mock.patch.object(collection_pipeline, "resolve_model_app_cls"),
        mock.patch.object(collection_pipeline, "resolve_manifest"),
        mock.patch.object(collection_pipeline, "get_model_kwargs", return_value={}),
        pytest.raises(ValueError, match="compute-quant-cpu-accuracy"),
    ):
        collection_pipeline.evaluate_model(
            source_dir=Path("/recipe"),
            device=hub.Device("fake"),
            local_model_dir="/some/dir",
            compute_quant_cpu_accuracy=True,
        )

# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from qai_hub_models.utils import precompiled_qnn_onnx
from qai_hub_models.utils.precompiled_qnn_onnx import (
    PrecompiledQnnOnnxModel,
    find_component_onnx,
)


def _fake_session(
    input_types: dict[str, str], outputs: list[np.ndarray]
) -> mock.MagicMock:
    session = mock.MagicMock()
    session.get_inputs.return_value = [
        SimpleNamespace(name=n, type=t) for n, t in input_types.items()
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name=f"out{i}") for i in range(len(outputs))
    ]
    session.run.return_value = outputs
    return session


@pytest.fixture
def fake_qnn() -> Generator[mock.MagicMock]:
    precompiled_qnn_onnx._register_qnn_ep.cache_clear()
    qnn = mock.MagicMock()
    qnn.get_library_path.return_value = "/fake/libqnn_ep.so"
    qnn.get_qnn_htp_path.return_value = "/fake/libQnnHtp.so"
    with (
        mock.patch.object(precompiled_qnn_onnx, "onnxruntime_qnn", qnn),
        mock.patch.object(precompiled_qnn_onnx, "ort") as ort,
    ):
        ort.get_ep_devices.return_value = [
            SimpleNamespace(ep_name="QNNExecutionProvider")
        ]
        qnn.ort = ort
        yield qnn


def test_find_component_onnx(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_component_onnx(tmp_path)
    (tmp_path / "a.onnx").touch()
    assert find_component_onnx(tmp_path) == tmp_path / "a.onnx"
    (tmp_path / "b.onnx").touch()
    with pytest.raises(FileNotFoundError):
        find_component_onnx(tmp_path)


def test_ep_registered_once_across_instances(fake_qnn: mock.MagicMock) -> None:
    ort = fake_qnn.ort
    ort.InferenceSession.return_value = _fake_session(
        {"x": "tensor(float)"}, [np.zeros((1,), np.float32)]
    )
    PrecompiledQnnOnnxModel("a.onnx", input_names=["x"])
    PrecompiledQnnOnnxModel("b.onnx", input_names=["x"])
    assert ort.register_execution_provider_library.call_count == 1


def test_call_maps_inputs_by_name_and_dtype(fake_qnn: mock.MagicMock) -> None:
    ort = fake_qnn.ort
    session = _fake_session(
        {"tokens": "tensor(int32)", "x": "tensor(float)"},
        [np.ones((2,), np.float32)],
    )
    ort.InferenceSession.return_value = session
    model = PrecompiledQnnOnnxModel("m.onnx", input_names=["x", "tokens"])

    out = model(torch.zeros(1, 3), torch.arange(4, dtype=torch.int64))

    assert isinstance(out, torch.Tensor)
    feed = session.run.call_args.args[1]
    assert feed["tokens"].dtype == np.int32
    assert feed["x"].dtype == np.float32


def test_multiple_outputs_return_tuple(fake_qnn: mock.MagicMock) -> None:
    ort = fake_qnn.ort
    ort.InferenceSession.return_value = _fake_session(
        {"x": "tensor(float)"}, [np.zeros(1, np.float32), np.ones(1, np.float32)]
    )
    out = PrecompiledQnnOnnxModel("m.onnx", input_names=["x"])(torch.zeros(1))
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_graph_input_missing_from_input_names_raises(
    fake_qnn: mock.MagicMock,
) -> None:
    ort = fake_qnn.ort
    ort.InferenceSession.return_value = _fake_session(
        {"x": "tensor(float)", "y": "tensor(float)"}, [np.zeros(1, np.float32)]
    )
    with pytest.raises(ValueError, match="not in input_names"):
        PrecompiledQnnOnnxModel("m.onnx", input_names=["x"])

# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Run a precompiled_qnn_onnx component (EPContext wrapper + QNN .bin) locally."""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

try:
    import onnxruntime_qnn
except ImportError:
    onnxruntime_qnn = None

_EP_NAME = "QNNExecutionProvider"
_ORT_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


@functools.cache
def _register_qnn_ep() -> None:
    # ORT raises if the same EP library is registered twice.
    ort.register_execution_provider_library(
        _EP_NAME, onnxruntime_qnn.get_library_path()
    )


def find_component_onnx(component_dir: str | os.PathLike) -> Path:
    """The single wrapper ``.onnx`` in a component directory."""
    candidates = sorted(Path(component_dir).glob("*.onnx"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .onnx in {component_dir}, found {candidates}."
        )
    return candidates[0]


class PrecompiledQnnOnnxModel:
    """
    An ``ExecutableModelProtocol`` over a precompiled_qnn_onnx wrapper.

    Accepts torch tensors positionally (in ``input_names`` order) or by keyword,
    and returns a tensor for a single output or a tuple in graph output order.
    """

    def __init__(
        self, onnx_path: str | os.PathLike, input_names: Sequence[str]
    ) -> None:
        if onnxruntime_qnn is None:
            raise ImportError(
                "onnxruntime-qnn is required to run a precompiled_qnn_onnx model."
            )
        _register_qnn_ep()
        htp_path = onnxruntime_qnn.get_qnn_htp_path()
        os.environ["ADSP_LIBRARY_PATH"] = os.path.dirname(htp_path)

        options = ort.SessionOptions()
        options.add_provider_for_devices(
            [d for d in ort.get_ep_devices() if d.ep_name == _EP_NAME],
            {"backend_path": htp_path},
        )
        self.session = ort.InferenceSession(str(onnx_path), sess_options=options)
        self.input_names = list(input_names)
        self._input_types = {i.name: i.type for i in self.session.get_inputs()}
        self._output_names = [o.name for o in self.session.get_outputs()]
        missing = set(self._input_types) - set(self.input_names)
        if missing:
            raise ValueError(f"{onnx_path} has inputs not in input_names: {missing}")

    def __call__(
        self, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        named: dict[str, Any] = dict(zip(self.input_names, args, strict=False))
        named.update(kwargs)
        feed = {
            name: np.ascontiguousarray(
                named[name].detach().cpu().numpy(),
                dtype=_ORT_TO_NUMPY[dtype],
            )
            for name, dtype in self._input_types.items()
        }
        outputs = [torch.from_numpy(o) for o in self.session.run(None, feed)]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

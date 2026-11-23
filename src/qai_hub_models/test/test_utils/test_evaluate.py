# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.utils.data import DataLoader

from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate.helpers import (
    evaluate,
)
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, make_torch_inputs
from qai_hub_models.utils.onnx.torch_wrapper import OnnxModelTorchWrapper


class VariableIODummyModel(BaseModel):
    """Dummy AI Hub Workbench model that allows changing the number of inputs / outputs."""

    DEFAULT_IO_SHAPE = (1, 3, 2, 2)

    def __init__(
        self,
        num_inputs: int = 1,
        num_outputs: int = 1,
        shape: tuple[int, ...] = DEFAULT_IO_SHAPE,
    ) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.shape = shape
        assert self.num_inputs > 0
        assert self.num_outputs > 0

    @staticmethod
    def from_pretrained(
        num_inputs: int = 1,
        num_outputs: int = 1,
        shape: tuple[int, ...] = DEFAULT_IO_SHAPE,
    ) -> VariableIODummyModel:
        return VariableIODummyModel(num_inputs, num_outputs, shape)

    def forward(
        self, *args: torch.Tensor | float, **kwargs: torch.Tensor | float
    ) -> torch.Tensor | int | float | tuple[torch.Tensor | int | float, ...]:
        inputs: dict[str, torch.Tensor | int | float] = {}

        for i, arg in enumerate(args):
            inputs[f"in{i}"] = arg

        for kwarg_name, kwarg in kwargs.items():
            assert kwarg_name not in inputs, f"Specified input arg {kwarg_name} twice."
            assert kwarg_name in self.get_input_spec(), (
                f"Unknown input arg {kwarg_name}"
            )
            inputs[kwarg_name] = kwarg

        if len(inputs) < len(self.get_input_spec()):
            raise ValueError(
                f"Missing inputs: {self.get_input_spec().keys() - inputs.keys()}"
            )
        if len(inputs) > len(self.get_input_spec()):
            raise ValueError(
                f"{len(self.get_input_spec())} Provided additional inputs: {inputs.keys() - self.get_input_spec().keys()}"
            )

        out = [
            inputs[f"in{min(self.num_inputs - 1, outIdx)}"] * 2
            for outIdx in range(self.num_outputs)
        ]
        if len(out) < self.num_inputs:
            for i in range(len(out), self.num_inputs):
                out[min(self.num_outputs - 1, i - len(out))] *= inputs[f"in{i}"]

        return tuple(out) if len(out) != 1 else out[0]

    def get_input_spec(self) -> InputSpec:
        return {
            f"out{i}": TensorSpec(
                shape=self.shape,
                dtype="float32",
                apply_runtime_channel_reordering=len(self.shape) == 4,
            )
            for i in range(self.num_inputs)
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            f"out{i}": TensorSpec(apply_runtime_channel_reordering=len(self.shape) == 4)
            for i in range(self.num_outputs)
        }


class DummyEvaluator(BaseEvaluator):
    def __init__(self, num_outputs: int) -> None:
        self.num_outputs = num_outputs
        self.dummy_metric = 0
        assert self.num_outputs > 0

    @classmethod
    def from_dummy_model(cls, model: VariableIODummyModel) -> DummyEvaluator:
        return cls(model.num_outputs)

    @classmethod
    def get_evaluator_from_model_func(
        cls, model: VariableIODummyModel
    ) -> Callable[[], DummyEvaluator]:
        def get_evaluator() -> DummyEvaluator:
            return cls.from_dummy_model(model)

        return get_evaluator

    def add_batch(
        self,
        output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
        gt: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    ) -> None:
        if self.num_outputs != 1:
            assert isinstance(output, (tuple, list))
            assert isinstance(gt, (tuple, list))
        else:
            output = tuple(
                output,
            )
            gt = tuple(
                gt,
            )

        assert len(output) == len(gt)
        for single_output, single_gt in zip(output, gt, strict=False):
            assert isinstance(single_output, (torch.Tensor, float, int))
            assert isinstance(single_gt, (torch.Tensor, float, int))

        self.dummy_metric = self.dummy_metric + 1

    def reset(self) -> None:
        self.dummy_metric = 0

    def get_accuracy_score(self) -> int:
        return self.dummy_metric

    def formatted_accuracy(self) -> str:
        return f"batch count: {self.dummy_metric}"


class DummyDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit,
        num_inputs: int,
        num_outputs: int,
        shape: tuple[int, ...],
        num_samples: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__("", split)
        assert num_samples >= self.default_samples_per_job()
        self.data = [
            tuple(torch.rand(shape, dtype=dtype) for _ in range(num_inputs))
            for __ in range(num_samples)
        ]
        self.gt = [
            tuple(torch.rand(shape, dtype=dtype) for _ in range(num_outputs))
            for __ in range(num_samples)
        ]

    @classmethod
    def from_dummy_model(
        cls, model: VariableIODummyModel, split: DatasetSplit, num_samples: int
    ) -> DummyDataset:
        return cls(
            split,
            model.num_inputs,
            model.num_outputs,
            model.shape[1:],  # remove batch dimension, dataloader will handle it
            num_samples,
            torch.float32,
        )

    def __len__(self) -> int:
        return len(self.data)

    def _download_data(self) -> None:
        return

    def _validate_data(self) -> bool:
        return True

    @staticmethod
    def default_samples_per_job() -> int:
        return 10

    @classmethod
    def default_num_calibration_samples(cls) -> int:
        return cls.default_samples_per_job()

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor | tuple[torch.Tensor, ...]
    ]:
        data = self.data[index]
        gt = self.gt[index]
        return data if len(data) != 1 else data[0], gt if len(gt) != 1 else gt[0]


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize(
    ("num_samples", "dataloader_batch_size", "model_batch_size"),
    [
        (100, 10, 1),
        (100, 10, 5),
        (20, 1, 1),
        (20, 20, 20),
        (100, 18, 16),
        (100, 16, 32),
    ],
)
@pytest.mark.parametrize(
    ("num_inputs", "num_outputs"),
    [
        (1, 1),
        (1, 3),
        (2, 1),
        (3, 7),
        (4, 2),
    ],
)
def test_local_evaluate(
    num_inputs: int,  # number of model inputs
    num_outputs: int,  # number of model output
    num_samples: int,  # number of samples to eval
    dataloader_batch_size: int,  # batch size of data loader passed to evaluate
    model_batch_size: int,  # batch size of compiled ONNX model
    shuffle: bool,  # random dataloader shuffle on/off
) -> None:
    """Test local evaluation for PyTorch and ONNX models."""
    model = VariableIODummyModel(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        shape=(model_batch_size, 3, 2, 2),
    )
    with qaihm_temp_dir() as tmpdir:
        onnx_path = f"{tmpdir}/model.onnx"
        torch.onnx.export(
            model,
            tuple(make_torch_inputs(model.get_input_spec())),
            onnx_path,
            dynamo=False,
        )
        onnx_model = OnnxModelTorchWrapper.OnCPU(onnx_path)

    evaluator_func = DummyEvaluator.get_evaluator_from_model_func(model)
    dataset = DummyDataset.from_dummy_model(model, DatasetSplit.VAL, num_samples)
    dataloader = DataLoader(dataset, dataloader_batch_size, shuffle)

    if dataloader_batch_size % model_batch_size != 0:
        with pytest.raises(
            ValueError, match=r".*must evenly divide the DataLoader's batch size.*"
        ):
            _ = evaluate(
                dataloader,
                evaluator_func,
                {"torch": model, "onnx": onnx_model},
                model_batch_size,
                verbose=False,
            )
    else:
        out, _ = evaluate(
            dataloader,
            evaluator_func,
            {"torch": model, "onnx": onnx_model},
            model_batch_size,
            verbose=False,
        )
        assert out["torch"].get_accuracy_score() == num_samples // model_batch_size
        assert out["onnx"].get_accuracy_score() == num_samples // model_batch_size

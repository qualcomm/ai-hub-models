# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
from collections import defaultdict
from collections.abc import Callable, Generator
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import qai_hub as hub
from qai_hub.client import SourceModelType
from tabulate import tabulate

from qai_hub_models import TargetRuntime
from qai_hub_models.protocols import FromPretrainedProtocol
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.envvars import DeploymentEnvvar
from qai_hub_models.scorecard.utils.testing_async_utils import append_line_to_file
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.base_dataset import BaseDataset, get_folder_name
from qai_hub_models.utils.base_model import (
    PrecompiledWorkbenchModel,
    WorkbenchModel,
)
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.evaluate.helpers import get_torch_val_dataloader
from qai_hub_models.utils.hub_clients import (
    default_hub_client_as,
    get_hub_client,
)
from qai_hub_models.utils.inference import (
    AsyncOnDeviceResult,
    dataset_entries_from_batch,
)
from qai_hub_models.utils.input_spec import (
    InputSpec,
    get_channel_last,
    is_input_spec,
    is_input_spec_dict,
)
from qai_hub_models.utils.printing import format_metric
from qai_hub_models.utils.quantization import get_calibration_data

__all__ = [
    "get_hub_val_dataset",
    "get_val_dataset_id_keys",
    "has_get_unsupported_reason",
    "hub_test_deployment",
    "mock_get_calibration_data",
    "mock_on_device_model_call",
    "mock_tabulate_fn",
    "patch_qai_hub",
    "skip_invalid_runtime_device",
]

AnyModel = (
    WorkbenchModel
    | MultiGraphWorkbenchModel
    | PrecompiledWorkbenchModel
    | CollectionModel
    | MultiGraphCollectionModel
)
AnyModelCls = type[AnyModel]


@pytest.fixture(scope="session", autouse=True)
def hub_test_deployment() -> Generator[str, None, None]:
    """
    Set the global Hub client to match ``QAIHM_TEST_DEPLOYMENT`` for the whole session.

    Reads the deployment from :func:`DeploymentEnvvar.get` and swaps the
    global Hub client via :func:`default_hub_client_as`, so ``hub.*`` calls
    use the correct deployment regardless of what ``~/.qai_hub/client.ini``
    points at. Runs autouse so every test — including compile/link/profile
    submissions that don't explicitly request a fixture — hits the requested
    deployment.

    If no matching profile is configured under ``~/.qai_hub/``, the fixture
    is a no-op so tests that never touch Hub can still run without
    credentials.

    Yields
    ------
    str
        The deployment name (e.g. ``"prod"``, ``"dev"``).
    """
    deployment = DeploymentEnvvar.get()
    client = get_hub_client(deployment)
    if client is None:
        yield deployment
        return
    with default_hub_client_as(client):
        yield deployment


def mock_on_device_model_call(inference_job: hub.InferenceJob) -> Callable:
    def mock_call(self: Any, *args: Any) -> AsyncOnDeviceResult:
        return AsyncOnDeviceResult(
            inference_job,
            self.target_runtime,
            self.channel_last_output,
            self.output_names,
            num_retries=0,
        )

    return mock_call


def mock_tabulate_fn(df: pd.DataFrame, **kwargs: Any) -> tuple[list[str], str]:
    psnr_values = []
    for _, value in df.iterrows():
        psnr_values.append(format_metric(value.psnr))
    return psnr_values, tabulate(df, **kwargs)  # pyright: ignore[reportArgumentType]


def mock_get_calibration_data(
    model: WorkbenchModel | CollectionModel,
    input_spec_arg: InputSpec | dict[str, InputSpec] | None = None,
    num_samples: int | None = None,
    component_name: str | None = None,
    dataset_options: dict | None = None,
    app: Any = None,
) -> hub.Dataset:
    """
    Gets the calibration data needed to quantize the input model.

    Since many models use the same calibration data, we save the workbench dataset
    id to a file and re-use it across models to avoid re-uploading the same data
    to hub.

    Each dataset is uniquely identified by the dataset name and input spec.

    Parameters
    ----------
    model
        Model or collection model to get calibration data for.
    input_spec_arg
        For a single model: an InputSpec or None.
        For a collection model: a dict mapping component names to InputSpecs.
    num_samples
        Number of samples used to calibrate.
    component_name
        For collection models, the name of the component being calibrated.
    dataset_options
        Additional options to pass to the dataset constructor.
    app
        Application object. Default is None.

    Returns
    -------
    hub.Dataset
        Hub dataset object that was uploaded
    """
    # Resolve component spec for cache key
    component_spec: InputSpec | None = None
    if isinstance(model, CollectionModel):
        assert component_name is not None and is_input_spec_dict(input_spec_arg)
        component_spec = input_spec_arg.get(component_name)
        cache_prefix_name = f"{model.__class__.__name__}_{component_name}"
    else:
        assert isinstance(model, WorkbenchModel) and is_input_spec(input_spec_arg)
        component_spec = input_spec_arg
        calib_cls = model.get_calibration_dataset_cls()
        cache_prefix_name = (
            calib_cls.dataset_name() if calib_cls else model.__class__.__name__
        )

    cache_prefix = get_folder_name(cache_prefix_name, component_spec)
    cache_key = cache_prefix + "_train"
    dataset_ids_file = ScorecardArtifact.DATASET_IDS.touch()
    dataset_ids = load_yaml(dataset_ids_file)
    if dataset_ids and cache_key in dataset_ids:
        return hub.get_dataset(dataset_ids[cache_key])
    dataset = get_calibration_data(
        model,
        input_spec_arg,
        num_samples,
        component_name=component_name,
        app=app,
    )
    hub_dataset = hub.upload_dataset(dataset)
    append_line_to_file(dataset_ids_file, f"{cache_key}: {hub_dataset.dataset_id}")
    return hub_dataset


def get_val_dataset_id_keys(
    dataset_name: str, apply_channel_transpose: bool
) -> tuple[str, str]:
    """
    Gets the keys used to store the workbench dataset id in a yaml file.

    This gives the keys for the validation split used to compute accuracy.

    Some runtimes have channel transpose applied while others don't, so
    we need separate hub datasets for each.

    Parameters
    ----------
    dataset_name
        Name of dataset to get keys
    apply_channel_transpose
        Whether to apply channel last transpose to input data

    Returns
    -------
    input_key : str
        Dataset id key for input.
    gt_key : str
        Dataset id key for ground truth.
    """
    base_name = f"{dataset_name}_val_"
    if not apply_channel_transpose:
        base_name += "nt_"  # no transpose
    return base_name + "input", base_name + "gt"


def get_hub_val_dataset(
    dataset_cls: type[BaseDataset],
    ids_file: Path,
    model_cls: AnyModelCls,
    apply_channel_transpose: bool,
    num_samples: int | None = None,
) -> hub.Dataset:
    """
    Creates a hub dataset with a chunk of validation data from the given dataset.

    If the dataset ids are already present in the file, the function
    just loads and returns it.

    Otherwise, it creates the dataset and writes the ids to the file.

    The dataset is sampled by taking every N samples for the largest possible N
    that produces the number of requested samples.

    The dataset is produced using the input spec and channel last inputs of a
    representative model. The assumption being that all models using this dataset
    have the same values for both of those things.

    Parameters
    ----------
    dataset_cls
        Dataset class (subclass of BaseDataset).
    ids_file
        Path to file where dataset ids are stored.
    model_cls
        The model class using this data. Used to determine input spec
        and channel last inputs.
    apply_channel_transpose
        If False, returns all inputs in channel first format.
        If True, applies channel last transpose for the inputs specified by the model.
    num_samples
        Number of samples to sample from the full dataset. Default is None.

    Returns
    -------
    hub.Dataset
        Hub dataset containing validation data.
    """
    assert issubclass(model_cls, WorkbenchModel), (
        "CollectionModel is not yet supported by this function."
    )
    assert issubclass(model_cls, FromPretrainedProtocol)
    dataset_ids = load_yaml(ids_file)
    instance = model_cls.from_pretrained()
    input_spec = instance.get_input_spec()
    dataset_name = dataset_cls.dataset_name()
    folder_name = get_folder_name(dataset_name, input_spec)
    input_key, gt_key = get_val_dataset_id_keys(folder_name, apply_channel_transpose)
    if dataset_ids and input_key in dataset_ids:
        assert gt_key in dataset_ids
        return hub.get_dataset(dataset_ids[input_key])
    dataloader = get_torch_val_dataloader(dataset_cls, num_samples, input_spec)
    batch = next(iter(dataloader))
    input_entries, gt_entries = dataset_entries_from_batch(
        batch,
        list(input_spec.keys()),
        get_channel_last(input_spec) if apply_channel_transpose else [],
    )

    input_dataset = hub.upload_dataset(input_entries)
    gt_dataset = hub.upload_dataset(gt_entries)
    append_line_to_file(ids_file, f"{input_key}: {input_dataset.dataset_id}")
    append_line_to_file(ids_file, f"{gt_key}: {gt_dataset.dataset_id}")
    return input_dataset


@contextlib.contextmanager
def patch_qai_hub(
    model_type: SourceModelType = SourceModelType.ONNX,
) -> Generator[SimpleNamespace, None, None]:
    """
    Generic AI Hub Workbench patch with assertable mocks. Example::

        with patch_qai_hub() as mock_hub:
            hub.submit_inference_job(...)
            mock_hub.submit_inference_job.assert_called()
    """
    # Model
    model = mock.Mock(hub.Model)
    model.model_id = "mabcd1234"
    model.model_type = model_type
    model.metadata = defaultdict(lambda: "2.33")

    def _store_device(mock_obj: mock.Mock, *args: Any, **kwargs: Any) -> mock.Mock:
        if "device" in kwargs:
            mock_obj.device = kwargs["device"]
        return mock_obj

    # Compile
    mock_compile = mock.Mock(hub.CompileJob)
    mock_compile.job_id = "jcbcd1234"
    mock_compile.get_target_model.return_value = model
    mock_submit_compile_job = mock.Mock()
    mock_submit_compile_job.return_value = mock_compile

    # Inference
    mock_inference = mock.Mock(hub.InferenceJob)
    mock_inference.name = "Inference job"
    mock_inference.job_id = "jibcd1234"
    mock_inference.model = model
    # heavily abridged profile
    mock_inference.download_profile.return_value = {
        "execution_detail": {},
        "execution_summary": {
            "inference_memory_peak_range": [50000, 100000],
            "estimated_inference_time": 10000,
            "peak_memory_bytes": 12000,
        },
    }
    mock_inference.download_output_data.return_value = {
        "output0": [np.array([1.0, 2.0])],
        "output1": [np.array([3.0])],
    }
    mock_submit_inference_job = mock.Mock()
    mock_submit_inference_job.return_value = mock_inference
    mock_submit_inference_job.side_effect = partial(_store_device, mock_inference)

    # Link
    mock_link = mock.Mock(hub.LinkJob)
    mock_link.name = "Link job"
    mock_link.job_id = "jlbcd1234"
    mock_link.get_target_model.return_value = model
    mock_submit_link_job = mock.Mock()
    mock_submit_link_job.return_value = mock_link

    # Quantize
    mock_quantize = mock.Mock(hub.QuantizeJob)
    mock_quantize.name = "Quantize job"
    mock_quantize.job_id = "jqbcd1234"
    mock_quantize.get_target_model.return_value = model
    mock_submit_quantize_job = mock.Mock()
    mock_submit_quantize_job.return_value = mock_quantize

    patch_hub_compile = mock.patch(
        "qai_hub.submit_compile_job", mock_submit_compile_job
    )
    patch_hub_inference = mock.patch(
        "qai_hub.submit_inference_job", mock_submit_inference_job
    )
    patch_hub_link = mock.patch("qai_hub.submit_link_job", mock_submit_link_job)
    patch_hub_quantize = mock.patch(
        "qai_hub.submit_quantize_job", mock_submit_quantize_job
    )

    with (
        patch_hub_compile,
        patch_hub_inference,
        patch_hub_link,
        patch_hub_quantize,
    ):
        # Yield mocks to allow assertions
        yield SimpleNamespace(
            submit_compile_job=mock_submit_compile_job,
            submit_inference_job=mock_submit_inference_job,
            submit_link_job=mock_submit_link_job,
            submit_quantize_job=mock_submit_quantize_job,
        )


def has_get_unsupported_reason(cls: type, stop_at_classes: list[type]) -> bool:
    """
    Check whether the 'get_unsupported_reason' attribute is defined in the given class
    or any of its parent classes up to (but not including) any class in stop_at_classes.

    Parameters
    ----------
    cls
        The class to check.
    stop_at_classes
        A list of classes at which to stop the search in the MRO.

    Returns
    -------
    has_method : bool
        True if 'get_unsupported_reason' is found in cls or one of its parent classes
        before reaching any of the stop_at_classes; False otherwise.
    """
    for base in cls.__mro__:
        if base in stop_at_classes:
            break
        if "get_unsupported_reason" in base.__dict__:
            return True
    return False


def skip_invalid_runtime_device(
    model_cls: AnyModelCls,
    runtime: TargetRuntime,
    device: ScorecardDevice,
) -> None:
    model: AnyModel
    if issubclass(model_cls, FromPretrainedProtocol):
        model = model_cls.from_pretrained()
    else:
        raise NotImplementedError()

    if reason := model.get_unsupported_reason(runtime, device.execution_device):
        pytest.xfail(reason)

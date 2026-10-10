# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import contextlib
import itertools
import math
import tempfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, TypeAlias, TypeVar, cast
from unittest import mock

import numpy as np
import qai_hub as hub
import torch
from filelock import FileLock
from mypy_boto3_s3.service_resource import Bucket

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.artifacts import (
    INTERMEDIATES_DIR,
    ScorecardArtifact,
)
from qai_hub_models.scorecard.device import cs_universal
from qai_hub_models.scorecard.envvars import (
    CompileSingleInstantiationEnvvar,
    IgnoreDeviceJobCacheEnvvar,
    S3ArtifactsDirEnvvar,
)
from qai_hub_models.scorecard.errors import CachedScorecardJobError
from qai_hub_models.scorecard.params import ScExportTestParams
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.scorecard_job import (
    JobTypeVar,
    ScorecardJobTypeVar,
)
from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    ComponentNamesYaml,
    GraphNamesYaml,
    InferenceScorecardJobYaml,
    LinkScorecardJobYaml,
    PreQDQCompileScorecardJobYaml,
    ProfileScorecardJobYaml,
    QuantizeScorecardJobYaml,
    ScorecardAssetYaml,
    ScorecardJobYaml,
    ToolVersionsByPathYaml,
)
from qai_hub_models.scorecard.utils.testing import (
    get_and_sync_datasets_cache_dir,
    get_hub_val_dataset,
    mock_get_calibration_data,
    mock_on_device_model_call,
    mock_tabulate_fn,
)
from qai_hub_models.scorecard.utils.testing_async_utils import (
    CompileJobsAreIdenticalCache,
    append_line_to_file,
    cache_dataset,
    callable_side_effect,
    get_cached_dataset_entries,
    write_accuracy,
)
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    get_qaihm_s3,
    s3_multipart_upload,
)
from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    WorkbenchModelCollection,
)
from qai_hub_models.utils.base_model import WorkbenchModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
    MultiGraphWorkbenchModelCollection,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.evaluate.helpers import (
    DEFAULT_NUM_EVAL_SAMPLES,
    _load_quant_cpu_onnx,
    evaluate_on_dataset,
    get_torch_val_dataloader,
)
from qai_hub_models.utils.export.context import resolve_model_app_cls
from qai_hub_models.utils.export.result import (
    CollectionExportResult,
    ComponentGroup,
    ExportResult,
    LegacyCollectionExportResult,
    MultiGraphCollectionExportResult,
    MultiGraphComponentGroup,
    MultiGraphExportResult,
    MultiGraphGroup,
)
from qai_hub_models.utils.hub_clients import (
    deployment_is_prod,
    get_default_hub_deployment,
)
from qai_hub_models.utils.inference import AsyncOnDeviceModel
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT
from qai_hub_models.utils.qai_hub_helpers import assert_success_and_get_target_models

__all__ = [
    "ExportFunc",
    "_pad_and_concatenate",
    "compile_via_export",
    "export_test_e2e",
    "fetch_cached_jobs_if_compile_jobs_are_identical",
    "fetch_compile_or_link_jobs",
    "inference_via_export",
    "link_via_export",
    "on_device_inference_for_accuracy_validation",
    "patch_hub_with_cached_jobs",
    "pre_quantize_compile_via_export",
    "profile_via_export",
    "quantize_via_export",
    "run_llm_compile",
    "torch_inference_for_accuracy_validation",
    "torch_inference_for_accuracy_validation_outputs",
]

ExportFunc = Callable[
    ...,
    ExportResult
    | CollectionExportResult
    | MultiGraphExportResult
    | MultiGraphCollectionExportResult
    | LegacyCollectionExportResult,
]
JobFunc = Callable[..., hub.Job | dict[str, hub.Job]]


QAIHMModelT: TypeAlias = (
    WorkbenchModel
    | CollectionModel
    | MultiGraphWorkbenchModel
    | MultiGraphCollectionModel
)


def _maybe_restrict_to_single_instantiation(model: Any) -> None:
    """Restrict an LLM model to a single instantiation (for quicker testing)"""
    if not CompileSingleInstantiationEnvvar.get():
        return
    restrict = getattr(model, "restrict_to_single_instantiation", None)
    if callable(restrict):
        restrict()


def _get_components_and_graph_names(
    model: QAIHMModelT | type[QAIHMModelT],
    model_id: str,
) -> tuple[list[str] | None, list[str] | None, ComponentGroup[list[str]] | None]:
    components: list[str] | None = None
    graph_names: list[str] | None = None
    component_graph_names: ComponentGroup[list[str]] | None = None

    if isinstance(model, type):
        # A model does not support getting the input spec without being instantiated.
        # Try to collect the cached component/graph names from disk instead.
        if issubclass(model, MultiGraphCollectionModel):
            components = ComponentNamesYaml.from_test_artifacts().get(model_id)
            if components:
                gn_cache = GraphNamesYaml.from_test_artifacts()
                cgns = {cn: gn_cache.get(model_id, cn) for cn in components}
                component_graph_names = ComponentGroup(
                    {k: v for k, v in cgns.items() if v is not None}
                )
        elif issubclass(model, CollectionModel):
            components = ComponentNamesYaml.from_test_artifacts().get(model_id)
        elif issubclass(model, MultiGraphWorkbenchModel):
            graph_names = GraphNamesYaml.from_test_artifacts().get(model_id)

    elif isinstance(model, MultiGraphCollectionModel):
        components = model.component_names
        cgn = ComponentGroup[list[str]]({})
        if isinstance(model, MultiGraphWorkbenchModelCollection):
            for component_name, component in model.components.items():
                if isinstance(component, MultiGraphWorkbenchModel):
                    cgn[component_name] = component.graph_names
                else:
                    cgn[component_name] = [component.context_graph_name]
        if cgn:
            component_graph_names = cgn
    elif isinstance(model, CollectionModel):
        components = model.component_names
        cgn = ComponentGroup[list[str]]({})
        if isinstance(model, WorkbenchModelCollection):
            for component_name, component in model.components.items():
                if isinstance(component, MultiGraphWorkbenchModel):
                    cgn[component_name] = component.graph_names
        if cgn:
            component_graph_names = cgn
    elif isinstance(model, MultiGraphWorkbenchModel):
        graph_names = model.graph_names

    _stash_component_graph_names(
        model_id, components, graph_names, component_graph_names
    )

    return components, graph_names, component_graph_names


def _stash_component_graph_names(
    model_id: str,
    components: list[str] | None,
    graph_names: list[str] | None,
    component_graph_names: ComponentGroup[list[str]] | None,
) -> None:
    """Write component and graph names to scorecard artifact YAMLs."""
    if components is not None:
        comp_cache = ComponentNamesYaml.from_test_artifacts()
        comp_cache.set(model_id, components)
        comp_cache.to_file(ScorecardArtifact.COMPONENT_NAMES.touch())

    if component_graph_names is not None:
        gn_cache = GraphNamesYaml.from_test_artifacts()
        for comp_name, gn_list in component_graph_names.items():
            gn_cache.set(model_id, comp_name, gn_list)
        gn_cache.to_file(ScorecardArtifact.GRAPH_NAMES.touch())
    elif graph_names is not None:
        gn_cache = GraphNamesYaml.from_test_artifacts()
        gn_cache.set(model_id, model_id, graph_names)
        gn_cache.to_file(ScorecardArtifact.GRAPH_NAMES.touch())


def _invalid_job_submission(*args: Any, **kwargs: Any) -> None:
    raise ValueError(
        "Attempted to submit a job when a cached job should have been present."
    )


def _get_sim_cpu_key(model_id: str, precision: Precision) -> str:
    return f"{model_id}_{precision}_sim"


def _get_torch_cpu_key(model_id: str) -> str:
    return f"{model_id}_torch"


def patch_hub_with_cached_jobs(
    params: ScExportTestParams,
    patch_quantization: bool = False,
    patch_compile: bool = False,
    patch_link: bool = False,
    patch_profile: bool = False,
    patch_inference: bool = False,
) -> tuple[
    mock._patch,
    mock._patch | nullcontext,
    mock._patch | nullcontext,
    mock._patch | nullcontext,
    mock._patch | nullcontext,
    mock._patch | nullcontext,
    mock._patch | nullcontext,
]:
    """
    Many tests use the export scripts to submit jobs.
    However, there is no path to break the export script into pieces; eg.
        * compile in one test
        * profile in another test
        * etc.
    We could modify the export script parameters to be more expressive, but this
    would come at the cost of readability.

    Instead, we "mock" various hub APIs to return "cached" jobs from previous tests.
    This allows us to test various parts of the export script "asyncronously" (without each test neeing to wait for Hub).

    This function:
        1. Gathers previous cached jobs
        2. Mocks several hub APIs (eg. submit_profile_job) to return those jobs instead of creating new ones.

    NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    params
        Export test params.
    patch_quantization
        Whether to patch previously cached quantization jobs. Default is False.
    patch_compile
        Whether to patch previously cached compile jobs. Default is False.
    patch_link
        Whether to patch previously cached link jobs. Default is False.
    patch_profile
        Whether to patch previously cached profile jobs. Default is False.
    patch_inference
        Whether to patch previously cached inference jobs. Default is False.

    Returns
    -------
    device_patch : mock._patch
        Patch for device selection.
    calibration_data_patch : mock._patch | nullcontext
        Patch for calibration data retrieval.
    quantize_job_patch : mock._patch | nullcontext
        Patch for quantization jobs.
    compile_job_patch : mock._patch | nullcontext
        Patch for compilation jobs.
    link_job_patch : mock._patch | nullcontext
        Patch for link jobs.
    profile_job_patch : mock._patch | nullcontext
        Patch for profiling jobs.
    inference_job_patch : mock._patch | nullcontext
        Patch for inference jobs.

    Notes
    -----
    For each "type" of job, returns a patch.
    If the associated "patch_job_type" param is False, the corresponding patch will do nothing.
    If cached jobs of a specific type aren't found, the corresponding patch will do nothing.

    Raises
    ------
    ValueError
        If jobs are still running or if any job failed.
    """
    device_patch = mock.patch(
        "qai_hub.get_devices",
        return_value=[params.device.reference_device] if params.device else [],
    )

    calibration_datas_to_patch: list[hub.Dataset] = []
    quantize_jobs_to_patch: list[hub.QuantizeJob] = []
    compile_jobs_to_patch: list[hub.CompileJob] = []
    link_jobs_to_patch: list[hub.LinkJob] = []
    profile_jobs_to_patch: list[hub.ProfileJob] = []
    inference_jobs_to_patch: list[hub.InferenceJob] = []

    def _get_jobs(
        YamlT: type[ScorecardJobYaml[ScorecardJobTypeVar]],
    ) -> Iterable[ScorecardJobTypeVar] | None:
        yaml = YamlT.from_test_artifacts()
        jobs_can_be_missing = (
            params.precision == Precision.mixed_with_float
            and YamlT == QuantizeScorecardJobYaml
        )
        jobs = yaml.get_all_jobs(
            params,
            raise_if_not_successful=True,
            raise_if_jobs_are_missing=not jobs_can_be_missing,
        ).values()
        if jobs_can_be_missing:
            # All jobs must be defined unless we're targeting mixed_with_float
            return [x for x in jobs if x is not None]
        return cast(Iterable[ScorecardJobTypeVar], jobs)

    with ThreadPoolExecutor() as pool:
        quantize_future = (
            pool.submit(
                _get_jobs,
                QuantizeScorecardJobYaml,
            )
            if patch_quantization
            else None
        )

        compile_future = (
            pool.submit(
                _get_jobs,
                CompileScorecardJobYaml,
            )
            if patch_compile
            else None
        )

        link_future = (
            pool.submit(
                _get_jobs,
                LinkScorecardJobYaml,
            )
            if patch_link
            else None
        )

        profile_future = (
            pool.submit(
                _get_jobs,
                ProfileScorecardJobYaml,
            )
            if patch_profile
            else None
        )

        inference_future = (
            pool.submit(
                _get_jobs,
                InferenceScorecardJobYaml,
            )
            if patch_inference
            else None
        )

    # Collect pre-quantization (to ONNX) compile jobs & quantize jobs
    if quantize_future is not None:
        if quantize_jobs := quantize_future.result():
            pre_quantize_compile_jobs = [
                cast(hub.CompileJob, component_job.job.model.get_producer())
                for component_job in quantize_jobs
            ]

            # Don't create a compile patch here yet since we may need to also patch the main compile jobs later.
            compile_jobs_to_patch.extend(pre_quantize_compile_jobs)
            quantize_jobs_to_patch.extend([x.job for x in quantize_jobs])
            calibration_datas_to_patch.extend(
                [x.job.calibration_dataset for x in quantize_jobs]
            )
        elif params.precision != Precision.float:
            raise CachedScorecardJobError("Could not find cached quantize jobs.")

    if compile_future is not None:
        if compile_jobs := compile_future.result():
            compile_jobs_to_patch.extend([x.job for x in compile_jobs])
        else:
            raise CachedScorecardJobError("Could not find cached compile jobs.")

    if link_future is not None:
        if link_jobs := link_future.result():
            link_jobs_to_patch.extend([x.job for x in link_jobs])
        else:
            raise CachedScorecardJobError("Could not find cached link jobs.")

    if profile_future is not None:
        if profile_jobs := profile_future.result():
            profile_jobs_to_patch.extend([x.job for x in profile_jobs])
        else:
            raise CachedScorecardJobError("Could not find cached profile jobs.")

    if inference_future is not None:
        if inference_jobs := inference_future.result():
            inference_jobs_to_patch.extend([x.job for x in inference_jobs])
        else:
            raise CachedScorecardJobError("Could not find cached inference jobs.")

    calib_side_effect = itertools.chain(
        calibration_datas_to_patch, itertools.repeat(mock_get_calibration_data)
    )
    calibration_data_patch = mock.patch(
        "qai_hub_models.utils.quantization.get_calibration_data",
        side_effect=callable_side_effect(calib_side_effect),
    )

    quantize_side_effect = itertools.chain(
        quantize_jobs_to_patch, itertools.repeat(_invalid_job_submission)
    )
    quantize_job_patch = (
        mock.patch(
            "qai_hub.submit_quantize_job",
            side_effect=callable_side_effect(quantize_side_effect),
        )
        if patch_quantization or quantize_jobs_to_patch
        else nullcontext()
    )

    # When patching quantize but not compile, the first set of compile jobs
    # need to be patched and the following calls to `submit_compile_job` should
    # actually submit jobs. Any subsequent calls should throw an error.
    if not patch_compile and compile_jobs_to_patch:
        compile_side_effect = itertools.chain(
            compile_jobs_to_patch,
            itertools.repeat(
                hub.submit_compile_job, len(params.component_gn_pairs or [None])
            ),
            itertools.repeat(_invalid_job_submission),
        )
    else:
        compile_side_effect = itertools.chain(
            compile_jobs_to_patch, itertools.repeat(_invalid_job_submission)
        )
    compile_job_patch = (
        mock.patch(
            "qai_hub.submit_compile_job",
            side_effect=callable_side_effect(compile_side_effect),
        )
        if patch_compile or compile_jobs_to_patch
        else nullcontext()
    )

    link_side_effect = itertools.chain(
        link_jobs_to_patch, itertools.repeat(_invalid_job_submission)
    )
    link_job_patch = (
        mock.patch(
            "qai_hub.submit_link_job",
            side_effect=callable_side_effect(link_side_effect),
        )
        if patch_link or link_jobs_to_patch
        else nullcontext()
    )

    profile_side_effect = itertools.chain(
        profile_jobs_to_patch, itertools.repeat(_invalid_job_submission)
    )
    profile_job_patch = (
        mock.patch(
            "qai_hub.submit_profile_job",
            side_effect=callable_side_effect(profile_side_effect),
        )
        if patch_profile or profile_jobs_to_patch
        else nullcontext()
    )

    inference_side_effect = itertools.chain(
        inference_jobs_to_patch, itertools.repeat(_invalid_job_submission)
    )
    inference_job_patch = (
        mock.patch(
            "qai_hub.submit_inference_job",
            side_effect=callable_side_effect(inference_side_effect),
        )
        if patch_inference or inference_jobs_to_patch
        else nullcontext()
    )

    return (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        link_job_patch,
        profile_job_patch,
        inference_job_patch,
    )


def pre_quantize_compile_via_export(
    compile_model: Callable[
        ...,
        hub.CompileJob
        | ComponentGroup[hub.CompileJob]
        | MultiGraphComponentGroup[hub.CompileJob],
    ],
    model_id: str,
    model: QAIHMModelT,
    upload_model: Callable[..., Any] | None = None,
) -> None:
    """
    Use the provided export script function to submit ONNX compile jobs (before quantization).

    If async testing is enabled:
        Submitted jobs are added to the async testing cache,
        and this method returns immediately.

    Otherwise:
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    compile_model
        Export script function to submit compile jobs.
    model_id
        Model ID.
    model
        QAIHM instance of the model.
    upload_model
        Export script function to serialize and upload models.
    """
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model, model_id)
    )
    assert component_graph_names is None, (
        "Auto-quantization of multi-graph models is not supported"
    )
    test_params = ScExportTestParams(
        model_id,
        path=None,
        precision=None,
        component_names=component_names,
        graph_names=graph_names,
        component_graph_names=component_graph_names,
    )

    assert upload_model is not None
    source_models = upload_model(model)

    # Run ONNX compile jobs
    compile_output = compile_model(
        model,
        model_id,
        cs_universal.reference_device,
        TargetRuntime.ONNX,
        Precision.float,
        source_models,
    )

    # Verify success or cache job IDs to a file.
    cache = PreQDQCompileScorecardJobYaml.from_test_artifacts()
    cache.update_from_export_output(compile_output, test_params)
    cache.to_file()


def quantize_via_export(
    quantize_model: Callable[
        ..., hub.QuantizeJob | ComponentGroup[hub.QuantizeJob] | None
    ],
    model_id: str,
    model: QAIHMModelT,
    precision: Precision,
) -> None:
    """
    Use the provided export script function to submit quantize jobs.

    If async testing is enabled:
        Submitted jobs are added to the async testing cache,
        and this method returns immediately.

    Otherwise:
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    quantize_model
        Export script function to submit quantize jobs.
    model_id
        Model ID.
    model
        QAIHM instance of the model.
    precision
        Model precision.
    """
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model, model_id)
    )
    assert component_graph_names is None, (
        "Auto-quantization of multi-graph models is not supported"
    )
    test_params = ScExportTestParams(
        model_id,
        path=None,
        precision=precision,
        component_names=component_names,
        graph_names=graph_names,
        component_graph_names=component_graph_names,
    )

    # Fetch ONNX compile jobs + target models from pre_quantize_compile_via_export
    onnx_compile_jobs = (
        PreQDQCompileScorecardJobYaml.from_test_artifacts().get_export_output(
            test_params
        )
    )
    onnx_model_inputs = (
        assert_success_and_get_target_models(onnx_compile_jobs)
        if onnx_compile_jobs
        else None
    )

    # Run quantize jobs
    quantize_kwargs: dict[str, Any] = {}
    if isinstance(model, CollectionModel):
        quantize_kwargs["app"] = resolve_model_app_cls(QAIHM_MODELS_ROOT / model_id)

    with mock.patch(
        "qai_hub_models.utils.quantization.get_calibration_data",
        mock_get_calibration_data,
    ):
        quantize_output = quantize_model(
            precision,
            model,
            model_id,
            onnx_model_inputs,
            None,
            **quantize_kwargs,
        )

    # Verify success or cache job IDs to a file.
    cache = QuantizeScorecardJobYaml.from_test_artifacts()
    cache.update_from_export_output(quantize_output, test_params)
    cache.to_file()


def compile_via_export(
    compile_model: Callable[
        ...,
        hub.CompileJob
        | ComponentGroup[hub.CompileJob]
        | MultiGraphComponentGroup[hub.CompileJob],
    ],
    model_id: str,
    model: QAIHMModelT,
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
    is_aimet: bool = False,
    upload_model: Callable[..., Any] | None = None,
) -> None:
    """
    Use the provided export script function to submit compile jobs.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the compile job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    compile_model
        Export script function to submit compile jobs.
    model_id
        Model ID.
    model:
        QAIHM instance of the model
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    is_aimet
        Whether the model uses local aimet encodings during compilation.
    upload_model
        Export script function to serialize and upload models.
    """
    _maybe_restrict_to_single_instantiation(model)
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model, model_id)
    )
    test_params = ScExportTestParams(
        model_id,
        scorecard_path,
        precision,
        device,
        component_names,
        graph_names,
        component_graph_names,
    )

    if is_aimet:
        assert upload_model is not None
        source_models = upload_model(model)
        compile_output = compile_model(
            model,
            model_id,
            device.execution_device,
            scorecard_path.runtime,
            precision,
            source_models,
            extra_options=scorecard_path.get_compile_options(),
        )
    else:
        quantize_jobs = (
            QuantizeScorecardJobYaml.from_test_artifacts().get_export_output(
                test_params,
                raise_if_jobs_are_missing=precision != Precision.mixed_with_float,
            )
        )
        if precision != Precision.float and quantize_jobs is None:
            raise CachedScorecardJobError(
                test_params.str_with_description("Could not find cached quantize jobs.")
            )
        quantize_job_input = (
            assert_success_and_get_target_models(quantize_jobs)
            if quantize_jobs
            else None
        )

        if quantize_job_input is not None:
            if isinstance(quantize_job_input, ComponentGroup) and len(
                quantize_job_input
            ) != len(component_names or []):
                assert upload_model is not None
                source_models = upload_model(model)
                source_models |= quantize_job_input
            else:
                source_models = quantize_job_input
        else:
            assert upload_model is not None
            source_models = upload_model(model)

        compile_output = compile_model(
            model,
            model_id,
            device.execution_device,
            scorecard_path.runtime,
            precision,
            source_models,
            extra_options=scorecard_path.get_compile_options(),
        )

    # Verify success or cache job IDs to a file.
    cache = CompileScorecardJobYaml.from_test_artifacts()
    cache.update_from_export_output(compile_output, test_params)
    cache.to_file()


def link_via_export(
    link_model: Callable[..., hub.LinkJob | ComponentGroup[hub.LinkJob]],
    model_id: str,
    model: QAIHMModelT,
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
) -> None:
    """
    Use the provided export script function to submit link jobs.

    If async testing is enabled:
        * Fetches previously cached compile jobs.

        * Submitted link jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    link_model
        Export script function to submit link jobs.
    model_id
        Model ID.
    model
        QAIHM instance of the model.
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    """
    assert scorecard_path.runtime.uses_hub_link

    _maybe_restrict_to_single_instantiation(model)
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model, model_id)
    )
    test_params = ScExportTestParams(
        model_id,
        path=scorecard_path,
        precision=precision,
        device=device,
        component_names=component_names,
        graph_names=graph_names,
        component_graph_names=component_graph_names,
    )

    # Fetch ONNX compile jobs + target models from pre_quantize_compile_via_export
    compile_jobs = CompileScorecardJobYaml.from_test_artifacts().get_export_output(
        test_params
    )
    compiled_models = (
        assert_success_and_get_target_models(compile_jobs) if compile_jobs else None
    )

    # Call link_model from export script
    link_output = link_model(
        compiled_models,
        device.execution_device,
        model_id,
        model,
        scorecard_path.runtime,
        extra_options=scorecard_path.get_link_options(),
    )

    cache = LinkScorecardJobYaml.from_test_artifacts()
    cache.update_from_export_output(link_output, test_params)
    cache.to_file()


def run_llm_compile(
    export_model: ExportFunc,
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
    skip_compile_options: bool = False,
    extra_model_arguments: dict[str, Any] | None = None,
    skip_downloading: bool = True,
) -> MultiGraphCollectionExportResult | LegacyCollectionExportResult:
    """
    Use the provided export script function to submit compile jobs for llm tests.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the compile job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    export_model
        Export script function.
    model_id
        Model ID.
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    component_names
        Name of all model components (if applicable), or None of there are no components.
        Default is None.
    skip_compile_options
        Whether to skip compile options. Default is False.
    extra_model_arguments
        Additional model arguments to pass to export. Default is None.
    skip_downloading
        Whether to skip downloading. Default is True.

    Returns
    -------
    result : MultiGraphCollectionExportResult | LegacyCollectionExportResult
        The export result returned by ``export_model``. Callers can inspect
        compile/link job IDs and the downloaded bundle path via this object.
    """
    test_params = ScExportTestParams(
        model_id,
        path=scorecard_path,
        precision=precision,
        component_names=component_names,
    )

    result = export_model(
        device=device.execution_device,
        precision=precision,
        skip_downloading=skip_downloading,
        skip_profiling=True,
        skip_inferencing=True,
        skip_summary=True,
        compile_options=(
            scorecard_path.get_compile_options() if not skip_compile_options else ""
        ),
        target_runtime=scorecard_path.runtime,
        **extra_model_arguments or {},
    )

    # Verify success or cache job IDs to a file.
    cache = CompileScorecardJobYaml.from_test_artifacts()
    if isinstance(result, MultiGraphCollectionExportResult):
        cache.update_from_export_output(result.compile_jobs, test_params)
    else:
        result = cast(LegacyCollectionExportResult, result)
        cache.update_from_export_output(
            ComponentGroup(
                {
                    name: er.compile_job
                    for name, er in result.components.items()
                    if er.compile_job is not None
                }
            ),
            test_params,
        )
    cache.to_file()
    return result


def fetch_cached_jobs_if_compile_jobs_are_identical(
    params: ScExportTestParams,
) -> (
    JobTypeVar
    | MultiGraphGroup[JobTypeVar]
    | ComponentGroup[JobTypeVar]
    | MultiGraphComponentGroup[JobTypeVar]
    | None
):
    """
    Checks if the compile jobs are the same, the QAIRT version matches, and the override flag is not set.
    If all conditions are met, returns the cached profile job and saves the job to the YAML cache.
    Otherwise, returns None.

    Parameters
    ----------
    params
        Export test parameters.

    Returns
    -------
    cached_result : JobTypeVar | MultiGraphGroup[JobTypeVar] | ComponentGroup[JobTypeVar] | MultiGraphComponentGroup[JobTypeVar] | None
        The cached Jobs, or None if no cached job is found.
    """
    assert isinstance(params.path, ScorecardProfilePath)

    # Check if the QAIRT version matches the API version and if the override flag is set.
    # Previous scorecard QAIRT version is stored at /scorecard/intermediates/environment.env dump.
    is_override = IgnoreDeviceJobCacheEnvvar.get()
    if (
        #
        # don't run if user disabled caching
        is_override
        #
        # only prod jobs are cached
        or not deployment_is_prod(get_default_hub_deployment() or "")
        #
        # if the tool versions do not match, profiling for all paths must be re-run
        or params.path.tool_versions
        != ToolVersionsByPathYaml.from_dir(INTERMEDIATES_DIR).tool_versions.get(
            params.path, ToolVersions()
        )
    ):
        return None

    yaml = ProfileScorecardJobYaml.from_intermediates()

    compile_jobs_identical_cache_file = (
        ScorecardArtifact.COMPILE_JOBS_IDENTICAL_CACHE.touch()
    )
    compile_jobs_identical_cache = CompileJobsAreIdenticalCache.from_yaml(
        compile_jobs_identical_cache_file, create_empty_if_no_file=True
    )
    compile_jobs_are_identical = compile_jobs_identical_cache.is_identical(params)
    compile_jobs_identical_cache.to_yaml(compile_jobs_identical_cache_file)
    if not compile_jobs_are_identical:
        return None

    try:
        return yaml.get_export_output(params)
    except CachedScorecardJobError:
        # No cached profile jobs for this model, or the cached jobs failed.
        return None


CompileOrLinkT = TypeVar("CompileOrLinkT", hub.CompileJob, hub.LinkJob)


def fetch_compile_or_link_jobs(
    test_params: ScExportTestParams,
) -> (
    CompileOrLinkT
    | MultiGraphGroup[CompileOrLinkT]
    | ComponentGroup[CompileOrLinkT]
    | MultiGraphComponentGroup[CompileOrLinkT]
    | None
):
    """Fetch cached compile or link jobs depending on runtime type."""
    assert test_params.path is not None
    if test_params.path.runtime.uses_hub_link:
        return LinkScorecardJobYaml.from_test_artifacts().get_export_output(test_params)
    return CompileScorecardJobYaml.from_test_artifacts().get_export_output(test_params)


def profile_via_export(
    profile_model: Callable[
        ...,
        hub.ProfileJob
        | ComponentGroup[hub.ProfileJob]
        | MultiGraphComponentGroup[hub.ProfileJob],
    ],
    model_id: str,
    model: QAIHMModelT,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    components: list[str] | None = None,
) -> None:
    """
    Use the provided export script function to submit profile jobs.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the profile job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    profile_model
        Export script function to submit profile jobs.
    model_id
        Model ID.
    model:
        QAIHM instance of the model
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    components
        Profile only these components. Used by hybrid LLMs, whose backbone parts are
        measured end-to-end rather than individually. Defaults to every component.
    """
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model, model_id)
    )

    if components is not None:
        # Narrow after the call above, which stashes the full component list that
        # compile and link still need.
        assert component_names is not None, (
            f"{model_id} has no components to restrict to {components}"
        )
        unknown = sorted(set(components) - set(component_names))
        assert not unknown, f"{model_id}: cannot profile unknown component(s) {unknown}"
        component_names = [c for c in component_names if c in components]
        if component_graph_names is not None:
            component_graph_names = ComponentGroup(
                {
                    name: gn
                    for name, gn in component_graph_names.items()
                    if name in components
                }
            )

    test_params = ScExportTestParams(
        model_id,
        path=scorecard_path,
        precision=precision,
        device=device,
        component_names=component_names,
        graph_names=graph_names,
        component_graph_names=component_graph_names,
    )

    if profile_output := fetch_cached_jobs_if_compile_jobs_are_identical(test_params):
        print(
            test_params.str_with_description(
                "The compiled assets from the previous scorecard are identical. Copying over profile job(s).",
            )
        )
    else:
        compile_jobs = fetch_compile_or_link_jobs(test_params)
        target_models = (
            assert_success_and_get_target_models(compile_jobs) if compile_jobs else None
        )
        inference_options = model.get_hub_inference_options(
            scorecard_path.runtime,
            scorecard_path.get_inference_options(precision, device),
        )
        # Only collection profile runners accept `components`; single-model ones
        # have nothing to restrict, and `components` is never set for them.
        component_kwarg = {} if components is None else {"components": components}
        profile_output = profile_model(  # type: ignore[assignment]
            model_id,
            device.execution_device,
            inference_options,
            target_models,
            **component_kwarg,
        )

    cache = ProfileScorecardJobYaml.from_test_artifacts()
    cache.update_from_export_output(profile_output, test_params)
    cache.to_file()


def inference_via_export(
    inference_model: Callable[
        ...,
        hub.InferenceJob
        | ComponentGroup[hub.InferenceJob]
        | MultiGraphComponentGroup[hub.InferenceJob],
    ],
    model_id: str,
    model: QAIHMModelT,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
) -> None:
    """
    Use the provided export script function to submit inference jobs.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the inference job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    inference_model
        Export script function to submit inference jobs.
    model_id
        Model ID.
    model:
        QAIHM instance of the model
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    """
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model, model_id)
    )
    test_params = ScExportTestParams(
        model_id,
        path=scorecard_path,
        precision=precision,
        device=device,
        component_names=component_names,
        graph_names=graph_names,
        component_graph_names=component_graph_names,
    )

    # TODO(#15111): Reenable caching for Inference Jobs. Inference jobs also must check that input datasets are the same, not just the compiled assets.
    compile_jobs = fetch_compile_or_link_jobs(test_params)
    target_models = (
        assert_success_and_get_target_models(compile_jobs) if compile_jobs else None
    )

    runtime = scorecard_path.runtime
    inference_inputs = model.sample_inputs(
        use_channel_last_format=runtime.channel_last_native_execution
    )
    inference_options = model.get_hub_inference_options(
        scorecard_path.runtime, scorecard_path.get_inference_options(precision, device)
    )
    inference_output = inference_model(
        inference_inputs,
        model_id,
        device.execution_device,
        inference_options,
        target_models,
    )

    cache = InferenceScorecardJobYaml.from_test_artifacts()
    cache.update_from_export_output(inference_output, test_params)
    cache.to_file()


def _all_cached_profile_jobs_succeeded(test_params: ScExportTestParams) -> bool:
    """
    True iff every profile job that should exist for these test params is
    cached and succeeded. Used by export_test_e2e to decide whether to patch
    in cached profile jobs vs. skip profiling and still upload the release
    asset.

    Returns False (do not patch) when the profile YAML is missing entirely,
    when any profile job entry is absent for this (model, precision, device,
    runtime), or when any cached job is still running / failed.
    """
    if not ScorecardArtifact.PROFILE_YAML.exists():
        return False
    yaml = ProfileScorecardJobYaml.from_test_artifacts()
    try:
        jobs = yaml.get_all_jobs(
            test_params,
            wait_for_job=False,
            raise_if_not_successful=False,
            raise_if_jobs_are_missing=False,
        )
    except Exception:
        return False
    return bool(jobs) and all(j is not None and j.success for j in jobs.values())


def export_test_e2e(
    export_model: ExportFunc,
    model_cls: type[QAIHMModelT],
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> None:
    """
    Verifies the export script function provided works end to end.

    If async testing is enabled:
        * If found, existing Hub jobs are are used, rather than submitting new ones.

    Otherwise:
        * Submits all (quantize, compile, profile) jobs on Hub
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    export_model
        Export script function.
    model_cls
        The model class used during export.
    model_id
        Model ID.
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    component_names
        Name of all model components (if applicable), or None of there are no components.
        Default is None.
    """
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model_cls, model_id)
    )
    test_params = ScExportTestParams(
        model_id,
        scorecard_path,
        precision,
        device,
        component_names,
        graph_names,
        component_graph_names,
    )

    # Profile is a best-effort prerequisite: if every profile job for this
    # specific (model, precision, device, runtime) exists and succeeded,
    # patch them in so the export script's profile call returns the cached
    # job. Otherwise skip profiling entirely so we still upload the release
    # asset (compile artifacts) for this path.
    has_cached_profile_jobs = _all_cached_profile_jobs_succeeded(test_params)

    # Patch previous jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        link_job_patch,
        profile_job_patch,
        _,
    ) = patch_hub_with_cached_jobs(
        test_params,
        patch_quantization=not QAIHMModelManifest.from_model(model_id).is_aimet,
        patch_compile=True,
        patch_link=scorecard_path.runtime.uses_hub_link,
        patch_profile=has_cached_profile_jobs,
        patch_inference=False,
    )

    # Export will always trace the model.
    # However, that trace goes unused during this test because we are using pre-created compile jobs.
    # Patch over tracing / export to speed up the test.
    mocks: list[mock._patch] = []
    mocks.append(
        mock.patch.object(
            torch.jit,
            "trace",
            mock.MagicMock(return_value=None),
        )
    )
    mocks.append(
        mock.patch(
            "qai_hub.upload_model",
            mock.MagicMock(return_value=mock.MagicMock(spec=hub.Model)),
        )
    )
    if issubclass(model_cls, CollectionModel):
        mocks.append(
            mock.patch.object(
                model_cls,
                "serialize_component",
                mock.MagicMock(return_value=None),
            )
        )
    elif issubclass(model_cls, MultiGraphCollectionModel):
        mocks.append(
            mock.patch.object(
                model_cls,
                "serialize_component_graph",
                mock.MagicMock(return_value=None),
            )
        )
    elif issubclass(model_cls, WorkbenchModel):
        mocks.append(
            mock.patch.object(
                model_cls,
                "serialize",
                mock.MagicMock(return_value=None),
            )
        )

    # Skip calibration data loading step; the result is also unused during this test.
    mocks.append(
        mock.patch(
            "qai_hub_models.utils.quantization.get_calibration_data",
            mock.MagicMock(return_value=None),
        )
    )

    s3_bucket: Bucket | None = None
    with contextlib.suppress(ValueError):
        s3_bucket = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)[0]
    upload_to_s3 = (
        s3_bucket is not None
        and scorecard_path.is_published
        and not S3ArtifactsDirEnvvar.is_default()
    )

    # Skip model download when we aren't uploading to S3 to speed up the test.
    # Downloading the model for each test is slow and is not needed for basic export tests.
    if not upload_to_s3:
        mocks.append(
            mock.patch.object(
                hub.Model,
                "download",
                side_effect=[
                    Path(f"{component}.onnx")
                    for component in (component_names or [model_id])
                ],
            )
        )
        mocks.append(
            mock.patch(
                "qai_hub_models.utils.export.download.download_and_unzip_workbench_onnx_model",
                mock.MagicMock(
                    side_effect=[
                        ONNXBundle(Path(), f"{component}.onnx")
                        for component in (component_names or [model_id])
                    ]
                ),
            )
        )
        mocks.append(
            mock.patch(
                "qai_hub_models.utils.export.download.get_onnx_model_bundle_metadata",
                mock.MagicMock(
                    side_effect=[
                        (component, {}) for component in (component_names or [model_id])
                    ]
                ),
            )
        )
    # Test export script end to end.
    #
    # Capture any exception raised during the export call so we can still
    # upload the asset to S3 and record it in release-assets.yaml when the
    # export returned a download_path before raising (e.g. a post-download
    # check failed). We re-raise after the upload so the test runner sees
    # the failure. If the export raised without producing a download_path
    # we fail immediately.
    export_error: Exception | None = None
    result = None
    with (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        link_job_patch,
        profile_job_patch,
        tempfile.TemporaryDirectory() as tmpdir,
        mock.patch("qai_hub.upload_model", return_value=None),
    ):
        try:
            with contextlib.ExitStack() as stack:
                for m in mocks:
                    stack.enter_context(m)
                result = export_model(
                    device=device.execution_device,
                    precision=precision,
                    target_runtime=scorecard_path.runtime,
                    compile_options=scorecard_path.compile_path.get_compile_options(),
                    inference_options=scorecard_path.get_inference_options(
                        precision, device
                    ),
                    skip_profiling=not has_cached_profile_jobs,
                    skip_inferencing=True,
                    output_dir=tmpdir,
                    zip_assets=True,
                )
        except Exception as e:
            export_error = e

        if result is None or result.download_path is None:
            if export_error is not None:
                raise export_error
            raise RuntimeError(
                f"Export for {model_id} returned no download_path "
                f"(precision={precision}, runtime={scorecard_path.runtime}, "
                f"device={device})."
            )
        download_path: Path = result.download_path
        assert result.tool_versions is not None
        tool_versions: ToolVersions = result.tool_versions

        if upload_to_s3:
            assert s3_bucket is not None  # mypy
            assets_path = ScorecardArtifact.RELEASE_ASSETS.path
            chipset_key = (
                device if scorecard_path.runtime.is_aot_compiled else cs_universal
            )
            # FileLock guards read-modify-write against concurrent xdist workers
            # exporting different (precision, runtime) combos into the same yaml.
            with FileLock(f"{assets_path}.lock"):
                assets_cache = ScorecardAssetYaml.from_yaml(
                    assets_path, create_empty_if_no_file=True
                )
                already_cached = (
                    assets_cache.get_asset(
                        model_id, precision, chipset_key, scorecard_path
                    )
                    is not None
                )
            if not already_cached:
                s3_key = str(S3ArtifactsDirEnvvar.get() / download_path.name)
                s3_multipart_upload(
                    bucket=s3_bucket,
                    key=s3_key,
                    local_file_path=download_path,
                )
                with FileLock(f"{assets_path}.lock"):
                    assets_cache = ScorecardAssetYaml.from_yaml(
                        assets_path, create_empty_if_no_file=True
                    )
                    assets_cache.add_asset(
                        QAIHMModelReleaseAssets.AssetDetails(
                            s3_key=s3_key, tool_versions=tool_versions
                        ),
                        model_id,
                        precision,
                        chipset_key,
                        scorecard_path,
                    )
                    assets_cache.to_yaml(assets_path)

        if export_error is not None:
            raise export_error


def on_device_inference_for_accuracy_validation(
    model: type[QAIHMModelT],
    dataset_cls: type[BaseDataset],
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
) -> None:
    """
    Runs an inference job on the given dataset.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    model
        Model class to run inference on.
    dataset_cls
        Dataset class to use for evaluation.
    model_id
        Model ID.
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    """
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model, model_id)
    )
    assert graph_names is None and component_graph_names is None, (
        "Graph names are not supported for on-device inference"
    )
    test_params = ScExportTestParams(
        model_id,
        path=scorecard_path,
        precision=precision,
        device=device,
        component_names=component_names,
        graph_names=graph_names,
        component_graph_names=component_graph_names,
    )

    compile_jobs = fetch_compile_or_link_jobs(test_params)
    raw_target_models = (
        assert_success_and_get_target_models(compile_jobs) if compile_jobs else None
    )
    target_models_dict: dict[str | None, hub.Model]
    if component_names:
        target_models_dict = cast(dict[str | None, hub.Model], raw_target_models)
    else:
        target_model = cast(hub.Model, raw_target_models)
        target_models_dict = {None: target_model}

    job: hub.InferenceJob | None = None
    jobs_dict: dict[str, hub.InferenceJob] = {}
    for component_name, target_model in target_models_dict.items():
        hub_val_dataset = get_hub_val_dataset(
            dataset_cls,
            ScorecardArtifact.DATASET_IDS.touch(),
            model,
            apply_channel_transpose=scorecard_path.runtime.channel_last_native_execution,
            num_samples=get_num_eval_samples(dataset_cls),
        )
        ijob = hub.submit_inference_job(
            device=device.execution_device,
            inputs=hub_val_dataset,
            model=target_model,
            name=model_id,
        )

        if not component_name:
            job = ijob
        else:
            jobs_dict[component_name] = ijob

    cache = InferenceScorecardJobYaml.from_test_artifacts()
    cache.update_from_export_output(
        job if job else ComponentGroup(jobs_dict), test_params
    )
    cache.to_file()


def torch_inference_for_accuracy_validation(
    model: QAIHMModelT, dataset_cls: type[BaseDataset], model_id: str
) -> None:
    """
    Runs torch inference job on the given dataset.
    Uploads the results to hub and caches them.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    model
        Model instance to run inference on.
    dataset_cls
        Dataset class to use for evaluation.
    model_id
        Model ID.

    """
    assert isinstance(model, WorkbenchModel), (
        "This function is not yet supported for CollectionModel."
    )
    # Get the first dim of the first input. This is always the batch size.
    compiled_batch_size = next(iter(model.get_input_spec().values()))[0][0]

    inputs, *_ = next(
        iter(
            get_torch_val_dataloader(
                dataset_cls,
                get_num_eval_samples(dataset_cls),
                model.get_input_spec(),
            )
        )
    )
    if not isinstance(inputs, list) and not isinstance(inputs, tuple):
        # Generalize: treat "single-input" model as a list of 1 input.
        # This allows us to support single and multi-input models in 1 loop.
        inputs = [inputs]

    num_batches = inputs[0].shape[0]
    output_names = list(model.get_output_spec())
    outputs: list[list[np.ndarray]] = [[] for _ in output_names]
    for b in range(math.ceil(num_batches / compiled_batch_size)):
        # Complete N batches at a time to substantially reduces memory pressure
        # (not all inputs / outputs need to be in memory at once)
        # Without this, scorecard jobs can easily run OOM.
        #
        # TODO(#15497): We should strive to disable single-batch inference. Multi-batch inference is much faster.
        model_inputs = [
            x[b * compiled_batch_size : min((b + 1) * compiled_batch_size, x.shape[0])]
            for x in inputs
        ]
        model_outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] = model(
            *model_inputs
        )
        if not isinstance(model_outputs, (list, tuple)):
            # Generalize: treat "single-output" model as a list of 1 output.
            # This allows us to support single and multi-output models in 1 loop.
            model_outputs = [model_outputs]
        for i, output in enumerate(model_outputs):
            outputs[i].append(output.numpy())
    hub_entries = dict(zip(output_names, outputs, strict=False))
    cache_dataset(
        model_id,
        "torch_val",
        hub.upload_dataset(hub_entries),
    )


def torch_eval_smoke_on_two_samples(
    model: QAIHMModelT, dataset_cls: type[BaseDataset], model_id: str
) -> None:
    """
    Smoke-test the dataset + evaluator plumbing on two samples via torch.

    No accuracy threshold — this only proves the (dataset -> torch forward ->
    evaluator) chain runs end-to-end so nightly catches evaluator or dataset
    regressions cheaply.
    """
    assert isinstance(model, WorkbenchModel), (
        "This function is not yet supported for CollectionModel."
    )
    result = evaluate_on_dataset(
        evaluator_func=model.get_evaluator,
        model_executors={"torch": model},
        input_spec=model.get_input_spec(),
        dataset_cls=dataset_cls,
        num_samples=2,
        samples_per_job=2,
        use_cache=False,
    )
    assert result.torch_accuracy is not None, (
        f"Torch smoke eval for {model_id} did not produce an accuracy score."
    )


def _pad_and_concatenate(tensor_list: list[np.ndarray]) -> np.ndarray:
    """Concatenate arrays along axis 0, padding other dims if shapes differ.

    Detection models may produce variable-length outputs (e.g., different
    number of detections per image after NMS). Zero-padding is safe because
    downstream evaluators filter by score threshold.
    """
    max_ndim = max(arr.ndim for arr in tensor_list)
    if any(arr.ndim != max_ndim for arr in tensor_list):
        raise ValueError(
            "All arrays must have the same number of dimensions, "
            f"got ndims {[arr.ndim for arr in tensor_list]}"
        )
    max_shape = [max(arr.shape[dim] for arr in tensor_list) for dim in range(max_ndim)]

    needs_padding = any(
        arr.shape[dim] != max_shape[dim]
        for arr in tensor_list
        for dim in range(1, max_ndim)
    )

    if not needs_padding:
        return np.concatenate(tensor_list, axis=0)

    total_rows = sum(arr.shape[0] for arr in tensor_list)
    out = np.zeros((total_rows, *max_shape[1:]), dtype=tensor_list[0].dtype)
    row = 0
    for arr in tensor_list:
        slices = (slice(row, row + arr.shape[0]), *(slice(0, s) for s in arr.shape[1:]))
        out[slices] = arr
        row += arr.shape[0]
    return out


def torch_inference_for_accuracy_validation_outputs(model_id: str) -> list[np.ndarray]:
    """
    Fetches torch inference results computed by torch_inference_for_accuracy_validation
    Async testing must be enabled to run this method.

    Parameters
    ----------
    model_id
        Model ID.

    Returns
    -------
    inference_outputs : list[np.ndarray]
        List of results, in order of output from the torch model.
        [ output_0_array, output_1_array, ... ]
    """
    dataset = get_cached_dataset_entries(model_id, "torch_val")
    if not dataset:
        raise ValueError(f"Missing inference output dataset for model {model_id}")

    # Hub DatasetEntries is a dict of format {'name' [ batch_val_1, batch_val_2, etc.]}
    #
    # This flattens the dict into a list of the same order,
    # and merges the list of batch outputs for each dictionary entry into a single tensor.
    return [
        _pad_and_concatenate(tensor_list) if len(tensor_list) > 1 else tensor_list[0]
        for tensor_list in dataset.values()
    ]


def split_and_group_accuracy_validation_output_batches(
    torch_inference_outputs: list[np.ndarray],
) -> list[torch.Tensor | tuple[torch.Tensor, ...]]:
    """
    Converts output generated by torch_inference_for_accuracy_validation_outputs to a different format.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    torch_inference_outputs
        Return value of torch_inference_for_accuracy_validation_outputs.

    Returns
    -------
    batched_outputs : list[torch.Tensor | tuple[torch.Tensor, ...]]
        If torch_inference_outputs is length 1:
            [output_0::batch_0, output_0::batch_1, ...]

        otherwise:
            [(output_0::batch_0, output_1::batch_0, ...),
             (output_0::batch_1, output_1::batch_1, ...),
             ...]

        Note that the batch dimension is preserved in all returned tensors (it is always 1).
    """
    num_outputs = len(torch_inference_outputs)
    if num_outputs == 1:
        output = torch.tensor(torch_inference_outputs[0])
        return list(output.split(1))

    num_batches = len(torch_inference_outputs[0])
    outputs_per_batch: list[torch.Tensor | tuple[torch.Tensor, ...]] = [
        tuple(
            torch.Tensor(output_n[batch_idx]).unsqueeze(0)
            for output_n in torch_inference_outputs
        )
        for batch_idx in range(num_batches)
    ]
    return outputs_per_batch


def accuracy_on_sample_inputs_via_export(
    export_model: ExportFunc,
    model_id: str,
    model: QAIHMModelT,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> None:
    """
    Computes accuracy for the given model's sample inputs and saves it to disk.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    export_model
        Code-generated export function from export.py.
    model_id
        Model ID.
    model
        QAIHM instance of the model
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    component_names
        Name of all model components (if applicable), or None of there are no components.
        Default is None.
    """
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model, model_id)
    )
    assert graph_names is None and component_graph_names is None, (
        "Graph names are not supported for on-device inference"
    )
    test_params = ScExportTestParams(
        model_id,
        path=scorecard_path,
        precision=precision,
        device=device,
        component_names=component_names,
        graph_names=graph_names,
        component_graph_names=component_graph_names,
    )

    # Patch previous jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        link_job_patch,
        _,  # profile_job_patch
        inference_job_patch,
    ) = patch_hub_with_cached_jobs(
        test_params,
        patch_quantization=not QAIHMModelManifest.from_model(model_id).is_aimet,
        patch_compile=True,
        patch_link=scorecard_path.runtime.uses_hub_link,
        patch_profile=False,
        patch_inference=True,
    )

    psnr_values: list[str] = []

    def _mock_tabulate_fn(df: Any, **kwargs: Any) -> str:
        new_psnr_values, tabulate_results = mock_tabulate_fn(df)
        psnr_values.extend(new_psnr_values)
        return tabulate_results

    tabulate_patch = mock.patch(
        "qai_hub_models.utils.printing.tabulate",
        side_effect=_mock_tabulate_fn,
    )

    with (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        link_job_patch,
        inference_job_patch,
        tabulate_patch,
    ):
        export_model(
            device=device.execution_device,
            target_runtime=scorecard_path.runtime,
            precision=precision,
            skip_downloading=True,
            skip_profiling=True,
        )

    write_accuracy(model_id, device.chipset, precision, scorecard_path, psnr_values)


def _get_dataset_cache_patch(
    dataset_cls: type[BaseDataset],
    scorecard_path: ScorecardProfilePath,
    model_cls: type[WorkbenchModel | CollectionModel],
) -> mock._patch:
    dataset_dir = get_and_sync_datasets_cache_dir(
        scorecard_path.runtime.channel_last_native_execution,
        dataset_cls,
        get_num_eval_samples(dataset_cls),
        model_cls,
    )
    return mock.patch(
        "qai_hub_models.utils.evaluate.helpers.get_hub_datasets_path",
        return_value=dataset_dir.parent,
    )


def get_num_eval_samples(dataset_cls: type[BaseDataset]) -> int:
    """
    Resolve how many samples to evaluate for a given dataset.

    This needs to be set in multiple callsites and for both num_samples and samples_per_job.

    Parameters
    ----------
    dataset_cls
        The dataset class.

    Returns
    -------
    num_samples : int
        Number of samples to evaluate.
    """
    return min(
        dataset_cls.default_samples_per_job(),
        DEFAULT_NUM_EVAL_SAMPLES,
    )


def accuracy_on_dataset_via_evaluate_and_export(
    export_model: ExportFunc,
    model: WorkbenchModel,
    dataset_cls: type[BaseDataset],
    torch_val_outputs: list[np.ndarray],
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
) -> None:
    """
    Computes accuracy for the given model and dataset and saves it to disk.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    export_model
        Code-generated export function from export.py.
    model
        Model instance to run inference on.
    dataset_cls
        Dataset class to use for evaluation.
    torch_val_outputs
        Torch validation outputs.
    torch_evaluate_mock_outputs
        The outputs of the torch forward passes in the format expected by the evaluate function.
    model_id
        Model ID.
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.

    """
    dataset_name = dataset_cls.dataset_name()
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model, model_id)
    )
    assert graph_names is None and component_graph_names is None, (
        "Graph names are not supported for on-device inference"
    )
    test_params = ScExportTestParams(
        model_id,
        path=scorecard_path,
        precision=precision,
        device=device,
        component_names=component_names,
        graph_names=graph_names,
        component_graph_names=component_graph_names,
    )

    cache_path_patch = _get_dataset_cache_patch(
        dataset_cls, scorecard_path, model.__class__
    )

    cpu_accuracy = load_yaml(ScorecardArtifact.CPU_ACCURACY.touch())
    sim_acc = cpu_accuracy.get(_get_sim_cpu_key(model_id, precision), None)

    torch_key = _get_torch_cpu_key(model_id)
    if torch_key not in cpu_accuracy:
        raise CachedScorecardJobError(
            "Torch accuracy data is missing. Accuracy data collection (test_torch_accuracy) probably failed."
        )
    torch_acc = float(cpu_accuracy[torch_key])

    dataset_metadata = None
    metric_metadata = None
    num_samples = None
    with suppress(NotImplementedError):
        dataset_metadata = dataset_cls.get_dataset_metadata()
        num_samples = get_num_eval_samples(dataset_cls)
        metric_metadata = model.get_evaluator().get_metric_metadata()

    try:
        # Get existing inference jobs, then create related patches
        # This will raise a ValueError if any of the jobs failed
        inference_sc_jobs = (
            InferenceScorecardJobYaml.from_test_artifacts().get_all_jobs(
                test_params,
                raise_if_not_successful=True,
                raise_if_jobs_are_missing=True,
            )
        )
    except (CachedScorecardJobError, ValueError):
        # If no on-device accuracy numbers, we still want to write torch, sim numbers
        write_accuracy(
            model_id,
            device.chipset,
            precision,
            scorecard_path,
            [],
            torch_acc,
            None,
            float(sim_acc) if sim_acc is not None else None,
            dataset_name,
            dataset_metadata,
            metric_metadata,
            num_samples,
        )
        raise

    inference_jobs = [x.job for x in inference_sc_jobs.values() if x is not None]
    inference_output_datas = [x.download_output_data() for x in inference_jobs]
    dataset_download_patch = mock.patch(
        "qai_hub.client.Dataset.download", side_effect=inference_output_datas
    )
    inference_job_dataset_download_patch = mock.patch(
        "qai_hub.client.InferenceJob.download_output_data",
        side_effect=inference_output_datas,
    )
    on_device_call_patch = mock.patch.object(
        AsyncOnDeviceModel,
        "__call__",
        new=callable_side_effect(
            iter([mock_on_device_model_call(x) for x in inference_jobs])
        ),
    )
    torch_call_patch = mock.patch(
        "qai_hub_models.utils.evaluate.helpers.WorkbenchModel.__call__",
        side_effect=torch_evaluate_mock_outputs,
    )
    compare_torch_inference_patch = mock.patch(
        "qai_hub_models.utils.compare._torch_inference_impl",
        side_effect=[torch_val_outputs],
    )

    # Run eval script to collect accuracy metrics
    num_samples = get_num_eval_samples(dataset_cls)
    inference_job = next(iter(inference_jobs))
    on_device_model = AsyncOnDeviceModel(
        model=inference_job.model,
        input_names=None,
        device=inference_job.device,
    )
    input_spec = on_device_model.get_input_spec()
    with (
        cache_path_patch,
        dataset_download_patch,
        on_device_call_patch,
        torch_call_patch,
    ):
        evaluate_result = evaluate_on_dataset(
            evaluator_func=model.get_evaluator,
            model_executors={"on-device": on_device_model},
            input_spec=input_spec,
            dataset_cls=dataset_cls,
            use_cache=True,
            num_samples=num_samples,
            samples_per_job=num_samples,
        )

    # Patch previous jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        link_job_patch,
        _,  # profile_job_patch
        inference_job_patch,
    ) = patch_hub_with_cached_jobs(
        params=test_params,
        patch_quantization=not QAIHMModelManifest.from_model(model_id).is_aimet,
        patch_compile=True,
        patch_link=scorecard_path.runtime.uses_hub_link,
        patch_inference=True,
    )

    psnr_values: list[str] = []

    def _mock_tabulate_fn(df: Any, **kwargs: Any) -> str:
        new_psnr_values, tabulate_results = mock_tabulate_fn(df)
        psnr_values.extend(new_psnr_values)
        return tabulate_results

    tabulate_patch = mock.patch(
        "qai_hub_models.utils.printing.tabulate",
        side_effect=_mock_tabulate_fn,
    )
    with (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        link_job_patch,
        inference_job_patch,
        compare_torch_inference_patch,
        inference_job_dataset_download_patch,
        tabulate_patch,
    ):
        export_model(
            device=device.execution_device,
            target_runtime=scorecard_path.runtime,
            precision=precision,
            skip_downloading=True,
            skip_profiling=True,
        )

    write_accuracy(
        model_id,
        device.chipset,
        precision,
        scorecard_path,
        psnr_values,
        torch_acc,
        evaluate_result.device_accuracy,
        float(sim_acc) if sim_acc is not None else None,
        dataset_name,
        dataset_metadata,
        metric_metadata,
        num_samples,
    )


def torch_accuracy_on_dataset(
    model: WorkbenchModel,
    dataset_cls: type[BaseDataset],
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
    model_id: str,
) -> None:
    """
    Computes accuracy for the given model on pytorch.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    model
        Model instance to run inference on.
    dataset_cls
        Dataset class to use for evaluation.
    torch_evaluate_mock_outputs
        The outputs of the torch forward passes in the format
        expected by the evaluate function.
    model_id
        Model ID.

    """
    # Create evaluator BEFORE mock is active so train_model() uses the real model.
    # Some evaluators (e.g. ClassificationEvaluator) call the model many times
    # during __init__, which would exhaust the mock's side_effect list.
    evaluator = model.get_evaluator()

    torch_call_patch = mock.patch(
        "qai_hub_models.utils.evaluate.helpers.WorkbenchModel.__call__",
        side_effect=torch_evaluate_mock_outputs,
    )
    scorecard_path = ScorecardProfilePath.ONNX
    cache_path_patch = _get_dataset_cache_patch(
        dataset_cls, scorecard_path, model.__class__
    )
    num_samples = get_num_eval_samples(dataset_cls)
    input_spec = model.get_input_spec()
    with torch_call_patch, cache_path_patch:
        evaluate_result = evaluate_on_dataset(
            evaluator_func=lambda: evaluator,
            model_executors={"torch": model},
            input_spec=input_spec,
            use_cache=True,
            dataset_cls=dataset_cls,
            num_samples=num_samples,
            samples_per_job=num_samples,
        )
    cache_key = _get_torch_cpu_key(model_id)
    append_line_to_file(
        ScorecardArtifact.CPU_ACCURACY.touch(),
        f"{cache_key}: {evaluate_result.torch_accuracy:.3g}",
    )


def sim_accuracy_on_dataset(
    model: WorkbenchModel,
    dataset_cls: type[BaseDataset],
    model_id: str,
    precision: Precision,
) -> None:
    """
    Computes accuracy for the given model on quantsim.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    model
        Model instance to run inference on.
    dataset_cls
        Dataset class to use for evaluation.
    model_id
        Model ID.
    precision
        Model precision.

    """
    component_names, graph_names, component_graph_names = (
        _get_components_and_graph_names(model, model_id)
    )
    assert (
        component_names is None
        and graph_names is None
        and component_graph_names is None
    )
    test_params = ScExportTestParams(
        model_id,
        path=None,
        precision=precision,
        component_names=component_names,
        graph_names=graph_names,
        component_graph_names=component_graph_names,
    )

    if precision == Precision.float:
        return
    scorecard_path = ScorecardProfilePath.ONNX
    cache_path_patch = _get_dataset_cache_patch(
        dataset_cls, scorecard_path, model.__class__
    )
    quantize_sc_jobs = QuantizeScorecardJobYaml.from_test_artifacts().get_all_jobs(
        test_params, raise_if_not_successful=True, raise_if_jobs_are_missing=True
    )
    quantize_jobs = [x.job for x in quantize_sc_jobs.values() if x is not None]
    num_samples = get_num_eval_samples(dataset_cls)
    quantized_hub_model = next(iter(quantize_jobs)).get_target_model()
    assert quantized_hub_model is not None
    quantized_producer = quantized_hub_model.get_producer()
    assert isinstance(quantized_producer, (hub.QuantizeJob, hub.CompileJob))
    input_spec = cast(InputSpec, quantized_producer.shapes)
    with cache_path_patch:
        evaluate_result = evaluate_on_dataset(
            evaluator_func=model.get_evaluator,
            model_executors={"quant cpu": _load_quant_cpu_onnx(quantized_hub_model)},
            input_spec=input_spec,
            dataset_cls=dataset_cls,
            use_cache=True,
            num_samples=num_samples,
            samples_per_job=num_samples,
        )
        cache_key = _get_sim_cpu_key(model_id, precision)
        append_line_to_file(
            ScorecardArtifact.CPU_ACCURACY.touch(),
            f"{cache_key}: {evaluate_result.sim_accuracy:.3g}",
        )

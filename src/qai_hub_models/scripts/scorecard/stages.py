# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
The scorecard's stages for a single, collection or multi-graph PyTorch recipe
model, with or without an eval dataset.

Each stage calls the pipeline type's ``export_model`` directly. The caller owns
the pipeline type's ``*ExportIntermediates`` (so source models and quantize jobs can be
reused across submissions) and gets the updated ones back. The job-ID YAMLs are
written for the collect phase.

There is one inference job per (precision, path, device) (qai_hub 0.56 merged
profiling into ``hub.InferenceJob``), written to inference-jobs.yaml. Models
without an eval dataset submit it via ``export_model`` on sample inputs. Models
with one submit it via ``evaluate_model`` on the dataset. Torch and
quantsim accuracy are computed once per model and once per precision
respectively, and recorded in cpu-accuracy.yaml. Collection and multi-graph
models never use the eval dataset: they always go through ``export_model``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, ParamSpec

import qai_hub as hub
from filelock import FileLock

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.envvars import S3ArtifactsDirEnvvar, ScopeEnvvar
from qai_hub_models.scorecard.execution_helpers import wait_for_prerequisite_job
from qai_hub_models.scorecard.params import ScExportTestParams
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    InferenceScorecardJobYaml,
    LinkScorecardJobYaml,
    QuantizeScorecardJobYaml,
    ScorecardAssetYaml,
    ScorecardJobYaml,
)
from qai_hub_models.scorecard.utils.testing_async_utils import (
    has_previous_accuracy,
    write_accuracy,
)
from qai_hub_models.scorecard.utils.testing_export_eval import (
    _get_components_and_graph_names,
    _get_sim_cpu_key,
    _get_torch_cpu_key,
    fetch_cached_jobs_if_model_assets_are_identical,
    get_num_eval_samples,
)
from qai_hub_models.scripts.scorecard.types import (
    InferenceJobs,
    Intermediates,
    PipelineType,
    Result,
    compile_of,
    empty_intermediates,
    inference_of,
    link_of,
    pipeline_type,
    quantize_of,
    typed_jobs,
    update_from_result,
    update_intermediates,
)
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    NoAWSCredsError,
    get_qaihm_s3,
    s3_multipart_upload,
)
from qai_hub_models.utils.base_config import dump_yaml_dict
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.evaluate.helpers import EvaluateIntermediates
from qai_hub_models.utils.evaluate.pipeline import evaluate_model
from qai_hub_models.utils.export.context import resolve_model_cls
from qai_hub_models.utils.export.dispatch import select_pipeline
from qai_hub_models.utils.export.result import ExportIntermediates
from qai_hub_models.utils.inference import WorkbenchJobFailed, job_failed_error
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT
from qai_hub_models.utils.printing import format_metric
from qai_hub_models.utils.qai_hub_helpers import get_target_models

P = ParamSpec("P")


class JobKind(IntEnum):
    """Ordered so each kind depends only on the kinds before it."""

    QUANTIZE = 1
    COMPILE = 2
    LINK = 3
    INFERENCE = 4


class PrerequisiteJobNotFound(Exception):
    """A stage's upstream job is absent, usually because that stage failed."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"prerequisite {kind} job not found")


@dataclass
class StageOutcome:
    """A stage's result (if it succeeded), wall time and exception (if it failed)."""

    result: Intermediates | None
    seconds: float
    error: Exception | None = None


def describe_error(error: BaseException) -> str:
    """
    A failed stage's error for the timings CSV.

    Job failures keep their own message. Anything else is "<file>:<line>:
    <Type>: <message>", with the file and line of the last traceback frame.
    """
    if isinstance(error, WorkbenchJobFailed):
        return str(error)
    description = f"{type(error).__name__}: {error}"
    if not (frames := traceback.extract_tb(error.__traceback__)):
        return description
    last = frames[-1]
    return f"{'/'.join(Path(last.filename).parts[-2:])}:{last.lineno}: {description}"


def timed_stage(fn: Callable[P, Intermediates | None]) -> Callable[P, StageOutcome]:
    """Make a stage return a StageOutcome instead of raising."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> StageOutcome:
        start = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except PrerequisiteJobNotFound as e:
            print(f"{fn.__name__} blocked: {e}")
            return StageOutcome(None, time.monotonic() - start, e)
        except Exception as e:
            print(f"{fn.__name__} failed:\n{traceback.format_exc()}")
            return StageOutcome(None, time.monotonic() - start, e)
        return StageOutcome(result, time.monotonic() - start)

    return wrapper


def _recipe_dir(model_id: str) -> Path:
    return QAIHM_MODELS_ROOT / model_id


def _eval_dataset_cls(model_id: str) -> type[BaseDataset] | None:
    """The eval dataset class of a single model, or None if it has none or is not single."""
    if pipeline_type(model_id) != PipelineType.SINGLE:
        return None
    model_cls = resolve_model_cls(_recipe_dir(model_id))
    eval_dataset_classes = model_cls.get_eval_dataset_classes()
    return eval_dataset_classes[0] if eval_dataset_classes else None


def _measures_accuracy(model_id: str, device: ScorecardDevice) -> bool:
    """Accuracy is only measured on the model's default device, in perf_and_accuracy scopes."""
    if ScopeEnvvar.get() != ScopeEnvvar.PERF_AND_ACCURACY:
        return False
    default_device = QAIHMModelManifest.from_model(model_id).default_device
    return device == ScorecardDevice.get(default_device)


@functools.cache
def _component_and_graph_names(
    model_id: str,
) -> tuple[list[str] | None, list[str] | None, Any]:
    """
    Component / graph names for the model. Instantiates it if the artifacts
    YAMLs don't have them yet, which also records them for the collect phase.
    """
    model_cls = resolve_model_cls(_recipe_dir(model_id))
    names = _get_components_and_graph_names(model_cls, model_id)
    if pipeline_type(model_id) != PipelineType.SINGLE and names == (None, None, None):
        names = _get_components_and_graph_names(model_cls.from_pretrained(), model_id)
    return names


def _test_params(
    model_id: str,
    precision: Precision,
    path: ScorecardInferencePath,
    device: ScorecardDevice,
) -> ScExportTestParams[ScorecardInferencePath]:
    """The identity of one parameterization, with the model's component / graph names."""
    component_names, graph_names, component_graph_names = _component_and_graph_names(
        model_id
    )
    return ScExportTestParams(
        model_id,
        path=path,
        precision=precision,
        device=device,
        component_names=component_names,
        graph_names=graph_names,
        component_graph_names=component_graph_names,
    )


def _is_llm(model_id: str) -> bool:
    return pipeline_type(model_id) == PipelineType.MULTI_GRAPH_COLLECTION


def _export(
    model_id: str,
    intermediates: Intermediates,
    stub_llm_checkpoint: bool = False,
    **kwargs: Any,
) -> Result:
    """Run the recipe's export pipeline, starting from `intermediates`."""
    with contextlib.ExitStack() as stack:
        if _is_llm(model_id):
            # Deferred: scorecard.llm pulls in transformers, absent from non-LLM venvs.
            from qai_hub_models.scripts.scorecard.llm import (
                llm_export_context,
                llm_export_kwargs,
            )

            kwargs.update(llm_export_kwargs(kwargs["precision"]))
            stack.enter_context(llm_export_context(model_id, stub_llm_checkpoint))
        result = select_pipeline(_recipe_dir(model_id))(
            export_intermediates=intermediates, **kwargs
        )
    assert isinstance(result, Result)
    return result


def _each_job(jobs: hub.Job | Mapping[Any, hub.Job]) -> Iterable[hub.Job]:
    """The jobs in a single job or a component / graph group of them."""
    return jobs.values() if isinstance(jobs, Mapping) else [jobs]


def _load_cpu_accuracy() -> dict:
    """cpu-accuracy.yaml's contents; empty until the first accuracy is recorded."""
    return load_yaml(ScorecardArtifact.CPU_ACCURACY.touch()) or {}


def record_cpu_accuracy(key: str, value: float) -> None:
    """Set `key` in cpu-accuracy.yaml, replacing any recorded value."""
    path = ScorecardArtifact.CPU_ACCURACY.touch()
    with FileLock(f"{path}.lock"):
        recorded = load_yaml(path) or {}
        recorded[key] = float(f"{value:.3g}")
        dump_yaml_dict(recorded, path)


def _write_inference_jobs(jobs: InferenceJobs, test_params: ScExportTestParams) -> None:
    cache = InferenceScorecardJobYaml.from_test_artifacts()
    cache.update_from_export_output(jobs, test_params)
    cache.to_file()


def _wait_and_assert_success(jobs: hub.Job | Mapping[Any, hub.Job]) -> None:
    for job in _each_job(jobs):
        status = wait_for_prerequisite_job(job)
        if not status.success:
            raise job_failed_error(job)


def _is_quantized(model_id: str, precision: Precision) -> bool:
    """Multi-graph models are never quantized by the scorecard."""
    return precision != Precision.float and pipeline_type(model_id) not in (
        PipelineType.MULTI_GRAPH,
        PipelineType.MULTI_GRAPH_COLLECTION,
    )


def _load_recorded_jobs(
    yaml_cls: type[ScorecardJobYaml], test_params: ScExportTestParams
) -> object:
    """Load the jobs a previous run saved in `yaml_cls`'s file in the artifacts dir, without waiting on them. None if there are none."""
    jobs = yaml_cls.from_test_artifacts().get_export_output(
        test_params,
        wait_for_job=False,
        raise_if_not_successful=False,
        raise_if_jobs_are_missing=False,
    )
    return jobs or None


def load_intermediates(
    model_id: str,
    precision: Precision,
    path: ScorecardInferencePath,
    device: ScorecardDevice,
    reuse: frozenset[JobKind],
) -> Intermediates:
    """
    The jobs of the kinds in `reuse` that are recorded in the artifacts dir's
    job-ID YAMLs, as unwaited handles. Every other applicable kind will be
    submitted fresh.
    """
    test_params = _test_params(model_id, precision, path, device)
    intermediates = empty_intermediates(pipeline_type(model_id))
    if JobKind.QUANTIZE in reuse and _is_quantized(model_id, precision):
        intermediates = update_intermediates(
            intermediates,
            quantize_jobs=typed_jobs(
                _load_recorded_jobs(QuantizeScorecardJobYaml, test_params),
                hub.QuantizeJob,
            ),
        )
    if JobKind.COMPILE in reuse:
        intermediates = update_intermediates(
            intermediates,
            compile_jobs=typed_jobs(
                _load_recorded_jobs(CompileScorecardJobYaml, test_params),
                hub.CompileJob,
            ),
        )
    if JobKind.LINK in reuse and path.runtime.uses_hub_link:
        intermediates = update_intermediates(
            intermediates,
            link_jobs=typed_jobs(
                _load_recorded_jobs(LinkScorecardJobYaml, test_params), hub.LinkJob
            ),
        )
    if JobKind.INFERENCE in reuse:
        intermediates = update_intermediates(
            intermediates,
            inference_jobs=typed_jobs(
                _load_recorded_jobs(InferenceScorecardJobYaml, test_params),
                hub.InferenceJob,
            ),
        )
    return intermediates


@timed_stage
def submit_jobs_through_link(
    model_id: str,
    precision: Precision,
    path: ScorecardInferencePath,
    device: ScorecardDevice,
    intermediates: Intermediates,
) -> Intermediates:
    """
    Submit (upload, quantize,) compile and link without waiting.
    Any source model or job in `intermediates` is reused; everything else is
    submitted fresh and overwrites the entry in the quantize, compile and link
    job-ID YAMLs.

    Link is submitted here, ahead of the identical-assets check in
    ``submit_inference_jobs``, which compares its output with last week's.
    """
    test_params = _test_params(model_id, precision, path, device)

    updated = update_from_result(
        intermediates,
        _export(
            model_id,
            intermediates,
            device=device.execution_device,
            precision=precision,
            target_runtime=path.runtime,
            skip_inferencing=True,
            skip_downloading=True,
            skip_summary=True,
            compile_options=path.compile_path.get_compile_options(
                include_target_runtime=True
            ),
            link_options=path.compile_path.get_link_options(),
        ),
    )

    quantize_jobs = quantize_of(updated)
    compile_jobs = compile_of(updated)
    link_jobs = link_of(updated)
    assert compile_jobs is not None
    if quantize_jobs is not None:
        quantize_yaml = QuantizeScorecardJobYaml.from_test_artifacts()
        quantize_yaml.update_from_export_output(quantize_jobs, test_params)
        quantize_yaml.to_file()
    compile_yaml = CompileScorecardJobYaml.from_test_artifacts()
    compile_yaml.update_from_export_output(compile_jobs, test_params)
    compile_yaml.to_file()
    if link_jobs is not None:
        link_yaml = LinkScorecardJobYaml.from_test_artifacts()
        link_yaml.update_from_export_output(link_jobs, test_params)
        link_yaml.to_file()
    return updated


@timed_stage
def submit_inference_jobs(
    model_id: str,
    precision: Precision,
    path: ScorecardInferencePath,
    device: ScorecardDevice,
    intermediates: Intermediates,
) -> Intermediates:
    """
    Submit the inference job(s). `intermediates` must hold stage 1's compile
    jobs (and link jobs for AOT runtimes). Waits on the jobs whose output runs
    on device (the link jobs for AOT runtimes, the compile jobs otherwise) and
    asserts they succeeded, so nothing is submitted behind a failed job.
    Writes inference-jobs.yaml. A recorded inference job in `intermediates` is
    reused.

    If the asset that runs on device (the link job's for AOT runtimes, the
    compile job's otherwise) matches last week's (prod-only md5 gate) and
    accuracy.csv already has a row for it, last week's inference job is reused
    and nothing is submitted. This job's own link job stays in link-jobs.yaml:
    its output is identical.
    """
    test_params = _test_params(model_id, precision, path, device)
    if inference_of(intermediates) is not None:
        return intermediates

    prerequisite = (
        link_of(intermediates)
        if path.runtime.uses_hub_link
        else compile_of(intermediates)
    )
    if prerequisite is None:
        raise PrerequisiteJobNotFound(
            "link" if path.runtime.uses_hub_link else "compile"
        )
    _wait_and_assert_success(prerequisite)

    dataset_cls = _eval_dataset_cls(model_id)
    measures_accuracy = _measures_accuracy(model_id, device)
    if has_previous_accuracy(
        model_id,
        device.chipset,
        precision,
        path,
        dataset_cls is not None and measures_accuracy,
    ) and (
        cached_jobs := typed_jobs(
            fetch_cached_jobs_if_model_assets_are_identical(test_params),
            hub.InferenceJob,
        )
    ):
        print(f"{model_id} {precision} {path.value}: assets unchanged, reusing jobs")
        _write_inference_jobs(cached_jobs, test_params)
        return update_intermediates(intermediates, inference_jobs=cached_jobs)

    if dataset_cls is not None and measures_accuracy:
        assert isinstance(intermediates, ExportIntermediates)
        chain_job = intermediates.link_job or intermediates.compile_job
        assert chain_job is not None
        num_samples = get_num_eval_samples(dataset_cls)
        evaluate_result = evaluate_model(
            source_dir=_recipe_dir(model_id),
            device=device.execution_device,
            target_runtime=path.runtime,
            precision=precision,
            hub_model_id=get_target_models(chain_job).model_id,
            dataset_cls=dataset_cls,
            num_samples=num_samples,
            samples_per_job=num_samples,
            inference_options=path.get_inference_options(precision, device),
            use_dataset_cache=True,
            skip_torch_accuracy=True,
            skip_waiting=True,
        )
        assert len(evaluate_result.inference_jobs) == 1
        inference_job = evaluate_result.inference_jobs[0]
        _write_inference_jobs(inference_job, test_params)
        return dataclasses.replace(intermediates, inference_job=inference_job)

    updated = update_from_result(
        intermediates,
        _export(
            model_id,
            intermediates,
            device=device.execution_device,
            precision=precision,
            target_runtime=path.runtime,
            skip_downloading=True,
            skip_summary=True,
            inference_options=path.get_inference_options(precision, device),
        ),
    )
    inference_jobs = inference_of(updated)
    assert inference_jobs is not None
    _write_inference_jobs(inference_jobs, test_params)
    return updated


@timed_stage
def compute_cpu_accuracy(
    model_id: str,
    precision: Precision,
    path: ScorecardInferencePath,
    device: ScorecardDevice,
    intermediates: Intermediates,
    use_saved_accuracy: bool,
    computed: set[str],
) -> None:
    """
    Compute torch accuracy (once per model) and quantsim accuracy (once per
    precision) and write them to cpu-accuracy.yaml.

    Torch and quantsim accuracy don't depend on the device, so they are only
    computed once per run. `computed` holds the keys already computed in this
    run. Values in the file from earlier runs are replaced, unless
    `use_saved_accuracy` is set, in which case they are kept and not recomputed.

    Does nothing for models without an eval dataset, collection / multi-graph
    models, non-default devices, and outside perf_and_accuracy scopes.
    """
    dataset_cls = _eval_dataset_cls(model_id)
    if dataset_cls is None or not _measures_accuracy(model_id, device):
        return

    num_samples = get_num_eval_samples(dataset_cls)
    cpu_accuracy = _load_cpu_accuracy()
    if not use_saved_accuracy:
        cpu_accuracy = {k: v for k, v in cpu_accuracy.items() if k in computed}
    torch_key = _get_torch_cpu_key(model_id)
    if torch_key not in cpu_accuracy:
        torch_result = evaluate_model(
            source_dir=_recipe_dir(model_id),
            device=device.execution_device,
            dataset_cls=dataset_cls,
            num_samples=num_samples,
            samples_per_job=num_samples,
            use_dataset_cache=True,
            skip_device_accuracy=True,
        )
        assert torch_result.torch_accuracy is not None
        record_cpu_accuracy(torch_key, torch_result.torch_accuracy)
        computed.add(torch_key)

    sim_key = _get_sim_cpu_key(model_id, precision)
    if precision == Precision.float:
        return
    if sim_key in cpu_accuracy:
        return
    assert isinstance(intermediates, ExportIntermediates)
    quantize_job = intermediates.quantize_job
    if quantize_job is None:
        raise PrerequisiteJobNotFound("quantize")
    _wait_and_assert_success(quantize_job)
    sim_result = evaluate_model(
        source_dir=_recipe_dir(model_id),
        device=device.execution_device,
        target_runtime=path.runtime,
        precision=precision,
        hub_model_id=get_target_models(quantize_job).model_id,
        dataset_cls=dataset_cls,
        num_samples=num_samples,
        samples_per_job=num_samples,
        use_dataset_cache=True,
        skip_torch_accuracy=True,
        skip_device_accuracy=True,
        compute_quant_cpu_accuracy=True,
    )
    assert sim_result.sim_accuracy is not None
    record_cpu_accuracy(sim_key, sim_result.sim_accuracy)
    computed.add(sim_key)


def _psnr_values(inference_metrics: Any) -> list[str] | None:
    """
    PSNR per output, formatted. Collection components are concatenated in
    order. None if the pipeline did not compute metrics (multi-graph models).
    """
    if inference_metrics is None:
        return None
    frames = (
        list(inference_metrics.values())
        if isinstance(inference_metrics, dict)
        else [inference_metrics]
    )
    return [format_metric(v) for frame in frames for v in frame["psnr"].tolist()]


@timed_stage
def finalize_metrics(
    model_id: str,
    precision: Precision,
    path: ScorecardInferencePath,
    device: ScorecardDevice,
    intermediates: Intermediates,
) -> Intermediates | None:
    """
    Wait on the inference job(s) in `intermediates` and write the row to accuracy.csv.
    Submits nothing: compile, link and inference jobs are all supplied, so the
    pipeline skips those steps. With an eval dataset (single models only) the
    row has on-device accuracy next to the torch and sim accuracy from
    ``compute_cpu_accuracy``, and no PSNR. Otherwise the row has the PSNR from
    ``export_model``. Does nothing on non-default devices, which only collect
    latency, outside perf_and_accuracy scopes, or for multi-graph models, whose
    pipeline computes no PSNR.
    """
    if inference_of(intermediates) is None:
        raise PrerequisiteJobNotFound("inference")

    if not _measures_accuracy(model_id, device):
        return None

    dataset_cls = _eval_dataset_cls(model_id)
    if dataset_cls is not None:
        assert isinstance(intermediates, ExportIntermediates)
        inference_job = intermediates.inference_job
        assert inference_job is not None
        evaluate_result = evaluate_model(
            source_dir=_recipe_dir(model_id),
            device=device.execution_device,
            target_runtime=path.runtime,
            precision=precision,
            hub_model_id=inference_job.model.model_id,
            dataset_cls=dataset_cls,
            num_samples=get_num_eval_samples(dataset_cls),
            samples_per_job=get_num_eval_samples(dataset_cls),
            use_dataset_cache=True,
            skip_torch_accuracy=True,
            evaluate_intermediates=EvaluateIntermediates(
                inference_jobs=[inference_job]
            ),
        )
        cpu_accuracy = _load_cpu_accuracy()
        torch_key = _get_torch_cpu_key(model_id)
        assert torch_key in cpu_accuracy, (
            f"Torch accuracy for {model_id} is missing; the cpu_accuracy stage "
            "probably failed."
        )
        sim_accuracy = cpu_accuracy.get(_get_sim_cpu_key(model_id, precision))
        dataset_metadata = None
        metric_metadata = None
        num_samples = None
        with contextlib.suppress(NotImplementedError):
            dataset_metadata = dataset_cls.get_dataset_metadata()
            num_samples = get_num_eval_samples(dataset_cls)
            model = resolve_model_cls(_recipe_dir(model_id)).from_pretrained()
            metric_metadata = model.get_evaluator().get_metric_metadata()
        write_accuracy(
            model_id,
            device.chipset,
            precision,
            path,
            [],
            torch_accuracy=float(cpu_accuracy[torch_key]),
            device_accuracy=evaluate_result.device_accuracy,
            sim_accuracy=sim_accuracy,
            dataset_name=dataset_cls.dataset_name(),
            dataset_metadata=dataset_metadata,
            metric_metadata=metric_metadata,
            num_samples=num_samples,
        )
        return intermediates

    export_result = _export(
        model_id,
        intermediates,
        device=device.execution_device,
        precision=precision,
        target_runtime=path.runtime,
        skip_downloading=True,
    )
    psnr_values = _psnr_values(export_result.inference_metrics)
    if psnr_values is not None:
        write_accuracy(model_id, device.chipset, precision, path, psnr_values)
    return update_from_result(intermediates, export_result)


@timed_stage
def upload_assets(
    model_id: str,
    precision: Precision,
    path: ScorecardInferencePath,
    device: ScorecardDevice,
    intermediates: Intermediates,
) -> None:
    """
    Download the compiled bundle, upload it to S3, and record the
    key in release-assets.yaml. Needs compile (and link) in `intermediates`.
    If the inference jobs are there and succeeded, their tool versions are
    recorded. A no-op if this path isn't published or no S3 destination is
    configured.
    """
    if not path.is_published:
        return

    s3_bucket = None
    with contextlib.suppress(NoAWSCredsError):
        s3_bucket = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)[0]
    if s3_bucket is None or S3ArtifactsDirEnvvar.is_default():
        return

    compile_jobs = compile_of(intermediates)
    link_jobs = link_of(intermediates)
    if compile_jobs is None:
        raise PrerequisiteJobNotFound("compile")
    _wait_and_assert_success(compile_jobs)
    if link_jobs is not None:
        _wait_and_assert_success(link_jobs)
    chipset_key = device if path.runtime.is_aot_compiled else cs_universal

    assets_path = ScorecardArtifact.RELEASE_ASSETS.path
    assets_cache = ScorecardAssetYaml.from_yaml(
        assets_path, create_empty_if_no_file=True
    )
    # Non-AOT runtimes share one asset across devices, so skip the repeats.
    if assets_cache.get_asset(model_id, precision, chipset_key, path) is not None:
        return

    # Inference jobs report the tool versions that ran on device, which are more
    # complete than the compile job's. Skip them if any failed.
    inference_jobs = inference_of(intermediates)
    if inference_jobs is not None and not all(
        wait_for_prerequisite_job(job).success for job in _each_job(inference_jobs)
    ):
        inference_jobs = None

    export_result = _export(
        model_id,
        update_intermediates(
            empty_intermediates(pipeline_type(model_id)),
            compile_jobs=compile_jobs,
            link_jobs=link_jobs,
            inference_jobs=inference_jobs,
        ),
        stub_llm_checkpoint=True,
        device=device.execution_device,
        precision=precision,
        target_runtime=path.runtime,
        skip_inferencing=True,
        skip_summary=True,
        zip_assets=True,
    )
    assert export_result.download_path is not None
    assert export_result.tool_versions is not None

    s3_key = str(S3ArtifactsDirEnvvar.get() / export_result.download_path.name)
    s3_multipart_upload(
        bucket=s3_bucket,
        key=s3_key,
        local_file_path=export_result.download_path,
    )
    assets_cache.add_asset(
        QAIHMModelReleaseAssets.AssetDetails(
            s3_key=s3_key, tool_versions=export_result.tool_versions
        ),
        model_id,
        precision,
        chipset_key,
        path,
    )
    assets_cache.to_yaml(assets_path)

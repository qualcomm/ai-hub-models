# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Run the scorecard for exactly one model: compile (through link), submit
inference jobs, compute torch/quantsim accuracy, finalize accuracy, then
(optionally) collect the model's results into perf.yaml / numerics.yaml and
results.csv. perf.yaml, numerics.yaml and the checked-in intermediates are
updated only with `--publish-numbers`, and release assets are uploaded only with
`--upload-assets`.

Nothing is reused by default: every job is submitted fresh and overwrites its
job-ID YAML entry. `--use-cached <kind>` uses that job kind and every kind
before it (quantize > compile > link > inference) from those YAMLs instead, and
`--end-before` stops at a stage. Per-stage wall times are appended to
stage-timings.csv in the artifacts dir. Models outside the supported pipeline types
(see ``_unsupported_pipeline_reason``) are skipped with the reason printed.
Single, collection and multi-graph models are supported; the latter two
never use the eval dataset, so their accuracy column is PSNR from sample inputs.
"""

from __future__ import annotations

import argparse
import csv
import functools
import inspect
import shutil
import sys
from collections.abc import Callable
from enum import IntEnum
from pathlib import Path

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    DeploymentEnvvar,
    EnabledDevicesEnvvar,
    EnabledModelsEnvvar,
    EnabledPathsEnvvar,
    EnabledPrecisionsEnvvar,
    IgnoreExistingIntermediateJobsDuringCollectionEnvvar,
    ScopeEnvvar,
    StaticModelsDirEnvvar,
)
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.scorecard.results.scorecard_summary import ModelTestConfig
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.scripts.aggregate_scorecard_results import main as aggregate_main
from qai_hub_models.scripts.collect_scorecard_results import collect
from qai_hub_models.scripts.collect_scorecard_results import (
    parse_args as parse_collect_args,
)
from qai_hub_models.scripts.scorecard.stages import (
    JobKind,
    PrerequisiteJobNotFound,
    StageOutcome,
    compute_cpu_accuracy,
    describe_error,
    finalize_metrics,
    load_intermediates,
    submit_inference_jobs,
    submit_jobs_through_link,
    upload_assets,
)
from qai_hub_models.scripts.scorecard.types import (
    CalibrationData,
    CompileJobs,
    InferenceJobs,
    Intermediates,
    LinkJobs,
    PipelineType,
    QuantizeJobs,
    SourceModels,
    calibration_data_of,
    compile_of,
    empty_intermediates,
    inference_of,
    link_of,
    pipeline_type,
    quantize_of,
    source_of,
    update_intermediates,
)
from qai_hub_models.utils._cached_from_pretrained import clear_from_pretrained_caches
from qai_hub_models.utils.asset_loaders import get_hub_datasets_path
from qai_hub_models.utils.aws import QAIHM_PRIVATE_S3_BUCKET, get_qaihm_s3_or_exit
from qai_hub_models.utils.envvars import CacheFromPretrainedEnvvar
from qai_hub_models.utils.export.context import resolve_model_cls
from qai_hub_models.utils.hub_clients import (
    get_default_hub_deployment,
    get_scorecard_client_or_raise,
    set_default_hub_client,
)
from qai_hub_models.utils.inference import WorkbenchJobFailed
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

TIMINGS_HEADER = ["model", "stage", "precision", "path", "device", "seconds", "status"]


class Stage(IntEnum):
    """Ordered so stages can be compared with --end-before. Names match its
    values (lowercased).
    """

    COMPILE = 1
    INFERENCE = 2
    CPU_ACCURACY = 3
    UPLOAD_ASSETS = 4
    DEVICE_ACCURACY = 5
    COLLECT = 6
    DONE = 7


def _unsupported_pipeline_reason(model_id: str) -> str | None:
    """None if this model fits the single-model shape
    scripts/scorecard/stages.py implements; otherwise the reason it doesn't.
    """
    sc = QAIHMModelInternalConfig.from_model(model_id)
    if not sc.runs_in_scorecard:
        return "skipped by internal-config.yaml"
    if sc.skip_export:
        return "no recipe code (skip_export)"
    return None


# LLM perf and accuracy run through QDC Genie, not Workbench inference jobs.
_LLM_SKIPPED_STAGES = frozenset(
    {Stage.INFERENCE, Stage.CPU_ACCURACY, Stage.DEVICE_ACCURACY}
)


Entry = tuple[Precision, ScorecardInferencePath, ScorecardDevice]


def _resolve_matrix(model_id: str) -> list[Entry]:
    """This model's enabled (precision, path, device) tuples, honoring
    --precisions/--runtimes/--devices (set as envvars by their CLI args).
    Entries are grouped by precision so each stage, iterating in order, reuses
    one cached from_pretrained instance per precision.
    """
    manifest = QAIHMModelManifest.from_model(model_id)
    tests = ModelTestConfig.from_recipe_model(manifest).inference_tests
    precisions = list(dict.fromkeys(precision for precision, _, _ in tests))
    return sorted(tests, key=lambda entry: precisions.index(entry[0]))


def _compile_key(entry: Entry) -> tuple[Precision, ScorecardInferencePath, str | None]:
    """Compile jobs are per (precision, path); AOT-compiled paths are also per device."""
    precision, path, device = entry
    return precision, path, device.name if path.runtime.is_aot_compiled else None


class _ModelState:
    """In-memory jobs for one model, shared across stages and submissions."""

    def __init__(self, model_id: str) -> None:
        self.pipeline_type = pipeline_type(model_id)
        model_cls = resolve_model_cls(QAIHM_MODELS_ROOT / model_id)
        self.per_precision_source = (
            "precision" in inspect.signature(model_cls.from_pretrained).parameters
        )
        self.sources: dict[Precision | None, SourceModels] = {}
        self.calibration_data: CalibrationData | None = None
        self.quantize_jobs: dict[Precision, QuantizeJobs] = {}
        self.compile_jobs: dict[tuple, CompileJobs] = {}
        self.link_jobs: dict[Entry, LinkJobs] = {}
        self.inference_jobs: dict[Entry, InferenceJobs] = {}

    def _source_key(self, entry: Entry) -> Precision | None:
        """Models built per precision upload a source per precision."""
        return entry[0] if self.per_precision_source else None

    def intermediates(self, entry: Entry) -> Intermediates:
        """Every job known so far that applies to `entry`."""
        return update_intermediates(
            empty_intermediates(self.pipeline_type),
            source=self.sources.get(self._source_key(entry)),
            calibration_data=self.calibration_data,
            quantize_jobs=self.quantize_jobs.get(entry[0]),
            compile_jobs=self.compile_jobs.get(_compile_key(entry)),
            link_jobs=self.link_jobs.get(entry),
            inference_jobs=self.inference_jobs.get(entry),
        )

    def update(self, entry: Entry, intermediates: Intermediates) -> None:
        """Record whatever jobs `intermediates` carries."""
        if (source := source_of(intermediates)) is not None:
            self.sources.setdefault(self._source_key(entry), source)
        self.calibration_data = self.calibration_data or calibration_data_of(
            intermediates
        )
        if (quantize := quantize_of(intermediates)) is not None:
            self.quantize_jobs.setdefault(entry[0], quantize)
        if (compile_ := compile_of(intermediates)) is not None:
            self.compile_jobs.setdefault(_compile_key(entry), compile_)
        if (link := link_of(intermediates)) is not None:
            self.link_jobs[entry] = link
        if (inference := inference_of(intermediates)) is not None:
            self.inference_jobs[entry] = inference

    def load_recorded(
        self, model_id: str, matrix: list[Entry], reuse: frozenset[JobKind]
    ) -> None:
        """Fill in the `reuse` kinds of jobs from the job-ID YAMLs."""
        for entry in matrix:
            self.update(entry, load_intermediates(model_id, *entry, reuse))


def _run_stage(
    log_row: Callable[[list[str]], object],
    model_id: str,
    stage: Stage,
    matrix: list[Entry],
    state: _ModelState,
    stage_fn: Callable[..., StageOutcome],
) -> bool:
    """Run `stage_fn` over the matrix, updating `state` and logging timings."""
    ok = True
    for entry in matrix:
        precision, path, device = entry
        outcome = stage_fn(model_id, *entry, state.intermediates(entry))
        if outcome.result is not None:
            state.update(entry, outcome.result)
        if outcome.error is None:
            status = "ok"
        elif isinstance(outcome.error, PrerequisiteJobNotFound):
            # Not a failure itself: the stage that failed upstream already flagged the run.
            status = f"blocked: {outcome.error}"
        else:
            # A model failing on Workbench is a result, reported in the summary.
            ok &= isinstance(outcome.error, WorkbenchJobFailed)
            status = f"failed: {describe_error(outcome.error)}"
        log_row(
            [
                model_id,
                stage.name.lower(),
                str(precision),
                path.value,
                device.name,
                f"{outcome.seconds:.1f}",
                status,
            ]
        )
    return ok


def _run_stages(
    model_id: str,
    artifacts_dir: Path,
    use_cached: JobKind | None = None,
    start_at: Stage = Stage.COMPILE,
    end_before: Stage = Stage.DONE,
    upload: bool = False,
    publish_numbers: bool = False,
) -> bool:
    """Run the scorecard stages before `end_before` for one model, then (if
    in range) collect its results.

    The job kind `use_cached` and every kind before it in JobKind are
    taken from the job-ID YAMLs in artifacts_dir; later kinds are submitted
    fresh, overwriting their YAML entries. The upload_assets stage only runs
    when `upload` is set, and collect updates perf.yaml, numerics.yaml and the
    checked-in intermediates only when `publish_numbers` is set. Writes the same job-ID YAMLs, accuracy.csv,
    perf.yaml and numerics.yaml a CI run writes, under artifacts_dir. Returns
    False if any stage failed.
    """
    reuse = (
        frozenset(k for k in JobKind if k <= use_cached) if use_cached else frozenset()
    )

    def runs(stage: Stage) -> bool:
        return start_at <= stage < end_before

    ok = True
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timings_path = artifacts_dir / "stage-timings.csv"
    needs_header = not timings_path.exists() or timings_path.stat().st_size == 0
    with timings_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(TIMINGS_HEADER)
        if reason := _unsupported_pipeline_reason(model_id):
            print(f"{model_id} | SKIPPED: {reason}")
            writer.writerow([model_id, "skipped", "", "", "", "0.0", reason])
        else:
            matrix = _resolve_matrix(model_id)
            state = _ModelState(model_id)
            if reuse:
                state.load_recorded(model_id, matrix, reuse)

            stages: list[tuple[Stage, Callable[..., StageOutcome]]] = [
                (Stage.COMPILE, submit_jobs_through_link),
                (Stage.INFERENCE, submit_inference_jobs),
                (
                    Stage.CPU_ACCURACY,
                    functools.partial(
                        compute_cpu_accuracy,
                        use_saved_accuracy=JobKind.INFERENCE in reuse,
                        computed=set(),
                    ),
                ),
                (Stage.DEVICE_ACCURACY, finalize_metrics),
            ]
            if upload:
                stages.insert(-1, (Stage.UPLOAD_ASSETS, upload_assets))

            if state.pipeline_type == PipelineType.MULTI_GRAPH_COLLECTION:
                stages = [s for s in stages if s[0] not in _LLM_SKIPPED_STAGES]

            # Each stage submits its whole matrix before the next drains it,
            # so entries queue on device in parallel rather than serially.
            try:
                for stage, stage_fn in stages:
                    if runs(stage):
                        ok &= _run_stage(
                            writer.writerow, model_id, stage, matrix, state, stage_fn
                        )
            finally:
                clear_from_pretrained_caches()

    if runs(Stage.COLLECT):
        _run_collect(model_id, artifacts_dir, publish_numbers)

    return ok


def _wipe_dataset_cache() -> None:
    shutil.rmtree(get_hub_datasets_path(), ignore_errors=True)


def run_scorecard_single(
    model_id: str,
    artifacts_dir: Path,
    use_cached: JobKind | None = None,
    start_at: Stage = Stage.COMPILE,
    end_before: Stage = Stage.DONE,
    upload: bool = False,
    publish_numbers: bool = False,
) -> bool:
    """Run `_run_stages` with the eval dataset cache wiped before and after.

    The cache is never reused across scorecard runs.
    """
    _wipe_dataset_cache()
    try:
        return _run_stages(
            model_id,
            artifacts_dir,
            use_cached,
            start_at,
            end_before,
            upload,
            publish_numbers,
        )
    finally:
        _wipe_dataset_cache()


def _run_collect(model_id: str, artifacts_dir: Path, publish_numbers: bool) -> None:
    """Call collect() in-process with this model as its only --models scope."""
    old_argv = sys.argv
    try:
        sys.argv = [
            sys.argv[0],
            "--models",
            model_id,
            "--gen-csv",
            "--gen-perf-summary",
        ]
        collect_args = parse_collect_args()
    finally:
        sys.argv = old_argv
    collect_args.artifacts_dir = artifacts_dir
    collect_args.publish_numbers = publish_numbers
    collect(collect_args)

    # Aggregation reads export-summary.csv and accuracy.csv from the artifacts dir.
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        aggregate_main()
    finally:
        sys.argv = old_argv


def add_args(parser: argparse.ArgumentParser) -> None:
    """The scorecard arguments shared by run_scorecard.py and this script."""
    EnabledModelsEnvvar.add_arg(parser, setenv=True)
    EnabledPrecisionsEnvvar.add_arg(parser, setenv=True)
    EnabledPathsEnvvar.add_arg(parser, setenv=True)
    EnabledDevicesEnvvar.add_arg(parser, setenv=True)
    ArtifactsDirEnvvar.add_arg(parser, setenv=True)
    StaticModelsDirEnvvar.add_arg(parser, setenv=True)
    ScopeEnvvar.add_arg(parser, setenv=True)
    IgnoreExistingIntermediateJobsDuringCollectionEnvvar.add_arg(parser, setenv=True)
    DeploymentEnvvar.add_arg(parser, default=get_default_hub_deployment())
    parser.add_argument(
        "--use-cached",
        choices=[k.name.lower() for k in JobKind],
        default=None,
        help="Use the cached jobs of this kind and every kind before it (in "
        f"the order {' > '.join(k.name.lower() for k in JobKind)}) from the job-ID YAMLs in the "
        "artifacts dir. Later kinds are submitted fresh and overwrite their YAML "
        "entry. The torch / quantsim accuracy in cpu-accuracy.yaml is reused "
        "only with 'inference'. By default nothing is reused.",
    )
    parser.add_argument(
        "--start-at",
        choices=[s.name.lower() for s in Stage],
        default=Stage.COMPILE.name.lower(),
        help="Start the window at this stage (inclusive). Use with --use-cached "
        "to run a later phase on another machine from the job-ID YAMLs.",
    )
    parser.add_argument(
        "--end-before",
        choices=[s.name.lower() for s in Stage],
        default=Stage.DONE.name.lower(),
        help="End the window before this stage (exclusive).",
    )
    parser.add_argument(
        "--upload-assets",
        action="store_true",
        help="Also run the upload_assets stage, which publishes release assets "
        "to S3. Off by default.",
    )
    parser.add_argument(
        "--publish-numbers",
        action="store_true",
        help="Also update perf.yaml, numerics.yaml and the checked-in "
        "intermediates. Off by default, which only produces results.csv.",
    )


def init_hub_client(args: argparse.Namespace) -> None:
    """Check S3 access if uploading, and point the default Hub client at the deployment."""
    if args.upload_assets:
        get_qaihm_s3_or_exit(QAIHM_PRIVATE_S3_BUCKET)
    set_default_hub_client(get_scorecard_client_or_raise(args.deployment))


def run_from_args(model_id: str, args: argparse.Namespace) -> bool:
    """run_scorecard_single() with the options parsed by add_args()."""
    CacheFromPretrainedEnvvar.set(True)
    return run_scorecard_single(
        model_id=model_id,
        artifacts_dir=args.artifacts_dir,
        use_cached=JobKind[args.use_cached.upper()] if args.use_cached else None,
        start_at=Stage[args.start_at.upper()],
        end_before=Stage[args.end_before.upper()],
        upload=args.upload_assets,
        publish_numbers=args.publish_numbers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the scorecard for one model in the current environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args(parser)
    args = parser.parse_args()
    init_hub_client(args)

    pytorch_models, static_models = validate_and_split_enabled_models(
        args.models, args.static_models_dir
    )
    if static_models or len(pytorch_models) != 1:
        raise SystemExit(
            f"run_scorecard_single.py runs exactly one PyTorch recipe model, got "
            f"{sorted(pytorch_models | static_models)}."
        )
    if not run_from_args(next(iter(pytorch_models)), args):
        sys.exit(1)


if __name__ == "__main__":
    main()

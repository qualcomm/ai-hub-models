# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import copy
import datetime
import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import pandas as pd
import ruamel.yaml

from qai_hub_models import Precision
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.devices_and_chipsets_yaml import load_similar_devices
from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    BranchEnvvar,
    DateFormatEnvvar,
    DeploymentEnvvar,
    EnabledModelsEnvvar,
    EnabledPrecisionsEnvvar,
    IgnoreExistingIntermediateJobsDuringCollectionEnvvar,
    SpecialModelSetting,
    SpecialPrecisionSetting,
    StaticModelsDirEnvvar,
)
from qai_hub_models.scorecard.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.results.code_gen import (
    remove_numerics_failures,
    remove_perf_failures,
    update_code_gen_accuracy_failure_reasons,
    update_code_gen_failure_reasons,
    update_model_publish_status,
)
from qai_hub_models.scorecard.results.numerics_diff import NumericsDiff
from qai_hub_models.scorecard.results.performance_diff import PerformanceDiff
from qai_hub_models.scorecard.results.scorecard_summary import (
    ModelTestConfig,
)
from qai_hub_models.scorecard.results.spreadsheet import ResultsSpreadsheet
from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    ComponentNamesYaml,
    GraphNamesYaml,
    InferenceScorecardJobYaml,
    LinkScorecardJobYaml,
    PreQDQCompileScorecardJobYaml,
    ProfileScorecardJobYaml,
    QuantizeScorecardJobYaml,
    ToolVersionsByPathYaml,
    get_model_component_and_graph_names,
)
from qai_hub_models.scorecard.scorecard_config_yaml import QAIHMModelScorecardConfig
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.scorecard.static.model_config import ScorecardModelConfig
from qai_hub_models.scorecard.utils.numerics_yaml_helpers import (
    create_numerics_yaml,
    get_chipset_registry,
)
from qai_hub_models.scorecard.utils.testing_async_utils import accuracy_row_key
from qai_hub_models.scripts.download_scorecard_results import (
    download_single_artifact,
    find_latest_run,
)
from qai_hub_models.utils.hub_clients import (
    default_hub_client_as,
    deployment_is_prod,
    get_default_hub_deployment,
    get_scorecard_client_or_raise,
    set_default_hub_client,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS

# If the precision is any one of these two values, add it to the branch column
# to allow tableau to differentiate different types of scorecards
SPECIAL_PRECISIONS = ["bench", "default_quantized"]


def drop_names_with_replacements(
    component_names_yaml: ComponentNamesYaml,
    graph_names_yaml: GraphNamesYaml,
    fresh_component_names: ComponentNamesYaml,
    fresh_graph_names: GraphNamesYaml,
    models: Iterable[str],
) -> None:
    """Drop committed component/graph names only where this run has replacements.

    Names are recipe-derived, and nothing regenerates them for a model this run
    did not exercise, so clearing unconditionally loses them permanently. The
    per-model clear is still needed where a replacement exists, so a dropped
    component leaves no stale ``<model>_<component>`` graph-name key.
    """
    for model in models:
        if fresh_component_names.get(model) is not None:
            component_names_yaml.clear(model)
        if fresh_graph_names.has_model(model):
            graph_names_yaml.clear(model)


@dataclass
class _IntermediateState:
    component_names: ComponentNamesYaml
    graph_names: GraphNamesYaml
    pre_qdq_jobs: PreQDQCompileScorecardJobYaml
    quantize_jobs: QuantizeScorecardJobYaml
    compile_jobs: CompileScorecardJobYaml
    link_jobs: LinkScorecardJobYaml
    profile_jobs: ProfileScorecardJobYaml
    inference_jobs: InferenceScorecardJobYaml
    # Pre-clear snapshot for the regression reports' "Previous *" columns.
    previous_compile_jobs: CompileScorecardJobYaml
    previous_inference_jobs: InferenceScorecardJobYaml


def _load_intermediate_state(
    using_prod_hub: bool,
    ignore_existing: bool,
    model_list: list[str],
    all_models: bool,
) -> _IntermediateState:
    if not using_prod_hub:
        # Previous scorecard state is applicable only on prod, but the committed
        # prod job IDs still back the "Previous * (prod)" columns on dev runs.
        return _IntermediateState(
            ComponentNamesYaml(),
            GraphNamesYaml(),
            PreQDQCompileScorecardJobYaml(),
            QuantizeScorecardJobYaml(),
            CompileScorecardJobYaml(),
            LinkScorecardJobYaml(),
            ProfileScorecardJobYaml(),
            InferenceScorecardJobYaml(),
            CompileScorecardJobYaml.from_intermediates(),
            InferenceScorecardJobYaml.from_intermediates(),
        )

    component_names_yaml = ComponentNamesYaml.from_intermediates()
    graph_names_yaml = GraphNamesYaml.from_intermediates()
    pre_qdq_job_yamls = PreQDQCompileScorecardJobYaml.from_intermediates()
    quantize_job_yamls = QuantizeScorecardJobYaml.from_intermediates()
    compile_job_yamls = CompileScorecardJobYaml.from_intermediates()
    link_job_yamls = LinkScorecardJobYaml.from_intermediates()
    profile_job_yamls = ProfileScorecardJobYaml.from_intermediates()
    inference_job_yamls = InferenceScorecardJobYaml.from_intermediates()

    # Snapshot before the ignore-existing clear below; CI always sets it, and
    # snapshotting after left every "Previous *" job ID empty (tetracode #21471).
    previous_compile_jobs = CompileScorecardJobYaml(dict(compile_job_yamls.mapping))
    previous_inference_jobs = InferenceScorecardJobYaml(
        dict(inference_job_yamls.mapping)
    )

    # Erase jobs for models we're collecting results for, if applicable
    if ignore_existing:
        # Fresh names needed to build the scoped export-params list.
        fresh_component_names = ComponentNamesYaml.from_test_artifacts()
        fresh_graph_names = GraphNamesYaml.from_test_artifacts()

        drop_names_with_replacements(
            component_names_yaml,
            graph_names_yaml,
            fresh_component_names,
            fresh_graph_names,
            model_list,
        )

        if all_models:
            pre_qdq_job_yamls.clear()
            quantize_job_yamls.clear()
            compile_job_yamls.clear()
            link_job_yamls.clear()
            profile_job_yamls.clear()
            inference_job_yamls.clear()
        else:
            for model in model_list:
                # Job yamls: scoped drop preserves out-of-scope committed IDs.
                try:
                    manifest = QAIHMModelManifest.from_model(model)
                    scope_params = _resolve_test_params(
                        manifest, fresh_component_names, fresh_graph_names
                    ).get_all_export_params()
                except Exception:
                    # Non-recipe models fall back to model-wide clear.
                    pre_qdq_job_yamls.clear(model)
                    quantize_job_yamls.clear(model)
                    compile_job_yamls.clear(model)
                    link_job_yamls.clear(model)
                    profile_job_yamls.clear(model)
                    inference_job_yamls.clear(model)
                    continue
                pre_qdq_job_yamls.drop_in_scope(scope_params)
                quantize_job_yamls.drop_in_scope(scope_params)
                compile_job_yamls.drop_in_scope(scope_params)
                link_job_yamls.drop_in_scope(scope_params)
                profile_job_yamls.drop_in_scope(scope_params)
                inference_job_yamls.drop_in_scope(scope_params)

    return _IntermediateState(
        component_names_yaml,
        graph_names_yaml,
        pre_qdq_job_yamls,
        quantize_job_yamls,
        compile_job_yamls,
        link_job_yamls,
        profile_job_yamls,
        inference_job_yamls,
        previous_compile_jobs,
        previous_inference_jobs,
    )


def _resolve_test_params(
    manifest: QAIHMModelManifest,
    component_names_yaml: ComponentNamesYaml,
    graph_names_yaml: GraphNamesYaml,
) -> ModelTestConfig:
    """Build the ModelTestConfig for a recipe model."""
    if manifest.id is None:
        raise ValueError("Cannot resolve test params for a manifest with no id.")
    component_names, graph_names, component_graph_names = (
        get_model_component_and_graph_names(
            manifest.id, component_names_yaml, graph_names_yaml
        )
    )
    return ModelTestConfig.from_recipe_model(
        manifest, component_names, graph_names, component_graph_names
    )


def _scope_from_test_config(
    test_params: ModelTestConfig,
) -> set[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """The (precision, path, device) tuples this run was configured to measure."""
    return {(prec, path, device) for prec, path, device in test_params.profile_tests}


def _perf_components_owned_here(sc: QAIHMModelScorecardConfig) -> set[str] | None:
    """
    perf.yaml component names this writer may modify, or None for all of them.

    A hybrid LLM's perf.yaml is co-owned: apply_llm_perf_updates writes the consolidated
    backbone entry from QDC measurements, and this writer contributes the standalone
    components it profiled through Workbench. Restricting the scoped drop to our own
    components keeps each writer from deleting the other's results.
    """
    if not sc.is_llm or not sc.standalone_components:
        return None
    return set(sc.standalone_components.values())


def _accuracy_scope_from_test_config(
    model_id: str, test_params: ModelTestConfig
) -> set[tuple[str, str, str, str]]:
    """accuracy.csv row keys for the tuples this run was configured to measure."""
    return {
        accuracy_row_key(model_id, device.chipset, precision, path)
        for precision, path, device in test_params.profile_tests
    }


def _drop_accuracy_rows_in_scope(
    df: pd.DataFrame, in_scope_keys: set[tuple[str, str, str, str]]
) -> pd.DataFrame:
    """Drop rows whose (model_id, chipset, precision, runtime) key is in scope."""
    if not in_scope_keys:
        return df
    key_index = pd.MultiIndex.from_tuples(
        in_scope_keys, names=["model_id", "chipset", "precision", "runtime"]
    )
    old_index = pd.MultiIndex.from_frame(
        df[["model_id", "chipset", "precision", "runtime"]]
    )
    return df[~old_index.isin(key_index)]


def _merge_numerics(
    committed: QAIHMModelNumerics, fresh: QAIHMModelNumerics
) -> QAIHMModelNumerics:
    """Merge fresh (this run's) numerics into the committed (pre-scope-drop) struct.

    Metrics are keyed by (dataset_name, metric_name, metric_unit). Fresh wins
    at the (device, precision, path) grain; committed entries the fresh metric
    doesn't cover are folded in; committed-only metrics are preserved.
    """

    def key(m: QAIHMModelNumerics.MetricDetails) -> tuple[str, str, str]:
        return (m.dataset_name, m.metric_name, m.metric_unit)

    committed_by_key = {key(m): m for m in committed.metrics}
    merged: list[QAIHMModelNumerics.MetricDetails] = []
    seen: set[tuple[str, str, str]] = set()
    for fresh_metric in fresh.metrics:
        k = key(fresh_metric)
        seen.add(k)
        prev = committed_by_key.get(k)
        if prev is not None:
            # Fold at (device, precision, path) so a fresh float/onnx on a
            # device doesn't shadow the committed float/qnn_dlc, float/tflite.
            for device, prev_prec_dict in prev.device_metric.items():
                fresh_prec_dict = fresh_metric.device_metric.setdefault(device, {})
                for precision, prev_path_dict in prev_prec_dict.items():
                    fresh_path_dict = fresh_prec_dict.setdefault(precision, {})
                    for path, details in prev_path_dict.items():
                        fresh_path_dict.setdefault(path, details)
        merged.append(fresh_metric)
    for k, prev in committed_by_key.items():
        if k not in seen:
            merged.append(prev)
    return QAIHMModelNumerics(metrics=merged)


def read_jobs_config(config_path: str) -> dict:
    """Read yaml files."""
    yaml = ruamel.yaml.YAML()
    with open(config_path) as file:
        return yaml.load(file)


def _merge_existing_accuracy_data(
    new_df: pd.DataFrame,
    in_scope_keys: set[tuple[str, str, str, str]],
) -> pd.DataFrame:
    """Merge this run's accuracy rows on top of the committed CSV, dropping in-scope keys."""
    intermediates_path = ScorecardArtifact.ACCURACY_CSV.intermediates_path
    if not os.path.exists(intermediates_path):
        return new_df
    old_df = _drop_accuracy_rows_in_scope(
        pd.read_csv(intermediates_path), in_scope_keys
    )
    return pd.concat([old_df, new_df])


def remove_failed_jobs(config: dict) -> None:
    """
    Failed jobs need to be in the config to get their job ids for summary but
    we want to delete them before writing to perf.yaml
    """
    for model_config in config["models"]:
        for perf_metrics in model_config["performance_metrics"]:
            for key in list(perf_metrics.keys()):
                if (
                    "job_id" in perf_metrics[key]
                    and perf_metrics[key]["inference_time"] == "null"
                ):
                    del perf_metrics[key]


def _load_previous_tool_versions(deployment: str) -> ToolVersionsByPathYaml:
    """Load the previous run's tool-versions.yaml for a per-deployment diff.

    Sourcing baseline:
      1. Most recent completed run of the same deployment in the S3 history
         (dev diffs against dev, prod against prod).
      2. Checked-in intermediates as a fallback — used the first time a new
         deployment shows up, and on test branches that can't reach S3.

    Without the S3 baseline the diff falls back to intermediates, which only
    refreshes when a prod scorecard PR merges to main — so a dev run's
    "previous" would silently stay pinned to the last merged prod run.
    """
    try:
        previous = find_latest_run(deployment)
    except Exception:
        logging.warning(
            "Could not query scorecard S3 history; falling back to checked-in "
            "intermediates for the toolchain diff baseline.",
            exc_info=True,
        )
        previous = None

    if previous is not None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "tool-versions.yaml"
            downloaded = download_single_artifact(previous, "tool-versions.yaml", dest)
            if downloaded is not None:
                print(
                    f"Loaded previous {deployment} tool-versions from S3 "
                    f"run {previous.run_id} (baseline for toolchain diff)."
                )
                return ToolVersionsByPathYaml.from_yaml(
                    downloaded, create_empty_if_no_file=True
                )
        print(
            f"Previous {deployment} run {previous.run_id} has no "
            f"tool-versions.yaml in S3 (likely uploaded before the artifact "
            f"was added). Falling back to intermediates."
        )
    else:
        print(
            f"No previous {deployment} run found in scorecard S3 history; "
            f"falling back to intermediates for the toolchain diff baseline."
        )

    return ToolVersionsByPathYaml.from_yaml(
        ScorecardArtifact.TOOL_VERSIONS.intermediates_path,
        create_empty_if_no_file=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    EnabledModelsEnvvar.add_arg(parser)
    IgnoreExistingIntermediateJobsDuringCollectionEnvvar.add_arg(parser)
    StaticModelsDirEnvvar.add_arg(parser)
    parser.add_argument(
        "--gen-csv",
        action="store_true",
        help="Generate a csv that summarizes the profile and compile steps.",
    )
    parser.add_argument(
        "--gen-perf-summary",
        action="store_true",
        help="Generate a summary of the performance per model, and update perf yaml files.",
    )
    parser.add_argument(
        "--sync-code-gen",
        action="store_true",
        help="Sync code generation YAML with failures & successes in the scorecard YAML. If compile job fails for any device, that path is skipped. If the profile job for the default export device fails, that path is skipped.",
    )
    DeploymentEnvvar.add_arg(parser, default=get_default_hub_deployment())
    EnabledPrecisionsEnvvar.add_arg(parser)
    BranchEnvvar.add_arg(parser)
    DateFormatEnvvar.add_arg_group(parser)
    ArtifactsDirEnvvar.add_arg(parser)
    parser.add_argument(
        "--accuracy-csv-path",
        type=str,
        default=str(ScorecardArtifact.ACCURACY_CSV.path),
        help="Accuracy CSV input. When present and non-empty, drives numerics-yaml updates.",
    )
    return parser.parse_args()


def process_model(
    model_id: str,
    deployment: str,
    static_models_dir: Path,
    component_names_yaml: ComponentNamesYaml,
    graph_names_yaml: GraphNamesYaml,
    pre_qdq_job_yamls: PreQDQCompileScorecardJobYaml,
    quantize_job_yamls: QuantizeScorecardJobYaml,
    compile_job_yamls: CompileScorecardJobYaml,
    link_job_yamls: LinkScorecardJobYaml,
    profile_job_yamls: ProfileScorecardJobYaml,
    inference_job_yamls: InferenceScorecardJobYaml,
    gen_csv: bool,
    sync_code_gen: bool,
    gen_perf_summary: bool,
    write_model_card: bool,
) -> tuple[ResultsSpreadsheet, QAIHMModelPerf | None, QAIHMModelPerf | None] | None:
    """
    Process results for a single model.

    Returns ``None`` on per-model failure (logged to stderr) so one bad model
    doesn't take down the whole multiprocessing batch.

    Parameters
    ----------
    model_id
        Model identifier.
    deployment
        Deployment environment.
    static_models_dir
        Directory containing static model configurations.
    component_names_yaml
        YAML containing component names for each model.
    graph_names_yaml
        YAML containing graph names for each model component.
    pre_qdq_job_yamls
        YAML containing pre qdq compile job information.
    quantize_job_yamls
        YAML containing quantize job information.
    compile_job_yamls
        YAML containing compile job information.
    link_job_yamls
        YAML containing link job information.
    profile_job_yamls
        YAML containing profile job information.
    inference_job_yamls
        YAML containing inference job information.
    gen_csv
        Whether to generate CSV spreadsheet.
    sync_code_gen
        Whether to sync code generation.
    gen_perf_summary
        Whether to generate performance summary.
    write_model_card
        Whether to write model card.

    Returns
    -------
    result : tuple[ResultsSpreadsheet, QAIHMModelPerf | None, QAIHMModelPerf | None] | None
        ``None`` if processing this model raised an exception. Otherwise a
        3-tuple of:

        * spreadsheet — model spreadsheet, or empty spreadsheet if not gen_csv.
        * previous_perf — previous (on disk) perf yaml, or None if not
          gen_perf_summary. Always None for static models.
        * current_perf — current (from this scorecard) perf yaml, or None if
          not gen_perf_summary. Always None for static models.
    """
    try:
        if model_id in MODEL_IDS:
            # This model has an end to end pyTorch recipe.
            return process_e2e_recipe_model(
                model_id,
                component_names_yaml,
                graph_names_yaml,
                pre_qdq_job_yamls,
                quantize_job_yamls,
                compile_job_yamls,
                link_job_yamls,
                profile_job_yamls,
                inference_job_yamls,
                gen_csv,
                sync_code_gen,
                gen_perf_summary,
                write_model_card,
            )
        # This model was uploaded statically (as a single file).
        if gen_csv:
            spreadsheet = process_static_file_model(
                model_id,
                deployment,
                static_models_dir,
                compile_job_yamls,
                link_job_yamls,
                profile_job_yamls,
                inference_job_yamls,
            )
        else:
            spreadsheet = ResultsSpreadsheet()
        return (spreadsheet, None, None)
    except Exception:
        # Skip this model so one bad input doesn't kill the multiprocessing
        # pool; the aggregate raise at the end of __main__ still fails the
        # job after assets land.
        print(
            f"{model_id} result processing failed:\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return None


def _get_pytorch_tags(manifest: QAIHMModelManifest) -> list[str]:
    assert manifest.status is not MODEL_STATUS.UNSET
    tags = [tag.value for tag in manifest.tags]
    tags.append("pytorch")
    tags.append(manifest.status.value)
    return tags


def _get_static_tags(model_info: ScorecardModelConfig) -> list[str]:
    tags = [tag.name for tag in model_info.tags]
    tags.append("static")
    tags.append("private")
    tags.append(f"bu-{model_info.bu_owner.value}")
    return tags


def process_e2e_recipe_model(
    model_id: str,
    component_names_yaml: ComponentNamesYaml,
    graph_names_yaml: GraphNamesYaml,
    pre_qdq_job_yamls: PreQDQCompileScorecardJobYaml,
    quantize_job_yamls: QuantizeScorecardJobYaml,
    compile_job_yamls: CompileScorecardJobYaml,
    link_job_yamls: LinkScorecardJobYaml,
    profile_job_yamls: ProfileScorecardJobYaml,
    inference_job_yamls: InferenceScorecardJobYaml,
    gen_csv: bool,
    sync_code_gen: bool,
    gen_perf_summary: bool,
    write_model_card: bool,
) -> tuple[ResultsSpreadsheet, QAIHMModelPerf | None, QAIHMModelPerf | None]:
    """
    Process results for a model with an end-to-end pyTorch recipe.

    Parameters
    ----------
    model_id
        Model identifier.
    component_names_yaml
        YAML containing component names for each model.
    graph_names_yaml
        YAML containing graph names for each model component.
    pre_qdq_job_yamls
        YAML containing pre qdq compile job information.
    quantize_job_yamls
        YAML containing quantize job information.
    compile_job_yamls
        YAML containing compile job information.
    link_job_yamls
        YAML containing link job information.
    profile_job_yamls
        YAML containing profile job information.
    inference_job_yamls
        YAML containing inference job information.
    gen_csv
        Whether to generate CSV spreadsheet.
    sync_code_gen
        Whether to sync code generation.
    gen_perf_summary
        Whether to generate performance summary.
    write_model_card
        Whether to write model card.

    Returns
    -------
    spreadsheet : ResultsSpreadsheet
        Model spreadsheet, or empty spreadsheet if not gen_csv.
    previous_perf : QAIHMModelPerf | None
        Previous (on disk) perf yaml, or None if not gen_perf_summary.
    current_perf : QAIHMModelPerf | None
        Current (from this scorecard) perf yaml, or None if not gen_perf_summary.
    """

    def print_with_id(pstr: str) -> None:
        print(f"{model_id} | {pstr}")

    # Load configs
    manifest = QAIHMModelManifest.from_model(model_id)
    sc = QAIHMModelScorecardConfig.from_model(model_id)

    # Skip certain models
    if manifest.is_precompiled or sc.skip_hub_tests_and_scorecard or sc.skip_scorecard:
        return ResultsSpreadsheet(), None, None

    # Get enabled test paths for this model
    test_params = _resolve_test_params(manifest, component_names_yaml, graph_names_yaml)

    # Get summaries for this model and its components.
    print_with_id("Loading summary")
    summaries = test_params.get_all_export_test_summaries(
        pre_qdq_job_yamls,
        quantize_job_yamls,
        compile_job_yamls,
        link_job_yamls,
        profile_job_yamls,
        inference_job_yamls,
    )

    assert manifest.domain is not None
    assert manifest.use_case is not None
    entries: ResultsSpreadsheet = ResultsSpreadsheet()
    entries.set_model_metadata(
        model_id,
        manifest.domain,
        manifest.use_case,
        _get_pytorch_tags(manifest),
        known_failure_reasons=manifest.disabled_paths,
        default_quantized_precision=manifest.default_quantized_precision,
        default_device=ScorecardDevice.get(manifest.default_device),
    )
    if gen_csv:
        print_with_id("Adding to Spreadsheet")
        for export_test_summary in summaries:
            entries.append_export_test_summary(export_test_summary)

    if sync_code_gen and not sc.freeze_perf_yaml and not sc.is_llm:
        # Enable or disable runtimes on this model depending on whether the default device has passing jobs
        update_code_gen_failure_reasons(summaries, test_params.enabled_paths, manifest)
        manifest_path = manifest.to_model_yaml()
        print_with_id(f"Updated Runtime Failure Reasons in {manifest_path}")

        # Update model status & reason, if applicable
        if update_model_publish_status(manifest):
            manifest_path = manifest.to_model_yaml()
            print_with_id(pstr=f"Updated publish status at {manifest_path}")

    model_card = QAIHMModelPerf()
    prev_model_card = QAIHMModelPerf()
    if gen_perf_summary:
        print_with_id("Writing Performance YAML")

        # Build model card
        model_card = QAIHMModelPerf()
        model_card_without_failures = QAIHMModelPerf() if write_model_card else None
        for summary in summaries:
            summary.add_to_perf(model_card, include_failures=True)
            if model_card_without_failures and summary.params.path.is_published:
                summary.add_to_perf(model_card_without_failures, include_failures=False)

        # Load old model card and write new model card. Skip the write when this shard
        # produced no published summaries -- otherwise the scoped merge deletes committed entries.
        prev_model_card = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
        # A hybrid LLM's standalone components are profiled through Workbench, so this
        # writer owns their entries while apply_llm_perf_updates owns the backbone entry.
        owned_components = _perf_components_owned_here(sc)
        if (
            not sc.freeze_perf_yaml
            and (not sc.is_llm or owned_components)
            and model_card_without_failures
            and not model_card_without_failures.empty
        ):
            # Scoped merge: drop in-scope (precision, path, device) tuples
            # from the committed card, then upsert this run's results.
            merged = copy.deepcopy(prev_model_card)
            merged.drop_entries_in_scope(
                _scope_from_test_config(test_params), only_components=owned_components
            )
            for summary in summaries:
                params = summary.params
                assert params.precision is not None and params.device is not None
                # Jobs can pass on some devices for a path disabled by failures elsewhere.
                if params.path.is_published and manifest.is_supported(
                    params.precision,
                    params.path.runtime,
                    device=params.device.device_name,
                ):
                    summary.add_to_perf(merged, include_failures=False)
            # The merge inherits supported_chipsets from the committed card and
            # only ever unions onto it, so a chipset the scorecard never ran is
            # never dropped. Skip a co-owned card: it would lose the chipsets
            # the other writer's components establish.
            if sync_code_gen and owned_components is None:
                merged.drop_chipsets_without_assets(model_id)
            merged.apply_similar_devices(load_similar_devices())
            card_path = merged.to_model_yaml(model_id)
            print_with_id(f"Wrote {card_path}")

    return entries, prev_model_card, model_card


def process_static_file_model(
    model_id: str,
    deployment: str,
    models_dir: Path,
    compile_job_yamls: CompileScorecardJobYaml | None,
    link_job_yamls: LinkScorecardJobYaml | None,
    profile_job_yamls: ProfileScorecardJobYaml | None,
    inference_job_yamls: InferenceScorecardJobYaml | None,
) -> ResultsSpreadsheet:
    """
    Process results for a static model (uploaded onnx or traced pyTorch file).

    Returns model spreadsheet.
    """

    def print_with_id(pstr: str) -> None:
        print(f"{model_id} | {pstr}")

    # Load config
    model_info = ScorecardModelConfig.from_yaml(models_dir / (model_id + ".yaml"))
    test_params = ModelTestConfig.from_static_model(model_info)

    # Get summaries for this model and its components.
    with default_hub_client_as(
        get_scorecard_client_or_raise(deployment, model_info.restrict_access)
    ):
        summaries = test_params.get_all_export_test_summaries(
            None,
            None,
            compile_job_yamls,
            link_job_yamls,
            profile_job_yamls,
            inference_job_yamls,
        )

        print_with_id("Adding to Spreadsheet")
        entries = ResultsSpreadsheet()
        entries.set_model_metadata(
            model_id,
            model_info.domain,
            model_info.use_case,
            _get_static_tags(model_info),
            default_quantized_precision=None,
            default_device=model_info.devices[0],
        )
        for export_test_summary in summaries:
            entries.append_export_test_summary(export_test_summary)

        return entries


if __name__ == "__main__":
    args = parse_args()
    static_model_dir: Path = args.static_models_dir

    # Verify args are compatible with the chosen deployment.
    using_prod_hub = deployment_is_prod(args.deployment)
    if not using_prod_hub and args.sync_code_gen:
        print("Warning: Can't sync code gen if deployment is not prod.")
        args.sync_code_gen = False

    os.makedirs(args.artifacts_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # List of models for which to generate perf.
    pytorch_models, static_models = validate_and_split_enabled_models(
        args.models, static_model_dir
    )
    all_models = SpecialModelSetting.ALL in args.models
    model_list = sorted(pytorch_models.union(static_models))

    # Load datestr
    date = DateFormatEnvvar.parse(args.date, args.date_format)

    # Set client to use target deployment
    set_default_hub_client(get_scorecard_client_or_raise(args.deployment))

    # Load Base YAMLs
    state = _load_intermediate_state(
        using_prod_hub,
        args.ignore_existing_intermediate_jobs,
        model_list,
        all_models,
    )
    component_names_yaml = state.component_names
    graph_names_yaml = state.graph_names
    pre_qdq_job_yamls = state.pre_qdq_jobs
    quantize_job_yamls = state.quantize_jobs
    compile_job_yamls = state.compile_jobs
    link_job_yamls = state.link_jobs
    profile_job_yamls = state.profile_jobs
    inference_job_yamls = state.inference_jobs
    previous_compile_jobs = state.previous_compile_jobs
    previous_inference_jobs = state.previous_inference_jobs

    # Append job results from test artifacts
    component_names_yaml.mapping.update(
        ComponentNamesYaml.from_test_artifacts().mapping
    )
    graph_names_yaml.mapping.update(GraphNamesYaml.from_test_artifacts().mapping)
    pre_qdq_job_yamls.update(PreQDQCompileScorecardJobYaml.from_test_artifacts())
    quantize_job_yamls.update(QuantizeScorecardJobYaml.from_test_artifacts())
    compile_job_yamls.update(CompileScorecardJobYaml.from_test_artifacts())
    link_job_yamls.update(LinkScorecardJobYaml.from_test_artifacts())
    profile_job_yamls.update(ProfileScorecardJobYaml.from_test_artifacts())
    inference_job_yamls.update(InferenceScorecardJobYaml.from_test_artifacts())
    current_compile_jobs = CompileScorecardJobYaml.from_test_artifacts()
    current_inference_jobs = InferenceScorecardJobYaml.from_test_artifacts()

    # Extract Data from Models
    if len(model_list) > 1:
        # Use multiprocessing for multiple models because getting jobs from Hub is slow
        pool = multiprocessing.Pool(processes=15)
        model_summaries = pool.starmap(
            process_model,
            zip(
                model_list,
                cycle([args.deployment]),
                cycle([static_model_dir]),
                cycle([component_names_yaml]),
                cycle([graph_names_yaml]),
                cycle([pre_qdq_job_yamls]),
                cycle([quantize_job_yamls]),
                cycle([compile_job_yamls]),
                cycle([link_job_yamls]),
                cycle([profile_job_yamls]),
                cycle([inference_job_yamls]),
                cycle([args.gen_csv]),
                cycle([args.sync_code_gen]),
                cycle([args.gen_perf_summary]),
                cycle([using_prod_hub]),
            ),
        )
        pool.close()
        # join() ensures worker stderr is fully flushed before we print the
        # failure summary at the end of __main__.
        pool.join()
    else:
        # Single model option for that allows breakpoints
        model_summaries = [
            process_model(
                model_list[0],
                args.deployment,
                static_model_dir,
                component_names_yaml,
                graph_names_yaml,
                pre_qdq_job_yamls,
                quantize_job_yamls,
                compile_job_yamls,
                link_job_yamls,
                profile_job_yamls,
                inference_job_yamls,
                args.gen_csv,
                args.sync_code_gen,
                args.gen_perf_summary,
                using_prod_hub,
            )
        ]

    perf_report: PerformanceDiff | None = None
    if args.gen_perf_summary:
        perf_report = PerformanceDiff(
            current_compile_jobs=current_compile_jobs,
            previous_compile_jobs=previous_compile_jobs,
        )
    spreadsheet = ResultsSpreadsheet() if args.gen_csv else None
    if spreadsheet is not None:
        spreadsheet.set_date(date)
        # Tableau wants to differentiate between different types of scorecards
        # So mark them as such in the branch column.
        branch = args.branch
        precisions = args.precisions
        for precision in precisions:
            if isinstance(precision, SpecialPrecisionSetting):
                branch += f" - {precision.value}"
        spreadsheet.set_branch(branch)

    # Numerics setup. Skipped (no-op) when the accuracy CSV is missing/empty —
    # this is the perf-only run path.
    accuracy_path = Path(args.accuracy_csv_path)
    accuracy_csv_present = accuracy_path.exists() and accuracy_path.stat().st_size > 0
    accuracy_df = pd.read_csv(accuracy_path) if accuracy_csv_present else None
    chipset_registry = get_chipset_registry() if accuracy_csv_present else None
    global_numerics_diff: NumericsDiff | None = (
        NumericsDiff(
            current_inference_jobs=current_inference_jobs,
            previous_inference_jobs=previous_inference_jobs,
        )
        if accuracy_csv_present
        else None
    )

    failed_model_ids: list[str] = []
    # Row keys this run measured; scopes the CSV drop below.
    accuracy_scope_keys: set[tuple[str, str, str, str]] = set()
    for model_id, model_summary in zip(model_list, model_summaries, strict=False):
        if model_summary is None:
            failed_model_ids.append(model_id)
            continue
        model_spreadsheet, prev_model_card, curr_model_card = model_summary

        # Combine model spreadsheet with group spreadsheet
        if spreadsheet is not None:
            spreadsheet.combine(model_spreadsheet)

        # Update performance report with model card diff
        if perf_report is not None:
            # Summary is made between the existing perf.yaml and the newly
            # created model card.
            perf_report.update_summary(
                model_id,
                previous_report=prev_model_card,
                new_report=curr_model_card,
            )

        # Numerics is pytorch-only; static models don't have inference jobs.
        if not (accuracy_csv_present and model_id in pytorch_models):
            continue
        assert accuracy_df is not None
        assert chipset_registry is not None
        assert global_numerics_diff is not None
        try:
            manifest = QAIHMModelManifest.from_model(model_id)
            sc = QAIHMModelScorecardConfig.from_model(model_id)
            if (
                sc.skip_hub_tests_and_scorecard
                or sc.skip_scorecard
                or sc.freeze_perf_yaml
            ):
                continue

            model_diff = NumericsDiff(
                current_inference_jobs=current_inference_jobs,
                previous_inference_jobs=previous_inference_jobs,
            )
            numerics = create_numerics_yaml(
                model_id,
                accuracy_df,
                chipset_registry,
                model_diff,
                benchmark=manifest.numerics_benchmark,
                threshold_override=sc.numerics_threshold_override,
            )
            global_numerics_diff.merge_from(model_diff)

            # Scoped merge into the committed numerics.yaml.
            test_params = _resolve_test_params(
                manifest, component_names_yaml, graph_names_yaml
            )
            numerics_scope = _scope_from_test_config(test_params)
            accuracy_scope_keys |= _accuracy_scope_from_test_config(
                model_id, test_params
            )
            committed_numerics = QAIHMModelNumerics.from_model(
                model_id, not_exists_ok=True
            )
            if committed_numerics is not None:
                committed_numerics.drop_entries_in_scope(numerics_scope)

            if numerics is None:
                # Nothing measured; write the committed remainder (may be empty).
                out = committed_numerics or QAIHMModelNumerics()
                out.to_model_yaml(model_id)
                continue

            if committed_numerics is not None and not committed_numerics.is_empty():
                numerics = _merge_numerics(committed_numerics, numerics)

            if numerics.metrics:
                # Update failure reasons according to what NumericsDiff says is
                # above the acceptable accuracy threshold.
                update_code_gen_accuracy_failure_reasons(model_id, manifest, model_diff)

                # Update numerics.yaml to remove failing paths
                numerics = remove_numerics_failures(numerics, manifest.disabled_paths)

                if args.sync_code_gen and using_prod_hub:
                    # If sync-code-gen is on, save the updated failure reasons to disk.
                    manifest.to_model_yaml()

                    # Do not remove failing paths if frozen or LLM
                    # LLMs because it is handled by apply_llm_perf_updates.
                    if not sc.freeze_perf_yaml and not sc.is_llm:
                        perf = remove_perf_failures(
                            perf=QAIHMModelPerf.from_model(
                                model_id, not_exists_ok=True
                            ),
                            failure_reason=manifest.disabled_paths,
                        )
                        perf.apply_similar_devices(load_similar_devices())
                        perf.to_model_yaml(model_id)

                    # Un-publish or re-publish the model if needed by updating manifest.yaml.
                    if update_model_publish_status(manifest):
                        manifest.to_model_yaml()

            numerics.to_model_yaml(model_id)
            print(f"{model_id} numerics update complete")
        except Exception:
            # Skip this model so one bad input doesn't kill the batch; the
            # aggregate raise at the end of __main__ still fails the job
            # after assets land.
            print(
                f"{model_id} numerics update failed:\n{traceback.format_exc()}",
                file=sys.stderr,
            )
            failed_model_ids.append(model_id)

    # Write spreadsheet to disk
    if spreadsheet is not None:
        summary_path = os.path.join(args.artifacts_dir, "export-summary.csv")
        spreadsheet.to_csv(summary_path)
        print(f"Spreadsheet written to {os.path.realpath(summary_path)}")

    # Write performance summary to disk
    if perf_report is not None:
        report_path = os.path.join(
            args.artifacts_dir, f"performance-summary-{now_str}.txt"
        )
        current_tool_versions = ToolVersionsByPathYaml.from_yaml(
            ScorecardArtifact.TOOL_VERSIONS.path,
            create_empty_if_no_file=True,
        )
        previous_tool_versions = _load_previous_tool_versions(args.deployment)
        toolchain_changes = current_tool_versions.diff(previous_tool_versions)
        perf_report.dump_summary(report_path, toolchain_changes=toolchain_changes)

        regressions_path = os.path.join(
            args.artifacts_dir, f"perf-regressions-2x-{now_str}.json"
        )
        perf_report.dump_severe_regressions_json(regressions_path)

    # Write numerics summary to disk
    if global_numerics_diff is not None:
        numerics_summary_path = os.path.join(
            args.artifacts_dir, f"numerics-summary-{now_str}.txt"
        )
        global_numerics_diff.dump_summary(numerics_summary_path)

        numerics_regressions_path = os.path.join(
            args.artifacts_dir, f"numerics-regressions-{now_str}.json"
        )
        global_numerics_diff.dump_regressions_json(numerics_regressions_path)

        newly_disabled_path = os.path.join(
            args.artifacts_dir, f"newly-disabled-{now_str}.json"
        )
        global_numerics_diff.dump_newly_disabled_json(newly_disabled_path)

        quantized_failures_path = os.path.join(
            args.artifacts_dir, f"quantized-failures-{now_str}.json"
        )
        global_numerics_diff.dump_quantized_failures_json(quantized_failures_path)

        # Write accuracy to intermediates folder, scoped-merged onto the committed CSV.
        if args.sync_code_gen and using_prod_hub:
            assert accuracy_df is not None
            accuracy_df = _merge_existing_accuracy_data(
                accuracy_df, accuracy_scope_keys
            )
            accuracy_df.to_csv(
                ScorecardArtifact.ACCURACY_CSV.intermediates_path, index=False
            )
    else:
        print("No accuracy CSV found. Skipping numerics-yaml updates.")

    # Write jobs and environment to intermediates folder.
    if using_prod_hub:
        component_names_yaml.to_file()
        graph_names_yaml.to_file()
        quantize_job_yamls.to_file()
        compile_job_yamls.to_file()
        link_job_yamls.to_file()
        profile_job_yamls.to_file()
        inference_job_yamls.to_file()
        print(f"Component Names written to {component_names_yaml.path}")
        print(f"Graph Names written to {graph_names_yaml.path}")
        print(f"Quantize Job IDs written to {quantize_job_yamls.path}")
        print(f"Compile Job IDs written to {compile_job_yamls.path}")
        print(f"Link Job IDs written to {link_job_yamls.path}")
        print(f"Profile Job IDs written to {profile_job_yamls.path}")
        print(f"Inference Job IDs written to {inference_job_yamls.path}")

        try:
            shutil.copy(
                ScorecardArtifact.TOOL_VERSIONS.path,
                ScorecardArtifact.TOOL_VERSIONS.intermediates_path,
            )
            print(
                f"Tool versions written to {ScorecardArtifact.TOOL_VERSIONS.intermediates_path}"
            )
        except (shutil.SameFileError, FileNotFoundError):
            pass

        try:
            shutil.copy(
                ScorecardArtifact.ENVIRONMENT_FILE.path,
                ScorecardArtifact.ENVIRONMENT_FILE.intermediates_path,
            )
            print(
                f"Test envvars written to {ScorecardArtifact.ENVIRONMENT_FILE.intermediates_path}"
            )
        except (shutil.SameFileError, FileNotFoundError):
            pass

    # Fail loudly only AFTER all assets are on disk for downstream uploads.
    if failed_model_ids:
        raise RuntimeError(
            f"{len(failed_model_ids)} model(s) failed during result "
            f"collection (assets were still written; see stderr above for "
            f"per-model tracebacks): {', '.join(failed_model_ids)}"
        )

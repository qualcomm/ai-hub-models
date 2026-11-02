# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from argparse import ArgumentDefaultsHelpFormatter
from datetime import date, datetime, time
from io import StringIO
from pathlib import Path

from qai_hub.client import JobType

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.envvars import (
    EnabledDevicesEnvvar,
    EnabledModelsEnvvar,
    EnabledPathsEnvvar,
    EnabledPrecisionsEnvvar,
    IgnoreKnownFailuresEnvvar,
    SpecialModelSetting,
    StaticModelsDirEnvvar,
)
from qai_hub_models.scorecard.execution_helpers import (
    get_evaluation_parameterized_pytest_config,
    get_model_test_parameterizations,
)
from qai_hub_models.scorecard.internal_config_yaml import (
    QAIHMModelInternalConfig,
    TestRunnerSplit,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.yaml import ComponentNamesYaml
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.scorecard.static.model_config import ScorecardModelConfig
from qai_hub_models.scorecard.static.model_exec import (
    get_static_model_test_parameterizations,
)
from qai_hub_models.utils.device import FormFactor

# Scorecard limits
SC_DAYTIME_START_TIME = time(17, 0)  # 5:00 PM
SC_DAYTIME_END_TIME = time(23, 15)  # 11:15 PM
MAX_DAYTIME_DEVICE_JOBS = 30
MAX_DAYTIME_AUTO_DEVICE_JOBS = 10


def _extract_codegen_test_options(
    model_id: str,
) -> tuple[
    bool,
    list[str] | None,
    dict[Precision, list[TargetRuntime]],
    dict[Precision, list[TargetRuntime]],
    ScorecardDevice,
    bool,
    bool,
]:
    sc = QAIHMModelInternalConfig.from_model(model_id)
    manifest = sc.manifest
    return (
        sc.skip_hub_tests_and_scorecard or sc.skip_scorecard,
        (
            ComponentNamesYaml.from_intermediates().get(model_id)
            or ["dummy_single_component"]
        )
        if manifest.is_collection_model
        else None,
        manifest.get_supported_paths_for_testing(),
        manifest.get_supported_paths_for_testing(only_include_passing=True),
        ScorecardDevice.get(manifest.default_device),
        manifest.can_use_quantize_job,
        manifest.requires_aot_prepare,
    )


def get_torch_recipe_profile_parameterizations(
    model_id: str,
) -> list[tuple[str | None, Precision, ScorecardProfilePath, ScorecardDevice]]:
    """
    Get all valid profile parameterizations in the current testing environment for the given PyTorch recipe model ID.
    Each parameterization corresponds to a single profile job.
    """
    (
        skipped,
        component_names,
        test_enabled_precision_runtimes,
        test_passing_precision_runtimes,
        _,
        can_use_quantize_job,
        _,
    ) = _extract_codegen_test_options(model_id)

    if skipped:
        return []

    params = get_model_test_parameterizations(
        model_id=model_id,
        model_supported_test_paths=test_enabled_precision_runtimes,
        model_passing_test_paths=test_passing_precision_runtimes,
        path_type=ScorecardProfilePath,
        can_use_quantize_job=can_use_quantize_job,
        devices=None,
        include_unsupported_paths=None,
    )
    return [(n, *p) for p in params for n in (component_names or [None])]  # type: ignore[list-item]


def get_torch_recipe_inference_parameterizations(
    model_id: str,
) -> list[tuple[str | None, Precision, ScorecardProfilePath, ScorecardDevice]]:
    """
    Get all valid inference parameterizations in the current testing environment for the given PyTorch recipe model ID.
    Each parameterization corresponds to a single inference job.
    """
    (
        skipped,
        component_names,
        test_enabled_precision_runtimes,
        test_passing_precision_runtimes,
        default_device,
        can_use_quantize_job,
        _,
    ) = _extract_codegen_test_options(model_id)

    if skipped:
        return []
    params = get_evaluation_parameterized_pytest_config(
        model_id=model_id,
        device=default_device,
        enabled_test_paths=test_enabled_precision_runtimes,
        passing_test_paths=test_passing_precision_runtimes,
        can_use_quantize_job=can_use_quantize_job,
    )
    return [(n, *p) for p in params for n in (component_names or [None])]  # type: ignore[list-item]


def get_static_model_parameterizations(
    model_id: str, static_models_dir: Path, devices: list[ScorecardDevice]
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """
    Get all valid parameterizations in the current testing environment for the given static recipe model ID.
    Each parameterization corresponds to a single device job.
    """
    info = ScorecardModelConfig.from_scorecard_model_id(model_id, static_models_dir)
    return get_static_model_test_parameterizations(
        model_id,
        JobType.PROFILE,
        ScorecardProfilePath,
        info.precision,
        devices,
        info.enabled_profile_runtimes,
    )


def count_device_jobs(
    models: set[str | SpecialModelSetting] | None = None,
    static_models_dir: Path | None = None,
    run_accuracy_tests: bool = True,
    show_precision_in_summary: bool = False,
    bu_owner: ScorecardModelConfig.BU | None = None,
) -> tuple[
    int,
    dict[ScorecardDevice, int],
    dict[FormFactor, int],
    dict[ScorecardProfilePath, int],
    dict[str, int],
    dict[str, dict[str, int]],
    dict[str, int],
]:
    """
    Count the number of device jobs that would be submitted in the current test environment.

    Parameters
    ----------
    models
        Set of model IDs or special settings.
    static_models_dir
        Path to directory containing static models.
    run_accuracy_tests
        Whether to include accuracy test jobs in count.
    show_precision_in_summary
        Whether to show precision in summary keys.
    bu_owner
        Business unit owner filter.

    Returns
    -------
    total_jobs : int
        Total number of jobs.
    job_count_by_device : dict[ScorecardDevice, int]
        Dictionary mapping device to job count.
    job_count_by_form_factor : dict[FormFactor, int]
        Dictionary mapping form factor to job count.
    job_count_by_scorecard_path : dict[ScorecardProfilePath, int]
        Dictionary mapping scorecard path to job count.
    job_count_by_pytorch_recipe : dict[str, int]
        Dictionary mapping model key to job count.
        If show_precision_in_summary is True, key format is "model_id (precision)".
        Otherwise, key is model_id.
    job_count_by_pytorch_recipe_component : dict[str, dict[str, int]]
        Dictionary mapping model key to component counts.
        dict[key0, dict[key1, count]]
        keys:
            key0: If show_precision_in_summary is True, key0 is "model_id (precision)"
                Otherwise, key0 is "model_id".
            key1: Model component name.
        If a model has no components, it will not appear in this dictionary.
    job_count_by_static_model : dict[str, int]
        Dictionary mapping static model ID to job count.
    """
    # Get models to be tested in this environment.
    enabled_torch_models, enabled_static_models = validate_and_split_enabled_models(
        models, static_models_dir
    )

    # Job count tracking objects
    total_jobs: int = 0
    job_count_by_device: dict[ScorecardDevice, int] = dict.fromkeys(
        ScorecardDevice.all_devices(enabled=True, is_mirror=False), 0
    )
    job_count_by_form_factor: dict[FormFactor, int] = dict.fromkeys(FormFactor, 0)
    del job_count_by_device[cs_universal]  # compile only device
    job_count_by_path: dict[ScorecardProfilePath, int] = {
        p: 0 for p in ScorecardProfilePath if p.enabled
    }
    job_count_by_static_model: dict[str, int] = {}
    job_count_by_pytorch_recipe: dict[str, int] = {}
    job_count_by_pytorch_recipe_component: dict[str, dict[str, int]] = {}

    # Count PyTorch Recipe Jobs
    if bu_owner is not None and bu_owner != ScorecardModelConfig.BU.AI_HUB:
        # BUs do not own PyTorch model recipes.
        enabled_torch_models = set()
    # LLMs run through the separate LLM On-Device workflow, not the profile
    # path enumerated here — counting them inflates the total and trips the
    # daytime quota on splits that will never submit those jobs.
    enabled_torch_models = {
        m
        for m in enabled_torch_models
        if QAIHMModelInternalConfig.from_model(m).test_split is not TestRunnerSplit.LLM
    }
    for model_id in enabled_torch_models:
        # Get profile & inference job paramaterizations, concat them
        jobs = get_torch_recipe_profile_parameterizations(model_id)
        if run_accuracy_tests:
            jobs.extend(get_torch_recipe_inference_parameterizations(model_id))

        # Add to job counts
        total_jobs += len(jobs)
        for component, precision, path, device in jobs:
            # Count jobs per model (+ precision)
            model_id_precision = (
                f"{model_id} ({precision!s})" if show_precision_in_summary else model_id
            )
            if model_id_precision not in job_count_by_pytorch_recipe:
                job_count_by_pytorch_recipe[model_id_precision] = 0
            job_count_by_pytorch_recipe[model_id_precision] += 1

            # Count jobs per component
            if component is not None:
                if model_id_precision not in job_count_by_pytorch_recipe_component:
                    job_count_by_pytorch_recipe_component[model_id_precision] = {}
                if (
                    component
                    not in job_count_by_pytorch_recipe_component[model_id_precision]
                ):
                    job_count_by_pytorch_recipe_component[model_id_precision][
                        component
                    ] = 0
                job_count_by_pytorch_recipe_component[model_id_precision][
                    component
                ] += 1

            # Count per-device / per-path
            job_count_by_device[device] += 1
            job_count_by_form_factor[device.form_factor] += 1
            job_count_by_path[path] += 1

    # Count Static Model jobs
    for model_id in enabled_static_models:
        # Inference jobs are supported but not enabled by default for static models, so we don't count them.
        config = ScorecardModelConfig.from_scorecard_model_id(model_id)
        if bu_owner is not None and config.bu_owner != bu_owner:
            continue

        static_jobs = get_static_model_parameterizations(
            model_id,
            (
                static_models_dir
                if static_models_dir is not None
                else StaticModelsDirEnvvar.get()
            ),
            config.devices,
        )
        if run_accuracy_tests:
            inference_jobs = get_static_model_parameterizations(
                model_id,
                (
                    static_models_dir
                    if static_models_dir is not None
                    else StaticModelsDirEnvvar.get()
                ),
                [config.eval_device],
            )
            static_jobs.extend(inference_jobs)
        total_jobs += len(static_jobs)
        job_count_by_static_model[model_id] = len(static_jobs)
        for _precision, path, device in static_jobs:
            job_count_by_device[device] += 1
            job_count_by_form_factor[device.form_factor] += 1
            job_count_by_path[path] += 1

    return (
        total_jobs,
        job_count_by_device,
        job_count_by_form_factor,
        job_count_by_path,
        job_count_by_pytorch_recipe,
        job_count_by_pytorch_recipe_component,
        job_count_by_static_model,
    )


def device_job_counts_to_printstr(
    total_jobs: int,
    job_count_by_device: dict[ScorecardDevice, int],
    job_count_by_form_factor: dict[FormFactor, int],
    job_count_by_path: dict[ScorecardProfilePath, int],
    job_count_by_pytorch_recipe: dict[str, int],
    job_count_by_pytorch_recipe_component: dict[str, dict[str, int]],
    job_count_by_static_model: dict[str, int],
    run_accuracy_tests: bool = False,
    show_components_in_summary: bool = False,
) -> str:
    out = StringIO()
    # Print summary
    pandi = "profile & inference" if run_accuracy_tests else "profile"
    print(file=out)
    print(f"TOTAL* {pandi.upper()} JOB COUNT: {total_jobs}", file=out)
    print(file=out)
    print("----------", file=out)
    print(file=out)
    print(f"Max count* of {pandi} Jobs Submitted Per Device:", file=out)
    for device, count in sorted(
        job_count_by_device.items(),
        key=lambda kv: (-kv[1], kv[0].reference_device_name),
    ):
        print(f"    {device.reference_device_name}: {count}", file=out)
    print(file=out)
    print("----------", file=out)
    print(file=out)
    print(f"Max count* of {pandi} Jobs Submitted Per Device Form Factor:", file=out)
    for ff, count in sorted(
        job_count_by_form_factor.items(),
        key=lambda kv: (-kv[1], kv[0].value),
    ):
        print(f"    {ff.value}: {count}", file=out)
    print(file=out)
    print("----------", file=out)
    print(file=out)
    print(f"Max count* of {pandi} Jobs Submitted Per Scorecard Path:", file=out)
    for path, count in sorted(
        job_count_by_path.items(), key=lambda kv: (-kv[1], kv[0].name)
    ):
        print(f"    {path.name}: {count}", file=out)
    print(file=out)
    if len(job_count_by_pytorch_recipe) > 0:
        print("----------", file=out)
        print(file=out)
        print(f"Max count* of {pandi} Jobs Submitted Per PYTORCH RECIPE:\n")
        for model_id, count in sorted(
            job_count_by_pytorch_recipe.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"    {model_id}: {count}", file=out)
            if show_components_in_summary:
                for cid, ccount in sorted(
                    job_count_by_pytorch_recipe_component.get(model_id, {}).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                ):
                    print(f"        {cid}: {ccount}", file=out)
        print(file=out)
    if len(job_count_by_static_model) > 0:
        print("----------", file=out)
        print(file=out)
        print(
            f"Max count* of {pandi} Jobs Submitted Per STATIC ONNX / TORCHSCRIPT MODEL:",
            file=out,
        )
        for model_id, count in sorted(
            job_count_by_static_model.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"    {model_id}: {count}", file=out)
        print(file=out)
    print("----------", file=out)
    print(file=out)
    print(f"TOTAL* {pandi.upper()} JOB COUNT: {total_jobs}", file=out)
    print(file=out)
    print("----------", file=out)
    print(file=out)
    print(
        "* NOTE: THIS SCRIPT COMPUTES THE WORST CASE SCENARIO. The actual number of jobs is likely to be significantly less either due to compile job caching or compile job failures.",
        file=out,
    )
    print(file=out)
    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="""
        Count the maximum number (if all compile jobs succeed and aren't identical to the previous scorecard run) of profile jobs to be submitted given the provided testing configuration.

        If no flags are passed, this will count the number of jobs submitted in the current environment given all set QAIHM test environment variables.
        """,
    )
    EnabledModelsEnvvar.add_arg(parser, setenv=True)
    EnabledPathsEnvvar.add_arg(parser, setenv=True)
    EnabledPrecisionsEnvvar.add_arg(parser, setenv=True)
    EnabledDevicesEnvvar.add_arg(parser, setenv=True)
    IgnoreKnownFailuresEnvvar.add_arg(parser, setenv=True)
    StaticModelsDirEnvvar.add_arg(parser, setenv=True)
    parser.add_argument(
        "--bu-owner",
        type=str,
        default=None,
        choices=[x.value for x in ScorecardModelConfig.BU],
        help="BU Owner. Applies only to static onnx models.",
    )
    parser.add_argument(
        "--run-accuracy-tests",
        action="store_true",
        default=False,
        help="Whether or not accuracy testing (and therefore inference jobs) is enabled.",
    )
    parser.add_argument(
        "--assert-valid-daytime-scorecard",
        action="store_true",
        default=False,
        help="Raises an error if there are too many jobs to be run during the daytime scorecard.",
    )
    parser.add_argument(
        "--show-precision-in-summary",
        action="store_true",
        default=False,
        help="Break apart pyTorch models by precision when listing the count of jobs for each model.",
    )
    parser.add_argument(
        "--show-components-in-summary",
        action="store_true",
        default=False,
        help="Show pyTorch recipe model components when listing the count of jobs for each model.",
    )
    args = parser.parse_args()

    models: set[str | SpecialModelSetting] = args.models
    static_models_dir: Path = args.static_models_dir
    run_accuracy_tests: bool = args.run_accuracy_tests
    assert_valid_daytime_scorecard: int = args.assert_valid_daytime_scorecard
    show_precision_in_summary: bool = args.show_precision_in_summary
    show_components_in_summary: bool = args.show_components_in_summary
    bu_owner: ScorecardModelConfig.BU | None = (
        ScorecardModelConfig.BU(args.bu_owner) if args.bu_owner else None
    )

    (
        total_jobs,
        job_count_by_device,
        job_count_by_device_form_factor,
        job_count_by_path,
        job_count_by_pytorch_recipe,
        job_count_by_pytorch_recipe_component,
        job_count_by_static_model,
    ) = count_device_jobs(
        models,
        static_models_dir,
        run_accuracy_tests,
        show_precision_in_summary,
        bu_owner,
    )

    # Print summary
    print(
        device_job_counts_to_printstr(
            total_jobs,
            job_count_by_device,
            job_count_by_device_form_factor,
            job_count_by_path,
            job_count_by_pytorch_recipe,
            job_count_by_pytorch_recipe_component,
            job_count_by_static_model,
            run_accuracy_tests,
            show_components_in_summary,
        )
    )

    if (
        assert_valid_daytime_scorecard
        and date.today().weekday() < 5
        and not (SC_DAYTIME_START_TIME <= datetime.now().time() <= SC_DAYTIME_END_TIME)
    ):
        if total_jobs > MAX_DAYTIME_DEVICE_JOBS:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(
                f"ERROR: Total number of device jobs submitted by this scorecard ({total_jobs}) meets or exceeds the maximum daytime quota ({MAX_DAYTIME_DEVICE_JOBS})."
            )
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            sys.exit(1)

        auto_jobs = 0
        for device, count in job_count_by_device.items():
            if device.form_factor == FormFactor.AUTO:
                auto_jobs += count
        if auto_jobs > MAX_DAYTIME_AUTO_DEVICE_JOBS:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(
                f"ERROR: Total number of automotive device jobs submitted by this scorecard ({auto_jobs}) meets or exceeds the maximum daytime quota ({MAX_DAYTIME_AUTO_DEVICE_JOBS})."
            )
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            sys.exit(1)


if __name__ == "__main__":
    main()

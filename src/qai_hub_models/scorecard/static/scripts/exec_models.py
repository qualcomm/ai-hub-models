# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qai_hub import UserError

from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    DeploymentEnvvar,
    EnabledModelsEnvvar,
    SpecialModelSetting,
    StaticModelsDirEnvvar,
)
from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    InferenceScorecardJobYaml,
)
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.scorecard.static.model_config import ScorecardModelConfig
from qai_hub_models.scorecard.static.model_exec import (
    compile_model,
    inference_model,
)
from qai_hub_models.scorecard.static.model_sync import sync_model_assets
from qai_hub_models.utils.hub_clients import get_scorecard_client_or_raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    EnabledModelsEnvvar.add_arg(parser, {SpecialModelSetting.STATIC})
    StaticModelsDirEnvvar.add_arg(parser)
    ArtifactsDirEnvvar.add_arg(parser)
    DeploymentEnvvar.add_arg(parser)
    parser.add_argument(
        "--skip-compile", action="store_true", help="Skip compile step."
    )
    parser.add_argument(
        "--skip-inference", action="store_true", help="Skip inference step."
    )
    parser.add_argument(
        "--compile-ids",
        type=str,
        metavar="\b",
        default=os.environ.get("QAIHM_BENCH_COMPILE_JOBS_FILE", None),
        help="Compile Jobs YAML path. Applicable only if compile step is skipped.",
    )
    parser.add_argument(
        "--compile-timeout",
        type=int,
        metavar="\b",
        default=None,
        help=(
            "Wait a max of this many seconds for *each* compile job to finish before executing the inference job. "
            "If a timeout is reached, the inference job submission for that compiled model is skipped. "
            "If you do not want to wait for compile jobs, this should be set to 0. If you wish to wait indefinitely, "
            "this should be left at the default value."
        ),
    )
    parser.add_argument(
        "--persist-asset-sync",
        action="store_true",
        help=(
            "By default, this script will upload models and temporary datasets to Hub if an asset is not already  "
            "uploaded to the target Hub deployment. These assets are not persisted beyond the lifetime of this script. "
            "If this arg is set, permanent datasets will be synced, and asset IDs will be saved to the on-disk model config. "
            "Generally you should set this only if you anticipate this deployment being used multiple times and you plan to commit the config changes."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Collect args
    models_path: Path = args.static_models_dir
    models: set[str | SpecialModelSetting] = args.models
    _, model_id_list = validate_and_split_enabled_models(models, models_path)
    compile_ids = (
        Path(args.compile_ids)
        if args.compile_ids
        else ScorecardArtifact.COMPILE_YAML.touch()
    )
    inference_ids = ScorecardArtifact.INFERENCE_YAML.touch()
    deployment: str = args.deployment
    compile_job_timeout: int | None = args.compile_timeout
    permanent_asset_sync: bool = args.persist_asset_sync

    enable_compile: bool = not args.skip_compile
    enable_inference: bool = not args.skip_inference
    if not enable_compile and not enable_inference:
        print("Please enable compile or inference")
        sys.exit(1)

    # Get model configs
    model_configs = [
        ScorecardModelConfig.from_scorecard_model_id(model_id)
        for model_id in model_id_list
    ]

    # Sync assets for selected models
    for config in model_configs:
        print(f"Sync Assets for {config.id}")
        sync_model_assets(config, deployment, permanent_asset_sync)
        if permanent_asset_sync:
            config.to_scorecard_yaml(models_path)

    compile_results = None
    if enable_compile:
        compile_results = CompileScorecardJobYaml.from_file(
            compile_ids, create_empty_if_no_file=True
        )
        for config in model_configs:
            # Disable client verbosity; compile/inference model will print more succinct summaries instead
            hub = get_scorecard_client_or_raise(deployment, config.restrict_access)
            hub.set_verbose(False)

            try:
                compile_model(
                    config.id,
                    config.hub_model_ids_automated[deployment],
                    hub,
                    config.get_hub_api_input_specs(),
                    config.precision,
                    config.devices,
                    list({x.compile_path for x in config.enabled_profile_runtimes}),
                    config.output_names,
                    config.channel_first_inputs,
                    config.channel_first_outputs,
                    config.extra_compile_options,
                    compile_results,
                )
            except UserError as e:
                print(f"Failed to submit compile job for model `{config.id}`: {e}")
        compile_results.to_file()

    elif enable_inference:
        compile_results = CompileScorecardJobYaml.from_file(
            compile_ids, create_empty_if_no_file=True
        )

    if enable_inference:
        assert compile_results  # for mypy
        inference_results = InferenceScorecardJobYaml.from_file(
            inference_ids, create_empty_if_no_file=True
        )
        for config in model_configs:
            # Disable client verbosity; compile/inference model will print more succinct summaries instead
            hub = get_scorecard_client_or_raise(deployment, config.restrict_access)
            hub.set_verbose(False)

            try:
                inference_model(
                    config.id,
                    config.hub_input_dataset_ids_automated[deployment],
                    config.hub_input_channel_last_dataset_ids_automated.get(deployment),
                    compile_results,
                    hub,
                    config.precision,
                    config.devices,
                    config.enabled_profile_runtimes,
                    inference_results,
                    compile_job_timeout,
                )
            except UserError as e:
                print(f"Failed to submit inference job for model `{config.id}`: {e}")
        inference_results.to_file()

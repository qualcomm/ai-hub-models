# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import subprocess
from pathlib import Path
from typing import cast

from qai_hub_models.scorecard.envvars import (
    DeploymentListEnvvar,
    EnabledModelsEnvvar,
    SpecialModelSetting,
    StaticModelsDirEnvvar,
)
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.scorecard.static.model_config import ScorecardModelConfig
from qai_hub_models.scorecard.static.model_sync import (
    DEFAULT_DEPLOYMENTS,
    sync_model_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    EnabledModelsEnvvar.add_arg(parser)
    StaticModelsDirEnvvar.add_arg(parser)
    DeploymentListEnvvar.add_arg(parser, list(DEFAULT_DEPLOYMENTS))
    parser.add_argument(
        "--clear-existing-assets",
        action="store_true",
        help="Re-uploads all assets to all deployments, regardless of whether they exist already in a config or not.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    models_path: Path = args.static_models_dir
    models: set[str | SpecialModelSetting] = args.models
    _, model_id_list = validate_and_split_enabled_models(args.models, models_path)
    deployments_list = cast(list[str], args.deployments)
    modified_files = []
    for model_id in model_id_list:
        config = ScorecardModelConfig.from_scorecard_model_id(model_id)
        sync_model_assets(
            config,
            deployments_list,
            clear_existing=args.clear_existing_assets,
        )
        config.to_scorecard_yaml(models_path)
        modified_files.append(str(models_path / f"{model_id}.yaml"))

    # Run pre-commit on re-generated files
    if modified_files:
        subprocess.run(["pre-commit", "run", "--files", *modified_files], check=False)

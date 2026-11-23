# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Run the scorecard for every requested model, one after another.

Each model runs run_scorecard_single.py in its own process. Models compatible
with the global requirements use this environment; the others get their own
venv (created on demand and reused under --envs-dir). A model that fails does
not stop the others; the exit code is non-zero if any failed. See run_scorecard_single.py
for the stages and the options shared with it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.scripts.run_scorecard_single import add_args, init_hub_client
from qai_hub_models.scripts.scorecard.envs import (
    default_envs_dir,
    ensure_model_env,
    needs_own_env,
)

_SINGLE_MODULE = "qai_hub_models.scripts.run_scorecard_single"


def _argv_for_model(model_id: str) -> list[str]:
    """This process's CLI args with --models narrowed to `model_id`, minus --envs-dir."""
    out: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
        elif arg in ("--models", "--envs-dir"):
            skip_next = True
        elif not arg.startswith(("--models=", "--envs-dir=")):
            out.append(arg)
    return [*out, "--models", model_id]


def _run_model(model_id: str, envs_dir: Path) -> bool:
    """
    Run run_scorecard_single for `model_id` in its own process: in its own
    (created on demand) venv if it needs one, else in this environment.
    A process per model keeps one model's imports (e.g. an external repo's
    top-level `models` package) and crashes from affecting the next.
    """
    try:
        python = (
            ensure_model_env(model_id, envs_dir)
            if needs_own_env(model_id)
            else Path(sys.executable)
        )
    except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
        print(f"{model_id} | ENV SETUP FAILED: {e}")
        return False
    result = subprocess.run(
        [str(python), "-m", _SINGLE_MODULE, *_argv_for_model(model_id)],
        env=os.environ.copy(),
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args(parser)
    parser.add_argument(
        "--envs-dir",
        type=Path,
        default=default_envs_dir(),
        help="Where venvs for models incompatible with the global requirements "
        "are created and reused.",
    )
    args = parser.parse_args()
    init_hub_client(args)

    pytorch_models, static_models = validate_and_split_enabled_models(
        args.models, args.static_models_dir
    )
    if static_models:
        raise SystemExit(
            f"run_scorecard.py only runs PyTorch recipe models, got {sorted(static_models)}."
        )

    failed = [
        model_id
        for model_id in sorted(pytorch_models)
        if not _run_model(model_id, args.envs_dir)
    ]
    if failed:
        print(f"run_scorecard failed for: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Parse the compound ``options`` input passed through scorecard.yml.

Input is a comma-separated list of ``key=value`` segments; see ``_OPTIONS`` for
the accepted keys and the tests for the full accepted / rejected set. Every
option is written to ``$GITHUB_OUTPUT`` under its own name and, when
``--emit-env`` is set, also to ``$GITHUB_ENV`` under the ``QAIHM_TEST_*`` name
in ``_OPTIONS`` so downstream shell steps in the same job pick them up as env
vars.

scorecard.yml is at GitHub's 10-input limit for workflow_dispatch, so new
scorecard knobs are added here rather than as new workflow inputs. Adding one is
a single ``_OPTIONS`` entry.

``checkpoint`` additionally requires ``--models`` (or MODELS) to name exactly one
model, since one checkpoint cannot be correct for a whole scorecard split.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable


def _qairt_version(key: str, val: str) -> str:
    # Downstream QAIRTVersionEnvvar expects this sentinel rather than an empty string.
    return val or "qaihm_default"


def _geniex_version(key: str, val: str) -> str:
    if val.startswith("v"):
        raise ValueError(
            f"{key} must not start with 'v' (use bare SemVer, e.g. '0.3.7'); got {val!r}"
        )
    return val


_GENIE_SCOPES = ("perf", "accuracy", "perf_and_accuracy")


def _genie_scope(key: str, val: str) -> str:
    # Empty means Genie follows the scorecard's test_scope.
    if val and val not in _GENIE_SCOPES:
        raise ValueError(
            f"{key} must be one of {', '.join(_GENIE_SCOPES)}; got {val!r}"
        )
    return val


def _checkpoint(key: str, val: str) -> str:
    if not val:
        return val
    if "://" in val:
        raise ValueError(
            f"{key} must be a bare key in the private bucket, not a URL; strip the "
            f"s3://<bucket>/ prefix. Got {val!r}"
        )
    if val.startswith("/") or "\\" in val or ".." in val.split("/"):
        raise ValueError(f"{key} must be a relative key without '..'; got {val!r}")
    if not val.endswith(".zip"):
        raise ValueError(
            f"{key} must name a .zip produced by quantize_and_upload; got {val!r}"
        )
    return val


# option name -> (envvar emitted by --emit-env, value when the option is unset, normalizer)
_OPTIONS: dict[str, tuple[str, str, Callable[[str, str], str]]] = {
    "qairt_version": ("QAIHM_TEST_QAIRT_VERSION", "qaihm_default", _qairt_version),
    "geniex_version": ("QAIHM_TEST_GENIEX_VERSION", "", _geniex_version),
    "genie_scope": ("QAIHM_TEST_GENIE_SCOPE", "", _genie_scope),
    "checkpoint": ("QAIHM_TEST_CHECKPOINT", "", _checkpoint),
    # Device-eval dataset for LLM accuracy; empty means grace2.
    "eval_dataset": ("QAIHM_TEST_EVAL_DATASET", "", lambda key, val: val),
}

# Values of SpecialModelSetting (scorecard/envvars.py). Duplicated rather than
# imported: scripts/ci is stdlib-only so it can run without the qaihm venv.
_SPECIAL_MODEL_SETTINGS = frozenset(
    {
        "all",
        "pytorch",
        "pytorch_no_llm",
        "static",
        "bench",
        "llm_week1",
        "llm_week2",
        "llm_downloadable",
    }
)


def validate_single_model(checkpoint: str, models: str | None) -> None:
    """Reject a checkpoint override that is not scoped to exactly one model.

    One checkpoint can only be correct for one model. Aimed at a multi-model run
    every other model would silently load these weights and report plausible but
    wrong numbers, so this fails instead.

    Parameters
    ----------
    checkpoint
        The resolved checkpoint option; "" when unset.
    models
        The scorecard `models` input, as passed to QAIHM_TEST_MODELS. None when
        the caller has no models input to check against -- scorecard.yml's LLM
        run reads its model list from environment.env after this step, so it
        passes nothing and the scoping is enforced by the job that does have it.

    Raises
    ------
    ValueError
        If checkpoint is set and models is given but is not one model id.
    """
    if not checkpoint or models is None:
        return
    models = models.strip()
    named = [m.strip() for m in models.split(",") if m.strip()]
    if len(named) != 1 or named[0].lower() in _SPECIAL_MODEL_SETTINGS:
        raise ValueError(
            f"checkpoint={checkpoint} applies to one model, so models must name "
            f"exactly one model id; got {models!r}. Pass models=<model_id>."
        )


def parse(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for seg in raw.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "=" not in seg:
            raise ValueError(
                f"Options must be key=value; got bare value {seg!r}. "
                f"Known keys: {', '.join(_OPTIONS)}."
            )
        key, _, val = seg.partition("=")
        key = key.strip()
        val = val.strip()
        if key not in _OPTIONS:
            raise ValueError(f"Unknown option key: {key!r}")
        if key in values:
            raise ValueError(f"Duplicate option key: {key!r}")
        values[key] = _OPTIONS[key][2](key, val)

    return {
        key: values.get(key, unset_value)
        for key, (_, unset_value, _) in _OPTIONS.items()
    }


def _write_kv_file(path: str, pairs: list[tuple[str, str]]) -> None:
    # GITHUB_OUTPUT/GITHUB_ENV are line-oriented; newlines in a value would
    # inject an extra key=value line and let a crafted input override the
    # intended one.
    for key, val in pairs:
        if "\n" in val or "\r" in val:
            raise ValueError(f"{key} must not contain newlines: {val!r}")
    with open(path, "a") as f:
        for key, val in pairs:
            f.write(f"{key}={val}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.environ.get("RAW", ""))
    ap.add_argument(
        "--models",
        # None, not "": a caller with no models input to offer must not be read
        # as a caller whose models input is empty.
        default=os.environ.get("MODELS"),
        help="The scorecard `models` input, checked against the checkpoint option.",
    )
    ap.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    ap.add_argument("--github-env", default=os.environ.get("GITHUB_ENV"))
    ap.add_argument(
        "--emit-env",
        action="store_true",
        help="Also write each option's QAIHM_TEST_* envvar to $GITHUB_ENV.",
    )
    args = ap.parse_args()

    try:
        options = parse(args.raw)
        validate_single_model(options["checkpoint"], args.models)
        print("Parsed: " + " ".join(f"{k}={v!r}" for k, v in options.items()))
        if args.github_output:
            _write_kv_file(args.github_output, list(options.items()))
        if args.emit_env and args.github_env:
            _write_kv_file(
                args.github_env,
                [(_OPTIONS[key][0], val) for key, val in options.items()],
            )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

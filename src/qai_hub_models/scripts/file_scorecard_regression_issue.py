#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Generate a GitHub issue body for scorecard regressions (perf 2x+ and accuracy).

Reads the JSON files produced by PerformanceDiff.dump_severe_regressions_json(),
NumericsDiff.dump_regressions_json(), and NumericsDiff.dump_newly_disabled_json(),
then renders a Jinja template.

The actual issue creation is done by the GitHub Action that calls this script.

Usage (from GitHub Actions):
    python3 -m qai_hub_models.scripts.file_scorecard_regression_issue \
        --perf-regressions-json path/to/perf-regressions-2x-*.json \
        --newly-disabled-json path/to/newly-disabled-*.json \
        --output regression-issue.json \
        --run-url "https://github.com/..." \
        --perf-diff-url "https://..." \
        --numerics-diff-url "https://..."
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from qai_hub_models import QAIRTVersion
from qai_hub_models.scorecard.envvars import DateFormatEnvvar
from qai_hub_models.scorecard.results.yaml import ToolVersionsByPathYaml
from qai_hub_models.scripts.download_scorecard_results import (
    download_single_artifact,
    find_latest_run,
)
from qai_hub_models.utils.hub_clients import deployment_is_prod

TEMPLATES_DIR = Path(__file__).parent / "templates"

# GitHub's hard limit on issue bodies is 65536 characters. Leave a small margin
# so the truncation footer we may append still fits.
MAX_ISSUE_BODY_LEN = 65000

# AI Hub job IDs are short alphanumeric tokens. Anything else is rejected to
# avoid markdown injection in the rendered issue body.
_JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# GitHub Actions run IDs are numeric; keep an alphanumeric allowlist for the
# rare case someone hand-invokes with a non-numeric label. Anything outside
# this shape would be rendered verbatim in the issue body.
_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Dates on S3 manifests are ISO-8601 (YYYY-MM-DD). Allowlist matches that shape
# to avoid markdown injection via manifest.date rendered into the context grid.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _job_url(job_id: str, deployment: str = "workbench") -> str:
    """Build an AI Hub job URL from a job ID."""
    if job_id == "null" or not job_id:
        return ""
    return f"https://{deployment}.aihub.qualcomm.com/jobs/{job_id}/"


def _job_link(job_id: str, deployment: str = "workbench") -> str:
    """Build a markdown link for a job ID, or N/A."""
    if job_id == "null" or not job_id or not _JOB_ID_RE.match(job_id):
        return "N/A"
    return f"[{job_id}]({_job_url(job_id, deployment)})"


_DEPLOYMENT_SUFFIX_RE = re.compile(r"\((prod|dev|staging)\)")


def _env_label(deployment: str) -> str:
    """Map an AI Hub URL subdomain to its env name shown in column headers.

    For non-prod deployments the subdomain and env name are identical
    (dev/staging); only prod maps subdomain "workbench" -> env "prod".
    """
    return "prod" if deployment_is_prod(deployment) else deployment.lower()


def _subdomain_for_env(env: str) -> str:
    """Inverse of _env_label: env name -> AI Hub URL subdomain."""
    return "workbench" if deployment_is_prod(env) else env.lower()


def _deployment_for_column(column_name: str, default: str) -> str:
    """Pick the AI Hub subdomain to use for a Job ID column's links.

    Columns may carry a (prod|dev|staging) suffix recorded by an earlier
    pipeline stage; that suffix wins. Columns without a suffix fall back to
    the caller-supplied default — typically the current run's deployment for
    new-run columns, and the previous baseline's deployment for "Previous *"
    columns.
    """
    match = _DEPLOYMENT_SUFFIX_RE.search(column_name)
    if match:
        return _subdomain_for_env(match.group(1))
    return default


# Canonical job-id column prefixes. The order matters — 'Previous *' must come
# first so the prefix-match below doesn't classify 'Previous Job ID' as 'Job ID'.
_PREVIOUS_PREFIXES = (
    "Previous Job ID",
    "Previous Compile Job ID",
    "Previous Inference Job ID",
)
_NEW_PREFIXES = ("Job ID", "Compile Job ID", "Inference Job ID")


def _is_previous_column(column_name: str) -> bool:
    return any(column_name.startswith(p) for p in _PREVIOUS_PREFIXES)


def _retag_columns(
    rows: list[dict], deployment: str, previous_deployment: str
) -> list[dict]:
    """Append a (env) suffix to canonical job-id columns based on which side
    of the comparison they describe.

    The diff scripts emit canonical column names with no env suffix
    ('Job ID', 'Previous Compile Job ID', etc). We add the suffix here
    because only the issue builder knows both deployments.
    """
    new_env = _env_label(deployment)
    prev_env = _env_label(previous_deployment)
    out = []
    for row in rows:
        new_row = {}
        for key, val in row.items():
            if _DEPLOYMENT_SUFFIX_RE.search(key):
                # Already tagged (e.g. historical S3 JSON); leave it.
                new_row[key] = val
            elif _is_previous_column(key):
                new_row[f"{key} ({prev_env})"] = val
            elif any(key == p or key.startswith(p + " ") for p in _NEW_PREFIXES):
                new_row[f"{key} ({new_env})"] = val
            else:
                new_row[key] = val
        out.append(new_row)
    return out


def _linkify_job_ids(
    rows: list[dict],
    deployment: str = "workbench",
    previous_deployment: str | None = None,
) -> list[dict]:
    """Convert job ID values to markdown links in-place.

    Any column whose name contains "Job ID" gets its value converted from a
    plain ID string to a markdown link. The link's deployment subdomain is
    derived from the column name when it carries a (prod|dev|staging) suffix;
    otherwise it falls back to `previous_deployment` for "Previous *" columns
    and to `deployment` for the rest.
    """
    if previous_deployment is None:
        previous_deployment = deployment
    result = []
    for row in rows:
        new_row = {}
        for key, val in row.items():
            if "Job ID" in key:
                fallback = (
                    previous_deployment if _is_previous_column(key) else deployment
                )
                col_deployment = _deployment_for_column(key, fallback)
                new_row[key] = _job_link(str(val), col_deployment)
            else:
                new_row[key] = val
        result.append(new_row)
    return result


@dataclass
class ScorecardContextCell:
    """One cell of the Scorecard Context table (deployment vs latest/previous)."""

    run_id: str = ""
    date: str = ""
    qairt: str = ""


@dataclass
class ScorecardContext:
    """2x2 grid summarising both deployments' latest + previous runs.

    Rendered above the regression tables so a reader can tell at a glance
    which side of the dev/prod split this issue was filed for, and whether
    the previous same-deployment baseline is recent enough to be trusted.
    """

    dev_latest: ScorecardContextCell = field(default_factory=ScorecardContextCell)
    dev_previous: ScorecardContextCell = field(default_factory=ScorecardContextCell)
    prod_latest: ScorecardContextCell = field(default_factory=ScorecardContextCell)
    prod_previous: ScorecardContextCell = field(default_factory=ScorecardContextCell)
    current_env: str = ""


# tool-versions.yaml stores one QAIRT string per profile path; scorecard runs
# use one QAIRT for the whole run today, but a per-path field lets that
# change. Pick this path when summarising because it exists for every
# scorecard configuration we ship.
_CONTEXT_QAIRT_PATH = "qnn_context_binary"


def _short_qairt(version: QAIRTVersion) -> str:
    """Format a QAIRT version as major.minor.patch for the context grid.

    QAIRT.full_version_with_flavor includes the build id (2.48.0.260626120635);
    the grid only shows the semver portion so cells stay narrow enough for
    GitHub's markdown renderer.
    """
    fw = version.framework
    if fw.patch is None:
        return fw.api_version
    return f"{fw.api_version}.{fw.patch}"


def _cell_from_manifest_and_versions(
    run_id: str, date: str, tool_versions: ToolVersionsByPathYaml
) -> ScorecardContextCell:
    """Build one context cell. QAIRT is left blank if we can't find it."""
    # Sanitize inputs before they land in the issue markdown. run_id/date can
    # come from an S3 manifest (traceable to a free-form workflow input), so
    # apply the same allowlist we already use for --current-run-id and job IDs.
    if not _RUN_ID_RE.match(run_id):
        run_id = ""
    if not _DATE_RE.match(date):
        date = ""
    qairt = ""
    for path, versions in tool_versions.tool_versions.items():
        if path.name.lower() == _CONTEXT_QAIRT_PATH and versions.qairt is not None:
            qairt = _short_qairt(versions.qairt)
            break
    if not qairt:
        # Fall back to whichever path has a qairt set — better than blank.
        for versions in tool_versions.tool_versions.values():
            if versions.qairt is not None:
                qairt = _short_qairt(versions.qairt)
                break
    return ScorecardContextCell(run_id=run_id, date=date, qairt=qairt)


def _lookup_context_cell(
    deployment: str,
    exclude_run_id: str = "",
    scheduled_run_ids: list[str] | None = None,
) -> ScorecardContextCell:
    """Pull the most recent (or previous) run's cell from S3 for a deployment.

    Returns an empty cell on any S3 miss or error — this table is context,
    not load-bearing, so a lookup failure must not break issue filing.
    """
    try:
        manifest = find_latest_run(
            deployment,
            exclude_run_id=exclude_run_id,
            scheduled_run_ids=scheduled_run_ids,
        )
    except Exception:
        logging.warning(
            "S3 lookup failed while building scorecard context for %s.",
            deployment,
            exc_info=True,
        )
        return ScorecardContextCell()
    if manifest is None:
        return ScorecardContextCell()

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "tool-versions.yaml"
        try:
            downloaded = download_single_artifact(manifest, "tool-versions.yaml", dest)
        except Exception:
            logging.warning(
                "S3 download failed for %s tool-versions.yaml.",
                manifest.run_id,
                exc_info=True,
            )
            downloaded = None
        tool_versions = (
            ToolVersionsByPathYaml.from_yaml(downloaded, create_empty_if_no_file=True)
            if downloaded is not None
            else ToolVersionsByPathYaml()
        )
    return _cell_from_manifest_and_versions(
        manifest.run_id, manifest.date, tool_versions
    )


def build_scorecard_context(
    current_deployment: str,
    current_run_id: str,
    current_tool_versions: ToolVersionsByPathYaml,
    scheduled_run_ids: list[str] | None = None,
) -> ScorecardContext:
    """Assemble the Scorecard Context table.

    Parameters
    ----------
    current_deployment
        AI Hub subdomain: "workbench" (=prod), "dev", "staging".
    current_run_id
        The run ID for this scorecard. Used to (a) fill the "latest" cell of
        this run's deployment, and (b) exclude the current run from the
        opposite-deployment lookup in case the current run is somehow already
        indexed as latest for a different deployment (partial retry).
    current_tool_versions
        This run's tool-versions.yaml (already loaded upstream).
    scheduled_run_ids
        GitHub run IDs of scheduled weekly scorecards, newest first. When set,
        only these runs fill the non-current cells.

    Returns
    -------
    ScorecardContext
        Populated grid with the current run in the correct deployment column
        and the other cells filled from S3 (or left empty on miss).
    """
    current_env = _env_label(current_deployment)
    # Kickoff run's start date (set by the collect workflow); same date S3 records.
    run_date = DateFormatEnvvar.get().strftime("%Y-%m-%d")
    current_cell = _cell_from_manifest_and_versions(
        current_run_id, run_date, current_tool_versions
    )

    # Previous run for THIS deployment (baseline used by the toolchain diff).
    # S3 manifests store the normalized env label ("prod"/"dev"/"staging"), not
    # the raw AI Hub URL subdomain, so look up by current_env not current_deployment.
    same_prev = _lookup_context_cell(
        current_env,
        exclude_run_id=current_run_id,
        scheduled_run_ids=scheduled_run_ids,
    )

    # Other deployment's latest + previous (context only). Same reason as above:
    # compare env labels, not subdomains.
    other_env = "dev" if current_env == "prod" else "prod"
    other_latest = _lookup_context_cell(
        other_env,
        exclude_run_id=current_run_id,
        scheduled_run_ids=scheduled_run_ids,
    )
    other_prev = _lookup_context_cell(
        other_env,
        exclude_run_id=other_latest.run_id or current_run_id,
        scheduled_run_ids=scheduled_run_ids,
    )

    ctx = ScorecardContext(current_env=current_env)
    if current_env == "prod":
        ctx.prod_latest = current_cell
        ctx.prod_previous = same_prev
        ctx.dev_latest = other_latest
        ctx.dev_previous = other_prev
    else:
        # Non-prod deployments (dev, staging, …) show under the "Dev" column;
        # the current-env annotation clarifies which one it actually is.
        ctx.dev_latest = current_cell
        ctx.dev_previous = same_prev
        ctx.prod_latest = other_latest
        ctx.prod_previous = other_prev
    return ctx


def _render(
    today: str,
    perf_regressions: list[dict],
    numerics_regressions: list[dict],
    newly_disabled_configs: list[dict],
    newly_failing_benchmarks: list[dict],
    run_url: str,
    perf_diff_url: str,
    numerics_diff_url: str,
    context: ScorecardContext | None = None,
    perf_dropped: int = 0,
    numerics_dropped: int = 0,
    accuracy_dropped: int = 0,
    quantized_failures: list[dict] | None = None,
) -> str:
    template = _env.get_template("scorecard_regression_issue_template.j2")
    return template.render(
        today=today,
        perf_regressions=perf_regressions,
        numerics_regressions=numerics_regressions,
        newly_disabled_configs=newly_disabled_configs,
        newly_failing_benchmarks=newly_failing_benchmarks,
        quantized_failures=quantized_failures or [],
        run_url=run_url,
        perf_diff_url=perf_diff_url,
        numerics_diff_url=numerics_diff_url,
        context=context,
        perf_dropped=perf_dropped,
        numerics_dropped=numerics_dropped,
        accuracy_dropped=accuracy_dropped,
    )


def build_issue_body(
    perf_regressions: list[dict],
    numerics_regressions: list[dict],
    run_url: str,
    perf_diff_url: str,
    numerics_diff_url: str,
    newly_disabled_configs: list[dict] | None = None,
    newly_failing_benchmarks: list[dict] | None = None,
    deployment: str = "workbench",
    previous_deployment: str | None = None,
    context: ScorecardContext | None = None,
    quantized_failures: list[dict] | None = None,
) -> str:
    """Build the GitHub issue body with regression tables.

    GitHub rejects issue bodies longer than 65536 characters. If the rendered
    body would exceed that, drop rows from the largest table first until it
    fits, and append a note pointing readers to the linked diff artifacts for
    the full list.
    """
    if previous_deployment is None:
        previous_deployment = deployment
    if newly_disabled_configs is None:
        newly_disabled_configs = []
    if newly_failing_benchmarks is None:
        newly_failing_benchmarks = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    perf_tagged = _retag_columns(perf_regressions, deployment, previous_deployment)
    numerics_tagged = _retag_columns(
        numerics_regressions, deployment, previous_deployment
    )
    perf = _linkify_job_ids(perf_tagged, deployment, previous_deployment)
    numerics = _linkify_job_ids(numerics_tagged, deployment, previous_deployment)
    # Accuracy rows don't carry job IDs, so no re-tagging / linkifying needed.
    disabled = list(newly_disabled_configs)
    benchmarks = list(newly_failing_benchmarks)
    quantized = list(quantized_failures or [])

    body = _render(
        today,
        perf,
        numerics,
        disabled,
        benchmarks,
        run_url,
        perf_diff_url,
        numerics_diff_url,
        context=context,
        quantized_failures=quantized,
    )
    perf_dropped = numerics_dropped = accuracy_dropped = 0
    while len(body) > MAX_ISSUE_BODY_LEN and (
        perf or numerics or disabled or benchmarks or quantized
    ):
        # Bulk-drop based on current overage so we don't re-render once per row.
        total_rows = (
            len(perf) + len(numerics) + len(disabled) + len(benchmarks) + len(quantized)
        )
        chars_per_row = max(1, len(body) // max(total_rows, 1))
        rows_to_drop = max(1, (len(body) - MAX_ISSUE_BODY_LEN) // chars_per_row)
        for _ in range(rows_to_drop):
            if not (perf or numerics or disabled or benchmarks or quantized):
                break
            # Drop from whichever table currently has the most rows.
            max_size = max(
                len(perf), len(numerics), len(disabled), len(benchmarks), len(quantized)
            )
            if len(perf) == max_size and perf:
                perf.pop()
                perf_dropped += 1
            elif len(numerics) == max_size and numerics:
                numerics.pop()
                numerics_dropped += 1
            elif len(disabled) == max_size and disabled:
                disabled.pop()
                accuracy_dropped += 1
            elif len(benchmarks) == max_size and benchmarks:
                benchmarks.pop()
                accuracy_dropped += 1
            elif quantized:
                quantized.pop()
                accuracy_dropped += 1
        body = _render(
            today,
            perf,
            numerics,
            disabled,
            benchmarks,
            run_url,
            perf_diff_url,
            numerics_diff_url,
            context=context,
            perf_dropped=perf_dropped,
            numerics_dropped=numerics_dropped,
            accuracy_dropped=accuracy_dropped,
            quantized_failures=quantized,
        )
    # Defensive hard cap: if the template's fixed overhead alone (headers,
    # URLs, footers) exceeds the limit, GitHub would still 422 us. Truncate.
    if len(body) > MAX_ISSUE_BODY_LEN:
        body = body[: MAX_ISSUE_BODY_LEN - 3] + "..."
    return body


def _resolve_glob(pattern: str) -> str | None:
    """Resolve a glob pattern to a single file path, or None."""
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _load_json(path: str | None) -> list[dict]:
    """Load a JSON file and return its contents, or empty list."""
    if not path:
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GitHub issue body for 2x+ scorecard regressions"
    )
    parser.add_argument(
        "--perf-regressions-json",
        required=True,
        help="Path (or glob) to perf-regressions-2x-*.json",
    )
    parser.add_argument(
        "--numerics-regressions-json",
        default="",
        help="Path (or glob) to numerics-regressions-*.json",
    )
    parser.add_argument(
        "--newly-disabled-json",
        default="",
        help=(
            "Path (or glob) to newly-disabled-*.json. Contains a flat list of "
            "rows tagged with `kind` (disabled_configuration or benchmark_failure) "
            "used to render the accuracy sections of the issue body."
        ),
    )
    parser.add_argument(
        "--quantized-failures-json",
        default="",
        help=(
            "Path (or glob) to quantized-failures-*.json: every disabled quantized "
            "configuration with its off-target Sim accuracy."
        ),
    )
    parser.add_argument(
        "--run-url",
        default="N/A",
        help="URL to the scorecard GitHub Actions run",
    )
    parser.add_argument(
        "--perf-diff-url",
        default="N/A",
        help="URL to the performance diff artifact",
    )
    parser.add_argument(
        "--numerics-diff-url",
        default="N/A",
        help="URL to the numerics diff artifact",
    )
    parser.add_argument(
        "--deployment",
        default="workbench",
        help="AI Hub deployment subdomain for the new scorecard run's job URLs (default: workbench)",
    )
    parser.add_argument(
        "--previous-deployment",
        default=None,
        help=(
            "AI Hub deployment subdomain for the previous baseline's job URLs. "
            "Defaults to --deployment, since most runs compare against the same "
            "deployment's prior run. Override when the baseline came from a "
            "different deployment (e.g. dev run comparing against main/prod)."
        ),
    )
    parser.add_argument(
        "--labels",
        default="p1,scorecard",
        help="Comma-separated labels for the filed issue (default: p1,scorecard)",
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Optional prefix for the issue title (e.g. '[TEST] ')",
    )
    parser.add_argument(
        "--current-run-id",
        default="",
        help=(
            "GitHub Actions run ID for the scorecard being filed. Enables the "
            "Scorecard Context table when combined with --tool-versions-path. "
            "Omitting either skips the context block."
        ),
    )
    parser.add_argument(
        "--tool-versions-path",
        default="",
        help=(
            "Path to this run's tool-versions.yaml. Used to fill the current "
            "run's QAIRT in the Scorecard Context table."
        ),
    )
    parser.add_argument(
        "--scheduled-run-ids",
        default=None,
        help=(
            "Comma-separated GitHub run IDs of scheduled weekly scorecards "
            "(prod + dev), newest first. Restricts the Scorecard Context's "
            "Previous cells to real weekly runs."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the issue JSON (title + body)",
    )
    args = parser.parse_args()

    scheduled_run_ids: list[str] | None = None
    if args.scheduled_run_ids is not None:
        scheduled_run_ids = [i for i in args.scheduled_run_ids.split(",") if i]
        if not all(i.isdigit() for i in scheduled_run_ids):
            parser.error(f"Invalid --scheduled-run-ids {args.scheduled_run_ids!r}")
        scheduled_run_ids.sort(key=int, reverse=True)

    # Load structured regression data
    perf_path = _resolve_glob(args.perf_regressions_json)
    if not perf_path:
        print(f"No perf regressions JSON found matching: {args.perf_regressions_json}")
        return
    perf_regressions = _load_json(perf_path)

    numerics_path = (
        _resolve_glob(args.numerics_regressions_json)
        if args.numerics_regressions_json
        else None
    )
    numerics_regressions = _load_json(numerics_path)

    newly_disabled_path = (
        _resolve_glob(args.newly_disabled_json) if args.newly_disabled_json else None
    )
    newly_disabled_rows = _load_json(newly_disabled_path)
    disabled_configs = [
        {k: v for k, v in r.items() if k != "kind"}
        for r in newly_disabled_rows
        if r.get("kind") == "disabled_configuration"
    ]
    newly_failing_benchmarks = [
        {k: v for k, v in r.items() if k != "kind"}
        for r in newly_disabled_rows
        if r.get("kind") == "benchmark_failure"
    ]
    has_accuracy = bool(disabled_configs or newly_failing_benchmarks)
    # Context only: these rows alone never file an issue, or one would be filed every week.
    quantized_failures = _load_json(
        _resolve_glob(args.quantized_failures_json)
        if args.quantized_failures_json
        else None
    )

    if not perf_regressions and not numerics_regressions and not has_accuracy:
        print("No 2x+ regressions found — skipping issue creation.")
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    env_label = _env_label(args.deployment).capitalize()
    if has_accuracy:
        title = (
            f"[Scorecard - {env_label}] Regressions Detected "
            f"(perf 2x+, accuracy) - {today}"
        )
    else:
        title = f"[Scorecard - {env_label}] 2x+ Regressions Detected - {today}"

    context: ScorecardContext | None = None
    if args.current_run_id and args.tool_versions_path:
        if not _RUN_ID_RE.match(args.current_run_id):
            print(
                f"Invalid --current-run-id {args.current_run_id!r}; "
                "expected alphanumeric/_/- (up to 64 chars). Skipping context block."
            )
        else:
            try:
                current_tool_versions = ToolVersionsByPathYaml.from_yaml(
                    args.tool_versions_path, create_empty_if_no_file=True
                )
                context = build_scorecard_context(
                    current_deployment=args.deployment,
                    current_run_id=args.current_run_id,
                    current_tool_versions=current_tool_versions,
                    scheduled_run_ids=scheduled_run_ids,
                )
            except Exception:
                logging.warning(
                    "Failed to build Scorecard Context; issue will be filed without it.",
                    exc_info=True,
                )

    body = build_issue_body(
        perf_regressions,
        numerics_regressions,
        args.run_url,
        args.perf_diff_url,
        args.numerics_diff_url,
        newly_disabled_configs=disabled_configs,
        newly_failing_benchmarks=newly_failing_benchmarks,
        deployment=args.deployment,
        previous_deployment=args.previous_deployment,
        context=context,
        quantized_failures=quantized_failures,
    )

    perf_count = len(perf_regressions)
    numerics_count = len(numerics_regressions)
    print(
        f"Found {perf_count} perf regression(s), "
        f"{numerics_count} numerics regression(s), "
        f"{len(disabled_configs)} newly-disabled configuration(s), "
        f"{len(newly_failing_benchmarks)} newly-failing benchmark(s)."
    )

    title = f"{args.title_prefix}{title}"
    labels = [l.strip() for l in args.labels.split(",")]
    output = {"title": title, "body": body, "labels": labels}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Issue JSON written to {args.output}")


if __name__ == "__main__":
    main()

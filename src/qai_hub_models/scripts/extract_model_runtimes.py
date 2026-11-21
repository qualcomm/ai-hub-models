# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Extract per-model wall-clock runtimes from scorecard JUnit XMLs.

Reads the three combined JUnit XMLs from a scorecard run (job submission,
export test, accuracy) and writes a YAML of per-model, per-stage seconds.
``split_torch_models`` reads this to load-balance CI splits.

Either pass ``--action-id`` to pull the artifact from a past scorecard
run via ``gh``, or pass the XML paths directly with the per-stage flags.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from qai_hub_models_cli.proto_helpers._common import dump_yaml

from qai_hub_models.scorecard.artifacts import (
    RUNTIME_ALL_STAGES,
    RUNTIME_STAGE_ACCURACY,
    RUNTIME_STAGE_EXPORT_TEST,
    RUNTIME_STAGE_JOB_SUBMISSION,
    ScorecardArtifact,
)
from qai_hub_models.utils.base_config import sorted_mappings

SCORECARD_ARTIFACT_NAME = "test-results-scorecard"
JOB_SUBMISSION_XML_NAME = "qaihm-model-tests-junit.xml"
EXPORT_TEST_XML_NAME = "qaihm-export-zip-tests-junit.xml"
ACCURACY_XML_NAME = "qaihm-device-accuracy-tests-junit.xml"

_MODEL_CLASSNAME_RE = re.compile(r"^qai_hub_models\.models\.([^.]+)\.")


def _extract_model_id(classname: str, name: str) -> str | None:
    match = _MODEL_CLASSNAME_RE.match(classname)
    if match:
        return match.group(1)
    # Some pytest configs put the dotted path in 'name'.
    match = _MODEL_CLASSNAME_RE.match(name)
    if match:
        return match.group(1)
    return None


def parse_junit_per_model_seconds(xml_path: Path) -> dict[str, float]:
    """Sum testcase ``time`` per model in one JUnit XML.

    Attribution is per-testcase (via ``classname``) rather than
    per-testsuite — the combined XMLs have many testsuites and an
    unknown one would otherwise leak time into the wrong bucket.
    """
    if not xml_path.exists():
        return {}

    totals: dict[str, float] = defaultdict(float)
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        print(f"Warning: could not parse {xml_path}: {exc}", file=sys.stderr)
        return {}
    root = tree.getroot()

    for testcase in root.iter("testcase"):
        classname = testcase.get("classname", "")
        name = testcase.get("name", "")
        model_id = _extract_model_id(classname, name)
        if model_id is None:
            continue
        time_attr = testcase.get("time", "0")
        try:
            totals[model_id] += float(time_attr)
        except ValueError:
            continue

    return dict(totals)


def build_runtime_estimates(
    job_submission_xml: Path | None,
    export_test_xml: Path | None,
    accuracy_xml: Path | None,
) -> dict[str, dict[str, float]]:
    """Return ``{model_id: {stage: seconds}}`` aggregated across the three XMLs."""
    stage_inputs: dict[str, Path | None] = {
        RUNTIME_STAGE_JOB_SUBMISSION: job_submission_xml,
        RUNTIME_STAGE_EXPORT_TEST: export_test_xml,
        RUNTIME_STAGE_ACCURACY: accuracy_xml,
    }

    per_stage: dict[str, dict[str, float]] = {}
    for stage, xml_path in stage_inputs.items():
        if xml_path is None:
            per_stage[stage] = {}
            continue
        per_stage[stage] = parse_junit_per_model_seconds(xml_path)

    all_models = set().union(*(s.keys() for s in per_stage.values()))
    estimates: dict[str, dict[str, float]] = {}
    for model_id in sorted(all_models):
        entry: dict[str, float] = {}
        for stage in RUNTIME_ALL_STAGES:
            if model_id in per_stage[stage]:
                entry[stage] = round(per_stage[stage][model_id], 1)
        estimates[model_id] = entry
    return estimates


def write_runtime_estimates_yaml(
    output_path: Path,
    estimates: dict[str, dict[str, float]],
    source_action_id: str | None = None,
) -> None:
    """Write the runtime estimates YAML in the canonical schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
    if source_action_id:
        payload["source_action_id"] = source_action_id
    payload["models"] = estimates

    with open(output_path, "w") as f:
        dump_yaml(sorted_mappings(payload), f)


_ACTION_ID_RE = re.compile(r"[0-9]+")
_REPO_RE = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")


def _download_scorecard_artifact(
    action_id: str,
    repo: str | None,
    dest: Path,
) -> None:
    """Download the ``test-results-scorecard`` artifact via ``gh run download``.

    Inputs are regex-validated before shelling out — a value starting
    with ``-`` would otherwise be parsed by ``gh`` as a flag.
    """
    if shutil.which("gh") is None:
        raise SystemExit("gh CLI not found on PATH; cannot download artifacts.")
    if not _ACTION_ID_RE.fullmatch(action_id):
        raise ValueError(f"action_id must be numeric, got: {action_id!r}")
    if repo is not None and not _REPO_RE.fullmatch(repo):
        raise ValueError(f"repo must be 'owner/name', got: {repo!r}")
    cmd = [
        "gh",
        "run",
        "download",
        "-n",
        SCORECARD_ARTIFACT_NAME,
        "-D",
        str(dest),
    ]
    if repo:
        cmd.extend(["-R", repo])
    cmd.append("--")
    cmd.append(action_id)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-id",
        default=None,
        help=(
            f"Scorecard run id. Downloads '{SCORECARD_ARTIFACT_NAME}' via gh "
            "and reads the JUnit XMLs inside. Mutually exclusive with --*-xml."
        ),
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo (owner/name) to pass to gh. Defaults to gh's auto-detection.",
    )
    parser.add_argument(
        "--job-submission-xml",
        type=Path,
        default=None,
        help="Path to combined model-tests JUnit XML (qaihm-model-tests-junit.xml).",
    )
    parser.add_argument(
        "--export-test-xml",
        type=Path,
        default=None,
        help="Path to combined export JUnit XML (qaihm-export-zip-tests-junit.xml).",
    )
    parser.add_argument(
        "--accuracy-xml",
        type=Path,
        default=None,
        help="Path to combined accuracy JUnit XML (qaihm-device-accuracy-tests-junit.xml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ScorecardArtifact.MODEL_RUNTIME_ESTIMATES.intermediates_path,
        help="Output YAML path. Defaults to the checked-in intermediates copy.",
    )
    parser.add_argument(
        "--source-action-id",
        default=None,
        help="Override the source_action_id stamped into the YAML (default: --action-id).",
    )

    args = parser.parse_args()

    explicit_xmls = (args.job_submission_xml, args.export_test_xml, args.accuracy_xml)
    if args.action_id and any(explicit_xmls):
        parser.error("--action-id cannot be combined with --*-xml path flags.")
    if not args.action_id and not any(explicit_xmls):
        parser.error(
            "Pass --action-id to download from a scorecard run, or pass at least "
            "one of --job-submission-xml/--export-test-xml/--accuracy-xml."
        )

    source_action_id = args.source_action_id or args.action_id

    if args.action_id:
        with tempfile.TemporaryDirectory(prefix="runtime-estimates-") as tmp:
            tmp_path = Path(tmp)
            _download_scorecard_artifact(args.action_id, args.repo, tmp_path)
            job_xml = tmp_path / JOB_SUBMISSION_XML_NAME
            export_xml = tmp_path / EXPORT_TEST_XML_NAME
            accuracy_xml = tmp_path / ACCURACY_XML_NAME
            estimates = build_runtime_estimates(
                job_xml if job_xml.exists() else None,
                export_xml if export_xml.exists() else None,
                accuracy_xml if accuracy_xml.exists() else None,
            )
    else:
        estimates = build_runtime_estimates(*explicit_xmls)

    if not estimates:
        print("No model runtimes recovered; not writing YAML.")
        return

    write_runtime_estimates_yaml(args.output, estimates, source_action_id)
    print(f"Wrote runtime estimates for {len(estimates)} models to {args.output}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Report models missing design assets (static banner required for publishing).

Scans:
  1. All internal-config.yaml files checked into the repo on the current branch.
  2. All open PRs in the repo for internal-config.yaml files with missing banners.

status, status_reason and has_static_banner all live in internal-config.yaml;
name, use_case and source_repo come from the model's manifest.yaml.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

import yaml

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
SCORECARD_MODELS_ROOT = Path(__file__).resolve().parents[1] / "scorecard" / "models"
CONFIG_NAME = "internal-config.yaml"


def _local_manifest(model_id: str) -> dict:
    path = MODELS_ROOT / model_id / "manifest.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


REPO = "qcom-ai-hub/ai-hub-models-internal"

BANNER_REASON_KEYWORDS = ("banner", "design asset", "static banner")
SAFE_BRANCH_RE = re.compile(r"^[\w./_+-]+$")
MAX_YAML_BYTES = 512 * 1024  # 512 KB


def _blocked_by_banners(info: dict) -> bool:
    """Return True if missing banners are the likely blocker for this model.

    - pending models missing banners: banners are a blocker (scorecard wants to
      promote but can't).
    - unpublished models: only if the status_reason explicitly mentions banners.
    """
    # Static banner is required for publishing; animated is optional.
    if info.get("has_static_banner", False):
        return False

    status = info.get("status", "")
    reason = str(info.get("status_reason") or "").lower()
    if status == "pending":
        return not reason or any(kw in reason for kw in BANNER_REASON_KEYWORDS)
    if status == "unpublished":
        return any(kw in reason for kw in BANNER_REASON_KEYWORDS)
    return False


# -----------------------------------------------------------------------------
# Local scan
# -----------------------------------------------------------------------------
def scan_local_models() -> list[dict[str, str]]:
    """Return models where missing banners are the blocker."""
    missing: list[dict[str, str]] = []
    for config_path in sorted(SCORECARD_MODELS_ROOT.glob(f"*/{CONFIG_NAME}")):
        model_id = config_path.parent.name
        with open(config_path, encoding="utf-8") as f:
            info = yaml.safe_load(f)
        if info is None:
            continue

        if _blocked_by_banners(info):
            manifest = _local_manifest(model_id)
            missing.append(
                {
                    "model_id": model_id,
                    "name": manifest.get("name", model_id),
                    "status": info.get("status", "unknown"),
                    "use_case": manifest.get("use_case", ""),
                    "source_repo": manifest.get("source_repo", ""),
                }
            )
    return missing


# -----------------------------------------------------------------------------
# Open PR scan
# -----------------------------------------------------------------------------
def scan_open_prs() -> list[dict[str, str]]:
    """Check open PRs for internal-config.yaml files where banners are the blocker."""
    missing: list[dict[str, str]] = []

    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                REPO,
                "--state",
                "open",
                "--limit",
                "200",
                "--json",
                "number,title,headRefName,files",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        logger.warning("Failed to list open PRs: %s", e)
        return missing

    prs = json.loads(result.stdout) if result.stdout.strip() else []
    seen_models: set[str] = set()

    for pr in prs:
        pr_number = pr.get("number")
        pr_title = pr.get("title", "")

        # Filter for scorecard configs from the files list (no extra subprocess)
        pr_files = pr.get("files") or []
        config_yamls = [
            f.get("path", "")
            for f in pr_files
            if f.get("path", "").endswith(CONFIG_NAME)
            and "/scorecard/models/" in f.get("path", "")
        ]

        if not config_yamls:
            continue

        # Validate branch name before using in API call
        branch = pr.get("headRefName", "")
        if not branch or not SAFE_BRANCH_RE.match(branch):
            logger.warning("Skipping PR %s: unsafe branch name %r", pr_number, branch)
            continue

        for config_path in config_yamls:
            # Path like src/qai_hub_models/scorecard/models/<model_id>/internal-config.yaml
            parts = config_path.split("/")
            try:
                models_idx = parts.index("models")
                model_id = parts[models_idx + 1]
            except (ValueError, IndexError):
                continue

            if model_id in seen_models:
                continue

            try:
                content_result = subprocess.run(
                    [
                        "gh",
                        "api",
                        f"repos/{REPO}/contents/{config_path}",
                        "-H",
                        "Accept: application/vnd.github.raw+json",
                        "--method",
                        "GET",
                        "-F",
                        f"ref={branch}",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                continue

            raw = content_result.stdout
            if len(raw.encode()) > MAX_YAML_BYTES:
                logger.warning(
                    "Skipping oversized %s for model in PR %s", CONFIG_NAME, pr_number
                )
                continue
            try:
                info = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                logger.warning("Failed to parse YAML for PR %s: %s", pr_number, exc)
                continue
            if info is None:
                continue

            if _blocked_by_banners(info):
                seen_models.add(model_id)
                # A brand-new model's manifest only exists on the PR branch, so
                # these display fields fall back to empty for it.
                manifest = _local_manifest(model_id)
                missing.append(
                    {
                        "model_id": model_id,
                        "name": manifest.get("name", model_id),
                        "status": info.get("status", "unknown"),
                        "use_case": manifest.get("use_case", ""),
                        "source_repo": manifest.get("source_repo", ""),
                        "pr": f"#{pr_number}",
                        "pr_title": pr_title,
                    }
                )
    return missing


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    logger.info("Scanning local models for missing design assets...")
    local_missing = scan_local_models()
    logger.info("Found %d local models missing both banners", len(local_missing))

    logger.info("Scanning open PRs for models missing design assets...")
    pr_missing = scan_open_prs()
    logger.info("Found %d models in open PRs missing both banners", len(pr_missing))

    # Deduplicate: if a model is in both local and a PR, skip the PR entry
    local_ids = {m["model_id"] for m in local_missing}
    pr_only = [m for m in pr_missing if m["model_id"] not in local_ids]

    def _bullet(m: dict[str, str], pr_info: bool = False) -> str:
        parts = [m["name"], m["status"]]
        if m.get("use_case"):
            parts.append(m["use_case"])
        if m.get("source_repo"):
            parts.append(m["source_repo"])
        if pr_info and m.get("pr"):
            parts.append(f"PR {m['pr']}")
        return "- " + " | ".join(parts)

    print("\n" + "=" * 80)
    print("MODELS MISSING DESIGN ASSETS (both static + animated banners are false)")
    print("=" * 80)

    if local_missing:
        print(f"\nChecked into repo ({len(local_missing)} models):\n")
        for m in local_missing:
            print(_bullet(m))
    else:
        print("\nNo models in the repo are missing both banners.")

    if pr_only:
        print(f"\nIn open PRs only ({len(pr_only)} models):\n")
        for m in pr_only:
            print(_bullet(m, pr_info=True))
    elif pr_missing:
        print("\nAll models in open PRs are also in the repo already.")
    else:
        print("\nNo models in open PRs are missing both banners.")

    total = len(local_missing) + len(pr_only)
    print(f"\nTotal: {total} models missing design assets\n")


if __name__ == "__main__":
    main()

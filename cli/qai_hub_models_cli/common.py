# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import platformdirs
from packaging.version import Version

CACHE_DIR = Path(platformdirs.user_cache_dir("qai_hub_models")) / "cli"
STORE_URL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com"
ASSET_FOLDER = "qai-hub-models/models/{model_id}/releases/v{version}"

GITHUB_REPO_URL = "https://github.com/qualcomm/ai-hub-models"
AIHUB_MODELS_URL = "https://aihub.qualcomm.com/models"


@functools.cache
def is_heavy_package_installed() -> bool:
    """Whether the heavy ``qai_hub_models`` package is importable.

    Uses ``find_spec`` rather than ``try: import`` to avoid pulling torch and
    other heavy deps into the lean CLI's process.
    """
    return importlib.util.find_spec("qai_hub_models") is not None


def model_repo_url(model_id: str, version: Version) -> str:
    """URL to the model's source code on GitHub for the given release version."""
    ref = f"v{version}" if not version.is_devrelease else "main"
    return f"{GITHUB_REPO_URL}/tree/{ref}/src/qai_hub_models/models/{model_id}"


def parse_sdk_version_filters(queries: list[str]) -> dict[str, str]:
    """Parse ``-s tool=version`` CLI args into a ``{tool: version}`` map.

    Each query uses ``tool=version`` syntax (e.g. ``"litert=1.4.4"``). Only the
    syntax is validated here; the tool name is normalized but resolved to a proto
    field later, in :func:`tool_versions_match`.

    Parameters
    ----------
    queries
        SDK version filter strings.

    Returns
    -------
    dict[str, str]
        Map of (normalized) tool name to (lower-cased) version substring.

    Raises
    ------
    ValueError
        If any query is not of the form ``tool=version``.
    """
    parsed: dict[str, str] = {}
    for query in queries:
        if "=" not in query:
            raise ValueError(
                f"Invalid SDK version filter {query!r}. "
                "Use 'tool=version' syntax, e.g. 'litert=1.4.4'."
            )
        tool, _, version = query.partition("=")
        tool = tool.strip().lower().replace("-", "_").replace(" ", "_")
        version = version.strip().lower()
        parsed[tool] = version
    return parsed


# The CLI is invoked as both ``qai_hub_models`` and ``qai-hub-models``; sample
# commands shown to the user consistently use the dash form.
CLI_NAME = "qai-hub-models"


def sample_command(*parts: str) -> str:
    """Join *parts* after the ``qai-hub-models`` program name, dropping empties.

    The single source of truth for the program name. Pass ``version_flag(version)``
    as a part to pin a release (it is ``""`` for the installed version, and empty
    parts are dropped). E.g. ``sample_command("info", model, version_flag(v))``.
    """
    return " ".join(p for p in (CLI_NAME, *parts) if p)


def _filter_flag(flag: str, values: list[str] | None, placeholder: str) -> str:
    """``flag 'v1' 'v2'`` for *values*, else ``flag placeholder``.

    Values are quoted (runtime display / chipset / device names can contain
    spaces); the placeholder is left unquoted.
    """
    if not values:
        return f"{flag} {placeholder}"
    return f"{flag} " + " ".join(f"'{v}'" for v in values)


def build_filter_command(
    command: str,
    model: str,
    version_flag: str = "",
    runtimes: list[str] | None = None,
    precisions: list[str] | None = None,
    chipsets: list[str] | None = None,
    devices: list[str] | None = None,
    show_chipset_placeholder: bool = True,
    extra_flags: list[str] | None = None,
) -> str:
    """Build a full ``qai-hub-models <command> <model> -r ... -p ... [-c/-d ...]``.

    Echoes the given filter values (or placeholders) for the ``fetch`` download
    hint and the ``perf``/``numerics`` "filter these results" hint.
    *show_chipset_placeholder* controls whether the
    ``[ -c '<chipset>' || -d '<device>' ]`` hint appears when neither is given.
    *extra_flags* are appended verbatim (e.g. ``["--url-only"]``).
    """
    if devices:
        target = "-d " + " ".join(f"'{d}'" for d in devices)
    elif chipsets:
        target = "-c " + " ".join(f"'{c}'" for c in chipsets)
    elif show_chipset_placeholder:
        target = "[ -c '<chipset>' || -d '<device>' ]"
    else:
        target = ""
    return sample_command(
        command,
        model,
        version_flag,
        _filter_flag("-r", runtimes, "<runtime>"),
        _filter_flag("-p", precisions, "<precision>"),
        target,
        *(extra_flags or []),
    )


def format_command_sections(sections: dict[str, list[tuple[str, str]]]) -> str:
    """Print pre-built commands grouped by section header, labels aligned.

    *sections* maps a section title to its ``(label, command)`` entries (commands
    already built via :func:`sample_command`). Labels are right-padded to a common
    width across all sections so the commands line up; empty sections are skipped.
    """
    width = max((len(label) for es in sections.values() for label, _ in es), default=0)
    return "\n\n".join(
        f"{title}:\n"
        + "\n".join(f"  {label + ':':<{width + 1}}  {cmd}" for label, cmd in entries)
        for title, entries in sections.items()
        if entries
    )

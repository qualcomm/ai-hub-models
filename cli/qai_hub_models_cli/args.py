# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared argparse arguments reused across CLI subcommands."""

from __future__ import annotations

import argparse
from typing import TypeVar

from packaging.version import Version
from packaging.version import parse as parse_version

from qai_hub_models_cli._version import __version__
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers.platform_enums import (
    precision_proto_to_str,
    runtime_proto_to_str,
)
from qai_hub_models_cli.proto_helpers.tool_versions import SDK_FILTER_TOOL_NAMES
from qai_hub_models_cli.versions import CURRENT_VERSION, normalize_version

T = TypeVar("T")

# Comma-separated lists of the known runtime/precision/sdk tokens, computed once
# and reused in every filter's help text.
RUNTIME_VALUES = ", ".join(
    runtime_proto_to_str(r)
    for r in Runtime.values()
    if r != Runtime.RUNTIME_UNSPECIFIED
)
PRECISION_VALUES = ", ".join(
    precision_proto_to_str(p)
    for p in Precision.values()
    if p != Precision.PRECISION_UNSPECIFIED
)
KNOWN_SDKS = ", ".join(SDK_FILTER_TOOL_NAMES)


def parse_version_arg(s: str) -> Version:
    """Argparse type function: normalize and parse a version string."""
    return parse_version(normalize_version(s))


def flatten_multi_arg(value: list[list[T]] | None) -> list[T] | None:
    """Flatten an ``action="append", nargs="+"`` value (list-of-lists) into one list."""
    if not value:
        return None
    return [item for group in value for item in group]


def add_version_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``-v/--version`` argument (stored as ``qaihm_version``)."""
    parser.add_argument(
        "-v",
        "--version",
        default=CURRENT_VERSION,
        type=parse_version_arg,
        dest="qaihm_version",
        help=f"AI Hub Models version tag (e.g. v0.45.0 or 0.45.0). Default: {__version__}.",
    )


def add_quiet_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the shared ``-q/--quiet`` flag with a command-specific help string."""
    parser.add_argument("-q", "--quiet", action="store_true", help=help_text)


def add_sdk_version_arg(parser: argparse.ArgumentParser, subject: str) -> None:
    """Add the shared ``-s/--sdk-version`` filter (*subject* names the matched thing)."""
    parser.add_argument(
        "-s",
        "--sdk-version",
        nargs="+",
        default=None,
        type=str.lower,
        help="Filter by SDK/tool version using 'tool=version' syntax (e.g. "
        f"'litert=1.4.4' or 'qairt=2.20'). Accepts multiple values; {subject} "
        f"must match all of them. Known SDKs: {KNOWN_SDKS}.",
    )


def add_asset_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the single-value asset filters shared by ``fetch`` and ``find``."""
    parser.add_argument(
        "-r",
        "--runtime",
        default=None,
        help=f"Target runtime. Known values: {RUNTIME_VALUES}. "
        "Older releases may support different values.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        default=None,
        type=str.lower,
        help=f"Model precision. Known values: {PRECISION_VALUES}. "
        "Older releases may support different values.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "-c",
        "--chipset",
        default=None,
        type=str.lower,
        help="Chipset name for device-specific (AOT compiled) runtimes. "
        "Run `qai-hub-models chipsets` to see supported chipsets.",
    )
    target.add_argument(
        "-d",
        "--device",
        default=None,
        help="Device name for device-specific (AOT compiled) runtimes. "
        "Run `qai-hub-models devices` to see supported devices. Cannot be specified with chipset.",
    )
    add_sdk_version_arg(parser, "an asset")


def add_model_metric_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the multi-value filters shared by ``perf`` and ``numerics``.

    Each flag accepts multiple values and a record matches any of them; read the
    parsed values with :func:`flatten_multi_arg`.
    """
    parser.add_argument(
        "-r",
        "--runtime",
        nargs="+",
        action="append",
        default=None,
        help="Filter by runtime(s); a record matches any of them. "
        "May be repeated or given multiple values. "
        f"Known values: {RUNTIME_VALUES}.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        nargs="+",
        action="append",
        default=None,
        type=str.lower,
        help="Filter by precision(s); a record matches any of them. "
        "May be repeated or given multiple values. "
        f"Known values: {PRECISION_VALUES}.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "-c",
        "--chipset",
        nargs="+",
        action="append",
        default=None,
        type=str.lower,
        help="Filter by chipset(s); a record matches any of them. "
        "May be repeated or given multiple values. "
        "Run `qai-hub-models chipsets` to see supported chipsets.",
    )
    target.add_argument(
        "-d",
        "--device",
        nargs="+",
        action="append",
        default=None,
        help="Filter by device(s); a record matches any of them. "
        "May be repeated or given multiple values. "
        "Run `qai-hub-models devices` to see supported devices. "
        "Cannot be combined with --chipset.",
    )
    add_sdk_version_arg(parser, "a record")


def add_json_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--json`` flag for machine-readable output."""
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output results as JSON for programmatic consumption.",
    )


def add_chipset_attribute_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the chipset-attribute filters shared by the devices/chipsets commands."""
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Only show entries whose chipset supports fp16.",
    )
    parser.add_argument(
        "--htp-version",
        nargs="+",
        action="append",
        type=int,
        default=None,
        help="Filter by chipset HTP version(s).",
    )
    parser.add_argument(
        "--soc-model",
        nargs="+",
        action="append",
        type=int,
        default=None,
        help="Filter by chipset SoC model(s).",
    )

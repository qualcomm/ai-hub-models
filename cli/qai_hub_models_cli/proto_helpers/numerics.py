# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from collections.abc import Iterable
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli.proto.numerics_pb2 import ModelNumerics
from qai_hub_models_cli.proto.platform_pb2 import PlatformInfo, RuntimeInfo
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers._common import fetch_model_proto
from qai_hub_models_cli.proto_helpers.platform import device_names_for_filter
from qai_hub_models_cli.proto_helpers.platform_enums import (
    precision_proto_to_str,
    precisions_str_to_proto_set,
    runtime_proto_to_str,
    runtimes_str_to_proto_set,
)
from qai_hub_models_cli.proto_helpers.tool_versions import (
    format_tool_versions,
    tool_versions_match,
    validate_sdk_tools,
)
from qai_hub_models_cli.utils import build_table
from qai_hub_models_cli.versions import CURRENT_VERSION


@functools.lru_cache(maxsize=1)
def get_model_numerics(
    model: str,
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> ModelNumerics:
    """
    Fetch and cache the model numerics protobuf for a given model.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``) or display name
        (e.g. ``"MobileNet-v2"``).
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local numerics protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    ModelNumerics
        Parsed model numerics protobuf containing per-device numerical
        accuracy metrics and benchmark values.

    Raises
    ------
    KeyError
        If *model* is not found in the manifest for *version*.
    """
    return fetch_model_proto(
        model,
        version,
        ModelNumerics,
        cache_filename="numerics.pb",
        manifest_url_field="numerics",
        source_getter="get_numerics_proto",
        local_path=local_path,
    )


def filter_numerics(
    numerics: ModelNumerics,
    platform: PlatformInfo,
    runtime: Runtime.ValueType | str | list[Runtime.ValueType | str] | None = None,
    precision: Precision.ValueType
    | str
    | list[Precision.ValueType | str]
    | None = None,
    chipset: str | list[str] | None = None,
    device: str | list[str] | None = None,
    sdk_versions: dict[str, str] | None = None,
) -> ModelNumerics:
    """
    Return a copy of *numerics* keeping only per-device results matching the filters.

    The filters apply to each metric's per-device results; a metric is dropped
    entirely if none of its device results match. Any filter left as ``None`` is
    not applied. *chipset* and *device* are mutually exclusive: *device* matches
    the named device(s), while *chipset* matches every device with the named
    chipset(s).

    Parameters
    ----------
    numerics
        The model's numerics to filter.
    platform
        Platform registry used to resolve *runtime* display names and
        *chipset*/*device*.
    runtime
        Runtime enum value or string to filter on (e.g. ``"tflite"``), or a list
        of them; a result matches if its runtime is any of them.
    precision
        Precision enum value or string to filter on (e.g. ``"float"``), or a list
        of them; a result matches if its precision is any of them.
    chipset
        Chipset reference (canonical ID, name, or alias) to filter on, or a list
        of them. Mutually exclusive with *device*.
    device
        Device name to filter on, or a list of them. Mutually exclusive with
        *chipset*.
    sdk_versions
        Map of tool name to version substring (see
        :func:`parse_sdk_version_filters`). A result must match every entry; its
        tool versions are cross-referenced from perf at build time (empty for
        older releases, which then never match an SDK filter).

    Returns
    -------
    ModelNumerics
        A new ``ModelNumerics`` keeping only matching per-device results.

    Raises
    ------
    ValueError
        If both *chipset* and *device* are provided.
    KeyError
        If *runtime*, *chipset*, or *device* is not known.
    """
    if sdk_versions:
        validate_sdk_tools(sdk_versions)
    runtime_vals = runtimes_str_to_proto_set(runtime, platform.runtimes)
    precision_vals = precisions_str_to_proto_set(precision)
    device_names = device_names_for_filter(
        platform.chipsets, platform.devices, chipset, device
    )

    filtered = ModelNumerics(
        aihm_version=numerics.aihm_version,
        model_id=numerics.model_id,
    )
    for metric in numerics.metrics:
        matching = [
            dm
            for dm in metric.device_metrics
            if (runtime_vals is None or dm.runtime in runtime_vals)
            and (precision_vals is None or dm.precision in precision_vals)
            and (device_names is None or dm.device.lower() in device_names)
            and (
                not sdk_versions or tool_versions_match(dm.tool_versions, sdk_versions)
            )
        ]
        if not matching:
            continue
        new_metric = filtered.metrics.add()
        new_metric.CopyFrom(metric)
        del new_metric.device_metrics[:]
        new_metric.device_metrics.extend(matching)
    return filtered


def format_numerics_table(
    numerics: ModelNumerics,
    title: str | None = "Numerics (Accuracy)",
    runtimes: Iterable[RuntimeInfo] | None = None,
) -> str:
    """Format a model's numerical accuracy metrics as a table.

    One row per (metric, device result). The torch reference value for each
    metric is shown alongside each on-device value for comparison. The SDK
    Versions column is sourced from each result's tool versions (cross-referenced
    from perf at build time).

    *runtimes* (``platform.runtimes``), when provided, is used to render each
    runtime by its human display name (e.g. ``TensorFlow Lite``) instead of its
    token.

    Returns a message string when *numerics* has no per-device results.
    """
    columns = [
        "Dataset",
        "Metric",
        "Precision",
        "Runtime",
        "Device",
        "Accuracy",
        "Torch Ref",
        "SDK Versions",
    ]

    rows = [
        [
            metric.dataset_name,
            metric.metric_name,
            precision_proto_to_str(dm.precision),
            runtime_proto_to_str(dm.runtime, runtimes, display_name=True),
            dm.device,
            f"{dm.partial_metric:.3g}{metric.metric_unit}",
            f"{metric.partial_torch_metric:.3g}{metric.metric_unit}",
            format_tool_versions(dm.tool_versions),
        ]
        for metric in numerics.metrics
        for dm in metric.device_metrics
    ]

    if not rows:
        return "No numerics match the given filters."
    return build_table(
        columns,
        rows,
        wrap_column="SDK Versions",
        title=title,
        wrap_on_commas=True,
    )

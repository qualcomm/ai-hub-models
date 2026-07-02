# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli.proto.perf_pb2 import ModelPerf
from qai_hub_models_cli.proto.platform_pb2 import PlatformInfo
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.range_pb2 import DoubleRange, IntRange
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
def get_model_perf(
    model: str,
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> ModelPerf:
    """
    Fetch and cache the model perf protobuf for a given model.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``) or display name
        (e.g. ``"MobileNet-v2"``).
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local perf protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    ModelPerf
        Parsed model perf protobuf containing per-device performance
        metrics such as inference time, memory usage, and layer counts.

    Raises
    ------
    KeyError
        If *model* is not found in the manifest for *version*.
    """
    return fetch_model_proto(
        model,
        version,
        ModelPerf,
        cache_filename="perf.pb",
        manifest_url_field="perf",
        source_getter="get_perf_proto",
        local_path=local_path,
    )


def filter_perf(
    perf: ModelPerf,
    platform: PlatformInfo,
    runtime: Runtime.ValueType | str | list[Runtime.ValueType | str] | None = None,
    precision: Precision.ValueType
    | str
    | list[Precision.ValueType | str]
    | None = None,
    chipset: str | list[str] | None = None,
    device: str | list[str] | None = None,
    sdk_versions: dict[str, str] | None = None,
    components: list[str] | None = None,
) -> ModelPerf:
    """
    Return a copy of *perf* keeping only performance records matching the filters.

    Any filter left as ``None`` is not applied. *chipset* and *device* are
    mutually exclusive: *device* matches the named device(s), while *chipset*
    matches every device with the named chipset(s).

    Parameters
    ----------
    perf
        The model's performance metrics to filter.
    platform
        Platform registry used to resolve *runtime* display names and
        *chipset*/*device*.
    runtime
        Runtime enum value or string to filter on (e.g. ``"tflite"``), or a list
        of them; a record matches if its runtime is any of them.
    precision
        Precision enum value or string to filter on (e.g. ``"float"``), or a list
        of them; a record matches if its precision is any of them.
    chipset
        Chipset reference (canonical ID, name, or alias) to filter on, or a list
        of them. Mutually exclusive with *device*.
    device
        Device name to filter on, or a list of them. Mutually exclusive with
        *chipset*.
    sdk_versions
        Map of tool name to version substring (see
        :func:`parse_sdk_version_filters`). A record must match every entry;
        records without tool versions never match.
    components
        Component names to filter on (case-insensitive); a record matches if its
        component is any of them. For multi-component models. Records without a
        component never match.

    Returns
    -------
    ModelPerf
        A new ``ModelPerf`` with the same metadata and only the records that
        match every provided filter.

    Raises
    ------
    ValueError
        If both *chipset* and *device* are provided.
    KeyError
        If *runtime*, *chipset*, *device*, or a component is not known.
    """
    if sdk_versions:
        validate_sdk_tools(sdk_versions)
    runtime_vals = runtimes_str_to_proto_set(runtime, platform)
    precision_vals = precisions_str_to_proto_set(precision)
    device_names = device_names_for_filter(platform, chipset, device)

    component_set = {c.lower() for c in components} if components else None
    if component_set is not None:
        known = {
            r.component for r in perf.performance_metrics if r.HasField("component")
        }
        if unknown := component_set - {c.lower() for c in known}:
            valid = ", ".join(sorted(known)) or "(this model has no components)"
            raise KeyError(
                f"Unknown component(s): {', '.join(sorted(unknown))}. "
                f"Valid components: {valid}."
            )

    filtered = ModelPerf(
        aihm_version=perf.aihm_version,
        model_id=perf.model_id,
        supported_devices=perf.supported_devices,
        supported_chipsets=perf.supported_chipsets,
    )
    for record in perf.performance_metrics:
        if runtime_vals is not None and record.runtime not in runtime_vals:
            continue
        if precision_vals is not None and record.precision not in precision_vals:
            continue
        if device_names is not None and record.device.lower() not in device_names:
            continue
        if component_set is not None and (
            not record.HasField("component")
            or record.component.lower() not in component_set
        ):
            continue
        if sdk_versions and not (
            record.HasField("tool_versions")
            and tool_versions_match(record.tool_versions, sdk_versions)
        ):
            continue
        filtered.performance_metrics.add().CopyFrom(record)
    return filtered


def _format_int_range(r: IntRange) -> str:
    """Format an ``IntRange`` as ``min-max`` (or a single value if min == max)."""
    has_min, has_max = r.HasField("min"), r.HasField("max")
    if has_min and has_max:
        return str(r.min) if r.min == r.max else f"{r.min}-{r.max}"
    if has_min:
        return str(r.min)
    if has_max:
        return str(r.max)
    return ""


def _format_double_range(r: DoubleRange) -> str:
    """Format a ``DoubleRange`` as ``min-max`` (or a single value if min == max)."""
    has_min, has_max = r.HasField("min"), r.HasField("max")
    if has_min and has_max:
        return f"{r.min:.1f}" if r.min == r.max else f"{r.min:.1f}-{r.max:.1f}"
    if has_min:
        return f"{r.min:.1f}"
    if has_max:
        return f"{r.max:.1f}"
    return ""


def _sdk_versions_str(record: ModelPerf.PerformanceDetails) -> str:
    """Render an asset's tool versions, or ``—`` when unset."""
    return (
        format_tool_versions(record.tool_versions)
        if record.HasField("tool_versions")
        else "—"
    )


def format_perf_table(
    perf: ModelPerf,
    title: str | None = "Performance",
    platform: PlatformInfo | None = None,
) -> str:
    """Format a model's performance metrics as one or two tables.

    Standard (non-LLM) records and LLM records have different metrics, so they
    are rendered as separate tables joined by a blank line. The ``Component``
    column is shown only when the model has 2+ distinct components (a single
    component adds a column of identical values, so it's omitted).

    *platform*, when provided, is used to render each runtime by its human
    display name (e.g. ``TensorFlow Lite``) instead of its token.

    Returns a message string when *perf* has no records.
    """
    # A record carries either standard metrics or LLM metrics; guard against a
    # record with both so it can't appear in both tables.
    standard = [
        r
        for r in perf.performance_metrics
        if r.HasField("metrics") and not r.llm_metrics
    ]
    llm = [r for r in perf.performance_metrics if r.llm_metrics]
    components = {
        r.component for r in perf.performance_metrics if r.HasField("component")
    }
    show_component = len(components) >= 2

    # Precompute runtime -> display name once; runtime_proto_to_str otherwise
    # rescans platform.runtimes for every record.
    runtime_names: dict[int, str] = {}
    # "Similar" devices (those with a reference_chipset) borrow their metrics;
    # mark them with a "*" and explain it in a footnote.
    similar_device_names: set[str] = set()
    if platform is not None:
        runtime_names = {
            rt.runtime: rt.display_name for rt in platform.runtimes if rt.display_name
        }
        similar_device_names = {
            d.name.lower() for d in platform.devices if d.reference_chipset
        }

    def _runtime_name(runtime: Runtime.ValueType) -> str:
        return runtime_names.get(runtime) or runtime_proto_to_str(runtime)

    marked_similar = False

    def _device_cell(device: str) -> str:
        """Device name, suffixed with "*" when it is a "similar" device."""
        nonlocal marked_similar
        if device.lower() in similar_device_names:
            marked_similar = True
            return f"{device} *"
        return device

    tables: list[str] = []

    if standard:
        columns = ["Precision", "Runtime", "Device"]
        if show_component:
            columns.append("Component")
        columns += [
            "Inference (ms)",
            "Peak Memory (MB)",
            "Compute Unit",
            "SDK Versions",
        ]
        rows = []
        for r in standard:
            m = r.metrics
            row = [
                precision_proto_to_str(r.precision),
                _runtime_name(r.runtime),
                _device_cell(r.device),
            ]
            if show_component:
                row.append(r.component if r.HasField("component") else "")
            row += [
                f"{m.inference_time_milliseconds:.2f}",
                _format_int_range(m.estimated_peak_memory_range_mb)
                if m.HasField("estimated_peak_memory_range_mb")
                else "",
                m.primary_compute_unit,
                _sdk_versions_str(r),
            ]
            rows.append(row)
        tables.append(
            build_table(
                columns,
                rows,
                wrap_column="SDK Versions",
                title=title,
                wrap_on_commas=True,
            )
        )

    if llm:
        columns = ["Precision", "Runtime", "Device"]
        if show_component:
            columns.append("Component")
        columns += [
            "Context Len",
            "Compute Unit",
            "Tokens/sec",
            "Time to First Token (ms)",
            "Prefill Tokens/sec",
            "SDK Versions",
        ]
        rows = []
        for r in llm:
            for lm in r.llm_metrics:
                row = [
                    precision_proto_to_str(r.precision),
                    _runtime_name(r.runtime),
                    _device_cell(r.device),
                ]
                if show_component:
                    row.append(r.component if r.HasField("component") else "")
                row += [
                    str(lm.context_length),
                    # desired_compute_unit is unset on records from releases
                    # before 0.57.0; assume npu in that case.
                    (lm.desired_compute_unit or "npu").upper(),
                    f"{lm.tokens_per_second:.1f}",
                    _format_double_range(lm.time_to_first_token_range_milliseconds)
                    if lm.HasField("time_to_first_token_range_milliseconds")
                    else "",
                    f"{lm.prefill_tokens_per_second:.1f}"
                    if lm.HasField("prefill_tokens_per_second")
                    else "",
                    _sdk_versions_str(r),
                ]
                rows.append(row)
        tables.append(
            build_table(
                columns,
                rows,
                wrap_column="SDK Versions",
                title="LLM Performance" if standard else title,
                wrap_on_commas=True,
            )
        )

    if not tables:
        return "No performance metrics match the given filters."
    output = "\n\n".join(tables)
    if marked_similar:
        output += (
            "\n\n* Devices marked with '*' in the Device column above are "
            "'similar' devices (not tested directly). Their metrics are copied "
            "from a reference device that serves as a substitute compilation "
            "target. Run `qai-hub-models devices` to see each similar device's "
            "reference."
        )
    return output

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from qai_hub_models_cli.proto import numerics_pb2
from qai_hub_models_cli.proto.shared import range_pb2, tool_versions_pb2
from qai_hub_models_cli.proto_helpers._common import (
    read_proto_yaml,
    write_proto_yaml,
)

from qai_hub_models import Precision
from qai_hub_models.configs.proto_helpers import (
    precision_from_proto,
    precision_to_proto,
    runtime_from_proto,
    runtime_to_proto,
)
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

# Key identifying a measured path, shared by perf and numerics: a numerics result
# and a perf record describe the same measurement when these three match.
_PathKey = tuple[str, int, int]


def _perf_tool_versions_by_path(
    perf: QAIHMModelPerf | None,
) -> dict[_PathKey, tool_versions_pb2.ToolVersions]:
    """Map each perf entry's (device, precision, runtime) to its tool versions.

    Numerics is produced without tool-version info, so it borrows the versions
    from the matching perf record when building the release proto.
    """
    if perf is None:
        return {}
    versions: dict[_PathKey, tool_versions_pb2.ToolVersions] = {}

    def _collect(
        precision: Precision,
        component: str,
        device: ScorecardDevice,
        path: ScorecardProfilePath,
        details: QAIHMModelPerf.PerformanceDetails,
    ) -> None:
        key = (
            str(device),
            precision_to_proto(precision),
            runtime_to_proto(path.runtime),
        )
        versions[key] = details.tool_versions.to_proto()

    perf.for_each_entry(_collect)
    return versions


def get_numerics_yaml_path(model_id: str) -> Path:
    return QAIHM_MODELS_ROOT / model_id / "numerics.yaml"


class QAIHMModelNumerics(BaseQAIHMConfig):
    class DeviceDetails(BaseQAIHMConfig):
        partial_metric: float
        # Set only for runtimes measured on several compute units.
        compute_unit: str | None = None

    class Range(BaseQAIHMConfig):
        min: float | None = None
        max: float | None = None

    class MetricDetails(BaseQAIHMConfig):
        dataset_name: str
        dataset_link: str
        dataset_split_description: str
        metric_name: str
        metric_description: str
        metric_unit: str
        metric_range: QAIHMModelNumerics.Range
        # False for error/distance metrics (WER, NME, perplexity, ...), where a
        # smaller metric is a better model.
        higher_is_better: bool = True
        # Maximum allowed deviation between torch and device accuracy, or
        # between actual accuracy and the benchmark value.
        # Paths exceeding this threshold are disabled by the scorecard.
        metric_enablement_threshold: float | None = None
        # Expected accuracy from info.yaml's numerics_benchmark.
        benchmark_value: float | None = None
        num_partial_samples: int
        partial_torch_metric: float
        device_metric: dict[
            ScorecardDevice,
            dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ],
        ] = Field(default_factory=dict)

        @classmethod
        def from_proto(
            cls, proto: numerics_pb2.ModelNumerics.NumericsMetric
        ) -> QAIHMModelNumerics.MetricDetails:
            # Insert in (device, precision, path) name order, the order the old
            # name-keyed YAML loaded in, so the release proto's order is unchanged.
            results = sorted(
                (
                    (
                        ScorecardDevice.parse(dm.device),
                        precision_from_proto(dm.precision),
                        ScorecardProfilePath.from_runtime(
                            runtime_from_proto(dm.runtime)
                        ),
                        dm,
                    )
                    for dm in proto.device_metrics
                ),
                key=lambda r: (str(r[0]), str(r[1]), str(r[2])),
            )
            device_metric: dict[
                ScorecardDevice,
                dict[
                    Precision,
                    dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails],
                ],
            ] = {}
            for device, precision, path, dm in results:
                device_metric.setdefault(device, {}).setdefault(precision, {})[path] = (
                    QAIHMModelNumerics.DeviceDetails(
                        partial_metric=dm.partial_metric,
                        compute_unit=(
                            dm.compute_unit if dm.HasField("compute_unit") else None
                        ),
                    )
                )

            return cls(
                dataset_name=proto.dataset_name,
                dataset_link=proto.dataset_link,
                dataset_split_description=proto.dataset_split_description,
                metric_name=proto.metric_name,
                metric_description=proto.metric_description,
                metric_unit=proto.metric_unit,
                metric_range=QAIHMModelNumerics.Range(
                    min=(
                        proto.metric_range.min
                        if proto.metric_range.HasField("min")
                        else None
                    ),
                    max=(
                        proto.metric_range.max
                        if proto.metric_range.HasField("max")
                        else None
                    ),
                ),
                higher_is_better=(
                    proto.higher_is_better
                    if proto.HasField("higher_is_better")
                    else True
                ),
                metric_enablement_threshold=(
                    proto.metric_enablement_threshold
                    if proto.HasField("metric_enablement_threshold")
                    else None
                ),
                benchmark_value=(
                    proto.benchmark_value if proto.HasField("benchmark_value") else None
                ),
                num_partial_samples=proto.num_partial_samples,
                partial_torch_metric=proto.partial_torch_metric,
                device_metric=device_metric,
            )

    metrics: list[MetricDetails] = Field(default_factory=list)

    def to_model_yaml(self, model_id: str) -> Path:
        path = get_numerics_yaml_path(model_id)
        self.to_yaml(path)
        return path

    def is_empty(self) -> bool:
        return len(self.metrics) == 0

    def drop_entries_in_scope(
        self,
        scope: set[tuple[Precision, ScorecardProfilePath, ScorecardDevice]],
    ) -> None:
        """Delete every entry whose (precision, path, device) key is in scope."""
        by_device: dict[
            ScorecardDevice, set[tuple[Precision, ScorecardProfilePath]]
        ] = {}
        for precision, path, device in scope:
            by_device.setdefault(device, set()).add((precision, path))
        surviving_metrics: list[QAIHMModelNumerics.MetricDetails] = []
        for metric in self.metrics:
            for device, in_scope in by_device.items():
                prec_dict = metric.device_metric.get(device)
                if prec_dict is None:
                    continue
                empty_precisions: list[Precision] = []
                for precision, path_dict in prec_dict.items():
                    for path in [p for p in path_dict if (precision, p) in in_scope]:
                        path_dict.pop(path, None)
                    if not path_dict:
                        empty_precisions.append(precision)
                for precision in empty_precisions:
                    prec_dict.pop(precision, None)
                if not prec_dict:
                    metric.device_metric.pop(device, None)
            if metric.device_metric:
                surviving_metrics.append(metric)
        self.metrics = surviving_metrics

    def to_proto(
        self,
        aihm_version: str,
        model_id: str,
        perf: QAIHMModelPerf | None = None,
    ) -> numerics_pb2.ModelNumerics:
        # SDK/tool versions aren't stored in numerics; cross-reference them from
        # the matching perf record (same device/precision/runtime), when given.
        tool_versions_by_path = _perf_tool_versions_by_path(perf)
        metrics: list[numerics_pb2.ModelNumerics.NumericsMetric] = []
        for m in self.metrics:
            device_metrics: list[
                numerics_pb2.ModelNumerics.NumericsMetric.DeviceNumericsMetrics
            ] = []
            for device, prec_dict in m.device_metric.items():
                for precision, path_dict in prec_dict.items():
                    for path, details in path_dict.items():
                        precision_proto = precision_to_proto(precision)
                        runtime_proto = runtime_to_proto(path.runtime)
                        tool_versions = tool_versions_by_path.get(
                            (str(device), precision_proto, runtime_proto)
                        )
                        device_metrics.append(
                            numerics_pb2.ModelNumerics.NumericsMetric.DeviceNumericsMetrics(
                                device=str(device),
                                precision=precision_proto,
                                runtime=runtime_proto,
                                partial_metric=details.partial_metric,
                                tool_versions=tool_versions,
                                compute_unit=details.compute_unit,
                            )
                        )

            metric_range = range_pb2.DoubleRange(
                min=m.metric_range.min, max=m.metric_range.max
            )
            metrics.append(
                numerics_pb2.ModelNumerics.NumericsMetric(
                    dataset_name=m.dataset_name,
                    dataset_link=m.dataset_link,
                    dataset_split_description=m.dataset_split_description,
                    metric_name=m.metric_name,
                    metric_description=m.metric_description,
                    metric_unit=m.metric_unit,
                    metric_range=metric_range,
                    higher_is_better=m.higher_is_better,
                    metric_enablement_threshold=m.metric_enablement_threshold,
                    benchmark_value=m.benchmark_value,
                    num_partial_samples=m.num_partial_samples,
                    partial_torch_metric=m.partial_torch_metric,
                    device_metrics=device_metrics,
                )
            )

        return numerics_pb2.ModelNumerics(
            aihm_version=aihm_version,
            model_id=model_id,
            metrics=metrics,
        )

    @classmethod
    def from_proto(cls, proto: numerics_pb2.ModelNumerics) -> QAIHMModelNumerics:
        """Build the in-memory representation from a ``ModelNumerics`` proto."""
        return cls(metrics=[cls.MetricDetails.from_proto(m) for m in proto.metrics])

    @classmethod
    def from_yaml(
        cls, path: str | os.PathLike, create_empty_if_no_file: bool = False
    ) -> QAIHMModelNumerics:
        """Read a numerics.yaml, which is the YAML form of a ``ModelNumerics`` proto."""
        if create_empty_if_no_file and (
            not os.path.exists(path) or os.path.getsize(path) == 0
        ):
            return cls()
        return cls.from_proto(read_proto_yaml(path, numerics_pb2.ModelNumerics))

    def to_yaml(self, path: str | os.PathLike, **kwargs: Any) -> bool:  # type: ignore[override]
        """
        Write this numerics to ``path`` as the YAML form of a ``ModelNumerics`` proto.

        If there are no metrics, any existing file at ``path`` is deleted instead.

        Parameters
        ----------
        path
            Where to write the file.
        **kwargs
            Unused. Accepted so the signature matches ``BaseQAIHMConfig.to_yaml``.

        Returns
        -------
        written : bool
            True if the file was written.
        """
        path = Path(path)
        if self.is_empty():
            path.unlink(missing_ok=True)
            return False
        proto = self.to_proto("", "")
        for metric in proto.metrics:
            metric.device_metrics.sort(
                key=lambda d: (d.device, d.precision, d.runtime, d.compute_unit)
            )
        write_proto_yaml(proto, path)
        return True

    @classmethod
    def from_model(
        cls: type[QAIHMModelNumerics], model_id: str, not_exists_ok: bool = False
    ) -> QAIHMModelNumerics | None:
        numerics_path = get_numerics_yaml_path(model_id)
        if not_exists_ok and not os.path.exists(numerics_path):
            return None
        return cls.from_yaml(numerics_path)

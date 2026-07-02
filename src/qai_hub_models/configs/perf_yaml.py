# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path

from pydantic import Field
from qai_hub_models_cli.proto import perf_pb2
from qai_hub_models_cli.proto.shared import range_pb2

from qai_hub_models import Precision
from qai_hub_models.configs.proto_helpers import precision_to_proto, runtime_to_proto
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.scorecard import ScorecardDevice, ScorecardProfilePath
from qai_hub_models.scorecard.results.chipset_helpers import sorted_chipsets
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


@cache
def _similar_device_names() -> frozenset[str]:
    """Names of "similar" (non-workbench) devices excluded from the perf proto."""
    from qai_hub_models.configs.devices_and_chipsets_yaml import (
        _load_similar_devices_raw,
    )

    return frozenset(_load_similar_devices_raw().devices.keys())


class QAIHMModelPerf(BaseQAIHMConfig):
    """Schema for perf.yaml files."""

    class PerformanceDetails(BaseQAIHMConfig):
        class TimeToFirstTokenRangeMilliseconds(BaseQAIHMConfig):
            min: float
            max: float

        class LLMMetricsPerContextLength(BaseQAIHMConfig):
            """LLM performance metrics for a specific context length."""

            context_length: int
            tokens_per_second: float
            time_to_first_token_range_milliseconds: (
                QAIHMModelPerf.PerformanceDetails.TimeToFirstTokenRangeMilliseconds
            )
            prefill_tokens_per_second: float | None = None

            # Compute unit these metrics are intended to run on ("npu", "gpu",
            # or "cpu"). Required so it is always written out; most models use
            # "npu", while runtimes that target other compute units (e.g.
            # geniex_llamacpp) set it accordingly.
            desired_compute_unit: str

        class PeakMemoryRangeMB(BaseQAIHMConfig):
            min: int
            max: int

            @staticmethod
            def from_bytes(
                mmin: int, mmax: int
            ) -> QAIHMModelPerf.PerformanceDetails.PeakMemoryRangeMB:
                return QAIHMModelPerf.PerformanceDetails.PeakMemoryRangeMB(
                    min=round(mmin / (1 << 20)),
                    max=round(mmax / (1 << 20)),
                )

        class LayerCounts(BaseQAIHMConfig):
            total: int
            npu: int = 0
            gpu: int = 0
            cpu: int = 0

            @staticmethod
            def from_layers(
                npu: int = 0, gpu: int = 0, cpu: int = 0
            ) -> QAIHMModelPerf.PerformanceDetails.LayerCounts:
                return QAIHMModelPerf.PerformanceDetails.LayerCounts(
                    total=npu + gpu + cpu,
                    npu=npu,
                    gpu=gpu,
                    cpu=cpu,
                )

            @property
            def primary_compute_unit(self) -> str:
                if self.npu == 0 and self.gpu == 0 and self.cpu == 0:
                    return "null"
                compute_unit_for_most_layers = max(self.cpu, self.gpu, self.npu)
                if compute_unit_for_most_layers == self.npu:
                    return "NPU"
                if compute_unit_for_most_layers == self.gpu:
                    return "GPU"
                return "CPU"

        # Only set for non-LLMs.
        job_id: str | None = None
        job_status: str | None = None

        # Only set for successful non-LLM jobs.
        inference_time_milliseconds: float | None = None
        estimated_peak_memory_range_mb: (
            QAIHMModelPerf.PerformanceDetails.PeakMemoryRangeMB | None
        ) = None
        primary_compute_unit: str | None = None
        layer_counts: QAIHMModelPerf.PerformanceDetails.LayerCounts | None = None

        # The tool versions used by the profile jobs to execute this model.
        # All jobs will include QAIRT version, + the inference engine version used (tflite, onnx, etc.)
        tool_versions: ToolVersions = Field(default_factory=ToolVersions)

        # Only set for LLMs.
        llm_metrics: (
            list[QAIHMModelPerf.PerformanceDetails.LLMMetricsPerContextLength] | None
        ) = None

    class ComponentDetails(BaseQAIHMConfig):
        precision: Precision | None = (
            None  # Used to clarify each component's precision when a component model is mixed precision
        )
        performance_metrics: dict[
            ScorecardDevice,
            dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails],
        ] = Field(default_factory=dict)

    class PrecisionDetails(BaseQAIHMConfig):
        components: dict[str, QAIHMModelPerf.ComponentDetails] = Field(
            default_factory=dict
        )

    supported_devices: list[ScorecardDevice] = Field(default_factory=list)
    supported_chipsets: list[str] = Field(default_factory=list)
    precisions: dict[Precision, QAIHMModelPerf.PrecisionDetails] = Field(
        default_factory=dict
    )

    @property
    def empty(self) -> bool:
        return (
            not self.supported_chipsets
            and not self.supported_devices
            and not self.precisions
        )

    def apply_similar_devices(self, mapping: dict[str, tuple[str, list[str]]]) -> None:
        """
        Duplicate performance entries from supported devices to similar unsupported devices.

        Parameters
        ----------
        mapping
            Dict of unsupported_device_name -> (real_chipset, list of reference device names).
            The first reference device with results is used as the perf source. The
            real_chipset is added to ``supported_chipsets`` when perf data is placed.
        """
        similar_device_objs = {
            name: ScorecardDevice(name, name, register=False) for name in mapping
        }

        # Names this call actually copied perf data onto. Only these drive the
        # chipset-insertion below; pre-existing entries keep whatever
        # supported_chipsets already had (avoids "fix-up" surprises and keeps
        # the function purely additive on its own outputs).
        newly_placed: set[str] = set()

        for precision_details in self.precisions.values():
            for component in precision_details.components.values():
                existing_devices = {str(d): d for d in component.performance_metrics}

                for unsupported_name, (_, reference_names) in mapping.items():
                    if unsupported_name in existing_devices:
                        continue

                    for name in reference_names:
                        if name in existing_devices:
                            component.performance_metrics[
                                similar_device_objs[unsupported_name]
                            ] = copy.copy(
                                component.performance_metrics[existing_devices[name]]
                            )
                            newly_placed.add(unsupported_name)
                            break

        # Build reverse lookup: reference_device_name -> [unsupported_names]
        reference_to_similar: dict[str, list[str]] = {}
        for unsupported_name, (_, reference_names) in mapping.items():
            for name in reference_names:
                reference_to_similar.setdefault(name, []).append(unsupported_name)

        existing_names = {str(d) for d in self.supported_devices}
        new_devices: list[ScorecardDevice] = []
        inserted: set[str] = set()
        for device in self.supported_devices:
            new_devices.append(device)
            for unsupported_name in reference_to_similar.get(str(device), []):
                if (
                    unsupported_name not in existing_names
                    and unsupported_name not in inserted
                ):
                    new_devices.append(similar_device_objs[unsupported_name])
                    inserted.add(unsupported_name)
        self.supported_devices = new_devices

        # Add the real chipset for each newly-placed similar device.
        existing_chipsets = set(self.supported_chipsets)
        new_chipsets = {
            mapping[name][0]
            for name in newly_placed
            if mapping[name][0] not in existing_chipsets
        }
        if new_chipsets:
            self.supported_chipsets = sorted_chipsets(existing_chipsets | new_chipsets)

    def for_each_entry(
        self,
        callback: Callable[
            [
                Precision,
                str,
                ScorecardDevice,
                ScorecardProfilePath,
                QAIHMModelPerf.PerformanceDetails,
            ],
            bool | None,
        ],
        include_paths: list[ScorecardProfilePath] | None = None,
    ) -> None:
        """
        Walk over each valid perf.yaml job entry and call the callback.

        Parameters
        ----------
        callback
            A function to call for each perf.yaml job entry.
            Func Params:
                precision: Precision
                    The precision for this entry,
                component: str
                    Component name. Will be Model Name if there is 1 component.
                device: ScorecardDevice,
                    Device for this entry.
                path: ScorecardProfilePath
                    Path for this entry.
                QAIHMModelPerf.PerformanceDetails
                    Actual entry perf data (includes llm_metrics for LLMs)

            Func Returns:
                Boolean or None.
                If None or True, for_each_entry continues to walk over more entries.
                If False, for_each_entry will stop walking over additional entries.

        include_paths
            Scorecard Profile Paths to loop over. If None, uses all enabled paths.
        """
        for precision, precision_perf in self.precisions.items():
            for component_name, component_detail in precision_perf.components.items():
                for (
                    device,
                    device_detail,
                ) in component_detail.performance_metrics.items():
                    for path, profile_perf_details in device_detail.items():
                        if include_paths and path not in include_paths:
                            continue
                        res = callback(
                            precision,
                            component_name,
                            device,
                            path,
                            profile_perf_details,
                        )
                        # Note that res may be None. We ignore the return value in that case.
                        if res is False:
                            # If res is explicitly false, stop and return
                            return

    @classmethod
    def from_model(
        cls: type[QAIHMModelPerf], model_id: str, not_exists_ok: bool = False
    ) -> QAIHMModelPerf:
        perf_path = QAIHM_MODELS_ROOT / model_id / "perf.yaml"
        if not_exists_ok and not os.path.exists(perf_path):
            return QAIHMModelPerf()
        return cls.from_yaml(perf_path)

    def to_model_yaml(self, model_id: str) -> Path:
        out = QAIHM_MODELS_ROOT / model_id / "perf.yaml"
        self.to_yaml(out)
        return out

    def to_proto(
        self,
        aihm_version: str,
        model_id: str,
        exclude_similar_devices: bool = True,
    ) -> perf_pb2.ModelPerf:
        """Serialize this perf data to a ``ModelPerf`` proto.

        When *exclude_similar_devices* is True (default), "similar" devices
        (those from similar_devices.yaml, whose perf is borrowed from a reference
        device rather than measured) are dropped from both ``performance_metrics``
        and ``supported_devices``.
        """
        perf_details: list[perf_pb2.ModelPerf.PerformanceDetails] = []
        similar_devices = (
            _similar_device_names() if exclude_similar_devices else frozenset()
        )

        def _collect(
            precision: Precision,
            component: str,
            device: ScorecardDevice,
            path: ScorecardProfilePath,
            details: QAIHMModelPerf.PerformanceDetails,
        ) -> None:
            if str(device) in similar_devices:
                return
            profile_job = None
            if details.job_id is not None:
                profile_job = perf_pb2.ModelPerf.ProfileJob(
                    id=details.job_id,
                    status=details.job_status or "",
                )

            metrics = None
            if details.inference_time_milliseconds is not None:
                layer_counts = None
                if details.layer_counts is not None:
                    layer_counts = perf_pb2.ModelPerf.PerfMetrics.LayerCounts(
                        total=details.layer_counts.total,
                        npu=details.layer_counts.npu,
                        gpu=details.layer_counts.gpu,
                        cpu=details.layer_counts.cpu,
                    )
                mem_range = None
                if details.estimated_peak_memory_range_mb is not None:
                    mem_range = range_pb2.IntRange(
                        min=details.estimated_peak_memory_range_mb.min,
                        max=details.estimated_peak_memory_range_mb.max,
                    )
                metrics = perf_pb2.ModelPerf.PerfMetrics(
                    inference_time_milliseconds=details.inference_time_milliseconds,
                    estimated_peak_memory_range_mb=mem_range,
                    primary_compute_unit=details.primary_compute_unit or "",
                    layer_counts=layer_counts,
                )

            llm_metrics = []
            if details.llm_metrics:
                for lm in details.llm_metrics:
                    ttft = None
                    if lm.time_to_first_token_range_milliseconds:
                        ttft = range_pb2.DoubleRange(
                            min=lm.time_to_first_token_range_milliseconds.min,
                            max=lm.time_to_first_token_range_milliseconds.max,
                        )
                    llm_metrics.append(
                        perf_pb2.ModelPerf.LLMPerfMetrics(
                            context_length=lm.context_length,
                            tokens_per_second=lm.tokens_per_second,
                            time_to_first_token_range_milliseconds=ttft,
                            prefill_tokens_per_second=lm.prefill_tokens_per_second,
                            desired_compute_unit=lm.desired_compute_unit,
                        )
                    )

            perf_details.append(
                perf_pb2.ModelPerf.PerformanceDetails(
                    precision=precision_to_proto(precision),
                    component=component,
                    device=str(device),
                    runtime=runtime_to_proto(path.runtime),
                    tool_versions=details.tool_versions.to_proto(),
                    profile_job=profile_job,
                    metrics=metrics,
                    llm_metrics=llm_metrics,
                )
            )

        self.for_each_entry(_collect)

        return perf_pb2.ModelPerf(
            aihm_version=aihm_version,
            model_id=model_id,
            supported_devices=[
                str(d) for d in self.supported_devices if str(d) not in similar_devices
            ],
            supported_chipsets=self.supported_chipsets,
            performance_metrics=perf_details,
        )

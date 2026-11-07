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
from typing import Any

from pydantic import Field
from qai_hub_models_cli.proto import perf_pb2
from qai_hub_models_cli.proto.shared import range_pb2
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
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.scorecard import ScorecardDevice, ScorecardProfilePath
from qai_hub_models.scorecard.device import compute_peer_chipsets
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.chipset_helpers import (
    get_supported_devices,
    sorted_chipsets,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


@cache
def _similar_device_names() -> frozenset[str]:
    """Names of "similar" (non-workbench) devices excluded from the perf proto."""
    from qai_hub_models.scorecard.devices_and_chipsets_yaml import (
        _load_similar_devices_raw,
    )

    return frozenset(_load_similar_devices_raw().devices.keys())


class QAIHMModelPerf(BaseQAIHMConfig):
    """Schema for perf.yaml files."""

    class PerformanceDetails(BaseQAIHMConfig):
        class LLMMetricsPerContextLength(BaseQAIHMConfig):
            """LLM performance metrics for a specific context length."""

            context_length: int
            tokens_per_second: float
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

        @classmethod
        def from_proto(
            cls, proto: perf_pb2.ModelPerf.PerformanceDetails
        ) -> QAIHMModelPerf.PerformanceDetails:
            details = cls(
                tool_versions=ToolVersions.from_proto(proto.tool_versions),
            )
            if proto.HasField("profile_job"):
                details.job_id = proto.profile_job.id or None
                details.job_status = proto.profile_job.status or None
            if proto.HasField("metrics"):
                metrics = proto.metrics
                details.inference_time_milliseconds = (
                    metrics.inference_time_milliseconds
                )
                details.primary_compute_unit = metrics.primary_compute_unit or None
                if metrics.HasField("estimated_peak_memory_range_mb"):
                    details.estimated_peak_memory_range_mb = (
                        QAIHMModelPerf.PerformanceDetails.PeakMemoryRangeMB(
                            min=metrics.estimated_peak_memory_range_mb.min,
                            max=metrics.estimated_peak_memory_range_mb.max,
                        )
                    )
                if metrics.HasField("layer_counts"):
                    details.layer_counts = (
                        QAIHMModelPerf.PerformanceDetails.LayerCounts(
                            total=metrics.layer_counts.total,
                            npu=metrics.layer_counts.npu,
                            gpu=metrics.layer_counts.gpu,
                            cpu=metrics.layer_counts.cpu,
                        )
                    )
            if proto.llm_metrics:
                details.llm_metrics = [
                    QAIHMModelPerf.PerformanceDetails.LLMMetricsPerContextLength(
                        context_length=m.context_length,
                        tokens_per_second=m.tokens_per_second,
                        prefill_tokens_per_second=(
                            m.prefill_tokens_per_second
                            if m.HasField("prefill_tokens_per_second")
                            else None
                        ),
                        desired_compute_unit=m.desired_compute_unit,
                    )
                    for m in proto.llm_metrics
                ]
            return details

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

        Any pre-existing similar-device entry is overwritten from the first reference
        device that currently has results, so a re-run against a perf.yaml already
        populated by a prior call refreshes the borrowed metrics rather than leaving
        them frozen at the earlier values. This matters for the LLM pipeline, where
        apply_llm_perf_updates.py loads the committed perf.yaml before calling this.

        Parameters
        ----------
        mapping
            Dict of unsupported_device_name -> (real_chipset, list of reference device names).
            The first reference device with results is used as the perf source.
            real_chipset is added to ``supported_chipsets`` whenever the similar
            device lands in ``supported_devices``.
        """
        similar_device_objs = {
            name: ScorecardDevice(name=name, reference_device_name=name, register=False)
            for name in mapping
        }

        for precision_details in self.precisions.values():
            for component in precision_details.components.values():
                existing_devices = {str(d): d for d in component.performance_metrics}

                for unsupported_name, (_, reference_names) in mapping.items():
                    for name in reference_names:
                        if name in existing_devices:
                            component.performance_metrics[
                                similar_device_objs[unsupported_name]
                            ] = copy.copy(
                                component.performance_metrics[existing_devices[name]]
                            )
                            break

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

        # Every similar device in supported_devices contributes its chipset.
        present_similar = {str(d) for d in self.supported_devices} & set(mapping)
        existing_chipsets = set(self.supported_chipsets)
        new_chipsets = {
            mapping[name][0]
            for name in present_similar
            if mapping[name][0] not in existing_chipsets
        }
        if new_chipsets:
            self.supported_chipsets = sorted_chipsets(existing_chipsets | new_chipsets)

    def apply_compute_peer_chipsets(self) -> None:
        """
        Credit interchangeable compute chipsets (see ``compute_peer_chipsets``).

        Additive only: devices already listed are kept, so QDC-only devices absent
        from the Hub device query survive. The non-LLM scorecard writer gets this
        from ``ScorecardDevice.extended_supported_chipsets``; the LLM pipeline needs
        it applied separately, and must not pick up that method's mobile proxying.
        """
        existing_chipsets = set(self.supported_chipsets)
        peers = {c for chip in existing_chipsets for c in compute_peer_chipsets(chip)}
        if not (new_chipsets := peers - existing_chipsets):
            return

        self.supported_chipsets = sorted_chipsets(peers)
        existing_names = {str(d) for d in self.supported_devices}
        # Appended rather than re-sorted: supported_devices may hold similar devices
        # that are absent from Hub, and sorting resolves each chipset through Hub.
        self.supported_devices += [
            d
            for d in get_supported_devices(new_chipsets)
            if str(d) not in existing_names
        ]

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

    def drop_entries_in_scope(
        self,
        scope: set[tuple[Precision, ScorecardProfilePath, ScorecardDevice]],
        only_components: set[str] | None = None,
    ) -> None:
        """
        Delete every entry whose (precision, path, device) key is in scope.

        Parameters
        ----------
        scope
            The (precision, path, device) keys to drop.
        only_components
            Restrict the drop to these component names. A hybrid LLM's perf.yaml has two
            writers -- the QDC pipeline owns the consolidated backbone entry and the
            scorecard owns the standalone components -- and each must only clear its own,
            or whichever runs second deletes the other's results.
        """
        for precision, path, device in scope:
            precision_details = self.precisions.get(precision)
            if precision_details is None:
                continue
            empty_components: list[str] = []
            for (
                component_name,
                component_details,
            ) in precision_details.components.items():
                if (
                    only_components is not None
                    and component_name not in only_components
                ):
                    continue
                device_metrics = component_details.performance_metrics.get(device)
                if device_metrics is None:
                    continue
                device_metrics.pop(path, None)
                if not device_metrics:
                    component_details.performance_metrics.pop(device, None)
                if not component_details.performance_metrics:
                    empty_components.append(component_name)
            for name in empty_components:
                precision_details.components.pop(name, None)
            if not precision_details.components:
                self.precisions.pop(precision, None)

    def drop_chipsets_without_assets(self, model_id: str) -> None:
        """
        Drop claimed chipsets this model ships no context binary for.

        Subtractive only, and a no-op when the model publishes a universal
        asset: that one file runs on any chipset via CPU fallback, so its reach
        isn't enumerable from the asset list. Call before
        ``apply_similar_devices``, which re-adds the borrowed devices on top.

        Parameters
        ----------
        model_id
            Model whose ``release-assets.yaml`` lists the published assets.
        """
        assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
        asset_chipsets: set[str] = set()
        for precision_details in assets.precisions.values():
            if precision_details.universal_assets:
                return
            for chipset in precision_details.chipset_assets:
                asset_chipsets |= compute_peer_chipsets(chipset)
        if not asset_chipsets:
            return

        supported_chipsets = set(self.supported_chipsets) & asset_chipsets
        self.supported_chipsets = sorted_chipsets(supported_chipsets)
        self.supported_devices = get_supported_devices(supported_chipsets)

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
            if details.job_id is not None or details.job_status is not None:
                profile_job = perf_pb2.ModelPerf.ProfileJob(
                    id=details.job_id or "",
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

            llm_metrics = [
                perf_pb2.ModelPerf.LLMPerfMetrics(
                    context_length=lm.context_length,
                    tokens_per_second=lm.tokens_per_second,
                    prefill_tokens_per_second=lm.prefill_tokens_per_second,
                    desired_compute_unit=lm.desired_compute_unit,
                )
                for lm in details.llm_metrics or []
            ]

            component_precision = (
                self.precisions[precision].components[component].precision
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
                    component_precision=(
                        precision_to_proto(component_precision)
                        if component_precision is not None
                        else None
                    ),
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

    @classmethod
    def from_proto(cls, proto: perf_pb2.ModelPerf) -> QAIHMModelPerf:
        """Build the in-memory representation from a ``ModelPerf`` proto."""
        precisions: dict[Precision, QAIHMModelPerf.PrecisionDetails] = {}
        for entry in proto.performance_metrics:
            components = precisions.setdefault(
                precision_from_proto(entry.precision), cls.PrecisionDetails()
            ).components
            component = components.setdefault(entry.component, cls.ComponentDetails())
            if entry.HasField("component_precision"):
                component.precision = precision_from_proto(entry.component_precision)
            path = ScorecardProfilePath.from_runtime(runtime_from_proto(entry.runtime))
            component.performance_metrics.setdefault(
                ScorecardDevice.parse(entry.device), {}
            )[path] = cls.PerformanceDetails.from_proto(entry)

        return cls(
            supported_devices=[
                ScorecardDevice.parse(d) for d in proto.supported_devices
            ],
            supported_chipsets=list(proto.supported_chipsets),
            precisions=precisions,
        )

    @classmethod
    def from_yaml(
        cls, path: str | os.PathLike, create_empty_if_no_file: bool = False
    ) -> QAIHMModelPerf:
        """Read a perf.yaml, which is the YAML form of a ``ModelPerf`` proto."""
        if create_empty_if_no_file and (
            not os.path.exists(path) or os.path.getsize(path) == 0
        ):
            return cls()
        return cls.from_proto(read_proto_yaml(path, perf_pb2.ModelPerf))

    def to_yaml(self, path: str | os.PathLike, **kwargs: Any) -> bool:  # type: ignore[override]
        """
        Write this perf to ``path`` as the YAML form of a ``ModelPerf`` proto.

        Entries are sorted so regenerated files have a stable order. If this perf
        is empty, any existing file at ``path`` is deleted instead.

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
        if self.empty:
            path.unlink(missing_ok=True)
            return False
        proto = self.to_proto("", "", exclude_similar_devices=False)
        for entry in proto.performance_metrics:
            if not entry.tool_versions.ListFields():
                entry.ClearField("tool_versions")
        proto.performance_metrics.sort(
            key=lambda e: (e.precision, e.component, e.device, e.runtime)
        )
        write_proto_yaml(proto, path)
        return True

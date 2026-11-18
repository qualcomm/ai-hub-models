# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import dataclasses
import json
from typing import Any, NamedTuple

from prettytable import PrettyTable
from pydantic import ConfigDict, Field, model_validator
from qai_hub_models_cli.proto import perf_pb2

from qai_hub_models import Precision
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.scorecard.params import ScJobParams
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    ToolVersionChange,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.device import FormFactor

InferenceInfo = tuple[
    str,  # Model ID
    Precision,
    str,  # Component Name (same as Model ID if model does not have multiple components)
    ScorecardDevice,
    ScorecardProfilePath,
    float,  # prev inference time (millisecs)
    float,  # new inference time (millisecs),
    float,  # inference time diff
    str,  # New Profile Job ID
    str,  # Previous Profile Job ID
    str,  # New Compile Job ID
    str,  # Previous Compile Job ID
]


class SevereRegression(BaseQAIHMConfig):
    """A single 2x+ regression entry from perf-regressions-2x.json.

    Shared struct between the writer (get_severe_regressions / dump_severe_regressions_json)
    and readers (detect_scorecard_trends, file_scorecard_regression_issue).

    Field aliases match PrettyTable column names used in the JSON files.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model_id: str = Field(default="", alias="Model ID")
    precision: str = Field(default="", alias="Precision")
    component: str = Field(default="", alias="Component")
    device: str = Field(default="", alias="Device")
    runtime: str = Field(default="", alias="Runtime")
    prev_inference_time: str = Field(default="", alias="Prev Inference time")
    new_inference_time: str = Field(default="", alias="New Inference time")
    factor: str = Field(default="", alias="Kx slower")
    job_id: str = Field(default="", alias="Job ID")
    previous_job_id: str = Field(default="", alias="Previous Job ID")
    compile_job_id: str = Field(default="", alias="Compile Job ID")
    previous_compile_job_id: str = Field(default="", alias="Previous Compile Job ID")

    @model_validator(mode="before")
    @classmethod
    def _normalize_dynamic_job_id(cls, values: Any) -> Any:
        """Rename suffixed job-id keys (e.g. "Job ID (prod)") to their bare form.

        Historical JSON artifacts and PrettyTable headers tag job-id columns
        with the deployment env ("(prod)", "(dev)", "(staging)"). The field
        aliases use the bare form, so re-route any suffixed key to its
        canonical name before pydantic validates. Longer prefixes are tried
        first so "Previous Job ID (prod)" isn't greedily matched as "Job ID".
        """
        if not isinstance(values, dict):
            return values
        canonicals = (
            "Previous Compile Job ID",
            "Previous Job ID",
            "Compile Job ID",
            "Job ID",
        )
        for canonical in canonicals:
            if canonical in values:
                continue
            for key in list(values.keys()):
                if key.startswith(canonical):
                    values[canonical] = values.pop(key)
                    break
        return values

    @property
    def key(self) -> str:
        """Unique identity for dedup/trend detection."""
        return f"{self.model_id}|{self.precision}|{self.component}|{self.device}|{self.runtime}"


class NPUNotPrimaryCUInfo(NamedTuple):
    model_id: str  # Model ID
    precision: Precision
    component: str  # Component Name (same as Model ID if model does not have multiple components)
    path: ScorecardProfilePath
    cpu_count: int  # cpu compute unit count
    gpu_count: int  # gpu compute unit count
    npu_count: int  # npu compute unit count
    same_as_last_week: str  # True if last week's run also had non-NPU as primary CU, N/A if no run from last week.


class PerformanceDiff:
    """
    Generates Performance Difference between of two 'performance_metrics' from perf.yaml

    Perf summary is generated w.r.t 'perf_buckets' to summarize difference in decreasing order
        - "INF" -> Inference failure toggled.
        - 10 -> Speedup difference >= 10 and so on ...

    Why use speedup difference?
        - Speedup is relative to baseline measured with similar constraints and changes
        - Speedup difference gives a generate sense on the how Tetra performance diverged w.r.t. baseline at that point

    What all to capture in the summary (Summary of Interest) ?
        1. Inferences that started to fail or work (Speedup = "INF")
        2. Speedup difference >= 0.1 (check models closely from higher buckets)
        3. Missing devices (new runs missing data for certain devices)
        4. New models (models with new perf.yamls)
        5. Empty perf reports (models with no passing jobs)
    """

    # Regressions with absolute diff at or below this threshold (ms) are excluded
    _MIN_REGRESSION_MS = 1.0

    def __init__(
        self,
        current_compile_jobs: CompileScorecardJobYaml | None = None,
        previous_compile_jobs: CompileScorecardJobYaml | None = None,
    ) -> None:
        # Compile job ID lookup yamls. Empty defaults are safe —
        # _compile_job_id_for() returns "null" on miss.
        self._current_compile_jobs = current_compile_jobs or CompileScorecardJobYaml()
        self._previous_compile_jobs = previous_compile_jobs or CompileScorecardJobYaml()

        # Perf buckets to track
        self.perf_buckets: list[float] = [
            float("inf"),
            10,
            5,
            2,
            1.5,
        ]

        self.empty_models: list[str] = []

        self.missing_models: list[str] = []
        self.new_models: list[str] = []

        self.missing_precisions: list[tuple[str, Precision]] = []
        self.new_precisions: list[tuple[str, Precision]] = []

        self.missing_components: list[tuple[str, Precision, str]] = []
        self.new_components: list[tuple[str, Precision, str]] = []

        self.missing_devices: list[tuple[str, Precision, str, ScorecardDevice]] = []
        self.new_devices: list[tuple[str, Precision, str, ScorecardDevice]] = []

        self.progressions: dict[float | str, list[InferenceInfo]] = {
            x: [] for x in self.perf_buckets
        }
        self.regressions: dict[float | str, list[InferenceInfo]] = {
            x: [] for x in self.perf_buckets
        }

        # Map<Info -> # of devices affected>
        self.npu_not_primary_cu: dict[NPUNotPrimaryCUInfo, int] = {}

        # Exclude automotive devices (too noisy for regression tracking)
        yaml_config = DevicesAndChipsetsYaml.load()
        self._excluded_device_names: set[str] = {
            name
            for name, details in yaml_config.devices.items()
            if details.form_factor == FormFactor.AUTO
        }

    @staticmethod
    def _format_speedup(num: float | None) -> str | float:
        if not num:
            return "null"
        return float(format(num, ".5f"))

    def _compile_job_id_for(
        self,
        compile_jobs: CompileScorecardJobYaml,
        model_id: str,
        key: QAIHMModelPerf.Key,
    ) -> str:
        """Return the compile-job ID for this perf entry, or "null" if missing.

        The linkifier renders "null" as N/A.

        Single-component models are stored in the cache with component=None
        (see ScorecardJobYaml.update_from_export_output), while perf.yaml
        surfaces a display component name for the same rows. Fall back to a
        component-less lookup when the component-keyed one misses so those
        rows still get a link in the regression issue.
        """
        params = ScJobParams(
            model_id=model_id,
            path=key.path,
            precision=key.precision,
            device=key.device,
            component=key.component,
        )
        if not params.has_compile_job:
            return "null"
        if job_id := compile_jobs.get_job_id(params):
            return job_id
        params_no_component = dataclasses.replace(params, component=None)
        if job_id := compile_jobs.get_job_id(params_no_component):
            return job_id
        return "null"

    def update_summary(
        self,
        model_id: str,
        previous_report: QAIHMModelPerf | None,
        new_report: QAIHMModelPerf | None,
    ) -> None:
        """
        Record the differences between a model's previous and new perf reports.

        Precisions, components and devices are each reported as missing (only in the
        previous report) or new (only in the new report). A component is only
        reported if its precision is in both reports, and a device only if its
        component is in both. That way a new precision is listed once, not again
        for every component and device inside it.
        """
        if new_report is None and previous_report is None:
            return
        if previous_report is None:
            self.new_models.append(model_id)
            return
        if new_report is None:
            self.missing_models.append(model_id)
            return
        if new_report.empty:
            self.empty_models.append(model_id)
            return

        previous_entries = previous_report.entries
        new_entries = new_report.entries

        previous_devices = {
            (k.precision, k.component, k.device) for k in previous_entries
        }
        new_devices = {(k.precision, k.component, k.device) for k in new_entries}
        shared_devices = previous_devices & new_devices

        previous_components = {(prec, comp) for prec, comp, _ in previous_devices}
        new_components = {(prec, comp) for prec, comp, _ in new_devices}
        shared_components = previous_components & new_components

        previous_precisions = {prec for prec, _ in previous_components}
        new_precisions = {prec for prec, _ in new_components}
        shared_precisions = previous_precisions & new_precisions

        self.missing_precisions.extend(
            (model_id, prec) for prec in previous_precisions - new_precisions
        )
        self.new_precisions.extend(
            (model_id, prec) for prec in new_precisions - previous_precisions
        )
        self.missing_components.extend(
            (model_id, prec, comp)
            for prec, comp in previous_components - new_components
            if prec in shared_precisions
        )
        self.new_components.extend(
            (model_id, prec, comp)
            for prec, comp in new_components - previous_components
            if prec in shared_precisions
        )
        self.missing_devices.extend(
            (model_id, prec, comp, device)
            for prec, comp, device in previous_devices - new_devices
            if (prec, comp) in shared_components and not self._is_excluded(device)
        )
        self.new_devices.extend(
            (model_id, prec, comp, device)
            for prec, comp, device in new_devices - previous_devices
            if (prec, comp) in shared_components and not self._is_excluded(device)
        )

        no_entry = perf_pb2.ModelPerf.PerformanceDetails()
        for key in previous_entries.keys() | new_entries.keys():
            prec, comp, device = key.precision, key.component, key.device
            if self._is_excluded(device) or (prec, comp, device) not in shared_devices:
                continue

            prev_entry = previous_entries.get(key, no_entry)
            new_entry = new_entries.get(key, no_entry)

            # LLM entries have no inference time to compare.
            if prev_entry.llm_metrics or new_entry.llm_metrics:
                continue

            prev_inference_time = prev_entry.metrics.inference_time_milliseconds
            new_inference_time = new_entry.metrics.inference_time_milliseconds
            if prev_inference_time and new_inference_time:
                progression_speedup = prev_inference_time / new_inference_time
                regression_speedup = new_inference_time / prev_inference_time
                is_progression = progression_speedup >= 1
                diff = progression_speedup if is_progression else regression_speedup
            elif prev_inference_time:
                is_progression = False
                diff = float("inf")
            elif new_inference_time:
                is_progression = True
                diff = float("inf")
            else:
                # Neither report has a result for this path, so there is nothing to compare.
                continue

            # A regression this small is within measurement noise.
            is_noise = (
                not is_progression
                and prev_inference_time
                and new_inference_time
                and abs(new_inference_time - prev_inference_time)
                <= self._MIN_REGRESSION_MS
            )
            if not is_noise:
                bucket = next((b for b in self.perf_buckets if diff >= b), None)
                if bucket is not None:
                    append_to = (
                        self.progressions if is_progression else self.regressions
                    )
                    append_to[bucket].append(
                        (
                            model_id,
                            prec,
                            comp,
                            device,
                            key.path,
                            prev_inference_time or float("-inf"),
                            new_inference_time or float("-inf"),
                            diff,
                            new_entry.profile_job.id or "null",
                            prev_entry.profile_job.id or "null",
                            self._compile_job_id_for(
                                self._current_compile_jobs, model_id, key
                            ),
                            self._compile_job_id_for(
                                self._previous_compile_jobs, model_id, key
                            ),
                        )
                    )

            new_metrics = new_entry.metrics
            if (
                new_metrics.primary_compute_unit
                and new_metrics.primary_compute_unit != "NPU"
            ):
                # Count identical results together so several devices make one entry.
                info = NPUNotPrimaryCUInfo(
                    model_id,
                    prec,
                    comp,
                    key.path,
                    new_metrics.layer_counts.cpu,
                    new_metrics.layer_counts.gpu,
                    new_metrics.layer_counts.npu,
                    "N/A"
                    if not prev_entry.metrics.primary_compute_unit
                    else str(prev_entry.metrics.primary_compute_unit != "NPU"),
                )
                self.npu_not_primary_cu[info] = self.npu_not_primary_cu.get(info, 0) + 1

    def _is_excluded(self, device: ScorecardDevice) -> bool:
        """True if this device is left out of the performance diff."""
        return str(device) in self._excluded_device_names

    def _get_summary_table(
        self, bucket_id: float | str, get_progressions: bool = True
    ) -> PrettyTable:
        """
        Returns Summary Table for given bucket.

        Parameters
        ----------
        bucket_id
            Bucket ID from perf_buckets.
        get_progressions
            If True, returns progressions table. If False, returns regressions table.

        Returns
        -------
        table : PrettyTable
            Summary table for the given bucket.
        """
        # Column names are canonical (no env suffix). The issue-body builder
        # re-tags them with the deployment of each side at render time, since
        # 'previous' isn't always prod (e.g. dev scorecard runs compare against
        # the same dev branch's prior run).
        table = PrettyTable(
            [
                "Model ID",
                "Precision",
                "Component",
                "Device",
                "Runtime",
                "Prev Inference time",
                "New Inference time",
                "Kx faster" if get_progressions else "Kx slower",
                "Job ID",
                "Previous Job ID",
                "Compile Job ID",
                "Previous Compile Job ID",
            ]
        )
        data = self.progressions if get_progressions else self.regressions
        rows = data[bucket_id]
        rows.sort(key=lambda k: k[2])  # sort by component name
        table.add_rows(rows)
        return table

    def _has_perf_changes(self) -> bool:
        """Returns True if there are perf changes"""
        return bool(
            self.new_components
            or self.missing_components
            or self.new_devices
            or self.missing_devices
            or self.new_models
            or self.missing_models
            or self.new_precisions
            or self.missing_precisions
            or self.progressions
            or self.regressions
        )

    def dump_summary(
        self,
        summary_file_path: str,
        toolchain_changes: list[ToolVersionChange] | None = None,
    ) -> None:
        """Dumps Perf change summary captured so far to the provided path.

        Parameters
        ----------
        summary_file_path
            Output path for the summary text file.
        toolchain_changes
            Optional toolchain version changes since the previous scorecard.
            Section is omitted when None or empty.
        """
        with open(summary_file_path, "w") as sf:
            sf.write("================= Perf Change Summary =================")
            if toolchain_changes:
                sf.write(
                    "\n\n----------------- Toolchain Version Changes -----------------\n"
                )
                table = PrettyTable(["Path", "Tool", "Previous", "Current"])
                table.add_rows(
                    [(c.path, c.tool, c.previous, c.current) for c in toolchain_changes]
                )
                sf.write(str(table))
            if self._has_perf_changes():
                sf.write("\n\n----------------- Regressions -----------------\n")
                # Dumps Point 1 and 2 from Summary of Interest
                # 1. Inferences that started to fail (Speedup = "INF")
                # 2. Slower than previous run
                for bucket in self.perf_buckets:
                    if len(self.regressions[bucket]) > 0:
                        sf.write(
                            f"\n----------------- >= {bucket}x slower -----------------\n"
                        )
                        sf.write(
                            str(self._get_summary_table(bucket, get_progressions=False))
                        )

                sf.write("\n\n----------------- Progressions -----------------\n")

                # Dumps Point 1 and 2 from Summary of Interest
                # 1. Inferences that started to work (Speedup = "INF")
                # 2. Faster than previous run
                for bucket in self.perf_buckets:
                    if len(self.progressions[bucket]) > 0:
                        sf.write(
                            f"\n----------------- >= {bucket}x faster -----------------\n"
                        )
                        sf.write(str(self._get_summary_table(bucket)))
            else:
                sf.write("\nNo significant changes observed.")

            if len(self.npu_not_primary_cu) > 0:
                sf.write(
                    "\n----------------- NPU is not Primary Compute Unit -----------------\n"
                )
                table = PrettyTable(
                    [
                        "Model ID",
                        "Precision",
                        "Component",
                        "Runtime",
                        "CPU Layers",
                        "GPU Layers",
                        "NPU Layers",
                        "Same as last week?",
                        "# of Devices",
                    ]
                )
                table.add_rows([(*x, y) for x, y in self.npu_not_primary_cu.items()])
                sf.write(str(table))

            if len(self.missing_models) > 0:
                sf.write("\n----------------- Missing Models -----------------\n")
                table = PrettyTable(["Model ID"])
                table.add_rows([x] for x in self.missing_models)
                sf.write(str(table))

            if len(self.missing_precisions) > 0:
                sf.write("\n----------------- Missing Precisions -----------------\n")
                table = PrettyTable(["Model ID", "Precision"])
                table.add_rows(self.missing_precisions)
                sf.write(str(table))

            if len(self.missing_components) > 0:
                sf.write("\n----------------- Missing Components -----------------\n")
                table = PrettyTable(["Model ID", "Precision", "Component"])
                table.add_rows(self.missing_components)
                sf.write(str(table))

            if len(self.missing_devices) > 0:
                sf.write("\n----------------- Missing Devices -----------------\n")
                table = PrettyTable(["Model ID", "Precision", "Component", "Device"])
                table.add_rows(self.missing_devices)
                sf.write(str(table))

            if len(self.new_models) > 0:
                sf.write("\n----------------- New Models -----------------\n")
                table = PrettyTable(["Model ID"])
                table.add_rows([x] for x in self.new_models)
                sf.write(str(table))

            if len(self.new_precisions) > 0:
                sf.write("\n----------------- New Precisions -----------------\n")
                table = PrettyTable(["Model ID", "Precision"])
                table.add_rows(self.new_precisions)
                sf.write(str(table))

            if len(self.new_components) > 0:
                # 3. Missing devices (New runs missing data for certain devices)
                sf.write("\n----------------- New Components -----------------\n")
                table = PrettyTable(["Model ID", "Precision", "Component"])
                table.add_rows(self.new_components)
                sf.write(str(table))

            if len(self.new_devices) > 0:
                # 3. Missing devices (New runs missing data for certain devices)
                sf.write("\n----------------- New Devices -----------------\n")
                table = PrettyTable(["Model ID", "Precision", "Component", "Device"])
                table.add_rows(self.new_devices)
                sf.write(str(table))

            if len(self.empty_models) > 0:
                # 5. Empty reports (Models with no passing jobs)
                sf.write(
                    "\n----------------- Empty reports (No passing jobs) -----------------\n"
                )
                table = PrettyTable(["Model ID"])
                table.add_rows([x] for x in self.empty_models)
                sf.write(str(table))

        print(f"Perf change summary written to {summary_file_path}")

    def get_severe_regressions(self, min_factor: float = 2.0) -> list[SevereRegression]:
        """Return regressions at or above min_factor as SevereRegression structs.

        Parameters
        ----------
        min_factor
            Minimum regression factor to include (default 2.0).

        Returns
        -------
        list[SevereRegression]
            Regressions at or above min_factor.
        """
        results: list[SevereRegression] = []
        for bucket in self.perf_buckets:
            if bucket < min_factor:
                continue
            table = self._get_summary_table(bucket, get_progressions=False)
            field_names = table.field_names
            for row in self.regressions[bucket]:
                d = {k: str(v) for k, v in zip(field_names, row, strict=False)}
                results.append(SevereRegression.model_validate(d))
        return results

    def dump_severe_regressions_json(
        self, json_path: str, min_factor: float = 2.0
    ) -> None:
        """Write 2x+ regressions to a JSON file for downstream consumption."""
        data = self.get_severe_regressions(min_factor)
        serialized = [r.model_dump(by_alias=True) for r in data]
        with open(json_path, "w") as f:
            json.dump(serialized, f, indent=2)
        print(f"Severe regressions JSON written to {json_path} ({len(data)} entries)")

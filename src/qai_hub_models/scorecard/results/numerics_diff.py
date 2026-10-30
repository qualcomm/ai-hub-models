# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json

from prettytable import PrettyTable

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import (
    NumericsAccuracyBenchmark,
    QAIHMModelManifest,
)
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.scorecard.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.scorecard.params import ScJobParams
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.yaml import InferenceScorecardJobYaml
from qai_hub_models.utils.device import FormFactor


class NumericsDiff:
    """Generates Numerics Difference between two numerics.yaml"""

    def __init__(
        self,
        current_inference_jobs: InferenceScorecardJobYaml | None = None,
        previous_inference_jobs: InferenceScorecardJobYaml | None = None,
    ) -> None:
        # Inference job ID lookup yamls. Empty defaults are safe —
        # _inference_job_id_for() returns "null" on miss.
        self._current_inference_jobs = (
            current_inference_jobs or InferenceScorecardJobYaml()
        )
        self._previous_inference_jobs = (
            previous_inference_jobs or InferenceScorecardJobYaml()
        )

        self.missing_models: list[str] = []
        self.new_models: list[str] = []
        self.empty_models: list[str] = []

        # tuple<Model ID, Dataset Name, Metric Name>
        self.missing_metrics: list[tuple[str, str, str]] = []
        self.new_metrics: list[tuple[str, str, str]] = []

        # tuple<Model ID, Dataset Name, Metric Name, Device>
        self.missing_devices: list[tuple[str, str, str, ScorecardDevice]] = []
        self.new_devices: list[tuple[str, str, str, ScorecardDevice]] = []

        # tuple<Model ID, Dataset Name, Metric Name, Device, Precision>
        self.missing_precisions: list[
            tuple[str, str, str, ScorecardDevice, Precision]
        ] = []
        self.new_precisions: list[tuple[str, str, str, ScorecardDevice, Precision]] = []

        # tuple<Model ID, Dataset Name, Metric Name, Device, Precision, Path>
        self.missing_paths: list[
            tuple[str, str, str, ScorecardDevice, Precision, ScorecardProfilePath]
        ] = []
        self.new_paths: list[
            tuple[str, str, str, ScorecardDevice, Precision, ScorecardProfilePath]
        ] = []

        # Off-target (quantized ONNX on CPU) accuracy, keyed by
        # (Model ID, Dataset Name, Metric Name, Precision). Not per-path: sim runs once per precision.
        self.sim_accuracies: dict[tuple[str, str, str, Precision], float] = {}

        # tuple<Model ID, Dataset Name, Metric Name, Device, Precision, Path,
        #       FP Accuracy, Sim Accuracy, Device Accuracy, Previous FP Accuracy, Previous Device Accuracy,
        #       Inference Job ID, Previous Inference Job ID>
        # Progression == Accuracy is closer to target than previous. Regression == Accuracy is further from target than previous.
        self.progressions: list[
            tuple[
                str,
                str,
                str,
                ScorecardDevice,
                Precision,
                ScorecardProfilePath,
                str,
                str,
                str,
                str,
                str,
                str,
                str,
            ]
        ] = []
        self.regressions: list[
            tuple[
                str,
                str,
                str,
                ScorecardDevice,
                Precision,
                ScorecardProfilePath,
                str,
                str,
                str,
                str,
                str,
                str,
                str,
            ]
        ] = []

        # tuple<Model ID, Dataset Name, Metric Name, Accuracy Type ("torch" or "device"), Precision | None, Path | None, Sim Value, Device Value, Benchmark Value, Difference, Threshold, Newly Disabled>
        # Precision and Path are None for torch failures (torch accuracy is not per-device/path).
        self.benchmark_failures: list[
            tuple[
                str,
                str,
                str,
                str,
                Precision | None,
                ScorecardProfilePath | None,
                str,
                str,
                str,
                str,
                str,
                bool,
            ]
        ] = []

        # Exclude automotive devices (too noisy for regression tracking)
        yaml_config = DevicesAndChipsetsYaml.load()
        self._excluded_device_names: set[str] = {
            name
            for name, details in yaml_config.devices.items()
            if details.form_factor == FormFactor.AUTO
        }

        # tuple<Model ID, Dataset Name, Metric Name, Device, Precision, Path, FP Accuracy, Sim Accuracy, Current Device Accuracy, Difference, Difference Threshold, Newly Disabled>
        self.device_vs_float_greater_than_enablement_threshold: list[
            tuple[
                str,
                str,
                str,
                ScorecardDevice,
                Precision,
                ScorecardProfilePath,
                str,
                str,
                str,
                str,
                str,
                bool,
            ]
        ] = []

    def _inference_job_id_for(
        self,
        inference_jobs: InferenceScorecardJobYaml,
        model_id: str,
        precision: Precision,
        path: ScorecardProfilePath,
        device: ScorecardDevice,
    ) -> str:
        """Return the inference-job ID for these params, or "null" if missing.

        The linkifier renders "null" as N/A. Component / graph_name are
        intentionally not threaded through — numerics regressions don't track
        them today, so component-model lookups will miss and render N/A.
        """
        params = ScJobParams(
            model_id=model_id,
            path=path,
            precision=precision,
            device=device,
        )
        return inference_jobs.get_job_id(params) or "null"

    def _sim_cell(
        self,
        model_id: str,
        metric_details: QAIHMModelNumerics.MetricDetails,
        precision: Precision | None,
    ) -> str:
        """Return the Sim Accuracy cell, blank when sim is unavailable."""
        if precision is None:
            return ""
        sim = self.sim_accuracies.get(
            (
                model_id,
                metric_details.dataset_name,
                metric_details.metric_name,
                precision,
            )
        )
        return "" if sim is None else f"{sim} {metric_details.metric_unit}"

    def merge_from(self, other: NumericsDiff) -> None:
        self.missing_models.extend(other.missing_models)
        self.new_models.extend(other.new_models)
        self.empty_models.extend(other.empty_models)
        self.missing_metrics.extend(other.missing_metrics)
        self.new_metrics.extend(other.new_metrics)
        self.missing_devices.extend(other.missing_devices)
        self.new_devices.extend(other.new_devices)
        self.missing_precisions.extend(other.missing_precisions)
        self.new_precisions.extend(other.new_precisions)
        self.missing_paths.extend(other.missing_paths)
        self.new_paths.extend(other.new_paths)
        self.progressions.extend(other.progressions)
        self.regressions.extend(other.regressions)
        self.benchmark_failures.extend(other.benchmark_failures)
        self.device_vs_float_greater_than_enablement_threshold.extend(
            other.device_vs_float_greater_than_enablement_threshold
        )
        self.sim_accuracies.update(other.sim_accuracies)

    def _update_summary_for_path(
        self,
        model_id: str,
        previous_metric_details: QAIHMModelNumerics.MetricDetails | None,
        new_metric_details: QAIHMModelNumerics.MetricDetails,
        device: ScorecardDevice,
        precision: Precision,
        path: ScorecardProfilePath,
        previous_report: dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
        | None,
        new_report: dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails],
        benchmark: NumericsAccuracyBenchmark | None = None,
    ) -> None:
        prev_metric = previous_report.get(path) if previous_report else None
        new_metric = new_report.get(path)

        if not prev_metric and not new_metric:
            return
        if prev_metric and not new_metric:
            self.missing_paths.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                    precision,
                    path,
                )
            )
        elif not prev_metric and new_metric:
            self.new_paths.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                    precision,
                    path,
                )
            )

        # Skip device checks if torch already failed the benchmark
        torch_already_failed = any(
            f[0] == model_id
            and f[3] == "torch"
            and f[1] == new_metric_details.dataset_name
            and f[2] == new_metric_details.metric_name
            for f in self.benchmark_failures
        )

        device_vs_float = (
            (new_metric.partial_metric - new_metric_details.partial_torch_metric)
            if new_metric
            else None
        )

        device_vs_float_prev = (
            (prev_metric.partial_metric - previous_metric_details.partial_torch_metric)
            if prev_metric and previous_metric_details
            else None
        )

        if (
            not torch_already_failed
            and new_metric
            and device_vs_float
            and new_metric_details.metric_enablement_threshold
            and (abs(device_vs_float) >= new_metric_details.metric_enablement_threshold)
        ):
            info = QAIHMModelManifest.from_model(model_id)
            reasons = info.disabled_paths.get_disable_reasons(precision, path.runtime)
            self.device_vs_float_greater_than_enablement_threshold.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                    precision,
                    path,
                    f"{new_metric_details.partial_torch_metric} {new_metric_details.metric_unit}",
                    self._sim_cell(model_id, new_metric_details, precision),
                    f"{new_metric.partial_metric} {new_metric_details.metric_unit}",
                    f"{device_vs_float:.3f} {new_metric_details.metric_unit}",
                    f"{new_metric_details.metric_enablement_threshold} {new_metric_details.metric_unit}",
                    reasons.scorecard_accuracy_failure is None,
                )
            )

        # Check device accuracy against benchmark (skip if torch already failed)
        if (
            new_metric
            and benchmark
            and not torch_already_failed
            and new_metric_details.metric_enablement_threshold
            and benchmark.dataset_name == new_metric_details.dataset_name
            and benchmark.metric_name == new_metric_details.metric_name
            and benchmark.unit == new_metric_details.metric_unit
        ):
            device_vs_benchmark = new_metric.partial_metric - benchmark.value
            if (
                abs(device_vs_benchmark)
                >= new_metric_details.metric_enablement_threshold
            ):
                info = QAIHMModelManifest.from_model(model_id)
                reasons = info.disabled_paths.get_disable_reasons(
                    precision, path.runtime
                )
                self.benchmark_failures.append(
                    (
                        model_id,
                        new_metric_details.dataset_name,
                        new_metric_details.metric_name,
                        "device",
                        precision,
                        path,
                        self._sim_cell(model_id, new_metric_details, precision),
                        f"{new_metric.partial_metric} {new_metric_details.metric_unit}",
                        f"{benchmark.value} {new_metric_details.metric_unit}",
                        f"{device_vs_benchmark:.3f} {new_metric_details.metric_unit}",
                        f"{new_metric_details.metric_enablement_threshold} {new_metric_details.metric_unit}",
                        reasons.scorecard_accuracy_failure is None,
                    )
                )

        if (
            new_metric
            and prev_metric
            and previous_metric_details
            and new_metric_details.metric_enablement_threshold
            and device_vs_float
            and device_vs_float_prev
            and new_metric_details.metric_range.min
            == previous_metric_details.metric_range.min
            and new_metric_details.metric_range.max
            == previous_metric_details.metric_range.max
        ):
            # If both metrics are available and the range has not changed, we can compare the
            # difference between FP and Device Accuracy to find progressions and regressions.
            metric_diff_diff = abs(device_vs_float) - abs(device_vs_float_prev)
            if (
                abs(metric_diff_diff)
                > new_metric_details.metric_enablement_threshold / 10
            ):
                proreg = self.regressions if metric_diff_diff > 0 else self.progressions
                proreg.append(
                    (
                        model_id,
                        new_metric_details.dataset_name,
                        new_metric_details.metric_name,
                        device,
                        precision,
                        path,
                        f"{new_metric_details.partial_torch_metric} {new_metric_details.metric_unit}",
                        self._sim_cell(model_id, new_metric_details, precision),
                        f"{new_metric.partial_metric} {new_metric_details.metric_unit}",
                        f"{previous_metric_details.partial_torch_metric} {previous_metric_details.metric_unit}",
                        f"{prev_metric.partial_metric} {previous_metric_details.metric_unit}",
                        self._inference_job_id_for(
                            self._current_inference_jobs,
                            model_id,
                            precision,
                            path,
                            device,
                        ),
                        self._inference_job_id_for(
                            self._previous_inference_jobs,
                            model_id,
                            precision,
                            path,
                            device,
                        ),
                    )
                )

    def _update_summary_for_precision(
        self,
        model_id: str,
        previous_metric_details: QAIHMModelNumerics.MetricDetails | None,
        new_metric_details: QAIHMModelNumerics.MetricDetails,
        device: ScorecardDevice,
        precision: Precision,
        previous_report: dict[
            Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
        ]
        | None,
        new_report: dict[
            Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
        ],
        benchmark: NumericsAccuracyBenchmark | None = None,
    ) -> None:
        previous_report_precision = (
            previous_report.get(precision) if previous_report else None
        )
        new_report_precision = new_report.get(precision)

        if not previous_report_precision and not new_report_precision:
            return
        if previous_report_precision and not new_report_precision:
            self.missing_precisions.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                    precision,
                )
            )
        elif not previous_report_precision and new_report_precision:
            self.new_precisions.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                    precision,
                )
            )

        if new_report_precision:
            all_paths: set[ScorecardProfilePath] = set()
            if previous_report_precision:
                all_paths.update(previous_report_precision.keys())
            all_paths.update(new_report_precision.keys())

            for path in all_paths:
                self._update_summary_for_path(
                    model_id,
                    previous_metric_details,
                    new_metric_details,
                    device,
                    precision,
                    path,
                    previous_report_precision,
                    new_report_precision,
                    benchmark=benchmark,
                )

    def _update_summary_for_device(
        self,
        model_id: str,
        previous_metric_details: QAIHMModelNumerics.MetricDetails | None,
        new_metric_details: QAIHMModelNumerics.MetricDetails,
        device: ScorecardDevice,
        previous_report: dict[
            ScorecardDevice,
            dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ],
        ]
        | None,
        new_report: dict[
            ScorecardDevice,
            dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ],
        ],
        benchmark: NumericsAccuracyBenchmark | None = None,
    ) -> None:
        if str(device) in self._excluded_device_names:
            return
        previous_report_device = (
            previous_report.get(device) if previous_report else None
        )
        new_report_device = new_report.get(device)

        if not previous_report_device and not new_report_device:
            return
        if previous_report_device and not new_report_device:
            self.missing_devices.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                )
            )
        elif not previous_report_device and new_report_device:
            self.new_devices.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                )
            )

        if new_report_device:
            all_precisions: set[Precision] = set()
            if previous_report_device:
                all_precisions.update(previous_report_device.keys())
            all_precisions.update(new_report_device.keys())

            for precision in all_precisions:
                self._update_summary_for_precision(
                    model_id,
                    previous_metric_details,
                    new_metric_details,
                    device,
                    precision,
                    previous_report_device,
                    new_report_device,
                    benchmark=benchmark,
                )

    def _update_summary_for_metric(
        self,
        model_id: str,
        dataset: str,
        metric: str,
        previous_report: list[QAIHMModelNumerics.MetricDetails] | None,
        new_report: list[QAIHMModelNumerics.MetricDetails],
        benchmark: NumericsAccuracyBenchmark | None = None,
    ) -> None:
        previous_report_metric: QAIHMModelNumerics.MetricDetails | None = None
        new_report_metric: QAIHMModelNumerics.MetricDetails | None = None

        for metric_details in previous_report or []:
            if (
                metric_details.dataset_name == dataset
                and metric_details.metric_name == metric
            ):
                previous_report_metric = metric_details
                break

        for metric_details in new_report:
            if (
                metric_details.dataset_name == dataset
                and metric_details.metric_name == metric
            ):
                new_report_metric = metric_details
                break

        if not previous_report_metric and not new_report_metric:
            return
        if new_report_metric and not previous_report_metric:
            self.new_metrics.append((model_id, dataset, metric))
        elif previous_report_metric and not new_report_metric:
            self.missing_metrics.append((model_id, dataset, metric))

        if new_report_metric:
            # Check torch accuracy against benchmark
            if (
                benchmark
                and new_report_metric.metric_enablement_threshold
                and benchmark.dataset_name == new_report_metric.dataset_name
                and benchmark.metric_name == new_report_metric.metric_name
                and benchmark.unit == new_report_metric.metric_unit
            ):
                torch_vs_benchmark = (
                    new_report_metric.partial_torch_metric - benchmark.value
                )
                if (
                    abs(torch_vs_benchmark)
                    >= new_report_metric.metric_enablement_threshold
                ):
                    self.benchmark_failures.append(
                        (
                            model_id,
                            new_report_metric.dataset_name,
                            new_report_metric.metric_name,
                            "torch",
                            None,
                            None,
                            "",
                            f"{new_report_metric.partial_torch_metric} {new_report_metric.metric_unit}",
                            f"{benchmark.value} {new_report_metric.metric_unit}",
                            f"{torch_vs_benchmark:.3f} {new_report_metric.metric_unit}",
                            f"{new_report_metric.metric_enablement_threshold} {new_report_metric.metric_unit}",
                            False,
                        )
                    )

            all_devices: set[ScorecardDevice] = set()

            previous_report_metric_devices = None
            if previous_report_metric:
                previous_report_metric_devices = previous_report_metric.device_metric
                all_devices.update(previous_report_metric_devices.keys())

            new_report_metric_devices = new_report_metric.device_metric
            all_devices.update(new_report_metric_devices.keys())
            for device in all_devices:
                self._update_summary_for_device(
                    model_id,
                    previous_report_metric,
                    new_report_metric,
                    device,
                    previous_report_metric_devices,
                    new_report_metric_devices,
                    benchmark=benchmark,
                )

    def update_summary(
        self,
        model_id: str,
        previous_report: QAIHMModelNumerics | None,
        new_report: QAIHMModelNumerics | None,
        benchmark: NumericsAccuracyBenchmark | None = None,
    ) -> None:
        if not new_report and not previous_report:
            return
        if new_report and not previous_report:
            self.new_models.append(model_id)
        elif previous_report and not new_report:
            self.missing_models.append(model_id)

        if new_report:
            all_metrics: set[tuple[str, str]] = set()
            all_metrics.update(
                (metric.dataset_name, metric.metric_name)
                for metric in (previous_report.metrics if previous_report else [])
            )
            all_metrics.update(
                (metric.dataset_name, metric.metric_name)
                for metric in new_report.metrics
            )
            for dataset_name, metric_name in all_metrics:
                self._update_summary_for_metric(
                    model_id,
                    dataset_name,
                    metric_name,
                    previous_report.metrics if previous_report else None,
                    new_report.metrics,
                    benchmark=benchmark,
                )

    def _get_summary_table_proreg(
        self,
        data: list[
            tuple[
                str,
                str,
                str,
                ScorecardDevice,
                Precision,
                ScorecardProfilePath,
                str,
                str,
                str,
                str,
                str,
                str,
                str,
            ]
        ],
    ) -> PrettyTable:
        """
        Returns Summary Table for given bucket.

        Parameters
        ----------
        data
            List of tuples containing model id, dataset name, metric name, device,
            precision, path, FP accuracy, sim accuracy, device accuracy, previous
            FP accuracy, previous device accuracy, inference job id, and previous
            inference job id.

        Returns
        -------
        table : PrettyTable
            Summary table for the given data.
        """
        # Column names are canonical (no env suffix). The issue-body builder
        # re-tags them with the deployment of each side at render time, since
        # 'previous' isn't always prod.
        table = PrettyTable(
            [
                "Model ID",
                "Dataset Name",
                "Metric Name",
                "Device",
                "Precision",
                "Runtime",
                "FP Accuracy",
                "Sim Accuracy",
                "Device Accuracy",
                "Previous FP Accuracy",
                "Previous Device Accuracy",
                "Inference Job ID",
                "Previous Inference Job ID",
            ]
        )
        data.sort(key=lambda k: k[0])  # sort by model id
        table.add_rows(data)
        return table

    def dump_summary(self, summary_file_path: str) -> None:
        """Dumps Perf change summary captured so far to the provided path."""
        with open(summary_file_path, "w") as sf:
            sf.write("================= Perf Change Summary =================")
            sf.write(
                "\n\n----------------- Disabled Configurations -----------------\n"
            )
            if self.device_vs_float_greater_than_enablement_threshold:
                table = PrettyTable(
                    [
                        "Model ID",
                        "Dataset Bane",
                        "Metric name",
                        "Device",
                        "Precision",
                        "Runtime",
                        "FP Accuracy",
                        "Sim Accuracy",
                        "Device Accuracy",
                        "Difference",
                        "Difference Threshold",
                        "Newly Disabled",
                    ]
                )
                self.device_vs_float_greater_than_enablement_threshold.sort(
                    key=lambda k: k[0]
                )  # sort by model id
                table.add_rows(self.device_vs_float_greater_than_enablement_threshold)
                sf.write(str(table))
            else:
                sf.write("No disabled configurations found.\n")
            sf.write("\n")

            sf.write("\n\n----------------- Benchmark Failures -----------------\n")
            if self.benchmark_failures:
                table = PrettyTable(
                    [
                        "Model ID",
                        "Dataset Name",
                        "Metric Name",
                        "Accuracy Type",
                        "Precision",
                        "Runtime",
                        "Sim Value",
                        "Device Value",
                        "Benchmark Value",
                        "Difference",
                        "Threshold",
                        "Newly Disabled",
                    ]
                )
                self.benchmark_failures.sort(key=lambda k: k[0])  # sort by model id
                table.add_rows(self.benchmark_failures)
                sf.write(str(table))
            else:
                sf.write("No benchmark failures found.\n")
            sf.write("\n")

            sf.write("\n\n----------------- Regressions -----------------\n")
            if self.regressions:
                sf.write(str(object=self._get_summary_table_proreg(self.regressions)))
            else:
                sf.write("No significant regressions observed.\n")
            sf.write("\n")

            sf.write("\n\n----------------- Progressions -----------------\n")
            if self.progressions:
                sf.write(str(object=self._get_summary_table_proreg(self.progressions)))
            else:
                sf.write("No significant progressions observed.\n")
            sf.write("\n")

            if self.missing_models:
                sf.write("\n----------------- Missing Models -----------------\n")
                table = PrettyTable(["Model ID"])
                table.add_rows([x] for x in self.missing_models)
                sf.write(str(table))
                sf.write("\n")

            if self.new_models:
                sf.write("\n----------------- New Models -----------------\n")
                table = PrettyTable(["Model ID"])
                table.add_rows([x] for x in self.new_models)
                sf.write(str(table))
                sf.write("\n")

            if self.missing_metrics:
                sf.write("\n----------------- Missing Metrics -----------------\n")
                table = PrettyTable(["Model ID", "Dataset Name", "Metric Name"])
                table.add_rows(self.missing_metrics)
                sf.write(str(table))
                sf.write("\n")

            if self.new_metrics:
                sf.write("\n----------------- New Metrics -----------------\n")
                table = PrettyTable(["Model ID", "Dataset Name", "Metric Name"])
                table.add_rows(self.new_metrics)
                sf.write(str(table))
                sf.write("\n")

            if self.missing_devices:
                sf.write("\n----------------- Missing Devices -----------------\n")
                table = PrettyTable(
                    ["Model ID", "Dataset Name", "Metric Name", "Device"]
                )
                table.add_rows(self.missing_devices)
                sf.write(str(table))
                sf.write("\n")

            if self.new_devices:
                sf.write("\n----------------- New Devices -----------------\n")
                table = PrettyTable(
                    ["Model ID", "Dataset Name", "Metric Name", "Device"]
                )
                table.add_rows(self.new_devices)
                sf.write(str(table))
                sf.write("\n")

            if self.missing_precisions:
                sf.write("\n----------------- Missing Precisions -----------------\n")
                table = PrettyTable(
                    ["Model ID", "Dataset Name", "Metric Name", "Device", "Precision"]
                )
                table.add_rows(self.missing_precisions)
                sf.write(str(table))
                sf.write("\n")

            if self.new_precisions:
                sf.write("\n----------------- New Precisions -----------------\n")
                table = PrettyTable(
                    ["Model ID", "Dataset Name", "Metric Name", "Device", "Precision"]
                )
                table.add_rows(self.new_precisions)
                sf.write(str(table))
                sf.write("\n")

            if self.missing_paths:
                sf.write("\n----------------- Missing Runtimes -----------------\n")
                table = PrettyTable(
                    [
                        "Model ID",
                        "Dataset Name",
                        "Metric Name",
                        "Device",
                        "Precision",
                        "Runtime",
                    ]
                )
                table.add_rows(self.missing_paths)
                sf.write(str(table))
                sf.write("\n")

            if self.new_precisions:
                sf.write("\n----------------- New Runtimes -----------------\n")
                table = PrettyTable(
                    [
                        "Model ID",
                        "Dataset Name",
                        "Metric Name",
                        "Device",
                        "Precision",
                        "Runtime",
                    ]
                )
                table.add_rows(self.new_paths)
                sf.write(str(table))
                sf.write("\n")

        print(f"Perf change summary written to {summary_file_path}")

    def get_regressions_list(self) -> list[dict[str, str]]:
        """Return numerics regressions as a list of dicts.

        Uses _get_summary_table_proreg() column names as the single source
        of truth for key definitions, with values stringified for JSON
        compatibility.

        Returns
        -------
        list[dict[str, str]]
            Each dict has keys matching the PrettyTable column names.
        """
        table = self._get_summary_table_proreg(self.regressions)
        field_names = table.field_names
        return [
            {k: str(v) for k, v in zip(field_names, row, strict=False)}
            for row in self.regressions
        ]

    def dump_regressions_json(self, json_path: str) -> None:
        """Write numerics regressions to a JSON file for downstream consumption."""
        data = self.get_regressions_list()
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Numerics regressions JSON written to {json_path} ({len(data)} entries)")

    # Column names for the JSON export of newly-disabled rows. Kept clean here
    # (dump_summary has legacy typos we don't want in downstream tooling).
    _DISABLED_CONFIG_COLUMNS = [
        "Model ID",
        "Dataset Name",
        "Metric Name",
        "Device",
        "Precision",
        "Runtime",
        "FP Accuracy",
        "Sim Accuracy",
        "Device Accuracy",
        "Difference",
        "Difference Threshold",
        "Newly Disabled",
    ]
    _BENCHMARK_FAILURE_COLUMNS = [
        "Model ID",
        "Dataset Name",
        "Metric Name",
        "Accuracy Type",
        "Precision",
        "Runtime",
        "Sim Value",
        "Device Value",
        "Benchmark Value",
        "Difference",
        "Threshold",
        "Newly Disabled",
    ]

    def dump_newly_disabled_json(self, json_path: str) -> None:
        """Write newly-disabled accuracy rows from both buckets to one JSON file.

        Powers the "Newly Disabled Configurations" and "Newly Failing Benchmarks"
        sections of the scorecard auto-issue. Each row carries a `kind` key so
        the issue builder can split the flat list back into its source bucket.
        Stale (already-disabled) rows are left in the downloadable diff only.
        """
        rows: list[dict[str, str]] = []
        for disabled_row in self.device_vs_float_greater_than_enablement_threshold:
            if disabled_row[-1]:
                entry: dict[str, str] = {"kind": "disabled_configuration"}
                entry.update(
                    {
                        k: str(v)
                        for k, v in zip(
                            self._DISABLED_CONFIG_COLUMNS,
                            disabled_row,
                            strict=False,
                        )
                    }
                )
                rows.append(entry)
        for benchmark_row in self.benchmark_failures:
            if benchmark_row[-1]:
                entry = {"kind": "benchmark_failure"}
                entry.update(
                    {
                        k: str(v)
                        for k, v in zip(
                            self._BENCHMARK_FAILURE_COLUMNS,
                            benchmark_row,
                            strict=False,
                        )
                    }
                )
                rows.append(entry)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(
            f"Newly-disabled accuracy JSON written to {json_path} ({len(rows)} entries)"
        )

    _QUANTIZED_FAILURE_COLUMNS = [
        "Model ID",
        "Device",
        "Precision",
        "Runtime",
        "FP Accuracy",
        "Sim Accuracy",
        "Device Accuracy",
    ]

    def dump_quantized_failures_json(self, json_path: str) -> None:
        """Write every disabled configuration that has a Sim value, new or not.

        Unlike dump_newly_disabled_json, already-disabled rows are kept so the
        issue always shows whether each quantized failure is off-target or on-target.
        """
        rows: list[dict[str, str]] = []
        for disabled_row in self.device_vs_float_greater_than_enablement_threshold:
            full = {
                k: str(v)
                for k, v in zip(
                    self._DISABLED_CONFIG_COLUMNS, disabled_row, strict=False
                )
            }
            if full["Sim Accuracy"]:
                rows.append({k: full[k] for k in self._QUANTIZED_FAILURE_COLUMNS})
        rows.sort(key=lambda r: r["Model ID"])
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Quantized failures JSON written to {json_path} ({len(rows)} entries)")

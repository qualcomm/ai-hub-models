# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import cast

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import MODEL_DOMAIN, MODEL_USE_CASE
from qai_hub_models.configs.model_disable_reasons import ModelDisableReasonsMapping
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.scorecard.results.scorecard_job import ScorecardJobTypeVar
from qai_hub_models.scorecard.results.scorecard_summary import (
    ScorecardExportTestSummary,
    ScorecardJobSummary,
)
from qai_hub_models.utils.path_helpers import get_git_branch

MAX_ERROR_LENGTH = 250
DEFAULT_MODEL_SUMMARY_ID = "__DEFAULT_SUMMARY_FOR_MODEL"


@dataclass
class ResultsSpreadsheet(list):
    has_compile_jobs: bool = True
    has_profile_jobs: bool = True
    has_inference_jobs: bool = True
    _model_metadata: dict[str, ResultsSpreadsheet.ModelMetadata] = field(
        default_factory=dict
    )
    _datestr: str | None = None
    _branchstr: str | None = None

    @dataclass
    class ModelMetadata:
        domain: MODEL_DOMAIN
        use_case: MODEL_USE_CASE
        tags: list[str]
        known_failure_reasons: ModelDisableReasonsMapping
        default_quantized_precision: Precision | None
        default_device: ScorecardDevice

    @dataclass
    class Entry:
        model_id: str
        component_id: str | None
        graph_name: str | None
        precision: Precision
        chipset: str
        runtime: ScorecardInferencePath
        quantize_status: str
        quantize_url: str | None
        compile_status: str
        compile_url: str | None
        link_status: str
        link_url: str | None
        profile_status: str
        profile_url: str | None
        inference_time: float | None
        first_load_time: float | None
        warm_load_time: float | None
        NPU: int | None
        GPU: int | None
        CPU: int | None
        inference_status: str
        inference_url: str | None

        @property
        def model_component_graph_id(self) -> str:
            return (
                self.model_id
                + (f"::{self.component_id}" if self.component_id else "")
                + (f"::{self.graph_name}" if self.graph_name else "")
            )

    def to_csv(
        self, path: str | os.PathLike, combine_model_and_component_id: bool = True
    ) -> None:
        # Default datetime if not set
        date: str
        if not self._datestr:
            self.set_date(datetime.now())
            date = cast(str, self._datestr)
            self._datestr = None
        else:
            date = self._datestr

        # Default to branch if not set
        branch = self._branchstr or get_git_branch()

        # Get CSV fields
        field_names = [field.name for field in fields(ResultsSpreadsheet.Entry)]

        # Add metadata fields
        field_names.insert(2, "domain")
        field_names.insert(3, "use_case")
        field_names.insert(4, "tags")
        field_names.insert(5, "branch")
        field_names.insert(6, "known_issue")

        # Add date field
        field_names.insert(2, "date")

        # Remove fields that are not applicable
        fields_to_remove: list[str] = []
        if not self.has_compile_jobs:
            fields_to_remove.extend(
                ["compile_status", "compile_url", "link_status", "link_url"]
            )
        if not self.has_profile_jobs:
            fields_to_remove.extend(
                ["profile_status", "profile_url", "inference_time", "NPU", "GPU", "CPU"]
            )
        if not self.has_inference_jobs:
            fields_to_remove.extend(["inference_status", "inference_url"])
        if combine_model_and_component_id:
            fields_to_remove.extend(["component_id", "graph_name"])
        for field_name in fields_to_remove:
            field_names.remove(field_name)

        # Save CSV
        with open(path, "w") as csvfile:
            scorecard_csv = csv.writer(csvfile)

            # Header
            scorecard_csv.writerow(field_names)

            def _get_value(field_name: str, entry: ResultsSpreadsheet.Entry) -> str:
                model_id: str = entry.model_id
                if field_name == "model_id":
                    return (
                        entry.model_component_graph_id
                        if combine_model_and_component_id
                        else entry.model_id
                    )
                if field_name == "runtime":
                    return entry.runtime.spreadsheet_name
                if field_name == "date":
                    return date
                if field_name == "domain":
                    return (
                        self._model_metadata[model_id].domain.name
                        if model_id in self._model_metadata
                        else ""
                    )
                if field_name == "use_case":
                    return (
                        self._model_metadata[model_id].use_case.name
                        if model_id in self._model_metadata
                        else ""
                    )
                if field_name == "tags":
                    if model_id not in self._model_metadata:
                        return ""
                    metadata = self._model_metadata[model_id]
                    tags = list(metadata.tags)
                    if (
                        metadata.default_quantized_precision is not None
                        and metadata.default_quantized_precision == entry.precision
                    ):
                        tags.append("default_quantized")
                    if metadata.default_device.chipset == entry.chipset:
                        tags.append("default_chipset")
                    return ", ".join(tags)
                if field_name == "branch":
                    return branch
                if field_name == "known_issue":
                    if (meta := self._model_metadata.get(model_id)) and (
                        failure_reasons
                        := meta.known_failure_reasons.get_disable_reasons(
                            entry.precision, entry.runtime.runtime
                        )
                    ):
                        # Rows only carry a chipset, so match scoped devices by chipset.
                        row_devices = [
                            d.device_name
                            for d in failure_reasons.disable_devices_as_registered()
                            if d.chipset == entry.chipset
                        ]
                        return (
                            failure_reasons.failure_reason_for_device(
                                row_devices[0] if row_devices else None
                            )
                            or ""
                        )
                    return ""

                val = getattr(entry, field_name)
                if isinstance(val, bool):
                    return "YES" if val else "NO"
                return str(val)

            # Rows
            for entry in self:
                scorecard_csv.writerow(
                    [_get_value(name, entry) for name in field_names]
                )

    @dataclass
    class ModelInfo:
        model_id: str
        components: list[str] | None
        quantized: bool

    def append_export_test_summary(self, summaries: ScorecardExportTestSummary) -> None:
        self.extend(
            ResultsSpreadsheet.get_inference_job_summary_entry(inference_job_summary)
            for inference_job_summary in summaries.job_summaries
        )

    def set_model_metadata(
        self,
        model_id: str,
        domain: MODEL_DOMAIN,
        use_case: MODEL_USE_CASE,
        tags: list[str],
        default_quantized_precision: Precision | None,
        default_device: ScorecardDevice,
        known_failure_reasons: ModelDisableReasonsMapping | None = None,
    ) -> None:
        self._model_metadata[model_id] = ResultsSpreadsheet.ModelMetadata(
            domain,
            use_case,
            tags,
            known_failure_reasons or ModelDisableReasonsMapping(),
            default_quantized_precision,
            default_device,
        )

    def set_date(self, date: datetime | None) -> None:
        self._datestr = date.strftime("%m/%d/%Y") if date else None

    def set_branch(self, branch: str | None) -> None:
        self._branchstr = branch

    @staticmethod
    def get_inference_job_summary_entry(
        summary: ScorecardJobSummary,
    ) -> ResultsSpreadsheet.Entry:
        assert isinstance(summary.params.path, ScorecardInferencePath)
        assert summary.params.device is not None
        assert summary.params.precision is not None

        def _get_url_and_status(
            sjob: ScorecardJobTypeVar | None,
        ) -> tuple[str, str | None]:
            # Replace all whitespace with space character
            if not sjob:
                return ("skipped", None)

            status = (
                re.sub(r"\s+", " ", sjob.status_message[:MAX_ERROR_LENGTH])
                if sjob.status_message
                else None
            )
            return (
                sjob.job_status + (f" ({status})" if sjob.status_message else ""),
                sjob.job.url,
            )

        # Job status
        quantize_status, quantize_url = _get_url_and_status(summary.quantize_job)
        compile_status, compile_url = _get_url_and_status(summary.compile_job)
        link_status, link_url = _get_url_and_status(summary.link_job)
        # "profile" column reads the single inference job that both profiles and
        # infers (see summary.inference_job).
        profile_status, profile_url = _get_url_and_status(summary.inference_job)
        inference_status, inference_url = _get_url_and_status(summary.inference_job)

        # Inference job results
        inference_job = summary.inference_job
        if inference_job and inference_job.success:
            inference_time = inference_job.inference_time_milliseconds
            first_load_time = inference_job.first_load_time_milliseconds
            warm_load_time = inference_job.warm_load_time_milliseconds
            NPU = inference_job.layer_counts.npu
            GPU = inference_job.layer_counts.gpu
            CPU = inference_job.layer_counts.cpu
        else:
            inference_time = None
            first_load_time = None
            warm_load_time = None
            NPU = None
            GPU = None
            CPU = None

        # Create Entry
        return ResultsSpreadsheet.Entry(
            model_id=summary.params.model_id,
            component_id=summary.params.component,
            graph_name=summary.params.graph_name,
            precision=summary.params.precision,
            chipset=summary.params.device.chipset,
            runtime=summary.params.path,
            quantize_status=quantize_status,
            quantize_url=quantize_url,
            compile_status=compile_status.replace(
                ",", "."
            ),  # remove commas for CSV compatibliity
            compile_url=compile_url,
            link_status=link_status.replace(
                ",", "."
            ),  # remove commas for CSV compatibliity
            link_url=link_url,
            profile_status=profile_status.replace(
                ",", "."
            ),  # remove commas for CSV compatibliity
            profile_url=profile_url,
            inference_time=inference_time,
            first_load_time=first_load_time,
            warm_load_time=warm_load_time,
            NPU=NPU,
            GPU=GPU,
            CPU=CPU,
            inference_status=inference_status,
            inference_url=inference_url,
        )

    def combine(self, other: ResultsSpreadsheet) -> None:
        self.extend(other)
        self._model_metadata.update(other._model_metadata)

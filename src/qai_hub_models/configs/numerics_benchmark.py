# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models_cli.proto import numerics_pb2

from qai_hub_models.utils.base_config import BaseQAIHMConfig


class NumericsAccuracyBenchmark(BaseQAIHMConfig):
    """A published accuracy number for the float model, used as a scorecard baseline."""

    # Dataset the benchmark was measured on.
    dataset_name: str

    # Metric name, which must match a metric the model's evaluator reports.
    metric_name: str

    value: float

    # Metric unit, which must match the evaluator's unit for metric_name.
    unit: str

    # Where the number came from (a paper, a repo, our own measurement).
    source: str

    def to_proto(self) -> numerics_pb2.NumericsAccuracyBenchmark:
        return numerics_pb2.NumericsAccuracyBenchmark(
            dataset_name=self.dataset_name,
            metric_name=self.metric_name,
            value=self.value,
            unit=self.unit,
            source=self.source,
        )

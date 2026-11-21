# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import glob as glob_module
import os
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

from qai_hub_models_cli.proto_helpers._common import dump_yaml, safe_load_yaml
from typing_extensions import Self

from qai_hub_models.scorecard.envvars import ArtifactsDirEnvvar
from qai_hub_models.utils.base_config import sorted_mappings
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_PACKAGE_ROOT

ValT = TypeVar("ValT")

INTERMEDIATES_DIR = QAIHM_PACKAGE_ROOT / "scorecard" / "intermediates"

# Stage keys in model-runtime-estimates.yaml; shared between writer
# (extract_model_runtimes) and reader (split_torch_models).
RUNTIME_STAGE_JOB_SUBMISSION = "job_submission"
RUNTIME_STAGE_EXPORT_TEST = "export_test"
RUNTIME_STAGE_ACCURACY = "accuracy"
RUNTIME_ALL_STAGES = (
    RUNTIME_STAGE_JOB_SUBMISSION,
    RUNTIME_STAGE_EXPORT_TEST,
    RUNTIME_STAGE_ACCURACY,
)


def test_artifacts_dir() -> Path:
    """Get the path in which all test artifacts are stored."""
    return ArtifactsDirEnvvar.get()


class ScorecardArtifact(Enum):
    # Results
    ACCURACY_CSV = "accuracy.csv"
    EXPORT_CSV = "export-summary.csv"
    RESULTS_CSV = "results.csv"

    # History (uploaded to S3 for long-term storage)
    PERFORMANCE_SUMMARY = "performance-summary-*.txt"
    NUMERICS_SUMMARY = "numerics-summary-*.txt"
    PERF_REGRESSIONS_2X = "perf-regressions-2x-*.json"
    NUMERICS_REGRESSIONS = "numerics-regressions-*.json"
    SCORECARD_FAILURE_ANALYSIS = "scorecard_failure_analysis.csv"

    # Cached State
    DATE = "date.txt"
    ENVIRONMENT_FILE = "environment.env"
    TOOL_VERSIONS = "tool-versions.yaml"
    COMPONENT_NAMES = "component-names.yaml"
    GRAPH_NAMES = "graph-names.yaml"
    QUANTIZE_YAML = "quantize-jobs.yaml"
    COMPILE_YAML = "compile-jobs.yaml"
    COMPILE_JOBS_IDENTICAL_CACHE = "compile-jobs-are-identical-cache.yaml"
    LINK_YAML = "link-jobs.yaml"
    PROFILE_YAML = "profile-jobs.yaml"
    INFERENCE_YAML = "inference-jobs.yaml"
    RELEASE_ASSETS = "release-assets.yaml"
    DATASET_IDS = "dataset-ids.yaml"
    CPU_ACCURACY = "cpu-accuracy.yaml"
    MODEL_RUNTIME_ESTIMATES = "model-runtime-estimates.yaml"

    def touch(self) -> Path:
        """Get the path for this test artifact. Will touch() the artifact if it does not exist."""
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    @property
    def path(self) -> Path:
        """Get the path for this test artifact.

        For glob-pattern artifacts (containing '*'), resolves to the latest
        matching file. Returns the unresolved pattern path if no match exists.
        """
        raw = test_artifacts_dir() / self.value
        if "*" in self.value:
            matches = sorted(glob_module.glob(str(raw)))
            if matches:
                return Path(matches[-1])
        return raw

    def exists(self) -> bool:
        """Returns true if the artifact exists and is non-empty."""
        path = self.path
        return path.exists() and path.stat().st_size > 0

    @property
    def intermediates_path(self) -> Path:
        """Get the path for this artifact in the checked-in scorecard intermediates."""
        return INTERMEDIATES_DIR / self.value


class ScorecardYamlFile(Generic[ValT]):
    """Base class for YAML files backed by a dict[str, ValT] mapping."""

    ARTIFACT_TYPE: ScorecardArtifact

    def __init__(
        self,
        mapping: dict[str, ValT] | None = None,
        path: str | os.PathLike | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.mapping: dict[str, ValT] = mapping or {}

    @classmethod
    def from_file(
        cls, config_path: str | os.PathLike, create_empty_if_no_file: bool = False
    ) -> Self:
        if not os.path.exists(config_path):
            if create_empty_if_no_file:
                return cls({}, config_path)
            raise FileNotFoundError(f"File not found: {config_path}")
        with open(config_path) as f:
            data = safe_load_yaml(f) or {}
        return cls(data, config_path)

    @classmethod
    def from_intermediates(cls) -> Self:
        return cls.from_file(
            cls.ARTIFACT_TYPE.intermediates_path, create_empty_if_no_file=True
        )

    @classmethod
    def from_test_artifacts(cls) -> Self:
        return cls.from_file(cls.ARTIFACT_TYPE.path, create_empty_if_no_file=True)

    def _model_keys(self, model_id: str) -> list[str]:
        """Keys belonging to ``model_id``.

        Keys are ``<model>`` or ``<model>_<component>``, and model ids contain
        ``_`` -- 12 ids are a prefix of another (llama_v3_2_3b_instruct vs
        llama_v3_2_3b_instruct_ssd, qwen3_4b vs qwen3_4b_instruct_2507). A bare
        prefix test lets the shorter id claim its sibling's keys, so the longest
        matching model id owns the key.
        """
        siblings = [
            m
            for m in MODEL_IDS
            if len(m) > len(model_id) and m.startswith(f"{model_id}_")
        ]
        return [
            key
            for key in self.mapping
            if key == model_id
            or (
                key.startswith(f"{model_id}_")
                and not any(key == s or key.startswith(f"{s}_") for s in siblings)
            )
        ]

    def clear(self, model_id: str | None = None) -> None:
        if not model_id:
            self.mapping.clear()
            return
        for key in self._model_keys(model_id):
            del self.mapping[key]

    def has_model(self, model_id: str) -> bool:
        """Whether any key belongs to this model. Matches ``clear``'s key rule."""
        return bool(self._model_keys(model_id))

    def to_file(self, path: str | Path | None = None) -> None:
        path = path or self.path
        assert path is not None
        if self.mapping:
            with open(path, "w") as f:
                dump_yaml(sorted_mappings(self.mapping), f)
        else:
            Path(path).touch()

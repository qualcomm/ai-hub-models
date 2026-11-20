# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import functools
import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PrivateAttr, model_validator
from qai_hub_models_cli.proto import model_manifest_pb2
from qai_hub_models_cli.proto_helpers._common import (
    read_proto_yaml,
    write_proto_yaml,
)

from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs._info_yaml_llm_details import LLMDetails
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.proto_helpers import (
    call_to_action_from_proto,
    call_to_action_to_proto,
    form_factor_from_proto,
    form_factor_to_proto,
    status_from_proto,
    status_to_proto,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.device import FormFactor
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    QAIHM_MODELS_ROOT,
    SCORECARD_MODELS_ROOT,
)


class TestRunnerSplit(Enum):
    """Named GitHub actions runner splits for grouping models in CI test runs."""

    DEFAULT = "default"
    LLM = "llm"
    PI0_5 = "pi05"

    @property
    def name(self) -> str:
        return self.value

    @property
    def runs_on(self) -> dict[Literal["group", "labels"], str | list[str]] | None:
        """
        Runner configuration for this split, matching GitHub Actions runs-on syntax.

        Returns None (use workflow default) or a dict with "group" and/or "labels"
        (e.g. {"group": "GPU", "labels": ["self-hosted"]}).
        """
        if self is TestRunnerSplit.LLM:
            return {"group": "GPU"}
        return None

    @property
    def max_models_per_split(self) -> int:
        if self is TestRunnerSplit.LLM:
            return 5
        return 10**9


_Internal = model_manifest_pb2.ModelManifest.Internal

_TEST_SPLIT_TO_PROTO: dict[TestRunnerSplit, int] = {
    TestRunnerSplit.DEFAULT: _Internal.TEST_SPLIT_DEFAULT,
    TestRunnerSplit.LLM: _Internal.TEST_SPLIT_LLM,
    TestRunnerSplit.PI0_5: _Internal.TEST_SPLIT_PI05,
}
_PROTO_TO_TEST_SPLIT = {v: k for k, v in _TEST_SPLIT_TO_PROTO.items()}


class QAIHMModelInternalConfig(BaseQAIHMConfig):
    """Schema for model internal-config.yaml — fields consumed only by internal CI."""

    # Publish status. Required: every in-tree model must say whether it is in the
    # public catalog. Scorecard promotes pending -> published and demotes back.
    status: MODEL_STATUS

    # Why the model is not published. Required for unpublished models (with a
    # tracking issue link), optional for pending, forbidden for published.
    status_reason: str | None = None

    # If set, skips
    #  - generating `test_generated.py`
    #  - weekly scorecard
    #  - generating perf.yaml
    skip_hub_tests_and_scorecard: bool = False

    # Second knob for skipping of scorecard generation. Use case, skip scorecard but run hub tests.
    skip_scorecard: bool = False

    # If set to true, Scorecard will still run this model, but perf.yaml and associated manifest.yaml / README.md changes will not be written to disk.
    # This is useful for models whose assets cannot be changed in a release, but we still want to continue testing said models.
    freeze_perf_yaml: bool = False

    # Places this model into a named CI test split.
    test_split: TestRunnerSplit = TestRunnerSplit.DEFAULT

    # If set, skips all code generation for this model.
    skip_export: bool = False

    # If set, omits the evaluate/accuracy tests from the generated test file.
    skip_evaluate: bool = False

    # Overrides the scorecard acceptance threshold. E.g., set this to 15 to
    # allow up to a 15-point drop and still be considered successful.
    numerics_threshold_override: float | None = None

    # When possible, package versions in a model's specific `requirements.txt`
    # should match the versions in `qai_hub_models/global_requirements.txt`.
    # When this is not possible, set this field to indicate an inconsistency.
    global_requirements_incompatible: bool = False

    # If true, this LLM is excluded from the weekly schedule and only runs when
    # explicitly requested via the 'llm_on_demand' token.
    on_demand_only: bool = False

    # Components benchmarked standalone, mapped to their model card tab label (also the
    # component's perf.yaml key). Unlisted components are LLM backbone parts, measured
    # together end-to-end and consolidated into one tab keyed by the model name.
    standalone_components: dict[str, str] = Field(default_factory=dict)

    # =========================================================================
    # Website / model-card fields
    #
    # Read only when building the AI Hub website feed and the in-tree model card.
    # Nothing on the external recipe path (install / export / evaluate / demo /
    # generate-files / upload-to-hf) touches them, so they live here rather than in
    # manifest.yaml. WebsiteModelInfo pairs them back up with the manifest.
    # =========================================================================

    # A list of real-world applications for which this model could be used.
    # This is free-form and almost anything reasonable here is fine.
    applicable_scenarios: list[str] = Field(default_factory=list)

    # A list of other similar models in the repo.
    # Typically, any model that performs the same task is fine.
    # If nothing fits, this can be left blank. Limit to 3 models.
    related_models: list[str] = Field(default_factory=list)

    # Whether the model has a static image uploaded in S3. All published models must have this.
    has_static_banner: bool = False

    # Whether the model has an animated asset uploaded in S3. This is optional.
    has_animated_banner: bool = False

    # Device form factors for which we don't publish performance data.
    private_perf_form_factors: list[FormFactor] | None = None

    # Some models are made by company
    model_maker_id: str | None = None

    # Whether the model is compatible with the IMSDK Plugin for IOT devices
    imsdk_supported: bool = False

    # Add per device, download, app and if the model is available for purchase.
    llm_details: LLMDetails | None = None

    # Set by from_model so `manifest` can pull manifest.yaml on demand. None on
    # directly-constructed configs, which must pass a manifest in explicitly.
    _model_id: str | None = PrivateAttr(default=None)

    @functools.cached_property
    def manifest(self) -> QAIHMModelManifest:
        """
        This model's manifest.yaml, parsed once and cached on this instance.

        Read-only. from_model is cached, so this instance is shared process-wide;
        anything that writes to the manifest (disabled_paths, to_model_yaml) must load
        its own copy via QAIHMModelManifest.from_model instead.
        """
        if self._model_id is None:
            raise ValueError(
                "manifest is only available on configs loaded via from_model(); "
                "pass a manifest explicitly instead."
            )
        return QAIHMModelManifest.from_model(self._model_id)

    @model_validator(mode="after")
    def _validate_status_reason(self) -> QAIHMModelInternalConfig:
        if self.status is MODEL_STATUS.UNPUBLISHED and not self.status_reason:
            raise ValueError(
                "Unpublished models must set status_reason with a link to the "
                "related issue."
            )
        if self.status is MODEL_STATUS.PUBLISHED and self.status_reason:
            raise ValueError("status_reason must not be set for published models.")
        return self

    @model_validator(mode="after")
    def _validate_standalone_component_labels(self) -> QAIHMModelInternalConfig:
        # Labels are perf.yaml component keys, so a duplicate silently merges two
        # components' metrics.
        labels = list(self.standalone_components.values())
        if len(labels) != len(set(labels)):
            raise ValueError(
                f"standalone_components labels must be unique, got {sorted(labels)}."
            )
        return self

    def validate_standalone_components(
        self, manifest: QAIHMModelManifest | None = None
    ) -> None:
        """
        Check standalone_components against the model's manifest.

        Cross-file, so it cannot live in the model_validator: manifest.yaml is a
        separate file. Defaults to this config's own cached manifest.
        """
        if not self.standalone_components:
            return
        manifest = manifest if manifest is not None else self.manifest

        # Non-LLM collection models already get a tab per component.
        if not manifest.model_type_llm:
            raise ValueError(
                "standalone_components can only be set on LLM/VLM models "
                "(model_type_llm must be true). Non-LLM collection models already "
                "get one model card tab per component."
            )

        # `name` keys the consolidated backbone entry, so reusing it overwrites it.
        if (
            manifest.name is not None
            and manifest.name in self.standalone_components.values()
        ):
            raise ValueError(
                f"standalone_components label {manifest.name!r} collides with the "
                f"model name, which keys the consolidated backbone entry in perf.yaml."
            )

    def validate_llm_details(self, manifest: QAIHMModelManifest | None = None) -> None:
        """
        Check llm_details against the model's manifest.

        Cross-file, so it cannot live in a model_validator: model_type_llm is in
        manifest.yaml and llm_details is here. Defaults to this config's own manifest.
        """
        manifest = manifest if manifest is not None else self.manifest
        if manifest.model_type_llm and not self.llm_details:
            raise ValueError("llm_details must be set if model type is LLM")
        if self.llm_details and not manifest.model_type_llm:
            raise ValueError("Model type must be LLM if llm_details is set")

    def perf_component_key(
        self, component: str | None, manifest: QAIHMModelManifest | None = None
    ) -> str:
        """
        perf.yaml key for a component. Standalone components use their declared label.
        Any other component on a model that declares standalone ones is an LLM backbone
        part, which collapses to the model name because the parts are only measured
        together end-to-end. Regular collection models keep their raw component names.

        Only the fallback needs the manifest, so a raw component name on a model with
        no standalone components resolves without loading one at all.
        """
        if component is not None:
            if component in self.standalone_components:
                return self.standalone_components[component]
            if not self.standalone_components:
                return component
        manifest = manifest if manifest is not None else self.manifest
        assert manifest.id is not None, "perf_component_key needs a full model manifest"
        return manifest.name or manifest.id

    @classmethod
    @functools.cache
    def from_model(cls, model_id: str) -> QAIHMModelInternalConfig:
        """
        Load internal-config.yaml for the given model.

        Cached: these files are read-only config that nothing rewrites in-process, and
        callers like ScorecardJobSummary.add_to_perf ask for the same model's config
        once per export test.
        """
        if not os.path.exists(QAIHM_MODELS_ROOT / model_id):
            raise ValueError(f"{model_id} does not exist")

        internal_config_path = SCORECARD_MODELS_ROOT / model_id / "internal-config.yaml"
        # No empty fallback: it would silently drop the website fields and
        # skip_scorecard for a model whose config went missing.
        if not internal_config_path.exists():
            raise FileNotFoundError(
                f"{model_id} has no {internal_config_path}. Every in-tree model needs one."
            )
        config = cls.from_yaml(internal_config_path)
        config._model_id = model_id
        # Gated on the field being set so the common case doesn't pay a manifest load.
        if config.standalone_components:
            config.validate_standalone_components()
        config.validate_llm_details()
        return config

    def to_proto(self) -> model_manifest_pb2.ModelManifest.Internal:
        """Serialize to the ``ModelManifest.Internal`` proto."""
        msg = _Internal(
            status=status_to_proto(self.status),
            status_reason=self.status_reason,
            applicable_scenarios=self.applicable_scenarios,
            related_models=self.related_models,
            model_maker_id=self.model_maker_id,
            has_static_banner=self.has_static_banner,
            has_animated_banner=self.has_animated_banner,
            imsdk_supported=self.imsdk_supported,
            private_perf_form_factors=[
                form_factor_to_proto(ff)
                for ff in (self.private_perf_form_factors or [])
            ],
            standalone_components=self.standalone_components,
            skip_hub_tests_and_scorecard=self.skip_hub_tests_and_scorecard,
            skip_scorecard=self.skip_scorecard,
            freeze_perf_yaml=self.freeze_perf_yaml,
            test_split=_TEST_SPLIT_TO_PROTO[self.test_split],
            skip_export=self.skip_export,
            skip_evaluate=self.skip_evaluate,
            numerics_threshold_override=self.numerics_threshold_override,
            global_requirements_incompatible=self.global_requirements_incompatible,
            on_demand_only=self.on_demand_only,
        )
        if self.llm_details is not None:
            msg.llm_details.CopyFrom(
                _Internal.LLMDetails(
                    call_to_action=call_to_action_to_proto(
                        self.llm_details.call_to_action
                    ),
                    genie_compatible=self.llm_details.genie_compatible,
                    geniex_llamacpp_compatible=self.llm_details.geniex_llamacpp_compatible,
                )
            )
        return msg

    @classmethod
    def from_proto(
        cls, proto: model_manifest_pb2.ModelManifest.Internal
    ) -> QAIHMModelInternalConfig:
        """Build a config from a ``ModelManifest.Internal`` proto."""
        if (
            proto.status
            == model_manifest_pb2.ModelManifest.Internal.MODEL_STATUS_UNSPECIFIED
        ):
            raise ValueError("status is required in internal-config.yaml.")
        llm_details = None
        if proto.HasField("llm_details"):
            llm_details = LLMDetails(
                call_to_action=call_to_action_from_proto(
                    proto.llm_details.call_to_action
                ),
                genie_compatible=proto.llm_details.genie_compatible,
                geniex_llamacpp_compatible=proto.llm_details.geniex_llamacpp_compatible,
            )
        return cls(
            status=status_from_proto(proto.status),
            status_reason=(
                proto.status_reason if proto.HasField("status_reason") else None
            ),
            applicable_scenarios=list(proto.applicable_scenarios),
            related_models=list(proto.related_models),
            model_maker_id=(
                proto.model_maker_id if proto.HasField("model_maker_id") else None
            ),
            has_static_banner=proto.has_static_banner,
            has_animated_banner=proto.has_animated_banner,
            imsdk_supported=proto.imsdk_supported,
            private_perf_form_factors=[
                form_factor_from_proto(ff) for ff in proto.private_perf_form_factors
            ]
            or None,
            standalone_components=dict(proto.standalone_components),
            skip_hub_tests_and_scorecard=proto.skip_hub_tests_and_scorecard,
            skip_scorecard=proto.skip_scorecard,
            freeze_perf_yaml=proto.freeze_perf_yaml,
            test_split=_PROTO_TO_TEST_SPLIT[proto.test_split],
            skip_export=proto.skip_export,
            skip_evaluate=proto.skip_evaluate,
            numerics_threshold_override=(
                proto.numerics_threshold_override
                if proto.HasField("numerics_threshold_override")
                else None
            ),
            global_requirements_incompatible=proto.global_requirements_incompatible,
            on_demand_only=proto.on_demand_only,
            llm_details=llm_details,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | os.PathLike,
        create_empty_if_no_file: bool = False,
    ) -> QAIHMModelInternalConfig:
        """Read an internal-config.yaml, the YAML form of ``ModelManifest.Internal``."""
        return cls.from_proto(read_proto_yaml(path, _Internal))

    def to_yaml(self, path: str | os.PathLike, **kwargs: Any) -> bool:  # type: ignore[override]
        """Write this config to ``path`` as the YAML form of ``ModelManifest.Internal``."""
        write_proto_yaml(self.to_proto(), path)
        return True

    @property
    def runs_in_scorecard(self) -> bool:
        """Whether the model runs in scorecard."""
        return not self.skip_hub_tests_and_scorecard and not self.skip_scorecard

    @property
    def is_llm(self) -> bool:
        """
        True if this model is a large language model and produces perf updates
        through a QDC workflow.
        """
        return self.test_split is TestRunnerSplit.LLM


def write_publish_status(
    model_id: str, status: MODEL_STATUS, status_reason: str | None
) -> Path:
    """Set status / status_reason in the model's internal-config.yaml."""
    path = SCORECARD_MODELS_ROOT / model_id / "internal-config.yaml"
    config = QAIHMModelInternalConfig.from_yaml(path)
    # model_validate re-runs the status / status_reason coupling check.
    updated = QAIHMModelInternalConfig.model_validate(
        {**config.model_dump(), "status": status, "status_reason": status_reason}
    )
    updated.to_yaml(path)

    # from_model is cached process-wide; drop the now-stale copy.
    QAIHMModelInternalConfig.from_model.cache_clear()
    return path


def model_perf_component_key(model_id: str, component: str | None) -> str:
    """perf.yaml key for one component of the given model. See perf_component_key."""
    return QAIHMModelInternalConfig.from_model(model_id).perf_component_key(component)


def _scorecard_llm_configs() -> dict[str, QAIHMModelInternalConfig]:
    """{model_id: internal_config} for every scorecard-eligible test_split: llm model."""
    out: dict[str, QAIHMModelInternalConfig] = {}
    for model_id in MODEL_IDS:
        sc = QAIHMModelInternalConfig.from_model(model_id)
        if sc.is_llm and sc.runs_in_scorecard:
            out[model_id] = sc
    return out


def get_llm_model_ids() -> set[str]:
    """Scorecard-eligible pytorch recipes with test_split == llm."""
    return set(_scorecard_llm_configs())


def get_weekly_llm_model_ids() -> set[str]:
    """Scorecard LLM model IDs that run on the weekly schedule."""
    return {m for m, sc in _scorecard_llm_configs().items() if not sc.on_demand_only}


def get_on_demand_llm_model_ids() -> set[str]:
    """Scorecard LLM model IDs that only run when explicitly requested."""
    return {m for m, sc in _scorecard_llm_configs().items() if sc.on_demand_only}

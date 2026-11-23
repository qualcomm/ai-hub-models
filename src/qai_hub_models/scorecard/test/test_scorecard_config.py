# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from qai_hub_models_cli.proto import model_manifest_pb2

from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs._info_yaml_llm_details import (
    LLM_CALL_TO_ACTION,
    LLMDetails,
)
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.envvars import SpecialModelSetting
from qai_hub_models.scorecard.internal_config_yaml import (
    QAIHMModelInternalConfig,
    TestRunnerSplit,
    get_llm_model_ids,
    get_on_demand_llm_model_ids,
    get_weekly_llm_model_ids,
    write_publish_status,
)
from qai_hub_models.scorecard.results.yaml import ComponentNamesYaml
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.device import FormFactor
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    QAIHM_MODELS_ROOT,
    SCORECARD_MODELS_ROOT,
)

# Pinned rather than imported so that moving a field back must edit this test.
WEBSITE_FIELDS = {
    "status",
    "status_reason",
    "applicable_scenarios",
    "has_animated_banner",
    "has_static_banner",
    "imsdk_supported",
    "llm_details",
    "model_maker_id",
    "preview",
    "private_perf_form_factors",
    "related_models",
}

# Regional Llama variants: test_split: llm, on_demand_only: true. Never picked
# up by the weekly schedule; only enabled via the llm_on_demand token.
REGIONAL_MODELS = {
    "llama_v3_1_sea_lion_3_5_8b_r",
    "llama_v3_elyza_jp_8b",
    "llama_v3_taide_8b_chat",
}


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_every_model_has_scorecard_config(model_id: str) -> None:
    path = SCORECARD_MODELS_ROOT / model_id / "internal-config.yaml"
    assert path.exists(), (
        f"{model_id}: missing {path.relative_to(SCORECARD_MODELS_ROOT)}"
    )
    QAIHMModelInternalConfig.from_model(model_id)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_website_fields_not_in_manifest(model_id: str) -> None:
    # The manifest loader silently drops these keys for old standalone recipes, so
    # this is what keeps in-tree manifests from carrying them.
    raw = load_yaml(QAIHM_MODELS_ROOT / model_id / "manifest.yaml") or {}
    stale = sorted(WEBSITE_FIELDS & set(raw))
    assert not stale, (
        f"{model_id}: {stale} belong in internal-config.yaml, not manifest.yaml."
    )


def test_no_orphan_scorecard_configs() -> None:
    orphans = sorted(
        p.parent.name
        for p in SCORECARD_MODELS_ROOT.glob("*/internal-config.yaml")
        if p.parent.name not in MODEL_IDS
    )
    assert not orphans, f"internal-config.yaml for models that do not exist: {orphans}"


def test_website_fields_are_scorecard_config_fields() -> None:
    assert set(QAIHMModelInternalConfig.model_fields) >= WEBSITE_FIELDS
    assert not WEBSITE_FIELDS & set(QAIHMModelManifest.model_fields)


def test_every_internal_config_field_is_in_the_proto() -> None:
    proto_fields = {
        f.name for f in model_manifest_pb2.ModelManifest.Internal.DESCRIPTOR.fields
    }
    assert set(QAIHMModelInternalConfig.model_fields) == proto_fields


def test_manifest_ignores_unrecognized_keys() -> None:
    manifest = QAIHMModelManifest.model_validate(
        {"id": "x", "name": "X", "status": "published", "form_factors": ["Phone"]}
    )
    assert manifest.name == "X"
    assert not {"status", "form_factors"} & set(manifest.to_yaml_dict())


def test_missing_scorecard_config_raises(tmp_path: Path) -> None:
    model_id = sorted(MODEL_IDS)[0]
    with (
        mock.patch(
            "qai_hub_models.scorecard.internal_config_yaml.SCORECARD_MODELS_ROOT",
            tmp_path,
        ),
        pytest.raises(FileNotFoundError, match="Every in-tree model needs one"),
    ):
        QAIHMModelInternalConfig.from_model.__wrapped__(
            QAIHMModelInternalConfig, model_id
        )


def test_status_is_required() -> None:
    with pytest.raises(ValueError, match="status"):
        QAIHMModelInternalConfig.model_validate({})


@pytest.mark.parametrize(
    ("status", "reason", "error"),
    [
        (MODEL_STATUS.UNPUBLISHED, None, "Unpublished models must set"),
        (MODEL_STATUS.PUBLISHED, "tracked in #1", "must not be set"),
        (MODEL_STATUS.UNPUBLISHED, "tracked in #1", None),
        (MODEL_STATUS.PENDING, None, None),
        (MODEL_STATUS.PENDING, "waiting on banner", None),
        (MODEL_STATUS.PUBLISHED, None, None),
    ],
)
def test_status_reason_coupling(
    status: MODEL_STATUS, reason: str | None, error: str | None
) -> None:
    def build() -> QAIHMModelInternalConfig:
        return QAIHMModelInternalConfig(status=status, status_reason=reason)

    if error is None:
        build()
    else:
        with pytest.raises(ValueError, match=error):
            build()


def test_write_publish_status_round_trips_other_fields(tmp_path: Path) -> None:
    model_id = sorted(MODEL_IDS)[0]
    config = tmp_path / model_id / "internal-config.yaml"
    config.parent.mkdir()
    QAIHMModelInternalConfig(
        status=MODEL_STATUS.PUBLISHED,
        has_animated_banner=True,
        related_models=["b", "a"],
    ).to_yaml(config)
    with mock.patch(
        "qai_hub_models.scorecard.internal_config_yaml.SCORECARD_MODELS_ROOT",
        tmp_path,
    ):
        write_publish_status(model_id, MODEL_STATUS.PENDING, "no runtimes")
        pending = QAIHMModelInternalConfig.from_yaml(config)
        assert pending.status is MODEL_STATUS.PENDING
        assert pending.status_reason == "no runtimes"
        assert pending.has_animated_banner
        assert pending.related_models == ["b", "a"]

        write_publish_status(model_id, MODEL_STATUS.PUBLISHED, None)
        published = QAIHMModelInternalConfig.from_yaml(config)
        assert published.status is MODEL_STATUS.PUBLISHED
        assert published.status_reason is None
        assert "status_reason" not in config.read_text()

        with pytest.raises(ValueError, match="Unpublished models must set"):
            write_publish_status(model_id, MODEL_STATUS.UNPUBLISHED, None)


def test_internal_config_proto_round_trip() -> None:
    config = QAIHMModelInternalConfig(
        status=MODEL_STATUS.UNPUBLISHED,
        status_reason="tracked in #1",
        test_split=TestRunnerSplit.LLM,
        on_demand_only=True,
        numerics_threshold_override=15.0,
        private_perf_form_factors=[FormFactor.AUTO],
        standalone_components={"vision_encoder": "Encoder"},
        llm_details=LLMDetails(call_to_action=LLM_CALL_TO_ACTION.DOWNLOAD),
    )
    assert QAIHMModelInternalConfig.from_proto(config.to_proto()) == config


def test_internal_config_missing_status_rejected(tmp_path: Path) -> None:
    path = tmp_path / "internal-config.yaml"
    path.write_text("has_static_banner: true\n")
    with pytest.raises(ValueError, match="status is required"):
        QAIHMModelInternalConfig.from_yaml(path)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_pip_commands_require_global_requirements_incompatible(model_id: str) -> None:
    """If manifest.yaml sets any pre/post pip install commands, the model's
    internal-config.yaml must have global_requirements_incompatible: true.
    """
    sc = QAIHMModelInternalConfig.from_model(model_id)
    manifest = sc.manifest
    if not manifest.pre_pip_install_commands and not manifest.post_pip_install_commands:
        return
    assert sc.global_requirements_incompatible, (
        f"{model_id}: pre_pip_install_commands/post_pip_install_commands is set in "
        "manifest.yaml, but global_requirements_incompatible is not true in "
        "internal-config.yaml."
    )


def test_duplicate_standalone_component_labels_rejected() -> None:
    # Two components sharing a label would silently merge in perf.yaml.
    with pytest.raises(ValueError, match="labels must be unique"):
        QAIHMModelInternalConfig(
            status=MODEL_STATUS.PENDING,
            standalone_components={
                "vision_encoder": "Encoder",
                "audio_encoder": "Encoder",
            },
        )


class TestStandaloneComponentsValidation:
    """
    standalone_components lives in internal-config.yaml but is checked against
    manifest.yaml, so these cross-file checks run through
    QAIHMModelInternalConfig.validate_standalone_components rather than a validator.
    """

    @staticmethod
    def _config(standalone_components: dict[str, str]) -> QAIHMModelInternalConfig:
        return QAIHMModelInternalConfig(
            status=MODEL_STATUS.PENDING, standalone_components=standalone_components
        )

    @staticmethod
    def _manifest(
        model_type_llm: bool = True, name: str | None = None
    ) -> QAIHMModelManifest:
        return QAIHMModelManifest(model_type_llm=model_type_llm, name=name)

    def test_llm_with_standalone_component_is_valid(self) -> None:
        config = self._config({"vision_encoder": "Vision-Encoder"})
        manifest = self._manifest()
        config.validate_standalone_components(manifest)
        assert config.perf_component_key("vision_encoder", manifest) == "Vision-Encoder"

    def test_directly_constructed_config_has_no_manifest(self) -> None:
        # `manifest` only resolves on from_model configs; a directly-constructed one
        # must be given a manifest rather than silently guessing a model id.
        config = self._config({})
        with pytest.raises(ValueError, match="only available on configs loaded via"):
            _ = config.manifest

    def test_non_llm_rejected(self) -> None:
        config = self._config({"encoder": "Encoder"})
        with pytest.raises(ValueError, match="can only be set on LLM/VLM models"):
            config.validate_standalone_components(self._manifest(model_type_llm=False))

    def test_label_colliding_with_model_name_rejected(self) -> None:
        # `name` keys the consolidated backbone entry, so reusing it would
        # overwrite the end-to-end numbers with a single component's.
        config = self._config({"vision_encoder": "My-VLM"})
        with pytest.raises(ValueError, match="collides with the"):
            config.validate_standalone_components(self._manifest(name="My-VLM"))


def test_from_model_and_manifest_are_cached() -> None:
    """
    add_to_perf asks for the same model's config once per export test, so a repeat
    from_model must not re-parse internal-config.yaml or manifest.yaml.
    """
    model_id = sorted(MODEL_IDS)[0]
    first = QAIHMModelInternalConfig.from_model(model_id)
    assert QAIHMModelInternalConfig.from_model(model_id) is first
    assert first.manifest is first.manifest


def test_standalone_components_are_real_components() -> None:
    """
    Every standalone_components key must name a component the model actually declares
    in Python, otherwise the scorecard would ask to profile a component that does not
    exist and the tab would silently never appear.
    """
    known_components = ComponentNamesYaml.from_intermediates()
    errors: list[str] = []
    for model_id in MODEL_IDS:
        standalone_components = QAIHMModelInternalConfig.from_model(
            model_id
        ).standalone_components
        if not standalone_components:
            continue
        components = known_components.get(model_id)
        if components is None:
            errors.append(
                f"{model_id}: declares standalone_components but has no recorded "
                f"component names; it must be a collection model."
            )
            continue
        unknown = set(standalone_components) - set(components)
        if unknown:
            errors.append(
                f"{model_id}: standalone_components {sorted(unknown)} are not "
                f"components of this model (has {sorted(components)})."
            )
    assert not errors, "\n".join(errors)


def test_weekly_and_on_demand_partition_all_llms() -> None:
    weekly = get_weekly_llm_model_ids()
    on_demand = get_on_demand_llm_model_ids()
    assert weekly
    assert on_demand
    assert weekly.isdisjoint(on_demand)
    assert weekly | on_demand == get_llm_model_ids()


def test_regional_models_are_on_demand() -> None:
    """Regional variants run only on-demand via llm_on_demand, never on a schedule."""
    llm_ids = get_llm_model_ids()
    on_demand = get_on_demand_llm_model_ids()
    weekly = get_weekly_llm_model_ids()
    for model_id in REGIONAL_MODELS:
        assert model_id in llm_ids
        assert model_id in on_demand
        assert model_id not in weekly


def test_tokens_resolve_to_expected_llm_sets() -> None:
    for token, expected in [
        (SpecialModelSetting.LLM, get_weekly_llm_model_ids()),
        (SpecialModelSetting.LLM_ON_DEMAND, get_on_demand_llm_model_ids()),
    ]:
        torch_ids, static_ids = validate_and_split_enabled_models({token})
        assert torch_ids == expected
        assert not static_ids


def test_combined_weekend_tokens_round_trip() -> None:
    """The workflow's 'pytorch_no_llm,static,llm' string is valid."""
    tokens: set = {
        SpecialModelSetting.PYTORCH_NO_LLM,
        SpecialModelSetting.STATIC,
        SpecialModelSetting.LLM,
    }
    torch_ids, static_ids = validate_and_split_enabled_models(tokens)
    assert get_weekly_llm_model_ids() <= torch_ids
    assert static_ids

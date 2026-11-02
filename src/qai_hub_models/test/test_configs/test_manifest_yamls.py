# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import MODEL_DOMAIN_USE_CASES
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.configs.manifest_yaml import (
    MODEL_DOMAIN,
    MODEL_USE_CASE,
    LMQuantizationDetails,
    QAIHMModelManifest,
    TechnicalDetails,
)
from qai_hub_models.scorecard.model_info import WebsiteModelInfo
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, QAIHM_WEB_ASSET
from qai_hub_models.utils.device import CANARY_DEVICES
from qai_hub_models.utils.metrics import VALID_METRIC_PAIRS
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_MODELS_ROOT
from qai_hub_models.utils.url_check import validate_urls_exist

HF_PIPELINE_TAGS = {
    "keypoint-detection",
    "text-classification",
    "token-classification",
    "table-question-answering",
    "question-answering",
    "zero-shot-classification",
    "translation",
    "summarization",
    "conversational",
    "feature-extraction",
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "sentence-similarity",
    "text-to-speech",
    "text-to-audio",
    "automatic-speech-recognition",
    "audio-to-audio",
    "audio-classification",
    "voice-activity-detection",
    "depth-estimation",
    "gaze-estimation",
    "image-classification",
    "object-detection",
    "image-segmentation",
    "text-to-image",
    "image-to-text",
    "image-to-image",
    "image-to-video",
    "unconditional-image-generation",
    "video-classification",
    "reinforcement-learning",
    "robotics",
    "tabular-classification",
    "tabular-regression",
    "tabular-to-text",
    "table-to-text",
    "multiple-choice",
    "text-retrieval",
    "time-series-forecasting",
    "text-to-video",
    "visual-question-answering",
    "document-question-answering",
    "zero-shot-image-classification",
    "graph-ml",
    "mask-generation",
    "zero-shot-object-detection",
    "text-to-3d",
    "image-to-3d",
    "video-object-tracking",
    "other",
}


def test_all_domains_accounted_for() -> None:
    # Verify all use cases and domains are accounted for in the mapping
    assert len(MODEL_DOMAIN_USE_CASES) == len(MODEL_DOMAIN)
    use_cases = {
        ucase for ucases in MODEL_DOMAIN_USE_CASES.values() for ucase in ucases
    }
    assert len(use_cases) == len(MODEL_USE_CASE)


def test_model_usecase_to_hf_pipeline_tag() -> None:
    for use_case in MODEL_USE_CASE:
        assert use_case.map_to_hf_pipeline_tag() in HF_PIPELINE_TAGS


def _check_website_facing(info: WebsiteModelInfo) -> None:
    """Enforce website / scorecard / publication rules for an in-tree recipe.

    These checks live here, not in ``QAIHMModelManifest.check_fields``, so
    that ``check_fields`` remains a universal-correctness validator that
    external recipes can pass without pulling in Qualcomm-catalog rules
    (banner presence, canary devices, headline copy style, etc.).
    """
    manifest, internal = info.manifest, info.internal
    assert manifest.id is not None
    assert manifest.name is not None
    assert manifest.headline is not None

    if " " in manifest.name:
        raise ValueError("Model Name must not have a space.")
    if "_" in manifest.name:
        raise ValueError("Model Name should use dashes (-) instead of underscores.")

    if not manifest.headline.endswith("."):
        raise ValueError("Model headlines must end with a period.")

    for r_model in internal.related_models:
        if r_model == manifest.id:
            raise ValueError(f"Model {r_model} cannot be related to itself.")

    if (
        manifest.research_paper is not None
        and manifest.research_paper.startswith("https://arxiv.org/")
        and "/abs/" not in manifest.research_paper
    ):
        raise ValueError(
            "Arxiv links should be `abs` links, not link directly to pdfs."
        )

    # status_reason coupling is enforced by QAIHMModelInternalConfig's validator.
    if info.is_published:
        can_be_published, reason = info.can_promote_to_published()
        if not can_be_published:
            raise ValueError(f"Model cannot be published: {reason}")

    if manifest.numerics_benchmark is not None:
        pair = (
            manifest.numerics_benchmark.metric_name,
            manifest.numerics_benchmark.unit,
        )
        if pair not in VALID_METRIC_PAIRS:
            valid_pairs_str = ", ".join(
                f"({n!r}, {u!r})" for n, u in sorted(VALID_METRIC_PAIRS)
            )
            raise ValueError(
                f"numerics_benchmark metric_name={pair[0]!r} with "
                f"unit={pair[1]!r} does not match any known metric. "
                f"Valid pairs:\n  {valid_pairs_str}"
            )

    if info.is_published:
        if not os.path.exists(manifest.get_package_path() / "manifest.yaml"):
            raise ValueError("All published models must have a manifest.yaml")
        if not os.path.exists(manifest.get_package_path() / "perf.yaml"):
            raise ValueError("All published models must have a perf.yaml")
        if not manifest.supports_at_least_1_runtime:
            raise ValueError("Published models must support at least one export path")
        if not internal.has_static_banner:
            raise ValueError("Published models must have a static asset.")

    expected_qaihm_repo = Path("src") / "qai_hub_models" / "models" / manifest.id
    if expected_qaihm_repo != ASSET_CONFIG.get_qaihm_repo(manifest.id):
        raise ValueError("QAIHM repo not pointing to expected relative path")

    if manifest.model_type_llm and internal.llm_details is not None:
        if internal.llm_details.call_to_action in {
            LLM_CALL_TO_ACTION.DOWNLOAD,
            LLM_CALL_TO_ACTION.DOWNLOAD_AND_VIEW_README,
        }:
            if manifest.restrict_model_sharing:
                raise ValueError(
                    "LLM call to action cannot be 'download' when restrict "
                    "model sharing is enabled."
                )
        elif not manifest.restrict_model_sharing and os.path.exists(
            QAIHM_MODELS_ROOT / manifest.id / "release-assets.yaml"
        ):
            raise ValueError(
                "LLM has downloadable assets but the call to action is not 'download'."
            )

    if manifest.default_device not in CANARY_DEVICES:
        raise ValueError(
            f"Default device must be any of these canary devices: {CANARY_DEVICES}"
        )


def _collect_website_urls(info: WebsiteModelInfo) -> list[tuple[str, str]]:
    """Build the URL HEAD-check list for an in-tree recipe."""
    manifest, internal = info.manifest, info.internal
    assert manifest.id is not None
    urls: list[tuple[str, str]] = []
    if internal.has_static_banner:
        urls.append(
            (
                ASSET_CONFIG.get_web_asset_url(manifest.id, QAIHM_WEB_ASSET.STATIC_IMG),
                "Static banner does not exist",
            )
        )
    if internal.has_animated_banner:
        urls.append(
            (
                ASSET_CONFIG.get_web_asset_url(
                    manifest.id, QAIHM_WEB_ASSET.ANIMATED_MOV
                ),
                "Animated banner does not exist",
            )
        )
    if manifest.license:
        urls.append((manifest.license, "License does not exist"))
    if manifest.research_paper:
        urls.append((manifest.research_paper, "Research paper does not exist"))
    if manifest.source_repo:
        urls.append((manifest.source_repo, "Source repo does not exist"))

    return urls


def _validate_model(model_id: str) -> None:
    info = WebsiteModelInfo.from_model(model_id)
    manifest = info.manifest
    QAIHMModelManifest.model_validate(manifest)
    assert manifest.id == model_id, (
        f"{model_id} config ID does not match the model's folder name"
    )
    _check_website_facing(info)
    validate_urls_exist(_collect_website_urls(info))
    manifest.check_geniex_runtime_technical_details()


def test_manifest_yaml() -> None:
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_validate_model, model_id): model_id for model_id in MODEL_IDS
        }
        for future in as_completed(futures):
            model_id = futures[future]
            try:
                future.result()
            except Exception as err:
                errors.append(f"{model_id}: {err!s}")
    assert not errors, f"{len(errors)} model(s) failed validation:\n" + "\n".join(
        errors
    )


def test_export_paths_include_aot_on_jit() -> None:
    """
    JIT models (`requires_aot_prepare=False`) can still be exported to AOT
    runtimes via the JIT-compile + link path. `get_supported_paths_for_export`
    must include those AOT runtimes even though
    `get_supported_paths_for_testing` filters them out.
    """
    manifest = QAIHMModelManifest.from_model("vit")
    assert not manifest.requires_aot_prepare

    testing_paths = manifest.get_supported_paths_for_testing()
    export_paths = manifest.get_supported_paths_for_export()

    for precision, runtimes in testing_paths.items():
        assert precision in export_paths
        assert set(runtimes).issubset(set(export_paths[precision]))

    assert any(TargetRuntime.QNN_CONTEXT_BINARY in rts for rts in export_paths.values())
    assert not any(
        TargetRuntime.QNN_CONTEXT_BINARY in rts for rts in testing_paths.values()
    )


# ---------------------------------------------------------------------------
# lm_quantization_details validation (QAIHMModelManifest.check_fields):
#   1. the section is LLM/VLM-only (requires model_type_llm)
#   2. every recipe key must be in supported_precisions
# These are template-shaped manifests (id=None) so the website-facing
# checks are skipped and only the build/export invariants run.
# ---------------------------------------------------------------------------
def _lm_details() -> LMQuantizationDetails:
    """A minimal valid recipe (contract W4A16 precision + a single Calibration)."""
    # model_validate, not the constructor: an omitted `precision` is filled by
    # _fill_precision, which is the path manifests take.
    return LMQuantizationDetails.model_validate({"recipe": [{"name": "Calibration"}]})


class TestLMQuantizationDetailsValidation:
    def test_llm_with_supported_precision_is_valid(self) -> None:
        manifest = QAIHMModelManifest(
            model_type_llm=True,
            supported_precisions=[Precision.w4a16],
            lm_quantization_details={Precision.w4a16: _lm_details()},
        )
        assert manifest.lm_quantization_details[Precision.w4a16].recipe.backbone

    def test_non_llm_with_recipe_rejected(self) -> None:
        with pytest.raises(ValueError, match="can only be set on LLM/VLM models"):
            QAIHMModelManifest(
                model_type_llm=False,
                supported_precisions=[Precision.w4a16],
                lm_quantization_details={Precision.w4a16: _lm_details()},
            )

    def test_recipe_for_unsupported_precision_rejected(self) -> None:
        # A recipe keyed by a precision absent from supported_precisions is
        # unreachable, so it must fail loud (and name the offending precision).
        with pytest.raises(ValueError, match="are not in"):
            QAIHMModelManifest(
                model_type_llm=True,
                supported_precisions=[Precision.w4],
                lm_quantization_details={Precision.w4a16: _lm_details()},
            )

    def test_omitted_precision_is_re_emitted_in_full(self) -> None:
        # The scorecard rewrites manifest.yaml wholesale (collect_scorecard_results
        # --sync-code-gen) with exclude_defaults=True, which deletes any field that
        # merely equals its default -- and in lm_schema the defaults *are* the
        # W4A16 contract. The schema fills the contract at validation instead, so a
        # section that omits `precision:` re-emits it fully fleshed out.
        details = _lm_details()
        dumped = details.model_dump(exclude_defaults=True)

        assert dumped["precision"]["activations"] == "int16"
        assert dumped["precision"]["kv_cache"] == "int8"
        assert dumped["precision"]["embedding"] == "int16"
        assert dumped["precision"]["lm_head"] == {"qtype": "int8", "granularity": "PCQ"}
        # Sugar forms the lm_schema before-validators widen are re-narrowed.
        assert dumped["precision"]["blocks"] == {"qtype": "int4", "granularity": "PCQ"}
        assert dumped["recipe"] == [{"name": "Calibration"}]
        assert LMQuantizationDetails.model_validate(dumped) == details

    def test_partial_precision_is_re_emitted_in_full(self) -> None:
        # A partially written block keeps what was authored and gains the rest.
        details = LMQuantizationDetails.model_validate(
            {
                "precision": {"lm_head": {"qtype": "int16"}},
                "recipe": [{"name": "Calibration"}],
            }
        )
        dumped = details.model_dump(exclude_defaults=True)

        assert dumped["precision"]["lm_head"] == {
            "qtype": "int16",
            "granularity": "PCQ",
        }
        assert dumped["precision"]["kv_cache"] == "int8"

    def test_serialization_keeps_components_for_vlms(self) -> None:
        # A recipe with a visual chain has to keep the component form; only the
        # single-component case collapses to the bare list.
        details = LMQuantizationDetails.model_validate(
            {
                "recipe": {
                    "backbone": [{"name": "Calibration"}],
                    "visual": [{"name": "Calibration"}],
                }
            }
        )
        dumped = details.model_dump(exclude_defaults=True)

        assert dumped["recipe"]["backbone"] == [{"name": "Calibration"}]
        assert dumped["recipe"]["visual"] == [{"name": "Calibration"}]
        assert LMQuantizationDetails.model_validate(dumped) == details

    def test_empty_recipe_section_is_valid_for_non_llm(self) -> None:
        # An empty section must not trip the LLM-only guard (the guard is on a
        # non-empty section only), so ordinary non-LLM manifests still validate.
        manifest = QAIHMModelManifest(
            model_type_llm=False,
            supported_precisions=[Precision.float],
        )
        assert manifest.lm_quantization_details == {}


# ---------------------------------------------------------------------------
# check_geniex_runtime_technical_details, model_infer_id half: model_infer_id
# points the website at a HuggingFace repo to download, so it is required
# exactly when we may redistribute weights (restrict_model_sharing=False).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "runtime", [TargetRuntime.GENIEX_QAIRT, TargetRuntime.GENIEX_LLAMACPP]
)
@pytest.mark.parametrize(
    ("restricted", "infer_id", "expected_error"),
    [
        (False, "ai-hub-models/Test-Model", None),
        (False, None, "must define model_infer_id"),
        (True, "ai-hub-models/Test-Model", "must not define model_infer_id"),
        (True, None, None),
    ],
)
def test_geniex_model_infer_id(
    runtime: TargetRuntime,
    restricted: bool,
    infer_id: str | None,
    expected_error: str | None,
) -> None:
    details: TechnicalDetails = {"Supported Input Modalities": "text"}
    if infer_id is not None:
        details["model_infer_id"] = infer_id
    manifest = QAIHMModelManifest(
        id="test_model",
        restrict_model_sharing=restricted,
        runtime_technical_details={runtime: details},
    )

    if expected_error is None:
        manifest.check_geniex_runtime_technical_details()
    else:
        with pytest.raises(ValueError, match=expected_error):
            manifest.check_geniex_runtime_technical_details()


def test_geniex_model_infer_id_skipped_without_geniex_details() -> None:
    # Non-GenieX models have no GenieX section; don't force one into existence.
    QAIHMModelManifest(id="test_model").check_geniex_runtime_technical_details()


def test_geniex_runtime_technical_details_reports_all_errors() -> None:
    # A manifest that trips both halves must surface both, not just the first.
    manifest = QAIHMModelManifest(
        id="test_model",
        restrict_model_sharing=False,
        orchestrator_runtimes=[TargetRuntime.GENIEX_LLAMACPP],
        runtime_technical_details={TargetRuntime.GENIEX_QAIRT: {}},
    )
    with pytest.raises(ValueError, match="geniex_llamacpp") as excinfo:
        manifest.check_geniex_runtime_technical_details()
    assert "must define model_infer_id" in str(excinfo.value)

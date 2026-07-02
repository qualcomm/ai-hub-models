# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor, as_completed

from qai_hub_models.configs._info_yaml_enums import MODEL_DOMAIN_USE_CASES
from qai_hub_models.configs.info_yaml import (
    MODEL_DOMAIN,
    MODEL_USE_CASE,
    QAIHMModelInfo,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS

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


def _validate_model(model_id: str) -> None:
    info_spec = QAIHMModelInfo.from_model(model_id)
    QAIHMModelInfo.model_validate(info_spec, context=dict(validate_urls_exist=True))
    assert info_spec.id == model_id, (
        f"{model_id} config ID does not match the model's folder name"
    )
    info_spec.check_geniex_runtime_technical_details()


def test_info_yaml() -> None:
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

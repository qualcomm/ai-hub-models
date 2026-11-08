# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models_cli.proto import info_pb2, platform_pb2
from qai_hub_models_cli.proto.shared import precision_pb2, runtime_pb2

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import (
    MODEL_DOMAIN,
    MODEL_LICENSE,
    MODEL_STATUS,
    MODEL_TAG,
    MODEL_USE_CASE,
    VOICE_AI_SDK,
)
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.utils.device import FormFactor

_PRECISION_TO_PROTO: dict[str, int] = {
    "float": precision_pb2.PRECISION_FLOAT,
    "w8a8": precision_pb2.PRECISION_W8A8,
    "w8a16": precision_pb2.PRECISION_W8A16,
    "w16a16": precision_pb2.PRECISION_W16A16,
    "w4a16": precision_pb2.PRECISION_W4A16,
    "w4": precision_pb2.PRECISION_W4,
    "w8a8_mixed_int16": precision_pb2.PRECISION_W8A8_MIXED_INT16,
    "w8a16_mixed_int16": precision_pb2.PRECISION_W8A16_MIXED_INT16,
    "w8a8_mixed_fp16": precision_pb2.PRECISION_W8A8_MIXED_FP16,
    "w8a16_mixed_fp16": precision_pb2.PRECISION_W8A16_MIXED_FP16,
    "mxfp4": precision_pb2.PRECISION_MXFP4,
    "q8_0": precision_pb2.PRECISION_Q8_0,
    "q4_0": precision_pb2.PRECISION_Q4_0,
    "mixed": precision_pb2.PRECISION_MIXED,
    "mixed_with_float": precision_pb2.PRECISION_MIXED_WITH_FLOAT,
}

_RUNTIME_TO_PROTO: dict[str, int] = {
    "tflite": runtime_pb2.RUNTIME_TFLITE,
    "qnn_dlc": runtime_pb2.RUNTIME_QNN_DLC,
    "qnn_context_binary": runtime_pb2.RUNTIME_QNN_CONTEXT_BINARY,
    "onnx": runtime_pb2.RUNTIME_ONNX,
    "precompiled_qnn_onnx": runtime_pb2.RUNTIME_PRECOMPILED_QNN_ONNX,
    "genie": runtime_pb2.RUNTIME_GENIE,
    "geniex_qairt": runtime_pb2.RUNTIME_GENIEX_QAIRT,
    "voice_ai": runtime_pb2.RUNTIME_VOICE_AI,
    "geniex_llamacpp": runtime_pb2.RUNTIME_GENIEX_LLAMACPP,
}


def precision_to_proto(precision: Precision) -> int:
    key = str(precision)
    if key not in _PRECISION_TO_PROTO:
        raise ValueError(f"Unknown precision: {key}")
    return _PRECISION_TO_PROTO[key]


def runtime_to_proto(runtime: TargetRuntime) -> int:
    if runtime.value not in _RUNTIME_TO_PROTO:
        raise ValueError(f"Unknown runtime: {runtime.value}")
    return _RUNTIME_TO_PROTO[runtime.value]


_PROTO_TO_PRECISION: dict[int, str] = {v: k for k, v in _PRECISION_TO_PROTO.items()}
_PROTO_TO_RUNTIME: dict[int, str] = {v: k for k, v in _RUNTIME_TO_PROTO.items()}


def precision_from_proto(value: int) -> Precision:
    if value not in _PROTO_TO_PRECISION:
        raise ValueError(f"Unknown precision proto value: {value}")
    return Precision.parse(_PROTO_TO_PRECISION[value])


def runtime_from_proto(value: int) -> TargetRuntime:
    if value not in _PROTO_TO_RUNTIME:
        raise ValueError(f"Unknown runtime proto value: {value}")
    return TargetRuntime(_PROTO_TO_RUNTIME[value])


_FORM_FACTOR_TO_PROTO: dict[str, int] = {
    "Phone": platform_pb2.FORM_FACTOR_PHONE,
    "Tablet": platform_pb2.FORM_FACTOR_TABLET,
    "Auto": platform_pb2.FORM_FACTOR_AUTO,
    "XR": platform_pb2.FORM_FACTOR_XR,
    "Compute": platform_pb2.FORM_FACTOR_COMPUTE,
    "IoT": platform_pb2.FORM_FACTOR_IOT,
}


def form_factor_to_proto(ff: FormFactor) -> int:
    if ff.value not in _FORM_FACTOR_TO_PROTO:
        raise ValueError(f"Unknown form factor: {ff.value}")
    return _FORM_FACTOR_TO_PROTO[ff.value]


def form_factor_from_proto(value: int) -> FormFactor:
    for name, proto_value in _FORM_FACTOR_TO_PROTO.items():
        if proto_value == value:
            return FormFactor(name)
    raise ValueError(f"Unknown form factor proto value: {value}")


_LICENSE_TO_PROTO: dict[str, int] = {
    MODEL_LICENSE.UNLICENSED.value: info_pb2.MODEL_LICENSE_UNLICENSED,
    MODEL_LICENSE.COMMERCIAL.value: info_pb2.MODEL_LICENSE_COMMERCIAL,
    MODEL_LICENSE.AI_HUB_MODELS_LICENSE.value: info_pb2.MODEL_LICENSE_AI_HUB_MODELS_LICENSE,
    MODEL_LICENSE.APACHE_2_0.value: info_pb2.MODEL_LICENSE_APACHE_2_0,
    MODEL_LICENSE.MIT.value: info_pb2.MODEL_LICENSE_MIT,
    MODEL_LICENSE.BSD_3_CLAUSE.value: info_pb2.MODEL_LICENSE_BSD_3_CLAUSE,
    MODEL_LICENSE.CC_BY_4_0.value: info_pb2.MODEL_LICENSE_CC_BY_4_0,
    MODEL_LICENSE.AGPL_3_0.value: info_pb2.MODEL_LICENSE_AGPL_3_0,
    MODEL_LICENSE.GPL_3_0.value: info_pb2.MODEL_LICENSE_GPL_3_0,
    MODEL_LICENSE.CREATIVEML_OPENRAIL_M.value: info_pb2.MODEL_LICENSE_CREATIVEML_OPENRAIL_M,
    MODEL_LICENSE.CC_BY_NON_COMMERCIAL_4_0.value: info_pb2.MODEL_LICENSE_CC_BY_NON_COMMERCIAL_4_0,
    MODEL_LICENSE.OTHER_NON_COMMERCIAL.value: info_pb2.MODEL_LICENSE_OTHER_NON_COMMERCIAL,
    MODEL_LICENSE.LLAMA2.value: info_pb2.MODEL_LICENSE_LLAMA2,
    MODEL_LICENSE.LLAMA3.value: info_pb2.MODEL_LICENSE_LLAMA3,
    MODEL_LICENSE.TAIDE.value: info_pb2.MODEL_LICENSE_TAIDE,
    MODEL_LICENSE.FALCON3.value: info_pb2.MODEL_LICENSE_FALCON3,
    MODEL_LICENSE.GEMMA.value: info_pb2.MODEL_LICENSE_GEMMA,
    MODEL_LICENSE.LFM1_0.value: info_pb2.MODEL_LICENSE_LFM1_0,
    MODEL_LICENSE.AIMET_MODEL_ZOO.value: info_pb2.MODEL_LICENSE_AIMET_MODEL_ZOO,
    MODEL_LICENSE.SAM3.value: info_pb2.MODEL_LICENSE_SAM3,
}


def license_to_proto(lic: MODEL_LICENSE) -> int:
    if lic.value not in _LICENSE_TO_PROTO:
        raise ValueError(f"Unknown license: {lic.value}")
    return _LICENSE_TO_PROTO[lic.value]


_DOMAIN_TO_PROTO: dict[str, int] = {
    MODEL_DOMAIN.COMPUTER_VISION.value: info_pb2.MODEL_DOMAIN_COMPUTER_VISION,
    MODEL_DOMAIN.MULTIMODAL.value: info_pb2.MODEL_DOMAIN_MULTIMODAL,
    MODEL_DOMAIN.AUDIO.value: info_pb2.MODEL_DOMAIN_AUDIO,
    MODEL_DOMAIN.GENERATIVE_AI.value: info_pb2.MODEL_DOMAIN_GENERATIVE_AI,
}


def domain_to_proto(domain: MODEL_DOMAIN) -> int:
    if domain.value not in _DOMAIN_TO_PROTO:
        raise ValueError(f"Unknown domain: {domain.value}")
    return _DOMAIN_TO_PROTO[domain.value]


_TAG_TO_PROTO: dict[str, int] = {
    MODEL_TAG.BACKBONE.value: info_pb2.MODEL_TAG_BACKBONE,
    MODEL_TAG.REAL_TIME.value: info_pb2.MODEL_TAG_REAL_TIME,
    MODEL_TAG.FOUNDATION.value: info_pb2.MODEL_TAG_FOUNDATION,
    MODEL_TAG.LLM.value: info_pb2.MODEL_TAG_LLM,
    MODEL_TAG.GENERATIVE_AI.value: info_pb2.MODEL_TAG_GENERATIVE_AI,
    MODEL_TAG.BU_IOT.value: info_pb2.MODEL_TAG_BU_IOT,
    MODEL_TAG.BU_AUTO.value: info_pb2.MODEL_TAG_BU_AUTO,
    MODEL_TAG.BU_COMPUTE.value: info_pb2.MODEL_TAG_BU_COMPUTE,
    MODEL_TAG.MOE.value: info_pb2.MODEL_TAG_MOE,
    MODEL_TAG.VLM.value: info_pb2.MODEL_TAG_VLM,
    MODEL_TAG.ROBOTICS.value: info_pb2.MODEL_TAG_ROBOTICS,
}


def tag_to_proto(tag: MODEL_TAG) -> int:
    if tag.value not in _TAG_TO_PROTO:
        raise ValueError(f"Unknown tag: {tag.value}")
    return _TAG_TO_PROTO[tag.value]


_STATUS_TO_PROTO: dict[str, int] = {
    MODEL_STATUS.PUBLISHED.value: info_pb2.MODEL_STATUS_PUBLISHED,
    MODEL_STATUS.UNPUBLISHED.value: info_pb2.MODEL_STATUS_UNPUBLISHED,
    MODEL_STATUS.PENDING.value: info_pb2.MODEL_STATUS_PENDING,
}


def status_to_proto(status: MODEL_STATUS) -> int:
    if status.value not in _STATUS_TO_PROTO:
        raise ValueError(f"Unknown status: {status.value}")
    return _STATUS_TO_PROTO[status.value]


_USE_CASE_TO_PROTO: dict[str, int] = {
    MODEL_USE_CASE.IMAGE_CLASSIFICATION.value: info_pb2.MODEL_USE_CASE_IMAGE_CLASSIFICATION,
    MODEL_USE_CASE.IMAGE_EDITING.value: info_pb2.MODEL_USE_CASE_IMAGE_EDITING,
    MODEL_USE_CASE.IMAGE_GENERATION.value: info_pb2.MODEL_USE_CASE_IMAGE_GENERATION,
    MODEL_USE_CASE.SUPER_RESOLUTION.value: info_pb2.MODEL_USE_CASE_SUPER_RESOLUTION,
    MODEL_USE_CASE.SEMANTIC_SEGMENTATION.value: info_pb2.MODEL_USE_CASE_SEMANTIC_SEGMENTATION,
    MODEL_USE_CASE.DEPTH_ESTIMATION.value: info_pb2.MODEL_USE_CASE_DEPTH_ESTIMATION,
    MODEL_USE_CASE.GAZE_ESTIMATION.value: info_pb2.MODEL_USE_CASE_GAZE_ESTIMATION,
    MODEL_USE_CASE.IMAGE_TO_TEXT.value: info_pb2.MODEL_USE_CASE_IMAGE_TO_TEXT,
    MODEL_USE_CASE.OBJECT_DETECTION.value: info_pb2.MODEL_USE_CASE_OBJECT_DETECTION,
    MODEL_USE_CASE.POSE_ESTIMATION.value: info_pb2.MODEL_USE_CASE_POSE_ESTIMATION,
    MODEL_USE_CASE.DRIVER_ASSISTANCE.value: info_pb2.MODEL_USE_CASE_DRIVER_ASSISTANCE,
    MODEL_USE_CASE.ROBOTICS.value: info_pb2.MODEL_USE_CASE_ROBOTICS,
    MODEL_USE_CASE.SPEECH_RECOGNITION.value: info_pb2.MODEL_USE_CASE_SPEECH_RECOGNITION,
    MODEL_USE_CASE.AUDIO_ENHANCEMENT.value: info_pb2.MODEL_USE_CASE_AUDIO_ENHANCEMENT,
    MODEL_USE_CASE.AUDIO_CLASSIFICATION.value: info_pb2.MODEL_USE_CASE_AUDIO_CLASSIFICATION,
    MODEL_USE_CASE.AUDIO_GENERATION.value: info_pb2.MODEL_USE_CASE_AUDIO_GENERATION,
    MODEL_USE_CASE.VOICE_ACTIVITY_DETECTION.value: info_pb2.MODEL_USE_CASE_VOICE_ACTIVITY_DETECTION,
    MODEL_USE_CASE.SPEAKER_DIARIZATION.value: info_pb2.MODEL_USE_CASE_SPEAKER_DIARIZATION,
    MODEL_USE_CASE.VIDEO_CLASSIFICATION.value: info_pb2.MODEL_USE_CASE_VIDEO_CLASSIFICATION,
    MODEL_USE_CASE.VIDEO_GENERATION.value: info_pb2.MODEL_USE_CASE_VIDEO_GENERATION,
    MODEL_USE_CASE.VIDEO_OBJECT_TRACKING.value: info_pb2.MODEL_USE_CASE_VIDEO_OBJECT_TRACKING,
    MODEL_USE_CASE.TEXT_GENERATION.value: info_pb2.MODEL_USE_CASE_TEXT_GENERATION,
}


def use_case_to_proto(use_case: MODEL_USE_CASE) -> int:
    if use_case.value not in _USE_CASE_TO_PROTO:
        raise ValueError(f"Unknown use case: {use_case.value}")
    return _USE_CASE_TO_PROTO[use_case.value]


_CALL_TO_ACTION_TO_PROTO: dict[str, int] = {
    LLM_CALL_TO_ACTION.DOWNLOAD.value: info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_DOWNLOAD,
    LLM_CALL_TO_ACTION.VIEW_README.value: info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_VIEW_README,
    LLM_CALL_TO_ACTION.DOWNLOAD_AND_VIEW_README.value: info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_DOWNLOAD_AND_VIEW_README,
    LLM_CALL_TO_ACTION.CONTACT_FOR_PURCHASE.value: info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_FOR_PURCHASE,
    LLM_CALL_TO_ACTION.CONTACT_FOR_DOWNLOAD.value: info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_FOR_DOWNLOAD,
    LLM_CALL_TO_ACTION.COMING_SOON.value: info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_COMING_SOON,
    LLM_CALL_TO_ACTION.CONTACT_US.value: info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_US,
}


def call_to_action_to_proto(cta: LLM_CALL_TO_ACTION) -> int:
    if cta.value not in _CALL_TO_ACTION_TO_PROTO:
        raise ValueError(f"Unknown call to action: {cta.value}")
    return _CALL_TO_ACTION_TO_PROTO[cta.value]


_VOICE_AI_SDK_TO_PROTO: dict[str, int] = {
    VOICE_AI_SDK.ASR.value: info_pb2.VOICE_AI_SDK_ASR,
    VOICE_AI_SDK.TTS.value: info_pb2.VOICE_AI_SDK_TTS,
    VOICE_AI_SDK.TRANSLATION.value: info_pb2.VOICE_AI_SDK_TRANSLATION,
}


def voice_ai_sdk_to_proto(sdk: VOICE_AI_SDK | None) -> int:
    if sdk is None:
        return info_pb2.VOICE_AI_SDK_UNSPECIFIED
    if sdk.value not in _VOICE_AI_SDK_TO_PROTO:
        raise ValueError(f"Unknown voice AI SDK: {sdk.value}")
    return _VOICE_AI_SDK_TO_PROTO[sdk.value]


def _invert(mapping: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in mapping.items()}


_PROTO_TO_LICENSE = _invert(_LICENSE_TO_PROTO)
_PROTO_TO_DOMAIN = _invert(_DOMAIN_TO_PROTO)
_PROTO_TO_TAG = _invert(_TAG_TO_PROTO)
_PROTO_TO_STATUS = _invert(_STATUS_TO_PROTO)
_PROTO_TO_USE_CASE = _invert(_USE_CASE_TO_PROTO)
_PROTO_TO_CALL_TO_ACTION = _invert(_CALL_TO_ACTION_TO_PROTO)
_PROTO_TO_VOICE_AI_SDK = _invert(_VOICE_AI_SDK_TO_PROTO)


def license_from_proto(value: int) -> MODEL_LICENSE:
    return MODEL_LICENSE(_PROTO_TO_LICENSE[value])


def domain_from_proto(value: int) -> MODEL_DOMAIN:
    return MODEL_DOMAIN(_PROTO_TO_DOMAIN[value])


def tag_from_proto(value: int) -> MODEL_TAG:
    return MODEL_TAG(_PROTO_TO_TAG[value])


def status_from_proto(value: int) -> MODEL_STATUS:
    return MODEL_STATUS(_PROTO_TO_STATUS[value])


def use_case_from_proto(value: int) -> MODEL_USE_CASE:
    return MODEL_USE_CASE(_PROTO_TO_USE_CASE[value])


def call_to_action_from_proto(value: int) -> LLM_CALL_TO_ACTION:
    return LLM_CALL_TO_ACTION(_PROTO_TO_CALL_TO_ACTION[value])


def voice_ai_sdk_from_proto(value: int) -> VOICE_AI_SDK | None:
    if value == info_pb2.VOICE_AI_SDK_UNSPECIFIED:
        return None
    return VOICE_AI_SDK(_PROTO_TO_VOICE_AI_SDK[value])

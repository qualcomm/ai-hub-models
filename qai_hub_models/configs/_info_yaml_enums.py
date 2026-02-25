# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from enum import Enum, unique

HF_AVAILABLE_LICENSES = {
    "apache-2.0",
    "mit",
    "openrail",
    "bigscience-openrail-m",
    "creativeml-openrail-m",
    "bigscience-bloom-rail-1.0",
    "bigcode-openrail-m",
    "afl-3.0",
    "artistic-2.0",
    "bsl-1.0",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-3-clause-clear",
    "c-uda",
    "cc",
    "cc0-1.0",
    "cc-by-2.0",
    "cc-by-2.5",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
    "cc-by-nc-2.0",
    "cc-by-nc-3.0",
    "cc-by-nc-4.0",
    "cc-by-nd-4.0",
    "cc-by-nc-nd-3.0",
    "cc-by-nc-nd-4.0",
    "cc-by-nc-sa-2.0",
    "cc-by-nc-sa-3.0",
    "cc-by-nc-sa-4.0",
    "cdla-sharing-1.0",
    "cdla-permissive-1.0",
    "cdla-permissive-2.0",
    "wtfpl",
    "ecl-2.0",
    "epl-1.0",
    "epl-2.0",
    "etalab-2.0",
    "eupl-1.1",
    "agpl-3.0",
    "gfdl",
    "gpl",
    "gpl-2.0",
    "gpl-3.0",
    "lgpl",
    "lgpl-2.1",
    "lgpl-3.0",
    "isc",
    "intel-research",
    "lppl-1.3c",
    "ms-pl",
    "apple-ascl",
    "apple-amlr",
    "mpl-2.0",
    "odc-by",
    "odbl",
    "openrail++",
    "osl-3.0",
    "postgresql",
    "ofl-1.1",
    "ncsa",
    "unlicense",
    "zlib",
    "pddl",
    "lgpl-lr",
    "deepfloyd-if-license",
    "llama2",
    "llama3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "stability-ai",
    "gemma",
    "lfm1.0",
    "unknown",
    "other",
    "array",
}


class MODEL_LICENSE(Enum):
    # General
    UNLICENSED = "unlicensed"
    AI_HUB_MODELS_LICENSE = "ai-hub-models-license"
    COMMERCIAL = "commercial"

    # Non-restrictive
    APACHE_2_0 = "apache-2.0"
    MIT = "mit"
    BSD_3_CLAUSE = "bsd-3-clause"
    CC_BY_4_0 = "cc-by-4.0"

    # Copyleft (derivative works must have the same license)
    AGPL_3_0 = "agpl-3.0"
    GPL_3_0 = "gpl-3.0"
    CREATIVEML_OPENRAIL_M = "creativeml-openrail-m"

    # Commecial use prohibited
    OTHER_NON_COMMERCIAL = "other-non-commercial"
    CC_BY_NON_COMMERCIAL_4_0 = "cc-by-non-commercial-4.0"

    # Licenses for specific models
    AIMET_MODEL_ZOO = "aimet-model-zoo"
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    TAIDE = "taide"
    FALCON3 = "falcon3"
    GEMMA = "gemma"
    LFM1_0 = "lfm1.0"

    @property
    def is_copyleft(self) -> bool:
        # Return true if any derivative works must also use this license.
        return self in [
            MODEL_LICENSE.COMMERCIAL,
            MODEL_LICENSE.AGPL_3_0,
            MODEL_LICENSE.GPL_3_0,
            MODEL_LICENSE.CREATIVEML_OPENRAIL_M,
            MODEL_LICENSE.LLAMA2,
            MODEL_LICENSE.LLAMA3,
            MODEL_LICENSE.TAIDE,
            MODEL_LICENSE.FALCON3,
            MODEL_LICENSE.GEMMA,
        ]

    @property
    def is_non_commerical(self) -> bool:
        # If true if this license does not permit commercial use.
        return self in [
            MODEL_LICENSE.OTHER_NON_COMMERCIAL,
            MODEL_LICENSE.CC_BY_NON_COMMERCIAL_4_0,
        ]

    @property
    def huggingface_name(self) -> str:
        hf_str = None
        if self == MODEL_LICENSE.CC_BY_NON_COMMERCIAL_4_0:
            hf_str = "cc-by-nc-4.0"
        elif self == MODEL_LICENSE.UNLICENSED:
            hf_str = "unknown"
        else:
            hf_str = str(self)
        if hf_str in HF_AVAILABLE_LICENSES:
            return hf_str
        return "other"

    @property
    def url(self) -> str | None:
        # If this license has a known URL, return it.
        if self == MODEL_LICENSE.AI_HUB_MODELS_LICENSE:
            return "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf"
        if self == MODEL_LICENSE.LLAMA2:
            return "https://github.com/facebookresearch/llama/blob/main/LICENSE"
        if self == MODEL_LICENSE.TAIDE:
            return "https://en.taide.tw/download.html"
        if self == MODEL_LICENSE.FALCON3:
            return "https://falconllm.tii.ae/falcon-terms-and-conditions.html"
        if self == MODEL_LICENSE.GEMMA:
            return "https://ai.google.dev/gemma/terms"
        if self == MODEL_LICENSE.LFM1_0:
            return "https://www.liquid.ai/lfm-license"
        return None


@unique
class MODEL_DOMAIN(Enum):
    COMPUTER_VISION = "Computer Vision"
    MULTIMODAL = "Multimodal"
    AUDIO = "Audio"
    GENERATIVE_AI = "Generative AI"


@unique
class MODEL_TAG(Enum):
    BACKBONE = "backbone"
    REAL_TIME = "real-time"
    FOUNDATION = "foundation"
    LLM = "llm"
    GENERATIVE_AI = "generative-ai"
    BU_IOT = "bu-iot"
    BU_AUTO = "bu-auto"
    BU_COMPUTE = "bu-compute"
    MOE = "moe"


@unique
class MODEL_STATUS(Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    # proprietary models are released only internally
    PROPRIETARY = "proprietary"


@unique
class MODEL_USE_CASE(Enum):
    # Image: 100 - 199
    IMAGE_CLASSIFICATION = "Image Classification"
    IMAGE_EDITING = "Image Editing"
    IMAGE_GENERATION = "Image Generation"
    SUPER_RESOLUTION = "Super Resolution"
    SEMANTIC_SEGMENTATION = "Semantic Segmentation"
    DEPTH_ESTIMATION = "Depth Estimation"
    GAZE_ESTIMATION = "Gaze Estimation"

    # Ex: OCR, image caption
    IMAGE_TO_TEXT = "Image To Text"
    OBJECT_DETECTION = "Object Detection"
    POSE_ESTIMATION = "Pose Estimation"
    DRIVER_ASSISTANCE = "Driver Assistance"
    ROBOTICS = "Robotics"

    # Audio: 200 - 299
    SPEECH_RECOGNITION = "Speech Recognition"
    AUDIO_ENHANCEMENT = "Audio Enhancement"
    AUDIO_CLASSIFICATION = "Audio Classification"
    AUDIO_GENERATION = "Audio Generation"

    # Video: 300 - 399
    VIDEO_CLASSIFICATION = "Video Classification"
    VIDEO_GENERATION = "Video Generation"
    VIDEO_OBJECT_TRACKING = "Video Object Tracking"

    # LLM: 400 - 499
    TEXT_GENERATION = "Text Generation"

    def map_to_hf_pipeline_tag(self) -> str:
        """Map our usecase to pipeline-tag used by huggingface."""
        if self.name in {"IMAGE_EDITING", "SUPER_RESOLUTION"}:
            return "image-to-image"
        if self.name == "SEMANTIC_SEGMENTATION":
            return "image-segmentation"
        if self.name == "POSE_ESTIMATION":
            return "keypoint-detection"
        if self.name == "GAZE_ESTIMATION":
            return "gaze-estimation"
        if self.name == "AUDIO_ENHANCEMENT":
            return "audio-to-audio"
        if self.name == "VIDEO_GENERATION":
            return "image-to-video"
        if self.name == "IMAGE_GENERATION":
            return "unconditional-image-generation"
        if self.name == "SPEECH_RECOGNITION":
            return "automatic-speech-recognition"
        if self.name == "AUDIO_CLASSIFICATION":
            return "audio-classification"
        if self.name == "AUDIO_GENERATION":
            return "text-to-audio"
        if self.name == "DRIVER_ASSISTANCE":
            return "other"
        return self.name.replace("_", "-").lower()

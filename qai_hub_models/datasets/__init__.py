# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import importlib
import inspect
from typing import Any, cast

from qai_hub_models.utils.input_spec import InputSpec

from .common import BaseDataset, DatasetSplit

_ALL_DATASETS_IMPORT_ERRORS: dict[str, Exception] = {}
DATASET_NAME_MAP: dict[str, type[BaseDataset]] = {}


# We don't want to require a user to install requirements for all datasets just to
# import the datasets folder. Therefore we only include the datasets that can
# be imported.
def _try_import_dataset(module_name: str, cls: str, name: str | None = None) -> None:
    """
    Import the dataset and add it to the DATASET_NAME_MAP, or pass
    if dependencies for the dataset aren't installed.
    """
    if name is None:
        assert module_name[0] == "."
        name = module_name[1:]
    try:
        module = importlib.import_module(module_name, package="qai_hub_models.datasets")
    except NotImplementedError as e:
        if "AIMET-ONNX" in str(e):
            # stable diffusion dataset requires aimet-onnx
            _ALL_DATASETS_IMPORT_ERRORS[name] = e
            return
        raise
    except Exception as e:
        if (
            isinstance(e, ModuleNotFoundError)
            and str(e) == f"No module named 'qai_hub_models.datasets{module_name}"
        ):
            # this module legitimately does not exist
            raise

        # The module couldn't be loaded for some other reason.
        #
        # By default, the name of the dataset is the name of its module.
        # We add it to this import errors list to hopefully raise the error
        # at a later time (when the user requests this dataset).
        _ALL_DATASETS_IMPORT_ERRORS[name] = e
        return

    if x := getattr(module, cls, None):
        xds = cast(type[BaseDataset], x)
        assert name == xds.dataset_name(), (
            f"Name is not consistent with call to dataset_name(): {name} vs. {xds.dataset_name()}"
        )
        DATASET_NAME_MAP[name] = xds
    else:
        raise ValueError(
            f"Could not import {cls}. {cls} was not found in {module_name}"
        )


_try_import_dataset(".common_voice", "CommonVoiceDataset")
_try_import_dataset(".common_voice", "CommonVoiceText", name="common_voice_text")
_try_import_dataset(".bsd300", "BSD300Dataset")
_try_import_dataset(".cityscapes", "CityscapesDataset")
_try_import_dataset(".cityscapes_lowres", "CityscapesLowResDataset")
_try_import_dataset(".ade20k", "ADESegmentationDataset")
_try_import_dataset(".ade20k", "ADE10SegmentationDataset", name="ade20k_10")
_try_import_dataset(".coco", "CocoDataset")
_try_import_dataset(".coco_ppe", "CocoPPEDataset")
_try_import_dataset(".coco_seg", "CocoSegDataset")
_try_import_dataset(".ppe", "PPEDataset")
_try_import_dataset(".coco91class", "Coco91ClassDataset")
_try_import_dataset(".coco_face", "CocoFaceDataset")
_try_import_dataset(".human_faces", "HumanFacesDataset")
_try_import_dataset(".human_faces", "HumanFaces192Dataset", name="human_faces_192")
_try_import_dataset(".human_poses", "HumanPosesDataset")
_try_import_dataset(".coco_panoptic_seg", "CocoPanopticSegmentationDataset")
_try_import_dataset(".foot_track_dataset", "FootTrackDataset")
_try_import_dataset(".gear_guard_dataset", "GearGuardDataset")
_try_import_dataset(".gear_guard_v2_dataset", "GearGuardV2Dataset")
_try_import_dataset(".facemap_3dmm_dataset", "FaceMap3DMMDataset")
_try_import_dataset(".face_attrib_enhanced_dataset", "FaceAttribEnhancedDataset")
_try_import_dataset(".face_attrib_dataset", "FaceAttribDataset")
_try_import_dataset(".face_det_lite", "FaceDetLiteDataset")
_try_import_dataset(".kinetics400", "Kinetics400Dataset")
_try_import_dataset(".imagenet", "ImagenetDataset")
_try_import_dataset(".imagenet_colorization", "ImagenetColorizationDataset")
_try_import_dataset(".imagenet_256", "Imagenet_256Dataset")
_try_import_dataset(".imagenette", "ImagenetteDataset")
_try_import_dataset(".imagenette_colorization", "ImagenetteColorizationDataset")
_try_import_dataset(".imagenette_256", "Imagenette_256Dataset")
_try_import_dataset(".nuscenes", "NuscenesDataset")
_try_import_dataset(".wider_face", "WIDERFaceDataset")
_try_import_dataset(".nuscenes_bev", "NuscenesBevDataset")
_try_import_dataset(".nuscenes_bev", "NuscenesBevCVTDataset", name="nuscenes_bev_cvt")
_try_import_dataset(".nuscenes_bev", "NuscenesBevGKTDataset", name="nuscenes_bev_gkt")
_try_import_dataset(".nyuv2", "NyUv2Dataset")
_try_import_dataset(".cofw", "COFWDataset")
_try_import_dataset(".pascal_voc", "VOCSegmentationDataset")
_try_import_dataset(".mpii", "MPIIDataset")
_try_import_dataset(".cocobody", "CocoBodyDataset")
_try_import_dataset(".cocowholebody", "CocoWholeBodyDataset")
_try_import_dataset(".camvid", "CamVidSegmentationDataset")
_try_import_dataset(".valentini", "ValentiniDataset")
_try_import_dataset(".carvana", "CarvanaDataset")
_try_import_dataset(".camouflage_dataset", "CamouflageDataset")
_try_import_dataset(".eg1800", "eg1800SegmentationDataset")
_try_import_dataset(".kitti", "KittiDataset")
_try_import_dataset(".semantic_kitti", "SemanticKittiDataset")
_try_import_dataset(".sav", "SaVDataset")
_try_import_dataset(".icdar2015", "ICDAR2015Dataset")
_try_import_dataset(
    ".stable_diffusion_calib",
    "StableDiffusionCalibDatasetTextEncoder",
    name="stable_diffusion_calib_text_encoder",
)
_try_import_dataset(
    ".stable_diffusion_calib",
    "StableDiffusionCalibDatasetUnet",
    name="stable_diffusion_calib_unet",
)
_try_import_dataset(
    ".stable_diffusion_calib",
    "StableDiffusionCalibDatasetVae",
    name="stable_diffusion_calib_vae",
)
_try_import_dataset(
    ".stable_diffusion_calib",
    "StableDiffusionCalibDatasetControlNet",
    name="stable_diffusion_calib_controlnet",
)
_try_import_dataset(".celebahq", "CelebAHQDataset")
_try_import_dataset(".wikitext", "WikiText")
_try_import_dataset(".wikitext_masked", "WikiTextMasked")
_try_import_dataset(".wikitext_ja", "WikiText_Japanese")
_try_import_dataset(".tiny_mmlu", "TinyMMLU")
_try_import_dataset(
    ".tricky_llm_prompts", "TrickyLLMPromptsPhi35", name="tricky_llm_prompts_phi35"
)
_try_import_dataset(".mmlu", "MMLU")
_try_import_dataset(".mmmlu", "MMMLU")
_try_import_dataset(".mmmlu", "MMMLU_AR", name="mmmlu_ar")
_try_import_dataset(".mmmlu", "MMMLU_BN", name="mmmlu_bn")
_try_import_dataset(".mmmlu", "MMMLU_DE", name="mmmlu_de")
_try_import_dataset(".mmmlu", "MMMLU_ES", name="mmmlu_es")
_try_import_dataset(".mmmlu", "MMMLU_FR", name="mmmlu_fr")
_try_import_dataset(".mmmlu", "MMMLU_HI", name="mmmlu_hi")
_try_import_dataset(".mmmlu", "MMMLU_ID", name="mmmlu_id")
_try_import_dataset(".mmmlu", "MMMLU_IT", name="mmmlu_it")
_try_import_dataset(".mmmlu", "MMMLU_JA", name="mmmlu_ja")
_try_import_dataset(".mmmlu", "MMMLU_KO", name="mmmlu_ko")
_try_import_dataset(".mmmlu", "MMMLU_PT", name="mmmlu_pt")
_try_import_dataset(".mmmlu", "MMMLU_SW", name="mmmlu_sw")
_try_import_dataset(".mmmlu", "MMMLU_YO", name="mmmlu_yo")
_try_import_dataset(".mmmlu", "MMMLU_ZH", name="mmmlu_zh")
_try_import_dataset(".mpiigaze", "MPIIGazeDataset")
_try_import_dataset(".human_matting", "HumanMattingDataset")
_try_import_dataset(".libri_speech", "LibriSpeechDataset")
_try_import_dataset(
    ".amazon_counterfactual", "AmazonCounterfactualClassificationDataset"
)
_try_import_dataset(".audioset", "AudioSetDataset")
_try_import_dataset(".hagrid", "PalmDetectorDataset", name="hagrid_palmdetector")
_try_import_dataset(".hagrid", "HandLandmarkDataset", name="hagrid_handlandmark")
_try_import_dataset(
    ".hagrid", "GestureClassifierDataset", name="hagrid_gesturerecognizer"
)


def get_dataset_from_name(
    name: str,
    split: DatasetSplit,
    input_spec: InputSpec | None = None,
    **kwargs: Any,
) -> BaseDataset:
    dataset_cls = DATASET_NAME_MAP.get(name)
    if not dataset_cls:
        if name in _ALL_DATASETS_IMPORT_ERRORS:
            raise _ALL_DATASETS_IMPORT_ERRORS[name]
        raise ValueError(f"Unable to find dataset with name {name}")

    # Some datasets can be configured to different input specs
    if (
        input_spec is not None
        and "input_spec" in inspect.signature(dataset_cls.__init__).parameters
    ):
        kwargs["input_spec"] = input_spec
    return dataset_cls(split=split, **kwargs)

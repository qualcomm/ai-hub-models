# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from easyocr.craft import CRAFT as CRAFTDetector
from easyocr.DBNet.DBNet import DBNet as DBNetDetector
from easyocr.easyocr import Reader
from easyocr.model.model import Model as Recognizer
from easyocr.model.modules import BidirectionalLSTM
from easyocr.model.vgg_model import Model as VGGRecognizer
from typing_extensions import Self

from qai_hub_models.models.easyocr.dataset import (
    CocoTextCalibrationDataset,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import (
    BaseModel,
)
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)
from qai_hub_models.utils.rnn import UnrolledLSTM

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
LANG_LIST = ["en"]
MODEL_STORAGE_DIRECTORY = None
USER_NETWORK_DIRECTORY = None


class EasyOCRDetector(BaseModel):
    def __init__(self, detector_model: CRAFTDetector | DBNetDetector) -> None:
        super().__init__()
        self.model = detector_model

    @classmethod
    def from_pretrained(
        cls, lang_list: list = LANG_LIST, detect_network: str = "craft"
    ) -> Self:
        ocr_reader = Reader(
            lang_list,
            gpu=False,
            quantize=False,
            detect_network=detect_network,
            model_storage_directory=MODEL_STORAGE_DIRECTORY,
            user_network_directory=USER_NETWORK_DIRECTORY,
        )

        return cls(cast(CRAFTDetector | DBNetDetector, ocr_reader.detector))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run detector on `image`, and produce a ((text score map, link score map), features).

        Parameters
        ----------
        image
            Pixel values pre-processed for detector consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        detection_results : torch.Tensor
            Detection results containing text score map and link score map.
        """
        # Run network
        if isinstance(self.model, CRAFTDetector):
            image = normalize_image_torchvision(image)
            return self.model(image)[0]
        if isinstance(self.model, DBNetDetector):
            image *= 255
            assert self.model.model is not None
            return self.model.model(image, training=False)[0]
        raise NotImplementedError("Unknown detector model")

    def get_input_spec(
        self,
        batch_size: int = 1,
        detector_height: int = 608,
        detector_width: int = 800,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, detector_height, detector_width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "results": TensorSpec(),
        }


class EasyOCRRecognizer(BaseModel):
    """VGG based model is the default model of text recognition"""

    def __init__(self, recognizer_model: VGGRecognizer | Recognizer) -> None:
        super().__init__()
        self.model = recognizer_model

    @classmethod
    def from_pretrained(
        cls,
        lang_list: list = LANG_LIST,
        recog_network: str = "standard",
        unroll_lstm: bool = False,
    ) -> Self:
        ocr_reader = Reader(
            lang_list,
            gpu=False,
            quantize=False,
            recog_network=recog_network,
            model_storage_directory=MODEL_STORAGE_DIRECTORY,
            user_network_directory=USER_NETWORK_DIRECTORY,
        )

        recognizer = cast(VGGRecognizer | Recognizer, ocr_reader.recognizer)

        # Patch LSTM modules with UnrolledLSTM wrappers if needed
        # Unrolled LSTM is required to run on NPU via LiteRT (TFLite).
        if unroll_lstm:
            for rnn in recognizer.SequenceModeling:
                bilstm = cast(BidirectionalLSTM, rnn)
                bilstm.rnn = UnrolledLSTM(bilstm.rnn)  # type: ignore[assignment, unused-ignore]

        return cls(recognizer)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run recognizer on `image`, and produce a score matrix corresponding to character classes.

        Parameters
        ----------
        image
            Pixel values pre-processed for detector consumption.
            Range: float[0, 1]
            1-channel Color Space: grey

        Returns
        -------
        character_predictions : torch.Tensor
            segmented mask per class: Shape [batch, T, classes]
        """
        return self.model((image - 0.5) / 0.5, None)

    def get_input_spec(
        self,
        batch_size: int = 1,
        recognizer_height: int = 64,
        recognizer_width: int = 800,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 1, recognizer_height, recognizer_width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.GRAYSCALE,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "output_preds": TensorSpec(),
        }


class EasyOCR(WorkbenchModelCollection):
    def __init__(
        self,
        detector: EasyOCRDetector,
        recognizer: EasyOCRRecognizer,
        lang_list: list[str],
    ) -> None:
        super().__init__({"detector": detector, "recognizer": recognizer})
        self.lang_list = lang_list
        self.detector = detector
        self.recognizer = recognizer

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoTextCalibrationDataset

    def get_input_spec(
        self,
        batch_size: int = 1,
        detector_height: int = 608,
        detector_width: int = 800,
        recognizer_height: int = 64,
        recognizer_width: int = 800,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(
            batch_size=batch_size,
            detector_height=detector_height,
            detector_width=detector_width,
            recognizer_height=recognizer_height,
            recognizer_width=recognizer_width,
        )

    @classmethod
    def from_pretrained(
        cls,
        lang_list: list[str] = LANG_LIST,
        detect_network: str = "craft",
        recog_network: str = "standard",
        unroll_lstm: bool = False,
    ) -> Self:
        """
        Create an EasyOCR model.

        Parameters
        ----------
        lang_list
            Language List
        detect_network
            Detector network architecture
            Valid options: craft, dbnet18
        recog_network
            Recognizer network architecture
            Valid options: standard (determine based on language list), generation1, generation2
        unroll_lstm
            Whether to unroll LSTM layers in the recognizer.
            Unrolled LSTM is required to run on NPU via LiteRT (TFLite).

        Returns
        -------
        model_instance : Self
            The EasyOCR model instance.
        """
        detector = EasyOCRDetector.from_pretrained(lang_list, detect_network)
        recognizer = EasyOCRRecognizer.from_pretrained(
            lang_list, recog_network, unroll_lstm
        )
        return cls(detector, recognizer, lang_list)

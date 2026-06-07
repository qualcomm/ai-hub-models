# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import shutil
from pathlib import Path

import onnxruntime
import torch
from typing_extensions import Self

from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)
from qai_hub_models.utils.onnx.torch_wrapper import OnnxSessionTorchWrapper

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Apache-2.0 ONNX exports of the PaddleOCR v5 pipeline (no pickle, no opaque
# binary model formats). Mirrored on Hugging Face.
SOURCE_REPO = "https://huggingface.co/monkt/paddleocr-onnx"
DET_ONNX_URL = f"{SOURCE_REPO}/resolve/main/detection/v3/det.onnx"
ZH_REC_ONNX_URL = f"{SOURCE_REPO}/resolve/main/languages/chinese/rec.onnx"
KO_REC_ONNX_URL = f"{SOURCE_REPO}/resolve/main/languages/korean/rec.onnx"
EN_REC_ONNX_URL = f"{SOURCE_REPO}/resolve/main/languages/english/rec.onnx"
LATIN_REC_ONNX_URL = f"{SOURCE_REPO}/resolve/main/languages/latin/rec.onnx"
ESLAV_REC_ONNX_URL = f"{SOURCE_REPO}/resolve/main/languages/eslav/rec.onnx"
THAI_REC_ONNX_URL = f"{SOURCE_REPO}/resolve/main/languages/thai/rec.onnx"
GREEK_REC_ONNX_URL = f"{SOURCE_REPO}/resolve/main/languages/greek/rec.onnx"

# Each recognition language ships a character dictionary (one character per
# UTF-8 line) alongside its ONNX graph. CTC decoding maps logit indices to the
# characters in this dictionary.
ZH_DICT_URL = f"{SOURCE_REPO}/resolve/main/languages/chinese/dict.txt"
KO_DICT_URL = f"{SOURCE_REPO}/resolve/main/languages/korean/dict.txt"
EN_DICT_URL = f"{SOURCE_REPO}/resolve/main/languages/english/dict.txt"
LATIN_DICT_URL = f"{SOURCE_REPO}/resolve/main/languages/latin/dict.txt"
ESLAV_DICT_URL = f"{SOURCE_REPO}/resolve/main/languages/eslav/dict.txt"
THAI_DICT_URL = f"{SOURCE_REPO}/resolve/main/languages/thai/dict.txt"
GREEK_DICT_URL = f"{SOURCE_REPO}/resolve/main/languages/greek/dict.txt"

_REC_DICT_URLS = {
    "zh": ZH_DICT_URL,
    "ko": KO_DICT_URL,
    "en": EN_DICT_URL,
    "latin": LATIN_DICT_URL,
    "eslav": ESLAV_DICT_URL,
    "thai": THAI_DICT_URL,
    "greek": GREEK_DICT_URL,
}

# Recognition models read fixed-height text crops with a width chosen by the
# caller. We pin a representative export width that matches our benchmark crops.
DETECTION_RESOLUTION = (640, 640)
RECOGNITION_HEIGHT = 48
RECOGNITION_WIDTH = 320


def _fetch_onnx(url: str, filename: str) -> str:
    """Fetch the source ONNX file to the local asset cache and return its path."""
    asset = CachedWebModelAsset(url, MODEL_ID, MODEL_ASSET_VERSION, filename)
    return str(asset.fetch())


class _OnnxSourceModel(BaseModel):
    """Base class for PP-OCRv5 components backed by a source ONNX graph.

    The model source is provided directly as ONNX. ``forward`` runs the graph
    with ONNX Runtime so the component is usable for local (host) inference,
    while ``serialize`` hands the original ONNX file to the export pipeline so
    AI Hub Workbench compiles the real graph instead of tracing a Python module
    that wraps an inference session.
    """

    def __init__(self, onnx_path: str) -> None:
        super().__init__()
        self.onnx_path = onnx_path
        self._wrapper = OnnxSessionTorchWrapper(
            onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._wrapper(image)

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Provide the source ONNX directly to the export/compile pipeline."""
        dst = Path(output_dir) / f"{self.name}.onnx"
        if dst.resolve() != Path(self.onnx_path).resolve():
            shutil.copyfile(self.onnx_path, dst)
        return dst


class PPOCRv5Detector(_OnnxSourceModel):
    """PP-OCRv5 text detection (PP-HGNetV2 backbone + DB++ head)."""

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(_fetch_onnx(DET_ONNX_URL, "det.onnx"))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run text detection on `image` and produce a per-pixel text probability map.

        Parameters
        ----------
        image
            Pixel values pre-processed for detector consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        prob_map : torch.Tensor
            Per-pixel text probability map. Shape [batch, 1, H, W].
        """
        return super().forward(image)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = DETECTION_RESOLUTION[0],
        width: int = DETECTION_RESOLUTION[1],
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    def get_output_names(self) -> list[str]:
        return ["prob_map"]

    def get_channel_last_inputs(self) -> list[str]:
        return ["image"]


class PPOCRv5Recognizer(_OnnxSourceModel):
    """PP-OCRv5 text recognition (SVTR backbone + CTC head).

    A single recognition architecture serves multiple languages; the only
    difference between the language variants is the character dictionary that
    maps CTC logit indices to characters.
    """

    def __init__(self, onnx_path: str, lang: str | None = None) -> None:
        super().__init__(onnx_path)
        self.lang = lang
        self._charlist: list[str] | None = None

    @classmethod
    def from_pretrained(cls, lang: str = "zh") -> Self:
        urls = {
            "zh": ZH_REC_ONNX_URL,
            "ko": KO_REC_ONNX_URL,
            "en": EN_REC_ONNX_URL,
            "latin": LATIN_REC_ONNX_URL,
            "eslav": ESLAV_REC_ONNX_URL,
            "thai": THAI_REC_ONNX_URL,
            "greek": GREEK_REC_ONNX_URL,
        }
        if lang not in urls:
            raise ValueError(f"Unsupported recognition language: {lang!r}")
        return cls(_fetch_onnx(urls[lang], f"{lang}_rec.onnx"), lang=lang)

    def get_charlist(self) -> list[str]:
        """Return the CTC character list for this recognizer.

        PaddleOCR CTC decoding uses the layout
        ``["<blank>"] + dictionary + [" "]`` so that index 0 is the CTC blank
        and the final index is the space character. The resulting list length
        equals the recognizer's logit vocabulary (e.g. 18385 for Chinese).

        The dictionary characters are read from the ``character`` metadata
        baked into the ONNX graph when present; otherwise they are loaded from
        the per-language ``dict.txt`` shipped alongside the ONNX weights.
        """
        if self._charlist is not None:
            return self._charlist

        import onnx

        chars: list[str] = []
        model = onnx.load(self.onnx_path, load_external_data=False)
        for prop in model.metadata_props:
            if prop.key == "character" and prop.value:
                chars = prop.value.splitlines()
                break

        if not chars:
            if self.lang is None or self.lang not in _REC_DICT_URLS:
                raise ValueError(
                    "Cannot resolve a character dictionary: the ONNX graph has "
                    "no 'character' metadata and the recognizer language is "
                    f"unknown ({self.lang!r})."
                )
            dict_path = _fetch_onnx(_REC_DICT_URLS[self.lang], f"{self.lang}_dict.txt")
            with open(dict_path, encoding="utf-8") as handle:
                chars = handle.read().splitlines()

        self._charlist = ["<blank>", *chars, " "]
        return self._charlist

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run text recognition on a text-line crop and produce CTC logits.

        Parameters
        ----------
        image
            Pixel values pre-processed for recognizer consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        logits : torch.Tensor
            CTC logits over the character dictionary. Shape [batch, T, vocab].
        """
        return super().forward(image)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = RECOGNITION_HEIGHT,
        width: int = RECOGNITION_WIDTH,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    def get_output_names(self) -> list[str]:
        return ["logits"]

    def get_channel_last_inputs(self) -> list[str]:
        return ["image"]


@CollectionModel.add_component(PPOCRv5Detector, "det")
@CollectionModel.add_component(PPOCRv5Recognizer, "zh_rec")
@CollectionModel.add_component(PPOCRv5Recognizer, "ko_rec")
@CollectionModel.add_component(PPOCRv5Recognizer, "en_rec")
@CollectionModel.add_component(PPOCRv5Recognizer, "latin_rec")
@CollectionModel.add_component(PPOCRv5Recognizer, "eslav_rec")
@CollectionModel.add_component(PPOCRv5Recognizer, "thai_rec")
@CollectionModel.add_component(PPOCRv5Recognizer, "greek_rec")
class PPOCRv5(PretrainedCollectionModel):
    """PP-OCRv5 detection + multilingual recognition collection.

    A shared text detector plus the seven PP-OCRv5 recognition languages,
    each an independently-compiled component:
        * det       : text detection (PP-HGNetV2 + DB++)
        * zh_rec    : Simplified Chinese + English recognition (SVTR/CTC)
        * ko_rec    : Korean + English recognition (SVTR/CTC)
        * en_rec    : English recognition (SVTR/CTC)
        * latin_rec : Latin-script recognition (SVTR/CTC)
        * eslav_rec : Cyrillic / East-Slavic recognition (SVTR/CTC)
        * thai_rec  : Thai recognition (SVTR/CTC)
        * greek_rec : Greek recognition (SVTR/CTC)
    """

    def __init__(
        self,
        det: PPOCRv5Detector,
        zh_rec: PPOCRv5Recognizer,
        ko_rec: PPOCRv5Recognizer,
        en_rec: PPOCRv5Recognizer,
        latin_rec: PPOCRv5Recognizer,
        eslav_rec: PPOCRv5Recognizer,
        thai_rec: PPOCRv5Recognizer,
        greek_rec: PPOCRv5Recognizer,
    ) -> None:
        super().__init__(
            det,
            zh_rec,
            ko_rec,
            en_rec,
            latin_rec,
            eslav_rec,
            thai_rec,
            greek_rec,
        )
        self.det = det
        self.zh_rec = zh_rec
        self.ko_rec = ko_rec
        self.en_rec = en_rec
        self.latin_rec = latin_rec
        self.eslav_rec = eslav_rec
        self.thai_rec = thai_rec
        self.greek_rec = greek_rec

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(
            PPOCRv5Detector.from_pretrained(),
            PPOCRv5Recognizer.from_pretrained("zh"),
            PPOCRv5Recognizer.from_pretrained("ko"),
            PPOCRv5Recognizer.from_pretrained("en"),
            PPOCRv5Recognizer.from_pretrained("latin"),
            PPOCRv5Recognizer.from_pretrained("eslav"),
            PPOCRv5Recognizer.from_pretrained("thai"),
            PPOCRv5Recognizer.from_pretrained("greek"),
        )

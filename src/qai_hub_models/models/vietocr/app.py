# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.vietocr.model import IMAGE_HEIGHT, IMAGE_WIDTH


class VietOCRApp:
    """
    Light-weight "app code" for running the VietOCR recognition CNN backbone.

    For a given text-line image, the app will:
        * convert to RGB and resize to the backbone input shape
        * normalize pixel values to [0, 1]
        * run the backbone to produce a per-column feature sequence
    """

    def __init__(
        self,
        backbone: Callable[[torch.Tensor], torch.Tensor],
        image_shape: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
    ) -> None:
        self.backbone = backbone
        self.image_shape = image_shape

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Resize and normalize a PIL image into a backbone input tensor."""
        height, width = self.image_shape
        resized = image.convert("RGB").resize((width, height))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        chw = arr.transpose(2, 0, 1)[np.newaxis]
        return torch.from_numpy(np.ascontiguousarray(chw))

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.predict_features_from_image(*args, **kwargs)

    def predict_features_from_image(
        self, pixel_values_or_image: np.ndarray | torch.Tensor | Image.Image
    ) -> np.ndarray:
        """
        Produce the per-column feature sequence for a text-line image.

        Parameters
        ----------
        pixel_values_or_image
            Input PIL image (before pre-processing), or a tensor / array
            already shaped [batch, 3, H, W] with values in [0, 1].

        Returns
        -------
        features : np.ndarray
            Per-column feature sequence. Shape [W', batch, 256].
        """
        pixel_values = self._to_input_tensor(pixel_values_or_image)
        return np.asarray(self.backbone(pixel_values))

    def _to_input_tensor(
        self, pixel_values_or_image: np.ndarray | torch.Tensor | Image.Image
    ) -> torch.Tensor:
        if isinstance(pixel_values_or_image, Image.Image):
            return self.preprocess_image(pixel_values_or_image)
        if isinstance(pixel_values_or_image, np.ndarray):
            return torch.from_numpy(pixel_values_or_image)
        return pixel_values_or_image

    # ------------------------------------------------------------------
    # End-to-end recognition.
    #
    # The AI Hub model exported here is the *CNN backbone* (the first of the
    # two VietOCR recognizer components). To turn the backbone's per-column
    # feature sequence into recognized text, those features are fed through the
    # *second* component -- the Transformer seq2seq encoder/decoder plus the
    # output vocabulary -- taken from the installed ``vietocr`` package. The
    # combination demonstrates full end-to-end Vietnamese text recognition
    # while keeping the exported on-device artifact limited to the backbone.
    # ------------------------------------------------------------------
    def recognize_text(
        self,
        image: Image.Image,
        transformer: torch.nn.Module,
        vocab: Any,
        max_seq_length: int = 128,
        sos_token: int = 1,
        eos_token: int = 2,
    ) -> str:
        """
        Recognize the Vietnamese text in a text-line image, end to end.

        The backbone (this AI Hub model) produces the feature sequence; the
        ``transformer`` (VietOCR's Transformer seq2seq head) greedily decodes
        that sequence into token ids, which ``vocab`` maps back to characters.

        Parameters
        ----------
        image
            A text-line PIL image.
        transformer
            VietOCR's ``LanguageTransformer`` head (provides
            ``forward_encoder`` / ``forward_decoder``).
        vocab
            VietOCR's ``Vocab`` (provides ``decode``).
        max_seq_length, sos_token, eos_token
            Greedy-decode controls matching VietOCR's defaults.

        Returns
        -------
        text : str
            The recognized Vietnamese text.
        """
        pixel_values = self._to_input_tensor(image)

        with torch.no_grad():
            # Component 1: exported CNN backbone -> per-column features.
            src = self.backbone(pixel_values)
            if isinstance(src, np.ndarray):
                src = torch.from_numpy(src)

            # Component 2: Transformer seq2seq decode of the features.
            memory = transformer.forward_encoder(src)

            translated = [sos_token]
            for _ in range(max_seq_length):
                tgt = torch.LongTensor([translated]).transpose(0, 1)
                output, memory = transformer.forward_decoder(tgt, memory)
                next_token = int(output[:, -1, :].argmax(dim=-1).item())
                translated.append(next_token)
                if next_token == eos_token:
                    break

        return vocab.decode(translated)

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import jiwer
import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    WORD_ERROR_RATE,
    MetricMetadata,
)


class WEREvaluator(BaseEvaluator):
    """WER evaluator for Parakeet TDT on LibriSpeech test-clean dataset."""

    def __init__(self) -> None:
        self.transform = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.SubstituteRegexes({"-": " "}),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ]
        )
        self.reset()

    def add_batch(
        self,
        output: list[str],
        target: list[str],
    ) -> None:
        """
        Parameters
        ----------
        output
            List of predicted transcription strings from app.transcribe().
        target
            List of ground truth transcription strings (lowercased).
        """
        for pred, ref in zip(output, target, strict=True):
            # Decode ASCII tensor → string
            if isinstance(pred, torch.Tensor):
                pred = "".join(chr(int(c)) for c in pred if int(c) != 0)
            if isinstance(ref, torch.Tensor):
                ref = "".join(chr(int(c)) for c in ref if int(c) != 0)
            # Normalize using jiwer standard ASR transforms
            pred = self.transform(pred)
            ref = self.transform(ref)
            self.predictions.append(pred)
            self.references.append(ref)

    def reset(self) -> None:
        self.predictions: list[str] = []
        self.references: list[str] = []

    def get_accuracy_score(self) -> float:
        """Return WER as percentage."""
        return jiwer.wer(self.references, self.predictions) * 100

    def formatted_accuracy(self) -> str:
        wer_score = self.get_accuracy_score()
        return f"Word Error Rate: {wer_score:.3f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return WORD_ERROR_RATE

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Zero-shot image classification evaluator (full 1000-way argmax).

Used with models such as SigLIP2 that produce sigmoid-loss logits between
an image and all N class prompts simultaneously.  A prediction is correct
when ``argmax(logits)`` matches the ground-truth class index.
"""

from __future__ import annotations

import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    ACCURACY_TOP1,
    MetricMetadata,
)


class ZeroShotClassificationEvaluator(BaseEvaluator):
    """Evaluator for zero-shot image classification via full N-way argmax.

    Expects the model to return logits of shape ``[B, N]`` where N is the
    number of candidate class prompts (e.g. 1000 for ImageNet).  A sample is
    counted as correct when ``argmax(logits, dim=-1)`` equals the ground-truth
    class index.
    """

    def __init__(self) -> None:
        self.reset()

    def add_batch(
        self,
        output: torch.Tensor,
        gt: torch.Tensor | int,
    ) -> None:
        """Accumulate predictions for a batch.

        Parameters
        ----------
        output
            Logits of shape ``[B, N]`` where B is the batch size and N is the
            number of candidate classes.  A 1-D tensor is treated as a single
            sample and unsqueezed to ``[1, N]``.
        gt
            Ground-truth class indices of shape ``[B]`` or a scalar integer.
            Each value must be in ``[0, N)``.
        """
        if isinstance(output, (list, tuple)):
            output = output[0]
        if output.dim() == 1:
            output = output.unsqueeze(0)
        batch_size = output.shape[0]
        self.total_samples += batch_size

        gt_tensor = torch.tensor([gt]) if isinstance(gt, int) else gt.view(-1).long()
        gt_tensor = gt_tensor.to(output.device)
        assert gt_tensor.shape[0] == batch_size, (
            f"gt size {gt_tensor.shape[0]} != output batch size {batch_size}"
        )
        predicted = torch.argmax(output, dim=-1).long()
        self.correct_count += int((predicted == gt_tensor).sum().item())

    def reset(self) -> None:
        self.correct_count: int = 0
        self.total_samples: int = 0

    def get_accuracy_score(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return (self.correct_count / self.total_samples) * 100.0

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.1f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return ACCURACY_TOP1


class CollectionZeroShotEvaluator(ZeroShotClassificationEvaluator):
    """
    Wraps ZeroShotClassificationEvaluator for SigLIP2 and OpenAI-CLIP.

    The on-device image encoder returns raw L2-normalised image embeddings
    ``[B, D]``.  Text embeddings are pre-computed once before the eval loop
    and passed at construction time.

    Parameters
    ----------
    text_embeds
        Pre-computed L2-normalised text embeddings ``[N, D]``.
    logit_scale
        Scalar multiplier (``exp(log_scale)`` from the pretrained model).
    logit_bias
        Scalar bias added after the dot product.
    """

    def __init__(
        self,
        text_embeds: torch.Tensor,
        logit_scale: float,
        logit_bias: float,
    ) -> None:
        super().__init__()
        self.text_embeds = text_embeds
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias

    def add_batch(
        self,
        output: torch.Tensor | tuple[torch.Tensor, ...],
        gt: torch.Tensor | int,
    ) -> None:
        """Accumulate predictions for a batch.

        Converts raw image embeddings to logits before delegating to the base
        class evaluator.

        Parameters
        ----------
        output
            L2-normalised image embeddings of shape ``[B, D]``, or a tuple/list
            whose first element is such a tensor.  ``D`` must match the embedding
            dimension of ``self.text_embeds``.
        gt
            Ground-truth class indices of shape ``[B]`` or a scalar integer.
            Each value must be in ``[0, N)`` where N is the number of text classes.
        """
        image_embeds = output[0] if isinstance(output, (list, tuple)) else output
        # On-device models return fp16; cast to match text_embeds dtype before matmul.
        logits = (
            self.logit_scale
            * image_embeds.to(self.text_embeds.dtype)
            @ self.text_embeds.t()
            + self.logit_bias
        )
        super().add_batch(logits, gt)

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Generator
from typing import cast

import torch

from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    AsyncOnDeviceResult,
    OnDeviceModel,
    dataset_entries_from_batch,
)


class ZeroShotCollectionApp(
    CollectionAppEvaluateProtocol, CollectionAppQuantizeProtocol
):
    """Base class for two-encoder zero-shot image-text apps (CLIP, SigLIP2, etc.).

    Provides shared _run_encoder and run_model_for_eval logic.
    Subclasses must set self.image_encoder and implement get_evaluator /
    get_calibration_data.
    """

    image_encoder: ExecutableModelProtocol[torch.Tensor] | AsyncOnDeviceModel

    @staticmethod
    def _run_encoder(
        encoder: ExecutableModelProtocol[torch.Tensor] | AsyncOnDeviceModel,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Run encoder on N inputs as a single Hub job (batch-1 compiled model).

        Compiled on-device models have batch size 1.  Passing a raw [N, ...]
        tensor would submit it as a single entry of shape [N, ...] which Hub
        rejects.  Instead we split into N batch-1 entries and submit one job.
        Local models are passed the tensor directly.
        """
        if isinstance(encoder, (OnDeviceModel, AsyncOnDeviceModel)):
            entries, _ = dataset_entries_from_batch(
                (inputs, torch.empty(0)),
                list(encoder.input_names),
                encoder.channel_last_input,
            )
            result = encoder(entries)
            waited = (
                result.wait() if isinstance(result, AsyncOnDeviceResult) else result
            )
            return waited[0] if isinstance(waited, tuple) else waited
        return encoder(inputs)

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        """Run image encoder only; evaluator uses pre-computed text embeddings.

        Local path: model_input is a tuple; image encoder runs locally.
        On-device path: model_input is a generator of input tensors; image
        encoder job is submitted async and yielded for harness tracking.
        """
        if isinstance(model_input, tuple):
            image_embeds = self._run_encoder(self.image_encoder, model_input[0])
            assert isinstance(image_embeds, torch.Tensor)
            embeds: tuple[torch.Tensor, ...] = (image_embeds,)
            yield embeds
            return embeds
        else:
            image_tensor = cast(torch.Tensor, next(model_input))
            async_result = cast(AsyncOnDeviceResult, self.image_encoder(image_tensor))
            yield async_result
            return async_result

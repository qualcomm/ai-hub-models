# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage
from PIL.Image import Image
from qai_hub.client import DatasetEntries
from safetensors import safe_open
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.imagenet.imagenet_zeroshot import imagenet_prompts
from qai_hub_models.datasets.open_images.open_images_zeroshot import (
    OpenImagesV7ZeroshotDataset,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.siglip2.evaluator import SigLIP2Evaluator
from qai_hub_models.models.siglip2.model import DEFAULT_WEIGHTS, TEXT_SEQ_LEN
from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    AsyncOnDeviceResult,
    OnDeviceModel,
    dataset_entries_from_batch,
)
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


class SigLIP2App(CollectionAppEvaluateProtocol, CollectionAppQuantizeProtocol):
    """
    End-to-end app for SigLIP2 zero-shot image-text similarity.

    Orchestrates two independently compiled components:
      * image_encoder — encodes images to L2-normalised embeddings
      * text_encoder  — encodes tokenised prompts to L2-normalised embeddings

    Logits are computed in the app as:
        logit_scale * image_embeds @ text_embeds.T + logit_bias

    For a given set of images and text prompts the app will:
      * Pre-process images to float32 [0, 1] tensors
      * Tokenize text prompts with the SigLIP2 tokenizer
      * Run both encoders
      * Return logits_per_image [num_images, num_texts]
    """

    def __init__(
        self,
        image_encoder: ExecutableModelProtocol[torch.Tensor] | AsyncOnDeviceModel,
        text_encoder: ExecutableModelProtocol[torch.Tensor] | AsyncOnDeviceModel,
        logit_scale: float,
        logit_bias: float,
        weights: str = DEFAULT_WEIGHTS,
    ) -> None:
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias
        self.tokenizer = AutoTokenizer.from_pretrained(weights, use_fast=True)
        self._cached_text_embeds: torch.Tensor | None = None

    def predict(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.predict_similarity(*args, **kwargs)

    def predict_similarity(
        self,
        images_or_paths: Sequence[Image | str | Path],
        texts: Sequence[str],
    ) -> torch.Tensor:
        """
        Compute image-text similarity logits.

        Parameters
        ----------
        images_or_paths
            PIL Images or file paths / URLs.
        texts
            Text prompts to compare against.

        Returns
        -------
        logits_per_image : torch.Tensor
            Shape [num_images, num_texts].
        """
        pil_images: list[Image] = []
        for item in images_or_paths:
            if isinstance(item, (str, Path)):
                item = load_image(item)
            pil_images.append(item)

        token_out = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=TEXT_SEQ_LEN,
        )
        input_ids = token_out["input_ids"].to(torch.int32)

        # Encode all text prompts once.
        text_embeds = self._run_encoder(self.text_encoder, input_ids)
        assert isinstance(text_embeds, torch.Tensor)

        # Encode each image and compute logits
        image_tensors = torch.cat(
            [
                preprocess_PIL_image(img.resize((224, 224), PILImage.BILINEAR))
                for img in pil_images
            ]
        )  # [N, 3, 224, 224]
        image_embeds = self._run_encoder(self.image_encoder, image_tensors)
        assert isinstance(image_embeds, torch.Tensor)
        return (
            self.logit_scale * image_embeds @ text_embeds.t() + self.logit_bias
        )  # [N, num_texts]

    @classmethod
    def from_components(
        cls,
        models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel],
    ) -> SigLIP2App:
        """
        Construct a SigLIP2App from a list of model components.

        Parameters
        ----------
        models
            [image_encoder, text_encoder] — either local ExecutableModelProtocol
            instances or compiled AsyncOnDeviceModel instances.

        Returns
        -------
        SigLIP2App
            App instance ready for inference and evaluation.
        """
        weights_path = hf_hub_download(DEFAULT_WEIGHTS, filename="model.safetensors")
        with safe_open(weights_path, framework="pt") as f:
            logit_scale = math.exp(f.get_tensor("logit_scale").item())
            logit_bias = f.get_tensor("logit_bias").item()
        return cls(
            image_encoder=models[0],
            text_encoder=models[1],
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )

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

    # ------------------------------------------------------------------
    # CollectionAppEvaluateProtocol
    # ------------------------------------------------------------------
    def get_evaluator(self) -> SigLIP2Evaluator:
        """Return a SigLIP2Evaluator backed by this app's text encoder.

        On the first call the text encoder (local or on-device) is run once
        over all 1 000 ImageNet zero-shot prompts and the resulting embeddings
        are cached on ``self._cached_text_embeds``.  Subsequent calls return a
        new evaluator that reuses the cached embeddings, so the text encoder is
        never run more than once per app instance regardless of how many times
        the harness calls this method.
        """
        if self._cached_text_embeds is None:
            token_out = self.tokenizer(
                imagenet_prompts(),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=TEXT_SEQ_LEN,
            )
            input_ids = token_out["input_ids"].to(torch.int32)
            with torch.no_grad():
                text_embeds = self._run_encoder(self.text_encoder, input_ids)
            self._cached_text_embeds = F.normalize(text_embeds, dim=-1)
        return SigLIP2Evaluator(
            self._cached_text_embeds, self.logit_scale, self.logit_bias
        )

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        """Run image encoder and return raw L2-normalised image embeddings.

        Both local and on-device paths return image embeddings ``[B, D]``.
        The evaluator (SigLIP2Evaluator) converts them to logits using
        pre-computed text embeddings.

        Local eval: model_input is a tuple of tensors; image encoder runs
        locally and embeddings are returned directly.

        On-device eval: model_input is a generator of per-component input
        tensors.  We call the image encoder (an AsyncOnDeviceModel) with the
        image tensor, yield the AsyncOnDeviceResult so the harness can track
        the job, then return it as the StopIteration value.  The harness calls
        .wait() and passes the raw embeddings to SigLIP2Evaluator.add_batch.
        """
        if isinstance(model_input, tuple):
            # Local torch eval path: return image embeddings directly.
            image_embeds = self._run_encoder(self.image_encoder, model_input[0])
            assert isinstance(image_embeds, torch.Tensor)
            embeds: tuple[torch.Tensor, ...] = (image_embeds,)
            yield embeds
            return embeds
        else:
            # On-device eval path: submit image encoder job, yield async result.
            image_tensor = cast(torch.Tensor, next(model_input))
            async_result = cast(AsyncOnDeviceResult, self.image_encoder(image_tensor))
            yield async_result
            return async_result

    # ------------------------------------------------------------------
    # CollectionAppQuantizeProtocol
    # ------------------------------------------------------------------

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        """
        Return calibration data for the requested component.

        Uses ``open_images_v7_zeroshot`` (TRAIN). The image encoder uses the
        image-only mode (``(image, label)``); the text encoder uses joint mode
        (``tokenizer_id`` set, ``(image, input_ids, label)``) and only
        ``input_ids`` is fed to it.
        """
        component = collection_model.components[component_name]
        input_spec = (input_specs or {}).get(
            component_name
        ) or component.get_input_spec()
        batch_size = get_batch_size(input_spec) or 1

        is_text = component_name == "text_encoder"
        dataset = OpenImagesV7ZeroshotDataset(
            split=DatasetSplit.TRAIN,
            tokenizer_id=DEFAULT_WEIGHTS if is_text else None,
            text_seq_len=TEXT_SEQ_LEN,
        )
        num_samples = num_samples or dataset.default_num_calibration_samples()
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Loading {num_samples} calibration samples.")
        dataloader = DataLoader(
            sample_dataset(dataset, num_samples), batch_size=batch_size
        )
        inputs: list[list[torch.Tensor | np.ndarray]] = [[]]
        for batch in dataloader:
            inputs[0].append(batch[1] if is_text else batch[0])
        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))

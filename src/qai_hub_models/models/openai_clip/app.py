# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import clip
import numpy as np
import torch
from PIL.Image import Image
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.models.openai_clip.dataset import (
    CLIP_IMAGE_PREPROCESSOR,
    ClipOpenImagesZeroshotDataset,
)
from qai_hub_models.models.openai_clip.model import (
    CLIP_CALIBRATION_TEMPLATES,
    CLIP_ZEROSHOT_TEMPLATES,
    build_ensembled_text_embeds,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.zeroshot_classifier.app import (
    ZeroShotCollectionApp,
)
from qai_hub_models.models.templates.zeroshot_classifier.evaluator import (
    CollectionZeroShotEvaluator,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    OnDeviceModel,
)
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.labels import get_class_names
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


class ClipApp(ZeroShotCollectionApp):
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Clip.

    The app uses 2 models:
        * ClipImageEncoder
        * ClipTextEncoder

    For a given image input, the app will:
        * pre-process the image
        * pre-process the text
        * Run image encoder and text encoder
        * Return similarity scores
    """

    def __init__(
        self,
        image_encoder: ExecutableModelProtocol[torch.Tensor] | AsyncOnDeviceModel,
        text_encoder: ExecutableModelProtocol[torch.Tensor] | AsyncOnDeviceModel,
        logit_scale: float = 100.0,
    ) -> None:
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_tokenizer = clip.tokenize
        self.image_preprocessor = CLIP_IMAGE_PREPROCESSOR
        # logit_scale: Temperature scale applied to cosine similarities.
        # Released CLIP weights saturate the training cap at 100.0.
        self._logit_scale = logit_scale
        self._cached_text_embeds: torch.Tensor | None = None

    def predict(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.predict_similarity(*args, **kwargs)

    def predict_similarity(
        self,
        images_or_paths: Sequence[Image | str | Path],
        texts: Sequence[str],
    ) -> torch.Tensor:
        """
        Compute image-text similarity scores.

        Parameters
        ----------
        images_or_paths
            PIL Images or file paths / URLs.
        texts
            Text prompt candidates.

        Returns
        -------
        similarities : torch.Tensor
            Shape [num_images, num_texts]. logit_scale * cosine similarity.
        """
        pil_images: list[Image] = []
        for item in images_or_paths:
            if isinstance(item, (str, Path)):
                item = load_image(item)
            pil_images.append(item)

        image_tensors = torch.cat(
            [self.image_preprocessor(img).unsqueeze(0) for img in pil_images]
        )  # [N, 3, H, W]

        text_tokens = torch.cat([self.text_tokenizer(t) for t in texts])  # [M, 77]

        image_embeds = self._run_encoder(self.image_encoder, image_tensors)
        text_embeds = self._run_encoder(self.text_encoder, text_tokens)

        assert isinstance(image_embeds, torch.Tensor)
        assert isinstance(text_embeds, torch.Tensor)
        return self._logit_scale * image_embeds @ text_embeds.t()

    @classmethod
    def from_components(
        cls,
        models: Sequence[ExecutableModelProtocol] | Sequence[AsyncOnDeviceModel],
    ) -> ClipApp:
        """
        Construct a ClipApp from a list of model components.

        Parameters
        ----------
        models
            [image_encoder, text_encoder] — either local ExecutableModelProtocol
            instances or compiled AsyncOnDeviceModel instances.

        Returns
        -------
        ClipApp
            App instance ready for inference and evaluation.
        """
        return cls(
            image_encoder=models[0],
            text_encoder=models[1],
        )

    # ------------------------------------------------------------------
    # CollectionAppEvaluateProtocol
    # ------------------------------------------------------------------

    def get_evaluator(self) -> CollectionZeroShotEvaluator:
        """Return a CollectionZeroShotEvaluator backed by this app's text encoder.

        Per-class text embeddings are ensembled across CLIP_ZEROSHOT_TEMPLATES,
        for both local (torch) and on-device evaluation.

        Embeddings are pre-computed once and cached on ``self._cached_text_embeds``.
        """
        if self._cached_text_embeds is None:
            with torch.no_grad():
                text_embeds = build_ensembled_text_embeds(
                    get_class_names("clip_imagenet"),
                    CLIP_ZEROSHOT_TEMPLATES,
                    lambda tokens: self._run_encoder(self.text_encoder, tokens),
                )
                # Cast to match image encoder dtype. Local models have .parameters();
                # on-device models default to fp32.
                if isinstance(self.image_encoder, (OnDeviceModel, AsyncOnDeviceModel)):
                    image_dtype = torch.float32
                else:
                    image_dtype = next(self.image_encoder.parameters()).dtype  # type: ignore[attr-defined]
                self._cached_text_embeds = text_embeds.to(image_dtype)
        return CollectionZeroShotEvaluator(
            self._cached_text_embeds, self._logit_scale, 0.0
        )

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
        Produces calibration data for a quantize job for the given component.

        Image encoder: images sampled from ClipOpenImagesZeroshotDataset (TRAIN).
        Text encoder: tokenized text from 1000 Open Images classes x 7 CLIP templates
            (7000 prompts total).
        """
        model = collection_model.components[component_name]
        input_spec = (
            input_specs[component_name] if input_specs else model.get_input_spec()
        )
        batch_size = get_batch_size(input_spec) or 1

        if component_name == "image_encoder":
            dataset = ClipOpenImagesZeroshotDataset(
                split=DatasetSplit.TRAIN, input_spec=input_spec
            )
            n = num_samples or dataset.default_num_calibration_samples()
            n = (n // batch_size) * batch_size
            print(f"Loading {n} calibration samples for {component_name}.")
            dataloader = DataLoader(sample_dataset(dataset, n), batch_size=batch_size)
            inputs: list[list[torch.Tensor | np.ndarray]] = [[]]
            for image, _ in dataloader:
                inputs[0].append(image)
            return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))

        if component_name == "text_encoder":
            labels = get_class_names("clip_imagenet")
            all_prompts = [
                tmpl.format(name)
                for name in labels
                for tmpl in CLIP_CALIBRATION_TEMPLATES
            ]
            n = num_samples or 128
            n = min((n // batch_size) * batch_size, len(all_prompts))
            print(f"Loading {n} calibration samples for {component_name}.")
            g = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(all_prompts), generator=g)[:n].tolist()
            sampled = [all_prompts[i] for i in indices]
            text_inputs: list[list[torch.Tensor | np.ndarray]] = [[]]
            for start in range(0, n, batch_size):
                batch = sampled[start : start + batch_size]
                tokens = torch.cat([clip.tokenize(p) for p in batch])
                text_inputs[0].append(tokens)
            return make_hub_dataset_entries(tuple(text_inputs), list(input_spec.keys()))

        raise NotImplementedError(
            f"Calibration data not implemented for component: {component_name!r}"
        )

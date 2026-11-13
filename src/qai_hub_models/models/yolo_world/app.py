# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import cast

import numpy as np
import torch
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.yolo.app import YoloWorldPromptDetectionApp
from qai_hub_models.models.yolo_world.model import YoloWorldTextEncoder
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.labels import get_class_names
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


class YoloWorldDetectionApp(
    YoloWorldPromptDetectionApp,
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
):
    """
    Light-weight "app code" for end-to-end inference with YoloWorld.

    Works in two modes:
    - Single-model mode (demo/test): pass ``detector`` and ``text_encoder`` directly.
      ``prompt_text`` is supplied per inference call to ``predict_boxes_from_image``.
    - Collection mode (export/evaluate): build via ``from_components([detector, text_encoder])``.
      ``prompt_text`` is set once on the app and used by ``run_model_for_eval``.

    The two-stage pipeline:
      1. text_encoder.encode_classes(prompt_text) → txt_feats
      2. detector(image, txt_feats) → boxes, scores, classes
    """

    def __init__(
        self,
        detector: ExecutableModelProtocol | AsyncOnDeviceModel,
        text_encoder: ExecutableModelProtocol | AsyncOnDeviceModel,
    ) -> None:
        """
        Initialize YoloWorldDetectionApp.

        Parameters
        ----------
        detector
            YoloWorldDetector callable: ``(image, txt_feats) -> (boxes, scores, classes)``.
        text_encoder
            YoloWorldTextEncoder callable: ``encode_classes(class_names) -> txt_feats``.
        """
        super().__init__(model=detector)  # type: ignore[arg-type]
        self.detector = detector
        self.text_encoder = text_encoder

    # Set by from_components() for on-device evaluation.
    _collection_detector: ExecutableModelProtocol | AsyncOnDeviceModel | None = None
    _collection_text_encoder: ExecutableModelProtocol | AsyncOnDeviceModel | None = None
    # Set by from_components() to fix prompt_text for eval runs.
    prompt_text: list[str] = []

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        h, w = pixel_values.shape[-2:]
        if h != 640 or w != 640:
            raise ValueError(
                f"YoloWorld only supports 640x640 input. Received: {h}x{w}"
            )

    @classmethod
    def from_components(
        cls,
        models: Sequence[ExecutableModelProtocol] | Sequence[AsyncOnDeviceModel],
        prompt_text: list[str] | str | None = None,
    ) -> YoloWorldDetectionApp:
        """
        Create an app instance from a list of model components.

        Parameters
        ----------
        models
            [detector, text_encoder] — components in collection order (detector first).
        prompt_text
            Class names for eval runs. Defaults to COCO 80 classes if not provided.

        Returns
        -------
        YoloWorldDetectionApp
            WorkbenchModel Collection application supporting Evaluation and Quantization
        """
        if prompt_text is None:
            prompt_text = get_class_names("coco")

        detector_component = models[0]
        text_encoder_component = models[1]

        app = cls(
            detector=detector_component,
            text_encoder=text_encoder_component,
        )
        app._collection_text_encoder = text_encoder_component
        app._collection_detector = detector_component
        app.prompt_text = (
            prompt_text
            if isinstance(prompt_text, list)
            else [t.strip() for t in prompt_text.split(",") if t.strip()]
        )
        return app

    def _get_coco_tokens(self) -> torch.Tensor:
        """Return pre-tokenized COCO-80 token IDs."""
        return YoloWorldTextEncoder.tokenize_classes(self.prompt_text)

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        model = collection_model.components[component_name]
        input_spec = (
            input_specs[component_name] if input_specs else model.get_input_spec()
        )
        batch_size = get_batch_size(input_spec) or 1

        if component_name == "text_encoder":
            # Text encoder takes fixed COCO token IDs — produce N identical copies.
            tokens = YoloWorldTextEncoder.tokenize_classes(get_class_names("coco"))
            num_samples = num_samples or 128
            token_data: list[torch.Tensor | np.ndarray] = [tokens] * num_samples
            return make_hub_dataset_entries((token_data,), ["tokens"])

        # For the detector component: provide (image, txt_feats) pairs.
        text_encoder_component = cast(
            YoloWorldTextEncoder, collection_model.components["text_encoder"]
        )
        detector_component = collection_model.components["detector"]
        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        detector_image_spec = (input_specs or {}).get(
            "detector", detector_component.get_input_spec()
        )

        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=detector_image_spec,
        )
        num_samples = num_samples or dataset.default_num_calibration_samples()
        num_samples = (num_samples // batch_size) * batch_size
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)

        txt_feats = text_encoder_component.encode_classes(get_class_names("coco"))

        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]
        for sample_input, _ in dataloader:
            if isinstance(sample_input, (tuple, list)):
                image = sample_input[0]
            else:
                image = sample_input
            batch_txt = txt_feats.expand(image.shape[0], -1, -1)
            inputs[0].append(image)
            inputs[1].append(batch_txt)

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        # model_input is (image,) from the COCO eval dataset.
        # In the tuple path image is a Tensor; in the generator path next() returns
        # a pre-split tuple[Tensor, ...] produced by evaluate.py's split loop.
        if isinstance(model_input, tuple):
            image: torch.Tensor | tuple[torch.Tensor, ...] = model_input[0]
        else:
            raw = next(model_input)
            image = cast(tuple[torch.Tensor, ...], raw)

        text_encoder = self._collection_text_encoder or self.text_encoder
        detector = self._collection_detector or self.detector

        assert text_encoder is not None
        text_output: AsyncOnDeviceResult | tuple[torch.Tensor, ...]
        if isinstance(text_encoder, AsyncOnDeviceModel):
            # On-device path: text encoder expects raw token IDs.
            tokens = YoloWorldTextEncoder.tokenize_classes(self.prompt_text)
            text_output = text_encoder(tokens)
        else:
            # Local path: use encode_classes which handles tokenization internally.
            text_output = (
                cast(YoloWorldTextEncoder, text_encoder).encode_classes(
                    self.prompt_text
                ),
            )
        yield text_output

        if isinstance(text_output, AsyncOnDeviceResult):
            txt_feats_raw = text_output.wait()
            txt_feats = (
                txt_feats_raw[0] if isinstance(txt_feats_raw, tuple) else txt_feats_raw
            )
            # wait() on a single-output model returns a plain Tensor; ensure (1, N, 512).
            if isinstance(txt_feats, torch.Tensor) and txt_feats.ndim == 2:
                txt_feats = txt_feats.unsqueeze(0)
        else:
            # Local path: text_output is a 1-tuple wrapping the tensor.
            txt_feats = text_output[0]

        # Run detector with (image, txt_feats) passthrough.
        if isinstance(image, torch.Tensor):
            txt_feats = txt_feats.expand(image.shape[0], -1, -1).contiguous()
            det_output = detector(image, txt_feats)
        else:
            # On-device path: image is a tuple of per-job tensor chunks.
            image_chunks = cast(tuple[torch.Tensor, ...], image)
            txt_feats_chunks = tuple(
                txt_feats.expand(chunk.shape[0], -1, -1).contiguous()
                for chunk in image_chunks
            )
            det_output = detector(image_chunks, txt_feats_chunks)  # type: ignore[arg-type]
        yield det_output
        return det_output

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from PIL.Image import Image
from qai_hub.public_rest_api import DatasetEntries
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, nms
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models.grounding_dino_t.dataset import (
    IMG_HEIGHT,
    IMG_WIDTH,
    MAX_TEXT_LEN,
    tokenize_caption,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.util.utils import (
    get_phrases_from_posmap,
)
from qai_hub_models.models.grounding_dino_t.model import (
    NUM_FEATURE_LEVELS,
    GroundingDINOTCollection,
    load_tokenizer,
)
from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.draw import draw_box_from_xyxy
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.image_processing import (
    letterbox_resize_pad_pil,
    preprocess_PIL_image,
)
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

_LOGGER = logging.getLogger(__name__)

# box_threshold/text_threshold defaults match upstream:
# https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/util/inference.py#L156-L157
DEFAULT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25
DEFAULT_NMS_THRESHOLD = 0.5


def _resize_image(image: Image) -> Image:
    """Letterbox-resize to (IMG_WIDTH, IMG_HEIGHT) via the shared
    letterbox_resize_pad_pil helper, so the app's inference path sees the
    same image distribution as calibration/eval (GroundingDINOCOCODataset)
    and the evaluator's letterbox unscaling of predicted boxes stays valid.
    """
    return letterbox_resize_pad_pil(image, IMG_HEIGHT, IMG_WIDTH)


def _postprocess_predictions(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenized: dict,
    tokenizer: PreTrainedTokenizerBase,
    resized_image: Image,
    box_threshold: float,
    text_threshold: float,
    nms_threshold: float,
) -> tuple[list[npt.NDArray[np.uint8]], torch.Tensor, list[str], torch.Tensor]:
    """Decode raw model outputs into boxes/phrases/scores and draw them on `resized_image`.

    Shared post-processing for collection predictions. The three compiled
    components produce `pred_logits`/`pred_boxes`; decode/threshold/NMS/draw
    happens here.

    Parameters
    ----------
    pred_logits
        Per-query, per-token contrastive logits, shape [1, num_queries, MAX_TEXT_LEN].
    pred_boxes
        Per-query predicted boxes (cx, cy, w, h), shape [1, num_queries, 4].
    attention_mask
        Padding mask used to zero out logits at padded token positions, shape [1, MAX_TEXT_LEN].
    tokenized
        Raw (unpadded) tokenizer output, used to decode phrases.
    tokenizer
        Tokenizer used to decode token ids back to strings.
    resized_image
        The letterboxed (IMG_WIDTH, IMG_HEIGHT) RGB PIL image to draw detections onto.
    box_threshold
        Minimum per-box confidence score to keep a detection.
    text_threshold
        Minimum per-token score used when decoding the predicted phrase label.
    nms_threshold
        IoU threshold above which overlapping detections are suppressed, keeping
        only the higher-scoring box. Applied class-agnostically across all
        surviving detections, since GroundingDINO's classes are open-vocabulary
        decoded text rather than a fixed label set.

    Returns
    -------
    annotated_frames : list[npt.NDArray[np.uint8]]
        Single-element list containing `resized_image` (the letterboxed
        (IMG_WIDTH, IMG_HEIGHT) canvas, HxWx3 uint8, black-padded for
        non-4:3 inputs) with bounding boxes and labels drawn on it.
    scores : torch.Tensor
        1-D float tensor of per-detection confidence scores, shape [N].
    phrases : list[str]
        Decoded text label for each detection, length N.
    boxes_xyxy : torch.Tensor
        Bounding boxes in (x1, y1, x2, y2) pixel coordinates within the
        letterboxed (IMG_WIDTH, IMG_HEIGHT) canvas -- *not* unscaled back to
        the original input image's resolution (unlike evaluator.py, which
        does unletterbox for COCO scoring). Shape [N, 4].
    """
    annotated_frame: npt.NDArray[np.uint8] = np.array(resized_image, dtype=np.uint8)

    # Only batch size 1 is supported: resized_image is a single PIL image, so
    # there's nowhere to attach per-batch-item results beyond index 0.
    assert pred_logits.shape[0] == 1, (
        f"_postprocess_predictions only supports batch size 1, got {pred_logits.shape[0]}."
    )

    prediction_logits = pred_logits.cpu().sigmoid()[0]  # [900, 256]
    prediction_boxes = pred_boxes.cpu()[0]

    # Zero out logits at padding positions
    real_len = int(attention_mask[0].sum().item())
    prediction_logits[:, real_len:] = 0.0

    keep = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[keep]
    boxes = prediction_boxes[keep]

    phrases = [
        get_phrases_from_posmap(
            logit > text_threshold, tokenized, cast(Any, tokenizer)
        ).replace(".", "")
        for logit in logits
    ]

    boxes_scaled = boxes * torch.tensor(
        [IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT], dtype=torch.float32
    )
    boxes_xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy")
    scores = logits.max(dim=1)[0]

    keep_nms = nms(boxes_xyxy, scores, nms_threshold)
    boxes_xyxy = boxes_xyxy[keep_nms]
    scores = scores[keep_nms]
    phrases = [phrases[i] for i in keep_nms.tolist()]

    for box, phrase, score in zip(boxes_xyxy, phrases, scores, strict=True):
        draw_box_from_xyxy(
            annotated_frame,
            box[0:2].int(),
            box[2:4].int(),
            color=(0, 255, 0),
            size=2,
            text=f"{phrase} {score:.2f}",
        )

    return [annotated_frame], scores, phrases, boxes_xyxy


class GroundingDINOTCollectionApp(
    CollectionAppEvaluateProtocol, CollectionAppQuantizeProtocol
):
    """Pre/post-processing wrapper around the split GroundingDINO-T
    (vision + bert + fusion_decoder components) for open-vocabulary object
    detection.

    Calls the three components separately: `predict` runs the vision and BERT
    text encoder inline, and `predict_from_outputs` runs the fusion/decoder half
    on their outputs and decodes the final detections.
    """

    def __init__(
        self,
        vision: ExecutableModelProtocol | AsyncOnDeviceModel,
        bert: ExecutableModelProtocol | AsyncOnDeviceModel,
        fusion_decoder: ExecutableModelProtocol | AsyncOnDeviceModel,
        text: str,
    ) -> None:
        """
        Parameters
        ----------
        vision
            Vision component (local model or on-device wrapper).
        bert
            BERT text-encoder component (local model or on-device wrapper).
        fusion_decoder
            Fusion/decoder component (local model or on-device wrapper).
        text
            Grounding text prompt, tokenized once here and reused across
            every `predict`/`predict_from_outputs` call.
        """
        self.vision = vision
        self.bert = bert
        self.fusion_decoder = fusion_decoder
        self.tokenizer = load_tokenizer()
        (
            self._input_ids,
            self._attention_mask,
            self._token_type_ids,
            self._position_ids,
            self._text_self_attention_masks,
            _,  # cate_to_token_mask — not needed in app, used by dataset/evaluator
        ) = tokenize_caption(self.tokenizer, text, MAX_TEXT_LEN)
        self._tokenized = {"input_ids": self._input_ids[0].tolist()}

    def predict_from_outputs(
        self,
        encoded_text: torch.Tensor,
        *vision_outputs: torch.Tensor,
        resized_image: Image,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    ) -> tuple[list[npt.NDArray[np.uint8]], torch.Tensor, list[str], torch.Tensor]:
        """Run the fusion/decoder component on `encoded_text` + `vision_outputs`, and
        decode detections, drawing them on the already-resized `resized_image`.
        """
        with torch.no_grad():
            fusion_decoder_out = self.fusion_decoder(
                encoded_text,
                self._attention_mask,
                self._position_ids,
                self._text_self_attention_masks,
                *vision_outputs,
            )
            if isinstance(fusion_decoder_out, AsyncOnDeviceResult):
                fusion_decoder_out = fusion_decoder_out.wait()
            pred_logits, pred_boxes = cast(
                "tuple[torch.Tensor, torch.Tensor]", fusion_decoder_out
            )

        return _postprocess_predictions(
            pred_logits,
            pred_boxes,
            self._attention_mask,
            self._tokenized,
            self.tokenizer,
            resized_image,
            box_threshold,
            text_threshold,
            nms_threshold,
        )

    def predict(
        self,
        image: Image,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    ) -> tuple[list[npt.NDArray[np.uint8]], torch.Tensor, list[str], torch.Tensor]:
        """
        Run the full vision + bert + fusion_decoder pipeline on `image` against the
        app's configured text prompt, and decode detections.

        Parameters
        ----------
        image
            Input image to run detection on.
        box_threshold
            Minimum per-box confidence score to keep a detection.
        text_threshold
            Minimum per-token score used when decoding the predicted phrase label.
        nms_threshold
            IoU threshold above which overlapping detections are suppressed.

        Returns
        -------
        annotated_frames : list[npt.NDArray[np.uint8]]
            List containing one HWC uint8 image with detections drawn on it.
            This is the letterboxed (IMG_WIDTH, IMG_HEIGHT) canvas `image`
            was resized/padded into, not `image` itself -- for non-4:3
            inputs it includes black letterbox bars.
        scores : torch.Tensor
            Per-detection confidence scores.
        phrases : list[str]
            Per-detection decoded phrase labels.
        boxes_xyxy : torch.Tensor
            Per-detection bounding boxes in (x1, y1, x2, y2) format, in the
            letterboxed (IMG_WIDTH, IMG_HEIGHT) canvas's pixel coordinates --
            *not* unscaled back to `image`'s original resolution. Use
            `qai_hub_models.utils.image_processing.letterbox_geometry` to map
            these back to `image`'s coordinate space if needed (see
            evaluator.py's `add_batch` for the unscale math).
        """
        resized_image = _resize_image(image)
        image_tensor = preprocess_PIL_image(resized_image)
        with torch.no_grad():
            vision_out = self.vision(image_tensor)
            if isinstance(vision_out, AsyncOnDeviceResult):
                vision_out = vision_out.wait()
            vision_outputs = cast("tuple[torch.Tensor, ...]", vision_out)

            bert_out = self.bert(
                self._input_ids,
                self._token_type_ids,
                self._position_ids,
                self._text_self_attention_masks,
            )
            if isinstance(bert_out, AsyncOnDeviceResult):
                bert_out = bert_out.wait()
            encoded_text = cast("torch.Tensor", bert_out)
        return self.predict_from_outputs(
            encoded_text,
            *vision_outputs,
            resized_image=resized_image,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_threshold=nms_threshold,
        )

    @classmethod
    def from_components(
        cls,
        models: Sequence[ExecutableModelProtocol] | Sequence[AsyncOnDeviceModel],
        prompt_text: list[str] | str | None = None,
    ) -> GroundingDINOTCollectionApp:
        """
        Create a collection app instance from the collection's 3 model components.

        Parameters
        ----------
        models
            The collection's components in `(vision, bert, fusion_decoder)` order. Can
            be either local models or AsyncOnDeviceModel wrappers.
        prompt_text
            Grounding text prompt(s). A list is joined into a single caption
            with " . " separators. If None, the app is created with an empty prompt.

        Returns
        -------
        GroundingDINOTCollectionApp
            A collection app instance initialized with the provided models and prompt.
        """
        vision_component = models[0]
        bert_component = models[1]
        fusion_decoder_component = models[2]
        text = (
            prompt_text
            if isinstance(prompt_text, str)
            else (" . ".join(prompt_text) + " ." if prompt_text else "")
        )
        if not text:
            _LOGGER.warning(
                "No prompt_text provided; the app will be created with an "
                "empty grounding text prompt and will not detect anything."
            )
        return cls(
            vision=vision_component,
            bert=bert_component,
            fusion_decoder=fusion_decoder_component,
            text=text,
        )

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: GroundingDINOTCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        """
        Produces a numpy dataset to be used for calibration data of a quantize job.

        Parameters
        ----------
        collection_model
            The parent collection model.
        component_name
            The name of the component being calibrated. One of
            "vision", "bert", or "fusion_decoder".
        input_specs
            Per-component input specs. If None, uses each component's defaults.
        num_samples
            Number of data samples to use. If not specified, uses
            default specified on dataset.

        Returns
        -------
        DatasetEntries
            Dataset compatible with the format expected by AI Hub Workbench.
        """
        vision_input_spec = (
            input_specs["vision"]
            if input_specs
            else collection_model.components["vision"].get_input_spec()
        )
        batch_size = get_batch_size(vision_input_spec) or 1
        if input_specs:
            for other_name in ("bert", "fusion_decoder"):
                other_spec = input_specs.get(other_name)
                other_batch_size = get_batch_size(other_spec) if other_spec else None
                assert other_batch_size is None or other_batch_size == batch_size, (
                    f"input_specs batch size mismatch: vision={batch_size}, "
                    f"{other_name}={other_batch_size}. All components share one "
                    "calibration dataloader, so their batch sizes must agree."
                )

        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=vision_input_spec,
        )
        num_samples = num_samples or dataset.default_num_calibration_samples()
        if num_samples < batch_size:
            raise ValueError(
                f"num_samples ({num_samples}) must be >= batch_size ({batch_size})."
            )
        # Round down to a multiple of batch_size
        num_samples = (num_samples // batch_size) * batch_size
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)

        if component_name == "vision":
            images: list[torch.Tensor | np.ndarray] = []
            for sample_input, _ in dataloader:
                images.append(sample_input[0])
            return make_hub_dataset_entries((images,), ["image"])

        if component_name == "bert":
            bert_input_names = [
                "input_ids",
                "token_type_ids",
                "position_ids",
                "text_self_attention_masks",
            ]
            per_text_tensors: list[list[torch.Tensor | np.ndarray]] = [
                [] for _ in bert_input_names
            ]
            for sample_input, _ in dataloader:
                (
                    _,
                    input_ids,
                    _,
                    token_type_ids,
                    position_ids,
                    text_self_attention_masks,
                ) = sample_input
                for j, t in enumerate(
                    [input_ids, token_type_ids, position_ids, text_self_attention_masks]
                ):
                    per_text_tensors[j].append(t)
            return make_hub_dataset_entries(tuple(per_text_tensors), bert_input_names)

        if component_name == "fusion_decoder":
            vision = collection_model.components["vision"]
            bert = collection_model.components["bert"]
            device = next(vision.parameters()).device

            input_names = [
                "encoded_text",
                "attention_mask",
                "position_ids",
                "text_self_attention_masks",
            ] + [f"src_{i}" for i in range(NUM_FEATURE_LEVELS)]
            per_input_tensors: list[list[torch.Tensor | np.ndarray]] = [
                [] for _ in input_names
            ]
            with torch.no_grad():
                for sample_input, _ in dataloader:
                    (
                        image,
                        input_ids,
                        attention_mask,
                        token_type_ids,
                        position_ids,
                        text_self_attention_masks,
                    ) = sample_input
                    vision_out = vision(image.to(device))
                    encoded_text = bert(
                        input_ids.to(device),
                        token_type_ids.to(device),
                        position_ids.to(device),
                        text_self_attention_masks.to(device),
                    )
                    sample_tensors = [
                        encoded_text,
                        attention_mask,
                        position_ids,
                        text_self_attention_masks,
                        *vision_out,
                    ]
                    for j, t in enumerate(sample_tensors):
                        per_input_tensors[j].append(t)

            return make_hub_dataset_entries(tuple(per_input_tensors), input_names)

        raise NotImplementedError(f"Unknown component: {component_name!r}")

    def run_model_for_eval(
        self,
        model_input: Any,
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        """Run the vision, bert, and fusion_decoder components in sequence for evaluation.

        model_input is either a tuple of (image, *text_inputs) torch.Tensor for
        local execution, or a generator yielding AsyncOnDeviceResult per
        component for on-device execution (image first, then the 5 text
        tensors). model_batch_size is used to re-split on-device outputs for
        the next component in the pipeline.

        Yields output from each of the vision, bert, and fusion_decoder components;
        returns the final fusion_decoder output (pred_logits, pred_boxes).
        """
        if isinstance(model_input, tuple):
            image: Any = model_input[0]
            text_inputs: tuple[Any, ...] = model_input[1:]
        else:
            # model_input yields one original tensor's batch-split chunks per
            # next() call (image, then the 5 text tensors, in dataset order).
            image = next(model_input)
            text_inputs = tuple(next(model_input) for _ in range(5))

        vision_out: tuple[torch.Tensor, ...] | AsyncOnDeviceResult = self.vision(image)
        yield vision_out

        (
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            text_self_attention_masks,
        ) = text_inputs
        bert_out: torch.Tensor | AsyncOnDeviceResult = self.bert(
            input_ids, token_type_ids, position_ids, text_self_attention_masks
        )
        yield bert_out if isinstance(bert_out, AsyncOnDeviceResult) else (bert_out,)

        if isinstance(vision_out, AsyncOnDeviceResult):
            # wait() concatenates chunks back into one full-batch tensor per
            # output; re-split to match the still-chunked text tensors below.
            vision_tensors = tuple(
                t.split(model_batch_size, dim=0) for t in vision_out.wait()
            )
        else:
            vision_tensors = vision_out

        if isinstance(bert_out, AsyncOnDeviceResult):
            encoded_text = cast(torch.Tensor, bert_out.wait()).split(
                model_batch_size, dim=0
            )
        else:
            encoded_text = bert_out

        det_out: tuple[torch.Tensor, ...] | AsyncOnDeviceResult = self.fusion_decoder(
            encoded_text,
            attention_mask,
            position_ids,
            text_self_attention_masks,
            *vision_tensors,
        )
        yield det_out
        return det_out

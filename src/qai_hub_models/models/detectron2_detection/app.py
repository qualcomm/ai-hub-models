# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence

import numpy as np
import torch
from PIL import Image
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.models.templates.corevision.draw import (
    create_color_map,
    draw_box_from_xyxy,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.models.templates.corevision.labels import get_class_names
from qai_hub_models.models.templates.proposal_based_detection.app import (
    ProposalBasedDetectionApp,
)
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


class Detectron2DetectionApp(
    ProposalBasedDetectionApp,
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
):
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with Detectron2 Detection.

    For a given image input, the app will:
        * Preprocess the image (normalize, resize, etc).
        * Run Detectron2 Detection Inference.
        * Convert the raw output into box coordinates and corresponding label and confidence.
        * Return numpy image with boxes.
    """

    def __init__(
        self,
        proposal_generator: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        roi_head: Callable[
            [torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        proposal_iou_threshold: float = 0.7,
        boxes_iou_threshold: float = 0.5,
        boxes_score_threshold: float = 0.8,
        max_det_pre_nms: int = 6000,
        max_det_post_nms: int = 200,
        max_vis_boxes: int = 100,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        Initialize Detectron2DetectionApp.

        Parameters
        ----------
        proposal_generator
            Callable that generates proposals from image tensor.
        roi_head
            Callable that processes ROI features and proposals.
        proposal_iou_threshold
            IOU threshold for proposal filtering.
        boxes_iou_threshold
            IOU threshold for NMS on boxes.
        boxes_score_threshold
            Score threshold for filtering boxes.
        max_det_pre_nms
            Maximum detections before NMS.
        max_det_post_nms
            Maximum detections after NMS.
        max_vis_boxes
            Maximum boxes to visualize.
        input_spec
            Model input spec. If None, defaults to 800x800.
        """
        super().__init__(
            proposal_iou_threshold,
            boxes_iou_threshold,
            boxes_score_threshold,
            max_det_pre_nms,
            max_det_post_nms,
            input_spec=input_spec,
        )
        self.proposal_generator = proposal_generator
        self.roi_head = roi_head
        self.max_vis_boxes = max_vis_boxes
        self.num_classes = 80

    def predict(
        self,
        images: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        raw_output: bool = False,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | list[Image.Image]
    ):
        """
        From the provided image, detect the objects.

        Parameters
        ----------
        images
            PIL image, numpy array (N H W C x uint8) or (H W C x uint8) with RGB channel layout,
            or pyTorch tensor (N C H W x fp32, value range is [0, 1]) with RGB channel layout.
        raw_output
            If true, returns raw detection outputs (boxes, scores, class indices).
            Otherwise, returns annotated images.

        Returns
        -------
        result : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | list[Image.Image]
            If raw_output is True:
                boxes
                    List of bounding box tensors per batch. Each tensor has shape [num preds, 4]
                    where 4 represents (x1, y1, x2, y2).
                scores
                    List of class score tensors per batch multiplied by confidence.
                    Each tensor has shape [num_preds, num_classes].
                class_idx
                    List of class index tensors. Each tensor has shape [num_preds] with values
                    indicating the most probable class.
            If raw_output is False:
                List of PIL images with detected objects drawn and labeled.
        """
        NHWC_int_numpy_frames, image_tensor = app_to_net_image_inputs(images)

        image_tensor, scale_factor, pad = resize_pad(
            image_tensor, (self.model_image_height, self.model_image_width)
        )

        # Run Proposal Generator Inference
        with torch.no_grad():
            feature, proposals, objectness_logits = self.proposal_generator(
                image_tensor
            )

        padded_proposals = self.filter_proposals([proposals], [objectness_logits])[0]

        # Run ROI Head Inference
        pred_boxes, pred_scores, pred_classes = [], [], []
        for i in range(len(padded_proposals)):
            with torch.no_grad():
                boxes, scores, classes = self.roi_head(
                    feature[i : i + 1], padded_proposals[i : i + 1]
                )
            pred_boxes.append(boxes)
            pred_scores.append(scores)
            pred_classes.append(classes)

        boxes = torch.concat(pred_boxes)
        scores = torch.concat(pred_scores)
        classes = torch.concat(pred_classes)

        boxes[..., 0::2] = boxes[..., 0::2].clamp(
            min=0, max=self.model_image_width - pad[0]
        )
        boxes[..., 1::2] = boxes[..., 1::2].clamp(
            min=0, max=self.model_image_height - pad[1]
        )

        batched_boxes, batched_scores, batched_classes = batched_nms(
            self.boxes_iou_threshold,
            self.boxes_score_threshold,
            boxes.float(),
            scores.float(),
            classes,
        )

        color = create_color_map(self.num_classes + 1)
        labels_list = get_class_names("coco")

        out_images = []
        for i, (boxes, scores, labels) in enumerate(
            zip(batched_boxes, batched_scores, batched_classes, strict=False)
        ):
            h, w, _ = NHWC_int_numpy_frames[i].shape
            boxes = boxes[: self.max_vis_boxes]
            scores = scores[: self.max_vis_boxes]
            labels = labels[: self.max_vis_boxes]

            # resize coordinates to original image size.
            boxes[:, 0::2] = ((boxes[:, 0::2] - pad[0]) / scale_factor).clamp(
                min=0, max=w
            )
            boxes[:, 1::2] = ((boxes[:, 1::2] - pad[1]) / scale_factor).clamp(
                min=0, max=h
            )

            batched_boxes[i] = boxes
            batched_scores[i] = scores
            batched_classes[i] = labels

            for box, score, label in zip(boxes, scores, labels, strict=False):
                draw_box_from_xyxy(
                    NHWC_int_numpy_frames[i],
                    box[0:2].int(),
                    box[2:4].int(),
                    color=color[label + 1].tolist(),
                    size=2,
                    text=labels_list[label] + f" {score:.2f}",
                )
            out_images.append(Image.fromarray(NHWC_int_numpy_frames[i]))

        if raw_output:
            return batched_boxes, batched_scores, batched_classes

        return out_images

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

        proposal_generator = collection_model.components["proposal_generator"]
        pg_spec = (input_specs or {}).get(
            "proposal_generator", proposal_generator.get_input_spec()
        )

        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=pg_spec,
        )
        num_samples = num_samples or dataset.default_num_calibration_samples()
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Loading {num_samples} calibration samples.")
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)
        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]
        for sample_input, _ in dataloader:
            if component_name == "roi_head":
                features, proposals, objectness_logits = proposal_generator(
                    sample_input
                )
                app = cls(
                    proposal_generator=proposal_generator,
                    roi_head=model,
                    input_spec=pg_spec,
                )
                padded_proposals = app.filter_proposals(
                    [proposals], [objectness_logits]
                )[0]
                sample_input = (features, padded_proposals)
            if isinstance(sample_input, (tuple, list)):
                for i, tensor in enumerate(sample_input):
                    inputs[i].append(tensor)
            else:
                inputs[0].append(sample_input)
        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        model_output = self.proposal_generator(*model_input)
        yield model_output
        if isinstance(model_output, AsyncOnDeviceResult):
            features, proposals, objectness_logits = model_output.wait()
        else:
            features, proposals, objectness_logits = model_output
        padded_proposals = self.filter_proposals([proposals], [objectness_logits])[0]
        model_inputs = (features, padded_proposals)
        model_output = (
            self.roi_head(*(x.split(model_batch_size, dim=0) for x in model_inputs))
            if isinstance(model_output, AsyncOnDeviceResult)
            else self.roi_head(*model_inputs)
        )
        yield model_output
        return model_output

    @classmethod
    def from_components(
        cls,
        models: Sequence[ExecutableModelProtocol] | Sequence[AsyncOnDeviceModel],
    ) -> Detectron2DetectionApp:
        return cls(
            proposal_generator=models[0],  # type: ignore[arg-type]
            roi_head=models[1],  # type: ignore[arg-type]
        )

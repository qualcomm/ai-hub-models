# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import get_config_file
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models.detectron2_panoptic_seg.evaluator import (
    DEFAULT_BOXES_IOU_THRESHOLD,
    DEFAULT_BOXES_SCORE_THRESHOLD,
    DEFAULT_INSTANCES_SCORE_THRESH,
    DEFAULT_MAX_DET_POST_NMS,
    DEFAULT_OVERLAP_THRESHOLD,
    DEFAULT_STUFF_AREA_THRESH,
    build_panoptic_segments,
    paste_masks_in_image,
    select_instances,
)
from qai_hub_models.models.detectron2_panoptic_seg.model import (
    DEFAULT_CONFIG,
    DEFAULT_NUM_PROPOSALS,
    filter_rpn_proposals,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
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


class Detectron2PanopticSegApp(
    ProposalBasedDetectionApp,
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
):
    """
    End-to-end app for Detectron2 Panoptic Segmentation.

    For a given image input, the app will:
        * Preprocess the image (normalize, resize, etc).
        * Run the proposal generator (FPN backbone + sem-seg head + RPN).
        * Filter proposals.
        * Run the ROI head (box + mask prediction).
        * Merge instance and semantic predictions into a panoptic segmentation.
        * Return an annotated PIL image with per-segment colored overlays and labels.
    """

    def __init__(
        self,
        proposal_generator: Callable,
        roi_head: Callable,
        proposal_iou_threshold: float = 0.7,
        boxes_iou_threshold: float = DEFAULT_BOXES_IOU_THRESHOLD,
        boxes_score_threshold: float = DEFAULT_BOXES_SCORE_THRESHOLD,
        max_det_pre_nms: int = 1000,
        max_det_post_nms: int = DEFAULT_MAX_DET_POST_NMS,
        instances_score_thresh: float = DEFAULT_INSTANCES_SCORE_THRESH,
        overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
        stuff_area_thresh: int = DEFAULT_STUFF_AREA_THRESH,
        num_proposals: int = DEFAULT_NUM_PROPOSALS,
        input_spec: InputSpec | None = None,
    ) -> None:
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
        self.num_proposals = num_proposals
        self.instances_score_thresh = instances_score_thresh
        self.overlap_threshold = overlap_threshold
        self.stuff_area_thresh = stuff_area_thresh

    def _run_inference(
        self,
        image_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the two-stage pipeline: proposal generator, RPN proposal
        filtering (NMS off device), then the ROI head.

        Parameters
        ----------
        image_tensor
            Preprocessed image, shape (1, 3, model_image_height,
            model_image_width), fp32.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            boxes             : (1, num_proposals, num_box_classes, 4) xyxy.
            scores            : (1, num_proposals, num_classes), background dropped.
            mask_logits       : (num_proposals, num_mask_classes, mask_h, mask_w).
            filtered_proposals: (num_proposals, 4) xyxy proposals fed to the ROI head.
            sem_seg_logits    : semantic segmentation logits from the proposal generator.
        """
        with torch.no_grad():
            (
                feature_p2,
                feature_p3,
                feature_p4,
                feature_p5,
                proposals,
                scores_rpn,
                sem_seg_logits,
            ) = self.proposal_generator(image_tensor)

        filtered_proposals = filter_rpn_proposals(
            proposals,
            scores_rpn,
            self.model_image_height,
            self.model_image_width,
            self.num_proposals,
            pre_nms_topk=self.max_det_pre_nms,
            nms_thresh=self.proposal_iou_threshold,
        )

        with torch.no_grad():
            boxes, scores, mask_logits = self.roi_head(
                feature_p2,
                feature_p3,
                feature_p4,
                feature_p5,
                filtered_proposals,
            )

        return boxes, scores, mask_logits, filtered_proposals, sem_seg_logits

    def predict(
        self,
        images: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        raw_output: bool = False,
    ) -> (
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
        ]
        | list[Image.Image]
    ):
        """
        Run end-to-end panoptic segmentation.

        Parameters
        ----------
        images
            PIL image, numpy array (H W C uint8 RGB), or torch tensor (N C H W fp32 [0,1]).
        raw_output
            If True, returns (batched_boxes, batched_scores, batched_classes,
            mask_probs, sem_seg_results) in model-image pixel space.
            If False, returns annotated PIL images with panoptic overlay.

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor] | list[Image.Image]
            Raw model outputs tuple (raw_output=True) or annotated PIL images (raw_output=False).
        """
        NHWC_int_numpy_frames, image_tensor = app_to_net_image_inputs(images)

        image_tensor, scale, pad = resize_pad(
            image_tensor, (self.model_image_height, self.model_image_width)
        )

        boxes, scores, mask_logits, filtered_proposals, sem_seg_results = (
            self._run_inference(image_tensor)
        )

        # Dense per-class outputs → (proposal, class) score filter + per-class
        # NMS, matching detectron2's fast_rcnn_inference_single_image.
        # The model runs with batch_size=1, so operate on batch index 0.
        inst_boxes, inst_scores, inst_classes, mask_probs, paste_boxes = (
            select_instances(
                boxes[0],
                scores[0],
                mask_logits,
                filtered_proposals,
                self.boxes_score_threshold,
                self.boxes_iou_threshold,
                self.max_det_post_nms,
            )
        )

        left_pad, top_pad = pad
        inst_boxes = inst_boxes.clone()
        inst_boxes[:, 0::2] = inst_boxes[:, 0::2].clamp(
            min=0, max=self.model_image_width
        )
        inst_boxes[:, 1::2] = inst_boxes[:, 1::2].clamp(
            min=0, max=self.model_image_height
        )
        batched_boxes = [inst_boxes]
        batched_scores = [inst_scores]
        batched_classes = [inst_classes]

        if raw_output:
            for i in range(len(batched_boxes)):
                h, w, _ = NHWC_int_numpy_frames[i].shape
                batched_boxes[i][:, 0::2] = (
                    (batched_boxes[i][:, 0::2] - left_pad) / scale
                ).clamp(min=0, max=w)
                batched_boxes[i][:, 1::2] = (
                    (batched_boxes[i][:, 1::2] - top_pad) / scale
                ).clamp(min=0, max=h)
            return (
                batched_boxes,
                batched_scores,
                batched_classes,
                mask_probs,
                sem_seg_results,
            )

        # detectron2 Visualizer metadata is image-independent; build it once.
        cfg = get_cfg()
        cfg.merge_from_file(get_config_file(DEFAULT_CONFIG))
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        # Crop sem_seg to the valid (unpadded) content in stride space, using
        # resize_pad's math.floor rounding so upsampling maps back exactly.
        stride = self.model_image_height // sem_seg_results.shape[-2]

        # Build panoptic visualization
        out_images = []
        for i, img_array in enumerate(NHWC_int_numpy_frames):
            orig_h, orig_w, _ = img_array.shape

            # Valid content extent on the model canvas (before padding).
            content_h = math.floor(orig_h * scale)
            content_w = math.floor(orig_w * scale)
            sem_top = top_pad // stride
            sem_left = left_pad // stride
            sem_h = content_h // stride
            sem_w = content_w // stride
            sem_seg_crop = sem_seg_results[
                i : i + 1,
                :,
                sem_top : sem_top + sem_h,
                sem_left : sem_left + sem_w,
            ]
            sem_seg_full = F.interpolate(
                sem_seg_crop,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )[0].argmax(dim=0)  # (orig_h, orig_w) - already 0-53 stuff indices

            # Rescale the mask pooling boxes to original image coords; masks
            # must be pasted with the same boxes they were pooled on.
            paste_boxes_orig = paste_boxes.clone()
            paste_boxes_orig[:, 0::2] = (
                (paste_boxes_orig[:, 0::2] - left_pad) / scale
            ).clamp(0, orig_w - 1)
            paste_boxes_orig[:, 1::2] = (
                (paste_boxes_orig[:, 1::2] - top_pad) / scale
            ).clamp(0, orig_h - 1)

            inst_scores_i = batched_scores[i]
            inst_classes_i = batched_classes[i]

            # Paste masks at original image resolution
            inst_masks_orig = paste_masks_in_image(
                mask_probs,
                paste_boxes_orig,
                orig_h,
                orig_w,
            )  # (M, orig_h, orig_w) bool

            # Build panoptic map at original resolution. category_id holds
            # detectron2 contiguous thing ids (things) / stuff ids (stuff),
            # which the Visualizer's metadata maps to names and colors.
            panoptic_seg, segments = build_panoptic_segments(
                inst_masks_orig,
                inst_scores_i,
                sem_seg_full,
                self.instances_score_thresh,
                self.overlap_threshold,
                self.stuff_area_thresh,
            )
            segments_info: list[dict] = []
            for seg in segments:
                if seg.is_thing:
                    segments_info.append(
                        {
                            "id": seg.id,
                            "isthing": True,
                            "score": inst_scores_i[seg.source_index].item(),
                            "category_id": int(inst_classes_i[seg.source_index].item()),
                        }
                    )
                else:
                    segments_info.append(
                        {
                            "id": seg.id,
                            "isthing": False,
                            "category_id": seg.source_index,
                            "area": seg.area,
                        }
                    )

            # detectron2 Visualizer for high-quality rendering.
            v = Visualizer(
                img_array[:, :, ::-1],  # BGR for Visualizer
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE,
            )
            vis_out = v.draw_panoptic_seg(
                panoptic_seg.to("cpu"), segments_info, alpha=0.7
            )
            out_images.append(Image.fromarray(vis_out.get_image()))

        return out_images

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        # NOTE: CocoPanopticSegmentationDataset only supports DatasetSplit.VAL
        # (see its docstring), so calibration and eval draw from the same
        # split/image pool here. This mirrors the existing precedent in
        # yolov11_pose/CocoKeypointsDataset, which has the same limitation.
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
            DatasetSplit.VAL,
            input_spec=pg_spec,
        )
        num_samples = num_samples or dataset.default_num_calibration_samples()
        num_samples = (num_samples // batch_size) * batch_size
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)
        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]
        image_height, image_width = pg_spec["image"][0][2:]
        num_proposals = (
            input_spec["proposals"][0][0]
            if "proposals" in input_spec
            else DEFAULT_NUM_PROPOSALS
        )
        for sample_input, _ in dataloader:
            if component_name == "roi_head":
                # filter_rpn_proposals (and the roi_head it feeds) only
                # supports batch_size=1, so loop over each sample in the
                # dataloader batch individually rather than passing the
                # whole batch through at once.
                with torch.no_grad():
                    (
                        feature_p2,
                        feature_p3,
                        feature_p4,
                        feature_p5,
                        proposals,
                        scores_rpn,
                        _sem_seg,
                    ) = proposal_generator(sample_input)
                for b in range(proposals.shape[0]):
                    filtered_proposals = filter_rpn_proposals(
                        proposals[b : b + 1],
                        scores_rpn[b : b + 1],
                        image_height,
                        image_width,
                        num_proposals,
                    )
                    per_sample_input = (
                        feature_p2[b : b + 1],
                        feature_p3[b : b + 1],
                        feature_p4[b : b + 1],
                        feature_p5[b : b + 1],
                        filtered_proposals,
                    )
                    for i, tensor in enumerate(per_sample_input):
                        inputs[i].append(tensor)
            elif isinstance(sample_input, (tuple, list)):
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
        pg_output = self.proposal_generator(*model_input)
        yield pg_output

        if isinstance(pg_output, AsyncOnDeviceResult):
            (
                feature_p2,
                feature_p3,
                feature_p4,
                feature_p5,
                proposals,
                scores_rpn,
                sem_seg_results,
            ) = pg_output.wait()
        else:
            (
                feature_p2,
                feature_p3,
                feature_p4,
                feature_p5,
                proposals,
                scores_rpn,
                sem_seg_results,
            ) = pg_output

        # filter_rpn_proposals and the ROI head are batch_size=1 only, so run
        # them per sample and stack, letting a job carry many samples while the
        # host post-processing stays per image. The evaluator indexes all five
        # outputs by sample.
        features = (feature_p2, feature_p3, feature_p4, feature_p5)
        batch_size = proposals.shape[0]
        boxes_l, scores_l, mask_logits_l, proposals_l = [], [], [], []
        for i in range(batch_size):
            filtered_proposals = filter_rpn_proposals(
                proposals[i : i + 1],
                scores_rpn[i : i + 1],
                self.model_image_height,
                self.model_image_width,
                self.num_proposals,
                pre_nms_topk=self.max_det_pre_nms,
                nms_thresh=self.proposal_iou_threshold,
            )
            roi_inputs = (*(f[i : i + 1] for f in features), filtered_proposals)
            roi_output = self.roi_head(*roi_inputs)
            yield roi_output

            if isinstance(roi_output, AsyncOnDeviceResult):
                boxes, scores, mask_logits = roi_output.wait()
            else:
                boxes, scores, mask_logits = roi_output
            boxes_l.append(boxes)
            scores_l.append(scores)
            mask_logits_l.append(mask_logits)
            proposals_l.append(filtered_proposals)

        return (
            torch.cat(boxes_l, dim=0),
            torch.cat(scores_l, dim=0),
            torch.stack(mask_logits_l, dim=0),
            torch.stack(proposals_l, dim=0),
            sem_seg_results,
        )

    @classmethod
    def from_components(
        cls,
        models: Sequence[ExecutableModelProtocol] | Sequence[AsyncOnDeviceModel],
    ) -> Detectron2PanopticSegApp:
        return cls(
            proposal_generator=models[0],
            roi_head=models[1],
        )

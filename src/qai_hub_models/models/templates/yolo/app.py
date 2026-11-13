# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast, overload

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize

from qai_hub_models.datasets.coco import COCO_SKELETON
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
    rotated_batched_nms,
)
from qai_hub_models.models.templates.corevision.draw import (
    create_color_map,
    draw_box_from_xyxy,
    draw_connections,
    draw_obb_on_image,
    draw_points,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.models.templates.yolo.utils import detect_postprocess
from qai_hub_models.utils.input_spec import InputSpec


def _parse_prompt_text(prompt_text: list[str] | str | None) -> list[str]:
    """Normalize prompt text from a list, comma-separated string, or labels file."""
    if prompt_text is None:
        return []
    if isinstance(prompt_text, str):
        if prompt_text.endswith(".txt"):
            with open(prompt_text) as f:
                return [t.rstrip("\r\n").strip() for t in f if t.strip()]
        return [t.strip() for t in prompt_text.split(",") if t.strip()]
    return [t.strip() for t in prompt_text if t.strip()]


class YoloObjectDetectionApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference
    with Yolo object detection models.

    The app works with following models:
        * YoloV7
        * YoloV8Detection
        * YoloV10Detection
        * YoloV11Detection
        * Yolo26Detection

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run Yolo inference
        * if requested, post-process YoloV7 output using non maximum suppression
        * if requested, draw the predicted bounding boxes on the input image
    """

    def __init__(
        self,
        model: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        model_includes_postprocessing: bool = True,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        Initialize a YoloObjectDetectionApp application.

        Parameters
        ----------
        model
            Yolo object detection model.

            Inputs:
                Tensor of shape (N C H W x float32) with range [0, 1] and RGB channel layout.

            Outputs:
                boxes: Tensor of shape [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
                            The output are in the range of the input image's dimensions (NOT [0-1])

                scores: Tensor of shape [batch, num_preds, # of classes (typically 80)]

                class_idx: Tensor of shape [num_preds] where the values are the indices
                            of the most probable class of the prediction.

        nms_score_threshold
            Score threshold for non maximum suppression.

        nms_iou_threshold
            Intersection over Union threshold for non maximum suppression.

        model_includes_postprocessing
            Whether the model includes postprocessing steps beyond the detector.

        input_spec
            Model input spec. If provided, input images are resized to the
            ``image`` input's (height, width) before inference.
            If None, no resizing is performed.
        """
        self.model = model
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.model_includes_postprocessing = model_includes_postprocessing
        if input_spec is not None:
            _, _, h, w = input_spec["image"][0]
            self.model_image_input_shape: tuple[int, int] | None = (h, w)
        else:
            self.model_image_input_shape = None

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """Verify image size is valid model input."""
        raise NotImplementedError

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | list[np.ndarray]
    ):
        # See predict_boxes_from_image.
        return self.predict_boxes_from_image(*args, **kwargs)

    @overload
    def predict_boxes_from_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: Literal[False] = ...,
        prompt_text: list[str] | str | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def predict_boxes_from_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: Literal[True],
        prompt_text: list[str] | str | None = ...,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]: ...

    def predict_boxes_from_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: bool = False,
        prompt_text: list[str] | str | None = None,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | list[np.ndarray]
    ):
        """
        From the provided image or tensor, predict the bounding boxes & classes of objects detected within.

        Parameters
        ----------
        pixel_values_or_image
            PIL image
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        raw_output
            See "returns" doc section for details.

        prompt_text
            Class names to detect. Only used by YoloWorldPromptDetectionApp subclasses.
            List or comma-separated string. Falls back to ``self.prompt_text`` when set.
            Ignored by non-prompt models.

        Returns
        -------
        output : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | list[np.ndarray]
            If raw_output is True, returns:
                boxes : list[torch.Tensor]
                    Bounding box locations per batch.
                    List element shape is [num preds, 4] where 4 == (x1, y1, x2, y2).
                scores : list[torch.Tensor]
                    Class scores per batch multiplied by confidence.
                    List element shape is [num_preds, # of classes (typically 80)].
                class_idx : list[torch.Tensor]
                    Shape is [num_preds] where the values are the indices of the most probable class of the prediction.

            If raw_output is False, returns:
                images : list[np.ndarray]
                    A list of predicted RGB, [H, W, C] images (one list element per batch).
                    Each image will have bounding boxes drawn.
        """
        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )
        scale = None
        padding = None
        if self.model_image_input_shape is not None:
            NCHW_fp32_torch_frames, scale, padding = resize_pad(
                NCHW_fp32_torch_frames, self.model_image_input_shape
            )
        self.check_image_size(NCHW_fp32_torch_frames)

        txt_feats: torch.Tensor | None = None
        filter_indices: set[int] | None = None
        if isinstance(self, YoloWorldPromptDetectionApp):
            classes: list[str]
            if prompt_text is not None:
                classes = (
                    [t.strip() for t in prompt_text.split(",") if t.strip()]
                    if isinstance(prompt_text, str)
                    else list(prompt_text)
                )
            elif self.prompt_text:
                classes = self.prompt_text
            else:
                raise ValueError(
                    "prompt_text must be provided either as an argument or via from_components."
                )
            if not classes:
                raise ValueError("prompt_text must contain at least one class name.")
            with torch.no_grad():
                txt_feats = self.text_encoder.encode_classes(classes)
            filter_indices = set(range(len(classes)))

        # Run prediction
        if txt_feats is not None:
            world_app = cast("YoloWorldPromptDetectionApp", self)
            model_output_typing: tuple[torch.Tensor, ...] = self.model(
                NCHW_fp32_torch_frames,
                world_app._txt_feats_for_image(NCHW_fp32_torch_frames, txt_feats),
            )
        else:
            model_output_typing = self.model(NCHW_fp32_torch_frames)
        model_output = model_output_typing
        if self.model_includes_postprocessing:
            pred_boxes, pred_scores, pred_class_idx = model_output
        else:
            if isinstance(model_output, torch.Tensor):
                model_output = (model_output,)
            pred_boxes, pred_scores, pred_class_idx = self.pre_nms_postprocess(
                *model_output
            )

        # Non Maximum Suppression on each batch
        pred_post_nms_boxes, pred_post_nms_scores, pred_post_nms_class_idx = (
            batched_nms(
                self.nms_iou_threshold,
                self.nms_score_threshold,
                pred_boxes,
                pred_scores,
                pred_class_idx,
            )
        )

        # Transform box coordinates back to original image space
        if scale is not None and padding is not None:
            pad_x, pad_y = padding
            for i in range(len(pred_post_nms_boxes)):
                boxes = pred_post_nms_boxes[i]
                if boxes.numel() > 0:
                    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
                    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

        # Subclass-defined post-NMS filtering (no-op when filter_indices is None)
        pred_post_nms_boxes, pred_post_nms_scores, pred_post_nms_class_idx = (
            self._post_nms_filter(
                pred_post_nms_boxes,
                pred_post_nms_scores,
                pred_post_nms_class_idx,
                filter_indices,
            )
        )
        # Return raw output if requested
        if raw_output or isinstance(pixel_values_or_image, torch.Tensor):
            return (pred_post_nms_boxes, pred_post_nms_scores, pred_post_nms_class_idx)

        # Add boxes to each batch
        for batch_idx in range(len(pred_post_nms_boxes)):
            pred_boxes_batch = pred_post_nms_boxes[batch_idx]
            for box in pred_boxes_batch:
                draw_box_from_xyxy(
                    NHWC_int_numpy_frames[batch_idx],
                    box[0:2].int(),
                    box[2:4].int(),
                    color=(0, 255, 0),
                    size=2,
                )

        return NHWC_int_numpy_frames

    def _post_nms_filter(
        self,
        boxes: list[torch.Tensor],
        scores: list[torch.Tensor],
        class_idx: list[torch.Tensor],
        filter_indices: set[int] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Hook for subclasses to filter post-NMS detections.

        The default implementation is a no-op; override in a subclass to
        remove unwanted detections after NMS has been applied.
        """
        return boxes, scores, class_idx

    def pre_nms_postprocess(
        self, *predictions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the output of the YOLO detector for input to NMS.

        Parameters
        ----------
        *predictions
            Variable number of tensor outputs from the Yolo detection model.
            Tensor shapes vary by model implementation.

        Returns
        -------
        boxes : torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
        scores : torch.Tensor
            Class scores multiplied by confidence. Shape is [batch, num_preds].
        class_idx : torch.Tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
        """
        return detect_postprocess(predictions[0])


class YoloWorldPromptDetectionApp(YoloObjectDetectionApp):
    """
    Light-weight app code for end-to-end inference with Yolo World prompt-based
    object detection models (e.g. YoloWorld).

    The app owns all prompt handling:
      * parses ``prompt_text`` provided by the user,
      * generates text embeddings from the model text encoder, and
      * passes those embeddings to the detector together with the image.
    """

    # Declared here so predict_boxes_from_image (base class) can access them
    # after an isinstance(self, YoloWorldPromptDetectionApp) guard.
    # Subclasses (e.g. YoloWorldDetectionApp) set these at construction time.
    prompt_text: list[str] = []
    text_encoder: Any  # implements encode_classes(list[str]) -> torch.Tensor

    def __init__(
        self,
        model: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        model_includes_postprocessing: bool = True,
    ) -> None:
        """
        Initialize a YoloWorldPromptDetectionApp.

        Parameters
        ----------
        model
            Yolo World detector. It must accept ``(image, txt_feats)`` as inputs.
        nms_score_threshold
            Score threshold for non maximum suppression.
        nms_iou_threshold
            Intersection over Union threshold for non maximum suppression.
        model_includes_postprocessing
            Whether the model includes postprocessing steps beyond the detector.
        """
        super().__init__(
            model,
            nms_score_threshold,
            nms_iou_threshold,
            model_includes_postprocessing,
        )

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """Verify image size is valid model input."""
        invalid_dims = [s for s in pixel_values.shape[-2:] if s % 32 != 0]
        if invalid_dims:
            raise ValueError(
                f"Invalid image size: dimensions {invalid_dims} are not divisible by 32."
            )

    def _txt_feats_for_image(
        self, pixel_values: torch.Tensor, txt_feats: torch.Tensor
    ) -> torch.Tensor:
        """Return text embeddings shaped for the current image batch/model input."""
        txt_feats = txt_feats.to(device=pixel_values.device)
        input_spec_fn = getattr(self.model, "get_input_spec", None)
        txt_feats_spec = (
            input_spec_fn().get("txt_feats") if input_spec_fn is not None else None
        )
        if txt_feats_spec is not None:
            expected_num_classes = txt_feats_spec[0][1]
            if txt_feats.shape[1] > expected_num_classes:
                raise ValueError(
                    "prompt_text contains more classes than the model txt_feats input supports. "
                    f"Expected at most {expected_num_classes}, got {txt_feats.shape[1]}."
                    f"Model supports only COCO classes to identify"
                )
            if txt_feats.shape[1] < expected_num_classes:
                padding = torch.zeros(
                    txt_feats.shape[0],
                    expected_num_classes - txt_feats.shape[1],
                    txt_feats.shape[2],
                    dtype=txt_feats.dtype,
                    device=txt_feats.device,
                )
                txt_feats = torch.cat([txt_feats, padding], dim=1)

        batch_size = pixel_values.shape[0]
        if txt_feats.shape[0] == batch_size:
            return txt_feats
        if txt_feats.shape[0] != 1:
            raise ValueError(
                f"Text embedding batch size {txt_feats.shape[0]} does not match image batch size {batch_size}."
            )
        return txt_feats.expand(batch_size, -1, -1)

    def _post_nms_filter(
        self,
        boxes: list[torch.Tensor],
        scores: list[torch.Tensor],
        class_idx: list[torch.Tensor],
        filter_indices: set[int] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Retain only detections whose class index maps to prompt_text."""
        if filter_indices is None:
            return boxes, scores, class_idx
        for batch_idx in range(len(class_idx)):
            if len(class_idx[batch_idx]) == 0:
                continue
            keep = torch.tensor(
                [
                    int(cls_idx.item()) in filter_indices
                    for cls_idx in class_idx[batch_idx]
                ],
                dtype=torch.bool,
                device=class_idx[batch_idx].device,
            )
            boxes[batch_idx] = boxes[batch_idx][keep]
            scores[batch_idx] = scores[batch_idx][keep]
            class_idx[batch_idx] = class_idx[batch_idx][keep]
        return boxes, scores, class_idx


class YoloSegmentationApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference
    with Yolo segmentation model.

    The app works with following models:
        * YoloV8Segmentation
        * YoloV11Segmentation

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run Yolo inference
        * By default,
            - post-processes output using non-maximum-suppression
            - applies predicted mask on input image
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        Initialize a YoloSegmentationApp application.

        Parameters
        ----------
        model
            Yolo Segmentation model

            Inputs:
                Tensor of shape (N H W C x float32) with range [0, 1] and RGB channel layout.

            Outputs:
                boxes: torch.Tensor
                    Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
                scores: torch.Tensor
                    Class scores multiplied by confidence: Shape is [batch, num_preds]
                masks: torch.Tensor
                    Predicted masks: Shape is [batch, num_preds, 32]
                classes: torch.Tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
                protos: torch.Tensor
                    Tensor of shape[batch, 32, mask_h, mask_w]
                    Multiply masks and protos to generate output masks.

        nms_score_threshold
            Score threshold for non maximum suppression.

        nms_iou_threshold
            Intersection over Union threshold for non maximum suppression.

        input_spec
            Model input spec. If None, defaults to 640x640.
        """
        self.model = model
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        if input_spec is not None:
            _, _, self.input_height, self.input_width = input_spec["image"][0]
        else:
            self.input_height = 640
            self.input_width = 640

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """Verify image size is valid model input."""
        if not all(s % 32 == 0 for s in pixel_values.shape[-2:]):
            raise ValueError(
                f"Image spatial dimensions must be multiples of 32, got {pixel_values.shape[-2:]}"
            )

    def preprocess_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        img_size = (self.input_height, self.input_width)
        return Resize(img_size)(pixel_values)

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> (
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ]
        | list[Image.Image]
    ):
        # See predict_boxes_from_image.
        return self.predict_segmentation_from_image(*args, **kwargs)

    def filter_predictions(
        self,
        pred_post_nms_boxes: list[torch.Tensor],
        pred_post_nms_scores: list[torch.Tensor],
        pred_post_nms_class_idx: list[torch.Tensor],
        pred_post_nms_masks: list[torch.Tensor],
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """
        Filter post-NMS predictions before mask processing.

        Override in a subclass to apply custom filtering (e.g. class-based).
        The base implementation is a no-op and returns all inputs unchanged.

        Parameters
        ----------
        pred_post_nms_boxes
            Per-batch bounding-box tensors, each of shape [num_boxes, 4].
        pred_post_nms_scores
            Per-batch score tensors, each of shape [num_boxes].
        pred_post_nms_class_idx
            Per-batch class-index tensors, each of shape [num_boxes].
        pred_post_nms_masks
            Per-batch mask-coefficient tensors, each of shape [num_boxes, 32].

        Returns
        -------
        pred_post_nms_boxes : list[torch.Tensor]
            Filtered per-batch bounding-box tensors.
        pred_post_nms_scores : list[torch.Tensor]
            Filtered per-batch score tensors.
        pred_post_nms_class_idx : list[torch.Tensor]
            Filtered per-batch class-index tensors.
        pred_post_nms_masks : list[torch.Tensor]
            Filtered per-batch mask-coefficient tensors.
        """
        return (
            pred_post_nms_boxes,
            pred_post_nms_scores,
            pred_post_nms_class_idx,
            pred_post_nms_masks,
        )

    def process_and_resize_masks(
        self,
        pred_post_nms_masks: list[torch.Tensor],
        pred_post_nms_boxes: list[torch.Tensor],
        proto: torch.Tensor,
        input_h: int,
        input_w: int,
    ) -> list[np.ndarray]:
        """
        Apply proto coefficients to mask predictions, upsample to model input
        size, then resize to the original image dimensions.

        Override in a subclass to change mask processing behaviour (e.g. to
        handle variable-length batches or use a different interpolation path).

        Parameters
        ----------
        pred_post_nms_masks
            Per-batch mask-coefficient tensors, each of shape [num_boxes, 32].
        pred_post_nms_boxes
            Per-batch bounding-box tensors, each of shape [num_boxes, 4].
        proto
            Proto tensor of shape [batch, 32, mask_h, mask_w].
        input_h
            Original image height (resize target).
        input_w
            Original image width (resize target).

        Returns
        -------
        list[np.ndarray]
            Per-batch float32 mask arrays of shape [num_boxes, input_h, input_w].
        """
        from ultralytics.utils.ops import process_mask

        processed = [
            process_mask(
                proto[batch_idx],
                pred_post_nms_masks[batch_idx],
                pred_post_nms_boxes[batch_idx],
                (self.input_height, self.input_width),
                upsample=True,
            ).numpy()
            for batch_idx in range(len(pred_post_nms_masks))
        ]

        resized: torch.Tensor = F.interpolate(
            input=torch.Tensor(processed),
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )
        return list(resized.numpy())

    def create_output_images(
        self,
        NHWC_int_numpy_frames: list[np.ndarray],
        resized_masks: list[np.ndarray],
    ) -> list[Image.Image]:
        """
        Overlay segmentation masks on the input images and return annotated PIL images.

        Override in a subclass to change the visualisation strategy (e.g. to
        use per-instance threshold-based colouring instead of argmax).

        Parameters
        ----------
        NHWC_int_numpy_frames
            Per-batch uint8 RGB image arrays of shape [H, W, 3].
        resized_masks
            Per-batch float32 mask arrays of shape [num_boxes, H, W].

        Returns
        -------
        list[Image.Image]
            Annotated PIL images, one per batch element.
        """
        pred_post_nms_resized_masks = torch.from_numpy(np.stack(resized_masks))
        pred_mask_img = torch.argmax(pred_post_nms_resized_masks, 1)

        color_map = create_color_map(int(pred_mask_img.max().item()) + 1)
        out = []
        for i, img_tensor in enumerate(NHWC_int_numpy_frames):
            out.append(
                Image.blend(
                    Image.fromarray(img_tensor),
                    Image.fromarray(color_map[pred_mask_img[i]]),
                    alpha=0.5,
                )
            )
        return out

    def predict_segmentation_from_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: bool = False,
    ) -> (
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ]
        | list[Image.Image]
    ):
        """
        From the provided image or tensor, predict the bounding boxes & classes of objects detected within.

        Parameters
        ----------
        pixel_values_or_image
            PIL image
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        raw_output
            See "returns" doc section for details.

        Returns
        -------
        output : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | list[Image.Image]
            If raw_output is True, returns:
                pred_boxes : list[torch.Tensor]
                    List of predicted boxes for all the batches.
                    Each pred_box is of shape [num_boxes, 4].
                pred_scores : list[torch.Tensor]
                    List of scores for each predicted box for all the batches.
                    Each pred_score is of shape [num_boxes].
                pred_masks : list[torch.Tensor]
                    List of predicted masks for all the batches.
                    Each pred_mask is of shape [num_boxes, input_h, input_w].
                pred_classes : list[torch.Tensor]
                    List of predicted class for all the batches.
                    Each pred_class is of shape [num_boxes].

            If raw_output is False, returns:
                image_with_masks : list[Image.Image]
                    Input image with predicted masks applied.
        """
        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )

        # Cache input spatial dimension to use for post-processing
        input_h, input_w = NCHW_fp32_torch_frames.shape[2:]
        NCHW_fp32_torch_frames = self.preprocess_input(NCHW_fp32_torch_frames)

        self.check_image_size(NCHW_fp32_torch_frames)

        # Run prediction
        pred_boxes, pred_scores, pred_masks, pred_class_idx, proto = self.model(
            NCHW_fp32_torch_frames
        )

        # Non Maximum Suppression on each batch
        (
            pred_post_nms_boxes,
            pred_post_nms_scores,
            pred_post_nms_class_idx,
            pred_post_nms_masks,
        ) = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            pred_boxes,
            pred_scores,
            pred_class_idx,
            pred_masks,
        )

        # Filter predictions (override filter_predictions to customise)
        (
            pred_post_nms_boxes,
            pred_post_nms_scores,
            pred_post_nms_class_idx,
            pred_post_nms_masks,
        ) = self.filter_predictions(
            pred_post_nms_boxes,
            pred_post_nms_scores,
            pred_post_nms_class_idx,
            pred_post_nms_masks,
        )

        # Process masks and resize to original image dimensions
        # (override process_and_resize_masks to customise)
        resized_masks: list[np.ndarray] = self.process_and_resize_masks(
            pred_post_nms_masks,
            pred_post_nms_boxes,
            proto,
            input_h,
            input_w,
        )

        # Return raw output if requested
        if raw_output or isinstance(pixel_values_or_image, torch.Tensor):
            return (
                pred_post_nms_boxes,
                pred_post_nms_scores,
                [torch.from_numpy(m) for m in resized_masks],
                pred_post_nms_class_idx,
            )

        # Overlay masks on images (override create_output_images to customise)
        return self.create_output_images(NHWC_int_numpy_frames, resized_masks)


class YoloPoseApp:
    """
    Light-weight app for end-to-end inference with YOLO pose estimation models.

    For a given image input, the app will:
        * pre-process the image (convert to range [0, 1])
        * Run YOLO pose inference
        * Post-process the output using NMS to extract confident detections
        * Optionally draw keypoints and skeleton on the image
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        Initialize a YoloPoseApp application.

        Parameters
        ----------
        model
            YOLO pose estimation model.

            Inputs:
                Tensor of shape (N, 3, H, W) with range [0, 1] and RGB channel layout.

            Outputs:
                boxes: Tensor of shape [batch, num_preds, 4] where 4 == (x1, y1, x2, y2).
                scores: Tensor of shape [batch, num_preds].
                keypoints: Tensor of shape [batch, num_preds, num_keypoints, 3]
                          where 3 == (x, y, visibility).

        nms_score_threshold
            Confidence score threshold for NMS; detections below this are discarded.

        nms_iou_threshold
            IoU threshold for NMS duplicate suppression.

        input_spec
            Model input spec. If None, defaults to 640x640.
        """
        self.model = model
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        if input_spec is not None:
            _, _, self.input_height, self.input_width = input_spec["image"][0]
        else:
            self.input_height = 640
            self.input_width = 640

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """Verify image size is valid model input."""
        if not all(s % 32 == 0 for s in pixel_values.shape[-2:]):
            raise ValueError(
                f"Image dimensions must be divisible by 32. Got {pixel_values.shape[-2:]}"
            )

    def preprocess_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Resize or otherwise prepare the input tensor before inference."""
        img_size = (self.input_height, self.input_width)
        return Resize(img_size)(pixel_values)

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray | list[Image.Image]:
        # See predict_pose_keypoints.
        return self.predict_pose_keypoints(*args, **kwargs)

    def predict_pose_keypoints(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
        raw_output: bool = False,
    ) -> np.ndarray | list[Image.Image]:
        """
        Predicts pose keypoints for persons in the image.

        Parameters
        ----------
        pixel_values_or_image
            PIL image(s)
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        raw_output
            See "returns" doc section for details.

        Returns
        -------
        result : np.ndarray | list[Image.Image]
            If raw_output is True, returns:
            keypoints
                List of numpy arrays (one per batch image), each of shape
                [num_detections, num_keypoints, 3].
                Each keypoint is an (x, y, visibility) tuple within the image.

            If raw_output is False, returns:
            predicted_images
                Images with keypoints and skeleton drawn.
        """
        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )
        NCHW_fp32_torch_frames = self.preprocess_input(NCHW_fp32_torch_frames)
        self.check_image_size(NCHW_fp32_torch_frames)

        # Rebuild NHWC uint8 frames from the (possibly resized) tensor so that
        # keypoints, which are in the preprocessed coordinate space, are drawn
        # on an image of the same spatial dimensions.
        NHWC_int_numpy_frames = [
            (NCHW_fp32_torch_frames[i].permute(1, 2, 0).numpy() * 255)
            .clip(0, 255)
            .astype(np.uint8)
            for i in range(NCHW_fp32_torch_frames.shape[0])
        ]

        # Run inference
        boxes, scores, keypoints = self.model(NCHW_fp32_torch_frames)

        # Apply NMS to filter low-confidence and duplicate detections.
        # keypoints shape [B, N, num_kpts, 3] is passed as an additional gather arg
        # so it is filtered in lock-step with boxes/scores.
        _post_nms_boxes, _post_nms_scores, post_nms_keypoints = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            boxes,
            scores,
            None,  # no class indices for single-class (person) detector
            keypoints,  # gathered alongside boxes/scores
        )

        if raw_output:
            return np.array([kpts.numpy() for kpts in post_nms_keypoints])

        # Draw keypoints and skeleton on images
        predicted_images = []
        for batch_idx in range(len(NHWC_int_numpy_frames)):
            img = NHWC_int_numpy_frames[batch_idx].copy()
            batch_keypoints = post_nms_keypoints[batch_idx].numpy()  # [D, num_kpts, 3]

            for kpts in batch_keypoints:  # kpts: [num_kpts, 3]
                visible = kpts[:, 2] > 0.5  # visibility threshold

                # Draw skeleton connections (only between visible keypoints)
                for i, j in COCO_SKELETON:
                    if visible[i] and visible[j]:
                        draw_connections(
                            img,
                            np.array([[kpts[i, :2], kpts[j, :2]]]),
                            color=(0, 255, 0),
                            size=2,
                        )

                # Draw keypoint dots
                visible_pts = kpts[visible, :2]
                if len(visible_pts) > 0:
                    draw_points(img, visible_pts, color=(255, 0, 0), size=6)

            predicted_images.append(Image.fromarray(img))

        return predicted_images


class YoloOBBApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference
    with Yolo OBB (Oriented Bounding Box) models.

    The app works with following models:
        * YoloV8-OBB

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run Yolo inference
        * Post-process output Non Maximum Suppression rotated
        * Draw the predicted oriented bounding boxes on the input image
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        nms_score_threshold: float = 0.5,
        nms_iou_threshold: float = 0.1,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        Initialize a YoloOBBApp application.

        Parameters
        ----------
        model
            Yolo OBB model

            Inputs:
                Tensor of shape (N C H W x float32) with range [0, 1] and RGB channel layout.

            Outputs:
                boxes: torch.Tensor
                    Bounding box locations (xywh). Shape is [batch, num preds, 4].
                angles: torch.Tensor
                    Box rotation angle in radians. Shape is [batch, num_preds].
                scores: torch.Tensor
                    Class scores multiplied by confidence: Shape is [batch, num_preds].
                classes: torch.Tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class.

        nms_score_threshold
            Score threshold for non maximum suppression.

        nms_iou_threshold
            Intersection over Union threshold for non maximum suppression.

        input_spec
            Model input spec. If None, defaults to 640x640.
        """
        self.model = model
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        if input_spec is not None:
            _, _, self.input_height, self.input_width = input_spec["image"][0]
        else:
            self.input_height = 640
            self.input_width = 640

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """Verify image size is valid model input."""
        if not all(s % 32 == 0 for s in pixel_values.shape[-2:]):
            raise ValueError(
                f"Image spatial dimensions must be multiples of 32, got {pixel_values.shape[-2:]}"
            )

    def preprocess_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        scaled_img, _, _ = resize_pad(
            pixel_values, (self.input_height, self.input_width)
        )
        return scaled_img

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> (
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ]
        | list[Image.Image]
    ):
        return self.predict_obb_from_image(*args, **kwargs)

    def predict_obb_from_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: bool = False,
    ) -> (
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ]
        | list[Image.Image]
    ):
        """
        From the provided image or tensor, predict the oriented bounding boxes & classes of objects.

        Parameters
        ----------
        pixel_values_or_image
            PIL image
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        raw_output
            See "returns" doc section for details.

        Returns
        -------
        output : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | list[Image.Image]
            If raw_output is True, returns:
                boxes : list[torch.Tensor]
                    Bounding box locations per batch.
                    List element shape is [num preds, 4] where 4 == (x_center, y_center, w, h).
                scores : list[torch.Tensor]
                    Confidence score per box.
                    List element shape is [num_preds].
                angles : list[torch.Tensor]
                    Rotation angles corresponding to each bounding box (in radians).
                    List element shape is [num_preds]
                class_idx : list[torch.Tensor]
                    Shape is [num_preds] where the values are the indices of the most probable class of the prediction.

            If raw_output is False, returns:
                images : list[Image.Image]
                    A list of predicted RGB, [H, W, C] images.
                    Each image will have oriented bounding boxes drawn.
        """
        # Input Prep
        _, NCHW_fp32_torch_frames = app_to_net_image_inputs(pixel_values_or_image)

        NCHW_fp32_torch_frames = self.preprocess_input(NCHW_fp32_torch_frames)
        self.check_image_size(NCHW_fp32_torch_frames)

        # 1. Run prediction
        # Expecting:
        # boxes:  [Batch, N, 4] (cx, cy, w, h)
        # angles: [Batch, N, 1] (radians)
        # scores: [Batch, C, N] OR [Batch, N, C] (class probabilities)
        pred_boxes, pred_angles, pred_scores, pred_class_idx = self.model(
            NCHW_fp32_torch_frames
        )

        final_boxes, final_scores, final_angles, final_classes = rotated_batched_nms(
            pred_boxes_xywh=pred_boxes,
            pred_angles_rad=pred_angles,
            pred_scores=pred_scores,
            pred_class_idx=pred_class_idx,
            score_thr=self.nms_score_threshold,
            iou_thr=self.nms_iou_threshold,
            class_aware=True,
            canonicalize=False,
        )
        out_images = []
        if not raw_output:
            for i in range(len(final_boxes)):
                img_np = (
                    NCHW_fp32_torch_frames[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    * 255.0
                ).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                if final_boxes[i].numel() > 0:
                    draw_obb_on_image(img_pil, final_boxes[i], final_angles[i])
                out_images.append(img_pil)
            return out_images

        return final_boxes, final_scores, final_angles, final_classes

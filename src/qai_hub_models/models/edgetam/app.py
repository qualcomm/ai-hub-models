# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qai_hub.client import DatasetEntries
from sam2.modeling.sam2_base import SAM2Base as Sam2
from torch.utils.data import DataLoader

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models.templates.corevision.draw import create_color_map
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.sam2.model_patches import (
    mask_postprocessing,
)
from qai_hub_models.utils.base_app import CollectionAppQuantizeProtocol
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


class EdgeTAMApp:
    """
    App for single-frame image segmentation using EdgeTAM.

    This is the image-mode variant of EdgeTAM, equivalent to the existing
    EdgeTAM model that operates on a single frame. For multi-frame video
    tracking, use :class:`EdgeTAMVideoApp`.

    Accepts a single image (numpy array) and point prompts, and returns the
    segmented image with the mask overlaid.
    """

    def __init__(
        self,
        encoder: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        video_decoder: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        memory_encoder: Callable[
            [torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        sam2: Any,
        maskmem_pos_enc: torch.Tensor,
        encoder_input_img_size: int = 1024,
        mask_threshold: float = 0.0,
    ) -> None:
        self._video_app = EdgeTAMVideoApp(
            encoder=encoder,
            video_decoder=video_decoder,
            memory_encoder=memory_encoder,
            sam2=sam2,
            maskmem_pos_enc=maskmem_pos_enc,
            encoder_input_img_size=encoder_input_img_size,
            mask_threshold=mask_threshold,
        )

    def predict(
        self,
        image: np.ndarray,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        raw_output: bool = False,
    ) -> np.ndarray:
        """
        Segment an object in a single image.

        Parameters
        ----------
        image
          RGB numpy array of shape (H, W, C) with uint8 values.
        point_coords
          Point coordinates. Shape: [N, 2] in pixel coords.
        point_labels
          Point labels (1=positive, 0=negative). Shape: [N].
        raw_output
          If True, returns a binary mask instead of a painted image.

        Returns
        -------
        np.ndarray
          Painted RGB image with the segmentation mask overlaid, or a binary
          mask if ``raw_output`` is True.
        """
        results = self._video_app.track(
            [image], point_coords, point_labels, raw_output=raw_output
        )
        return results[0]


class EdgeTAMVideoApp(CollectionAppQuantizeProtocol):
    """
    App for video object tracking using EdgeTAM.

    Uses the 3 exported component interfaces (encoder, memory_encoder,
    video_decoder) for both CPU and on-device inference.
    Components may be PyTorch modules (CPU) or OnDeviceModel wrappers
    (on-device). Memory attention always runs on CPU via sam2.
    """

    def __init__(
        self,
        encoder: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        video_decoder: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        memory_encoder: Callable[
            [torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        sam2: Sam2,
        maskmem_pos_enc: torch.Tensor,
        encoder_input_img_size: int = 1024,
        mask_threshold: float = 0.0,
    ) -> None:
        self.encoder = encoder
        self.video_decoder = video_decoder
        self.memory_encoder = memory_encoder
        self.sam2 = sam2
        self.encoder_input_img_size = encoder_input_img_size
        self.mask_threshold = mask_threshold

        self.maskmem_pos_enc = maskmem_pos_enc

        sam2.binarize_mask_from_pts_for_mem_enc = True
        self._precompute_backbone_encodings(sam2, encoder_input_img_size)

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        components = collection_model.components
        encoder = components["encoder"]
        encoder_spec = input_specs.get("encoder") if input_specs else None
        if encoder_spec is None:
            encoder_spec = encoder.get_input_spec()
        component_spec = (
            input_specs[component_name]
            if input_specs
            else components[component_name].get_input_spec()
        )
        batch_size = get_batch_size(component_spec) or 1

        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=encoder_spec,
        )
        num_samples = num_samples or dataset.default_num_calibration_samples()
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Loading {num_samples} calibration samples.")
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)

        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(component_spec))
        ]

        for sample_input, _ in dataloader:
            # Dataset returns (image, coords, labels). Discard dataset coords
            # and use image-centre point prompts instead — they reliably hit
            # the object so object_score_logits stays positive across all
            # calibration samples.
            image, _, _ = sample_input
            B = image.shape[0]
            point_coords = torch.full((B, 2, 2), 0.5)
            point_labels = torch.ones(B, 2)

            if component_name == "encoder":
                tensors: tuple[torch.Tensor, ...] = (image, point_coords, point_labels)

            elif component_name == "video_decoder":
                _, hr1, hr2, sparse, pix_feat = encoder(
                    image, point_coords, point_labels
                )
                tensors = (pix_feat, hr1, hr2, sparse)

            elif component_name == "memory_encoder":
                raw_embeddings, hr1, hr2, sparse, pix_feat = encoder(
                    image, point_coords, point_labels
                )
                low_res_masks, _, _, _ = components["video_decoder"](
                    pix_feat, hr1, hr2, sparse
                )
                img_size = encoder_spec["image"][0][-1]  # H == W for square input
                mask_for_mem = F.interpolate(
                    low_res_masks.float(),
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                mask_for_mem = (mask_for_mem > 0).float()
                tensors = (raw_embeddings, mask_for_mem)

            else:
                raise ValueError(
                    f"Unknown component '{component_name}'. "
                    "Expected one of: encoder, video_decoder, memory_encoder."
                )

            for i, tensor in enumerate(tensors):
                inputs[i].append(tensor)

        return make_hub_dataset_entries(tuple(inputs), list(component_spec.keys()))

    def _precompute_backbone_encodings(self, sam2: Sam2, img_size: int) -> None:
        """
        Run a dummy forward pass to precompute the fixed backbone positional
        encodings needed at tracking time.

        _cached_pos_embeds / _cached_feat_sizes — backbone spatial embeddings
            used by _prepare_memory_conditioned_features on every tracking frame.
        """
        dummy_image = torch.zeros(1, 3, img_size, img_size)
        backbone_out = sam2.forward_image(dummy_image)
        pos_encs = backbone_out["vision_pos_enc"][-sam2.num_feature_levels :]
        self._cached_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in pos_encs]
        self._cached_feat_sizes = [(x.shape[-2], x.shape[-1]) for x in pos_encs]

    def predict(self, *args: Any, **kwargs: Any) -> list[np.ndarray]:
        return self.track(*args, **kwargs)

    def track(
        self,
        frames: list[np.ndarray],
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        raw_output: bool = False,
    ) -> list[np.ndarray]:
        """
        Track an object across video frames.

        Parameters
        ----------
        frames
          List of numpy arrays (H W C x uint8) with RGB channel layout.
        point_coords
          Point coordinates on the first frame. Shape: [N, 2] in pixel coords.
        point_labels
          Point labels (1=positive, 0=negative). Shape: [N].
        raw_output
          If True, returns binary masks instead of painted frames.

        Returns
        -------
        list[np.ndarray]
          If ``raw_output`` is False, returns painted RGB frames with the tracked
          mask overlaid. Otherwise, returns binary masks for each frame.
        """
        h, w = frames[0].shape[:2]
        img_size = self.encoder_input_img_size

        # Scale coords to [0, 1] for the patched prompt encoder
        scaled_coords = point_coords.clone().float()
        if len(scaled_coords.shape) == 2:
            scaled_coords = scaled_coords.unsqueeze(0)
        if len(point_labels.shape) == 1:
            point_labels = point_labels.unsqueeze(0)
        scaled_coords[..., 0] = scaled_coords[..., 0] / w
        scaled_coords[..., 1] = scaled_coords[..., 1] / h

        output_dict: dict[str, dict[int, Any]] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }

        painted_frames = []
        out_masks = []

        for frame_idx, frame_np in enumerate(frames):
            # Resize to encoder input size then convert to NCHW float [0, 1].
            pil_img = (
                Image.fromarray(frame_np).convert("RGB").resize((img_size, img_size))
            )
            _, frame_tensor = app_to_net_image_inputs(np.array(pil_img))  # [1, C, H, W]

            if frame_idx == 0:
                # Conditioning frame: encoder returns pix_feat (image_embeddings
                # + no_mem_embed) directly — no extra backbone pass needed.
                raw_embeddings, hr1, hr2, sparse, pix_feat = self.encoder(
                    frame_tensor,
                    scaled_coords,
                    point_labels,
                )
                is_mask_from_pts = True
            else:
                # Tracking frame: no no_mem_embed.
                # labels=-1 is the model's "no prompt" signal — the prompt encoder
                # produces a learned null embedding, so the decoder relies on memory.
                num_points = (
                    point_coords.shape[-2]
                    if len(point_coords.shape) == 3
                    else point_coords.shape[0]
                )
                raw_embeddings, hr1, hr2, sparse, _ = self.encoder(
                    frame_tensor,
                    torch.zeros(1, num_points, 2),
                    torch.full((1, num_points), -1.0),
                )
                is_mask_from_pts = False

                B, C, _, _ = raw_embeddings.shape
                vision_feat_flat = raw_embeddings.view(B, C, -1).permute(2, 0, 1)

                pix_feat = self.sam2._prepare_memory_conditioned_features(
                    frame_idx=frame_idx,
                    is_init_cond_frame=False,
                    current_vision_feats=[vision_feat_flat],
                    current_vision_pos_embeds=self._cached_pos_embeds[-1:],
                    feat_sizes=self._cached_feat_sizes[-1:],
                    output_dict=output_dict,
                    num_frames=len(frames),
                    track_in_reverse=False,
                )

            # Decoder
            low_res_masks, _, obj_ptr, object_score_logits = self.video_decoder(
                pix_feat, hr1, hr2, sparse
            )

            # Apply object-score gating on CPU (moved out of the exported decoder
            # to avoid a quantization cliff when object_score_logits is near zero).
            if self.sam2.pred_obj_scores:
                is_obj_appearing = object_score_logits > 0
                # Mask gating: absent object → -1024 sentinel
                low_res_masks = torch.where(
                    is_obj_appearing[:, :, None, None],
                    low_res_masks,
                    torch.tensor(-1024.0, device=low_res_masks.device),
                )
                # obj_ptr blending
                if self.sam2.soft_no_obj_ptr:
                    lambda_is_obj = object_score_logits.sigmoid()
                else:
                    lambda_is_obj = is_obj_appearing.float()
                if self.sam2.fixed_no_obj_ptr:
                    obj_ptr = lambda_is_obj * obj_ptr
                obj_ptr = obj_ptr + (1 - lambda_is_obj) * self.sam2.no_obj_ptr

            # Upscale masks for memory encoder input
            high_res_masks = F.interpolate(
                low_res_masks.float(),
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )

            # Conditioning: binarize. Tracking: sigmoid to get [0,1] probs.
            if is_mask_from_pts:
                mask_for_mem = (high_res_masks > 0).float()
            else:
                mask_for_mem = torch.sigmoid(high_res_masks)

            # Memory encoder: use clean backbone features (no no_mem_embed)
            maskmem_features = self.memory_encoder(raw_embeddings, mask_for_mem)

            # Store in output_dict for memory attention
            current_out = {
                "maskmem_features": maskmem_features.to(torch.bfloat16)
                if maskmem_features is not None
                else None,
                "maskmem_pos_enc": [self.maskmem_pos_enc],
                "obj_ptr": obj_ptr,
                "object_score_logits": object_score_logits,
            }
            if frame_idx == 0:
                output_dict["cond_frame_outputs"][frame_idx] = current_out
            else:
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out
                # Prune the non-conditioning memory bank to a sliding window of
                # (num_maskmem - 1) frames so memory usage stays bounded for long
                # videos. The SAM2 attention code only ever reads the most recent
                # (num_maskmem - 1) * memory_temporal_stride_for_eval frames, so
                # anything older can be safely discarded.
                stride = self.sam2.memory_temporal_stride_for_eval
                max_non_cond = (self.sam2.num_maskmem - 1) * stride
                non_cond = output_dict["non_cond_frame_outputs"]
                if len(non_cond) > max_non_cond:
                    oldest = min(non_cond)
                    del non_cond[oldest]

            # Post-process already-upscaled mask to original frame resolution
            upscaled_mask = mask_postprocessing(high_res_masks, (h, w))
            binary_mask = (
                (upscaled_mask > self.mask_threshold)
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            out_masks.append(binary_mask)

            if not raw_output:
                # Only overlay the foreground (mask==1) pixels; leave background unchanged.
                # Using Image.blend with a black background color map darkens the entire
                # image. Instead, composite the color only where the mask is active.
                frame_rgba = np.array(Image.fromarray(frame_np).convert("RGBA"))
                if binary_mask.max() > 0:
                    color_map = create_color_map(binary_mask.max() + 1)
                    overlay_color = color_map[1]  # foreground color (RGB)
                    overlay = np.zeros_like(frame_rgba)
                    overlay[..., :3] = overlay_color
                    overlay[..., 3] = (binary_mask * 128).astype(
                        np.uint8
                    )  # 50% alpha on mask
                    frame_rgba = np.array(
                        Image.alpha_composite(
                            Image.fromarray(frame_rgba),
                            Image.fromarray(overlay),
                        )
                    )
                painted_frames.append(
                    np.asarray(Image.fromarray(frame_rgba).convert("RGB"))
                )

        if raw_output:
            return out_masks
        return painted_frames

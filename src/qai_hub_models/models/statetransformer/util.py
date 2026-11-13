# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os

import cv2
import numpy as np
import torch

from qai_hub_models.models.templates.corevision.draw import create_color_map


def save_raster(
    inputs: dict[str, torch.Tensor],
    sample_index: int,
    file_index: int = 0,
    prediction_trajectory: np.ndarray | torch.Tensor | None = None,
    path_to_save: str | None = None,
    high_scale: float = 4.0,
    low_scale: float = 0.77,
    prediction_key_point: np.ndarray | torch.Tensor | None = None,
    prediction_key_point_by_gen: np.ndarray | torch.Tensor | None = None,
    prediction_trajectory_by_gen: np.ndarray | torch.Tensor | None = None,
) -> dict[str, np.ndarray] | None:
    """
    Generate and optionally save rasterized bird's-eye-view (BEV) visualizations.

    This function processes rasterized representation tensors (road layout, route,
    agents, traffic signals, etc.) and overlays model context actions, ground truth
    trajectory labels, predicted trajectories, and predicted keypoints.
    It produces both high- and low-resolution raster images. If `path_to_save` is
    provided, the images will be written to disk; otherwise, the rendered rasters are returned.

    Parameters
    ----------
    inputs
        A dictionary containing raster tensors and metadata. Expected keys:
        - `"high_res_raster"` : Tensor of shape `(C, H, W)`
        - `"low_res_raster"` : Tensor of shape `(C, H, W)`
        - `"context_actions"` : Tensor of shape `(T, 2)` with past ego-agent positions
        - `"trajectory_label"` : (Optional) ground truth future trajectory
    sample_index
        Index of the sample within the batch to visualize.
    file_index
        File numbering index used when saving multiple visualizations.
    prediction_trajectory
        Predicted trajectory coordinates with shape `(N, 2)` in meters.
    path_to_save
        Directory path to save output rasters. If None, no saving occurs and output is returned instead.
    high_scale
        Scaling factor applied when converting coordinate values to pixel locations for high-resolution raster.
    low_scale
        Scaling factor for low-resolution raster coordinates.
    prediction_key_point
        Prediction keypoints used for visual debugging of intermediate model states.
    prediction_key_point_by_gen
        per-generation keypoint predictions plotted in a separate color.
    prediction_trajectory_by_gen
        per-generation predicted trajectories used for sequence decoding visualization.

    Returns
    -------
    result : dict[str, np.ndarray] | None
        If `path_to_save` is None, returns a dictionary:
            `{"high_res_raster": ndarray, "low_res_raster": ndarray}`
        Otherwise, writes files to disk and returns None.

    """
    # save rasters
    image_to_save = {}
    past_frames_num = inputs["context_actions"][sample_index].shape[0]
    agent_type_num = 8
    for each_key in ["high_res_raster", "low_res_raster"]:
        """
        # channels:
        # 0: route raster
        # 1-20: road raster
        # 21-24: traffic raster
        # 25-56: agent raster (32=8 (agent_types) * 4 (sample_frames_in_past))
        """
        each_img = inputs[each_key][sample_index]
        if isinstance(each_img, torch.Tensor):
            each_img_np = each_img.cpu().numpy()
        goal = each_img_np[:, :, 0]
        road = each_img_np[:, :, :21]
        traffic_lights = each_img_np[:, :, 21:25]
        agent = each_img_np[:, :, 25:]
        # generate a color pallet of 20 in RGB space
        color_pallet = create_color_map(21)
        target_image = np.zeros(
            [each_img_np.shape[0], each_img_np.shape[1], 3], dtype=float
        )

        for i in range(21):
            if i in [0, 11]:
                continue
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        for i in [0, 11]:
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        for i in range(3):
            traffic_light_per_channel = traffic_lights[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(traffic_light_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][traffic_light_per_channel == 1] = (
                        color_pallet[i, k]
                    )
        target_image[:, :, 0][goal == 1] = 255
        # generate 9 values interpolated from 0 to 1
        agent_colors = np.array(
            [
                [0.01 * 255] * past_frames_num,
                np.linspace(0, 255, past_frames_num),
                np.linspace(255, 0, past_frames_num),
            ]
        ).transpose()
        for i in range(past_frames_num):
            for j in range(agent_type_num):
                agent_per_channel = agent[:, :, j * past_frames_num + i].copy()
                # agent_per_channel = agent_per_channel[:, :, None].repeat(3, axis=2)
                if np.sum(agent_per_channel) > 0:
                    for k in range(3):
                        target_image[:, :, k][agent_per_channel == 1] = agent_colors[
                            i, k
                        ]
        if "high" in each_key:
            scale = high_scale
        elif "low" in each_key:
            scale = 300
            # scale = low_scale
        # draw context actions, and trajectory label
        for each_traj_key in ["context_actions", "trajectory_label"]:
            if each_traj_key not in inputs:
                continue
            pts: np.ndarray | torch.Tensor = inputs[each_traj_key][sample_index]
            if isinstance(inputs[each_traj_key], torch.Tensor):
                pts = inputs[each_traj_key][sample_index].cpu().numpy()

            for i in range(pts.shape[0]):
                x = int(pts[i, 0] * scale) + target_image.shape[0] // 2
                y = int(pts[i, 1] * scale) + target_image.shape[1] // 2
                if 0 < x < target_image.shape[0] and 0 < y < target_image.shape[1]:
                    if "actions" in each_traj_key:
                        target_image[x, y, :] = [255, 0, 255]
                    elif "label" in each_traj_key:
                        target_image[x, y, :] = [255, 255, 0]

        tray_point_size = max(2, int(0.75 * scale * 4 / 7 / 20))
        key_point_size = max(2, int(3 * scale * 4 / 7))
        # draw prediction trajectory
        if prediction_trajectory is not None:
            for i in range(prediction_trajectory.shape[0]):
                x = (
                    int(prediction_trajectory[i, 0] * scale)
                    + target_image.shape[0] // 2
                )
                y = (
                    int(prediction_trajectory[i, 1] * scale)
                    + target_image.shape[1] // 2
                )
                if 0 < x < target_image.shape[0] and 0 < y < target_image.shape[1]:
                    target_image[
                        x - tray_point_size : x + tray_point_size,
                        y - tray_point_size : y + tray_point_size,
                        1:,
                    ] += 200

            x = int(0 * scale) + target_image.shape[0] // 2
            y = int(0 * scale) + target_image.shape[1] // 2
            if 0 < x < target_image.shape[0] and 0 < y < target_image.shape[1]:
                target_image[
                    x - tray_point_size : x + tray_point_size,
                    y - tray_point_size : y + tray_point_size,
                    2,
                ] += 200

        # draw prediction trajectory by generation
        if prediction_trajectory_by_gen is not None:
            for i in range(prediction_trajectory_by_gen.shape[0]):
                x = (
                    int(prediction_trajectory_by_gen[i, 0] * scale)
                    + target_image.shape[0] // 2
                )
                y = (
                    int(prediction_trajectory_by_gen[i, 1] * scale)
                    + target_image.shape[1] // 2
                )
                if 0 < x < target_image.shape[0] and 0 < y < target_image.shape[1]:
                    target_image[
                        x - tray_point_size : x + tray_point_size,
                        y - tray_point_size : y + tray_point_size,
                        :2,
                    ] += 100

        # draw key points
        if prediction_key_point is not None:
            for i in range(prediction_key_point.shape[0]):
                x = int(prediction_key_point[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_key_point[i, 1] * scale) + target_image.shape[1] // 2
                if 0 < x < target_image.shape[0] and 0 < y < target_image.shape[1]:
                    target_image[
                        x - key_point_size : x + key_point_size,
                        y - key_point_size : y + key_point_size,
                        1,
                    ] += 200

        # draw prediction key points during generation
        if prediction_key_point_by_gen is not None:
            for i in range(prediction_key_point_by_gen.shape[0]):
                x = (
                    int(prediction_key_point_by_gen[i, 0] * scale)
                    + target_image.shape[0] // 2
                )
                y = (
                    int(prediction_key_point_by_gen[i, 1] * scale)
                    + target_image.shape[1] // 2
                )
                if 0 < x < target_image.shape[0] and 0 < y < target_image.shape[1]:
                    target_image[
                        x - key_point_size : x + key_point_size,
                        y - key_point_size : y + key_point_size,
                        2,
                    ] += 200

        target_image = np.clip(target_image, 0, 255)
        image_to_save[each_key] = target_image
    if path_to_save is not None:
        for each_key, img in image_to_save.items():
            cv2.imwrite(
                os.path.join(
                    path_to_save,
                    "test"
                    "_"
                    + str(file_index)
                    + "_"
                    + str(sample_index)
                    + "_"
                    + str(each_key)
                    + ".png",
                ),
                img,
            )
    else:
        return image_to_save

    print(
        "length of action and labels: ", inputs["context_actions"][sample_index].shape
    )
    print("debug images saved to: ", path_to_save, file_index)
    return None

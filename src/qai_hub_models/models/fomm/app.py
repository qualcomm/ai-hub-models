# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from scipy.spatial import ConvexHull
from skimage.transform import resize

from qai_hub_models.models.fomm.model import FOMM
from qai_hub_models.models.templates.corevision.image_processing import pad_to_square

EXPECTED_SIZE = 256


def resize_frame(
    frame: np.ndarray, size: tuple[int, int] = (EXPECTED_SIZE, EXPECTED_SIZE)
) -> np.ndarray:
    """Resize helper that keeps inputs in [0,1] regardless of original scale."""
    frame = frame.astype(np.float32)
    if frame.max() > 1.0:
        frame /= 255.0
    return resize(pad_to_square(frame, pad_value=1.0), size)[..., :3]


# This comes from https://github.com/AliaksandrSiarohin/first-order-model/blob/master/animate.py
# We copy it here to use between the keypoint detector and the generator,
# so it is not baked into the models by tracing.
# The only modification is instead of inputs being dictionaries with keys "value" and "jacobian",
# these are split into separate variables
def normalize_kp(
    kp_source_value: torch.Tensor,
    kp_source_jacobian: torch.Tensor,
    kp_driving_value: torch.Tensor,
    kp_driving_jacobian: torch.Tensor,
    kp_driving_initial_value: torch.Tensor,
    kp_driving_initial_jacobian: torch.Tensor,
    adapt_movement_scale: bool = False,
    use_relative_movement: bool = False,
    use_relative_jacobian: bool = False,
) -> dict[str, torch.Tensor]:
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source_value[0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial_value[0].data.cpu().numpy()).volume
        adapt_movement_scale_num = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale_num = 1

    kp_new = {"value": kp_driving_value, "jacobian": kp_driving_jacobian}

    if use_relative_movement:
        kp_value_diff = kp_driving_value - kp_driving_initial_value
        kp_value_diff *= adapt_movement_scale_num
        kp_new["value"] = kp_value_diff + kp_source_value

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving_jacobian, torch.inverse(kp_driving_initial_jacobian)
            )
            kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source_jacobian)

    return kp_new


def load_video_frames(
    video_filepath: str | Path,
    sample_rate: int = 1,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """
    Loads video frames from a .mp4 file.

    Parameters
    ----------
    video_filepath
        Filepath to the .mp4 file.
    sample_rate
        Only return one every {sample_rate} frames
    max_frames
        Maximum number of frames to return (default: None, all frames).

    Returns
    -------
    frames : list[np.ndarray]
        List of frames as numpy arrays.
    """
    try:
        reader = imageio.get_reader(video_filepath)
    except RuntimeError as e:
        if "ffmpeg" in str(e):
            raise ValueError(
                "This demo requires the ffmpeg package to be installed on the system. "
                "For example, on a mac, you can run `brew install ffmpeg`."
            ) from None
        raise

    # Drop any misbehaving frames
    driving_video_frames = []
    i = 0
    try:
        for im in reader.iter_data():
            curr_frame = resize_frame(im)
            if i % sample_rate == 0:
                driving_video_frames.append(curr_frame)
                if max_frames is not None and len(driving_video_frames) >= max_frames:
                    break
            i += 1
    except RuntimeError:
        pass
    reader.close()
    return driving_video_frames


class FOMMApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FOMM.

    The app uses 1 model:
        * FOMM

    For a given source image and driving video input, the app will:
        * detect keypoints from each frame of the driving video
        * detect keypoints from the source image
        * generate new frames with the source image performing the motion of the driving video
    """

    def __init__(self, model: FOMM) -> None:
        self.kp_detector = model.detector
        self.generator = model.generator

    def predict(self, *args: Any, **kwargs: Any) -> list[np.ndarray]:
        # See copy_motion_to_video.
        return self.copy_motion_to_video(*args, **kwargs)

    def copy_motion_to_video(
        self,
        source_image: np.ndarray,
        driving_video: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Given a source image and a driving video, returns a new video of the
        source image mirroring the movements in the driving video.

        Parameters
        ----------
        source_image
            Numpy array (H W C x float[0 - 1], RGB).
        driving_video
            List of numpy array frames of the target animation.
            Each array is shape (C, H, W). float[0 - 1], RGB.

        Returns
        -------
        predicted_images : list[np.ndarray]
            A list of predicted images. Each image is a numpy array
            (H W C x float32 [0, 1]).
        """
        # open source image
        source_image = resize_frame(source_image)
        source_frame = torch.tensor(
            source_image[np.newaxis].astype(np.float32)
        ).permute(0, 3, 1, 2)

        driving_tensors = torch.tensor(
            np.array(driving_video)[np.newaxis].astype(np.float32)
        ).permute(0, 4, 1, 2, 3)

        # Get keypoints in source image
        source_kp_v, source_kp_j = self.kp_detector(source_frame)

        # get keypoints in initial target frame
        driving_kp_initial_v, driving_kp_initial_j = self.kp_detector(
            driving_tensors[:, :, 0]
        )
        predictions = []
        for frame_idx in range(driving_tensors.shape[2]):
            driving_frame = driving_tensors[:, :, frame_idx]
            # get keypoints for each frame of driving video
            driving_kp_v, driving_kp_j = self.kp_detector(driving_frame)

            # normalize the keypoints
            kp_norm = normalize_kp(
                kp_source_value=source_kp_v,
                kp_source_jacobian=source_kp_j,
                kp_driving_value=driving_kp_v,
                kp_driving_jacobian=driving_kp_j,
                kp_driving_initial_value=driving_kp_initial_v,
                kp_driving_initial_jacobian=driving_kp_initial_j,
                use_relative_movement=True,
                use_relative_jacobian=True,
                adapt_movement_scale=True,
            )
            # predict new frame
            out = self.generator(
                source_frame,
                source_kp_v,
                source_kp_j,
                kp_norm["value"],
                kp_norm["jacobian"],
            )

            predictions.append(np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])[0])

        return predictions

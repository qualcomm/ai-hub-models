# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import torch

from qai_hub_models.datasets.kinetics400 import (
    DEFAULT_NUM_CLIPS,
    DEFAULT_NUM_CROPS,
    multi_crop,
    preprocess_video_kinetics_400,
    read_video_at_fps,
    read_video_per_second,
    sample_clips,
    sample_video,
)
from qai_hub_models.models.templates.video_classifier.model import KineticsClassifier
from qai_hub_models.utils.labels import get_class_names


def recognize_action_kinetics_400(prediction: torch.Tensor) -> list[str]:
    """
    Return the top 5 class names.

    Parameters
    ----------
    prediction
        Get the probability for all classes.

    Returns
    -------
    class_names : list[str]
        List of class ids from Kinetics-400 dataset is returned.
    """
    # Get top 5 class probabilities
    prediction = torch.topk(prediction.flatten(), 5).indices

    actions = get_class_names("kinetics400")
    return [actions[pred] for pred in prediction]


class KineticsClassifierApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with an KineticsClassifier.

    For a given video input, the app will:
        * Pre-process the video into multiple views (temporal clips x crops)
        * Run single-clip Video Classification on each view
        * Sum the per-view softmax scores and return the top 5 class names.
    """

    def __init__(
        self,
        model: KineticsClassifier,
        num_frames: int = 16,
        num_clips: int = DEFAULT_NUM_CLIPS,
        num_crops: int = DEFAULT_NUM_CROPS,
        frame_sample_rate: int = 1,
    ) -> None:
        self.model = model
        self.num_frames = num_frames
        self.video_dim: int = model.get_input_spec()["video"][0][-1]
        self.num_clips = num_clips
        self.num_crops = num_crops
        self.frame_sample_rate = frame_sample_rate

    def preprocess_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """Apply model-specific normalisation / resize to a single clip."""
        return preprocess_video_kinetics_400(clip)

    def predict(self, path: str | Path) -> list[str]:
        """
        Run each view (temporal clip x spatial crop) as an independent
        single-clip inference, sum the per-view softmax scores, and return
        the top 5 Kinetics class names.

        Parameters
        ----------
        path
            Path to the raw video.

        Returns
        -------
        predicted_classes : list[str]
            Top 5 most probable classes for a given video.
        """
        if self.video_dim == 112:
            raw_video = read_video_at_fps(str(path), target_fps=15)
        else:
            raw_video = read_video_per_second(str(path))

        # VideoMAE strides the video before windowing; torchvision uses stride 1.
        if self.frame_sample_rate > 1:
            raw_video = raw_video[:: self.frame_sample_rate]

        if self.num_clips > 1:
            all_clips = sample_clips(raw_video, self.num_frames, self.num_clips)
        else:
            all_clips = [sample_video(raw_video, self.num_frames)]

        views: list[torch.Tensor] = []
        for clip in all_clips:
            preprocessed = self.preprocess_clip(clip)
            if self.num_crops > 1:
                views.extend(multi_crop(preprocessed, self.video_dim, self.num_crops))
            else:
                views.append(preprocessed)

        # Sum per-view softmax probabilities (the model applies softmax).
        score_sum = self.model(views[0].unsqueeze(0))
        for view in views[1:]:
            score_sum = score_sum + self.model(view.unsqueeze(0))

        return recognize_action_kinetics_400(score_sum)

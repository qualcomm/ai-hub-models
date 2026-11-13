# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch
from torch_audioset.yamnet.model import YAMNet
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models.yamnet.dataset import AudioSetDataset
from qai_hub_models.models.yamnet.evaluator import AudioSetOutputEvaluator
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, IoType, OutputSpec, TensorSpec
from qai_hub_models.utils.labels import write_labels_file

YAMNET_PROXY_REPOSITORY = "https://github.com/w-hc/torch_audioset.git"
YAMNET_PROXY_REPO_COMMIT = "e8852c5"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "yamnet.pth"

INPUT_AUDIO_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "speech_whistling2.wav"
)

# The number of Mel features per audio context
N_MELS = 64
# Audio length per MEL feature
MELS_AUDIO_LEN = 96


class YamNet(BaseModel):
    """Defines the YAMNet waveform-to-class-scores model."""

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> Self:
        model = _load_yamnet_source_model_from_weights(weights_path)
        return cls(model)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Run Yamnet  on audio, and produce class probabilities

        Parameters
        ----------
        audio
            preprocessed 1x1x96x64 tensor(log mel spectrogram patches of a 1-D waveform)

        Returns
        -------
        class_scores : torch.Tensor
            Scores is a matrix of (time_frames, num_classes) classifier scores, class_scores of shape (1,521)
        """
        return self.model(audio)

    def get_input_spec(self) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "audio": TensorSpec(
                shape=(1, 1, MELS_AUDIO_LEN, N_MELS),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        from qai_hub_models.models.yamnet.app import (
            load_audiofile,
            preprocessing_yamnet_from_source,
        )

        input_tensor_np = load_audiofile(path=str(INPUT_AUDIO_ADDRESS.fetch()))[0]
        input_tensor = torch.tensor(input_tensor_np)
        input_tensor, _ = preprocessing_yamnet_from_source(input_tensor)
        preprocessed_tensor = input_tensor[0:1, :, :, :]
        return {"audio": [preprocessed_tensor.numpy()]}

    def get_output_spec(self) -> OutputSpec:
        return {
            "class_scores": TensorSpec(),
        }

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [AudioSetDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return AudioSetDataset

    def get_evaluator(self) -> BaseEvaluator:
        return AudioSetOutputEvaluator()

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_labels_file("yamnet", output_dir, metadata)


def _load_yamnet_source_model_from_weights(
    weights_path_yamnet: str | None = None,
) -> torch.nn.Module:
    # Load yamnet model from the source repository using the given weights.
    if not weights_path_yamnet:
        weights_path_yamnet = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        ).fetch()

    # download the weights file

    model = YAMNet()
    pretrained_dict = torch.load(
        str(weights_path_yamnet),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.load_state_dict(pretrained_dict)
    model.to("cpu").eval()
    return model

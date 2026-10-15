# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from pathlib import Path

import pytest

from qai_hub_models.utils.checkpoint import CheckpointType


@pytest.mark.parametrize("subfolder", ["", "component"])
@pytest.mark.parametrize(
    ("filenames", "expected"),
    [
        ([], CheckpointType.INVALID),
        (["model.encodings"], CheckpointType.INVALID),
        (["model.encodings", "model.pt"], CheckpointType.TORCH_STATE_DICT),
        (["model.encodings", "model.pth"], CheckpointType.TORCH_STATE_DICT),
        (["model.encodings", "model.onnx"], CheckpointType.AIMET_ONNX_EXPORT),
        (["model.encodings", "model_part1.onnx"], CheckpointType.AIMET_ONNX_EXPORT),
        (["model.onnx"], CheckpointType.INVALID),
        (["model.encodings", "unrelated.onnx"], CheckpointType.INVALID),
    ],
)
def test_from_checkpoint_local_directory(
    tmp_path: Path,
    subfolder: str,
    filenames: list[str],
    expected: CheckpointType,
) -> None:
    checkpoint_dir = tmp_path / subfolder
    checkpoint_dir.mkdir(exist_ok=True)
    for filename in filenames:
        (checkpoint_dir / filename).touch()

    assert CheckpointType.from_checkpoint(tmp_path, subfolder=subfolder) is expected


@pytest.mark.parametrize("has_torch_checkpoint", [False, True])
def test_from_checkpoint_ignores_onnx_directory(
    tmp_path: Path, has_torch_checkpoint: bool
) -> None:
    (tmp_path / "model.encodings").touch()
    (tmp_path / "model.onnx").mkdir()
    if has_torch_checkpoint:
        (tmp_path / "model.pt").touch()

    expected = (
        CheckpointType.TORCH_STATE_DICT
        if has_torch_checkpoint
        else CheckpointType.INVALID
    )
    assert CheckpointType.from_checkpoint(tmp_path) is expected

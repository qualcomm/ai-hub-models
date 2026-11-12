# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from qai_hub_models.datasets.kitti_lidar_place_recognition.kitti_lidar_place_recognition import (
    lidar_scan_to_range_image,
)
from qai_hub_models.datasets.nclt_lidar_place_recognition.nclt_lidar_place_recognition import (
    fill_empty_rows,
    nearest_positions,
    read_velodyne_sync_bin,
)
from qai_hub_models.models.overlap_transformer.app import OverlapTransformerApp
from qai_hub_models.models.overlap_transformer.demo import (
    QUERY_SCAN,
    REFERENCE_SCAN_DISTANT,
    REFERENCE_SCAN_REVISIT,
    load_scan,
)
from qai_hub_models.models.overlap_transformer.demo import main as demo_main
from qai_hub_models.models.overlap_transformer.evaluator import (
    LidarPlaceRecognitionEvaluator,
)
from qai_hub_models.models.overlap_transformer.external_repos.overlap_transformer.tools.utils.utils import (
    range_projection,
)
from qai_hub_models.models.overlap_transformer.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OverlapTransformer,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.test_helpers import assert_most_close

# Revisit labels come from NCLT's groundtruth.csv (query<->revisit 2.99m apart,
# query<->distant 336.2m); retrieval accuracy itself is tested separately below.


def test_task() -> None:
    model = OverlapTransformer.from_pretrained()
    input_spec = model.get_input_spec()
    app = OverlapTransformerApp(model, input_spec)

    query = load_scan(QUERY_SCAN)
    ref_revisit = load_scan(REFERENCE_SCAN_REVISIT)
    ref_distant = load_scan(REFERENCE_SCAN_DISTANT)
    descriptors = app.predict(query, ref_revisit, ref_distant)

    assert descriptors.shape == (3, 256)
    norms = np.linalg.norm(descriptors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    # Query is a closer descriptor match to the true revisit than to the
    # distant scan -- real signal, not just an I/O smoke test.
    des_query, des_revisit, des_distant = descriptors
    assert np.linalg.norm(des_query - des_revisit) < np.linalg.norm(
        des_query - des_distant
    )

    expected_descriptors = load_numpy(
        CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "expected_descriptors.npy"
        )
    )
    assert_most_close(
        descriptors, expected_descriptors, diff_tol=0.0, rtol=0.01, atol=1e-4
    )


def test_demo() -> None:
    demo_main(is_test=True)


def test_lidar_place_recognition_evaluator() -> None:
    # A0/A1 and C0/C1 are true-revisit pairs (<3m), B0 has none; descriptors are
    # rigged so C0/C1 miss (nearest-match B0 instead), giving a known 50% recall.
    positions = torch.tensor(
        [[0, 0, 0], [0, 0, 2], [50, 0, 0], [100, 0, 0], [100, 0, 1]],
        dtype=torch.float32,
    )
    frame_indices = torch.tensor([0, 300, 600, 900, 1200], dtype=torch.long)
    descriptors = torch.tensor(
        [[0, 0], [0.1, 0], [5, 5], [9, 9], [4.95, 4.95]],
        dtype=torch.float32,
    )

    evaluator = LidarPlaceRecognitionEvaluator()
    evaluator.add_batch(descriptors, (positions, frame_indices))
    assert evaluator.get_accuracy_score() == pytest.approx(50.0)


def test_lidar_scan_to_range_image_projects_known_point() -> None:
    # A point straight ahead (y=0, z=0) has pitch=0 and projects to the
    # horizontal center column, row floor((1 - 25/28) * height) = 0.
    points = np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    height, width = 8, 16

    range_image = lidar_scan_to_range_image(points, height, width)

    assert range_image.shape == (height, width)
    assert np.count_nonzero(range_image) == 1
    row, col = np.nonzero(range_image)
    assert col[0] == width // 2
    assert row[0] == 0
    assert range_image[row[0], col[0]] == pytest.approx(10.0, abs=1e-4)


def test_lidar_scan_to_range_image_matches_upstream_range_projection() -> None:
    # Fixed (seeded) point cloud, run through both our range-image rewrite and
    # upstream's own tools/utils/utils.py::range_projection -- same FOV/max_range
    # defaults, so cell assignment and nearest-point-wins collisions should match
    # exactly rather than just agreeing on cosine similarity downstream.
    rng = np.random.default_rng(0)
    n = 300
    xyz = rng.uniform(-30.0, 30.0, size=(n, 3)).astype(np.float32)
    points = np.concatenate([xyz, np.ones((n, 1), dtype=np.float32)], axis=1)
    height, width = 16, 32

    ours = lidar_scan_to_range_image(points, height, width)
    upstream_range, _, _, _ = range_projection(points, proj_H=height, proj_W=width)

    valid = upstream_range > 0
    assert valid.any()
    np.testing.assert_allclose(ours[valid], upstream_range[valid], rtol=1e-5, atol=1e-5)
    assert np.count_nonzero(ours) == int(valid.sum())


def test_read_velodyne_sync_bin_parses_known_point(tmp_path: Path) -> None:
    # x=y=z_raw=20000 decodes to 0.0m via scale=0.005, offset=-100.0;
    # intensity=7, laser_id=3 pass through unchanged.
    raw_record = struct.pack("<HHHBB", 20000, 40000, 20000, 7, 3)
    scan_path = tmp_path / "123456.bin"
    scan_path.write_bytes(raw_record)

    points = read_velodyne_sync_bin(scan_path)

    assert points.shape == (1, 5)
    np.testing.assert_allclose(points[0, :3], [0.0, 100.0, 0.0], atol=1e-4)
    assert points[0, 3] == pytest.approx(7.0)
    assert points[0, 4] == pytest.approx(3.0)


def test_nearest_positions_picks_closest_utime() -> None:
    gt_utimes = np.array([0, 100, 300, 1000], dtype=np.float64)
    gt_xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float64)
    query_utimes = np.array([5, 250, 999], dtype=np.float64)

    result = nearest_positions(query_utimes, gt_utimes, gt_xyz)

    np.testing.assert_allclose(result, [[0, 0, 0], [2, 2, 2], [3, 3, 3]])


def test_fill_empty_rows_replicates_nearest_valid_row() -> None:
    range_image = np.array(
        [[1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]], dtype=np.float32
    )

    filled = fill_empty_rows(range_image)

    np.testing.assert_allclose(filled, [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])

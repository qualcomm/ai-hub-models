# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import importlib.util
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

from qai_hub_models.extern.mmdet import patch_mmdet_no_build_deps


@contextmanager
def patch_mmpose_no_build_deps():
    """
    Make MMPose work without dependencies that require native builds.

    - Patches some build dependencies of mmpose that aren't needed by AI Hub Models if those deps aren't installed.
      xtcocotools in particular is problematic on python > 3.10

    - Patches MMCV so it can be imported without natively-built CUDA & CPU ops.
    """
    patched_coco = importlib.util.find_spec("xtcocotools") is None
    if patched_coco:
        patched_coco = True
        sys.modules["xtcocotools"] = MagicMock()
        sys.modules["xtcocotools.coco"] = MagicMock()
        sys.modules["xtcocotools.cocoeval"] = MagicMock()
        sys.modules["xtcocotools.mask"] = MagicMock()

    patched_munkres = importlib.util.find_spec("munkres") is None
    if patched_munkres:
        sys.modules["munkres"] = MagicMock()
    try:
        with patch_mmdet_no_build_deps():  # also patches mmcv
            yield
    finally:
        if patched_coco:
            del sys.modules["xtcocotools"]
            del sys.modules["xtcocotools.coco"]
            del sys.modules["xtcocotools.cocoeval"]
            del sys.modules["xtcocotools.mask"]

        if patched_munkres:
            del sys.modules["munkres"]

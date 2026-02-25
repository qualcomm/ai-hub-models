# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import importlib.util
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

from qai_hub_models.extern.mmcv import patch_mmcv_no_extensions


@contextmanager
def patch_mmdet_no_build_deps():
    """
    Make MMDet work without dependencies that require native builds.

    - Patches some build dependencies of mmdet that aren't needed by AI Hub Models if those deps aren't installed.
      pycocotools in particular is problematic on python > 3.10 and numpy 2.x

    - Patches MMCV so it can be imported without natively-built CUDA & CPU ops.
    """
    patched_coco = importlib.util.find_spec("pycocotools") is None
    if patched_coco:
        patched_coco = True
        sys.modules["pycocotools"] = MagicMock()
        sys.modules["pycocotools.coco"] = MagicMock()
        sys.modules["pycocotools.cocoeval"] = MagicMock()
        sys.modules["pycocotools.mask"] = MagicMock()

    try:
        with patch_mmcv_no_extensions():
            yield
    finally:
        if patched_coco:
            del sys.modules["pycocotools"]
            del sys.modules["pycocotools.coco"]
            del sys.modules["pycocotools.cocoeval"]
            del sys.modules["pycocotools.mask"]

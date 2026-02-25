# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import types
from contextlib import contextmanager

from qai_hub_models.utils.bounding_box_processing import nms as torchvision_nms


@contextmanager
def patch_mmcv_no_extensions():
    """
    Patch MMCV so that the native / CUDA extension loader does nothing while importing dependencies.

    This allows:
     - mmcv-lite to be used throughout the repository instead of full MMCV
     - newer torch versions to be used. MMCV is shipped with 2.4.1 compatibility by default.

    USE AT YOUR OWN RISK.
        - AI Hub Models should not depend on any native MMCV ops to trace or run models.
          If a native op is required, it should be patched with a python implementation below.

        - If native ops are 'loaded' while this patch is in effect, they will all fail when used.
    """
    from mmcv.utils import ext_loader

    load_ext = ext_loader.load_ext

    def load_ext_dummy(*args, **kwargs):
        try:
            load_ext(*args, **kwargs)
        except (ImportError, ModuleNotFoundError):
            return types.SimpleNamespace()

    ext_loader.load_ext = load_ext_dummy

    # Patch NMS, the only native op we use, to use a native Python implementation instead.
    from mmcv.ops.nms import ext_module as nms_ext_module

    if not hasattr(nms_ext_module, "nms"):
        # This only runs when there is no working 'nms' extension installed.
        def mmcv_nms(boxes, scores, iou_threshold, offset):
            assert offset == 0, "Only offset of zero is supported for NMS."
            return torchvision_nms(boxes, scores, iou_threshold)

        nms_ext_module.nms = mmcv_nms

    try:
        yield
    finally:
        ext_loader.load_ext = load_ext

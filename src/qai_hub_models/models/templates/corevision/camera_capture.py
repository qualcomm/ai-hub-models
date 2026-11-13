# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import sys
from collections.abc import Callable

import cv2
import numpy as np

ESCAPE_KEY_ID = 27


def capture_and_display_processed_frames(
    frame_processor: Callable[[np.ndarray], np.ndarray],
    window_display_name: str,
    cap_device: int = 0,
) -> None:
    """
    Capture frames from the given input camera device, run them through
    the frame processor, and display the outputs in a window with the given name.

    User should press Esc to exit.

    Parameters
    ----------
    frame_processor
        Processes frames. Input and output are numpy arrays of shape (H W C)
        with RGB channel layout and dtype uint8 / byte.
    window_display_name
        Name of the window used to display frames.
    cap_device
        Identifier for the camera to use to capture frames.
    """
    if sys.platform.startswith("win"):
        print(
            "WARNING: On windows, it make take up to a minute for camera capture to start and display on screen."
        )

    cv2.namedWindow(window_display_name)
    capture = cv2.VideoCapture(cap_device)
    if not capture.isOpened():
        raise ValueError("Unable to open video capture.")

    frame_count = 0
    has_frame, frame = capture.read()
    while has_frame:
        assert isinstance(frame, np.ndarray)

        frame_count = frame_count + 1
        # mirror frame
        frame = np.ascontiguousarray(frame[:, ::-1, ::-1])

        # process & show frame
        processed_frame = frame_processor(frame)
        cv2.imshow(window_display_name, processed_frame[:, :, ::-1])

        has_frame, frame = capture.read()

        # detect if user closes GUI window (https://stackoverflow.com/a/45564409)
        gui_close = cv2.getWindowProperty(window_display_name, cv2.WND_PROP_VISIBLE) < 1

        # or hits escape key
        key = cv2.waitKey(1)
        escape_close = key == ESCAPE_KEY_ID

        if escape_close or gui_close:
            break

    capture.release()

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL.Image import Image
from PIL.ImageShow import IPythonViewer, _viewers  # type: ignore[attr-defined]

ALWAYS_DISPLAY_VAR = "QAIHM_ALWAYS_DISPLAY_OUTPUT"


def is_running_in_notebook() -> bool:
    try:
        from IPython.core.getipython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def save_image(image: Image, base_dir: str, filename: str, desc: str) -> None:
    os.makedirs(base_dir, exist_ok=True)
    filename = os.path.join(base_dir, filename)
    image.save(filename)
    print(f"Saving {desc} to {filename}")


def is_headless() -> bool:
    if os.environ.get(ALWAYS_DISPLAY_VAR) == "1":
        return False

    return (
        os.environ.get("SSH_TTY") is not None
        or os.environ.get("SSH_CLIENT") is not None
    )


def generate_video_from_frames(
    frames: list[np.ndarray],
    output_dir: str | None,
    output_filename: str,
    fps: int = 30,
) -> None:
    """
    Generates a video from a list of frames.

    Parameters
    ----------
    frames
        The frames to include in the video.
    output_dir
        The folder path to save the generated video.
    output_filename
        The name to save the file.
    fps
        The frame rate of the output video. Defaults to 30.
    """
    height, width, _layers = frames[0].shape
    fourcc = 0x39307076  # hex code for "vp09" format

    if output_dir is None:
        output_dir = os.path.join(Path.cwd(), "build")

    output_path = os.path.join(output_dir, output_filename)
    print("Saving video to ", output_path)
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()


def display_image(image: Image, desc: str = "image") -> bool:
    """
    Attempt to display image.
    Return true if displaying was attempted without exceptions.
    """
    # Display IPython viewer first
    # Remote server notebooks will be caught here as well
    if is_running_in_notebook():
        for viewer in _viewers:
            if isinstance(viewer, IPythonViewer):
                viewer.show(image)
                return True

    try:
        if not is_headless():
            print(f"Displaying {desc}")
            image.show()
            return True
        print(
            "\nDemo image display is disabled by default for remote servers. "
            f"To override, set `{ALWAYS_DISPLAY_VAR}=1` in your environment.\n"
        )
    except Exception:
        print("Failure to display demo images displayed on screen.")
        print(
            "If you are using a notebook environment like Jupyter/Collab, please use %run -m to run the script instead of python -m."
        )
    return False


def display_or_save_image(
    image: Image,
    output_dir: str | None = None,
    filename: str = "image.png",
    desc: str = "image",
) -> bool:
    """
    If output_dir is set, save image to disk and return.
    Else try to display image.
    If displaying image fails, save to disk in a default location.

    Parameters
    ----------
    image
        PIL Image to save.
    output_dir
        If set, saves image to this directory.
    filename
        If saving to directory, the filename to use.
    desc
        Description of what the image is, used in a print statement.

    Returns
    -------
    was_displayed : bool
        True if displaying was attempted.
    """
    if output_dir is not None:
        save_image(image, output_dir, filename, desc)
        return False

    if display_image(image, desc):
        return True

    save_image(image, os.path.join(Path.cwd(), "build"), filename, desc)
    return False


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Converts a numpy array image to uint8 type. Values are clipped to the
    range [0, 255].

    Parameters
    ----------
    image
        The input image array.

    Returns
    -------
    processed_image : np.ndarray
        The processed image in uint8 format.
    """
    clipped_image = np.clip(image, 0, 1)
    return np.round(clipped_image * 255).astype(np.uint8)

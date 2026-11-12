# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Literal, TypeVar

import cv2
import numpy as np
import torch
from PIL import Image as PILImageModule
from PIL.Image import Image, Resampling
from PIL.Image import fromarray as ImageFromArray
from torch.nn.functional import interpolate, pad
from torchvision import transforms

IMAGENET_DIM = 224
IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGENET_DIM),
        transforms.ToTensor(),
    ]
)


def make_imagenet_transform(
    crop_size: int,
    resize_size: int | None = None,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = True,
) -> transforms.Compose:
    """
    Build a standard ImageNet-style transform for a given crop size.

    By default resizes to int(crop_size * 256 / 224) then center-crops to
    crop_size, matching the torchvision convention. Pass resize_size,
    interpolation, and antialias explicitly to match a specific model's
    official weights transform (e.g. efficientnet_v2_s uses resize_size=384,
    efficientnet_b4 uses BICUBIC interpolation).

    CenterCrop(crop_size) is always applied after Resize, even when
    resize_size == crop_size. This is necessary to produce square outputs
    from non-square input images: Resize scales the shorter edge to
    resize_size, leaving the longer edge larger; CenterCrop then squares
    it. This matches the behaviour of the official torchvision
    ImageClassification transform.

    Normalization (mean/std subtraction) is intentionally excluded.
    Models routing inputs through ImagenetClassifier.forward() apply
    normalization internally via normalize_image_torchvision(). Callers
    using this transform for other purposes (e.g. quantization calibration)
    must apply normalization separately.
    """
    if resize_size is None:
        resize_size = round(crop_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=interpolation, antialias=antialias
            ),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )


def app_to_net_image_inputs(
    pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
    image_layout: str = "RGB",
    to_float: bool = True,
) -> tuple[list[np.ndarray], torch.Tensor]:
    """
    Convert the provided images to application inputs.

    This does not change channel order. RGB stays RGB etc.

    Parameters
    ----------
    pixel_values_or_image
        PIL image, or list of PIL images, or
        numpy array (H W C x uint8) or (N H W C x uint8) -- both RGB or grayscale channel layout, or
        pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB or grayscale channel layout.
    image_layout
        Image layout string (e.g., "RGB").
    to_float
        Whether to denormalize images to [0,1] (fp32) or keep as uint8.

    Returns
    -------
    NHWC_int_numpy_frames : list[np.ndarray]
        List of numpy arrays, one per input image with uint8 dtype, [H W C] shape,
        and RGB or grayscale layout.
        This output is typically used for drawing/displaying images with PIL and CV2.
    NCHW_torch_frames : torch.Tensor
        Tensor of images with shape [Batch, Channels, Height, Width],
        and RGB or grayscale layout.

    Notes
    -----
    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """
    NHWC_int_numpy_frames: list[np.ndarray] = []
    NCHW_torch_frames: torch.Tensor
    if isinstance(pixel_values_or_image, Image):
        pixel_values_or_image = [pixel_values_or_image]
    if isinstance(pixel_values_or_image, list):
        frames = []
        for image in pixel_values_or_image:
            if image_layout == "BGR":
                # PIL does not support BGR natively; decode as RGB then flip channels.
                rgb = np.array(image.convert("RGB"))
                NHWC_int_numpy_frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            else:
                NHWC_int_numpy_frames.append(np.array(image.convert(image_layout)))
            frames.append(preprocess_PIL_image(image, to_float=to_float))
        NCHW_torch_frames = torch.cat(frames)
    elif isinstance(pixel_values_or_image, torch.Tensor):
        NCHW_torch_frames = pixel_values_or_image
        NHWC_int_numpy_frames.extend(
            (b_img.permute(1, 2, 0) * 255).byte().numpy()
            for b_img in pixel_values_or_image
        )
    else:
        assert isinstance(pixel_values_or_image, np.ndarray)
        NHWC_int_numpy_frames = (
            [pixel_values_or_image]
            if len(pixel_values_or_image.shape) == 3
            else list(pixel_values_or_image)
        )
        NCHW_torch_frames = numpy_image_to_torch(pixel_values_or_image)

    return NHWC_int_numpy_frames, NCHW_torch_frames


def preprocess_PIL_image(image: Image, to_float: bool = True) -> torch.Tensor:
    """Convert a PIL image into a pyTorch tensor with range [0, 1] and shape NCHW."""
    transform = transforms.PILToTensor()  # rgb image
    img = transform(image).unsqueeze(0)
    if to_float:
        return img.float() / 255.0  # int 0 - 255 to float 0.0 - 1.0
    return img


def preprocess_PIL_image_mask(image_mask: Image) -> torch.Tensor:
    """Convert a PIL mask image into a pyTorch tensor with values 0. or 1."""
    transform = transforms.PILToTensor()
    mask = transform(image_mask.convert("L"))
    mask = mask.unsqueeze(0).float()
    return (mask > 1.0) * 1.0


def numpy_image_to_torch(image: np.ndarray, to_float: bool = True) -> torch.Tensor:
    """Convert a Numpy image (dtype uint8, shape [H W C] or [N H W C]) into a pyTorch tensor with range [0, 1] and shape NCHW."""
    image_torch = torch.from_numpy(image)

    if len(image.shape) == 3:
        image_torch = image_torch.unsqueeze(0)
    image_torch = image_torch.permute(0, 3, 1, 2)
    if to_float:
        return image_torch.float() / 255.0
    return image_torch


def torch_image_to_numpy(image: torch.Tensor, to_int: bool = True) -> np.ndarray:
    """Convert a pyTorch tensor (shape NCHW) with range [0, 1] to a Numpy image (dtype uint8, shape [N H W C])."""
    assert len(image.shape) == 4
    if to_int:
        image = (image * 255.0).byte()
    return image.permute(0, 2, 3, 1).numpy().squeeze(0)


def torch_tensor_to_PIL_image(data: torch.Tensor) -> Image:
    """Convert a Torch tensor (dtype float32) with range [0, 1] and shape CHW into PIL image CHW."""
    out = torch.clip(data, min=0.0, max=1.0)
    np_out = (out.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    if np_out.shape[2] == 1:
        np_out = np_out.squeeze(2)
    return ImageFromArray(np_out)


def normalize_image_torchvision(
    image_tensor: torch.Tensor,
    image_tensor_has_batch: bool = True,
    is_video: bool = False,
) -> torch.Tensor:
    """
    Normalizes according to standard torchvision constants.

    Due to issues with FX Graph tracing in AIMET, image_tensor_has_batch is a constant passed in,
    rather than determining the image rank using len(image_tensor.shape).

    There are many PyTorch models that expect input images normalized with
    these specific constants, so this utility can be re-used across many models.

    Parameters
    ----------
    image_tensor
        Image tensor to normalize.
    image_tensor_has_batch
        Whether the tensor has a batch dimension.
    is_video
        Whether the tensor represents a video.

    Returns
    -------
    normalized_tensor : torch.Tensor
        Normalized image tensor.
    """
    shape = [-1, 1, 1]
    if image_tensor_has_batch:
        shape.insert(0, 1)
    if is_video:
        shape.append(1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(*shape)
    std = torch.Tensor([[0.229, 0.224, 0.225]]).reshape(*shape)
    return (image_tensor - mean) / std


def normalize_image_transform() -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a torchvision transform that returns a torch tensor normalized according to some constants.

    There are many PyTorch models that expect input images normalized with
    these specific constants, so this utility can be re-used across many models.
    """
    return functools.partial(normalize_image_torchvision, image_tensor_has_batch=False)


def letterbox_geometry(
    orig_h: float, orig_w: float, dst_h: int, dst_w: int
) -> tuple[float, int, int, int, int]:
    """
    Compute the scale, resized size, and centered integer padding for a
    letterbox resize to (dst_h, dst_w), preserving aspect ratio.

    Uses round() for the resized size and floor division for padding
    (not math.floor()-based resize_pad), matching GroundingDINO's own
    preprocessing convention exactly so unscaling predictions stays exact.

    Parameters
    ----------
    orig_h
        Original image height.
    orig_w
        Original image width.
    dst_h
        Target (letterboxed) height.
    dst_w
        Target (letterboxed) width.

    Returns
    -------
    scale : float
        Scale factor applied to the original image.
    new_h : int
        Resized image height, before padding.
    new_w : int
        Resized image width, before padding.
    pad_top : int
        Padding added above the resized image.
    pad_left : int
        Padding added to the left of the resized image.
    """
    scale = min(dst_h / orig_h, dst_w / orig_w)
    new_h = round(orig_h * scale)
    new_w = round(orig_w * scale)
    pad_top = (dst_h - new_h) // 2
    pad_left = (dst_w - new_w) // 2
    return scale, new_h, new_w, pad_top, pad_left


def letterbox_resize_pad_pil(image: Image, dst_h: int, dst_w: int) -> Image:
    """
    Letterbox-resize `image` to (dst_h, dst_w) using letterbox_geometry's
    convention (round-based resize, floor-divided centered padding),
    zero-padding the remainder. RGB-converts first.

    A single shared implementation for any caller that must stay pixel-exact
    with letterbox_geometry's convention across both PIL and tensor outputs
    (e.g. matching inference preprocessing to calibration/eval preprocessing)
    -- callers needing a tensor can wrap this with e.g. torchvision's
    to_tensor(). Not a generic resize_pad() replacement: resize_pad() uses a
    different (floor-based) resize/padding convention.

    Parameters
    ----------
    image
        Source image, any mode/size.
    dst_h
        Target (letterboxed) height.
    dst_w
        Target (letterboxed) width.

    Returns
    -------
    canvas : Image
        RGB image of exact size (dst_w, dst_h), aspect-preserved and
        zero-padded.
    """
    img = image.convert("RGB")
    orig_w, orig_h = img.size
    _scale, new_h, new_w, pad_top, pad_left = letterbox_geometry(
        orig_h, orig_w, dst_h, dst_w
    )
    resized = img.resize((new_w, new_h), resample=Resampling.BILINEAR)
    canvas = PILImageModule.new("RGB", (dst_w, dst_h))
    canvas.paste(resized, (pad_left, pad_top))
    return canvas


def pad_to_square(frame: np.ndarray, pad_value: float = 255) -> np.ndarray:
    """
    Pad an image or video frame to square dimensions with whitespace.

    Assumes the input shape is of format (H, W, C).

    Parameters
    ----------
    frame
        Image or video frame to pad.
    pad_value
        Value to use for padding (default: 255).

    Returns
    -------
    padded_frame : np.ndarray
        Padded square frame.
    """
    h, w, _ = frame.shape
    if h < w:
        top_pad = (w - h) // 2
        pad_values = ((top_pad, w - h - top_pad), (0, 0), (0, 0))
    else:
        top_pad = (h - w) // 2
        pad_values = ((0, 0), (top_pad, h - w - top_pad), (0, 0))
    return np.pad(frame, pad_values, constant_values=pad_value)


def resize_pad(
    image: torch.Tensor,
    dst_size: tuple[int, ...],
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    vertical_float: Literal["center", "top", "bottom"] = "center",
    horizontal_float: Literal["center", "left", "right"] = "center",
) -> tuple[torch.Tensor, float, tuple[int, int]]:
    """
    Resize and pad image to be shape [..., dst_size[0], dst_size[1]].

    This will not warp or crop the image. It will be resized as large as it can
    possibly be without being cropped and while maintaining aspect ratio.
    The image is then padded so that it's in the center of the returned image "frame"
    of the desired size (dst_size).

    Parameters
    ----------
    image
        Image to reshape with shape (..., H, W).
    dst_size
        Size to which the image should be reshaped (height, width).
    pad_mode
        Padding mode.
    pad_value
        Padding value.
    vertical_float
        Where the image should float vertically in the resulting canvas.
        For example, if "float" is bottom, the image will be written
        in the bottom of the frame, and the remainder of the frame
        (above the image) is padded.
    horizontal_float
        Where the image should float horizontally in the resulting canvas.
        For example, if "float" is left, the image will be written
        in the left of the frame, and the remainder of the frame
        (to the right of the image) is padded.

    Returns
    -------
    rescaled_padded_image : torch.Tensor
        Tensor with shape (..., dst_size[0], dst_size[1]).
    scale : float
        Scale factor between original image and dst_size image.
    pad : tuple[int, int]
        Pixels of padding added to the rescaled image (left_padding, top_padding).

    Notes
    -----
    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """
    height, width = image.shape[-2:]
    dst_frame_height, dst_frame_width = dst_size

    h_ratio = dst_frame_height / height
    w_ratio = dst_frame_width / width
    scale = min(h_ratio, w_ratio)
    if h_ratio < w_ratio:
        scale = h_ratio
        new_height = dst_frame_height
        new_width = math.floor(width * scale)
    else:
        scale = w_ratio
        new_height = math.floor(height * scale)
        new_width = dst_frame_width

    new_height = math.floor(height * scale)
    new_width = math.floor(width * scale)

    def _split_padding(
        float_img_in_frame: Literal["center", "left", "right", "top", "bottom"],
        pad_size: int,
    ) -> tuple[int, int]:
        """
        Split padding into (Left, Rright), or (Top, Bottom)
        splits based on where the image should float in the frame.
        """
        if float_img_in_frame == "center":
            return (int(pad_size // 2), int(pad_size // 2 + pad_size % 2))
        if float_img_in_frame in ["right", "bottom"]:
            return (pad_size, 0)
        if float_img_in_frame in ["left", "top"]:
            return (0, pad_size)
        raise ValueError(f"Invalid pad type: {float_img_in_frame}")

    pad_top, pad_bottom = _split_padding(vertical_float, dst_frame_height - new_height)
    pad_left, pad_right = _split_padding(horizontal_float, dst_frame_width - new_width)

    rescaled_image = interpolate(
        image, size=[int(new_height), int(new_width)], mode="bilinear"
    )
    rescaled_padded_image = pad(
        rescaled_image,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode=pad_mode,
        value=pad_value,
    )
    padding = (pad_left, pad_top)

    return rescaled_padded_image, scale, padding


def resize_and_normalize(image: torch.Tensor) -> torch.Tensor:
    """Resize with padding to 224x224 and normalize to [-1, 1]."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image, _, _ = resize_pad(
        image, (224, 224), vertical_float="top", horizontal_float="left"
    )
    return image * 2.0 - 1.0


def undo_resize_pad(
    image: torch.Tensor,
    orig_size_wh: tuple[int, int],
    scale: float,
    padding: tuple[int, int],
) -> torch.Tensor:
    """
    Undos the effect of resize_pad.

    Instead of scale, the original size (in order width, height) is provided to prevent an off-by-one size.

    Parameters
    ----------
    image
        Image to undo resize and pad.
    orig_size_wh
        Original size (width, height).
    scale
        Scale factor used in resize_pad.
    padding
        Padding used in resize_pad.

    Returns
    -------
    restored_image : torch.Tensor
        Image restored to original size.
    """
    width, height = orig_size_wh

    rescaled_image = interpolate(image, scale_factor=1 / scale, mode="bilinear")

    scaled_padding = [round(padding[0] / scale), round(padding[1] / scale)]

    return rescaled_image[
        ...,
        scaled_padding[1] : scaled_padding[1] + height,
        scaled_padding[0] : scaled_padding[0] + width,
    ]


def transform_resize_pad_coordinates(
    coordinates: torch.Tensor, scale_factor: float, pad: torch.Tensor | tuple[int, int]
) -> torch.Tensor:
    """
    Transform integer (pixel space) coordinates from their location in an image to the equivalent location in a scaled / padded version of the same image.

    NOTE: x and y can be swapped (passed in order of y, x) if padding order is also swapped.

    Parameters
    ----------
    coordinates
        Coordinate tensor of shape [..., 2], where 2 == [x, y].
        Coordinates must be in pixel space.
    scale_factor
        Scale factor applied to the image.
    pad
        Padding values [padding_left, padding_top], in pixel space.

    Returns
    -------
    transformed_coordinates : torch.Tensor
        Modified coordinates in pixel space.
    """
    return coordinates * scale_factor + torch.Tensor([*pad]).int()


def transform_resize_pad_normalized_coordinates(
    coordinates: torch.Tensor,
    src_image_shape: torch.Tensor | torch.Size | tuple[int, int],
    resized_image_shape: torch.Tensor | torch.Size | tuple[int, int],
    scale_factor: float,
    pad: tuple[int, int],
) -> torch.Tensor:
    """
    Convert normalized ([0-1] float space) coordinates from their location in an image to the equivalent location in a scaled / padded version of the same image.

    NOTE: x and y can be swapped (passed in order of y, x) if shape and padding order are also swapped.

    Parameters
    ----------
    coordinates
        Coordinate tensor of shape [..., 2], where 2 == [x, y].
        Coordinates must be in normalized [0-1] float space.
    src_image_shape
        Source image shape (width, height).
    resized_image_shape
        Resized image shape (width, height).
    scale_factor
        Scale factor applied to the image.
    pad
        Padding values [padding_left, padding_top], in PIXEL space.

    Returns
    -------
    transformed_coordinates : torch.Tensor
        Modified coordinates in normalized [0-1] float space.
    """
    coordinates = coordinates * torch.Tensor([*src_image_shape]).int()
    coordinates = transform_resize_pad_coordinates(coordinates, scale_factor, pad)
    coordinates /= torch.Tensor([*resized_image_shape]).int()
    return coordinates


def pil_resize_pad(
    image: Image,
    dst_size: tuple[int, int],
    pad_mode: str = "constant",
) -> tuple[Image, float, tuple[int, int]]:
    torch_image = preprocess_PIL_image(image)
    torch_out_image, scale, padding = resize_pad(
        torch_image,
        dst_size,
        pad_mode=pad_mode,
    )
    pil_out_image = torch_tensor_to_PIL_image(torch_out_image[0])
    return (pil_out_image, scale, padding)


def pil_undo_resize_pad(
    image: Image, orig_size_wh: tuple[int, int], scale: float, padding: tuple[int, int]
) -> Image:
    torch_image = preprocess_PIL_image(image)
    torch_out_image = undo_resize_pad(torch_image, orig_size_wh, scale, padding)
    return torch_tensor_to_PIL_image(torch_out_image[0])


def denormalize_coordinates(
    coordinates: torch.Tensor,
    input_img_size: tuple[int, int],
    scale: float = 1.0,
    pad: tuple[int, int] = (0, 0),
) -> None:
    """
    Maps detection coordinates from [0,1] to coordinates in the original image.

    This function can be exported and run inside inference frameworks if desired.

    Note: If included in the model, this code is likely to be unfriendly to quantization.
          This is because of the high range and variability of the output tensor.
          For best quantization accuracy, this code should be run separately from the model,
          or the model should de-quantize activations before running these layers.

    Parameters
    ----------
    coordinates
        Coordinate tensor with shape [..., 2]. Range must be [0, 1].
    input_img_size
        The size of the tensor that was fed to the NETWORK (NOT the original image size).
        H / W is the same order as coordinates.
    scale
        Scale factor used to resize the image to be fed to the network.
    pad
        Padding used during resizing of input image to network input tensor.
        This is the absolute number of padding pixels in the network input tensor, NOT in the original image.
        H / W is in the same order as coordinates.

    Notes
    -----
    The coordinates input is modified in place.
    The output coordinates are in the format [..., 2] where 2 is (y0, x0).
    """
    img_0, img_1 = input_img_size
    pad_0, pad_1 = pad

    coordinates[..., 0] = ((coordinates[..., 0] * img_0 - pad_0) / scale).int()
    coordinates[..., 1] = ((coordinates[..., 1] * img_1 - pad_1) / scale).int()


def apply_batched_affines_to_frame(
    frame: np.ndarray, affines: list[np.ndarray], output_image_size: tuple[int, int]
) -> np.ndarray:
    """
    Generate one image per affine applied to the given frame.

    I/O is numpy since this uses cv2 APIs under the hood.

    Parameters
    ----------
    frame
        Frame on which to apply the affine. Shape is [H W C], dtype must be np.byte.
    affines
        List of 2x3 affine matrices to apply to the frame.
    output_image_size
        Size of each output frame.

    Returns
    -------
    transformed_images : np.ndarray
        Computed images with shape [B H W C].
    """
    assert (
        frame.dtype == np.byte or frame.dtype == np.uint8  # noqa: PLR1714 Using a set for comparison is not equivalent to using == on both of these individually.
    )  # cv2 does not work correctly otherwise. Don't remove this assertion.

    imgs = []
    for affine in affines:
        img = cv2.warpAffine(frame, affine, output_image_size)
        imgs.append(img)
    return np.stack(imgs)


ndarrayOrTensor = TypeVar("ndarrayOrTensor", np.ndarray, torch.Tensor)


def apply_affine_to_coordinates(
    coordinates: ndarrayOrTensor, affine: ndarrayOrTensor
) -> ndarrayOrTensor:
    """
    Apply the given affine matrix to the given coordinates.

    Parameters
    ----------
    coordinates
        Coordinates on which to apply the affine. Shape is [..., 2], where 2 == [X, Y].
    affine
        Affine matrix to apply to the coordinates.

    Returns
    -------
    ndarrayOrTensor
        Transformed coordinates with shape [..., 2], where 2 == [X, Y].
    """
    return (affine[:, :2] @ coordinates.T + affine[:, 2:]).T


def compute_vector_rotation(
    vec_start: torch.Tensor,
    vec_end: torch.Tensor,
    offset_rads: float | torch.Tensor = 0,
) -> torch.Tensor:
    """
    From the given vector, compute the rotation of the vector with added offset.

    Parameters
    ----------
    vec_start
        Starting point of the vector. Shape is [B, 2], where 2 == (x, y).
    vec_end
        Ending point of the vector. Shape is [B, 2], where 2 == (x, y).
    offset_rads
        Offset to subtract from the rotation calculation. Can be size [1] or [Batch].

    Returns
    -------
    rotation_angle : torch.Tensor
        Computed rotation angle in radians. Shape is [Batch].
    """
    return (
        torch.atan2(
            vec_start[..., 1] - vec_end[..., 1], vec_start[..., 0] - vec_end[..., 0]
        )
        - offset_rads
    )


def compute_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: int,
    output_size: tuple[int, int],
    shift: np.ndarray | None = None,
    inv: bool = False,
) -> np.ndarray:
    """
    Get the affine transformation matrix.

    Parameters
    ----------
    center
        Center coordinates, with shape of [2,].
    scale
        Scale factors, with shape of [2,].
    rot
        Rotation angle in degrees.
    output_size
        Output image size (width, height).
    shift
        Shift values [2,], defaults to [0.0, 0.0].
    inv
        If False, get affine transform for source image size to output image size,
        otherwise get affine transform for output image size to source image size.

    Returns
    -------
    affine_matrix : np.ndarray
        Affine transform for transformation, with shape (2, 3).
    """
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # get directions
    rot_rad = np.pi * rot / 180
    src_point = np.array([0, src_w * -0.5], np.float32)
    src_dir = get_dir(src_point, rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    if shift is None:
        shift = np.array([0.0, 0.0])
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        affine = cv2.getAffineTransform(dst, src)
    else:
        affine = cv2.getAffineTransform(src, dst)

    return affine


def get_dir(src_point: np.ndarray, rot_rad: float) -> np.ndarray:
    """
    Get the direction of source point based on rotation.

    Parameters
    ----------
    src_point
        The point with shape [2,].
    rot_rad
        The rotation radian.

    Returns
    -------
    rotated_point : np.ndarray
        The point after rotation with shape [2,].
    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = np.array([0.0, 0.0], np.float32)
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(point_x: np.ndarray, point_y: np.ndarray) -> np.ndarray:
    """
    Gets the 3rd point from two points.

    Parameters
    ----------
    point_x
        The point with shape (2,).
    point_y
        The point with shape (2,).

    Returns
    -------
    third_point : np.ndarray
        The third point with shape (2,).
    """
    direct = point_x - point_y
    return point_y + np.array([-direct[1], direct[0]], dtype=np.float32)


def pre_process_with_affine(
    image: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    rot: int,
    in_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Pre-processes an input image with affine transformations.

    Parameters
    ----------
    image
        The input image as a NumPy array (H, W, C) as int32 data type.
    center
        Center coordinates, with shape of [2,].
    scale
        Scale factors, with shape of [2,].
    rot
        Rotation angle in degrees.
    in_shape
        The target input shape (height, width) for the model.

    Returns
    -------
    preprocessed_image : torch.Tensor
        The pre-processed image with shape (1, C, H, W).
    """
    inp_height, inp_width = in_shape

    trans_input = compute_affine_transform(center, scale, rot, (inp_width, inp_height))
    trans_image = apply_batched_affines_to_frame(
        image,
        [trans_input],
        (inp_width, inp_height),
    )
    return numpy_image_to_torch(trans_image)


def denormalize_coordinates_affine(
    coords: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    rot: int,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    Denormalize coordinates to the original image space.

    Parameters
    ----------
    coords
        Coordinates with shape (N, 2).
    center
        Original image center used during pre-processing. Shape (2,).
    scale
        Original image scale used during pre-processing. Shape (2,).
    rot
        Rotation angle in degrees used during pre-processing.
    output_size
        The dimensions (width, height) of the model's output feature map.

    Returns
    -------
    denormalized_coords : np.ndarray
        Transformed coordinates in the original image space. Shape (N, 2).
    """
    trans = compute_affine_transform(center, scale, rot, output_size, inv=True)
    return apply_affine_to_coordinates(coords, trans)


def get_post_rot_and_tran(
    resize: float, crop: tuple, rotate: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get post rotation and post translation.

    Used to transform points based on resize, crop and rotate in the image.

    Parameters
    ----------
    resize
        The scaling factor applied to the image.
    crop
        The cropping boundaries in pixels (left, upper, right, lower).
    rotate
        The rotation angle in degrees.

    Returns
    -------
    post_rot : torch.Tensor
        Shape [3, 3]. Post rotation matrix in camera coordinate system.
    post_tran : torch.Tensor
        Shape [3,]. Post translation tensor in camera coordinate system.
    """
    post_rot = torch.eye(3)
    post_tran = torch.zeros(3)

    # post-homography transformation
    post_rot[:2, :2] *= resize
    post_tran[:2] -= torch.Tensor(crop[:2])

    rotate_angle = torch.tensor(rotate / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    A = torch.Tensor([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot[:2, :2] = A.matmul(post_rot[:2, :2])
    post_tran[:2] = A.matmul(post_tran[:2]) + b

    return post_rot, post_tran

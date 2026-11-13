# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    IMAGENET_DIM,
    IMAGENET_TRANSFORM,
    app_to_net_image_inputs,
    apply_affine_to_coordinates,
    apply_batched_affines_to_frame,
    compute_affine_transform,
    compute_vector_rotation,
    denormalize_coordinates,
    denormalize_coordinates_affine,
    get_3rd_point,
    get_dir,
    get_post_rot_and_tran,
    normalize_image_torchvision,
    normalize_image_transform,
    numpy_image_to_torch,
    pad_to_square,
    pil_resize_pad,
    pil_undo_resize_pad,
    pre_process_with_affine,
    preprocess_PIL_image,
    preprocess_PIL_image_mask,
    resize_pad,
    torch_image_to_numpy,
    torch_tensor_to_PIL_image,
    transform_resize_pad_coordinates,
    transform_resize_pad_normalized_coordinates,
    undo_resize_pad,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixture providing a seeded numpy random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


class TestImageConversions:
    """Test image conversion functions between PIL, numpy, and torch."""

    def test_preprocess_PIL_image_to_float(self) -> None:
        """Test PIL image conversion to torch tensor with float normalization."""
        # Create a simple RGB image
        img = Image.new("RGB", (100, 50), color=(255, 128, 0))
        result = preprocess_PIL_image(img, to_float=True)

        assert result.shape == (1, 3, 50, 100)  # NCHW format
        assert result.dtype == torch.float32
        assert torch.allclose(result[0, 0, 0, 0], torch.tensor(1.0))  # Red channel
        assert torch.allclose(result[0, 1, 0, 0], torch.tensor(128.0 / 255.0))  # Green
        assert torch.allclose(result[0, 2, 0, 0], torch.tensor(0.0))  # Blue

    def test_preprocess_PIL_image_no_float(self) -> None:
        """Test PIL image conversion without float normalization."""
        img = Image.new("RGB", (100, 50), color=(255, 128, 0))
        result = preprocess_PIL_image(img, to_float=False)

        assert result.shape == (1, 3, 50, 100)
        assert result.dtype == torch.uint8
        assert result[0, 0, 0, 0] == 255

    def test_preprocess_PIL_image_mask(self) -> None:
        """Test PIL mask image conversion."""
        # Create a mask with black and white regions
        img_mask = Image.new("L", (100, 50), color=0)
        result = preprocess_PIL_image_mask(img_mask)

        assert result.shape == (1, 1, 50, 100)
        assert result.dtype == torch.float32
        assert torch.all(result == 0.0)

        # Test with white mask
        img_mask_white = Image.new("L", (100, 50), color=255)
        result_white = preprocess_PIL_image_mask(img_mask_white)
        assert torch.all(result_white == 1.0)

    def test_numpy_image_to_torch_3d(self, rng: np.random.Generator) -> None:
        """Test numpy to torch conversion with 3D array (single image)."""
        # Create HWC numpy array
        img_np = rng.integers(0, 256, (50, 100, 3), dtype=np.uint8)
        result = numpy_image_to_torch(img_np, to_float=True)

        assert result.shape == (1, 3, 50, 100)  # NCHW
        assert result.dtype == torch.float32
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_numpy_image_to_torch_4d(self, rng: np.random.Generator) -> None:
        """Test numpy to torch conversion with 4D array (batched images)."""
        # Create NHWC numpy array
        img_np = rng.integers(0, 256, (2, 50, 100, 3), dtype=np.uint8)
        result = numpy_image_to_torch(img_np, to_float=True)

        assert result.shape == (2, 3, 50, 100)  # NCHW
        assert result.dtype == torch.float32

    def test_numpy_image_to_torch_no_float(self, rng: np.random.Generator) -> None:
        """Test numpy to torch conversion without float normalization."""
        img_np = rng.integers(0, 256, (50, 100, 3), dtype=np.uint8)
        result = numpy_image_to_torch(img_np, to_float=False)

        assert result.dtype == torch.uint8

    def test_torch_image_to_numpy(self) -> None:
        """Test torch to numpy conversion."""
        # Function uses squeeze(0) which requires batch size of 1
        img_torch = torch.rand(1, 3, 50, 100)
        result = torch_image_to_numpy(img_torch, to_int=True)

        # squeeze(0) removes the batch dimension when it's size 1
        assert result.shape == (50, 100, 3)
        assert result.dtype == np.uint8

    def test_torch_image_to_numpy_no_int(self) -> None:
        """Test torch to numpy conversion without int conversion."""
        # Function uses squeeze(0) which requires batch size of 1
        img_torch = torch.rand(1, 3, 50, 100)
        result = torch_image_to_numpy(img_torch, to_int=False)

        assert result.shape == (50, 100, 3)
        assert result.dtype == np.float32

    def test_torch_tensor_to_PIL_image(self) -> None:
        """Test torch tensor to PIL image conversion."""
        # Create a solid color tensor
        img_torch = torch.ones(3, 50, 100) * 0.5
        result = torch_tensor_to_PIL_image(img_torch)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 50)
        result_array = np.array(result)
        assert result_array.shape == (50, 100, 3)

    def test_torch_tensor_to_PIL_image_grayscale(self) -> None:
        """Test torch tensor to PIL image conversion for grayscale."""
        img_torch = torch.ones(1, 50, 100) * 0.5
        result = torch_tensor_to_PIL_image(img_torch)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 50)
        result_array = np.array(result)
        assert result_array.shape == (50, 100)  # Grayscale has no channel dim

    def test_torch_tensor_to_PIL_image_clipping(self) -> None:
        """Test that values are clipped to [0, 1]."""
        img_torch = torch.tensor(
            [[[1.5, -0.5]]]
        )  # 1x1x2 image with out of range values
        result = torch_tensor_to_PIL_image(img_torch)

        result_array = np.array(result)
        assert result_array.max() == 255
        assert result_array.min() == 0

    def test_app_to_net_image_inputs_single_pil(self) -> None:
        """Test app_to_net_image_inputs with single PIL image."""
        img = Image.new("RGB", (100, 50), color=(255, 0, 0))
        nhwc_list, nchw_tensor = app_to_net_image_inputs(img)

        assert len(nhwc_list) == 1
        assert nhwc_list[0].shape == (50, 100, 3)
        assert nchw_tensor.shape == (1, 3, 50, 100)

    def test_app_to_net_image_inputs_list_pil(self) -> None:
        """Test app_to_net_image_inputs with list of PIL images."""
        imgs = [
            Image.new("RGB", (100, 50), color=(255, 0, 0)),
            Image.new("RGB", (100, 50), color=(0, 255, 0)),
        ]
        nhwc_list, nchw_tensor = app_to_net_image_inputs(imgs)

        assert len(nhwc_list) == 2
        assert nchw_tensor.shape == (2, 3, 50, 100)

    def test_app_to_net_image_inputs_torch(self) -> None:
        """Test app_to_net_image_inputs with torch tensor."""
        img_torch = torch.rand(2, 3, 50, 100)
        nhwc_list, nchw_tensor = app_to_net_image_inputs(img_torch)

        assert len(nhwc_list) == 2
        assert nchw_tensor.shape == (2, 3, 50, 100)
        assert torch.equal(nchw_tensor, img_torch)

    def test_app_to_net_image_inputs_numpy_3d(self, rng: np.random.Generator) -> None:
        """Test app_to_net_image_inputs with 3D numpy array."""
        img_np = rng.integers(0, 256, (50, 100, 3), dtype=np.uint8)
        nhwc_list, nchw_tensor = app_to_net_image_inputs(img_np)

        assert len(nhwc_list) == 1
        assert nchw_tensor.shape == (1, 3, 50, 100)

    def test_app_to_net_image_inputs_numpy_4d(self, rng: np.random.Generator) -> None:
        """Test app_to_net_image_inputs with 4D numpy array."""
        img_np = rng.integers(0, 256, (2, 50, 100, 3), dtype=np.uint8)
        nhwc_list, nchw_tensor = app_to_net_image_inputs(img_np)

        assert len(nhwc_list) == 2
        assert nchw_tensor.shape == (2, 3, 50, 100)


class TestNormalization:
    """Test image normalization functions."""

    def test_normalize_image_torchvision_with_batch(self) -> None:
        """Test torchvision normalization with batch dimension."""
        img = torch.ones(2, 3, 224, 224) * 0.5
        result = normalize_image_torchvision(img, image_tensor_has_batch=True)

        assert result.shape == (2, 3, 224, 224)
        # Check that normalization was applied
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        expected = (0.5 - mean[0]) / std[0]
        assert torch.allclose(result[0, 0, 0, 0], expected, atol=1e-5)

    def test_normalize_image_torchvision_no_batch(self) -> None:
        """Test torchvision normalization without batch dimension."""
        img = torch.ones(3, 224, 224) * 0.5
        result = normalize_image_torchvision(img, image_tensor_has_batch=False)

        assert result.shape == (3, 224, 224)

    def test_normalize_image_torchvision_video(self) -> None:
        """Test torchvision normalization with video."""
        img = torch.ones(2, 3, 16, 224, 224) * 0.5  # Batch, Channels, Frames, H, W
        result = normalize_image_torchvision(
            img, image_tensor_has_batch=True, is_video=True
        )

        assert result.shape == (2, 3, 16, 224, 224)

    def test_normalize_image_transform(self) -> None:
        """Test normalize_image_transform returns a callable."""
        transform = normalize_image_transform()

        assert callable(transform)

        # Test the transform
        img = torch.ones(3, 224, 224) * 0.5
        result = transform(img)
        assert result.shape == (3, 224, 224)

    def test_imagenet_transform(self) -> None:
        """Test IMAGENET_TRANSFORM is properly configured."""
        img = Image.new("RGB", (300, 300), color=(255, 0, 0))
        result = IMAGENET_TRANSFORM(img)

        assert result.shape == (3, IMAGENET_DIM, IMAGENET_DIM)
        assert result.dtype == torch.float32


class TestPaddingAndResizing:
    """Test padding and resizing functions."""

    def test_pad_to_square_horizontal(self, rng: np.random.Generator) -> None:
        """Test padding a horizontal image to square."""
        # Create a horizontal image (wider than tall)
        img = rng.integers(0, 256, (50, 100, 3), dtype=np.uint8)
        result = pad_to_square(img)

        assert result.shape == (100, 100, 3)
        # Check padding is white (255)
        assert np.all(result[0, :, :] == 255)

    def test_pad_to_square_vertical(self, rng: np.random.Generator) -> None:
        """Test padding a vertical image to square."""
        # Create a vertical image (taller than wide)
        img = rng.integers(0, 256, (100, 50, 3), dtype=np.uint8)
        result = pad_to_square(img)

        assert result.shape == (100, 100, 3)
        assert np.all(result[:, 0, :] == 255)

    def test_pad_to_square_already_square(self, rng: np.random.Generator) -> None:
        """Test padding an already square image."""
        img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        result = pad_to_square(img)

        assert result.shape == (100, 100, 3)

    def test_resize_pad_basic(self) -> None:
        """Test basic resize_pad functionality."""
        img = torch.rand(1, 3, 100, 200)  # NCHW
        dst_size = (150, 150)

        result, scale, padding = resize_pad(img, dst_size)

        assert result.shape == (1, 3, 150, 150)
        assert isinstance(scale, float)
        assert len(padding) == 2

    def test_resize_pad_center_float(self) -> None:
        """Test resize_pad with center float (default)."""
        img = torch.rand(1, 3, 100, 200)
        dst_size = (150, 150)

        result, _scale, _padding = resize_pad(
            img, dst_size, vertical_float="center", horizontal_float="center"
        )

        assert result.shape == (1, 3, 150, 150)

    def test_resize_pad_top_left_float(self) -> None:
        """Test resize_pad with top-left float."""
        img = torch.rand(1, 3, 100, 200)
        dst_size = (150, 150)

        result, _scale, padding = resize_pad(
            img, dst_size, vertical_float="top", horizontal_float="left"
        )

        assert result.shape == (1, 3, 150, 150)
        assert padding[0] == 0  # Left padding should be 0
        assert padding[1] == 0  # Top padding should be 0

    def test_resize_pad_bottom_right_float(self) -> None:
        """Test resize_pad with bottom-right float."""
        img = torch.rand(1, 3, 100, 200)
        dst_size = (150, 150)

        result, _scale, _padding = resize_pad(
            img, dst_size, vertical_float="bottom", horizontal_float="right"
        )

        assert result.shape == (1, 3, 150, 150)

    def test_resize_pad_with_pad_value(self) -> None:
        """Test resize_pad with custom pad value."""
        img = torch.rand(1, 3, 100, 200)
        dst_size = (150, 150)
        pad_value = 1.0

        result, _scale, _padding = resize_pad(img, dst_size, pad_value=pad_value)

        assert result.shape == (1, 3, 150, 150)

    def test_undo_resize_pad(self) -> None:
        """Test undo_resize_pad restores original size."""
        img = torch.rand(1, 3, 100, 200)
        dst_size = (150, 150)
        orig_size_wh = (200, 100)

        # Apply resize_pad
        resized, scale, padding = resize_pad(img, dst_size)

        # Undo it
        restored = undo_resize_pad(resized, orig_size_wh, scale, padding)

        # Check dimensions match (may not be exact due to rounding)
        assert restored.shape[2] == 100
        assert restored.shape[3] == 200

    def test_pil_resize_pad(self) -> None:
        """Test PIL version of resize_pad."""
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        dst_size = (150, 150)

        result, scale, padding = pil_resize_pad(img, dst_size)

        assert isinstance(result, Image.Image)
        assert result.size == (150, 150)
        assert isinstance(scale, float)
        assert len(padding) == 2

    def test_pil_undo_resize_pad(self) -> None:
        """Test PIL version of undo_resize_pad."""
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        dst_size = (150, 150)
        orig_size_wh = (200, 100)

        # Apply pil_resize_pad
        resized, scale, padding = pil_resize_pad(img, dst_size)

        # Undo it
        restored = pil_undo_resize_pad(resized, orig_size_wh, scale, padding)

        assert isinstance(restored, Image.Image)
        assert restored.size == orig_size_wh


class TestCoordinateTransforms:
    """Test coordinate transformation functions."""

    def test_transform_resize_pad_coordinates(self) -> None:
        """Test coordinate transformation for resize_pad."""
        coords = torch.tensor([[10, 20], [30, 40]])  # x, y coordinates
        scale_factor = 2.0
        pad = (5, 10)

        result = transform_resize_pad_coordinates(coords, scale_factor, pad)

        expected = torch.tensor(
            [[25, 50], [65, 90]]
        )  # (10*2+5, 20*2+10), (30*2+5, 40*2+10)
        assert torch.equal(result, expected)

    def test_transform_resize_pad_coordinates_with_tensor_pad(self) -> None:
        """Test coordinate transformation with tensor padding."""
        coords = torch.tensor([[10, 20]])
        scale_factor = 2.0
        pad = torch.tensor([5, 10])

        result = transform_resize_pad_coordinates(coords, scale_factor, pad)

        expected = torch.tensor([[25, 50]])
        assert torch.equal(result, expected)

    def test_transform_resize_pad_normalized_coordinates(self) -> None:
        """Test normalized coordinate transformation."""
        coords = torch.tensor([[0.5, 0.5]])  # Center of image
        src_shape = (100, 200)  # width, height
        dst_shape = (150, 150)
        scale_factor = 0.75
        pad = (37, 0)  # Padding added

        result = transform_resize_pad_normalized_coordinates(
            coords, src_shape, dst_shape, scale_factor, pad
        )

        # Result should be normalized coordinates in resized image
        assert result.shape == coords.shape
        assert torch.all(result >= 0.0) and torch.all(result <= 1.0)

    def test_denormalize_coordinates(self) -> None:
        """Test denormalize_coordinates modifies in place."""
        coords = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32)
        input_img_size = (100, 200)  # height, width
        scale = 2.0
        pad = (10, 20)

        # Should modify in place
        denormalize_coordinates(coords, input_img_size, scale, pad)

        # Check that coordinates are now in pixel space
        assert coords.dtype == torch.float32
        # Expected: ((0.5 * 100 - 10) / 2, (0.5 * 200 - 20) / 2) = (20, 40)
        expected_0 = torch.tensor([20, 40], dtype=torch.float32)
        assert torch.allclose(coords[0], expected_0)


class TestAffineTransforms:
    """Test affine transformation functions."""

    def test_apply_batched_affines_to_frame(self, rng: np.random.Generator) -> None:
        """Test applying batched affines to a frame."""
        frame = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        # Identity affine
        affine1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        # Translation affine
        affine2 = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float32)
        affines = [affine1, affine2]

        result = apply_batched_affines_to_frame(frame, affines, (100, 100))

        assert result.shape == (2, 100, 100, 3)
        assert result.dtype == np.uint8

    def test_apply_affine_to_coordinates_numpy(self) -> None:
        """Test applying affine to coordinates with numpy."""
        coords = np.array([[10, 20], [30, 40]], dtype=np.float32)
        # Identity affine
        affine = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 10.0]], dtype=np.float32)

        result = apply_affine_to_coordinates(coords, affine)

        expected = np.array([[15, 30], [35, 50]], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_apply_affine_to_coordinates_torch(self) -> None:
        """Test applying affine to coordinates with torch."""
        coords = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        affine = torch.tensor([[1.0, 0.0, 5.0], [0.0, 1.0, 10.0]])

        result = apply_affine_to_coordinates(coords, affine)

        expected = torch.tensor([[15.0, 30.0], [35.0, 50.0]])
        assert torch.allclose(result, expected)

    def test_compute_vector_rotation(self) -> None:
        """Test computing vector rotation."""
        vec_start = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        vec_end = torch.tensor([[10.0, 0.0], [20.0, 10.0]])
        offset_rads = 0.0

        result = compute_vector_rotation(vec_start, vec_end, offset_rads)

        assert result.shape == (2,)
        # atan2(start_y - end_y, start_x - end_x) = atan2(0 - 0, 0 - 10) = atan2(0, -10) = pi
        assert torch.allclose(result[0], torch.tensor(torch.pi), atol=1e-5)

    def test_compute_vector_rotation_with_offset(self) -> None:
        """Test computing vector rotation with offset."""
        vec_start = torch.tensor([[0.0, 0.0]])
        vec_end = torch.tensor([[10.0, 0.0]])
        offset_rads = torch.pi / 4

        result = compute_vector_rotation(vec_start, vec_end, offset_rads)

        # atan2(0, -10) - pi/4 = pi - pi/4 = 3*pi/4
        assert torch.allclose(result[0], torch.tensor(3 * torch.pi / 4), atol=1e-5)

    def test_compute_affine_transform_no_rotation(self) -> None:
        """Test compute_affine_transform without rotation."""
        center = np.array([50.0, 50.0])
        scale = np.array([100.0, 100.0])
        rot = 0
        output_size = (200, 200)

        affine = compute_affine_transform(center, scale, rot, output_size)

        assert affine.shape == (2, 3)
        # cv2.getAffineTransform returns float64
        assert affine.dtype in [np.float32, np.float64]
        # Just verify it's a valid transformation matrix
        assert not np.any(np.isnan(affine))

    def test_compute_affine_transform_with_rotation(self) -> None:
        """Test compute_affine_transform with rotation."""
        center = np.array([50.0, 50.0])
        scale = np.array([100.0, 100.0])
        rot = 45
        output_size = (200, 200)

        affine = compute_affine_transform(center, scale, rot, output_size)

        assert affine.shape == (2, 3)

    def test_compute_affine_transform_inverse(self) -> None:
        """Test compute_affine_transform with inverse."""
        center = np.array([50.0, 50.0])
        scale = np.array([100.0, 100.0])
        rot = 0
        output_size = (200, 200)

        affine_forward = compute_affine_transform(
            center, scale, rot, output_size, inv=False
        )
        affine_inverse = compute_affine_transform(
            center, scale, rot, output_size, inv=True
        )

        assert affine_forward.shape == (2, 3)
        assert affine_inverse.shape == (2, 3)
        # They should be different
        assert not np.allclose(affine_forward, affine_inverse)

    def test_compute_affine_transform_with_shift(self) -> None:
        """Test compute_affine_transform with shift."""
        center = np.array([50.0, 50.0])
        scale = np.array([100.0, 100.0])
        rot = 0
        output_size = (200, 200)
        shift = np.array([10.0, 20.0])

        affine = compute_affine_transform(center, scale, rot, output_size, shift=shift)

        assert affine.shape == (2, 3)

    def test_get_dir(self) -> None:
        """Test get_dir rotation calculation."""
        src_point = np.array([1.0, 0.0])
        rot_rad = np.pi / 2  # 90 degrees

        result = get_dir(src_point, rot_rad)

        expected = np.array([0.0, 1.0])
        assert np.allclose(result, expected, atol=1e-5)

    def test_get_dir_no_rotation(self) -> None:
        """Test get_dir with no rotation."""
        src_point = np.array([1.0, 0.0])
        rot_rad = 0.0

        result = get_dir(src_point, rot_rad)

        assert np.allclose(result, src_point)

    def test_get_3rd_point(self) -> None:
        """Test get_3rd_point calculation."""
        point_x = np.array([0.0, 0.0])
        point_y = np.array([1.0, 0.0])

        result = get_3rd_point(point_x, point_y)

        # direct = [0, 0] - [1, 0] = [-1, 0]
        # return [1, 0] + [-0, -1] = [1, -1]
        expected = np.array([1.0, -1.0])
        assert np.allclose(result, expected)

    def test_get_3rd_point_different_points(self) -> None:
        """Test get_3rd_point with different points."""
        point_x = np.array([0.0, 0.0])
        point_y = np.array([0.0, 1.0])

        result = get_3rd_point(point_x, point_y)

        # direct = [0, 0] - [0, 1] = [0, -1]
        # return [0, 1] + [1, 0] = [1, 1]
        expected = np.array([1.0, 1.0])
        assert np.allclose(result, expected)

    def test_pre_process_with_affine(self, rng: np.random.Generator) -> None:
        """Test pre_process_with_affine."""
        image = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        center = np.array([50.0, 50.0])
        scale = np.array([100.0, 100.0])
        rot = 0
        in_shape = (64, 64)

        result = pre_process_with_affine(image, center, scale, rot, in_shape)

        assert result.shape == (1, 3, 64, 64)
        assert isinstance(result, torch.Tensor)

    def test_denormalize_coordinates_affine(self) -> None:
        """Test denormalize_coordinates_affine."""
        coords = np.array([[32.0, 32.0], [16.0, 48.0]], dtype=np.float32)
        center = np.array([50.0, 50.0])
        scale = np.array([100.0, 100.0])
        rot = 0
        output_size = (64, 64)

        result = denormalize_coordinates_affine(coords, center, scale, rot, output_size)

        assert result.shape == coords.shape
        assert isinstance(result, np.ndarray)

    def test_get_post_rot_and_tran_no_rotation(self) -> None:
        """Test get_post_rot_and_tran without rotation."""
        resize = 2.0
        crop = (0, 0, 100, 100)
        rotate = 0

        post_rot, post_tran = get_post_rot_and_tran(resize, crop, rotate)

        assert post_rot.shape == (3, 3)
        assert post_tran.shape == (3,)
        # With no rotation and zero crop, should just be identity scaled by resize
        assert torch.allclose(post_rot[0, 0], torch.tensor(2.0))
        assert torch.allclose(post_rot[1, 1], torch.tensor(2.0))

    def test_get_post_rot_and_tran_with_rotation(self) -> None:
        """Test get_post_rot_and_tran with rotation."""
        resize = 1.0
        crop = (10, 20, 110, 120)
        rotate = 90

        post_rot, post_tran = get_post_rot_and_tran(resize, crop, rotate)

        assert post_rot.shape == (3, 3)
        assert post_tran.shape == (3,)

    def test_get_post_rot_and_tran_with_crop(self) -> None:
        """Test get_post_rot_and_tran with crop."""
        resize = 1.0
        crop = (10, 20, 110, 120)
        rotate = 0

        _post_rot, post_tran = get_post_rot_and_tran(resize, crop, rotate)

        # Translation should account for crop
        assert not torch.equal(post_tran, torch.zeros(3))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_resize_pad_invalid_float_vertical(self) -> None:
        """Test resize_pad with invalid vertical float."""
        img = torch.rand(1, 3, 100, 200)
        dst_size = (150, 150)

        with pytest.raises(ValueError, match="Invalid pad type"):
            resize_pad(img, dst_size, vertical_float="invalid")  # type: ignore[arg-type]

    def test_resize_pad_invalid_float_horizontal(self) -> None:
        """Test resize_pad with invalid horizontal float."""
        img = torch.rand(1, 3, 100, 200)
        dst_size = (150, 150)

        with pytest.raises(ValueError, match="Invalid pad type"):
            resize_pad(img, dst_size, horizontal_float="invalid")  # type: ignore[arg-type]

    def test_apply_batched_affines_wrong_dtype(self, rng: np.random.Generator) -> None:
        """Test apply_batched_affines_to_frame with wrong dtype."""
        # Should fail with float32 dtype
        frame = rng.random((100, 100, 3)).astype(np.float32)
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        with pytest.raises(AssertionError):
            apply_batched_affines_to_frame(frame, [affine], (100, 100))

    def test_torch_image_to_numpy_wrong_shape(self) -> None:
        """Test torch_image_to_numpy with wrong shape."""
        img_torch = torch.rand(3, 50, 100)  # Missing batch dimension

        with pytest.raises(AssertionError):
            torch_image_to_numpy(img_torch)


class TestIntegration:
    """Test integration scenarios combining multiple functions."""

    def test_roundtrip_pil_torch_pil(self) -> None:
        """Test roundtrip conversion PIL -> Torch -> PIL."""
        original = Image.new("RGB", (100, 50), color=(128, 64, 32))

        # PIL -> Torch
        torch_img = preprocess_PIL_image(original)

        # Torch -> PIL
        restored = torch_tensor_to_PIL_image(torch_img[0])

        # Check dimensions
        assert restored.size == original.size

        # Check colors are approximately preserved
        original_array = np.array(original)
        restored_array = np.array(restored)
        assert np.allclose(original_array, restored_array, atol=2)

    def test_roundtrip_numpy_torch_numpy(self, rng: np.random.Generator) -> None:
        """Test roundtrip conversion Numpy -> Torch -> Numpy."""
        original = rng.integers(0, 256, (50, 100, 3), dtype=np.uint8)

        # Numpy -> Torch
        torch_img = numpy_image_to_torch(original)

        # Torch -> Numpy (expand to batch)
        torch_img_batched = torch_img  # Already has batch dim
        restored = torch_image_to_numpy(torch_img_batched)

        # Check shape
        assert restored.shape == original.shape

        # Check values are approximately preserved
        assert np.allclose(original, restored, atol=2)

    def test_resize_pad_undo_roundtrip(self) -> None:
        """Test resize_pad and undo_resize_pad roundtrip."""
        img = torch.rand(1, 3, 100, 200)
        dst_size = (300, 300)
        orig_size_wh = (200, 100)

        # Apply resize_pad
        resized, scale, padding = resize_pad(img, dst_size)

        # Undo
        restored = undo_resize_pad(resized, orig_size_wh, scale, padding)

        # Check size is restored
        assert restored.shape[2:] == (100, 200)

        # Note: We don't check exact value match because bilinear interpolation
        # with large scale factors introduces significant artifacts.
        # The important property is that the shape is correctly restored.

    def test_affine_transform_roundtrip(self) -> None:
        """Test affine transform and inverse roundtrip."""
        coords = np.array([[50.0, 50.0]], dtype=np.float32)
        center = np.array([50.0, 50.0])
        scale = np.array([100.0, 100.0])
        rot = 45
        output_size = (100, 100)

        # Forward transform
        affine_forward = compute_affine_transform(
            center, scale, rot, output_size, inv=False
        )
        transformed = apply_affine_to_coordinates(coords, affine_forward)

        # Inverse transform
        affine_inverse = compute_affine_transform(
            center, scale, rot, output_size, inv=True
        )
        restored = apply_affine_to_coordinates(transformed, affine_inverse)

        # Should get back approximately the same coordinates
        assert np.allclose(coords, restored, atol=1.0)

    def test_pil_resize_pad_undo_roundtrip(self) -> None:
        """Test PIL resize_pad and undo_resize_pad roundtrip."""
        original = Image.new("RGB", (200, 100), color=(255, 0, 0))
        dst_size = (300, 300)
        orig_size_wh = (200, 100)

        # Apply resize_pad
        resized, scale, padding = pil_resize_pad(original, dst_size)

        # Undo
        restored = pil_undo_resize_pad(resized, orig_size_wh, scale, padding)

        # Check size is restored
        assert restored.size == orig_size_wh

    def test_coordinate_transform_pipeline(self) -> None:
        """Test a pipeline of coordinate transformations."""
        # Start with normalized coordinates
        coords = torch.tensor([[0.5, 0.5]])
        src_shape = (100, 200)
        dst_shape = (150, 150)
        scale_factor = 0.75
        pad = (37, 0)

        # Transform to resized space
        transformed = transform_resize_pad_normalized_coordinates(
            coords, src_shape, dst_shape, scale_factor, pad
        )

        # Should still be normalized
        assert torch.all(transformed >= 0.0)
        assert torch.all(transformed <= 1.0)

        # Convert to pixel space
        pixel_coords = transformed * torch.tensor([dst_shape[0], dst_shape[1]])

        # Should be valid pixel coordinates
        assert torch.all(pixel_coords >= 0)
        assert torch.all(pixel_coords <= torch.tensor([dst_shape[0], dst_shape[1]]))

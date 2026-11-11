# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging
import sys
import types
from typing import Any, cast

import cv2
import mmcv as _mmcv
import numpy as np
import torch
from torch import nn

from qai_hub_models.models.templates.centernet.model_patches import (
    custom_deformconv2d as _custom_deformconv2d,
)


def grouped_modulated_deform_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    offset: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    deform_groups: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the single-offset-group helper across deformable groups."""
    if groups != 1:
        raise ValueError("Grouped convolution weights are not supported.")
    if x.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported.")
    if x.shape[1] % deform_groups != 0:
        raise ValueError("Input channels must be divisible by deform_groups.")

    kernel_points = weight.shape[2] * weight.shape[3]
    channels_per_group = x.shape[1] // deform_groups
    offset_channels_per_group = 2 * kernel_points
    mask_channels_per_group = kernel_points
    output: torch.Tensor | None = None

    for group_index in range(deform_groups):
        channel_start = group_index * channels_per_group
        channel_end = channel_start + channels_per_group
        offset_start = group_index * offset_channels_per_group
        offset_end = offset_start + offset_channels_per_group
        mask_start = group_index * mask_channels_per_group
        mask_end = mask_start + mask_channels_per_group

        group_output = _custom_deformconv2d(
            x[:, channel_start:channel_end],
            weight[:, channel_start:channel_end],
            None,
            offset[:, offset_start:offset_end],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            mask=mask[:, mask_start:mask_end],
        )
        output = group_output if output is None else output + group_output

    if output is None:
        raise ValueError("deform_groups must be positive.")
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output


def _make_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def install() -> None:
    """Populate sys.modules with mmcv v1 stubs so mmedit imports don't fail.

    Why not patch_mmcv_no_extensions() + mmcv==1.7.0 (BEVFormer approach)?
    BasicVSR++ calls mmcv.ops.ModulatedDeformConv2d, a native C++ op that
    patch_mmcv_no_extensions() explicitly does not replace — it only silences
    the extension loader.  This shim provides a pure-Python replacement so the
    model runs without any compiled mmcv extensions.

    Leak safety: sys.modules entries added here are permanent for the process
    lifetime (matching how real package imports work).  The np.bool8 / int0 /
    uint0 aliases are additive only (guarded by hasattr), so they are safe in a
    shared pytest session — they never remove or overwrite existing attributes.
    """
    if "mmcv.runner" in sys.modules:
        return  # Already installed

    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_  # type: ignore[attr-defined]
    if not hasattr(np, "int0"):
        np.int0 = np.intp  # type: ignore[attr-defined]
    if not hasattr(np, "uint0"):
        np.uint0 = np.uintp  # type: ignore[attr-defined]

    # ---- mmcv.runner -------------------------------------------------------
    runner = _make_module("mmcv.runner")

    def _load_checkpoint(
        model: nn.Module, filename: str, strict: bool = False, logger: Any = None
    ) -> None:
        # weights_only=False is safe: filename is always the pinned SPYNET_WEIGHTS_URL cache asset, never caller-controlled.
        ckpt = torch.load(filename, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=strict)

    runner.load_checkpoint = _load_checkpoint  # type: ignore[attr-defined]
    runner.get_dist_info = lambda: (0, 1)  # type: ignore[attr-defined]

    def _auto_fp16(
        fn: Any = None, apply_to: Any = None, out_fp32: bool = False, **kw: Any
    ) -> Any:
        """Decorator no-op: returns the function unchanged."""
        if fn is not None:
            return fn
        return lambda f: f

    runner.auto_fp16 = _auto_fp16  # type: ignore[attr-defined]

    class _Hook:
        pass

    class _RunnerRegistry:
        def __init__(self, name: str = "", parent: Any = None, **kw: Any) -> None:
            self._name = name
            self._module_dict: dict[str, Any] = {}

        def register_module(
            self,
            cls: Any = None,
            name: Any = None,
            force: bool = False,
            module: Any = None,
        ) -> Any:
            def _reg(c: Any) -> Any:
                self._module_dict[name or c.__name__] = c
                return c

            if module is not None:
                return _reg(module)
            return _reg if cls is None else _reg(cls)

        def get(self, key: str) -> Any:
            return self._module_dict.get(key)

    runner.HOOKS = _RunnerRegistry("hooks")  # type: ignore[attr-defined]
    runner.Hook = _Hook  # type: ignore[attr-defined]
    runner.IterBasedRunner = object  # type: ignore[attr-defined]
    runner.LrUpdaterHook = _Hook  # type: ignore[attr-defined]
    runner.obj_from_dict = lambda d, parent, **kw: None  # type: ignore[attr-defined]
    runner.build_optimizer = lambda model, cfg: None  # type: ignore[attr-defined]
    runner.init_dist = lambda launcher, **kw: None  # type: ignore[attr-defined]

    class _DistUtils:
        master_only = staticmethod(lambda fn: fn)

    runner.dist_utils = _DistUtils()  # type: ignore[attr-defined]
    sys.modules["mmcv.runner.dist_utils"] = _DistUtils()  # type: ignore[assignment]

    # ---- mmcv.cnn ----------------------------------------------------------
    cnn = _make_module("mmcv.cnn")

    class ConvModule(nn.Module):
        """Minimal mmcv ConvModule: stores sub-modules as named attrs to match checkpoint keys."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Any,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            norm_cfg: Any = None,
            act_cfg: Any = None,
            bias: bool = True,
            **kw: Any,
        ) -> None:
            assert norm_cfg is None, (
                "norm_cfg is not supported; passing non-None would silently drop the norm layer and corrupt checkpoint keys"
            )
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            self.activate: nn.Module | None = None
            if act_cfg is not None:
                act_type = act_cfg.get("type", "ReLU")
                if act_type == "ReLU":
                    self.activate = nn.ReLU(inplace=True)
                elif act_type == "LeakyReLU":
                    self.activate = nn.LeakyReLU(
                        negative_slope=act_cfg.get("negative_slope", 0.1), inplace=True
                    )
                elif act_type == "Sigmoid":
                    self.activate = nn.Sigmoid()
                elif act_type == "Tanh":
                    self.activate = nn.Tanh()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            if self.activate is not None:
                x = self.activate(x)
            return x

    def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
        w = cast(torch.Tensor, module.weight) if hasattr(module, "weight") else None
        b = cast(torch.Tensor, module.bias) if hasattr(module, "bias") else None
        if w is not None:
            nn.init.constant_(w, val)
        if b is not None:
            nn.init.constant_(b, bias)

    def kaiming_init(
        module: nn.Module,
        a: float = 0,
        mode: str = "fan_out",
        nonlinearity: str = "relu",
        bias: float = 0,
        distribution: str = "normal",
    ) -> None:
        w = cast(torch.Tensor, module.weight)
        if distribution == "uniform":
            nn.init.kaiming_uniform_(w, a=a, mode=mode, nonlinearity=nonlinearity)  # type: ignore[arg-type]
        else:
            nn.init.kaiming_normal_(w, a=a, mode=mode, nonlinearity=nonlinearity)  # type: ignore[arg-type]
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(cast(torch.Tensor, module.bias), bias)

    def xavier_init(
        module: nn.Module,
        gain: float = 1,
        bias: float = 0,
        distribution: str = "normal",
    ) -> None:
        w = cast(torch.Tensor, module.weight)
        if distribution == "uniform":
            nn.init.xavier_uniform_(w, gain=gain)
        else:
            nn.init.xavier_normal_(w, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(cast(torch.Tensor, module.bias), bias)

    def normal_init(
        module: nn.Module,
        mean: float = 0,
        std: float = 1,
        bias: float = 0,
    ) -> None:
        nn.init.normal_(cast(torch.Tensor, module.weight), mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(cast(torch.Tensor, module.bias), bias)

    def build_activation_layer(cfg: dict[str, Any]) -> nn.Module:
        act_type = cfg.get("type", "ReLU")
        if act_type == "ReLU":
            return nn.ReLU(inplace=cfg.get("inplace", True))
        if act_type == "LeakyReLU":
            return nn.LeakyReLU(
                negative_slope=cfg.get("negative_slope", 0.01),
                inplace=cfg.get("inplace", False),
            )
        if act_type == "Sigmoid":
            return nn.Sigmoid()
        return nn.ReLU()

    class _ConvLayers:
        """Minimal Registry stub for CONV_LAYERS."""

        def __init__(self) -> None:
            self._module_dict: dict[str, Any] = {
                "Conv2d": nn.Conv2d,
                "Conv1d": nn.Conv1d,
                "Conv3d": nn.Conv3d,
            }

        def register_module(
            self,
            cls: Any = None,
            name: Any = None,
            force: bool = False,
            module: Any = None,
        ) -> Any:
            if module is not None:
                self._module_dict[name or module.__name__] = module
                return module

            def _reg(c: Any) -> Any:
                self._module_dict[name or c.__name__] = c
                return c

            return _reg if cls is None else _reg(cls)

        def build(self, cfg: dict[str, Any], **kw: Any) -> Any:
            t = cfg.pop("type")
            return self._module_dict[t](**cfg)

    cnn.ConvModule = ConvModule  # type: ignore[attr-defined]
    cnn.constant_init = constant_init  # type: ignore[attr-defined]
    cnn.kaiming_init = kaiming_init  # type: ignore[attr-defined]
    cnn.xavier_init = xavier_init  # type: ignore[attr-defined]
    cnn.normal_init = normal_init  # type: ignore[attr-defined]
    cnn.build_activation_layer = build_activation_layer  # type: ignore[attr-defined]
    cnn.CONV_LAYERS = _ConvLayers()  # type: ignore[attr-defined]

    class _MMCVModels:
        def __init__(self) -> None:
            self._d: dict[str, Any] = {}

        def register_module(
            self,
            cls: Any = None,
            name: Any = None,
            force: bool = False,
        ) -> Any:
            def _reg(c: Any) -> Any:
                self._d[name or c.__name__] = c
                return c

            return _reg if cls is None else _reg(cls)

    cnn.MODELS = _MMCVModels()  # type: ignore[attr-defined]

    # mmcv.cnn.utils sub-module
    cnn_utils = _make_module("mmcv.cnn.utils")
    cnn_utils.get_model_complexity_info = lambda *a, **kw: (0, "0")  # type: ignore[attr-defined]
    cnn.utils = cnn_utils  # type: ignore[attr-defined]

    # ---- mmcv.ops ----------------------------------------------------------
    ops = _make_module("mmcv.ops")

    class ModulatedDeformConv2d(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Any,
            stride: Any = 1,
            padding: Any = 0,
            dilation: Any = 1,
            groups: int = 1,
            deform_groups: int = 1,
            bias: bool = True,
            **kw: Any,
        ) -> None:
            super().__init__()
            ks = (
                (kernel_size, kernel_size)
                if isinstance(kernel_size, int)
                else kernel_size
            )
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = ks
            self.stride = (stride, stride) if isinstance(stride, int) else stride
            self.padding = (padding, padding) if isinstance(padding, int) else padding
            self.dilation = (
                (dilation, dilation) if isinstance(dilation, int) else dilation
            )
            self.groups = groups
            self.deform_groups = deform_groups
            self.weight = nn.Parameter(
                torch.empty(out_channels, in_channels // groups, *ks)
            )
            self.bias: nn.Parameter | None = (
                nn.Parameter(torch.zeros(out_channels)) if bias else None
            )
            nn.init.kaiming_uniform_(self.weight, a=0)

        def forward(
            self,
            x: torch.Tensor,
            offset: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            return grouped_modulated_deform_conv2d(
                x,
                self.weight,
                self.bias,
                offset,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                deform_groups=self.deform_groups,
                mask=mask,
            )

    def modulated_deform_conv2d(
        x: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: Any,
        padding: Any,
        dilation: Any,
        groups: int,
        deform_groups: int,
    ) -> torch.Tensor:
        return grouped_modulated_deform_conv2d(
            x,
            weight,
            bias,
            offset,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups,
            mask=mask,
        )

    ops.ModulatedDeformConv2d = ModulatedDeformConv2d  # type: ignore[attr-defined]
    ops.modulated_deform_conv2d = modulated_deform_conv2d  # type: ignore[attr-defined]
    ops.get_onnxruntime_op_path = lambda: ""  # type: ignore[attr-defined]

    # ---- mmcv.utils --------------------------------------------------------
    utils = _make_module("mmcv.utils")
    utils.get_logger = (  # type: ignore[attr-defined]
        lambda name, log_file=None, log_level=logging.INFO: logging.getLogger(name)
    )
    utils.print_log = lambda msg, logger=None, level=logging.INFO: None  # type: ignore[attr-defined]
    utils.build_from_cfg = lambda cfg, registry, **kw: None  # type: ignore[attr-defined]
    utils.collect_env = dict  # type: ignore[attr-defined]
    utils.get_git_hash = lambda *a, **kw: ""  # type: ignore[attr-defined]

    class _Registry:
        def __init__(self, name: str = "", parent: Any = None, **kw: Any) -> None:
            self._name = name
            self._module_dict: dict[str, Any] = {}
            if parent is not None and hasattr(parent, "_d"):
                self._module_dict.update(parent._d)
            elif parent is not None and hasattr(parent, "_module_dict"):
                self._module_dict.update(parent._module_dict)

        def register_module(
            self,
            cls: Any = None,
            name: Any = None,
            force: bool = False,
            module: Any = None,
        ) -> Any:
            def _reg(c: Any) -> Any:
                self._module_dict[name or c.__name__] = c
                return c

            if module is not None:
                return _reg(module)
            return _reg if cls is None else _reg(cls)

        def get(self, key: str) -> Any:
            return self._module_dict.get(key)

        def build(self, cfg: dict[str, Any], **kw: Any) -> Any:
            t = cfg.pop("type")
            cls = self._module_dict[t]
            return cls(**cfg)

    utils.Registry = _Registry  # type: ignore[attr-defined]

    parrots = _make_module("mmcv.utils.parrots_wrapper")
    parrots._BatchNorm = nn.BatchNorm2d  # type: ignore[attr-defined]
    utils.parrots_wrapper = parrots  # type: ignore[attr-defined]

    parallel = _make_module("mmcv.parallel")
    parallel.collate = lambda batch, **kw: batch  # type: ignore[attr-defined]
    parallel.scatter = lambda data, target_gpus, **kw: data  # type: ignore[attr-defined]
    parallel.MMDataParallel = nn.DataParallel  # type: ignore[attr-defined]
    parallel.MMDistributedDataParallel = nn.parallel.DistributedDataParallel  # type: ignore[attr-defined]
    parallel.MODULE_WRAPPERS = []  # type: ignore[attr-defined]
    parallel.is_module_wrapper = lambda m: False  # type: ignore[attr-defined]

    class _DataContainer:
        def __init__(self, data: Any, **kw: Any) -> None:
            self.data = data

    parallel.DataContainer = _DataContainer  # type: ignore[attr-defined]
    sys.modules["mmcv.parallel.DataContainer"] = _DataContainer  # type: ignore[assignment]

    sg = _make_module("mmcv.parallel.scatter_gather")
    sg.scatter_kwargs = lambda inputs, kwargs, target_gpus, **kw: (inputs, kwargs)  # type: ignore[attr-defined]

    fileio = _make_module("mmcv.fileio")
    fileio.FileClient = object  # type: ignore[attr-defined]

    if not hasattr(_mmcv, "scandir"):
        _mmcv.scandir = lambda *a, **kw: iter([])
    if not hasattr(_mmcv, "imresize"):
        _mmcv.imresize = lambda img, size, **kw: cv2.resize(img, size)
    if not hasattr(_mmcv, "build_from_cfg"):
        _mmcv.build_from_cfg = lambda cfg, registry, **kw: None

    # Patch mmcv.utils.Registry into mmcv directly
    _mmcv.utils = utils
    if not hasattr(_mmcv, "Registry"):
        _mmcv.Registry = _Registry

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution

from mmengine import utils as mmengine_utils
from mmengine.config import config
from mmengine.runner import checkpoint
from mmengine.utils import package_utils


@contextmanager
def patch_mmengine_pkgresources():
    """Patched mmengine's use of pkg_resources APIs that are now deprecated & removed from setuptools."""

    def is_installed(package: str) -> bool:
        try:
            distribution(package)
            return True
        except PackageNotFoundError:
            return False

    def get_installed_path(package: str) -> str:
        """Copied from mmengine.utils.package_utils; modified to remove pkg_resources"""
        import importlib.util
        import os.path as osp

        # if the package name is not the same as module name, module name should be
        # inferred. For example, mmcv-full is the package name, but mmcv is module
        # name. If we want to get the installed path of mmcv-full, we should concat
        # the pkg.location and module name
        try:
            dist = distribution(package)
        except PackageNotFoundError as e:
            # if the package is not installed, package path set in PYTHONPATH
            # can be detected by `find_spec`
            spec = importlib.util.find_spec(package)
            if spec is not None:
                if spec.origin is not None:
                    return osp.dirname(spec.origin)
                # `get_installed_path` cannot get the installed path of
                # namespace packages
                raise RuntimeError(
                    f"{package} is a namespace package, which is invalid "
                    "for `get_install_path`"
                ) from e
            raise

        location = str(dist.locate_file(""))
        possible_path = osp.join(location, package)
        if osp.exists(possible_path):
            return possible_path
        return osp.join(location, package2module(package))

    def package2module(package: str):
        """Copied from mmengine.utils.package_utils; modified to remove pkg_resources"""
        dist = distribution(package)
        try:
            top_level = dist.read_text("top_level.txt")
            assert top_level is not None
            return top_level.split("\n")[0]
        except FileNotFoundError as e:
            raise ValueError(f"can not infer the module name of {package}") from e

    is_installed_old = package_utils.is_installed
    package2module_old = package_utils.package2module
    get_installed_path_old = package_utils.get_installed_path

    config.is_installed = is_installed
    config.get_installed_path = get_installed_path
    package_utils.is_installed = is_installed
    package_utils.package2module = package2module
    package_utils.get_installed_path = get_installed_path
    mmengine_utils.get_installed_path = get_installed_path
    try:
        yield
    finally:
        config.is_installed = is_installed_old
        config.get_installed_path = get_installed_path_old
        package_utils.is_installed = is_installed_old
        package_utils.package2module = package2module_old
        package_utils.get_installed_path = get_installed_path_old
        mmengine_utils.get_installed_path = get_installed_path_old


@contextmanager
def patch_mmengine_torch_load_no_weights_only():
    """
    Within this context manager, weights_only=False is always passed to torch.load()
    for compatibility with torch 2.6 and newer.
    """
    load_url = checkpoint.load_url

    def load_url_no_weights_only(*args, **kwargs):
        return load_url(*args, **kwargs, weights_only=False)

    checkpoint.load_url = load_url_no_weights_only

    torch_load = checkpoint.torch.load

    def torch_load_no_weights_only(*args, **kwargs):
        return torch_load(*args, **kwargs, weights_only=False)

    checkpoint.torch.load = torch_load_no_weights_only

    try:
        yield
    finally:
        checkpoint.load_url = load_url
        checkpoint.torch.load = torch_load

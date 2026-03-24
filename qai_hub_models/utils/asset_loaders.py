# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import fileinput
import json
import logging
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager, suppress
from enum import Enum
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock
from zipfile import ZipFile

import gdown
import h5py
import numpy as np
import requests
import ruamel.yaml
import torch
from PIL import Image
from qai_hub.util.dataset_entries_converters import h5_to_dataset_entries
from schema import And, Schema, SchemaError
from tqdm import tqdm

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.aws import can_access_private_s3
from qai_hub_models.utils.envvars import IsOnCIEnvvar
from qai_hub_models.utils.version_helpers import QAIHMVersion

ASSET_BASES_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "asset_bases.yaml"
)

QAIHM_STORE_ROOT = os.environ.get("QAIHM_STORE_ROOT", os.path.expanduser("~"))
LOCAL_STORE_DEFAULT_PATH = os.path.join(QAIHM_STORE_ROOT, ".qaihm")
EXECUTING_IN_CI_ENVIRONMENT = os.getenv("QAIHM_CI", "0") == "1"
SOURCE_AS_ROOT_LOCK = threading.Lock()

PathLike = os.PathLike | str
VersionType = str | int


# If non-None, always enter this for yes (True)/no (False) prompts
_always_answer = None


@contextmanager
def always_answer_prompts(answer: bool) -> Generator[None, None, None]:
    global _always_answer  # noqa: PLW0603
    old_value = _always_answer
    _always_answer = answer
    try:
        yield
    finally:
        _always_answer = old_value


@contextmanager
def set_log_level(log_level: int) -> Generator[None, None, None]:
    logger = logging.getLogger()
    old_level = logger.level
    try:
        logger.setLevel(log_level)
        yield
    finally:
        logger.setLevel(old_level)


@contextmanager
def tmp_os_env(env_values: dict[str, str]) -> Generator[None, None, None]:
    """
    Creates a context where the os environment variables are replaced with
        the given values. After exiting the context, the previous env is restored.
    """
    previous_env = os.environ.copy()
    try:
        os.environ.update(env_values)
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous_env)


UNPUBLISHED_MODEL_WARNING = (
    "This model is not published. Use with caution; "
    "it may not meet performance/accuracy standards "
    "and may not support some runtimes or chipsets/devices. "
    "We do not provide support for unpublished models. "
    "If this model was previously published, use earlier releases."
)


def query_yes_no(question: str, default: str | None = "yes") -> bool:
    """
    Ask a yes/no question and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Sourced from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    if _always_answer is not None:
        return _always_answer

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"invalid default answer: {default}")

    while True:
        print(question + prompt, end="")
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        if choice in valid:
            return valid[choice]
        print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def maybe_clone_git_repo(
    git_file_path: str,
    commit_hash: str,
    model_name: str,
    model_version: VersionType,
    patches: list[str] | None = None,
    ask_to_clone: bool = not EXECUTING_IN_CI_ENVIRONMENT,
) -> Path:
    """Clone (or pull) a repository, save it to disk in a standard location,
    and return the absolute path to the cloned location. Patches can be applied
    by providing a list of paths to diff files.
    """
    # http://blah.come/author/name.git -> name, author
    repo_name = os.path.basename(git_file_path).split(".")[0]
    repo_author = os.path.basename(os.path.dirname(git_file_path))
    local_path = ASSET_CONFIG.get_local_store_model_path(
        model_name, model_version, f"{repo_author}_{repo_name}_git"
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(os.path.join(local_path, ".git")):
        # Clone repo
        should_clone = (
            True
            if not ask_to_clone
            else query_yes_no(
                f"{model_name} requires repository {git_file_path} . Ok to clone?",
            )
        )
        if should_clone:
            print(f"Cloning {git_file_path} to {local_path}...")

            try:
                from git import (  # noqa:  TID251 Import is wrapped in try / catch as requested by precommit rules.
                    Repo,
                )
            except ImportError as e:
                if e.msg.startswith("Failed to initialize: Bad git executable."):
                    raise ImportError(
                        f"Git is not installed or can't be found on your system. Git is required to load {model_name}.\n"
                        "Follow these instructions to install git: https://github.com/git-guides/install-git\n"
                        "After Git is installed on your system, please try again."
                    ) from None

            repo = Repo.clone_from(git_file_path, local_path)
            repo.git.checkout(commit_hash)
            for patch_path in patches or []:
                git_cmd = ["git", "apply"]
                if platform.system() == "Windows":
                    # We pass ignore-space-change,
                    # which is used when finding patch content in files.
                    # Windows has trouble understanding non-windows EOL markers.
                    #
                    # There are more specific flags available that only change how
                    # git looks at line endings, but they are not available in all
                    # versions of git on windows.
                    git_cmd.append("--ignore-space-change")
                git_cmd.append(patch_path)
                repo.git.execute(git_cmd)
            print("Done")
        else:
            raise ValueError(
                f"Unable to load {model_name} without its required repository."
            )

    return local_path


def wipe_sys_modules(module: ModuleType) -> None:
    """
    Wipe all modules from sys.modules whose names start with the given module name.

    An alternative to `importlib.reload`, which only reloads the top-level module
        but may still reference the old package for submodules.
    """
    module_name = module.__name__
    dep_modules = [name for name in sys.modules if name.startswith(module_name)]
    for submodule_name in dep_modules:
        sys.modules.pop(submodule_name)


def _load_file(
    file: PathType,
    loader_func: Callable[[str], Any],
    dst_folder_path: tempfile.TemporaryDirectory | str | None = None,
) -> Any:
    if isinstance(file, (str, Path)):
        file = str(file)
        if file.startswith("http"):
            if dst_folder_path is None:
                dst_folder_path = tempfile.TemporaryDirectory()
            if isinstance(dst_folder_path, tempfile.TemporaryDirectory):
                dst_folder_path_str = dst_folder_path.name
            else:
                dst_folder_path_str = dst_folder_path
            dst_path = os.path.join(dst_folder_path_str, os.path.basename(file))
            download_file(file, dst_path)
            return loader_func(dst_path)
        return loader_func(file)
    if isinstance(file, CachedWebAsset):
        return loader_func(str(file.fetch()))
    raise NotImplementedError()


def load_image(
    image: PathType, verbose: bool = False, desc: str = "image"
) -> Image.Image:
    if verbose:
        print(f"Loading {desc} from {image}")
    return _load_file(image, Image.open)


def load_numpy(file: PathType) -> Any:
    return _load_file(file, np.load)


def load_torch(pt: PathType) -> Any:
    return _load_file(pt, partial(torch.load, map_location="cpu", weights_only=False))


def load_json(json_filepath: PathType) -> dict:
    def _load_json_helper(file_path: str) -> Any:
        with open(file_path) as json_file:
            return json.load(json_file)

    return _load_file(json_filepath, _load_json_helper)


def load_yaml(yaml_filepath: PathType) -> dict:
    def _load_yaml_helper(file_path: str) -> Any:
        with open(file_path) as yaml_file:
            return ruamel.yaml.YAML(typ="safe", pure=True).load(yaml_file)

    return _load_file(yaml_filepath, _load_yaml_helper)


def load_h5(h5_filepath: PathType) -> dict:
    def _load_h5_helper(file_path: str) -> Any:
        with h5py.File(file_path, "r") as h5f:
            return h5_to_dataset_entries(h5f)

    return _load_file(h5_filepath, _load_h5_helper)


def load_raw_file(filepath: PathType) -> str:
    def _load_raw_file_helper(file_path: str) -> Any:
        with open(file_path) as f:
            return f.read()

    return _load_file(filepath, _load_raw_file_helper)


def load_path(file: PathType, tmpdir: tempfile.TemporaryDirectory | str) -> str | Path:
    """
    Get asset path on disk.
    If `file` is a string URL, downloads the file to tmpdir.name.
    """

    def return_path(path: str) -> str:
        return path

    return _load_file(file, return_path, tmpdir)


def get_hub_datasets_path() -> Path:
    """Get the path where cached hub data for evaluation can be stored."""
    return Path(LOCAL_STORE_DEFAULT_PATH) / "hub_datasets"


@contextmanager
def SourceAsRoot(
    source_repo_url: str,
    source_repo_commit_hash: str,
    source_repo_name: str,
    source_repo_version: int | str,
    source_repo_patches: list[str] | None = None,
    keep_sys_modules: bool = True,
    ask_to_clone: bool = not EXECUTING_IN_CI_ENVIRONMENT,
    # These modules are imported but unused during model loading.
    # They are mocked so they can be imported without requiring us to install them.
    imported_but_unused_modules: list[str] | None = None,
    source_root_subdir: str | None = None,
) -> Generator[str, None, None]:
    """
    Context manager that runs code with:
     * the source repository added to the system path,
     * cwd set to the source repo's root directory.

    Only one of this class should be active per Python session.
    """
    repository_path = str(
        maybe_clone_git_repo(
            source_repo_url,
            source_repo_commit_hash,
            source_repo_name,
            source_repo_version,
            patches=source_repo_patches,
            ask_to_clone=ask_to_clone,
        )
    )
    if source_root_subdir:
        repository_path = os.path.join(repository_path, source_root_subdir)
    SOURCE_AS_ROOT_LOCK.acquire()
    original_path = list(sys.path)
    original_modules = dict(sys.modules)
    cwd = os.getcwd()
    try:
        # These modules are imported but unused during use.
        # They are mocked so they can be imported without error
        # without requiring us to install them.
        for module_name in imported_but_unused_modules or []:
            if module_name not in sys.modules:
                sys.modules[module_name] = MagicMock()

        # If repo path already in sys.path from previous load,
        # delete it and put it first
        if repository_path in sys.path:
            sys.path.remove(repository_path)
        # Patch path for this load only, since the model source
        # code references modules via a global scope.
        # Insert with highest priority (see #7666)
        sys.path.insert(0, repository_path)
        os.chdir(repository_path)
        yield repository_path
    finally:
        # Be careful editing these lines (failure means partial clean-up)
        os.chdir(cwd)
        sys.path = original_path
        if not keep_sys_modules:
            # When you call something like `import models`, it loads the `models` module
            # into sys.modules so all future `import models` point to that module.
            #
            # We want all imports done within the sub-repo to be either deleted from
            # sys.modules or restored to the previous module if one was overwritten.
            for name, module in list(sys.modules.items()):
                if (getattr(module, "__file__", "") or "").startswith(
                    repository_path
                ) or name in (imported_but_unused_modules or []):
                    if name in original_modules:
                        sys.modules[name] = original_modules[name]
                    else:
                        del sys.modules[name]
        SOURCE_AS_ROOT_LOCK.release()


def find_replace_in_repo(
    repo_path: str, filepaths: str | list[str], find_str: str, replace_str: str
) -> None:
    """
    When loading models from external repos, sometimes small modifications
    need to be made to the repo code to get it working in the zoo env.

    This does a simple find + replace within a single file.

    Parameters
    ----------
    repo_path
        Local filepath to the repo of interest.
    filepaths
        Filepath within the repo to the file to change.
    find_str
        The string that needs to be replaced.
    replace_str
        The string with which to replace all instances of `find_str`.
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    for filepath in filepaths:
        with fileinput.FileInput(
            Path(repo_path) / filepath,
            inplace=True,
            backup=".bak",
        ) as file:
            for line in file:
                print(line.replace(find_str, replace_str), end="")


class QAIHM_WEB_ASSET(Enum):
    STATIC_IMG = 0
    ANIMATED_MOV = 1


class ModelZooAssetConfig:
    def __init__(
        self,
        asset_url: str,
        web_asset_folder: str,
        static_web_banner_filename: str,
        animated_web_banner_filename: str,
        model_asset_folder: str,
        dataset_asset_folder: str,
        local_store_path: str,
        qaihm_repo: str,
        example_use: str,
        huggingface_path: str,
        repo_url: str,
        models_website_url: str,
        models_website_relative_path: str,
        genie_url: str,
        released_asset_folder: str,
        released_asset_filename: str,
        released_asset_with_chipset_filename: str,
    ) -> None:
        self.local_store_path = local_store_path
        self.asset_url = asset_url
        self.web_asset_folder = web_asset_folder
        self.static_web_banner_filename = static_web_banner_filename
        self.animated_web_banner_filename = animated_web_banner_filename
        self.model_asset_folder = model_asset_folder
        self.dataset_asset_folder = dataset_asset_folder
        self.qaihm_repo = qaihm_repo
        self.example_use = example_use
        self.huggingface_path = huggingface_path
        self.repo_url = repo_url
        self.models_website_url = models_website_url
        self.models_website_relative_path = models_website_relative_path
        self.genie_url = genie_url
        self.released_asset_folder = released_asset_folder
        self.released_asset_filename = released_asset_filename
        self.released_asset_with_chipset_filename = released_asset_with_chipset_filename

    def get_hugging_face_url(self, model_name: str) -> str:
        return f"https://huggingface.co/{self.get_huggingface_path(model_name)}"

    def get_huggingface_path(self, model_name: str) -> str:
        return self.huggingface_path.lstrip("/").replace(
            "{model_name}", str(model_name)
        )

    def get_web_asset_url(self, model_id: str, asset_type: QAIHM_WEB_ASSET) -> str:
        if asset_type == QAIHM_WEB_ASSET.STATIC_IMG:
            file = self.static_web_banner_filename
        elif asset_type == QAIHM_WEB_ASSET.ANIMATED_MOV:
            file = self.animated_web_banner_filename
        else:
            raise NotImplementedError("unsupported web asset type")
        return (
            f"{self.asset_url.rstrip('/')}/"
            + (
                Path(self.web_asset_folder.lstrip("/").format(model_id=model_id)) / file
            ).as_posix()
        )

    def get_local_store_path(self) -> Path:
        return Path(self.local_store_path)

    def get_local_store_model_path(
        self, model_name: str, version: VersionType, filename: Path | str
    ) -> Path:
        return self.local_store_path / self.get_relative_model_asset_path(
            model_name, version, filename
        )

    def get_local_store_dataset_path(
        self, dataset_name: str, version: VersionType, filename: Path | str
    ) -> Path:
        return self.local_store_path / self.get_relative_dataset_asset_path(
            dataset_name, version, filename
        )

    def get_relative_model_asset_path(
        self, model_id: str, version: int | str, file_name: Path | str
    ) -> Path:
        return Path(
            self.model_asset_folder.lstrip("/").format(
                model_id=model_id, version=version
            )
        ) / Path(file_name)

    def get_relative_dataset_asset_path(
        self, dataset_id: str, version: int | str, file_name: Path | str
    ) -> Path:
        return Path(
            self.dataset_asset_folder.lstrip("/").format(
                dataset_id=dataset_id, version=version
            )
        ) / Path(file_name)

    def get_asset_url(self, file: Path | str) -> str:
        return f"{self.asset_url.rstrip('/')}/{(file.as_posix() if isinstance(file, Path) else file).lstrip('/')}"

    def get_model_asset_url(
        self, model_id: str, version: int | str, file_name: Path | str
    ) -> str:
        return self.get_asset_url(
            self.get_relative_model_asset_path(model_id, version, file_name)
        )

    def get_dataset_asset_url(
        self, dataset_id: str, version: int | str, file_name: Path | str
    ) -> str:
        return self.get_asset_url(
            self.get_relative_dataset_asset_path(dataset_id, version, file_name)
        )

    def get_qaihm_repo(
        self,
        model_id: str | None,
        relative: bool = True,
        qaihm_version_tag: str | None = None,
    ) -> Path | str:
        relative_path = (
            Path(self.qaihm_repo.lstrip("/").format(model_id=model_id))
            if model_id
            else Path("qai_hub_models")
        )
        repo_url = self.repo_url
        if qaihm_version_tag:
            repo_url = repo_url.replace("/blob/main", f"/refs/tags/{qaihm_version_tag}")
        if not relative:
            return f"{repo_url.rstrip('/')}/{relative_path.as_posix()}"
        return relative_path

    def get_qaihm_repo_download_url(
        self, model_id: str | None, file_name: str, qaihm_version_tag: str | None = None
    ) -> str:
        repo_url = self.get_qaihm_repo(model_id, False, qaihm_version_tag)
        repo_url = os.path.join(str(repo_url), file_name)
        return repo_url.replace("github.com", "raw.githubusercontent.com")

    def get_website_url(self, model_id: str, relative: bool = False) -> Path | str:
        relative_path = Path(
            self.models_website_relative_path.lstrip("/").format(model_id=model_id)
        )
        if not relative:
            return f"{self.models_website_url.rstrip('/')}/{relative_path.as_posix()}"
        return relative_path

    def get_example_use(self, model_id: str) -> str:
        return self.example_use.lstrip("/").format(model_id=model_id)

    def get_release_asset_filename(
        self,
        model_id: str,
        runtime: TargetRuntime,
        precision: Precision,
        chipset: str | None,
    ) -> str:
        if runtime.is_aot_compiled:
            if chipset is None:
                raise ValueError("Chipset must be provided for AOT compiled runtimes")
            return self.released_asset_with_chipset_filename.format(
                model_id=model_id,
                runtime=runtime.value,
                precision=str(precision),
                chipset_with_underscores=chipset.replace("-", "_"),
            )
        return self.released_asset_filename.format(
            model_id=model_id, runtime=runtime.value, precision=str(precision)
        )

    def get_release_asset_name(
        self,
        model_id: str,
        runtime: TargetRuntime,
        precision: Precision,
        chipset: str | None,
    ) -> str:
        return os.path.splitext(
            self.get_release_asset_filename(model_id, runtime, precision, chipset)
        )[0]

    def get_release_asset_s3_key(
        self,
        model_id: str,
        version: str,
        runtime: TargetRuntime,
        precision: Precision,
        chipset: str | None,
    ) -> str:
        return self.released_asset_folder.format(
            model_id=model_id, version=QAIHMVersion.tag_from_string(version)[1:]
        ) + self.get_release_asset_filename(model_id, runtime, precision, chipset)

    def get_release_asset_url(
        self,
        model_id: str,
        version: str,
        runtime: TargetRuntime,
        precision: Precision,
        chipset: str | None,
    ) -> str:
        return self.get_asset_url(
            self.get_release_asset_s3_key(
                model_id, version, runtime, precision, chipset
            )
        )

    ###
    # Load from CFG
    ###
    @staticmethod
    def from_cfg(
        asset_cfg_path: str = ASSET_BASES_DEFAULT_PATH,
        local_store_path: str = LOCAL_STORE_DEFAULT_PATH,
    ) -> ModelZooAssetConfig:
        # Load CFG and params
        asset_cfg = ModelZooAssetConfig.load_asset_cfg(asset_cfg_path)

        return ModelZooAssetConfig(
            asset_cfg["store_url"],
            asset_cfg["web_asset_folder"],
            asset_cfg["static_web_banner_filename"],
            asset_cfg["animated_web_banner_filename"],
            asset_cfg["model_asset_folder"],
            asset_cfg["dataset_asset_folder"],
            local_store_path,
            asset_cfg["qaihm_repo"],
            asset_cfg["example_use"],
            asset_cfg["huggingface_path"],
            asset_cfg["repo_url"],
            asset_cfg["models_website_url"],
            asset_cfg["models_website_relative_path"],
            asset_cfg["genie_url"],
            asset_cfg["released_asset_folder"],
            asset_cfg["released_asset_filename"],
            asset_cfg["released_asset_with_chipset_filename"],
        )

    ASSET_CFG_SCHEMA = Schema(
        And(
            {
                "store_url": str,
                "web_asset_folder": str,
                "dataset_asset_folder": str,
                "static_web_banner_filename": str,
                "animated_web_banner_filename": str,
                "model_asset_folder": str,
                "qaihm_repo": str,
                "labels_path": str,
                "example_use": str,
                "huggingface_path": str,
                "repo_url": str,
                "models_website_url": str,
                "models_website_relative_path": str,
                "email_template": str,
                "genie_url": str,
                "released_asset_folder": str,
                "released_asset_filename": str,
                "released_asset_with_chipset_filename": str,
            }
        )
    )

    @staticmethod
    def load_asset_cfg(path: str) -> dict[str, Any]:
        data = load_yaml(path)
        try:
            # Validate high level-schema
            ModelZooAssetConfig.ASSET_CFG_SCHEMA.validate(data)
        except SchemaError as e:
            raise ValueError(f"{e.code} in {path}") from None
        return data


ASSET_CONFIG = ModelZooAssetConfig.from_cfg()


class CachedWebAsset:
    """Helper class for downloading files for storage in the QAIHM asset cache."""

    def __init__(
        self,
        url: str,
        local_cache_path: str | os.PathLike,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
        model_downloader: Callable[[str, str, int], str] | None = None,
        downloader_num_retries: int = 4,
        private_s3_key: str | None = None,
        local_cache_extracted_path: str | os.PathLike | None = None,
    ) -> None:
        """
        Parameters
        ----------
        url
            URL to download the asset from.
        local_cache_path
            Path to store the downloaded asset on disk.
        asset_config
            Asset config to use to save this file.
        model_downloader
            Callable to download the file. Defaults to `download_file`.
        downloader_num_retries
            Number of retries when downloading.
        private_s3_key
            If set, the asset will be fetched from the private S3 bucket
            using this key on CI or when the user has valid AWS credentials.
        local_cache_extracted_path
            Path where extracted archive
            contents will live. Defaults to `local_cache_path` with its
            file extension stripped (e.g. `data/foo.zip` → `data/foo`).
            Set this explicitly when the archive already contains a top-level
            directory to avoid double-nesting.
        """
        if private_s3_key and can_access_private_s3():
            from qai_hub_models.utils.aws import QAIHM_PRIVATE_S3_BUCKET

            url = f"s3://{QAIHM_PRIVATE_S3_BUCKET}/{private_s3_key}"
            model_downloader = download_from_private_s3

        self.url = url
        self.local_cache_path = Path(local_cache_path).absolute()
        self.asset_config: ModelZooAssetConfig = asset_config
        self._downloader: Callable = model_downloader or download_file
        self.downloader_num_retries = downloader_num_retries
        self._local_cache_extracted_path: Path | None

        # Determine whether this is an archive, and what path to extract that archive to.
        assert self.local_cache_path.suffixes, (
            "CachedWebAsset does not support fetching directories."
        )
        self.archive_ext: str | None = None
        for ext in [".zip", ".tar", ".tar.gz", ".tgz"]:
            if str(self.local_cache_path).endswith(ext):
                self.archive_ext = ext
                break

        if self.archive_ext:
            if local_cache_extracted_path is not None:
                self._local_cache_extracted_path = Path(
                    local_cache_extracted_path
                ).absolute()
            else:
                self._local_cache_extracted_path = Path(
                    str(self.local_cache_path).removesuffix(self.archive_ext)
                )
        else:
            self._local_cache_extracted_path = None

    def __repr__(self) -> str:
        return self.url

    @staticmethod
    def from_asset_store(
        relative_store_file_path: str,
        num_retries: int = 4,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
    ) -> CachedWebAsset:
        """
        File from the online qaihm asset store.

        Parameters
        ----------
        relative_store_file_path
            Path relative to `qai_hub_models` cache root to store this asset.
            (also relative to the root of the online file store)
        num_retries
            Number of retries when downloading thie file.
        asset_config
            Asset config to use to save this file.

        Returns
        -------
        asset : CachedWebAsset
            CachedWebAsset instance for the file.
        """
        return CachedWebAsset(
            asset_config.get_asset_url(relative_store_file_path),
            Path(relative_store_file_path),
            asset_config,
            download_file,
            num_retries,
        )

    @staticmethod
    def from_google_drive(
        gdrive_file_id: str,
        relative_store_file_path: str | Path,
        num_retries: int = 4,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
    ) -> CachedWebAsset:
        """
        File from google drive.

        Parameters
        ----------
        gdrive_file_id
            Unique identifier of the file in Google Drive.
            Typically found in the URL.
        relative_store_file_path
            Path relative to `qai_hub_models` cache root to store this asset.
        num_retries
            Number of retries when downloading thie file.
        asset_config
            Asset config to use to save this file.

        Returns
        -------
        asset : CachedWebAsset
            CachedWebAsset instance for the file.
        """
        return CachedWebAsset(
            f"https://drive.google.com/uc?id={gdrive_file_id}",
            Path(relative_store_file_path),
            asset_config,
            download_and_cache_google_drive,
            num_retries,
        )

    @property
    def is_extracted(self) -> bool:
        ext_path = self.extracted_path
        return ext_path.is_dir() and any(ext_path.iterdir())

    @property
    def extracted_path(self) -> Path:
        """Get the path of this asset on disk (when extracted)."""
        if self._local_cache_extracted_path is None:
            raise ValueError(
                "Cannot determine extracted path of an asset that is not an archive."
            )
        return self._local_cache_extracted_path

    @property
    def path(self) -> Path:
        """
        Get the path of this asset on disk.

        Returns
        -------
        path : Path
            Path to the asset on disk.

            For archived (.zip, .tar, .etc) assets, path() will return the extracted path if the asset
            has been extracted, and the original archive file's path if it has not been extracted.
        """
        if self.archive_ext is not None and self.is_extracted:
            return self.extracted_path
        return self.local_cache_path

    def fetch(
        self,
        extract: bool = False,
        local_path: str | Path | None = None,
    ) -> Any:
        """
        Fetch this file from the web if it does not exist on disk.

        Parameters
        ----------
        extract
            Extract the asset after downloading it. Ignored if the asset is already extracted.

        local_path
            Path to a local file to use instead of downloading.
            The file is copied into the asset cache so the original is
            not modified.

        Returns
        -------
        path : Any
            Path to the fetched asset on disk. If the asset has been extracted (or extract is True), the extracted path will always be returned.
        """
        if self.archive_ext and self.is_extracted:
            return self.extracted_path

        # Download file
        if not self.local_cache_path.exists():
            # Create dirs
            os.makedirs(os.path.dirname(p=self.local_cache_path), exist_ok=True)

            if local_path is not None:
                shutil.copy2(local_path, self.local_cache_path)
            else:
                # Downloader should return path we expect.
                p1 = self._downloader(self.url, self.local_cache_path)
                assert str(p1) == str(self.local_cache_path)

        # Extract asset if requested
        if extract:
            self.extract()
            return self.extracted_path

        return self.local_cache_path

    def extract(self) -> Path:
        """Extract this asset if it is compressed. Updates the path of this asset to the folder to which the zip file was extracted."""
        if not self.archive_ext:
            raise ValueError("Cannot extract an asset that is not an archive.")
        if not self.is_extracted:
            try:
                if self.archive_ext == ".zip":
                    extract_zip_file(self.local_cache_path, self.extracted_path)
                    os.remove(self.local_cache_path)  # Deletes zip file
                elif self.archive_ext in [".tar", ".tar.gz", ".tgz"]:
                    extract_tar_file(self.local_cache_path, self.extracted_path)
                    os.remove(self.local_cache_path)  # Deletes tar file
                else:
                    raise ValueError(
                        f"Unsupported compressed file type: {self.archive_ext}"
                    )
            except BaseException:
                # Cleanup the folder if the extraction failed, so we don't falsely think the asset was extracted already.
                if self.extracted_path.exists():
                    shutil.rmtree(self.extracted_path)
                raise

        return self.extracted_path


class CachedWebModelAsset(CachedWebAsset):
    """Helper class for downloading files for storage in the QAIHM asset cache."""

    def __init__(
        self,
        url: str,
        model_id: str,
        model_asset_version: int | str,
        filename: Path | str,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
        model_downloader: Callable[[str, str, int], str] | None = None,
        downloader_num_retries: int = 4,
    ) -> None:
        local_cache_path = asset_config.get_local_store_model_path(
            model_id, model_asset_version, filename
        )
        super().__init__(
            url,
            local_cache_path,
            asset_config,
            model_downloader,
            downloader_num_retries,
        )
        self.model_id = model_id
        self.model_version = model_asset_version

    @staticmethod
    def from_asset_store(
        model_id: str,
        model_asset_version: str | int,
        filename: str | Path,
        num_retries: int = 4,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
    ) -> Any:
        """
        File from the online qaihm asset store.

        Parameters
        ----------
        model_id
            Model ID
        model_asset_version
            Asset version for this model.
        filename
            Filename for this asset on disk.
        num_retries
            Number of retries when downloading thie file.
        asset_config
            Asset config to use to save this file.

        Returns
        -------
        asset : Any
            CachedWebModelAsset instance for the file.
        """
        web_store_path = asset_config.get_model_asset_url(
            model_id, model_asset_version, filename
        )
        return CachedWebModelAsset(
            web_store_path,
            model_id,
            model_asset_version,
            filename,
            asset_config,
            download_file,
            num_retries,
        )

    @staticmethod
    def from_google_drive(
        gdrive_file_id: str,
        model_id: str,
        model_asset_version: str | int,
        filename: str,
        num_retries: int = 4,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
    ) -> CachedWebModelAsset:
        """
        File from google drive.

        Parameters
        ----------
        gdrive_file_id
            Unique identifier of the file in Google Drive.
            Typically found in the URL.
        model_id
            Model ID
        model_asset_version
            Asset version for this model.
        filename
            Filename for this asset on disk.
        num_retries
            Number of retries when downloading thie file.
        asset_config
            Asset config to use to save this file.

        Returns
        -------
        asset : CachedWebModelAsset
            CachedWebModelAsset instance for the file.
        """
        return CachedWebModelAsset(
            f"https://drive.google.com/uc?id={gdrive_file_id}",
            model_id,
            model_asset_version,
            filename,
            asset_config,
            download_and_cache_google_drive,
            num_retries,
        )


class CachedWebDatasetAsset(CachedWebAsset):
    """
    Class representing dataset-specific files that needs stored in the local cache once downloaded.

    These files should correspond to a single (or group) of datasets in `qai_hub_models/dataset`.
    """

    def __init__(
        self,
        url: str,
        dataset_id: str,
        dataset_version: int | str,
        filename: str,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
        model_downloader: Callable[[str, str, int], str] | None = None,
        downloader_num_retries: int = 4,
        private_s3_key: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        url
            URL to download the asset from.
        dataset_id
            Dataset ID.
        dataset_version
            Asset version for this dataset.
        filename
            Filename for this asset on disk.
        asset_config
            Asset config to use to save this file.
        model_downloader
            Callable to download the file. Defaults to `download_file`.
        downloader_num_retries
            Number of retries when downloading.
        private_s3_key
            If set, the asset will be fetched from the private S3 bucket
            using this key on CI or when the user has valid AWS credentials.
        """
        local_cache_path = asset_config.get_local_store_dataset_path(
            dataset_id, dataset_version, filename
        )
        super().__init__(
            url,
            local_cache_path,
            asset_config,
            model_downloader,
            downloader_num_retries,
            private_s3_key,
        )
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version

    @staticmethod
    def from_asset_store(
        dataset_id: str,
        dataset_version: str | int,
        filename: str,
        num_retries: int = 4,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
    ) -> CachedWebDatasetAsset:
        """
        File from the online qaihm asset store.

        Parameters
        ----------
        dataset_id
            Dataset ID
        dataset_version
            Asset version for this dataset.
        filename
            Filename for this asset on disk.
        num_retries
            Number of retries when downloading thie file.
        asset_config
            Asset config to use to save this file.

        Returns
        -------
        asset : CachedWebDatasetAsset
            CachedWebDatasetAsset instance for the file.
        """
        web_store_path = asset_config.get_dataset_asset_url(
            dataset_id, dataset_version, filename
        )
        return CachedWebDatasetAsset(
            web_store_path,
            dataset_id,
            dataset_version,
            filename,
            asset_config,
            download_file,
            num_retries,
        )

    @staticmethod
    def from_google_drive(
        gdrive_file_id: str,
        model_id: str,
        model_asset_version: str | int,
        filename: str,
        num_retries: int = 4,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
    ) -> CachedWebDatasetAsset:
        """
        File from google drive.

        Parameters
        ----------
        gdrive_file_id
            Unique identifier of the file in Google Drive.
            Typically found in the URL.
        model_id
            Model ID
        model_asset_version
            Asset version for this model.
        filename
            Filename for this asset on disk.
        num_retries
            Number of retries when downloading thie file.
        asset_config
            Asset config to use to save this file.

        Returns
        -------
        asset : CachedWebDatasetAsset
            CachedWebDatasetAsset instance for the file.
        """
        return CachedWebDatasetAsset(
            f"https://drive.google.com/uc?id={gdrive_file_id}",
            model_id,
            model_asset_version,
            filename,
            asset_config,
            download_and_cache_google_drive,
            num_retries,
        )


def download_from_private_s3(
    s3_url: str, dst_path: str | Path, num_retries: int = 4
) -> str:
    """
    Download a file from S3 to a local path.

    Parameters
    ----------
    s3_url
        S3 URL in the format s3://bucket-name/key.
    dst_path
        Local destination path.
    num_retries
        Unused. Present for compatibility with the downloader interface.

    Returns
    -------
    dst_path : str
        The local path where the file was saved.
    """
    dst_path = str(dst_path)
    if not os.path.exists(dst_path):
        from qai_hub_models.utils.aws import get_qaihm_s3, s3_download

        if not s3_url.startswith("s3://"):
            raise ValueError(f"Expected s3:// URL, got: {s3_url}")
        without_prefix = s3_url[len("s3://") :]
        bucket_name, key = without_prefix.split("/", 1)
        bucket, _ = get_qaihm_s3(bucket_name)
        s3_download(bucket, key, dst_path)
    return dst_path


def download_file(
    web_url: str, dst_path: str, num_retries: int = 4, verbose: bool = True
) -> str:
    """
    Downloads data from the internet and stores in `dst_folder`.
    `dst_folder` should be relative to the local cache root for qai_hub_models.

    Supports resuming partial downloads on connection failures.
    """
    if not os.path.exists(dst_path):
        with qaihm_temp_dir() as tmp_dir:
            tmp_filepath = os.path.join(tmp_dir, Path(dst_path).name)

            for attempt in range(num_retries + 1):
                # Resume from where we left off if the temp file exists
                bytes_downloaded = (
                    os.path.getsize(tmp_filepath) if os.path.exists(tmp_filepath) else 0
                )
                headers = {}
                if bytes_downloaded > 0:
                    headers["Range"] = f"bytes={bytes_downloaded}-"

                try:
                    response = requests.get(web_url, stream=True, headers=headers)
                    if response.status_code not in (200, 206):
                        raise ValueError(
                            f"Unable to download file at {web_url}"
                            f" (status {response.status_code})"
                        )

                    # If server doesn't support range requests and returned
                    # full content, restart from scratch.
                    if bytes_downloaded > 0 and response.status_code == 200:
                        bytes_downloaded = 0

                    # Prefer Content-Range header for total file size
                    # (e.g. "bytes 1000-1999/5000" -> 5000) when available.
                    content_range = response.headers.get("content-range", "")
                    range_match = re.search(r"/(\d+)\s*$", content_range)
                    if range_match:
                        total_size = int(range_match.group(1))
                    else:
                        total_size = bytes_downloaded + int(
                            response.headers.get("content-length", 0)
                        )
                    block_size = 1024

                    mode = "ab" if bytes_downloaded > 0 else "wb"
                    with tqdm(
                        total=total_size,
                        initial=bytes_downloaded,
                        unit="B",
                        unit_scale=True,
                    ) as progress_bar:
                        progress_bar.set_description(
                            f"Downloading data at {web_url} to {dst_path}"
                        )
                        with open(tmp_filepath, mode) as file:
                            for data in response.iter_content(block_size):
                                if not IsOnCIEnvvar.get():
                                    progress_bar.update(len(data))
                                file.write(data)

                        if IsOnCIEnvvar.get():
                            progress_bar.set_postfix_str("Done", refresh=False)
                            progress_bar.update(total_size)
                        else:
                            progress_bar.set_postfix_str("Done")

                    # Verify the download is complete. A server may
                    # return a partial range even when asked for the
                    # full remainder of the file.
                    actual_size = os.path.getsize(tmp_filepath)
                    if total_size and actual_size < total_size:
                        if attempt < num_retries:
                            print(
                                f"Download incomplete ({actual_size}/{total_size} bytes), "
                                f"retrying ({attempt + 1}/{num_retries})..."
                            )
                            continue
                        raise ValueError(
                            f"Download incomplete after {num_retries} retries: "
                            f"got {actual_size}/{total_size} bytes from {web_url}"
                        )

                    # Download completed successfully
                    break

                except (
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ContentDecodingError,
                    requests.exceptions.Timeout,
                ) as e:
                    if attempt < num_retries:
                        print(
                            f"Download interrupted ({e.__class__.__name__}), "
                            f"retrying ({attempt + 1}/{num_retries})..."
                        )
                    else:
                        raise

            shutil.move(tmp_filepath, dst_path)
    return dst_path


def download_and_cache_google_drive(
    web_url: str, dst_path: str, num_retries: int = 4
) -> str:
    """
    Download file from google drive to the local directory.

    Parameters
    ----------
    web_url
        URL of the file in Google Drive.
    dst_path
        Destination path where the file will be saved.
    num_retries
        Number of times to retry in case download fails.

    Returns
    -------
    dst_path : str
        Filepath within the local filesystem.
    """
    for i in range(num_retries):
        print(f"Downloading data at {web_url} to {dst_path}... ")
        with suppress(Exception):
            gdown.download(web_url, dst_path, quiet=False)
        if os.path.exists(dst_path):
            print("Done")
            return dst_path
        print(f"Failed to download file at {web_url}")
        if i < num_retries - 1:
            print("Retrying in 3 seconds.")
            time.sleep(3)
    return dst_path


def copyfile(src: str, dst: str, num_retries: int = 4) -> str:
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copyfile(src, dst)
    return dst


def extract_tar_file(
    tar_path: os.PathLike | str, out_path: str | os.PathLike | None = None
) -> Path:
    """
    Extract a tar file's contents into out_path.

    If the archive contains a single top-level directory, its contents
    are unwrapped directly into out_path. If the archive contains multiple
    top-level entries, they are all placed into out_path as-is.

    Parameters
    ----------
    tar_path
        Path to the tar file.
    out_path
        Destination directory. If None, defaults to the tar file path
        without its archive extension. Must not already exist.

    Returns
    -------
    out_path : Path
        Path to the extracted directory.
    """
    archive_ext: str | None = None
    for ext in [".tar", ".tar.gz", ".tgz"]:
        if str(tar_path).endswith(ext):
            archive_ext = ext
            break

    if not archive_ext:
        raise ValueError(f"{tar_path} is not an archive.")

    out_path = Path(out_path if out_path else str(tar_path).removesuffix(archive_ext))

    if out_path.exists():
        raise ValueError(f"Cannot extract to an existing directory: {out_path}")

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tar_path) as f:
            f.extractall(tmp)

        top_level_files = list(Path(tmp).iterdir())
        if len(top_level_files) == 1 and top_level_files[0].is_dir():
            # Single top-level dir. Extract contents directly to out_path.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(top_level_files[0], out_path)
        else:
            # Multiple top-level entries: extract everything directly into out_path
            out_path.mkdir(parents=True, exist_ok=True)
            for file in top_level_files:
                shutil.move(file, out_path / file.name)

    return out_path


def extract_zip_file(
    filepath_str: os.PathLike | str, out_path: str | os.PathLike | None = None
) -> Path:
    """
    Extract a zip file's contents into out_path.

    If the archive contains a single top-level directory, its contents
    are unwrapped directly into out_path. If the archive contains multiple
    top-level entries, they are all placed into out_path as-is.

    Parameters
    ----------
    filepath_str
        Path to the zip file.
    out_path
        Destination directory. If None, defaults to the zip file path
        without the .zip extension. Must not already exist.

    Returns
    -------
    out_path : Path
        Path to the extracted directory.
    """
    filepath = Path(filepath_str)
    out_path = Path(out_path if out_path else filepath.parent / filepath.stem)

    if out_path.exists():
        raise ValueError(f"Cannot extract to an existing directory: {out_path}")

    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(filepath, "r") as zf:
            zf.extractall(path=tmp)

        top_level_files = list(Path(tmp).iterdir())
        if len(top_level_files) == 1 and top_level_files[0].is_dir():
            # Single top-level dir. Extract contents directly to out_path.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(top_level_files[0], out_path)
        else:
            # Multiple top-level entries: extract everything directly into out_path
            out_path.mkdir(parents=True, exist_ok=True)
            for file in top_level_files:
                shutil.move(file, out_path / file.name)

    return out_path


# TODO (#12708): Remove this and rely on client
def zip_model(output_dir_path: PathLike, model_path: PathLike) -> str:
    model_path = os.path.realpath(model_path)
    package_name = os.path.basename(model_path)
    compresslevel = 1

    output_path = os.path.join(output_dir_path, package_name + ".zip")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel
    ) as f:
        walk: Iterable[tuple[str, list[str], list[str]]]
        if os.path.isfile(model_path):
            root_path = os.path.dirname(model_path)
            walk = [(root_path, [], [model_path])]
        else:
            root_path = os.path.join(model_path, "..")
            walk = os.walk(model_path)
        for root, _, files in walk:
            # Create directory entry (can use f.mkdir from Python 3.11)
            rel_root = os.path.relpath(root, root_path)
            if rel_root != ".":
                f.writestr(rel_root + "/", "")
            for file in files:
                f.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), root_path),
                )
    return output_path


def callback_with_retry(
    num_retries: int,
    callback: Callable,
    *args: Any | None,
    **kwargs: Any | None,
) -> Any:
    """Allow retries when running provided function."""
    if num_retries == 0:
        raise RuntimeError(f"Unable to run function {callback.__name__}")
    try:
        return callback(*args, **kwargs)
    except Exception as error:
        error_msg = f"Error: {getattr(error, 'message', str(error))}"
        print(error_msg)
        if hasattr(error, "status_code"):
            print(f"Status code: {error.status_code}")
        time.sleep(10)
        return callback_with_retry(num_retries - 1, callback, *args, **kwargs)


@contextmanager
def qaihm_temp_dir(debug_base_dir: str | None = None) -> Generator[str, None, None]:
    """
    Keep temp file under LOCAL_STORE_DEFAULT_PATH instead of /tmp which has
    limited space.

    Parameters
    ----------
    debug_base_dir
        If provided, use this directory instead of creating a temp directory.
        If None, creates a temporary directory as usual.

    Yields
    ------
    str
        Path to the temporary directory.
    """
    if debug_base_dir is not None:
        # Use the debug directory directly
        os.makedirs(debug_base_dir, exist_ok=True)
        yield debug_base_dir
    else:
        # Use temporary directory as before
        path = os.path.join(LOCAL_STORE_DEFAULT_PATH, "tmp")
        os.makedirs(path, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path) as tempdir:
            yield tempdir


PathType = str | Path | CachedWebAsset

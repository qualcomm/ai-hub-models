# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Resolution tests for ``qai-hub-models configure-dataset``."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models.test.mock_imports import install_mock_packages

# cv2 is only needed to import the dataset modules, not by these tests.
install_mock_packages(["cv2"])

from qai_hub_models.cli.configure_dataset import _resolve_class, main  # noqa: E402
from qai_hub_models.datasets.cityscapes.cityscapes import (  # noqa: E402
    CITYSCAPES_CONFIGURE_FILES,
    CITYSCAPES_GT_ASSET,
    CITYSCAPES_IMAGES_ASSET,
    CityscapesDataset,
)
from qai_hub_models.utils.private_asset_loaders import (  # noqa: E402
    UnfetchableDatasetError,
    configure_dataset_command,
)

_STANDALONE_DATASET = """
from qai_hub_models.utils.base_dataset import BaseDataset


class MyEvalDataset(BaseDataset):
    def __init__(self, split=None, input_spec=None, files=None):
        self.configured_with = files

    def _download_data(self) -> None: ...

    def __len__(self) -> int:
        return 0

    def __getitem__(self, i):
        raise IndexError

    @staticmethod
    def default_samples_per_job() -> int:
        return 1

    @classmethod
    def configure(cls, files) -> None:
        cls(files=files)
"""


def _write_recipe(root: Path) -> Path:
    recipe = root / "my_recipe"
    recipe.mkdir()
    (recipe / "__init__.py").write_text("")
    (recipe / "dataset.py").write_text(_STANDALONE_DATASET)
    return recipe


def test_resolves_an_in_tree_dotted_path() -> None:
    resolved = _resolve_class(
        "qai_hub_models.datasets.cityscapes.cityscapes.CityscapesDataset"
    )
    assert resolved is CityscapesDataset


def test_a_bare_name_is_rejected() -> None:
    """No dataset registry exists to resolve one against."""
    with pytest.raises(CLIUserError, match="dotted import path"):
        _resolve_class("cityscapes")


def test_unimportable_module_names_the_module() -> None:
    with pytest.raises(CLIUserError, match="Could not import module"):
        _resolve_class("qai_hub_models.datasets.nope_missing.Thing")


def test_missing_attribute_names_the_attribute() -> None:
    with pytest.raises(CLIUserError, match="has no attribute"):
        _resolve_class("qai_hub_models.datasets.cityscapes.cityscapes.NotAClass")


def test_a_non_dataset_class_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be a subclass"):
        _resolve_class("pathlib.Path")


def test_resolves_a_standalone_recipe_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A console script does not put cwd on sys.path; the resolver must.

    Without this, a dataset class in a standalone recipe folder -- the case the
    generated command is meant to cover -- would never resolve.
    """
    _write_recipe(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "my_recipe", raising=False)
    monkeypatch.delitem(sys.modules, "my_recipe.dataset", raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(tmp_path)])

    resolved = _resolve_class("my_recipe.dataset.MyEvalDataset")
    assert resolved.__name__ == "MyEvalDataset"


def test_main_forwards_files_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_recipe(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "my_recipe", raising=False)
    monkeypatch.delitem(sys.modules, "my_recipe.dataset", raising=False)

    cls = _resolve_class("my_recipe.dataset.MyEvalDataset")
    with patch.object(cls, "configure") as mock_configure:
        main(["my_recipe.dataset.MyEvalDataset", "--files", "a.zip", "b.zip"])
    mock_configure.assert_called_once_with(["a.zip", "b.zip"])


# ── the generated instruction ───────────────────────────────────────


def test_generated_command_uses_the_runtime_import_path() -> None:
    assert configure_dataset_command(CityscapesDataset, CITYSCAPES_CONFIGURE_FILES) == (
        "qai-hub-models configure-dataset "
        "qai_hub_models.datasets.cityscapes.cityscapes.CityscapesDataset "
        "--files leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip"
    )


def test_generated_command_round_trips_through_the_resolver() -> None:
    """The command the error prints must be one the command can consume."""
    dotted = configure_dataset_command(
        CityscapesDataset, CITYSCAPES_CONFIGURE_FILES
    ).split()[2]
    assert _resolve_class(dotted) is CityscapesDataset


def test_error_appends_the_command_when_given_a_class() -> None:
    err = UnfetchableDatasetError(
        "cityscapes",
        ["Make an account at https://example.com"],
        CityscapesDataset,
        configure_files=CITYSCAPES_CONFIGURE_FILES,
    )
    message = str(err)
    assert "1. Make an account at https://example.com" in message
    assert "2. Run `qai-hub-models configure-dataset" in message
    assert "python -m" not in message


def test_error_without_a_class_is_unchanged() -> None:
    """Legacy behavior: no class, no generated step."""
    err = UnfetchableDatasetError(
        "cityscapes", ["Make an account"], configure_files=CITYSCAPES_CONFIGURE_FILES
    )
    assert "configure-dataset" not in str(err)


def test_internal_only_dataset_ignores_the_class() -> None:
    """installation_steps=None means the dataset is not publicly available."""
    err = UnfetchableDatasetError(
        "secret",
        None,
        CityscapesDataset,
        configure_files=CITYSCAPES_CONFIGURE_FILES,
    )
    assert "Qualcomm-internal usage only" in str(err)
    assert "configure-dataset" not in str(err)


def test_real_dataset_error_names_its_files_in_order() -> None:
    """End to end on a live two-file dataset, not a synthetic class."""
    err = UnfetchableDatasetError(
        "cityscapes",
        ["Make an account"],
        CityscapesDataset,
        configure_files=CITYSCAPES_CONFIGURE_FILES,
    )
    assert "--files leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip`" in str(err)


def test_both_cityscapes_assets_carry_the_same_file_list() -> None:
    """Either asset can be the one that fails, so both must name the full set."""
    for asset in (CITYSCAPES_IMAGES_ASSET, CITYSCAPES_GT_ASSET):
        err = asset.access_denied_error
        assert isinstance(err, UnfetchableDatasetError)
        assert err.configure_files == CITYSCAPES_CONFIGURE_FILES


def test_cityscapes_file_count_matches_its_configure_arity() -> None:
    """A wrong-length list would print a command that fails the arity check."""
    with pytest.raises(ValueError, match="expects 2 file"):
        CityscapesDataset.configure(["only-one.zip"])
    assert len(CITYSCAPES_CONFIGURE_FILES) == 2


# ── the full loop: real fetch failure -> printed command -> configure() ──


@pytest.fixture
def unreachable_cityscapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point both Cityscapes assets at an empty store and deny S3.

    Reproduces what an external user hits, without consulting the real cache
    (whose contents would otherwise decide whether this test exercises
    anything) and without deleting it.
    """
    store = tmp_path / "store" / "data"
    for asset in (CITYSCAPES_IMAGES_ASSET, CITYSCAPES_GT_ASSET):
        archive = store / asset.local_cache_path.name
        monkeypatch.setattr(asset, "local_cache_path", archive)
        monkeypatch.setattr(
            asset,
            "_local_cache_extracted_path",
            store / archive.name.removesuffix(".zip"),
        )
    monkeypatch.setattr(
        "qai_hub_models.utils.private_asset_loaders.can_access_private_s3",
        lambda: False,
    )
    monkeypatch.setattr(
        "qai_hub_models.utils.private_asset_loaders.is_internal_repo", lambda: False
    )


def test_two_file_dataset_prints_a_command_that_actually_runs(
    unreachable_cityscapes: None,
) -> None:
    """The generated instruction is executable, and its file order is real.

    Goes through the product path -- asset fetch denied, `download_data`
    rebuilding the error with the concrete class -- then runs the command it
    printed and checks `configure()` receives those files in that order.
    """
    with pytest.raises(UnfetchableDatasetError) as caught:
        CityscapesDataset()
    message = str(caught.value)

    printed = re.search(r"Run `([^`]+)`", message)
    assert printed is not None, f"no command in message:\n{message}"
    argv = shlex.split(printed.group(1))
    assert argv[:2] == ["qai-hub-models", "configure-dataset"]

    with patch.object(CityscapesDataset, "configure") as configure:
        main(argv[2:])

    files = configure.call_args.args[0]
    assert files == CITYSCAPES_CONFIGURE_FILES
    # Same order the human steps list them in: images, then ground truth.
    assert [message.index(f) for f in files] == sorted(message.index(f) for f in files)


def test_printed_filenames_are_the_ones_the_steps_tell_you_to_download(
    unreachable_cityscapes: None,
) -> None:
    """Guards against a configure_files list invented rather than read off upstream."""
    with pytest.raises(UnfetchableDatasetError) as caught:
        CityscapesDataset()
    human_steps = "\n".join(
        line
        for line in str(caught.value).splitlines()
        if "configure-dataset" not in line
    )
    for filename in CITYSCAPES_CONFIGURE_FILES:
        assert filename in human_steps


def test_the_second_asset_can_be_the_one_that_fails(
    unreachable_cityscapes: None,
) -> None:
    """With images already present, ground truth is what raises -- same file list.

    Bypasses ``__init__`` so ``download_data`` cannot wipe the store it just
    populated; the fetch order under test is ``_download_data``'s.
    """
    (CITYSCAPES_IMAGES_ASSET.extracted_path / "train").mkdir(parents=True)
    dataset = CityscapesDataset.__new__(CityscapesDataset)
    dataset.input_images_zip = None
    dataset.input_gt_zip = None

    with pytest.raises(UnfetchableDatasetError) as caught:
        dataset._download_data()

    assert caught.value is CITYSCAPES_GT_ASSET.access_denied_error
    assert caught.value.configure_files == CITYSCAPES_CONFIGURE_FILES

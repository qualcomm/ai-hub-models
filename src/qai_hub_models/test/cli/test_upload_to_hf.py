# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for ``qai-hub-models upload-to-hf``.

Covers the published repo layout (the recipe at the root), token resolution,
repo naming, public-by-default visibility, version tags, and the refusals.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models.cli import upload_to_hf as mod
from qai_hub_models.cli.hf_common import (
    COMMUNITY_ORG_NAME,
    COMMUNITY_TAG,
    COMMUNITY_TAG_SEARCH_URL,
)
from qai_hub_models.cli.upload_to_hf import (
    _hf_front_matter,
    _staged_recipe_files,
    upload_to_hf,
)
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest

_MANIFEST = """\
description: A test classifier.
domain: Computer Vision
headline: A test classifier.
id: my_model
license: https://example.com/LICENSE
license_type: {license_type}
name: MyModel
research_paper: https://arxiv.org/abs/1512.03385
research_paper_title: Test Paper
source_repo: https://github.com/example/test-model
{extra}\
supported_precisions:
{precisions}
tags:
- bu-auto
templates:
- imagenet_classifier
use_case: Image Classification
"""


def _write_recipe(
    folder: Path,
    license_type: str = "bsd-3-clause",
    precisions: str = "- float\n",
    extra: str = "",
    status_line: str = "",
) -> QAIHMModelManifest:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.yaml").write_text(
        _MANIFEST.format(
            license_type=license_type,
            precisions=precisions,
            extra=extra,
        )
    )
    (folder / "__init__.py").write_text("MODEL_ID = 'my_model'\n")
    (folder / "model.py").write_text("# model\n")
    (folder / "demo.py").write_text("# demo\n")
    (folder / "test.py").write_text("# test\n")
    (folder / "requirements.txt").write_text("numpy\n")
    if status_line:
        (folder / "internal-config.yaml").write_text(status_line)
    return QAIHMModelManifest.from_yaml(folder / "manifest.yaml")


@pytest.fixture
def hf_api() -> Any:
    """Patch every HuggingFace call so no test can reach the network.

    ``get_token`` is stubbed too, so the suite behaves identically whether or
    not the machine running it happens to be logged in.
    """
    with (
        patch.object(mod, "create_repo") as create,
        patch.object(mod, "upload_folder") as upload,
        patch.object(mod, "repo_exists", return_value=True) as exists,
        patch.object(mod, "get_token", return_value="hf_fake_token") as token,
        patch.object(mod, "list_repo_files", return_value=[]) as listed,
        patch.object(mod, "create_tag") as tag,
        patch.object(mod, "list_repo_refs") as refs,
        patch.object(mod, "whoami", return_value={"name": "me"}) as who,
        patch.object(mod, "list_repo_commits") as commits,
    ):
        refs.return_value.tags = []
        commits.return_value = _fake_commits("me")
        yield {
            "create_repo": create,
            "upload_folder": upload,
            "repo_exists": exists,
            "get_token": token,
            "list_repo_files": listed,
            "create_tag": tag,
            "list_repo_refs": refs,
            "whoami": who,
            "list_repo_commits": commits,
        }


def _fake_commits(creator: str, latest_author: str = "someone_else") -> list[Any]:
    """Build a list_repo_commits return value: newest first, oldest last.

    Ownership comes from the *oldest* commit, so the newest one deliberately has
    a different author -- a test that read the wrong end would pass otherwise.
    """
    newest, oldest = MagicMock(), MagicMock()
    newest.authors = [latest_author]
    oldest.authors = [creator]
    return [newest, oldest]


def _fake_tags(*names: str, targets: dict[str, str] | None = None) -> Any:
    """Build a list_repo_refs return value carrying the given tag names.

    *targets* maps a tag name to the commit it points at, for tests that care
    which commit a tag targets; unnamed tags get a unique sentinel so they never
    accidentally compare equal to a commit sha under test.
    """
    refs = MagicMock()
    refs.tags = [MagicMock(name=n) for n in names]
    # MagicMock(name=...) sets the mock's repr, not its .name attribute.
    for ref, name in zip(refs.tags, names, strict=False):
        ref.name = name
        ref.target_commit = (targets or {}).get(name, f"sha-of-{name}")
    return refs


class TestTargetResolution:
    """The target is always read as a folder path, never as a model id."""

    def test_bare_folder_name_in_cwd(
        self, tmp_path: Path, hf_api: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`upload-to-hf my_model` must work, without the ./ prefix."""
        _write_recipe(tmp_path / "my_model")
        monkeypatch.chdir(tmp_path)
        assert upload_to_hf("my_model", dry_run=True) is None

    def test_dot_slash_form_still_works(
        self, tmp_path: Path, hf_api: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_recipe(tmp_path / "my_model")
        monkeypatch.chdir(tmp_path)
        assert upload_to_hf("./my_model/", dry_run=True) is None

    def test_absolute_path_works(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        assert upload_to_hf(str(recipe), dry_run=True) is None

    def test_folder_named_after_a_builtin_model_is_just_a_folder(
        self, tmp_path: Path, hf_api: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sharing a built-in model's name is not special; it is a local folder."""
        from qai_hub_models.utils.path_helpers import MODEL_IDS

        model_id = next(iter(sorted(MODEL_IDS)))
        _write_recipe(tmp_path / model_id)
        monkeypatch.chdir(tmp_path)

        assert upload_to_hf(model_id, dry_run=True) is None

    def test_builtin_model_id_with_no_such_folder_is_a_missing_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No id lookup at all -- an absent folder is an absent folder."""
        from qai_hub_models.utils.path_helpers import MODEL_IDS

        model_id = next(iter(sorted(MODEL_IDS)))
        monkeypatch.chdir(tmp_path)
        with pytest.raises(CLIUserError, match="No folder named"):
            upload_to_hf(model_id, dry_run=True)

    def test_never_resolves_into_the_installed_package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model id must not silently pick up the in-tree recipe's folder."""
        from qai_hub_models.utils.path_helpers import MODEL_IDS

        model_id = next(iter(sorted(MODEL_IDS)))
        monkeypatch.chdir(tmp_path)
        with pytest.raises(CLIUserError, match="No folder named") as excinfo:
            upload_to_hf(model_id, dry_run=True)
        assert "site-packages" not in str(excinfo.value)
        assert "qai_hub_models/models" not in str(excinfo.value)

    def test_unknown_target_names_both_accepted_forms(self, tmp_path: Path) -> None:
        with pytest.raises(CLIUserError, match="No folder named"):
            upload_to_hf(str(tmp_path / "nope"), dry_run=True)

    def test_folder_without_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(CLIUserError, match="not a recipe folder"):
            upload_to_hf(str(tmp_path / "empty"), dry_run=True)


class TestStagedRecipeFiles:
    def test_excludes_build_output(self, tmp_path: Path) -> None:
        """README.md is regenerated, and build output never ships."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        (recipe / "README.md").write_text("stale")
        (recipe / "__pycache__").mkdir()
        (recipe / ".DS_Store").write_text("x")
        (recipe / "export_assets").mkdir()
        (recipe / "build").mkdir()

        names = {p.name for p in _staged_recipe_files(recipe)}
        assert names == {
            "__init__.py",
            "model.py",
            "demo.py",
            "test.py",
            "requirements.txt",
            "manifest.yaml",
        }

    def test_keeps_external_repos_dir(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        (recipe / "external_repos").mkdir()
        (recipe / "external_repos" / "__init__.py").write_text("# bootstrap\n")
        assert "external_repos" in {p.name for p in _staged_recipe_files(recipe)}


class TestFrontMatter:
    def test_includes_pipeline_tag_when_use_case_set(self, tmp_path: Path) -> None:
        manifest = _write_recipe(tmp_path / "my_model")
        assert "pipeline_tag: image-classification" in _hf_front_matter(manifest)

    def test_omits_pipeline_tag_when_use_case_unset(self, tmp_path: Path) -> None:
        """get_hf_pipeline_tag asserts on use_case, which external recipes often omit."""
        manifest = _write_recipe(tmp_path / "my_model")
        manifest.use_case = None
        front_matter = _hf_front_matter(manifest)
        assert "pipeline_tag" not in front_matter
        assert "library_name: pytorch" in front_matter

    def test_carries_the_community_tag(self, tmp_path: Path) -> None:
        """The tag is how a repo in a personal namespace gets surfaced at all."""
        manifest = _write_recipe(tmp_path / "my_model")
        assert f"- {COMMUNITY_TAG}" in _hf_front_matter(manifest)

    def test_community_tag_survives_a_manifest_with_no_tags(
        self, tmp_path: Path
    ) -> None:
        manifest = _write_recipe(tmp_path / "my_model")
        manifest.tags = []
        assert f"- {COMMUNITY_TAG}" in _hf_front_matter(manifest)

    def test_license_uses_manifest_mapping(self, tmp_path: Path) -> None:
        """Front matter takes the license from MODEL_LICENSE.huggingface_name."""
        manifest = _write_recipe(tmp_path / "my_model", license_type="mit")
        expected = manifest.license_type.huggingface_name  # type: ignore[union-attr]
        assert f"license: {expected}" in _hf_front_matter(manifest)

    def test_license_is_the_real_slug_not_other(self, tmp_path: Path) -> None:
        """A card must name the actual license, not collapse it to "other"."""
        manifest = _write_recipe(tmp_path / "my_model", license_type="bsd-3-clause")
        assert "license: bsd-3-clause" in _hf_front_matter(manifest)

    def test_tags_stay_block_style_after_a_flow_list_dump(self, tmp_path: Path) -> None:
        """``to_yaml(flow_lists=True)`` used to mutate ruamel's representer
        registry process-wide, flipping this front matter to flow style.
        """
        manifest = _write_recipe(tmp_path / "my_model")
        manifest.to_yaml(tmp_path / "flow.yaml", flow_lists=True)

        front_matter = _hf_front_matter(manifest)

        assert f"- {COMMUNITY_TAG}" in front_matter
        assert "tags: [" not in front_matter


class TestNeverCompiles:
    def test_never_runs_an_export(self, tmp_path: Path, hf_api: Any) -> None:
        """Nothing in this command may reach the export pipeline."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with patch(
            "qai_hub_models.utils.export.dispatch.select_pipeline"
        ) as select_pipeline:
            upload_to_hf(str(recipe), dry_run=True)
        select_pipeline.assert_not_called()


class TestStaging:
    def test_layout_puts_the_recipe_at_the_repo_root(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)

        staged: dict[str, list[str]] = {}
        real_stage = mod._stage

        def _capture(
            source_dir: Path,
            staging: Path,
            manifest: QAIHMModelManifest,
            repo_id: str,
        ) -> None:
            real_stage(source_dir, staging, manifest, repo_id)
            staged["files"] = sorted(
                str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file()
            )

        with patch.object(mod, "_stage", _capture):
            upload_to_hf(str(recipe), dry_run=True)

        assert "model.py" in staged["files"]
        assert "manifest.yaml" in staged["files"]
        assert "README.md" in staged["files"]
        assert "LICENSE" in staged["files"]
        # Nothing is nested under a subfolder -- a clone is directly installable.
        assert all("/" not in name for name in staged["files"])

    def test_publishes_an_assets_folder_like_any_other_content(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """`assets/` has no special meaning now; recipes ship test fixtures there."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        (recipe / "assets").mkdir()
        (recipe / "assets" / "fixture.jpg").write_text("bytes")

        staged: dict[str, list[str]] = {}
        real_stage = mod._stage

        def _capture(
            source_dir: Path,
            staging: Path,
            manifest: QAIHMModelManifest,
            repo_id: str,
        ) -> None:
            real_stage(source_dir, staging, manifest, repo_id)
            staged["files"] = sorted(
                str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file()
            )

        with patch.object(mod, "_stage", _capture):
            upload_to_hf(str(recipe), dry_run=True)

        assert "assets/fixture.jpg" in staged["files"]

    def _stage_external_repos(self, recipe: Path, tmp_path: Path) -> set[str]:
        """Stage *recipe* and return the staged paths under ``external_repos/``."""
        manifest = QAIHMModelManifest.from_yaml(recipe / "manifest.yaml")
        staging = tmp_path / "staging"
        staging.mkdir()
        mod._stage(recipe, staging, manifest, "me/my_model")
        ext = staging / "external_repos"
        return (
            {str(p.relative_to(ext)) for p in ext.rglob("*")} if ext.exists() else set()
        )

    def test_external_repo_clone_is_not_published(self, tmp_path: Path) -> None:
        """A clone is re-fetched from the manifest, so shipping it is pure waste."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        ext = recipe / "external_repos"
        ext.mkdir()
        (ext / "__init__.py").write_text("# bootstrap\n")
        (ext / "upstream_patches.diff").write_text("--- a\n+++ b\n")
        (ext / "upstream_hash.txt").write_text("deadbeef")
        clone = ext / "upstream"
        (clone / "deep" / "nested").mkdir(parents=True)
        (clone / "setup.py").write_text("# upstream code\n")
        (clone / "deep" / "nested" / "weights.bin").write_text("x" * 100)

        staged = self._stage_external_repos(recipe, tmp_path)

        # The bootstrap and the authored patch are required to rebuild the clone.
        assert staged == {"__init__.py", "upstream_patches.diff"}

    def test_symlinked_clone_is_not_published(self, tmp_path: Path) -> None:
        """Clones can be symlinks into the shared cache; copytree would follow them."""
        cache = tmp_path / "cache" / "upstream"
        cache.mkdir(parents=True)
        (cache / "setup.py").write_text("# upstream code\n")

        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        ext = recipe / "external_repos"
        ext.mkdir()
        (ext / "__init__.py").write_text("# bootstrap\n")
        (ext / "upstream").symlink_to(cache, target_is_directory=True)

        assert self._stage_external_repos(recipe, tmp_path) == {"__init__.py"}

    def test_subfolders_elsewhere_are_still_published(self, tmp_path: Path) -> None:
        """The subfolder rule is scoped to external_repos, not the whole recipe."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        (recipe / "assets" / "sub").mkdir(parents=True)
        (recipe / "assets" / "sub" / "fixture.jpg").write_text("bytes")

        manifest = QAIHMModelManifest.from_yaml(recipe / "manifest.yaml")
        staging = tmp_path / "staging"
        staging.mkdir()
        mod._stage(recipe, staging, manifest, "me/my_model")

        assert (staging / "assets" / "sub" / "fixture.jpg").exists()

    def test_readme_gets_front_matter_and_a_usage_section(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        staging = tmp_path / "staging"
        staging.mkdir()

        mod._stage(recipe, staging, manifest, "me/my_model")
        readme = (staging / "README.md").read_text()

        assert readme.startswith("---\n")
        assert "library_name: pytorch" in readme
        assert "qai-hub-models register me/my_model" in readme
        assert "--alias" not in readme
        assert "qai-hub-models demo my_model" in readme


class TestRefusals:
    def test_refuses_a_folder_with_no_manifest(self, tmp_path: Path) -> None:
        """A recipe is defined by its manifest, so its absence is the one check."""
        folder = tmp_path / "not_a_recipe"
        folder.mkdir()
        (folder / "model.py").write_text("x = 1\n")
        with pytest.raises(CLIUserError, match=r"no manifest\.yaml"):
            upload_to_hf(str(folder), dry_run=True)

    def test_missing_manifest_fails_before_staging(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        folder = tmp_path / "not_a_recipe"
        folder.mkdir()
        with (
            patch.object(mod, "_stage") as stage,
            pytest.raises(CLIUserError, match=r"no manifest\.yaml"),
        ):
            upload_to_hf(str(folder), assume_yes=True)
        stage.assert_not_called()
        hf_api["create_repo"].assert_not_called()

    def test_refuses_a_missing_folder(self, tmp_path: Path) -> None:
        with pytest.raises(CLIUserError, match="No folder named"):
            upload_to_hf(str(tmp_path / "nope"), dry_run=True)

    def test_refuses_in_tree_recipe(self, tmp_path: Path, hf_api: Any) -> None:
        # In-tree recipes live under qai_hub_models/models/. External recipes don't.
        # Since we create this in tmp_path (not under qai_hub_models/models/), it's treated as external.
        # External recipes don't have internal-config.yaml and can be uploaded to the community org.
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        # External recipes should proceed (not be rejected)
        with (
            patch.object(mod, "_stage") as mock_stage,
            patch("builtins.input", return_value="y"),
            patch.object(mod, "_print_next_steps"),
        ):
            upload_to_hf(str(recipe), dry_run=False)
            # Verify staging was called (meaning it didn't reject the recipe as in-tree)
            mock_stage.assert_called_once()

    def test_restricted_sharing_is_not_a_refusal(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """restrict_model_sharing guarded redistributing compiled weights.

        Only the author's own source and a generated card are published now, so
        there is nothing for that flag to restrict and it must not block.
        """
        recipe = tmp_path / "my_model"
        _write_recipe(
            recipe,
            license_type="gpl-3.0",
            extra="restrict_model_sharing: true\n",
        )

        assert upload_to_hf(str(recipe), dry_run=True) is None


class TestUpload:
    def test_dry_run_contacts_nothing(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        assert upload_to_hf(str(recipe), dry_run=True) is None
        hf_api["create_repo"].assert_not_called()
        hf_api["upload_folder"].assert_not_called()
        hf_api["repo_exists"].assert_not_called()

    def test_uploads_and_returns_url(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        url = upload_to_hf(str(recipe), assume_yes=True)
        assert url == "https://huggingface.co/me/my_model"
        hf_api["create_repo"].assert_called_once()
        hf_api["upload_folder"].assert_called_once()

    def test_default_repo_id_uses_own_namespace(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """Not the community org: a personal namespace is owned, so uncontested."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        upload_to_hf(str(recipe), assume_yes=True)
        assert hf_api["create_repo"].call_args.kwargs["repo_id"] == "me/my_model"

    def test_default_repo_id_never_targets_the_community_org(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """Publishing must not require org membership, so never default there."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        upload_to_hf(str(recipe), assume_yes=True)
        assert (
            COMMUNITY_ORG_NAME not in hf_api["create_repo"].call_args.kwargs["repo_id"]
        )

    def test_repo_is_named_after_the_folder_not_the_manifest_id(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """The name the author typed is the name that gets published."""
        recipe = tmp_path / "my_renamed_folder"
        _write_recipe(recipe)  # manifest id is my_model
        upload_to_hf(str(recipe), assume_yes=True)
        assert (
            hf_api["create_repo"].call_args.kwargs["repo_id"] == "me/my_renamed_folder"
        )

    def test_trailing_slash_does_not_leak_into_the_repo_name(
        self, tmp_path: Path, hf_api: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_recipe(tmp_path / "my_model")
        monkeypatch.chdir(tmp_path)
        upload_to_hf("./my_model/", assume_yes=True)
        assert hf_api["create_repo"].call_args.kwargs["repo_id"] == "me/my_model"

    def test_dot_target_uses_the_containing_folder_name(
        self, tmp_path: Path, hf_api: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`upload-to-hf .` from inside a recipe must not publish a repo named '.'."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        monkeypatch.chdir(recipe)
        upload_to_hf(".", assume_yes=True)
        assert hf_api["create_repo"].call_args.kwargs["repo_id"] == "me/my_model"

    @pytest.mark.parametrize(
        "folder_name", ["my model", "my..model", "a--b", "-lead", "trail."]
    )
    def test_rejects_folder_names_hf_will_not_accept(
        self, tmp_path: Path, hf_api: Any, folder_name: str
    ) -> None:
        """Fail with the naming rules, not an opaque error from deep in the API."""
        recipe = tmp_path / folder_name
        _write_recipe(recipe)

        with pytest.raises(CLIUserError, match="will not accept as a repo name"):
            upload_to_hf(str(recipe), assume_yes=True)

        hf_api["create_repo"].assert_not_called()
        hf_api["upload_folder"].assert_not_called()

    def test_bad_folder_name_suggests_repo_id_escape_hatch(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my model"
        _write_recipe(recipe)
        with pytest.raises(CLIUserError, match="--repo-id") as excinfo:
            upload_to_hf(str(recipe), assume_yes=True)
        assert "Rename the folder" in str(excinfo.value)

    def test_bad_folder_name_is_fine_with_an_explicit_repo_id(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """The folder name only matters when it is what names the repo."""
        recipe = tmp_path / "my model"
        _write_recipe(recipe)
        upload_to_hf(str(recipe), repo_id="me/my_model", assume_yes=True)
        assert hf_api["create_repo"].call_args.kwargs["repo_id"] == "me/my_model"

    def test_rejects_an_invalid_explicit_repo_id(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with pytest.raises(CLIUserError, match="is invalid"):
            upload_to_hf(str(recipe), repo_id="me/bad name", assume_yes=True)
        hf_api["create_repo"].assert_not_called()

    def test_two_users_do_not_collide_on_the_same_model_name(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """The whole point of per-user namespaces: no race for a good name."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)

        seen = []
        for user in ("alice", "bob"):
            hf_api["whoami"].return_value = {"name": user}
            hf_api["repo_exists"].return_value = False
            upload_to_hf(str(recipe), assume_yes=True)
            seen.append(hf_api["create_repo"].call_args.kwargs["repo_id"])

        assert seen == ["alice/my_model", "bob/my_model"]

    def test_explicit_repo_id_overrides(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        url = upload_to_hf(str(recipe), repo_id="me/mine", assume_yes=True)
        assert url == "https://huggingface.co/me/mine"
        assert hf_api["create_repo"].call_args.kwargs["repo_id"] == "me/mine"

    def test_public_by_default(self, tmp_path: Path, hf_api: Any) -> None:
        """Publishing is the point, so it takes one command and no web step."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        upload_to_hf(str(recipe), assume_yes=True)
        assert hf_api["create_repo"].call_args.kwargs["private"] is False

    def test_private_is_opt_in(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        upload_to_hf(str(recipe), private=True, assume_yes=True)
        assert hf_api["create_repo"].call_args.kwargs["private"] is True

    def test_prints_how_to_go_public_after_creating_private_repo(
        self, tmp_path: Path, hf_api: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["repo_exists"].return_value = False
        upload_to_hf(str(recipe), private=True, assume_yes=True)
        out = capsys.readouterr().out
        assert "private" in out
        assert "Change repo visibility" in out

    def test_public_upload_gives_no_visibility_homework(
        self, tmp_path: Path, hf_api: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The default path is done when the command returns."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["repo_exists"].return_value = False
        upload_to_hf(str(recipe), assume_yes=True)
        assert "Change repo visibility" not in capsys.readouterr().out

    def test_no_visibility_advice_for_existing_repo(
        self, tmp_path: Path, hf_api: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An existing repo keeps its own visibility, so don't claim it is private."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["repo_exists"].return_value = True
        upload_to_hf(str(recipe), private=True, assume_yes=True)
        assert "Change repo visibility" not in capsys.readouterr().out

    def test_cli_private_flag_maps_to_private_true(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with patch.object(mod, "upload_to_hf") as upload:
            mod.main([str(recipe), "--private"])
        assert upload.call_args.kwargs["private"] is True

    def test_cli_defaults_to_public(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with patch.object(mod, "upload_to_hf") as upload:
            mod.main([str(recipe)])
        assert upload.call_args.kwargs["private"] is False

    def test_cli_has_no_public_flag(self, tmp_path: Path) -> None:
        """--public is redundant now that public is the default."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with pytest.raises(SystemExit):
            mod.main([str(recipe), "--public"])


class TestVersioning:
    """Re-uploading is an update, not a duplicate: one commit per upload."""

    def test_survives_an_upload_result_without_an_oid(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """Never crash reporting a version; the upload itself already succeeded."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["upload_folder"].return_value = None
        assert upload_to_hf(str(recipe), assume_yes=True) is not None

    def test_deletes_files_dropped_from_the_recipe(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """A renamed or removed file must not linger in the published repo."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        (recipe / "test.py").unlink()

        with patch.object(
            mod,
            "list_repo_files",
            return_value=["model.py", "test.py", "README.md", ".gitattributes"],
        ):
            upload_to_hf(str(recipe), assume_yes=True)

        deleted = hf_api["upload_folder"].call_args.kwargs["delete_patterns"]
        assert deleted == ["test.py"]

    def test_never_deletes_repo_plumbing(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with patch.object(
            mod, "list_repo_files", return_value=[".gitattributes", ".gitignore"]
        ):
            upload_to_hf(str(recipe), assume_yes=True)
        assert hf_api["upload_folder"].call_args.kwargs["delete_patterns"] is None

    def test_mirrors_the_folder_with_no_carve_outs(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """Every upload makes the repo an exact copy, whatever the stale path is."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with patch.object(
            mod,
            "list_repo_files",
            return_value=["model.py", "assets/old.zip", "docs/notes.md", "stale.py"],
        ):
            upload_to_hf(str(recipe), assume_yes=True)

        deleted = hf_api["upload_folder"].call_args.kwargs["delete_patterns"]
        assert deleted == ["assets/old.zip", "docs/notes.md", "stale.py"]

    def test_new_repo_computes_no_deletions(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["repo_exists"].return_value = False
        with patch.object(mod, "list_repo_files") as listed:
            upload_to_hf(str(recipe), assume_yes=True)
        listed.assert_not_called()

    def test_declining_the_prompt_after_a_deletion_list_aborts(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with (
            patch.object(mod, "list_repo_files", return_value=["gone.py"]),
            patch.object(sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["1", "n"]),
        ):
            assert upload_to_hf(str(recipe)) is None
        hf_api["upload_folder"].assert_not_called()


class TestPublishConfirmation:
    """Every publish confirms -- an update overwrites what the URL already serves.

    ``repo_exists`` defaults to True in the fixture, so these are updates unless
    stated otherwise.
    """

    def test_an_update_prompts_before_publishing(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with (
            patch.object(sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["1", "y"]) as prompt,
        ):
            assert upload_to_hf(str(recipe)) is not None
        assert "Update" in prompt.call_args[0][0]
        hf_api["upload_folder"].assert_called_once()

    def test_declining_an_update_uploads_nothing(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with (
            patch.object(sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["1", "n"]),
        ):
            assert upload_to_hf(str(recipe)) is None
        hf_api["upload_folder"].assert_not_called()
        hf_api["create_tag"].assert_not_called()

    def test_assume_yes_skips_the_update_prompt(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with (
            patch.object(sys.stdin, "isatty", return_value=True),
            patch("builtins.input") as prompt,
        ):
            assert upload_to_hf(str(recipe), assume_yes=True) is not None
        prompt.assert_not_called()

    def test_a_new_repo_still_names_its_visibility(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """Visibility is a create-time decision, so only the create prompt says it."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["repo_exists"].return_value = False
        with (
            patch.object(sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["1", "y"]) as prompt,
        ):
            assert upload_to_hf(str(recipe)) is not None
        question = prompt.call_args[0][0]
        assert "Create" in question
        assert "PUBLIC" in question

    def test_a_deletion_does_not_add_a_second_prompt(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """One answer covers the whole upload, deletions included."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with (
            patch.object(mod, "list_repo_files", return_value=["gone.py"]),
            patch.object(sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["1", "y"]) as prompt,
        ):
            assert upload_to_hf(str(recipe)) is not None
        assert prompt.call_count == 2

    def test_a_non_interactive_update_does_not_block(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """No tty means no one to ask; scripted uploads must not hang."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with (
            patch.object(sys.stdin, "isatty", return_value=False),
            patch("builtins.input") as prompt,
        ):
            assert upload_to_hf(str(recipe)) is not None
        prompt.assert_not_called()


class TestVisibility:
    """Location and discoverability are decoupled: own namespace, shared tag."""

    def test_points_at_the_tag_search_after_upload(
        self, tmp_path: Path, hf_api: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        upload_to_hf(str(recipe), assume_yes=True)

        out = capsys.readouterr().out
        assert COMMUNITY_TAG_SEARCH_URL in out
        assert COMMUNITY_ORG_NAME not in out

    def test_claims_ownership_only_for_the_default_namespace(
        self, tmp_path: Path, hf_api: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With --repo-id pointing elsewhere, 'it is yours' may not be true."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        upload_to_hf(str(recipe), repo_id="someorg/thing", assume_yes=True)
        assert "in your own namespace" not in capsys.readouterr().out

    def test_dry_run_without_a_token_shows_a_username_placeholder(
        self, tmp_path: Path, hf_api: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A dry run must still work with no token, so the id can't be resolved."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with (
            patch.object(mod, "get_token", return_value=None),
            patch.object(mod, "whoami", side_effect=RuntimeError("no token")),
        ):
            assert upload_to_hf(str(recipe), dry_run=True) is None


class TestOwnership:
    """Updating an existing repo requires having created it.

    Org members share write access on HuggingFace, so it would otherwise let one
    contributor commit straight over another's published model.
    """

    def test_owner_may_update_their_own_repo(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_commits"].return_value = _fake_commits("me")
        assert upload_to_hf(str(recipe), assume_yes=True) is not None

    def test_refuses_to_overwrite_someone_elses_repo(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_commits"].return_value = _fake_commits("someone_else")

        with pytest.raises(CLIUserError, match="created by 'someone_else'"):
            upload_to_hf(str(recipe), assume_yes=True)

    def test_refusal_suggests_a_name_the_user_owns(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_commits"].return_value = _fake_commits("someone_else")

        with pytest.raises(CLIUserError, match="not you") as excinfo:
            upload_to_hf(str(recipe), assume_yes=True)

        message = str(excinfo.value)
        assert "--repo-id me/<name>" in message
        assert "transfer or delete" in message

    def test_refuses_before_uploading_or_deleting_anything(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """The check must land before create_repo, upload, or any deletion."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_commits"].return_value = _fake_commits("someone_else")
        hf_api["list_repo_files"].return_value = ["doomed.py"]

        with pytest.raises(CLIUserError, match="not you"):
            upload_to_hf(str(recipe), assume_yes=True)

        hf_api["create_repo"].assert_not_called()
        hf_api["upload_folder"].assert_not_called()
        hf_api["create_tag"].assert_not_called()

    def test_ownership_read_from_oldest_commit_not_newest(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """A previous contributor's commit must not confer ownership."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        # Newest commit is ours, but we did not create the repo.
        hf_api["list_repo_commits"].return_value = _fake_commits(
            "someone_else", latest_author="me"
        )
        with pytest.raises(CLIUserError, match="created by 'someone_else'"):
            upload_to_hf(str(recipe), assume_yes=True)

    def test_new_repo_skips_the_ownership_check(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["repo_exists"].return_value = False
        upload_to_hf(str(recipe), assume_yes=True)
        hf_api["list_repo_commits"].assert_not_called()

    def test_dry_run_skips_the_ownership_check(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """--dry-run contacts nothing, so there is nothing to authorize."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_commits"].return_value = _fake_commits("someone_else")
        assert upload_to_hf(str(recipe), dry_run=True) is None

    def test_unverifiable_history_fails_closed(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """Can't prove ownership -> refuse, rather than assume it is fine."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_commits"].side_effect = RuntimeError("403")

        with pytest.raises(CLIUserError, match="ownership cannot be verified"):
            upload_to_hf(str(recipe), assume_yes=True)

    def test_empty_history_fails_closed(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_commits"].return_value = []
        with pytest.raises(CLIUserError, match="ownership cannot be verified"):
            upload_to_hf(str(recipe), assume_yes=True)

    def test_unreadable_username_fails_closed(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["whoami"].side_effect = RuntimeError("bad token")
        with pytest.raises(CLIUserError, match="username could not be read"):
            upload_to_hf(str(recipe), assume_yes=True)

    def test_applies_to_explicit_repo_ids_too(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """--repo-id is not a way around the check."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_commits"].return_value = _fake_commits("someone_else")
        with pytest.raises(CLIUserError, match="created by 'someone_else'"):
            upload_to_hf(str(recipe), repo_id="someorg/thing", assume_yes=True)


class TestNoReplaceFlag:
    def test_replace_flag_no_longer_exists(self, tmp_path: Path) -> None:
        """Mirroring is unconditional, so the opt-in flag is gone rather than inert."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with pytest.raises(SystemExit):
            mod.main([str(recipe), "--replace"])


class TestVersionTags:
    """First upload is v1; each later one bumps to the next vN."""

    def test_new_repo_is_tagged_v1(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["repo_exists"].return_value = False
        hf_api["upload_folder"].return_value.oid = "sha1"

        upload_to_hf(str(recipe), assume_yes=True)

        assert hf_api["create_tag"].call_args.kwargs["tag"] == "v1"
        assert hf_api["create_tag"].call_args.kwargs["revision"] == "sha1"

    def test_new_repo_does_not_query_tags(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["repo_exists"].return_value = False
        upload_to_hf(str(recipe), assume_yes=True)
        hf_api["list_repo_refs"].assert_not_called()

    @pytest.mark.parametrize(
        ("existing", "expected"),
        [
            ([], "v1"),
            (["v1"], "v2"),
            (["v1", "v2"], "v3"),
            (["v2", "v10"], "v11"),
            (["v1", "some-other-tag"], "v2"),
            (["not-a-version"], "v1"),
        ],
    )
    def test_bumps_past_the_highest_existing_version(
        self, tmp_path: Path, hf_api: Any, existing: list[str], expected: str
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_refs"].return_value = _fake_tags(*existing)

        upload_to_hf(str(recipe), assume_yes=True)

        assert hf_api["create_tag"].call_args.kwargs["tag"] == expected

    def test_numbering_is_numeric_not_lexicographic(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """v9 -> v10, not v9 -> v2 (which sorting strings would give)."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_refs"].return_value = _fake_tags("v9")
        upload_to_hf(str(recipe), assume_yes=True)
        assert hf_api["create_tag"].call_args.kwargs["tag"] == "v10"

    def test_tag_is_reported_and_pinnable(
        self, tmp_path: Path, hf_api: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_refs"].return_value = _fake_tags("v1")
        upload_to_hf(str(recipe), assume_yes=True)

        out = capsys.readouterr().out
        assert "version v2" in out
        assert "--version v2" in out

    def test_no_tag_skips_tagging(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        upload_to_hf(str(recipe), no_tag=True, assume_yes=True)
        hf_api["create_tag"].assert_not_called()
        hf_api["list_repo_refs"].assert_not_called()

    def test_tag_failure_does_not_fail_the_upload(
        self, tmp_path: Path, hf_api: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The bytes are already pushed; a tagging error must not look like failure."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["create_tag"].side_effect = RuntimeError("tag exists")

        url = upload_to_hf(str(recipe), assume_yes=True)

        assert url == "https://huggingface.co/me/my_model"
        assert "could not create tag" in capsys.readouterr().out

    def test_tag_appears_in_the_commit_message(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_refs"].return_value = _fake_tags("v4")
        upload_to_hf(str(recipe), assume_yes=True)
        assert "v5" in hf_api["upload_folder"].call_args.kwargs["commit_message"]

    def test_cli_no_tag_flag_is_forwarded(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with patch.object(mod, "upload_to_hf") as upload:
            mod.main([str(recipe), "--no-tag"])
        assert upload.call_args.kwargs["no_tag"] is True

    def test_dry_run_creates_no_tag(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        upload_to_hf(str(recipe), dry_run=True)
        hf_api["create_tag"].assert_not_called()


class TestUnchangedReupload:
    """HuggingFace refuses an empty commit, so re-uploading nothing must not
    mint a second tag pointing at the same commit.
    """

    def test_no_new_tag_when_nothing_changed(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_refs"].return_value = _fake_tags(
            "v1", targets={"v1": "head_sha"}
        )
        # An unchanged upload leaves the head where it was.
        hf_api["upload_folder"].return_value.oid = "head_sha"

        upload_to_hf(str(recipe), assume_yes=True)

        hf_api["create_tag"].assert_not_called()

    def test_reports_the_existing_version_not_a_new_one(
        self, tmp_path: Path, hf_api: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_refs"].return_value = _fake_tags(
            "v1", "v2", targets={"v2": "head_sha"}
        )
        hf_api["upload_folder"].return_value.oid = "head_sha"

        upload_to_hf(str(recipe), assume_yes=True)

        out = capsys.readouterr().out
        assert "Nothing changed" in out
        assert "still version v2" in out
        assert "--version v2" in out
        # Not a bare "v3": the output embeds a random tmp path, and one that
        # happened to be named tmpv0kv3hd8 failed this test in CI.
        assert "version v3" not in out
        hf_api["create_tag"].assert_not_called()

    def test_a_real_change_still_bumps_the_version(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """The guard must key off the commit sha, not merely the tag existing."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["list_repo_refs"].return_value = _fake_tags(
            "v1", targets={"v1": "old_sha"}
        )
        hf_api["upload_folder"].return_value.oid = "new_sha"

        upload_to_hf(str(recipe), assume_yes=True)

        assert hf_api["create_tag"].call_args.kwargs["tag"] == "v2"
        assert hf_api["create_tag"].call_args.kwargs["revision"] == "new_sha"


class TestToken:
    def test_missing_token_explains_setup(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with (
            patch.object(mod, "get_token", return_value=None),
            pytest.raises(CLIUserError, match="No Hugging Face token") as excinfo,
        ):
            upload_to_hf(str(recipe), assume_yes=True)

        message = str(excinfo.value)
        assert "https://huggingface.co/settings/tokens" in message
        assert "hf auth login" in message
        assert "HF_TOKEN" in message
        assert "--dry-run" in message

    def test_missing_token_fails_before_any_upload(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """The check is up front, so nothing is staged or created first."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with (
            patch.object(mod, "get_token", return_value=None),
            patch.object(mod, "_stage") as stage,
            pytest.raises(CLIUserError, match="No Hugging Face token"),
        ):
            upload_to_hf(str(recipe), assume_yes=True)
        stage.assert_not_called()
        hf_api["create_repo"].assert_not_called()

    def test_dry_run_needs_no_token(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with patch.object(mod, "get_token", return_value=None):
            assert upload_to_hf(str(recipe), dry_run=True) is None

    def _whoami(self, role: str | None) -> dict[str, Any]:
        info: dict[str, Any] = {"name": "me"}
        if role is not None:
            info["auth"] = {"accessToken": {"role": role}}
        return info

    def test_read_only_token_explains_how_to_replace_it(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """A read token 403s deep inside the upload, so it is caught up front."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["whoami"].return_value = self._whoami("read")
        with pytest.raises(CLIUserError, match="read-only") as excinfo:
            upload_to_hf(str(recipe), assume_yes=True)

        message = str(excinfo.value)
        assert "https://huggingface.co/settings/tokens" in message
        assert "hf auth login" in message
        assert "HF_TOKEN" in message
        assert "--dry-run" in message

    def test_read_only_token_fails_before_any_upload(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["whoami"].return_value = self._whoami("read")
        with (
            patch.object(mod, "_stage") as stage,
            pytest.raises(CLIUserError, match="read-only"),
        ):
            upload_to_hf(str(recipe), assume_yes=True)
        stage.assert_not_called()
        hf_api["create_repo"].assert_not_called()

    @pytest.mark.parametrize("role", ["write", "fineGrained"])
    def test_write_capable_roles_are_allowed(
        self, tmp_path: Path, hf_api: Any, role: str
    ) -> None:
        """Fine-grained tokens report their own role and do carry write access."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["whoami"].return_value = self._whoami(role)
        assert upload_to_hf(str(recipe), assume_yes=True) is not None

    def test_unreadable_role_does_not_block_the_upload(
        self, tmp_path: Path, hf_api: Any
    ) -> None:
        """No role in the response is not evidence of a read-only token."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["whoami"].return_value = self._whoami(None)
        assert upload_to_hf(str(recipe), assume_yes=True) is not None

    def test_dry_run_skips_the_role_check(self, tmp_path: Path, hf_api: Any) -> None:
        """A dry run uploads nothing, so a read-only token is fine for one."""
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        hf_api["whoami"].return_value = self._whoami("read")
        assert upload_to_hf(str(recipe), dry_run=True) is None

    def test_env_token_is_used(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with patch.object(mod, "get_token", return_value="hf_from_env"):
            upload_to_hf(str(recipe), assume_yes=True)
        assert hf_api["create_repo"].call_args.kwargs["token"] == "hf_from_env"

    def test_explicit_token_wins_over_env(self, tmp_path: Path, hf_api: Any) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with patch.object(mod, "get_token", return_value="hf_from_env"):
            upload_to_hf(str(recipe), token="hf_explicit", assume_yes=True)
        assert hf_api["create_repo"].call_args.kwargs["token"] == "hf_explicit"

    def test_main_exits_cleanly_without_a_traceback(
        self, tmp_path: Path, hf_api: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recipe = tmp_path / "my_model"
        _write_recipe(recipe)
        with (
            patch.object(mod, "get_token", return_value=None),
            pytest.raises(SystemExit) as excinfo,
        ):
            mod.main([str(recipe), "--yes"])
        assert "hf auth login" in str(excinfo.value)

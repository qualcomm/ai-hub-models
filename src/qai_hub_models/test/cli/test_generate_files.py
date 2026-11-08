# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for ``qai-hub-models generate-files`` README rendering.

Covers the external/internal split: external recipes (outside
``qai_hub_models/models/``) must get a README whose commands take the recipe
folder path, drops the catalog-only Quick Start block, drops the
``workbench.aihub.qualcomm.com`` model-page link, and drops the
unpublished-warning banner. In-tree recipes, whose status comes from
internal-config.yaml, keep the existing catalog-flavored README.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models.cli import generate_files as generate_files_mod
from qai_hub_models.cli.generate_files import _hf_register_cmd, write_readme
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest

_MINIMAL_MANIFEST = """\
description: A test classifier.
domain: Computer Vision
has_on_target_demo: true
headline: A test classifier.
id: {model_id}
license: https://example.com/LICENSE
license_type: bsd-3-clause
name: {model_name}
research_paper: https://arxiv.org/abs/1512.03385
research_paper_title: Test Paper
source_repo: https://github.com/example/test-model
supported_precisions:
{precisions}\
tags:
- bu-auto
templates:
- imagenet_classifier
use_case: Image Classification
"""


def _write_recipe(
    folder: Path,
    model_id: str = "my_model",
    precisions: str = "- float\n",
) -> QAIHMModelManifest:
    """Author a minimal recipe folder + manifest at *folder* and return the manifest."""
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.yaml"
    manifest_path.write_text(
        _MINIMAL_MANIFEST.format(
            model_id=model_id,
            model_name="MyModel",
            precisions=precisions,
        )
    )
    return QAIHMModelManifest.from_yaml(manifest_path)


class TestExternalReadme:
    """External recipes: folder outside qai_hub_models/models/ → external README."""

    def test_writes_without_status(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme_path = write_readme(recipe, manifest)
        assert readme_path == recipe / "README.md"
        assert readme_path.exists()

    def test_uses_bare_folder_name_for_install(self, tmp_path: Path) -> None:
        """A bare name resolves as a cwd folder, so no ./ prefix is advertised."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models install my_model" in readme
        assert "./my_model/" not in readme

    def test_uses_bare_folder_name_for_export(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models export my_model" in readme
        assert "./my_model/" not in readme

    def test_export_command_relies_on_defaults(self, tmp_path: Path) -> None:
        """The sample export command carries no flags.

        ``export`` resolves precision, runtime and device from the manifest
        itself, so spelling them out only invited readers to edit values that
        are already correct.
        """
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        export_lines = [
            line
            for line in readme.splitlines()
            if line.startswith("qai-hub-models export ")
        ]
        assert export_lines
        for line in export_lines:
            assert "--target-runtime" not in line
            assert "--precision" not in line
            assert "--device" not in line

    def test_demo_uses_the_cli_command(self, tmp_path: Path) -> None:
        """`qai-hub-models demo`, not a python -m module path."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models demo my_model" in readme
        assert "python -m" not in readme

    def test_demo_command_has_no_quantize_flag(self, tmp_path: Path) -> None:
        """--quantize is a redundant alias for --precision; it must not be advertised."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe, precisions="- float\n- w8a8\n")
        readme = write_readme(recipe, manifest).read_text()
        assert "--quantize" not in readme

    def test_no_catalog_quickstart_commands(self, tmp_path: Path) -> None:
        """``qai-hub-models info/perf/numerics/fetch`` need a catalog id — external skips."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models info " not in readme
        assert "qai-hub-models perf " not in readme
        assert "qai-hub-models numerics " not in readme
        assert "qai-hub-models fetch " not in readme

    def test_no_quick_start_section(self, tmp_path: Path) -> None:
        """Quick Start duplicates Setup + Run CLI Demo + Export for external — drop it."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "## Quick Start" not in readme

    def test_no_catalog_model_page_link(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        # The title link (aihub.qualcomm.com/models/<id>) is catalog-only.
        # The generic workbench.aihub.qualcomm.com landing link is still
        # fine to reference elsewhere.
        assert "aihub.qualcomm.com/models/" not in readme

    def test_no_unpublished_warning_banner(self, tmp_path: Path) -> None:
        """External recipes have no status; do not render the pending/unpublished banner."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "[!WARNING]" not in readme

    def test_missing_headline_raises(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest_path = recipe / "manifest.yaml"
        recipe.mkdir()
        manifest_path.write_text(
            _MINIMAL_MANIFEST.format(
                model_id="my_model",
                model_name="MyModel",
                precisions="- float\n",
            ).replace("headline: A test classifier.\n", "")
        )
        manifest = QAIHMModelManifest.from_yaml(manifest_path)
        with pytest.raises(CLIUserError, match="headline"):
            write_readme(recipe, manifest)


class TestInternalReadme:
    """In-tree recipes: status from internal-config.yaml → catalog-flavored README."""

    @pytest.fixture(autouse=True)
    def _in_tree_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            generate_files_mod,
            "resolve_publish_status",
            lambda _: MODEL_STATUS.PENDING,
        )

    def test_uses_bare_model_id_for_export(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models export my_model" in readme
        assert "qai-hub-models export ./my_model/" not in readme

    def test_demo_uses_the_cli_command(self, tmp_path: Path) -> None:
        """In-tree cards use the same CLI command, with the bare model id."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models demo my_model" in readme
        assert "python -m qai_hub_models.models.my_model.demo" not in readme

    def test_includes_catalog_quickstart(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models info " in readme
        assert "qai-hub-models perf " in readme
        assert "qai-hub-models numerics " in readme

    def test_includes_catalog_model_page_link(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "aihub.qualcomm.com/models/my_model" in readme

    def test_pending_shows_warning_banner(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "[!WARNING]" in readme

    def test_install_uses_cli_not_extras(self, tmp_path: Path) -> None:
        """Setup uses ``qai-hub-models install <id>``, not ``pip install "qai-hub-models[<id>]"``."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models install my_model" in readme
        assert 'pip install "qai-hub-models[my-model]"' not in readme
        assert 'pip install "qai-hub-models[my_model]"' not in readme


class TestPublishedCardRegisterStep:
    """A published card must fetch the recipe before anything uses it."""

    def test_no_register_line_without_a_repo_id(self, tmp_path: Path) -> None:
        """`generate-files` writes the on-disk README, which nobody registers."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(recipe, manifest).read_text()
        assert "qai-hub-models register" not in readme

    def test_register_sits_in_the_setup_step(self, tmp_path: Path) -> None:
        """Order matters: install the package, fetch the recipe, then use it."""
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        readme = write_readme(
            recipe, manifest, hf_repo_id="someone/my_model"
        ).read_text()

        pip = readme.index("pip install qai-hub-models")
        register = readme.index("qai-hub-models register someone/my_model")
        install = readme.index("qai-hub-models install my_model")
        demo = readme.index("qai-hub-models demo my_model")
        export = readme.index("qai-hub-models export my_model")
        assert pip < register < install < demo < export

        # One register line, in the setup step -- not a second copy up top.
        assert readme.count("qai-hub-models register") == 1
        assert register < readme.index("Configure Qualcomm")

    def test_a_repo_id_does_not_disturb_the_plain_install_block(
        self, tmp_path: Path
    ) -> None:
        """The published card adds lines; it must not reflow the shared ones.

        The register step is conditional inside `package_instructions`, and a
        careless `{% if %}` there changed the blank lines around the code fence
        for all 219 in-tree cards. This pins the unpublished rendering.
        """
        recipe = tmp_path / "my_model"
        manifest = _write_recipe(recipe)
        plain = write_readme(recipe, manifest).read_text()

        assert (
            "pip install qai-hub-models\n"
            "qai-hub-models install my_model\n"
            "```\n"
            "\n"
            "### 2. Configure" in plain
        )

    def test_alias_is_omitted_when_the_repo_name_already_matches(self) -> None:
        assert (
            _hf_register_cmd("someone/my_model", "my_model")
            == "qai-hub-models register someone/my_model"
        )

    def test_alias_is_added_when_the_repo_name_differs(self) -> None:
        """Every later command says `my_model`, so the alias has to be that."""
        assert (
            _hf_register_cmd("someone/my-model-v2", "my_model")
            == "qai-hub-models register someone/my-model-v2 --alias my_model"
        )

    def test_alias_is_added_when_no_alias_can_be_derived(self) -> None:
        """`default_alias` raises rather than mangling; --alias is then required."""
        assert (
            _hf_register_cmd("someone/Weird.Name!", "my_model")
            == "qai-hub-models register someone/Weird.Name! --alias my_model"
        )

    def test_no_repo_id_means_no_command(self) -> None:
        assert _hf_register_cmd(None, "my_model") is None

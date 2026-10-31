# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for ``qai-hub-models validate`` (report-card runner + wiring)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from packaging.specifiers import SpecifierSet
from typing_extensions import Self

from qai_hub_models import Precision, SampleInputsType, TargetRuntime
from qai_hub_models.cli import validate as validate_mod
from qai_hub_models.cli.dispatch import run_model_script
from qai_hub_models.cli.install import InstallAborted
from qai_hub_models.cli.validate import (
    Report,
    Result,
    Status,
    _check_arxiv_abs,
    _check_default_device_canary,
    _check_external_repo_shas,
    _check_external_repos_init,
    _check_files_present,
    _check_headline_period,
    _check_id_matches_folder,
    _check_in_tree_status,
    _check_install,
    _check_manifest,
    _check_multi_graph_sample_inputs,
    _check_name_style,
    _check_no_self_referential_imports,
    _check_pip_commands,
    _check_related_not_self,
    _check_requirements_txt,
    _check_status_not_unset,
    _check_status_reason,
    _check_url_reachability,
    _check_website_fields_set,
    _collect_manifest_urls,
    _extract_pip_command_pkgs,
    _extract_pip_command_reqs,
    _extract_shape,
    _iter_requirements,
    _recipe_opts_out_of_global_requirements,
    _render_json,
    _render_text,
    _run_all_checks,
    _run_internal_checks,
    _specifier_is_satisfiable,
)
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec


def _make_manifest(**kwargs: Any) -> QAIHMModelManifest:
    """Bare stand-in for QAIHMModelManifest; each test sets only the fields it uses."""
    return cast(QAIHMModelManifest, SimpleNamespace(**kwargs))


class TestFilesPresent:
    def test_all_present(self, tmp_path: Path) -> None:
        for name in (
            "__init__.py",
            "model.py",
            "manifest.yaml",
            "app.py",
            "demo.py",
            "test.py",
        ):
            (tmp_path / name).touch()
        report = Report()
        _check_files_present(tmp_path, report)
        statuses = {r.name: r.status for r in report.rows}
        assert statuses["Required files present"] is Status.PASS
        assert "app.py present" not in statuses
        assert statuses["demo.py present"] is Status.PASS
        assert statuses["test.py present"] is Status.PASS

    def test_missing_required_hard_fails(self, tmp_path: Path) -> None:
        (tmp_path / "__init__.py").touch()
        (tmp_path / "manifest.yaml").touch()
        report = Report()
        _check_files_present(tmp_path, report)
        row = next(r for r in report.rows if r.name == "Required files present")
        assert row.status is Status.FAIL
        assert "model.py" in row.detail

    def test_missing_optional_are_warnings(self, tmp_path: Path) -> None:
        (tmp_path / "__init__.py").touch()
        (tmp_path / "model.py").touch()
        (tmp_path / "manifest.yaml").touch()
        report = Report()
        _check_files_present(tmp_path, report)
        statuses = {r.name: r.status for r in report.rows}
        assert "app.py present" not in statuses
        assert statuses["demo.py present"] is Status.WARN
        assert statuses["test.py present"] is Status.WARN


class TestIdMatchesFolder:
    def test_id_matches(self, tmp_path: Path) -> None:
        folder = tmp_path / "my_model"
        folder.mkdir()
        report = Report()
        _check_id_matches_folder(folder, _make_manifest(id="my_model"), report)
        assert report.rows[0].status is Status.PASS

    def test_id_mismatch_fails(self, tmp_path: Path) -> None:
        folder = tmp_path / "my_model"
        folder.mkdir()
        report = Report()
        _check_id_matches_folder(folder, _make_manifest(id="other_id"), report)
        assert report.rows[0].status is Status.FAIL
        assert "other_id" in report.rows[0].detail

    def test_id_none_skips(self, tmp_path: Path) -> None:
        report = Report()
        _check_id_matches_folder(tmp_path, _make_manifest(id=None), report)
        assert report.rows == []


class TestSelfReferentialImports:
    def test_clean_recipe_passes(self, tmp_path: Path) -> None:
        (tmp_path / "model.py").write_text("from .base import Foo\n")
        report = Report()
        _check_no_self_referential_imports(tmp_path, report)
        assert report.rows[0].status is Status.PASS

    def test_self_import_fails(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        recipe.mkdir()
        (recipe / "model.py").write_text(
            "from qai_hub_models.models.my_model.util import helper\n"
        )
        report = Report()
        _check_no_self_referential_imports(recipe, report)
        assert report.rows[0].status is Status.FAIL
        assert "model.py:1" in report.rows[0].detail

    def test_other_model_import_is_fine(self, tmp_path: Path) -> None:
        recipe = tmp_path / "my_model"
        recipe.mkdir()
        (recipe / "model.py").write_text(
            "from qai_hub_models.models.other_model.util import helper\n"
        )
        report = Report()
        _check_no_self_referential_imports(recipe, report)
        assert report.rows[0].status is Status.PASS

    def test_in_tree_recipe_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # In-tree recipes ARE `qai_hub_models.models.<id>`, so the import is
        # legitimate. Only external / standalone recipes need this rule.
        monkeypatch.setattr(validate_mod, "_is_in_tree", lambda _: True)
        recipe = tmp_path / "my_model"
        recipe.mkdir()
        (recipe / "model.py").write_text(
            "from qai_hub_models.models.my_model.util import helper\n"
        )
        report = Report()
        _check_no_self_referential_imports(recipe, report)
        assert report.rows == []


class TestStatusNotUnset:
    def test_in_tree_unset_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(validate_mod, "_is_in_tree", lambda _: True)
        report = Report()
        _check_status_not_unset(
            tmp_path, _make_manifest(status=MODEL_STATUS.UNSET), report
        )
        assert report.rows[0].status is Status.FAIL

    def test_in_tree_pending_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(validate_mod, "_is_in_tree", lambda _: True)
        report = Report()
        _check_status_not_unset(
            tmp_path, _make_manifest(status=MODEL_STATUS.PENDING), report
        )
        assert report.rows[0].status is Status.PASS

    def test_external_unset_is_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(validate_mod, "_is_in_tree", lambda _: False)
        report = Report()
        _check_status_not_unset(
            tmp_path, _make_manifest(status=MODEL_STATUS.UNSET), report
        )
        assert report.rows == []


class TestExternalReposInit:
    def test_no_external_repos_skips(self, tmp_path: Path) -> None:
        report = Report()
        _check_external_repos_init(
            tmp_path, _make_manifest(external_repos=None), report
        )
        assert report.rows == []

    def test_missing_init_fails(self, tmp_path: Path) -> None:
        report = Report()
        _check_external_repos_init(
            tmp_path, _make_manifest(external_repos={"foo": object()}), report
        )
        assert report.rows[0].status is Status.FAIL

    def test_present_init_passes(self, tmp_path: Path) -> None:
        (tmp_path / "external_repos").mkdir()
        (tmp_path / "external_repos" / "__init__.py").touch()
        report = Report()
        _check_external_repos_init(
            tmp_path, _make_manifest(external_repos={"foo": object()}), report
        )
        assert report.rows[0].status is Status.PASS


class TestExternalRepoShas:
    def test_valid_sha_passes(self) -> None:
        report = Report()
        cfg = SimpleNamespace(commit_sha="a" * 40)
        _check_external_repo_shas(_make_manifest(external_repos={"repo": cfg}), report)
        assert report.rows[0].status is Status.PASS

    def test_short_sha_fails(self) -> None:
        report = Report()
        cfg = SimpleNamespace(commit_sha="abc123")
        _check_external_repo_shas(_make_manifest(external_repos={"repo": cfg}), report)
        assert report.rows[0].status is Status.FAIL
        assert "abc123" in report.rows[0].detail

    def test_branch_name_fails(self) -> None:
        report = Report()
        cfg = SimpleNamespace(commit_sha="main")
        _check_external_repo_shas(_make_manifest(external_repos={"repo": cfg}), report)
        assert report.rows[0].status is Status.FAIL

    def test_no_external_repos_skips(self) -> None:
        report = Report()
        _check_external_repo_shas(_make_manifest(external_repos=None), report)
        assert report.rows == []


class TestRequirementsTxt:
    def test_no_file_is_silent(self, tmp_path: Path) -> None:
        report = Report()
        _check_requirements_txt(tmp_path, report)
        assert report.rows == []

    def test_pinned_lines_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "requirements.txt").write_text("foo==1.0\nbar>=2.0\n")
        monkeypatch.setattr(validate_mod, "_load_requirement_pins", lambda path: {})
        report = Report()
        _check_requirements_txt(tmp_path, report)
        statuses = {r.name: r.status for r in report.rows}
        assert statuses["requirements.txt entries pinned"] is Status.PASS
        assert statuses["requirements.txt vs. base package"] is Status.PASS

    def test_unpinned_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "requirements.txt").write_text("foo\n")
        monkeypatch.setattr(validate_mod, "_load_requirement_pins", lambda path: {})
        report = Report()
        _check_requirements_txt(tmp_path, report)
        row = next(
            r for r in report.rows if r.name == "requirements.txt entries pinned"
        )
        assert row.status is Status.FAIL
        assert "foo" in row.detail

    def test_conflict_with_base_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "requirements.txt").write_text("torch==2.99\n")
        monkeypatch.setattr(
            validate_mod,
            "_load_requirement_pins",
            lambda path: {"torch": SpecifierSet(">=2.4,<=2.11.0")},
        )
        report = Report()
        _check_requirements_txt(tmp_path, report)
        row = next(
            r for r in report.rows if r.name == "requirements.txt vs. base package"
        )
        assert row.status is Status.FAIL
        assert "torch" in row.detail
        assert "2.99" in row.detail

    def test_standalone_ignores_global_requirements_only_pins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A standalone recipe may move a package the base package never pins.

        transformers is the real case: absent from the base package's
        requirements.txt, but pinned in global_requirements.txt because many
        in-tree models agree on one version. A standalone recipe brings its own
        environment, so that agreement is not binding on it.
        """
        (tmp_path / "requirements.txt").write_text("transformers==5.7.0\n")
        base = tmp_path / "base.txt"
        base.write_text("torch>=2.4,<=2.11.0\n")
        globals_ = tmp_path / "global.txt"
        globals_.write_text("torch>=2.4,<=2.11.0\ntransformers==4.56.2\n")
        monkeypatch.setattr(validate_mod, "_BASE_REQUIREMENTS_PATH", base)
        monkeypatch.setattr(validate_mod, "_GLOBAL_REQUIREMENTS_PATH", globals_)
        monkeypatch.setattr(validate_mod, "_is_in_tree", lambda source_dir: False)
        validate_mod._load_requirement_pins.cache_clear()
        report = Report()
        _check_requirements_txt(tmp_path, report)
        row = next(
            r for r in report.rows if r.name == "requirements.txt vs. base package"
        )
        assert row.status is Status.PASS

    def test_in_tree_still_held_to_global_requirements(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same pin is still a conflict for an in-tree recipe."""
        (tmp_path / "requirements.txt").write_text("transformers==5.7.0\n")
        base = tmp_path / "base.txt"
        base.write_text("torch>=2.4,<=2.11.0\n")
        globals_ = tmp_path / "global.txt"
        globals_.write_text("torch>=2.4,<=2.11.0\ntransformers==4.56.2\n")
        monkeypatch.setattr(validate_mod, "_BASE_REQUIREMENTS_PATH", base)
        monkeypatch.setattr(validate_mod, "_GLOBAL_REQUIREMENTS_PATH", globals_)
        monkeypatch.setattr(validate_mod, "_is_in_tree", lambda source_dir: True)
        validate_mod._load_requirement_pins.cache_clear()
        report = Report()
        _check_requirements_txt(tmp_path, report)
        statuses = {r.name: r.status for r in report.rows}
        assert statuses["requirements.txt vs. base package"] is Status.PASS
        assert statuses["requirements.txt vs. global_requirements.txt"] is Status.FAIL

    def test_compatible_pin_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "requirements.txt").write_text("torch==2.5\n")
        monkeypatch.setattr(
            validate_mod,
            "_load_requirement_pins",
            lambda path: {"torch": SpecifierSet(">=2.4,<=2.11.0")},
        )
        report = Report()
        _check_requirements_txt(tmp_path, report)
        statuses = {r.name: r.status for r in report.rows}
        assert statuses["requirements.txt vs. base package"] is Status.PASS

    def _in_tree_statuses(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        recipe_line: str,
        base_line: str,
        global_line: str,
    ) -> dict[str, Status]:
        (tmp_path / "requirements.txt").write_text(recipe_line)
        base = tmp_path / "base.txt"
        base.write_text(base_line)
        globals_ = tmp_path / "global.txt"
        globals_.write_text(global_line)
        monkeypatch.setattr(validate_mod, "_BASE_REQUIREMENTS_PATH", base)
        monkeypatch.setattr(validate_mod, "_GLOBAL_REQUIREMENTS_PATH", globals_)
        monkeypatch.setattr(validate_mod, "_is_in_tree", lambda source_dir: True)
        validate_mod._load_requirement_pins.cache_clear()
        report = Report()
        _check_requirements_txt(tmp_path, report, skip_global_conflicts=True)
        return {r.name: r.status for r in report.rows}

    def test_global_conflict_skipped_when_opted_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        statuses = self._in_tree_statuses(
            tmp_path,
            monkeypatch,
            "transformers==5.7.0\n",
            "torch>=2.4,<=2.11.0\n",
            "transformers==4.56.2\n",
        )
        assert statuses["requirements.txt vs. base package"] is Status.PASS
        assert statuses["requirements.txt vs. global_requirements.txt"] is Status.SKIP

    def test_base_conflict_still_fails_when_opted_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        statuses = self._in_tree_statuses(
            tmp_path,
            monkeypatch,
            "torch==2.99\n",
            "torch>=2.4,<=2.11.0\n",
            "torch>=2.4,<=2.11.0\n",
        )
        assert statuses["requirements.txt vs. base package"] is Status.FAIL
        assert statuses["requirements.txt vs. global_requirements.txt"] is Status.SKIP


class TestRecipeOptsOut:
    def test_none_manifest_skips_global(self) -> None:
        assert _recipe_opts_out_of_global_requirements(None) is True

    def test_external_model_skips_global(self) -> None:
        manifest = _make_manifest(id="__model_that_does_not_exist__")
        assert _recipe_opts_out_of_global_requirements(manifest) is True

    def test_in_tree_opt_out_is_detected(self) -> None:
        manifest = _make_manifest(id="gemma_4_e4b_it")
        assert _recipe_opts_out_of_global_requirements(manifest) is True


class TestRequirementsHelpers:
    def test_iter_requirements_skips_comments_and_flags(self) -> None:
        rows = _iter_requirements("# comment\nfoo==1.0\n-r other.txt\n\nbar>=2\n")
        assert rows == [(2, "foo==1.0"), (5, "bar>=2")]

    def test_specifier_is_satisfiable_intersects(self) -> None:
        assert _specifier_is_satisfiable(SpecifierSet("==2.5,>=2.0,<=3.0")) is True
        assert _specifier_is_satisfiable(SpecifierSet("==2.99,>=2.4,<=2.11.0")) is False
        assert _specifier_is_satisfiable(SpecifierSet(">=2.0,<=3.0")) is True

    def test_extract_pip_command_pkgs_basic(self) -> None:
        assert _extract_pip_command_pkgs("pip install foo==1.0 bar>=2") == [
            "foo",
            "bar",
        ]

    def test_extract_pip_command_pkgs_skips_editable_and_vcs(self) -> None:
        assert _extract_pip_command_pkgs("pip install -e ./src") == []
        assert _extract_pip_command_pkgs("pip install git+https://x/y.git") == []

    def test_extract_pip_command_reqs(self) -> None:
        reqs = _extract_pip_command_reqs(
            "pip install foo==1.0 bar>=2 -e ./src --extra-index-url https://pypi.org git+https://x/y.git"
        )
        assert [r.name for r in reqs] == ["foo", "bar"]
        assert [str(r.specifier) for r in reqs] == ["==1.0", ">=2"]


class TestCheckPipCommands:
    PINS = {
        "torch": SpecifierSet(">=2.4,<=2.11.0"),
        "torchvision": SpecifierSet(">=0.19,<=0.26.0"),
        "onnx": SpecifierSet("==1.18.0"),
        "onnxruntime": SpecifierSet("==1.22.1"),
    }

    def _run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *commands: str
    ) -> Report:
        monkeypatch.setattr(
            validate_mod, "_load_requirement_pins", lambda path: self.PINS
        )
        manifest = _make_manifest(
            pre_pip_install_commands=[SimpleNamespace(command=c) for c in commands],
            post_pip_install_commands=[],
        )
        report = Report()
        _check_pip_commands(tmp_path, manifest, report)
        return report

    def test_no_commands_is_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert not self._run(tmp_path, monkeypatch).rows

    @pytest.mark.parametrize(
        "command",
        [
            "pip install torchvision==0.24.0 torch==2.9.0",
            "pip install torch==2.9.0 torchvision==0.24.0",
            "pip install onnxruntime==1.22.1 onnx==1.18.0",
            "pip install onnx==1.18.0 onnxruntime==1.22.1",
            "pip install other-package==1.0.0 torch",
        ],
    )
    def test_compatible_commands_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
    ) -> None:
        (row,) = self._run(tmp_path, monkeypatch, command).rows
        assert row.status is Status.PASS

    def test_detects_real_conflict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (row,) = self._run(tmp_path, monkeypatch, "pip install torch==1.12.0").rows
        assert row.status is Status.FAIL
        assert "torch" in row.detail
        assert "1.12.0" in row.detail


class TestExtractShape:
    def test_tuple_form(self) -> None:
        assert _extract_shape(((1, 3, 224, 224), "float32")) == (1, 3, 224, 224)

    def test_tensor_spec_object(self) -> None:
        spec = SimpleNamespace(shape=(1, 100, 4))
        assert _extract_shape(spec) == (1, 100, 4)

    def test_empty_shape_is_none(self) -> None:
        # An empty tuple means "shape unknown/undeclared" — skip the check.
        assert _extract_shape(((), "float32")) is None
        assert _extract_shape(SimpleNamespace(shape=())) is None

    def test_unknown_form(self) -> None:
        assert _extract_shape("weird") is None


class _StubMultiGraph(MultiGraphWorkbenchModel):
    """Multi-graph stub that keeps the base class's real sample-input generation."""

    def __init__(
        self,
        specs: dict[str, InputSpec],
        sample_override: Callable[[str], SampleInputsType] | None = None,
        spec_error: Exception | None = None,
    ) -> None:
        self._specs = specs
        self._sample_override = sample_override
        self._spec_error = spec_error

    @property
    def graph_names(self) -> list[str]:
        return list(self._specs)

    def get_graph_input_spec(
        self, graph_name: str, *args: Any, **kwargs: Any
    ) -> InputSpec:
        if self._spec_error is not None:
            raise self._spec_error
        return self._specs[graph_name]

    def get_graph_output_spec(self, graph_name: str) -> OutputSpec:
        return {"out": TensorSpec(shape=(1, 2), dtype="float32")}

    def get_graph_sample_inputs(
        self,
        graph_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        if self._sample_override is not None:
            return self._sample_override(graph_name)
        return super().get_graph_sample_inputs(
            graph_name, input_spec, use_channel_last_format
        )

    def serialize_graph(
        self, graph_name: str, output_dir: Any, input_spec: InputSpec | None = None
    ) -> Path:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Self:
        raise NotImplementedError


def _one_graph(
    sample_override: Callable[[str], SampleInputsType] | None = None,
    spec_error: Exception | None = None,
) -> _StubMultiGraph:
    return _StubMultiGraph(
        {"g0": {"x": TensorSpec(shape=(1, 4), dtype="float32")}},
        sample_override=sample_override,
        spec_error=spec_error,
    )


def _run_mg_check(model: MultiGraphWorkbenchModel) -> Result:
    report = Report()
    _check_multi_graph_sample_inputs(model, report)
    assert len(report.rows) == 1
    return report.rows[0]


class TestMultiGraphSampleInputs:
    def test_matching_spec_passes(self) -> None:
        row = _run_mg_check(_one_graph())
        assert row.status is Status.PASS
        assert "1 graph(s)" in row.detail

    def test_every_graph_counted(self) -> None:
        spec: InputSpec = {"x": TensorSpec(shape=(1, 4), dtype="float32")}
        row = _run_mg_check(_StubMultiGraph({"g0": spec, "g1": dict(spec)}))
        assert row.status is Status.PASS
        assert "2 graph(s)" in row.detail

    def test_channel_last_input_passes(self) -> None:
        # Regression: the spec is always channel-first, so sample inputs have to
        # be generated in that layout or every channel-last input fails.
        row = _run_mg_check(
            _StubMultiGraph(
                {
                    "g0": {
                        "image": TensorSpec(
                            shape=(1, 3, 8, 8),
                            dtype="float32",
                            apply_runtime_channel_reordering=True,
                        )
                    }
                }
            )
        )
        assert row.status is Status.PASS

    def test_missing_key_fails(self) -> None:
        row = _run_mg_check(_one_graph(sample_override=lambda _: {}))
        assert row.status is Status.FAIL
        assert "missing ['x']" in row.detail

    def test_extra_key_fails(self) -> None:
        row = _run_mg_check(
            _one_graph(
                sample_override=lambda _: {
                    "x": [np.zeros((1, 4), dtype=np.float32)],
                    "y": [np.zeros((1, 4), dtype=np.float32)],
                }
            )
        )
        assert row.status is Status.FAIL
        assert "unexpected ['y']" in row.detail

    def test_shape_mismatch_fails(self) -> None:
        row = _run_mg_check(
            _one_graph(
                sample_override=lambda _: {"x": [np.zeros((1, 8), dtype=np.float32)]}
            )
        )
        assert row.status is Status.FAIL
        assert "has shape (1, 8)" in row.detail

    def test_dtype_mismatch_fails(self) -> None:
        row = _run_mg_check(
            _one_graph(
                sample_override=lambda _: {"x": [np.zeros((1, 4), dtype=np.int32)]}
            )
        )
        assert row.status is Status.FAIL
        assert "has dtype int32" in row.detail

    def test_spec_exception_reported(self) -> None:
        row = _run_mg_check(_one_graph(spec_error=ValueError("boom")))
        assert row.status is Status.FAIL
        assert "ValueError: boom" in row.detail


class TestUrlReachability:
    def test_no_urls_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manifest = _make_manifest(
            license=None,
            source_repo=None,
            research_paper=None,
            external_repos=None,
        )
        called = False

        def _spy(_pairs: list[tuple[str, str]]) -> list[str]:
            nonlocal called
            called = True
            return []

        monkeypatch.setattr(validate_mod, "head_check_urls", _spy)
        report = Report()
        _check_url_reachability(manifest, report)
        assert not called
        assert report.rows[0].status is Status.SKIP

    def test_all_reachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manifest = _make_manifest(
            license="https://example.com/lic",
            source_repo="https://example.com/repo",
            research_paper=None,
            external_repos=None,
        )
        monkeypatch.setattr(validate_mod, "head_check_urls", lambda _pairs: [])
        report = Report()
        _check_url_reachability(manifest, report)
        statuses = {r.name: r.status for r in report.rows}
        assert statuses == {"license": Status.PASS, "source_repo": Status.PASS}

    def test_broken_url_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manifest = _make_manifest(
            license="https://example.com/broken",
            source_repo="https://example.com/ok",
            research_paper=None,
            external_repos=None,
        )
        monkeypatch.setattr(
            validate_mod,
            "head_check_urls",
            lambda _pairs: [
                "License URL unreachable at https://example.com/broken (status: 404)"
            ],
        )
        report = Report()
        _check_url_reachability(manifest, report)
        by_name = {r.name: r for r in report.rows}
        assert by_name["license"].status is Status.FAIL
        assert by_name["source_repo"].status is Status.PASS

    def test_collects_external_repo_urls(self) -> None:
        cfg = SimpleNamespace(repo_url="https://github.com/foo/bar")
        manifest = _make_manifest(
            license="https://x/lic",
            source_repo=None,
            research_paper=None,
            external_repos={"bar": cfg},
        )
        urls = _collect_manifest_urls(manifest)
        names = [row_name for row_name, _, _ in urls]
        assert "external_repos.bar.repo_url" in names
        assert "license" in names


class TestRender:
    def _report_with_mixed_rows(self) -> Report:
        r = Report()
        r.add(Result("a", "Folder shape", Status.PASS, "ok"))
        r.add(Result("b", "Manifest", Status.FAIL, "boom"))
        r.add(Result("c", "Model code", Status.WARN, "hmm"))
        return r

    def test_text_groups_by_category(self) -> None:
        text = _render_text(self._report_with_mixed_rows())
        assert "Folder shape" in text
        assert "Manifest" in text
        assert "Model code" in text
        assert "1 passed, 1 failed, 1 warnings, 0 skipped." in text

    def test_json_is_valid(self) -> None:
        import json as _json

        payload = _json.loads(_render_json(self._report_with_mixed_rows()))
        assert payload["counts"]["FAIL"] == 1
        assert payload["failed"] is True
        assert len(payload["rows"]) == 3


class TestCheckManifest:
    MANIFEST = """\
id: my_recipe
name: My Recipe
headline: A model for testing.
description: A model for testing manifest validation.
domain: Computer Vision
use_case: Image Classification
license_type: apache-2.0
license: https://example.com/license
source_repo: https://example.com/repo
research_paper: https://arxiv.org/abs/1234.5678
research_paper_title: A Paper
form_factors:
- Phone
applicable_scenarios:
- Inventory Management
supported_precisions:
- float
"""

    def _write(self, tmp_path: Path, extra: str = "") -> Path:
        source_dir = tmp_path / "my_recipe"
        source_dir.mkdir()
        (source_dir / "manifest.yaml").write_text(self.MANIFEST + extra)
        return source_dir

    def _validates_row(self, report: Report) -> Result:
        rows = [r for r in report.rows if r.name == "manifest.yaml validates"]
        assert len(rows) == 1
        return rows[0]

    def test_plain_manifest_validates(self, tmp_path: Path) -> None:
        report = Report()
        assert _check_manifest(self._write(tmp_path), report) is not None
        assert self._validates_row(report).status is Status.PASS

    def test_disabled_paths_validates(self, tmp_path: Path) -> None:
        """
        disabled_paths is keyed by Precision/TargetRuntime, and
        ModelDisableReasonsMapping.__init__ takes **kwargs, so a python-mode
        model_dump round-trip raised "TypeError: keywords must be strings" and
        every recipe recording a failed path failed validation.
        """
        extra = """\
disabled_paths:
  float:
    tflite:
      scorecard_accuracy_failure: "On-device 58.7 vs 92.6 torch — not investigated"
"""
        report = Report()
        manifest = _check_manifest(self._write(tmp_path, extra), report)
        assert manifest is not None
        row = self._validates_row(report)
        assert row.status is Status.PASS, row.detail
        assert manifest.disabled_paths is not None
        reasons = manifest.disabled_paths.peek_disable_reasons(
            Precision.float, TargetRuntime.TFLITE
        )
        assert reasons is not None
        assert reasons.has_failure


class TestDispatchValidate:
    def test_dispatch_forwards_to_validate_main(self) -> None:
        with patch("qai_hub_models.cli.dispatch.validate_main") as mock_main:
            run_model_script(
                model_id="./my_recipe/",
                script="validate",
                forwarded=["--json"],
            )
            mock_main.assert_called_once_with(["./my_recipe/", "--json"])


class TestInstallStep:
    def test_install_success_returns_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(validate_mod, "install_model", lambda *_a, **_kw: None)
        report = Report()
        assert _check_install(tmp_path, report) is True
        assert report.rows[0].status is Status.PASS

    def test_install_declined_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_aborted(*_a: Any, **_kw: Any) -> None:
            raise InstallAborted("user said no")

        monkeypatch.setattr(validate_mod, "install_model", _raise_aborted)
        report = Report()
        assert _check_install(tmp_path, report) is False
        assert report.rows[0].status is Status.FAIL
        assert "--no-install" in report.rows[0].detail

    def test_install_error_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_runtime(*_a: Any, **_kw: Any) -> None:
            raise RuntimeError("pip broke")

        monkeypatch.setattr(validate_mod, "install_model", _raise_runtime)
        report = Report()
        assert _check_install(tmp_path, report) is False
        assert report.rows[0].status is Status.FAIL


class TestNoInstallFlag:
    def test_no_install_emits_skip_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = False

        def _spy(*_a: Any, **_kw: Any) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(validate_mod, "install_model", _spy)
        # Stub everything else so we don't need a real recipe folder.
        for fn in (
            "_check_files_present",
            "_check_no_self_referential_imports",
            "_check_manifest",
            "_check_init_exports",
            "_check_model_code",
            "_check_app",
            "_check_evaluator",
            "_check_url_reachability",
        ):
            monkeypatch.setattr(validate_mod, fn, lambda *_a, **_kw: None)
        report = Report()
        _run_all_checks(tmp_path, report, skip_install=True)
        assert not called
        install_row = next(r for r in report.rows if r.category == "Install")
        assert install_row.status is Status.SKIP
        assert "--no-install" in install_row.detail

    def test_install_failure_skips_downstream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_runtime(*_a: Any, **_kw: Any) -> None:
            raise RuntimeError("pip broke")

        monkeypatch.setattr(validate_mod, "install_model", _raise_runtime)
        for fn in (
            "_check_files_present",
            "_check_no_self_referential_imports",
            "_check_manifest",
            "_check_url_reachability",
        ):
            monkeypatch.setattr(validate_mod, fn, lambda *_a, **_kw: None)

        model_import_called = False

        def _spy_model_code(*_a: Any, **_kw: Any) -> Any:
            nonlocal model_import_called
            model_import_called = True
            return None

        monkeypatch.setattr(validate_mod, "_check_model_code", _spy_model_code)

        report = Report()
        _run_all_checks(tmp_path, report, skip_install=False)
        assert not model_import_called
        # SKIP rows for downstream categories present.
        categories = {r.category for r in report.rows}
        assert {"Model code", "App", "Datasets / evaluator"} <= categories


class TestReportExitCode:
    def test_failed_report_exits_nonzero(self, tmp_path: Path) -> None:
        with (
            patch(
                "qai_hub_models.cli.validate.resolve_recipe_dir",
                return_value=tmp_path,
            ),
            patch(
                "qai_hub_models.cli.validate._run_all_checks",
                side_effect=lambda _sd, report, **_kw: report.add(
                    Result("x", "Manifest", Status.FAIL, "bad")
                ),
            ),
            pytest.raises(SystemExit) as exc,
        ):
            validate_mod.main(["fake-target"])
        assert exc.value.code == 1

    def test_passing_report_exits_zero(self, tmp_path: Path) -> None:
        with (
            patch(
                "qai_hub_models.cli.validate.resolve_recipe_dir",
                return_value=tmp_path,
            ),
            patch(
                "qai_hub_models.cli.validate._run_all_checks",
                side_effect=lambda _sd, report, **_kw: report.add(
                    Result("x", "Manifest", Status.PASS)
                ),
            ),
            pytest.raises(SystemExit) as exc,
        ):
            validate_mod.main(["fake-target"])
        assert exc.value.code == 0

    def test_warnings_do_not_fail(self, tmp_path: Path) -> None:
        with (
            patch(
                "qai_hub_models.cli.validate.resolve_recipe_dir",
                return_value=tmp_path,
            ),
            patch(
                "qai_hub_models.cli.validate._run_all_checks",
                side_effect=lambda _sd, report, **_kw: report.add(
                    Result("x", "Manifest", Status.WARN, "watch out")
                ),
            ),
            pytest.raises(SystemExit) as exc,
        ):
            validate_mod.main(["fake-target"])
        assert exc.value.code == 0


class TestInternalFlag:
    """--internal wires the extra check pass; unflagged runs skip it."""

    def _stub_run_all(
        self, source_dir: Path, report: Report, **_kw: Any
    ) -> QAIHMModelManifest:
        report.add(Result("x", "Manifest", Status.PASS))
        return _make_manifest(id=source_dir.name)

    def test_flag_absent_skips_internal(self, tmp_path: Path) -> None:
        with (
            patch(
                "qai_hub_models.cli.validate.resolve_recipe_dir",
                return_value=tmp_path,
            ),
            patch(
                "qai_hub_models.cli.validate._run_all_checks",
                side_effect=self._stub_run_all,
            ),
            patch("qai_hub_models.cli.validate._run_internal_checks") as run_internal,
            pytest.raises(SystemExit),
        ):
            validate_mod.main(["fake-target"])
        run_internal.assert_not_called()

    def test_flag_present_runs_internal(self, tmp_path: Path) -> None:
        with (
            patch(
                "qai_hub_models.cli.validate.resolve_recipe_dir",
                return_value=tmp_path,
            ),
            patch(
                "qai_hub_models.cli.validate._run_all_checks",
                side_effect=self._stub_run_all,
            ),
            patch("qai_hub_models.cli.validate._run_internal_checks") as run_internal,
            pytest.raises(SystemExit),
        ):
            validate_mod.main(["fake-target", "--internal"])
        run_internal.assert_called_once()


class TestInternalChecks:
    """Unit tests for individual --internal check functions."""

    def test_in_tree_status_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with patch("qai_hub_models.cli.validate._is_in_tree", return_value=True):
            report = Report()
            _check_in_tree_status(Path("/anywhere"), report)
        assert report.rows[0].status is Status.PASS

    def test_in_tree_status_fail(self) -> None:
        with patch("qai_hub_models.cli.validate._is_in_tree", return_value=False):
            report = Report()
            _check_in_tree_status(Path("/tmp/external"), report)
        assert report.rows[0].status is Status.FAIL

    def test_website_fields_missing(self) -> None:
        report = Report()
        _check_website_fields_set(
            _make_manifest(id="mymodel", name=None, headline=None), report
        )
        row = report.rows[0]
        assert row.status is Status.FAIL
        assert "name" in row.detail
        assert "headline" in row.detail

    def test_website_fields_all_set(self) -> None:
        report = Report()
        _check_website_fields_set(
            _make_manifest(id="mymodel", name="MyModel", headline="A model."),
            report,
        )
        assert report.rows[0].status is Status.PASS

    def test_name_style_underscore_fails(self) -> None:
        report = Report()
        _check_name_style(_make_manifest(name="my_model"), report)
        assert report.rows[0].status is Status.FAIL

    def test_name_style_space_fails(self) -> None:
        report = Report()
        _check_name_style(_make_manifest(name="my model"), report)
        assert report.rows[0].status is Status.FAIL

    def test_name_style_dashes_pass(self) -> None:
        report = Report()
        _check_name_style(_make_manifest(name="My-Model"), report)
        assert report.rows[0].status is Status.PASS

    def test_headline_missing_period_fails(self) -> None:
        report = Report()
        _check_headline_period(_make_manifest(headline="A model"), report)
        assert report.rows[0].status is Status.FAIL

    def test_headline_with_period_passes(self) -> None:
        report = Report()
        _check_headline_period(_make_manifest(headline="A model."), report)
        assert report.rows[0].status is Status.PASS

    def test_related_self_link_fails(self) -> None:
        report = Report()
        _check_related_not_self(
            _make_manifest(id="my_model", related_models=["my_model"]), report
        )
        assert report.rows[0].status is Status.FAIL

    def test_related_clean_passes(self) -> None:
        report = Report()
        _check_related_not_self(
            _make_manifest(id="my_model", related_models=["other"]), report
        )
        assert report.rows[0].status is Status.PASS

    def test_arxiv_pdf_link_fails(self) -> None:
        report = Report()
        _check_arxiv_abs(
            _make_manifest(research_paper="https://arxiv.org/pdf/1234.pdf"),
            report,
        )
        assert report.rows[0].status is Status.FAIL

    def test_arxiv_abs_link_passes(self) -> None:
        report = Report()
        _check_arxiv_abs(
            _make_manifest(research_paper="https://arxiv.org/abs/1234"), report
        )
        assert report.rows[0].status is Status.PASS

    def test_arxiv_non_arxiv_silent(self) -> None:
        report = Report()
        _check_arxiv_abs(_make_manifest(research_paper="https://acm.org/paper"), report)
        assert report.rows == []

    def test_status_reason_unpublished_missing_fails(self) -> None:
        report = Report()
        _check_status_reason(
            _make_manifest(status=MODEL_STATUS.UNPUBLISHED, status_reason=None),
            report,
        )
        assert report.rows[0].status is Status.FAIL

    def test_status_reason_published_present_fails(self) -> None:
        report = Report()
        _check_status_reason(
            _make_manifest(
                status=MODEL_STATUS.PUBLISHED,
                status_reason="tracked in #123",
            ),
            report,
        )
        assert report.rows[0].status is Status.FAIL

    def test_status_reason_unpublished_with_reason_passes(self) -> None:
        report = Report()
        _check_status_reason(
            _make_manifest(
                status=MODEL_STATUS.UNPUBLISHED,
                status_reason="tracked in #123",
            ),
            report,
        )
        assert report.rows[0].status is Status.PASS

    def test_default_device_canary_fails(self) -> None:
        with patch(
            "qai_hub_models.cli.validate.CANARY_DEVICES", {"Samsung Galaxy S24"}
        ):
            report = Report()
            _check_default_device_canary(
                _make_manifest(default_device="Pixel 4"), report
            )
        assert report.rows[0].status is Status.FAIL

    def test_default_device_canary_passes(self) -> None:
        with patch(
            "qai_hub_models.cli.validate.CANARY_DEVICES", {"Samsung Galaxy S24"}
        ):
            report = Report()
            _check_default_device_canary(
                _make_manifest(default_device="Samsung Galaxy S24"), report
            )
        assert report.rows[0].status is Status.PASS

    def test_run_internal_checks_skips_when_manifest_none(self, tmp_path: Path) -> None:
        with patch("qai_hub_models.cli.validate._is_in_tree", return_value=True):
            report = Report()
            _run_internal_checks(tmp_path, None, report)
        assert any(r.status is Status.SKIP for r in report.rows)


_SKIP_DETAIL = "skipped via --skip-url-checks"

_URL_MANIFEST = """\
id: skip_url_recipe
name: Skip-Url
headline: A model for testing.
description: A model for testing manifest validation.
domain: Computer Vision
use_case: Image Classification
license_type: apache-2.0
license: https://example.invalid/license
source_repo: https://example.invalid/repo
research_paper: https://arxiv.org/abs/1234.5678
research_paper_title: A Paper
status: pending
form_factors:
- Phone
applicable_scenarios:
- Inventory Management
supported_precisions:
- float
has_static_banner: true
has_animated_banner: true
"""


def _forbid_url_checks(_pairs: list[tuple[str, str]]) -> list[str]:
    raise AssertionError("head_check_urls should not run")


class TestSkipUrlChecks:
    """``--skip-url-checks`` omits reachability probes and reports SKIP."""

    def test_parser_defaults_and_documents_flag(self) -> None:
        parser = validate_mod.build_parser()
        assert parser.parse_args(["my_model"]).skip_url_checks is False
        assert parser.parse_args(["my_model", "--skip-url-checks"]).skip_url_checks
        help_text = " ".join(parser.format_help().split())
        assert "--skip-url-checks" in help_text
        assert "Skip explicit URL reachability checks only" in help_text
        assert (
            "Installation, initialization, and tests can still require network access."
            in help_text
        )

    def test_main_forwards_flag_to_ordinary_and_internal(self, tmp_path: Path) -> None:
        seen: list[tuple[str, bool]] = []

        def _all(_source: Path, report: Report, **kwargs: Any) -> Any:
            seen.append(("all", kwargs["skip_url_checks"]))
            report.add(Result("x", "Manifest", Status.PASS))
            return _make_manifest(id="mymodel")

        def _internal(
            _source: Path, _manifest: Any, _report: Report, **kwargs: Any
        ) -> None:
            seen.append(("internal", kwargs["skip_url_checks"]))

        def _invoke(argv: list[str]) -> None:
            with (
                patch(
                    "qai_hub_models.cli.validate.resolve_recipe_dir",
                    return_value=tmp_path,
                ),
                patch(
                    "qai_hub_models.cli.validate._run_all_checks",
                    side_effect=_all,
                ),
                patch(
                    "qai_hub_models.cli.validate._run_internal_checks",
                    side_effect=_internal,
                ),
                pytest.raises(SystemExit) as exc,
            ):
                validate_mod.main(argv)
            assert exc.value.code == 0

        _invoke(["fake-target", "--internal"])
        _invoke(["fake-target", "--internal", "--skip-url-checks"])
        _invoke(["fake-target", "--skip-url-checks"])
        assert seen == [
            ("all", False),
            ("internal", False),
            ("all", True),
            ("internal", True),
            ("all", True),
        ]

    def test_skip_alone_exits_zero_and_unrelated_failure_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = _make_manifest(
            license="https://example.invalid/license",
            source_repo=None,
            research_paper=None,
            external_repos=None,
        )
        monkeypatch.setattr(validate_mod, "head_check_urls", _forbid_url_checks)
        fail = False

        def _run(_source: Path, report: Report, **kwargs: Any) -> None:
            _check_url_reachability(
                manifest, report, skip_url_checks=kwargs["skip_url_checks"]
            )
            if fail:
                report.add(
                    Result("manifest.yaml loads", "Manifest", Status.FAIL, "nope")
                )

        with (
            patch(
                "qai_hub_models.cli.validate.resolve_recipe_dir",
                return_value=tmp_path,
            ),
            patch("qai_hub_models.cli.validate._run_all_checks", side_effect=_run),
            pytest.raises(SystemExit) as exc,
        ):
            validate_mod.main(["fake-target", "--skip-url-checks"])
        assert exc.value.code == 0

        fail = True
        with (
            patch(
                "qai_hub_models.cli.validate.resolve_recipe_dir",
                return_value=tmp_path,
            ),
            patch("qai_hub_models.cli.validate._run_all_checks", side_effect=_run),
            pytest.raises(SystemExit) as exc,
        ):
            validate_mod.main(["fake-target", "--skip-url-checks"])
        assert exc.value.code == 1

    def test_text_and_json_reports_show_skip_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = _make_manifest(
            id="skip_url_recipe",
            name="Skip-Url",
            license="https://example.invalid/license",
            source_repo="https://example.invalid/repo",
            research_paper=None,
            external_repos=None,
            has_static_banner=True,
            has_animated_banner=True,
        )
        monkeypatch.setattr(validate_mod, "head_check_urls", _forbid_url_checks)
        report = Report()
        _check_url_reachability(manifest, report, skip_url_checks=True)
        validate_mod._check_web_asset_urls(manifest, report, skip_url_checks=True)
        assert not report.failed

        text = _render_text(report)
        assert "SKIP" in text
        assert _SKIP_DETAIL in text
        assert "URL reachability" in text
        assert "Internal" in text

        payload = json.loads(_render_json(report))
        assert payload["failed"] is False
        assert payload["counts"]["FAIL"] == 0
        assert payload["counts"]["SKIP"] == len(payload["rows"])
        names = set()
        for row in payload["rows"]:
            assert row["status"] == "SKIP"
            assert row["detail"] == _SKIP_DETAIL
            names.add(row["name"])
        assert {"license", "source_repo", "static banner", "animated banner"} <= names

    def test_empty_urls_keep_existing_skip_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = _make_manifest(
            license=None,
            source_repo=None,
            research_paper=None,
            external_repos=None,
            id="skip_url_recipe",
            has_static_banner=False,
            has_animated_banner=False,
        )
        calls: list[list[tuple[str, str]]] = []

        def _spy(pairs: list[tuple[str, str]]) -> list[str]:
            calls.append(pairs)
            return []

        monkeypatch.setattr(validate_mod, "head_check_urls", _spy)
        report = Report()
        _check_url_reachability(manifest, report, skip_url_checks=True)
        validate_mod._check_web_asset_urls(manifest, report, skip_url_checks=True)
        assert calls == []
        assert report.rows == [
            Result(
                "URLs reachable",
                "URL reachability",
                Status.SKIP,
                "no URLs declared in manifest.",
            )
        ]

    @pytest.mark.parametrize("skip_install", [False, True])
    @pytest.mark.parametrize("skip_url_checks", [False, True])
    def test_missing_manifest_does_not_url_check(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        skip_install: bool,
        skip_url_checks: bool,
    ) -> None:
        source = tmp_path / "bare_recipe"
        source.mkdir()
        calls: list[list[tuple[str, str]]] = []

        def _spy(pairs: list[tuple[str, str]]) -> list[str]:
            calls.append(pairs)
            return []

        def _install(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("pip broke")

        monkeypatch.setattr(validate_mod, "head_check_urls", _spy)
        monkeypatch.setattr(validate_mod, "install_model", _install)
        report = Report()
        manifest = _run_all_checks(
            source,
            report,
            skip_install=skip_install,
            skip_url_checks=skip_url_checks,
        )
        assert manifest is None
        assert calls == []
        assert not any(r.category == "URL reachability" for r in report.rows)
        assert any(
            r.name == "manifest.yaml loads" and r.status is Status.FAIL
            for r in report.rows
        )

    @pytest.mark.parametrize(
        ("mode", "skip_url_checks"),
        [
            ("no_install", True),
            ("install_success", True),
            ("install_failure", True),
            ("no_install", False),
            ("install_success", False),
            ("install_failure", False),
        ],
    )
    def test_orchestration_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
        skip_url_checks: bool,
    ) -> None:
        source = tmp_path / "skip_url_recipe"
        source.mkdir()
        (source / "__init__.py").write_text("")
        (source / "model.py").write_text("")
        (source / "manifest.yaml").write_text(_URL_MANIFEST)

        calls: list[list[tuple[str, str]]] = []
        install_calls = 0
        downstream = {"init": 0, "model": 0, "pytest": 0}

        def _spy(pairs: list[tuple[str, str]]) -> list[str]:
            calls.append(list(pairs))
            return []

        def _install(*_args: Any, **_kwargs: Any) -> None:
            nonlocal install_calls
            install_calls += 1
            if mode == "install_failure":
                raise RuntimeError("pip broke")

        def _count(key: str) -> Any:
            def _inner(*_args: Any, **_kwargs: Any) -> None:
                downstream[key] += 1

            return _inner

        monkeypatch.setattr(validate_mod, "head_check_urls", _spy)
        monkeypatch.setattr(validate_mod, "install_model", _install)
        monkeypatch.setattr(validate_mod, "_is_in_tree", lambda _path: True)
        monkeypatch.setattr(validate_mod, "_check_init_exports", _count("init"))
        monkeypatch.setattr(validate_mod, "_check_model_code", _count("model"))
        monkeypatch.setattr(validate_mod, "_check_pytest", _count("pytest"))

        report = Report()
        manifest = _run_all_checks(
            source,
            report,
            skip_install=mode == "no_install",
            skip_url_checks=skip_url_checks,
        )
        assert manifest is not None
        _run_internal_checks(source, manifest, report, skip_url_checks=skip_url_checks)

        url_rows = [r for r in report.rows if r.category == "URL reachability"]
        assert {r.name for r in url_rows} == {
            "license",
            "source_repo",
            "research_paper",
        }
        banners = [
            r for r in report.rows if r.name in {"static banner", "animated banner"}
        ]
        assert {r.name for r in banners} == {"static banner", "animated banner"}
        if skip_url_checks:
            assert calls == []
            assert all(
                r.status is Status.SKIP and r.detail == _SKIP_DETAIL for r in url_rows
            )
            assert all(
                r.status is Status.SKIP and r.detail == _SKIP_DETAIL for r in banners
            )
        else:
            assert len(calls) == 2
            assert {url for url, _label in calls[0]} == {
                "https://example.invalid/license",
                "https://example.invalid/repo",
                "https://arxiv.org/abs/1234.5678",
            }
            assert {label for _url, label in calls[1]} == {
                "static banner unreachable",
                "animated banner unreachable",
            }
            assert all(r.status is Status.PASS for r in url_rows)
            assert all(r.status is Status.PASS for r in banners)

        assert any(
            r.name == "Required files present" and r.status is Status.PASS
            for r in report.rows
        )
        assert any(
            r.name == "demo.py present" and r.status is Status.WARN for r in report.rows
        )
        assert any(
            r.name == "name style" and r.status is Status.PASS for r in report.rows
        )
        install_row = next(r for r in report.rows if r.category == "Install")
        if mode == "no_install":
            assert install_calls == 0
            assert install_row.status is Status.SKIP
            assert install_row.detail == "skipped via --no-install"
        elif mode == "install_success":
            assert install_calls == 1
            assert install_row.status is Status.PASS
        else:
            assert install_calls == 1
            assert install_row.status is Status.FAIL
            assert "install failed" in {r.detail for r in report.rows}

        if mode == "install_failure":
            assert downstream == {"init": 0, "model": 0, "pytest": 0}
            assert report.failed
        else:
            assert downstream == {"init": 1, "model": 1, "pytest": 1}
            assert not report.failed

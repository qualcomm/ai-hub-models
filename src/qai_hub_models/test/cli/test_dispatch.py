# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for the heavy-side export/evaluate dispatcher."""

from __future__ import annotations

import argparse
import io
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.cli.dispatch import (
    _confirm_run_ok,
    build_evaluate_parser_for,
    build_export_parser_for,
    run_model_script,
)
from qai_hub_models.cli.upload_to_hf import build_parser as build_upload_parser
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.context import RecipeSourceUnavailableError


def test_dispatch_export_builds_parser_and_runs() -> None:
    """Export path: parser is built once from resolved recipe and pipeline invoked."""
    fake_parser = argparse.ArgumentParser()
    fake_parser.add_argument("--device")
    fake_parsed = argparse.Namespace(device="S25")

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/fake_model"),
        ) as mock_resolve,
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ) as mock_build,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
        patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=True),
    ):
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("fake_model", "export", ["--device", "S25"])

        mock_resolve.assert_called_once_with("fake_model")
        mock_build.assert_called_once()
        mock_select.assert_called_once()


def test_dispatch_export_prompts_for_unpublished_model() -> None:
    """Export for `status: unpublished` recipe prompts and exits early on decline."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/sam"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ),
        patch(
            "qai_hub_models.cli.dispatch.resolve_publish_status",
            return_value=MODEL_STATUS.UNPUBLISHED,
        ),
        patch("qai_hub_models.cli.dispatch.resolve_manifest"),
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
            return_value=False,
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
    ):
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("sam", "export", [])

        mock_check.assert_called_once()
        mock_select.assert_not_called()


def test_dispatch_export_prompts_for_pending_model() -> None:
    """Export for ``status: pending`` recipe prompts — only PUBLISHED skips."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/mobile_facenet"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ),
        patch(
            "qai_hub_models.cli.dispatch.resolve_publish_status",
            return_value=MODEL_STATUS.PENDING,
        ),
        patch("qai_hub_models.cli.dispatch.resolve_manifest"),
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
            return_value=False,
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
    ):
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("mobile_facenet", "export", [])

        mock_check.assert_called_once()
        mock_select.assert_not_called()


def test_dispatch_export_does_not_prompt_for_external_recipe() -> None:
    """External recipes have no status, and the catalog warning does not apply."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/external_recipe"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ),
        patch(
            "qai_hub_models.cli.dispatch.resolve_publish_status",
            return_value=None,
        ),
        patch("qai_hub_models.cli.dispatch.resolve_manifest"),
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
            return_value=False,
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
    ):
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("external_recipe", "export", [])

        mock_check.assert_not_called()
        mock_select.assert_called_once()


def test_dispatch_export_skips_prompt_for_published_model() -> None:
    """Export for ``status: published`` recipe runs silently — no prompt."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/resnet50"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ),
        patch(
            "qai_hub_models.cli.dispatch.resolve_publish_status",
            return_value=MODEL_STATUS.PUBLISHED,
        ),
        patch("qai_hub_models.cli.dispatch.resolve_manifest"),
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
    ):
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("resnet50", "export", [])

        mock_check.assert_not_called()
        mock_select.assert_called_once()


def test_dispatch_evaluate_builds_parser_and_runs() -> None:
    """Evaluate path: parser is built once from resolved recipe and pipeline invoked."""
    fake_parser = argparse.ArgumentParser()
    fake_parser.add_argument("--device")
    fake_parsed = argparse.Namespace(device="S25")

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/fake_model"),
        ) as mock_resolve,
        patch(
            "qai_hub_models.cli.dispatch.build_evaluate_parser_for",
            return_value=fake_parser,
        ) as mock_build,
        patch("qai_hub_models.cli.dispatch.select_evaluate_pipeline") as mock_select,
        patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=True),
        patch("qai_hub_models.cli.dispatch.resolve_manifest") as mock_manifest,
    ):
        mock_manifest.return_value.model_type_llm = False
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("fake_model", "evaluate", ["--device", "S25"])

        mock_resolve.assert_called_once_with("fake_model")
        mock_build.assert_called_once()
        mock_select.assert_called_once()


def test_dispatch_evaluate_prompts_for_unpublished_model() -> None:
    """Evaluate for `status: unpublished` recipe prompts and exits early on decline."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()
    fake_manifest = Mock()
    fake_manifest.model_type_llm = False

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_recipe_dir",
            return_value=Path("/tmp/sam"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_evaluate_parser_for",
            return_value=fake_parser,
        ),
        patch(
            "qai_hub_models.cli.dispatch.resolve_publish_status",
            return_value=MODEL_STATUS.UNPUBLISHED,
        ),
        patch(
            "qai_hub_models.cli.dispatch.resolve_manifest",
            return_value=fake_manifest,
        ),
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
            return_value=False,
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_evaluate_pipeline") as mock_select,
    ):
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("sam", "evaluate", [])

        mock_check.assert_called_once()
        mock_select.assert_not_called()


def test_export_parser_uses_export_paths() -> None:
    """
    Bug fix: JIT models were rejecting `--target-runtime qnn_context_binary`
    because the parser was fed `get_supported_paths_for_testing()`, which
    drops AOT runtimes for JIT models. The parser must be fed
    `get_supported_paths_for_export()`.
    """
    manifest = Mock()
    export_paths = {Precision.w8a8: [TargetRuntime.QNN_CONTEXT_BINARY]}
    testing_paths = {Precision.w8a8: [TargetRuntime.QNN_DLC]}
    manifest.get_supported_paths_for_export.return_value = export_paths
    manifest.get_supported_paths_for_testing.return_value = testing_paths
    manifest.default_device = None
    manifest.separate_quantize_script = False

    with (
        patch("qai_hub_models.cli.dispatch.resolve_manifest", return_value=manifest),
        patch("qai_hub_models.cli.dispatch.resolve_model_cls", return_value=Mock()),
        patch("qai_hub_models.cli.dispatch.select_pipeline"),
        patch("qai_hub_models.cli.dispatch.export_parser") as mock_export_parser,
    ):
        build_export_parser_for(Path("/tmp/fake_model"))

        kwargs = mock_export_parser.call_args.kwargs
        assert kwargs["supported_precision_runtimes"] is export_paths
        assert kwargs["supported_precision_runtimes"] is not testing_paths


def test_evaluate_parser_uses_export_paths() -> None:
    """
    Same bug applies to evaluate: JIT models must accept AOT runtimes via
    the JIT-compile + link path. Parser must be fed
    `get_supported_paths_for_export()`.
    """
    manifest = Mock()
    export_paths = {Precision.w8a8: [TargetRuntime.QNN_CONTEXT_BINARY]}
    testing_paths = {Precision.w8a8: [TargetRuntime.QNN_DLC]}
    manifest.get_supported_paths_for_export.return_value = export_paths
    manifest.get_supported_paths_for_testing.return_value = testing_paths
    manifest.default_device = None
    manifest.num_calibration_samples = None
    manifest.can_use_quantize_job = False
    manifest.supports_quantization = False

    model_cls = Mock(spec=BaseModel)
    model_cls.get_eval_dataset_classes.return_value = []

    with (
        patch("qai_hub_models.cli.dispatch.resolve_manifest", return_value=manifest),
        patch("qai_hub_models.cli.dispatch.resolve_model_cls", return_value=model_cls),
        patch("qai_hub_models.cli.dispatch.evaluate_parser") as mock_evaluate_parser,
    ):
        build_evaluate_parser_for(Path("/tmp/fake_model"))

        kwargs = mock_evaluate_parser.call_args.kwargs
        assert kwargs["supported_precision_runtimes"] is export_paths
        assert kwargs["supported_precision_runtimes"] is not testing_paths


class TestDisabledPathHandling:
    """A path with a manifest failure is still selectable, but not the default,
    and not runnable without confirming.
    """

    @staticmethod
    def _manifest(failing: set[tuple[Precision, TargetRuntime]]) -> Mock:
        manifest = Mock()
        paths = {Precision.float: [TargetRuntime.TFLITE, TargetRuntime.QNN_DLC]}
        manifest.get_supported_paths_for_export.return_value = paths
        manifest.failure_reason.side_effect = (
            lambda p, r: "recorded failure" if (p, r) in failing else None
        )
        manifest.default_device = None
        manifest.separate_quantize_script = False
        return manifest

    def _build(self, manifest: Mock) -> Mock:
        with (
            patch(
                "qai_hub_models.cli.dispatch.resolve_manifest", return_value=manifest
            ),
            patch("qai_hub_models.cli.dispatch.resolve_model_cls", return_value=Mock()),
            patch("qai_hub_models.cli.dispatch.select_pipeline"),
            patch("qai_hub_models.cli.dispatch.export_parser") as mock_parser,
        ):
            build_export_parser_for(Path("/tmp/fake_model"))
        return mock_parser

    def test_default_skips_the_failing_runtime(self) -> None:
        """Tflite is first in TargetRuntime order, so without this the default
        would be a runtime the manifest records as broken.
        """
        manifest = self._manifest({(Precision.float, TargetRuntime.TFLITE)})
        mock_parser = self._build(manifest)
        preferred = mock_parser.return_value.set_preferred_precision_runtimes
        preferred.assert_called_once_with({Precision.float: [TargetRuntime.QNN_DLC]})

    def test_all_runtimes_stay_selectable(self) -> None:
        manifest = self._manifest({(Precision.float, TargetRuntime.TFLITE)})
        mock_parser = self._build(manifest)
        kwargs = mock_parser.call_args.kwargs
        assert kwargs["supported_precision_runtimes"][Precision.float] == [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN_DLC,
        ]

    def test_no_failures_prefers_everything(self) -> None:
        mock_parser = self._build(self._manifest(set()))
        preferred = mock_parser.return_value.set_preferred_precision_runtimes
        preferred.assert_called_once_with(
            {Precision.float: [TargetRuntime.TFLITE, TargetRuntime.QNN_DLC]}
        )

    def test_running_a_failing_path_prompts_and_can_decline(self) -> None:
        manifest = self._manifest({(Precision.float, TargetRuntime.TFLITE)})
        args = argparse.Namespace(
            precision=Precision.float, target_runtime=TargetRuntime.TFLITE
        )
        with (
            patch(
                "qai_hub_models.cli.dispatch.resolve_manifest", return_value=manifest
            ),
            patch(
                "qai_hub_models.cli.dispatch.check_disabled_path_warning",
                return_value=False,
            ) as mock_check,
        ):
            assert _confirm_run_ok(Path("/tmp/fake_model"), args) is False
        mock_check.assert_called_once()
        assert "recorded failure" in mock_check.call_args.args[1]

    def test_running_a_passing_path_does_not_prompt(self) -> None:
        manifest = self._manifest({(Precision.float, TargetRuntime.TFLITE)})
        args = argparse.Namespace(
            precision=Precision.float, target_runtime=TargetRuntime.QNN_DLC
        )
        with (
            patch(
                "qai_hub_models.cli.dispatch.resolve_manifest", return_value=manifest
            ),
            patch(
                "qai_hub_models.cli.dispatch.check_disabled_path_warning"
            ) as mock_check,
        ):
            assert _confirm_run_ok(Path("/tmp/fake_model"), args) is True
        mock_check.assert_not_called()


class TestRecipeCommandHelp:
    """`<cmd> --help` with no target should show real flags where it can.

    Lives in the heavy suite because it imports the heavy-side parsers, which
    the lean suite's conftest deliberately blocks.
    """

    def test_every_recipe_command_gets_one_or_the_other(self) -> None:
        """No recipe command falls through both branches."""
        from qai_hub_models_cli.cli import _RECIPE_COMMANDS, _print_recipe_command_help

        for script in _RECIPE_COMMANDS:
            stream = io.StringIO()
            _print_recipe_command_help(script, stream)
            assert stream.getvalue().strip(), script

    @pytest.mark.parametrize(
        ("script", "expected_flag"),
        [
            ("upload-to-hf", "--private"),
            ("install", "--dry-run"),
            ("validate", "--no-install"),
            ("generate-files", "target"),
        ],
    )
    def test_static_commands_print_their_real_flags(
        self, capsys: pytest.CaptureFixture[str], script: str, expected_flag: str
    ) -> None:
        from qai_hub_models_cli.cli import _print_recipe_command_help

        _print_recipe_command_help(script)
        out = capsys.readouterr().out
        assert expected_flag in out
        assert f"qai-hub-models {script}" in out

    @pytest.mark.parametrize("script", ["export", "evaluate"])
    def test_recipe_dependent_commands_explain_why_they_cannot(
        self, capsys: pytest.CaptureFixture[str], script: str
    ) -> None:
        """export/evaluate flags come from the recipe, so say that rather than
        pretending a flag list exists.
        """
        from qai_hub_models_cli.cli import _print_recipe_command_help

        _print_recipe_command_help(script)
        out = capsys.readouterr().out
        assert "come from the recipe" in out
        assert f"qai-hub-models {script} <target> --help" in out

    def test_upload_to_hf_help_says_the_target_is_a_folder(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """It reads the target as a folder; advertising ids would be wrong."""
        from qai_hub_models_cli.cli import _print_recipe_command_help

        _print_recipe_command_help("upload-to-hf")
        out = capsys.readouterr().out
        assert "not a model id" in out


def test_dispatch_upload_to_hf_forwards_argv() -> None:
    """upload-to-hf resolves its own target, so dispatch must not pre-resolve it."""
    forwarded = ["--dry-run", "--no-tag"]
    # Parsed for real: main is mocked below, so nothing else here would notice
    # if one of these flags were removed from the command.
    build_upload_parser().parse_args(["my_model", *forwarded])

    with (
        patch("qai_hub_models.cli.dispatch.upload_to_hf_main") as mock_main,
        patch("qai_hub_models.cli.dispatch.resolve_recipe_dir") as mock_resolve,
    ):
        run_model_script("./my_model/", "upload-to-hf", forwarded)

    mock_main.assert_called_once_with(["./my_model/", *forwarded])
    mock_resolve.assert_not_called()


class TestDispatchDemo:
    """`demo` hands its argv tail to the recipe's own demo.main()."""

    def _run(self, forwarded: list[str], confirm: bool = True) -> list[str]:
        """Run the demo branch and return the sys.argv its main() observed."""
        seen: list[str] = []
        demo_module = types.ModuleType("fake_model.demo")
        demo_module.main = lambda: seen.extend(sys.argv)  # type: ignore[attr-defined]
        recipe_module = types.ModuleType("fake_model")

        with (
            patch.dict(
                sys.modules,
                {"fake_model": recipe_module, "fake_model.demo": demo_module},
            ),
            patch(
                "qai_hub_models.cli.dispatch.resolve_recipe_dir",
                return_value=Path("/tmp/fake_model"),
            ),
            patch(
                "qai_hub_models.cli.dispatch.import_recipe_module",
                return_value=recipe_module,
            ),
            patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=confirm),
        ):
            run_model_script("fake_model", "demo", forwarded)
        return seen

    def test_forwards_the_tail_as_argv(self) -> None:
        """Demo scripts read sys.argv, so the tail has to arrive that way."""
        assert self._run(["--eval-mode", "on-device"]) == [
            "fake_model.demo",
            "--eval-mode",
            "on-device",
        ]

    def test_restores_argv_afterwards(self) -> None:
        before = list(sys.argv)
        self._run(["--image", "x.png"])
        assert sys.argv == before

    def test_declining_the_prompt_skips_the_demo(self) -> None:
        assert self._run([], confirm=False) == []


def test_dispatch_evaluate_prefers_recipe_build_parser(tmp_path: Path) -> None:
    """LLM recipes add flags (--task, --use-presplit) the generic parser lacks.

    Building the generic parser for them rejected every one of those flags.
    """
    (tmp_path / "evaluate.py").write_text("")
    seen: list[argparse.Namespace] = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    evaluate_module = types.ModuleType("fake_model.evaluate")
    evaluate_module.build_parser = lambda: parser  # type: ignore[attr-defined]
    evaluate_module.main = seen.append  # type: ignore[attr-defined]

    with (
        patch.dict(sys.modules, {"fake_model.evaluate": evaluate_module}),
        patch("qai_hub_models.cli.dispatch.resolve_recipe_dir", return_value=tmp_path),
        patch(
            "qai_hub_models.cli.dispatch.import_recipe_module",
            return_value=types.ModuleType("fake_model"),
        ),
        patch("qai_hub_models.cli.dispatch.resolve_manifest") as mock_manifest,
        patch("qai_hub_models.cli.dispatch._passing_paths", return_value={}),
        patch("qai_hub_models.cli.dispatch.build_evaluate_parser_for") as generic,
        patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=True),
    ):
        mock_manifest.return_value.model_type_llm = False
        parser.set_preferred_precision_runtimes = Mock()  # type: ignore[attr-defined]
        run_model_script("fake_model", "evaluate", ["--task", "mmlu"])

    generic.assert_not_called()
    assert [ns.task for ns in seen] == ["mmlu"]


def test_dispatch_demo_without_main_runs_main_block(tmp_path: Path) -> None:
    """22 recipes' demo.py has only an `if __name__ == "__main__":` block.

    Calling `.main()` on those crashed with AttributeError instead of running.
    """
    pkg = tmp_path / "fake_recipe_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    out = tmp_path / "argv.txt"
    (pkg / "demo.py").write_text(
        "import sys\n"
        'if __name__ == "__main__":\n'
        f"    open({str(out)!r}, 'w').write(' '.join(sys.argv[1:]))\n"
    )
    with (
        patch.object(sys, "path", [str(tmp_path), *sys.path]),
        patch("qai_hub_models.cli.dispatch.resolve_recipe_dir", return_value=pkg),
        patch(
            "qai_hub_models.cli.dispatch.import_recipe_module",
            return_value=types.ModuleType("fake_recipe_pkg"),
        ),
        patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=True),
    ):
        run_model_script("fake_recipe_pkg", "demo", ["--prompt", "hi"])
    sys.modules.pop("fake_recipe_pkg.demo", None)
    sys.modules.pop("fake_recipe_pkg", None)

    assert out.read_text() == "--prompt hi"


class TestDispatchLLMEvaluate:
    """LLM `evaluate` runs the recipe's host-side evaluate.py, not the Workbench pipeline."""

    def _run(
        self, tmp_path: Path, forwarded: list[str], confirm: bool = True
    ) -> tuple[Mock, list[str]]:
        """Run the LLM evaluate branch; return the run_module mock and the argv it saw."""
        argv_seen: list[str] = []
        with (
            patch(
                "qai_hub_models.cli.dispatch.resolve_recipe_dir",
                return_value=tmp_path,
            ),
            patch("qai_hub_models.cli.dispatch.resolve_manifest") as mock_manifest,
            patch(
                "qai_hub_models.cli.dispatch.import_recipe_module",
                return_value=types.ModuleType("fake_llm"),
            ),
            patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=confirm),
            patch(
                "qai_hub_models.cli.dispatch.runpy.run_module",
                side_effect=lambda *a, **k: argv_seen.extend(sys.argv),
            ) as mock_run,
            patch(
                "qai_hub_models.cli.dispatch.build_evaluate_parser_for",
                side_effect=AssertionError("Workbench evaluate parser used for LLM"),
            ),
        ):
            mock_manifest.return_value.model_type_llm = True
            run_model_script("fake_llm", "evaluate", forwarded)
        return mock_run, argv_seen

    def test_runs_the_recipe_evaluate_script(self, tmp_path: Path) -> None:
        (tmp_path / "evaluate.py").touch()
        before = list(sys.argv)
        mock_run, argv_seen = self._run(tmp_path, ["--task", "wikitext"])

        mock_run.assert_called_once_with(
            "fake_llm.evaluate", run_name="__main__", alter_sys=False
        )
        assert argv_seen == ["fake_llm.evaluate", "--task", "wikitext"]
        assert sys.argv == before

    def test_declining_the_prompt_skips_evaluate(self, tmp_path: Path) -> None:
        (tmp_path / "evaluate.py").touch()
        mock_run, _ = self._run(tmp_path, [], confirm=False)
        mock_run.assert_not_called()

    def test_missing_evaluate_script_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RecipeSourceUnavailableError, match="no evaluate script"):
            self._run(tmp_path, [])

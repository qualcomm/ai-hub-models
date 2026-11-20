# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import sys
import types
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from qai_hub_models_cli.cli import (
    _build_parser,
    _check_version_match,
    main,
)
from qai_hub_models_cli.common import SUPPORT_EMAIL
from qai_hub_models_cli.errors import CLIUserError


def _subcommand_choices(parser: argparse.ArgumentParser) -> set[str]:
    """Return the set of subcommand names registered on the top-level parser."""
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    return set(subparsers_action.choices)


class _StubRecipeSourceUnavailableError(CLIUserError):
    """Stands in for the heavy side's ``RecipeSourceUnavailableError``."""


def _stub_heavy_modules(
    model_ids: set[str],
    run_model_script: MagicMock | None = None,
    static_help: bool = False,
) -> dict[str, types.ModuleType]:
    """Patch the heavy ``qai_hub_models`` modules the dispatch code imports.

    ``static_help`` is what ``print_command_help`` returns: False stands for a
    command whose flags come from the resolved recipe, so the lean CLI prints
    its own hint instead.
    """
    path_helpers = types.ModuleType("qai_hub_models.utils.path_helpers")
    path_helpers.MODEL_IDS = model_ids  # type: ignore[attr-defined]
    dispatch = types.ModuleType("qai_hub_models.cli.dispatch")
    dispatch.run_model_script = run_model_script or MagicMock()  # type: ignore[attr-defined]
    command_help = types.ModuleType("qai_hub_models.cli.command_help")
    command_help.print_command_help = lambda script, stream: static_help  # type: ignore[attr-defined]
    context = types.ModuleType("qai_hub_models.utils.export.context")
    context.RecipeSourceUnavailableError = _StubRecipeSourceUnavailableError  # type: ignore[attr-defined]
    return {
        "qai_hub_models": types.ModuleType("qai_hub_models"),
        "qai_hub_models.utils": types.ModuleType("qai_hub_models.utils"),
        "qai_hub_models.utils.path_helpers": path_helpers,
        "qai_hub_models.utils.export": types.ModuleType("qai_hub_models.utils.export"),
        "qai_hub_models.utils.export.context": context,
        "qai_hub_models.cli": types.ModuleType("qai_hub_models.cli"),
        "qai_hub_models.cli.dispatch": dispatch,
        "qai_hub_models.cli.command_help": command_help,
    }


@pytest.mark.parametrize(
    ("cli_v", "models_v", "should_exit"),
    [
        ("1.0.0", "1.0.0", False),  # match
        ("1.0.0", "2.0.0", True),  # mismatch
        ("1.0.0", None, False),  # qai_hub_models not installed
    ],
)
def test_check_version_match(
    cli_v: str, models_v: str | None, should_exit: bool
) -> None:
    """`_check_version_match` exits iff cli/models versions are both installed and differ."""

    def _version(pkg: str) -> str:
        if pkg == "qai_hub_models_cli":
            return cli_v
        if models_v is None:
            raise PackageNotFoundError(pkg)
        return models_v

    with patch("qai_hub_models_cli.cli.version", side_effect=_version):
        if should_exit:
            with pytest.raises(SystemExit):
                _check_version_match()
        else:
            _check_version_match()


# ── two-phase dispatch for export/evaluate/install ──────────────────


@pytest.mark.parametrize("script", ["export", "evaluate", "install"])
def test_dispatch_forwards_remaining_args_to_model_parser(script: str) -> None:
    """`<script> <model> --flag value` hands model-specific args to dispatch verbatim.

    When the model arg is already a valid installed model ID, dispatch skips the
    (slow) manifest lookup and forwards straight to the model's parser.
    """
    fake_entry = MagicMock()
    fake_entry.id = "mobilenet_v2"
    mock_run = MagicMock()
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch("qai_hub_models_cli.cli.CURRENT_VERSION", "9.9.9"),
        patch(
            "qai_hub_models_cli.cli.get_release_manifest_entry", return_value=fake_entry
        ) as mock_get_entry,
        patch.dict(
            sys.modules,
            _stub_heavy_modules({"mobilenet_v2"}, run_model_script=mock_run),
        ),
    ):
        main([script, "mobilenet_v2", "--target-runtime", "tflite"])
    # mobilenet_v2 is a valid installed model ID, so the manifest lookup is skipped.
    mock_get_entry.assert_not_called()
    mock_run.assert_called_once_with(
        model_id="mobilenet_v2",
        script=script,
        forwarded=["--target-runtime", "tflite"],
    )


def test_dispatch_surfaces_source_unavailable_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A source-less recipe prints its own message, not the support-email text."""
    mock_run = MagicMock(
        side_effect=_StubRecipeSourceUnavailableError(
            "this recipe cannot be exported here"
        )
    )
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch.dict(
            sys.modules, _stub_heavy_modules({"sam3"}, run_model_script=mock_run)
        ),
        pytest.raises(SystemExit),
    ):
        main(["export", "sam3", "--help"])
    out = capsys.readouterr().out
    assert "this recipe cannot be exported here" in out
    assert "ai-hub-support@qti.qualcomm.com" not in out


def test_dispatch_surfaces_user_error_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A CLIUserError prints only its own text: no stack trace, no support email.

    Unknown ids, bad flags and missing datasets are the user's to fix; a stack
    trace or bug report prompt would bury the one line that says how.
    """
    mock_run = MagicMock(
        side_effect=CLIUserError("Unknown model 'typo'. Did you mean 'mobilenet_v2'?")
    )
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch.dict(
            sys.modules,
            _stub_heavy_modules({"mobilenet_v2"}, run_model_script=mock_run),
        ),
        pytest.raises(SystemExit),
    ):
        main(["export", "mobilenet_v2"])
    captured = capsys.readouterr()
    assert "Did you mean 'mobilenet_v2'?" in captured.out
    assert "Traceback" not in captured.err
    assert SUPPORT_EMAIL not in captured.out + captured.err


def test_dispatch_unexpected_error_shows_traceback_and_support_email(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-CLIUserError is a bug: users need the trace and where to send it."""
    mock_run = MagicMock(side_effect=AttributeError("'NoneType' has no attribute 'x'"))
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch.dict(
            sys.modules,
            _stub_heavy_modules({"mobilenet_v2"}, run_model_script=mock_run),
        ),
        pytest.raises(SystemExit) as exc,
    ):
        main(["export", "mobilenet_v2"])
    err = capsys.readouterr().err
    assert exc.value.code == 1
    assert "Traceback" in err
    assert "'NoneType' has no attribute 'x'" in err
    assert SUPPORT_EMAIL in err


def test_dispatch_missing_model_arg_exits_with_usage_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`export` (no target) prints our usage hint as a usage error.

    Stderr and exit 2, matching argparse, since a missing argument is an error
    rather than a help request.
    """
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch.dict(sys.modules, _stub_heavy_modules(set())),
        pytest.raises(SystemExit) as exc_info,
    ):
        main(["export"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Usage:" in err
    assert "export <target>" in err


def test_dispatch_help_flag_exits_zero_on_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`export -h` is a help request, so stdout and exit 0."""
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch.dict(sys.modules, _stub_heavy_modules(set())),
        pytest.raises(SystemExit) as exc_info,
    ):
        main(["export", "-h"])
    assert exc_info.value.code == 0
    assert "export <target>" in capsys.readouterr().out


@pytest.mark.parametrize(
    "script", ["demo", "export", "evaluate", "install", "generate-files"]
)
def test_dispatch_resolves_a_registered_alias_for_every_command(script: str) -> None:
    """A registered folder is a valid target for every recipe command.

    ``install`` and ``generate-files`` used to refuse one, even though both
    resolve an absolute folder path — which is exactly what ``resolve_alias``
    returns. That left a registered recipe with no way to install its deps.
    """
    mock_run = MagicMock()
    folder = Path("/recipes/my_recipe")
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch("qai_hub_models_cli.cli.resolve_alias", return_value=folder),
        patch.dict(sys.modules, _stub_heavy_modules(set(), run_model_script=mock_run)),
    ):
        main([script, "my_recipe"])
    mock_run.assert_called_once_with(model_id=folder, script=script, forwarded=[])


def test_dispatch_builtin_id_wins_over_alias() -> None:
    """A registered alias must not shadow a built-in model id.

    ``register_alias``'s collision check only fires when heavy is installed,
    so someone could register e.g. ``mobilenet_v2`` from a lean CLI and later
    install heavy — dispatch has to prefer the built-in either way.
    """
    mock_run = MagicMock()
    mock_resolve_alias = MagicMock(return_value=None)
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch("qai_hub_models_cli.cli.resolve_alias", mock_resolve_alias),
        patch.dict(
            sys.modules,
            _stub_heavy_modules({"mobilenet_v2"}, run_model_script=mock_run),
        ),
    ):
        main(["export", "mobilenet_v2"])
    mock_resolve_alias.assert_not_called()
    mock_run.assert_called_once_with(
        model_id="mobilenet_v2", script="export", forwarded=[]
    )


# ── register ────────────────────────────────────────────────────────


class TestRegisterAlias:
    """`register <target> [--alias]` derives the name unless one is given."""

    def _run(self, argv: list[str]) -> MagicMock:
        mock_register = MagicMock(return_value="/tmp/recipe")
        with (
            patch("qai_hub_models_cli.cli._check_version_match"),
            patch("qai_hub_models_cli.cli.register_alias", mock_register),
        ):
            main(argv)
        return mock_register

    def test_derives_the_name_from_the_target(self) -> None:
        mock = self._run(["register", "ashwmurt/yolov8_pose"])
        assert mock.call_args.args == ("yolov8_pose", "ashwmurt/yolov8_pose")

    def test_alias_overrides_the_derived_name(self) -> None:
        mock = self._run(["register", "ashwmurt/yolov8_pose", "--alias", "my_yolo"])
        assert mock.call_args.args == ("my_yolo", "ashwmurt/yolov8_pose")

    def test_version_is_forwarded_as_the_hf_revision(self) -> None:
        mock = self._run(["register", "ashwmurt/yolov8_pose", "--version", "v2"])
        assert mock.call_args.kwargs["revision"] == "v2"

    def test_no_version_means_latest(self) -> None:
        mock = self._run(["register", "ashwmurt/yolov8_pose"])
        assert mock.call_args.kwargs["revision"] is None

    def test_prints_the_name_it_registered(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(["register", "ashwmurt/yolov8_pose"])
        assert "'yolov8_pose'" in capsys.readouterr().out

    def test_an_underivable_name_exits_before_registering(self) -> None:
        mock_register = MagicMock()
        with (
            patch("qai_hub_models_cli.cli._check_version_match"),
            patch("qai_hub_models_cli.cli.register_alias", mock_register),
            pytest.raises(SystemExit, match="--alias"),
        ):
            main(["register", "owner/mod:el!"])
        mock_register.assert_not_called()


# ── top-level help completeness ─────────────────────────────────────


def test_every_registered_subcommand_is_listed_in_help() -> None:
    """A command missing from `sections` works but is invisible in --help.

    `_GroupedHelpFormatter` renders only the names `sections` lists, so the two
    hand-maintained lists in `_build_parser` can drift silently.
    """
    parser = _build_parser()
    # formatter_class is a functools.partial carrying the sections table.
    sections = cast(Any, parser.formatter_class).keywords["sections"]
    listed = {name for names in sections.values() for name in names}
    assert _subcommand_choices(parser) - listed == set()


# ── configure-dataset ───────────────────────────────────────────────


def test_configure_dataset_forwards_the_class_path_and_files() -> None:
    """Its target is a dotted dataset class, not a recipe, so it must pass through."""
    mock_run = MagicMock()
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.is_heavy_package_installed", return_value=True),
        patch("qai_hub_models_cli.cli.resolve_alias", return_value=None),
        patch.dict(sys.modules, _stub_heavy_modules(set(), run_model_script=mock_run)),
    ):
        main(
            [
                "configure-dataset",
                "qai_hub_models.datasets.kitti.kitti.KittiDataset",
                "--files",
                "a.zip",
                "b.zip",
            ]
        )
    mock_run.assert_called_once_with(
        model_id="qai_hub_models.datasets.kitti.kitti.KittiDataset",
        script="configure-dataset",
        forwarded=["--files", "a.zip", "b.zip"],
    )

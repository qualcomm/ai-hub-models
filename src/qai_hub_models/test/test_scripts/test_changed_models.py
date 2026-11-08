# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

# changes.py lives in scripts/tasks/ outside the installed package.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TASKS_DIR = _REPO_ROOT / "scripts" / "tasks"

# Register 'tasks' as a namespace package so submodule imports work.
_tasks_pkg = types.ModuleType("tasks")
_tasks_pkg.__path__ = [str(_TASKS_DIR)]
sys.modules["tasks"] = _tasks_pkg


def _load(name: str) -> types.ModuleType:
    path = _TASKS_DIR / (name.split(".")[-1] + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("tasks.constants")
_load("tasks.github")
_load("tasks.plan")
_load("tasks.util")
_changes = _load("tasks.changes")

get_python_import_expression = _changes.get_python_import_expression
resolve_affected_models = _changes.resolve_affected_models
prune_llm_groups = _changes.prune_llm_groups


def test_import_expression_matches_real_imports() -> None:
    """
    Verify get_python_import_expression produces import paths that
    actually appear in the codebase (no stale 'src.' prefix, etc.).
    """
    filepath = "src/qai_hub_models/models/mobilenet_v2/model.py"
    expr = get_python_import_expression(filepath)

    assert not expr.startswith("src."), (
        f"Import expression '{expr}' has 'src.' prefix — "
        "imports in code use 'qai_hub_models.*', not 'src.qai_hub_models.*'"
    )
    assert expr == "qai_hub_models.models.mobilenet_v2.model"


def test_import_expression_for_template_module() -> None:
    """Template modules under templates/ should also resolve correctly."""
    filepath = "src/qai_hub_models/models/templates/diffusion/model.py"
    expr = get_python_import_expression(filepath)
    assert expr == "qai_hub_models.models.templates.diffusion.model"


def test_import_expression_for_init() -> None:
    """__init__.py should strip the filename."""
    filepath = "src/qai_hub_models/models/mobilenet_v2/__init__.py"
    expr = get_python_import_expression(filepath)
    assert expr == "qai_hub_models.models.mobilenet_v2"


def test_template_change_detects_dependent_models() -> None:
    """
    Changing a templates module should detect the concrete models
    that import from it.
    """
    changed = ["src/qai_hub_models/models/templates/diffusion/model.py"]
    models = resolve_affected_models(changed)
    assert "stable_diffusion_v1_5" in models
    assert "controlnet_canny" in models


# ── End-to-end: simulated git diffs → affected models ──────────────


def test_direct_model_file_change() -> None:
    """Changing a model's own model.py should detect that model."""
    models = resolve_affected_models(
        ["src/qai_hub_models/models/mobilenet_v2/model.py"]
    )
    assert "mobilenet_v2" in models


def test_template_whisper_change_detects_whisper_models() -> None:
    """Changing templates/hf_whisper should detect whisper variants."""
    models = resolve_affected_models(
        ["src/qai_hub_models/models/templates/hf_whisper/model.py"]
    )
    assert "whisper_tiny" in models


def test_utility_with_manual_edges_detects_representative_models() -> None:
    """
    Changing a utility file listed in MANUAL_EDGES should detect
    the representative models (DFS is short-circuited).
    """
    models = resolve_affected_models(["src/qai_hub_models/utils/input_spec.py"])
    assert "sinet" in models
    assert "whisper_tiny" in models


def test_base_model_change_detects_sd() -> None:
    """
    Changing base_model.py should detect stable_diffusion_v1_5
    via its MANUAL_EDGES entry.
    """
    models = resolve_affected_models(["src/qai_hub_models/utils/base_model.py"])
    assert "stable_diffusion_v1_5" in models
    assert "sinet" in models
    assert "whisper_tiny" in models


def test_unrelated_file_detects_nothing() -> None:
    """Changing a file outside qai_hub_models should detect no models."""
    models = resolve_affected_models(["scripts/tasks/changes.py"])
    assert len(models) == 0


def test_template_change_does_not_detect_template_as_model() -> None:
    """
    templates/ directories are not models — they should never appear
    in the output even though they're under models/.
    """
    models = resolve_affected_models(
        ["src/qai_hub_models/models/templates/diffusion/model.py"]
    )
    for m in models:
        assert not m.startswith("templates")


def test_model_inherited_by_others_detects_dependents() -> None:
    """
    SAM's model.py is imported by mobilesam, edgetam, etc.
    Changing it should detect those downstream models.
    """
    models = resolve_affected_models(["src/qai_hub_models/models/sam/model.py"])
    assert "sam" in models
    assert "mobilesam" in models


def test_utils_not_in_manual_edges_fans_out() -> None:
    """
    Utils files NOT in MANUAL_EDGES should follow real imports
    and find all affected models (potentially many).
    """
    models = resolve_affected_models(["src/qai_hub_models/utils/dataset_util.py"])
    # dataset_util.py fans through SD quantize + LLM quantize.
    assert "stable_diffusion_v1_5" in models
    assert "llama_v3_2_1b_instruct" in models


def test_aimet_onnx_utils_change_detects_sd() -> None:
    """
    Changing quantization_aimet_onnx.py should detect stable_diffusion_v1_5
    via its MANUAL_EDGES entry for AIMET coverage.
    """
    models = resolve_affected_models(
        ["src/qai_hub_models/utils/quantization_aimet_onnx.py"]
    )
    assert "stable_diffusion_v1_5" in models


# ── get_ci_test_models: manifest.yaml detection ─────────────────────

get_ci_test_models = _changes.get_ci_test_models
get_changed_files_in_package = _changes.get_changed_files_in_package


def _run_get_ci_test_models_with_changed_files(changed_files_content: str) -> set[str]:
    """
    Helper that runs get_ci_test_models with a fake changed-files file.

    Simulates the CI environment where build/changed-qaihm-files.txt
    is pre-populated with the list of changed files.
    """
    # Clear the lru_caches so previous test state doesn't leak
    get_changed_files_in_package.cache_clear()
    _changes.get_affected_files.cache_clear()

    with tempfile.TemporaryDirectory() as tmpdir:
        build_dir = os.path.join(tmpdir, "build")
        os.makedirs(build_dir)
        changed_file = os.path.join(build_dir, "changed-qaihm-files.txt")
        with open(changed_file, "w") as f:
            f.write(changed_files_content)

        with (
            patch.object(_changes, "on_github", return_value=True),
            patch.object(_changes, "REPO_ROOT", tmpdir),
        ):
            result = get_ci_test_models()

    # Clean up caches after test
    get_changed_files_in_package.cache_clear()
    _changes.get_affected_files.cache_clear()
    return result


def test_manifest_yaml_change_detects_model() -> None:
    """
    Changing only a model's manifest.yaml should detect that model.

    This was the bug in issue #19031: config yaml changes were never
    fed into get_ci_test_models because get_changed_files_in_package
    was only called with suffix='.py' and suffix='requirements.txt'.
    """
    changed_files = "src/qai_hub_models/models/cvt/manifest.yaml\n"
    models = _run_get_ci_test_models_with_changed_files(changed_files)
    assert "cvt" in models


# ── LLM grouping tests ──────────────────────────────────────────────


def test_llm_group_collapses_multiple_llamas() -> None:
    """
    Given a set with multiple llama variants, prune_llm_groups keeps
    only llama_v3_2_1b_instruct (the first in LLM_GROUPS[0]).
    """
    models = {
        "llama_v3_1_8b_instruct",
        "llama_v3_2_1b_instruct",
        "llama_v3_2_3b_instruct",
        "mobilenet_v2",
    }
    pruned = prune_llm_groups(models)
    assert "llama_v3_2_1b_instruct" in pruned
    assert "llama_v3_1_8b_instruct" not in pruned
    assert "llama_v3_2_3b_instruct" not in pruned
    assert "mobilenet_v2" in pruned


def test_llm_group_collapses_multiple_qwens() -> None:
    """qwen3_0_6b is first (smallest) in LLM_GROUPS[1], so it survives."""
    models = {
        "qwen3_0_6b",
        "qwen3_4b",
        "qwen3_8b",
        "qwen2_7b_instruct",
        "mobilenet_v2",
    }
    pruned = prune_llm_groups(models)
    assert "qwen3_0_6b" in pruned
    assert "qwen3_4b" not in pruned
    assert "qwen3_8b" not in pruned
    assert "qwen2_7b_instruct" not in pruned
    assert "mobilenet_v2" in pruned


def test_llm_groups_are_independent() -> None:
    """A set containing both a llama and a qwen keeps one from each group."""
    models = {
        "llama_v3_8b_instruct",
        "qwen3_8b",
        "mobilenet_v2",
    }
    pruned = prune_llm_groups(models)
    # llama_v3_8b_instruct is the only llama present, so it's kept
    assert "llama_v3_8b_instruct" in pruned
    # qwen3_8b is the only qwen present, so it's kept
    assert "qwen3_8b" in pruned
    assert "mobilenet_v2" in pruned


def test_prune_llm_groups_preserves_non_llm_models() -> None:
    """Non-LLM models are untouched."""
    models = {
        "mobilenet_v2",
        "sinet",
        "whisper_tiny",
        "stable_diffusion_v1_5",
    }
    pruned = prune_llm_groups(models)
    assert pruned == models


def test_prune_llm_groups_no_op_when_single_present() -> None:
    """If only one llama is present, it's kept as-is."""
    models = {"llama_v3_2_1b_instruct", "mobilenet_v2"}
    pruned = prune_llm_groups(models)
    assert pruned == models


def test_llm_group_mistral_and_falcon_in_llama_group() -> None:
    """Mistral and falcon are grouped with llama-family."""
    models = {
        "llama_v3_2_1b_instruct",
        "mistral_7b_instruct_v0_3",
        "falcon_v3_7b_instruct",
        "mobilenet_v2",
    }
    pruned = prune_llm_groups(models)
    # llama_v3_2_1b_instruct comes first in LLM_GROUPS[0]
    assert "llama_v3_2_1b_instruct" in pruned
    assert "mistral_7b_instruct_v0_3" not in pruned
    assert "falcon_v3_7b_instruct" not in pruned
    assert "mobilenet_v2" in pruned


def test_gemma4_e4b_alone_is_not_pruned() -> None:
    """
    E4B is the only pytorch gemma4 recipe (E2B is llama.cpp-only), so it is
    always the gemma4 representative and is never pruned away.
    """
    pruned = prune_llm_groups({"gemma_4_e4b_it", "mobilenet_v2"})
    assert pruned == {"gemma_4_e4b_it", "mobilenet_v2"}


# ── gemma4 template-file routing ────────────────────────────────────


def test_template_gemma4_change_detects_gemma4_representative() -> None:
    """Changing templates/gemma4/ should test the gemma4 representative."""
    models = resolve_affected_models(
        ["src/qai_hub_models/models/templates/gemma4/model.py"]
    )
    assert "gemma_4_e4b_it" in models


def test_template_gemma4_quantize_change_detects_gemma4() -> None:
    """
    quantize.py's only consumers are per-model quantize.py files, which
    resolve_affected_models filters out (not in its filename allowlist).
    Without a MANUAL_EDGES entry this would resolve to no models at all,
    so a change to the template quantize workflow would go untested.
    """
    models = resolve_affected_models(
        ["src/qai_hub_models/models/templates/gemma4/quantize.py"]
    )
    assert "gemma_4_e4b_it" in models


def test_lm_driver_change_detects_gemma4_too() -> None:
    """
    templates/lm_driver/ is imported by templates/gemma4/model.py as well as by
    llama and qwen-VL, so all three representatives must be tested.
    """
    models = resolve_affected_models(
        ["src/qai_hub_models/models/templates/lm_driver/generator.py"]
    )
    assert "gemma_4_e4b_it" in models
    assert "llama_v3_2_1b_instruct" in models
    assert "qwen3_vl_4b_instruct" in models

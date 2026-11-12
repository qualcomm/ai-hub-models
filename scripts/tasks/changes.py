# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import difflib
import functools
import os
import re
from collections.abc import Iterable
from pathlib import Path

from .constants import (
    PUBLIC_BENCH_MODELS,
    PY_PACKAGE_INSTALL_ROOT,
    PY_PACKAGE_MODELS_ROOT,
    PY_PACKAGE_RELATIVE_MODELS_ROOT,
    PY_PACKAGE_RELATIVE_SRC_ROOT,
    REPO_ROOT,
    SCORECARD_PACKAGE_MODELS_RELATIVE_ROOT,
    STATIC_MODELS_ROOT,
)
from .github import on_github
from .plan import Task
from .util import check_manifest_field, new_cd, run, run_and_get_output

SINET_MODEL = "sinet"
WHISPER_TINY_MODEL = "whisper_tiny"
REPRESENTATIVE_MODELS = [SINET_MODEL, WHISPER_TINY_MODEL]
CODEGEN_FALLBACK_LLM_MODEL = "llama_v3_2_1b_instruct"
SINET_MODEL_FILE = f"src/qai_hub_models/models/{SINET_MODEL}/model.py"
WHISPER_TINY_MODEL_FILE = f"src/qai_hub_models/models/{WHISPER_TINY_MODEL}/model.py"
REPRESENTATIVE_MODEL_FILES = [SINET_MODEL_FILE, WHISPER_TINY_MODEL_FILE]

# stable_diffusion_v1_5 is an AIMET collection model (PretrainedCollectionModel
# with quantized components). Including it in representative sets for base_model.py
# and quantization_aimet_onnx.py ensures PR CI covers the AIMET/collection model
# code path, which differs significantly from standard single-component models.
REPRESENTATIVE_AIMET_MODEL_FILE = (
    "src/qai_hub_models/models/stable_diffusion_v1_5/model.py"
)

# For LLM families, testing a single representative is enough coverage
# for cross-family refactors. Within each sublist, if multiple entries end up
# in the resolved set of models to test, keep only the FIRST one (highest
# priority — put the cheapest/smallest model first).
LLM_GROUPS: list[list[str]] = [
    [
        "llama_v3_2_1b_instruct",  # smallest, primary representative
        "llama_v3_2_3b_instruct",
        "llama_v3_2_3b_instruct_ssd",
        "llama_v3_1_8b_instruct",
        "llama_v3_8b_instruct",
        "llama_v3_1_sea_lion_3_5_8b_r",
        "llama_v3_elyza_jp_8b",
        "llama_v3_taide_8b_chat",
        "mistral_7b_instruct_v0_3",
        "falcon_v3_7b_instruct",
    ],
    [
        "qwen3_0_6b",  # smallest, primary representative
        "qwen3_1_7b",
        "qwen3_4b",
        "qwen3_4b_instruct_2507",
        "qwen3_8b",
        "qwen2_7b_instruct",
        "qwen2_5_vl_7b_instruct",
        "qwen3_vl_4b_instruct",
    ],
    [
        "gemma_4_e4b_it",  # only gemma4 pytorch recipe (E2B is llama.cpp-only)
    ],
]

# Template dirs route to their representative model file:
# llm/llama3→llama, qwen3/qwen2→qwen, qwen2_vl/qwen3_vl/vlm→qwen_vl,
# gemma4→gemma4, lm_driver→all three reps.
LLAMA_REPRESENTATIVE_MODEL_FILE = (
    "src/qai_hub_models/models/llama_v3_2_1b_instruct/model.py"
)
QWEN_REPRESENTATIVE_MODEL_FILE = "src/qai_hub_models/models/qwen3_0_6b/model.py"
QWEN_VL_REPRESENTATIVE_MODEL_FILE = (
    "src/qai_hub_models/models/qwen3_vl_4b_instruct/model.py"
)
GEMMA4_REPRESENTATIVE_MODEL_FILE = "src/qai_hub_models/models/gemma_4_e4b_it/model.py"
PI05_REPRESENTATIVE_MODEL_FILE = "src/qai_hub_models/models/pi05/model.py"
PRECOMPILED_REPRESENTATIVE_MODEL_FILE = (
    "src/qai_hub_models/models/qwen2_7b_instruct/model.py"
)
_TEMPLATES_DIR = Path(REPO_ROOT, "src/qai_hub_models/models/templates")
_LLAMA_REP_FILES = sorted(
    p.relative_to(REPO_ROOT).as_posix()
    for sub in ("llm", "llama3")
    for p in (_TEMPLATES_DIR / sub).rglob("*.py")
)
_QWEN_REP_FILES = sorted(
    p.relative_to(REPO_ROOT).as_posix()
    for sub in ("qwen3", "qwen2")
    for p in (_TEMPLATES_DIR / sub).rglob("*.py")
)
_QWEN_VL_REP_FILES = sorted(
    p.relative_to(REPO_ROOT).as_posix()
    for sub in ("qwen2_vl", "qwen3_vl", "vlm")
    for p in (_TEMPLATES_DIR / sub).rglob("*.py")
)
_GEMMA4_REP_FILES = sorted(
    p.relative_to(REPO_ROOT).as_posix()
    for p in (_TEMPLATES_DIR / "gemma4").rglob("*.py")
)
_LM_DRIVER_REP_FILES = sorted(
    p.relative_to(REPO_ROOT).as_posix()
    for p in (_TEMPLATES_DIR / "lm_driver").rglob("*.py")
)

# Utils used only by LLM text + LLM VL template code.
_LLM_TEXT_AND_VL_UTILS = [
    "src/qai_hub_models/utils/system_info.py",
]
# aimet/encodings.py is used by text LLMs + VL LLMs + pi05.
_LLM_AND_PI05_UTILS = [
    "src/qai_hub_models/utils/aimet/encodings.py",
]


# For certain files that are imported by many models, manually override
# which files to test. For example, quantization_aimet is imported by all
# aimet models. Testing a representative set of aimet models is probably
# good enough rather than testing all of them.
MANUAL_EDGES = {
    "src/qai_hub_models/datasets/__init__.py": [
        "src/qai_hub_models/models/yolov7_quantized/model.py"
    ],
    "src/qai_hub_models/utils/base_model.py": [
        *REPRESENTATIVE_MODEL_FILES,
        REPRESENTATIVE_AIMET_MODEL_FILE,
    ],
    # Hosts the shared AIMET quantize path (quantize_from_steps) used by LLMs and
    # VLMs in addition to the SD collection model, so a change here must exercise
    # a representative beyond SD. The text-LLM path is covered by the qwen-VL
    # backbone (a text transformer), so llama/qwen text reps are omitted to keep
    # this fan-out cheap.
    "src/qai_hub_models/utils/quantization_aimet_onnx.py": [
        REPRESENTATIVE_AIMET_MODEL_FILE,
        LLAMA_REPRESENTATIVE_MODEL_FILE,
        QWEN_VL_REPRESENTATIVE_MODEL_FILE,
        GEMMA4_REPRESENTATIVE_MODEL_FILE,
        PI05_REPRESENTATIVE_MODEL_FILE,
    ],
    "src/qai_hub_models/utils/export/pipeline.py": [SINET_MODEL_FILE],
    "src/qai_hub_models/utils/export/collection_pipeline.py": [WHISPER_TINY_MODEL_FILE],
    "src/qai_hub_models/utils/export/multi_graph_pipeline.py": [
        LLAMA_REPRESENTATIVE_MODEL_FILE
    ],
    "src/qai_hub_models/utils/export/multi_graph_collection_pipeline.py": [
        LLAMA_REPRESENTATIVE_MODEL_FILE
    ],
    "src/qai_hub_models/utils/export/precompiled_pipeline.py": [
        PRECOMPILED_REPRESENTATIVE_MODEL_FILE
    ],
    "src/qai_hub_models/utils/export/dispatch.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/device.py": REPRESENTATIVE_MODEL_FILES,
    **{f: [LLAMA_REPRESENTATIVE_MODEL_FILE] for f in _LLAMA_REP_FILES},
    **{f: [QWEN_REPRESENTATIVE_MODEL_FILE] for f in _QWEN_REP_FILES},
    **{f: [QWEN_VL_REPRESENTATIVE_MODEL_FILE] for f in _QWEN_VL_REP_FILES},
    **{f: [GEMMA4_REPRESENTATIVE_MODEL_FILE] for f in _GEMMA4_REP_FILES},
    **{
        f: [
            LLAMA_REPRESENTATIVE_MODEL_FILE,
            QWEN_VL_REPRESENTATIVE_MODEL_FILE,
            GEMMA4_REPRESENTATIVE_MODEL_FILE,
        ]
        for f in _LM_DRIVER_REP_FILES
    },
    **{
        f: [LLAMA_REPRESENTATIVE_MODEL_FILE, QWEN_VL_REPRESENTATIVE_MODEL_FILE]
        for f in _LLM_TEXT_AND_VL_UTILS
    },
    **{
        f: [
            LLAMA_REPRESENTATIVE_MODEL_FILE,
            QWEN_VL_REPRESENTATIVE_MODEL_FILE,
            PI05_REPRESENTATIVE_MODEL_FILE,
        ]
        for f in _LLM_AND_PI05_UTILS
    },
    "src/qai_hub_models/common.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/configs/_info_yaml_enums.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/configs/_info_yaml_llm_details.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/configs/lm_recipe_yaml.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/configs/manifest_yaml.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/scorecard/devices_and_chipsets_yaml.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/configs/model_disable_reasons.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/configs/model_metadata.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/scorecard/numerics_yaml.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/scorecard/perf_yaml.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/configs/proto_helpers.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/scorecard/release_assets_yaml.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/configs/tensor_spec.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/configs/tool_versions.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/protocols.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/scorecard/device.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/scorecard/envvars.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/scorecard/execution_helpers.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/scorecard/utils/testing.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/scorecard/utils/testing_export_eval.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/asset_loaders.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/aws.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/args.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/base_config.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/base_dataset.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/base_evaluator.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/collection_model_helpers.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/envvars.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/evaluate.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/inference.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/input_spec.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/models/templates/corevision/labels.py": [
        "src/qai_hub_models/models/yamnet/model.py",
        "src/qai_hub_models/models/yolov8_det/model.py",
    ],
    "src/qai_hub_models/utils/onnx/torch_wrapper.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/path_helpers.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/printing.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/qai_hub_helpers.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/quantization.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/runtime_torch_wrapper.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/tflite/torch_wrapper.py": REPRESENTATIVE_MODEL_FILES,
    "src/qai_hub_models/utils/transpose_channel.py": REPRESENTATIVE_MODEL_FILES,
}


def prune_llm_groups(models: set[str]) -> set[str]:
    """
    For each sublist in LLM_GROUPS, keep at most the first model that is present.
    Non-LLM models are untouched.
    """
    out = set(models)
    for group in LLM_GROUPS:
        present = [m for m in group if m in out]
        for extra in present[1:]:
            out.discard(extra)
    return out


def get_python_import_expression(filepath: str) -> str:
    """
    Given a filepath, return the expression used to import the file
    in other modules.

    For example, src/qai_hub_models/models/trocr/model.py ->
        qai_hub_models.models.trocr.model
    """
    rel_path = os.path.relpath(filepath, PY_PACKAGE_INSTALL_ROOT)
    init_suffix = "/__init__.py"
    if rel_path.endswith(init_suffix):
        rel_path = rel_path[: -len(init_suffix)]
    else:
        rel_path = rel_path[: -len(".py")]
    return rel_path.replace("/", ".")


def _get_file_edges(filename: str) -> set[str]:
    """Resolve which files directly import from `filename`."""
    file_import = get_python_import_expression(filename)
    grep_out = run_and_get_output(
        f"grep -r --include='*.py' '{file_import}' {PY_PACKAGE_RELATIVE_SRC_ROOT}",
        check=False,
    )
    if grep_out.strip() == "":
        return set()

    # Determine which files depend on the current file, and thus
    # also may be affected by the current change
    # i.e. resolve the edges of the current node for DFS
    dependent_files = set()
    for grep_result in grep_out.strip().split("\n"):
        dependent_file = grep_result.split(":")[0]
        dependent_files.add(dependent_file)

    return dependent_files


@functools.lru_cache(maxsize=1)
def get_affected_files(changed_files: Iterable[str]) -> set[str]:
    """
    Given a list of changed python files, performs a Depth-First Search (DFS)
    over the qai_hub_models directory to figure out which files were affected.

    Cached so that the graph traversal is done once, and `resolve_affected_models`
    can be run with different args using the same base set of files.
    """
    changed_files = list(changed_files)
    seen = set(changed_files)
    while len(changed_files) > 0:
        # Pop off stack
        curr_file = changed_files.pop()
        if not curr_file.endswith(".py"):
            continue
        if curr_file in MANUAL_EDGES:
            dependent_files = set(MANUAL_EDGES[curr_file])
        else:
            dependent_files = _get_file_edges(curr_file)
        # Add new nodes to stack
        for dependent_file in dependent_files:
            if dependent_file not in seen:
                seen.add(dependent_file)
                changed_files.append(dependent_file)
    return seen


def resolve_affected_models(
    changed_files: Iterable[str],
    include_model: bool = True,
    include_demo: bool = True,
    include_tests: bool = True,
    include_generated_tests: bool = True,
    include_cj_yaml: bool = True,
) -> set[str]:
    """
    Given a list of changed python files, performs a Depth-First Search (DFS)
    over the qai_hub_models directory to figure out which directories were affected.

    The source nodes are the files that were directly changed, and there's
    an edge from file A to file B if file B imports from file A.

    Note: If a zoo module is imported using a relative path, the dependency will not
    be detected. Imports should be done using "from qai_stac_models.<my_module>"
    in order to detect that current file depends on <my_module>.

    changed_files: List of filepaths to files that changed. Paths are
        relative to the root of this repository.
    """
    # Convert to tuple so it can be used as a cache key
    affected_files = get_affected_files(tuple(changed_files))
    changed_models = set()
    for f in affected_files:
        file_path = Path(f)
        # Only consider directories directly in the top-level `models/` folder
        # (i.e. ignore `models/templates`)
        if str(file_path.parent.parent) == PY_PACKAGE_RELATIVE_MODELS_ROOT:
            if file_path.name not in [
                "model.py",
                "test.py",
                "demo.py",
                "requirements.txt",
                "manifest.yaml",
                "lm_recipe.yaml",
            ]:
                continue
            if not include_model and file_path.name == "model.py":
                continue
            if not include_tests and file_path.name == "test.py":
                continue
            if not include_demo and file_path.name == "demo.py":
                continue
            if not include_cj_yaml and file_path.name in (
                "manifest.yaml",
                "lm_recipe.yaml",
            ):
                continue

            model_name = file_path.parent.name
            if (file_path.parent / "model.py").exists() and (
                file_path.parent / "manifest.yaml"
            ).exists():
                changed_models.add(model_name)
        elif (
            str(file_path.parent.parent) == SCORECARD_PACKAGE_MODELS_RELATIVE_ROOT
            and file_path.name == "test_generated.py"
            and include_generated_tests
        ):
            model_name = file_path.parent.name
            source_model_dir = Path(PY_PACKAGE_RELATIVE_MODELS_ROOT) / model_name
            if (source_model_dir / "model.py").exists() and (
                source_model_dir / "info.yaml"
            ).exists():
                changed_models.add(model_name)
    return changed_models


def _expand_diff_to_seed_py_files(diff_path: str) -> list[str]:
    """
    Map a changed patches-diff path (under models/<X>/ or models/templates/<Y>/) to
    a list of .py seed files whose "change" should trigger the same downstream
    tests. For a top-level model, seed model.py so model + export tests run. For
    a templates component, seed every .py in that component so the DFS finds every
    dependent model.
    """
    parts = Path(diff_path).parts
    try:
        models_idx = parts.index("models")
    except ValueError:
        return []
    tail = parts[models_idx + 1 :]
    if not tail:
        return []
    if tail[0] == "templates" and len(tail) >= 2:
        template_dir = Path(PY_PACKAGE_MODELS_ROOT, "templates", tail[1])
        return [p.relative_to(REPO_ROOT).as_posix() for p in template_dir.rglob("*.py")]
    model_py_rel = Path(PY_PACKAGE_RELATIVE_MODELS_ROOT, tail[0], "model.py")
    if Path(REPO_ROOT, model_py_rel).exists():
        return [model_py_rel.as_posix()]
    return []


@functools.lru_cache(maxsize=4)
def get_changed_files_in_package(
    prefix: str | None = None,
    suffix: str | None = None,
) -> Iterable[str]:
    """
    Returns the list of changed files in zoo based on git tracking.

    If the suffix argument is passed, restrict only to files ending in that suffix.
    """
    with new_cd(REPO_ROOT):
        changed_files_path = "build/changed-qaihm-files.txt"
        if not on_github():
            run(f"git diff origin/main --name-only > {changed_files_path}")
        if os.path.exists(changed_files_path):
            with open(changed_files_path) as f:
                changed_files = [
                    file
                    for file in f.read().split("\n")
                    if file.startswith(PY_PACKAGE_RELATIVE_SRC_ROOT)
                    and (prefix is None or file.startswith(prefix))
                    and (suffix is None or file.endswith(suffix))
                ]
                # Weed out duplicates
                return list(set(changed_files))
        return []


def get_ci_test_models(
    include_model: bool = True,
    include_demo: bool = True,
    include_tests: bool = True,
    include_generated_tests: bool = True,
    include_cj_yaml: bool = True,
) -> set[str]:
    """
    Resolve which models within zoo have changed to figure which ones need to be tested.

    First figures out which files have changed and then does a recursive search
    through all files that import from changed files. Then filters the final list
    to model directories to know which ones that need to be tested.

    Returns a list of model IDs (folder names) that have changed.
    """
    files = list(get_changed_files_in_package(suffix="requirements.txt"))
    files.extend(get_changed_files_in_package(suffix=".py"))
    files.extend(get_changed_files_in_package(suffix="manifest.yaml"))
    for diff in get_changed_files_in_package(suffix=".diff"):
        files.extend(_expand_diff_to_seed_py_files(diff))
    return resolve_affected_models(
        files,
        include_model,
        include_demo,
        include_tests,
        include_generated_tests,
        include_cj_yaml,
    )


def get_all_models() -> list[str]:
    """Resolve model IDs (folder names) of all models in QAIHM."""
    model_names: set[str] = set()
    for model_name in os.listdir(PY_PACKAGE_MODELS_ROOT):
        if os.path.exists(
            os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "manifest.yaml")
        ):
            model_names.add(model_name)

    bench_dir = os.getenv("QAIHM_BENCH_TEST_DIR", STATIC_MODELS_ROOT)
    static_models = {x[:-5] for x in os.listdir(bench_dir) if x.endswith(".yaml")}

    # Select a subset of models based on user input
    allowed_models_str = os.environ.get("QAIHM_TEST_MODELS", "all").lower()
    user_specified_models_list: list[str] | None = None
    if allowed_models_str and allowed_models_str not in ["all", "pytorch"]:
        if allowed_models_str == "bench":
            with open(PUBLIC_BENCH_MODELS) as f:
                model_names = set(f.read().strip().split("\n"))
        else:
            user_specified_models_list = [
                model.strip() for model in allowed_models_str.split(",")
            ]
            allowed_models = set(user_specified_models_list) - static_models
            for model in allowed_models:
                if model not in model_names:
                    raise ValueError(f"Unknown model selected: {model}")
            model_names = allowed_models

    if user_specified_models_list:
        return [
            x for x in user_specified_models_list if x in model_names
        ]  # preserve order specified by user

    return list(model_names)


class ValidateQuantizeInputsTask(Task):
    """
    Check the quantize_and_upload inputs (the QAIHM_QUANTIZE_MODEL_ID,
    QAIHM_QUANTIZE_PRECISION and QAIHM_QUANTIZE_ARGS env vars).

    Runs before a GPU pod is claimed. Both manifest fields are flat top-level
    keys, so they are matched as text and no venv is needed.
    """

    def __init__(self) -> None:
        super().__init__("Validate quantize inputs")

    def does_work(self) -> bool:
        return True

    def run_task(self) -> bool:
        model_id = os.environ.get("QAIHM_QUANTIZE_MODEL_ID", "").strip()
        precision = os.environ.get("QAIHM_QUANTIZE_PRECISION", "").strip()
        quantize_args = os.environ.get("QAIHM_QUANTIZE_ARGS", "")

        precision_parser = argparse.ArgumentParser(add_help=False)
        precision_parser.add_argument("--precision", nargs="?", const="")
        parsed, _ = precision_parser.parse_known_args(quantize_args.split())
        if parsed.precision is not None:
            raise ValueError(
                "quantize_args must not contain --precision; set the precision "
                "input instead."
            )

        known = get_all_models()
        if model_id not in known:
            hint = difflib.get_close_matches(model_id, known, n=3, cutoff=0.4)
            print(
                f"::error::model_id {model_id!r} is not a model with a manifest.yaml."
                + (
                    f" Did you mean {' or '.join(repr(h) for h in hint)}?"
                    if hint
                    else ""
                )
            )
            return False

        model_dir = Path(PY_PACKAGE_MODELS_ROOT, model_id)
        ok = True

        if not check_manifest_field(model_id, "separate_quantize_script"):
            print(
                f"::error::{model_id} does not set 'separate_quantize_script: true' "
                "in its manifest.yaml, so it has no quantize.py to run."
            )
            ok = False

        # quantize only runs a precision that has a recipe, which also implies it is
        # in supported_precisions; failing here saves claiming a GPU pod.
        recipe_path = model_dir / "lm_recipe.yaml"
        recipe = recipe_path.read_text(encoding="utf-8") if recipe_path.exists() else ""
        if not re.search(rf"^{re.escape(precision)}:\s*$", recipe, re.MULTILINE):
            print(
                f"::error::{model_id} has no lm_recipe.yaml recipe for "
                f"precision {precision!r}, so quantize has nothing to run."
            )
            ok = False

        return ok


class PrintCITestModelsTask(Task):
    def __init__(self, group_name: str | None = None) -> None:
        super().__init__(group_name)

    def does_work(self) -> bool:
        return False

    def run_task(self) -> bool:
        # model / demo / test changed, or manifest.yaml changed
        model_or_yaml_changed = get_ci_test_models(include_generated_tests=False)

        # test_generated.py changed, but the rest of the model, including manifest.yaml was not affected
        # models will hit this when global templates change
        only_code_generation_changed = get_ci_test_models(
            include_model=False,
            include_demo=False,
            include_tests=False,
            include_cj_yaml=False,
        )

        out = model_or_yaml_changed
        codegen_only_models = only_code_generation_changed - model_or_yaml_changed
        if codegen_only_models:
            # We don't run tests for every model whose codegen files was changed.
            # However, if there are models with only codegen changes,
            # we run a representative model set to make sure codegen didn't break for regular + component models.
            out = out.union(REPRESENTATIVE_MODELS)
            llm_models = {m for group in LLM_GROUPS for m in group}
            if codegen_only_models & llm_models:
                out.add(CODEGEN_FALLBACK_LLM_MODEL)

        out = prune_llm_groups(out)
        print(",".join(sorted(out)))
        return True

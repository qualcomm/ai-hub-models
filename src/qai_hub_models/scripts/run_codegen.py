# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.scripts.generate_global_readme import generate_global_readme
from qai_hub_models.scripts.generate_model_readme import generate_and_write_model_readme
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    QAIHM_MODELS_ROOT,
    QAIHM_PACKAGE_ROOT,
    QAIHM_PACKAGE_SRC_ROOT,
    QAIHM_REPO_ROOT,
)

HEADER = "# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY."


def _is_auto_generated(path: Path) -> bool:
    """Return True if the file was created by codegen."""
    with open(path) as f:
        head = f.read(2048)
    return HEADER in head


def _generate_unit_tests(
    environment: Environment,
    model_name: str,
    manifest: dict[str, Any],
    test_path: Path,
) -> str:
    template = environment.get_template("unit_test_template.j2")
    file_contents = template.render(
        manifest,
        model_name=model_name,
        header=HEADER,
    )

    with open(test_path, "w") as f:
        f.write(file_contents)
    return str(test_path)


def _generate_conftest(
    environment: Environment,
    model_name: str,
    manifest: dict[str, Any],
    conftest_path: Path,
) -> str:
    template = environment.get_template("conftest_template.j2")
    file_contents = template.render(
        manifest,
        model_name=model_name,
        header=HEADER,
    )
    with open(conftest_path, "w") as f:
        f.write(file_contents)
    return str(conftest_path)


def _generate_external_repos_init(
    environment: Environment,
    model_name: str,
    model_dir: Path,
) -> str:
    external_repos_dir = model_dir / "external_repos"
    os.makedirs(external_repos_dir, exist_ok=True)

    template = environment.get_template("external_repos_init_template.j2")
    file_contents = template.render(
        model_id=model_name,
        header=HEADER,
    )
    file_path = os.path.join(external_repos_dir, "__init__.py")
    with open(file_path, "w") as f:
        f.write(file_contents)
    return file_path


def _generate_external_repos_for_subfolders(
    environment: Environment, root: Path
) -> list[str]:
    """Generate __init__.py for each <root>/*/manifest.yaml that declares external_repos."""
    if not root.exists():
        return []

    template = environment.get_template("external_repos_init_template.j2")
    generated = []
    for folder in sorted(root.iterdir()):
        manifest_path = folder / "manifest.yaml"
        if not manifest_path.exists():
            continue
        manifest = QAIHMModelManifest.from_yaml(manifest_path)
        if not manifest.external_repos:
            continue
        external_repos_dir = folder / "external_repos"
        os.makedirs(external_repos_dir, exist_ok=True)
        file_contents = template.render(
            template_name=folder.name,
            header=HEADER,
        )
        file_path = external_repos_dir / "__init__.py"
        with open(file_path, "w") as f:
            f.write(file_contents)
        generated.append(str(file_path))
    return generated


def _generate_template_external_repos(environment: Environment) -> list[str]:
    return _generate_external_repos_for_subfolders(
        environment, QAIHM_MODELS_ROOT / "templates"
    )


def _generate_example_external_repos(environment: Environment) -> list[str]:
    return _generate_external_repos_for_subfolders(
        environment, QAIHM_REPO_ROOT / "scripts" / "examples"
    )


def generate_code_for_model(model_name: str) -> list[str]:
    model_dir = QAIHM_MODELS_ROOT / model_name
    scorecard_config = QAIHMModelInternalConfig.from_model(model_name)
    manifest = scorecard_config.manifest

    if scorecard_config.skip_export:
        print(f"Skipping code generation for {model_name}.")
        return []

    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates/"),
        keep_trailing_newline=True,
    )
    generated_files: list[str] = []
    # Generate or clean up external repo files
    external_repos_dir = model_dir / "external_repos"
    if manifest.external_repos:
        generated_files.append(
            _generate_external_repos_init(environment, model_name, model_dir)
        )
    elif external_repos_dir.exists():
        shutil.rmtree(external_repos_dir)

    manifest_dict = manifest.model_dump()
    manifest_dict["can_use_quantize_job"] = manifest.can_use_quantize_job
    manifest_dict["is_llm"] = scorecard_config.is_llm
    manifest_dict["standalone_components"] = scorecard_config.standalone_components

    should_generate_tests = not scorecard_config.skip_hub_tests_and_scorecard
    scorecard_model_dir = QAIHM_PACKAGE_ROOT / "scorecard" / "models" / model_name
    test_path = scorecard_model_dir / "test_generated.py"
    if should_generate_tests:
        scorecard_model_dir.mkdir(parents=True, exist_ok=True)
        init_path = scorecard_model_dir / "__init__.py"
        if not init_path.exists():
            init_path.touch()
        generated_files.append(
            _generate_unit_tests(environment, model_name, manifest_dict, test_path)
        )
    elif test_path.exists() and _is_auto_generated(test_path):
        os.remove(test_path)

    # Transition-only: remove the auto-generated test file from its old home
    # under models/<id>/. Kept for one PR of grace so in-flight branches
    # don't leave orphans. Delete this block once no branches reference
    # the old location (tetracode#20296 PR 3 cleanup).
    old_test_path = model_dir / "test_generated.py"
    if old_test_path.exists() and _is_auto_generated(old_test_path):
        os.remove(old_test_path)

    should_generate_conftest = should_generate_tests and not manifest.is_precompiled
    # tetracode#20296: emit conftest.py at BOTH the hand-written test.py's
    # location (models/<id>/) and the moved test_generated.py's location
    # (scorecard/models/<id>/) so cached_from_pretrained applies to tests
    # in both dirs (pytest conftest discovery doesn't walk sideways). The
    # two files are identical today; the scorecard-side conftest may
    # diverge later as scorecard-specific fixtures are added.
    conftest_paths = [
        model_dir / "conftest.py",
        scorecard_model_dir / "conftest.py",
    ]
    for conftest_path in conftest_paths:
        if should_generate_conftest:
            generated_files.append(
                _generate_conftest(
                    environment, model_name, manifest_dict, conftest_path
                )
            )
        elif conftest_path.exists() and _is_auto_generated(conftest_path):
            os.remove(conftest_path)

    zoo_root = QAIHM_MODELS_ROOT
    return [os.path.join(zoo_root, gen_file) for gen_file in generated_files]


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models",
        "-m",
        nargs="+",
        type=str,
        help="Models for which to generate files.",
    )
    group.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="If set, generates files for all models.",
    )
    parser.add_argument(
        "--no-precommit",
        action="store_true",
        help="If set, skips running pre-commit on generated files.",
    )
    args = parser.parse_args()
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates/"),
        keep_trailing_newline=True,
    )

    models = args.models if args.models else MODEL_IDS
    modified_files = []

    # Generate template and scripts/examples external repos __init__.py files
    if args.all:
        modified_files.extend(_generate_template_external_repos(environment))
        modified_files.extend(_generate_example_external_repos(environment))

    for model in models:
        try:
            modified_files.extend(generate_code_for_model(model))
            modified_files.append(str(generate_and_write_model_readme(model)))
        except Exception as e:  # noqa: PERF203
            raise ValueError(f"Failed to generate export files for {model}") from e

    # Generate global README
    published = [
        sc.manifest
        for sc in (QAIHMModelInternalConfig.from_model(mid) for mid in MODEL_IDS)
        if sc.status is MODEL_STATUS.PUBLISHED
    ]
    modified_files.extend(
        str(p)
        for p in generate_global_readme(
            published, QAIHM_REPO_ROOT, QAIHM_PACKAGE_SRC_ROOT
        )
    )

    if not args.no_precommit:
        os.environ["SKIP"] = "mypy-src,mypy-cli"
        subprocess.run(["pre-commit", "run", "--files", *modified_files], check=False)


if __name__ == "__main__":
    main()

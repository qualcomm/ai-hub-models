# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from tempfile import TemporaryDirectory

from .changes import get_all_models, get_changed_files_in_package
from .constants import (
    BUILD_ROOT,
    PY_PACKAGE_MODELS_ROOT,
    PY_PACKAGE_SRC_ROOT,
    REPO_ROOT,
    STORE_ROOT_ENV_VAR,
)
from .task import (
    CompositeTask,
    PyTestTask,
    RunCommandsTask,
)
from .util import (
    can_support_aimet,
    check_code_gen_field,
    get_is_hub_quantized,
    get_model_python_version_requirements,
    is_quantized_llm_model,
    model_needs_aimet,
    on_ci,
)
from .venv import (
    CreateVenvTask,
    InstallGlobalRequirementsTask,
    RunCommandsWithVenvTask,
    SyncLocalQAIHMVenvTask,
    SyncModelRequirementsVenvTask,
    SyncModelVenvTask,
)


class PyTestQAIHMTask(PyTestTask):
    """Pytest utils."""

    def __init__(self, venv: str | None) -> None:
        all_dirs_except_models = [
            f"{PY_PACKAGE_SRC_ROOT}/{x}"
            for x in os.listdir(PY_PACKAGE_SRC_ROOT)
            if x not in {"models", "__pycache__", "scorecard"}
        ]

        # Internal scorecard tests are expensive (calls to Hub), so only run them if the internal scorecard changes.
        scorecard_files = [
            os.path.join(PY_PACKAGE_SRC_ROOT, "scorecard", x)
            for x in os.listdir(os.path.join(PY_PACKAGE_SRC_ROOT, "scorecard"))
        ]

        if not get_changed_files_in_package("qai_hub_models/scorecard/internal"):
            scorecard_files.remove(f"{PY_PACKAGE_SRC_ROOT}/scorecard/internal")
        all_dirs_except_models.extend(scorecard_files)

        all_dirs_except_models = [x for x in all_dirs_except_models if os.path.isdir(x)]

        # Get JUnit XML path from environment variable
        junit_xml_path = os.environ.get("QAIHM_JUNIT_XML_PATH")

        super().__init__(
            "Test QAIHM",
            venv=venv,
            files_or_dirs=" ".join(all_dirs_except_models),
            parallel=True,
            junit_xml_path=junit_xml_path,
        )


class GPUPyTestModelsTask(CompositeTask):
    """Run all tests for LLM models."""

    def __init__(
        self,
        venv: str | None,
        model_names: str = "all",  # Comma-separated list of model names, or "all" for all models.
        run_evaluate: bool = True,
        run_compile: bool = True,
        run_qdc: bool = True,
        run_demo: bool = True,
        raise_on_failure: bool = True,
        nightly_only: bool = False,  # If True, only run tests marked with @pytest.mark.nightly
    ) -> None:
        home_dir = "/local/mnt2/workspace2/qaihm_bot"
        junit_xml_path = os.environ.get("QAIHM_JUNIT_XML_PATH")
        tmp_dir = os.path.join(home_dir, "tmp")

        models_to_test = []
        for model_name in get_all_models():
            if os.path.exists(
                os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "test.py")
            ) and (is_quantized_llm_model(model_name)):
                models_to_test.append(model_name)
                # Only run llama_v2_7b_chat on weekly runs so nightly takes less time.
                if not nightly_only and model_name == "llama_v2_7b_chat":
                    models_to_test.append(model_name)
        # Run llama_v2_7b_chat last to avoid resource contention or conflicts with other models.
        if "llama_v2_7b_chat" in models_to_test:
            models_to_test.sort(key=lambda x: x == "llama_v2_7b_chat")
        tasks = []
        common_command = f"export HOME={home_dir} && mkdir -p {tmp_dir} && export TMPDIR={tmp_dir} && rm -rf {home_dir}/.cache/huggingface/hub/models--* {home_dir}/.qaihm/models/* {tmp_dir}/*"
        for model_name in models_to_test:
            base_dir = os.path.dirname(junit_xml_path)
            filename_parts = os.path.splitext(os.path.basename(junit_xml_path))
            test_suites = []
            if run_evaluate:
                test_suites.append("evaluate")
            if run_compile:
                test_suites.append("compile_ram_intensive")
            if run_qdc:
                test_suites.append("qdc")
            if run_demo:
                test_suites.append("demo")
            # Special case: llama_v2_7b_chat only has compile_ram_intensive tests.
            if model_name == "llama_v2_7b_chat":
                test_suites = ["compile_ram_intensive"]

            for test_suite in test_suites:
                model_filename = (
                    f"{filename_parts[0]}-{test_suite}-{model_name}{filename_parts[1]}"
                )
                model_junit_xml_path = os.path.join(base_dir, model_filename)
                marker_expr = (
                    f"{test_suite} and nightly" if nightly_only else test_suite
                )
                options = f"-m '{marker_expr}' --junit-xml={model_junit_xml_path}"
                if model_name == "llama_v2_7b_chat":
                    tasks.append(
                        RunCommandsWithVenvTask(
                            group_name=f"Install Dependencies For Model {model_name}",
                            venv=venv,
                            commands=[
                                f"{common_command} && pip install -r qai_hub_models/models/{model_name}/requirements.txt",
                            ],
                            raise_on_failure=False,
                            # Ignore "no tests collected" return code
                            ignore_return_codes=[5],
                        )
                    )
                tasks.append(
                    RunCommandsWithVenvTask(
                        group_name=f"Run {test_suite} Tests For Model {model_name}",
                        venv=venv,
                        commands=[
                            f"{common_command} && pytest -v -s --capture=no qai_hub_models/models/{model_name}/test.py {options}",
                        ],
                        raise_on_failure=False,
                        # Ignore "no tests collected" return code
                        ignore_return_codes=[5],
                    )
                )

        super().__init__(
            # If a group name is used here, you get two groups per model
            # printed to console when running these tasks, one of which is empty.
            None,
            list(tasks),
            continue_after_single_task_failure=True,
            raise_on_failure=raise_on_failure,
        )


class PyTestModelTask(CompositeTask):
    """Run all tests for a single model."""

    def __init__(
        self,
        model_name: str,
        python_executable: str,
        venv: (
            str | None
        ) = None,  # If None, creates a fresh venv for each model instead of using 1 venv for all models.
        use_shared_cache: bool = False,  # If True, uses a shared cache rather than the global QAIHM cache.
        run_mypy: bool = False,
        run_general: bool = True,
        run_pre_quantize_compile: bool = False,
        run_quantize: bool = False,
        run_compile: bool = True,
        run_profile: bool = False,
        run_inference: bool = False,
        run_compute_device_accuracy: bool = False,
        run_export: bool = False,
        run_trace: bool = True,
        install_deps: bool = True,
        raise_on_failure: bool = False,
        qaihm_wheel_dir: str | os.PathLike | None = None,
        junit_xml_path: str | None = None,
    ) -> None:
        tasks = []

        model_version_reqs = get_model_python_version_requirements(model_name)
        current_py_version = sys.version_info

        if model_version_reqs[0] and current_py_version < model_version_reqs[0]:
            tasks.append(  # greater than this python version
                RunCommandsTask(
                    f"Skip Model {model_name}",
                    f'echo "Skipping Tests For Model {model_name} -- Current Python ({current_py_version}) is too old (must be at least {model_version_reqs[0]})"',
                )
            )
        elif model_version_reqs[1] and current_py_version >= model_version_reqs[1]:
            tasks.append(  # less than this python version
                RunCommandsTask(
                    f"Skip Model {model_name}",
                    f'echo "Skipping Tests For Model {model_name} -- Current Python ({current_py_version}) is too new (must be less than {model_version_reqs[1]})"',
                )
            )
        elif model_needs_aimet(model_name) and not can_support_aimet():
            tasks.append(
                RunCommandsTask(
                    f"Skip Model {model_name}",
                    f'echo "Skipping Tests For Model {model_name} -- AIMET is required, but AIMET is not supported on this platform."',
                )
            )
        else:
            # Create test environment
            needs_model_venv = venv is None
            if needs_model_venv:
                model_venv = os.path.join(BUILD_ROOT, "test", "model_envs", model_name)
                tasks.append(CreateVenvTask(model_venv, python_executable))
                # Creates a new environment from scratch
                tasks.append(
                    SyncModelVenvTask(
                        model_name,
                        model_venv,
                        include_dev_deps=True,
                        qaihm_wheel_dir=qaihm_wheel_dir,
                    )
                )
            else:
                model_venv = venv
                if install_deps:
                    # Only install requirements.txt into existing venv
                    tasks.append(
                        SyncModelRequirementsVenvTask(
                            model_name, model_venv, pip_force_install=False
                        )
                    )

            if check_code_gen_field(model_name, "skip_hub_tests_and_scorecard"):
                tasks.append(  # greater than this python version
                    RunCommandsTask(
                        f"Skip Model {model_name} Hub Tests",
                        f'echo "Skipping Tests For Model {model_name} -- skip_hub_tests_and_scorecard is set in code gen"',
                    )
                )
            elif (
                check_code_gen_field(model_name, "skip_scorecard") and not run_general
            ):  # For scorecard runs, run_general is set to False because it is a test_compile_all_models task rather than a precheckin task with hub tests.
                tasks.append(  # greater than this python version
                    RunCommandsTask(
                        f"Skip Model {model_name} Scorecard",
                        f'echo "Skipping Scorecard For Model {model_name} -- skip_scorecard is set in code gen"',
                    )
                )
            else:
                # Extras arguments
                extras_args = ["-s"]

                # Generate flags
                test_flags = []
                if run_general:
                    test_flags.append("unmarked")
                if run_pre_quantize_compile:
                    test_flags.append("pre_quantize_compile")
                if run_compile:
                    test_flags.append("compile")
                if run_profile:
                    test_flags.append("profile")
                if run_inference:
                    test_flags.append("inference")
                if run_quantize:
                    test_flags.append("quantize")
                if run_trace:
                    test_flags.append("trace")
                if run_compute_device_accuracy:
                    test_flags.append("compute_device_accuracy")
                if run_export:
                    test_flags.append("export")
                if test_flags:
                    extras_args += ["-m", f'"{" or ".join(test_flags)}"']

                    # Create temporary directory for storing cloned & downloaded test artifacts.
                    with TemporaryDirectory() as tmpdir:
                        env = os.environ.copy()
                        if not use_shared_cache:
                            env[STORE_ROOT_ENV_VAR] = tmpdir

                        # Standard Test Suite
                        model_dir = os.path.join(PY_PACKAGE_MODELS_ROOT, model_name)
                        model_test = os.path.join(model_dir, "test.py")
                        generated_model_test = os.path.join(
                            model_dir, "test_generated.py"
                        )

                        if os.path.exists(model_test) or os.path.exists(
                            generated_model_test
                        ):
                            tasks.append(
                                PyTestTask(
                                    group_name=f"Model Tests: {model_name}",
                                    venv=model_venv,
                                    files_or_dirs=model_dir,
                                    parallel=False,
                                    extra_args=" ".join([*extras_args, "--no-header"]),
                                    env=env,
                                    raise_on_failure=not needs_model_venv,  # Do not raise on failure if a model venv was created, to make sure the venv is removed when the test finishes
                                    ignore_no_tests_return_code=True,
                                    include_pytest_cmd_in_status_message=False,
                                    junit_xml_path=junit_xml_path,
                                )
                            )

                if run_mypy:
                    tasks.append(
                        RunCommandsWithVenvTask(
                            f"MyPy: {model_name}",
                            model_venv,
                            # MyPy errors on "unused #type: ignore" from unrelated model code, if run in a non-global environment.
                            # Therefore we run mypy only on the specific model folder for specific model environments.
                            [
                                f'mypy --warn-unused-configs --config-file="{REPO_ROOT}/pyproject.toml" --package qai_hub_models.models.{model_name}'
                            ],
                            raise_on_failure=False,
                        )
                    )

            if not venv:
                tasks.append(
                    RunCommandsTask(
                        f"Remove virtual environment at {model_venv}",
                        f"rm -rf {model_venv}",
                    )
                )

        super().__init__(
            # If a group name is used here, you get two groups per model
            # printed to console when running these tasks, one of which is empty.
            None,
            list(tasks),
            continue_after_single_task_failure=True,
            raise_on_failure=raise_on_failure,
            show_subtasks_in_failure_message=False,
        )


class PyTestModelsTask(CompositeTask):
    """Run tests for the provided set of models."""

    def __init__(
        self,
        python_executable: str,
        models_for_testing: Iterable[str],
        models_to_test_export: Iterable[str],
        base_test_venv: str | None = None,  # Env with QAIHM installed
        venv_for_each_model: bool = True,  # Create a fresh venv for each model instead of using the base test venv instead.
        use_shared_cache: bool = False,  # Use the global QAIHM cache rather than a temporary one for tests.
        test_trace: bool = True,
        run_mypy: bool = False,  # Only run mypy for model folders that don't use global requirements. Global reqs are covered by the precommit.
        run_general: bool = True,
        run_export_pre_quantize_compile: bool = False,
        run_export_quantize: bool = False,
        run_export_compile: bool = True,
        run_export_profile: bool = False,
        run_export_inference: bool = False,
        run_compute_device_accuracy: bool = False,
        run_full_export: bool = False,
        exit_after_single_model_failure: bool = False,
        raise_on_failure: bool = True,
        qaihm_wheel_dir: str | os.PathLike | None = None,
        junit_xml_path: str | None = None,
    ) -> None:
        self.exit_after_single_model_failure = exit_after_single_model_failure

        # Get base JUnit XML path from environment variable if not provided
        base_junit_xml_path = None
        if junit_xml_path is None:
            base_junit_xml_path = os.environ.get("QAIHM_MODELS_JUNIT_XML_PATH")

        if len(models_for_testing) == 0 and len(models_to_test_export) == 0:
            super().__init__("All Per-Model Tests (Skipped)", [])
            return
        tasks = []

        # Whether or not export tests will be run asynchronously
        # (submit all jobs for all models at once, rather than one model at a time).
        test_hub_async: bool = bool(os.environ.get("QAIHM_TEST_HUB_ASYNC", "0"))

        if test_hub_async and not on_ci():
            for run_test, job_type in [
                (run_export_quantize, "quantize"),
                (run_export_compile, "compile"),
                (run_export_profile, "profile"),
                (run_export_inference, "inference"),
            ]:
                if run_test:
                    # Clean previous cached test jobs.
                    filename = os.getenv(
                        "QAIHM_TEST_ARTIFACTS_DOR",
                        os.path.join(
                            os.getcwd(), "qaihm_test_artifacts", f"{job_type}-jobs.yaml"
                        ),
                    )
                    tasks.append(
                        RunCommandsTask(
                            "Delete stored compile jobs from past test runs.",
                            f'if [ -f "{filename}" ]; then rm "{filename}"; fi',
                        )
                    )

        has_venv = base_test_venv is not None
        if not has_venv and (not venv_for_each_model or test_hub_async):
            # Create Venv
            base_test_venv = os.path.join(BUILD_ROOT, "test", "base_venv")
            tasks.append(CreateVenvTask(base_test_venv, python_executable))
            tasks.append(
                SyncLocalQAIHMVenvTask(
                    base_test_venv, ["dev"], qaihm_wheel_dir=qaihm_wheel_dir
                )
            )

        print(f"Tests to be run for models: {models_for_testing}")
        global_models = set()
        if not venv_for_each_model:
            for model_name in models_for_testing:
                if not check_code_gen_field(
                    model_name, "global_requirements_incompatible"
                ):
                    global_models.add(model_name)

            if len(global_models) > 0:
                tasks.append(InstallGlobalRequirementsTask(base_test_venv))

        # Sort models for ease of tracking how far along the tests are.
        # Do reverse order because whisper is slow to compile, so trigger earlier.
        export_models = models_to_test_export
        hub_quantized_models = []
        nonhub_quantized_models = []
        for model in models_for_testing:
            if get_is_hub_quantized(model) and model in export_models:
                hub_quantized_models.append(model)
            else:
                nonhub_quantized_models.append(model)

        if run_export_quantize:
            models_to_run = hub_quantized_models
        else:
            # Run hub quantized models last to give quantize job time to complete
            models_to_run = nonhub_quantized_models + hub_quantized_models
        for model_name in models_to_run:
            # Run standard test suite for this model.
            is_global_model = model_name in global_models

            # Create a model-specific JUnit XML path if base path is provided
            model_junit_xml_path = junit_xml_path
            if base_junit_xml_path and not model_junit_xml_path:
                base_dir = os.path.dirname(base_junit_xml_path)
                base_filename = os.path.basename(base_junit_xml_path)

                filename_parts = os.path.splitext(base_filename)
                model_filename = f"{filename_parts[0]}-{model_name}{filename_parts[1]}"
                model_junit_xml_path = os.path.join(base_dir, model_filename)

            tasks.append(
                PyTestModelTask(
                    model_name,
                    python_executable,
                    venv=base_test_venv if is_global_model else None,
                    use_shared_cache=use_shared_cache,
                    install_deps=not is_global_model,
                    run_trace=test_trace,
                    run_mypy=run_mypy and not is_global_model,
                    run_general=run_general,
                    run_pre_quantize_compile=run_export_pre_quantize_compile
                    and model_name in export_models,
                    run_quantize=run_export_quantize and model_name in export_models,
                    run_compile=run_export_compile and model_name in export_models,
                    run_profile=run_export_profile and model_name in export_models,
                    run_inference=run_export_inference and model_name in export_models,
                    run_compute_device_accuracy=run_compute_device_accuracy
                    and model_name in export_models,
                    run_export=run_full_export and model_name in export_models,
                    # Do not raise on failure; let PyTestModelsTask::run_tasks handle this
                    raise_on_failure=False,
                    qaihm_wheel_dir=qaihm_wheel_dir,
                    junit_xml_path=model_junit_xml_path,
                )
            )

        if test_hub_async and run_export_compile and not has_venv:
            # Cleanup venv
            tasks.append(RunCommandsTask(base_test_venv, f"rm -rf {base_test_venv}"))

        super().__init__(
            "All Per-Model Tests",
            list(tasks),
            continue_after_single_task_failure=True,
            raise_on_failure=raise_on_failure,
        )

    def run_task(self) -> bool:
        result: bool = True
        for task in self.tasks:
            try:
                task_result = task.run()
            except Exception:
                task_result = False
            if not task_result:
                if (
                    isinstance(task, PyTestModelTask)
                    and self.exit_after_single_model_failure
                ):
                    self.tasks[-1].run()  # cleanup venv
                    break
                if not self.continue_after_single_task_failure:
                    break
            result = result and task_result
        return result


class GenerateTestSummaryTask(RunCommandsTask):
    """Generate a test failure summary from JUnit XML files."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
    ) -> None:
        """
        Initialize the task.

        Parameters
        ----------
        input_dir
            Directory containing JUnit XML files (searched recursively).
        output_dir
            Output directory for combined.xml and summary.md.
        """
        super().__init__(
            group_name="Generate Test Failure Summary",
            commands=[
                f'python3 scripts/generate_test_summary.py --input="{input_dir}" --output="{output_dir}"'
            ],
        )


class LlamaCppBenchmarkTask(RunCommandsWithVenvTask):
    """Task to run Llama.CPP benchmarks on QDC devices.

    This task runs the benchmark script in a venv with QAIHM installed.
    Configuration is passed via environment variables:
    - QAIHM_MODELS: Comma-separated list of models or 'all'
    - QAIHM_TEST_DEVICES: Comma-separated list of devices
    - QDC_API_KEY: API token for QDC authentication
    - LLAMA_CPP_PATH: Path to Llama.CPP build directory
    - LLAMA_CPP_RESULTS_DIR: Directory to save perf.yaml files
    """

    def __init__(
        self,
        venv: str | None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        models = os.environ.get("QAIHM_MODELS", "all")
        devices = os.environ.get("QAIHM_TEST_DEVICES", "Snapdragon 8 Elite Gen 5 QRD")
        llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "llama.cpp")
        results_dir = os.environ.get("LLAMA_CPP_RESULTS_DIR", "llama_cpp_results")

        command = (
            f"python -m qai_hub_models.scripts.run_llama_cpp_benchmarks"
            f' --models "{models}"'
            f' --devices "{devices}"'
            f' --llama-cpp-path "{llama_cpp_path}"'
            f' --results-dir "{results_dir}"'
        )

        super().__init__(
            "Llama.CPP Benchmarks",
            venv,
            [command],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )

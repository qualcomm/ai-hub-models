#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import logging
import os
import sys
import textwrap
from collections.abc import Callable

from tasks.aws import REPO_ROOT, ValidateAwsCredentialsTask
from tasks.changes import (
    PrintCITestModelsTask,
    get_all_models,
)
from tasks.constants import BUILD_ROOT, DEFAULT_PYTHON, PY_PACKAGE_SRC_ROOT, VENV_PATH
from tasks.plan import (
    ALL_TASKS,
    PUBLIC_TASKS,
    SUMMARIZERS,
    TASK_DEPENDENCIES,
    Plan,
    depends,
    depends_if,
    public_task,
    task,
)
from tasks.release import (
    BuildPublicRepositoryTask,
    BuildWheelTask,
    CreateReleaseVenv,
    PublishWheelTask,
    PushRepositoryTask,
)
from tasks.task import (
    CompositeTask,
    ConditionalTask,
    ListTasksTask,
    NoOpTask,
    PyTestTask,
    RunCommandsWithVenvTask,
    Task,
)
from tasks.test import (
    GenerateTestSummaryTask,
    GPUPyTestModelsTask,
    InstallGlobalRequirementsTask,
    LlamaCppBenchmarkTask,
    PyTestModelsTask,
    PyTestQAIHMTask,
)
from tasks.util import echo, get_env_bool, on_ci, run
from tasks.venv import (
    AggregateScorecardResultsTask,
    CreateVenvTask,
    DownloadQDCWheelTask,
    GenerateGlobalRequirementsTask,
    SyncLocalQAIHMVenvTask,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and test all the things.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--task",
        "--tasks",
        dest="legacy_task",
        type=str,
        help="[deprecated] Comma-separated list of tasks to run; use --task=list_tasks to list all tasks.",
    )
    parser.add_argument(
        "task",
        type=str,
        nargs="*",
        help='Task(s) to run. Specify "list" to show all tasks.',
    )

    parser.add_argument(
        "--only",
        action="store_true",
        help="Run only the listed task(s), skipping any dependencies.",
    )

    parser.add_argument(
        "--print-task-graph",
        action="store_true",
        help="Print the task library in DOT format and exit. Combine with --task to highlight what would run.",
    )

    parser.add_argument(
        "--python",
        type=str,
        default=DEFAULT_PYTHON,
        help="Python executable path or name (only used when creating the venv).",
    )

    parser.add_argument(
        "--venv",
        type=str,
        metavar="...",
        default=VENV_PATH,
        help=textwrap.dedent(
            """\
                    [optional] Use the virtual env at the specified path.
                    - Creates a virtual env at that path if none exists.
                    - If omitted, creates and uses a virtual environment at """
            + VENV_PATH
            + """
                    - If [none], does not create or activate a virtual environment.
                    """
        ),
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan, rather than running it."
    )

    args = parser.parse_args()
    if args.legacy_task:
        args.task.extend(args.legacy_task.split(","))
    delattr(args, "legacy_task")
    return args


DEFAULT_RELEASE_DIRECTORY = os.path.join(BUILD_ROOT, "release")
RELEASE_VENV = os.path.join(BUILD_ROOT, "release_venv")
RELEASE_WHEEL_DIR = os.path.join(DEFAULT_RELEASE_DIRECTORY, "wheel")
RELEASE_REPO_DIR = os.path.join(DEFAULT_RELEASE_DIRECTORY, "repository")
PRIVATE_WHEEL_DIR = os.path.join(BUILD_ROOT, "wheel")


def get_test_venv_wheel_dir() -> str | None:
    """
    Get the directory with built wheels that should be used for testing.
    The wheel will exists so long as install_deps is a dependency of the current task.

    Returns None if an editable install should be used instead.
    """
    if get_env_bool("QAIHM_TEST_USE_PUBLIC_WHEEL"):
        return RELEASE_WHEEL_DIR
    if on_ci() and not get_env_bool("QAIHM_CI_USE_EDITABLE_INSTALL"):
        return PRIVATE_WHEEL_DIR
    return None  # editable install


class TaskLibrary:
    def __init__(
        self,
        python_executable: str,
        venv_path: str | None,
    ) -> None:
        self.python_executable = python_executable
        self.venv_path = venv_path

    @staticmethod
    def to_dot(highlight: list[str] | None = None) -> str:
        elements: list[str] = []
        for tsk in ALL_TASKS:
            task_attrs: list[str] = []
            if tsk in PUBLIC_TASKS:
                task_attrs.append("style=filled")
            if tsk in (highlight or []):
                task_attrs.append("penwidth=4.0")
            if len(task_attrs) > 0:
                elements.append(f"{tsk} [{' '.join(task_attrs)}]")
            else:
                elements.append(tsk)
        for tsk in TASK_DEPENDENCIES:
            for dep in TASK_DEPENDENCIES[tsk]:
                elements.append(f"{tsk} -> {dep}")
        elements_str = "\n".join([f"  {element};" for element in elements])
        return f"digraph {{\n{elements_str}\n}}"

    @public_task("Print a list of commonly used tasks; see also --task=list_all.")
    @depends(["list_public"])
    def list_tasks(self, plan: Plan) -> str:
        return plan.add_step("list_tasks", NoOpTask())

    @task
    def list_all(self, plan: Plan) -> str:
        return plan.add_step("list_all", ListTasksTask(ALL_TASKS))

    @task
    def list_public(self, plan: Plan) -> str:
        return plan.add_step("list_public", ListTasksTask(PUBLIC_TASKS))

    @public_task("precheckin")
    @depends(
        [
            "test_qaihm",
            "test_changed_models",
        ]
    )
    def precheckin(self, plan: Plan) -> str:
        # Excludes export tests, and uses the same environment for each model.
        return plan.add_step("precheckin", NoOpTask())

    @public_task("precheckin_long")
    @depends(
        [
            "test_qaihm",
            "test_changed_models_long",
        ]
    )
    def precheckin_long(self, plan: Plan) -> str:
        # Includes export tests, and creates a fresh environment for each model.
        return plan.add_step("precheckin_long", NoOpTask())

    @public_task("all_tests")
    @depends(
        [
            "test_qaihm",
            "test_all_models",
        ]
    )
    def all_tests(self, plan: Plan) -> str:
        return plan.add_step("all_tests", NoOpTask())

    @public_task("all_tests_long")
    @depends(["test_qaihm"])
    def all_tests_long(self, plan: Plan) -> str:
        return plan.add_step("all_tests_long", NoOpTask())

    @task
    @depends(["install_deps"])
    def validate_aws_credentials(
        self, plan: Plan, step_id: str = "validate_aws_credentials"
    ) -> str:
        return plan.add_step(step_id, ValidateAwsCredentialsTask(self.venv_path))

    @task
    def create_venv(self, plan: Plan, step_id: str = "create_venv") -> str:
        return plan.add_step(
            step_id,
            ConditionalTask(
                group_name=None,
                condition=lambda: self.venv_path is None
                or os.path.exists(self.venv_path),
                true_task=NoOpTask(
                    f"Using virtual environment at {self.venv_path}."
                    if self.venv_path
                    else "Using currently active python environment."
                ),
                false_task=CreateVenvTask(self.venv_path, self.python_executable),
            ),
        )

    @public_task("Install dependencies for model zoo.")
    @depends_if(
        get_test_venv_wheel_dir(),
        eq=[
            (RELEASE_WHEEL_DIR, ["build_public_wheel", "create_venv"]),
            (PRIVATE_WHEEL_DIR, ["build_internal_wheel", "create_venv"]),
            # no dependencies (editable install) otherwise
        ],
        default=["create_venv"],
    )
    def install_deps(self, plan: Plan, step_id: str = "install_deps") -> str:
        return plan.add_step(
            step_id,
            SyncLocalQAIHMVenvTask(
                self.venv_path, ["dev"], qaihm_wheel_dir=get_test_venv_wheel_dir()
            ),
        )

    @public_task("Install Global Requirements")
    @depends(["install_deps", "generate_global_requirements"])
    def install_global_requirements(
        self, plan: Plan, step_id: str = "install_deps"
    ) -> str:
        return plan.add_step(
            step_id,
            InstallGlobalRequirementsTask(self.venv_path),
        )

    @public_task("Generate Global Requirements")
    @depends(["install_deps"])
    def generate_global_requirements(
        self, plan: Plan, step_id: str = "generate_global_requirements"
    ) -> str:
        return plan.add_step(
            step_id,
            GenerateGlobalRequirementsTask(
                venv=self.venv_path,
            ),
        )

    @public_task("Aggregate Scorecard Results")
    @depends(["install_deps"])
    def aggregate_scorecard_results(
        self, plan: Plan, step_id: str = "aggregate_scorecard_results"
    ) -> str:
        return plan.add_step(
            step_id,
            AggregateScorecardResultsTask(
                venv=self.venv_path,
            ),
        )

    @public_task("Download QDC wheel")
    @depends(["install_deps", "validate_aws_credentials"])
    def download_qdc_wheel(
        self, plan: Plan, step_id: str = "download_qdc_wheel"
    ) -> str:
        return plan.add_step(
            step_id,
            DownloadQDCWheelTask(
                venv=self.venv_path,
            ),
        )

    @public_task("Run Llama.CPP benchmarks on QDC devices")
    @depends(["install_deps"])
    def run_llama_cpp_benchmark(
        self, plan: Plan, step_id: str = "run_llama_cpp_benchmark"
    ) -> str:
        """
        Run Llama.CPP benchmarks. Uses environment variables:
        - LLAMA_CPP_PATH: Path to Llama.CPP binaries
        - QAIHM_TEST_MODELS: Comma-separated model names, or "all" (default: "all")
        - QAIHM_TEST_DEVICES: Comma-separated list of QDC device names
        - QDC_API_KEY: QDC API token

        Model configs are defined in qai_hub_models/scripts/run_llama_cpp_benchmarks.py
        """
        return plan.add_step(
            step_id,
            LlamaCppBenchmarkTask(venv=self.venv_path),
        )

    @public_task("Model Test Setup")
    @depends(["install_deps", "generate_global_requirements"])
    def model_test_setup(self, plan: Plan, step_id: str = "model_test_setup") -> str:
        return plan.add_step(step_id, NoOpTask())

    @task
    def clean_pip(self, plan: Plan) -> str:
        class CleanPipTask(Task):
            def __init__(self, venv_path: str | None) -> None:
                super().__init__("Deleting python packages")
                self.venv_path = venv_path

            def does_work(self) -> bool:
                return True

            def run_task(self) -> bool:
                if self.venv_path is not None:
                    # Some sanity checking to make sure we don't accidentally "rm -rf /"
                    if not self.venv_path.startswith(os.environ["HOME"]):
                        run(f"rm -rI {self.venv_path}")
                    else:
                        run(f"rm -rf {self.venv_path}")
                return True

        return plan.add_step("clean_pip", CleanPipTask(self.venv_path))

    @public_task("Run tests for all files except models.")
    @depends(["install_deps"])
    def test_qaihm(self, plan: Plan, step_id: str = "test_qaihm") -> str:
        return plan.add_step(
            step_id,
            PyTestQAIHMTask(self.venv_path),
        )

    @public_task("Output list of models to test in CI as comma-separated list")
    def get_ci_test_models(self, plan: Plan) -> str:
        return plan.add_step("get_ci_test_models", PrintCITestModelsTask())

    @public_task("Run GPU weekly tests.")
    def test_gpu_models_weekly(
        self, plan: Plan, step_id: str = "test_gpu_models_weekly"
    ) -> str:
        return plan.add_step(
            step_id,
            GPUPyTestModelsTask(venv=self.venv_path),
        )

    @public_task("Run GPU nightly tests (only tests marked with @pytest.mark.nightly).")
    def test_gpu_models_nightly(
        self, plan: Plan, step_id: str = "test_gpu_models_nightly"
    ) -> str:
        return plan.add_step(
            step_id,
            GPUPyTestModelsTask(venv=self.venv_path, nightly_only=True),
        )

    @public_task("Generate perf.yamls.")
    @depends(["install_deps"])
    def create_perfs(self, plan: Plan, step_id: str = "generate_perfs") -> str:
        return plan.add_step(
            step_id,
            RunCommandsWithVenvTask(
                group_name=None,
                venv=self.venv_path,
                commands=[
                    "python qai_hub_models/scripts/collect_scorecard_results.py --gen-csv --gen-perf-summary --sync-code-gen"
                ],
            ),
        )

    @public_task("Generate numerics.yamls.")
    @depends(["install_deps"])
    def create_numerics(self, plan: Plan, step_id: str = "generate_numerics") -> str:
        return plan.add_step(
            step_id,
            RunCommandsWithVenvTask(
                group_name=None,
                venv=self.venv_path,
                commands=[
                    "python qai_hub_models/scripts/collect_scorecard_numerics_results.py --sync-code-gen"
                ],
            ),
        )

    @public_task("Generate release-assets.yaml files.")
    @depends(["install_deps"])
    def create_release_assets(
        self, plan: Plan, step_id: str = "create_release_assets"
    ) -> str:
        return plan.add_step(
            step_id,
            RunCommandsWithVenvTask(
                group_name=None,
                venv=self.venv_path,
                commands=[
                    "python qai_hub_models/scripts/collect_scorecard_assets_results.py"
                ],
            ),
        )

    def _make_unit_test_task(
        self,
        enable_compile: bool = False,
    ) -> PyTestModelsTask:
        models = get_all_models()
        return PyTestModelsTask(
            self.python_executable,
            models,
            models,
            self.venv_path,
            venv_for_each_model=False,
            use_shared_cache=True,
            run_general=True,
            run_export_compile=enable_compile,
            exit_after_single_model_failure=False,
            test_trace=False,
            qaihm_wheel_dir=get_test_venv_wheel_dir(),
            run_mypy=True,
        )

    @public_task("Run unit tests and mymy for all models.")
    @depends(["model_test_setup"])
    def test_unit_all_models(
        self, plan: Plan, step_id: str = "test_unit_all_models"
    ) -> str:
        return plan.add_step(step_id, self._make_unit_test_task())

    @public_task("Run unit tests, mypy, and compile jobs for all models.")
    @depends(["model_test_setup"])
    def test_unit_and_compile_all_models(
        self, plan: Plan, step_id: str = "test_unit_and_compile_all_models"
    ) -> str:
        return plan.add_step(step_id, self._make_unit_test_task(enable_compile=True))

    def _make_hub_scorecard_task(
        self,
        pre_quantize_compile: bool = False,
        quantize: bool = False,
        enable_compile: bool = False,
        enable_profile: bool = False,
        enable_inference: bool = False,
        enable_compute_device_accuracy: bool = False,
        enable_export_end2end: bool = False,
    ) -> PyTestModelsTask:
        models = get_all_models()
        return PyTestModelsTask(
            self.python_executable,
            models,
            models,
            self.venv_path,
            venv_for_each_model=False,
            use_shared_cache=True,
            run_general=False,
            run_export_pre_quantize_compile=pre_quantize_compile,
            run_export_quantize=quantize,
            run_export_compile=enable_compile,
            run_export_profile=enable_profile,
            run_export_inference=enable_inference,
            run_compute_device_accuracy=enable_compute_device_accuracy,
            run_full_export=enable_export_end2end,
            # If one model fails, we should still try the others.
            exit_after_single_model_failure=False,
            test_trace=False,
            qaihm_wheel_dir=get_test_venv_wheel_dir(),
        )

    @public_task("Run pre-quantize ONNX compile jobs for all models.")
    @depends(["model_test_setup"])
    def test_pre_quantize_compile_all_models(
        self, plan: Plan, step_id: str = "test_pre_quantize_compile_all_models"
    ) -> str:
        return plan.add_step(
            step_id, self._make_hub_scorecard_task(pre_quantize_compile=True)
        )

    @public_task("Run quantize jobs for all models.")
    @depends(["model_test_setup"])
    def test_quantize_all_models(
        self, plan: Plan, step_id: str = "test_quantize_all_models"
    ) -> str:
        return plan.add_step(step_id, self._make_hub_scorecard_task(quantize=True))

    @public_task("Run Compile jobs for all models.")
    @depends(["model_test_setup"])
    def test_compile_all_models(
        self, plan: Plan, step_id: str = "test_compile_all_models"
    ) -> str:
        return plan.add_step(
            step_id, self._make_hub_scorecard_task(enable_compile=True)
        )

    @public_task("Run profile jobs for all models.")
    @depends(["model_test_setup"])
    def test_profile_all_models(
        self, plan: Plan, step_id: str = "test_profile_all_models"
    ) -> str:
        return plan.add_step(
            step_id, self._make_hub_scorecard_task(enable_profile=True)
        )

    @public_task("Run inference jobs for all models.")
    @depends(["model_test_setup"])
    def test_inference_all_models(
        self, plan: Plan, step_id: str = "test_inference_all_models"
    ) -> str:
        return plan.add_step(
            step_id, self._make_hub_scorecard_task(enable_inference=True)
        )

    @public_task("Run profile and inference jobs for all models.")
    @depends(["model_test_setup"])
    def test_profile_inference_all_models(
        self, plan: Plan, step_id: str = "test_profile_inference_all_models"
    ) -> str:
        return plan.add_step(
            step_id,
            self._make_hub_scorecard_task(enable_profile=True, enable_inference=True),
        )

    @public_task(
        "Compute accuracy metrics using output of cached inference jobs submitted by `test_inference_all_models`"
    )
    @depends(["model_test_setup"])
    def compute_device_accuracy_metrics(
        self, plan: Plan, step_id: str = "compute_device_accuracy_metrics"
    ) -> str:
        return plan.add_step(
            step_id, self._make_hub_scorecard_task(enable_compute_device_accuracy=True)
        )

    @public_task("Verify all export scripts work end-to-end.")
    @depends(["model_test_setup"])
    def test_export_scripts(
        self, plan: Plan, step_id: str = "test_export_scripts"
    ) -> str:
        return plan.add_step(
            step_id, self._make_hub_scorecard_task(enable_export_end2end=True)
        )

    @public_task("Verify all async compile jobs completed successfully.")
    @depends(["install_deps"])
    def verify_compile_jobs(
        self, plan: Plan, step_id: str = "verify_compile_jobs"
    ) -> str:
        junit_xml_path = os.environ.get("QAIHM_JUNIT_XML_PATH")
        return plan.add_step(
            step_id,
            PyTestTask(
                group_name="Verify Compile Jobs Success",
                venv=self.venv_path,
                files_or_dirs=os.path.join(
                    PY_PACKAGE_SRC_ROOT, "test", "test_async_compile_jobs.py"
                ),
                parallel=False,
                extra_args="-s",
                junit_xml_path=junit_xml_path,
            ),
        )

    @public_task("Build & Install QAIHM Release Rependencies")
    def install_release_deps(
        self, plan: Plan, step_id: str = "install_release_deps"
    ) -> str:
        release_venv_task = CreateReleaseVenv(RELEASE_VENV, self.python_executable)
        return plan.add_step(step_id, release_venv_task)

    @public_task(
        "Build Public Copy of the Repository (with internal information removed)"
    )
    def build_public_repository(
        self, plan: Plan, step_id: str = "build_public_repository"
    ) -> str:
        return plan.add_step(
            step_id,
            CompositeTask(
                group_name=None,
                tasks=[
                    # "install_deps" will call this task if the user wants to use the public package for testing.
                    # To avoid a circular dependency, we use an editable install to first build the public package.
                    SyncLocalQAIHMVenvTask(
                        self.venv_path, ["dev"], qaihm_wheel_dir=None
                    ),
                    BuildPublicRepositoryTask(
                        self.venv_path,
                        RELEASE_REPO_DIR,
                    ),
                ],
            ),
        )

    @public_task(description="Build Public Python Wheel")
    @depends(["install_release_deps", "build_public_repository"])
    def build_public_wheel(
        self, plan: Plan, step_id: str = "build_public_wheel"
    ) -> str:
        return plan.add_step(
            step_id,
            BuildWheelTask(
                RELEASE_VENV,
                RELEASE_REPO_DIR,
                wheel_dir=RELEASE_WHEEL_DIR,
            ),
        )

    @public_task("Build Internal Python Wheel")
    @depends(["install_release_deps"])
    def build_internal_wheel(
        self, plan: Plan, step_id: str = "build_internal_wheel"
    ) -> str:
        return plan.add_step(
            step_id,
            BuildWheelTask(RELEASE_VENV, REPO_ROOT, PRIVATE_WHEEL_DIR),
        )

    @public_task("Release QAIHM Wheel to PyPi")
    @depends(["build_public_wheel"])
    def release_wheel(self, plan: Plan, step_id: str = "release_wheel") -> str:
        return plan.add_step(
            step_id,
            PublishWheelTask(RELEASE_WHEEL_DIR, RELEASE_VENV),
        )

    @public_task("Push QAIHM Code to GitHub")
    @depends(["build_public_repository"])
    def release_code(self, plan: Plan, step_id: str = "release_code") -> str:
        return plan.add_step(
            step_id,
            PushRepositoryTask(RELEASE_REPO_DIR),
        )

    @public_task("Push QAIHM Assets to AWS S3")
    @depends(["install_deps"])
    def release_assets(self, plan: Plan, step_id: str = "release_assets") -> str:
        return plan.add_step(
            step_id,
            RunCommandsWithVenvTask(
                group_name=None,
                venv=self.venv_path,
                commands=["python qai_hub_models/scripts/publish_release_assets.py"],
            ),
        )

    @public_task("Push QAIHM model Cards to Hugging Face")
    @depends(["install_deps"])
    def release_huggingface(
        self, plan: Plan, step_id: str = "release_huggingface"
    ) -> str:
        return plan.add_step(
            step_id,
            RunCommandsWithVenvTask(
                group_name=None,
                venv=self.venv_path,
                commands=[
                    "python qai_hub_models/scripts/generate_hf_model_readme.py --public --models ALL"
                ],
            ),
        )

    @public_task(
        "Push QAIHM Code, Wheel, and Assets (build repo & wheel, push repo, push assets, push HF model cards)"
    )
    @depends(["release_assets", "release_code", "release_wheel", "release_huggingface"])
    def release(self, plan: Plan, step_id: str = "release") -> str:
        return plan.add_step(
            step_id,
            NoOpTask("Release AI Hub Models"),
        )

    @public_task("Generate Test Failure Summary")
    def generate_test_summary(
        self, plan: Plan, step_id: str = "generate_test_summary"
    ) -> str:
        # Use the workspace directory for test results
        results_dir = os.path.join(os.getcwd(), "test-results")
        return plan.add_step(
            step_id,
            GenerateTestSummaryTask(input_dir=results_dir, output_dir=results_dir),
        )

    # This task has no depedencies and does nothing.
    @task
    def nop(self, plan: Plan) -> str:
        return plan.add_step("nop", NoOpTask())


def plan_from_dependencies(
    main_tasks: list[str],
    python_executable: str,
    venv_path: str,
) -> Plan:
    task_library = TaskLibrary(
        python_executable,
        venv_path,
    )
    plan = Plan()

    # We always run summarizers, which perform conditional work on the output
    # of other steps.
    work_list = SUMMARIZERS

    # The work list is processed as a stack, so LIFO. We reverse the user-specified
    # tasks so that they (and their dependencies) can be expressed in a natural order.
    work_list.extend(reversed(main_tasks))

    for task_name in work_list:
        if not hasattr(task_library, task_name):
            echo(f"Task '{task_name}' does not exist.", file=sys.stderr)
            sys.exit(1)

    while len(work_list) > 0:
        task_name = work_list.pop()
        unfulfilled_deps: list[str] = []
        for dep in TASK_DEPENDENCIES.get(task_name, []):
            if not plan.has_step(dep):
                unfulfilled_deps.append(dep)
                if not hasattr(task_library, dep):
                    echo(
                        f"Non-existent task '{dep}' was declared as a dependency for '{task_name}'.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
        if len(unfulfilled_deps) == 0:
            # add task_name to plan
            task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
            task_adder(plan)
        else:
            # Look at task_name again later when its deps are satisfied
            work_list.append(task_name)
            work_list.extend(reversed(unfulfilled_deps))

    return plan


def plan_from_task_list(
    tasks: list[str],
    python_executable: str,
    venv_path: str,
) -> Plan:
    task_library = TaskLibrary(
        python_executable,
        venv_path,
    )
    plan = Plan()
    for task_name in tasks:
        # add task_name to plan
        task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
        task_adder(plan)
    return plan


def build_and_test() -> None:
    log_format = "[%(asctime)s] [bnt] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format)

    args = parse_arguments()

    venv_path = args.venv if args.venv.lower() != "none" else None
    python_executable = args.python if venv_path else "python"

    plan = Plan()

    if len(args.task) > 0:
        planner = plan_from_task_list if args.only else plan_from_dependencies
        plan = planner(
            args.task,
            python_executable,
            venv_path,
        )

    if args.print_task_graph:
        print(TaskLibrary.to_dot(plan.steps))
        sys.exit(0)
    elif len(args.task) == 0:
        echo("At least one task or --print-task-graph is required.")

    if args.dry_run:
        plan.print()
    else:
        caught = None
        try:
            plan.run()
        except Exception as ex:
            caught = ex
        if plan.has_report():
            print()
            plan.print_report()
            print()
        if caught:
            raise caught


if __name__ == "__main__":
    build_and_test()

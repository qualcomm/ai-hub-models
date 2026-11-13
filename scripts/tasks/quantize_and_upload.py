# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
import shlex
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from .github import append_to_summary
from .task import (
    CompositeTask,
    RunCommandsTask,
    RunCommandsWithVenvTask,
    Task,
)
from .util import check_manifest_field, get_manifest_str_field

# (task, num_samples, metric line prefix) run on both the quantized checkpoint and
# the unquantized model; 0 runs the whole split.
EVAL_TASKS = (("wikitext", 0, "PPL"), ("mmlu", 1000, "MMLU:"))
FAILURE_LOG_TAIL_LINES = 40

# Profile ci-setup writes when given aws-role-arn.
AWS_PROFILE = "qaihm"
PRIVATE_BUCKET = "qai-hub-models-private-assets"
EPHEMERAL_QUANTIZE_ASSETS_PREFIX = "ephemeral_quantize_assets/gh_actions"

SHARING_RESTRICTED_FILENAME = "SHARING_RESTRICTED.txt"
# The exported graph and its weights: the model itself, as opposed to the
# encodings, which are our own work. Left out of the zip for restricted models.
WEIGHT_FILENAMES = ("model_dynamic.onnx", "model.data")


def format_eval_logs(checkpoint_dir: Path) -> str:
    """Task x checkpoint table of each eval log's metric line, else its last line."""
    if not any(checkpoint_dir.glob("eval_*.log")):
        return ""
    rows = ["| Task | Quantized | Unquantized |", "|---|---|---|"]
    for task, _, metric_prefix in EVAL_TASKS:
        cells = []
        for label in ("quantized", "unquantized"):
            log_path = checkpoint_dir / f"eval_{label}_{task}.log"
            text = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.exists()
                else ""
            )
            lines = [x.strip() for x in reversed(text.splitlines()) if x.strip()]
            # Last line is the error when eval failed.
            last = next(
                (x for x in lines if x.startswith(metric_prefix)),
                lines[0] if lines else "",
            )
            # A | in an error message would split the cell.
            cells.append(
                f"`{last.replace('|', chr(92) + '|')}`" if last else "(no output)"
            )
        rows.append(f"| {task} | {' | '.join(cells)} |")
    return "\n".join(rows)


def archive_checkpoint(
    checkpoint_dir: Path, zip_path: Path, exclude_weights: bool
) -> None:
    """Zip *checkpoint_dir*, leaving WEIGHT_FILENAMES out when *exclude_weights*."""
    skipped: list[str] = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(checkpoint_dir.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(checkpoint_dir).as_posix()
            if exclude_weights and path.name in WEIGHT_FILENAMES:
                skipped.append(name)
                continue
            archive.write(path, name)
    if skipped:
        print(f"Excluded from the archive (license-restricted): {', '.join(skipped)}")


def build_upload_report(
    model_id: str,
    precision: str,
    s3_key: str,
    eval_logs: str,
    restriction: str | None,
    local_commands: list[str],
) -> str:
    """Markdown report for the run summary: where the zip is and how to use it."""
    s3_url = f"s3://{PRIVATE_BUCKET}/{s3_key}"
    sections = [f"## Quantized `{model_id}` {precision}"]
    if eval_logs:
        sections.append(f"### Evaluation\n\n{eval_logs}")
    sections.append(
        "### Reproduce locally\n\n```bash\n" + "\n".join(local_commands) + "\n```"
    )
    sections.append(f"### Asset\n\n```\n{s3_url}\n```")
    sections.append(
        "### Run scorecard on it\n\nDispatch the scorecard workflow with "
        f"`models` = `{model_id}` and `options` = (paste exactly, the bucket name "
        f"is not part of it):\n\n```\ncheckpoint={s3_key}\n```"
    )
    if restriction:
        sections.append(
            f"### :warning: Sharing restricted\n\n`{model_id}` restricts model "
            f"sharing (license: `{restriction}`), so "
            f"`{'` and `'.join(WEIGHT_FILENAMES)}` are **excluded** from the "
            "archive above: it carries the encodings, configs and tokenizer "
            "only. A consumer re-exports the ONNX from the source weights under "
            "its own license acceptance. The archive must still not be copied to "
            "the public asset store."
        )
    return "\n\n".join(sections)


class FunctionTask(Task):
    """A Task that runs a python callable."""

    def __init__(self, group_name: str, fn: Callable[[], None]) -> None:
        super().__init__(group_name)
        self.fn = fn

    def does_work(self) -> bool:
        return True

    def run_task(self) -> bool:
        self.fn()
        return True


class QuantizeAndUploadTask(CompositeTask):
    """
    Quantize a model, evaluate the checkpoint and upload it to S3.

    Configured by the quantize_and_upload workflow's QAIHM_QUANTIZE_* env vars:
    MODEL_ID, PRECISION, ARGS, EVALUATE and CHECKPOINT_DIR, plus
    QAIHM_UNQUANTIZED_CHECKPOINT. Evaluation never fails the task: the checkpoint is the
    deliverable, and quantizing it may have taken many hours.
    """

    def __init__(self, venv: str | None) -> None:
        model_id = os.environ["QAIHM_QUANTIZE_MODEL_ID"]
        precision = os.environ["QAIHM_QUANTIZE_PRECISION"]
        unquantized_checkpoint = os.environ["QAIHM_UNQUANTIZED_CHECKPOINT"]
        out_dir = os.environ["QAIHM_QUANTIZE_CHECKPOINT_DIR"]
        # shlex.quote keeps every value inert: none of them is parsed as shell.
        q = shlex.quote
        extra_args = shlex.join(shlex.split(os.environ.get("QAIHM_QUANTIZE_ARGS", "")))

        self.quantize_log = os.path.join(
            os.environ.get("RUNNER_TEMP", tempfile.gettempdir()), "quantize.log"
        )
        self.model_id = model_id
        self.precision = precision

        def quantize_cmd(output_dir: str) -> str:
            return (
                f"python -m qai_hub_models.models.{model_id}.quantize "
                f"--precision {q(precision)} --checkpoint {q(unquantized_checkpoint)} "
                f"-o {q(output_dir)} {extra_args}"
            ).rstrip()

        def evaluate_cmd(checkpoint: str, task: str, num_samples: int) -> str:
            return (
                f"qai-hub-models evaluate {q(model_id)} --checkpoint {q(checkpoint)} "
                f"--task {q(task)} --num-samples {num_samples}"
            )

        def evaluate(checkpoint: str, label: str, task: str, num_samples: int) -> Task:
            # No pipefail: a failed metric is ignored, and the log it wrote is
            # what the upload report reads.
            return RunCommandsWithVenvTask(
                group_name=f"Evaluate {label} ({task})",
                venv=venv,
                commands=f"{evaluate_cmd(checkpoint, task, num_samples)} 2>&1 "
                f"| tee {q(os.path.join(out_dir, f'eval_{label}_{task}.log'))}",
                raise_on_failure=False,
            )

        # The summary's copy of the commands, with a local output dir in place of
        # the runner's.
        local_out = "./quantized_checkpoint"
        local_commands = [
            f"qai-hub-models install {q(model_id)}",
            quantize_cmd(local_out),
        ]

        tasks: list[Task] = [
            RunCommandsWithVenvTask(
                "Install model dependencies",
                venv,
                f"qai-hub-models install {q(model_id)} --yes",
            ),
            # pipefail so tee does not mask a failed quantize; the log feeds the
            # failure summary.
            RunCommandsWithVenvTask(
                "Quantize",
                venv,
                f"set -o pipefail && {quantize_cmd(out_dir)} 2>&1 "
                f"| tee {q(self.quantize_log)}",
            ),
        ]
        if os.environ.get("QAIHM_QUANTIZE_EVALUATE", "").lower() == "true":
            local_commands += [
                evaluate_cmd(checkpoint, task, num_samples)
                for checkpoint in (local_out, unquantized_checkpoint)
                for task, num_samples, _ in EVAL_TASKS
            ]
            # Quantized first: if the baseline dies, the numbers that matter survive.
            tasks.append(
                CompositeTask(
                    "Evaluate checkpoint",
                    [
                        evaluate(checkpoint, label, task, num_samples)
                        for checkpoint, label in (
                            (out_dir, "quantized"),
                            (unquantized_checkpoint, "unquantized"),
                        )
                        for task, num_samples, _ in EVAL_TASKS
                    ],
                    continue_after_single_task_failure=True,
                    raise_on_failure=False,
                )
            )
        # Namespaces the upload so two people quantizing the same model and
        # precision cannot overwrite each other.
        run_id = os.environ["GITHUB_RUN_ID"]
        checkpoint_dir = Path(out_dir)
        zip_path = checkpoint_dir.parent / f"{model_id}-{precision}.zip"
        s3_key = f"{EPHEMERAL_QUANTIZE_ASSETS_PREFIX}/{run_id}/{zip_path.name}"
        # The license when the manifest forbids redistributing weights, else None.
        restriction = (
            get_manifest_str_field(model_id, "license_type") or "unspecified"
            if check_manifest_field(model_id, "restrict_model_sharing")
            else None
        )

        def package() -> None:
            # Before zipping, so the marker rides along.
            if restriction:
                (checkpoint_dir / SHARING_RESTRICTED_FILENAME).write_text(
                    f"Sharing is restricted for {model_id} per its manifest.yaml "
                    f"(license: {restriction}).\n"
                    f"{' and '.join(WEIGHT_FILENAMES)} are excluded from this "
                    "archive; the consumer re-exports the ONNX from the source "
                    "weights under its own license acceptance.\n"
                )
            archive_checkpoint(checkpoint_dir, zip_path, bool(restriction))

        def report() -> None:
            append_to_summary(
                build_upload_report(
                    model_id,
                    precision,
                    s3_key,
                    format_eval_logs(checkpoint_dir),
                    restriction,
                    local_commands,
                )
            )

        tasks += [
            FunctionTask("Package checkpoint", package),
            RunCommandsTask(
                "Upload to S3",
                f"aws s3 cp {q(str(zip_path))} {q(f's3://{PRIVATE_BUCKET}/{s3_key}')} "
                f"--profile {AWS_PROFILE}",
            ),
            FunctionTask("Write upload report", report),
        ]
        super().__init__("Quantize and upload", tasks)

    def run_task(self) -> bool:
        result = super().run_task()
        if not result:
            self._write_failure_summary()
        return result

    def _write_failure_summary(self) -> None:
        # The report step only runs on success; without this a failed run's page
        # is empty.
        failed = next((t.group_name for t in self.tasks if t.last_result is False), "")
        lines = [
            f"## Quantizing `{self.model_id}` {self.precision} failed",
            "",
            f"Failed at **{failed}**. Nothing was uploaded to S3.",
            "",
        ]
        try:
            tail = Path(self.quantize_log).read_text().splitlines()
        except OSError:
            tail = []
        if tail:
            lines += [
                f"<details><summary>Last {FAILURE_LOG_TAIL_LINES} lines</summary>",
                "",
                "```",
                *tail[-FAILURE_LOG_TAIL_LINES:],
                "```",
                "</details>",
            ]
        else:
            lines.append("No output was captured; see the step log above.")
        append_to_summary("\n".join(lines))

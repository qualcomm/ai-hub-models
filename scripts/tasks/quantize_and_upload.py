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

# Only wikitext runs its whole split (0). The task is free text, so everything
# else is capped: the mmlu/mmmlu family is thousands of questions per language.
WHOLE_SPLIT_EVAL_PREFIX = "wikitext"
CAPPED_EVAL_SAMPLES = 1000
FAILURE_LOG_TAIL_LINES = 40

# Profile ci-setup writes when given aws-role-arn.
AWS_PROFILE = "qaihm"
PRIVATE_BUCKET = "qai-hub-models-private-assets"
EPHEMERAL_QUANTIZE_ASSETS_PREFIX = "ephemeral_quantize_assets/gh_actions"
# Lifecycle rule on the ephemeral_* prefixes; reported so the reader knows their
# deadline to copy.
EPHEMERAL_RETENTION_DAYS = 7

SHARING_RESTRICTED_FILENAME = "SHARING_RESTRICTED.txt"
# The exported graph and its weights: the model itself, as opposed to the
# encodings, which are our own work. Left out of the zip for restricted models.
WEIGHT_FILENAMES = ("model_dynamic.onnx", "model.data")


def format_eval_logs(checkpoint_dir: Path) -> str:
    """The last line of each eval_<label>.log: the metric, or the error if it failed."""
    lines = []
    for log_path in sorted(checkpoint_dir.glob("eval_*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        last = next((x.strip() for x in reversed(text.splitlines()) if x.strip()), "")
        label = log_path.stem.removeprefix("eval_")
        lines.append(f"- **{label}**: `{last or '(empty log)'}`")
    return "\n".join(lines)


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
) -> str:
    """Markdown report for the run summary: where the zip is and how to use it."""
    s3_url = f"s3://{PRIVATE_BUCKET}/{s3_key}"
    sections = [f"## Quantized `{model_id}` {precision}"]
    if eval_logs:
        sections.append(f"### Evaluation\n\n{eval_logs}")
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
            f"the public asset store, and is purged {EPHEMERAL_RETENTION_DAYS} "
            "days after upload."
        )
    else:
        sections.append(
            f"**Purged {EPHEMERAL_RETENTION_DAYS} days after upload** by the "
            "bucket's lifecycle rule."
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
    MODEL_ID, PRECISION, UNQUANTIZED_CHECKPOINT, ARGS, EVALUATE, EVAL_TASK and
    CHECKPOINT_DIR. Evaluation never fails the task: the checkpoint is the
    deliverable, and quantizing it may have taken many hours.
    """

    def __init__(self, venv: str | None) -> None:
        model_id = os.environ["QAIHM_QUANTIZE_MODEL_ID"]
        precision = os.environ["QAIHM_QUANTIZE_PRECISION"]
        unquantized = os.environ["QAIHM_QUANTIZE_UNQUANTIZED_CHECKPOINT"]
        out_dir = os.environ["QAIHM_QUANTIZE_CHECKPOINT_DIR"]
        eval_task = os.environ.get("QAIHM_QUANTIZE_EVAL_TASK", "wikitext")
        whole_split = eval_task.startswith(WHOLE_SPLIT_EVAL_PREFIX)
        num_samples = 0 if whole_split else CAPPED_EVAL_SAMPLES
        # shlex.quote keeps every value inert: none of them is parsed as shell.
        q = shlex.quote
        extra_args = shlex.join(shlex.split(os.environ.get("QAIHM_QUANTIZE_ARGS", "")))

        self.quantize_log = os.path.join(
            os.environ.get("RUNNER_TEMP", tempfile.gettempdir()), "quantize.log"
        )
        self.model_id = model_id
        self.precision = precision

        def evaluate(checkpoint: str, label: str) -> Task:
            # No pipefail: a failed metric is ignored, and the log it wrote is
            # what the upload report reads.
            return RunCommandsWithVenvTask(
                group_name=f"Evaluate {label} ({eval_task})",
                venv=venv,
                commands=f"qai-hub-models evaluate {q(model_id)} "
                f"--checkpoint {q(checkpoint)} --task {q(eval_task)} "
                f"--num-samples {num_samples} 2>&1 "
                f"| tee {q(os.path.join(out_dir, f'eval_{label}.log'))}",
                raise_on_failure=False,
            )

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
                f"set -o pipefail && qai-hub-models quantize {q(model_id)} "
                f"--precision {q(precision)} --checkpoint {q(unquantized)} "
                f"-o {q(out_dir)} {extra_args} 2>&1 | tee {q(self.quantize_log)}",
            ),
        ]
        if os.environ.get("QAIHM_QUANTIZE_EVALUATE", "").lower() == "true":
            # Quantized first: if the baseline dies, the number that matters survives.
            tasks.append(
                CompositeTask(
                    "Evaluate checkpoint",
                    [evaluate(out_dir, "quantized"), evaluate(unquantized, "float")],
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

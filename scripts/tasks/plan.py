# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import datetime
import functools
import re
import time
from collections.abc import Callable
from typing import TypeVar

from .task import Task
from .util import echo

F = TypeVar("F", bound=Callable[..., object])

ALL_TASKS: list[str] = []
PUBLIC_TASKS: list[str] = []
TASK_DEPENDENCIES: dict[str, list[str]] = {}
TASK_DESCRIPTIONS: dict[str, str] = {}
SUMMARIZERS: list[str] = []


def task(func: F) -> F:
    ALL_TASKS.append(func.__name__)
    return func


def public_task(description: str) -> Callable[[F], F]:
    def add_task(func: F) -> F:
        PUBLIC_TASKS.append(func.__name__)
        TASK_DESCRIPTIONS[func.__name__] = description
        task(func)
        return func

    return add_task


def depends(deps: list[str]) -> Callable[[F], F]:
    def add_dep(func: F) -> F:
        TASK_DEPENDENCIES[func.__name__] = deps
        return func

    return add_dep


def depends_if(
    obj: object, eq: list[tuple[object, list[str]]], default: list[str] | None = None
) -> Callable[[F], F]:
    if default is None:
        default = []
    deps = default
    for obj_ep, deps_candidate in eq:
        if obj == obj_ep:
            deps = deps_candidate
            break

    def add_dep(func: F) -> F:
        TASK_DEPENDENCIES[func.__name__] = deps
        return func

    return add_dep


def summarizer(func: F) -> F:
    SUMMARIZERS.append(func.__name__)
    return func


class Step:
    """A named Task within a Plan."""

    def __init__(self, step_id: str, task: Task) -> None:
        self._step_id = step_id
        self._task = task

    def __repr__(self) -> str:
        return self._step_id

    @property
    def step_id(self) -> str:
        return self._step_id

    @property
    def task(self) -> Task:
        return self._task


class Plan:
    """An ordered list of Tasks to execute."""

    _steps: list[Step]
    _skips: list[re.Pattern]
    _plan_duration = datetime.timedelta | None
    _step_durations: list[tuple[str, datetime.timedelta]]

    def __init__(self) -> None:
        self._steps = []
        self._skips = []
        self._plan_duration = None
        self._step_durations = []

    def add_step(self, step_id: str, task: Task) -> str:
        if self.count_step(step_id) > 10:
            raise RuntimeError(
                f"Refusing to add step '{step_id}' more than 10 times. Perhaps the planner is in an infinite loop?"
            )
        self._steps.append(Step(step_id, task))
        return step_id

    def count_step(self, step_id: str) -> int:
        step_count = 0
        for s in self._steps:
            if s.step_id == step_id:
                step_count += 1
        return step_count

    def for_each(self, func: Callable[[str, Task], None]) -> None:
        for s in self._steps:
            func(s.step_id, s.task)

    def has_step(self, step_id: str) -> bool:
        return any(s.step_id == step_id for s in self._steps)

    def is_skipped(self, step_id: str) -> bool:
        return any(r.match(step_id) for r in self._skips)

    def print(self) -> None:
        for step in self._steps:
            step_msg = step.step_id
            if not step.task.does_work():
                step_msg += " (no-op)"
            if self.is_skipped(step.step_id):
                step_msg += " (skipped)"
            print(step_msg)

    def has_report(self) -> bool:
        """Return True if there are step durations to report."""
        return len(self._step_durations) > 0

    def print_report(self) -> None:
        """Print a report on how long steps in the plan took."""
        if not self.has_report():
            return

        step_id_lens = [len(s) for s, d in self._step_durations]
        max_step_id_len = functools.reduce(lambda a, b: max(b, a), step_id_lens)
        print(f"{'Step':^{max_step_id_len}} {'Duration':^14}")
        print(f"{'-':-^{max_step_id_len}} {'-':-^14}")
        for step_id, duration in self._step_durations:
            print(f"{step_id:<{max_step_id_len}} {duration!s:<14}")
        if self._plan_duration:
            print(f"{'-':-^{max_step_id_len}} {'-':-^14}")
            print(f"{'Total':<{max_step_id_len}} {self._plan_duration!s:<14}")

    def run(self) -> None:
        start_time = time.monotonic()

        def run_task(step_id: str, task: Task) -> None:
            if self.is_skipped(step_id):
                echo(f"Skipping {step_id}")
            else:
                step_start_time = time.monotonic()

                caught: Exception | None = None
                try:
                    result = task.run()
                except Exception as ex:
                    caught = ex
                    result = False
                step_end_time = time.monotonic()
                if task.does_work():
                    self._step_durations.append(
                        (
                            step_id,
                            datetime.timedelta(seconds=step_end_time - step_start_time),
                        )
                    )

                if not result:
                    raise caught or ValueError(
                        f"Task {step_id} failed (returned {result})"
                    )

            return result

        try:
            self.for_each(run_task)
        finally:
            end_time = time.monotonic()
            self._plan_duration = datetime.timedelta(seconds=end_time - start_time)

    def skip(self, pattern: str) -> None:
        self._skips.append(re.compile(pattern))

    @property
    def steps(self) -> list[str]:
        return [s.step_id for s in self._steps]

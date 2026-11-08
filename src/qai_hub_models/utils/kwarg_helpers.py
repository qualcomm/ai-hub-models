# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import inspect
import re
import types
from collections.abc import Callable, Mapping
from typing import Any

from qai_hub_models.utils.export.result import (
    ComponentGroup,
    MultiGraphComponentGroup,
    MultiGraphGroup,
)

__all__ = [
    "filter_component_graph_name_kwargs",
    "filter_kwargs",
    "filter_per_component_kwargs",
    "filter_per_graph_kwargs",
    "get_params",
    "to_ns",
]


def get_params(func: Callable) -> dict[str, inspect.Parameter]:
    """Return the non-self parameters of the given function, ignoring variadic params (*args, **kwargs)."""
    sig = inspect.signature(func)
    return {
        name: param
        for name, param in sig.parameters.items()
        if name not in {"self", "cls"}
        and param.kind
        not in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]
    }


def filter_kwargs(func: Callable, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Given a dict with many args, pull out the ones relevant to the provided function."""
    return {k: v for k, v in kwargs.items() if k in get_params(func)}


def filter_per_graph_kwargs(
    per_graph_funcs: MultiGraphGroup[Callable],
    per_graph_kwargs: MultiGraphGroup[dict[str, Any]] | None = None,
    global_kwargs: dict[str, Any] | None = None,
) -> MultiGraphGroup[dict[str, Any]]:
    """Returns a dict of all matching kwargs for the function on each graph."""
    out: MultiGraphGroup[dict[str, Any]] = MultiGraphGroup()

    for graph_name, graph_func in per_graph_funcs.items():
        graph_kwargs: dict[str, Any] = {}
        if global_kwargs:
            graph_kwargs.update(filter_kwargs(graph_func, global_kwargs))
        if per_graph_kwargs:
            graph_kwargs.update(per_graph_kwargs.get(graph_name, {}))
        if graph_kwargs:
            out[graph_name] = graph_kwargs

    return out


def filter_per_component_kwargs(
    per_component_funcs: ComponentGroup[Callable],
    per_component_kwargs: ComponentGroup[dict[str, Any]] | None,
    global_kwargs: dict[str, Any] | None,
) -> ComponentGroup[dict[str, Any]]:
    """Returns a dict of all matching kwargs for the function on each component."""
    out: ComponentGroup[dict[str, Any]] = ComponentGroup()

    for component_name, component_func in per_component_funcs.items():
        component_kwargs: dict[str, Any] = {}
        if global_kwargs:
            component_kwargs.update(filter_kwargs(component_func, global_kwargs))
        if per_component_kwargs:
            component_kwargs.update(per_component_kwargs.get(component_name, {}))
        if component_kwargs:
            out[component_name] = component_kwargs

    return out


def filter_component_graph_name_kwargs(
    per_component_graph_funcs: MultiGraphComponentGroup[Callable],
    per_component_kwargs: ComponentGroup[dict[str, Any]] | None = None,
    per_component_graph_kwargs: MultiGraphComponentGroup[dict[str, Any]] | None = None,
    global_kwargs: dict[str, Any] | None = None,
) -> MultiGraphComponentGroup[dict[str, Any]]:
    """Returns a dict of all matching kwargs for the function on each (graph + component) pair."""
    out: MultiGraphComponentGroup[dict[str, Any]] = MultiGraphComponentGroup()

    for (component_name, graph_name), func in per_component_graph_funcs.items():
        kwargs: dict[str, Any] = {}
        if global_kwargs:
            kwargs.update(filter_kwargs(func, global_kwargs))
        if per_component_kwargs:
            kwargs.update(per_component_kwargs.get(component_name, {}))
        if per_component_graph_kwargs:
            kwargs.update(
                per_component_graph_kwargs.get((component_name, graph_name), {})
            )
        if kwargs:
            out[(component_name, graph_name)] = kwargs

    return out


def cli_friendly_class_name(class_name: str) -> str:
    """CLI-friendly name derived from the python class name."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", class_name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return name.lower()


def to_ns(d: dict) -> types.SimpleNamespace:
    """Recursively convert a nested dict to SimpleNamespace for attribute access."""
    ns = types.SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, to_ns(v) if isinstance(v, dict) else v)
    return ns

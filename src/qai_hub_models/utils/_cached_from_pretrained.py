# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import gc
import inspect
from collections.abc import Callable
from typing import Any, _ProtocolMeta

import torch

from qai_hub_models.utils.envvars import CacheFromPretrainedEnvvar, IsOnCIEnvvar


def _freeze(value: Any) -> Any:
    """Hashable stand-in for `value`: lists become tuples, dicts and sets become frozensets."""
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_freeze(v) for v in value))
    if isinstance(value, dict):
        return (
            "dict",
            frozenset((_freeze(k), _freeze(v)) for k, v in value.items()),
        )
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, frozenset(_freeze(v) for v in value))
    return value


class _CallKey:
    """
    Cache key built from a call's arguments. Equality and hashing use a frozen
    copy of the arguments, so equal lists or dicts match. The original arguments
    are kept too, because `from_pretrained` must be called with them, not with
    the frozen copies.
    """

    def __init__(self, cls: type, args: tuple, kwargs: dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs
        self._frozen = (cls, _freeze(args), _freeze(kwargs))
        self._hash = hash(self._frozen)

    def is_stale(self) -> bool:
        """Whether the original arguments no longer match the snapshot taken at build time."""
        return self._frozen[1:] != (_freeze(self.args), _freeze(self.kwargs))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _CallKey) and self._frozen == other._frozen


def _caching_enabled() -> bool:
    """Caching is opt-in: on in CI, when the scorecard runs, or with the envvar."""
    return CacheFromPretrainedEnvvar.get() or IsOnCIEnvvar.get()


def _cache_by_full_signature(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Remember the result of the last call to `func`. A new call reuses it if every
    argument is equal, counting defaults, so `f()` and `f(x=<default>)` match.
    Lists, dicts and sets are compared by value. Calls with other unhashable
    arguments always call `func`. Does nothing unless caching is enabled (see
    `_caching_enabled`).
    """
    signature = inspect.signature(func)

    @functools.lru_cache(maxsize=1)
    @functools.wraps(func)
    def build(cls: type, key: _CallKey) -> tuple[_CallKey, Any]:
        # Instances built under inference_mode hold inference tensors that torch.export rejects.
        with torch.inference_mode(False):
            return key, func(cls, *key.args, **key.kwargs)

    @functools.wraps(func)
    def cached(cls: type, *args: Any, **kwargs: Any) -> Any:
        if not _caching_enabled():
            return func(cls, *args, **kwargs)
        bound = signature.bind(cls, *args, **kwargs)
        bound.apply_defaults()
        try:
            key = _CallKey(cls, bound.args[1:], bound.kwargs)
        except TypeError:
            return func(cls, *args, **kwargs)
        stored_key, result = build(cls, key)
        if stored_key.is_stale():
            # An argument was mutated after the entry was built; rebuild.
            build.cache_clear()
            stored_key, result = build(cls, key)
        return result

    return cached


class CachedFromPretrainedMeta(_ProtocolMeta):
    """
    Lets a class's own `from_pretrained` return its previous instance when called
    again with the same arguments. Calling it with different arguments replaces
    the saved instance, so each class keeps at most one.

    Only active when `_caching_enabled()`. A class and its subclasses opt out by
    setting `_cache_from_pretrained = False`.
    """

    def __init__(cls, name: str, bases: tuple, namespace: dict, **kwargs: Any) -> None:
        super().__init__(name, bases, namespace, **kwargs)

        descriptor = namespace.get("from_pretrained")
        if (
            getattr(cls, "_is_protocol", False)
            or not getattr(cls, "_cache_from_pretrained", True)
            or not isinstance(descriptor, classmethod)
        ):
            return
        func = descriptor.__func__
        if getattr(func, "__isabstractmethod__", False):
            return

        cls.from_pretrained = classmethod(_cache_by_full_signature(func))


def clear_from_pretrained_caches() -> None:
    """Drop every cached `from_pretrained` instance and free the memory."""
    for obj in gc.get_objects():
        if (
            isinstance(obj, functools._lru_cache_wrapper)
            and getattr(obj.__wrapped__, "__name__", None) == "from_pretrained"
        ):
            obj.cache_clear()
    gc.collect()

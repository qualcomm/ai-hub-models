# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Mock out third-party packages that are not installed in the test environment.

Must not import ``qai_hub_models``: ``test_export_e2e`` loads this file by path
from a ``sitecustomize.py`` before anything else is imported.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from unittest.mock import MagicMock


class DynamicMockModule(types.ModuleType):
    """Mock module that auto-creates submodules in sys.modules on access."""

    def __init__(self, name: str, *args: object, **kwargs: object) -> None:
        super().__init__(name)
        self.__path__: list[str] = []
        self.__all__: list[str] = []
        # Some mocked packages (e.g. transformers) have their __version__
        # inspected at import time. Provide a real value so version comparisons
        # don't trip the dunder guard in __getattr__ below.
        self.__version__ = "0.0.0"

    def __getattr__(self, name: str) -> object:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # CamelCase names are treated as classes (e.g. transformers'
        # GenerationMixin, PretrainedConfig). Returning a real, subclassable
        # class lets modules that subclass these at import time (e.g. the LLM
        # generators: `class Generator(GenerationMixin, torch.nn.Module)`) load
        # under the mock. Lowercase names are treated as submodules.
        if name[:1].isupper():
            klass = type(name, (), {})
            setattr(self, name, klass)
            return klass
        submodule_name = f"{self.__name__}.{name}"
        if submodule_name not in sys.modules:
            sub = DynamicMockModule(submodule_name)
            sys.modules[submodule_name] = sub
        return sys.modules[submodule_name]

    def __call__(self, *args: object, **kwargs: object) -> MagicMock:
        return MagicMock()


class MockFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that intercepts imports of specified packages."""

    def __init__(self, prefixes: list[str]) -> None:
        self._prefixes = prefixes

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        for prefix in self._prefixes:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return importlib.machinery.ModuleSpec(fullname, self)  # type: ignore[arg-type]
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> DynamicMockModule:
        return DynamicMockModule(spec.name)

    def exec_module(self, module: types.ModuleType) -> None:
        pass


def install_mock_packages(prefixes: list[str]) -> MockFinder:
    """Intercept imports of ``prefixes`` for the rest of this process."""
    finder = MockFinder(prefixes)
    sys.meta_path.insert(0, finder)
    return finder

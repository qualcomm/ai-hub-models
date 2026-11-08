# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Shared export pipelines. ``qai-hub-models export`` calls
``select_pipeline(resolve_recipe_dir(model_id))`` from
:mod:`qai_hub_models.utils.export.dispatch` to pick one of:

    - :mod:`qai_hub_models.utils.export.pipeline` — single-graph, non-precompiled
    - :mod:`qai_hub_models.utils.export.collection_pipeline` — collection models
    - :mod:`qai_hub_models.utils.export.multi_graph_collection_pipeline` — sharded LLMs
    - :mod:`qai_hub_models.utils.export.precompiled_pipeline` — precompiled

Step implementations are one file per step at this package's root
(``upload.py``, ``compile.py``, etc.). Each step file holds the regular,
collection, and (where applicable) multi-graph variants.
"""

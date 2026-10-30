# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import warnings

from qai_hub_models.models.qwen3_5_4b.model import (
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_5_4B_PreSplit,
    Qwen3_5_4B_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm.evaluate import (
    build_llm_evaluate_parser,
    llm_evaluate,
)
from qai_hub_models.models.templates.llm.model import LLM_QNN
from qai_hub_models.models.templates.qwen3_5.model import END_TOKENS
from qai_hub_models.utils.args import QAIHMArgumentParser


def build_parser() -> QAIHMArgumentParser:
    return build_llm_evaluate_parser(
        quantized_model_cls=QuantizedSplitModelWrapper,
        fp_model_cls=FPSplitModelWrapper,
        has_presplit=True,
    )


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_parser()
    if args is None:
        warnings.warn(
            "Running `python -m qai_hub_models.models.qwen3_5_4b.evaluate` is "
            "deprecated and will be removed in a future release. "
            "Use `qai-hub-models evaluate qwen3_5_4b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args = parser.parse_args()
    use_presplit = args.use_presplit
    llm_evaluate(
        quantized_model_cls=Qwen3_5_4B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=Qwen3_5_4B_PreSplit if use_presplit else FPSplitModelWrapper,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        parser=parser,
        args=args,
        end_tokens=END_TOKENS,
    )


if __name__ == "__main__":
    main()

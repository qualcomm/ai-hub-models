# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.qwen3_5_4b.model import (
    MODEL_ID,
    SUPPORTED_PRECISIONS,
    Qwen3_5_4B_PreSplit,
    Qwen3_5_4B_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm.quantize import llm_quantize

if __name__ == "__main__":
    llm_quantize(
        quantized_model_cls=Qwen3_5_4B_QuantizablePreSplit,
        fp_model_cls=Qwen3_5_4B_PreSplit,
        model_id=MODEL_ID,
        supported_precisions=SUPPORTED_PRECISIONS,
    )

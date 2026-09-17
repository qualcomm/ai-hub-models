# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models.bge_large.app import BGELargeApp
from qai_hub_models.models.bge_large.model import MODEL_ID, BGELarge
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(BGELarge)
    parser = get_on_device_demo_parser(parser, add_output_dir=False)
    parser.add_argument(
        "--text",
        nargs="+",
        default=[
            "This is a sentence about AI.",
            "Machine learning models run on mobile devices.",
            "The weather is sunny today.",
        ],
        help="Sentences to encode",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(BGELarge, MODEL_ID, args)
    app = BGELargeApp(model)

    print(f"\nEncoding {len(args.text)} sentences with {MODEL_ID}:\n")
    embeddings = app.encode(args.text)

    for i, sentence in enumerate(args.text):
        print(f'  [{i}] "{sentence}"')
        print(
            f"      shape: {tuple(embeddings[i].shape)}, norm: {embeddings[i].norm().item():.4f}"
        )

    if len(args.text) > 1:
        print("\nPairwise cosine similarities:")
        for i in range(len(args.text)):
            for j in range(i + 1, len(args.text)):
                sim = torch.nn.functional.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0),
                ).item()
                print(f"  [{i}] vs [{j}]: {sim:.4f}")


if __name__ == "__main__":
    main()

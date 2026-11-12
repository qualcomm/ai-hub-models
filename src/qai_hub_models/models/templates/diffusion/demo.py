# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import copy
import os

import torch
from PIL import Image

from qai_hub_models import (
    Precision,
    TargetRuntime,
)
from qai_hub_models.models.templates.corevision.display import (
    display_or_save_image,
    to_uint8,
)
from qai_hub_models.models.templates.diffusion.app import (
    OUT_H,
    OUT_W,
    StableDiffusionApp,
)
from qai_hub_models.models.templates.diffusion.controlnet import (
    ControlNetBase,
    ControlUnetBase,
)
from qai_hub_models.models.templates.diffusion.stable_diffusion import (
    StableDiffusionBase,
    TextEncoderBase,
    UnetBase,
    VaeDecoderBase,
)
from qai_hub_models.models.templates.diffusion.utils import make_canny
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_model_kwargs,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.evaluate.helpers import EvalMode

DEFAULT_PROMPT = "Painting - She Danced By The Light Of The Moon by Steve Henderson"


# Run Stable Diffuison end-to-end on a given prompt. The demo will output an
# AI-generated image based on the description in the prompt.
def stable_diffusion_demo(
    model_id: str,
    model_cls: type[StableDiffusionBase],
    is_test: bool = False,
    default_guidance_scale: float = 7.5,
    default_num_steps: int = 5,
    use_controlnet: bool = False,
    default_prompt: str = DEFAULT_PROMPT,
    default_image: str | None = None,
) -> None:
    """
    Run Stable Diffusion demo for generating images from text prompts.

    Parameters
    ----------
    model_id
        Identifier for the model.
    model_cls
        The StableDiffusion model class.
    is_test
        If True, run in test mode.
    default_guidance_scale
        Default guidance scale for generation.
    default_num_steps
        Default number of diffusion steps.
    use_controlnet
        Whether to use ControlNet for conditional generation.
    default_prompt
        Default text prompt for image generation.
    default_image
        Default image path (only used if use_controlnet is True).
    """
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(
        parser=parser,
        supported_eval_modes=[EvalMode.QUANTSIM, EvalMode.FP, EvalMode.ON_DEVICE],
        supported_precisions={Precision.w8a16},
        available_target_runtimes=[
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
        default_device="Snapdragon X Elite CRD",
        add_output_dir=True,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help="Prompt for stable diffusion",
    )
    if use_controlnet:
        # TODO: provide default for this
        parser.add_argument(
            "--image",
            type=str,
            default=default_image,
            help="Use canny image generated from this image as conditional image.",
        )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=default_num_steps,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random generator seed",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=default_guidance_scale,
        help="Guidance scale",
    )
    args = parser.parse_args([] if is_test else None)

    validate_on_device_demo_args(args, model_id)

    if args.eval_mode == EvalMode.ON_DEVICE:
        host_device = torch.device("cpu")
    else:
        host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    canny_image = None
    if use_controlnet:
        # TODO: load this into torch.Tensor
        canny_image_pil = Image.open(args.image)
        canny_image = make_canny(canny_image_pil, OUT_H, OUT_W).to(host_device)

    controlnet = None
    if args.eval_mode == EvalMode.FP:
        model_kwargs = dict(get_model_kwargs(model_cls, vars(args)))
        model_kwargs["host_device"] = host_device

        text_encoder_cls = model_cls.component_classes["text_encoder"]
        assert issubclass(text_encoder_cls, TextEncoderBase)
        text_encoder = text_encoder_cls.torch_from_pretrained(**model_kwargs)

        unet_cls = model_cls.component_classes["unet"]
        assert issubclass(unet_cls, (UnetBase, ControlUnetBase))
        unet = unet_cls.torch_from_pretrained(**model_kwargs)

        vae_cls = model_cls.component_classes["vae"]
        assert issubclass(vae_cls, VaeDecoderBase)
        vae_decoder = vae_cls.torch_from_pretrained(**model_kwargs)

        if use_controlnet:
            controlnet_cls = model_cls.component_classes["controlnet"]
            assert issubclass(controlnet_cls, ControlNetBase)
            controlnet = controlnet_cls.torch_from_pretrained(**model_kwargs)
    else:
        cli_args = copy.copy(args)
        cli_args.host_device = host_device
        _, components = demo_model_components_from_cli_args(
            model_cls, model_id, cli_args
        )
        if use_controlnet:
            text_encoder, unet, vae_decoder, controlnet = components  # type: ignore[assignment]
        else:
            text_encoder, unet, vae_decoder = components  # type: ignore[assignment]

    assert issubclass(model_cls, StableDiffusionBase)
    tokenizer = model_cls.make_tokenizer()
    scheduler = model_cls.make_scheduler(args.checkpoint)

    app = StableDiffusionApp(
        text_encoder=text_encoder,
        vae_decoder=vae_decoder,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        # OnDeviceModel account for channel_last already
        channel_last_latent=False,
        host_device=host_device,
        controlnet=controlnet,
    )

    image = app.generate_image(
        args.prompt,
        num_steps=args.num_steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        cond_image=canny_image,
    )

    pil_img = Image.fromarray(to_uint8(image.detach().cpu().numpy())[0])

    if args.eval_mode == EvalMode.FP:
        default_output_dir = os.path.join("build", "torch_fp32")
    elif args.eval_mode == EvalMode.ON_DEVICE:
        default_output_dir = os.path.join("build", "on_device_e2e")
    else:  # quantsim
        default_output_dir = os.path.join("build", "quantsim")
    output_dir = args.output_dir or default_output_dir
    if not is_test:
        display_or_save_image(pil_img, output_dir)

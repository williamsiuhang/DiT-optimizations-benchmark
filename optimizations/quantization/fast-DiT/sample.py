# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
import os
import json
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from torchvision.datasets import ImageFolder
from models import DiT_models
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    # 1. apply PTQ if requested
    if args.quantize:
        # 2. optional: move to bfloat16
        #model = model.to(torch.bfloat16)

        print("Quantization enabled: applying int8 dynamic quantization")
        import torchao
        from torchao.quantization.quant_api import quantize_, int8_weight_only, Int8WeightOnlyConfig
        from torchao.utils import unwrap_tensor_subclass, TORCH_VERSION_AT_LEAST_2_5

        model = torchao.autoquant(torch.compile(model, mode='max-autotune'))
        if not TORCH_VERSION_AT_LEAST_2_5:
            unwrap_tensor_subclass(model)

    w = model.blocks[0].attn.qkv.weight
    print(type(w), w.dtype)

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels you want to condition on
    human_labels = [35, 19, 53, 100, 159, 101, 157, 157]

    # Load the mapping file
    mapping_path = os.path.join(os.path.dirname(__file__), "imagenet256-map.json")

    with open(mapping_path, "r") as f:
        class_name_to_idx = json.load(f)

    # List of all folder names (integers as strings)
    folder_names = sorted([str(v) for v in class_name_to_idx.values()])

    # Build mapping: folder name -> internal model class index (0 to 199)
    folder_to_model_idx = {folder_name: idx for idx, folder_name in enumerate(folder_names)}

    # Now map human labels to model indices
    class_labels = []
    for lbl in human_labels:
        folder_name = str(lbl)
        if folder_name not in folder_to_model_idx:
            raise ValueError(f"Folder name {folder_name} not found in folder_to_model_idx")

        mapped_idx = folder_to_model_idx[folder_name]
        class_labels.append(mapped_idx)

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    os.makedirs("samples", exist_ok=True)
    save_image(samples, "samples/sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantize", action="store_true", help="Apply PTQ (dynamic quantization)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)

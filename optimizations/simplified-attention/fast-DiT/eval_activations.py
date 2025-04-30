# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Get block activations from sampling of a DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL # type: ignore
from att_models import DiT_models
import argparse
import os
import pandas as pd

def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        if args.ema:
            checkpoint = checkpoint["ema"]
        else:
            checkpoint = checkpoint["model"]
    return checkpoint

def store_activations(activations, block, t):
    def hook_fn(module, input, output):
        if block not in activations:
            activations[block] = []
        activations[block].append(output.detach())
    return hook_fn

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        att = None,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device) # type: ignore

    # Labels to condition the model with (feel free to change):
    torch.manual_seed(args.seed)
    class_labels = torch.randint(0, 200, (16,), device=device)
    class_labels = torch.cat([class_labels, class_labels], 0)

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([200] * n, device=device)
    y = torch.cat([class_labels, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    activations = {}
    hooks = {}
    for name, module in model.named_modules():
        if name.startswith("blocks.") and len(name) <= 9:
            if name not in hooks:
                hooks[name] = {}
            for t in range(args.num_sampling_steps):
                hooks[name][t] = module.register_forward_hook(store_activations(activations, name, t))

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    #samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    #samples = vae.decode(samples / 0.18215).sample # type: ignore

    # Organisation activations:
    for block in activations:
        activations[block] = torch.stack(activations[block], dim=1) #B, T, N, C
        print(f"Block: {block}, Activations shape: {activations[block].shape}")
    
    # Calculate average normalized L1 distances:
    diffs = {}
    for block in activations:
        t0 = activations[block][:, 0:args.num_sampling_steps-1, :, :].mean(dim=0)  
        t1 = activations[block][:, 1:args.num_sampling_steps, :, :].mean(dim=0)  
        block_diff = torch.abs(t0 - t1, dim=(0, 2, 3))  # T
        diffs[block] = block_diff
    
    df = pd.DataFrame(diffs)
    df.to_csv(f"{args.name}_diffs.csv", index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--name", type=str, default="sample")
    args = parser.parse_args()
    main(args)

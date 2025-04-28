# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Example usage::
# python sample_selected.py --model "DiT-S/2" --num-classes 200 --ckpt path_to_checkpoint --name samples/filename


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "external/fast-DiT")))

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse

selected_samples = [
  {"class_id": 31, "cfg_scale": 8.499075157903865, "sampling_steps": 750, "seed": 1076791978}, # pretzel
  {"class_id": 164, "cfg_scale": 9.76093649217647, "sampling_steps": 950, "seed": 2673622937}, # fireboat
  # {"class_id": 193, "cfg_scale": 8.032737443124347, "sampling_steps": 700, "seed": 2826454853}, # lakeside
  # {"class_id": 83, "cfg_scale": 9.898068555401121, "sampling_steps": 300, "seed": 1152390534}, # tow_truck
  {"class_id": 185, "cfg_scale": 7.090231738866984, "sampling_steps": 700, "seed": 1780152141}, # jacamar
  {"class_id": 149, "cfg_scale": 4.346629991682841, "sampling_steps": 150, "seed": 3519135134}, # daisy
  {"class_id": 1, "cfg_scale": 5.674450242164038, "sampling_steps": 100, "seed": 322554063}, # artichoke
  # {"class_id": 29, "cfg_scale": 7.708030233678128, "sampling_steps": 350, "seed": 1545022854}, # pomegranate
  # {"class_id": 64, "cfg_scale": 7.6388345305630105, "sampling_steps": 850, "seed": 691962240}, #strawberry
  {"class_id": 16, "cfg_scale": 8.065961455068354, "sampling_steps": 150, "seed": 3173231053}, #mushroom
  # {"class_id": 3, "cfg_scale": 6.426364107105464, "sampling_steps": 1000, "seed": 2262811645}, # lorikeet
  # {"class_id": 183, "cfg_scale": 6.2497647208281775, "sampling_steps": 100, "seed": 2360190866}, # house_finch
  # {"class_id": 128, "cfg_scale": 6.544628875180937, "sampling_steps": 350, "seed": 3213048256}, # bulbul
  # {"class_id": 46, "cfg_scale": 6.014363790858858, "sampling_steps": 200, "seed": 2232959682}, #sea_cucumber
  {"class_id": 44, "cfg_scale": 6.177363583506423, "sampling_steps": 100, "seed": 1136259584}, # sea_anemone
  {"class_id": 179, "cfg_scale": 9.128145269339797, "sampling_steps": 500, "seed": 1438490416}, # hay
]
def main(args):
  # Setup PyTorch:
  torch.set_grad_enabled(False)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Load model:
  latent_size = args.image_size // 8
  model = DiT_models[args.model](
    input_size=latent_size,
    num_classes=args.num_classes
  ).to(device)
  ckpt_path = args.ckpt
  state_dict = find_model(ckpt_path)
  model.load_state_dict(state_dict)
  model.eval()  # important!
  vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

  # Generate images for each configuration in selected_samples:
  all_samples = []
  for sample_config in selected_samples:
    class_id = sample_config["class_id"]
    cfg_scale = sample_config["cfg_scale"]
    sampling_steps = sample_config["sampling_steps"]
    seed = sample_config["seed"]

    # Set seed for reproducibility:
    torch.manual_seed(seed)

    # Create sampling noise:
    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    y = torch.tensor([class_id], device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes], device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Create diffusion process:
    diffusion = create_diffusion(str(sampling_steps))

    # Sample images:
    samples = diffusion.p_sample_loop(
      model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    decoded_samples = vae.decode(samples / 0.18215).sample
    all_samples.append(decoded_samples)

  # Combine all samples into a grid:
  all_samples = torch.cat(all_samples, dim=0)
  save_image(all_samples, args.name + ".png", nrow=8, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=200)
    # parser.add_argument("--cfg-scale", type=float, default=4.0)
    # parser.add_argument("--num-sampling-steps", type=int, default=250)
    # parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--name", type=str, default="sample")
    args = parser.parse_args()
    main(args)
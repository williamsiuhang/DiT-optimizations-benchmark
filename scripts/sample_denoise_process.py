import sys
import os

############################################
# Replace your path with the correct method:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../optimizations/base")))
############################################

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

def main(args):
  # Setup PyTorch:
  torch.manual_seed(args.seed)
  torch.set_grad_enabled(False)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Load checkpoints from the specified folder:
  checkpoint_dir = args.checkpoint_dir
  checkpoints = sorted(
      [os.path.join(checkpoint_dir, ckpt) for ckpt in os.listdir(checkpoint_dir) if ckpt.endswith(".pt")]
  )

  if not checkpoints:
    print(f"No checkpoints found in {checkpoint_dir} folder.")
    return

  # Load model:
  latent_size = args.image_size // 8
  model = DiT_models[args.model](
    input_size=latent_size,
    num_classes=args.num_classes
  ).to(device)
  diffusion = create_diffusion(str(args.num_sampling_steps))
  vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

  # Class label to condition the model with:
  class_labels = [args.classID]
  print(f"Class label: {class_labels}")

  # Create sampling noise:
  z = torch.randn(1, 4, latent_size, latent_size, device=device)
  y = torch.tensor(class_labels, device=device)

  # Setup classifier-free guidance:
  z = torch.cat([z, z], 0)
  y_null = torch.tensor([args.num_classes], device=device)
  y = torch.cat([y, y_null], 0)
  model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

  # Iterate over checkpoints and generate samples:
  samples_list = []
  for ckpt_path in checkpoints:
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    # Sample images:
    samples = diffusion.p_sample_loop(
      model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    samples_list.append(samples)

  # Concatenate all samples into a single row:
  all_samples = torch.cat(samples_list, dim=0)
  save_image(all_samples, args.name + ".png", nrow=len(checkpoints), normalize=True, value_range=(-1, 1))
  print(f"Saved samples to {args.name}.png")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
  parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
  parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
  parser.add_argument("--num-classes", type=int, default=200)
  parser.add_argument("--cfg-scale", type=float, default=4.0)
  parser.add_argument("--num-sampling-steps", type=int, default=250)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--name", type=str, default="sample_row", help="Output image name")
  parser.add_argument("--classID", type=int, required=True, help="Class ID to condition the model with")
  parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing model checkpoints")
  args = parser.parse_args()
  main(args)
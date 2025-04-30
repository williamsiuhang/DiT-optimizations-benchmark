# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT, now with knowledge distillation and enhanced terminal output.
"""
import torch
import torch.nn as nn

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator
from tqdm import tqdm
import datetime

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def format_time(seconds):
    """Format time in a human-readable format."""
    return str(datetime.timedelta(seconds=int(seconds)))


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


def print_gpu_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 * 1024 * 1024)
            cached = torch.cuda.memory_reserved(i) / (1024 * 1024 * 1024)
            print(f"GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached")


def print_terminal_separator(title=None, width=80):
    """Print a separator line in the terminal with optional title."""
    if title:
        padding = (width - len(title) - 4) // 2
        print("=" * padding + f"[ {title} ]" + "=" * padding)
    else:
        print("=" * width)


def print_training_config(args, logger, dataset_size, model_params):
    """Print a summary of the training configuration."""
    print_terminal_separator("TRAINING CONFIGURATION", 100)
    logger.info(f"Model:                   {args.model}")
    logger.info(f"Teacher model:           DiT-S/2 from {args.teacher_ckpt}")
    logger.info(f"Image size:              {args.image_size}x{args.image_size}")
    logger.info(f"Batch size:              {args.global_batch_size}")
    logger.info(f"Dataset size:            {dataset_size:,} images")
    logger.info(f"KD alpha:                {args.kd_alpha}")
    logger.info(f"Epochs:                  {args.epochs}")
    logger.info(f"Model parameters:        {model_params:,}")
    logger.info(f"VAE:                     stabilityai/sd-vae-ft-{args.vae}")
    print_terminal_separator(width=100)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model, optionally distilling from a pretrained teacher.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create student model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    # --- Knowledge distillation setup ---
    if accelerator.is_main_process:
        print_terminal_separator("LOADING TEACHER MODEL", 100)

    teacher_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher = DiT_models["DiT-S/2"](input_size=latent_size, num_classes=args.num_classes)
    # load either from a dict with key "model" or directly
    teacher.load_state_dict(teacher_ckpt.get("model", teacher_ckpt))
    teacher = teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    kd_criterion = nn.MSELoss()
    kd_alpha = args.kd_alpha

    if accelerator.is_main_process:
        logger.info(f"Teacher model loaded from {args.teacher_ckpt}")
        logger.info(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    # -------------------------------------

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")

    if accelerator.is_main_process:
        print_terminal_separator("LOADING VAE", 100)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    if accelerator.is_main_process:
        logger.info(f"VAE loaded: stabilityai/sd-vae-ft-{args.vae}")
        model_parameters = sum(p.numel() for p in model.parameters())
        logger.info(f"Student DiT Parameters: {model_parameters:,}")

    # Setup optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    if accelerator.is_main_process:
        print_terminal_separator("LOADING DATASET", 100)

    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")
        print_training_config(args, logger, len(dataset), model_parameters)

    # Resume logic...
    train_steps = 0
    if args.resume is not None:
        if accelerator.is_main_process:
            print_terminal_separator("RESUMING FROM CHECKPOINT", 100)
            logger.info(f"Resuming from checkpoint {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

        def strip_module_prefix(sd):
            return {k.replace("module.", ""): v for k, v in sd.items()}

        model.load_state_dict(strip_module_prefix(ckpt["model"]))
        ema.load_state_dict(strip_module_prefix(ckpt["ema"]))
        opt.load_state_dict(ckpt["opt"])
        train_steps = ckpt.get("train_steps", 0)
        if accelerator.is_main_process:
            logger.info(f"Resumed from step {train_steps}")

    # Prepare for training:
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()
    model, opt, loader = accelerator.prepare(model, opt, loader)

    if accelerator.is_main_process:
        # Initialize metrics tracking
        metrics = {
            "loss": 0.0,
            "diffusion_loss": 0.0,
            "kd_loss": 0.0,
            "steps_per_sec": 0.0,
            "epoch_times": [],
            "total_start_time": time(),
        }

        print_terminal_separator("STARTING TRAINING", 100)
        logger.info(f"Training for {args.epochs} epochs with {len(loader)} steps per epoch")
        logger.info(f"Distillation weight (alpha): {kd_alpha}")
        logger.info(f"Logging frequency: Every {args.log_every} steps")
        logger.info(f"Checkpoint frequency: Every {args.ckpt_every} steps")
        print_gpu_memory_usage()

    log_steps = 0
    start_time = time()

    for epoch in range(args.epochs):
        epoch_start = time()

        if accelerator.is_main_process:
            print_terminal_separator(f"EPOCH {epoch + 1}/{args.epochs}", 100)
            logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
            epoch_loader = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}",
                                disable=not accelerator.is_main_process)
        else:
            epoch_loader = loader

        # Reset running stats for each epoch
        running_loss = 0.0
        running_diffusion_loss = 0.0
        running_kd_loss = 0.0

        for x, y in epoch_loader:
            x, y = x.to(device), y.to(device)
            x, y = x.squeeze(dim=1), y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.size(0),), device=device)
            model_kwargs = dict(y=y)

            # Diffusion loss
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            diffusion_loss = loss_dict["loss"].mean()

            # Distillation loss
            with torch.no_grad():
                teacher_pred = teacher(x, t, y)
            student_pred = model(x, t, y)
            kd_loss = kd_criterion(student_pred, teacher_pred)

            # Combined loss
            loss = (1 - kd_alpha) * diffusion_loss + kd_alpha * kd_loss

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Logging
            running_loss += loss.item()
            running_diffusion_loss += diffusion_loss.item()
            running_kd_loss += kd_loss.item()
            log_steps += 1
            train_steps += 1

            # Update progress bar on main process
            if accelerator.is_main_process:
                epoch_loader.set_postfix({
                    'loss': loss.item(),
                    'diff_loss': diffusion_loss.item(),
                    'kd_loss': kd_loss.item(),
                    'steps': train_steps
                })

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time() - start_time
                steps_per_sec = log_steps / elapsed
                avg_loss = running_loss / log_steps / accelerator.num_processes
                avg_diffusion_loss = running_diffusion_loss / log_steps / accelerator.num_processes
                avg_kd_loss = running_kd_loss / log_steps / accelerator.num_processes

                if accelerator.is_main_process:
                    # Detailed log message
                    log_msg = (
                        f"[Step {train_steps:07d}] "
                        f"Loss: {avg_loss:.4f} "
                        f"(Diffusion: {avg_diffusion_loss:.4f}, KD: {avg_kd_loss:.4f}) | "
                        f"Steps/Sec: {steps_per_sec:.2f} | "
                        f"GPU Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB | "
                        f"Time elapsed: {format_time(time() - metrics['total_start_time'])}"
                    )
                    logger.info(log_msg)

                    # Update metrics
                    metrics["loss"] = avg_loss
                    metrics["diffusion_loss"] = avg_diffusion_loss
                    metrics["kd_loss"] = avg_kd_loss
                    metrics["steps_per_sec"] = steps_per_sec

                running_loss, running_diffusion_loss, running_kd_loss, log_steps = 0, 0, 0, 0
                start_time = time()

            # Checkpointing student-only
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    print_terminal_separator("SAVING CHECKPOINT", 100)
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps
                    }
                    path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, path)
                    logger.info(f"Saved student checkpoint to {path}")
                    print_gpu_memory_usage()

        # End of epoch
        epoch_time = time() - epoch_start
        if accelerator.is_main_process:
            metrics["epoch_times"].append(epoch_time)
            avg_epoch_time = sum(metrics["epoch_times"]) / len(metrics["epoch_times"])
            est_remaining = avg_epoch_time * (args.epochs - epoch - 1)

            print_terminal_separator(f"END OF EPOCH {epoch + 1}/{args.epochs}", 100)
            logger.info(f"Epoch {epoch + 1} completed in {format_time(epoch_time)}")
            logger.info(f"Average epoch time: {format_time(avg_epoch_time)}")
            logger.info(f"Estimated remaining time: {format_time(est_remaining)}")
            logger.info(f"Total time elapsed: {format_time(time() - metrics['total_start_time'])}")

    model.eval()
    if accelerator.is_main_process:
        print_terminal_separator("TRAINING COMPLETE", 100)
        logger.info("Done with distillation training!")
        logger.info(f"Total training time: {format_time(time() - metrics['total_start_time'])}")

        # Save final model
        final_checkpoint = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "train_steps": train_steps
        }
        final_path = f"{checkpoint_dir}/final.pt"
        torch.save(final_checkpoint, final_path)
        logger.info(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--teacher-ckpt", type=str, default="checkpoints/s2-0200000.pt",
                        help="Path to pretrained teacher checkpoint for distillation")
    parser.add_argument("--kd-alpha", type=float, default=0.5,
                        help="Weight for distillation loss vs. diffusion loss")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .pt checkpoint to continue training")
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.global_seed)
    np.random.seed(args.global_seed)

    main(args)
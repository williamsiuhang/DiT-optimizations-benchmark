# This file downloads the CIFAR-10 dataset,
# converts it to image files,
# and removes the original CIFAR-10 dataset after conversion.

import shutil
import os, pickle
from torchvision.datasets import CIFAR10
from PIL import Image

data_dir = "data/cifar10"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/val"
CIFAR_DIR = "data/cifar10/cifar-10-batches-py"

# Download CIFAR-10 dataset ===================================================
if not os.path.exists(data_dir):
  CIFAR10(data_dir, train=True, download=True)
  CIFAR10(data_dir, train=False, download=True)
else:
  print(f"Data already exists in {data_dir}. Skipping download.")

# Convert CIFAR-10 to image files =============================================
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
  META = pickle.load(
    open(f"{CIFAR_DIR}/batches.meta", "rb"),
    encoding="bytes"
  )
  labels = [l.decode() for l in META[b'label_names']]

  def dump_batch(batch_file, split):
      batch = pickle.load(
        open(f"{CIFAR_DIR}/{batch_file}", "rb"),
        encoding="bytes"
      )
      data, lbls = batch[b'data'], batch[b'labels']
      data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
      for i, (img, y) in enumerate(zip(data, lbls)):
          out_dir = f"data/cifar10/{split}/{labels[y]}"
          os.makedirs(out_dir, exist_ok=True)
          Image.fromarray(img).save(f"{out_dir}/{batch_file}_{i}.png")

  for i in range(1, 6):
    dump_batch(f"data_batch_{i}", split="train")  # make train folders
  dump_batch("test_batch", split="val")  # make val folder
else:
  print(f"Train and val folders already exist in {data_dir}. Skipping conversion.")


# Remove the original CIFAR-10 dataset  ==========================================
if os.path.exists(CIFAR_DIR):
  shutil.rmtree(CIFAR_DIR)  # extracted CIFAR-10 dataset directory
if os.path.exists(f"{data_dir}/cifar-10-python.tar.gz"):
  os.remove(f"{data_dir}/cifar-10-python.tar.gz")  # downloaded tar.gz file

# Reference:
# https://www.cs.toronto.edu/~kriz/cifar.html
# https://gist.github.com/aabobakr/4333ee3bc6817674c700067b657067e5
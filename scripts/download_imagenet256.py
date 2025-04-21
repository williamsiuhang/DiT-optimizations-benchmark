import os
import kagglehub

# Set KAGGLEHUB_CACHE environment variable to change the cache directory
project_dir = input("Enter the project directory of DiT-optimizations (default: ~): ").strip() or "~"
data_dir = project_dir + "/data/imagenet256"
os.environ["KAGGLEHUB_CACHE"] = project_dir + "/cache"

# Download the ImageNet 256x256 dataset ============================
def download_imagenet256(output_dir):
  os.makedirs(output_dir, exist_ok=True)
  print("Downloading ImageNet 256x256 dataset...")
  path = kagglehub.dataset_download("dimensi0n/imagenet-256")
  print("Path to dataset files:", path)

# Download and store the dataset ===============================================
if not os.path.exists(data_dir):
  download_imagenet256(data_dir)
else:
  print(f"ImageNet 256x256 dataset already exists in {data_dir}. Skipping download.")

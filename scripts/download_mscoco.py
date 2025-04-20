import os
import zipfile
import requests
from tqdm import tqdm

# This file downloads the CIFAR-10 dataset,
data_dir = "data/mscoco"
train_dir = f"{data_dir}/train2017"
val_dir = f"{data_dir}/val2017"
annotations_dir = f"{data_dir}/annotations"

# MS COCO URLs
mscoco_urls = {
  "train2017": "http://images.cocodataset.org/zips/train2017.zip",
  "val2017": "http://images.cocodataset.org/zips/val2017.zip",
  "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

# Function to download and extract files =======================================
def download_and_extract(url, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  local_zip = os.path.join(output_dir, os.path.basename(url))
  
  if not os.path.exists(local_zip):
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
      total_size = int(r.headers.get('content-length', 0))
      with open(local_zip, 'wb') as f, tqdm(
        desc=local_zip,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
      ) as bar:
        for chunk in r.iter_content(chunk_size=1024):
          f.write(chunk)
          bar.update(len(chunk))
  else:
    print(f"{local_zip} already exists. Skipping download.")
  
  print(f"Extracting {local_zip}...")
  with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall(output_dir)
  print(f"Extraction complete for {local_zip}.")

# Download and extract MS COCO dataset =========================================
if not os.path.exists(train_dir):
  download_and_extract(mscoco_urls["train2017"], data_dir)
else:
  print(f"Train images already exist in {train_dir}. Skipping download.")

if not os.path.exists(val_dir):
  download_and_extract(mscoco_urls["val2017"], data_dir)
else:
  print(f"Validation images already exist in {val_dir}. Skipping download.")

if not os.path.exists(annotations_dir):
  download_and_extract(mscoco_urls["annotations"], data_dir)
else:
  print(f"Annotations already exist in {annotations_dir}. Skipping download.")
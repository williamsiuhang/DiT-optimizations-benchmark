import os
import subprocess
from datetime import datetime
import random

ROOT = "~/scratch/DiT-optimizations-benchmark/optimizations/mixture-of-experts"
env  = os.environ.copy()
env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH','')}"

batch = datetime.now().strftime("%y%m%d%H%M%S")

def run_sampling(n):
  # log_file_path = os.path.join(os.path.dirname(__file__), "samples/sampling_log.txt")
  log_file_path = "./samples/sampling_log.txt"
  
  for i in range(n):
    cfg_scale = random.uniform(4, 10)
    num_sampling_steps = random.choice(range(100, 1050, 50))
    class_labels = ",".join(map(str, random.sample(range(1, 200), 16)))
    # class_labels = '127,67,174,53,10,125,157,113,101,84,34,86,140,139,93,80'
    seed = random.randint(0, 2**32 - 1)
    command = [
      "python", "./optimizations/mixture-of-experts/sample.py",
      "--model", "DiT-MoE-S/2-4E1A",
      "--num-classes", "200",
      "--ckpt", "results/012-DiT-MoE-S-2-4E1A/checkpoints/0200000.pt",
      "--name", f"samples/{batch}_sample{i + 1}",
      "--cfg-scale", str(cfg_scale),
      "--num-sampling-steps", str(num_sampling_steps),
      "--class-labels", class_labels,
      "--seed", str(seed),
    ]
    
    with open(log_file_path, "a") as log_file:
      log_file.write(f"Sample Name: samples/{batch}_sample{i + 1}, CFG Scale: {cfg_scale}, Sampling Steps: {num_sampling_steps}, Class Labels: {class_labels}, Seed: {seed}\n")
    
    print(f"Running iteration {i + 1} of {n}...")
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
      print(f"Error during iteration {i + 1}: {result.stderr}")
    else:
      print(f"Iteration {i + 1} completed successfully.")

if __name__ == "__main__":
  try:
    N = int(input("Enter the number of times to run the script: "))
    if N <= 0:
      print("Please enter a positive integer.")
    else:
      run_sampling(N)
  except ValueError:
    print("Invalid input. Please enter a valid integer.")
    
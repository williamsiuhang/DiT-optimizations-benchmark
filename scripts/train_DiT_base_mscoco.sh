# Run from main project folder
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "CUDA is available. Proceeding with GPU training."
  torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    DiT/train.py \
      --model DiT-S/4 \
      --data-path data/mscoco/train2017 \
      --global-batch-size 256
else
  echo "CUDA is not available. Please ensure a GPU is accessible or modify the training script for CPU support."
  exit 1
fi
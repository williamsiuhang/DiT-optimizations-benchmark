import torch
import torch.nn as nn
import argparse
import json
import sys
import os

############################################
# Replace your path with the correct method:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../base")))
#############################################

from models import DiT_models
from download import find_model

# Unstructured method of pruning: prune attention heads by L2 Norm
def prune_attention(attn, num_keep, device): 
    h = attn.num_heads
    d = attn.qkv.weight.shape[1]
    head_dim = d // h # per-head dimension
    assert num_keep > 0
  
    # Get q, k, and v weights
    qkv = attn.qkv.weight.data.to(device)
    q, k, v = qkv.chunk(3, dim=0)

    # Calculate L2 Norm score per head
    scores = []
    for i in range(h):
        q_norm = q[i*head_dim:(i+1)*head_dim].norm()
        k_norm = k[i*head_dim:(i+1)*head_dim].norm()
        v_norm = v[i*head_dim:(i+1)*head_dim].norm()
        scores.append((q_norm + k_norm + v_norm).item())

    # keep the top scoring heads
    scores_tensor = torch.tensor(scores, device=device)
    keep_inds = torch.topk(scores_tensor, num_keep).indices
    # so mask out the lowest scoring ones
    mask_inds = torch.tensor([i for i in range(h) if i not in keep_inds], device=device)

    # Prune weights for lowest scoring head(s)
    for i in mask_inds:
        start = i * head_dim
        end = (i + 1) * head_dim
        q[start:end] = 0
        k[start:end] = 0
        v[start:end] = 0    
    # Update qkv
    attn.qkv.weight = nn.Parameter(torch.cat([q, k, v], dim=0).cpu())

    # Prune biases for the lowest scoring head(s)
    qkv_bias = attn.qkv.bias.data.to(device)
    bq, bk, bv = qkv_bias.chunk(3)
    for i in mask_inds:
        start = i * head_dim
        end = (i + 1) * head_dim
        bq[start:end] = 0
        bk[start:end] = 0
        bv[start:end] = 0
    attn.qkv.bias = nn.Parameter(torch.cat([bq, bk, bv], dim=0).cpu())
    
    print("Kept ", num_keep, " attention heads")

def main(args):
    # Set GPU
    if torch.cuda.is_available():
        # Set the device to CUDA
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load model checkpoint and config
    ckpt = find_model(args.ckpt_in)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt

    model = DiT_models[args.model](
        input_size=32,
        num_classes=200
    )
    model.to(device)
    model.load_state_dict(state_dict) # load weights

    model_name = state_dict.get("name", "model")
    print("Pruning", model_name, ", ", args.model)

    for name, m in model.named_modules():
        # Prune attention layers(s)
        if hasattr(m, 'qkv'):
            prune_attention(m, args.att_num_keep, device)
    
    # Save the pruned model checkpoint
    os.makedirs("pruned_checkpoints", exist_ok=True)
    if args.ckpt_out is not None:
        out_dir = args.ckpt_out
    else: 
        out_dir = os.path.join("pruned_checkpoints", f"{model_name}-att-{args.att_num_keep}-pruned.pt")

    torch.save(model.state_dict(), out_dir)
    print(f"Wrote pruned checkpoint to ", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt-in", type=str, required=True)
    parser.add_argument("--ckpt-out", type=str, required=False)
    parser.add_argument("att_num_keep", type=int, default=5)
    args = parser.parse_args()
    main(args)

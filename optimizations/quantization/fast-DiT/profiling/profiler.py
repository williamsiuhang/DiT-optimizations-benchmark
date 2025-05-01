#!/usr/bin/env python3
import time
import json
import argparse

from torchao.quantization.quant_api import quantize_, Int8WeightOnlyConfig
from torchao.utils import unwrap_tensor_subclass, TORCH_VERSION_AT_LEAST_2_5
import torchao
import io
import torch
import pandas as pd
from fvcore.nn import FlopCountAnalysis

from prof_models import DiT_models  # your DiT model registry
torch.set_float32_matmul_precision('high')

def profile_model(model, x, t, y, device, iters=20):
    """Measure peak GPU memory and throughput (inferences/sec)."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    # Warm-up
    for _ in range(3):
        _ = model(x, t, y)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    # Timed runs
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x, t, y)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    end = time.perf_counter()
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else None
    throughput = iters / (end - start)
    return peak_mem_gb, throughput

def main():
    parser = argparse.ArgumentParser(description="Profile DiT models with optional quantization")
    parser.add_argument("-f", "--model_file", required=True,
                        help="JSON file listing models to profile")
    parser.add_argument("-o", "--output_file", required=True,
                        help="Base name for the CSV output")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply int8 weight-only PTQ via TorchAO")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.model_file, "r") as f:
        model_entries = json.load(f)["models"]

    rows = []
    for entry in model_entries:
        name  = entry.get("name", "unnamed")
        mtype = entry.get("type", "DiT-S/2")

        # 1) build & eval
        model = DiT_models[mtype](input_size=32, num_classes=200, profile=True)
        model.to(device).eval()

        # 2) quantize (if requested)
        if args.quantize:
            quantize_(model, Int8WeightOnlyConfig())
            if not TORCH_VERSION_AT_LEAST_2_5:
                unwrap_tensor_subclass(model)

        # ─── compute serialized state_dict size ───
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        qsize_mb = buf.getbuffer().nbytes / 1e6
        print(f"serialized state_dict size = {qsize_mb:.2f} MB")

        # 3) FLOP count on the raw (quantized but uncompiled) model
        x = torch.randn(1, 4, 32, 32, device=device)
        t = torch.randint(0, 500, (1,), device=device)
        y = torch.randint(0, 200, (1,), device=device)
        flops_total = FlopCountAnalysis(model, (x, t, y)).total() / 1e9

        # 4) count params & model size
        num_params    = sum(p.numel() for p in model.parameters())
        params_M      = num_params / 1e6
        total_bytes   = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_MB = total_bytes / 1e6

        # 5) now compile _and_ profile runtime & memory
        model_c = torch.compile(model, mode="max-autotune")
        peak_mem_gb, throughput = profile_model(model_c, x, t, y, device, iters=50)

        # 6) record metrics including state_dict size
        rows.append({
            "name": name,
            "type": mtype,
            "params_M": round(params_M, 3),
            "model_size_MB": round(model_size_MB, 3),
            "state_dict_size_MB": round(qsize_mb, 3),  # added field
            "flops_G": round(flops_total, 3),
            "peak_mem_GB": round(peak_mem_gb, 3) if peak_mem_gb is not None else "",
            "throughput_it_per_s": round(throughput, 2),
        })

    pd.DataFrame(rows).to_csv(f"{args.output_file}.csv", index=False)
    print(f"Wrote profiling results to {args.output_file}.csv")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import json
import argparse
from att_train import create_logger
from att_models import DiT_models
from fvcore.nn import FlopCountAnalysis
import pandas as pd


def main(args):
    # Set GPU
    if torch.cuda.is_available():
        # Set the device to CUDA
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available, using CPU.")
    
    output_dir = args.output_dir
    logger = create_logger(output_dir, "profile2504pd")
    logger.info("Profiler started.")
    
    # Model file has structure:
    # {models: [{name: <identifying name>, type: <"DiT-S/2", "S/4", etc>}, att: <attention type, "plain", "med" etc>, mediator_dim: <int, optional>]}
    with open(args.model_file, "r") as f:
        model_list = json.load(f)["models"]

    rows = []
    for m in model_list:
        name = m.get("name", "No name")
        mtype = m.get("type", "DiT-S/4")
        if mtype in ["DiT-XS/2", "DiT-XS/4"]:
            depth = 6
        else:
            depth = 12
        att = m.get("att", None)
        med = m.get("mediator_dim", 4)
        logger.info(f"Model: {name}")
        model = DiT_models[mtype](
                    input_size=32,
                    num_classes=200,
                    att = att,
                    mediator_dim=med,
                    profile=True
                    ) 
        model.to(device)
        model.eval()
        
        params_M = sum(p.numel() for p in model.parameters())/1e6
        x = torch.randn((1, 4, 32, 32)).to(device)  # Example input tensor
        t = torch.randint(0, 500, (1,)).to(device)
        y = torch.randint(0, 200, (1,)).to(device)
        flops = FlopCountAnalysis(model, (x, t, y))
        flops_counter = flops.by_module()
        total_g = flops_counter[""] / 1e9
        total_attn = sum([flops_counter["blocks." + str(i) + ".attn"] for i in range(depth)]) / 1e9
        total_mlp = sum([flops_counter["blocks." + str(i) + ".mlp"] for i in range(depth)]) / 1e9
        total_other = total_g - total_attn - total_mlp
        rows.append(pd.Series({
            "name": name,
            "type": mtype,
            "att": att,
            "mediator_dim": med,
            "params_M": round(params_M, 3),
            "flops": round(total_g, 3),
            "attn_flops": round(total_attn, 3),
            "mlp_flops": round(total_mlp, 3),
            "other_flops": round(total_other, 3)
        }))
        logger.info(f"FLOPs by module: {flops.by_module()}")
        logger.info(f"FLOPs by operator: {flops.by_operator()}")
        logger.info(f"FLOPs by module and operator: {flops.by_module_and_operator()}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/model_flops.csv", index=False)
    logger.info(f"Profiler finished. Results saved to {output_dir}/model_flops.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("-f", "--model_file", type=str)
    args = parser.parse_args()
    main(args)
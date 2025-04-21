import torch
import argparse
import os

def find_model(model_name):
    """
    Loads a model from a local path.
    """
    # Load a custom DiT checkpoint:
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if args.ema and ("ema" in checkpoint):  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint

def main(args):
    model = find_model(args.ckpt)
    print(f"EMA: {args.ema}")
    print(f"Model loaded from {args.ckpt}")
    print(f"Model: {model['model']}")
    print(f"EMA: {model['ema']}")
    print(f"Model module: {model['model'].module}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--ema", action="store_true", help="Load EMA weights if available")
    args = parser.parse_args()
    main(args)
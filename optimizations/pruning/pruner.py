import torch
import torch.nn as nn
import json
import argparse

def main(args):
    # Set GPU
    if torch.cuda.is_available():
        # Set the device to CUDA
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--input_model", type=str)
    parser.add_argument("-f", "--output_model", type=str)
    args = parser.parse_args()
    main(args)
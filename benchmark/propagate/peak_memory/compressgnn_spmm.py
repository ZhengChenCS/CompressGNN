import torch 
import os
import sys
sys.path.append("../../../loader")
sys.path.append("../../../layer")
import time

from typing import List
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import argparse
import numpy as np 
from torch import Tensor
from torch_scatter import scatter
from compressgnn_runtime import spmm_func
from CompressgnnSPMM import compress_spmm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input data")
args = parser.parse_args()

def main():
    dataset = torch.load(args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device)
    dataset = dataset.to(device)
    #Warm up
    out = compress_spmm(
            dataset.edge_index,
            dataset.x,
            )
    torch.cuda.synchronize()
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    print("Peak memory {:.4f}(MB)".format(peak_memory_bytes/(1024*1024)))

if __name__ == "__main__":
    main()

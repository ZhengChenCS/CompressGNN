import torch 
import os
import sys
sys.path.append("../../../loader")
import time
from origin_data import Data
from typing import List
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import argparse
import numpy as np 
from torch import Tensor
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input data")
args = parser.parse_args()


def spmm(A: SparseTensor, B: Tensor):
    vlist, elist, value = A.csr()
    return matmul(A, B, reduce="add")

def main():
    dataset = torch.load(args.data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device)
    dataset = dataset.to(device)
    #Warm up
    out = spmm(
            dataset.edge_index,
            dataset.x
            )
    torch.cuda.synchronize()
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    print("Peak memory {:.4f}(MB)".format(peak_memory_bytes/(1024*1024)))

if __name__ == "__main__":
    main()

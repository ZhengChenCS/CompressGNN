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
import dgl.sparse as dglsp


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input data")
args = parser.parse_args()


def spmm(A, B:Tensor):
    return dglsp.spmm(A, B)

def main():
    dataset = torch.load(args.data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlist, elist, value = dataset.edge_index.csr()
    sp_mat = dglsp.from_csr(
        indptr=vlist,
        indices=elist,
        val=value
    )
    torch.cuda.reset_peak_memory_stats(device)
    mat = dataset.x 
    mat = mat.to(device)
    sp_mat = sp_mat.to(device)
    torch.cuda.synchronize()
    out = spmm(sp_mat,mat)
    torch.cuda.synchronize()
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    print("Peak memory {:.4f}(MB)".format(peak_memory_bytes/(1024*1024)))

if __name__ == "__main__":
    main()

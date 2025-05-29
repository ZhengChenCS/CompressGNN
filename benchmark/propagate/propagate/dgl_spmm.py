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
import dgl.function as fn

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input data")
parser.add_argument('--device', type=str, default='cpu', help="device")
parser.add_argument('--run_nums', type=int, default=10, help="number of run iteration")
args = parser.parse_args()


def spmm(A, B:Tensor):
    return spmm_with_update_all(A, B)
    # return dglsp.spmm(A, B)

def spmm_with_update_all(g, x):
    g.srcdata["h"] = x
    g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    return g.dstdata["h"]

def main():
    dataset = torch.load(args.data)
    
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlist, elist, value = dataset.edge_index.csr()
    sp_mat = dglsp.from_csr(
        indptr=vlist,
        indices=elist,
        val=value
    )
    mat = dataset.x 
    mat = mat.to(device)
    sp_mat = sp_mat.to(device)

    #Warm up
    out = spmm(sp_mat,mat)

    if not args.device == 'cpu':
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    for i in range(args.run_nums):
        out = spmm(sp_mat,mat)
    if not args.device == 'cpu':
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    print("{} platform for {} run_times: {:.4f}(s)".format(args.device, args.run_nums, t_end-t_start))

if __name__ == "__main__":
    main()

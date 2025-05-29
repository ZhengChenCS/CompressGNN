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


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input data")
args = parser.parse_args()


def propagate(src: Tensor, dst: Tensor, weight:Tensor,  B:Tensor):
    edge_content = B.index_select(-2, src)
    edge_content = weight.view(-1, 1) * edge_content
    return scatter(edge_content, dst, dim=-2, dim_size=B.size()[0], reduce="add")

def main():
    dataset = torch.load(args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device)
    dataset = dataset.to(device)
    #Warm up
    out = propagate(
            dataset.edge_index[0],
            dataset.edge_index[1],
            dataset.edge_weight,
            dataset.x,
            )
    torch.cuda.synchronize()
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    print("Peak memory {:.4f}(MB)".format(peak_memory_bytes/(1024*1024)))

if __name__ == "__main__":
    main()

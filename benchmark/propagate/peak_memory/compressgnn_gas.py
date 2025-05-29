import torch 
import os
import sys
sys.path.append("../../../loader")
import time

from typing import List
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import argparse
import numpy as np 
from torch import Tensor
from torch_scatter import scatter

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input data")
args = parser.parse_args()


def compress_propagate(edge_index: List[Tensor], 
                        weight:Tensor, B:Tensor, vertex_cnt:int, rule_cnt:int):
    out = torch.zeros(
        (vertex_cnt + rule_cnt, B.size()[1]), dtype=B.dtype, device=B.device
    )
    step = len(edge_index)
    for i in range(step):
        if i == 0:
            src = B.index_select(-2, edge_index[i][0])
        else:
            src = out.index_select(-2, edge_index[i][0])
        src = weight[i].view(-1, 1) * src
        out = scatter(src, edge_index[i][1], dim=-2, dim_size=vertex_cnt+rule_cnt, reduce="add", out=out)
    return out[:vertex_cnt]

def main():
    dataset = torch.load(args.data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.reset_peak_memory_stats(device)
    dataset = dataset.to(device)
    #Warm up
    out = compress_propagate(
            dataset.edge_index,
            dataset.edge_weight,
            dataset.x,
            dataset.vertex_cnt,
            dataset.rule_cnt
            )
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    
    print("Peak memory {:.4f}(MB)".format(peak_memory_bytes/(1024*1024)))

if __name__ == "__main__":
    main()

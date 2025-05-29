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
parser.add_argument('--device', type=str, default='cpu', help="device")
parser.add_argument('--run_nums', type=int, default=10, help="number of run iteration")
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
    
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = dataset.to(device)

    #Warm up
    out = compress_propagate(
            dataset.edge_index,
            dataset.edge_weight,
            dataset.x,
            dataset.vertex_cnt,
            dataset.rule_cnt
            )
    if not args.device == 'cpu':
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    for i in range(args.run_nums):
        out = compress_propagate(
            dataset.edge_index,
            dataset.edge_weight,
            dataset.x,
            dataset.vertex_cnt,
            dataset.rule_cnt
            )
    if not args.device == 'cpu':
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    print("{} platform for {} run_times: {:.4f}(s)".format(args.device, args.run_nums, t_end-t_start))

if __name__ == "__main__":
    main()

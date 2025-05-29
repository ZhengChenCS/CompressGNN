import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader
import math
from torch_geometric.loader import DataLoader
import shutil
import logging
import time
import argparse
import numpy as np
import sys
from tqdm import tqdm
sys.path.append("../../layer/")
sys.path.append("../../loader/")
sys.path.append("../../model/dgl")
from CompressgnnData import CompressgnnData
# from sgc_loader import SGCData, SGCDataLoader
from gcn import GCN
import dgl

path = "/mnt/disk1/GNNDataset/Reddit/origin/data_csr.pt"
dataset = torch.load(path)
graph = dataset.edge_index

src, dst, _ = graph.coo()
print(src)
print(dst)
g = dgl.graph((src, dst))
print(g)

import torch
import sys
sys.path.append('../../layer')
import PUPropagate
from PUPropagate import PUPropagate
from torch import Tensor
import torch.nn.functional as F
from metric import cosine_distance
from metric import batch_distance
from typing import Dict, List, Optional, Tuple, Union
import time


class SGC(torch.nn.Module):
    def __init__(self, in_features, out_features, K):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features)
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.P = PUPropagate(cached=True, K=K)
        self.p_time = 0
    
    def reset_cache(self):
        self.P.reset_cache()
        self.p_time = 0
        
    def propagate(self, x, edge_index):
        stime = time.perf_counter()
        x = self.P(x, edge_index)
        etime = time.perf_counter()
        self.p_time += etime-stime
        return x

    def transform(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight=None) -> Tensor:
        x = self.propagate(x, edge_index)
        x = self.transform(x)
        return x
    
    def inference(self, x: Tensor, edge_index: Tensor, edge_weight=None) -> Tensor:
        x = self.propagate(x, edge_index)
        x = self.transform(x)
        return x

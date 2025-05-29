import torch
import sys
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import time
import dgl.sparse as dglsp
from torch_sparse import SparseTensor


class Propagate(torch.nn.Module):
    def __init__(self, K: int=-1, cached: bool=False):
        super().__init__()
        self.K = K 
        self.cached = cached
        self._cached_x = None
    
    def reset_parameters(self):
        self._cached_x = None

    def forward(self, x: Tensor, graph: SparseTensor):
        cache = self._cached_x
        if cache is None:
            vlist, elist, value = graph.csr()
            sp_mat = dglsp.from_csr(
                indptr=vlist,
                indices=elist,
                val=value
            )
            for k in range(self.K):
                x = dglsp.spmm(sp_mat, x)
            if self.cached:
                self._cached_x = x
            return x
        else:
            x = cache.detach()
            return x

class SGC(torch.nn.Module):
    def __init__(self, in_features, out_features, K):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features)
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.P = Propagate(cached=True, K=K)
        self.p_time = 0
    
    def reset_cache(self):
        self.P.reset_parameters()
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

from typing import Optional, Tuple, List
import sys
import numpy as np

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul
from torch_sparse import sum as sparsesum
from type import Adj, OptTensor

from CompressgnnPropagation import Propagate


class KPropagate(Propagate):

    def __init__(self, spmm_impl: str = "inplace", **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index, edge_weight=None, vertex_cnt=None, rule_cnt=None) -> Tensor:
        """"""
        out = self.compressgnn_propagate(edge_index=edge_index, x=x, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt,
                                      size=None)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


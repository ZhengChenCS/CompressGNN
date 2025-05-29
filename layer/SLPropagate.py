from typing import Optional, Tuple
import sys
import numpy as np

import torch
from torch import Tensor

from CompressgnnPropagation import Propagate
from type import Adj, OptTensor
from torch_sparse import matmul, SparseTensor


class SLPropagate(Propagate):

    def __init__(self, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x_j: Tensor) -> Tensor:
        return matmul(adj_t, x_j, reduce=self.aggr)

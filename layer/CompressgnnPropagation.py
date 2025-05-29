import inspect
import os
import os.path as osp
import re
from collections import OrderedDict
from inspect import Parameter
from itertools import chain
from typing import Callable, List, Optional, Set, Union, get_type_hints
from uuid import uuid1

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from torch_scatter import gather_csr, scatter, segment_csr
from torch_sparse import SparseTensor
from CompressgnnSPMM import Compressgnn_SPMM
from type import Adj, Size

import sys
sys.path.append("../loader/")
from compress_graph import CompressGraph

AGGRS = {'add', 'sum', 'mean', 'min', 'max', 'mul'}


class Propagate(torch.nn.Module):
    def __init__(
            self, aggr: Optional[Union[str, List[str]]] = 'add', node_dim: int = -2):
        super().__init__()
        if aggr is None or isinstance(aggr, str):
            assert aggr is None or aggr in AGGRS
            self.aggr: Optional[str] = aggr
            self.aggrs: List[str] = []
        else:
            raise ValueError(
                f"Only strings are vaild"
                f"aggregation schemes(got '{type(aggr)}'")
        self.node_dim = node_dim
        self.rule_out = None
        self.out = None
        self.SPMM = Compressgnn_SPMM()

    def __lift__(self, src: Tensor, edge_index, dim):
        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return src.index_select(self.node_dim, index)
        elif isinstance(edge_index, SparseTensor):
            return src

    def propagate(self, edge_index: Adj, x: Tensor,
                  edge_weight: Optional[Tensor] = None, size: Size = None):
        out = self.__lift__(x, edge_index, 0)
        if isinstance(edge_index, SparseTensor):
            out = self.message_and_aggregate(edge_index, x)
            out = self.update(out)
        elif isinstance(edge_index, Tensor):
            out = self.message(out, edge_weight)
            out = self.aggregate(out, edge_index[1], x.size()[0])
            out = self.update(out)
        return out

    def compressgnn_propagate(self, x: Tensor, edge_index,  edge_weight: Optional[List[Tensor]]
                           = None, size: Size = None, vertex_cnt: int = None, rule_cnt: int = None):
        if isinstance(edge_index, CompressGraph):
            out = self.SPMM(
                edge_index=edge_index, 
                x_j=x)
        elif isinstance(edge_index[0], Tensor):
            out = torch.zeros(
                (vertex_cnt + rule_cnt,
                 x.size()[1]),
                dtype=x.dtype,
                device=x.device)
            step = len(edge_index)
            for i in range(step):
                if i == 0:
                    src = self.__lift__(x, edge_index[i], 0)
                else:
                    src = self.__lift__(out, edge_index[i], 0)
                src = self.message(src, edge_weight[i])
                out = self.aggregate(
                    inputs=src, index=edge_index[i][1], dim_size=vertex_cnt + rule_cnt, out=out)
            out = out[:vertex_cnt]
        out = self.update(out)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: int, ptr: Optional[Tensor] = None, out: Optional[Tensor] = None):
        if ptr is not None:
            pass
        else:
            return scatter(inputs, index, dim=self.node_dim,
                           dim_size=dim_size, reduce=self.aggr, out=out)

    def update(self, out: Tensor) -> Tensor:
        return out

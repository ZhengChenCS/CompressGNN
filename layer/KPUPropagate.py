from typing import Optional

from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.typing import Adj, OptTensor
from CompressgnnPropagation import Propagate


class KPUPropagate(Propagate):
    _cached_x: Optional[Tensor]

    def __init__(self, K: int = 1,
                 cached: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.cached = cached

        self._cached_x = None
        self.reset_cache()

    def reset_cache(self):
        self._cached_x = None

    def forward(self, x: Tensor, edge_index, edge_weight=None, vertex_cnt=None, rule_cnt=None) -> Tensor:
        """"""
        cache = self._cached_x
        if cache is None:
            for k in range(self.K):
                x = self.compressgnn_propagate(x=x, edge_index=edge_index,  edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt,
                                              size=None)
            if self.cached:
                self._cached_x = x
            return x
        else:
            x = cache.detach()
            return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')

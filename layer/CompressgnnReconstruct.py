import torch
from torch import Tensor
import time 


class Compressgnn_Reconstruct(torch.nn.Module):
    def __init__(self, node_dim: int = -2):
        super(Compressgnn_Reconstruct, self).__init__()
        self.node_dim = node_dim

    def forward(self, src: Tensor, index: Tensor) -> Tensor:
        x =  src.index_select(self.node_dim, index)
        return x

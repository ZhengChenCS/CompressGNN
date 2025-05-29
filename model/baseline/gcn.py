import torch
import sys
sys.path.append('../../layer')
import SLPropagate
from SLPropagate import SLPropagate
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from metric import cosine_distance
from type import OptTensor


class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        self.P1 = SLPropagate()
        self.P2 = SLPropagate()
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight:OptTensor = None) -> Tensor:
        x = self.P1(x, edge_index, edge_weight)
        x = self.lin1(x).relu()
        x = self.P2(x, edge_index, edge_weight)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=1)  # [N, out_c]
        return x

    def inference(self, x: Tensor, edge_index: Tensor, edge_weight:OptTensor = None) -> Tensor:
        x = self.P1(x, edge_index, edge_weight)
        x = self.lin1(x).relu()
        x = self.P2(x, edge_index, edge_weight)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=1)  # [N, out_c]
        return x

import torch
import sys
sys.path.append('../../layer')
import SLPropagate
from SLPropagate import SLPropagate
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from metric import cosine_distance


class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features, K):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features, bias=False)
        self.P1 = SLPropagate(add_self_loops=True, normalize=True)
        self.P2 = SLPropagate(add_self_loops=True, normalize=True)
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.K = K

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for k in range(self.K):
            x = self.lin1(x)
            x = self.P1(x, edge_index)
        print(cosine_distance(x))
        x = self.lin2(x)
        x = self.P2(x, edge_index)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return x

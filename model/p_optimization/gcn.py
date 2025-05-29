import torch
import sys
sys.path.append('../../layer')
from KPropagate import KPropagate
from torch import Tensor
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features)
        self.P1 = KPropagate(add_self_loops=False, normalize=False)
        self.P2 = KPropagate(add_self_loops=False, normalize=False)
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor, vertex_cnt:int, rule_cnt:int) -> Tensor:
        x = self.lin1(x)
        x = self.P1(x=x, edge_index=edge_index, edge_mask=edge_mask, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt).relu()
        x = self.lin2(x)
        x = self.P2(x=x, edge_index=edge_index, edge_mask=edge_mask, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return x

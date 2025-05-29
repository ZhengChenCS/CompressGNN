import torch
import sys
sys.path.append('../../layer')
from KPUPropagate import KPUPropagate
from torch import Tensor
import torch.nn.functional as F

class SGC(torch.nn.Module):
    def __init__(self, in_features, out_features, K):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.P = KPUPropagate(cached=True, K=K, add_self_loops=False, normalize=False)

    def reset_cache(self):
        self.P.reset_parameters()
        
    def propagate(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor, vertex_cnt:int, rule_cnt:int):
        return self.P(x=x, edge_index=edge_index, edge_mask=edge_mask, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)

    def transform(self, x):
        x = self.lin(x)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor, vertex_cnt:int, rule_cnt:int) -> Tensor:
        x = self.propagate(x=x, edge_index=edge_index, edge_mask=edge_mask, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        x = self.transform(x)
        return x

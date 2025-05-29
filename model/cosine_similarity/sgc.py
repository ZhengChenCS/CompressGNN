import torch
import sys
sys.path.append('../../layer')
import PUPropagate
from PUPropagate import PUPropagate
from torch import Tensor
import torch.nn.functional as F
from metric import cosine_distance


class SGC(torch.nn.Module):
    def __init__(self, in_features, out_features, K):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.P = PUPropagate(cached=True, K=K, normalize=True)
    
    def reset_cache(self):
        self.P.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.P(x, edge_index)
        print(cosine_distance(x))
        # out = torch.chunk(x, 5, dim=1)
        x = self.lin(x)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return x

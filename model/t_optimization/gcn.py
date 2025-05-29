import torch
import sys
sys.path.append('../../layer')
from SLPropagate import SLPropagate
from CompressgnnCluster import Compressgnn_Cluster
from CompressgnnReconstruct import Compressgnn_Reconstruct
from torch import Tensor
import torch.nn.functional as F
from metric import cosine_distance

class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features, param_H=1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features)
        self.P1 = SLPropagate()
        self.P2 = SLPropagate()
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.cluster = Compressgnn_Cluster(in_feature=in_features, param_H=param_H, training=True, cache=True)
        self.reconstruct = Compressgnn_Reconstruct()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.lin1(x)
        # x = self.P1(x, edge_index).relu()
        x = self.P1(x, edge_index)
        x, index = self.cluster(x)
        print(x.size())
        x = self.lin2(x)
        x = self.reconstruct(x, index)
        x = self.P2(x, edge_index)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return x

import torch
import sys
sys.path.append('../../layer')
from SLPropagate import SLPropagate
from CompressgnnCluster import Compressgnn_Cluster
from CompressgnnReconstruct import Compressgnn_Reconstruct
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from metric import cosine_distance
from type import OptTensor


class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features, param_H=1, **kwargs):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        self.P1 = SLPropagate()
        self.P2 = SLPropagate()
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        index_cache = kwargs.get('index_cache', False)
        self.cluster1 = Compressgnn_Cluster(in_feature=in_features, param_H=param_H, training=True, cache=True, index_cache=False)
        self.cluster2 = Compressgnn_Cluster(in_feature=in_features, param_H=param_H, training=True, cache=True, index_cache=False)
        self.reconstruct = Compressgnn_Reconstruct()
        self.x_size = []
    
    def reset_cache(self):
        self.cluster2.reset_cache()
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight:OptTensor = None) -> Tensor:
        x_size = []
        x_size.append(x.size(0))
        x = self.P1(x, edge_index, edge_weight)
        x, index1 = self.cluster1(x)
        x_size.append(x.size(0))
        x = self.lin1(x).relu()
        x = self.reconstruct(x, index1)
        x = self.P2(x, edge_index, edge_weight)
        x, index2 = self.cluster2(x)
        x_size.append(x.size(0))
        x = self.lin2(x)
        x = self.reconstruct(x, index2)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        self.x_size = x_size
        return out

    def inference(self, x: Tensor, edge_index: Tensor, edge_weight:OptTensor = None) -> Tensor:
        x = self.P1(x, edge_index, edge_weight)
        x = self.lin1(x).relu()
        x = self.P2(x, edge_index, edge_weight)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=1)  # [N, out_c]
        return x

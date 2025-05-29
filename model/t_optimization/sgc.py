import torch
import sys
sys.path.append('../../layer')
import PUPropagate
from PUPropagate import PUPropagate
from CompressgnnCluster import Compressgnn_Cluster
from CompressgnnReconstruct import Compressgnn_Reconstruct
from torch import Tensor
import torch.nn.functional as F
from metric import cosine_distance
from kmeans_pytorch import kmeans



class SGC(torch.nn.Module):
    def __init__(self, in_features, out_features, K, param_H=1):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.P = PUPropagate(cached=True, K=K, normalize=True)
        self.cluster = Compressgnn_Cluster(in_feature=in_features, param_H=param_H, training=True)
        self.reconstruct = Compressgnn_Reconstruct()
    
    def reset_cache(self):
        self.P.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.P(x, edge_index)
        x, index = self.cluster(x)
        print(x.size()[0])
        x = self.lin(x)
        x = self.reconstruct(x, index)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return x

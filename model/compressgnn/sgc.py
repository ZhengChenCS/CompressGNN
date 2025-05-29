import torch
import sys
sys.path.append('../../layer')
from torch import Tensor
import torch.nn.functional as F
from metric import cosine_distance
from metric import batch_distance
sys.path.append('../../layer')
from KPUPropagate import KPUPropagate
from PUPropagate import PUPropagate
from CompressgnnCluster import Compressgnn_Cluster
from CompressgnnReconstruct import Compressgnn_Reconstruct
import time


class SGC(torch.nn.Module):
    def __init__(self, in_features, out_features, K, param_H=1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features)
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.P = KPUPropagate(cached=True, K=K)
        # self.P = PUPropagate(cached=True, K=K)
        self.cluster = Compressgnn_Cluster(in_feature=in_features, param_H=param_H, training=True, cache=True, index_cache=False)
        self.reconstruct = Compressgnn_Reconstruct()
        self.p_time = 0

    def reset_cache(self):
        self.P.reset_cache()
        self.cluster.reset_cache()
    
    def propagate(self, x: Tensor, edge_index, edge_weight=None, vertex_cnt=None, rule_cnt=None):
        t_start = time.perf_counter()
        x = self.P(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        # x = self.P(x=x, edge_index=edge_index, edge_weight=edge_weight)
        t_end = time.perf_counter()
        self.p_time += t_end - t_start
        return x    
    
    def transform(self, x: Tensor, is_cluster: bool):
        if is_cluster == True:
            x, index = self.cluster(x)
            print(x.size())
            x = self.lin1(x).relu()
            x = self.lin2(x)
            x = self.reconstruct(x, index)
        else:
            x = self.lin1(x).relu()
            x = self.lin2(x)
        out = F.log_softmax(x, dim=1)
        return out

    def forward(self, x: Tensor, edge_index, is_cluster : bool, edge_weight=None, vertex_cnt=None, rule_cnt=None) -> Tensor:
        x = self.propagate(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        x = self.transform(x, is_cluster)
        return x
    
    def inference(self, x: Tensor, edge_index: Tensor, edge_weight=None, vertex_cnt=None, rule_cnt=None) -> Tensor:
        x = self.propagate(x, edge_index)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out

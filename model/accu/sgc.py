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
from metric import batch_distance
import time


class SGC(torch.nn.Module):
    def __init__(self, in_features, out_features, K, param_H=1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=in_features)
        self.lin2 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.P = PUPropagate(cached=True, K=K)
        self.cluster = Compressgnn_Cluster(in_feature=in_features, param_H=param_H, training=True, cache=True)
        self.reconstruct = Compressgnn_Reconstruct()
        self.propagate_time = 0
        self.cluster_time = 0
        self.reconstruct_time = 0
        self.transform_time = 0
        self.all_time = 0
    
    def reset_cache(self):
        self.P.reset_cache()
        self.cluster.reset_cache()
        self.propagate_time = 0
        self.cluster_time = 0
        self.reconstruct_time = 0
        self.all_time = 0
        self.transform_time = 0

        
    def propagate(self, x, edge_index):
        s_time = time.perf_counter()
        x = self.P(x, edge_index)
        e_time = time.perf_counter()
        self.propagate_time += e_time - s_time
        return x

    def transform(self, x):
        s_time = time.perf_counter()
        x, index = self.cluster(x)
        print(x.size())
        e_time = time.perf_counter()
        self.cluster_time += e_time - s_time
        t_start = time.perf_counter()
        x = self.lin1(x).relu()
        x = self.lin2(x)
        t_end = time.perf_counter()
        self.transform_time += (t_end-t_start)
        s_time = time.perf_counter()
        x = self.reconstruct(x, index)
        e_time = time.perf_counter()
        self.reconstruct_time += e_time - s_time
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        stime = time.perf_counter()
        x = self.propagate(x, edge_index)
        x = self.transform(x)
        etime = time.perf_counter()
        self.all_time += (etime-stime)
        return x
    
    def inference(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.propagate(x, edge_index)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out
    
    

import torch
import sys
sys.path.append('../../layer')
from KPropagate import KPropagate
from torch import Tensor
import torch.nn.functional as F
from CompressgnnCluster import Compressgnn_Cluster
from CompressgnnReconstruct import Compressgnn_Reconstruct
from metric import batch_distance
from typing import Dict, List, Optional, Tuple, Union

class GNNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, param_H=1, use_transformation=False, refresh_interval=1):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.P = KPropagate()
        self.use_transformation = use_transformation
        self.cluster = Compressgnn_Cluster(in_feature=in_features, param_H=param_H, training=True, cache=True, index_cache=False, refresh_interval=refresh_interval)
        self.reconstruct = Compressgnn_Reconstruct()
    
    def set_epoch(self, epoch, loss=None):
        self.cluster.set_epoch(epoch, loss=loss)

    def reset_cache(self):
        self.cluster.reset_cache()
    
    def forward(self, x: Tensor, edge_index, edge_weight=None, vertex_cnt:int=0, rule_cnt:int=0) -> Tensor:
        x = self.P(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        if self.use_transformation:
            reps, index = self.cluster(x)
            x = self.reconstruct(F.linear(reps, self.lin.weight, None), index, bias=self.lin.bias)
        else:
            x = self.lin(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128, param_H=1, dropout=0.3, use_transformation=False, refresh_interval=1):
        super().__init__()
        self.layer1 = GNNLayer(in_features=in_features, out_features=hidden_features, param_H=param_H, use_transformation=use_transformation, refresh_interval=refresh_interval)
        self.layer2 = GNNLayer(in_features=hidden_features, out_features=hidden_features, param_H=param_H, use_transformation=use_transformation, refresh_interval=refresh_interval)
        self.layer3 = GNNLayer(in_features=hidden_features, out_features=out_features, param_H=param_H, use_transformation=use_transformation, refresh_interval=refresh_interval)
        self.dropout = dropout
    
    def set_epoch(self, epoch, loss=None):
        for layer in (self.layer1, self.layer2, self.layer3):
            layer.set_epoch(epoch, loss=loss)

    def reset_cache(self):
        self.layer1.reset_cache()
        self.layer2.reset_cache()
        self.layer3.reset_cache()

    def forward(self, x: Tensor, edge_index, edge_weight=None, vertex_cnt:int=0, rule_cnt:int=0) -> Tensor:
        x = self.layer1(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer3(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        out = F.log_softmax(x, dim=1) 
        return out
    
    def inference(self, x: Tensor, edge_index, edge_weight=None, vertex_cnt:int=0, rule_cnt:int=0) -> Tensor:
        x = self.layer1(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        x = x.relu()
        x = self.layer2(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        x = x.relu()
        x = self.layer3(x=x, edge_index=edge_index, edge_weight=edge_weight, vertex_cnt=vertex_cnt, rule_cnt=rule_cnt)
        out = F.log_softmax(x, dim=1) 
        return out
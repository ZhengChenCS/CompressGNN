import torch
import sys
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import time
import dgl.sparse as dglsp
from torch_sparse import SparseTensor
from type import OptTensor
import dgl
from dgl import function as fn

class Propagate(torch.nn.Module):
    def __init__(self, add_self_loop=True):
        super().__init__()
        self.cached_norm = None
        self.add_self_loop = add_self_loop
    

    def gcn_norm(self, g: dgl.DGLGraph):
        if self.cached_norm is None:
            if self.add_self_loop:
                g = g.add_self_loop()
            in_degs = g.in_degrees().float().clamp(min=1)
            out_degs = g.out_degrees().float().clamp(min=1)
            norm_in = torch.pow(in_degs, -0.5)
            norm_out = torch.pow(out_degs, -0.5)
            self.cached_norm = (norm_in, norm_out)
        return self.cached_norm

    def forward(self, x: Tensor, g: dgl.DGLGraph) -> Tensor:
        norm_in, norm_out = self.gcn_norm(g)
        x = x * norm_in.unsqueeze(1)
        g.ndata['h'] = x
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        h = h * norm_out.unsqueeze(1)
        return h

class GCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # self.conv = dgl.nn.GraphConv(in_feats=in_features, 
        #                                out_feats=out_features,
        #                                norm='both',
        #                                weight=True,
        #                                bias=True,
        #                                allow_zero_in_degree=True)
        self.propagate = Propagate()
        self.lin = torch.nn.Linear(in_features, out_features)
    
    def forward(self, x, g):
        x = self.propagate(x, g)
        x = self.lin(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128, dropout=0.3):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, hidden_features)
        self.layer3 = GCNLayer(hidden_features, out_features)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: dgl.DGLGraph) -> Tensor:
        x = self.layer1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer3(x, edge_index)
        x = F.log_softmax(x, dim=1)  # [N, out_c]
        return x

    def inference(self, x: Tensor, edge_index: dgl.DGLGraph) -> Tensor:
        x = self.layer1(x, edge_index).relu()
        x = self.layer2(x, edge_index).relu()
        x = self.layer3(x, edge_index)
        x = F.log_softmax(x, dim=1)  # [N, out_c]
        return x




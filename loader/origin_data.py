import numpy as np
import torch
from torch import Tensor
import copy
from compressgnn_offline import add_self_loop_csr, add_self_loop_coo
from compressgnn_offline import gcn_norm_coo, get_norm_degree, gcn_norm_csr
from torch_sparse import SparseTensor


class Data():
    def __init__(self, x, edge_index, y,
                 train_mask, vaild_mask, test_mask,
                 add_self_loop: bool = True, normalize: bool = True,
                 graph_type: str="coo"):
        x = x.astype(np.float32)
        self.x = torch.as_tensor(x)
        self.y = torch.as_tensor(np.array(y))
        self.train_mask = torch.as_tensor(train_mask)
        self.vaild_mask = torch.as_tensor(vaild_mask)
        self.test_mask = torch.as_tensor(test_mask)
        self.vertex_cnt = self.x.size()[0]
        self.feature_dim = self.x.size()[1]
        self.graph_type = graph_type

        if self.graph_type == "coo":
            if add_self_loop:
                src = copy.deepcopy(edge_index[0])
                dst = copy.deepcopy(edge_index[1])
                src, dst = add_self_loop_coo(
                    src, dst, self.vertex_cnt)
                edge_index = np.vstack((src, dst))
            if normalize:
                norm_degree = get_norm_degree(edge_index[1], self.vertex_cnt)
                edge_weight = gcn_norm_coo(
                    edge_index[0],
                    edge_index[1],
                    norm_degree,
                    self.vertex_cnt)
            else:
                edge_weight = np.ones(edge_index.shape[1], dtype=float)
            self.edge_index = edge_index
            self.edge_cnt = self.edge_index.shape[1]
            self.edge_index = self.edge_index.astype(int)
            self.edge_index = torch.as_tensor(self.edge_index, dtype=torch.long)
            self.edge_weight = torch.as_tensor(edge_weight)
        elif self.graph_type == "csr":
            if add_self_loop:
                vlist, elist = add_self_loop_csr(edge_index[0], edge_index[1])
                edge_index = (vlist, elist)
            self.edge_cnt = edge_index[1].shape[0]
            if normalize:
                norm_degree = get_norm_degree(edge_index[1], self.vertex_cnt)
                edge_weight = gcn_norm_csr(edge_index[0], edge_index[1], norm_degree, self.vertex_cnt)
            else:
                edge_weight = np.ones(edge_index[1].shape, dtype=float)
            vlist = vlist.astype(int)
            elist = elist.astype(int)
            vlist = torch.as_tensor(vlist, dtype=torch.long)
            elist = torch.as_tensor(elist, dtype=torch.long)
            edge_weight = torch.as_tensor(edge_weight, dtype=torch.float)
            edge_index = SparseTensor(rowptr=vlist, col=elist, value=edge_weight)
            t_vlist, t_elist, t_value = edge_index.csc()
            self.edge_index = SparseTensor(rowptr=t_vlist, col=t_elist, value=t_value)
            
            

    def __repr__(self):
        return 'Data(vertex_cnt:"%s", edge_cnt:"%s", feature_dim:"%s", graph_type:"%s")' % (
            self.vertex_cnt, self.edge_cnt, self.feature_dim, self.graph_type)

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.vaild_mask = self.vaild_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        if self.graph_type == "coo":
            self.edge_index = self.edge_index.to(device)
            self.edge_weight = self.edge_weight.to(device)
        elif self.graph_type == "csr":
            self.edge_index = self.edge_index.to(device)
        return self

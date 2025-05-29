import numpy as np
import torch
from torch import Tensor
import copy
from compressgnn_offline import add_self_loop_coo, add_self_loop_csr
from compressgnn_offline import get_norm_degree, gcn_norm_coo_compress, gcn_norm_csr_compress
from compressgnn_offline import compress_csr, check_csr, depth_filter_csr
from compressgnn_offline import coo2csr, csr2coo
from compressgnn_offline import filter_csr
from compressgnn_offline import gen_mask_vertex, gen_mask_edge
from torch_sparse import SparseTensor
from compressgnn_offline import splice_csr, splice_csr_condense_row, splice_csr_condense_col, splice_csr_condense_row_and_col
from compressgnn_offline import check_csr
from compressgnn_offline import get_active_row_and_col
from compressgnn_offline import get_out_degree
from compressgnn_offline import hybird_partition
from compress_graph import CompressGraph


def compress(vlist, elist, threshold, max_depth, min_edge):
    new_vlist, new_elist, new_vertex_cnt, new_rule_cnt = compress_csr(
        vlist, elist, vlist.shape[0]-1)
    print(
        "compression ratio: {} / {} = {:.4f}".format(
            vlist.shape[0] +
            elist.shape[0],
            new_vlist.shape[0] +
            new_elist.shape[0], (vlist.shape[0] + elist.shape[0]) / (new_vlist.shape[0] + new_elist.shape[0])))
    new_vlist, new_elist, new_vertex_cnt, new_rule_cnt = filter_csr(
        new_vlist, new_elist, new_vertex_cnt, new_rule_cnt, threshold)
    print(
        "After filter compression ratio: {} / {} = {:.4f}".format(
            vlist.shape[0] +
            elist.shape[0],
            new_vlist.shape[0] +
            new_elist.shape[0], (vlist.shape[0] + elist.shape[0]) / (new_vlist.shape[0] + new_elist.shape[0])))
    new_vlist, new_elist, new_vertex_cnt, new_rule_cnt = depth_filter_csr(
        new_vlist, new_elist, new_vertex_cnt, new_rule_cnt, max_depth, min_edge)
    print(
        "After depth filter compression ratio: {} / {} = {:.4f}".format(
            vlist.shape[0] +
            elist.shape[0],
            new_vlist.shape[0] +
            new_elist.shape[0], (vlist.shape[0] + elist.shape[0]) / (new_vlist.shape[0] + new_elist.shape[0])))
    return new_vlist, new_elist, new_vertex_cnt, new_rule_cnt




class CompressgnnData():
    def __init__(self, x, edge_index, y,
                 train_mask, vaild_mask, test_mask,
                 add_self_loop: bool = True, normalize: bool = True,
                 graph_type: str = "coo",
                 max_depth: int = 3,
                 min_edge: int = 1000000,
                 threshold: int = 16,
                 homo: bool = False):
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
                edge_index = (src, dst)
            if normalize:
                norm_degree = get_norm_degree(edge_index[1], self.vertex_cnt)
            '''
            Compression
            '''
            vlist, elist = coo2csr(
                edge_index[0], edge_index[1], self.vertex_cnt)
            new_vlist, new_elist, new_vertex_cnt, new_rule_cnt = compress(
                vlist, elist, threshold, max_depth, min_edge)
            self.vertex_cnt = new_vertex_cnt
            self.rule_cnt = new_rule_cnt
            self.edge_cnt = new_elist.shape[0]
            src, dst = csr2coo(new_vlist, new_elist)
            edge_index = np.vstack((src, dst))
            '''
            normalize
            '''
            if normalize:
                edge_weight = gcn_norm_coo_compress(
                    edge_index[0],
                    edge_index[1],
                    norm_degree,
                    self.vertex_cnt,
                    self.rule_cnt)
            else:
                edge_weight = np.ones(self.edge_cnt, dtype=np.float)
            '''
            Generate multi phase tensor
            '''
            edge_mask, step = gen_mask_edge(
                edge_index[0], edge_index[1], self.vertex_cnt, self.rule_cnt)
            print("step:{}".format(step))
            self.step = step
            edge_index = edge_index.astype(np.int64)
            edge_index = torch.from_numpy(edge_index)
            edge_weight = torch.from_numpy(edge_weight)
            self.edge_index = []
            self.edge_weight = []
            for i in range(step):
                mask = edge_mask[i]
                src = edge_index[0][mask]
                dst = edge_index[1][mask]
                part_edge_index = torch.stack((src, dst), 0)
                part_edge_weight = edge_weight[mask]
                self.edge_index.append(part_edge_index)
                self.edge_weight.append(part_edge_weight)
        elif self.graph_type == "csr":
            if add_self_loop:
                vlist, elist = add_self_loop_csr(edge_index[0], edge_index[1])
                edge_index = (vlist, elist)
            if normalize:
                norm_degree = get_norm_degree(edge_index[1], self.vertex_cnt)
            '''
            Compression
            '''
            new_vlist, new_elist, vertex_cnt, rule_cnt = compress(
                edge_index[0], edge_index[1], threshold, max_depth, min_edge)
            self.vertex_cnt = vertex_cnt
            self.rule_cnt = rule_cnt
            self.edge_cnt = new_elist.shape[0]
            new_vlist = new_vlist.astype(np.int64)
            new_vlist = torch.as_tensor(new_vlist)
            new_elist = new_elist.astype(np.int64)
            new_elist = torch.as_tensor(new_elist)
            edge_index = (new_vlist, new_elist)
            '''
            Generate weighted graph
            '''
            if normalize:
                edge_weight = gcn_norm_csr_compress(
                    edge_index[0], edge_index[1], norm_degree, self.vertex_cnt, self.rule_cnt)
            else:
                edge_weight = np.ones(self.edge_cnt, dtype=np.float)
            edge_weight = torch.as_tensor(edge_weight)
            graph = SparseTensor(
                rowptr=edge_index[0],
                col=edge_index[1],
                value=edge_weight)
            '''
            Generate multi phase sparse tensor
            '''
            mask, step = gen_mask_vertex(
                edge_index[0], edge_index[1], self.vertex_cnt, self.rule_cnt, "csc")
            self.step = step
            vlist, elist, value = graph.csc()
            graph = CompressGraph(vlist, elist, value, self.vertex_cnt)
            self.edge_index = graph
        if homo:
            pass
    def __repr__(self):
        return 'CompressgnnData(vertex_cnt:"%s", rule_cnt:"%s", edge_cnt:"%s", step:"%s", graph_type:"%s")' % (
            self.vertex_cnt, self.rule_cnt, self.edge_cnt, self.step, self.graph_type)

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.vaild_mask = self.vaild_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        if self.graph_type == "coo":
            for i in range(self.step):
                self.edge_index[i] = self.edge_index[i].to(device)
                self.edge_weight[i] = self.edge_weight[i].to(device)
        elif self.graph_type == "csr":
            self.edge_index = self.edge_index.to(device)    
        return self


class CompressgnnBatchData():
    def __init__(self, path):
        self.data = torch.load(path)
        self.len = len(self.data)

    def __len__(self):
        return self.len


class CompressgnnDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data, **kwargs):
        self.data = data

        super().__init__(range(len(data)), collate_fn=self.__collate__,
                         **kwargs)

    def __collate__(self, batch):
        idx = int(batch[0])
        data = copy.copy(self.data.data[idx])
        return data
    

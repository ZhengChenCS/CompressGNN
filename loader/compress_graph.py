import torch 
import numpy as np
from torch_sparse import SparseTensor
from compressgnn_offline import hybird_partition, topo_partition, gen_degree

class CompressGraph():
    def __init__(self, vlist, elist, value, vertex_cnt):
        v2v_graph, v2r_graph, r2v_graph, r2r_graph = hybird_partition(vlist, elist, value, vertex_cnt)
        v2v_vlist = torch.from_numpy(v2v_graph[0])
        v2v_elist = torch.from_numpy(v2v_graph[1])
        v2v_value = torch.from_numpy(v2v_graph[2])
        v2r_vlist = torch.from_numpy(v2r_graph[0])
        v2r_elist = torch.from_numpy(v2r_graph[1])
        v2r_value = torch.from_numpy(v2r_graph[2])
        r2v_vlist = torch.from_numpy(r2v_graph[0])
        r2v_elist = torch.from_numpy(r2v_graph[1])
        r2v_value = torch.from_numpy(r2v_graph[2])
        r2r_list = topo_partition(r2r_graph[0], r2r_graph[1], r2r_graph[2])   
        self.v2v_graph = SparseTensor(rowptr=v2v_vlist, col=v2v_elist, value=v2v_value)
        self.v2r_graph = SparseTensor(rowptr=v2r_vlist, col=v2r_elist, value=v2r_value)
        self.r2v_graph = SparseTensor(rowptr=r2v_vlist, col=r2v_elist, value=r2v_value)
        self.step = len(r2r_list)
        self.r2r_graph = []
        r2r_vlists = []
        r2r_elists = []
        for i in range(self.step):
            r2r_vlists.append(r2r_list[i][0])
            r2r_elists.append(r2r_list[i][1])
            r2r_vlist = torch.from_numpy(r2r_list[i][0])
            r2r_elist = torch.from_numpy(r2r_list[i][1])
            r2r_value = torch.from_numpy(r2r_list[i][2])
            bi_graph = SparseTensor(rowptr=r2r_vlist, col=r2r_elist, value=r2r_value)
            self.r2r_graph.append(bi_graph)
        vertex_degree, rule_degree = gen_degree(v2v_vlist, v2v_elist,
                                                v2r_vlist, v2r_elist, 
                                                r2v_vlist, r2v_elist, 
                                                r2r_vlists, r2r_elists)
        vertex_degree = torch.from_numpy(vertex_degree)
        rule_degree = torch.from_numpy(rule_degree)
        self.vertex_degree = vertex_degree
        self.rule_degree = rule_degree
        
    
    def __repr__(self):
        return 'CompressGraph' 
    
    def to(self, device):
        self.v2v_graph = self.v2v_graph.to(device)
        self.v2r_graph = self.v2r_graph.to(device)
        self.r2v_graph = self.r2v_graph.to(device)
        for s in range(self.step):
            self.r2r_graph[s] = self.r2r_graph[s].to(device) 
        self.vertex_degree = self.vertex_degree.to(device)
        self.rule_degree = self.rule_degree.to(device)
        return self
    
















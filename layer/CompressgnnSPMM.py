import torch
from torch import Tensor
from compressgnn_runtime import spmm_func
# from kmeans_pytorch import kmeans
from torch_scatter import scatter
from torch_sparse import SparseTensor
from type import List

import sys
sys.path.append("../loader/")
from compress_graph import CompressGraph

def compress_spmm(graph, x_j):
    v2v_vlist, v2v_elist, v2v_value = graph.v2v_graph.csr()
    v2r_vlist, v2r_elist, v2r_value = graph.v2r_graph.csr()
    r2v_vlist, r2v_elist, r2v_value = graph.r2v_graph.csr()
    if not x_j.is_cuda:
        rule_out = spmm_func(r2v_vlist, r2v_elist, r2v_value, x_j, "rowbalance", None)
        for i in range(graph.step):
            part_vlist, part_elist, part_value = graph.r2r_graph[i].csr()
            rule_out = spmm_func(part_vlist, part_elist, part_value, rule_out, "rowbalance", rule_out)
        out = spmm_func(v2r_vlist, v2r_elist, v2r_value, rule_out, "rowbalance", None)
        out = spmm_func(v2v_vlist, v2v_elist, v2v_value, x_j, "rowbalance", out)
    else:
        rule_out = spmm_func(r2v_vlist, r2v_elist, r2v_value, x_j, "nnzbalance_rowcache", None)
        for i in range(graph.step):
            part_vlist, part_elist, part_value = graph.r2r_graph[i].csr()
            rule_out = spmm_func(part_vlist, part_elist, part_value, rule_out, "nnzbalance_rowcache", rule_out)
        out = spmm_func(v2r_vlist, v2r_elist, v2r_value, rule_out, "nnzbalance_rowcache", None)
        out = spmm_func(v2v_vlist, v2v_elist, v2v_value, x_j, "nnzbalance_rowcache", out)
    return out

class Compressgnn_SPMM_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, x_j):
        v2v_vlist, v2v_elist, v2v_value = graph.v2v_graph.csr()
        v2r_vlist, v2r_elist, v2r_value = graph.v2r_graph.csr()
        r2v_vlist, r2v_elist, r2v_value = graph.r2v_graph.csr()
        if not x_j.is_cuda:
            rule_out = spmm_func(r2v_vlist, r2v_elist, r2v_value, x_j, "rowbalance", None)
            for i in range(graph.step):
                part_vlist, part_elist, part_value = graph.r2r_graph[i].csr()
                rule_out = spmm_func(part_vlist, part_elist, part_value, rule_out, "rowbalance", rule_out)
            out = spmm_func(v2r_vlist, v2r_elist, v2r_value, rule_out, "rowbalance", None)
            out = spmm_func(v2v_vlist, v2v_elist, v2v_value, x_j, "rowbalance", out)
        else:
            rule_out = spmm_func(r2v_vlist, r2v_elist, r2v_value, x_j, "nnzbalance_rowcache", None)
            for i in range(graph.step):
                part_vlist, part_elist, part_value = graph.r2r_graph[i].csr()
                rule_out = spmm_func(part_vlist, part_elist, part_value, rule_out, "nnzbalance_rowcache", rule_out)
            out = spmm_func(v2r_vlist, v2r_elist, v2r_value, rule_out, "nnzbalance_rowcache", None)
            out = spmm_func(v2v_vlist, v2v_elist, v2v_value, x_j, "nnzbalance_rowcache", out)
        ctx.graph = graph
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        graph = ctx.graph
        v2v_vlist, v2v_elist, v2v_value = graph.v2v_graph.csc()
        v2r_vlist, v2r_elist, v2r_value = graph.r2v_graph.csc()
        r2v_vlist, r2v_elist, r2v_value = graph.v2r_graph.csc()
        if not grad_output.is_cuda:
            rule_out = spmm_func(r2v_vlist, r2v_elist, r2v_value, grad_output, "rowbalance", None)
            for i in range(graph.step, 0, 1):
                vlist, elist, value = graph.r2r_graph[i-1].csc()                                                      
                rult_out = spmm_func(vlist, elist, value, rule_out, "rowbalance", rule_out)
            rule_out = spmm_func(r2r_vlist, r2r_elist, r2r_value, rule_out, "rowbalance", rule_out)
            out = spmm_func(v2v_vlist, v2r_elist, v2r_value, grad_output, "rowbalance", None)
            out = spmm_func(v2r_vlist, v2v_elist, v2v_value, rule_out, "rowbalance", out)
        else:
            rule_out = spmm_func(r2v_vlist, r2v_elist, r2v_value, grad_output, "nnzbalance_rowcache", None)
            for i in range(graph.step, 0, 1):
                vlist, elist, value = graph.r2r_graph[i-1].csc()                                                      
                rult_out = spmm_func(vlist, elist, value, rule_out, "nnzbalance_rowcache", rule_out)
            out = spmm_func(v2v_vlist, v2v_elist, v2v_value, grad_output, "nnzbalance_rowcache", None)
            out = spmm_func(v2r_vlist, v2r_elist, v2r_value, rule_out, "nnzbalance_rowcache", out)
        return None, out

class Compressgnn_SPMM(torch.nn.Module):
    def __init__(self):
        super(Compressgnn_SPMM, self).__init__()

    def forward(self, edge_index: CompressGraph, x_j: Tensor) -> Tensor:
        out = Compressgnn_SPMM_Function.apply(
            edge_index, x_j)
        return out

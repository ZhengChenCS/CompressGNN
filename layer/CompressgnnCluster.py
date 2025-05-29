import torch
from torch import Tensor
from compressgnn_runtime import cluster_forward
# from kmeans_pytorch import kmeans
from torch_scatter import scatter
import time


class Compressgnn_Cluster_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, random_vectors, param_H, is_training):
        ret = cluster_forward(input, random_vectors, param_H)
        index = ret[0].long()
        # unique_index, unique_indices = torch.unique(tensor, return_inverse=True)
        active_bucket = ret[1]
        # outputs = scatter(
        #     input,
        #     index,
        #     dim=-2,
        #     dim_size=active_bucket + 1,
        #     reduce='mean')
        outputs = scatter(
            input,
            index,
            dim=-2,
            dim_size=active_bucket + 1,
            reduce='max')
        ctx.save_for_backward(index)
        return outputs, index, active_bucket

    @staticmethod
    def backward(ctx, gradOutput, _, __):
        index, = ctx.saved_tensors
        return gradOutput.index_select(-2, index), None, None, None


class Compressgnn_Cluster(torch.nn.Module):
    def __init__(self, in_feature: int, param_H: int, training=True,
                 device=None, dtype=None, cache=True, index_cache=False):
        super(Compressgnn_Cluster, self).__init__()
        factory_kwags = {'device': device, 'dtype': dtype}
        self.param_H = param_H
        self.in_feature = in_feature
        self.random_vectors = torch.nn.Parameter(
            torch.randn((in_feature, param_H), **factory_kwags)
        )
        self.is_training = training
        self.cache = cache
        self.output = None
        self.vertex_index = None
        self.active_bucket = None
        self.index_cache = index_cache

    def reset_cache(self):
        self.output = None
        self.vertex_index = None
        self.active_bucket = None

    def forward(self, input: Tensor) -> Tensor:
        if self.cache is True:
            output, vertex_index = None, None
            if self.output is not None:
                output = self.output.detach()
            if self.vertex_index is not None:
                vertex_index = self.vertex_index.detach()
            if self.index_cache is False:
                if output is None:
                    if not self.training:
                        output, _, _ = Compressgnn_Cluster_Function.apply(
                            input, self.random_vectors, self.param_H, self.is_training)
                        self.output = output
                        return output
                    else:
                        output, vertex_index, _ = Compressgnn_Cluster_Function.apply(
                            input, self.random_vectors, self.param_H, self.is_training)
                        self.output = output
                        self.vertex_index = vertex_index
                        return output, vertex_index
                else:
                    if not self.is_training:
                        output = self.output.detach()
                        return output
                    else:
                        output = self.output.detach()
                        vertex_index = self.vertex_index.detach()
                        return output, vertex_index
            else:
                if vertex_index is None:
                    if not self.is_training:
                        output, vertex_index, active_bucket = Compressgnn_Cluster_Function.apply(
                            input, self.random_vectors, self.param_H, self.is_training)
                        self.vertex_index = vertex_index
                        self.active_bucket = active_bucket
                        return output
                    else:
                        output, vertex_index, active_bucket = Compressgnn_Cluster_Function.apply(
                            input, self.random_vectors, self.param_H, self.is_training)
                        self.vertex_index = vertex_index
                        self.active_bucket = active_bucket
                        return output, vertex_index
                else:
                    if not self.is_training:
                        vertex_index = self.vertex_index.detach()
                        active_bucket = self.active_bucket.detach()
                        output = scatter(
                            input, index, dim=-2, dim_size=active_bucket + 1, reduce='mean')
                        return output
                    else:
                        vertex_index = self.vertex_index.detach()
                        active_bucket = self.active_bucket
                        output = scatter(
                            input, vertex_index, dim=-2, dim_size=active_bucket + 1, reduce='mean')
                        return output, vertex_index
        else:
            if not self.is_training:
                output, _, _ = Compressgnn_Cluster_Function.apply(
                    input, self.random_vectors, self.param_H, self.training)
                return output
            else:
                output, vertex_index, _ = Compressgnn_Cluster_Function.apply(
                    input, self.random_vectors, self.param_H, self.training)
                return output, vertex_index

"""Reconstruct node features, optionally fusing shared bias and ReLU.

Dropout remains at node level outside reconstruction, preserving independent
node masks. Consecutive shared transforms may run on representatives first.
"""
import torch
from torch.autograd.function import once_differentiable
import compressgnn_runtime as _native


class _Reconstruct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index, bias, relu):
        output = _native.cluster_reconstruct(src.contiguous(), index.contiguous(),
                                              None if bias is None else bias.contiguous(), relu)
        ctx.save_for_backward(index, output)
        ctx.groups, ctx.relu, ctx.has_bias = src.size(0), relu, bias is not None
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        index, output = ctx.saved_tensors
        reps_grad = _native.cluster_reconstruct_backward(grad.contiguous(), index.contiguous(),
                                                          output, ctx.groups, ctx.relu)
        return reps_grad, None, reps_grad.sum(0) if ctx.has_bias else None, None


class Compressgnn_Reconstruct(torch.nn.Module):
    def __init__(self, node_dim=-2):
        super().__init__()
        self.node_dim = node_dim

    def forward(self, src, index, bias=None, activation=None):
        if activation not in (None, 'relu'):
            raise ValueError('Supported activations: None, relu')
        if src.is_cuda and src.dim() == 2 and self.node_dim in (0, -2) and (bias is not None or activation):
            return _Reconstruct.apply(src, index, bias, activation == 'relu')
        output = src.index_select(self.node_dim, index)
        if bias is not None:
            output = output + bias
        return output.relu() if activation == 'relu' else output

"""Mean representatives with refreshable LSH assignments (multi-level reuse).

Only indices and cluster counts are cached. Representatives are recomputed from
current features on every call so gradients reach the current upstream model.
Call set_epoch with zero-based epochs when refresh_interval > 1. Train and eval
keep separate assignments; cache_key must identify a graph and node ordering.
"""
import math
import operator
import torch
from torch.autograd.function import once_differentiable
from torch_scatter import scatter_mean
import compressgnn_runtime as _native


def _mean_forward(x, index, counts):
    if x.is_cuda:
        return _native.cluster_mean(x.contiguous(), index, counts)
    return scatter_mean(x, index, dim=0, dim_size=counts.numel())


def _mean_backward(grad, index, counts):
    if grad.is_cuda:
        return _native.cluster_mean_backward(grad.contiguous(), index, counts)
    return grad.index_select(0, index) / counts.index_select(0, index).clamp_min(1).to(grad.dtype)[:, None]


class _Mean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index, counts):
        ctx.save_for_backward(index, counts)
        return _mean_forward(x, index, counts)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        index, counts = ctx.saved_tensors
        return _mean_backward(grad, index, counts), None, None


def _assign(input, planes, bits, workspace=None):
    with torch.no_grad():
        if input.is_cuda:
            if workspace is None:
                workspace = _native.cluster_workspace(input.contiguous(), bits)
            return _native.cluster_assign(input.contiguous(), planes.contiguous(), bits, workspace)
        index, active = _native.cluster_forward(input.contiguous(), planes.contiguous(), bits)
        index = index.long()
        counts = torch.bincount(index, minlength=active + 1).to(torch.int32)
        return index, counts, active


class Compressgnn_Cluster_Function(torch.autograd.Function):
    """Compatibility entry point; fixed-assignment mean gradients, no cache."""
    @staticmethod
    def forward(ctx, input, random_vectors, param_H, is_training):
        index, counts, active = _assign(input, random_vectors, param_H)
        ctx.save_for_backward(index, counts)
        ctx.mark_non_differentiable(index)
        return _mean_forward(input, index, counts), index, active

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, _index_grad, _active_grad):
        index, counts = ctx.saved_tensors
        return _mean_backward(grad, index, counts), None, None, None


class Compressgnn_Cluster(torch.nn.Module):
    """Reusable LSH indices; q is measured in epochs, never in forward calls.

    refresh_interval=1 is the reference policy. For q>1, call set_epoch(e), or
    pass epoch=e to forward. A cache-disabled call always refreshes. The legacy
    index_cache argument is accepted; both values now cache indices only, never
    detached features. Constructor training=False retains the old tensor-only
    return contract; training=True returns (representatives, index) in both
    train() and eval(). Module mode controls separate cache namespaces.
    """
    def __init__(self, in_feature, param_H, training=True, device=None, dtype=None,
                 cache=True, index_cache=False, refresh_interval=1,
                 refresh_policy='fixed', q_min=1, q_max=10, tau_loss=None):
        super().__init__()
        self.in_feature = operator.index(in_feature)
        self.param_H = operator.index(param_H)
        if self.in_feature < 0 or not 0 <= self.param_H <= 30:
            raise ValueError('Expected nonnegative feature width and 0 <= param_H <= 30')
        self.refresh_interval = operator.index(refresh_interval)
        self.q_min, self.q_max = operator.index(q_min), operator.index(q_max)
        if self.refresh_interval < 1 or not 1 <= self.q_min <= self.q_max:
            raise ValueError('Refresh intervals must be positive integers, q_min <= q_max')
        if refresh_policy not in ('fixed', 'loss'):
            raise ValueError('refresh_policy must be fixed or loss')
        if refresh_policy == 'loss' and (tau_loss is None or not math.isfinite(tau_loss) or tau_loss <= 0):
            raise ValueError('Loss policy requires a finite positive tau_loss')
        self.refresh_policy, self.tau_loss = refresh_policy, tau_loss
        self.cache, self.index_cache = bool(cache), bool(index_cache)
        self.is_training = bool(training)  # Legacy return contract, not cache mode.
        self.register_buffer('random_vectors', torch.randn(self.in_feature, self.param_H,
                                                          device=device, dtype=dtype))
        self._epoch = None
        self._previous_loss = None
        self._last_loss_epoch = None
        self.current_interval = self.refresh_interval if refresh_policy == 'fixed' else self.q_min
        self._states, self._workspaces = {}, {}
        self.vertex_index = self.active_bucket = None
        self.last_refreshed = False
        self.refresh_count = 0
        self.train(training)

    def set_epoch(self, epoch, loss=None):
        """Set zero-based epoch; loss is the previous completed training loss."""
        epoch = operator.index(epoch)
        if epoch < 0:
            raise ValueError('epoch must be nonnegative')
        if self._epoch is not None and epoch < self._epoch:
            self.reset_cache()
            self._previous_loss = self._last_loss_epoch = None
            self.current_interval = self.refresh_interval if self.refresh_policy == 'fixed' else self.q_min
        self._epoch = epoch
        if loss is not None and self.refresh_policy == 'loss' and epoch != self._last_loss_epoch:
            if torch.is_tensor(loss):
                raise TypeError('Pass a host loss value; do not hide a GPU synchronization in set_epoch')
            loss = float(loss)
            if not math.isfinite(loss):
                raise ValueError('loss must be finite')
            if self._previous_loss is not None:
                raw = self.q_min + (self.q_max - self.q_min) * math.exp(-abs(loss - self._previous_loss) / self.tau_loss)
                self.current_interval = min(self.q_max, max(self.q_min, int(math.floor(raw + .5))))
            self._previous_loss, self._last_loss_epoch = loss, epoch
        return self

    def reset_cache(self):
        self._states.clear()
        self.vertex_index = self.active_bucket = None
        self.last_refreshed = False

    def _apply(self, fn):
        self.reset_cache()
        self._workspaces.clear()
        return super()._apply(fn)

    def _load_from_state_dict(self, *args, **kwargs):
        self.reset_cache()
        self._previous_loss = self._last_loss_epoch = None
        self._epoch = None
        self.current_interval = self.refresh_interval if self.refresh_policy == 'fixed' else self.q_min
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, input, epoch=None, cache_key=None, force_refresh=False):
        if input.dim() != 2 or input.size(1) != self.in_feature:
            raise ValueError('Expected [nodes, in_feature] input')
        if input.device != self.random_vectors.device or input.dtype != self.random_vectors.dtype:
            raise ValueError('Move the module to the input device and dtype before use')
        if input.dtype not in (torch.float32, torch.float64):
            raise ValueError('Only float32 and float64 are supported')
        if epoch is not None:
            self.set_epoch(epoch)
        if self.cache and self._epoch is None and (self.refresh_interval > 1 or self.refresh_policy == 'loss'):
            raise ValueError('Cross-epoch reuse requires set_epoch(epoch) or forward(epoch=epoch)')
        # A workspace belongs to one device/stream; independent streams never share it.
        stream = torch.cuda.current_stream(input.device).cuda_stream if input.is_cuda else None
        slot = (self.training, input.device, stream)
        signature = (tuple(input.shape), input.dtype, cache_key, self.random_vectors.data_ptr(), self.random_vectors._version)
        state = self._states.get(slot)
        refresh = (not self.cache or force_refresh or self._epoch is None or state is None or
                   state['signature'] != signature or self._epoch - state['epoch'] >= self.current_interval)
        if refresh:
            workspace = None
            if input.is_cuda and input.size(0):
                workspace_key = (input.device, stream, self.param_H)
                if workspace_key not in self._workspaces:
                    self._workspaces[workspace_key] = _native.cluster_workspace(input.contiguous(), self.param_H)
                workspace = self._workspaces[workspace_key]
            if not input.size(0):
                index = torch.empty(0, device=input.device, dtype=torch.int64)
                counts = torch.zeros(1, device=input.device, dtype=torch.int32)
                active = 0
            else:
                index, counts, active = _assign(input, self.random_vectors, self.param_H, workspace)
            state = dict(index=index, counts=counts, active=active, signature=signature, epoch=self._epoch)
            if self.cache and self._epoch is not None:
                self._states[slot] = state
            self.refresh_count += 1
        self.last_refreshed = refresh
        self.vertex_index, self.active_bucket = state['index'], state['active']
        output = _Mean.apply(input, state['index'], state['counts'])
        return (output, state['index']) if self.is_training else output

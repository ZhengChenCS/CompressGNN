#include "cluster_gpu.h"
#include "cluster_ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <algorithm>

namespace {
constexpr int threads = 256;
int blocks(int64_t n) { return static_cast<int>(std::min<int64_t>((n + threads - 1) / threads, 65535)); }
struct Occupied {
    __host__ __device__ int operator()(const int &x) const { return x != 0; }
};
using Flags = cub::TransformInputIterator<int, Occupied, const int *>;
void check_float(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "Expected contiguous CUDA matrix");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64,
                "Only float32 and float64 are supported");
}
void check_index(torch::Tensor index, torch::Device device) {
    TORCH_CHECK(index.device() == device && index.dim() == 1 && index.is_contiguous() &&
                index.scalar_type() == torch::kInt64, "Expected contiguous int64 index on input device");
}
void check_count(torch::Tensor count, torch::Device device) {
    TORCH_CHECK(count.device() == device && count.dim() == 1 && count.is_contiguous() &&
                count.scalar_type() == torch::kInt32 && count.numel() >= 1,
                "Expected nonempty int32 counts on input device");
}
int bucket_size(int64_t bits) {
    TORCH_CHECK(bits >= 0 && bits <= 30, "Hash bits must be in [0,30]");
    return 1 << bits;
}
size_t scan_size(int buckets, cudaStream_t stream) {
    size_t bytes = 0;
    Flags flag(nullptr, Occupied{});
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, bytes, flag,
                  static_cast<int *>(nullptr), buckets, stream));
    return bytes;
}
template <typename scalar_t>
__global__ void hash_ids(const scalar_t *projected, int64_t *ids, int *counts, int64_t n, int bits) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < n; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int id = 0;
        for (int i = 0; i < bits; ++i) id = (id << 1) | (projected[row * bits + i] > 0);
        ids[row] = id;
        atomicAdd(counts + id, 1);
    }
}
__global__ void remap_ids(int64_t *ids, const int *prefix, int64_t n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) ids[i] = prefix[ids[i]];
}
__global__ void compact_counts(const int *bucket_count, const int *prefix, int *count, int n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x)
        if (bucket_count[i]) count[prefix[i]] = bucket_count[i];
}
template <typename scalar_t>
__global__ void sum_features(const scalar_t *x, const int64_t *index, scalar_t *out,
                             int64_t size, int64_t width, int64_t groups) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t group = index[i / width];
        CUDA_KERNEL_ASSERT(group >= 0 && group < groups);
        atomicAdd(out + group * width + i % width, x[i]);
    }
}
template <typename scalar_t>
__global__ void divide_counts(scalar_t *x, const int *count, int64_t size, int64_t width) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<int64_t>(blockDim.x) * gridDim.x)
        x[i] /= static_cast<scalar_t>(max(count[i / width], 1));
}
template <typename scalar_t>
__global__ void mean_grad(const scalar_t *grad, const int64_t *index, const int *count,
                         scalar_t *out, int64_t size, int64_t width, int64_t groups) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t group = index[i / width];
        CUDA_KERNEL_ASSERT(group >= 0 && group < groups);
        out[i] = grad[group * width + i % width] / static_cast<scalar_t>(max(count[group], 1));
    }
}
template <typename scalar_t>
__global__ void reconstruct(const scalar_t *reps, const int64_t *index, const scalar_t *bias,
                            scalar_t *out, int64_t size, int64_t width, int64_t groups, bool relu) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t group = index[i / width];
        CUDA_KERNEL_ASSERT(group >= 0 && group < groups);
        scalar_t value = reps[group * width + i % width];
        if (bias) value += bias[i % width];
        out[i] = relu && value < 0 ? scalar_t(0) : value;
    }
}
template <typename scalar_t>
__global__ void reconstruct_grad(const scalar_t *grad, const scalar_t *output,
    const int64_t *index, scalar_t *out, int64_t size, int64_t width, int64_t groups, bool relu) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t group = index[i / width];
        CUDA_KERNEL_ASSERT(group >= 0 && group < groups);
        scalar_t value = relu && output[i] <= 0 ? scalar_t(0) : grad[i];
        atomicAdd(out + group * width + i % width, value);
    }
}
} // namespace

std::vector<torch::Tensor> cluster_workspace_cuda(torch::Tensor input, int64_t bits) {
    check_float(input); c10::cuda::CUDAGuard guard(input.device());
    int n = bucket_size(bits); auto stream = at::cuda::getCurrentCUDAStream();
    auto ints = input.options().dtype(torch::kInt32);
    return {torch::empty({n}, ints), torch::empty({n}, ints),
            torch::empty({static_cast<int64_t>(scan_size(n, stream))}, input.options().dtype(torch::kUInt8))};
}
std::tuple<torch::Tensor, torch::Tensor, int64_t> cluster_assign_cuda(
    torch::Tensor input, torch::Tensor planes, int64_t bits, std::vector<torch::Tensor> workspace) {
    check_float(input); check_float(planes); c10::cuda::CUDAGuard guard(input.device());
    int n = bucket_size(bits);
    TORCH_CHECK(planes.device() == input.device() && planes.scalar_type() == input.scalar_type() &&
                planes.size(0) == input.size(1) && planes.size(1) == bits, "Hash plane shape/device/dtype mismatch");
    TORCH_CHECK(input.size(0) <= INT_MAX, "Node count exceeds int32 bucket count capacity");
    TORCH_CHECK(workspace.size() == 3, "Expected bucket, prefix and scan workspaces");
    auto stream = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < 3; ++i)
        TORCH_CHECK(workspace[i].device() == input.device() && workspace[i].dim() == 1 &&
                    workspace[i].is_contiguous(), "Workspace device/shape mismatch");
    TORCH_CHECK(workspace[0].scalar_type() == torch::kInt32 && workspace[1].scalar_type() == torch::kInt32 &&
                workspace[0].numel() == n && workspace[1].numel() == n &&
                workspace[2].scalar_type() == torch::kUInt8 && workspace[2].numel() >= scan_size(n, stream),
                "Workspace dtype/capacity mismatch");
    auto ids = torch::empty({input.size(0)}, input.options().dtype(torch::kInt64));
    if (!input.size(0)) return {ids, torch::zeros({1}, workspace[0].options()), 0};
    auto projected = input.mm(planes);
    auto buckets = workspace[0]; auto prefix = workspace[1]; buckets.zero_();
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cluster_hash_ids", [&] {
        hash_ids<scalar_t><<<blocks(input.size(0)), threads, 0, stream>>>(projected.data_ptr<scalar_t>(),
            ids.data_ptr<int64_t>(), buckets.data_ptr<int>(), input.size(0), bits);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    size_t bytes = workspace[2].numel(); Flags flags(buckets.data_ptr<int>(), Occupied{});
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(workspace[2].data_ptr(), bytes, flags,
                                               prefix.data_ptr<int>(), n, stream));
    // One scalar read only on assignment refresh; reuse epochs never take this path.
    int active = prefix.select(0, n - 1).item<int>();
    auto counts = torch::zeros({active + 1}, buckets.options());
    remap_ids<<<blocks(input.size(0)), threads, 0, stream>>>(ids.data_ptr<int64_t>(), prefix.data_ptr<int>(), input.size(0));
    compact_counts<<<blocks(n), threads, 0, stream>>>(buckets.data_ptr<int>(), prefix.data_ptr<int>(), counts.data_ptr<int>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {ids, counts, active};
}
std::tuple<torch::Tensor, ID_DATATYPE> cluster_forward_cuda(torch::Tensor input, torch::Tensor planes, const uint32_t bits) {
    auto result = cluster_assign_cuda(input, planes, bits, cluster_workspace_cuda(input, bits));
    return {std::get<0>(result).to(torch::kInt32), static_cast<int>(std::get<2>(result))};
}
torch::Tensor cluster_mean_cuda(torch::Tensor input, torch::Tensor index, torch::Tensor count) {
    check_float(input); check_index(index, input.device()); check_count(count, input.device());
    TORCH_CHECK(index.numel() == input.size(0), "Input/index length mismatch");
    c10::cuda::CUDAGuard guard(input.device()); auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::zeros({count.numel(), input.size(1)}, input.options());
    if (!input.numel()) return out;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cluster_mean", [&] {
        sum_features<scalar_t><<<blocks(input.numel()), threads, 0, stream>>>(input.data_ptr<scalar_t>(),
            index.data_ptr<int64_t>(), out.data_ptr<scalar_t>(), input.numel(), input.size(1), count.numel());
        divide_counts<scalar_t><<<blocks(out.numel()), threads, 0, stream>>>(out.data_ptr<scalar_t>(),
            count.data_ptr<int>(), out.numel(), input.size(1));
    }); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor cluster_mean_backward_cuda(torch::Tensor grad, torch::Tensor index, torch::Tensor count) {
    check_float(grad); check_index(index, grad.device()); check_count(count, grad.device());
    TORCH_CHECK(grad.size(0) == count.numel(), "Gradient/count shape mismatch");
    c10::cuda::CUDAGuard guard(grad.device()); auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({index.numel(), grad.size(1)}, grad.options());
    if (!out.numel()) return out;
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "cluster_mean_backward", [&] {
        mean_grad<scalar_t><<<blocks(out.numel()), threads, 0, stream>>>(grad.data_ptr<scalar_t>(),
            index.data_ptr<int64_t>(), count.data_ptr<int>(), out.data_ptr<scalar_t>(), out.numel(), grad.size(1), count.numel());
    }); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor cluster_reconstruct_cuda(torch::Tensor reps, torch::Tensor index,
                                      torch::optional<torch::Tensor> bias, bool relu) {
    check_float(reps); check_index(index, reps.device());
    if (bias.has_value()) TORCH_CHECK(bias->device() == reps.device() && bias->scalar_type() == reps.scalar_type() &&
        bias->dim() == 1 && bias->numel() == reps.size(1) && bias->is_contiguous(), "Invalid reconstruction bias");
    c10::cuda::CUDAGuard guard(reps.device()); auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({index.numel(), reps.size(1)}, reps.options());
    if (!out.numel()) return out;
    AT_DISPATCH_FLOATING_TYPES(reps.scalar_type(), "cluster_reconstruct", [&] {
        reconstruct<scalar_t><<<blocks(out.numel()), threads, 0, stream>>>(reps.data_ptr<scalar_t>(),
            index.data_ptr<int64_t>(), bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
            out.data_ptr<scalar_t>(), out.numel(), reps.size(1), reps.size(0), relu);
    }); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor cluster_reconstruct_backward_cuda(torch::Tensor grad, torch::Tensor index,
                                               torch::Tensor output, int64_t groups, bool relu) {
    check_float(grad); check_index(index, grad.device()); check_float(output);
    TORCH_CHECK(groups >= 0 && output.sizes() == grad.sizes() && output.device() == grad.device() &&
        output.scalar_type() == grad.scalar_type() && index.numel() == grad.size(0), "Invalid reconstruction gradient");
    c10::cuda::CUDAGuard guard(grad.device()); auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::zeros({groups, grad.size(1)}, grad.options());
    if (!grad.numel()) return out;
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "cluster_reconstruct_backward", [&] {
        reconstruct_grad<scalar_t><<<blocks(grad.numel()), threads, 0, stream>>>(grad.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), index.data_ptr<int64_t>(), out.data_ptr<scalar_t>(), grad.numel(), grad.size(1), groups, relu);
    }); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

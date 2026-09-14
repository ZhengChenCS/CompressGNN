#include "cluster.h"
#include "../util.h"
#include <Python.h>
#include <torch/script.h>

#include "cpu/cluster_cpu.h"
//#include "kongming_cluster_cpu.h"
#ifdef WITH_CUDA
#include "cuda/cluster_gpu.h"
#endif

std::tuple<torch::Tensor, ID_DATATYPE>
cluster_forward(torch::Tensor input, torch::Tensor random_vectors,
                         const uint32_t param_h) {

    TORCH_CHECK(input.dim() == 2 && random_vectors.dim() == 2 && input.is_contiguous() && random_vectors.is_contiguous(), "Expected contiguous matrices");
    TORCH_CHECK(param_h <= 30 && random_vectors.size(0) == input.size(1) && random_vectors.size(1) == param_h, "Invalid hash plane shape/bits");
    TORCH_CHECK(input.device() == random_vectors.device() && input.scalar_type() == random_vectors.scalar_type(), "Hash device/dtype mismatch");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64, "Expected float32/float64 features");
    if (input.device().is_cuda()) {
#ifdef WITH_CUDA
        return cluster_forward_cuda(input, random_vectors, param_h);
        // return std::vector<torch::Tensor>{input};
#else
        AT_ERROR("Not compiled with CUDA support");
#endif
    } else {
        return cluster_forward_cpu(input, random_vectors, param_h);
    }
}

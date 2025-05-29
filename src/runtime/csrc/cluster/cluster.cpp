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

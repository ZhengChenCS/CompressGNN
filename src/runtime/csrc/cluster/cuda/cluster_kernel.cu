#include "../../util.h"
#include "cluster_gpu.h"
#include "cluster_kernel.cuh"
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <utility>

void LSH(cudaStream_t &stream, const torch::Tensor &input,
         const torch::Tensor &random_vectors, const int32_t param_H,
         torch::Tensor &vector_index, ID_DATATYPE &active_bucket);

std::tuple<torch::Tensor, ID_DATATYPE>
cluster_forward_cuda(torch::Tensor input, torch::Tensor random_vectors,
                              const uint32_t param_H) {
    Timer tt;

    AT_ASSERTM(input.dim() == 2, "Input must be a 2-dim tensor");

    // int64_t inputHeight = input.size(0);
    int64_t inputWidth = input.size(1);

    AT_ASSERTM(random_vectors.size(0) == inputWidth,
               "Random vector width must be consistent with input width");

    AT_ASSERTM(param_H < 32, "Paramter H must <= 32");
    Timer timer;
    torch::Tensor vertex_index;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ID_DATATYPE active_bucket;
    LSH(stream, input, random_vectors, param_H, vertex_index, active_bucket);
    return {vertex_index, active_bucket};
}

void LSH(cudaStream_t &stream, const torch::Tensor &input,
         const torch::Tensor &random_vectors, const int32_t param_H,
         torch::Tensor &vertex_index, ID_DATATYPE &active_bucket) {
    const int64_t inputHeight = input.size(0);
    const int64_t inputWidth = input.size(1);
    ID_DATATYPE total_buckets = std::pow(2, param_H);
    torch::Tensor hash_vector = input.mm(random_vectors);
    vertex_index =
        torch::zeros(inputHeight, input.options().dtype(torch::kInt));
    get_id_cuda(stream, hash_vector, vertex_index, total_buckets,
                active_bucket);
}

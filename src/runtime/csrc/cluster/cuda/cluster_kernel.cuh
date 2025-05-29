#pragma once
#include "../../util.h"
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <utility>

#define CUDA_NUM_THREADS 1024

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                 \
         i += blockDim.x * gridDim.x)

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__global__ void
get_id_cuda_kernel(const int64_t n, // num_vectors
                   const int64_t vector_len, const scalar_t *hashed_vectors,
                   ID_DATATYPE *vector_ids, ID_DATATYPE *bucket_index) {

    CUDA_1D_KERNEL_LOOP(index, n) {
        const scalar_t *vector_ptr = hashed_vectors + index * vector_len;
        ID_DATATYPE id = 0;
#pragma unroll
        for (int i = 0; i < vector_len; i++) {
            if (vector_ptr[i] > 0) {
                id = (id << 1) | 1;
            } else {
                id = id << 1;
            }
        }
        vector_ids[index] = id;
        bucket_index[id] = 1;
    }
}

template <typename scalar_t>
__global__ void get_new_index_cuda_kernel(const int64_t n, // num_vectors
                                          ID_DATATYPE *vector_ids,
                                          ID_DATATYPE *bucket_index) {

    CUDA_1D_KERNEL_LOOP(index, n) {
        ID_DATATYPE old_id = vector_ids[index];
        ID_DATATYPE new_id = bucket_index[old_id];
        vector_ids[index] = new_id;
    }
}

void get_id_cuda(cudaStream_t &stream, const torch::Tensor &hash_vector,
                 torch::Tensor &vector_ids, ID_DATATYPE total_buckets,
                 ID_DATATYPE &active_bucket) {
    int64_t num_vectors = hash_vector.size(0);
    int64_t vector_len = hash_vector.size(1);
    int64_t n_rows = vector_ids.size(0);
    thrust::device_vector<ID_DATATYPE> bucket_index(total_buckets, 0);

    AT_DISPATCH_FLOATING_TYPES(
        hash_vector.scalar_type(), "get_id_count_cuda", ([&] {
            get_id_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(num_vectors), CUDA_NUM_THREADS, 0, stream>>>(
                    num_vectors, vector_len, hash_vector.data_ptr<scalar_t>(),
                    vector_ids.data_ptr<ID_DATATYPE>(),
                    thrust::raw_pointer_cast(&bucket_index[0]));
            thrust::inclusive_scan(bucket_index.begin(),
                                   bucket_index.begin() + total_buckets,
                                   bucket_index.begin());
            get_new_index_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(num_vectors), CUDA_NUM_THREADS, 0, stream>>>(
                    num_vectors, vector_ids.data_ptr<ID_DATATYPE>(),
                    thrust::raw_pointer_cast(&bucket_index[0]));
        }));
    AT_CUDA_CHECK(cudaGetLastError());
    active_bucket = bucket_index[total_buckets - 1];
}
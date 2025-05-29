#include "cuda_util.cuh"
#include "spmm_gpu.h"
#include "nnz_balance.cuh"
#include "row_balance.cuh"
#include <ATen/cuda/CUDAContext.h>

torch::Tensor spmm_cuda(torch::Tensor vlist, torch::Tensor elist,
                        torch::Tensor value, torch::Tensor mat,
                        std::string method,
                        torch::optional<torch::Tensor> optional_out) {
    CHECK_CUDA(vlist);
    CHECK_CUDA(elist);
    CHECK_CUDA(value);
    CHECK_CUDA(mat);
    CHECK_CONTIGUOUS(mat);
    if(optional_out.has_value())
        CHECK_CUDA(optional_out.value());
    
    torch::Tensor out;
    if(optional_out.has_value()){
        out = optional_out.value().contiguous();
    }else{
        auto sizes = mat.sizes().vec();
        sizes[0] = vlist.size(-1) - 1;
        out = torch::zeros(sizes, mat.options());
    }
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        mat.scalar_type(), "_", ([&] {
            auto mat_data = mat.data_ptr<scalar_t>();
            auto vlist_ptr = vlist.data_ptr<int64_t>();
            auto elist_ptr = elist.data_ptr<int64_t>();
            auto value_ptr = value.data_ptr<scalar_t>();
            auto out_data = out.data_ptr<scalar_t>();

            auto col_num = mat.size(-1);
            auto row_num = vlist.size(-1) - 1;
            auto nnz = elist.size(-1);
//
            if (method == "rowbalance_parreduce") {
                auto rowDim = (row_num * WARP_SIZE + THREADS - 1) / THREADS;
                auto colDim = (col_num + COL_TILE - 1) /  COL_TILE;
                auto BLOCKS = dim3(rowDim, colDim);
                spmm_rowbalance_parreduce_kernel<scalar_t>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                        vlist_ptr, elist_ptr, value_ptr, mat_data, out_data,
                        (int)row_num, (int)col_num);
            } else if (method == "rowbalance_seqreduce") {
                auto rowDim = (row_num + THREADS - 1) / THREADS;
                auto colDim = col_num;
                auto BLOCKS = dim3(rowDim, colDim);
                spmm_rowbalance_seqreduce_kernel<scalar_t>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                        vlist_ptr, elist_ptr, value_ptr, mat_data, out_data,
                        (int)row_num, (int)col_num);
            } else if (method == "rowbalance_rowcache") { 
                auto rowDim = (row_num * WARP_SIZE + THREADS - 1) / THREADS;
                auto colDim = (col_num + WARP_SIZE - 1) / 32; 
                auto BLOCKS = dim3(rowDim, colDim); 
                spmm_rowbalance_rowcache_kernel<scalar_t>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                        vlist_ptr, elist_ptr, value_ptr, mat_data, out_data,
                        (int)row_num, (int)col_num);
            } else if (method == "nnzbalance_parreduce") {
                auto nnzDim = (nnz + NE_PER_WARP - 1) / NE_PER_WARP;
                auto colDim = (col_num + COL_TILE - 1) / COL_TILE; 
                auto BLOCKS =
                    dim3((nnzDim * WARP_SIZE + THREADS - 1) / THREADS, colDim);
                spmm_nnzbalance_parreduce_kernel<scalar_t>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                        vlist_ptr, elist_ptr, value_ptr, mat_data, out_data,
                        (int)row_num, (int)col_num, (int64_t)nnz);
            } else if (method == "nnzbalance_seqreduce") {
                auto nnzDim = (nnz + NE_PER_THREAD - 1) / NE_PER_THREAD;
                auto colDim = (col_num + COL_TILE - 1) / COL_TILE;
                auto BLOCKS = dim3((nnzDim + THREADS - 1) / THREADS, colDim);
                spmm_nnzbalance_seqreduce_kernel<scalar_t>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                        vlist_ptr, elist_ptr, value_ptr, mat_data, out_data,
                        (int)row_num, (int)col_num, (int64_t)nnz);
            } else if (method == "nnzbalance_rowcache") {
                auto nnzDim = (nnz + NE_PER_WARP - 1) / NE_PER_WARP;
                auto colDim = (col_num + COL_TILE - 1) / COL_TILE;
                auto BLOCKS =
                    dim3((nnzDim * WARP_SIZE + THREADS - 1) / THREADS, colDim);
                spmm_nnzbalance_rowcache_kernel<scalar_t>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                        vlist_ptr, elist_ptr, value_ptr, mat_data, out_data,
                        (int)row_num, (int)col_num, (int64_t)nnz);
            }
            else{
                std::cerr << "Unvaild spmm method for CUDA." << std::endl;
            }
        }));
    cudaDeviceSynchronize();
    return out;
}

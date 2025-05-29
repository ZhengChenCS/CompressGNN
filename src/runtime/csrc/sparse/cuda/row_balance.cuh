#pragma once
#include "cuda_util.cuh"
#include <cstdint>

template <typename scalar_t>
__global__ void
spmm_rowbalance_parreduce_kernel(const int64_t *vlist, const int64_t *elist,
                                 const scalar_t *value,
                                 const scalar_t *mat_data, scalar_t *out_data,
                                 const int row_num, const int col_num) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t warpIdx = thread_id >> 5;
    int64_t row = warpIdx;
    if (row >= row_num)
        return;
    int laneIdx = thread_id & (32 - 1);
    int mat_col_idx = blockIdx.y << 5;
    int leftover = col_num - mat_col_idx;
    

    int64_t start = __ldg(vlist + row);
    int64_t end = __ldg(vlist + row + 1);

    int mat_row;
    scalar_t val;
    float result[COL_TILE] = {0.0};

    int eid = start + laneIdx;

    for (; eid < end; eid += WARP_SIZE) {
        if (eid < end) {
            mat_row = __ldg(elist + eid) * col_num;
            val = __ldg(value + eid);
        } else {
            val = 0.0;
            mat_row = 0;
        }
#pragma unroll
        for (int jj = 0; jj < COL_TILE; ++jj) {
            if(jj < leftover)
                result[jj] += __ldg(mat_data + mat_row + mat_col_idx + jj) * val;
        }
    }
    for (int jj = 0; jj < COL_TILE; ++jj) {
        
        SHFL_DOWN_REDUCE(result[jj]);
    }
    if (laneIdx == 0) {
        for (int kk = 0; kk < COL_TILE; ++kk) {
            if(kk < leftover)
                out_data[row * col_num + mat_col_idx + kk] += result[kk];
        }
    }
}

template <typename scalar_t>
__global__ void
spmm_rowbalance_seqreduce_kernel(const int64_t *vlist, const int64_t *elist,
                                 const scalar_t *value,
                                 const scalar_t *mat_data, scalar_t *out_data,
                                 const int row_num, const int col_num) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t col = blockIdx.y;
    if (row >= row_num)
        return;
    scalar_t result = 0;

    int64_t start = __ldg(vlist + row);
    int64_t end = __ldg(vlist + row + 1);

    int mat_row;
    scalar_t val;

    for (int eid = start; eid < end; ++eid) {
        mat_row = __ldg(elist + eid) * col_num;
        val = __ldg(value + eid);
        result += __ldg(mat_data + mat_row + col) * val;
    }
    out_data[row * col_num + col] += result;
}

template <typename scalar_t>
__global__ void
spmm_rowbalance_rowcache_kernel(const int64_t *vlist, const int64_t *elist,
                                const scalar_t *value, const scalar_t *mat_data,
                                scalar_t *out_data, const int row_num,
                                const int col_num) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row = thread_id >> 5;
    if (row >= row_num)
        return;
    int laneIdx = thread_id & (32 - 1);
    int mat_col_idx = (blockIdx.y << 5) + laneIdx;
    int leftover = col_num - (blockIdx.y << 5);

    int64_t start = __ldg(vlist + row);
    int64_t end = __ldg(vlist + row + 1);

    int mat_row, mat_rows[WARP_SIZE];
    scalar_t val, vals[WARP_SIZE];
    int eid = start + laneIdx;
    scalar_t result = 0;

    for (int ii = start; ii < end; ii += 32) {
        if (eid < end) {
            mat_row = __ldg(elist + eid) * col_num;
            val = __ldg(value + eid);
        } else {
            mat_row = -1;
            val = 0.0;
        }
        eid += 32;
#pragma unroll
        for (int jj = 0; jj < WARP_SIZE; jj++) {
            mat_rows[jj] = __shfl_sync(FULL_MASK, mat_row, jj);
            vals[jj] = __shfl_sync(FULL_MASK, val, jj);
        }
#pragma unroll
        for (int jj = 0; jj < WARP_SIZE; jj++) {
            if (laneIdx < leftover && mat_rows[jj] != -1) {
                val = vals[jj] * __ldg(mat_data + mat_rows[jj] + mat_col_idx);
                result += val;
            }
        }
    }
    if (laneIdx < leftover) {
        out_data[row * col_num + mat_col_idx] += result;
    }
}
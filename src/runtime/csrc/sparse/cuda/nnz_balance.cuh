#pragma once
#include "cuda_util.cuh"
#include <cstdint>

template <typename scalar_t>
__global__ void spmm_nnzbalance_parreduce_kernel(
    const int64_t *vlist, const int64_t *elist, const scalar_t *value,
    const scalar_t *mat_data, scalar_t *out_data, const int row_num,
    const int col_num, const int64_t nnz) {
    int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t col = blockIdx.y * 32;

    int64_t warpIdx = thread_idx >> 5;
    int laneIdx = thread_idx & (32 - 1);

    if (warpIdx * NE_PER_WARP > nnz)
        return;

    int64_t k;
    float v;

    float result[32] = {0.0};

    int64_t eid = warpIdx * NE_PER_WARP + laneIdx;

    for (; eid < (warpIdx + 1) * NE_PER_WARP; eid += WARP_SIZE) {
        // if(eid >= nnz) break;
        int row =
            binary_search_segment_number<int64_t>(vlist, row_num, nnz, eid);
        if (eid < nnz) {
            k = __ldg(elist + eid) * col_num;
            v = __ldg(value + eid);
        } else {
            k = 0;
            v = 0.0f;
        }
#pragma unroll
        for (int jj = 0; jj < COL_TILE; ++jj) {
            float val = __ldg(mat_data + k + col + jj);
            result[jj] = val * v;
        }
        int warp_start = __shfl_sync(FULL_MASK, row, 0);
        int warp_end = __shfl_sync(FULL_MASK, row, 31);

        if (warp_end == warp_start) {
#pragma unroll
            for (int jj = 0; jj < COL_TILE; jj++) {
                SHFL_DOWN_REDUCE(result[jj]);
            }
            if (laneIdx == 0) {
#pragma unroll
                for (int jj = 0; jj < COL_TILE; ++jj) {
                    if (col + jj < col_num)
                        atomicAdd(&out_data[row * col_num + col + jj],
                                  result[jj]);
                }
            }
        } else {
            bool is_seg_start =
                (__shfl_up_sync(FULL_MASK, row, 1) != row) || (laneIdx == 0);
            float tmpv = 0;
            int tmpr;
#pragma unroll
            for (int jj = 0; jj < COL_TILE; ++jj) {
                SEG_SHFL_SCAN(result[jj], tmpv, row, tmpr);
            }
            if (is_seg_start) {
#pragma unroll
                for (int jj = 0; jj < COL_TILE; ++jj) {
                    if (col + jj < col_num)
                        atomicAdd(&out_data[row * col_num + col + jj],
                                  result[jj]);
                }
            }
        }
    }
    return;
}

template <typename scalar_t>
__global__ void spmm_nnzbalance_seqreduce_kernel(
    const int64_t *vlist, const int64_t *elist, const scalar_t *value,
    const scalar_t *mat_data, scalar_t *out_data, const int row_num,
    const int col_num, const int64_t nnz) {
    int64_t nnz_start = blockIdx.x * blockDim.x + threadIdx.x;
    if (nnz_start >= (nnz + NE_PER_THREAD - 1) / NE_PER_THREAD)
        return;
    int64_t col = blockIdx.y * 32;

    float result[32] = {0.0};
    uint64_t colIdx = 0;
    float val = 0.0;

    int64_t eid = nnz_start * NE_PER_THREAD;
    int64_t row =
        binary_search_segment_number<int64_t>(vlist, row_num, nnz, eid);
    int step = __ldg(vlist + row + 1) - eid;

    for (int ii = 0; ii < NE_PER_THREAD; ii++) {
        if (eid > nnz)
            break;
        if (ii < step) {
            colIdx = __ldg(elist + eid) * col_num;
            val = __ldg(value + eid);
#pragma unroll
            for (int jj = 0; jj < COL_TILE; ++jj) {
                if (col + jj < col_num) {
                    result[jj] += __ldg(mat_data + colIdx + col + jj) * val;
                }
            }
            eid++;
        } else {
#pragma unroll
            for (int jj = 0; jj < 32; ++jj) {
                if (col + jj < col_num)
                    atomicAdd(&out_data[row * col_num + col + jj], result[jj]);
            }
            row =
                binary_search_segment_number<int64_t>(vlist, row_num, nnz, eid);
            step = __ldg(vlist + row + 1) - eid + ii;
            colIdx = __ldg(elist + eid) * col_num;
            val = __ldg(value + eid);
#pragma unroll
            for (int jj = 0; jj < COL_TILE; ++jj) {
                if (col + jj < col_num)
                    result[jj] = __ldg(mat_data + colIdx + col + jj) * val;
            }
            eid++;
        }
    }
#pragma unroll
    for (int jj = 0; jj < 32; ++jj) {
        if (col + jj < col_num)
            atomicAdd(&out_data[row * col_num + col + jj], result[jj]);
    }
}

template <typename scalar_t>
__global__ void
spmm_nnzbalance_rowcache_kernel(const int64_t *vlist, const int64_t *elist,
                                const scalar_t *value, const scalar_t *mat_data,
                                scalar_t *out_data, const int row_num,
                                const int col_num, const int64_t nnz) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t warpIdx = thread_id >> 5;
    if (warpIdx * NE_PER_WARP > nnz)
        return;
    int laneIdx = thread_id & (32 - 1);
    int mat_col_idx = (blockIdx.y << 5) + laneIdx;
    int leftover = col_num - (blockIdx.y << 5);

    int mat_row, mat_rows[WARP_SIZE];
    int row_id, row_ids[WARP_SIZE];
    scalar_t val, vals[WARP_SIZE];
    scalar_t results[WARP_SIZE] = {0.0};

    int64_t eid = warpIdx * NE_PER_WARP + laneIdx;
    for (int ii = 0; ii < NE_PER_WARP; ii += WARP_SIZE) {
        if (eid < nnz) {
            mat_row = __ldg(elist + eid) * col_num;
            val = __ldg(value + eid);
        } else {
            mat_row = -1;
            val = 0.0f;
        }
        row_id =
            binary_search_segment_number<int64_t>(vlist, row_num, nnz, eid);
#pragma unroll
        for (int jj = 0; jj < WARP_SIZE; ++jj) {
            mat_rows[jj] = __shfl_sync(FULL_MASK, mat_row, jj);
            vals[jj] = __shfl_sync(FULL_MASK, val, jj);
            row_ids[jj] = __shfl_sync(FULL_MASK, row_id, jj);
        }
#pragma unroll
        for (int jj = 0; jj < WARP_SIZE; ++jj) {
            if (mat_rows[jj] != -1) {
                val = __ldg(mat_data + mat_rows[jj] + mat_col_idx);
                results[jj] = val * vals[jj];
            }
        }
        int row_curr = row_ids[0], next_row;
        scalar_t result = results[0];
        // #pragma unroll
        for (int jj = 1; jj < WARP_SIZE; ++jj) {
            next_row = row_ids[jj];
            if (row_curr != next_row) {
                if (laneIdx < leftover) {
                    atomicAdd(&out_data[row_curr * col_num + mat_col_idx],
                              result);
                }
                row_curr = next_row;
                result = 0;
            }
            result = result + results[jj];
        }
        if (laneIdx < leftover)
            atomicAdd(&out_data[row_curr * col_num + mat_col_idx], result);
        eid += WARP_SIZE;
    }
    return;
}
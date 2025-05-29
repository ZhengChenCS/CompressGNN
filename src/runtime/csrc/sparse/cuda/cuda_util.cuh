#ifndef CUDA_UTIL
#define CUDA_UTIL

#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS 256
#define FULL_MASK 0xffffffff
#define NE_PER_THREAD 32
#define NE_PER_WARP 32
#define COL_TILE 32
#define WARP_SIZE 32

#define SHFL_DOWN_REDUCE(v)                                                    \
    v += __shfl_down_sync(FULL_MASK, v, 16);                                   \
    v += __shfl_down_sync(FULL_MASK, v, 8);                                    \
    v += __shfl_down_sync(FULL_MASK, v, 4);                                    \
    v += __shfl_down_sync(FULL_MASK, v, 2);                                    \
    v += __shfl_down_sync(FULL_MASK, v, 1);

#define SEG_SHFL_SCAN(v, tmpv, segid, tmps)                                    \
    tmpv = __shfl_down_sync(FULL_MASK, v, 1);                                  \
    tmps = __shfl_down_sync(FULL_MASK, segid, 1);                              \
    if (tmps == segid && laneIdx < 31)                                         \
        v += tmpv;                                                             \
    tmpv = __shfl_down_sync(FULL_MASK, v, 2);                                  \
    tmps = __shfl_down_sync(FULL_MASK, segid, 2);                              \
    if (tmps == segid && laneIdx < 30)                                         \
        v += tmpv;                                                             \
    tmpv = __shfl_down_sync(FULL_MASK, v, 4);                                  \
    tmps = __shfl_down_sync(FULL_MASK, segid, 4);                              \
    if (tmps == segid && laneIdx < 28)                                         \
        v += tmpv;                                                             \
    tmpv = __shfl_down_sync(FULL_MASK, v, 8);                                  \
    tmps = __shfl_down_sync(FULL_MASK, segid, 8);                              \
    if (tmps == segid && laneIdx < 24)                                         \
        v += tmpv;                                                             \
    tmpv = __shfl_down_sync(FULL_MASK, v, 16);                                 \
    tmps = __shfl_down_sync(FULL_MASK, segid, 16);                             \
    if (tmps == segid && laneIdx < 16)                                         \
        v += tmpv;

template <typename index_t>
__device__ __forceinline__ index_t
binary_search_segment_number(const index_t *seg_offsets, const index_t n_seg,
                             const index_t n_elem, const index_t elem_id) {
    index_t lo = 1, hi = n_seg, mid;
    while (lo < hi) {
        mid = (lo + hi) >> 1;
        if (seg_offsets[mid] <= elem_id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (hi - 1);
}

#endif
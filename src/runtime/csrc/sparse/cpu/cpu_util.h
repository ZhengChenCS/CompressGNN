#ifndef CPU_UTIL
#define CPU_UTIL
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <condition_variable>
#include <immintrin.h>
#include <emmintrin.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#define compiler_fence() asm volatile(""::: "memory")

#define NE_PER_THREAD 64

template <typename index_t>
inline index_t
binary_search_segment_number(const index_t *seg_offsets, const index_t n_seg,
                             const index_t n_elem, const index_t elem_id){
    index_t lo = 1, hi = n_seg, mid;
    while(lo < hi){
        mid = (lo + hi) >> 1;
        if(seg_offsets[mid] <= elem_id){
            lo = mid + 1;
        }else{
            hi = mid;
        }
    }
    return (hi-1);
}

template <typename T>
inline typename std::enable_if<sizeof(T) == 1, bool>::type cas(T *ptr, T oldv, T newv)
{
    static_assert(sizeof(char) == 1);
    return __sync_bool_compare_and_swap((char*)ptr, *((char*)&oldv), *((char*)&newv));
}

template <typename T>
inline typename std::enable_if<sizeof(T) == 2, bool>::type cas(T *ptr, T oldv, T newv)
{
    static_assert(sizeof(short) == 2);
    return __sync_bool_compare_and_swap((short*)ptr, *((short*)&oldv), *((short*)&newv));
}

template <typename T>
inline typename std::enable_if<sizeof(T) == 4, bool>::type cas(T *ptr, T oldv, T newv)
{
    static_assert(sizeof(int) == 4);
    return __sync_bool_compare_and_swap((int*)ptr, *((int*)&oldv), *((int*)&newv));
}

template <typename T>
inline typename std::enable_if<sizeof(T) == 8, bool>::type cas(T *ptr, T oldv, T newv)
{
    static_assert(sizeof(long) == 8);
    return __sync_bool_compare_and_swap((long*)ptr, *((long*)&oldv), *((long*)&newv));
}

template <typename T>
inline typename std::enable_if<sizeof(T) == 16, bool>::type cas(T *ptr, T oldv, T newv)
{
    static_assert(sizeof(__int128_t) == 16);
    return __sync_bool_compare_and_swap((__int128_t*)ptr, *((__int128_t*)&oldv), *((__int128_t*)&newv));
}

template <class T>
inline T write_add(T *a, T b) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "length not supported by cas.");
    T newV, oldV;
    do
    {
        compiler_fence();
        oldV = *a;
        newV = oldV + b;
    }
    while (!cas(a, oldV, newV));
    return newV;
}

#endif
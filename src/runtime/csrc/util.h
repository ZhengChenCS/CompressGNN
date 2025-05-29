#pragma once
#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__version_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__version_cpu(void) { return NULL; }
#endif
#endif

namespace {
class Timer {
  public:
    clock_t _tic;
    clock_t _toc;
    Timer() { _tic = clock(); }
    void tic() { _tic = clock(); }
    void toc(std::string s = "") {
        _toc = clock();
        // std::cout << s << ": " << 1.0 * (_toc - _tic) / CLOCKS_PER_SEC <<
        // std::endl;
        _tic = clock();
    }
};
} // namespace

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be GPU tensor")

using ID_DATATYPE = int;
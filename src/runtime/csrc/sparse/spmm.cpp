#include "spmm.h"
#include "../util.h"
#include <Python.h>
#include <torch/script.h>

#include "cpu/spmm_cpu.h"
#ifdef WITH_CUDA
#include "cuda/spmm_gpu.h"
#endif

torch::Tensor spmm_func(torch::Tensor vlist, torch::Tensor elist,
                   torch::Tensor value, torch::Tensor mat, std::string method,
                   torch::optional<torch::Tensor> optional_out) {
    if (mat.device().is_cuda()) {
#ifdef WITH_CUDA
        return spmm_cuda(vlist, elist, value, mat, method, optional_out);
#else
        AT_ERROR("Not compiled with CUDA support");
#endif
    } else {
        return spmm_cpu(vlist, elist, value, mat, method, optional_out);
    }
}
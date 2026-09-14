#include "cluster/cluster.h"
#include "sparse/spmm.h"
#include "util.h"
#ifdef WITH_CUDA
#include "cluster/cuda/cluster_ops.h"
#endif
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster_forward", &cluster_forward,
          "CompressGNN cluster forward");
#ifdef WITH_CUDA
    m.def("cluster_workspace", &cluster_workspace_cuda);
    m.def("cluster_assign", &cluster_assign_cuda);
    m.def("cluster_mean", &cluster_mean_cuda);
    m.def("cluster_mean_backward", &cluster_mean_backward_cuda);
    m.def("cluster_reconstruct", &cluster_reconstruct_cuda);
    m.def("cluster_reconstruct_backward", &cluster_reconstruct_backward_cuda);
#endif
    m.def("spmm_func", &spmm_func, "spmm function");
}

#include "cluster/cluster.h"
#include "sparse/spmm.h"
#include "util.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster_forward", &cluster_forward,
          "CompressGNN cluster forward");
    m.def("spmm_func", &spmm_func, "spmm function");
}

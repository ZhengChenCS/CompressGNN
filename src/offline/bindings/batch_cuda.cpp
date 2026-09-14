#include "batch_bridge.hpp"
#include <cuda_runtime.h>
PYBIND11_MODULE(compressgnn_batch_cuda,m) {
    namespace py=pybind11;
    m.def("is_available",[](){int count=0;auto err=cudaGetDeviceCount(&count);if(err!=cudaSuccess){cudaGetLastError();return false;}return count>0;});
    m.def("compress_csr_batch",[](const py::array& v,const py::array& e,int rounds,int frequency){
        return batch_bridge::run(compressgraph::compress_cuda,v,e,1,rounds,frequency);
    },py::arg("vlist"),py::arg("elist"),py::arg("rounds")=12,py::arg("min_pair_frequency")=16);
}

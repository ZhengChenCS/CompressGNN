#pragma once
#include "core/batch/compress.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <climits>
namespace batch_bridge {
namespace py = pybind11;
inline std::vector<int32_t> integers(const py::array& input) {
    const auto kind = py::str(input.dtype().attr("kind")).cast<std::string>();
    if (input.ndim()!=1 || (kind!="i" && kind!="u"))
        throw py::value_error("CSR arrays must be one-dimensional integer arrays");
    if (input.size()>INT_MAX) throw py::value_error("CSR array exceeds int32 capacity");
    if (kind=="i" && input.itemsize()==4) {
        py::array_t<int32_t,py::array::c_style|py::array::forcecast> values(input);
        return {values.data(),values.data()+values.size()};
    }
    py::array_t<int64_t,py::array::c_style|py::array::forcecast> values(input);
    std::vector<int32_t> output(values.size());
    for(py::ssize_t i=0;i<values.size();++i) {
        auto value=values.data()[i];
        if(value<0 || value>INT_MAX) throw py::value_error("CSR value exceeds int32 range");
        output[i]=static_cast<int32_t>(value);
    }
    return output;
}
inline py::array_t<int32_t> owned(std::vector<int32_t>&& values) {
    auto* storage=new std::vector<int32_t>(std::move(values));
    py::capsule owner(storage,[](void* p){delete static_cast<std::vector<int32_t>*>(p);});
    return py::array_t<int32_t>({storage->size()},{sizeof(int32_t)},storage->data(),owner);
}
using Function=compressgraph::Result(*)(const compressgraph::Csr&,const compressgraph::Options&);
inline py::tuple run(Function fn,const py::array& v,const py::array& e,int threads,int rounds,int frequency) {
    compressgraph::Csr input{integers(v),integers(e)};
    compressgraph::Options options;options.threads=threads;options.rounds=rounds;options.min_frequency=frequency;
    compressgraph::Result result;
    {py::gil_scoped_release release;result=fn(input,options);}
    return py::make_tuple(owned(std::move(result.graph.rowptr)),owned(std::move(result.graph.col)),result.vertices,result.rules);
}
} // namespace batch_bridge

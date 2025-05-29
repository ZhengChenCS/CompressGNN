#pragma once
#include <cstdio>
#include <iostream>
#include <vector>
#include <utility>
#include "assert.h"
#include "util.h"


std::tuple<py::array_t<int64_t>, py::array_t<int64_t>> 
gen_degree(
    const py::array_t<int64_t>& v2v_vlist,
    const py::array_t<int64_t>& v2v_elist,
    const py::array_t<int64_t>& v2r_vlist,
    const py::array_t<int64_t>& v2r_elist,
    const py::array_t<int64_t>& r2v_vlist,
    const py::array_t<int64_t>& r2v_elist,
    const std::vector<py::array_t<int64_t>>& r2r_vlists,
    const std::vector<py::array_t<int64_t>>& r2r_elists
)
{
    auto v2v_vlist_buf = v2v_vlist.request();
    auto v2v_elist_buf = v2v_elist.request();
    auto v2r_vlist_buf = v2r_vlist.request();
    auto v2r_elist_buf = v2r_elist.request();
    auto r2v_vlist_buf = r2v_vlist.request();
    auto r2v_elist_buf = r2v_elist.request();

    auto v2v_vlist_ptr = static_cast<int64_t*>(v2v_vlist_buf.ptr);
    auto v2v_elist_ptr = static_cast<int64_t*>(v2v_elist_buf.ptr);
    auto v2r_vlist_ptr = static_cast<int64_t*>(v2r_vlist_buf.ptr);
    auto v2r_elist_ptr = static_cast<int64_t*>(v2r_elist_buf.ptr);
    auto r2v_vlist_ptr = static_cast<int64_t*>(r2v_vlist_buf.ptr);
    auto r2v_elist_ptr = static_cast<int64_t*>(r2v_elist_buf.ptr);
    size_t step = r2r_vlists.size();

    std::vector<int64_t*> r2r_vlist_ptrs;
    std::vector<int64_t*> r2r_elist_ptrs;
    for (const auto& r2r_vlist : r2r_vlists) {
        r2r_vlist_ptrs.push_back(static_cast<int64_t*>(r2r_vlist.request().ptr));
    }
    for (const auto& r2r_elist : r2r_elists) {
        r2r_elist_ptrs.push_back(static_cast<int64_t*>(r2r_elist.request().ptr));
    }

    int64_t vertex_cnt = v2v_vlist_buf.size-1;
    int64_t rule_cnt = r2v_vlist_buf.size-1;

    std::vector<int64_t> v_degree(vertex_cnt, 0);
    std::vector<int64_t> r_degree(rule_cnt, 0);

    // add vertex degree in v2v
    for (int64_t v = 0; v < vertex_cnt; ++v) {
        v_degree[v] = v2v_vlist_ptr[v+1] - v2v_vlist_ptr[v];
    }

    // add rule degree in r2v
    for (int64_t r = 0; r < rule_cnt; ++r) {
        r_degree[r] = r2v_vlist_ptr[r+1] - r2v_vlist_ptr[r];
    }

    // add rule degree in r2r
    for(int64_t layer = step-1; layer >=0; layer--) {
        for(int64_t r = 0; r < rule_cnt; ++r) {
            int64_t start = r2r_vlist_ptrs[layer][r];
            int64_t end = r2r_vlist_ptrs[layer][r+1];
            for(int64_t i = start; i < end; ++i) {
                int64_t child = r2r_elist_ptrs[layer][i];
                r_degree[r] += r_degree[child];
            }
        }
    }
    
    // add vertex_degree in v2r
    for(int64_t v = 0; v < vertex_cnt; ++v) {
        int64_t start = v2r_vlist_ptr[v];
        int64_t end = v2r_vlist_ptr[v+1];
        for(int64_t i = start; i < end; ++i) {
            int64_t rule = v2r_elist_ptr[i];
            v_degree[v] += r_degree[rule];
        }
    }
    return std::make_tuple(py::array_t<int64_t>(v_degree.size(), v_degree.data()), 
                           py::array_t<int64_t>(r_degree.size(), r_degree.data()));
}






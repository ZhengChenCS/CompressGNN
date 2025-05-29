#pragma once
#include "assert.h"
#include "util.h"
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <vector>

/*
* hybird partition
* @input: vlist, elist, vertex_cnt
*/

std::tuple<
std::tuple<py::array_t<VertexT>, py::array_t<VertexT>, py::array_t<float>>,
std::tuple<py::array_t<VertexT>, py::array_t<VertexT>, py::array_t<float>>,
std::tuple<py::array_t<VertexT>, py::array_t<VertexT>, py::array_t<float>>,
std::tuple<py::array_t<VertexT>, py::array_t<VertexT>, py::array_t<float>>
>
hybird_partition(py::array_t<VertexT> &np_input_vlist, py::array_t<VertexT> &np_input_elist, py::array_t<float> &np_input_value, VertexT vertex_cnt)
{
    py::buffer_info vlist_buf = np_input_vlist.request();
    VertexT *vlist_ptr = (VertexT*) vlist_buf.ptr;
    size_t total_cnt = vlist_buf.size - 1;
    py::buffer_info elist_buf = np_input_elist.request();
    VertexT *elist_ptr = (VertexT*) elist_buf.ptr;
    py::buffer_info value_buf = np_input_value.request();
    float* value_ptr = (float*) value_buf.ptr;
    VertexT vid, eid, start, end, dst;


    std::vector<VertexT> v2v_vlist;
    std::vector<VertexT> v2v_elist;
    std::vector<float> v2v_value;

    std::vector<VertexT> v2r_vlist;
    std::vector<VertexT> v2r_elist;
    std::vector<float> v2r_value;

    std::vector<VertexT> r2v_vlist;
    std::vector<VertexT> r2v_elist;
    std::vector<float> r2v_value;

    std::vector<VertexT> r2r_vlist;
    std::vector<VertexT> r2r_elist;
    std::vector<float> r2r_value;

    v2v_vlist.emplace_back(0);
    v2r_vlist.emplace_back(0);
    r2v_vlist.emplace_back(0);
    r2r_vlist.emplace_back(0);
    
    VertexT v2v_len = 0, v2r_len = 0, r2v_len = 0, r2r_len = 0;
    for(vid = 0; vid < vertex_cnt; ++vid){
        start = vlist_ptr[vid];
        end = vlist_ptr[vid+1];
        for(eid = start; eid < end && elist_ptr[eid] < vertex_cnt; eid++){
            v2v_elist.emplace_back(elist_ptr[eid]);
            v2v_value.emplace_back(value_ptr[eid]);
            v2v_len++;
        }
        for(; eid < end; eid++){
            v2r_elist.emplace_back(elist_ptr[eid]-vertex_cnt);
            v2r_value.emplace_back(value_ptr[eid]);
            v2r_len++;
        }
        v2v_vlist.emplace_back(v2v_len);
        v2r_vlist.emplace_back(v2r_len);
    }

    for(vid = vertex_cnt; vid < total_cnt; ++vid){
        start = vlist_ptr[vid];
        end = vlist_ptr[vid+1];
        for(eid = start; eid < end && elist_ptr[eid] < vertex_cnt; eid++){
            r2v_elist.emplace_back(elist_ptr[eid]);
            r2v_value.emplace_back(value_ptr[eid]);
            r2v_len++;
        }
        for(; eid < end; eid++){
            r2r_elist.emplace_back(elist_ptr[eid]-vertex_cnt);
            r2r_value.emplace_back(value_ptr[eid]);
            r2r_len++;
        }
        r2v_vlist.emplace_back(r2v_len);
        r2r_vlist.emplace_back(r2r_len);
    }

    auto np_v2v_vlist = vector2numpy1D(v2v_vlist);
    auto np_v2v_elist = vector2numpy1D(v2v_elist);
    auto np_v2v_value = vector2numpy1D(v2v_value);

    auto np_v2r_vlist = vector2numpy1D(v2r_vlist);
    auto np_v2r_elist = vector2numpy1D(v2r_elist);
    auto np_v2r_value = vector2numpy1D(v2r_value);

    auto np_r2v_vlist = vector2numpy1D(r2v_vlist);
    auto np_r2v_elist = vector2numpy1D(r2v_elist);
    auto np_r2v_value = vector2numpy1D(r2v_value);

    auto np_r2r_vlist = vector2numpy1D(r2r_vlist);
    auto np_r2r_elist = vector2numpy1D(r2r_elist);
    auto np_r2r_value = vector2numpy1D(r2r_value);

    return {
        std::make_tuple(np_v2v_vlist, np_v2v_elist, np_v2v_value),
        std::make_tuple(np_v2r_vlist, np_v2r_elist, np_v2r_value),
        std::make_tuple(np_r2v_vlist, np_r2v_elist, np_r2v_value),
        std::make_tuple(np_r2r_vlist, np_r2r_elist, np_r2r_value)
    };
}

std::vector<
std::tuple<py::array_t<VertexT>, py::array_t<VertexT>, py::array_t<float>>
>
topo_partition(py::array_t<VertexT> &np_input_vlist,
            py::array_t<VertexT> &np_input_elist,
            py::array_t<float> &np_input_value)
{
    py::buffer_info vlist_buf = np_input_vlist.request();
    VertexT *vlist_ptr = (VertexT*) vlist_buf.ptr;
    VertexT r_cnt = vlist_buf.size - 1;
    py::buffer_info elist_buf = np_input_elist.request();
    VertexT *elist_ptr = (VertexT*) elist_buf.ptr;
    py::buffer_info value_buf = np_input_value.request();
    float* value_ptr = (float*) value_buf.ptr;

    std::vector<int> degree(r_cnt, 0);
    std::vector<int> order(r_cnt, -1);
    VertexT vid, eid, dst, start, end;
    for(vid = 0; vid < r_cnt; ++vid)
    {
        degree[vid] = vlist_ptr[vid+1] - vlist_ptr[vid];
    }

    VertexT count = r_cnt;
    for(vid = 0; vid < r_cnt; ++vid)
    {
        if(degree[vid] == 0){
            order[vid] = 0;
            count--;
        }
    }
    int step = 1;
    while(count > 0)
    {
        for(vid = 0; vid < r_cnt; ++vid)
        {
            if(order[vid] != -1) continue;
            VertexT active_count = 0;
            for(eid = vlist_ptr[vid]; eid < vlist_ptr[vid+1]; ++eid)
            {
                dst = elist_ptr[eid];
                if(order[dst] != -1)
                {
                    active_count++;
                }
            }
            if(active_count == degree[vid])
            {
                order[vid] = step;
                count--;
            }
        }
        step++;
    }
    int num_subgraph = step-1;
    std::vector<std::tuple<std::vector<VertexT>, std::vector<VertexT>, std::vector<float>>> r2r_graph(num_subgraph);
    std::vector<VertexT> offset(num_subgraph, 0);

    for(int i = 0; i < num_subgraph; ++i)
    {
        std::get<0>(r2r_graph[i]).emplace_back(0);
    }

    for(vid = 0; vid < r_cnt; vid++)
    {
        int v_step = order[vid]-1;
        if(v_step == -1) // no need to run
        {
            for(int i = 0; i < num_subgraph; ++i)
            {
                std::get<0>(r2r_graph[i]).emplace_back(offset[i]);
            }
        }
        else
        {
            start = vlist_ptr[vid];
            end = vlist_ptr[vid+1];
            for(eid = start; eid < end; ++eid)
            {
                dst = elist_ptr[eid];
                std::get<1>(r2r_graph[v_step]).emplace_back(dst);
                std::get<2>(r2r_graph[v_step]).emplace_back(value_ptr[eid]);
                offset[v_step]++;
            }
            std::get<0>(r2r_graph[v_step]).emplace_back(offset[v_step]);
            for(int i = 0; i < num_subgraph; ++i)
            {
                if(i == v_step) continue;
                std::get<0>(r2r_graph[i]).emplace_back(offset[i]);
            }
        }
    }

    //convert
    std::vector<std::tuple<py::array_t<VertexT>, py::array_t<VertexT>, py::array_t<float>>> np_r2r_graph(num_subgraph);
    
    for(int i = 0; i < num_subgraph; ++i)
    {
        std::get<0>(np_r2r_graph[i]) = vector2numpy1D(std::get<0>(r2r_graph[i]));
        std::get<1>(np_r2r_graph[i]) = vector2numpy1D(std::get<1>(r2r_graph[i]));
        std::get<2>(np_r2r_graph[i]) = vector2numpy1D(std::get<2>(r2r_graph[i]));
    }
    return np_r2r_graph;
    
}
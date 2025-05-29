// cz code
#pragma once
#include "Gorder.h"
// #include "Graph.h"

namespace Gorder {

void Graph::buildFromEdges(const std::vector<int> &src, const std::vector<int> &dst) {
    graph.clear();
    outedge.clear();
    inedge.clear();

    vsize = 0;
    edgenum = 0;

    for(int i = 0; i < src.size(); i++) {
        int src_node = src[i];
        int dst_node = dst[i];
        if(src_node == dst_node) continue;
        edgenum++;
        if(src_node >= vsize) vsize = src_node;
        if(dst_node >= vsize) vsize = dst_node;
    }
    vsize++;

    vector<pair<int, int>> edges;
    edges.reserve(edgenum);
    for(int i = 0; i < src.size(); i++) {
        if (src[i] != dst[i]){
            edges.push_back(std::make_pair(src[i], dst[i]));
        }
    }

    graph.resize(vsize+1);
    for(const auto &edge : edges){
        graph[edge.first].outdegree++;
        graph[edge.second].indegree++;
    }

    graph[0].outstart = 0;
    graph[0].instart = 0;
    for(int i = 1; i <= vsize; i++){
        graph[i].outstart = graph[i-1].outstart + graph[i-1].outdegree;
        graph[i].instart = graph[i-1].instart + graph[i-1].indegree;
    }

    sort(edges.begin(), edges.end(), 
        [](const pair<int, int>& a, const pair<int, int>& b) -> bool {
            if (a.first < b.first) return true;
            else if (a.first > b.first) return false;
            else return a.second <= b.second;
        });
    
    outedge.resize(edgenum);
    for (size_t i = 0; i < edges.size(); i++) {
        outedge[i] = edges[i].second;
    }

    vector<int> inpos(vsize);
    for (int i = 0; i < vsize; i++) {
        inpos[i] = graph[i].instart;
    }
    
    inedge.resize(edgenum);
    for (int u = 0; u < vsize; u++) {
        for (int j = graph[u].outstart; j < graph[u].outstart + graph[u].outdegree; j++) {
            inedge[inpos[outedge[j]]] = u;
            inpos[outedge[j]]++;
        }
    }

    graph[vsize].outstart = edgenum;
    graph[vsize].instart = edgenum;
    
    vector<pair<int, int>>().swap(edges);
}


void Graph::getReorderedEdges(const std::vector<int> &order, std::vector<int> &back_src, std::vector<int> &back_dst) {
    back_src.clear();
    back_dst.clear();
    back_src.reserve(edgenum);
    back_dst.reserve(edgenum);
    vector<vector<int>> ReOrderedGraph(vsize);
    int u, v;
    
    for(int i = 0; i < vsize; i++){
        u = order[i];
        ReOrderedGraph[u].reserve(graph[i+1].outstart - graph[i].outstart);
        for(int j = graph[i].outstart; j < graph[i+1].outstart; j++){
            v = order[outedge[j]];
            ReOrderedGraph[u].push_back(v);
        }
        sort(ReOrderedGraph[u].begin(), ReOrderedGraph[u].end());
    }

    for(int u = 0; u < vsize; u++){
        for(int j = 0; j < ReOrderedGraph[u].size(); j++){
            back_src.push_back(u);
            back_dst.push_back(ReOrderedGraph[u][j]);
        }
    }
    vector<vector<int>>().swap(ReOrderedGraph);
}

}

// return: src, dst, order
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
gorder_reorder(const std::vector<int> &src, const std::vector<int> &dst, int W) {
    Gorder::Graph g;
    g.buildFromEdges(src, dst);
    std::vector<int> first_order;
    g.Transform(first_order);



    std::vector<int> order;
    g.GorderGreedy(order, W);
    // std::vector<std::pair<int, int>> reordered_edges;_o
    std::vector<int> back_src, back_dst;
    g.getReorderedEdges(order, back_src, back_dst);
    std::vector<int> final_order(g.vsize);
    if(first_order.size() != g.vsize || order.size() != g.vsize){
        return {std::vector<int>(), std::vector<int>(), std::vector<int>()};
    }
    for(int i = 0; i < g.vsize; i++){
        if(first_order[i] >= g.vsize || order[i] >= g.vsize){
            std::cerr << "first_order or order is not valid" << std::endl;
            return {std::vector<int>(), std::vector<int>(), std::vector<int>()};
        }
        final_order[i] = order[first_order[i]];
    }
    
    

    return {back_src, back_dst, final_order};
}
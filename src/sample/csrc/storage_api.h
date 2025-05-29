#include <torch/extension.h> 
#include <stdexcept>
#include <vector>


int64_t get_neighbor(int64_t node_id, int64_t neighbor_index,
                     int64_t* v2v_vlist_ptr, int64_t* v2v_elist_ptr,
                     int64_t* v2r_vlist_ptr, int64_t* v2r_elist_ptr,
                     int64_t* r2v_vlist_ptr, int64_t* r2v_elist_ptr,
                     std::vector<int64_t*>& r2r_vlist_ptrs,
                     std::vector<int64_t*>& r2r_elist_ptrs,
                     int64_t* v_degree_ptr, int64_t* r_degree_ptr) {

    // search in v2v top down
    int64_t v2v_start = v2v_vlist_ptr[node_id];
    int64_t v2v_end = v2v_vlist_ptr[node_id + 1];
    int64_t v2v_count = v2v_end - v2v_start;
    if (neighbor_index < v2v_count) {
        return v2v_elist_ptr[v2v_start + neighbor_index];
    }
    // update neighbor index, now neighbor_index  = v2v_count + ...
    neighbor_index -= v2v_count;

    int64_t v2r_start = v2r_vlist_ptr[node_id];
    int64_t v2r_end = v2r_vlist_ptr[node_id + 1];
    int64_t v2r_count = v2r_end - v2r_start;

    for(int64_t i = 0; i < v2r_count; i++){
        int64_t rule_node = v2r_elist_ptr[v2r_start + i];
        int64_t degree = r_degree_ptr[rule_node];
        if (neighbor_index < degree) { //if in this rule, search it top down
            // search in r2v
            size_t layer = 0;
            while(layer < r2r_vlist_ptrs.size()){
                int64_t r2v_start = r2v_vlist_ptr[rule_node];
                int64_t r2v_end = r2v_vlist_ptr[rule_node + 1];
                int64_t r2v_count = r2v_end - r2v_start;
                if (neighbor_index < r2v_count) {
                    return r2v_elist_ptr[r2v_start + neighbor_index];
                }
                neighbor_index -= r2v_count;
                // continue push down
                int64_t r2r_start = r2r_vlist_ptrs[layer][rule_node];
                int64_t r2r_end = r2r_vlist_ptrs[layer][rule_node+1];
                int64_t r2r_count = r2r_end - r2r_start;
                for(int64_t r = 0; r < r2r_count; ++r){
                    int64_t cur_rule_node = r2r_elist_ptrs[layer][r2r_start+r];
                    int64_t cur_r_degree = r_degree_ptr[cur_rule_node];
                    if(neighbor_index < cur_r_degree){
                        rule_node = cur_rule_node;
                        break;
                    }
                    neighbor_index -= cur_r_degree;
                }
                layer++;
            }
            int64_t r2v_start = r2v_vlist_ptr[rule_node];
            int64_t r2v_end = r2v_vlist_ptr[rule_node + 1];
            int64_t r2v_count = r2v_end - r2v_start;
            if (neighbor_index < r2v_count) {
                return r2v_elist_ptr[r2v_start + neighbor_index];
            }
        }
        neighbor_index -= degree;
    }

    throw std::out_of_range("Neighbor index out of range");
}

std::vector<int64_t> get_neighbor_list(int64_t node_id, 
                    int64_t* v2v_vlist_ptr, int64_t* v2v_elist_ptr,
                    int64_t* v2r_vlist_ptr, int64_t* v2r_elist_ptr,
                    int64_t* r2v_vlist_ptr, int64_t* r2v_elist_ptr,
                    const std::vector<int64_t*>& r2r_vlist_ptrs,
                    const std::vector<int64_t*>& r2r_elist_ptrs){
    std::vector<int64_t> neighbor_list;
    int64_t v2v_start = v2v_vlist_ptr[node_id];
    int64_t v2v_end = v2v_vlist_ptr[node_id + 1];
    int64_t v2v_count = v2v_end - v2v_start;
    for(int64_t i = 0; i < v2v_count; i++){
        int64_t neighbor = v2v_elist_ptr[v2v_start + i];
        neighbor_list.push_back(neighbor);
    }
    int64_t v2r_start = v2r_vlist_ptr[node_id];
    int64_t v2r_end = v2r_vlist_ptr[node_id + 1];
    int64_t v2r_count = v2r_end - v2r_start;
    std::vector<int64_t> rule_nodes;
    for(int64_t i = 0; i < v2r_count; i++){
        int64_t rule_node = v2r_elist_ptr[v2r_start + i];
        rule_nodes.push_back(rule_node);
    }
    int64_t begin = 0, end = rule_nodes.size();
    for(size_t r2r_idx = 0; r2r_idx < r2r_vlist_ptrs.size(); r2r_idx++){
        for(int64_t i = begin; i < end; i++){
            int64_t rule_node = rule_nodes[i];
            int64_t r2r_start = r2r_vlist_ptrs[r2r_idx][rule_node];
            int64_t r2r_end = r2r_vlist_ptrs[r2r_idx][rule_node + 1];
            int64_t r2r_count = r2r_end - r2r_start;
            for(int64_t j = 0; j < r2r_count; j++){
                int64_t neighbor = r2r_elist_ptrs[r2r_idx][r2r_start + j];
                rule_nodes.push_back(neighbor);
            }
        }
        begin = end;
        end = rule_nodes.size();
    }
    for(int64_t i = 0; i < rule_nodes.size(); i++){
        int64_t rule_node = rule_nodes[i];
        int64_t r2v_start = r2v_vlist_ptr[rule_node];
        int64_t r2v_end = r2v_vlist_ptr[rule_node + 1];
        int64_t r2v_count = r2v_end - r2v_start;
        for(int64_t j = 0; j < r2v_count; j++){
            int64_t neighbor = r2v_elist_ptr[r2v_start + j];
            neighbor_list.push_back(neighbor);
        }
    }
    return neighbor_list;
}

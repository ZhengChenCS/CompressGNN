#include <torch/extension.h>
#include <vector>
#include "../../../third_party/parallel-hashmap/parallel_hashmap/phmap.h"
#include "storage_api.h"

// int64_t uniform_randint(int64_t max_val){
//     return rand() % max_val;
// }

inline int64_t uniform_randint(int64_t high) {
  auto options = torch::TensorOptions().dtype(torch::kInt64);
  auto ret = torch::randint(0, high, {1}, options);
  auto ptr = ret.data_ptr<int64_t>();
  return *ptr;
}

template <typename scalar_t>
inline torch::Tensor from_vector(const std::vector<scalar_t> &vec,
                                 bool inplace = false) {
  const auto size = (int64_t)vec.size();
  const auto out = torch::from_blob((scalar_t *)vec.data(), {size},
                                    c10::CppTypeToScalarType<scalar_t>::value);
  return inplace ? out : out.clone();
}
// neighbor: v2v, v2r->r2r->r2v, v2r->r2v
// return: output_node, row, col
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
neighbor_sample_sim_impl(const torch::Tensor v2v_vlist, const torch::Tensor v2v_elist, 
                    const torch::Tensor v2r_vlist, const torch::Tensor v2r_elist,
                    const torch::Tensor r2v_vlist, const torch::Tensor r2v_elist,
                    const std::vector<torch::Tensor> r2r_vlist, const std::vector<torch::Tensor> r2r_elist,
                    const torch::Tensor v_degree, const torch::Tensor r_degree,
                    const torch::Tensor &input_node,
                    const std::vector<int64_t> num_neighbors)
{
    // Initialize data structures for the sampling process
    std::vector<int64_t> samples;
    phmap::flat_hash_map<int64_t, int64_t> to_local_node;

    auto *v2v_vlist_ptr = v2v_vlist.data_ptr<int64_t>();
    auto *v2v_elist_ptr = v2v_elist.data_ptr<int64_t>();
    auto *v2r_vlist_ptr = v2r_vlist.data_ptr<int64_t>();
    auto *v2r_elist_ptr = v2r_elist.data_ptr<int64_t>();
    auto *r2v_vlist_ptr = r2v_vlist.data_ptr<int64_t>();
    auto *r2v_elist_ptr = r2v_elist.data_ptr<int64_t>();
    auto *v_degree_ptr = v_degree.data_ptr<int64_t>();
    auto *r_degree_ptr = r_degree.data_ptr<int64_t>();
    auto *input_node_data = input_node.data_ptr<int64_t>();

    // Initialize vectors to store r2r_vlist and r2r_elist pointers
    std::vector<int64_t*> r2r_vlist_ptrs;
    std::vector<int64_t*> r2r_elist_ptrs;

    // Convert each element in r2r_vlist to a data pointer and store it in r2r_vlist_ptrs
    for (const auto& tensor : r2r_vlist) {
        r2r_vlist_ptrs.push_back(tensor.data_ptr<int64_t>());
    }

    // Convert each element in r2r_elist to a data pointer and store it in r2r_elist_ptrs
    for (const auto& tensor : r2r_elist) {
        r2r_elist_ptrs.push_back(tensor.data_ptr<int64_t>());
    }

    int64_t vertex_cnt = v2v_vlist.size(0);
    int64_t rule_cnt = r2v_vlist.size(0);

    for(int64_t i = 0; i < input_node.numel(); i++){
        const auto &v = input_node_data[i];
        samples.push_back(v);
        to_local_node.insert({v, i});
    }

    std::vector<int64_t> rows, cols;

    int64_t begin = 0, end = samples.size();

    for(int64_t ell = 0; ell < (int64_t)num_neighbors.size(); ell++){
        const auto &num_samples = num_neighbors[ell];
        
        for(int64_t i = begin; i < end; i++){
            const int64_t w = samples[i];
            int64_t degree = v_degree_ptr[w];
            if(degree == 0){
                continue;
            }
            if(num_samples >= degree){
                auto neighbors = get_neighbor_list(w, 
                v2v_vlist_ptr, v2v_elist_ptr, 
                v2r_vlist_ptr, v2r_elist_ptr, 
                r2v_vlist_ptr, r2v_elist_ptr, 
                r2r_vlist_ptrs, r2r_elist_ptrs);
                for(auto neighbor : neighbors){
                    const auto res = to_local_node.insert({neighbor, samples.size()});
                    if(res.second){
                        samples.push_back(neighbor);
                    }
                    rows.push_back(i);
                    cols.push_back(res.first->second);
                }
            }
            else{
                std::unordered_set<int64_t> rnd_indices;
                for(int64_t j = degree - num_samples; j < degree; j++){
                    int64_t rnd = uniform_randint(j);
                    if(!rnd_indices.insert(rnd).second){
                        rnd = j;
                        rnd_indices.insert(j);
                    }
                    const int64_t neighbor = get_neighbor(w, rnd, 
                    v2v_vlist_ptr, v2v_elist_ptr, 
                    v2r_vlist_ptr, v2r_elist_ptr, 
                    r2v_vlist_ptr, r2v_elist_ptr, 
                    r2r_vlist_ptrs, r2r_elist_ptrs,
                    v_degree_ptr, r_degree_ptr);
                    const auto res = to_local_node.insert({neighbor, samples.size()});
                    if(res.second){
                        samples.push_back(neighbor);
                    }
                    rows.push_back(i);
                    cols.push_back(res.first->second);
                }
            }  
        }
        begin = end, end =samples.size();
    }
    return std::make_tuple(from_vector<int64_t>(samples), 
                        from_vector<int64_t>(rows), 
                        from_vector<int64_t>(cols));
    
}

// Returns 'output_node', 'row', 'col'
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
neighbor_sample_ori_impl(const torch::Tensor &colptr, const torch::Tensor &row,
                const torch::Tensor &input_node,
                const std::vector<int64_t> num_neighbors)
{
    
    // Initialize data structures for the sampling process
    std::vector<int64_t> samples;
    phmap::flat_hash_map<int64_t, int64_t> to_local_node;

    auto *colptr_data = colptr.data_ptr<int64_t>();
    auto *row_data = row.data_ptr<int64_t>();
    auto *input_node_data = input_node.data_ptr<int64_t>();

    // Insert input nodes into the sample and to_local_node map
    for (int64_t i = 0; i < input_node.numel(); i++) {
        const auto &v = input_node_data[i];
        samples.push_back(v);
        to_local_node.insert({v, i});
    }

    std::vector<int64_t> rows, cols, edges;

    int64_t begin = 0, end = samples.size();

    // Iterate over each layer (num_neighbors size)
    for (int64_t ell = 0; ell < (int64_t)num_neighbors.size(); ell++) {
        const auto &num_samples = num_neighbors[ell];

        for (int64_t i = begin; i < end; i++) {
            const auto &w = samples[i];
            const auto &col_start = colptr_data[w];
            const auto &col_end = colptr_data[w + 1];
            const auto col_count = col_end - col_start;

            if (col_count == 0)
                continue;

            // Since replace == false, we sample without replacement
            if (num_samples >= col_count) {
                
                // Take all neighbors since num_samples >= col_count
                for (int64_t offset = col_start; offset < col_end; offset++) {
                    const int64_t &v = row_data[offset];
                    const auto res = to_local_node.insert({v, samples.size()});
                    if (res.second) {
                        samples.push_back(v);  // Add new sample
                    }
                    cols.push_back(i);
                    rows.push_back(res.first->second);
                    edges.push_back(offset);
                }
            } else {
                // Sample without replacement when num_samples < col_count
                std::unordered_set<int64_t> rnd_indices;
                for (int64_t j = col_count - num_samples; j < col_count; j++) {
                    int64_t rnd = uniform_randint(j);
                    if (!rnd_indices.insert(rnd).second) {
                        rnd = j;
                        rnd_indices.insert(j);
                    }
                    const int64_t offset = col_start + rnd;
                    const int64_t &v = row_data[offset];
                    const auto res = to_local_node.insert({v, samples.size()});
                    if (res.second) {
                        samples.push_back(v);  // Add new sample
                    }
                    cols.push_back(i);
                    rows.push_back(res.first->second);
                    edges.push_back(offset);
                }
            }
        }
        begin = end, end = samples.size();
    }

    // Return the sampled results
    return std::make_tuple(from_vector<int64_t>(samples), 
                            from_vector<int64_t>(rows),
                           from_vector<int64_t>(cols));
}
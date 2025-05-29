import torch
import numpy as np
import dgl
from dgl.distributed import partition_graph
import os
# from ogb.nodeproppred import DglNodePropPredDataset, PygNodePropPredDataset

def partition_and_save(g, num_parts, save_path):
    """
    对图进行分区并保存
    
    Args:
        g: DGL图
        num_parts: 分区数量
        save_path: 保存路径
    """
    # 1. 首先使用 METIS 获取节点分区分配
    node_parts = dgl.metis_partition_assignment(g, num_parts)
    
    # 2. 使用节点分区信息进行图分区
    graphs = dgl.partition_graph_with_halo(
        g,                     # 输入图
        node_parts,           # 节点分区分配
        1,                    # halo hops
        reshuffle=True       # 重新排列节点
    )
    
    # 3. 保存分区
    os.makedirs(save_path, exist_ok=True)
    for part_id, subg in enumerate(graphs):
        # 保存子图
        dgl.save_graphs(
            os.path.join(save_path, f'part{part_id}.dgl'),
            [subg]
        )
        
        print(f"\n分区 {part_id}:")
        print(f"- 节点数: {subg.num_nodes()}")
        print(f"- 边数: {subg.num_edges()}")
    
    return graphs

def load_and_process_partitions(save_path, num_parts):
    """加载分区后的图"""
    parts = []
    for part_id in range(num_parts):
        # 加载子图
        subg, _ = dgl.load_graphs(
            os.path.join(save_path, f'part{part_id}.dgl')
        )
        parts.append(subg[0])
        
        # 加载映射关系
        mapping = torch.load(
            os.path.join(save_path, f'part{part_id}_mapping.pt')
        )
        print(f"Part {part_id}:")
        print(f"- Nodes: {subg[0].num_nodes()}")
        print(f"- Edges: {subg[0].num_edges()}")
    
    return parts

def sequential_partition(g, num_parts):
    """
    按照节点ID顺序进行简单的分区
    
    Args:
        g: DGL图
        num_parts: 分区数量
    Returns:
        parts: 列表，包含每个分区的节点ID
    """
    num_nodes = g.num_nodes()
    nodes_per_part = num_nodes // num_parts  # 每个分区的基本节点数
    remaining = num_nodes % num_parts        # 剩余的节点数
    
    parts = []
    start_idx = 0
    
    for i in range(num_parts):
        # 如果有剩余节点，则多分配一个节点
        part_size = nodes_per_part + (1 if i < remaining else 0)
        end_idx = start_idx + part_size
        
        # 获取这个分区的节点
        part_nodes = torch.arange(start_idx, end_idx)
        parts.append(part_nodes)
        
        start_idx = end_idx
    
    return parts

def save_partitions(g, parts, save_path):
    """
    保存分区后的子图
    
    Args:
        g: 原始DGL图
        parts: 分区节点列表
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    for i, part_nodes in enumerate(parts):
        # 创建子图
        subg = dgl.node_subgraph(g, part_nodes)
        
        # 保存子图
        save_file = os.path.join(save_path, f'part{i}.dgl')
        dgl.save_graphs(save_file, [subg])
        
        # 打印信息
        print(f"\n分区 {i}:")
        print(f"节点数: {subg.num_nodes()}")
        print(f"边数: {subg.num_edges()}")
        print(f"节点ID范围: {part_nodes[0].item()} - {part_nodes[-1].item()}")

if __name__ == "__main__":
    path = "/mnt/disk1/GNNDataset/OGB/dataset/ogbn_papers100M"
    
    datapath = f"{path}/raw/data.npz"
    data = np.load(datapath)
    '''
    node_feat
    edge_index
    num_nodes
    num_edges
    node_year
    '''
    # for key in data.files:
    #     print(key)
    # print(data)
    # print(data['node_feat'].shape)
    print(data['edge_index'].shape)
    # print(data['num_nodes'])
    # print(data['num_edges'])

    # 2. 创建DGL图
    g = dgl.graph((data['edge_index'][0], data['edge_index'][1]))
    
    # 3. 如果有节点特征，添加到图中
    if 'node_feat' in data:
        g.ndata['feat'] = torch.from_numpy(data['node_feat'])
    
    # 4. 进行分区
    num_parts = 8  # 分成8个部分
    save_path = './graph_parts'
    
    print("原始图信息:")
    print(f"节点数: {g.num_nodes()}")
    print(f"边数: {g.num_edges()}")
    
    # 5. 顺序分区
    print("\n开始分区...")
    parts = sequential_partition(g, num_parts)
    
    # 6. 保存分区
    print("\n保存分区...")
    save_partitions(g, parts, save_path)
    
    # 7. 验证分区结果
    total_nodes = sum(len(part) for part in parts)
    print(f"\n验证:")
    print(f"总节点数: {total_nodes}")
    print(f"原始节点数: {g.num_nodes()}")
    assert total_nodes == g.num_nodes(), "节点数不匹配！"

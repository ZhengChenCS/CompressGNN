import torch as th
import dgl
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader
import time
import sys
sys.path.append("../loader")

def sample(path):
    device = th.device("cuda")
    g_list, _ = dgl.load_graphs(path)
    g = g_list[0]
    # print(g)
    nfeat = g.ndata.pop("feat").to(device)
    labels = g.ndata['label'].to(device)

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    train_idx = th.nonzero(train_mask, as_tuple=True)[0]
    val_idx = th.nonzero(val_mask, as_tuple=True)[0]
    test_idx = th.nonzero(test_mask, as_tuple=True)[0]

    num_neighbors = [20, 10, 5]
    sampler = MultiLayerNeighborSampler(num_neighbors)

    # 创建数据加载器
    batch_size = 1024
    num_workers = 1  # 根据机器配置调整
    dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    start = time.perf_counter()
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        # 模拟训练步骤
        print("hello")
        break
    end = time.perf_counter()
    print("Total sampling time: {:.2f}s".format(end - start))

def main():

    dgl_data = "/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/origin/dgl_data.bin"
    
    compress_data = "/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/compress/dgl_data.bin"

    print("DGL Data")
    sample(dgl_data)
    print("CompressGNN Data")
    sample(compress_data)

    

    

if __name__ == '__main__':
    main()
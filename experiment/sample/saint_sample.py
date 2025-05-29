import torch as th
import dgl
from dgl.dataloading import MultiLayerNeighborSampler
from torch.utils.data import DataLoader
import time
import sys
from saint_sampler import SAINTEdgeSampler, SAINTNodeSampler, SAINTRandomWalkSampler
sys.path.append("../loader")

def sample(path, dataset_names):
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

    # num_neighbors = [20, 10, 5]
    kwargs = {
        "dn": dataset_names,
        "g": g,
        "train_nid": train_idx,
        "num_workers_sampler": 0
    }
    # saint_sampler = SAINTNodeSampler(2048, **kwargs)
    saint_sampler = SAINTEdgeSampler(2048, **kwargs)
    # sampler = MultiLayerNeighborSampler(num_neighbors)

    # 创建数据加载器
    batch_size = 1024
    num_workers = 1  # 根据机器配置调整
    dataloader = DataLoader(
        saint_sampler,
        collate_fn=saint_sampler.__collate_fn__,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )

    start = time.perf_counter()
    for step, subg in enumerate(dataloader):
        pass
        # print(step)
        # # break
    end = time.perf_counter()
    print("Total sampling time: {:.2f}s".format(end - start))

def main():

    dgl_data = "/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/origin/dgl_data.bin"
    
    compress_data = "/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/compress/dgl_data.bin"

    print("DGL Data")
    sample(dgl_data, "ogbn-products")
    print("CompressGNN Data")
    sample(compress_data, "ogbn-products-compress")

    

    

if __name__ == '__main__':
    main()
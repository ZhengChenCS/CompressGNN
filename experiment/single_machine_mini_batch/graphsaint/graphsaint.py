import torch as th
import dgl
from dgl.dataloading import MultiLayerNeighborSampler
from torch.utils.data import DataLoader
import time
import sys
from sampler import SAINTEdgeSampler, SAINTNodeSampler, SAINTRandomWalkSampler
sys.path.append("../loader")
import argparse
from model import GCNNet
import torch.nn.functional as F
import torch

args = argparse.ArgumentParser()
args.add_argument("--data", type=str, default="")
args.add_argument("--num-epochs", type=int, default=10)
args.add_argument("--gpu", type=int, default=0)
args.add_argument("--dataset-names", type=str, default="")
args = args.parse_args()

def train(path, dataset_names):
    device = th.device("cuda")
    g_list, _ = dgl.load_graphs(path)
    g = g_list[0]
    # print(g)
    nfeat = g.ndata["feat"]
    labels = g.ndata['label']

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

    train_idx = train_idx.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    in_feats = nfeat.shape[1]
    n_classes = labels.max().item() + 1
    hidden = 128
    arch = "1-1-0"
    dropout = 0
    batch_norm = False
    aggr = "concat"

    model = GCNNet(
        in_dim=in_feats,
        hid_dim=hidden,
        out_dim=n_classes,
        arch=arch,
        dropout=dropout
    )

    model.to(device)



    start = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(args.num_epochs):
        epoch_start = time.perf_counter()
        for step, subg in enumerate(dataloader):
            # forward
            subg = subg.to(device)
            model.train()
            pred = model(subg)
            batch_labels = subg.ndata["label"]
            loss = F.cross_entropy(pred, batch_labels, reduction="none")
            loss = (subg.ndata["l_n"] * loss).sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()
        epoch_end = time.perf_counter()
        print(f"Epoch {epoch} time: {epoch_end - epoch_start:.2f}s")

        model.eval()
        with th.no_grad():
            g = g.to(device)
            pred = model(g)
            pred = pred.to("cpu")
            test_label = labels[test_idx]
            test_pred = pred[test_idx]
            acc = (th.argmax(test_pred, dim=1) == test_label).float().sum() / len(test_pred)
            print(f"Epoch {epoch} test accuracy: {acc:.4f}")
            
    end = time.perf_counter()
    print("Total training time: {:.2f}s".format(end - start))

def main():
    train(args.data, args.dataset_names)

    

    

if __name__ == '__main__':
    main()
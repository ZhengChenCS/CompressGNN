import torch 
import numpy as np
import dgl


path = "/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/origin"

feature = np.load(f"{path}/features.npy")
label = np.load(f"{path}/labels.npy")
edge_index = np.load(f"{path}/edge.npy")
train_mask = np.load(f"{path}/train_mask.npy")
val_mask = np.load(f"{path}/val_mask.npy")
test_mask = np.load(f"{path}/test_mask.npy")

feature = torch.tensor(feature, dtype=torch.float)
label = torch.tensor(label, dtype=torch.long)
edge_index = torch.tensor(edge_index, dtype=torch.long)
train_mask = torch.tensor(train_mask, dtype=torch.bool)
val_mask = torch.tensor(val_mask, dtype=torch.bool)
test_mask = torch.tensor(test_mask, dtype=torch.bool)

g = dgl.graph((edge_index[0], edge_index[1]))


g.ndata['feat'] = feature
g.ndata['label'] = label
g.ndata['train_mask'] = train_mask
g.ndata['val_mask'] = val_mask
g.ndata['test_mask'] = test_mask

print(g)

dgl.save_graphs(f"{path}/dgl_data.bin", g)





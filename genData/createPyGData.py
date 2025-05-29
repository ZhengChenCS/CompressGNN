import torch 
from torch_geometric.data import Data
import numpy as np
from torch_sparse import SparseTensor



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

edge_index = SparseTensor.from_edge_index(edge_index)


data = Data(
    x=feature,
    y=label,
    adj_t=edge_index,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    num_nodes=feature.shape[0],
    num_classes=label.max().item() + 1
)

print(data)

torch.save(data, f"{path}/pyg_data_csr.pt")




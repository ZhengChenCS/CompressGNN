import torch 
import dgl
import numpy as np
import compressgnn_offline 

def reorder_data(edge_index):
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    print("start reorder")
    try:
        src, dst, order = compressgnn_offline.reorder(src, dst, 5)
        src = np.array(src)
        dst = np.array(dst)
        order = np.array(order)
        print("reorder done")
        return src, dst, order
    except Exception as e:
        print(e)
        return None, None, None

def compress(vlist, elist, threshold=4, max_depth=8, min_edge=100000):
    new_vlist, new_elist, new_vertex_cnt, new_rule_cnt = compressgnn_offline.compress_csr(
        vlist, elist, vlist.shape[0]-1)
    print(
        "compression ratio: {} / {} = {:.4f}".format(
            vlist.shape[0] +
            elist.shape[0],
            new_vlist.shape[0] +
            new_elist.shape[0], (vlist.shape[0] + elist.shape[0]) / (new_vlist.shape[0] + new_elist.shape[0])))
    new_vlist, new_elist, new_vertex_cnt, new_rule_cnt = compressgnn_offline.filter_csr(
        new_vlist, new_elist, new_vertex_cnt, new_rule_cnt, threshold)
    print(
        "After filter compression ratio: {} / {} = {:.4f}".format(
            vlist.shape[0] +
            elist.shape[0],
            new_vlist.shape[0] +
            new_elist.shape[0], (vlist.shape[0] + elist.shape[0]) / (new_vlist.shape[0] + new_elist.shape[0])))
    new_vlist, new_elist, new_vertex_cnt, new_rule_cnt = compressgnn_offline.depth_filter_csr(
        new_vlist, new_elist, new_vertex_cnt, new_rule_cnt, max_depth, min_edge)
    print(
        "After depth filter compression ratio: {} / {} = {:.4f}".format(
            vlist.shape[0] +
            elist.shape[0],
            new_vlist.shape[0] +
            new_elist.shape[0], (vlist.shape[0] + elist.shape[0]) / (new_vlist.shape[0] + new_elist.shape[0])))
    return new_vlist, new_elist, new_vertex_cnt, new_rule_cnt

path = "/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/origin"

feature = np.load(f"{path}/features.npy")
label = np.load(f"{path}/labels.npy")
edge_index = np.load(f"{path}/edge.npy")
train_mask = np.load(f"{path}/train_mask.npy")
val_mask = np.load(f"{path}/val_mask.npy")
test_mask = np.load(f"{path}/test_mask.npy")

src, dst, order = compressgnn_offline.reorder(edge_index[0].tolist(), edge_index[1].tolist(), 5)
order = np.argsort(order)

feature = feature[order]
label = label[order]
train_mask = train_mask[order]
val_mask = val_mask[order]
test_mask = test_mask[order]

vlist, elist = compressgnn_offline.coo2csr(src, dst, feature.shape[0])
vlist, elist, vertex_cnt, rule_cnt = compress(vlist, elist, threshold=4, max_depth=8, min_edge=100000)
src, dst = compressgnn_offline.csr2coo(vlist, elist)
edge_index = np.array([src, dst])

# generate rule data 
rule_label = np.zeros(rule_cnt)
rule_train_mask = np.zeros(rule_cnt)
rule_val_mask = np.zeros(rule_cnt)
rule_test_mask = np.zeros(rule_cnt)
rule_feature = np.zeros((rule_cnt, feature.shape[1]))

feature = np.vstack((feature, rule_feature))
label = np.hstack((label, rule_label))
train_mask = np.hstack((train_mask, rule_train_mask))
val_mask = np.hstack((val_mask, rule_val_mask))
test_mask = np.hstack((test_mask, rule_test_mask))

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

dgl.save_graphs(f"/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/compress/dgl_data.bin", g)
import os
import sys
sys.path.append("../loader")
import getopt
import torch
import numpy as np
import torch_geometric
from origin_data import Data



def main():
    data_dir = sys.argv[1]
    graph_dir = sys.argv[1] + "/origin"
    graph_type = sys.argv[2]
    train_mask_path = graph_dir + "/train_mask.npy"
    vaild_mask_path = graph_dir + "/val_mask.npy"
    test_mask_path = graph_dir + "/test_mask.npy"

    x = np.load(graph_dir + "/features.npy")
    y = np.load(graph_dir + "/labels.npy")
    if y.ndim == 2:
        y = [np.argmax(row_y) for row_y in y]
    y = np.array(y)
    vertex_cnt = y.size
    if not os.path.exists(train_mask_path):
        train_mask = np.ones(vertex_cnt, dtype=bool)
        vaild_mask = np.ones(vertex_cnt, dtype=bool)
        test_mask = np.ones(vertex_cnt, dtype=bool)
    else:
        train_mask = np.load(train_mask_path)
        vaild_mask = np.load(vaild_mask_path)
        test_mask = np.load(test_mask_path)
    if graph_type == "coo":
        edge_index = np.load(graph_dir + "/edge.npy")
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            vaild_mask=vaild_mask,
            test_mask=test_mask,
            graph_type="coo")
    elif graph_type == "csr":
        vlist = np.load(graph_dir + "/csr_vlist.npy")
        elist = np.load(graph_dir + "/csr_elist.npy")
        data = Data(
            x=x,
            y=y,
            edge_index=(
                vlist,
                elist),
            train_mask=train_mask,
            vaild_mask=vaild_mask,
            test_mask=test_mask,
            graph_type="csr")
    print(data)
    torch.save(data, graph_dir + "/data_" + graph_type + ".pt")


if __name__ == '__main__':
    main()

import os
import sys
sys.path.append("../loader/")
from CompressgnnData import CompressgnnData
from origin_data import Data
import getopt
import torch
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

def main():
    data_dir = sys.argv[1] + "/origin"
    graph_dir = sys.argv[1] + "/origin"
    save_dir = sys.argv[2]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    graph_type = sys.argv[3]
    train_mask_path = data_dir + "/train_mask.npy"
    vaild_mask_path = data_dir + "/val_mask.npy"
    test_mask_path = data_dir + "/test_mask.npy"

    x = np.load(data_dir + "/features.npy")
    y = np.load(data_dir + "/labels.npy")
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
    edge_index = np.load(graph_dir + "/edge.npy")
    src, dst, order = reorder_data(edge_index)
    order = np.argsort(order)
    np.save("order.npy", order)

    # gene reordered data
    x = x[order]
    y = y[order]
    train_mask = train_mask[order]
    vaild_mask = vaild_mask[order]
    test_mask = test_mask[order]

    if graph_type == "coo":
        eedge_index = (src, dst)
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            vaild_mask=vaild_mask,
            test_mask=test_mask,
            graph_type="coo")
    elif graph_type == "csr":
        v_cnt = max(src.max(), dst.max()) + 1
        vlist, elist = compressgnn_offline.coo2csr(src, dst, v_cnt)
        data = Data(
            x=x,
            y=y,
            edge_index=(
                vlist,
                elist),
            train_mask=train_mask,
            vaild_mask=vaild_mask,
            test_mask=test_mask,
            graph_type="csr"
        )
    
    print(data)
    torch.save(data, f"{save_dir}/data_{graph_type}_gorder.pt")


if __name__ == '__main__':
    main()
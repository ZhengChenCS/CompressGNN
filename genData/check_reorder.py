import os
import sys
sys.path.append("../loader/")
from CompressgnnData import CompressgnnData
from origin_data import Data
import getopt
import torch
import numpy as np
import compressgnn_offline
from torch_sparse import SparseTensor, matmul
from torch import Tensor



def spmm(A: SparseTensor, B: Tensor):
    vlist, elist, value = A.csr()
    return matmul(A, B, reduce="add")

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
    origin = torch.load("/mnt/disk1/GNNDataset/Reddit/origin/data_csr.pt")
    reorder_data = torch.load("/mnt/disk1/GNNDataset/Reddit/origin/data_csr_gorder.pt")
    order = np.load("order.npy")
    reorder = np.argsort(order)


    origin_c = spmm(origin.edge_index, origin.x)
    reorder_c = spmm(reorder_data.edge_index, reorder_data.x)

    print(origin_c[0][0], origin.y[0], origin.train_mask[0])
    
    print(reorder_c[reorder[0]][0], reorder_data.y[reorder[0]], reorder_data.train_mask[reorder[0]])
                      
if __name__ == '__main__':
    main()
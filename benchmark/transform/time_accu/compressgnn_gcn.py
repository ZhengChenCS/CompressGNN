import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import math
from torch_geometric.loader import DataLoader
import shutil
import logging
import time
import argparse
import numpy as np
import sys
from tqdm import tqdm
sys.path.append("../../../layer/")
sys.path.append("../../../loader/")
sys.path.append("../../../model")
sys.path.append("../../../model/compressgnn")

from sgc_loader import SGCData, SGCDataLoader
from gcn_trans import GCN

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamps

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input graph data")
parser.add_argument('--num_features', type=int, required=True, help="Input num of features")
parser.add_argument('--num_classes', type=int, required=True, help="Input num of classes")
parser.add_argument('--epochs', type=int, required=True, help="Input num of epoch")
parser.add_argument('--device', type=str, default='cpu', help="device")
parser.add_argument('--deltaH', type=int, default=0, help="delta H")
parser.add_argument('--reset', type=int, default=10, help="reset step")
parser.add_argument('--cache', action='store_true', default=True, help="cache")
parser.add_argument('--index_cache', action='store_true', default=False, help="index_cache")
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

def test(model, data, device):
    model = model.to(device)
    data = data.to(device)
    model.eval()
    _, prediction = model.inference(data.x, data.edge_index).max(dim=1)
    target = data.y.squeeze()

    test_correct = prediction[data.test_mask].eq(target[data.test_mask]).sum().item()
    test_number = data.test_mask.sum().item()

    logger.info("Accuracy of Test Samples: {:.2f}%".format( (test_correct / test_number * 100) ))
    print("Accuracy of Test Samples: {:.2f}%".format( (test_correct / test_number * 100) ))



def train_runtime(model, data, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    data = data.to(device)
    mask = data.train_mask.to(device)
    model.train()
    y = data.y[mask].squeeze()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    if data.graph_type == "coo":
        for epoch in tqdm(range(epochs)):
            if epoch % args.reset == 0:
                model.reset_cache()
            pred = model(data.x, data.edge_index, data.edge_weight, data.vertex_cnt, data.rule_cnt)
            loss = F.cross_entropy(pred[mask], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        for epoch in tqdm(range(epochs)):
            if epoch % args.reset == 0:
                model.reset_cache()
            pred = model(data.x, data.edge_index)
            loss = F.cross_entropy(pred[mask], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                test(model, dataset, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    return t_end - t_start



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = torch.load(args.data)
    # logger.info("Dataset=%s", args.data)
    # logger.info("Num of features=%s", args.num_features)
    # logger.info("Num of classes=%s", args.num_classes)
    logger.info(f"cache is {args.cache}, index_cache is {args.index_cache}, reset is {args.reset}")
    dataset.x = dataset.x.to(torch.float32)
    param_H = int(math.log(dataset.x.size()[0], 2))+args.deltaH
    param_H = param_H if param_H < 30 else 30

    model = GCN(in_features=args.num_features, out_features=args.num_classes, param_H=param_H,_cache=args.cache,_index_cache=args.index_cache)

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    warm_up_time = train_runtime(model, dataset, epochs=10, device=device)
    
    logger.info("Train time for {:d} epoch: {:.4f}s".format(10, warm_up_time))
    model.reset_cache()

    train_time = train_runtime(model, dataset, epochs=args.epochs, device=device)
    logger.info("Train time for {:d} epoch: {:.4f}s".format(args.epochs, train_time))
    print("Train time for {:d} epoch: {:.4f}s".format(args.epochs, train_time))
    
    test(model, dataset, device)
    logger.info("-------------------------------------------------------------------")
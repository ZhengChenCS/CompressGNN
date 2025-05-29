import sys
import os
import argparse
import torch
sys.path.append("../loader")
from origin_data import Data

import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input data")
parser.add_argument('--scale_factor', type=int, required=True, help="Dim of feature")
parser.add_argument('--output', type=str, required=True, help="Output data")
args = parser.parse_args()


if __name__ == '__main__':
    dataset = torch.load(args.data)
    features = np.random.randint(low=0, high=2, size=(dataset.vertex_cnt, args.scale_factor), dtype=np.compat.long)
    features = features.astype(np.float32)
    x = torch.from_numpy(features)
    dataset.x = x
    dataset.feature_dim = args.scale_factor
    torch.save(dataset, args.output)
    
    
    
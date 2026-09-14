# CompressGNN
CompressGNN is a framework for accelerating graph neural network (GNN) training through hierarchical compression. It reduces redundant neighbor aggregation by sharing graph propagation computations and reduces repeated feature transformations by grouping similar node features.

The project provides graph preprocessing tools, optimized CPU/GPU operators, and PyTorch model integrations. Propagation and transformation compression can be used independently or together, with configurable clustering reuse to explore training-time, memory, and accuracy trade-offs.

## Code Structure

The project's code base is organized in the following directory structure.

```shell
.
├── benchmark
├── dataset
├── experiment
├── genData
├── install.sh
├── layer
├── LICENSE
├── loader
├── model
├── README.md
├── requirements.txt
├── src
├── test
└── third_party
```

## Installaction

To use this repository, please follow the steps below to install the required dependencies.

### Prerequisites

- Python (version 3.8.13)
- pip (version 23.0.1)
- cuda (version 11.6)

### Installing Dependencies

1. Clone the repository

2. Navigate to the project directory

3. Install the required dependencies using pip:

```shell
pip install -r requirements.txt
```
This command will install all the necessary libraries and packages, including:

- numpy
- pytorch
- Pytorch Geometric (PyG)
- Deep Graph Library (DGL)
- torch_scatter
- torch_sparse
- pybind
- ...

4. Install CompressGNN

```shell
bash install.sh
```

## Data Preparation

### Download Dataset

We have uploaded the small dataset `Cora` and `cnr-2000`. 
Users can generate a dataset in a format that meets our data specifications from [WebGraph](https://webgraph.di.unimi.it/) and [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric.git).



### Input Data Format

```shell
.
├── csr_elist.npy
├── csr_vlist.npy
├── edge.npy
├── features.npy
├── labels.npy
├── test_mask.npy
├── train_mask.npy
└── val_mask.npy
```

### Generate Torch Format Dataset

```shell
cd genData
python createTorchDataset.py <input data folder> <output data folder> coo/csr
```

### Generate Compressed Torch Fromat Dataset

```shell
cd genData
python createCompressDataset <input data folder> <output data folder> coo/csr 
```

### Generate datasets with different feature lengths

```shell
cd genData
python datagen_feature.py --data=xxx.pt --scale_factor=length --output=xxx.pt
```

### Generate dataset using scripts.

```
cd genData
bash preprocess.sh
bash preprocess_compress.sh
bash datagen_feature.sh
```

## Run Evaluation


### End-to-end Performance

```shell
cd benchmark/end2end
bash run.sh
```

### Propagate Performance

- Speedup

```shell
cd benchmark/propagate/propagate
bash speedup.sh
```

- Performance with different feature dimension

```shell
cd benchmark/propagate/propagate
bash feature_scale.sh
```

- Peak memory

```shell
cd benchmark/propagate/peak_memory
bash run.sh
```

### Transformation Performance

- Time and accuracy

```shell
cd benchmark/transform/time_accu
bash run.sh
```

- Time breakdown

```shell
cd benchmark/transform/time_breakdown
bash run.sh
```








## Transformation compression and epoch reuse

Rebuild the runtime extension after updating the Python layers and CUDA sources.
Use the CUDA toolkit matching the installed PyTorch build (the tested environment
uses PyTorch 1.13.1+cu116 and CUDA 11.6).

```python
from CompressgnnCluster import Compressgnn_Cluster
from CompressgnnReconstruct import Compressgnn_Reconstruct

cluster = Compressgnn_Cluster(in_feature=128, param_H=16,
                             refresh_interval=5).cuda()
reconstruct = Compressgnn_Reconstruct()
# At the beginning of each zero-based training epoch:
cluster.set_epoch(epoch)
representatives, index = cluster(features, cache_key="graph-and-node-order")
output = reconstruct(linear(representatives), index)
```

The module caches assignments and counts, and recomputes mean representatives
from the current input on every call. The backward pass differentiates those
means with assignments held fixed. The random projection matrix is a buffer,
not an optimizer parameter. Recreate optimizer groups when migrating an old
checkpoint that included this matrix as a parameter.

- `refresh_interval=1` is the default. Larger intervals require `set_epoch(e)`
  or `forward(..., epoch=e)` and validation of training quality.
- Call `train()` / `eval()` as usual; their assignment caches are separate.
  For a new graph or changed node ordering, pass a different `cache_key` or
  call `reset_cache()`. Loading weights or moving device/dtype invalidates caches.
- Consecutive shared Linear/ReLU operations can run on representatives before
  one reconstruction. Reconstruct before node-mixing operations or independent
  node-level dropout. `reconstruct(reps, index, bias=bias, activation="relu")`
  optionally fuses bias and ReLU on CUDA.
- GCN exposes `use_transformation=True`, `refresh_interval`, and `set_epoch`;
  its default remains propagation-only. SGC and `gcn_trans` expose the epoch API.
- Optional `refresh_policy="loss"` requires positive `tau_loss`, `q_min`,
  `q_max`, and `set_epoch(epoch, loss=previous_training_loss)` with a host scalar.
  This changes assignment refresh frequency, not the switch to exact training.
- Cached state is not serialized; save refresh settings with training configs.
  The optimized autograd paths support first-order gradients only. Reuse does
  not guarantee unchanged training accuracy.

Run the transformation regression checks after building the extension:

```bash
python tests/test_transformation.py
```

The checks cover CPU/CUDA mean gradients, LSH assignments, epoch/cache behavior,
CUDA streams, fused reconstruction, and double-precision gradient checks.

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


## Offline compression with CompressGraph

New `CompressgnnData` objects use the exact batch compressor from
[CompressGraph](https://github.com/ZhengChenCS/CompressGraph), pinned as a Git
submodule. The default minimum pair frequency is now **16**, with **12 rounds**.
`compression_backend="auto"` chooses CUDA when the optional extension is installed
and a CUDA device is usable; otherwise it uses the new CPU backend. This replaces
the loader's default greedy Re-Pair construction. Existing saved graphs remain usable.

Initialize the dependency and rebuild the offline extension:

```bash
git submodule update --init src/offline/vendor/CompressGraph
cd src/offline
python setup.py install
```

For the optional GPU compressor (no PyTorch C++ ABI dependency), build with a CUDA
toolkit supporting C++17/CUB and an architecture appropriate to your GPU:

```bash
COMPRESSGNN_BUILD_CUDA=1 COMPRESSGNN_CUDA_HOME=/usr/local/cuda \
  COMPRESSGNN_CUDA_ARCH=86 python setup.py install
```

`COMPRESSGNN_CUDA_HOME` controls this standalone compressor only; it does not change
the CUDA toolkit used to build the training/runtime extension. The CUDA build emits
native code and PTX for the requested architecture (default 75). CPU-only installation
remains supported. To install all existing project extensions, use `bash install.sh`;
it also initializes the dependency and stops on build failures.

```python
data = CompressgnnData(
    x, edge_index, y, train_mask, valid_mask, test_mask,
    compression_backend="auto",  # or "cpu", "cuda", "legacy"
    compression_threads=20,     # CPU only; tune to the host
    compression_rounds=12,
    min_pair_frequency=16,
)
```

The selected backend is stored in `data.compression_backend`. Contribution filtering,
depth filtering, normalization and the existing COO/CSR runtime partitioning still run
after graph construction. Batch output expands exactly to the original adjacency
sequence, but rule selection, compression ratio and downstream execution costs can
change. Do not assume the previously measured raw GPU construction time covers these
postprocessing steps or guarantees unchanged training time.

CUDA allocation/kernel errors propagate; `auto` does not silently hide a runtime
failure by retrying on the CPU. Explicit `cuda` selection fails clearly when the
extension/device is unavailable. Batch frequency must be at least 3. For the original
construction use `compression_backend="legacy", min_pair_frequency=2`. The low-level
`compressgnn_offline.compress_csr` and `compress_csr_fast` APIs remain available for
compatibility; the legacy implementation is not safe for concurrent threaded calls.

Low-level new CPU API (returns rowptr, col, original vertex count, rule count):

```python
from compressgnn_offline import compress_csr_batch
v, e, n, nr = compress_csr_batch(rowptr, col, threads=20, rounds=12,
                                min_pair_frequency=16)
```

The optional `compressgnn_batch_cuda.compress_csr_batch` takes the same arguments
except `threads`. Input/output are CPU NumPy integer arrays; GPU transfer and device
allocation occur inside each call. No subprocess, temporary graph files or runtime
Git download is used during compression.

After rebuilding, run:

```bash
python tests/test_offline_compression.py
python tests/test_batch_compression.py
```

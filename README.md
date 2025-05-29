# CompressGNN
This is the KDD'25 repository for submitting paper **CompressGNN: Accelerating Graph Neural Network Training via
Hierarchical Compression**.

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








# GS-Scale: Unlocking Large-Scale 3D Gaussian Splatting Training via Host Offloading [[Paper](https://arxiv.org/pdf/2509.15645)]

## Overview
**GS-Scale** is a fast, memory-efficient, and scalable training framework built for large-scale 3DGS training. 
To reduce the GPU memory usage, GS-Scale stores all Gaussian parameters and optimizer states in host memory, 
transferring only a subset to the GPU on demand for each forward and backward pass. With various system-level
optimizations, **GS-Scale** significantly lowers GPU memory demands by 3.3-5.6x, while achieving training speeds 
comparable to GPU-Only systems, enabling large-scale 3DGS training on consumer-grade GPUs.

## Setup
```
source install.sh
```
Note:

   We recommend using CUDA 12.x; Currently, only Intel CPUs are supported;

## Prepare Dataset
#### Mill19 (Rubble, Building)
Download dataset [Rubble](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz) and [Building](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgz).
Download colmap from [CityGaussian](https://github.com/Linketic/CityGaussian/blob/main/doc/data_preparation.md) repo and downsample scenes with downsampling rate 4.
Merge train and test sets for both scenes and colmaps. (We do this to use "test\_every".)

#### GauU Scene (SZTU, LFLS, SZIIT)
Download dataset and colmap from [GauU Scene](https://saliteta.github.io/CUHKSZ_SMBU) and downsample scenes with downsampling rate 3.4175.
We use test\_every=10 and downsampling rate 3.4175 following CityGaussian.

#### MatrixCity (Aerial)
Download dataset [MatrixCity](https://huggingface.co/datasets/BoDai/MatrixCity/tree/main/small_city) and downsample scenes with downsampling rate 1.2.
We concat train/test sets and use "test\_after=5620", which means that scenes with ids greater than 5620 are used for test sets.
Download point cloud files from matrixcity\_point\_cloud\_ds20.zip [link](https://huggingface.co/datasets/BoDai/MatrixCity/tree/main/small_city_pointcloud).

## Experiments
#### GPU-Only (w/o host offloading)
```
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py [rubble, building, sztu, lfls, sziit, aerial] --data_dir [path-to-dataset] --result_dir [path-to-result-dir]
```

#### Baseline (host offloading w/o any optimizations)
```
CUDA_VISIBLE_DEVICES=0 python simple_trainer_hybrid_baseline.py [rubble, building, sztu, lfls, sziit, aerial] --data_dir [path-to-dataset] --result_dir [path-to-result-dir]
```

#### GS-Scale (host offloading w/ all optimizations)
```
# w/o split mode
CUDA_VISIBLE_DEVICES=0 python simple_trainer_hybrid_optimized.py [rubble, building, sztu, lfls, sziit, aerial] --data_dir [path-to-dataset] --result_dir [path-to-result-dir]

# w/ split mode
CUDA_VISIBLE_DEVICES=0 python simple_trainer_hybrid_optimized_split.py [rubble, building, sztu, lfls, sziit, aerial] --data_dir [path-to-dataset] --result_dir [path-to-result-dir]
```

## Acknowledgment
This repository is built on [gsplat](https://github.com/nerfstudio-project/gsplat.git) library.

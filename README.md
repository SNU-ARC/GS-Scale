# GS-Scale: Unlocking Large-Scale 3D Gaussian Splatting Training via Host Offloading [[Paper](https://99dhl.github.io/assets/pdf/asplos26_gsscale.pdf)]

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
Download the dataset and organize image/colmap folder as follows.
```
├── data
│   ├── colmap_results
│   │   ├── mill19
│   │   │   ├── rubble-pixsfm
│   │   │   │   ├── images
│   │   │   │   ├── sparse
│   │   │   │   │   ├── 0
│   │   │   │   │   │   ├── cameras.bin
│   │   │   │   │   │   ├── points3D.bin
│   │   │   │   │   │   ├── images.bin
│   │   │   ├── building-pixsfm
│   │   │   │   ├── images
│   │   │   │   ├── sparse
│   │   ├── GauU_Scene
│   │   │   ├── LFLS
│   │   │   │   ├── images
│   │   │   │   ├── sparse
│   │   │   ├── SZTU
│   │   │   │   ├── images
│   │   │   │   ├── sparse
│   │   │   ├── SZIIT
│   │   │   │   ├── images
│   │   │   │   ├── sparse
│   │   ├── MatrixCity
│   │   │   ├── aerial
│   │   │   │   ├── images
│   │   │   │   ├── sparse
│   │   │   │   │   ├── 0
│   │   │   │   │   │   ├── cameras.bin
│   │   │   │   │   │   ├── points3D.bin
│   │   │   │   │   │   ├── images.bin
│   │   │   │   │   │   ├── points.ply

```

#### Mill19 (Rubble, Building)
Download dataset [Rubble](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz) and [Building](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgz). We use downsampling rate 4, following [CityGaussian](https://github.com/Linketic/CityGaussian/blob/main/doc/data_preparation.md). Downsampling is performed automatically.

We downloaded preprocesed colmap from [CityGaussian](https://github.com/Linketic/CityGaussian/blob/main/doc/data_preparation.md). We merge train and test sets in a single directory for both images and colmap. (We do this to use "test\_every".) We provide merged colmap files through Google Drive [link](https://drive.google.com/drive/folders/1bCsIhR-_MFQ71uKNlg6JTROmCc290M4I).

#### GauU Scene (SZTU, LFLS, SZIIT)
Download dataset and colmap from [GauU Scene](https://saliteta.github.io/CUHKSZ_SMBU).
We use test\_every=10 and downsampling rate 3.4175 following CityGaussian. Downsampling is performed automatically.

#### MatrixCity (Aerial)
Download dataset [MatrixCity](https://huggingface.co/datasets/BoDai/MatrixCity/tree/main/small_city). We use downsampling rate 1.2, following CityGaussian. Downsampling is performed automatically. We merge images from train and test set to use "test\_after=5620", which means that scenes with ids greater than 5620 are used for test sets. You will need to rename the image files.

For Gaussian initialization, you can either use initial point cloud from ground truth depth (matrixcity\_point\_cloud\_ds20.zip [link](https://huggingface.co/datasets/BoDai/MatrixCity/tree/main/small_city_pointcloud)) or use colmap from [CityGaussian](https://github.com/Linketic/CityGaussian/blob/main/doc/data_preparation.md) repo. We use point cloud from ground truth depth for our experiments. You may change "init\_ply" option to False to use colmap. We provide both files through Google Drive [link](https://drive.google.com/drive/folders/1bCsIhR-_MFQ71uKNlg6JTROmCc290M4I).

## Training
#### GPU-Only (w/o host offloading)
Original gsplat training script. We recommend using original script when training small scenes. It is typically faster than host offloading versions as long as you don't encounter Out-of-Memory (OOM) issues.
```
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py [rubble, building, sztu, lfls, sziit, aerial] --data_dir [path-to-dataset] --result_dir [path-to-result-dir]
```

#### Baseline (host offloading w/o any optimizations)
A naive version of host offloading training. This saves substantial amount of GPU memory usage but is extremely slow. We do not recommend using this script.
```
CUDA_VISIBLE_DEVICES=0 python simple_trainer_hybrid_baseline.py [rubble, building, sztu, lfls, sziit, aerial] --data_dir [path-to-dataset] --result_dir [path-to-result-dir]
```

#### GS-Scale (host offloading w/ all optimizations)
An optimized version of host offloading training. We recommend this script for training large scenes on a single GPU. It prevents Out-of-Memory (OOM) errors while maintaining a training speed similar to GPU-only training.
Consider using split mode if the optimized version still results in OOM errors. It splits images that consume more Gaussians than `split_threshold * total_gaussians` to further reduce GPU memory usage. This does not affect training quality. A smaller `split_threshold` saves more memory but may slow down training. The default value is 0.3.

```
# w/o split mode
CUDA_VISIBLE_DEVICES=0 python simple_trainer_hybrid_optimized.py [rubble, building, sztu, lfls, sziit, aerial] --data_dir [path-to-dataset] --result_dir [path-to-result-dir]

# w/ split mode
CUDA_VISIBLE_DEVICES=0 python simple_trainer_hybrid_optimized_split.py [rubble, building, sztu, lfls, sziit, aerial] --data_dir [path-to-dataset] --result_dir [path-to-result-dir] --split_threshold [split-threshold]
```

## Adjusting Training Hyperparameters
We define configs for each dataset at the end of training script. You may change hyperparameters in there.
#### Learning rate (`position_lr`, `scaling_lr`)
We use the default learning rate settings from the original 3DGS paper except `position_lr` and `scaling_lr`. Decreasing these values helps improve redering quality. You may consider decreasing them further for larger scenes.

#### Densification Settings (`grow_grad2d`, `grow_scale3d`, `refine_stop_iter`)
We follow [Grendel](https://arxiv.org/abs/2406.18533)'s methodology to scale up or scale down the Gaussian counts for each scene. See Appendix C.3 for more details.
Example densification settings are shown in our training script.

## Acknowledgment
This repository is built on [gsplat](https://github.com/nerfstudio-project/gsplat.git) library.

## Citation
```tex
@inproceedings {gsscale,
    title={GS-Scale: Unlocking Large-Scale 3D Gaussian Splatting Training via Host Offloading},
    author={Donghyun Lee and Dawoon Jeong and Jae W. Lee and Hongil Yoon},
    booktitle = {31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems ({ASPLOS} 26)},
    year={2026},
}
```

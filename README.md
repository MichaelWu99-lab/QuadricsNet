# [QuadricsNet: Learning Concise Representation for Geometric Primitives in Point Clouds](https://github.com/MichaelWu99-lab/QuadricsNet)

*Authors:* Ji Wu, [Huai Yu](https://scholar.google.com/citations?user=lG7h27kAAAAJ&hl=zh-CN), [Wen Yang](http://www.captain-whu.com/yangwen_En.html), [Gui-Song Xia](http://www.captain-whu.com/xia_En.html)

This repository contains the official PyTorch implementation of the [QuadricsNet](https://arxiv.org/pdf/2309.14211.pdf) paper.

[Paper](https://arxiv.org/pdf/2309.14211.pdf) | [Video](https://www.youtube.com/watch?v=0EFhMufuUSI) | [Data](https://drive.google.com/drive/folders/1YRX1pkBt8LSIrlT_iYau_-qr1kMyyr9l?usp=drive_link)

# Introduction
QuadricsNet is an end-to-end framework to learn a concise representation with only 10 parameters for diverse geometric primitives in 3D point clouds. Experiments of primitive parsing on the collected dataset and structure mapping on real-world scenes demonstrate that the quadrics representation is effective and the QuadricsNet framework is robust.

![s](figures/QuadricsNet_brief.png)

# Network Architecture
![s](figures/QuadricsNet_architecture.png)

# Qualitative results
![s](figures/Qualitative_results.gif)

# Dataset
The Quadrics dataset is available: [Google Drive](https://drive.google.com/drive/folders/1YRX1pkBt8LSIrlT_iYau_-qr1kMyyr9l?usp=drive_link). 
* The file structure is as follows:
```
|-- dataset
    |-- quadrics_objects
        |-- train
            |-- h5_dense
                |-- 0.h5
                |-- ...
            |-- train.h5
        |-- test
            |-- h5_dense
                |-- 0.h5
                |-- ...
            |-- test.h5
    |-- quadrics_segments
        |-- train
            |-- plane.h5
            |-- ...
        |-- test
            |-- plane.h5
            |-- ...
```
* Please download this dataset and put in the **data/** folder.
# Installation
```shell
cd <QuadricsNet_Path>
git clone git@github.com:MichaelWu99-lab/QuadricsNet.git
cd QuadricsNet

conda env create --force environment.yml -n QuadricsNet
conda activate QuadricsNet
```
# Quickstart with the pretrained model
```shell
python test_e2e_vis.py configs/configs_test/config_e2e.yml
```

# Citation
If you find our work useful, please consider citing our paper:
```
@article{quadricsnet,
  title={QuadricsNet: Learning Concise Representation for Geometric Primitives in Point Clouds},
  author={Wu, Ji and Yu, Huai and Yang, Wen and Xia, Gui-Song},
  journal={arXiv preprint arXiv:2309.14211},
  year={2023}
}
```
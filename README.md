# [QuadricsNet: Learning Concise Representation for Geometric Primitives in Point Clouds](https://github.com/MichaelWu99-lab/QuadricsNet)

*Authors:* Ji Wu, [Huai Yu](https://scholar.google.com/citations?user=lG7h27kAAAAJ&hl=zh-CN), [Wen Yang](http://www.captain-whu.com/yangwen_En.html), [Gui-Song Xia](http://www.captain-whu.com/xia_En.html)

This repository contains the official PyTorch implementation of the [QuadricsNet](https://arxiv.org/pdf/2309.14211.pdf) paper.

[Paper](https://arxiv.org/pdf/2309.14211.pdf) | [Video](https://www.youtube.com/watch?v=0EFhMufuUSI) | [Data]()

# Introduction
QuadricsNet is an end-to-end framework to learn a concise representation with only 10 parameters for diverse geometric primitives in 3D point clouds. Experiments of primitive parsing on the collected dataset and structure mapping on real-world scenes demonstrate that the quadrics representation is effective and the QuadricsNet framework is robust.

![s](figures/QuadricsNet_brief.png)

# Network Architecture
![s](figures/QuadricsNet_architecture.png)

# Qualitative results
![s](figures/Qualitative_results.gif)

# Dataset
The Quadrics dataset is available: Google Drive.

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
# MbaRGCN

This is the official implementation of the following paper:

> Mamba-based Robust Graph Convolutional Network


<div align="center">
  <img src="https://github.com/hexin5515/MbaRGCN/blob/main/Image/MbaRGCN.png" width="1600px"/>
</div>

## Environment Setup

**Required Dependencies** :

* torch>=2.1.2
* torch_geometric>=2.5.2
* python>=3.8
* scipy>=1.12.0
* numpy>=1.23.5

## Quick Start

**Actor Dataset**

The main experiments:
```
python training_non_targeted.py --dataset Cora_ML --lr 0.01 --net GCN_mamba_Net_pro_max --layer_num 9 --d_model 128 --d_inner 128 --dt_rank 32 --d_state 32 --weight_decay 5e-3 --dropout 0.75 --mamba_dropout 0.1 --runs 10
```
Note: The dataset will be automatically downloaded when the code is executed

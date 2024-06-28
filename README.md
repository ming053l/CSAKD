# CSAKD: Knowledge Distillation with Cross Self-Attention for Hyperspectral and Multispectral Image Fusion

## [[Paper Link (arXiv)]](https://arxiv.org/abs/2404.15781) [[Model Zoo (.pth, .onnx)]](https://drive.google.com/drive/folders/18UAGosITMAch5f4TwaPuyuj5xYJpmZWK?usp=sharing) 

[Chih-Chung Hsu](https://cchsu.info/), Chih-Chien Ni, [Chia-Ming Lee](https://ming053l.github.io/), [Li-Wei Kang](https://scholar.google.com/citations?user=QwSzhgEAAAAJ&hl=zh-TW)

Advanced Computer Vision LAB, National Cheng Kung University

Department of Electrial Engineering, National Taiwan Normal University



## Overview

We introduce a novel knowledge distillation (KD) framework for HR-MSI/LR-HSI fusion to achieve SR of LR-HSI. Our KD framework integrates the proposed Cross-Layer Residual Aggregation (CLRA) block to enhance efficiency for constructing Dual Two-Streamed (DTS) network structure, designed to extract joint and distinct features from LR-HSI and HR-MSI simultaneously. To fully exploit the spatial and spectral feature representations of LR-HSI and HR-MSI, we propose a novel Cross Self-Attention (CSA) fusion module to adaptively fuse those features to improve the spatial and spectral quality of the reconstructed HR-HSI. Finally, the proposed KD-based joint loss function is employed to co-train the teacher and student networks.

<img src=".\figures\complexity.png" width="700"/>
<img src=".\figures\network.png" width="800"/>


## Environment

- [PyTorch >= 1.7](https://pytorch.org/)
- CUDA >= 11.2
- python==3.8.18
- pytorch==1.8.1 
- cudatoolkit=11.3 

### Installation
```
git clone https://github.com/ming053l/CSAKD.git
conda create --name csakd python=3.8 -y
conda activate csakd
# CUDA 11.3
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
cd CSAKD
pip install -r requirements.txt
```

## How To Test

```
python test_CSAKD.py
```

## How To Train

```
python train_CSAKD.py --batch_size 8 --epochs 800 --prefix KD_4bn_band4_AF3_1441 --msi_bands 4 --adaptive_fuse 3 --device='cuda:2' --lr 1e-4 --student_layers 1441
```

## Citations

If our work is helpful to your reaearch, please kindly cite our work. Thank!

#### BibTeX


## Contact
If you have any question, please email zuw408421476@gmail.com to discuss with the author.

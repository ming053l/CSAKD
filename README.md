# CSAKD: Knowledge Distillation with Cross Self-Attention for Hyperspectral and Multispectral Image Fusion

The offical pytorch implementation of "CSAKD: Knowledge Distillation with Cross Self-Attention for Hyperspectral and Multispectral Image Fusion". Submitted to IEEE Transaction on Image Processing (TIP 2024).

## [[Paper Link (arXiv)]](https://arxiv.org/abs/2404.15781) 

## [[Pre-trained Teacher Model]](https://drive.google.com/drive/folders/1qM5KeAuC44srurE536y8iAbtUH8-x0VR?usp=sharing)  [[Pre-trained Student Model]](https://drive.google.com/drive/folders/1qM5KeAuC44srurE536y8iAbtUH8-x0VR?usp=sharing) 

[Chih-Chung Hsu](https://cchsu.info/), Chih-Chien Ni, [Chia-Ming Lee](https://ming053l.github.io/), [Li-Wei Kang](https://scholar.google.com/citations?user=QwSzhgEAAAAJ&hl=zh-TW)

Advanced Computer Vision LAB, National Cheng Kung University

Department of Electrial Engineering, National Taiwan Normal University


## Overview

We introduce a novel knowledge distillation (KD) framework for HR-MSI/LR-HSI fusion to achieve SR of LR-HSI. Our KD framework integrates the proposed Cross-Layer Residual Aggregation (CLRA) block to enhance efficiency for constructing Dual Two-Streamed (DTS) network structure, designed to extract joint and distinct features from LR-HSI and HR-MSI simultaneously. To fully exploit the spatial and spectral feature representations of LR-HSI and HR-MSI, we propose a novel Cross Self-Attention (CSA) fusion module to adaptively fuse those features to improve the spatial and spectral quality of the reconstructed HR-HSI. Finally, the proposed KD-based joint loss function is employed to co-train the teacher and student networks.

<img src=".\figures\network.png" width="800"/>

## Performance evaluation and complexity comparison of the proposed method and other fusion models

**Note:** The methods marked with an asterisk (*) are unsupervised approaches. For the complexity parts, M and G indicate $10^6$ and $10^9$, respectively.

| Method                | PSNR↑  | SAM↓  | RMSE↓  | PSNR↑  | SAM↓  | RMSE↓  | - | - | - | - |
|-----------------------|--------|-------|--------|--------|-------|--------|---------|---------|----------|---------|
|            -          | **4 Bands LR-HSI** |   -    |    -   | **6 Bands LR-HSI** |  -  |  -   |  **Params**   |  **FLOPs**  |  **Run-time**  |  **Memory**  |
| PZRes-Net    | 34.963 | 1.934 | 35.498 | 37.427 | 1.478 | 28.234 | 40.15M  | 5262G   | 0.0141s  | 11059MB |
| MSSJFL       | 34.966 | 1.792 | 33.636 | 38.006 | 1.390 | 26.893 | 16.33M  | 175.56G | 0.0128s  | **1349M**  |
| Dual-UNet    | 35.423 | 1.892 | 33.183 | 38.453 | 1.548 | 26.148 | **2.97M**  | **88.65G** | 0.0127s  | 2152M  |
| DHIF-Net     | 34.458 | 1.829 | 34.769 | 39.146 | 1.239 | 25.309 | 57.04M  | 13795G  | 6.005s   | 29381M |
| *CUCaNet     | 28.848 | 4.140 | 71.710 | 35.509 | 2.205 | 38.973 | 3.0M    | 40.0G   | 2070.01s | -       |
| *USDN        | 30.069 | 3.688 | 93.408 | 35.208 | 2.650 | 53.987 | 0.006M  | 1.0G    | 28.83s   | -       |
| *U2MDN       | 30.127 | 3.235 | 59.071 | 33.356 | 2.243 | 41.528 | 0.01M   | 4.0G    | 547.28s  | -       |
| **CSAKD-Teacher**  | **35.967** | **1.527** | **30.928** | **40.046** | **1.095** | **23.785** | 26.8M   | 941.77G | 0.0134s  | 8733M  |
| **CSAKD-Student**  | 35.544 | 1.643 | 32.308 | 39.153 | 1.205 | 25.080 | 7.44M   | 144.77G | **0.0121s** | 1653M  |


<img src=".\figures\complexity.png" width="560"/>


## Environment

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

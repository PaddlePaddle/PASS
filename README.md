# PASSL
Under Develop.

## Introduction
PASSL is a Paddle based vision library for state-of-the-art Self-Supervised Learning research with PaddlePaddle. PASSL aims to accelerate research cycle in self-supervised learning: from designing a new self-supervised task to evaluating the learned representations. 

## Installation
- Install PaddlePaddle following the [official install guidence](https://www.paddlepaddle.org.cn/install/). 

## Data Preparation
- Download the ImageNet dataset from the [link](http://www.image-net.org/).
- Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
- Create a symlink under PASSL/data/ as the following examples: ```ln -s $YOUR_ILSVRC2012_PATH data/ILSVRC2012```
- At last, the folder looks like:
    ```
    PASSL
    ├── configs
    ├── LICENSE
    ├── passl
    ├── tools
    ├── README.md
    └── data
        └── ILSVRC2012
            ├── train
            └── val
    ```

## Implemented Models 
Models are all trained with ResNet-50 backbone.
|  | epochs |official results | passl results | Backbone| Model |
| ---|--- | ----  | ---- | ----| ---- |
| MoCo  | 200 |  60.6| 60.64| ResNet-50 | [download](https://passl.bj.bcebos.com/models/moco_v1_r50_e200_ckpt.pdparams)|
| MoCo v2 | 200 | 67.7 | 67.72| ResNet-50 | [download](https://passl.bj.bcebos.com/models/moco_v2_r50_e200_ckpt.pdparams)|
| MoCo-BYOL | 300 | 71.56 | 72.10| ResNet-50 | [download](https://passl.bj.bcebos.com/models/mocobyol_r50_ep300_ckpt.pdparams)|
| SimCLR | 100 | 64.5 | 65.3 | ResNet-50 | [download](https://passl.bj.bcebos.com/models/simclr_r50_ep100_ckpt.pdparams)|


## Getting Started

### 1. Train MoCo[v2]

#### single gpu
```
python tools/train.py -c configs/moco_v[1,2]_r50.yaml
```

#### multiple gpus

```
python tools/train.py -c configs/moco_v[1,2]_r50.yaml --num-gpus 8
```

Pretraining models with 200 epochs can be found at [MoCo v1](https://passl.bj.bcebos.com/models/moco_v1_r50_e200_ckpt.pdparams) and [MoCo v2](https://passl.bj.bcebos.com/models/moco_v2_r50_e200_ckpt.pdparams)

Note: The default learning rate in config files is for 8 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following ```new_lr = old_lr * new_ngpus / old_ngpus```. Replacing v[1,2] to v1 or v2 according to your requriement.

### 2. Extract backbone weights

``` 
python tools/extract_weight.py YOUR_TRAINED_MODEL_PATH --output $YOUR_EXTRACTED_BACKBONE_PATH
```

### 3. Evaluation on ImageNet Linear Classification

#### Train:
```
python tools/train.py -c configs/clas_r50.yaml --pretrained YOUR_EXTRACTED_BACKBONE_PATH --num-gpus 8
```

#### Evaluate:
```
python tools/train.py -c configs/clas_r50.yaml --load YOUR_TRAINED_CLS_MODEL --evaluate-only --num-gpus 8
```

The trained linear weights in conjuction with the backbone weights can be found at [MoCo v1 linear](https://passl.bj.bcebos.com/models/moco_v1_r50_clas.pdparams) and [MoCo v2 linear](https://passl.bj.bcebos.com/models/moco_v2_r50_clas.pdparams)

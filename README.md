<div align="center">

  <a href="">![TensorFLow](https://img.shields.io/badge/TensorFlow-2.X-orange?style=for-the-badge) ![License](https://img.shields.io/github/license/EMalagoli92/GCViT-TensorFlow?style=for-the-badge) </a>  
  
</div>

# GCViT-TensorFlow
TensorFlow 2.X implementation of [Global Context Vision Transformers](https://arxiv.org/pdf/2206.09959.pdf) [Ali Hatamizadeh](http://web.cs.ucla.edu/~ahatamiz),
[Hongxu (Danny) Yin](https://scholar.princeton.edu/hongxu), [Jan Kautz](https://jankautz.com/) [Pavlo Molchanov](https://www.pmolchanov.com/).

- Exact TensorFlow reimplementation of official Pytorch repo, including `timm` modules used by authors, preserving models and layers structure.
- ImageNet pretrained weights ported from Pytorch official implementation.

## Table of content
- [Abstract](#abstract)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgement](#acknowledgement)
- [Citations](#citations)
- [License](#license)

## Abstract
*GC ViT  achieves state-of-the-art results across image classification, object detection and semantic segmentation tasks. On ImageNet-1K dataset for classification, the tiny, small and base variants of GC ViT with `28M`, `51M` and `90M`, surpass comparably-sized prior art such as CNN-based ConvNeXt and ViT-based Swin Transformer by a large margin. Pre-trained GC ViT backbones in downstream tasks of object detection, instance segmentation, 
and semantic segmentation using MS COCO and ADE20K datasets outperform prior work consistently, sometimes by large margins.*

![Alt text](assets/images/comp_plots.png?raw=true "Fig1")
<p align = "center"> <sub>Top-1 accuracy vs. model FLOPs/parameter size on ImageNet-1K dataset. GC ViT achieves
new SOTA benchmarks for different model sizes as well as FLOPs, outperforming competing approaches by a
significant margin.</sub> </p>

![Alt text](assets/images/arch.png?raw=true "Fig2")
<p align = "center"><sub>Architecture of the Global Context ViT. The authors use alternating blocks of local and global
context self attention layers in each stage of the architecture.</sub></p>

## Results
TensorFlow implementation and ImageNet ported weights have been compared to the official Pytorch implementation on [ImageNet-V2](https://www.tensorflow.org/datasets/catalog/imagenet_v2) test set.

| Configuration  | Top-1 (Original) | Top-1 (Ported) | Top-5 (Original) | Top-5 (Ported)
| ------------- | ------------- | ------------- | ------------- | ------------- |
| GCViT-XXTiny  | 68.79 | 68.73 | 88.52 | 88.47 |
| GCViT-XTiny  | 70.97 | 71 | 89.8 | 89.79 |
| GCViT-Tiny  | 72.93 | 72.9| 90.7 | 90.7 |
| GCViT-Small  | 73.46 | 73.5 | 91.14 | 91.08 |
| GCViT-Base  | 74.13 | 74.16 | 91.66 | 91.69 |

Mean metrics difference: `3e-4`.

## Installation
- Clone the repo
```
git clone EMalagoli92/GCViT-TensorFlow
```

- Installing necessary packages 
```
pip install -r requirements.txt
```

## Usage
- Define a custom GCViT configuration.
```python
from models import GCViT

# Define a custom model configuration
model = GCViT(depths = [2, 2, 6, 2],
              num_heads = [2, 4, 8, 16],
              window_size = [7, 7, 14, 7],
              dim = 64,
              resolution = 224,
              in_chans = 3,
              mlp_ratio = 3,
              drop_path_rate = 0.2,
              data_format = "channels_last",
              num_classes = 100,
              classifier_activation = "softmax"
              )
```
- Use a a predefined GCviT configuration.
```python
from models import GCViT
    
model = GCViT(configuration = "base")
```
- Train from scratch the model.
```python    
# Example
model.compile(optimizer="sgd",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy","sparse_top_k_categorical_accuracy"]
              )
model.fit(x,y)              
```
- Use ImageNet Pretrained Weights
```python
# Example
from models import GCViT

model = GCViT(configuration="base",pretrained=True)
y_pred = model(image)
```

## Acknowledgement
- [GCViT](https://github.com/nvlabs/gcvit) (Official Pytorch implementation)
- [gcvit_tf](https://github.com/awsaf49/gcvit-tf)
- [tfgcvit](https://github.com/shkarupa-alex/tfgcvit)

## Citations
```bibtex
@article{hatamizadeh2022global,
  title={Global Context Vision Transformers},
  author={Hatamizadeh, Ali and Yin, Hongxu and Kautz, Jan and Molchanov, Pavlo},
  journal={arXiv preprint arXiv:2206.09959},
  year={2022}
}
```

Official PyTorch Implementation: []()

## License

[MIT](https://github.com/EMalagoli92/GCViT-TensorFlow/blob/main/LICENSE)

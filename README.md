# Multi-task MobileNetV2

## Introduction

A PyTorch implementation of MobileNetV2 for multi-task learning.  

## Usage

### Train

```
python main.py --train
```

Arguments:

 * `--data`: (default: `1`)
   * `0`: CIFAR-10
   * `1`: CIFAR-100
   * `2`: Omniglot
 * `--task`: Task ID (default: None) 
 * `--save`: A flag used to decide whether to save model or not.
 * `--load`: Load a pre-trained model before training.
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)
 * `--save_histroy`: A flag used to decide whether to save the accuracy during training or not.
 * `--verbose`: A flag used to decide whether to demonstrate verbose messages or not.

### Evaluate

```
python main.py --eval
```

Arguments:

 * `--data`: (default: `1`)
   * `0`: CIFAR-10
   * `1`: CIFAR-100
   * `2`: Omniglot
 * `--task`: Task ID (default: None)
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)

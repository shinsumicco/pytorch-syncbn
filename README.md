# pytorch-syncbn

Synchronized Multi-GPU Batch Normalization for PyTorch

Most of the source codes are based on [pytorch-syncbn](https://github.com/tamakoji/pytorch-syncbn) written by Tamaki Kojima ([@tamakoji](https://github.com/tamakoji)).

## Overview
This is an alternative implementation of "Synchronized Multi-GPU Batch Normalization" which computes global stats across GPUs instead of locally computed.

SyncBN are getting important for those input image is large, and must use multi-GPU to increase the mini-batch size for the training.

## Remarks
- Unlike [Pytorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), you don't need custom `nn.DataParallel`.
- Unlike [Inplace-ABN](https://github.com/mapillary/inplace_abn), you can just replace your `nn.BatchNorm2d` to this module implementation, since it will not mark for inplace operation.
- You can plug into arbitrary module written in PyTorch to enable Synchronized BatchNorm.
- Backward computation is rewritten and tested against behavior of `nn.BatchNorm2d`.

## Requirements
For PyTorch, please refer to https://pytorch.org/.

NOTE: The code is tested only with PyTorch v0.4.1, CUDA v8.0.44 and cuDNN v6.0.21 on Ubuntu 16.04.

To install all dependencies using pip, please run the following command:

```
pip install -r requirements.txt
```

## Build

Please use [`setup.py`](./setup.py) to build the extension as follows:

```
$ python setup.py [--cuda-path CUDA_PATH]
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Sun_Sep__4_22:14:01_CDT_2016
Cuda compilation tools, release 8.0, V8.0.44
=> building CUDA kernel
=> creating PyTorch extension
cc1: warning: command line option ‘-std=c++11’ is valid for C++/ObjC++ but not for C
cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
cc1plus: warning: command line option ‘-std=c99’ is valid for C/ObjC but not for C++
=> Please set PYTHONPATH as follows:

export PYTHONPATH="/path/to/dir/of/pytorch-syncbn:$PYTHONPATH"

$
```

Then, please set `PYTHONPATH` as indicated.

## Usage

Please refer to [`test.py`](./test.py) for testing the difference between non-synchronized and synchronized multi-GPU learning.

## Math

Please refer to [`README.md`](https://github.com/tamakoji/pytorch-syncbn/blob/master/README.md#math) of [original pytorch-syncbn](https://github.com/tamakoji/pytorch-syncbn).

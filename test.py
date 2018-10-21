# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2018 Shinya Sumikura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchsyncbn import nn as mm

torch.backends.cudnn.deterministic = True


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        elif isinstance(m, mm.BatchNorm2d) or isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()


print("Test script for torchsyncbn module")
print("")

num_gpu = torch.cuda.device_count()
print("=> number of GPUs: {}".format(num_gpu))
if num_gpu < 2:
    print("=> No multi-gpu is found. mm.BatchNorm2d will act as normal nn.BatchNorm2d")
print("")


def test1():
    print("Test1: single GPU (with torch) vs. multi GPU (with torch)")
    print("")

    print("=> initialize weights of two models with the same initial parameters")

    print("m1 [model for single GPU learning]")
    m1 = nn.Sequential(
        nn.Conv2d(3, 3, 1, 1, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(inplace=True),
        nn.Conv2d(3, 3, 1, 1, bias=False),
        nn.BatchNorm2d(3),
    ).cuda()
    torch.manual_seed(123)
    init_weight(m1)
    print(m1)

    print("m2 [model for non-synchronized multi-GPU learning]")
    m2 = nn.Sequential(
        nn.Conv2d(3, 3, 1, 1, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(inplace=True),
        nn.Conv2d(3, 3, 1, 1, bias=False),
        nn.BatchNorm2d(3),
    ).cuda()
    torch.manual_seed(123)
    init_weight(m2)
    m2 = nn.DataParallel(m2, device_ids=range(num_gpu))
    print(m2)

    o1 = torch.optim.SGD(m1.parameters(), 1e-3)
    o2 = torch.optim.SGD(m2.parameters(), 1e-3)

    y = torch.ones(num_gpu).float().cuda()

    print("=> learning the two models with the same inputs and targets")
    torch.manual_seed(123)
    for _ in range(100):
        x = torch.rand(num_gpu, 3, 2, 2).cuda()

        o1.zero_grad()
        z1 = m1(x)
        l1 = F.mse_loss(z1.mean(-1).mean(-1).mean(-1), y)
        l1.backward()
        o1.step()

        o2.zero_grad()
        z2 = m2(x)
        l2 = F.mse_loss(z2.mean(-1).mean(-1).mean(-1), y)
        l2.backward()
        o2.step()

    print("=> show the learned BN parameters, which are probably different between m1 and m2")
    m2 = m2.module
    print_parameters(m1, m2)


def test2():
    print("Test2: single GPU (with torch) vs. multi GPU (with torchsyncbn)")
    print("")

    print("=> initialize weights of two models with the same initial parameters")

    print("m1 [model for single GPU learning]")
    model_with_nn = nn.Sequential(
        nn.Conv2d(3, 3, 1, 1, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(inplace=True),
        nn.Conv2d(3, 3, 1, 1, bias=False),
        nn.BatchNorm2d(3),
    ).cuda()
    torch.manual_seed(123)
    init_weight(model_with_nn)
    print(model_with_nn)

    print("m2 [model for synchronized multi-GPU learning]")
    model_with_mm = nn.Sequential(
        nn.Conv2d(3, 3, 1, 1, bias=False),
        mm.BatchNorm2d(3),
        nn.ReLU(inplace=True),
        nn.Conv2d(3, 3, 1, 1, bias=False),
        mm.BatchNorm2d(3),
    ).cuda()
    torch.manual_seed(123)
    init_weight(model_with_mm)
    model_with_mm = nn.DataParallel(model_with_mm, device_ids=range(num_gpu))
    print(model_with_mm)

    o1 = torch.optim.SGD(model_with_nn.parameters(), 1e-3)
    o2 = torch.optim.SGD(model_with_mm.parameters(), 1e-3)

    y = torch.ones(num_gpu).float().cuda()

    print("=> learning the two models with the same inputs and targets")
    torch.manual_seed(123)
    for _ in range(100):
        x = torch.rand(num_gpu, 3, 2, 2).cuda()

        o1.zero_grad()
        z1 = model_with_nn(x)
        l1 = F.mse_loss(z1.mean(-1).mean(-1).mean(-1), y)
        l1.backward()
        o1.step()

        o2.zero_grad()
        z2 = model_with_mm(x)
        l2 = F.mse_loss(z2.mean(-1).mean(-1).mean(-1), y)
        l2.backward()
        o2.step()

    print("=> show the learned BN parameters, which should be the same between m1 and m2")
    model_with_mm = model_with_mm.module
    print_parameters(model_with_nn, model_with_mm)


def test3():
    print("Test3: multi GPU (with torch) vs. multi GPU (with torchsyncbn)")
    print("")

    print("=> initialize weights of two models with the same initial parameters")

    print("m1 [model for non-synchronized multi-GPU learning]")
    model_with_nn = nn.Sequential(
        nn.Conv2d(3, 3, 1, 1, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(inplace=True),
        nn.Conv2d(3, 3, 1, 1, bias=False),
        nn.BatchNorm2d(3),
    ).cuda()
    torch.manual_seed(123)
    init_weight(model_with_nn)
    model_with_nn = nn.DataParallel(model_with_nn, device_ids=range(num_gpu))
    print(model_with_nn)

    print("m2 [model for synchronized multi-GPU learning]")
    model_with_mm = nn.Sequential(
        nn.Conv2d(3, 3, 1, 1, bias=False),
        mm.BatchNorm2d(3),
        nn.ReLU(inplace=True),
        nn.Conv2d(3, 3, 1, 1, bias=False),
        mm.BatchNorm2d(3),
    ).cuda()
    torch.manual_seed(123)
    init_weight(model_with_mm)
    model_with_mm = nn.DataParallel(model_with_mm, device_ids=range(num_gpu))
    print(model_with_mm)

    o1 = torch.optim.SGD(model_with_nn.parameters(), 1e-3)
    o2 = torch.optim.SGD(model_with_mm.parameters(), 1e-3)

    y = torch.ones(num_gpu).float().cuda()

    print("=> learning the two models with the same inputs and targets")
    torch.manual_seed(123)
    for _ in range(100):
        x = torch.rand(num_gpu, 3, 2, 2).cuda()

        o1.zero_grad()
        z1 = model_with_nn(x)
        l1 = F.mse_loss(z1.mean(-1).mean(-1).mean(-1), y)
        l1.backward()
        o1.step()

        o2.zero_grad()
        z2 = model_with_mm(x)
        l2 = F.mse_loss(z2.mean(-1).mean(-1).mean(-1), y)
        l2.backward()
        o2.step()

    print("=> show the learned BN parameters, which are probably different between m1 and m2")
    model_with_nn = model_with_nn.module
    model_with_mm = model_with_mm.module
    print_parameters(model_with_nn, model_with_mm)


def print_parameters(model_with_nn, model_with_mm):
    print("")
    print("- m1(BatchNorm2d) running_mean")
    print(model_with_nn[1].running_mean)
    print(model_with_nn[-1].running_mean)
    print("- m2(BatchNorm2d) running_mean")
    print(model_with_mm[1].running_mean)
    print(model_with_mm[-1].running_mean)
    print("")
    print("- m1(BatchNorm2d) running_var")
    print(model_with_nn[1].running_var)
    print(model_with_nn[-1].running_var)
    print("- m2(BatchNorm2d) running_var")
    print(model_with_mm[1].running_var)
    print(model_with_mm[-1].running_var)
    print("")
    print("- m1(BatchNorm2d) weight")
    print(model_with_nn[1].weight.data)
    print(model_with_nn[-1].weight.data)
    print("- m2(BatchNorm2d) weight")
    print(model_with_mm[1].weight.data)
    print(model_with_mm[-1].weight.data)
    print("")
    print("- m1(BatchNorm2d) bias")
    print(model_with_nn[1].bias.data)
    print(model_with_nn[-1].bias.data)
    print("- m2(BatchNorm2d) bias")
    print(model_with_mm[1].bias.data)
    print(model_with_mm[-1].bias.data)
    print("")


if __name__ == "__main__":
    test1()
    test2()
    test3()

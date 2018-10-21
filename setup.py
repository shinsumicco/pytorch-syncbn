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

import os
import argparse
import subprocess
from pathlib import Path

from torch.utils.ffi import create_extension


def build_modules():
    parser = argparse.ArgumentParser(description="Install script for torchsyncbn package")
    parser.add_argument("--cuda-path", default=Path("/usr/local/cuda"), type=Path, metavar="PATH",
                        help="CUDA install path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose output")
    args = parser.parse_args()

    script_dir = Path(__file__).absolute().parent

    # CUDA building
    cuda_path = args.cuda_path
    cuda_include_dir = cuda_path / "include"
    nvcc = args.cuda_path / "bin" / "nvcc"
    nvcc_opt = "-std=c++11 -x cu --expt-extended-lambda -O3 -Xcompiler -fPIC".split()
    gen_code = "-gencode arch=compute_61,code=sm_61 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52".split()
    extension_dir = script_dir / "torchsyncbn" / "functional" / "_syncbn" / "src"

    # check if NVCC exists
    try:
        subprocess.call([nvcc, "-V"])
    except FileNotFoundError:
        raise RuntimeError("NVCC is not found: {}".format(nvcc))

    # build CUDA kernel
    print("=> building CUDA kernel")
    if args.verbose:
        result = subprocess.call([nvcc, "-v", "-c", "-o", "syncbn.cu.o", "syncbn.cu"] + ["-I", cuda_include_dir] + nvcc_opt + gen_code,
                                 cwd=extension_dir)
    else:
        result = subprocess.call([nvcc, "-c", "-o", "syncbn.cu.o", "syncbn.cu"] + ["-I", cuda_include_dir] + nvcc_opt + gen_code,
                                 cwd=extension_dir)
    if result != 0:
        raise RuntimeError("building syncbn.cu has been failed")

    # create PyTorch extension
    print("=> creating PyTorch extension")
    sources = [str(extension_dir / "syncbn.cpp")]
    headers = [str(extension_dir / "syncbn.h")]
    extra_objects = [str(extension_dir / "syncbn.cu.o")]

    os.environ["C_INCLUDE_PATH"] = str(cuda_include_dir)
    os.environ["CPLUS_INCLUDE_PATH"] = str(cuda_include_dir)
    ffi = create_extension(
        "_ext.syncbn",
        headers=headers,
        sources=sources,
        relative_to=extension_dir,
        with_cuda=True,
        extra_objects=extra_objects,
        extra_compile_args=["-std=c++11"],
        verbose=args.verbose
    )

    ffi.build()

    print("=> Please set PYTHONPATH as follows:")
    print("")
    print("export PYTHONPATH=\"{}:$PYTHONPATH\"".format(script_dir))
    print("")


if __name__ == "__main__":
    build_modules()

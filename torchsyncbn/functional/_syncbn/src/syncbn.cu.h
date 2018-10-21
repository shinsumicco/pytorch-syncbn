/*!
MIT License

Copyright (c) 2018 Tamaki Kojima

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
*/

#ifndef __SYNCBN__
#define __SYNCBN__

/*
 * Exported functions
 */
extern "C" int _syncbn_sum_sqsum_cuda(int N, int C, int S, const float *x,
                                  float *sum, float *sqsum,
                                  cudaStream_t stream);
extern "C" int _syncbn_forward_cuda(
    int N, int C, int S, float *z, const float *x,
    const float *gamma, const float *beta, const float *mean, const float *var,
    float eps, cudaStream_t stream);
extern "C" int _syncbn_backward_xhat_cuda(
    int N, int C, int S, const float *dz, const float *x,
    const float *mean, const float *var, float *sum_dz, float *sum_dz_xhat,
    float eps, cudaStream_t stream);
extern "C" int _syncbn_backward_cuda(
    int N, int C, int S, const float *dz, const float *x,
    const float *gamma, const float *beta, const float *mean, const float *var,
    const float *sum_dz, const float *sum_dz_xhat,
    float *dx, float *dweight, float *dbias,
    float eps, cudaStream_t stream);


#endif

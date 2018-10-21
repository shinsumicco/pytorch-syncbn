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

int syncbn_sum_sqsum_cuda(
    const THCudaTensor *x, THCudaTensor *sum, THCudaTensor *sqsum);
int syncbn_forward_cuda(
    THCudaTensor *z, const THCudaTensor *x,
    const THCudaTensor *gamma, const THCudaTensor *beta,
    const THCudaTensor *mean, const THCudaTensor *var, float eps);
int syncbn_backward_xhat_cuda(
    const THCudaTensor *dz, const THCudaTensor *x,
    const THCudaTensor *mean, const THCudaTensor *var,
    THCudaTensor *sum_dz, THCudaTensor *sum_dz_xhat,
    float eps);
int syncbn_backard_cuda(
    const THCudaTensor *dz, const THCudaTensor *x,
    const THCudaTensor *gamma, const THCudaTensor *beta,
    const THCudaTensor *mean, const THCudaTensor *var,
    const THCudaTensor *sum_dz, const THCudaTensor *sum_dz_xhat,
    THCudaTensor *dx, THCudaTensor *dgamma, THCudaTensor *dbeta, float eps);

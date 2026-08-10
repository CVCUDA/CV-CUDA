/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef INPAINT_UTILS_CUH
#define INPAINT_UTILS_CUH

#include "CvCudaUtils.cuh"

#include <limits>

using namespace nvcv::legacy::cuda_op;

#define KNOWN  0 //known outside narrow band
#define BAND   1 //narrow band (known)
#define INSIDE 2 //unknown
#define CHANGE 3 //servise

inline size_t InpaintCheckedAdd(size_t a, size_t b)
{
    if (a > std::numeric_limits<size_t>::max() - b)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Inpaint workspace size overflow");
    }
    return a + b;
}

inline size_t InpaintCheckedMul(size_t a, size_t b)
{
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Inpaint workspace size overflow");
    }
    return a * b;
}

inline size_t ComputeInpaintKernelSize(int maxBatchSize)
{
    return InpaintCheckedMul(
        InpaintCheckedMul(InpaintCheckedMul(sizeof(unsigned char), static_cast<size_t>(maxBatchSize)), 3), 3);
}

inline size_t ComputeInpaintWorkspaceSize(int maxBatchSize, nvcv::Size2D maxShape, int reduceGridSize)
{
    size_t batch = static_cast<size_t>(maxBatchSize);
    size_t erows = InpaintCheckedAdd(static_cast<size_t>(maxShape.h), 2);
    size_t ecols = InpaintCheckedAdd(static_cast<size_t>(maxShape.w), 2);

    size_t reduceSize
        = InpaintCheckedMul(sizeof(int), InpaintCheckedAdd(InpaintCheckedMul(reduceGridSize, batch), 1));
    size_t imageSize = InpaintCheckedMul(InpaintCheckedMul(InpaintCheckedMul(batch, erows), ecols),
                                         sizeof(float) + sizeof(unsigned char) * 3);

    return InpaintCheckedAdd(reduceSize, imageSize);
}

__inline__ __device__ float min4(float a, float b, float c, float d)
{
    a = min(a, b);
    c = min(c, d);
    return min(a, c);
}

__inline__ __device__ float VectorScalMult(float2 &v1, float2 &v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}

__inline__ __device__ float VectorLength(float2 &v1)
{
    return v1.x * v1.x + v1.y * v1.y;
}

__inline__ __device__ float FastMarching_solve(int i1, int j1, int i2, int j2, Ptr2dNHWC<unsigned char> f,
                                               Ptr2dNHWC<float> t)
{
    // printf("FastMarching_solve: %d %d %d %d\n", i1, j1, i2, j2);
    const int batch_idx = get_batch_idx();
    double    sol, a11, a22, m12;
    a11 = *t.ptr(batch_idx, i1, j1);
    a22 = *t.ptr(batch_idx, i2, j2);
    m12 = ((a11) > (a22) ? (a22) : (a11));

    if (*f.ptr(batch_idx, i1, j1) != INSIDE)
        if (*f.ptr(batch_idx, i2, j2) != INSIDE)
            if (fabs(a11 - a22) >= 1.0)
                sol = 1 + m12;
            else
                sol = (a11 + a22 + sqrt((double)(2 - (a11 - a22) * (a11 - a22)))) * 0.5;
        else
            sol = 1 + a11;
    else if (*f.ptr(batch_idx, i2, j2) != INSIDE)
        sol = 1 + a22;
    else
        sol = 1 + m12;

    return (float)sol;
}

inline int init_dilate_kernel(unsigned char *cross_kernel, cudaStream_t stream)
{
    unsigned char K[9] = {0, 1, 0, 1, 1, 1, 0, 1, 0};
    checkCudaErrors(cudaMemcpyAsync(cross_kernel, K, 9 * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));
    return 0;
}

template<typename T>
__global__ void fill_value_kernel(Ptr2dNHWC<T> dst, T value)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (x >= dst.cols || y >= dst.rows)
        return;

    for (int c = 0; c < dst.ch; ++c)
    {
        *dst.ptr(batch_idx, y, x, c) = value;
    }
}

template<typename T>
void fill_value(Ptr2dNHWC<T> dst, T value, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y), dst.batches);
    fill_value_kernel<<<grid, block, 0, stream>>>(dst, value);
    checkKernelErrors();
}

__global__ void deviceReducePoints(const int *g_in, int *g_out, const int N);

#endif // INPAINT_UTILS_CUH

/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvCudaUtils.cuh"
#include "cub/cub.cuh"

__global__ void triangle_cal(int *histogram, nvcv::cuda::Tensor1DWrap<double> thresh)
{
    int                     localid = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ int          hist[256];
    __shared__ volatile int reduce[256];
    hist[localid] = histogram[blockIdx.z * 256 + localid];

    int left_bound = INT_MAX, right_bound = -1;

    // find the left_bound of the histogram (the leftmost non-zero number).
    // Reduce to find the smallest localid
    if (hist[localid] > 0)
    {
        left_bound  = localid;
        right_bound = localid;
    }

    reduce[localid] = left_bound;
    __syncthreads();

    if (localid < 128)
        reduce[localid] = min(reduce[localid], reduce[localid + 128]);
    __syncthreads();
    if (localid < 64)
        reduce[localid] = min(reduce[localid], reduce[localid + 64]);
    __syncthreads();
    if (localid < 32)
    {
        reduce[localid] = min(reduce[localid], reduce[localid + 32]);
        reduce[localid] = min(reduce[localid], reduce[localid + 16]);
        reduce[localid] = min(reduce[localid], reduce[localid + 8]);
        reduce[localid] = min(reduce[localid], reduce[localid + 4]);
        reduce[localid] = min(reduce[localid], reduce[localid + 2]);
        reduce[localid] = min(reduce[localid], reduce[localid + 1]);
    }
    __syncthreads();

    left_bound = reduce[0];
    if (left_bound > 0)
        left_bound--;

    // find the right_bound of the histogram (the rightmost non-zero number).
    // Reduce to find the largest localid
    __syncthreads();
    reduce[localid] = right_bound;
    __syncthreads();

    if (localid < 128)
        reduce[localid] = max(reduce[localid], reduce[localid + 128]);
    __syncthreads();
    if (localid < 64)
        reduce[localid] = max(reduce[localid], reduce[localid + 64]);
    __syncthreads();
    if (localid < 32)
    {
        reduce[localid] = max(reduce[localid], reduce[localid + 32]);
        reduce[localid] = max(reduce[localid], reduce[localid + 16]);
        reduce[localid] = max(reduce[localid], reduce[localid + 8]);
        reduce[localid] = max(reduce[localid], reduce[localid + 4]);
        reduce[localid] = max(reduce[localid], reduce[localid + 2]);
        reduce[localid] = max(reduce[localid], reduce[localid + 1]);
    }
    __syncthreads();
    right_bound = reduce[0];

    if (right_bound < 255)
        right_bound++;
    __syncthreads();

    // find the coordinate with the maximum value in the histogram
    // reduce to find the maximum value and record the coordinate
    __shared__ nvcv::legacy::cuda_op::uchar idx[256];
    idx[localid]    = (nvcv::legacy::cuda_op::uchar)localid;
    reduce[localid] = hist[localid];
    __syncthreads();

    if (localid < 128 && reduce[localid + 128] >= reduce[localid])
    {
        if (reduce[localid + 128] == reduce[localid])
            idx[localid] = min(idx[localid], idx[localid + 128]);
        else
            idx[localid] = idx[localid + 128];
        reduce[localid] = reduce[localid + 128];
    }
    __syncthreads();
    if (localid < 64 && reduce[localid + 64] >= reduce[localid])
    {
        if (reduce[localid + 64] == reduce[localid])
            idx[localid] = min(idx[localid], idx[localid + 64]);
        else
            idx[localid] = idx[localid + 64];
        reduce[localid] = reduce[localid + 64];
    }
    __syncthreads();

    if (localid < 32)
    {
        if (reduce[localid + 32] >= reduce[localid])
        {
            if (reduce[localid + 32] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 32]);
            else
                idx[localid] = idx[localid + 32];
            reduce[localid] = reduce[localid + 32];
        }
        if (reduce[localid + 16] >= reduce[localid])
        {
            if (reduce[localid + 16] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 16]);
            else
                idx[localid] = idx[localid + 16];
            reduce[localid] = reduce[localid + 16];
        }
        if (reduce[localid + 8] >= reduce[localid])
        {
            if (reduce[localid + 8] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 8]);
            else
                idx[localid] = idx[localid + 8];
            reduce[localid] = reduce[localid + 8];
        }
        if (reduce[localid + 4] >= reduce[localid])
        {
            if (reduce[localid + 4] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 4]);
            else
                idx[localid] = idx[localid + 4];
            reduce[localid] = reduce[localid + 4];
        }
        if (reduce[localid + 2] >= reduce[localid])
        {
            if (reduce[localid + 2] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 2]);
            else
                idx[localid] = idx[localid + 2];
            reduce[localid] = reduce[localid + 2];
        }
        if (reduce[localid + 1] >= reduce[localid])
        {
            if (reduce[localid + 1] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 1]);
            else
                idx[localid] = idx[localid + 1];
            reduce[localid] = reduce[localid + 1];
        }
    }
    __syncthreads();
    int max = reduce[0], maxid = idx[0];

    // determine if the histogram needs to be flipped
    bool isfliped = false;
    if (maxid - left_bound < right_bound - maxid)
    {
        isfliped = true;
        int temp = hist[255 - localid];
        __syncthreads();
        hist[localid] = temp;
        left_bound    = 255 - right_bound;
        maxid         = 255 - maxid;
    }

    // from left_bound to the coordinate with the maximum value in the histogram(maxid),
    // calculate the distance : 'max_value * i + (left_bound - maxid) * histogram[i]'
    int val = -1;
    if (localid > left_bound && localid <= maxid)
        val = max * localid + (left_bound - maxid) * hist[localid];

    // find the coordinate with the largest distance
    // reduce to find the largest distance and record the coordinate
    __syncthreads();
    reduce[localid] = val;
    idx[localid]    = localid;
    __syncthreads();

    if (localid < 128 && reduce[localid + 128] >= reduce[localid])
    {
        if (reduce[localid + 128] == reduce[localid])
            idx[localid] = min(idx[localid], idx[localid + 128]);
        else
            idx[localid] = idx[localid + 128];
        reduce[localid] = reduce[localid + 128];
    }
    __syncthreads();
    if (localid < 64 && reduce[localid + 64] >= reduce[localid])
    {
        if (reduce[localid + 64] == reduce[localid])
            idx[localid] = min(idx[localid], idx[localid + 64]);
        else
            idx[localid] = idx[localid + 64];
        reduce[localid] = reduce[localid + 64];
    }
    __syncthreads();

    if (localid < 32)
    {
        if (reduce[localid + 32] >= reduce[localid])
        {
            if (reduce[localid + 32] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 32]);
            else
                idx[localid] = idx[localid + 32];
            reduce[localid] = reduce[localid + 32];
        }
        if (reduce[localid + 16] >= reduce[localid])
        {
            if (reduce[localid + 16] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 16]);
            else
                idx[localid] = idx[localid + 16];
            reduce[localid] = reduce[localid + 16];
        }
        if (reduce[localid + 8] >= reduce[localid])
        {
            if (reduce[localid + 8] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 8]);
            else
                idx[localid] = idx[localid + 8];
            reduce[localid] = reduce[localid + 8];
        }
        if (reduce[localid + 4] >= reduce[localid])
        {
            if (reduce[localid + 4] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 4]);
            else
                idx[localid] = idx[localid + 4];
            reduce[localid] = reduce[localid + 4];
        }
        if (reduce[localid + 2] >= reduce[localid])
        {
            if (reduce[localid + 2] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 2]);
            else
                idx[localid] = idx[localid + 2];
            reduce[localid] = reduce[localid + 2];
        }
        if (reduce[localid + 1] >= reduce[localid])
        {
            if (reduce[localid + 1] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 1]);
            else
                idx[localid] = idx[localid + 1];
            reduce[localid] = reduce[localid + 1];
        }
    }
    __syncthreads();

    // write to gpu memory
    if (localid == 0)
    {
        double res = (double)idx[0] - 1;
        if (isfliped)
            res = 255 - res;
        thresh[(int)blockIdx.z] = res;
    }
}

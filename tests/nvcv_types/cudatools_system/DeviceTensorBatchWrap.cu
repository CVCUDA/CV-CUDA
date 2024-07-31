/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeviceTensorBatchWrap.hpp"

#include <cvcuda/cuda_tools/MathOps.hpp>    // for operator == to allow EXPECT_EQ
#include <cvcuda/cuda_tools/StaticCast.hpp> // for StaticCast, etc.
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <gtest/gtest.h> // for EXPECT_EQ, etc.

namespace cuda = nvcv::cuda;

template<typename T, int NDIM, typename TensorWrap, typename... Coords>
__device__ T *tensor_ptr(TensorWrap &tensor, int *coords, Coords... vcoords)
{
    if constexpr (sizeof...(Coords) == NDIM)
    {
        return tensor.ptr(vcoords...);
    }
    else
    {
        return tensor_ptr<T, NDIM, TensorWrap>(tensor, coords, vcoords..., coords[sizeof...(Coords)]);
    }
}

template<typename TensorBatchWrapT, typename T>
__device__ void SetThroughTensor<TensorBatchWrapT, T>::Set(TensorBatchWrapT wrap, int sample, int *coords, T value)
{
    auto tensor                                                      = wrap.tensor(sample);
    *tensor_ptr<T, TensorBatchWrapT::kNumDimensions>(tensor, coords) = value;
}

template<typename TensorBatchWrapT, typename T>
__device__ void SetThroughSubscript<TensorBatchWrapT, T>::Set(TensorBatchWrapT wrap, int sample, int *coords, T value)
{
    constexpr int NDIM = TensorBatchWrapT::kNumDimensions;
    if constexpr (NDIM == 1)
    {
        wrap[int2{sample, coords[0]}] = value;
    }
    else if constexpr (NDIM == 2)
    {
        wrap[int3{sample, coords[1], coords[0]}] = value;
    }
    else if constexpr (NDIM == 3)
    {
        wrap[int4{sample, coords[2], coords[1], coords[0]}] = value;
    }
}

template<typename T, typename TensorBatchWrapT, typename... VCoords>
__device__ void SetThroughPtrHelper(TensorBatchWrapT wrap, int sample, int *coords, T value, VCoords... vcoords)
{
    constexpr int NDIM = TensorBatchWrapT::kNumDimensions;
    if constexpr (sizeof...(VCoords) == 0)
    {
        SetThroughPtrHelper<T>(wrap, sample, coords, value, sample, coords[0]);
    }
    else if constexpr (sizeof...(VCoords) < NDIM + 1)
    {
        SetThroughPtrHelper<T>(wrap, sample, coords, value, vcoords..., coords[sizeof...(VCoords) - 1]);
    }
    else
    {
        *wrap.ptr(vcoords...) = value;
    }
}

template<typename TensorBatchWrapT, typename T>
__device__ void SetThroughPtr<TensorBatchWrapT, T>::Set(TensorBatchWrapT wrap, int sample, int *coords, T value)
{
    SetThroughPtrHelper<T>(wrap, sample, coords, value);
}

template<typename SetValue, typename TensorBatchWrapT>
__global__ void SetReferenceKernel(TensorBatchWrapT wrap)
{
    int            sample    = blockIdx.x;
    const int64_t *shape     = wrap.shape(sample);
    int            id        = threadIdx.x;
    int64_t        tensorVol = 1;
    const int      ndim      = TensorBatchWrapT::kNumDimensions;
    for (int d = 0; d < ndim; ++d)
    {
        tensorVol *= shape[d];
    }
    for (int index = id; index < tensorVol; index += blockDim.x)
    {
        int coords[ndim];
        int tmp_i = index;
        for (int d = ndim - 1; d >= 0; --d)
        {
            coords[d] = tmp_i % shape[d];
            tmp_i /= shape[d];
        }
        SetValue::Set(wrap, sample, coords, cuda::SetAll<TensorBatchWrapT::ValueType>(index % 255));
    }
}

template<typename SetValue, typename TensorBatchWrapT>
void SetReference(TensorBatchWrapT wrap, cudaStream_t stream)
{
    int blocks = wrap.numTensors();
    SetReferenceKernel<SetValue, TensorBatchWrapT><<<blocks, 1024, 0, stream>>>(wrap);
}

#define SetReferenceSpec(SET_VALUE, TENSOR_BATCH_TYPE)                                                         \
    template __device__ void SET_VALUE<TENSOR_BATCH_TYPE, TENSOR_BATCH_TYPE::ValueType>::Set(                  \
        TENSOR_BATCH_TYPE, int, int *, TENSOR_BATCH_TYPE::ValueType);                                          \
    template void SetReference<SET_VALUE<TENSOR_BATCH_TYPE, TENSOR_BATCH_TYPE::ValueType>, TENSOR_BATCH_TYPE>( \
        TENSOR_BATCH_TYPE, cudaStream_t)

#define TB_PARAMS1 uchar1, -1, 32 * sizeof(uchar1), sizeof(uchar1)
SetReferenceSpec(SetThroughTensor, cuda::TensorBatchWrap<TB_PARAMS1>);

#define TB_PARAMS2 double4, 8 * sizeof(double4), sizeof(double4)
SetReferenceSpec(SetThroughTensor, cuda::TensorBatchWrap<TB_PARAMS2>);

#define TB_PARAMS3 float3, -1, -1, 8 * sizeof(float3), sizeof(float3)
SetReferenceSpec(SetThroughTensor, cuda::TensorBatchWrap<TB_PARAMS3>);

#define TB_PARAMS4 uchar2, sizeof(uchar2)
SetReferenceSpec(SetThroughSubscript, cuda::TensorBatchWrap<TB_PARAMS4>);

#define TB_PARAMS5 int3, -1, 16 * sizeof(int3), sizeof(int3)
SetReferenceSpec(SetThroughSubscript, cuda::TensorBatchWrap<TB_PARAMS5>);

#define TB_PARAMS6 ushort4, -1, sizeof(ushort4)
SetReferenceSpec(SetThroughSubscript, cuda::TensorBatchWrap<TB_PARAMS6>);

#define TB_PARAMS7 uchar4, -1, -1, 32 * sizeof(uchar4), sizeof(uchar4)
SetReferenceSpec(SetThroughPtr, cuda::TensorBatchWrap<TB_PARAMS7>);

#define TB_PARAMS8 float1, -1, -1, -1, 8 * sizeof(float1), sizeof(float1)
SetReferenceSpec(SetThroughPtr, cuda::TensorBatchWrap<TB_PARAMS8>);

#define TB_PARAMS9 float4, sizeof(float4)
SetReferenceSpec(SetThroughPtr, cuda::TensorBatchWrap<TB_PARAMS9>);

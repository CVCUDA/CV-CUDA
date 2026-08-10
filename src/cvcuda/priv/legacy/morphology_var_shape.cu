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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>

#include <type_traits>

using namespace nvcv::legacy::helpers;
using namespace nvcv::legacy::cuda_op;

namespace nvcv::legacy::cuda_op {

namespace {

static bool IsPlanar(DataFormat format)
{
    return format == kNCHW || format == kCHW;
}

} // namespace

__global__ void UpdateMasksAnchors(cuda::Tensor1DWrap<int2> masks, cuda::Tensor1DWrap<int2> anchors, int numImages,
                                   int iteration)
{
    int1 coord;
    coord.x = cuda::StaticCast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (coord.x >= numImages)
        return;

    int2 mask_size = masks[coord];
    int2 anchor    = anchors[coord];

    if (mask_size.x == -1 || mask_size.y == -1)
        mask_size.x = mask_size.y = 3;
    if (anchor.x < 0)
        anchor.x = mask_size.x / 2;
    if (anchor.y < 0)
        anchor.y = mask_size.y / 2;

    mask_size = mask_size + (iteration - 1) * (mask_size - 1);
    anchor    = anchor * iteration;

    masks[coord]   = mask_size;
    anchors[coord] = anchor;
}

template<class PT, class SrcWrapper>
__device__ __forceinline__ PT dilate3x3Interior(const SrcWrapper &src, int batch, int y, int x, PT res)
{
    res = cuda::max(res, *src.ptr(batch, y - 1, x - 1));
    res = cuda::max(res, *src.ptr(batch, y - 1, x));
    res = cuda::max(res, *src.ptr(batch, y - 1, x + 1));
    res = cuda::max(res, *src.ptr(batch, y, x - 1));
    res = cuda::max(res, *src.ptr(batch, y, x));
    res = cuda::max(res, *src.ptr(batch, y, x + 1));
    res = cuda::max(res, *src.ptr(batch, y + 1, x - 1));
    res = cuda::max(res, *src.ptr(batch, y + 1, x));
    res = cuda::max(res, *src.ptr(batch, y + 1, x + 1));
    return res;
}

template<class PT, class SrcWrapper>
__device__ __forceinline__ PT dilate3x3Interior(const SrcWrapper &src, int batch, int channel, int y, int x, PT res)
{
    res = cuda::max(res, *src.ptr(batch, channel, y - 1, x - 1));
    res = cuda::max(res, *src.ptr(batch, channel, y - 1, x));
    res = cuda::max(res, *src.ptr(batch, channel, y - 1, x + 1));
    res = cuda::max(res, *src.ptr(batch, channel, y, x - 1));
    res = cuda::max(res, *src.ptr(batch, channel, y, x));
    res = cuda::max(res, *src.ptr(batch, channel, y, x + 1));
    res = cuda::max(res, *src.ptr(batch, channel, y + 1, x - 1));
    res = cuda::max(res, *src.ptr(batch, channel, y + 1, x));
    res = cuda::max(res, *src.ptr(batch, channel, y + 1, x + 1));
    return res;
}

template<class PT, class SrcWrapper>
__device__ __forceinline__ PT erode3x3Interior(const SrcWrapper &src, int batch, int y, int x, PT res)
{
    res = cuda::min(res, *src.ptr(batch, y - 1, x - 1));
    res = cuda::min(res, *src.ptr(batch, y - 1, x));
    res = cuda::min(res, *src.ptr(batch, y - 1, x + 1));
    res = cuda::min(res, *src.ptr(batch, y, x - 1));
    res = cuda::min(res, *src.ptr(batch, y, x));
    res = cuda::min(res, *src.ptr(batch, y, x + 1));
    res = cuda::min(res, *src.ptr(batch, y + 1, x - 1));
    res = cuda::min(res, *src.ptr(batch, y + 1, x));
    res = cuda::min(res, *src.ptr(batch, y + 1, x + 1));
    return res;
}

template<class PT, class SrcWrapper>
__device__ __forceinline__ PT erode3x3Interior(const SrcWrapper &src, int batch, int channel, int y, int x, PT res)
{
    res = cuda::min(res, *src.ptr(batch, channel, y - 1, x - 1));
    res = cuda::min(res, *src.ptr(batch, channel, y - 1, x));
    res = cuda::min(res, *src.ptr(batch, channel, y - 1, x + 1));
    res = cuda::min(res, *src.ptr(batch, channel, y, x - 1));
    res = cuda::min(res, *src.ptr(batch, channel, y, x));
    res = cuda::min(res, *src.ptr(batch, channel, y, x + 1));
    res = cuda::min(res, *src.ptr(batch, channel, y + 1, x - 1));
    res = cuda::min(res, *src.ptr(batch, channel, y + 1, x));
    res = cuda::min(res, *src.ptr(batch, channel, y + 1, x + 1));
    return res;
}

__device__ __forceinline__ bool IsMorphInterior(int2 kernelSize, int2 anchor, int x, int y, int width, int height)
{
    return x >= anchor.x && y >= anchor.y && x + kernelSize.x - anchor.x <= width
        && y + kernelSize.y - anchor.y <= height;
}

template<class PT, class SrcWrapper>
__device__ __forceinline__ PT dilateInterior(const SrcWrapper &src, int batch, int y, int x, int2 kernelSize,
                                             int2 anchor, PT res)
{
    const int srcY0 = y - anchor.y;
    const int srcX0 = x - anchor.x;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        for (int j = 0; j < kernelSize.x; ++j)
        {
            res = cuda::max(res, *src.ptr(batch, srcY0 + i, srcX0 + j));
        }
    }

    return res;
}

template<class PT, class SrcWrapper>
__device__ __forceinline__ PT dilateInterior(const SrcWrapper &src, int batch, int channel, int y, int x,
                                             int2 kernelSize, int2 anchor, PT res)
{
    const int srcY0 = y - anchor.y;
    const int srcX0 = x - anchor.x;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        for (int j = 0; j < kernelSize.x; ++j)
        {
            res = cuda::max(res, *src.ptr(batch, channel, srcY0 + i, srcX0 + j));
        }
    }

    return res;
}

template<class PT, class SrcWrapper>
__device__ __forceinline__ PT erodeInterior(const SrcWrapper &src, int batch, int y, int x, int2 kernelSize,
                                            int2 anchor, PT res)
{
    const int srcY0 = y - anchor.y;
    const int srcX0 = x - anchor.x;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        for (int j = 0; j < kernelSize.x; ++j)
        {
            res = cuda::min(res, *src.ptr(batch, srcY0 + i, srcX0 + j));
        }
    }

    return res;
}

template<class PT, class SrcWrapper>
__device__ __forceinline__ PT erodeInterior(const SrcWrapper &src, int batch, int channel, int y, int x,
                                            int2 kernelSize, int2 anchor, PT res)
{
    const int srcY0 = y - anchor.y;
    const int srcX0 = x - anchor.x;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        for (int j = 0; j < kernelSize.x; ++j)
        {
            res = cuda::min(res, *src.ptr(batch, channel, srcY0 + i, srcX0 + j));
        }
    }

    return res;
}

template<bool UseGenericInterior, class SrcWrapper, class RawSrcWrapper, class DstWrapper,
         typename D = typename DstWrapper::ValueType, typename BT = typename cuda::BaseType<D>>
__global__ void dilate(const SrcWrapper src, const RawSrcWrapper rawSrc, DstWrapper dst,
                       cuda::Tensor1DWrap<int2> kernelSizeArr, cuda::Tensor1DWrap<int2> kernelAnchorArr, BT maxmin)
{
    D         res       = cuda::SetAll<D>(maxmin);
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int width  = dst.width(batch_idx);
    const int height = dst.height(batch_idx);

    if (x >= width || y >= height)
        return;

    int2 kernelSize = kernelSizeArr[batch_idx];
    int2 anchor     = kernelAnchorArr[batch_idx];

    if (kernelSize.x == 3 && kernelSize.y == 3 && anchor.x == 1 && anchor.y == 1 && x > 0 && y > 0 && x + 1 < width
        && y + 1 < height)
    {
        res = dilate3x3Interior(rawSrc, batch_idx, y, x, res);
    }
    else
    {
        if constexpr (UseGenericInterior)
        {
            if (IsMorphInterior(kernelSize, anchor, x, y, width, height))
            {
                res = dilateInterior(rawSrc, batch_idx, y, x, kernelSize, anchor, res);
            }
            else
            {
                int3 srcCoord = {0, 0, batch_idx};
                for (int i = 0; i < kernelSize.y; ++i)
                {
                    srcCoord.y = y - anchor.y + i;

                    for (int j = 0; j < kernelSize.x; ++j)
                    {
                        srcCoord.x = x - anchor.x + j;

                        res = cuda::max(res, src[srcCoord]);
                    }
                }
            }
        }
        else
        {
            int3 srcCoord = {0, 0, batch_idx};
            for (int i = 0; i < kernelSize.y; ++i)
            {
                srcCoord.y = y - anchor.y + i;

                for (int j = 0; j < kernelSize.x; ++j)
                {
                    srcCoord.x = x - anchor.x + j;

                    res = cuda::max(res, src[srcCoord]);
                }
            }
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<D>(res);
}

template<bool UseGenericInterior, class SrcWrapper, class RawSrcWrapper, class DstWrapper,
         typename D = typename DstWrapper::ValueType, typename BT = typename cuda::BaseType<D>>
__global__ void dilatePlanar(const SrcWrapper src, const RawSrcWrapper rawSrc, DstWrapper dst,
                             cuda::Tensor1DWrap<int2> kernelSizeArr, cuda::Tensor1DWrap<int2> kernelAnchorArr,
                             int channels, BT maxmin)
{
    D         res       = cuda::SetAll<D>(maxmin);
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel   = blockIdx.z % channels;
    const int batch_idx = blockIdx.z / channels;

    const int width  = dst.width(batch_idx, channel);
    const int height = dst.height(batch_idx, channel);

    if (x >= width || y >= height)
        return;

    int2 kernelSize = kernelSizeArr[batch_idx];
    int2 anchor     = kernelAnchorArr[batch_idx];

    if (kernelSize.x == 3 && kernelSize.y == 3 && anchor.x == 1 && anchor.y == 1 && x > 0 && y > 0 && x + 1 < width
        && y + 1 < height)
    {
        res = dilate3x3Interior(rawSrc, batch_idx, channel, y, x, res);
    }
    else
    {
        if constexpr (UseGenericInterior)
        {
            if (IsMorphInterior(kernelSize, anchor, x, y, width, height))
            {
                res = dilateInterior(rawSrc, batch_idx, channel, y, x, kernelSize, anchor, res);
            }
            else
            {
                int4 srcCoord = {0, 0, channel, batch_idx};
                for (int i = 0; i < kernelSize.y; ++i)
                {
                    srcCoord.y = y - anchor.y + i;

                    for (int j = 0; j < kernelSize.x; ++j)
                    {
                        srcCoord.x = x - anchor.x + j;

                        res = cuda::max(res, src[srcCoord]);
                    }
                }
            }
        }
        else
        {
            int4 srcCoord = {0, 0, channel, batch_idx};
            for (int i = 0; i < kernelSize.y; ++i)
            {
                srcCoord.y = y - anchor.y + i;

                for (int j = 0; j < kernelSize.x; ++j)
                {
                    srcCoord.x = x - anchor.x + j;

                    res = cuda::max(res, src[srcCoord]);
                }
            }
        }
    }

    *dst.ptr(batch_idx, channel, y, x) = cuda::SaturateCast<D>(res);
}

template<bool UseGenericInterior, class SrcWrapper, class RawSrcWrapper, class DstWrapper,
         typename D = typename DstWrapper::ValueType, typename BT = typename cuda::BaseType<D>>
__global__ void erode(const SrcWrapper src, const RawSrcWrapper rawSrc, DstWrapper dst,
                      cuda::Tensor1DWrap<int2> kernelSizeArr, cuda::Tensor1DWrap<int2> kernelAnchorArr, BT maxmin)
{
    D         res       = cuda::SetAll<D>(maxmin);
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int width  = dst.width(batch_idx);
    const int height = dst.height(batch_idx);

    if (x >= width || y >= height)
        return;

    int2 kernelSize = kernelSizeArr[batch_idx];
    int2 anchor     = kernelAnchorArr[batch_idx];

    if (kernelSize.x == 3 && kernelSize.y == 3 && anchor.x == 1 && anchor.y == 1 && x > 0 && y > 0 && x + 1 < width
        && y + 1 < height)
    {
        res = erode3x3Interior(rawSrc, batch_idx, y, x, res);
    }
    else
    {
        if constexpr (UseGenericInterior)
        {
            if (IsMorphInterior(kernelSize, anchor, x, y, width, height))
            {
                res = erodeInterior(rawSrc, batch_idx, y, x, kernelSize, anchor, res);
            }
            else
            {
                int3 srcCoord = {0, 0, batch_idx};
                for (int i = 0; i < kernelSize.y; ++i)
                {
                    srcCoord.y = y - anchor.y + i;

                    for (int j = 0; j < kernelSize.x; ++j)
                    {
                        srcCoord.x = x - anchor.x + j;

                        res = cuda::min(res, src[srcCoord]);
                    }
                }
            }
        }
        else
        {
            int3 srcCoord = {0, 0, batch_idx};
            for (int i = 0; i < kernelSize.y; ++i)
            {
                srcCoord.y = y - anchor.y + i;

                for (int j = 0; j < kernelSize.x; ++j)
                {
                    srcCoord.x = x - anchor.x + j;

                    res = cuda::min(res, src[srcCoord]);
                }
            }
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<D>(res);
}

template<bool UseGenericInterior, class SrcWrapper, class RawSrcWrapper, class DstWrapper,
         typename D = typename DstWrapper::ValueType, typename BT = typename cuda::BaseType<D>>
__global__ void erodePlanar(const SrcWrapper src, const RawSrcWrapper rawSrc, DstWrapper dst,
                            cuda::Tensor1DWrap<int2> kernelSizeArr, cuda::Tensor1DWrap<int2> kernelAnchorArr,
                            int channels, BT maxmin)
{
    D         res       = cuda::SetAll<D>(maxmin);
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel   = blockIdx.z % channels;
    const int batch_idx = blockIdx.z / channels;

    const int width  = dst.width(batch_idx, channel);
    const int height = dst.height(batch_idx, channel);

    if (x >= width || y >= height)
        return;

    int2 kernelSize = kernelSizeArr[batch_idx];
    int2 anchor     = kernelAnchorArr[batch_idx];

    if (kernelSize.x == 3 && kernelSize.y == 3 && anchor.x == 1 && anchor.y == 1 && x > 0 && y > 0 && x + 1 < width
        && y + 1 < height)
    {
        res = erode3x3Interior(rawSrc, batch_idx, channel, y, x, res);
    }
    else
    {
        if constexpr (UseGenericInterior)
        {
            if (IsMorphInterior(kernelSize, anchor, x, y, width, height))
            {
                res = erodeInterior(rawSrc, batch_idx, channel, y, x, kernelSize, anchor, res);
            }
            else
            {
                int4 srcCoord = {0, 0, channel, batch_idx};
                for (int i = 0; i < kernelSize.y; ++i)
                {
                    srcCoord.y = y - anchor.y + i;

                    for (int j = 0; j < kernelSize.x; ++j)
                    {
                        srcCoord.x = x - anchor.x + j;

                        res = cuda::min(res, src[srcCoord]);
                    }
                }
            }
        }
        else
        {
            int4 srcCoord = {0, 0, channel, batch_idx};
            for (int i = 0; i < kernelSize.y; ++i)
            {
                srcCoord.y = y - anchor.y + i;

                for (int j = 0; j < kernelSize.x; ++j)
                {
                    srcCoord.x = x - anchor.x + j;

                    res = cuda::min(res, src[srcCoord]);
                }
            }
        }
    }

    *dst.ptr(batch_idx, channel, y, x) = cuda::SaturateCast<D>(res);
}

template<bool UseGenericInterior, class SrcWrapper, class RawSrcWrapper, class DstWrapper,
         typename D = typename DstWrapper::ValueType, typename BT = typename cuda::BaseType<D>>
__global__ void dilatePlanarChannels(const SrcWrapper src, const RawSrcWrapper rawSrc, DstWrapper dst,
                                     cuda::Tensor1DWrap<int2> kernelSizeArr, cuda::Tensor1DWrap<int2> kernelAnchorArr,
                                     int channels, BT maxmin)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int2 kernelSize = kernelSizeArr[batch_idx];
    int2 anchor     = kernelAnchorArr[batch_idx];

    for (int channel = 0; channel < channels; ++channel)
    {
        const int width  = dst.width(batch_idx, channel);
        const int height = dst.height(batch_idx, channel);

        if (x >= width || y >= height)
            continue;

        D res = cuda::SetAll<D>(maxmin);

        if (kernelSize.x == 3 && kernelSize.y == 3 && anchor.x == 1 && anchor.y == 1 && x > 0 && y > 0 && x + 1 < width
            && y + 1 < height)
        {
            res = dilate3x3Interior(rawSrc, batch_idx, channel, y, x, res);
        }
        else
        {
            if constexpr (UseGenericInterior)
            {
                if (IsMorphInterior(kernelSize, anchor, x, y, width, height))
                {
                    res = dilateInterior(rawSrc, batch_idx, channel, y, x, kernelSize, anchor, res);
                }
                else
                {
                    int4 srcCoord = {0, 0, channel, batch_idx};
                    for (int i = 0; i < kernelSize.y; ++i)
                    {
                        srcCoord.y = y - anchor.y + i;

                        for (int j = 0; j < kernelSize.x; ++j)
                        {
                            srcCoord.x = x - anchor.x + j;

                            res = cuda::max(res, src[srcCoord]);
                        }
                    }
                }
            }
            else
            {
                int4 srcCoord = {0, 0, channel, batch_idx};
                for (int i = 0; i < kernelSize.y; ++i)
                {
                    srcCoord.y = y - anchor.y + i;

                    for (int j = 0; j < kernelSize.x; ++j)
                    {
                        srcCoord.x = x - anchor.x + j;

                        res = cuda::max(res, src[srcCoord]);
                    }
                }
            }
        }

        *dst.ptr(batch_idx, channel, y, x) = cuda::SaturateCast<D>(res);
    }
}

template<bool UseGenericInterior, class SrcWrapper, class RawSrcWrapper, class DstWrapper,
         typename D = typename DstWrapper::ValueType, typename BT = typename cuda::BaseType<D>>
__global__ void erodePlanarChannels(const SrcWrapper src, const RawSrcWrapper rawSrc, DstWrapper dst,
                                    cuda::Tensor1DWrap<int2> kernelSizeArr, cuda::Tensor1DWrap<int2> kernelAnchorArr,
                                    int channels, BT maxmin)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int2 kernelSize = kernelSizeArr[batch_idx];
    int2 anchor     = kernelAnchorArr[batch_idx];

    for (int channel = 0; channel < channels; ++channel)
    {
        const int width  = dst.width(batch_idx, channel);
        const int height = dst.height(batch_idx, channel);

        if (x >= width || y >= height)
            continue;

        D res = cuda::SetAll<D>(maxmin);

        if (kernelSize.x == 3 && kernelSize.y == 3 && anchor.x == 1 && anchor.y == 1 && x > 0 && y > 0 && x + 1 < width
            && y + 1 < height)
        {
            res = erode3x3Interior(rawSrc, batch_idx, channel, y, x, res);
        }
        else
        {
            if constexpr (UseGenericInterior)
            {
                if (IsMorphInterior(kernelSize, anchor, x, y, width, height))
                {
                    res = erodeInterior(rawSrc, batch_idx, channel, y, x, kernelSize, anchor, res);
                }
                else
                {
                    int4 srcCoord = {0, 0, channel, batch_idx};
                    for (int i = 0; i < kernelSize.y; ++i)
                    {
                        srcCoord.y = y - anchor.y + i;

                        for (int j = 0; j < kernelSize.x; ++j)
                        {
                            srcCoord.x = x - anchor.x + j;

                            res = cuda::min(res, src[srcCoord]);
                        }
                    }
                }
            }
            else
            {
                int4 srcCoord = {0, 0, channel, batch_idx};
                for (int i = 0; i < kernelSize.y; ++i)
                {
                    srcCoord.y = y - anchor.y + i;

                    for (int j = 0; j < kernelSize.x; ++j)
                    {
                        srcCoord.x = x - anchor.x + j;

                        res = cuda::min(res, src[srcCoord]);
                    }
                }
            }
        }

        *dst.ptr(batch_idx, channel, y, x) = cuda::SaturateCast<D>(res);
    }
}

template<typename D, NVCVBorderType B>
void MorphFilter2DCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                         const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &kMasks,
                         const TensorDataStridedCuda &kAnchors, NVCVMorphologyType morph_type,
                         bool enableGenericInterior, cudaStream_t stream)
{
    cuda::Tensor1DWrap<int2> kernelSizeTensor(kMasks);
    cuda::Tensor1DWrap<int2> kernelAnchorTensor(kAnchors);

    Size2D outMaxSize = outData.maxSize();
    int    maxWidth   = outMaxSize.w;
    int    maxHeight  = outMaxSize.h;

    dim3 block(16, 16);
    dim3 grid(divUp(maxWidth, block.x), divUp(maxHeight, block.y), outData.numImages());

    using BT = nvcv::cuda::BaseType<D>;
    BT val   = (morph_type == NVCVMorphologyType::NVCV_DILATE) ? std::numeric_limits<BT>::min()
                                                               : std::numeric_limits<BT>::max();

    cuda::BorderVarShapeWrap<const D, B>  src(inData, cuda::SetAll<D>(val));
    cuda::ImageBatchVarShapeWrap<const D> rawSrc(inData);
    cuda::ImageBatchVarShapeWrap<D>       dst(outData);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    if (morph_type == NVCVMorphologyType::NVCV_ERODE)
    {
        if (enableGenericInterior)
        {
            erode<true><<<grid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor, kernelAnchorTensor, val);
        }
        else
        {
            erode<false><<<grid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor, kernelAnchorTensor, val);
        }
        checkKernelErrors();
    }
    else if (morph_type == NVCVMorphologyType::NVCV_DILATE)
    {
        if (enableGenericInterior)
        {
            dilate<true><<<grid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor, kernelAnchorTensor, val);
        }
        else
        {
            dilate<false><<<grid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor, kernelAnchorTensor, val);
        }
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D, NVCVBorderType B>
void MorphFilter2DCallerPlanar(const ImageBatchVarShapeDataStridedCuda &inData,
                               const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &kMasks,
                               const TensorDataStridedCuda &kAnchors, NVCVMorphologyType morph_type, int channels,
                               bool enableGenericInterior, cudaStream_t stream)
{
    cuda::Tensor1DWrap<int2> kernelSizeTensor(kMasks);
    cuda::Tensor1DWrap<int2> kernelAnchorTensor(kAnchors);

    Size2D outMaxSize = outData.maxSize();
    int    maxWidth   = outMaxSize.w;
    int    maxHeight  = outMaxSize.h;

    dim3 block(16, 16);
    dim3 grid(divUp(maxWidth, block.x), divUp(maxHeight, block.y), channels * outData.numImages());

    using BT = nvcv::cuda::BaseType<D>;
    BT val   = (morph_type == NVCVMorphologyType::NVCV_DILATE) ? std::numeric_limits<BT>::min()
                                                               : std::numeric_limits<BT>::max();

    cuda::BorderVarShapeWrap<const D, B>  src(inData, cuda::SetAll<D>(val));
    cuda::ImageBatchVarShapeWrap<const D> rawSrc(inData);
    cuda::ImageBatchVarShapeWrap<D>       dst(outData);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    if (morph_type == NVCVMorphologyType::NVCV_ERODE)
    {
        if constexpr (std::is_same_v<D, uchar>)
        {
            dim3 channelGrid(divUp(maxWidth, block.x), divUp(maxHeight, block.y), outData.numImages());
            if (enableGenericInterior)
            {
                erodePlanarChannels<true><<<channelGrid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor,
                                                                             kernelAnchorTensor, channels, val);
            }
            else
            {
                erodePlanarChannels<false><<<channelGrid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor,
                                                                              kernelAnchorTensor, channels, val);
            }
        }
        else
        {
            if (enableGenericInterior)
            {
                erodePlanar<true>
                    <<<grid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor, kernelAnchorTensor, channels, val);
            }
            else
            {
                erodePlanar<false>
                    <<<grid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor, kernelAnchorTensor, channels, val);
            }
        }
        checkKernelErrors();
    }
    else if (morph_type == NVCVMorphologyType::NVCV_DILATE)
    {
        if constexpr (std::is_same_v<D, uchar>)
        {
            dim3 channelGrid(divUp(maxWidth, block.x), divUp(maxHeight, block.y), outData.numImages());
            if (enableGenericInterior)
            {
                dilatePlanarChannels<true><<<channelGrid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor,
                                                                              kernelAnchorTensor, channels, val);
            }
            else
            {
                dilatePlanarChannels<false><<<channelGrid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor,
                                                                               kernelAnchorTensor, channels, val);
            }
        }
        else
        {
            if (enableGenericInterior)
            {
                dilatePlanar<true>
                    <<<grid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor, kernelAnchorTensor, channels, val);
            }
            else
            {
                dilatePlanar<false>
                    <<<grid, block, 0, stream>>>(src, rawSrc, dst, kernelSizeTensor, kernelAnchorTensor, channels, val);
            }
        }
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void MorphFilter2D(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                   const TensorDataStridedCuda &kMasks, const TensorDataStridedCuda &kAnchors,
                   NVCVMorphologyType morph_type, NVCVBorderType borderMode, bool enableGenericInterior,
                   cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &kMasks,
                           const TensorDataStridedCuda &kAnchors, NVCVMorphologyType morph_type,
                           bool enableGenericInterior, cudaStream_t stream);

    static const func_t funcs[]
        = {MorphFilter2DCaller<D, NVCV_BORDER_CONSTANT>, MorphFilter2DCaller<D, NVCV_BORDER_REPLICATE>,
           MorphFilter2DCaller<D, NVCV_BORDER_REFLECT>, MorphFilter2DCaller<D, NVCV_BORDER_WRAP>,
           MorphFilter2DCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, kMasks, kAnchors, morph_type, enableGenericInterior, stream);
}

template<typename D>
void MorphFilter2DPlanar(const ImageBatchVarShapeDataStridedCuda &inData,
                         const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &kMasks,
                         const TensorDataStridedCuda &kAnchors, NVCVMorphologyType morph_type,
                         NVCVBorderType borderMode, int channels, bool enableGenericInterior, cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &kMasks,
                           const TensorDataStridedCuda &kAnchors, NVCVMorphologyType morph_type, int channels,
                           bool enableGenericInterior, cudaStream_t stream);

    static const func_t funcs[]
        = {MorphFilter2DCallerPlanar<D, NVCV_BORDER_CONSTANT>, MorphFilter2DCallerPlanar<D, NVCV_BORDER_REPLICATE>,
           MorphFilter2DCallerPlanar<D, NVCV_BORDER_REFLECT>, MorphFilter2DCallerPlanar<D, NVCV_BORDER_WRAP>,
           MorphFilter2DCallerPlanar<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, kMasks, kAnchors, morph_type, channels, enableGenericInterior, stream);
}

ErrorCode MorphologyVarShape::infer(const nvcv::ImageBatchVarShape &inBatch, const nvcv::ImageBatchVarShape &outBatch,
                                    NVCVMorphologyType morph_type, const TensorDataStridedCuda &masks,
                                    const TensorDataStridedCuda &anchors, bool noop, NVCVBorderType borderMode,
                                    bool enableGenericInterior, cudaStream_t stream)
{
    auto inData = inBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }
    auto outData = outBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    DataFormat input_format  = GetLegacyDataFormat(*inData);
    DataFormat output_format = GetLegacyDataFormat(*outData);
    DataType   data_type     = GetLegacyDataType(inData->uniqueFormat());

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;
    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    const bool isPlanar = IsPlanar(format);

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!(morph_type == NVCVMorphologyType::NVCV_ERODE || morph_type == NVCVMorphologyType::NVCV_DILATE))
    {
        LOG_ERROR("Invalid morph_type " << morph_type);
        return ErrorCode::INVALID_PARAMETER;
    }

    const int channels = inData->uniqueFormat().numChannels();

    if (!(channels == 1 || channels == 3 || channels == 4))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (isPlanar && static_cast<int64_t>(inData->numImages()) * channels > 65535)
    {
        LOG_ERROR("Planar Morphology requires numImages * channels <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_PARAMETER;
    }

    if (noop)
    {
        for (auto init = inBatch.begin(), outit = outBatch.begin(); init != inBatch.end() && outit != outBatch.end();
             ++init, ++outit)
        {
            const Image &inimg      = *init;
            const Image &outimg     = *outit;
            auto         inimgdata  = inimg.exportData<ImageDataStridedCuda>();
            auto         outimgdata = outimg.exportData<ImageDataStridedCuda>();
            for (int32_t p = 0; p < inimgdata->numPlanes(); ++p)
            {
                const ImagePlaneStrided &inplane  = inimgdata->plane(p);
                const ImagePlaneStrided &outplane = outimgdata->plane(p);
                const size_t             rowBytes
                    = static_cast<size_t>(inplane.width) * inData->uniqueFormat().planePixelStrideBytes(p);
                checkCudaErrors(cudaMemcpy2DAsync(outplane.basePtr, outplane.rowStride, inplane.basePtr,
                                                  inplane.rowStride, rowBytes, inplane.height, cudaMemcpyDeviceToDevice,
                                                  stream));
            }
        }
        return ErrorCode::SUCCESS;
    }

    dim3                     block(32), grid(divUp(inData->numImages(), 32));
    cuda::Tensor1DWrap<int2> kmasks(masks), kanchors(anchors);
    UpdateMasksAnchors<<<grid, block, 0, stream>>>(kmasks, kanchors, inData->numImages(), 1);

    typedef void (*filter2D_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                               const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &kMasks,
                               const TensorDataStridedCuda &kAnchors, NVCVMorphologyType morph_type,
                               NVCVBorderType borderMode, bool enableGenericInterior, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { MorphFilter2D<uchar>, 0,  MorphFilter2D<uchar3>,  MorphFilter2D<uchar4>},
        {                    0, 0,                      0,                      0},
        {MorphFilter2D<ushort>, 0, MorphFilter2D<ushort3>, MorphFilter2D<ushort4>},
        {                    0, 0,                      0,                      0},
        {                    0, 0,                      0,                      0},
        { MorphFilter2D<float>, 0,  MorphFilter2D<float3>,  MorphFilter2D<float4>},
    };

    if (isPlanar)
    {
        typedef void (*filter2D_planar_t)(
            const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
            const TensorDataStridedCuda &kMasks, const TensorDataStridedCuda &kAnchors, NVCVMorphologyType morph_type,
            NVCVBorderType borderMode, int channels, bool enableGenericInterior, cudaStream_t stream);

        static const filter2D_planar_t planarFuncs[6] = {
            MorphFilter2DPlanar<uchar>, 0, MorphFilter2DPlanar<ushort>, 0, 0, MorphFilter2DPlanar<float>,
        };

        planarFuncs[data_type](*inData, *outData, masks, anchors, morph_type, borderMode, channels,
                               enableGenericInterior, stream);
    }
    else
    {
        funcs[data_type][channels - 1](*inData, *outData, masks, anchors, morph_type, borderMode, enableGenericInterior,
                                       stream);
    }

    return ErrorCode::SUCCESS;
}
} // namespace nvcv::legacy::cuda_op

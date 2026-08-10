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
#include "threshold_util.cuh"

#include <type_traits>

using namespace nvcv::legacy::helpers;

using namespace nvcv::legacy::cuda_op;

using namespace nvcv::cuda;

constexpr int kU8BinaryElementsPerThread = sizeof(uint4) / sizeof(uchar);

static __device__ __forceinline__ uint4 SetAllU8Pack(uchar value)
{
    unsigned int word = 0x01010101u * value;
    uint4        out;
    out.x = word;
    out.y = word;
    out.z = word;
    out.w = word;
    return out;
}

static __device__ __forceinline__ uint4 BinaryThresholdU8Pack(uint4 in, uchar thresh, uchar maxval)
{
    uint4  out;
    uchar *inval  = reinterpret_cast<uchar *>(&in);
    uchar *outval = reinterpret_cast<uchar *>(&out);

#pragma unroll
    for (int i = 0; i < kU8BinaryElementsPerThread; i++) outval[i] = inval[i] > thresh ? maxval : 0;

    return out;
}

template<typename T, typename P = MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Binary_overflow(ImageBatchVarShapeWrapNHWC<T> src, ImageBatchVarShapeWrapNHWC<T> dst,
                                Tensor1DWrap<double, int32_t> _thresh, Tensor1DWrap<double, int32_t> _maxval,
                                int channel)
{
    int cn        = NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T      MAX     = TypeTraits<T>::max;
    T      MIN     = TypeTraits<T>::min;
    double maxv    = _maxval[batch];
    double th      = _thresh[batch];
    int    imaxval = round(maxv);
    T      maxval  = nvcv::cuda::SaturateCast<T>(imaxval);
    int    ithresh = floor(th);

    int loop = width * channel - w;
    int c    = w % channel;
    w        = w / channel;
    if (loop >= cn)
    {
        P out;
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T  thresh = (T)ithresh;
            P  in     = LoadPacked<P>(src.ptr(batch, h, w, c));
            T *inval  = reinterpret_cast<T *>((void *)&in);
            T  outval[4];
#pragma unroll
            for (int i = 0; i < cn; i++)
            {
                outval[i]             = inval[i] > thresh ? maxval : 0;
                GetElement<P>(out, i) = outval[i];
            }
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }
        if (ithresh < MIN)
        {
            out = SetAll<P>(maxval);
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }

        out = SetAll<P>(0);
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T thresh = (T)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
            {
                T inval                        = *(src.ptr(batch, h, w, c) + i);
                *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? maxval : 0;
            }
            return;
        }
        if (ithresh < MIN)
        {
#pragma unroll
            for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = maxval;
            return;
        }

#pragma unroll
        for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = 0;
    }
}

__global__ void Binary_overflow_u8_nix16(ImageBatchVarShapeWrapNHWC<uchar> src, ImageBatchVarShapeWrapNHWC<uchar> dst,
                                         Tensor1DWrap<double, int32_t> _thresh, Tensor1DWrap<double, int32_t> _maxval,
                                         int channel)
{
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch    = blockIdx.z;
    int width    = src.width(batch);
    int height   = src.height(batch);
    int rowElems = width * channel;
    if (rowElems == 0)
        return;

    int threadCol = (rowElems + kU8BinaryElementsPerThread - 1) / kU8BinaryElementsPerThread;
    int h         = globalid / threadCol;
    int elem      = (globalid % threadCol) * kU8BinaryElementsPerThread;
    if (h >= height || elem >= rowElems)
        return;

    int    imaxval = round(_maxval[batch]);
    uchar  maxval  = nvcv::cuda::SaturateCast<uchar>(imaxval);
    int    ithresh = floor(_thresh[batch]);
    int    loop    = rowElems - elem;
    int    c       = elem % channel;
    int    w       = elem / channel;
    uchar *srcRow  = src.ptr(batch, h, w, c);
    uchar *dstRow  = dst.ptr(batch, h, w, c);
    bool   aligned
        = ((reinterpret_cast<std::uintptr_t>(srcRow) | reinterpret_cast<std::uintptr_t>(dstRow)) & (alignof(uint4) - 1))
       == 0;

    if (loop >= kU8BinaryElementsPerThread)
    {
        uint4 out;
        if (ithresh >= TypeTraits<uchar>::min && ithresh <= TypeTraits<uchar>::max)
        {
            uchar thresh = (uchar)ithresh;
            if (aligned)
            {
                uint4 in                             = *(reinterpret_cast<uint4 *>(srcRow));
                *(reinterpret_cast<uint4 *>(dstRow)) = BinaryThresholdU8Pack(in, thresh, maxval);
            }
            else
            {
#pragma unroll
                for (int i = 0; i < kU8BinaryElementsPerThread; i++)
                {
                    uchar inval = srcRow[i];
                    dstRow[i]   = inval > thresh ? maxval : 0;
                }
            }
            return;
        }

        uchar value;
        if (ithresh < TypeTraits<uchar>::min)
        {
            out   = SetAllU8Pack(maxval);
            value = maxval;
        }
        else
        {
            out   = SetAllU8Pack(0);
            value = 0;
        }

        if (aligned)
            *(reinterpret_cast<uint4 *>(dstRow)) = out;
        else
        {
#pragma unroll
            for (int i = 0; i < kU8BinaryElementsPerThread; i++) dstRow[i] = value;
        }
    }
    else
    {
        if (ithresh >= TypeTraits<uchar>::min && ithresh <= TypeTraits<uchar>::max)
        {
            uchar thresh = (uchar)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
            {
                uchar inval = srcRow[i];
                dstRow[i]   = inval > thresh ? maxval : 0;
            }
            return;
        }

        uchar out = ithresh < TypeTraits<uchar>::min ? maxval : 0;
#pragma unroll
        for (int i = 0; i < loop; i++) dstRow[i] = out;
    }
}

template<typename T, typename P = MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Binary_Generic(ImageBatchVarShapeWrapNHWC<T> src, ImageBatchVarShapeWrapNHWC<T> dst,
                               Tensor1DWrap<double, int32_t> _thresh, Tensor1DWrap<double, int32_t> _maxval,
                               int channel)
{
    int cn        = NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T   maxval = (T)_maxval[batch];
    T   thresh = (T)_thresh[batch];
    int loop   = width * channel - w;
    int c      = w % channel;
    w          = w / channel;

    if (loop >= cn)
    {
        P  out;
        P  in    = LoadPacked<P>(src.ptr(batch, h, w, c));
        T *inval = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]             = inval[i] > thresh ? maxval : 0;
            GetElement<P>(out, i) = outval[i];
        }
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < loop; i++)
        {
            T inval                        = *(src.ptr(batch, h, w, c) + i);
            *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? maxval : 0;
        }
    }
}

template<typename T, typename P = MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void BinaryInv_overflow(ImageBatchVarShapeWrapNHWC<T> src, ImageBatchVarShapeWrapNHWC<T> dst,
                                   Tensor1DWrap<double, int32_t> _thresh, Tensor1DWrap<double, int32_t> _maxval,
                                   int channel)
{
    int cn        = NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T      MAX     = TypeTraits<T>::max;
    T      MIN     = TypeTraits<T>::min;
    double maxv    = _maxval[batch];
    double th      = _thresh[batch];
    int    imaxval = round(maxv);
    T      maxval  = nvcv::cuda::SaturateCast<T>(imaxval);
    int    ithresh = floor(th);

    int loop = width * channel - w;
    int c    = w % channel;
    w        = w / channel;
    if (loop >= cn)
    {
        P out;
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T  thresh = (T)ithresh;
            P  in     = LoadPacked<P>(src.ptr(batch, h, w, c));
            T *inval  = reinterpret_cast<T *>((void *)&in);
            T  outval[4];
#pragma unroll
            for (int i = 0; i < cn; i++)
            {
                outval[i]             = inval[i] > thresh ? 0 : maxval;
                GetElement<P>(out, i) = outval[i];
            }
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }
        if (ithresh < MIN)
        {
            out = SetAll<P>(0);
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }

        out = SetAll<P>(maxval);
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T thresh = (T)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
                *(dst.ptr(batch, h, w, c) + i) = *(src.ptr(batch, h, w, c) + i) > thresh ? 0 : maxval;
            return;
        }
        if (ithresh < MIN)
        {
#pragma unroll
            for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = 0;
            return;
        }

#pragma unroll
        for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = maxval;
    }
}

template<typename T, typename P = MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void BinaryInv_Generic(ImageBatchVarShapeWrapNHWC<T> src, ImageBatchVarShapeWrapNHWC<T> dst,
                                  Tensor1DWrap<double, int32_t> _thresh, Tensor1DWrap<double, int32_t> _maxval,
                                  int channel)
{
    int cn        = NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T   maxval = (T)_maxval[batch];
    T   thresh = (T)_thresh[batch];
    int loop   = width * channel - w;
    int c      = w % channel;
    w          = w / channel;

    if (loop >= cn)
    {
        P  out;
        P  in    = LoadPacked<P>(src.ptr(batch, h, w, c));
        T *inval = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]             = inval[i] > thresh ? 0 : maxval;
            GetElement<P>(out, i) = outval[i];
        }
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < loop; i++)
        {
            T inval                        = *(src.ptr(batch, h, w, c) + i);
            *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? 0 : maxval;
        }
    }
}

template<typename T, typename P = MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Trunc_overflow(ImageBatchVarShapeWrapNHWC<T> src, ImageBatchVarShapeWrapNHWC<T> dst,
                               Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T      MAX     = TypeTraits<T>::max;
    T      MIN     = TypeTraits<T>::min;
    double th      = _thresh[batch];
    int    ithresh = floor(th);

    int loop = width * channel - w;
    int c    = w % channel;
    w        = w / channel;
    if (loop >= cn)
    {
        P out;
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T  thresh = (T)ithresh;
            P  in     = LoadPacked<P>(src.ptr(batch, h, w, c));
            T *inval  = reinterpret_cast<T *>((void *)&in);
            T  outval[4];
#pragma unroll
            for (int i = 0; i < cn; i++)
            {
                outval[i]             = inval[i] > thresh ? thresh : inval[i];
                GetElement<P>(out, i) = outval[i];
            }
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }
        if (ithresh < MIN)
        {
            out = SetAll<P>(MIN);
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }

        StorePacked(dst.ptr(batch, h, w, c), LoadPacked<P>(src.ptr(batch, h, w, c)));
    }
    else
    {
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T thresh = (T)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
            {
                T inval                        = *(src.ptr(batch, h, w, c) + i);
                *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? thresh : inval;
            }
            return;
        }
        if (ithresh < MIN)
        {
#pragma unroll
            for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = MIN;
            return;
        }

#pragma unroll
        for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = *(src.ptr(batch, h, w, c) + i);
    }
}

template<typename T, typename P = MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Trunc_Generic(ImageBatchVarShapeWrapNHWC<T> src, ImageBatchVarShapeWrapNHWC<T> dst,
                              Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T   thresh = (T)_thresh[batch];
    int loop   = width * channel - w;
    int c      = w % channel;
    w          = w / channel;

    if (loop >= cn)
    {
        P  out;
        P  in    = LoadPacked<P>(src.ptr(batch, h, w, c));
        T *inval = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]             = inval[i] > thresh ? thresh : inval[i];
            GetElement<P>(out, i) = outval[i];
        }
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < loop; i++)
        {
            T inval                        = *(src.ptr(batch, h, w, c) + i);
            *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? thresh : inval;
        }
    }
}

template<typename T, typename P = MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Tozero_overflow(ImageBatchVarShapeWrapNHWC<T> src, ImageBatchVarShapeWrapNHWC<T> dst,
                                Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T      MAX     = TypeTraits<T>::max;
    T      MIN     = TypeTraits<T>::min;
    double th      = _thresh[batch];
    int    ithresh = floor(th);

    int loop = width * channel - w;
    int c    = w % channel;
    w        = w / channel;
    if (loop >= cn)
    {
        P out;
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T  thresh = (T)ithresh;
            P  in     = LoadPacked<P>(src.ptr(batch, h, w, c));
            T *inval  = reinterpret_cast<T *>((void *)&in);
            T  outval[4];
#pragma unroll
            for (int i = 0; i < cn; i++)
            {
                outval[i]             = inval[i] > thresh ? inval[i] : 0;
                GetElement<P>(out, i) = outval[i];
            }
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }
        if (ithresh < MIN)
        {
            StorePacked(dst.ptr(batch, h, w, c), LoadPacked<P>(src.ptr(batch, h, w, c)));
            return;
        }

        out = SetAll<P>(0);
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T thresh = (T)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
            {
                T inval                        = *(src.ptr(batch, h, w, c) + i);
                *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? inval : 0;
            }
            return;
        }
        if (ithresh < MIN)
        {
#pragma unroll
            for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = *(src.ptr(batch, h, w, c) + i);
            return;
        }

#pragma unroll
        for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = 0;
    }
}

template<typename T, typename P = MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Tozero_Generic(ImageBatchVarShapeWrapNHWC<T> src, ImageBatchVarShapeWrapNHWC<T> dst,
                               Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T   thresh = (T)_thresh[batch];
    int loop   = width * channel - w;
    int c      = w % channel;
    w          = w / channel;

    if (loop >= cn)
    {
        P  out;
        P  in    = LoadPacked<P>(src.ptr(batch, h, w, c));
        T *inval = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]             = inval[i] > thresh ? inval[i] : 0;
            GetElement<P>(out, i) = outval[i];
        }
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < loop; i++)
        {
            T inval                        = *(src.ptr(batch, h, w, c) + i);
            *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? inval : 0;
        }
    }
}

template<typename T, typename P = MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void TozeroInv_overflow(ImageBatchVarShapeWrapNHWC<T> src, ImageBatchVarShapeWrapNHWC<T> dst,
                                   Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T      MAX     = TypeTraits<T>::max;
    T      MIN     = TypeTraits<T>::min;
    double th      = _thresh[batch];
    int    ithresh = floor(th);

    int loop = width * channel - w;
    int c    = w % channel;
    w        = w / channel;
    if (loop >= cn)
    {
        P out;
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T  thresh = (T)ithresh;
            P  in     = LoadPacked<P>(src.ptr(batch, h, w, c));
            T *inval  = reinterpret_cast<T *>((void *)&in);
            T  outval[4];
#pragma unroll
            for (int i = 0; i < cn; i++)
            {
                outval[i]             = inval[i] > thresh ? 0 : inval[i];
                GetElement<P>(out, i) = outval[i];
            }
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }
        if (ithresh < MIN)
        {
            out = SetAll<P>(0);
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }

        StorePacked(dst.ptr(batch, h, w, c), LoadPacked<P>(src.ptr(batch, h, w, c)));
    }
    else
    {
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T thresh = (T)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
            {
                T inval                        = *(src.ptr(batch, h, w, c) + i);
                *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? 0 : inval;
            }
            return;
        }
        if (ithresh < MIN)
        {
#pragma unroll
            for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = 0;
            return;
        }

#pragma unroll
        for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = *(src.ptr(batch, h, w, c) + i);
    }
}

template<typename T, typename P = MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void TozeroInv_Generic(ImageBatchVarShapeWrapNHWC<T> src, ImageBatchVarShapeWrapNHWC<T> dst,
                                  Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T   thresh = (T)_thresh[batch];
    int loop   = width * channel - w;
    int c      = w % channel;
    w          = w / channel;

    if (loop >= cn)
    {
        P  out;
        P  in    = LoadPacked<P>(src.ptr(batch, h, w, c));
        T *inval = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]             = inval[i] > thresh ? 0 : inval[i];
            GetElement<P>(out, i) = outval[i];
        }
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < loop; i++)
        {
            T inval                        = *(src.ptr(batch, h, w, c) + i);
            *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? 0 : inval;
        }
    }
}

template<typename T>
__device__ __forceinline__ T ThresholdOverflowValue(T inval, double th, double maxv, NVCVThresholdType type)
{
    T   maxType = TypeTraits<T>::max;
    T   minType = TypeTraits<T>::min;
    int imaxval = round(maxv);
    T   maxval  = nvcv::cuda::SaturateCast<T>(imaxval);
    int ithresh = floor(th);

    switch (type)
    {
    case NVCV_THRESH_BINARY:
        if (ithresh >= minType && ithresh <= maxType)
        {
            T thresh = (T)ithresh;
            return inval > thresh ? maxval : 0;
        }
        return ithresh < minType ? maxval : 0;
    case NVCV_THRESH_BINARY_INV:
        if (ithresh >= minType && ithresh <= maxType)
        {
            T thresh = (T)ithresh;
            return inval > thresh ? 0 : maxval;
        }
        return ithresh < minType ? 0 : maxval;
    case NVCV_THRESH_TRUNC:
        if (ithresh >= minType && ithresh <= maxType)
        {
            T thresh = (T)ithresh;
            return inval > thresh ? thresh : inval;
        }
        return ithresh < minType ? minType : inval;
    case NVCV_THRESH_TOZERO:
        if (ithresh >= minType && ithresh <= maxType)
        {
            T thresh = (T)ithresh;
            return inval > thresh ? inval : 0;
        }
        return ithresh < minType ? inval : 0;
    default: // NVCV_THRESH_TOZERO_INV
        if (ithresh >= minType && ithresh <= maxType)
        {
            T thresh = (T)ithresh;
            return inval > thresh ? 0 : inval;
        }
        return ithresh < minType ? 0 : inval;
    }
}

template<typename T>
__device__ __forceinline__ T ThresholdGenericValue(T inval, double th, double maxv, NVCVThresholdType type)
{
    T thresh = (T)th;
    T maxval = (T)maxv;

    switch (type)
    {
    case NVCV_THRESH_BINARY:
        return inval > thresh ? maxval : 0;
    case NVCV_THRESH_BINARY_INV:
        return inval > thresh ? 0 : maxval;
    case NVCV_THRESH_TRUNC:
        return inval > thresh ? thresh : inval;
    case NVCV_THRESH_TOZERO:
        return inval > thresh ? inval : 0;
    default: // NVCV_THRESH_TOZERO_INV
        return inval > thresh ? 0 : inval;
    }
}

template<typename T>
__global__ void ThresholdPlanar(ImageBatchVarShapeWrap<T> src, ImageBatchVarShapeWrap<T> dst,
                                Tensor1DWrap<double, int32_t> _thresh, Tensor1DWrap<double, int32_t> _maxval,
                                int channels, NVCVThresholdType type, DataType data_type)
{
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch    = blockIdx.z / channels;
    int plane    = blockIdx.z % channels;
    int width    = src.width(batch);
    int height   = src.height(batch);

    if (globalid >= width * height)
        return;

    int y = globalid / width;
    int x = globalid % width;

    T inval                      = *src.ptr(batch, plane, y, x);
    T out                        = (data_type == kCV_32F || data_type == kCV_64F)
                                     ? ThresholdGenericValue(inval, _thresh[batch], _maxval[batch], type)
                                     : ThresholdOverflowValue(inval, _thresh[batch], _maxval[batch], type);
    *dst.ptr(batch, plane, y, x) = out;
}

__global__ void hist_kernel(ImageBatchVarShapeWrapNHWC<uchar> img, int *histogram)
{
    __shared__ int hist[256];
    int            localid = threadIdx.x;
    hist[localid]          = 0;
    __syncthreads();

    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int cols      = img.width(batch);
    int rows      = img.height(batch);
    int threadCol = ceil((float)cols / 16);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * 16;

    if (h < rows)
    {
        if (w + 16 > cols)
        {
            for (int i = w; i < cols; i++) atomicAdd(&hist[*img.ptr(batch, h, i)], 1);
        }
        else
        {
            int4   src   = LoadPacked<int4>(img.ptr(batch, h, w));
            uchar *inval = reinterpret_cast<uchar *>((void *)&src);
            for (int i = 0; i < 16; i++) atomicAdd(&hist[inval[i]], 1);
        }
    }
    __syncthreads();

    int val = hist[localid];
    if (val > 0)
        atomicAdd(&histogram[blockIdx.z * 256 + localid], val);
}

__global__ void otsu_cal_varshape(int *histogram, Tensor1DWrap<double, int32_t> thresh,
                                  ImageBatchVarShapeWrapNHWC<uchar> img)
{
    int            localid = threadIdx.y * blockDim.x + threadIdx.x;
    int            size    = img.width((int)blockIdx.z) * img.height((int)blockIdx.z);
    __shared__ int hist[256];
    hist[localid] = histogram[blockIdx.z * 256 + localid];
    __syncthreads();

    __shared__ volatile double reduce[256];
    double                     mu, scale = 1. / size;

    // reduce to calculate the sum of 'i * histogram[i]' (mu)
    reduce[localid] = localid * (double)hist[localid];
    __syncthreads();

    if (localid < 128)
        reduce[localid] = reduce[localid] + reduce[localid + 128];
    __syncthreads();
    if (localid < 64)
        reduce[localid] = reduce[localid] + reduce[localid + 64];
    __syncthreads();
    if (localid < 32)
    {
        reduce[localid] = reduce[localid] + reduce[localid + 32];
        reduce[localid] = reduce[localid] + reduce[localid + 16];
        reduce[localid] = reduce[localid] + reduce[localid + 8];
        reduce[localid] = reduce[localid] + reduce[localid + 4];
        reduce[localid] = reduce[localid] + reduce[localid + 2];
        reduce[localid] = reduce[localid] + reduce[localid + 1];
    }
    __syncthreads();

    mu = reduce[0] * scale;
    __syncthreads();

    // reduce to calculate the prefix sum of histogram[i] (q1)
    // the prefix sum of histogram[i] = histogram[0] + histogram[1] + ... + histogram[i-1] + histogram[i]
    double q1   = hist[localid] * scale;
    int    lane = localid % 32, warp = localid / 32;
    // sum of q1 in warp
    double temp = q1;
    temp += __shfl_xor_sync(0xffffffff, temp, 1);
    temp += __shfl_xor_sync(0xffffffff, temp, 2);
    temp += __shfl_xor_sync(0xffffffff, temp, 4);
    temp += __shfl_xor_sync(0xffffffff, temp, 8);
    temp += __shfl_xor_sync(0xffffffff, temp, 16);
    if (lane == 0)
        reduce[warp] = temp;
    __syncthreads();
    // prefix scan of the sum
    if (warp == 0)
    {
        temp = reduce[lane];
        reduce[lane + 1] += temp;
        temp = reduce[lane];
        reduce[lane + 2] += temp;
        temp = reduce[lane];
        reduce[lane + 4] += temp;
        temp             = reduce[lane];
        reduce[lane]     = 0;
        reduce[lane + 1] = temp;
    }
    __syncthreads();
    // prefix scan in warp
    temp = __shfl_up_sync(0xffffffff, q1, 1);
    if (lane >= 1)
        q1 += temp;
    temp = __shfl_up_sync(0xffffffff, q1, 2);
    if (lane >= 2)
        q1 += temp;
    temp = __shfl_up_sync(0xffffffff, q1, 4);
    if (lane >= 4)
        q1 += temp;
    temp = __shfl_up_sync(0xffffffff, q1, 8);
    if (lane >= 8)
        q1 += temp;
    temp = __shfl_up_sync(0xffffffff, q1, 16);
    if (lane >= 16)
        q1 += temp;
    q1 += reduce[warp];
    double q2 = 1 - q1;
    __syncthreads();

    // reduce to calculate the prefix sum of i * histogram[i] (one)
    // the prefix sum of i * histogram[i] = 0*histogram[0] + 1*histogram[1] + ... + (i-1)*histogram[i-1] + i*histogram[i]
    double one = localid * hist[localid] * scale;
    // sum of q1 in warp
    temp = one;
    temp += __shfl_xor_sync(0xffffffff, temp, 1);
    temp += __shfl_xor_sync(0xffffffff, temp, 2);
    temp += __shfl_xor_sync(0xffffffff, temp, 4);
    temp += __shfl_xor_sync(0xffffffff, temp, 8);
    temp += __shfl_xor_sync(0xffffffff, temp, 16);
    if (lane == 0)
        reduce[warp] = temp;
    __syncthreads();
    // prefix scan of the sum
    if (warp == 0)
    {
        temp = reduce[lane];
        reduce[lane + 1] += temp;
        temp = reduce[lane];
        reduce[lane + 2] += temp;
        temp = reduce[lane];
        reduce[lane + 4] += temp;
        temp             = reduce[lane];
        reduce[lane]     = 0;
        reduce[lane + 1] = temp;
    }
    __syncthreads();
    // prefix scan in warp
    temp = __shfl_up_sync(0xffffffff, one, 1);
    if (lane >= 1)
        one += temp;
    temp = __shfl_up_sync(0xffffffff, one, 2);
    if (lane >= 2)
        one += temp;
    temp = __shfl_up_sync(0xffffffff, one, 4);
    if (lane >= 4)
        one += temp;
    temp = __shfl_up_sync(0xffffffff, one, 8);
    if (lane >= 8)
        one += temp;
    temp = __shfl_up_sync(0xffffffff, one, 16);
    if (lane >= 16)
        one += temp;
    one += reduce[warp];
    __syncthreads();

    // calulate sigma
    double mu1 = one / q1, mu2 = (mu - q1 * mu1) / q2;
    double sigma;
    if (min(q1, q2) < FLT_EPSILON || max(q1, q2) > 1. - FLT_EPSILON)
        sigma = -1;
    else
        sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);

    // find the coordinate with the largest sigma
    // reduce to find the largest sigma and record the cooridinate
    reduce[localid] = sigma;
    __shared__ uchar idx[256];
    idx[localid] = localid;
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
        thresh[(int)blockIdx.z] = (double)idx[0];
}

template<typename T>
void thresholdDispatch(const nvcv::ImageBatchVarShapeDataStridedCuda &input,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &output,
                       const nvcv::TensorDataStridedCuda &_thresh, const nvcv::TensorDataStridedCuda &_maxval,
                       NVCVThresholdType type, DataType data_type, cudaStream_t stream)
{
    (void)data_type;
    Tensor1DWrap<double, int32_t> thresh(_thresh);
    Tensor1DWrap<double, int32_t> maxval(_maxval);

    nvcv::Size2D maxsize = input.maxSize();
    int          batch   = input.numImages();
    int          channel = input.uniqueFormat().numChannels();

    ImageBatchVarShapeWrapNHWC<T> src_ptr(input, channel);
    ImageBatchVarShapeWrapNHWC<T> dst_ptr(output, channel);

    dim3 block(256);
    int  N  = sizeof(T) == 8 ? 2 : 4;
    int  td = divUp(maxsize.w * channel, N) * maxsize.h;
    dim3 grid(divUp(td, 256), 1, batch);

    switch (type)
    {
    case NVCV_THRESH_BINARY:
        if constexpr (std::is_floating_point_v<T>)
            Binary_Generic<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channel);
        else if constexpr (std::is_same_v<T, uchar>)
        {
            int  tdU8 = divUp(maxsize.w * channel, kU8BinaryElementsPerThread) * maxsize.h;
            dim3 gridU8(divUp(tdU8, 256), 1, batch);
            Binary_overflow_u8_nix16<<<gridU8, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channel);
        }
        else
            Binary_overflow<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channel);
        break;
    case NVCV_THRESH_BINARY_INV:
        if constexpr (std::is_floating_point_v<T>)
            BinaryInv_Generic<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channel);
        else
            BinaryInv_overflow<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channel);
        break;
    case NVCV_THRESH_TRUNC:
        if constexpr (std::is_floating_point_v<T>)
            Trunc_Generic<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        else
            Trunc_overflow<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        break;
    case NVCV_THRESH_TOZERO:
        if constexpr (std::is_floating_point_v<T>)
            Tozero_Generic<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        else
            Tozero_overflow<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        break;
    default: //NVCV_THRESH_TOZERO_INV
        if constexpr (std::is_floating_point_v<T>)
            TozeroInv_Generic<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        else
            TozeroInv_overflow<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        break;
    }

    checkKernelErrors();
}

template<typename T>
void thresholdDispatchPlanar(const nvcv::ImageBatchVarShapeDataStridedCuda &input,
                             const nvcv::ImageBatchVarShapeDataStridedCuda &output,
                             const nvcv::TensorDataStridedCuda &_thresh, const nvcv::TensorDataStridedCuda &_maxval,
                             int channels, NVCVThresholdType type, DataType data_type, cudaStream_t stream)
{
    Tensor1DWrap<double, int32_t> thresh(_thresh);
    Tensor1DWrap<double, int32_t> maxval(_maxval);

    nvcv::Size2D maxsize = input.maxSize();
    int          batch   = input.numImages();

    ImageBatchVarShapeWrap<T> src_ptr(input);
    ImageBatchVarShapeWrap<T> dst_ptr(output);

    dim3 block(256);
    int  td = maxsize.w * maxsize.h;
    dim3 grid(divUp(td, block.x), 1, batch * channels);
    ThresholdPlanar<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channels, type, data_type);
    checkKernelErrors();
}

static void getThreshVal_Triangle(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                  const nvcv::TensorDataStridedCuda &threshold, int *histogram, cudaStream_t stream)
{
    int batch = inData.numImages();
    checkCudaErrors(cudaMemsetAsync(histogram, 0, sizeof(int) * 256 * batch, stream));

    ImageBatchVarShapeWrapNHWC<uchar> wrap(inData, inData.uniqueFormat().numChannels());
    Tensor1DWrap<double, int32_t>     thresh(threshold);
    nvcv::Size2D                      maxsize = inData.maxSize();

    dim3 block(256);
    int  td = divUp(maxsize.w, 16) * maxsize.h;
    dim3 grid(divUp(td, 256), 1, batch);
    hist_kernel<<<grid, block, 0, stream>>>(wrap, histogram);

    dim3 block2(256);
    dim3 grid2(1, 1, batch);
    triangle_cal<<<grid2, block2, 0, stream>>>(histogram, thresh);
}

static void getThreshVal_Otsu(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                              const nvcv::TensorDataStridedCuda &threshold, int *histogram, cudaStream_t stream)
{
    int batch = inData.numImages();
    checkCudaErrors(cudaMemsetAsync(histogram, 0, sizeof(int) * 256 * batch, stream));

    ImageBatchVarShapeWrapNHWC<uchar> wrap(inData, inData.uniqueFormat().numChannels());
    Tensor1DWrap<double, int32_t>     thresh(threshold);
    nvcv::Size2D                      maxsize = inData.maxSize();

    dim3 block(256);
    int  td = divUp(maxsize.w, 16) * maxsize.h;
    dim3 grid(divUp(td, 256), 1, batch);
    hist_kernel<<<grid, block, 0, stream>>>(wrap, histogram);

    dim3 block2(256);
    dim3 grid2(1, 1, batch);
    otsu_cal_varshape<<<grid2, block2, 0, stream>>>(histogram, thresh, wrap);
}

namespace nvcv::legacy::cuda_op {

ThresholdVarShape::ThresholdVarShape(DataShape max_input_shape, DataShape max_output_shape, uint32_t type,
                                     int maxBatchSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_histogram(nullptr)
    , m_type(type)
    , m_maxBatchSize(maxBatchSize)
{
    if (maxBatchSize < 0)
    {
        LOG_ERROR("Invalid num of max batch size " << maxBatchSize);
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Parameter error!");
    }
    m_automatic_thresh = (m_type & ~NVCV_THRESH_MASK);
    if (m_automatic_thresh != 0)
    {
        cudaError_t err = cudaMalloc(&m_histogram, sizeof(int) * 256 * maxBatchSize);
        if (err != cudaSuccess)
        {
            LOG_ERROR("CUDA memory allocation error of size: " << sizeof(int) * 256 * maxBatchSize);
            throw LegacyCudaAllocationError("CUDA memory allocation error!");
        }
    }
}

ThresholdVarShape::~ThresholdVarShape()
{
    if (m_automatic_thresh != 0)
    {
        cudaError_t err = cudaFree(m_histogram);
        if (err != cudaSuccess)
            LOG_ERROR("CUDA memory free error, possible memory leak!");
    }
}

ErrorCode ThresholdVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                   const ImageBatchVarShapeDataStridedCuda &outData,
                                   const TensorDataStridedCuda &thresh, const TensorDataStridedCuda &maxval,
                                   cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(input_format == kNHWC || input_format == kHWC || input_format == kNCHW || input_format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << input_format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    const bool isPlanar = (input_format == kNCHW || input_format == kCHW);

    DataType in_data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(in_data_type == kCV_8U || in_data_type == kCV_16S || in_data_type == kCV_16U || in_data_type == kCV_32F
          || in_data_type == kCV_64F))
    {
        LOG_ERROR("Invalid Data Type " << in_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType out_data_type = helpers::GetLegacyDataType(outData.uniqueFormat());
    if (in_data_type != out_data_type)
    {
        LOG_ERROR("DataType of input and output must be equal, but got " << in_data_type << " and " << out_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    const int channels = inData.uniqueFormat().numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType thresh_data_type = GetLegacyDataType(thresh.dtype());
    if (thresh_data_type != kCV_64F)
    {
        LOG_ERROR("Invalid thresh DataType " << thresh_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int thresh_dim = thresh.layout().rank();
    if (thresh_dim != 1)
    {
        LOG_ERROR("Invalid thresh Dim " << thresh_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType maxval_data_type = GetLegacyDataType(maxval.dtype());
    if (maxval_data_type != kCV_64F)
    {
        LOG_ERROR("Invalid maxval DataType " << maxval_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int maxval_dim = maxval.layout().rank();
    if (maxval_dim != 1)
    {
        LOG_ERROR("Invalid maxval Dim " << maxval_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (m_automatic_thresh == ((uint32_t)NVCV_THRESH_OTSU | (uint32_t)NVCV_THRESH_TRIANGLE))
    {
        LOG_ERROR("Invalid Threshold Type " << m_type);
        return ErrorCode::INVALID_PARAMETER;
    }

    m_type &= (uint32_t)NVCV_THRESH_MASK;
    if ((m_type & (m_type - 1)) != 0)
    {
        LOG_ERROR("Invalid Threhold Type " << m_type);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (m_automatic_thresh != 0 && inData.numImages() > m_maxBatchSize)
    {
        LOG_ERROR("Input batch exceeds maxBatchSize");
        return ErrorCode::INVALID_PARAMETER;
    }
    if (isPlanar && static_cast<int64_t>(inData.numImages()) * channels > 65535)
    {
        LOG_ERROR("Planar Threshold requires numImages * channels <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_PARAMETER;
    }

    if (m_automatic_thresh == NVCV_THRESH_OTSU)
    {
        if (in_data_type != kCV_8U)
        {
            LOG_ERROR("Invalid Data Type " << in_data_type);
            return ErrorCode::INVALID_DATA_TYPE;
        }
        if (inData.uniqueFormat().numChannels() != 1)
        {
            LOG_ERROR("Only support 1 channel");
            return ErrorCode::INVALID_DATA_FORMAT;
        }
        getThreshVal_Otsu(inData, thresh, m_histogram, stream);
    }
    else if (m_automatic_thresh == NVCV_THRESH_TRIANGLE)
    {
        if (in_data_type != kCV_8U)
        {
            LOG_ERROR("Invalid Data Type " << in_data_type);
            return ErrorCode::INVALID_DATA_TYPE;
        }
        if (inData.uniqueFormat().numChannels() != 1)
        {
            LOG_ERROR("Only support 1 channel");
            return ErrorCode::INVALID_DATA_FORMAT;
        }
        getThreshVal_Triangle(inData, thresh, m_histogram, stream);
    }

    typedef void (*threshold_t)(const ImageBatchVarShapeDataStridedCuda &input,
                                const ImageBatchVarShapeDataStridedCuda &output, const TensorDataStridedCuda &threshold,
                                const TensorDataStridedCuda &maxval, NVCVThresholdType type, DataType data_type,
                                cudaStream_t stream);
    static const threshold_t funcs[7] = {thresholdDispatch<uchar>, 0, thresholdDispatch<ushort>,
                                         thresholdDispatch<short>, 0, thresholdDispatch<float>,
                                         thresholdDispatch<double>};

    threshold_t       func    = funcs[in_data_type];
    NVCVThresholdType th_type = NVCVThresholdType(m_type);
    if (isPlanar)
    {
        typedef void (*threshold_planar_t)(
            const ImageBatchVarShapeDataStridedCuda &input, const ImageBatchVarShapeDataStridedCuda &output,
            const TensorDataStridedCuda &threshold, const TensorDataStridedCuda &maxval, int channels,
            NVCVThresholdType type, DataType data_type, cudaStream_t stream);
        static const threshold_planar_t planarFuncs[7]
            = {thresholdDispatchPlanar<uchar>, 0, thresholdDispatchPlanar<ushort>,
               thresholdDispatchPlanar<short>, 0, thresholdDispatchPlanar<float>,
               thresholdDispatchPlanar<double>};
        threshold_planar_t planarFunc = planarFuncs[in_data_type];
        planarFunc(inData, outData, thresh, maxval, channels, th_type, in_data_type, stream);
    }
    else
    {
        func(inData, outData, thresh, maxval, th_type, in_data_type, stream);
    }

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

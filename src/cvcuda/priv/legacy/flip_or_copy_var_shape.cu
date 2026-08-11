/* Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

// Var-shape flip is a pure pixel remap and memory/latency-bound at one element per thread (mirrors the
// tensor path: ncu showed the single-channel kernel latency-bound, DRAM well below peak). Each thread
// now processes NIX columns strided by the total x-thread count, issuing NIX independent loads/stores
// together (memory-level parallelism) while consecutive threads keep touching consecutive columns
// (coalesced); the per-thread var-shape height/width/flip-code lookups are also resolved once and
// amortized over NIX elements. NIX>1 needs the grid sized to divUp(maxWidth, NIX) x-threads. The remap
// is unchanged, so every output is bit-exact with the one-element-per-thread version.
constexpr int kFlipVarNIX       = 4;
constexpr int kFlipVarScalarNIX = 1;
constexpr int kFlipVarU8Vec     = 4;

template<int NIX, typename T>
__global__ void flip_kernel(const cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst,
                            const cuda::Tensor1DWrap<int> flipCode)
{
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);
    if (y >= out_height)
        return;

    const int x0        = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride    = gridDim.x * blockDim.x;
    const int flip_code = flipCode[batch_idx];

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int x = x0 + i * stride;
        if (x >= out_width)
            continue;

        if (flip_code == 1) // flip_code = 1, horizontal flip
        {
            *dst.ptr(batch_idx, y, x) = *src.ptr(batch_idx, y, (out_width - 1 - x));
        }
        else if (flip_code == 0) // flip_code = 0, vertical flip
        {
            *dst.ptr(batch_idx, y, x) = *src.ptr(batch_idx, (out_height - 1 - y), x);
        }
        else if (flip_code == -1) // flip_code = -1, horizontal and vertical flip
        {
            *dst.ptr(batch_idx, y, x) = *src.ptr(batch_idx, (out_height - 1 - y), (out_width - 1 - x));
        }
        else // just copy
        {
            *dst.ptr(batch_idx, y, x) = *src.ptr(batch_idx, y, x);
        }
    }
}

template<typename T, int NIX>
void flipImpl(const ImageBatchVarShapeDataStridedCuda &input, const ImageBatchVarShapeDataStridedCuda &output,
              const TensorDataStridedCuda &flipCode, cudaStream_t stream)
{
    dim3 blockSize(32, 8, 1);
    dim3 gridSize(divUp(divUp(input.maxSize().w, NIX), static_cast<int>(blockSize.x)),
                  divUp(input.maxSize().h, blockSize.y), output.numImages());

    cuda::ImageBatchVarShapeWrap<T> src(input);
    cuda::ImageBatchVarShapeWrap<T> dst(output);
    cuda::Tensor1DWrap<int>         flip_code(flipCode);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif // CUDA_DEBUG_LOG

    flip_kernel<NIX, T><<<gridSize, blockSize, 0, stream>>>(src, dst, flip_code);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif // CUDA_DEBUG_LOG
}

template<typename T>
void flip(const ImageBatchVarShapeDataStridedCuda &input, const ImageBatchVarShapeDataStridedCuda &output,
          const TensorDataStridedCuda &flipCode, cudaStream_t stream)
{
    flipImpl<T, kFlipVarNIX>(input, output, flipCode, stream);
}

template<typename T>
void flipScalar(const ImageBatchVarShapeDataStridedCuda &input, const ImageBatchVarShapeDataStridedCuda &output,
                const TensorDataStridedCuda &flipCode, cudaStream_t stream)
{
    flipImpl<T, kFlipVarScalarNIX>(input, output, flipCode, stream);
}

__device__ __forceinline__ uchar4 reverseU8Lanes(uchar4 value)
{
    return make_uchar4(value.w, value.z, value.y, value.x);
}

__device__ __forceinline__ void flipU8Group(const uchar *src_row, uchar *dst_row, int width, int dst_x, bool reverse_x)
{
    const int    src_x   = reverse_x ? width - kFlipVarU8Vec - dst_x : dst_x;
    uchar       *dst_ptr = dst_row + dst_x;
    const uchar *src_ptr = src_x >= 0 ? src_row + src_x : nullptr;
    const bool   full    = dst_x + kFlipVarU8Vec <= width;
    const bool   aligned = full && src_ptr != nullptr
                      && ((reinterpret_cast<std::uintptr_t>(src_ptr) | reinterpret_cast<std::uintptr_t>(dst_ptr))
                          & (alignof(uchar4) - 1))
                             == 0;

    if (aligned)
    {
        const uchar4 value                   = *reinterpret_cast<const uchar4 *>(src_ptr);
        *reinterpret_cast<uchar4 *>(dst_ptr) = reverse_x ? reverseU8Lanes(value) : value;
        return;
    }

#pragma unroll
    for (int k = 0; k < kFlipVarU8Vec; ++k)
    {
        const int x = dst_x + k;
        if (x >= width)
            break;
        dst_row[x] = src_row[reverse_x ? width - 1 - x : x];
    }
}

// Single-channel U8 VarShape uses four adjacent columns per thread. Consecutive threads therefore
// move a full 128-byte warp segment instead of only 32 bytes, while NIX retains the independent
// requests that hide the ImageBatchVarShape pointer-lookup latency. Images whose row pointers or
// mirrored block boundaries are not 4-byte aligned stay on the bit-exact scalar fallback below.
template<int NIX>
__global__ void flip_u8_wide_kernel(const cuda::ImageBatchVarShapeWrap<uchar> src,
                                    cuda::ImageBatchVarShapeWrap<uchar> dst, const cuda::Tensor1DWrap<int> flipCode)
{
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);
    if (y >= out_height)
        return;

    const int  group0     = blockIdx.x * blockDim.x + threadIdx.x;
    const int  group_step = gridDim.x * blockDim.x;
    const int  flip_code  = flipCode[batch_idx];
    const bool reverse_x  = flip_code == 1 || flip_code == -1;
    const int  src_y      = (flip_code == 0 || flip_code == -1) ? out_height - 1 - y : y;

    const uchar *src_row = src.ptr(batch_idx, src_y, 0);
    uchar       *dst_row = dst.ptr(batch_idx, y, 0);

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int dst_x = (group0 + i * group_step) * kFlipVarU8Vec;
        if (dst_x >= out_width)
            continue;

        flipU8Group(src_row, dst_row, out_width, dst_x, reverse_x);
    }
}

static void flipU8Wide(const ImageBatchVarShapeDataStridedCuda &input, const ImageBatchVarShapeDataStridedCuda &output,
                       const TensorDataStridedCuda &flipCode, cudaStream_t stream)
{
    dim3      blockSize(32, 8, 1);
    const int groups = divUp(input.maxSize().w, kFlipVarU8Vec);
    dim3      gridSize(divUp(divUp(groups, kFlipVarNIX), static_cast<int>(blockSize.x)),
                       divUp(input.maxSize().h, blockSize.y), output.numImages());

    cuda::ImageBatchVarShapeWrap<uchar> src(input);
    cuda::ImageBatchVarShapeWrap<uchar> dst(output);
    cuda::Tensor1DWrap<int>             flip_code(flipCode);

    flip_u8_wide_kernel<kFlipVarNIX><<<gridSize, blockSize, 0, stream>>>(src, dst, flip_code);
    checkKernelErrors();
}

template<int NIX>
__global__ void flip_planar_u8_wide_kernel(const cuda::ImageBatchVarShapeWrap<uchar> src,
                                           cuda::ImageBatchVarShapeWrap<uchar>       dst,
                                           const cuda::Tensor1DWrap<int> flipCode, int channels)
{
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);
    if (y >= out_height)
        return;

    const int  group0     = blockIdx.x * blockDim.x + threadIdx.x;
    const int  group_step = gridDim.x * blockDim.x;
    const int  flip_code  = flipCode[batch_idx];
    const bool reverse_x  = flip_code == 1 || flip_code == -1;
    const int  src_y      = (flip_code == 0 || flip_code == -1) ? out_height - 1 - y : y;

    for (int plane = 0; plane < channels; ++plane)
    {
        const uchar *src_row = src.ptr(batch_idx, plane, src_y, 0);
        uchar       *dst_row = dst.ptr(batch_idx, plane, y, 0);

#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            const int dst_x = (group0 + i * group_step) * kFlipVarU8Vec;
            if (dst_x < out_width)
                flipU8Group(src_row, dst_row, out_width, dst_x, reverse_x);
        }
    }
}

static void flip_planar_u8_wide(const ImageBatchVarShapeDataStridedCuda &input,
                                const ImageBatchVarShapeDataStridedCuda &output, const TensorDataStridedCuda &flipCode,
                                const int channels, cudaStream_t stream)
{
    dim3      blockSize(32, 8, 1);
    const int groups = divUp(input.maxSize().w, kFlipVarU8Vec);
    dim3      gridSize(divUp(divUp(groups, kFlipVarNIX), static_cast<int>(blockSize.x)),
                       divUp(input.maxSize().h, blockSize.y), output.numImages());

    cuda::ImageBatchVarShapeWrap<uchar> src(input);
    cuda::ImageBatchVarShapeWrap<uchar> dst(output);
    cuda::Tensor1DWrap<int>             flip_code(flipCode);

    flip_planar_u8_wide_kernel<kFlipVarNIX><<<gridSize, blockSize, 0, stream>>>(src, dst, flip_code, channels);
    checkKernelErrors();
}

// Planar (NCHW/CHW) flip. Flip remaps each output pixel to a source pixel independently of the
// channel, so the (src_x, src_y) mapping is computed once per output pixel and reused across every
// channel plane (grid.z runs over images, the kernel loops the planes). This avoids the per-plane
// redundant coordinate math of one thread per (image, plane) and keeps each plane's result
// bit-exact with the interleaved single-channel flip.
template<int NIX, typename T>
__global__ void flip_planar_kernel(const cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst,
                                   const cuda::Tensor1DWrap<int> flipCode, int channels)
{
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);
    if (y >= out_height)
        return;

    const int x0        = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride    = gridDim.x * blockDim.x;
    const int flip_code = flipCode[batch_idx];

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int x = x0 + i * stride;
        if (x >= out_width)
            continue;

        int src_x, src_y;
        if (flip_code == 1) // horizontal flip
        {
            src_x = out_width - 1 - x;
            src_y = y;
        }
        else if (flip_code == 0) // vertical flip
        {
            src_x = x;
            src_y = out_height - 1 - y;
        }
        else if (flip_code == -1) // horizontal and vertical flip
        {
            src_x = out_width - 1 - x;
            src_y = out_height - 1 - y;
        }
        else // just copy
        {
            src_x = x;
            src_y = y;
        }

        for (int plane = 0; plane < channels; ++plane)
        {
            *dst.ptr(batch_idx, plane, y, x) = *src.ptr(batch_idx, plane, src_y, src_x);
        }
    }
}

template<typename T>
void flip_planar(const ImageBatchVarShapeDataStridedCuda &input, const ImageBatchVarShapeDataStridedCuda &output,
                 const TensorDataStridedCuda &flipCode, const int channels, cudaStream_t stream)
{
    dim3 blockSize(32, 8, 1);
    dim3 gridSize(divUp(divUp(input.maxSize().w, kFlipVarNIX), static_cast<int>(blockSize.x)),
                  divUp(input.maxSize().h, blockSize.y), output.numImages());

    cuda::ImageBatchVarShapeWrap<T> src(input);
    cuda::ImageBatchVarShapeWrap<T> dst(output);
    cuda::Tensor1DWrap<int>         flip_code(flipCode);

    flip_planar_kernel<kFlipVarNIX, T><<<gridSize, blockSize, 0, stream>>>(src, dst, flip_code, channels);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif // CUDA_DEBUG_LOG
}

ErrorCode FlipOrCopyVarShape::infer(const ImageBatchVarShapeDataStridedCuda &input,
                                    const ImageBatchVarShapeDataStridedCuda &output,
                                    const TensorDataStridedCuda &flipCode, cudaStream_t stream)
{
    if (!input.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!output.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat inputFormat  = helpers::GetLegacyDataFormat(input);
    DataFormat outputFormat = helpers::GetLegacyDataFormat(output);

    if (inputFormat != outputFormat)
    {
        LOG_ERROR("Invalid DataFormat between input (" << inputFormat << ") and output (" << outputFormat << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = inputFormat;
    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    DataType dataType    = helpers::GetLegacyDataType(input.uniqueFormat());
    DataType outDataType = helpers::GetLegacyDataType(output.uniqueFormat());

    if (dataType != outDataType)
    {
        LOG_ERROR("DataType of input and output must be equal, but got " << dataType << " and " << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(dataType == kCV_8U || dataType == kCV_16U || dataType == kCV_16S || dataType == kCV_32S
          || dataType == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << dataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = input.uniqueFormat().numChannels();
    // Two-channel input is unsupported by both the interleaved dispatch table and planar formats.
    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // The planar path launches grid.z over images and loops the channel planes inside the kernel, so
    // numImages must fit CUDA's 65535 grid-z limit. Compute in 64-bit to avoid overflow.
    if (isPlanar && static_cast<int64_t>(output.numImages()) > 65535)
    {
        LOG_ERROR("Planar flip requires numImages <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // using flip_t = void(const ImageBatchVarShapeDataStridedCuda & input,
    //                     const ImageBatchVarShapeDataStridedCuda & output,
    //                     const TensorDataStridedCuda & flipCode,
    //                     cudaStream_t stream);
    if (isPlanar)
    {
        // Planar dispatch indexes by dtype only: each channel is flipped as a separate single-channel
        // plane, so one scalar specialization per dtype covers any channel count.
        typedef void (*planar_flip_t)(const ImageBatchVarShapeDataStridedCuda &input,
                                      const ImageBatchVarShapeDataStridedCuda &output,
                                      const TensorDataStridedCuda &flipCode, const int channels, cudaStream_t stream);

        static const planar_flip_t planar_funcs[6] = {
            flip_planar_u8_wide, 0 /*schar*/,      flip_planar<ushort>,
            flip_planar<short>,  flip_planar<int>, flip_planar<float>,
        };

        const planar_flip_t planar_func = planar_funcs[dataType];
        NVCV_ASSERT(planar_func != 0);
        planar_func(input, output, flipCode, channels, stream);
        return ErrorCode::SUCCESS;
    }

    typedef void (*flip_t)(const ImageBatchVarShapeDataStridedCuda &input,
                           const ImageBatchVarShapeDataStridedCuda &output, const TensorDataStridedCuda &flipCode,
                           cudaStream_t stream);

    static const flip_t funcs[6][4] = {
        {  flipU8Wide, 0,       flip<uchar3>,  flip<uchar4>},
        {           0, 0,                  0,             0},
        {flip<ushort>, 0,      flip<ushort3>, flip<ushort4>},
        { flip<short>, 0,       flip<short3>,  flip<short4>},
        {   flip<int>, 0,         flip<int3>,    flip<int4>},
        { flip<float>, 0, flipScalar<float3>,  flip<float4>}
    };

    funcs[dataType][channels - 1](input, output, flipCode, stream);

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

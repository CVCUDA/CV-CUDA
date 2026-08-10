/* Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../Assert.h"
#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <nvcv/util/CheckError.hpp>

#include <cstdint>
#include <type_traits>

#define BLOCK 32

namespace nvcv::legacy::cuda_op {

namespace {

static bool IsPlanar(DataFormat format)
{
    return format == kNCHW || format == kCHW;
}

static bool SameLayoutFamily(DataFormat lhs, DataFormat rhs)
{
    return IsPlanar(lhs) == IsPlanar(rhs);
}

static DataFormat ImageLayout(nvcv::ImageFormat fmt, int numImages)
{
    return helpers::GetLegacyDataFormat(fmt.numChannels(), fmt.numPlanes(), numImages);
}

} // namespace

template<typename T>
__global__ void channel_reorder_kernel(const cuda::ImageBatchVarShapeWrapNHWC<const T> src,
                                       cuda::ImageBatchVarShapeWrapNHWC<T> dst, const cuda::Tensor2DWrap<int> orders)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);
    if (dst_x >= out_width || dst_y >= out_height)
        return;

    const int *chOrder = orders.ptr(batch_idx);

    for (int ch = 0; ch < dst.numChannels(); ch++)
    {
        int src_ch = chOrder[ch];
        if (src_ch < 0)
        {
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = 0;
        }
        else
        {
            NVCV_CUDA_ASSERT(0 <= src_ch && src_ch < src.numChannels(),
                             "Index to source channel %d is out of bounds (%d)", src_ch, src.numChannels());
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = *src.ptr(batch_idx, dst_y, dst_x, src_ch);
        }
    }
}

template<typename T>
void reorder(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
             const TensorDataStridedCuda &orderData, int numSrcChannels, int numDstChannels, cudaStream_t stream)
{
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(inData.maxSize().w, blockSize.x), divUp(inData.maxSize().h, blockSize.y), batch_size);

    cuda::ImageBatchVarShapeWrapNHWC<const T> src_ptr(inData, numSrcChannels);
    cuda::ImageBatchVarShapeWrapNHWC<T>       dst_ptr(outData, numDstChannels);
    cuda::Tensor2DWrap<int>                   order_ptr(orderData);

    channel_reorder_kernel<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, order_ptr);

    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename T>
__global__ void channel_reorder_planar_kernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                              cuda::ImageBatchVarShapeWrap<T> dst, const cuda::Tensor2DWrap<int> orders,
                                              int numSrcChannels, int numDstChannels)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);
    if (dst_x >= out_width || dst_y >= out_height)
        return;

    const int *chOrder = orders.ptr(batch_idx);

    for (int ch = 0; ch < numDstChannels; ch++)
    {
        int src_ch = chOrder[ch];
        if (src_ch < 0)
        {
            *dst.ptr(batch_idx, ch, dst_y, dst_x) = 0;
        }
        else
        {
            NVCV_CUDA_ASSERT(0 <= src_ch && src_ch < numSrcChannels, "Index to source channel %d is out of bounds (%d)",
                             src_ch, numSrcChannels);
            *dst.ptr(batch_idx, ch, dst_y, dst_x) = *src.ptr(batch_idx, src_ch, dst_y, dst_x);
        }
    }
}

template<typename PackT>
constexpr uintptr_t kPackAlignmentMask = (sizeof(PackT) == sizeof(uint3) ? sizeof(uint) : sizeof(PackT)) - 1;

template<bool IsPlanar, int NumChannels, class SrcWrapper, class DstWrapper>
__global__ void channel_reorder_u8_kernel(const SrcWrapper src, DstWrapper dst, const cuda::Tensor2DWrap<int> orders)
{
    static constexpr int kNix = 4;

    const int dst_x0     = (blockIdx.x * blockDim.x + threadIdx.x) * kNix;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    const int out_height = dst.height(batch_idx);
    const int out_width  = dst.width(batch_idx);
    if (dst_x0 >= out_width || dst_y >= out_height)
        return;

    int chOrder[NumChannels];
#pragma unroll
    for (int ch = 0; ch < NumChannels; ++ch)
    {
        chOrder[ch] = orders.ptr(batch_idx)[ch];
        NVCV_CUDA_ASSERT(chOrder[ch] < NumChannels, "Index to source channel %d is out of bounds (%d)", chOrder[ch],
                         NumChannels);
    }

    if constexpr (IsPlanar)
    {
        using PackT = uint;

#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch)
        {
            const int src_ch = chOrder[ch];
            uchar    *dp     = dst.ptr(batch_idx, ch, dst_y, dst_x0);

            if (dst_x0 + kNix <= out_width && (reinterpret_cast<uintptr_t>(dp) & kPackAlignmentMask<PackT>) == 0)
            {
                PackT out{};
                if (src_ch >= 0)
                {
                    const uchar *sp = src.ptr(batch_idx, src_ch, dst_y, dst_x0);
                    if ((reinterpret_cast<uintptr_t>(sp) & kPackAlignmentMask<PackT>) == 0)
                    {
                        out                            = *reinterpret_cast<const PackT *>(sp);
                        *reinterpret_cast<PackT *>(dp) = out;
                        continue;
                    }
                }
                else
                {
                    *reinterpret_cast<PackT *>(dp) = out;
                    continue;
                }
            }

#pragma unroll
            for (int i = 0; i < kNix; ++i)
            {
                if (dst_x0 + i < out_width)
                {
                    dp[i] = src_ch < 0 ? 0 : *src.ptr(batch_idx, src_ch, dst_y, dst_x0 + i);
                }
            }
        }
    }
    else
    {
        using PackT = std::conditional_t<NumChannels == 3, uint3, uint4>;

        const uchar *sp = src.ptr(batch_idx, dst_y, dst_x0);
        uchar       *dp = dst.ptr(batch_idx, dst_y, dst_x0);
        if (dst_x0 + kNix <= out_width && (reinterpret_cast<uintptr_t>(sp) & kPackAlignmentMask<PackT>) == 0
            && (reinterpret_cast<uintptr_t>(dp) & kPackAlignmentMask<PackT>) == 0)
        {
            PackT in = *reinterpret_cast<const PackT *>(sp);
            PackT out;

            const uchar *inBytes  = reinterpret_cast<const uchar *>(&in);
            uchar       *outBytes = reinterpret_cast<uchar *>(&out);
#pragma unroll
            for (int i = 0; i < kNix; ++i)
            {
#pragma unroll
                for (int ch = 0; ch < NumChannels; ++ch)
                {
                    const int src_ch               = chOrder[ch];
                    outBytes[i * NumChannels + ch] = src_ch < 0 ? 0 : inBytes[i * NumChannels + src_ch];
                }
            }
            *reinterpret_cast<PackT *>(dp) = out;
        }
        else
        {
#pragma unroll
            for (int i = 0; i < kNix; ++i)
            {
                if (dst_x0 + i < out_width)
                {
#pragma unroll
                    for (int ch = 0; ch < NumChannels; ++ch)
                    {
                        const int src_ch         = chOrder[ch];
                        dp[i * NumChannels + ch] = src_ch < 0 ? 0 : *src.ptr(batch_idx, dst_y, dst_x0 + i, src_ch);
                    }
                }
            }
        }
    }
}

template<bool IsPlanar, int NumChannels>
void reorder_u8(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                const TensorDataStridedCuda &orderData, cudaStream_t stream)
{
    static constexpr int kNix = 4;

    int                     batch_size = inData.numImages();
    const auto              maxSize    = IsPlanar ? outData.maxSize() : inData.maxSize();
    dim3                    blockSize(BLOCK, BLOCK / 4, 1);
    dim3                    gridSize(divUp(maxSize.w, blockSize.x * kNix), divUp(maxSize.h, blockSize.y), batch_size);
    cuda::Tensor2DWrap<int> order_ptr(orderData);

    if constexpr (IsPlanar)
    {
        cuda::ImageBatchVarShapeWrap<const uchar> src_ptr(inData);
        cuda::ImageBatchVarShapeWrap<uchar>       dst_ptr(outData);
        channel_reorder_u8_kernel<true, NumChannels><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, order_ptr);
    }
    else
    {
        cuda::ImageBatchVarShapeWrapNHWC<const uchar> src_ptr(inData, NumChannels);
        cuda::ImageBatchVarShapeWrapNHWC<uchar>       dst_ptr(outData, NumChannels);
        channel_reorder_u8_kernel<false, NumChannels><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, order_ptr);
    }

    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename T>
void reorder_planar(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const TensorDataStridedCuda &orderData, int numSrcChannels, int numDstChannels, cudaStream_t stream)
{
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(outData.maxSize().w, blockSize.x), divUp(outData.maxSize().h, blockSize.y), batch_size);

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(inData);
    cuda::ImageBatchVarShapeWrap<T>       dst_ptr(outData);
    cuda::Tensor2DWrap<int>               order_ptr(orderData);

    channel_reorder_planar_kernel<T>
        <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, order_ptr, numSrcChannels, numDstChannels);

    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
}

ErrorCode ChannelReorderVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                        const ImageBatchVarShapeDataStridedCuda &outData,
                                        const TensorDataStridedCuda &orderData, cudaStream_t stream)
{
    if (inData.numImages() != outData.numImages())
    {
        LOG_ERROR("Input and output batches must have the same number of images");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (inData.numImages() == 0)
    {
        // nothing to do, move above the calling of GetLegacyDataType to avoid error: "All planes must have the same data type"
        return ErrorCode::SUCCESS;
    }

    DataType   data_type;
    DataFormat input_format;
    DataFormat output_format;
    int        channels;
    int        outChannels;
    {
        nvcv::ImageFormat fmt(inData.hostFormatList()[0]);
        nvcv::ImageFormat outFmt(outData.hostFormatList()[0]);
        data_type     = helpers::GetLegacyDataType(fmt);
        input_format  = ImageLayout(fmt, inData.numImages());
        output_format = ImageLayout(outFmt, outData.numImages());
        channels      = fmt.numChannels();
        outChannels   = outFmt.numChannels();
    }

    if (!(input_format == kNHWC || input_format == kHWC || input_format == kNCHW || input_format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << input_format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(output_format == kNHWC || output_format == kHWC || output_format == kNCHW || output_format == kCHW))
    {
        LOG_ERROR("Invalid output DataFormat " << output_format
                                               << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!SameLayoutFamily(input_format, output_format))
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = IsPlanar(input_format);

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (orderData.rank() != 2)
    {
        LOG_ERROR("order tensor must have 2 dimensions, not " << orderData.rank());
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (helpers::GetLegacyDataType(orderData.dtype()) != kCV_32S)
    {
        LOG_ERROR("Invalid Order tensor DataType " << helpers::GetLegacyDataType(orderData.dtype()));
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (orderData.layout()[0] != nvcv::LABEL_BATCH)
    {
        LOG_ERROR("Label of the first dimension of order tensor must be " << nvcv::LABEL_BATCH);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (orderData.shape(0) != inData.numImages())
    {
        LOG_ERROR("Order tensor must have same number of samples as number of input images");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (orderData.shape(1) > 4)
    {
        LOG_ERROR("Second dimension of order tensor must be at most 4, not " << orderData.shape(1));
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    for (int i = 0; i < inData.numImages(); ++i)
    {
        nvcv::ImageFormat inFmt(inData.hostFormatList()[i]);
        nvcv::ImageFormat outFmt(outData.hostFormatList()[i]);

        DataFormat imageInputFormat  = ImageLayout(inFmt, inData.numImages());
        DataFormat imageOutputFormat = ImageLayout(outFmt, outData.numImages());

        if (imageInputFormat != input_format)
        {
            LOG_ERROR("Input image #" << i << " has DataFormat " << imageInputFormat << ", expected " << input_format);
            return ErrorCode::INVALID_DATA_FORMAT;
        }

        if (!SameLayoutFamily(imageInputFormat, imageOutputFormat))
        {
            LOG_ERROR("Invalid DataFormat between input image #"
                      << i << " (" << imageInputFormat << ") and output image #" << i << " (" << imageOutputFormat
                      << ")");
            return ErrorCode::INVALID_DATA_FORMAT;
        }

        // Legacy code has this check, let's stick to it.
        if (inFmt.numChannels() != channels)
        {
            LOG_ERROR("Input channel " << inFmt.numChannels() << " differs from " << channels);
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        if (inFmt.numChannels() > 4 || (isPlanar && inFmt.numChannels() == 2))
        {
            LOG_ERROR("Invalid input channel number " << inFmt.numChannels());
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        if (outFmt.numChannels() != outChannels)
        {
            LOG_ERROR("Output channel " << outFmt.numChannels() << " differs from " << outChannels);
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        if (outFmt.numChannels() > 4 || (isPlanar && outFmt.numChannels() == 2))
        {
            LOG_ERROR("Invalid output channel number " << outFmt.numChannels());
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        if (outFmt.numChannels() > orderData.shape(1))
        {
            LOG_ERROR("Invalid output channel number " << outFmt.numChannels() << ", must be at most "
                                                       << orderData.shape(1));
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        // TODO: we can't check if order index is < channels, like legacy does. It'd incur in
        // perf penalty as the data is currently on device. Instead, we added an assertion in
        // the cuda kernel, but it's only triggered in debug builds, and leads to an unrecoverable
        // error (cuda kernel errors are sticky), process must be restarted.

        if (helpers::GetLegacyDataType(inFmt) != data_type)
        {
            LOG_ERROR("Format of input images must all have the same data type");
            return ErrorCode::INVALID_DATA_TYPE;
        }

        if (helpers::GetLegacyDataType(outFmt) != data_type)
        {
            LOG_ERROR("Format of output images must all have the same data type");
            return ErrorCode::INVALID_DATA_TYPE;
        }
    }

    if (isPlanar && static_cast<int64_t>(outData.numImages()) > 65535)
    {
        LOG_ERROR("Planar ChannelReorder requires numImages <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (isPlanar)
    {
        if (data_type == kCV_8U && channels == outChannels && channels == 3)
        {
            reorder_u8<true, 3>(inData, outData, orderData, stream);
            return ErrorCode::SUCCESS;
        }
        if (data_type == kCV_8U && channels == outChannels && channels == 4)
        {
            reorder_u8<true, 4>(inData, outData, orderData, stream);
            return ErrorCode::SUCCESS;
        }

        typedef void (*planar_func_t)(
            const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
            const TensorDataStridedCuda &orderData, int numSrcChannels, int numDstChannels, cudaStream_t stream);

        static const planar_func_t planar_funcs[6]
            = {reorder_planar<uchar>, 0, reorder_planar<ushort>, reorder_planar<short>, reorder_planar<int>,
               reorder_planar<float>};

        const planar_func_t planar_func = planar_funcs[data_type];
        NVCV_ASSERT(planar_func != 0);

        planar_func(inData, outData, orderData, channels, outChannels, stream);
        return ErrorCode::SUCCESS;
    }

    if (data_type == kCV_8U && channels == outChannels && channels == 3)
    {
        reorder_u8<false, 3>(inData, outData, orderData, stream);
        return ErrorCode::SUCCESS;
    }
    if (data_type == kCV_8U && channels == outChannels && channels == 4)
    {
        reorder_u8<false, 4>(inData, outData, orderData, stream);
        return ErrorCode::SUCCESS;
    }

    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &orderData,
                           int numSrcChannels, int numDstChannels, cudaStream_t stream);

    static const func_t funcs[6] = {reorder<uchar>, 0, reorder<ushort>, reorder<short>, reorder<int>, reorder<float>};

    const func_t func = funcs[data_type];
    NVCV_ASSERT(func != 0);

    func(inData, outData, orderData, channels, outChannels, stream);
    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

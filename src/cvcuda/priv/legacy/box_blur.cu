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

#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/priv/Types.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/TensorData.hpp>

#include <algorithm>
#include <cstdio>
#include <type_traits>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;
using namespace nvcv::cuda::osd;
using namespace cvcuda::priv;

namespace nvcv::legacy::cuda_op {

template<typename _T>
static __forceinline__ __device__ _T limit(_T value, _T low, _T high)
{
    return value < low ? low : (value > high ? high : value);
}

static bool IsPlanar(DataFormat format)
{
    return format == kNCHW || format == kCHW;
}

template<class SrcWrapper, class DstWrapper>
static __global__ void render_p2p_kernel(SrcWrapper src, DstWrapper dst, int batch, int height, int width, int channel)
{
    int       ix        = blockDim.x * blockIdx.x + threadIdx.x;
    int       iy        = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (ix >= width || iy >= height || batch_idx >= batch)
        return;

    if (channel == 3)
    {
        *(uchar3 *)(dst.ptr(batch_idx, iy, ix, 0)) = *(uchar3 *)(src.ptr(batch_idx, iy, ix, 0));
    }
    else
    {
        *(uchar4 *)(dst.ptr(batch_idx, iy, ix, 0)) = *(uchar4 *)(src.ptr(batch_idx, iy, ix, 0));
    }
}

template<class SrcWrapper, class DstWrapper>
static __global__ void render_p2p_planar_kernel(SrcWrapper src, DstWrapper dst, int batch, int height, int width,
                                                int channels)
{
    int       ix        = blockDim.x * blockIdx.x + threadIdx.x;
    int       iy        = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (ix >= width || iy >= height || batch_idx >= batch)
        return;

    for (int c = 0; c < channels; ++c)
    {
        *dst.ptr(batch_idx, c, iy, ix) = *src.ptr(batch_idx, c, iy, ix);
    }
}

template<class SrcWrapper, class DstWrapper>
static __global__ void render_blur_rgb_kernel(SrcWrapper src, DstWrapper dst, const BoxBlurCommand *commands,
                                              int num_command, int image_batch, int image_width, int image_height)
{
    if (blockIdx.x >= num_command)
        return;
    const BoxBlurCommand &box = commands[blockIdx.x];
    if (box.batch_index >= image_batch)
        return;

    using ChannelType = std::remove_const_t<typename SrcWrapper::ValueType>;
    using PixelType   = nvcv::cuda::MakeType<ChannelType, 3>;

    __shared__ PixelType crop[32][32];
    int                  ix = threadIdx.x;
    int                  iy = threadIdx.y;

    int boxwidth  = box.bounding_right - box.bounding_left;
    int boxheight = box.bounding_bottom - box.bounding_top;
    int sx        = limit((int)(ix / 32.0f * (float)boxwidth + 0.5f + box.bounding_left), 0, image_width);
    int sy        = limit((int)(iy / 32.0f * (float)boxheight + 0.5f + box.bounding_top), 0, image_height);

    crop[iy][ix] = *reinterpret_cast<const PixelType *>(src.ptr(box.batch_index, sy, sx, 0));
    __syncthreads();

    int3 color = make_int3(0, 0, 0);
    int  n     = 0;
    for (int i = -box.kernel_size / 2; i <= box.kernel_size / 2; ++i)
    {
        for (int j = -box.kernel_size / 2; j <= box.kernel_size / 2; ++j)
        {
            int u = i + iy;
            int v = j + ix;
            if (u >= 0 && u < 32 && v >= 0 && v < 32)
            {
                auto &c = crop[u][v];
                color.x += c.x;
                color.y += c.y;
                color.z += c.z;
                n++;
            }
        }
    }
    __syncthreads();
    crop[iy][ix] = PixelType{static_cast<ChannelType>(color.x / n), static_cast<ChannelType>(color.y / n),
                             static_cast<ChannelType>(color.z / n)};
    __syncthreads();

    int gap_width  = (boxwidth + 31) / 32;
    int gap_height = (boxheight + 31) / 32;
    for (int i = 0; i < gap_height; ++i)
    {
        for (int j = 0; j < gap_width; ++j)
        {
            int fx = ix * gap_width + j + box.bounding_left;
            int fy = iy * gap_height + i + box.bounding_top;
            if (fx >= 0 && fx < image_width && fy >= 0 && fy < image_height)
            {
                int sx = (ix * gap_width + j) / (float)boxwidth * 32;
                int sy = (iy * gap_height + i) / (float)boxheight * 32;
                if (sx < 32 && sy < 32)
                {
                    auto &pix                                                           = crop[sy][sx];
                    *reinterpret_cast<PixelType *>(dst.ptr(box.batch_index, fy, fx, 0)) = pix;
                }
            }
        }
    }
}

template<class SrcWrapper, class DstWrapper>
static __global__ void render_blur_rgba_kernel(SrcWrapper src, DstWrapper dst, const BoxBlurCommand *commands,
                                               int num_command, int image_batch, int image_width, int image_height)
{
    if (blockIdx.x >= num_command)
        return;
    const BoxBlurCommand &box = commands[blockIdx.x];
    if (box.batch_index >= image_batch)
        return;

    using ChannelType = std::remove_const_t<typename SrcWrapper::ValueType>;
    using Pixel3Type  = nvcv::cuda::MakeType<ChannelType, 3>;
    using Pixel4Type  = nvcv::cuda::MakeType<ChannelType, 4>;

    __shared__ Pixel3Type crop[32][32];
    int                   ix = threadIdx.x;
    int                   iy = threadIdx.y;

    int boxwidth  = box.bounding_right - box.bounding_left;
    int boxheight = box.bounding_bottom - box.bounding_top;
    int sx        = limit((int)(ix / 32.0f * (float)boxwidth + 0.5f + box.bounding_left), 0, image_width);
    int sy        = limit((int)(iy / 32.0f * (float)boxheight + 0.5f + box.bounding_top), 0, image_height);

    crop[iy][ix] = *reinterpret_cast<const Pixel3Type *>(src.ptr(box.batch_index, sy, sx, 0));
    __syncthreads();

    int3 color = make_int3(0, 0, 0);
    int  n     = 0;
    for (int i = -box.kernel_size / 2; i <= box.kernel_size / 2; ++i)
    {
        for (int j = -box.kernel_size / 2; j <= box.kernel_size / 2; ++j)
        {
            int u = i + iy;
            int v = j + ix;
            if (u >= 0 && u < 32 && v >= 0 && v < 32)
            {
                auto &c = crop[u][v];
                color.x += c.x;
                color.y += c.y;
                color.z += c.z;
                n++;
            }
        }
    }
    __syncthreads();
    crop[iy][ix] = Pixel3Type{static_cast<ChannelType>(color.x / n), static_cast<ChannelType>(color.y / n),
                              static_cast<ChannelType>(color.z / n)};
    __syncthreads();

    int gap_width  = (boxwidth + 31) / 32;
    int gap_height = (boxheight + 31) / 32;
    for (int i = 0; i < gap_height; ++i)
    {
        for (int j = 0; j < gap_width; ++j)
        {
            int fx = ix * gap_width + j + box.bounding_left;
            int fy = iy * gap_height + i + box.bounding_top;
            if (fx >= 0 && fx < image_width && fy >= 0 && fy < image_height)
            {
                int sx = (ix * gap_width + j) / (float)boxwidth * 32;
                int sy = (iy * gap_height + i) / (float)boxheight * 32;
                if (sx < 32 && sy < 32)
                {
                    auto &pix = crop[sy][sx];
                    *reinterpret_cast<Pixel4Type *>(dst.ptr(box.batch_index, fy, fx, 0))
                        = Pixel4Type{pix.x, pix.y, pix.z, nvcv::cuda::TypeTraits<ChannelType>::max};
                }
            }
        }
    }
}

template<class SrcWrapper, class DstWrapper>
static __global__ void render_blur_planar_kernel(SrcWrapper src, DstWrapper dst, const BoxBlurCommand *commands,
                                                 int num_command, int image_batch, int image_width, int image_height,
                                                 int channels)
{
    if (blockIdx.x >= num_command)
        return;
    const BoxBlurCommand &box = commands[blockIdx.x];
    if (box.batch_index >= image_batch)
        return;

    using ChannelType = std::remove_const_t<typename SrcWrapper::ValueType>;

    int plane = blockIdx.y;
    if (plane >= channels)
        return;

    int ix = threadIdx.x;
    int iy = threadIdx.y;

    int boxwidth  = box.bounding_right - box.bounding_left;
    int boxheight = box.bounding_bottom - box.bounding_top;

    int gap_width  = (boxwidth + 31) / 32;
    int gap_height = (boxheight + 31) / 32;

    if (plane == 3)
    {
        for (int i = 0; i < gap_height; ++i)
        {
            for (int j = 0; j < gap_width; ++j)
            {
                int fx = ix * gap_width + j + box.bounding_left;
                int fy = iy * gap_height + i + box.bounding_top;
                if (fx >= 0 && fx < image_width && fy >= 0 && fy < image_height)
                {
                    int sx = (ix * gap_width + j) / (float)boxwidth * 32;
                    int sy = (iy * gap_height + i) / (float)boxheight * 32;
                    if (sx < 32 && sy < 32)
                    {
                        *dst.ptr(box.batch_index, plane, fy, fx) = nvcv::cuda::TypeTraits<ChannelType>::max;
                    }
                }
            }
        }
        return;
    }

    __shared__ ChannelType crop[32][32];

    int sx = limit((int)(ix / 32.0f * (float)boxwidth + 0.5f + box.bounding_left), 0, image_width);
    int sy = limit((int)(iy / 32.0f * (float)boxheight + 0.5f + box.bounding_top), 0, image_height);

    crop[iy][ix] = *src.ptr(box.batch_index, plane, sy, sx);
    __syncthreads();

    int color = 0;
    int n     = 0;
    for (int i = -box.kernel_size / 2; i <= box.kernel_size / 2; ++i)
    {
        for (int j = -box.kernel_size / 2; j <= box.kernel_size / 2; ++j)
        {
            int u = i + iy;
            int v = j + ix;
            if (u >= 0 && u < 32 && v >= 0 && v < 32)
            {
                color += crop[u][v];
                n++;
            }
        }
    }
    __syncthreads();
    crop[iy][ix] = static_cast<ChannelType>(color / n);
    __syncthreads();

    for (int i = 0; i < gap_height; ++i)
    {
        for (int j = 0; j < gap_width; ++j)
        {
            int fx = ix * gap_width + j + box.bounding_left;
            int fy = iy * gap_height + i + box.bounding_top;
            if (fx >= 0 && fx < image_width && fy >= 0 && fy < image_height)
            {
                int sx = (ix * gap_width + j) / (float)boxwidth * 32;
                int sy = (iy * gap_height + i) / (float)boxheight * 32;
                if (sx < 32 && sy < 32)
                {
                    *dst.ptr(box.batch_index, plane, fy, fx) = crop[sy][sx];
                }
            }
        }
    }
}

static void cuosd_apply(cuOSDContext_t context, cudaStream_t stream)
{
    if (!context->blur_commands.empty())
    {
        if (context->gpu_blur_commands == nullptr)
        {
            context->gpu_blur_commands = std::make_unique<Memory<BoxBlurCommand>>();
        }

        context->gpu_blur_commands->alloc_or_resize_to(context->blur_commands.size());

        for (int i = 0; i < (int)context->blur_commands.size(); ++i)
        {
            auto &cmd = context->blur_commands[i];
            memcpy((void *)(context->gpu_blur_commands->host() + i), (void *)cmd.get(), sizeof(BoxBlurCommand));
        }

        context->gpu_blur_commands->copy_host_to_device(stream);
    }
}

template<typename SrcWrap, typename DstWrap>
inline void RenderBlur_RGB(SrcWrap src, DstWrap dst, const cuda_op::DataShape &inputShape, cuOSDContext_t context,
                           cudaStream_t stream, bool skipCopy)
{
    if (!skipCopy && src.ptr(0) != dst.ptr(0))
    {
        dim3 blockSize(32, 32);
        dim3 gridSize(divUp(int(inputShape.W + 1), (int)blockSize.x), divUp(int(inputShape.H + 1), (int)blockSize.y),
                      inputShape.N);

        render_p2p_kernel<<<gridSize, blockSize, 0, stream>>>(src, dst, inputShape.N, inputShape.H, inputShape.W,
                                                              inputShape.C);
        checkKernelErrors();
    }

    if (context->blur_commands.size() > 0)
    {
        dim3 blockSize(32, 32);
        dim3 gridSize(context->blur_commands.size(), 1);

        render_blur_rgb_kernel<<<gridSize, blockSize, 0, stream>>>(
            src, dst, context->gpu_blur_commands ? context->gpu_blur_commands->device() : nullptr,
            context->blur_commands.size(), inputShape.N, inputShape.W, inputShape.H);
        checkKernelErrors();
    }
}

template<typename T>
inline ErrorCode ApplyBoxBlur_RGB(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                  cuOSDContext_t context, cudaStream_t stream, bool skipCopy)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N
        || outputShape.C != inputShape.C || outputShape.C != 3)
    {
        LOG_ERROR("Invalid output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    cuosd_apply(context, stream);

    int64_t srcMaxStride = inAccess->sampleStride() * inAccess->numSamples();
    int64_t dstMaxStride = outAccess->sampleStride() * outAccess->numSamples();

    if (std::max(srcMaxStride, dstMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = nvcv::cuda::CreateTensorWrapNHWC<T, int32_t>(inData);
        auto dst = nvcv::cuda::CreateTensorWrapNHWC<T, int32_t>(outData);

        RenderBlur_RGB(src, dst, inputShape, context, stream, skipCopy);
        return ErrorCode::SUCCESS;
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
}

template<typename SrcWrap, typename DstWrap>
inline void RenderBlur_RGBA(SrcWrap src, DstWrap dst, const cuda_op::DataShape &inputShape, cuOSDContext_t context,
                            cudaStream_t stream, bool skipCopy)
{
    if (!skipCopy && src.ptr(0) != dst.ptr(0))
    {
        dim3 blockSize(32, 32);
        dim3 gridSize(divUp(int(inputShape.W + 1), (int)blockSize.x), divUp(int(inputShape.H + 1), (int)blockSize.y),
                      inputShape.N);

        render_p2p_kernel<<<gridSize, blockSize, 0, stream>>>(src, dst, inputShape.N, inputShape.H, inputShape.W,
                                                              inputShape.C);
        checkKernelErrors();
    }

    if (context->blur_commands.size() > 0)
    {
        dim3 blockSize(32, 32);
        dim3 gridSize(context->blur_commands.size(), 1);

        render_blur_rgba_kernel<<<gridSize, blockSize, 0, stream>>>(
            src, dst, context->gpu_blur_commands ? context->gpu_blur_commands->device() : nullptr,
            context->blur_commands.size(), inputShape.N, inputShape.W, inputShape.H);
        checkKernelErrors();
    }
}

template<typename T>
inline ErrorCode ApplyBoxBlur_RGBA(const nvcv::TensorDataStridedCuda &inData,
                                   const nvcv::TensorDataStridedCuda &outData, cuOSDContext_t context,
                                   cudaStream_t stream, bool skipCopy)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N
        || outputShape.C != inputShape.C)
    {
        LOG_ERROR("Invalid output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    cuosd_apply(context, stream);

    int64_t srcMaxStride = inAccess->sampleStride() * inAccess->numSamples();
    int64_t dstMaxStride = outAccess->sampleStride() * outAccess->numSamples();

    if (std::max(srcMaxStride, dstMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = nvcv::cuda::CreateTensorWrapNHWC<T, int32_t>(inData);
        auto dst = nvcv::cuda::CreateTensorWrapNHWC<T, int32_t>(outData);

        RenderBlur_RGBA(src, dst, inputShape, context, stream, skipCopy);
        return ErrorCode::SUCCESS;
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
}

template<typename SrcWrap, typename DstWrap>
inline void RenderBlur_Planar(SrcWrap src, DstWrap dst, const cuda_op::DataShape &inputShape, cuOSDContext_t context,
                              cudaStream_t stream, bool skipCopy)
{
    if (!skipCopy && src.ptr(0, 0, 0, 0) != dst.ptr(0, 0, 0, 0))
    {
        dim3 blockSize(32, 32);
        dim3 gridSize(divUp(int(inputShape.W + 1), (int)blockSize.x), divUp(int(inputShape.H + 1), (int)blockSize.y),
                      inputShape.N);

        render_p2p_planar_kernel<<<gridSize, blockSize, 0, stream>>>(src, dst, inputShape.N, inputShape.H, inputShape.W,
                                                                     inputShape.C);
        checkKernelErrors();
    }

    if (context->blur_commands.size() > 0)
    {
        dim3 blockSize(32, 32);
        dim3 gridSize(context->blur_commands.size(), inputShape.C);

        render_blur_planar_kernel<<<gridSize, blockSize, 0, stream>>>(
            src, dst, context->gpu_blur_commands ? context->gpu_blur_commands->device() : nullptr,
            context->blur_commands.size(), inputShape.N, inputShape.W, inputShape.H, inputShape.C);
        checkKernelErrors();
    }
}

template<typename T>
inline ErrorCode ApplyBoxBlur_Planar(const nvcv::TensorDataStridedCuda &inData,
                                     const nvcv::TensorDataStridedCuda &outData, cuOSDContext_t context,
                                     cudaStream_t stream, bool skipCopy)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N
        || outputShape.C != inputShape.C || (outputShape.C != 3 && outputShape.C != 4))
    {
        LOG_ERROR("Invalid output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    cuosd_apply(context, stream);

    int64_t maxStride
        = std::max({inAccess->sampleStride() * inAccess->numSamples(), inAccess->chStride() * inAccess->numChannels(),
                    inAccess->rowStride(), outAccess->sampleStride() * outAccess->numSamples(),
                    outAccess->chStride() * outAccess->numChannels(), outAccess->rowStride()});

    if (maxStride <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = nvcv::cuda::CreateTensorWrapNCHW<T, int32_t>(inData);
        auto dst = nvcv::cuda::CreateTensorWrapNCHW<T, int32_t>(outData);

        RenderBlur_Planar(src, dst, inputShape, context, stream, skipCopy);
        return ErrorCode::SUCCESS;
    }
    else
    {
        LOG_ERROR("Input or output stride exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
}

static ErrorCode cuosd_draw_boxblur(cuOSDContext_t context, int width, int height, NVCVBlurBoxesImpl *bboxes)
{
    for (int n = 0; n < bboxes->batch(); n++)
    {
        auto numBoxes = bboxes->numBoxesAt(n);

        for (int i = 0; i < numBoxes; i++)
        {
            auto bbox   = bboxes->boxAt(n, i);
            int  left   = max(min(bbox.box.x, width - 1), 0);
            int  top    = max(min(bbox.box.y, height - 1), 0);
            int  right  = max(min(left + bbox.box.width - 1, width - 1), 0);
            int  bottom = max(min(top + bbox.box.height - 1, height - 1), 0);

            if (left == right || top == bottom)
            {
                LOG_DEBUG("Skipped box_blur(" << bbox.box.x << ", " << bbox.box.y << ", " << bbox.box.width << ", "
                                              << bbox.box.height << ") in image(" << width << ", " << height << ")");
                continue;
            }

            if (bbox.box.width < 3 || bbox.box.height < 3 || bbox.kernelSize < 1)
            {
                LOG_DEBUG(
                    "This operation will be ignored because the region of interest is too small, or the kernel is too "
                    "small at box_blur("
                    << bbox.box.x << ", " << bbox.box.y << bbox.box.width << ", " << bbox.box.height
                    << ") with kernelSize=" << bbox.kernelSize);
                continue;
            }

            auto cmd             = std::make_shared<BoxBlurCommand>();
            cmd->batch_index     = n;
            cmd->kernel_size     = bbox.kernelSize;
            cmd->bounding_left   = left;
            cmd->bounding_right  = right;
            cmd->bounding_top    = top;
            cmd->bounding_bottom = bottom;
            context->blur_commands.emplace_back(cmd);
        }
    }
    return ErrorCode::SUCCESS;
}

BoxBlur::BoxBlur(DataShape max_input_shape, DataShape max_output_shape)
    : CudaBaseOp(max_input_shape, max_output_shape)
{
    m_context                    = new cuOSDContext();
    m_context->gpu_blur_commands = std::make_unique<Memory<BoxBlurCommand>>();
    m_context->gpu_blur_commands->alloc_or_resize_to(PREALLOC_CMD_NUM * sizeof(BoxBlurCommand));
}

BoxBlur::~BoxBlur()
{
    if (m_context)
    {
        m_context->blur_commands.clear();
        cuOSDContext *p = (cuOSDContext *)m_context;
        delete p;
    }
}

size_t BoxBlur::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return CudaBaseOp::calBufferSize(max_input_shape, max_output_shape, max_data_type);
}

ErrorCode BoxBlur::infer(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                         NVCVBlurBoxesI bboxes, cudaStream_t stream, bool skipCopy)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (!(input_format == kNHWC || input_format == kHWC || input_format == kNCHW || input_format == kCHW)
        || !(output_format == kNHWC || output_format == kHWC || output_format == kNCHW || output_format == kCHW)
        || IsPlanar(input_format) != IsPlanar(output_format))
    {
        LOG_ERROR("Invalid DataFormat both Input and Output must be kNHWC, kHWC, kNCHW or kCHW");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const cuda_op::DataType data_type = GetLegacyDataType(inData.dtype());

    if (!(data_type == kCV_8U || data_type == kCV_8S))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int cols     = inAccess->numCols();

    if (channels > 4 || channels < 3)
    {
        LOG_ERROR("Invalid channel number ch = " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    NVCVBlurBoxesImpl *_bboxes = (NVCVBlurBoxesImpl *)bboxes;
    if (_bboxes->batch() != batch)
    {
        LOG_ERROR("bboxes batch " << _bboxes->batch() << " != input batch " << batch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto ret = cuosd_draw_boxblur(m_context, cols, rows, _bboxes);
    if (ret != ErrorCode::SUCCESS)
    {
        return ret;
    }

    typedef ErrorCode (*func_t)(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                cuOSDContext_t context, cudaStream_t stream, bool skipCopy);

    static const func_t funcs[][2] = {
        {ApplyBoxBlur_RGB<uint8_t>, ApplyBoxBlur_RGBA<uint8_t>},
        { ApplyBoxBlur_RGB<int8_t>,  ApplyBoxBlur_RGBA<int8_t>},
    };

    const int dataTypeIdx = data_type == kCV_8S ? 1 : 0;
    ErrorCode status      = ErrorCode::SUCCESS;
    if (IsPlanar(input_format))
    {
        status = dataTypeIdx == 0 ? ApplyBoxBlur_Planar<uint8_t>(inData, outData, m_context, stream, skipCopy)
                                  : ApplyBoxBlur_Planar<int8_t>(inData, outData, m_context, stream, skipCopy);
    }
    else
    {
        const int channelIdx = channels - 3;
        status               = funcs[dataTypeIdx][channelIdx](inData, outData, m_context, stream, skipCopy);
    }
    m_context->blur_commands.clear(); // Clear the command buffer so next render does not contain previous boxes.
    return status;
}

} // namespace nvcv::legacy::cuda_op

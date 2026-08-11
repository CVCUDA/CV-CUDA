/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <cstdint>

#define BLOCK 32

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

__global__ void copyGammaValues(float *gammaArray, const uint8_t *gammaBase, int64_t gammaStride, const int numImages,
                                const int channelCount)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    const float gamma = *reinterpret_cast<const float *>(gammaBase + index * gammaStride);
    for (int i = 0; i < channelCount; i++)
    {
        gammaArray[index * channelCount + i] = gamma;
    }
}

__global__ void copyPerChannelGammaValues(float *gammaArray, const uint8_t *gammaBase, int64_t sampleStride,
                                          int64_t channelStride, const int numImages, const int channelCount)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages * channelCount)
    {
        return;
    }

    int imageIndex   = index / channelCount;
    int channelIndex = index % channelCount;
    gammaArray[index]
        = *reinterpret_cast<const float *>(gammaBase + imageIndex * sampleStride + channelIndex * channelStride);
}

static bool IsTensorDense(const TensorDataStridedCuda &tensor)
{
    int64_t expectedStride = sizeof(float);
    for (int dim = tensor.rank() - 1; dim >= 0; --dim)
    {
        if (tensor.stride(dim) != expectedStride)
        {
            return false;
        }
        expectedStride *= tensor.shape(dim);
    }
    return true;
}

static bool GetPerImageGammaStride(const TensorDataStridedCuda &gammas, int numImages, int64_t &sampleStride)
{
    if (IsTensorDense(gammas))
    {
        sampleStride = sizeof(float);
        return true;
    }

    const int sampleDim = gammas.layout().find(nvcv::LABEL_BATCH);
    if (sampleDim >= 0 && gammas.shape(sampleDim) == numImages)
    {
        sampleStride = gammas.stride(sampleDim);
        return true;
    }

    if (gammas.rank() == 1 && gammas.shape(0) == numImages)
    {
        sampleStride = gammas.stride(0);
        return true;
    }

    if (gammas.rank() == 2 && gammas.shape(0) == numImages && gammas.shape(1) == 1)
    {
        sampleStride = gammas.stride(0);
        return true;
    }

    return false;
}

static bool GetPerChannelGammaStrides(const TensorDataStridedCuda &gammas, int numImages, int channelCount,
                                      int64_t &sampleStride, int64_t &channelStride)
{
    if (gammas.rank() == 1 && gammas.shape(0) == numImages * channelCount)
    {
        sampleStride  = gammas.stride(0) * channelCount;
        channelStride = gammas.stride(0);
        return true;
    }

    const int sampleDim  = gammas.layout().find(nvcv::LABEL_BATCH);
    const int channelDim = gammas.layout().find(nvcv::LABEL_CHANNEL);
    if (sampleDim >= 0 && channelDim >= 0 && gammas.shape(sampleDim) == numImages
        && gammas.shape(channelDim) == channelCount)
    {
        sampleStride  = gammas.stride(sampleDim);
        channelStride = gammas.stride(channelDim);
        return true;
    }

    if (gammas.rank() == 2 && gammas.shape(0) == numImages && gammas.shape(1) == channelCount)
    {
        sampleStride  = gammas.stride(0);
        channelStride = gammas.stride(1);
        return true;
    }

    return false;
}

// apply 255*((x/255)**gamma) on each pixel
template<typename D, typename gamma_type>
__global__ void gamma_contrast_kernel(const cuda::ImageBatchVarShapeWrap<D> src, cuda::ImageBatchVarShapeWrap<D> dst,
                                      const cuda::Tensor1DWrap<gamma_type> gamma_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    gamma_type gamma = gamma_[batch_idx];
    gamma_type tmp   = (*src.ptr(batch_idx, dst_y, dst_x) + 0.0f) / 255.0f;

    D out                             = nvcv::cuda::SaturateCast<D>(cuda::pow(tmp, gamma) * 255.0f);
    *dst.ptr(batch_idx, dst_y, dst_x) = out;
}

// Single-channel U8 var-shape path: one thread owns sixteen adjacent pixels so image metadata and gamma
// lookup are amortized across sixteen independent power evaluations. Misaligned rows and trailing
// columns remain scalar.
__global__ void gamma_contrast_u8_kernel(const cuda::ImageBatchVarShapeWrap<uchar> src,
                                         cuda::ImageBatchVarShapeWrap<uchar>       dst,
                                         const cuda::Tensor1DWrap<float>           gamma_)
{
    constexpr int NIX = 16;

    const int dst_x     = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int width     = dst.width(batch_idx);
    if (dst_x >= width || dst_y >= dst.height(batch_idx))
        return;

    const float gamma = gamma_[batch_idx];
    auto        apply = [gamma](uchar value)
    {
        const float tmp = (value + 0.0f) / 255.0f;
        return nvcv::cuda::SaturateCast<uchar>(cuda::pow(tmp, gamma) * 255.0f);
    };

    const uchar *src_ptr = src.ptr(batch_idx, dst_y, dst_x);
    uchar       *dst_ptr = dst.ptr(batch_idx, dst_y, dst_x);
    const bool   aligned = ((reinterpret_cast<std::uintptr_t>(src_ptr) | reinterpret_cast<std::uintptr_t>(dst_ptr))
                          & (alignof(uchar4) - 1))
                      == 0;

    if (dst_x + NIX <= width && aligned)
    {
        const uchar4 input0  = *reinterpret_cast<const uchar4 *>(src_ptr);
        const uchar4 input1  = *reinterpret_cast<const uchar4 *>(src_ptr + 4);
        const uchar4 input2  = *reinterpret_cast<const uchar4 *>(src_ptr + 8);
        const uchar4 input3  = *reinterpret_cast<const uchar4 *>(src_ptr + 12);
        const uchar4 output0 = make_uchar4(apply(input0.x), apply(input0.y), apply(input0.z), apply(input0.w));
        const uchar4 output1 = make_uchar4(apply(input1.x), apply(input1.y), apply(input1.z), apply(input1.w));
        const uchar4 output2 = make_uchar4(apply(input2.x), apply(input2.y), apply(input2.z), apply(input2.w));
        const uchar4 output3 = make_uchar4(apply(input3.x), apply(input3.y), apply(input3.z), apply(input3.w));
        *reinterpret_cast<uchar4 *>(dst_ptr)      = output0;
        *reinterpret_cast<uchar4 *>(dst_ptr + 4)  = output1;
        *reinterpret_cast<uchar4 *>(dst_ptr + 8)  = output2;
        *reinterpret_cast<uchar4 *>(dst_ptr + 12) = output3;
    }
    else
    {
#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            if (dst_x + i < width)
            {
                dst_ptr[i] = apply(src_ptr[i]);
            }
        }
    }
}

template<typename D, typename gamma_type, int NIX>
__global__ void gamma_contrast_u8_batched_kernel(const cuda::ImageBatchVarShapeWrap<D> src,
                                                 cuda::ImageBatchVarShapeWrap<D>       dst,
                                                 const cuda::Tensor1DWrap<gamma_type>  gamma_)
{
    const int dst_x     = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int width     = dst.width(batch_idx);
    if (dst_x >= width || dst_y >= dst.height(batch_idx))
        return;

    const gamma_type gamma   = gamma_[batch_idx];
    const D         *src_ptr = src.ptr(batch_idx, dst_y, dst_x);
    D               *dst_ptr = dst.ptr(batch_idx, dst_y, dst_x);

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        if (dst_x + i < width)
        {
            const gamma_type tmp = (src_ptr[i] + 0.0f) / 255.0f;
            dst_ptr[i]           = nvcv::cuda::SaturateCast<D>(cuda::pow(tmp, gamma) * 255.0f);
        }
    }
}

// apply (x**gamma) on each pixel
template<typename D, typename gamma_type>
__global__ void gamma_contrast_float_kernel(const cuda::ImageBatchVarShapeWrap<D> src,
                                            cuda::ImageBatchVarShapeWrap<D>       dst,
                                            const cuda::Tensor1DWrap<gamma_type>  gamma_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    gamma_type gamma = gamma_[batch_idx];

    D out = nvcv::cuda::SaturateCast<D>(cuda::pow(cuda::StaticCast<float>(*src.ptr(batch_idx, dst_y, dst_x)), gamma));

    *dst.ptr(batch_idx, dst_y, dst_x) = cuda::clamp(cuda::StaticCast<float>(out), 0.f, 1.f);
}

template<typename D, typename gamma_type, int NIX>
__global__ void gamma_contrast_float_batched_kernel(const cuda::ImageBatchVarShapeWrap<D> src,
                                                    cuda::ImageBatchVarShapeWrap<D>       dst,
                                                    const cuda::Tensor1DWrap<gamma_type>  gamma_)
{
    const int dst_x     = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int width     = dst.width(batch_idx);
    if (dst_x >= width || dst_y >= dst.height(batch_idx))
        return;

    const gamma_type gamma   = gamma_[batch_idx];
    const D         *src_ptr = src.ptr(batch_idx, dst_y, dst_x);
    D               *dst_ptr = dst.ptr(batch_idx, dst_y, dst_x);

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        if (dst_x + i < width)
        {
            const D out = nvcv::cuda::SaturateCast<D>(cuda::pow(cuda::StaticCast<float>(src_ptr[i]), gamma));
            dst_ptr[i]  = cuda::clamp(cuda::StaticCast<float>(out), 0.f, 1.f);
        }
    }
}

// Planar (NCHW/CHW) gamma contrast. Gamma is applied per pixel per channel and is independent of the
// other channels, so grid.z runs over images and the kernel loops the channel planes -- reading the
// per-channel gamma as gammaValues[batch*channels + plane], exactly the value the interleaved kernel
// pulls from its vector gamma_[batch]. Each plane's result is bit-exact with the interleaved path.
template<typename D>
__global__ void gamma_contrast_planar_kernel(const cuda::ImageBatchVarShapeWrap<D> src,
                                             cuda::ImageBatchVarShapeWrap<D>       dst,
                                             const cuda::Tensor1DWrap<float> gamma_, int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        const float gamma = gamma_[batch_idx * channels + plane];
        const float tmp   = (*src.ptr(batch_idx, plane, dst_y, dst_x) + 0.0f) / 255.0f;

        *dst.ptr(batch_idx, plane, dst_y, dst_x) = nvcv::cuda::SaturateCast<D>(cuda::pow(tmp, gamma) * 255.0f);
    }
}

__global__ void gamma_contrast_planar_u8_kernel(const cuda::ImageBatchVarShapeWrap<uchar> src,
                                                cuda::ImageBatchVarShapeWrap<uchar>       dst,
                                                const cuda::Tensor1DWrap<float> gamma_, int channels)
{
    constexpr int NIX = 16;

    const int dst_x     = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int width     = dst.width(batch_idx);
    if (dst_x >= width || dst_y >= dst.height(batch_idx))
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        const float gamma = gamma_[batch_idx * channels + plane];
        auto        apply = [gamma](uchar value)
        {
            const float tmp = (value + 0.0f) / 255.0f;
            return nvcv::cuda::SaturateCast<uchar>(cuda::pow(tmp, gamma) * 255.0f);
        };

        const uchar *src_ptr = src.ptr(batch_idx, plane, dst_y, dst_x);
        uchar       *dst_ptr = dst.ptr(batch_idx, plane, dst_y, dst_x);
        const bool   aligned = ((reinterpret_cast<std::uintptr_t>(src_ptr) | reinterpret_cast<std::uintptr_t>(dst_ptr))
                              & (alignof(uchar4) - 1))
                          == 0;

        if (dst_x + NIX <= width && aligned)
        {
            const uchar4 input0  = *reinterpret_cast<const uchar4 *>(src_ptr);
            const uchar4 input1  = *reinterpret_cast<const uchar4 *>(src_ptr + 4);
            const uchar4 input2  = *reinterpret_cast<const uchar4 *>(src_ptr + 8);
            const uchar4 input3  = *reinterpret_cast<const uchar4 *>(src_ptr + 12);
            const uchar4 output0 = make_uchar4(apply(input0.x), apply(input0.y), apply(input0.z), apply(input0.w));
            const uchar4 output1 = make_uchar4(apply(input1.x), apply(input1.y), apply(input1.z), apply(input1.w));
            const uchar4 output2 = make_uchar4(apply(input2.x), apply(input2.y), apply(input2.z), apply(input2.w));
            const uchar4 output3 = make_uchar4(apply(input3.x), apply(input3.y), apply(input3.z), apply(input3.w));
            *reinterpret_cast<uchar4 *>(dst_ptr)      = output0;
            *reinterpret_cast<uchar4 *>(dst_ptr + 4)  = output1;
            *reinterpret_cast<uchar4 *>(dst_ptr + 8)  = output2;
            *reinterpret_cast<uchar4 *>(dst_ptr + 12) = output3;
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NIX; ++i)
            {
                if (dst_x + i < width)
                {
                    dst_ptr[i] = apply(src_ptr[i]);
                }
            }
        }
    }
}

// Planar gamma contrast for float images: apply (x**gamma) and clamp to [0, 1], per channel plane.
template<typename D>
__global__ void gamma_contrast_float_planar_kernel(const cuda::ImageBatchVarShapeWrap<D> src,
                                                   cuda::ImageBatchVarShapeWrap<D>       dst,
                                                   const cuda::Tensor1DWrap<float> gamma_, int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        const float gamma = gamma_[batch_idx * channels + plane];

        D out = nvcv::cuda::SaturateCast<D>(
            cuda::pow(cuda::StaticCast<float>(*src.ptr(batch_idx, plane, dst_y, dst_x)), gamma));

        *dst.ptr(batch_idx, plane, dst_y, dst_x) = cuda::clamp(cuda::StaticCast<float>(out), 0.f, 1.f);
    }
}

template<typename T>
void gamma_contrast_planar(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                           float *gammaValues, int channels, cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                            block(BLOCK, BLOCK / 4, 1);
    dim3                            grid(divUp(max_width, block.x), divUp(max_height, block.y), batch);
    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_planar_kernel<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma, channels);

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

static void gamma_contrast_planar_u8(const ImageBatchVarShapeDataStridedCuda &in,
                                     const ImageBatchVarShapeDataStridedCuda &out, float *gammaValues, int channels,
                                     cudaStream_t stream)
{
    constexpr int NIX = 16;

    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                                block(BLOCK, BLOCK / 4, 1);
    dim3                                grid(divUp(max_width, block.x * NIX), divUp(max_height, block.y), batch);
    cuda::ImageBatchVarShapeWrap<uchar> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<uchar> dst_ptr(out);

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_planar_u8_kernel<<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma, channels);
    checkKernelErrors();
}

template<typename T>
void gamma_contrast_float_planar(const ImageBatchVarShapeDataStridedCuda &in,
                                 const ImageBatchVarShapeDataStridedCuda &out, float *gammaValues, int channels,
                                 cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                            block(BLOCK, BLOCK / 4, 1);
    dim3                            grid(divUp(max_width, block.x), divUp(max_height, block.y), batch);
    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_float_planar_kernel<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma, channels);
    checkKernelErrors();
}

template<typename T>
void gamma_contrast(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                    float *gammaValues, cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                            block(BLOCK, BLOCK / 4, 1);
    dim3                            grid(divUp(max_width, block.x), divUp(max_height, block.y), batch);
    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

static void gamma_contrast_u8(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                              float *gammaValues, cudaStream_t stream)
{
    constexpr int NIX = 16;

    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                                block(BLOCK, BLOCK / 4, 1);
    dim3                                grid(divUp(max_width, block.x * NIX), divUp(max_height, block.y), batch);
    cuda::ImageBatchVarShapeWrap<uchar> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<uchar> dst_ptr(out);

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_u8_kernel<<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);
    checkKernelErrors();
}

template<typename T>
void gamma_contrast_u8_batched(const ImageBatchVarShapeDataStridedCuda &in,
                               const ImageBatchVarShapeDataStridedCuda &out, float *gammaValues, cudaStream_t stream)
{
    constexpr int NIX = sizeof(T) == sizeof(uchar4) ? 3 : 2;

    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                            block(BLOCK, BLOCK / 4, 1);
    dim3                            grid(divUp(max_width, block.x * NIX), divUp(max_height, block.y), batch);
    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_u8_batched_kernel<T, gamma_type, NIX><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);
    checkKernelErrors();
}

template<typename T>
void gamma_contrast_float(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                          float *gammaValues, cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                            block(BLOCK, BLOCK / 4, 1);
    dim3                            grid(divUp(max_width, block.x), divUp(max_height, block.y), batch);
    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_float_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);
    checkKernelErrors();
}

template<typename T>
void gamma_contrast_float_batched(const ImageBatchVarShapeDataStridedCuda &in,
                                  const ImageBatchVarShapeDataStridedCuda &out, float *gammaValues, cudaStream_t stream)
{
    constexpr int NIX = 2;

    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                            block(BLOCK, BLOCK / 4, 1);
    dim3                            grid(divUp(max_width, block.x * NIX), divUp(max_height, block.y), batch);
    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_float_batched_kernel<T, gamma_type, NIX><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);
    checkKernelErrors();
}

GammaContrastVarShape::GammaContrastVarShape(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount)
    : CudaBaseOp()
    , m_maxBatchSize(maxVarShapeBatchSize)
    , m_maxChannelCount(maxVarShapeChannelCount)
{
    if (m_maxBatchSize > 0 && m_maxChannelCount > 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(&m_gammaArray, m_maxBatchSize * m_maxChannelCount * sizeof(float)));
    }
}

GammaContrastVarShape::~GammaContrastVarShape()
{
    NVCV_CHECK_LOG(cudaFree(m_gammaArray));
}

ErrorCode GammaContrastVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                       const ImageBatchVarShapeDataStridedCuda &outData,
                                       const TensorDataStridedCuda &gammas, cudaStream_t stream)
{
    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (m_maxBatchSize <= 0 || inData.numImages() > m_maxBatchSize)
    {
        LOG_ERROR("Invalid maximum batch size");
        return ErrorCode::INVALID_PARAMETER;
    }

    if (m_maxChannelCount <= 0 || inData.uniqueFormat().numChannels() > m_maxChannelCount)
    {
        LOG_ERROR("Invalid maximum channel count");
        return ErrorCode::INVALID_PARAMETER;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid DataFormat " << format
                                        << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType out_data_type = helpers::GetLegacyDataType(outData.uniqueFormat());
    int      channels      = inData.uniqueFormat().numChannels();
    int      outChannels   = outData.uniqueFormat().numChannels();

    if (out_data_type != data_type || outChannels != channels)
    {
        LOG_ERROR("Input DataType " << data_type << " and channel count " << channels << " must match Output DataType "
                                    << out_data_type << " and channel count " << outChannels);
        return out_data_type != data_type ? ErrorCode::INVALID_DATA_TYPE : ErrorCode::INVALID_DATA_SHAPE;
    }

    // Planar 2-channel is rejected: there is no defined 2-plane planar format (matches Resize/Flip).
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // The planar path launches grid.z over images and loops the channel planes inside the kernel, so
    // numImages must fit CUDA's 65535 grid-z limit.
    if (isPlanar && static_cast<int64_t>(inData.numImages()) > 65535)
    {
        LOG_ERROR("Planar gamma contrast requires numImages <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int numElements = 1;
    for (int i = 0; i < gammas.rank(); i++)
    {
        numElements *= gammas.shape(i);
    }

    int numImages = inData.numImages();
    if (numElements != numImages && numElements != numImages * channels)
    {
        LOG_ERROR("Invalid gamma tensor length " << numElements << ", expected " << numImages << " or "
                                                 << numImages * channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (numImages * channels == numElements)
    {
        if (IsTensorDense(gammas))
        {
            checkCudaErrors(cudaMemcpyAsync(m_gammaArray, gammas.basePtr(), sizeof(float) * numImages * channels,
                                            cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            int64_t sampleStride;
            int64_t channelStride;
            if (!GetPerChannelGammaStrides(gammas, numImages, channels, sampleStride, channelStride))
            {
                LOG_ERROR("Per-channel gamma tensor must be dense or shaped as per-image channel slices");
                return ErrorCode::INVALID_DATA_SHAPE;
            }

            copyPerChannelGammaValues<<<divUp(numImages * channels, BLOCK), BLOCK, 0, stream>>>(
                m_gammaArray, reinterpret_cast<const uint8_t *>(gammas.basePtr()), sampleStride, channelStride,
                numImages, channels);
            checkKernelErrors();
        }
    }
    else
    {
        int64_t sampleStride;
        if (!GetPerImageGammaStride(gammas, numImages, sampleStride))
        {
            LOG_ERROR("Per-image gamma tensor must be dense or have a sample dimension matching the input batch");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        copyGammaValues<<<divUp(numImages, BLOCK), BLOCK, 0, stream>>>(
            m_gammaArray, reinterpret_cast<const uint8_t *>(gammas.basePtr()), sampleStride, numImages, channels);
        checkKernelErrors();
    }

    typedef void (*func_t)(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                           const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *gammas, cudaStream_t stream);

    static const func_t funcs[5][4] = {
        {          gamma_contrast_u8,      gamma_contrast<uchar2>, gamma_contrast_u8_batched<uchar3>,
         gamma_contrast_u8_batched<uchar4>                                                                                   },
        {0 /*gamma_contrast<schar>*/, 0 /*gamma_contrast<char2>*/,       0 /*gamma_contrast<char3>*/,
         0 /*gamma_contrast<char4>*/                                                                                         },
        {     gamma_contrast<ushort>,     gamma_contrast<ushort2>,           gamma_contrast<ushort3>, gamma_contrast<ushort4>},
        {      gamma_contrast<short>,      gamma_contrast<short2>,            gamma_contrast<short3>,  gamma_contrast<short4>},
        {        gamma_contrast<int>,        gamma_contrast<int2>,              gamma_contrast<int3>,    gamma_contrast<int4>},
    };

    static const func_t funcs_float[4] = {gamma_contrast_float_batched<float>, gamma_contrast_float<float2>,
                                          gamma_contrast_float<float3>, gamma_contrast_float<float4>};

    if (isPlanar)
    {
        // Planar dispatch indexes by dtype only: each channel is a separate single-channel plane, so
        // one scalar specialization per dtype covers any channel count (the kernel loops the planes).
        typedef void (*planar_func_t)(const ImageBatchVarShapeDataStridedCuda &in,
                                      const ImageBatchVarShapeDataStridedCuda &out, float *gammas, int channels,
                                      cudaStream_t stream);

        static const planar_func_t planar_funcs[5]
            = {gamma_contrast_planar_u8, 0 /*schar*/, gamma_contrast_planar<ushort>, gamma_contrast_planar<short>,
               gamma_contrast_planar<int>};

        if (data_type == kCV_32F)
        {
            gamma_contrast_float_planar<float>(inData, outData, m_gammaArray, channels, stream);
        }
        else
        {
            const planar_func_t planar_func = planar_funcs[data_type];
            NVCV_ASSERT(planar_func != nullptr);
            planar_func(inData, outData, m_gammaArray, channels, stream);
        }

        return ErrorCode::SUCCESS;
    }

    if (data_type == kCV_32F)
    {
        const func_t func = funcs_float[channels - 1];
        func(inData, outData, m_gammaArray, stream);
    }
    else
    {
        const func_t func = funcs[data_type][channels - 1];
        func(inData, outData, m_gammaArray, stream);
    }

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

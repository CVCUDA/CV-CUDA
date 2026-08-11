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
#include "gamma_contrast_common.cuh"

#include <nvcv/TensorDataAccess.hpp>

#include <cstdint>

#define BLOCK 32

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

namespace {

namespace detail = gamma_contrast_detail;

// Interleaved (NHWC/HWC) tensor gamma contrast: one thread per output pixel, applying the per-channel
// gamma vector gamma_[sample] -- the same dense [sample*channels] layout the var-shape path uses, so
// results are bit-exact with the var-shape kernel.
template<typename D, typename gamma_type, class SrcWrapper, class DstWrapper>
__global__ void gamma_contrast_tensor_kernel(SrcWrapper src, DstWrapper dst,
                                             const cuda::Tensor1DWrap<gamma_type> gamma_, int2 size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= size.x || y >= size.y)
        return;

    gamma_type gamma = gamma_[z];
    gamma_type tmp   = (src[int3{x, y, z}] + 0.0f) / 255.0f;

    dst[int3{x, y, z}] = nvcv::cuda::SaturateCast<D>(cuda::pow(tmp, gamma) * 255.0f);
}

template<typename D, typename gamma_type, class SrcWrapper, class DstWrapper, int NIX>
__global__ void gamma_contrast_tensor_u8_batched_kernel(SrcWrapper src, DstWrapper dst,
                                                        const cuda::Tensor1DWrap<gamma_type> gamma_, int2 size)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= size.x || y >= size.y)
        return;

    const gamma_type gamma = gamma_[z];

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        if (x + i < size.x)
        {
            const gamma_type tmp   = (src[int3{x + i, y, z}] + 0.0f) / 255.0f;
            dst[int3{x + i, y, z}] = nvcv::cuda::SaturateCast<D>(cuda::pow(tmp, gamma) * 255.0f);
        }
    }
}

template<typename D, typename gamma_type, class SrcWrapper, class DstWrapper>
__global__ void gamma_contrast_tensor_float_kernel(SrcWrapper src, DstWrapper dst,
                                                   const cuda::Tensor1DWrap<gamma_type> gamma_, int2 size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= size.x || y >= size.y)
        return;

    gamma_type gamma = gamma_[z];

    D out = nvcv::cuda::SaturateCast<D>(cuda::pow(cuda::StaticCast<float>(src[int3{x, y, z}]), gamma));

    dst[int3{x, y, z}] = cuda::clamp(cuda::StaticCast<float>(out), 0.f, 1.f);
}

// Planar (NCHW/CHW) tensor gamma contrast: one thread per output pixel, looping the channel planes and
// reading the per-channel gamma as gammaArray[sample*channels + plane]. Bit-exact with the interleaved
// and var-shape paths per channel.
template<typename D, class SrcWrapper, class DstWrapper>
__global__ void gamma_contrast_tensor_planar_kernel(SrcWrapper src, DstWrapper dst,
                                                    const cuda::Tensor1DWrap<float> gamma_, int2 size, int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= size.x || y >= size.y)
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        const float gamma = gamma_[z * channels + plane];
        const float tmp   = (src[int4{x, y, plane, z}] + 0.0f) / 255.0f;

        dst[int4{x, y, plane, z}] = nvcv::cuda::SaturateCast<D>(cuda::pow(tmp, gamma) * 255.0f);
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void gamma_contrast_tensor_planar_u8_kernel(SrcWrapper src, DstWrapper dst,
                                                       const cuda::Tensor1DWrap<float> gamma_, int2 size, int channels)
{
    constexpr int NIX = 8;

    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= size.x || y >= size.y)
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        const float gamma = gamma_[z * channels + plane];
        auto        apply = [gamma](uchar value)
        {
            const float tmp = (value + 0.0f) / 255.0f;
            return nvcv::cuda::SaturateCast<uchar>(cuda::pow(tmp, gamma) * 255.0f);
        };

        const uchar *src_ptr = &src[int4{x, y, plane, z}];
        uchar       *dst_ptr = &dst[int4{x, y, plane, z}];
        const bool   aligned = ((reinterpret_cast<std::uintptr_t>(src_ptr) | reinterpret_cast<std::uintptr_t>(dst_ptr))
                              & (alignof(uchar4) - 1))
                          == 0;

        if (x + NIX <= size.x && aligned)
        {
            const uchar4 input0  = *reinterpret_cast<const uchar4 *>(src_ptr);
            const uchar4 input1  = *reinterpret_cast<const uchar4 *>(src_ptr + 4);
            const uchar4 output0 = make_uchar4(apply(input0.x), apply(input0.y), apply(input0.z), apply(input0.w));
            const uchar4 output1 = make_uchar4(apply(input1.x), apply(input1.y), apply(input1.z), apply(input1.w));
            *reinterpret_cast<uchar4 *>(dst_ptr)     = output0;
            *reinterpret_cast<uchar4 *>(dst_ptr + 4) = output1;
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NIX; ++i)
            {
                if (x + i < size.x)
                {
                    dst_ptr[i] = apply(src_ptr[i]);
                }
            }
        }
    }
}

template<typename D, class SrcWrapper, class DstWrapper>
__global__ void gamma_contrast_tensor_planar_float_kernel(SrcWrapper src, DstWrapper dst,
                                                          const cuda::Tensor1DWrap<float> gamma_, int2 size,
                                                          int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= size.x || y >= size.y)
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        const float gamma = gamma_[z * channels + plane];

        D out = nvcv::cuda::SaturateCast<D>(cuda::pow(cuda::StaticCast<float>(src[int4{x, y, plane, z}]), gamma));

        dst[int4{x, y, plane, z}] = cuda::clamp(cuda::StaticCast<float>(out), 0.f, 1.f);
    }
}

// Scalar (host-float) gamma/gain variants. The gamma and gain are passed by value as kernel launch
// arguments (no device gamma array, no ExpandGamma copy) and applied as out = gain * in**gamma -- the
// torchvision adjust_gamma formula. Nearest rounding with gain == 1.0f (an exact IEEE no-op) is
// bit-exact with the device-tensor counterpart above fed a gamma tensor filled with the same value;
// truncation provides torchvision-compatible integer conversion.
template<bool kTruncate, typename D, typename U>
__device__ __forceinline__ D gamma_contrast_saturate_cast(U value)
{
    if constexpr (kTruncate)
    {
        value = nvcv::cuda::round<nvcv::cuda::RoundMode::ZERO>(value);
    }
    return nvcv::cuda::SaturateCast<D>(value);
}

template<bool kTruncate, typename D, typename gamma_type, class SrcWrapper, class DstWrapper>
__global__ void gamma_contrast_tensor_scalar_kernel(SrcWrapper src, DstWrapper dst, float gamma, float gain, int2 size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= size.x || y >= size.y)
        return;

    gamma_type g   = nvcv::cuda::SetAll<gamma_type>(gamma);
    gamma_type tmp = (src[int3{x, y, z}] + 0.0f) / 255.0f;

    dst[int3{x, y, z}] = gamma_contrast_saturate_cast<kTruncate, D>(cuda::pow(tmp, g) * gain * 255.0f);
}

template<typename D, typename gamma_type, class SrcWrapper, class DstWrapper>
__global__ void gamma_contrast_tensor_scalar_float_kernel(SrcWrapper src, DstWrapper dst, float gamma, float gain,
                                                          int2 size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= size.x || y >= size.y)
        return;

    gamma_type g = nvcv::cuda::SetAll<gamma_type>(gamma);

    D out = nvcv::cuda::SaturateCast<D>(cuda::pow(cuda::StaticCast<float>(src[int3{x, y, z}]), g) * gain);

    dst[int3{x, y, z}] = cuda::clamp(cuda::StaticCast<float>(out), 0.f, 1.f);
}

template<bool kTruncate, typename D, class SrcWrapper, class DstWrapper>
__global__ void gamma_contrast_tensor_planar_scalar_kernel(SrcWrapper src, DstWrapper dst, float gamma, float gain,
                                                           int2 size, int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= size.x || y >= size.y)
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        const float tmp = (src[int4{x, y, plane, z}] + 0.0f) / 255.0f;

        dst[int4{x, y, plane, z}] = gamma_contrast_saturate_cast<kTruncate, D>(cuda::pow(tmp, gamma) * gain * 255.0f);
    }
}

template<typename D, class SrcWrapper, class DstWrapper>
__global__ void gamma_contrast_tensor_planar_scalar_float_kernel(SrcWrapper src, DstWrapper dst, float gamma,
                                                                 float gain, int2 size, int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= size.x || y >= size.y)
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        D out
            = nvcv::cuda::SaturateCast<D>(cuda::pow(cuda::StaticCast<float>(src[int4{x, y, plane, z}]), gamma) * gain);

        dst[int4{x, y, plane, z}] = cuda::clamp(cuda::StaticCast<float>(out), 0.f, 1.f);
    }
}

// Interleaved launcher (T = vector pixel type for the channel count).
template<typename T>
void gamma_contrast_tensor(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                           float *gammaValues, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(size.x, block.x), divUp(size.y, block.y), batch);

    auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_tensor_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src, dst, gamma, size);
    checkKernelErrors();
}

template<typename T>
void gamma_contrast_tensor_u8_batched(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                      float *gammaValues, cudaStream_t stream)
{
    constexpr int NIX = 2;

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(size.x, block.x * NIX), divUp(size.y, block.y), batch);

    auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_tensor_u8_batched_kernel<T, gamma_type, decltype(src), decltype(dst), NIX>
        <<<grid, block, 0, stream>>>(src, dst, gamma, size);
    checkKernelErrors();
}

template<typename T>
void gamma_contrast_tensor_float(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                 float *gammaValues, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(size.x, block.x), divUp(size.y, block.y), batch);

    auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_tensor_float_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src, dst, gamma, size);
    checkKernelErrors();
}

// Planar launcher (T = scalar base type, one plane per channel).
template<typename T>
void gamma_contrast_tensor_planar(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                  float *gammaValues, int channels, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(size.x, block.x), divUp(size.y, block.y), batch);

    auto src = cuda::Tensor4DWrap<const T, int32_t>(inData.basePtr(), static_cast<int32_t>(srcAccess->sampleStride()),
                                                    static_cast<int32_t>(srcAccess->planeStride()),
                                                    static_cast<int32_t>(srcAccess->rowStride()));
    auto dst = cuda::Tensor4DWrap<T, int32_t>(outData.basePtr(), static_cast<int32_t>(dstAccess->sampleStride()),
                                              static_cast<int32_t>(dstAccess->planeStride()),
                                              static_cast<int32_t>(dstAccess->rowStride()));

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_tensor_planar_kernel<T><<<grid, block, 0, stream>>>(src, dst, gamma, size, channels);
    checkKernelErrors();
}

static void gamma_contrast_tensor_planar_u8(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                            float *gammaValues, int channels, cudaStream_t stream)
{
    constexpr int NIX = 8;

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(size.x, block.x * NIX), divUp(size.y, block.y), batch);

    auto src = cuda::Tensor4DWrap<const uchar, int32_t>(
        inData.basePtr(), static_cast<int32_t>(srcAccess->sampleStride()),
        static_cast<int32_t>(srcAccess->planeStride()), static_cast<int32_t>(srcAccess->rowStride()));
    auto dst = cuda::Tensor4DWrap<uchar, int32_t>(outData.basePtr(), static_cast<int32_t>(dstAccess->sampleStride()),
                                                  static_cast<int32_t>(dstAccess->planeStride()),
                                                  static_cast<int32_t>(dstAccess->rowStride()));

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_tensor_planar_u8_kernel<<<grid, block, 0, stream>>>(src, dst, gamma, size, channels);
    checkKernelErrors();
}

template<typename T>
void gamma_contrast_tensor_planar_float(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                        float *gammaValues, int channels, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(size.x, block.x), divUp(size.y, block.y), batch);

    auto src = cuda::Tensor4DWrap<const T, int32_t>(inData.basePtr(), static_cast<int32_t>(srcAccess->sampleStride()),
                                                    static_cast<int32_t>(srcAccess->planeStride()),
                                                    static_cast<int32_t>(srcAccess->rowStride()));
    auto dst = cuda::Tensor4DWrap<T, int32_t>(outData.basePtr(), static_cast<int32_t>(dstAccess->sampleStride()),
                                              static_cast<int32_t>(dstAccess->planeStride()),
                                              static_cast<int32_t>(dstAccess->rowStride()));

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_tensor_planar_float_kernel<T><<<grid, block, 0, stream>>>(src, dst, gamma, size, channels);
    checkKernelErrors();
}

// Scalar (host-float) launchers -- same grids/wrappers as the device-tensor launchers above, but the
// gamma/gain scalars are passed straight into the kernel launch (no gamma array).
template<typename T>
void gamma_contrast_tensor_scalar(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                  float gamma, float gain, NVCVRoundMode roundMode, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(size.x, block.x), divUp(size.y, block.y), batch);

    auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    if (roundMode == NVCV_ROUND_TRUNCATE)
    {
        gamma_contrast_tensor_scalar_kernel<true, T, gamma_type>
            <<<grid, block, 0, stream>>>(src, dst, gamma, gain, size);
    }
    else
    {
        gamma_contrast_tensor_scalar_kernel<false, T, gamma_type>
            <<<grid, block, 0, stream>>>(src, dst, gamma, gain, size);
    }
    checkKernelErrors();
}

template<typename T>
void gamma_contrast_tensor_scalar_float(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                        float gamma, float gain, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(size.x, block.x), divUp(size.y, block.y), batch);

    auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    gamma_contrast_tensor_scalar_float_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src, dst, gamma, gain, size);
    checkKernelErrors();
}

template<typename T>
void gamma_contrast_tensor_planar_scalar(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                         float gamma, float gain, int channels, NVCVRoundMode roundMode,
                                         cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(size.x, block.x), divUp(size.y, block.y), batch);

    auto src = cuda::Tensor4DWrap<const T, int32_t>(inData.basePtr(), static_cast<int32_t>(srcAccess->sampleStride()),
                                                    static_cast<int32_t>(srcAccess->planeStride()),
                                                    static_cast<int32_t>(srcAccess->rowStride()));
    auto dst = cuda::Tensor4DWrap<T, int32_t>(outData.basePtr(), static_cast<int32_t>(dstAccess->sampleStride()),
                                              static_cast<int32_t>(dstAccess->planeStride()),
                                              static_cast<int32_t>(dstAccess->rowStride()));

    if (roundMode == NVCV_ROUND_TRUNCATE)
    {
        gamma_contrast_tensor_planar_scalar_kernel<true, T>
            <<<grid, block, 0, stream>>>(src, dst, gamma, gain, size, channels);
    }
    else
    {
        gamma_contrast_tensor_planar_scalar_kernel<false, T>
            <<<grid, block, 0, stream>>>(src, dst, gamma, gain, size, channels);
    }
    checkKernelErrors();
}

template<typename T>
void gamma_contrast_tensor_planar_scalar_float(const TensorDataStridedCuda &inData,
                                               const TensorDataStridedCuda &outData, float gamma, float gain,
                                               int channels, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(size.x, block.x), divUp(size.y, block.y), batch);

    auto src = cuda::Tensor4DWrap<const T, int32_t>(inData.basePtr(), static_cast<int32_t>(srcAccess->sampleStride()),
                                                    static_cast<int32_t>(srcAccess->planeStride()),
                                                    static_cast<int32_t>(srcAccess->rowStride()));
    auto dst = cuda::Tensor4DWrap<T, int32_t>(outData.basePtr(), static_cast<int32_t>(dstAccess->sampleStride()),
                                              static_cast<int32_t>(dstAccess->planeStride()),
                                              static_cast<int32_t>(dstAccess->rowStride()));

    gamma_contrast_tensor_planar_scalar_float_kernel<T>
        <<<grid, block, 0, stream>>>(src, dst, gamma, gain, size, channels);
    checkKernelErrors();
}

struct TensorInfo
{
    DataType dataType;
    bool     isPlanar;
    int      numSamples;
    int      channels;
};

static ErrorCode validateTensorPair(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                    TensorInfo &info)
{
    DataFormat inputFormat  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat outputFormat = helpers::GetLegacyDataFormat(outData.layout());
    if (inputFormat != outputFormat)
    {
        LOG_ERROR("Invalid DataFormat between input (" << inputFormat << ") and output (" << outputFormat << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(inputFormat == kNHWC || inputFormat == kHWC || inputFormat == kNCHW || inputFormat == kCHW))
    {
        LOG_ERROR("Invalid DataFormat " << inputFormat
                                        << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (inputFormat == kNCHW || inputFormat == kCHW);

    DataType dataType    = GetLegacyDataType(inData.dtype());
    DataType outDataType = GetLegacyDataType(outData.dtype());
    if (dataType != outDataType)
    {
        LOG_ERROR("Input DataType " << dataType << " must match output DataType " << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(dataType == kCV_8U || dataType == kCV_16U || dataType == kCV_16S || dataType == kCV_32S
          || dataType == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << dataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!srcAccess || !dstAccess)
    {
        LOG_ERROR("Failed to create tensor access");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (srcAccess->numSamples() != dstAccess->numSamples() || srcAccess->numRows() != dstAccess->numRows()
        || srcAccess->numCols() != dstAccess->numCols())
    {
        LOG_ERROR("Input and output must have matching sample count, width, and height");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const int numSamples = srcAccess->numSamples();
    const int channels   = srcAccess->numChannels();
    if (channels != dstAccess->numChannels())
    {
        LOG_ERROR("Input and output channel counts must match");
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    info = {dataType, isPlanar, numSamples, channels};
    return ErrorCode::SUCCESS;
}

} // namespace

GammaContrast::GammaContrast(const int32_t maxBatchSize, const int32_t maxChannelCount)
    : CudaBaseOp()
    , m_maxBatchSize(maxBatchSize)
    , m_maxChannelCount(maxChannelCount)
{
    if (m_maxBatchSize > 0 && m_maxChannelCount > 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(&m_gammaArray, m_maxBatchSize * m_maxChannelCount * sizeof(float)));
    }
}

GammaContrast::~GammaContrast()
{
    NVCV_CHECK_LOG(cudaFree(m_gammaArray));
}

ErrorCode GammaContrast::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                               const TensorDataStridedCuda &gammas, cudaStream_t stream)
{
    TensorInfo info;
    ErrorCode  validateErr = validateTensorPair(inData, outData, info);
    if (validateErr != ErrorCode::SUCCESS)
    {
        return validateErr;
    }

    const DataType data_type  = info.dataType;
    const bool     isPlanar   = info.isPlanar;
    const int      numSamples = info.numSamples;
    const int      channels   = info.channels;

    if (m_maxBatchSize <= 0 || numSamples > m_maxBatchSize)
    {
        LOG_ERROR("Invalid maximum batch size");
        return ErrorCode::INVALID_PARAMETER;
    }
    if (m_maxChannelCount <= 0 || channels > m_maxChannelCount)
    {
        LOG_ERROR("Invalid maximum channel count");
        return ErrorCode::INVALID_PARAMETER;
    }

    // The planar path launches grid.z over samples; enforce CUDA's 65535 grid-z limit.
    if (static_cast<int64_t>(numSamples) > 65535)
    {
        LOG_ERROR("GammaContrast tensor requires numSamples <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    ErrorCode gammaErr = detail::ExpandGamma(gammas, numSamples, channels, m_gammaArray, stream);
    if (gammaErr != ErrorCode::SUCCESS)
    {
        return gammaErr;
    }

    if (isPlanar)
    {
        typedef void (*planar_func_t)(const TensorDataStridedCuda &, const TensorDataStridedCuda &, float *, int,
                                      cudaStream_t);
        static const planar_func_t planar_funcs[5]
            = {gamma_contrast_tensor_planar_u8, 0 /*schar*/, gamma_contrast_tensor_planar<ushort>,
               gamma_contrast_tensor_planar<short>, gamma_contrast_tensor_planar<int>};
        if (data_type == kCV_32F)
        {
            gamma_contrast_tensor_planar_float<float>(inData, outData, m_gammaArray, channels, stream);
        }
        else
        {
            NVCV_ASSERT(planar_funcs[data_type] != nullptr);
            planar_funcs[data_type](inData, outData, m_gammaArray, channels, stream);
        }
        return ErrorCode::SUCCESS;
    }

    typedef void (*func_t)(const TensorDataStridedCuda &, const TensorDataStridedCuda &, float *, cudaStream_t);

    static const func_t funcs[5][4] = {
        { gamma_contrast_tensor<uchar>,  gamma_contrast_tensor<uchar2>, gamma_contrast_tensor_u8_batched<uchar3>,
         gamma_contrast_tensor_u8_batched<uchar4>                                                                  },
        {                            0,                              0,                                        0, 0},
        {gamma_contrast_tensor<ushort>, gamma_contrast_tensor<ushort2>,           gamma_contrast_tensor<ushort3>,
         gamma_contrast_tensor<ushort4>                                                                            },
        { gamma_contrast_tensor<short>,  gamma_contrast_tensor<short2>,            gamma_contrast_tensor<short3>,
         gamma_contrast_tensor<short4>                                                                             },
        {   gamma_contrast_tensor<int>,    gamma_contrast_tensor<int2>,              gamma_contrast_tensor<int3>,
         gamma_contrast_tensor<int4>                                                                               },
    };
    static const func_t funcs_float[4] = {gamma_contrast_tensor_float<float>, gamma_contrast_tensor_float<float2>,
                                          gamma_contrast_tensor_float<float3>, gamma_contrast_tensor_float<float4>};

    if (data_type == kCV_32F)
    {
        funcs_float[channels - 1](inData, outData, m_gammaArray, stream);
    }
    else
    {
        NVCV_ASSERT(funcs[data_type][channels - 1] != nullptr);
        funcs[data_type][channels - 1](inData, outData, m_gammaArray, stream);
    }

    return ErrorCode::SUCCESS;
}

// Scalar (host-float) gamma/gain path. The scalars are baked into the kernel launch, so there is no
// gamma tensor, ExpandGamma copy, or m_gammaArray/max-batch/max-channel scratch bound.
ErrorCode GammaContrast::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, float gamma,
                               float gain, NVCVRoundMode roundMode, cudaStream_t stream)
{
    if (roundMode != NVCV_ROUND_NEAREST && roundMode != NVCV_ROUND_TRUNCATE)
    {
        LOG_ERROR("Invalid round mode " << static_cast<int>(roundMode));
        return ErrorCode::INVALID_PARAMETER;
    }

    TensorInfo info;
    ErrorCode  validateErr = validateTensorPair(inData, outData, info);
    if (validateErr != ErrorCode::SUCCESS)
    {
        return validateErr;
    }

    const DataType data_type  = info.dataType;
    const bool     isPlanar   = info.isPlanar;
    const int      numSamples = info.numSamples;
    const int      channels   = info.channels;

    // The planar path launches grid.z over samples; enforce CUDA's 65535 grid-z limit.
    if (static_cast<int64_t>(numSamples) > 65535)
    {
        LOG_ERROR("GammaContrast tensor requires numSamples <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (isPlanar)
    {
        typedef void (*planar_func_t)(const TensorDataStridedCuda &, const TensorDataStridedCuda &, float, float, int,
                                      NVCVRoundMode, cudaStream_t);
        static const planar_func_t planar_funcs[5]
            = {gamma_contrast_tensor_planar_scalar<uchar>, 0 /*schar*/, gamma_contrast_tensor_planar_scalar<ushort>,
               gamma_contrast_tensor_planar_scalar<short>, gamma_contrast_tensor_planar_scalar<int>};
        if (data_type == kCV_32F)
        {
            gamma_contrast_tensor_planar_scalar_float<float>(inData, outData, gamma, gain, channels, stream);
        }
        else
        {
            NVCV_ASSERT(planar_funcs[data_type] != nullptr);
            planar_funcs[data_type](inData, outData, gamma, gain, channels, roundMode, stream);
        }
        return ErrorCode::SUCCESS;
    }

    typedef void (*func_t)(const TensorDataStridedCuda &, const TensorDataStridedCuda &, float, float, NVCVRoundMode,
                           cudaStream_t);

    static const func_t funcs[5][4] = {
        { gamma_contrast_tensor_scalar<uchar>,  gamma_contrast_tensor_scalar<uchar2>,
         gamma_contrast_tensor_scalar<uchar3>,gamma_contrast_tensor_scalar<uchar4>                                                                                                                         },
        {                                   0,                                     0,                                  0,                                     0},
        {gamma_contrast_tensor_scalar<ushort>, gamma_contrast_tensor_scalar<ushort2>,
         gamma_contrast_tensor_scalar<ushort3>, gamma_contrast_tensor_scalar<ushort4>                                                                          },
        { gamma_contrast_tensor_scalar<short>,  gamma_contrast_tensor_scalar<short2>,
         gamma_contrast_tensor_scalar<short3>,  gamma_contrast_tensor_scalar<short4>                                                                           },
        {   gamma_contrast_tensor_scalar<int>,    gamma_contrast_tensor_scalar<int2>, gamma_contrast_tensor_scalar<int3>,
         gamma_contrast_tensor_scalar<int4>                                                                                                                    },
    };
    typedef void (*float_func_t)(const TensorDataStridedCuda &, const TensorDataStridedCuda &, float, float,
                                 cudaStream_t);
    static const float_func_t funcs_float[4]
        = {gamma_contrast_tensor_scalar_float<float>, gamma_contrast_tensor_scalar_float<float2>,
           gamma_contrast_tensor_scalar_float<float3>, gamma_contrast_tensor_scalar_float<float4>};

    if (data_type == kCV_32F)
    {
        funcs_float[channels - 1](inData, outData, gamma, gain, stream);
    }
    else
    {
        NVCV_ASSERT(funcs[data_type][channels - 1] != nullptr);
        funcs[data_type][channels - 1](inData, outData, gamma, gain, roundMode, stream);
    }

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

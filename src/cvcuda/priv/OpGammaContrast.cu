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

#include "CudaDeviceUtils.hpp"
#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpGammaContrast.hpp"
#include "TensorAccessValidation.hpp"

#include "ElementStreamCommon.cuh"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <cstdint>

namespace cvcuda::priv {

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

using uchar  = unsigned char;
using ushort = unsigned short;

// Thread-block geometry of every image kernel: 32 x 8 threads, gridDim.z over samples/images.
constexpr int BLOCK = 32;

// The two gamma-staging paths launched their copy kernels with different block sizes -- the tensor
// path's shared helper used 256, the var-shape path 32. Both are carried so neither launch changes.
constexpr int kTensorGammaBlock   = 256;
constexpr int kVarShapeGammaBlock = 32;

using legacy_format::ClassifyDataFormat;
using legacy_format::ClassifyDataType;
using legacy_format::DataFormatCode;
using legacy_format::DataTypeCode;

// GammaContrast's rejection messages print layout codes NHWC=0, HWC=1, NCHW=2, CHW=3 (not DataFormatCode's);
// do not reorder.
inline int PrintedFormat(DataFormatCode format)
{
    constexpr int kPrinted[] = {2, 0, 3, 1}; // indexed by DataFormatCode: NCHW, NHWC, CHW, HWC
    return kPrinted[static_cast<int>(format)];
}

inline bool IsSupportedDataType(DataTypeCode dataType)
{
    return dataType == DataTypeCode::kU8 || dataType == DataTypeCode::kU16 || dataType == DataTypeCode::kS16
        || dataType == DataTypeCode::kS32 || dataType == DataTypeCode::kF32 || dataType == DataTypeCode::kF16;
}

// ---------------------------------------------------------------------------------------------
// Gamma-value staging. Both paths normalize the user's gamma tensor (which may be per-sample or
// per-sample-per-channel, dense or strided) into a dense [numImages*channels] float array consumed
// identically by their kernels -- so the two paths stay bit-exact.
// ---------------------------------------------------------------------------------------------

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

inline bool IsTensorDense(const nvcv::TensorDataStridedCuda &tensor)
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

inline bool GetPerImageGammaStride(const nvcv::TensorDataStridedCuda &gammas, int numImages, int64_t &sampleStride)
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

inline bool GetPerChannelGammaStrides(const nvcv::TensorDataStridedCuda &gammas, int numImages, int channelCount,
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

// Validate the gamma tensor length and expand it into the dense [numImages*channels] gammaArray.
inline int ExpandGamma(const nvcv::TensorDataStridedCuda &gammas, int numImages, int channels, float *gammaArray,
                       int gammaBlock, cudaStream_t stream)
{
    // Every path below reads basePtr() as float: the dense branch copies sizeof(float) per element
    // and the strided branches reinterpret each element as a float. A different dtype would read the
    // wrong width or the wrong bits, so reject it here rather than downstream.
    if (gammas.dtype() != nvcv::TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Gamma tensor must be F32");
    }

    int numElements = 1;
    for (int i = 0; i < gammas.rank(); i++)
    {
        numElements *= gammas.shape(i);
    }

    if (numElements != numImages && numElements != numImages * channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid gamma tensor length %d, expected %d or %d",
                              numElements, numImages, numImages * channels);
    }

    if (numImages * channels == numElements)
    {
        if (IsTensorDense(gammas))
        {
            NVCV_CHECK_THROW(cudaMemcpyAsync(gammaArray, gammas.basePtr(), sizeof(float) * numImages * channels,
                                             cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            int64_t sampleStride;
            int64_t channelStride;
            if (!GetPerChannelGammaStrides(gammas, numImages, channels, sampleStride, channelStride))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Per-channel gamma tensor must be dense or shaped as per-image channel slices");
            }

            copyPerChannelGammaValues<<<util::DivUp(numImages * channels, gammaBlock), gammaBlock, 0, stream>>>(
                gammaArray, reinterpret_cast<const uint8_t *>(gammas.basePtr()), sampleStride, channelStride, numImages,
                channels);
            NVCV_CHECK_THROW(cudaGetLastError());
        }
    }
    else
    {
        int64_t sampleStride;
        if (!GetPerImageGammaStride(gammas, numImages, sampleStride))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Per-image gamma tensor must be dense or have a sample dimension matching the "
                                  "input batch");
        }

        copyGammaValues<<<util::DivUp(numImages, gammaBlock), gammaBlock, 0, stream>>>(
            gammaArray, reinterpret_cast<const uint8_t *>(gammas.basePtr()), sampleStride, numImages, channels);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    return numElements;
}

// ---------------------------------------------------------------------------------------------
// Tensor kernels
// ---------------------------------------------------------------------------------------------

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

    dst[int3{x, y, z}] = nvcv::cuda::SaturateCast<D>(
        cuda::clamp(cuda::pow(cuda::StaticCast<float>(src[int3{x, y, z}]), gamma), 0.f, 1.f));
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

        dst[int4{x, y, plane, z}] = nvcv::cuda::SaturateCast<D>(
            cuda::clamp(cuda::pow(cuda::StaticCast<float>(src[int4{x, y, plane, z}]), gamma), 0.f, 1.f));
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

    dst[int3{x, y, z}] = nvcv::cuda::SaturateCast<D>(
        cuda::clamp(cuda::pow(cuda::StaticCast<float>(src[int3{x, y, z}]), g) * gain, 0.f, 1.f));
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
        dst[int4{x, y, plane, z}] = nvcv::cuda::SaturateCast<D>(
            cuda::clamp(cuda::pow(cuda::StaticCast<float>(src[int4{x, y, plane, z}]), gamma) * gain, 0.f, 1.f));
    }
}

// Exhaustive U8 tabulation preserves bit-exactness only if Entry matches the per-element expression.

template<int kNCH, bool kPlanar, bool kTruncate>
struct GammaLutTable
{
    static constexpr int kTables = kNCH;

    struct Param
    {
        const float *gammaArray;
        float        scalarGamma;
        float        gain;
        int          channels;
    };

    inline static __device__ unsigned char Entry(const Param &p, int sample, int plane, int table, int value)
    {
        // gammaArray stays channel-strided even when uniform gamma collapses kNCH to 1.
        const int   c   = kPlanar ? plane : table;
        const float g   = p.gammaArray ? p.gammaArray[sample * p.channels + c] : p.scalarGamma;
        const float tmp = (value + 0.0f) / 255.0f;
        return gamma_contrast_saturate_cast<kTruncate, uchar>(cuda::pow(tmp, g) * p.gain * 255.0f);
    }
};

// Cap blocks by device capacity to amortize each block's LUT pow evaluations without starving small batches.
constexpr int kGammaLutGroups      = 1;
constexpr int kGammaLutBlocksPerSM = 32;

template<int C, bool kPlanar, bool kTruncate, bool kUniform>
inline bool gamma_contrast_tensor_u8_lut(const nvcv::TensorDataStridedCuda &inData,
                                         const nvcv::TensorDataStridedCuda &outData, const float *gammaArray,
                                         float scalarGamma, float gain, cudaStream_t stream)
{
    constexpr int kNCH = kUniform ? 1 : C;
    using Table        = GammaLutTable<kNCH, kPlanar, kTruncate>;
    using Carrier      = elem_stream::LutCarrier<Table>;
    using ValueT       = std::conditional_t<kPlanar, uchar, cuda::MakeType<uchar, C>>;

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!srcAccess || !dstAccess)
    {
        return false;
    }

    const typename Table::Param param{gammaArray, scalarGamma, gain, C};

    const int64_t regions    = std::max<int64_t>(static_cast<int64_t>(srcAccess->numSamples()) * (kPlanar ? C : 1), 1);
    const int     maxBlocksX = static_cast<int>(elem_stream::DeviceBlockBudget(kGammaLutBlocksPerSM, regions));

    return elem_stream::TryRun<kGammaLutGroups, Carrier, kPlanar, ValueT>(stream, inData, outData, *srcAccess,
                                                                          *dstAccess, param, maxBlocksX);
}

template<bool kTruncate>
inline bool gamma_contrast_tensor_u8_lut_dispatch(const nvcv::TensorDataStridedCuda &inData,
                                                  const nvcv::TensorDataStridedCuda &outData, const float *gammaArray,
                                                  float scalarGamma, float gain, int channels, bool isPlanar,
                                                  bool uniformChannels, cudaStream_t stream)
{
#define GAMMA_CONTRAST_LUT_CASE(C, PLANAR)                                                                    \
    (uniformChannels ? gamma_contrast_tensor_u8_lut<C, PLANAR, kTruncate, true>(inData, outData, gammaArray,  \
                                                                                scalarGamma, gain, stream)    \
                     : gamma_contrast_tensor_u8_lut<C, PLANAR, kTruncate, false>(inData, outData, gammaArray, \
                                                                                 scalarGamma, gain, stream))

    if (isPlanar)
    {
        // Planar regions need one table selected by plane, even with per-channel gamma.
        switch (channels)
        {
        case 1:
            return gamma_contrast_tensor_u8_lut<1, true, kTruncate, true>(inData, outData, gammaArray, scalarGamma,
                                                                          gain, stream);
        case 3:
            return gamma_contrast_tensor_u8_lut<3, true, kTruncate, true>(inData, outData, gammaArray, scalarGamma,
                                                                          gain, stream);
        case 4:
            return gamma_contrast_tensor_u8_lut<4, true, kTruncate, true>(inData, outData, gammaArray, scalarGamma,
                                                                          gain, stream);
        default:
            return false;
        }
    }

    switch (channels)
    {
    case 1:
        return gamma_contrast_tensor_u8_lut<1, false, kTruncate, true>(inData, outData, gammaArray, scalarGamma, gain,
                                                                       stream);
    case 2:
        return GAMMA_CONTRAST_LUT_CASE(2, false);
    case 3:
        return GAMMA_CONTRAST_LUT_CASE(3, false);
    case 4:
        return GAMMA_CONTRAST_LUT_CASE(4, false);
    default:
        return false;
    }
#undef GAMMA_CONTRAST_LUT_CASE
}

// Interleaved launcher (T = vector pixel type for the channel count).
template<typename T>
void gamma_contrast_tensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                           float *gammaValues, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(size.x, BLOCK), util::DivUp(size.y, BLOCK / 4), batch);

    auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_tensor_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src, dst, gamma, size);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast_tensor_u8_batched(const nvcv::TensorDataStridedCuda &inData,
                                      const nvcv::TensorDataStridedCuda &outData, float *gammaValues,
                                      cudaStream_t stream)
{
    constexpr int NIX = 2;

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(size.x, BLOCK * NIX), util::DivUp(size.y, BLOCK / 4), batch);

    auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_tensor_u8_batched_kernel<T, gamma_type, decltype(src), decltype(dst), NIX>
        <<<grid, block, 0, stream>>>(src, dst, gamma, size);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast_tensor_float(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                 float *gammaValues, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(size.x, BLOCK), util::DivUp(size.y, BLOCK / 4), batch);

    auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_tensor_float_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src, dst, gamma, size);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// Planar launcher (T = scalar base type, one plane per channel).
template<typename T>
void gamma_contrast_tensor_planar(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                  float *gammaValues, int channels, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(size.x, BLOCK), util::DivUp(size.y, BLOCK / 4), batch);

    auto src = cuda::Tensor4DWrap<const T, int32_t>(inData.basePtr(), static_cast<int32_t>(srcAccess->sampleStride()),
                                                    static_cast<int32_t>(srcAccess->planeStride()),
                                                    static_cast<int32_t>(srcAccess->rowStride()));
    auto dst = cuda::Tensor4DWrap<T, int32_t>(outData.basePtr(), static_cast<int32_t>(dstAccess->sampleStride()),
                                              static_cast<int32_t>(dstAccess->planeStride()),
                                              static_cast<int32_t>(dstAccess->rowStride()));

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_tensor_planar_kernel<T><<<grid, block, 0, stream>>>(src, dst, gamma, size, channels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void gamma_contrast_tensor_planar_u8(const nvcv::TensorDataStridedCuda &inData,
                                            const nvcv::TensorDataStridedCuda &outData, float *gammaValues,
                                            int channels, cudaStream_t stream)
{
    constexpr int NIX = 8;

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(size.x, BLOCK * NIX), util::DivUp(size.y, BLOCK / 4), batch);

    auto src = cuda::Tensor4DWrap<const uchar, int32_t>(
        inData.basePtr(), static_cast<int32_t>(srcAccess->sampleStride()),
        static_cast<int32_t>(srcAccess->planeStride()), static_cast<int32_t>(srcAccess->rowStride()));
    auto dst = cuda::Tensor4DWrap<uchar, int32_t>(outData.basePtr(), static_cast<int32_t>(dstAccess->sampleStride()),
                                                  static_cast<int32_t>(dstAccess->planeStride()),
                                                  static_cast<int32_t>(dstAccess->rowStride()));

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_tensor_planar_u8_kernel<<<grid, block, 0, stream>>>(src, dst, gamma, size, channels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast_tensor_planar_float(const nvcv::TensorDataStridedCuda &inData,
                                        const nvcv::TensorDataStridedCuda &outData, float *gammaValues, int channels,
                                        cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(size.x, BLOCK), util::DivUp(size.y, BLOCK / 4), batch);

    auto src = cuda::Tensor4DWrap<const T, int32_t>(inData.basePtr(), static_cast<int32_t>(srcAccess->sampleStride()),
                                                    static_cast<int32_t>(srcAccess->planeStride()),
                                                    static_cast<int32_t>(srcAccess->rowStride()));
    auto dst = cuda::Tensor4DWrap<T, int32_t>(outData.basePtr(), static_cast<int32_t>(dstAccess->sampleStride()),
                                              static_cast<int32_t>(dstAccess->planeStride()),
                                              static_cast<int32_t>(dstAccess->rowStride()));

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_tensor_planar_float_kernel<T><<<grid, block, 0, stream>>>(src, dst, gamma, size, channels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// Scalar (host-float) launchers -- same grids/wrappers as the device-tensor launchers above, but the
// gamma/gain scalars are passed straight into the kernel launch (no gamma array).
template<typename T>
void gamma_contrast_tensor_scalar(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                  float gamma, float gain, NVCVRoundMode roundMode, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(size.x, BLOCK), util::DivUp(size.y, BLOCK / 4), batch);

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
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast_tensor_scalar_float(const nvcv::TensorDataStridedCuda &inData,
                                        const nvcv::TensorDataStridedCuda &outData, float gamma, float gain,
                                        cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(size.x, BLOCK), util::DivUp(size.y, BLOCK / 4), batch);

    auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    gamma_contrast_tensor_scalar_float_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src, dst, gamma, gain, size);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast_tensor_planar_scalar(const nvcv::TensorDataStridedCuda &inData,
                                         const nvcv::TensorDataStridedCuda &outData, float gamma, float gain,
                                         int channels, NVCVRoundMode roundMode, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(size.x, BLOCK), util::DivUp(size.y, BLOCK / 4), batch);

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
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast_tensor_planar_scalar_float(const nvcv::TensorDataStridedCuda &inData,
                                               const nvcv::TensorDataStridedCuda &outData, float gamma, float gain,
                                               int channels, cudaStream_t stream)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    int2 size{static_cast<int>(srcAccess->numCols()), static_cast<int>(srcAccess->numRows())};
    int  batch = srcAccess->numSamples();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(size.x, BLOCK), util::DivUp(size.y, BLOCK / 4), batch);

    auto src = cuda::Tensor4DWrap<const T, int32_t>(inData.basePtr(), static_cast<int32_t>(srcAccess->sampleStride()),
                                                    static_cast<int32_t>(srcAccess->planeStride()),
                                                    static_cast<int32_t>(srcAccess->rowStride()));
    auto dst = cuda::Tensor4DWrap<T, int32_t>(outData.basePtr(), static_cast<int32_t>(dstAccess->sampleStride()),
                                              static_cast<int32_t>(dstAccess->planeStride()),
                                              static_cast<int32_t>(dstAccess->rowStride()));

    gamma_contrast_tensor_planar_scalar_float_kernel<T>
        <<<grid, block, 0, stream>>>(src, dst, gamma, gain, size, channels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// Var-shape kernels
// ---------------------------------------------------------------------------------------------

// apply 255*((x/255)**gamma) on each pixel
template<typename D, typename gamma_type>
__global__ void gamma_contrast_kernel(const cuda::ImageBatchVarShapeWrap<D> src, cuda::ImageBatchVarShapeWrap<D> dst,
                                      const cuda::Tensor1DWrap<gamma_type> gamma_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
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
    const int batch_idx = blockIdx.z;
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
    const int batch_idx = blockIdx.z;
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
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    gamma_type gamma = gamma_[batch_idx];

    *dst.ptr(batch_idx, dst_y, dst_x) = nvcv::cuda::SaturateCast<D>(
        cuda::clamp(cuda::pow(cuda::StaticCast<float>(*src.ptr(batch_idx, dst_y, dst_x)), gamma), 0.f, 1.f));
}

template<typename D, typename gamma_type, int NIX>
__global__ void gamma_contrast_float_batched_kernel(const cuda::ImageBatchVarShapeWrap<D> src,
                                                    cuda::ImageBatchVarShapeWrap<D>       dst,
                                                    const cuda::Tensor1DWrap<gamma_type>  gamma_)
{
    const int dst_x     = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
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
            dst_ptr[i] = nvcv::cuda::SaturateCast<D>(
                cuda::clamp(cuda::pow(cuda::StaticCast<float>(src_ptr[i]), gamma), 0.f, 1.f));
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
    const int batch_idx = blockIdx.z;
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
    const int batch_idx = blockIdx.z;
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
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        const float gamma = gamma_[batch_idx * channels + plane];

        *dst.ptr(batch_idx, plane, dst_y, dst_x) = nvcv::cuda::SaturateCast<D>(
            cuda::clamp(cuda::pow(cuda::StaticCast<float>(*src.ptr(batch_idx, plane, dst_y, dst_x)), gamma), 0.f, 1.f));
    }
}

template<typename T>
void gamma_contrast_planar(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                           const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *gammaValues, int channels,
                           cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                            block(BLOCK, BLOCK / 4, 1);
    dim3                            grid(util::DivUp(max_width, BLOCK), util::DivUp(max_height, BLOCK / 4), batch);
    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_planar_kernel<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma, channels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void gamma_contrast_planar_u8(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                                     const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *gammaValues,
                                     int channels, cudaStream_t stream)
{
    constexpr int NIX = 16;

    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(max_width, BLOCK * NIX), util::DivUp(max_height, BLOCK / 4), batch);

    cuda::ImageBatchVarShapeWrap<uchar> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<uchar> dst_ptr(out);

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_planar_u8_kernel<<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma, channels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast_float_planar(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                                 const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *gammaValues, int channels,
                                 cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                            block(BLOCK, BLOCK / 4, 1);
    dim3                            grid(util::DivUp(max_width, BLOCK), util::DivUp(max_height, BLOCK / 4), batch);
    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_float_planar_kernel<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma, channels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                    const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *gammaValues, cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                            block(BLOCK, BLOCK / 4, 1);
    dim3                            grid(util::DivUp(max_width, BLOCK), util::DivUp(max_height, BLOCK / 4), batch);
    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void gamma_contrast_u8(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                              const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *gammaValues,
                              cudaStream_t stream)
{
    constexpr int NIX = 16;

    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(max_width, BLOCK * NIX), util::DivUp(max_height, BLOCK / 4), batch);

    cuda::ImageBatchVarShapeWrap<uchar> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<uchar> dst_ptr(out);

    cuda::Tensor1DWrap<float> gamma(gammaValues);
    gamma_contrast_u8_kernel<<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast_u8_batched(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                               const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *gammaValues,
                               cudaStream_t stream)
{
    constexpr int NIX = sizeof(T) == sizeof(uchar4) ? 3 : 2;

    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(max_width, BLOCK * NIX), util::DivUp(max_height, BLOCK / 4), batch);

    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_u8_batched_kernel<T, gamma_type, NIX><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast_float(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                          const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *gammaValues, cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                            block(BLOCK, BLOCK / 4, 1);
    dim3                            grid(util::DivUp(max_width, BLOCK), util::DivUp(max_height, BLOCK / 4), batch);
    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_float_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void gamma_contrast_float_batched(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                                  const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *gammaValues,
                                  cudaStream_t stream)
{
    constexpr int NIX = 2;

    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(util::DivUp(max_width, BLOCK * NIX), util::DivUp(max_height, BLOCK / 4), batch);

    cuda::ImageBatchVarShapeWrap<T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(out);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_float_batched_kernel<T, gamma_type, NIX><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------------------------

struct TensorInfo
{
    DataTypeCode dataType;
    bool         isPlanar;
    int          numSamples;
    int          channels;
};

inline TensorInfo ValidateTensorPair(const nvcv::TensorDataStridedCuda &inData,
                                     const nvcv::TensorDataStridedCuda &outData)
{
    DataFormatCode inputFormat  = ClassifyDataFormat(inData.layout());
    DataFormatCode outputFormat = ClassifyDataFormat(outData.layout());
    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", PrintedFormat(inputFormat),
                              PrintedFormat(outputFormat));
    }

    const bool isPlanar = legacy_format::IsPlanar(inputFormat);

    DataTypeCode dataType    = ClassifyDataType(inData.dtype());
    DataTypeCode outDataType = ClassifyDataType(outData.dtype());
    if (dataType != outDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input DataType %d must match output DataType %d",
                              static_cast<int>(dataType), static_cast<int>(outDataType));
    }
    if (!IsSupportedDataType(dataType))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", static_cast<int>(dataType));
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!srcAccess || !dstAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Failed to create tensor access");
    }

    if (srcAccess->numSamples() != dstAccess->numSamples() || srcAccess->numRows() != dstAccess->numRows()
        || srcAccess->numCols() != dstAccess->numCols())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching sample count, width, and height");
    }

    const int numSamples = srcAccess->numSamples();
    const int channels   = srcAccess->numChannels();
    if (channels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output channel counts must match");
    }
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    return {dataType, isPlanar, numSamples, channels};
}

inline void RunTensorGamma(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                           const TensorInfo &info, float *gammaArray, bool uniformChannels, cudaStream_t stream)
{
    bool useLut = true;
    if (info.dataType == DataTypeCode::kU8 && info.isPlanar && info.channels == 3)
    {
        int sm = 0;
        NVCV_CHECK_THROW(GetCurrentDeviceSM(sm));
        if (sm == 89)
        {
            const auto fitsLegacy = [](const nvcv::TensorDataStridedCuda &data)
            {
                auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(data);
                if (access->numRows() > elem_stream::kMaxGridYZ * (BLOCK / 4))
                    return false;
                int64_t remaining = cuda::TypeTraits<int32_t>::max;
                for (int dimension = 0; dimension < data.rank(); ++dimension)
                {
                    const int64_t steps  = data.shape(dimension) - 1;
                    const int64_t stride = data.stride(dimension);
                    if (steps > 0)
                    {
                        if (stride < 0 || stride > remaining / steps)
                            return false;
                        remaining -= steps * stride;
                    }
                }
                return true;
            };
            useLut = !fitsLegacy(inData) || !fitsLegacy(outData);
        }
    }
    // Nearest rounding and the exact gain=1 identity preserve SaturateCast(pow(tmp, gamma) * 255.0f).
    if (info.dataType == DataTypeCode::kU8 && useLut
        && gamma_contrast_tensor_u8_lut_dispatch<false>(inData, outData, gammaArray, 0.0f, 1.0f, info.channels,
                                                        info.isPlanar, uniformChannels, stream))
    {
        return;
    }

    if (info.isPlanar)
    {
        typedef void (*planar_func_t)(const nvcv::TensorDataStridedCuda &, const nvcv::TensorDataStridedCuda &, float *,
                                      int, cudaStream_t);
        static const planar_func_t planar_funcs[5]
            = {gamma_contrast_tensor_planar_u8, 0 /*schar*/, gamma_contrast_tensor_planar<ushort>,
               gamma_contrast_tensor_planar<short>, gamma_contrast_tensor_planar<int>};
        if (info.dataType == DataTypeCode::kF32)
        {
            gamma_contrast_tensor_planar_float<float>(inData, outData, gammaArray, info.channels, stream);
        }
        // F16 follows the float path (pow with no /255 normalization, clamp to [0, 1]) with real
        // __half kernels; the gamma values remain float.
        else if (info.dataType == DataTypeCode::kF16)
        {
            gamma_contrast_tensor_planar_float<__half>(inData, outData, gammaArray, info.channels, stream);
        }
        else
        {
            NVCV_ASSERT(planar_funcs[static_cast<int>(info.dataType)] != nullptr);
            planar_funcs[static_cast<int>(info.dataType)](inData, outData, gammaArray, info.channels, stream);
        }
        return;
    }

    typedef void (*func_t)(const nvcv::TensorDataStridedCuda &, const nvcv::TensorDataStridedCuda &, float *,
                           cudaStream_t);

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
    // F16 instantiates the float-path kernels (pow with no /255 normalization, clamp to [0, 1])
    // with real __half image types; the gamma values remain float.
    static const func_t funcs_half[4] = {gamma_contrast_tensor_float<__half>, gamma_contrast_tensor_float<__half2>,
                                         gamma_contrast_tensor_float<half3>, gamma_contrast_tensor_float<half4>};

    if (info.dataType == DataTypeCode::kF32)
    {
        funcs_float[info.channels - 1](inData, outData, gammaArray, stream);
    }
    else if (info.dataType == DataTypeCode::kF16)
    {
        funcs_half[info.channels - 1](inData, outData, gammaArray, stream);
    }
    else
    {
        NVCV_ASSERT(funcs[static_cast<int>(info.dataType)][info.channels - 1] != nullptr);
        funcs[static_cast<int>(info.dataType)][info.channels - 1](inData, outData, gammaArray, stream);
    }
}

inline void RunTensorScalarGamma(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                 const TensorInfo &info, float gamma, float gain, NVCVRoundMode roundMode,
                                 cudaStream_t stream)
{
    if (info.dataType == DataTypeCode::kU8)
    {
        const bool ok = (roundMode == NVCV_ROUND_TRUNCATE)
                          ? gamma_contrast_tensor_u8_lut_dispatch<true>(inData, outData, nullptr, gamma, gain,
                                                                        info.channels, info.isPlanar, true, stream)
                          : gamma_contrast_tensor_u8_lut_dispatch<false>(inData, outData, nullptr, gamma, gain,
                                                                         info.channels, info.isPlanar, true, stream);
        if (ok)
        {
            return;
        }
    }

    if (info.isPlanar)
    {
        typedef void (*planar_func_t)(const nvcv::TensorDataStridedCuda &, const nvcv::TensorDataStridedCuda &, float,
                                      float, int, NVCVRoundMode, cudaStream_t);
        static const planar_func_t planar_funcs[5]
            = {gamma_contrast_tensor_planar_scalar<uchar>, 0 /*schar*/, gamma_contrast_tensor_planar_scalar<ushort>,
               gamma_contrast_tensor_planar_scalar<short>, gamma_contrast_tensor_planar_scalar<int>};
        if (info.dataType == DataTypeCode::kF32)
        {
            gamma_contrast_tensor_planar_scalar_float<float>(inData, outData, gamma, gain, info.channels, stream);
        }
        // F16 follows the float path (no /255 normalization, no round mode, clamp to [0, 1]).
        else if (info.dataType == DataTypeCode::kF16)
        {
            gamma_contrast_tensor_planar_scalar_float<__half>(inData, outData, gamma, gain, info.channels, stream);
        }
        else
        {
            NVCV_ASSERT(planar_funcs[static_cast<int>(info.dataType)] != nullptr);
            planar_funcs[static_cast<int>(info.dataType)](inData, outData, gamma, gain, info.channels, roundMode,
                                                          stream);
        }
        return;
    }

    typedef void (*func_t)(const nvcv::TensorDataStridedCuda &, const nvcv::TensorDataStridedCuda &, float, float,
                           NVCVRoundMode, cudaStream_t);

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
    typedef void (*float_func_t)(const nvcv::TensorDataStridedCuda &, const nvcv::TensorDataStridedCuda &, float, float,
                                 cudaStream_t);
    static const float_func_t funcs_float[4]
        = {gamma_contrast_tensor_scalar_float<float>, gamma_contrast_tensor_scalar_float<float2>,
           gamma_contrast_tensor_scalar_float<float3>, gamma_contrast_tensor_scalar_float<float4>};
    // F16 instantiates the float-path scalar kernels with real __half image types.
    static const float_func_t funcs_half[4]
        = {gamma_contrast_tensor_scalar_float<__half>, gamma_contrast_tensor_scalar_float<__half2>,
           gamma_contrast_tensor_scalar_float<half3>, gamma_contrast_tensor_scalar_float<half4>};

    if (info.dataType == DataTypeCode::kF32)
    {
        funcs_float[info.channels - 1](inData, outData, gamma, gain, stream);
    }
    else if (info.dataType == DataTypeCode::kF16)
    {
        funcs_half[info.channels - 1](inData, outData, gamma, gain, stream);
    }
    else
    {
        NVCV_ASSERT(funcs[static_cast<int>(info.dataType)][info.channels - 1] != nullptr);
        funcs[static_cast<int>(info.dataType)][info.channels - 1](inData, outData, gamma, gain, roundMode, stream);
    }
}

} // namespace

GammaContrastScratch::GammaContrastScratch(int32_t maxBatchSize, int32_t maxChannelCount)
{
    if (maxBatchSize > 0 && maxChannelCount > 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(&m_gammaArray, maxBatchSize * maxChannelCount * sizeof(float)));
    }
}

GammaContrastScratch::~GammaContrastScratch()
{
    NVCV_CHECK_LOG(cudaFree(m_gammaArray));
}

GammaContrast::GammaContrast(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount)
    // The gamma scratch is single-device by design: it is allocated once, sized from the
    // constructor's limits, and freed with the operator. PerDeviceResource creates one buffer per
    // CUDA device for transparent multi-GPU support. The tensor and var-shape paths keep separate
    // buffers, exactly as the two legacy operator instances did.
    : m_maxBatchSize(maxVarShapeBatchSize)
    , m_maxChannelCount(maxVarShapeChannelCount)
    , m_gammaScratch([maxVarShapeBatchSize, maxVarShapeChannelCount](int)
                     { return std::make_unique<GammaContrastScratch>(maxVarShapeBatchSize, maxVarShapeChannelCount); })
    , m_gammaScratchVarShape(
          [maxVarShapeBatchSize, maxVarShapeChannelCount](int)
          { return std::make_unique<GammaContrastScratch>(maxVarShapeBatchSize, maxVarShapeChannelCount); })
{
}

void GammaContrast::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                               const nvcv::Tensor &gamma) const
{
    CVCUDA_NVTX_RANGE("cvcuda::GammaContrast::operator()[Tensor]");

    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be device-acessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-acessible, pitch-linear tensor");
    }

    auto gammaData = gamma.exportData<nvcv::TensorDataStridedCuda>();
    if (gammaData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Gamma must be device-acessible, pitch-linear tensor");
    }

    float *gammaArray = m_gammaScratch.get().data();

    TensorInfo info = ValidateTensorPair(*inData, *outData);

    if (m_maxBatchSize <= 0 || info.numSamples > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid maximum batch size");
    }
    if (m_maxChannelCount <= 0 || info.channels > m_maxChannelCount)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid maximum channel count");
    }

    // The planar path launches grid.z over samples; enforce CUDA's 65535 grid-z limit.
    if (static_cast<int64_t>(info.numSamples) > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "GammaContrast tensor requires numSamples <= 65535 (CUDA grid-z limit)");
    }

    const int gammaElements
        = ExpandGamma(*gammaData, info.numSamples, info.channels, gammaArray, kTensorGammaBlock, stream);

    RunTensorGamma(*inData, *outData, info, gammaArray, gammaElements == info.numSamples, stream);
}

// Scalar (host-float) gamma/gain path. The scalars are baked into the kernel launch, so there is no
// gamma tensor, ExpandGamma copy, or gamma-scratch/max-batch/max-channel capacity bound.
void GammaContrast::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float gamma,
                               float gain, NVCVRoundMode roundMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::GammaContrast::operator()[Tensor,Scalar]");

    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be device-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-accessible, pitch-linear tensor");
    }

    if (roundMode != NVCV_ROUND_NEAREST && roundMode != NVCV_ROUND_TRUNCATE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid round mode %d",
                              static_cast<int>(roundMode));
    }

    TensorInfo info = ValidateTensorPair(*inData, *outData);

    // The planar path launches grid.z over samples; enforce CUDA's 65535 grid-z limit.
    if (static_cast<int64_t>(info.numSamples) > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "GammaContrast tensor requires numSamples <= 65535 (CUDA grid-z limit)");
    }

    RunTensorScalarGamma(*inData, *outData, info, gamma, gain, roundMode, stream);
}

void GammaContrast::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                               const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &gamma) const
{
    CVCUDA_NVTX_RANGE("cvcuda::GammaContrast::operator()[ImageBatchVarShape]");

    if (in.numImages() != out.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same number of images");
    }

    for (int i = 0; i < in.numImages(); ++i)
    {
        if (in[i].size() != out[i].size())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output images must have matching width and height");
        }
    }

    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be device-acessible, varshape pitch-linear image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-acessible, varshape pitch-linear image batch");
    }

    auto gammaData = gamma.exportData<nvcv::TensorDataStridedCuda>();
    if (gammaData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Gamma must be device-acessible, pitch-linear tensor");
    }

    float *gammaArray = m_gammaScratchVarShape.get().data();

    RequireUniqueFormat(*inData, "input");

    RequireUniqueFormat(*outData, "output");

    if (m_maxBatchSize <= 0 || inData->numImages() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid maximum batch size");
    }

    if (m_maxChannelCount <= 0 || inData->uniqueFormat().numChannels() > m_maxChannelCount)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid maximum channel count");
    }

    DataFormatCode input_format  = ClassifyDataFormat(*inData);
    DataFormatCode output_format = ClassifyDataFormat(*outData);
    if (input_format != output_format)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", PrintedFormat(input_format),
                              PrintedFormat(output_format));
    }

    DataFormatCode format = input_format;

    const bool isPlanar = legacy_format::IsPlanar(format);

    DataTypeCode data_type = ClassifyDataType(inData->uniqueFormat());

    if (!IsSupportedDataType(data_type))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", static_cast<int>(data_type));
    }

    DataTypeCode out_data_type = ClassifyDataType(outData->uniqueFormat());
    int          channels      = inData->uniqueFormat().numChannels();
    int          outChannels   = outData->uniqueFormat().numChannels();

    if (out_data_type != data_type || outChannels != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input DataType %d and channel count %d must match Output DataType %d and channel "
                              "count %d",
                              static_cast<int>(data_type), channels, static_cast<int>(out_data_type), outChannels);
    }

    // Planar 2-channel is rejected: there is no defined 2-plane planar format (matches Resize/Flip).
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    // The planar path launches grid.z over images and loops the channel planes inside the kernel, so
    // numImages must fit CUDA's 65535 grid-z limit.
    if (isPlanar && static_cast<int64_t>(inData->numImages()) > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar gamma contrast requires numImages <= 65535 (CUDA grid-z limit)");
    }

    int numImages = inData->numImages();

    ExpandGamma(*gammaData, numImages, channels, gammaArray, kVarShapeGammaBlock, stream);

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

    // F16 instantiates the float-path kernels (pow with no /255 normalization, clamp to [0, 1])
    // with real __half image types; the gamma values remain float.
    static const func_t funcs_half[4] = {gamma_contrast_float_batched<__half>, gamma_contrast_float<__half2>,
                                         gamma_contrast_float<half3>, gamma_contrast_float<half4>};

    if (isPlanar)
    {
        // Planar dispatch indexes by dtype only: each channel is a separate single-channel plane, so
        // one scalar specialization per dtype covers any channel count (the kernel loops the planes).
        typedef void (*planar_func_t)(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                                      const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *gammas, int channels,
                                      cudaStream_t stream);

        static const planar_func_t planar_funcs[5]
            = {gamma_contrast_planar_u8, 0 /*schar*/, gamma_contrast_planar<ushort>, gamma_contrast_planar<short>,
               gamma_contrast_planar<int>};

        if (data_type == DataTypeCode::kF32)
        {
            gamma_contrast_float_planar<float>(*inData, *outData, gammaArray, channels, stream);
        }
        // F16 follows the float path (pow with no /255 normalization, clamp to [0, 1]).
        else if (data_type == DataTypeCode::kF16)
        {
            gamma_contrast_float_planar<__half>(*inData, *outData, gammaArray, channels, stream);
        }
        else
        {
            const planar_func_t planar_func = planar_funcs[static_cast<int>(data_type)];
            NVCV_ASSERT(planar_func != nullptr);
            planar_func(*inData, *outData, gammaArray, channels, stream);
        }

        return;
    }

    if (data_type == DataTypeCode::kF32)
    {
        const func_t func = funcs_float[channels - 1];
        func(*inData, *outData, gammaArray, stream);
    }
    else if (data_type == DataTypeCode::kF16)
    {
        const func_t func = funcs_half[channels - 1];
        func(*inData, *outData, gammaArray, stream);
    }
    else
    {
        const func_t func = funcs[static_cast<int>(data_type)][channels - 1];
        func(*inData, *outData, gammaArray, stream);
    }
}

} // namespace cvcuda::priv

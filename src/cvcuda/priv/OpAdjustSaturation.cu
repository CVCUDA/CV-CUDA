/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpAdjustSaturation.hpp"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// torchvision luminance coefficients. Intentionally NOT cvtcolor's 0.299 (see OpAdjustSaturation.h).
constexpr float kR2Y = 0.2989f;
constexpr float kG2Y = 0.587f;
constexpr float kB2Y = 0.114f;

static bool IsPlanarLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

// Clamp + store to the output element type. Integers truncate toward zero (matching torchvision's
// `.to(uint8)`): round<ZERO> drops the fractional part and SaturateCast clamps into the dtype range,
// the same sequence cvcuda's ConvertTo uses for NVCV_ROUND_TRUNCATE (priv/legacy/convert_to.cu).
// Floats clamp to [0, 1] with no rounding.
template<typename BT>
inline __host__ __device__ BT StoreSaturation(float v)
{
    if constexpr (std::is_floating_point_v<BT>)
    {
        return static_cast<BT>(fminf(fmaxf(v, 0.0f), 1.0f));
    }
    else
    {
        return cuda::SaturateCast<BT>(cuda::round<cuda::RoundMode::ZERO>(v));
    }
}

// Core per-pixel saturation blend, shared by the interleaved and planar paths (and mirrored by the
// CPU gold). 1-channel pixels are returned unchanged (torchvision returns the image as-is); a
// 3-channel pixel is blended toward its luminance by `ratio` with weight `oneMinus = 1 - ratio`.
template<typename T>
inline __host__ __device__ T AdjustSaturationPixel(T pixel, float ratio, float oneMinus)
{
    using BT               = cuda::BaseType<T>;
    constexpr int numChans = cuda::NumElements<T>;

    T out{};
    if constexpr (numChans == 1)
    {
        out = pixel;
    }
    else
    {
        static_assert(numChans == 3, "AdjustSaturation supports 1- or 3-channel pixels");
        const float r = static_cast<float>(cuda::GetElement(pixel, 0));
        const float g = static_cast<float>(cuda::GetElement(pixel, 1));
        const float b = static_cast<float>(cuda::GetElement(pixel, 2));

        float gray = (kR2Y * r + kG2Y * g) + kB2Y * b;
        if constexpr (!std::is_floating_point_v<BT>)
        {
            gray = floorf(gray); // torchvision floors the grayscale for integer dtypes before the blend
        }

        cuda::GetElement(out, 0) = StoreSaturation<BT>(ratio * r + oneMinus * gray);
        cuda::GetElement(out, 1) = StoreSaturation<BT>(ratio * g + oneMinus * gray);
        cuda::GetElement(out, 2) = StoreSaturation<BT>(ratio * b + oneMinus * gray);
    }
    return out;
}

// Interleaved ((N)HWC) kernels ----------------------------------------------------------

template<typename T, class SrcWrapper, class DstWrapper>
__global__ void AdjustSaturationInterleaved(SrcWrapper src, DstWrapper dst, int2 size, float ratio, float oneMinus)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }
    dst[coord] = AdjustSaturationPixel<T>(src[coord], ratio, oneMinus);
}

template<typename T, class SrcWrapper, class DstWrapper>
__global__ void AdjustSaturationInterleavedVarShape(SrcWrapper src, DstWrapper dst, float ratio, float oneMinus)
{
    const int  z = blockIdx.z;
    const int2 size{dst.width(z), dst.height(z)};
    int3       coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }
    dst[coord] = AdjustSaturationPixel<T>(src[coord], ratio, oneMinus);
}

// Planar ((N)CHW) kernels ---------------------------------------------------------------

template<typename BT, int numChannels, class SrcWrapper, class DstWrapper>
inline __device__ void DoAdjustSaturationPlanar(SrcWrapper src, DstWrapper dst, int2 size, int z, float ratio,
                                                float oneMinus)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }
    if constexpr (numChannels == 1)
    {
        *dst.ptr(z, 0, coord.y, coord.x) = *src.ptr(z, 0, coord.y, coord.x);
    }
    else
    {
        using Vec = cuda::MakeType<BT, 3>;
        Vec pixel;
        cuda::GetElement(pixel, 0) = *src.ptr(z, 0, coord.y, coord.x);
        cuda::GetElement(pixel, 1) = *src.ptr(z, 1, coord.y, coord.x);
        cuda::GetElement(pixel, 2) = *src.ptr(z, 2, coord.y, coord.x);

        const Vec out = AdjustSaturationPixel<Vec>(pixel, ratio, oneMinus);

        *dst.ptr(z, 0, coord.y, coord.x) = cuda::GetElement(out, 0);
        *dst.ptr(z, 1, coord.y, coord.x) = cuda::GetElement(out, 1);
        *dst.ptr(z, 2, coord.y, coord.x) = cuda::GetElement(out, 2);
    }
}

template<typename BT, int numChannels, class SrcWrapper, class DstWrapper>
__global__ void AdjustSaturationPlanar(SrcWrapper src, DstWrapper dst, int2 size, float ratio, float oneMinus)
{
    DoAdjustSaturationPlanar<BT, numChannels>(src, dst, size, blockIdx.z, ratio, oneMinus);
}

template<typename BT, int numChannels, class SrcWrapper, class DstWrapper>
__global__ void AdjustSaturationPlanarVarShape(SrcWrapper src, DstWrapper dst, float ratio, float oneMinus)
{
    const int  z = blockIdx.z;
    const int2 size{dst.width(z), dst.height(z)};
    DoAdjustSaturationPlanar<BT, numChannels>(src, dst, size, z, ratio, oneMinus);
}

// Launchers -----------------------------------------------------------------------------

constexpr int kGridZLimit = 65535;

inline void CheckBatchLimit(int numSamples)
{
    if (numSamples > kGridZLimit)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Batch size exceeds the CUDA grid.z limit of 65535");
    }
}

template<typename T, int numChannels>
inline void RunTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                      const nvcv::TensorDataStridedCuda &dstData, bool isPlanar, float ratio, float oneMinus)
{
    using BT = cuda::BaseType<T>;

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    const int2 size = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});
    CheckBatchLimit(static_cast<int>(srcAccess->numSamples()));

    const int64_t inMaxStride  = srcAccess->sampleStride() * srcAccess->numSamples();
    const int64_t outMaxStride = dstAccess->sampleStride() * dstAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }

    const dim3 block(32, 4, 1);
    const dim3 grid(util::DivUp(size.x, block.x), util::DivUp(size.y, block.y),
                    static_cast<unsigned int>(srcAccess->numSamples()));

    if (isPlanar)
    {
        auto src = cuda::CreateTensorWrapNCHW<const BT, int32_t>(srcData);
        auto dst = cuda::CreateTensorWrapNCHW<BT, int32_t>(dstData);
        AdjustSaturationPlanar<BT, numChannels><<<grid, block, 0, stream>>>(src, dst, size, ratio, oneMinus);
    }
    else
    {
        auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(srcData);
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(dstData);
        AdjustSaturationInterleaved<T><<<grid, block, 0, stream>>>(src, dst, size, ratio, oneMinus);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, int numChannels>
inline void RunVarShape(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                        const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, bool isPlanar, float ratio,
                        float oneMinus)
{
    using BT = cuda::BaseType<T>;

    CheckBatchLimit(dstData.numImages());

    const int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};
    const dim3 block(32, 4, 1);
    const dim3 grid(util::DivUp(dstMaxSize.x, block.x), util::DivUp(dstMaxSize.y, block.y),
                    static_cast<unsigned int>(dstMaxSize.z));

    if (isPlanar)
    {
        cuda::ImageBatchVarShapeWrap<const BT> src(srcData);
        cuda::ImageBatchVarShapeWrap<BT>       dst(dstData);
        AdjustSaturationPlanarVarShape<BT, numChannels><<<grid, block, 0, stream>>>(src, dst, ratio, oneMinus);
    }
    else
    {
        cuda::ImageBatchVarShapeWrap<const T> src(srcData);
        cuda::ImageBatchVarShapeWrap<T>       dst(dstData);
        AdjustSaturationInterleavedVarShape<T><<<grid, block, 0, stream>>>(src, dst, ratio, oneMinus);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

// Dispatch over base dtype (u8 / f32) and channel count (1 / 3) -------------------------

template<typename Cb>
inline void RunChannelSwitch(int numChannels, nvcv::DataType dtype, const Cb &cb)
{
    const bool isU8  = (dtype == nvcv::TYPE_U8 || dtype == nvcv::TYPE_3U8);
    const bool isF32 = (dtype == nvcv::TYPE_F32 || dtype == nvcv::TYPE_3F32);

    if (!isU8 && !isF32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: AdjustSaturation supports 8-bit unsigned and 32-bit float");
    }

    if (numChannels == 1)
    {
        if (isU8)
        {
            cb(uchar1{}, std::integral_constant<int, 1>{});
        }
        else
        {
            cb(float1{}, std::integral_constant<int, 1>{});
        }
    }
    else if (numChannels == 3)
    {
        if (isU8)
        {
            cb(uchar3{}, std::integral_constant<int, 3>{});
        }
        else
        {
            cb(float3{}, std::integral_constant<int, 3>{});
        }
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of channels: AdjustSaturation supports 1 or 3 channels");
    }
}

// Validation ----------------------------------------------------------------------------

inline void ValidateSaturation(double saturation)
{
    if (!(saturation >= 0.0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The saturation factor must be >= 0 (and not NaN)");
    }
}

inline bool ValidateSrcDstTensors(bool &isEmpty, int &numChannels, nvcv::DataType &dtype,
                                  const nvcv::Optional<nvcv::TensorDataStridedCuda> &srcData,
                                  const nvcv::Optional<nvcv::TensorDataStridedCuda> &dstData)
{
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }
    if (srcData->layout() != dstData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }

    const bool isPlanar = IsPlanarLayout(srcData->layout());
    if (srcData->layout() != nvcv::TENSOR_HWC && srcData->layout() != nvcv::TENSOR_NHWC && !isPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC or (N)CHW layout");
    }
    if (srcData->dtype() != dstData->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same data type");
    }
    if (srcData->dtype().numChannels() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensor data type must be scalar; use the C dimension for image channels");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    if (!srcAccess || !dstAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input/output must be accessible as strided images");
    }
    if (srcAccess->numSamples() != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    numChannels = srcAccess->numChannels();
    if (numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }
    if (numChannels != 1 && numChannels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have 1 or 3 channels");
    }
    if (srcAccess->numCols() != dstAccess->numCols() || srcAccess->numRows() != dstAccess->numRows())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching width and height");
    }

    isEmpty = srcAccess->numSamples() == 0 || srcAccess->numRows() == 0 || srcAccess->numCols() == 0;
    dtype   = srcData->dtype();
    return isPlanar;
}

inline void ValidateImagePlanes(const nvcv::Image &image, const nvcv::ImageFormat &format)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    if (!data || data->numPlanes() != format.numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image plane descriptors must match the image format");
    }

    constexpr int64_t limit = cuda::TypeTraits<int32_t>::max;
    for (int p = 0; p < data->numPlanes(); ++p)
    {
        const nvcv::ImagePlaneStrided &plane    = data->plane(p);
        const nvcv::Size2D             expected = format.planeSize(image.size(), p);
        if (plane.width != expected.w || plane.height != expected.h)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Image plane descriptors must match the image format and size");
        }
        if (expected != image.size())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "All image planes must have matching width and height");
        }

        const int64_t pixelStride = format.planePixelStrideBytes(p);
        if (plane.rowStride < 0
            || (static_cast<int64_t>(plane.height - 1) * plane.rowStride
                    + static_cast<int64_t>(plane.width - 1) * pixelStride
                > limit))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW,
                                  "Input or output image-plane maximum byte offset exceeds %d",
                                  static_cast<int>(limit));
        }
    }
}

inline bool ValidateSrcDstVarBatch(bool &isEmpty, int &numChannels, nvcv::DataType &dtype,
                                   const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst,
                                   const nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda> &srcData,
                                   const nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda> &dstData)
{
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }
    if (srcData->numImages() != dstData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }
    isEmpty = srcData->numImages() == 0;
    if (isEmpty)
    {
        return false;
    }

    const auto &srcFormat = srcData->uniqueFormat();
    const auto &dstFormat = dstData->uniqueFormat();
    if (!srcFormat || !dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images in a batch must have the same format");
    }
    if (srcFormat != dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same format");
    }

    numChannels = srcFormat.numChannels();
    if (numChannels != 1 && numChannels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The input must have 1 or 3 channels");
    }
    if (numChannels == 3
        && (srcFormat.colorModel() != nvcv::ColorModel::RGB
            || (srcFormat.swizzle() != nvcv::Swizzle::S_XYZ0 && srcFormat.swizzle() != nvcv::Swizzle::S_XYZ1)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Three-channel input must use RGB channel order");
    }

    nvcv::ExtraChannelInfo extraChannels{};
    srcFormat.extraChannelInfo(&extraChannels);
    if (extraChannels.numChannels != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image formats with extra channels are not supported");
    }
    if (srcFormat.chromaSubsampling() != nvcv::ChromaSubsampling::CSS_444)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Chroma-subsampled image formats are not supported");
    }

    const int numPlanes = srcFormat.numPlanes();
    dtype               = srcFormat.planeDataType(0);
    if (numPlanes == 1)
    {
        if (dtype.numChannels() != numChannels || srcFormat.planePixelStrideBytes(0) != dtype.strideBytes())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Single-plane images must have one packed element per pixel");
        }
    }
    else if (numPlanes != numChannels || dtype.numChannels() != 1
             || srcFormat.planePixelStrideBytes(0) != dtype.strideBytes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar images must have one scalar, full-resolution plane per channel");
    }
    for (int p = 1; p < numPlanes; ++p)
    {
        if (srcFormat.planeDataType(p) != dtype || srcFormat.planeDataType(p).numChannels() != 1
            || srcFormat.planePixelStrideBytes(p) != dtype.strideBytes())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "All image planes must have the same packed scalar data type");
        }
    }

    for (int i = 0; i < src.numImages(); ++i)
    {
        const nvcv::Size2D srcSize = src[i].size();
        const nvcv::Size2D dstSize = dst[i].size();
        if (srcSize != dstSize)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output image %d sizes must match: input is %dx%d, output is %dx%d", i,
                                  srcSize.w, srcSize.h, dstSize.w, dstSize.h);
        }
        ValidateImagePlanes(src[i], srcFormat);
        ValidateImagePlanes(dst[i], dstFormat);
    }

    return numPlanes > 1;
}

} // anonymous namespace

namespace cvcuda::priv {

AdjustSaturation::AdjustSaturation() {}

// Tensor input variant
void AdjustSaturation::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                  double saturation) const
{
    ValidateSaturation(saturation);

    bool           isEmpty;
    int            numChannels;
    nvcv::DataType dtype;
    auto           srcData  = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData  = dst.exportData<nvcv::TensorDataStridedCuda>();
    const bool     isPlanar = ValidateSrcDstTensors(isEmpty, numChannels, dtype, srcData, dstData);
    if (isEmpty)
    {
        return;
    }

    const float ratio    = static_cast<float>(saturation);
    const float oneMinus = static_cast<float>(1.0 - saturation);

    RunChannelSwitch(numChannels, dtype,
                     [&](auto dummy, auto channelsIC)
                     {
                         using T                   = decltype(dummy);
                         constexpr int numChannels = decltype(channelsIC)::value;
                         RunTensor<T, numChannels>(stream, *srcData, *dstData, isPlanar, ratio, oneMinus);
                     });
}

// VarShape input variant
void AdjustSaturation::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                  const nvcv::ImageBatchVarShape &dst, double saturation) const
{
    ValidateSaturation(saturation);

    bool           isEmpty;
    int            numChannels;
    nvcv::DataType dtype;
    auto           srcData  = src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    auto           dstData  = dst.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    const bool     isPlanar = ValidateSrcDstVarBatch(isEmpty, numChannels, dtype, src, dst, srcData, dstData);
    if (isEmpty)
    {
        return;
    }

    const float ratio    = static_cast<float>(saturation);
    const float oneMinus = static_cast<float>(1.0 - saturation);

    RunChannelSwitch(numChannels, dtype,
                     [&](auto dummy, auto channelsIC)
                     {
                         using T                   = decltype(dummy);
                         constexpr int numChannels = decltype(channelsIC)::value;
                         RunVarShape<T, numChannels>(stream, *srcData, *dstData, isPlanar, ratio, oneMinus);
                     });
}

} // namespace cvcuda::priv

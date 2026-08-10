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

#include "OpAdjustSharpness.hpp"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// Blend clamp bound per base type: dtype max for unsigned integers, 1.0 for float (mirrors the
// torchvision `_max_value`).
template<typename BT>
inline __device__ float SharpnessBound()
{
    if constexpr (std::is_floating_point_v<BT>)
    {
        return 1.0f;
    }
    else
    {
        return static_cast<float>(cuda::TypeTraits<BT>::max);
    }
}

// Per-channel interior compute, shared bit-exactly with the CPU gold reference (see
// TestOpAdjustSharpness.cpp). Every multiply-add is an explicit `fmaf` and the accumulation order is
// fixed, so the correctly-rounded IEEE-754 result is identical on host and device regardless of the
// compiler's FMA-contraction setting. `c11` is the center (original) pixel; the eight edge taps
// carry weight 1/13 and the center 5/13. Integer types round the smoothed value to nearest (ties to
// even) before blending, then the clamped blend is truncated toward zero on the final cast — both
// matching torchvision's `blurred.round_()` + `.to(dtype)`.
template<typename BT>
inline __device__ BT AdjustSharpnessScalar(float c00, float c01, float c02, float c10, float c11, float c12, float c20,
                                           float c21, float c22, float oneMinusFactor, float bound)
{
    constexpr float kEdge   = 1.0f / 13.0f;
    constexpr float kCenter = 5.0f / 13.0f;

    float blur = c11 * kCenter;
    blur       = fmaf(c00, kEdge, blur);
    blur       = fmaf(c01, kEdge, blur);
    blur       = fmaf(c02, kEdge, blur);
    blur       = fmaf(c10, kEdge, blur);
    blur       = fmaf(c12, kEdge, blur);
    blur       = fmaf(c20, kEdge, blur);
    blur       = fmaf(c21, kEdge, blur);
    blur       = fmaf(c22, kEdge, blur);

    if constexpr (!std::is_floating_point_v<BT>)
    {
        blur = rintf(blur); // round-to-nearest-even, matching torch.round on the smoothed image
    }

    // out = in + (1 - factor) * (blur - in) = factor*in + (1-factor)*blur
    const float out     = fmaf(oneMinusFactor, blur - c11, c11);
    const float clamped = fminf(fmaxf(out, 0.0f), bound);
    return static_cast<BT>(clamped); // truncates toward zero for integer BT; identity for float
}

template<bool IsPlanar>
inline __device__ std::conditional_t<IsPlanar, int4, int3> CoordFor(int col, int row, int plane, int sample)
{
    if constexpr (!IsPlanar)
    {
        return int3{col, row, sample};
    }
    else
    {
        return int4{col, row, plane, sample};
    }
}

// Interior pixels are blended; the 1-pixel border is copied unchanged (there is no border
// extension). When a dimension is < 3 every pixel is a border pixel, so the whole image is copied —
// this reproduces torchvision returning the input for images with height or width <= 2.
template<bool IsPlanar, class SrcWrapper, class DstWrapper>
inline __device__ void DoAdjustSharpness(SrcWrapper src, DstWrapper dst, int2 size, int plane, float oneMinusFactor)
{
    using SrcT                       = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT                       = typename DstWrapper::ValueType;
    using BT                         = cuda::BaseType<SrcT>;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(numChannels == cuda::NumElements<DstT>);
    static_assert(!IsPlanar || numChannels == 1);

    const int col    = blockIdx.x * blockDim.x + threadIdx.x;
    const int row    = blockIdx.y * blockDim.y + threadIdx.y;
    const int sample = blockIdx.z;
    if (col >= size.x || row >= size.y)
    {
        return;
    }

    if (col == 0 || row == 0 || col == size.x - 1 || row == size.y - 1)
    {
        dst[CoordFor<IsPlanar>(col, row, plane, sample)] = src[CoordFor<IsPlanar>(col, row, plane, sample)];
        return;
    }

    const float bound = SharpnessBound<BT>();

    const SrcT n00 = src[CoordFor<IsPlanar>(col - 1, row - 1, plane, sample)];
    const SrcT n01 = src[CoordFor<IsPlanar>(col, row - 1, plane, sample)];
    const SrcT n02 = src[CoordFor<IsPlanar>(col + 1, row - 1, plane, sample)];
    const SrcT n10 = src[CoordFor<IsPlanar>(col - 1, row, plane, sample)];
    const SrcT n11 = src[CoordFor<IsPlanar>(col, row, plane, sample)];
    const SrcT n12 = src[CoordFor<IsPlanar>(col + 1, row, plane, sample)];
    const SrcT n20 = src[CoordFor<IsPlanar>(col - 1, row + 1, plane, sample)];
    const SrcT n21 = src[CoordFor<IsPlanar>(col, row + 1, plane, sample)];
    const SrcT n22 = src[CoordFor<IsPlanar>(col + 1, row + 1, plane, sample)];

    DstT out{};
#pragma unroll
    for (int ch = 0; ch < numChannels; ++ch)
    {
        cuda::GetElement(out, ch) = AdjustSharpnessScalar<BT>(
            static_cast<float>(cuda::GetElement(n00, ch)), static_cast<float>(cuda::GetElement(n01, ch)),
            static_cast<float>(cuda::GetElement(n02, ch)), static_cast<float>(cuda::GetElement(n10, ch)),
            static_cast<float>(cuda::GetElement(n11, ch)), static_cast<float>(cuda::GetElement(n12, ch)),
            static_cast<float>(cuda::GetElement(n20, ch)), static_cast<float>(cuda::GetElement(n21, ch)),
            static_cast<float>(cuda::GetElement(n22, ch)), oneMinusFactor, bound);
    }
    dst[CoordFor<IsPlanar>(col, row, plane, sample)] = out;
}

// Tensor variant
template<bool isPlanar, class SrcWrapper, class DstWrapper>
__global__ void AdjustSharpness(SrcWrapper src, DstWrapper dst, int2 size, int numPlanes, float oneMinusFactor)
{
    assert(isPlanar || numPlanes == 1);
    if constexpr (!isPlanar)
    {
        DoAdjustSharpness<isPlanar>(src, dst, size, 0, oneMinusFactor);
    }
    else
    {
        for (int p = 0; p < numPlanes; p++)
        {
            DoAdjustSharpness<isPlanar>(src, dst, size, p, oneMinusFactor);
        }
    }
}

// VarShape variant
template<bool isPlanar, class SrcWrapper, class DstWrapper>
__global__ void AdjustSharpness(SrcWrapper src, DstWrapper dst, int numPlanes, float oneMinusFactor)
{
    assert(isPlanar || numPlanes == 1);
    const int z = blockIdx.z;
    int2      size{dst.width(z), dst.height(z)};

    if constexpr (!isPlanar)
    {
        DoAdjustSharpness<isPlanar>(src, dst, size, 0, oneMinusFactor);
    }
    else
    {
        for (int p = 0; p < numPlanes; p++)
        {
            DoAdjustSharpness<isPlanar>(src, dst, size, p, oneMinusFactor);
        }
    }
}

// Run AdjustSharpness kernel -------------------------------------------------------------

template<bool isPlanar, typename ValueT, class SrcData, class DstData>
inline void RunAdjustSharpness(cudaStream_t stream, const SrcData &srcData, const DstData &dstData, float factor)
{
    const float oneMinusFactor = 1.0f - factor;
    dim3        block(32, 4, 1);
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
        int2 size      = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});

        // Each sample maps to one grid.z block (planar channels are looped inside the kernel);
        // CUDA caps grid.z at 65535.
        if (srcAccess->numSamples() > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Batch size exceeds the CUDA grid.z limit of 65535");
        }
        dim3 grid(util::DivUp(size.x, block.x), util::DivUp(size.y, block.y), srcAccess->numSamples());

        int64_t inMaxStride  = srcAccess->sampleStride() * srcAccess->numSamples();
        int64_t outMaxStride = dstAccess->sampleStride() * dstAccess->numSamples();
        if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }
        using StrideType = int32_t;

        if constexpr (!isPlanar)
        {
            auto src = cuda::CreateTensorWrapNHW<const ValueT, StrideType>(srcData);
            auto dst = cuda::CreateTensorWrapNHW<ValueT, StrideType>(dstData);
            AdjustSharpness<isPlanar><<<grid, block, 0, stream>>>(src, dst, size, 1, oneMinusFactor);
        }
        else
        {
            const int numPlanes = srcAccess->numPlanes();
            auto      src       = cuda::Tensor4DWrap<const ValueT, StrideType>(
                srcData.basePtr(), static_cast<int>(srcAccess->sampleStride()),
                static_cast<int>(srcAccess->planeStride()), static_cast<int>(srcAccess->rowStride()));
            auto dst = cuda::Tensor4DWrap<ValueT, StrideType>(
                dstData.basePtr(), static_cast<int>(dstAccess->sampleStride()),
                static_cast<int>(dstAccess->planeStride()), static_cast<int>(dstAccess->rowStride()));
            AdjustSharpness<isPlanar><<<grid, block, 0, stream>>>(src, dst, size, numPlanes, oneMinusFactor);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);
        // One grid.z block per image; CUDA caps grid.z at 65535.
        if (dstData.numImages() > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Batch size exceeds the CUDA grid.z limit of 65535");
        }
        int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};
        dim3 grid(util::DivUp(dstMaxSize.x, block.x), util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);

        const int numPlanes = dstData.uniqueFormat().numPlanes();

        cuda::ImageBatchVarShapeWrap<const ValueT> src(srcData);
        cuda::ImageBatchVarShapeWrap<ValueT>       dst(dstData);
        AdjustSharpness<isPlanar><<<grid, block, 0, stream>>>(src, dst, numPlanes, oneMinusFactor);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

// Dispatch over base data type (u8 / u16 / f32) and channel count (1 / 3 / 4) -------------

template<typename Cb>
inline void RunTypeSwitch(nvcv::DataType dType, const Cb &cb)
{
    using uchar  = unsigned char;
    using ushort = unsigned short;

#define NVCV_ADJUST_SHARPNESS_RUN_TYPED(DYN_BASE_TYPE, STATIC_BASE_TYPE)              \
    ((dType == nvcv::TYPE_4##DYN_BASE_TYPE) || (dType == nvcv::TYPE_3##DYN_BASE_TYPE) \
     || (dType == nvcv::TYPE_2##DYN_BASE_TYPE) || (dType == nvcv::TYPE_##DYN_BASE_TYPE)) cb(STATIC_BASE_TYPE{});

    // clang-format off
    if NVCV_ADJUST_SHARPNESS_RUN_TYPED(U8, uchar)
    else if NVCV_ADJUST_SHARPNESS_RUN_TYPED(U16, ushort)
    else if NVCV_ADJUST_SHARPNESS_RUN_TYPED(F32, float)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: AdjustSharpness supports 8-bit unsigned, 16-bit unsigned and 32-bit float");
    }
        // clang-format on

#undef NVCV_ADJUST_SHARPNESS_RUN_TYPED
}

template<typename Cb>
inline void RunChannelSwitch(int numChannels, int numPlanes, nvcv::DataType dType, const Cb &cb)
{
    RunTypeSwitch(dType,
                  [&numChannels, &numPlanes, &cb](auto dummyVal)
                  {
                      using ValBase = decltype(dummyVal);
                      // clang-format off
            if (numChannels == 1)
            {
                using Val = cuda::MakeType<ValBase, 1>;
                if (numPlanes == 1)
                {
                    cb(Val{}, std::integral_constant<bool, false>{});
                }
                else
                {
                    cb(Val{}, std::integral_constant<bool, true>{});
                }
            }
            else if (numChannels == 3)
            {
                cb(cuda::MakeType<ValBase, 3>{}, std::integral_constant<bool, false>{});
            }
            else if (numChannels == 4)
            {
                cb(cuda::MakeType<ValBase, 4>{}, std::integral_constant<bool, false>{});
            }
            else
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Invalid number of channels: AdjustSharpness supports 1, 3 or 4 channels");
            }
                      // clang-format on
                  });
}

// Validation ------------------------------------------------------------------------------

inline void ValidateSrcDstTensors(int &numInterleavedChannels, int &numPlanes, nvcv::DataType &dtype,
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
    if (!(srcData->layout() == nvcv::TENSOR_HWC || srcData->layout() == nvcv::TENSOR_NHWC
          || srcData->layout() == nvcv::TENSOR_CHW || srcData->layout() == nvcv::TENSOR_NCHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC or (N)CHW layout");
    }
    if (srcData->dtype() != dstData->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same data type");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    if (srcAccess->numSamples() != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    int numChannels = srcAccess->numChannels();
    if (numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }

    numPlanes = srcAccess->numPlanes();
    if (numPlanes != dstAccess->numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of planes");
    }
    if (numPlanes > 1 && numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar images are not supported");
    }

    if (srcAccess->numCols() != dstAccess->numCols() || srcAccess->numRows() != dstAccess->numRows())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching width and height");
    }

    dtype                  = srcData->dtype();
    numInterleavedChannels = srcAccess->infoLayout().isChannelLast() ? numChannels : 1;
}

inline auto ValidateSrcDstVarBatch(int &numInterleavedChannels, int &numPlanes, nvcv::DataType &dtype,
                                   cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                   const nvcv::ImageBatchVarShape &dst)
{
    using maybeVarShape = nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda>;
    std::tuple<maybeVarShape, maybeVarShape> srcDstData{
        src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream),
        dst.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream)};
    auto &[srcData, dstData] = srcDstData;

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

    int numSamples = srcData->numImages();
    if (numSamples != dstData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
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

    int numChannels = srcFormat.numChannels();
    numPlanes       = srcFormat.numPlanes();
    if (numPlanes > 1 && numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar images are not supported");
    }

    dtype = srcFormat.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (dtype != srcFormat.planeDataType(i))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "All planes in the input image must have the same data type");
        }
    }

    numInterleavedChannels = dtype.numChannels();

    for (int i = 0; i < numSamples; i++)
    {
        if (src[i].size() != dst[i].size())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output must have matching width and height");
        }
    }

    return srcDstData;
}

inline void ValidateSharpnessFactor(float sharpnessFactor)
{
    if (sharpnessFactor < 0.0f)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Sharpness factor must be non-negative");
    }
}

} // anonymous namespace

namespace cvcuda::priv {

AdjustSharpness::AdjustSharpness() {}

// Tensor input variant
void AdjustSharpness::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                 float sharpnessFactor) const
{
    ValidateSharpnessFactor(sharpnessFactor);

    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto           srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ValidateSrcDstTensors(numInterleavedChannels, numPlanes, dtype, srcData, dstData);

    RunChannelSwitch(numInterleavedChannels, numPlanes, dtype,
                     [&stream, &srcData, &dstData, sharpnessFactor](auto dummyVal, auto isPlanar)
                     {
                         using ValueT   = decltype(dummyVal);
                         using IsPlanar = decltype(isPlanar);
                         RunAdjustSharpness<IsPlanar::value, ValueT>(stream, *srcData, *dstData, sharpnessFactor);
                     });
}

// VarShape input variant
void AdjustSharpness::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                 const nvcv::ImageBatchVarShape &dst, float sharpnessFactor) const
{
    ValidateSharpnessFactor(sharpnessFactor);

    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto           srcDstData = ValidateSrcDstVarBatch(numInterleavedChannels, numPlanes, dtype, stream, src, dst);

    RunChannelSwitch(numInterleavedChannels, numPlanes, dtype,
                     [&stream, &srcDstData, sharpnessFactor](auto dummyVal, auto isPlanar)
                     {
                         using ValueT             = decltype(dummyVal);
                         using IsPlanar           = decltype(isPlanar);
                         auto &[srcData, dstData] = srcDstData;
                         RunAdjustSharpness<IsPlanar::value, ValueT>(stream, *srcData, *dstData, sharpnessFactor);
                     });
}

} // namespace cvcuda::priv

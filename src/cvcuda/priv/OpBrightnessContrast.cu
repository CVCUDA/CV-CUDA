/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "BrightnessContrastPolicy.hpp"
#include "CudaDeviceUtils.hpp"
#include "Nvtx.hpp"
#include "OpBrightnessContrast.hpp"

#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <tuple>
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

template<typename T, typename Ret>
constexpr __host__ __device__ Ret HalfRange()
{
    return std::is_integral<T>::value ? (1 << (8 * sizeof(T) - std::is_signed<T>::value - 1)) : 0.5;
}

template<typename T>
inline constexpr bool RequiresDouble = std::is_integral_v<T> && sizeof(T) >= 4;

template<typename SrcBT, typename DstBT>
using GetArgType = std::conditional_t<RequiresDouble<SrcBT> || RequiresDouble<DstBT>, double, float>;

template<typename ArgT>
using ArgWrapper = cuda::Tensor1DWrap<const ArgT, int32_t>;

template<typename BT>
struct SampleArgs
{
    BT   brightness;
    BT   contrast;
    BT   brightnessShift;
    BT   contrastCenter;
    bool clamp;
};

template<typename BT>
struct BatchArgsWrap
{
    int                  brightnessLen, contrastLen, brightnessShiftLen, contrastCenterLen;
    const ArgWrapper<BT> brightness;
    const ArgWrapper<BT> contrast;
    const ArgWrapper<BT> brightnessShift;
    const ArgWrapper<BT> contrastCenter;
    BT                   scalarBrightness;
    BT                   scalarContrast;
    BT                   scalarBrightnessShift;
    BT                   scalarContrastCenter;
    bool                 clamp;
};

template<typename BT>
inline __device__ BT GetArg(const ArgWrapper<BT> &tensorArg, int argLen, int sampleIdx, BT defaultVal, BT scalarVal)
{
    if (argLen < 0)
    {
        return scalarVal;
    }
    else if (argLen == 0)
    {
        return defaultVal;
    }
    else if (argLen == 1)
    {
        return tensorArg[0];
    }
    else
    {
        return tensorArg[sampleIdx];
    }
}

template<typename SrcBT, typename BT>
inline __device__ SampleArgs<BT> GetBrightnessContrastArg(const BatchArgsWrap<BT> &args, int sampleIdx)
{
    return {GetArg(args.brightness, args.brightnessLen, sampleIdx, BT{1}, args.scalarBrightness),
            GetArg(args.contrast, args.contrastLen, sampleIdx, BT{1}, args.scalarContrast),
            GetArg(args.brightnessShift, args.brightnessShiftLen, sampleIdx, BT{0}, args.scalarBrightnessShift),
            GetArg(args.contrastCenter, args.contrastCenterLen, sampleIdx, HalfRange<SrcBT, BT>(),
                   args.scalarContrastCenter),
            args.clamp};
}

template<typename DstT, typename T>
inline __device__ T ClampToImageRange(T value, bool clamp)
{
    if (!clamp)
    {
        return value;
    }

    using BT                         = cuda::BaseType<T>;
    using DstBT                      = cuda::BaseType<DstT>;
    static constexpr int numElements = cuda::NumElements<T>;
    const BT bound = std::is_floating_point_v<DstBT> ? BT{1} : static_cast<BT>(cuda::TypeTraits<DstBT>::max);
#pragma unroll
    for (int c = 0; c < numElements; ++c)
    {
        auto &v = cuda::GetElement(value, c);
        v       = v < BT{0} ? BT{0} : (v > bound ? bound : v);
    }
    return value;
}

template<bool IsPlanar>
inline __device__ std::conditional_t<IsPlanar, int4, int3> GetCoordForLayout(int3 nhwCoord, int p)
{
    if constexpr (!IsPlanar)
    {
        assert(p == 0);
        return nhwCoord;
    }
    else
    {
        return {nhwCoord.x, nhwCoord.y, p, nhwCoord.z};
    }
}

template<bool IsPlanar, class SrcWrapper, class DstWrapper, typename ArgT>
inline __device__ void DoBrightnessContrast(SrcWrapper src, DstWrapper dst, const SampleArgs<ArgT> arg, const int2 size,
                                            const int p)
{
    using SrcT                       = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT                       = typename DstWrapper::ValueType;
    using IntermediateT              = decltype(std::declval<ArgT>() * std::declval<SrcT>());
    using SrcBT                      = cuda::BaseType<SrcT>;
    using DstBT                      = cuda::BaseType<DstT>;
    using BI                         = cuda::BaseType<IntermediateT>;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(numChannels == cuda::NumElements<DstT>);
    // if planar then no interleaved channels
    static_assert(!IsPlanar || numChannels == 1);
    static_assert(std::is_same_v<BI, GetArgType<SrcBT, DstBT>>);

    int3 nhwCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (nhwCoord.x >= size.x || nhwCoord.y >= size.y)
    {
        return;
    }
    auto coord = GetCoordForLayout<IsPlanar>(nhwCoord, p);
    auto pixel = cuda::StaticCast<BI>(src[coord]);
    pixel = arg.brightnessShift + arg.brightness * (arg.contrastCenter + arg.contrast * (pixel - arg.contrastCenter));
    pixel = ClampToImageRange<DstT>(pixel, arg.clamp);
    dst[coord] = cuda::SaturateCast<DstT>(pixel);
}

// Per-pixel affine, factored out so the scalar and vectorized paths share byte-identical math.
template<typename DstT, typename SrcT, typename ArgT>
inline __device__ DstT ApplyBrightnessContrast(SrcT v, const SampleArgs<ArgT> &arg)
{
    using IntermediateT = decltype(std::declval<ArgT>() * std::declval<SrcT>());
    using BI            = cuda::BaseType<IntermediateT>;
    auto pixel          = cuda::StaticCast<BI>(v);
    pixel = arg.brightnessShift + arg.brightness * (arg.contrastCenter + arg.contrast * (pixel - arg.contrastCenter));
    pixel = ClampToImageRange<DstT>(pixel, arg.clamp);
    return cuda::SaturateCast<DstT>(pixel);
}

// Vector pack type for T: uint3 (12B) for 3-element T, else uint4 (16B). BC_NIX = pixels/thread.
template<typename T>
using BC_DPT = std::conditional_t<cuda::NumElements<T> == 3, uint3, uint4>;
template<typename T>
constexpr int BC_NIX = sizeof(BC_DPT<T>) / sizeof(T);
template<typename T>
using BC_BATCH_DPT = std::conditional_t<std::is_same_v<cuda::BaseType<T>, unsigned char> && cuda::NumElements<T> == 1,
                                        uint2, BC_DPT<T>>;
template<typename PackT>
constexpr uintptr_t BC_PACK_MSK = (sizeof(PackT) == sizeof(uint3) ? sizeof(uint) : sizeof(PackT)) - 1;
template<typename T>
constexpr uintptr_t BC_MSK = BC_PACK_MSK<BC_DPT<T>>;

// Vectorized interleaved (NHWC) brightness/contrast. Keep this path separate from planar batching so
// that the existing one-plane kernel does not pay for runtime plane selection.
template<class SrcWrapper, class DstWrapper, typename ArgT>
__global__ void BrightnessContrastVec(SrcWrapper src, DstWrapper dst, BatchArgsWrap<ArgT> batchArgs, int2 size)
{
    using SrcT             = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT             = typename DstWrapper::ValueType;
    using SrcBT            = cuda::BaseType<SrcT>;
    static constexpr int N = BC_NIX<SrcT>;

    const int z = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= size.y)
        return;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if (x0 >= size.x)
        return;

    const auto  arg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);
    const SrcT *sp  = &src[int3{x0, y, z}];
    DstT       *dp  = &dst[int3{x0, y, z}];

    if (x0 + N - 1 < size.x && (reinterpret_cast<uintptr_t>(sp) & BC_MSK<SrcT>) == 0
        && (reinterpret_cast<uintptr_t>(dp) & BC_MSK<DstT>) == 0)
    {
        alignas(BC_DPT<SrcT>) SrcT in[N];
        *reinterpret_cast<BC_DPT<SrcT> *>(in) = *reinterpret_cast<const BC_DPT<SrcT> *>(sp);
        alignas(BC_DPT<DstT>) DstT out[N];
#pragma unroll
        for (int i = 0; i < N; ++i) out[i] = ApplyBrightnessContrast<DstT>(in[i], arg);
        *reinterpret_cast<BC_DPT<DstT> *>(dp) = *reinterpret_cast<const BC_DPT<DstT> *>(out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
            if (x0 + i < size.x)
                dp[i] = ApplyBrightnessContrast<DstT>(sp[i], arg);
    }
}

// Batch adjacent planar or var-shape elements while reusing wrapper lookups and per-image arguments.
template<bool IsPlanar, bool IsVarShape, typename PackT, class SrcWrapper, class DstWrapper, typename ArgT>
__global__ void BrightnessContrastBatchVec(SrcWrapper src, DstWrapper dst, BatchArgsWrap<ArgT> batchArgs,
                                           int2 tensorSize, int numPlanes)
{
    using SrcT             = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT             = typename DstWrapper::ValueType;
    using SrcBT            = cuda::BaseType<SrcT>;
    static constexpr int N = sizeof(PackT) / sizeof(SrcT);

    const int z = blockIdx.z;
    int2      size{tensorSize};
    if constexpr (IsVarShape)
    {
        size = {dst.width(z), dst.height(z)};
    }
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= size.y)
        return;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if (x0 >= size.x)
        return;

    const auto arg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);
    assert(IsPlanar || numPlanes == 1);
    for (int p = 0; p < numPlanes; ++p)
    {
        auto        coord = GetCoordForLayout<IsPlanar>(int3{x0, y, z}, p);
        const SrcT *sp    = &src[coord];
        DstT       *dp    = &dst[coord];

        if (x0 + N - 1 < size.x && (reinterpret_cast<uintptr_t>(sp) & BC_PACK_MSK<PackT>) == 0
            && (reinterpret_cast<uintptr_t>(dp) & BC_PACK_MSK<PackT>) == 0)
        {
            alignas(PackT) SrcT in[N];
            *reinterpret_cast<PackT *>(in) = *reinterpret_cast<const PackT *>(sp);
            alignas(PackT) DstT out[N];
#pragma unroll
            for (int i = 0; i < N; ++i) out[i] = ApplyBrightnessContrast<DstT>(in[i], arg);
            *reinterpret_cast<PackT *>(dp) = *reinterpret_cast<const PackT *>(out);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < N; ++i)
                if (x0 + i < size.x)
                    dp[i] = ApplyBrightnessContrast<DstT>(sp[i], arg);
        }
    }
}

// BrightnessContrast kernel --------------------------------------------------------------

// Tensor variant
template<bool isPlanar, class SrcWrapper, class DstWrapper, typename ArgT>
__global__ void BrightnessContrast(SrcWrapper src, DstWrapper dst, BatchArgsWrap<ArgT> batchArgs, int2 size,
                                   int numPlanes)
{
    assert(isPlanar || numPlanes == 1);
    using SrcBT    = cuda::BaseType<typename SrcWrapper::ValueType>;
    int  z         = blockIdx.z;
    auto sampleArg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);

    if constexpr (!isPlanar)
    {
        DoBrightnessContrast<isPlanar>(src, dst, sampleArg, size, 0);
    }
    else
    {
        for (int p = 0; p < numPlanes; p++)
        {
            DoBrightnessContrast<isPlanar>(src, dst, sampleArg, size, p);
        }
    }
}

// VarBatch variant
template<bool isPlanar, class SrcWrapper, class DstWrapper, typename ArgT>
__global__ void BrightnessContrast(SrcWrapper src, DstWrapper dst, BatchArgsWrap<ArgT> batchArgs, int numPlanes)
{
    using SrcBT = cuda::BaseType<typename SrcWrapper::ValueType>;
    assert(isPlanar || numPlanes == 1);
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};
    auto sampleArg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);

    if constexpr (!isPlanar)
    {
        DoBrightnessContrast<isPlanar>(src, dst, sampleArg, size, 0);
    }
    else
    {
        for (int p = 0; p < numPlanes; p++)
        {
            DoBrightnessContrast<isPlanar>(src, dst, sampleArg, size, p);
        }
    }
}

// Run BrightnessContrast kernel ----------------------------------------------------------

template<bool isPlanar, typename SrcValueT, typename DstValueT, typename ArgT, class SrcData, class DstData>
inline void RunBrightnessContrast(cudaStream_t stream, const SrcData &srcData, const DstData &dstData,
                                  const BatchArgsWrap<ArgT> &batchArgs)
{
    using SrcBT                        = cuda::BaseType<SrcValueT>;
    static constexpr int  kNix         = BC_NIX<SrcValueT>;
    static constexpr int  kBatchNix    = sizeof(BC_BATCH_DPT<SrcValueT>) / sizeof(SrcValueT);
    static constexpr int  kNumElements = cuda::NumElements<SrcValueT>;
    static constexpr bool kUseBatchedPath
        = kNix > 1 && std::is_same_v<SrcBT, unsigned char> && (kNumElements == 1 || kNumElements == 3);
    dim3 block(32, 4, 1);
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        int2 size      = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});
        dim3 grid(util::DivUp(size.x, block.x), util::DivUp(size.y, block.y), srcAccess->numSamples());
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);

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
            auto src = cuda::CreateTensorWrapNHW<const SrcValueT, StrideType>(srcData);
            auto dst = cuda::CreateTensorWrapNHW<DstValueT, StrideType>(dstData);
            if constexpr (std::is_same_v<SrcValueT, DstValueT>)
            {
                // Vectorized interleaved path (BC_NIX pixels/thread). Same-type only so the load/store
                // pack widths match; mixed in/out types keep the scalar kernel.
                int sm = 0;
                NVCV_CHECK_THROW(cvcuda::priv::GetCurrentDeviceSM(sm));
                constexpr bool kIsUnsignedByte = std::is_same_v<SrcBT, unsigned char>;
                auto           policy
                    = cvcuda::priv::BrightnessContrastTensorKernelPolicyForSM(sm, kIsUnsignedByte, kNumElements);
                if (policy == cvcuda::priv::BrightnessContrastTensorKernelPolicy::kScalar)
                {
                    BrightnessContrast<isPlanar><<<grid, block, 0, stream>>>(src, dst, batchArgs, size, 1);
                }
                else
                {
                    dim3 gridV(util::DivUp(size.x, block.x * kNix), util::DivUp(size.y, block.y),
                               srcAccess->numSamples());
                    BrightnessContrastVec<<<gridV, block, 0, stream>>>(src, dst, batchArgs, size);
                }
            }
            else
            {
                BrightnessContrast<isPlanar><<<grid, block, 0, stream>>>(src, dst, batchArgs, size, 1);
            }
        }
        else
        {
            auto src = cuda::Tensor4DWrap<const SrcValueT, StrideType>(
                srcData.basePtr(), static_cast<int>(srcAccess->sampleStride()),
                static_cast<int>(srcAccess->planeStride()), static_cast<int>(srcAccess->rowStride()));
            auto dst = cuda::Tensor4DWrap<DstValueT, StrideType>(
                dstData.basePtr(), static_cast<int>(dstAccess->sampleStride()),
                static_cast<int>(dstAccess->planeStride()), static_cast<int>(dstAccess->rowStride()));
            int numPlanes = srcAccess->numPlanes();
            if constexpr (std::is_same_v<SrcValueT, DstValueT> && kUseBatchedPath)
            {
                dim3 gridV(util::DivUp(size.x, block.x * kBatchNix), util::DivUp(size.y, block.y),
                           srcAccess->numSamples());
                BrightnessContrastBatchVec<true, false, BC_BATCH_DPT<SrcValueT>>
                    <<<gridV, block, 0, stream>>>(src, dst, batchArgs, size, numPlanes);
            }
            else
            {
                BrightnessContrast<isPlanar><<<grid, block, 0, stream>>>(src, dst, batchArgs, size, numPlanes);
            }
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);
        int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};
        dim3 grid(util::DivUp(dstMaxSize.x, block.x), util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);

        cuda::ImageBatchVarShapeWrap<const SrcValueT> src(srcData);
        cuda::ImageBatchVarShapeWrap<DstValueT>       dst(dstData);

        int numPlanes = dstData.uniqueFormat().numPlanes();
        if constexpr (std::is_same_v<SrcValueT, DstValueT> && kUseBatchedPath)
        {
            dim3 gridV(util::DivUp(dstMaxSize.x, block.x * kBatchNix), util::DivUp(dstMaxSize.y, block.y),
                       dstMaxSize.z);
            BrightnessContrastBatchVec<isPlanar, true, BC_BATCH_DPT<SrcValueT>>
                <<<gridV, block, 0, stream>>>(src, dst, batchArgs, int2{}, numPlanes);
        }
        else
        {
            BrightnessContrast<isPlanar><<<grid, block, 0, stream>>>(src, dst, batchArgs, numPlanes);
        }

        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

template<typename Cb>
inline void RunTypeSwitch(nvcv::DataType dType, const Cb &cb)
{
    using uchar = unsigned char;

#define NVCV_BRIGHTNESS_RUN_TYPED(DYN_BASE_TYPE, STATIC_BASE_TYPE)                    \
    ((dType == nvcv::TYPE_4##DYN_BASE_TYPE) || (dType == nvcv::TYPE_3##DYN_BASE_TYPE) \
     || (dType == nvcv::TYPE_2##DYN_BASE_TYPE) || (dType == nvcv::TYPE_##DYN_BASE_TYPE)) cb(STATIC_BASE_TYPE{});

    // clang-format off
    if NVCV_BRIGHTNESS_RUN_TYPED(U8, uchar)
    else if NVCV_BRIGHTNESS_RUN_TYPED(S16, short)
    else if NVCV_BRIGHTNESS_RUN_TYPED(U16, ushort)
    else if NVCV_BRIGHTNESS_RUN_TYPED(S32, int)
    else if NVCV_BRIGHTNESS_RUN_TYPED(F32, float)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input/output data types");
    }
        // clang-format on

#undef NVCV_BRIGHTNESS_RUN_TYPED
}

template<typename Cb>
inline void RunTypeSwitch(int numChannels, int numPlanes, nvcv::DataType srcType, nvcv::DataType dstType,
                          nvcv::DataType argType, const Cb &cb)
{
#define NVCV_BRIGHTNESS_RUN_CHANNELS(NUM_CHANNELS)                             \
    (numChannels == NUM_CHANNELS)                                              \
    {                                                                          \
        using SrcVal = cuda::MakeType<SrcValBase, NUM_CHANNELS>;               \
        using DstVal = cuda::MakeType<DstValBase, NUM_CHANNELS>;               \
        cb(SrcVal{}, DstVal{}, ArgT{}, std::integral_constant<bool, false>{}); \
    }

    RunTypeSwitch(
        srcType,
        [&dstType, &argType, &numChannels, &numPlanes, &cb](auto dummySrcVal)
        {
            using SrcValBase = decltype(dummySrcVal);
            RunTypeSwitch(
                dstType,
                [&argType, &numChannels, &numPlanes, &cb](auto dummyDstVal)
                {
                    using DstValBase = decltype(dummyDstVal);
                    using ArgT       = GetArgType<SrcValBase, DstValBase>;
                    static_assert(std::is_same_v<ArgT, float> || std::is_same_v<ArgT, double>);
                    auto requiredArgDtype = std::is_same_v<ArgT, float> ? nvcv::TYPE_F32 : nvcv::TYPE_F64;
                    if (argType != NVCV_DATA_TYPE_NONE && argType != requiredArgDtype)
                    {
                        if (requiredArgDtype == nvcv::TYPE_F32)
                        {
                            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                                  "The brightness/contrast/brightness shift/contrast center arguments "
                                                  "are expected to have float32 type.");
                        }
                        else
                        {
                            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                                  "When the input or output type is (u)int32, the "
                                                  "brightness/contrast/brightness shift/contrast center arguments "
                                                  "must have double (float64) type.");
                        }
                    }
                    // clang-format off
                    if (numChannels == 1)
                    {
                        using SrcVal = cuda::MakeType<SrcValBase, 1>;
                        using DstVal = cuda::MakeType<DstValBase, 1>;
                        if (numPlanes == 1)
                        {
                            cb(SrcVal{}, DstVal{}, ArgT{}, std::integral_constant<bool, false>{});
                        }
                        else
                        {
                            cb(SrcVal{}, DstVal{}, ArgT{}, std::integral_constant<bool, true>{});
                        }
                    }
                    else if NVCV_BRIGHTNESS_RUN_CHANNELS (2)
                    else if NVCV_BRIGHTNESS_RUN_CHANNELS (3)
                    else if NVCV_BRIGHTNESS_RUN_CHANNELS (4)
                    else
                    {
                        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                            "Invalid number of channels");
                    }
                    // clang-format on
                });
        });
#undef NVCV_BRIGHTNESS_RUN_CHANNELS

} // namespace

// Argument validation helpers ----------------------------------------------------

inline void ValidateSrcDstTensors(int &numSamples, int &numInterleavedChannels, int &numPlanes,
                                  nvcv::DataType &srcDtype, nvcv::DataType &dstDtype,
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

    // Check layout before creating TensorDataAccessStridedImagePlanar, as Create() will fail for unsupported layouts
    if (srcData->layout() != dstData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }

    if (!(srcData->layout() == nvcv::TENSOR_HWC || srcData->layout() == nvcv::TENSOR_NHWC
          || srcData->layout() == nvcv::TENSOR_CHW || srcData->layout() == nvcv::TENSOR_NCHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC or (N)CHW layout");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    numSamples = srcAccess->numSamples();
    if (numSamples != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    int numChannels = srcAccess->numChannels();
    if (numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }
    if (numChannels > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", numChannels);
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

    srcDtype               = srcData->dtype();
    dstDtype               = dstData->dtype();
    numInterleavedChannels = srcAccess->infoLayout().isChannelLast() ? numChannels : 1;

    if (numInterleavedChannels > 1 && numPlanes > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in plannar format are expected to have one channel per plane");
    }
}

inline int GetArgNumSamples(const nvcv::Optional<nvcv::TensorDataStridedCuda> &argData)
{
    if (!argData)
    {
        return 0;
    }
    if (argData->rank() == 0)
    {
        return 1;
    }
    return argData->shape()[0];
}

template<typename ArgT>
inline BatchArgsWrap<ArgT> GetBatchArgsWrap(nvcv::Optional<nvcv::TensorDataStridedCuda> &brightnessData,
                                            nvcv::Optional<nvcv::TensorDataStridedCuda> &contrastData,
                                            nvcv::Optional<nvcv::TensorDataStridedCuda> &brightnessShiftData,
                                            nvcv::Optional<nvcv::TensorDataStridedCuda> &contrastCenterData)
{
    int brightnessLen      = GetArgNumSamples(brightnessData);
    int contrastLen        = GetArgNumSamples(contrastData);
    int brightnessShiftLen = GetArgNumSamples(brightnessShiftData);
    int constrastCenterLen = GetArgNumSamples(contrastCenterData);
    return {brightnessLen,
            contrastLen,
            brightnessShiftLen,
            constrastCenterLen,
            brightnessLen == 0 ? ArgWrapper<ArgT>{} : ArgWrapper<ArgT>(*brightnessData),
            contrastLen == 0 ? ArgWrapper<ArgT>{} : ArgWrapper<ArgT>(*contrastData),
            brightnessShiftLen == 0 ? ArgWrapper<ArgT>{} : ArgWrapper<ArgT>(*brightnessShiftData),
            constrastCenterLen == 0 ? ArgWrapper<ArgT>{} : ArgWrapper<ArgT>(*contrastCenterData),
            ArgT{},
            ArgT{},
            ArgT{},
            ArgT{},
            false};
}

template<typename ArgT>
inline BatchArgsWrap<ArgT> GetScalarBatchArgsWrap(double brightness, double contrast, double brightnessShift,
                                                  double contrastCenter, bool clamp)
{
    return {-1,
            -1,
            -1,
            -1,
            ArgWrapper<ArgT>{},
            ArgWrapper<ArgT>{},
            ArgWrapper<ArgT>{},
            ArgWrapper<ArgT>{},
            static_cast<ArgT>(brightness),
            static_cast<ArgT>(contrast),
            static_cast<ArgT>(brightnessShift),
            static_cast<ArgT>(contrastCenter),
            clamp};
}

inline auto validateSrcDstVarBatch(int &numSamples, int &numInterleavedChannels, int &numPlanes,
                                   nvcv::DataType &srcDtype, nvcv::DataType &dstDtype, cudaStream_t stream,
                                   const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst)
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

    numSamples = srcData->numImages();
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

    int numChannels = srcFormat.numChannels();
    if (numChannels != dstFormat.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }
    if (numChannels > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", numChannels);
    }

    numPlanes = srcFormat.numPlanes();
    if (numPlanes != dstFormat.numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of planes");
    }
    if (numPlanes > 1 && numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar images are not supported");
    }

    srcDtype = srcFormat.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (srcDtype != srcFormat.planeDataType(i))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Ale planes in the input image must have the same data type");
        }
    }

    dstDtype = dstFormat.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (dstDtype != dstFormat.planeDataType(i))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Ale planes in the output image must have the same data type");
        }
    }

    if (numPlanes > 1 && numPlanes != srcFormat.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in plannar format are expected to have one channel per plane");
    }

    // 1. In case of varbatch images, the number of interleaved channels is always baked
    // in the vectorized data type.
    // 2. The srcFormat.numChannels() is a sum of channels number for all planes. Thus,
    // if numPlanes == 1 -> srcDtype.numChannels() == srcFormat.numChannels(), otheriwse by
    // the check above enforcing numPlanes == srcFormat.numChannels() ->
    // srcDtype.numChannels() == 1.
    numInterleavedChannels = srcDtype.numChannels();

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

inline void ValidateTensorArgs(nvcv::DataType &argDType, nvcv::Optional<nvcv::TensorDataStridedCuda> &brightnessData,
                               nvcv::Optional<nvcv::TensorDataStridedCuda> &contrastData,
                               nvcv::Optional<nvcv::TensorDataStridedCuda> &brightnessShiftData,
                               nvcv::Optional<nvcv::TensorDataStridedCuda> &contrastCenterData, int numSamples,
                               const nvcv::Tensor &brightness, const nvcv::Tensor &contrast,
                               const nvcv::Tensor &brightnessShift, const nvcv::Tensor &contrastCenter)
{
    auto validateArgData
        = [&argDType, &numSamples](const std::string &argName, nvcv::Optional<nvcv::TensorDataStridedCuda> &argData,
                                   const nvcv::Tensor &argTensor)
    {
        if (argTensor)
        {
            argData = argTensor.exportData<nvcv::TensorDataStridedCuda>();
            if (!argData)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "The %s argument must be cuda-accessible, pitch-linear tensor", argName.c_str());
            }
            if (argData->rank() > 1)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "The %s argument must be a scalar or 1D tensor", argName.c_str());
            }
            else if (argData->rank() == 1)
            {
                int argNumSamples = argData->shape()[0];
                if (argNumSamples != 0 && argNumSamples != 1 && argNumSamples != numSamples)
                {
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                          "If the %s argument is specified, it must be a scalar or 1D tensor whose "
                                          "length must match the number of input images",
                                          argName.c_str());
                }
            }
            if (argData->dtype() != nvcv::TYPE_F32 && argData->dtype() != nvcv::TYPE_F64)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "The %s argument must have float type or double for int32 input types",
                                      argName.c_str());
            }
            if (argDType == NVCV_DATA_TYPE_NONE)
            {
                argDType = argData->dtype();
            }
            else if (argDType != argData->dtype())
            {
                throw nvcv::Exception(
                    nvcv::Status::ERROR_INVALID_ARGUMENT,
                    "The brightness/contrast/brightness shift and contrast center arguments must be of the same type");
            }
        }
    };

    validateArgData("brightness", brightnessData, brightness);
    validateArgData("contrast", contrastData, contrast);
    validateArgData("brightness shift", brightnessShiftData, brightnessShift);
    validateArgData("contrast center", contrastCenterData, contrastCenter);
}

} // anonymous namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

BrightnessContrast::BrightnessContrast() {}

// Operator --------------------------------------------------------------------

// Tensor input variant
void BrightnessContrast::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                    const nvcv::Tensor &brightness, const nvcv::Tensor &contrast,
                                    const nvcv::Tensor &brightnessShift, const nvcv::Tensor &contrastCenter) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BrightnessContrast::operator()[Tensor]");
    int            numSamples;
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType srcDtype;
    nvcv::DataType dstDtype;
    auto           srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ValidateSrcDstTensors(numSamples, numInterleavedChannels, numPlanes, srcDtype, dstDtype, srcData, dstData);
    nvcv::DataType                              argDType;
    nvcv::Optional<nvcv::TensorDataStridedCuda> brightnessData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> contrastData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> brightnessShiftData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> contrastCenterData;
    ValidateTensorArgs(argDType, brightnessData, contrastData, brightnessShiftData, contrastCenterData, numSamples,
                       brightness, contrast, brightnessShift, contrastCenter);
    RunTypeSwitch(numInterleavedChannels, numPlanes, srcDtype, dstDtype, argDType,
                  [&brightnessData, &contrastData, &brightnessShiftData, &contrastCenterData, &stream, &srcData,
                   &dstData](auto dummySrcVal, auto dummyDstVal, auto dummyArg, auto isPlanar)
                  {
                      using InT      = decltype(dummySrcVal);
                      using OutT     = decltype(dummyDstVal);
                      using ArgT     = decltype(dummyArg);
                      using IsPlanar = decltype(isPlanar);
                      auto args      = GetBatchArgsWrap<ArgT>(brightnessData, contrastData, brightnessShiftData,
                                                         contrastCenterData);
                      RunBrightnessContrast<IsPlanar::value, InT, OutT>(stream, *srcData, *dstData, args);
                  });
}

// VarShape input variant
void BrightnessContrast::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                    const nvcv::ImageBatchVarShape &dst, const nvcv::Tensor &brightness,
                                    const nvcv::Tensor &contrast, const nvcv::Tensor &brightnessShift,
                                    const nvcv::Tensor &contrastCenter) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BrightnessContrast::operator()[ImageBatchVarShape]");
    int            numSamples;
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType srcDtype;
    nvcv::DataType dstDtype;
    auto           srcDstData
        = validateSrcDstVarBatch(numSamples, numInterleavedChannels, numPlanes, srcDtype, dstDtype, stream, src, dst);
    nvcv::DataType                              argDType;
    nvcv::Optional<nvcv::TensorDataStridedCuda> brightnessData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> contrastData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> brightnessShiftData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> contrastCenterData;
    ValidateTensorArgs(argDType, brightnessData, contrastData, brightnessShiftData, contrastCenterData, numSamples,
                       brightness, contrast, brightnessShift, contrastCenter);
    RunTypeSwitch(numInterleavedChannels, numPlanes, srcDtype, dstDtype, argDType,
                  [&brightnessData, &contrastData, &brightnessShiftData, &contrastCenterData, &stream, &srcDstData](
                      auto dummySrcVal, auto dummyDstVal, auto dummyArg, auto isPlanar)
                  {
                      using InT      = decltype(dummySrcVal);
                      using OutT     = decltype(dummyDstVal);
                      using ArgT     = decltype(dummyArg);
                      using IsPlanar = decltype(isPlanar);
                      auto args      = GetBatchArgsWrap<ArgT>(brightnessData, contrastData, brightnessShiftData,
                                                         contrastCenterData);
                      auto &[srcData, dstData] = srcDstData;
                      RunBrightnessContrast<IsPlanar::value, InT, OutT>(stream, *srcData, *dstData, args);
                  });
}

void BrightnessContrast::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                    double brightness, double contrast, double brightnessShift, double contrastCenter,
                                    bool clamp) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BrightnessContrast::operator()[Tensor,Scalar]");
    int            numSamples;
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType srcDtype;
    nvcv::DataType dstDtype;
    auto           srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ValidateSrcDstTensors(numSamples, numInterleavedChannels, numPlanes, srcDtype, dstDtype, srcData, dstData);
    RunTypeSwitch(numInterleavedChannels, numPlanes, srcDtype, dstDtype, nvcv::DataType{},
                  [&, brightness, contrast, brightnessShift, contrastCenter, clamp](auto dummySrcVal, auto dummyDstVal,
                                                                                    auto dummyArg, auto isPlanar)
                  {
                      using InT      = decltype(dummySrcVal);
                      using OutT     = decltype(dummyDstVal);
                      using ArgT     = decltype(dummyArg);
                      using IsPlanar = decltype(isPlanar);
                      auto args
                          = GetScalarBatchArgsWrap<ArgT>(brightness, contrast, brightnessShift, contrastCenter, clamp);
                      RunBrightnessContrast<IsPlanar::value, InT, OutT>(stream, *srcData, *dstData, args);
                  });
}

void BrightnessContrast::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                    const nvcv::ImageBatchVarShape &dst, double brightness, double contrast,
                                    double brightnessShift, double contrastCenter, bool clamp) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BrightnessContrast::operator()[ImageBatchVarShape,Scalar]");
    int            numSamples;
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType srcDtype;
    nvcv::DataType dstDtype;
    auto           srcDstData
        = validateSrcDstVarBatch(numSamples, numInterleavedChannels, numPlanes, srcDtype, dstDtype, stream, src, dst);
    RunTypeSwitch(numInterleavedChannels, numPlanes, srcDtype, dstDtype, nvcv::DataType{},
                  [&, brightness, contrast, brightnessShift, contrastCenter, clamp](auto dummySrcVal, auto dummyDstVal,
                                                                                    auto dummyArg, auto isPlanar)
                  {
                      using InT      = decltype(dummySrcVal);
                      using OutT     = decltype(dummyDstVal);
                      using ArgT     = decltype(dummyArg);
                      using IsPlanar = decltype(isPlanar);
                      auto args
                          = GetScalarBatchArgsWrap<ArgT>(brightness, contrast, brightnessShift, contrastCenter, clamp);
                      auto &[srcData, dstData] = srcDstData;
                      RunBrightnessContrast<IsPlanar::value, InT, OutT>(stream, *srcData, *dstData, args);
                  });
}

} // namespace cvcuda::priv

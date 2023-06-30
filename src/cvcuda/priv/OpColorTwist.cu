/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpColorTwist.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/cuda/DropCast.hpp>
#include <nvcv/cuda/ImageBatchVarShapeWrap.hpp>
#include <nvcv/cuda/MathOps.hpp>
#include <nvcv/cuda/SaturateCast.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <nvcv/cuda/TensorWrap.hpp>
#include <nvcv/cuda/math/LinAlg.hpp>
#include <util/Assert.h>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

template<typename T, int N>
using Vec = cuda::math::Vector<T, N>;
template<typename T, int N, int M>
using Mat = cuda::math::Matrix<T, N, M>;

// Load explicit affine transform matrix from a tensor
template<class TwistWrap>
inline auto __device__ GetAffineTransform(const TwistWrap &twist)
{
    using ValueType = std::remove_const_t<typename TwistWrap::ValueType>;
    using BT        = cuda::BaseType<ValueType>;
    static_assert(cuda::NumElements<ValueType> == 4);

    Mat<BT, 3, 4> affineTransform;
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        ValueType row;
        if constexpr (TwistWrap::kNumDimensions == 1)
        {
            row = twist[i];
        }
        else
        {
            static_assert(TwistWrap::kNumDimensions == 2);
            int  z = blockIdx.z;
            int2 coord{i, z};
            row = twist[coord];
        }
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            affineTransform[i][j] = cuda::GetElement(row, j);
        }
    }
    return affineTransform;
}

// Do actual transformation of a pixel by an affine transform
template<class SrcWrapper, class DstWrapper, int N, typename TwistT>
inline void __device__ DoAffineTransform(SrcWrapper src, DstWrapper dst, const int2 size,
                                         const Mat<TwistT, N, N + 1> transform)
{
    using SrcT                       = typename SrcWrapper::ValueType;
    using DstT                       = typename DstWrapper::ValueType;
    using T                          = cuda::BaseType<DstT>;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(std::is_same_v<T, std::remove_const_t<cuda::BaseType<SrcT>>>);
    static_assert(numChannels == cuda::NumElements<DstT>);
    static_assert(numChannels >= N);

    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }

    auto               src_pixel = src[coord];
    Vec<TwistT, N + 1> in_vec;
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        in_vec[i] = cuda::GetElement(src_pixel, i);
    }
    in_vec[N]              = 1;
    Vec<TwistT, 3> out_vec = transform * in_vec;
    DstT           out_pixel;
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        cuda::GetElement(out_pixel, i) = cuda::SaturateCast<T>(out_vec[i]);
    }
    // rewrite the extra channels unaffected
#pragma unroll
    for (int i = N; i < numChannels; i++)
    {
        cuda::GetElement(out_pixel, i) = cuda::GetElement(src_pixel, i);
    }
    dst[coord] = out_pixel;
}

// Load affine transform ----------------------------------------------------------

template<class SrcWrapper, class DstWrapper, typename ValueType>
inline __device__ void DoColorTwist(SrcWrapper src, DstWrapper dst, const int2 size,
                                    const cuda::Tensor1DWrap<const ValueType> param)
{
    static_assert(cuda::NumElements<ValueType> == 4);
    auto transform = GetAffineTransform(param);
    DoAffineTransform(src, dst, size, transform);
}

template<class SrcWrapper, class DstWrapper, typename ValueType>
inline __device__ void DoColorTwist(SrcWrapper src, DstWrapper dst, const int2 size,
                                    const cuda::Tensor2DWrap<const ValueType> param)
{
    static_assert(cuda::NumElements<ValueType> == 4);
    auto transform = GetAffineTransform(param);
    DoAffineTransform(src, dst, size, transform);
}

// ColorTwist kernel --------------------------------------------------------------

// Tensor variant
template<class SrcWrapper, class DstWrapper, class ColorTwistParam>
__global__ void ColorTwist(SrcWrapper src, DstWrapper dst, int2 size, const ColorTwistParam param)
{
    DoColorTwist(src, dst, size, param);
}

// VarBatch variant
template<class SrcWrapper, class DstWrapper, class ColorTwistParam>
__global__ void ColorTwist(SrcWrapper src, DstWrapper dst, const ColorTwistParam param)
{
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};

    DoColorTwist(src, dst, size, param);
}

// Run ColorTwist kernel ----------------------------------------------------------

template<typename T, class SrcData, class DstData, class ColorTwistParam>
inline void RunColorTwist(cudaStream_t stream, const SrcData &srcData, const DstData &dstData,
                          const ColorTwistParam &param)
{
    dim3 block(32, 4, 1);
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto srcAccess = nvcv::TensorDataAccessStridedImage::Create(srcData);
        int2 size      = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});
        dim3 grid(util::DivUp(size.x, block.x), util::DivUp(size.y, block.y), srcAccess->numSamples());

        auto src = cuda::CreateTensorWrapNHW<const T>(srcData);
        auto dst = cuda::CreateTensorWrapNHW<T>(dstData);
        ColorTwist<<<grid, block, 0, stream>>>(src, dst, size, param);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);
        int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};
        dim3 grid(util::DivUp(dstMaxSize.x, block.x), util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);

        cuda::ImageBatchVarShapeWrap<const T> src(srcData);
        cuda::ImageBatchVarShapeWrap<T>       dst(dstData);

        ColorTwist<<<grid, block, 0, stream>>>(src, dst, param);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

template<typename SrcDestT, typename TwistT, class SrcData, class DstData>
inline void RunColorTwist(cudaStream_t stream, const SrcData &srcData, const DstData &dstData,
                          const nvcv::TensorDataStridedCuda &twistData, bool hasPerSampleTwist)
{
    if (!hasPerSampleTwist)
    {
        auto twist = cuda::Tensor1DWrap<const TwistT>(twistData);
        RunColorTwist<SrcDestT>(stream, srcData, dstData, twist);
    }
    else
    {
        auto twist = cuda::Tensor2DWrap<const TwistT>(twistData);
        RunColorTwist<SrcDestT>(stream, srcData, dstData, twist);
    }
}

// Src/twist/dst type-switch
template<typename Cb>
inline void RunSrcTypeSwitch(int numChannels, nvcv::DataType srcType, nvcv::DataType twistType, Cb &&cb)
{
    // The channels of input sample and the width of transform tensor may be baked into the data type.

#define NVCV_RUN_COLOR_TWIST(NUM_CHANNELS, SRC_TYPE, TWIST_TYPE, SRC_VEC_TYPE, TWIST_VEC_TYPE) \
    ((numChannels == NUM_CHANNELS)                                                             \
     && (srcType == nvcv::TYPE_##SRC_TYPE || srcType == nvcv::TYPE_##NUM_CHANNELS##SRC_TYPE)   \
     && (twistType == nvcv::TYPE_##TWIST_TYPE || twistType == nvcv::TYPE_4##TWIST_TYPE))       \
        cb(SRC_VEC_TYPE{}, TWIST_VEC_TYPE{})

    // clang-format off
    if NVCV_RUN_COLOR_TWIST (3, U8, F32, uchar3, float4);
    else if NVCV_RUN_COLOR_TWIST (4, U8, F32, uchar4, float4);
    else if NVCV_RUN_COLOR_TWIST (3, U16, F32, ushort3, float4);
    else if NVCV_RUN_COLOR_TWIST (4, U16, F32, ushort4, float4);
    else if NVCV_RUN_COLOR_TWIST (3, S16, F32, short3, float4);
    else if NVCV_RUN_COLOR_TWIST (4, S16, F32, short4, float4);
    else if NVCV_RUN_COLOR_TWIST (3, U32, F64, uint3, double4);
    else if NVCV_RUN_COLOR_TWIST (4, U32, F64, uint4, double4);
    else if NVCV_RUN_COLOR_TWIST (3, S32, F64, int3, double4);
    else if NVCV_RUN_COLOR_TWIST (4, S32, F64, int4, double4);
    else if NVCV_RUN_COLOR_TWIST (3, F32, F32, float3, float4);
    else if NVCV_RUN_COLOR_TWIST (4, F32, F32, float4, float4);
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input/twist/output data types");
    }
    // clang-format on

#undef NVCV_RUN_COLOR_TWIST
}

// Argument validation helpers ----------------------------------------------------

inline auto validateSrcDstTensors(int &numSamples, int &numChannels, nvcv::DataType &srcDstDtype,
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

    auto srcAccess = nvcv::TensorDataAccessStridedImage::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImage::Create(*dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    numSamples = srcAccess->numSamples();
    if (numSamples != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    numChannels = srcAccess->numChannels();
    if (numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }

    if (numChannels != 3 && numChannels != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have 3 or 4 channels");
    }

    srcDstDtype = srcData->dtype();

    if (srcDstDtype != dstData->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output data type are different, but must be the same.");
    }

    if (srcData->layout() != dstData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }

    if (srcData->layout() != nvcv::TENSOR_HWC && srcData->layout() != nvcv::TENSOR_NHWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must have (N)HWC layout");
    }

    return srcAccess;
}

inline void validateSrcDstVarBatch(int &numSamples, int &numChannels, nvcv::DataType &srcDstDtype,
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

    numSamples = srcData->numImages();
    if (numSamples != dstData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    const auto &srcFormat = srcData->uniqueFormat();
    if (srcFormat.numPlanes() > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image batches must have (N)HWC layout");
    }

    srcDstDtype = srcFormat.planeDataType(0);

    numChannels = srcFormat.numChannels();
    if (numChannels != 3 && numChannels != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The input must have 3 or 4 channels");
    }

    if (srcFormat != dstData->uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output data type are different, but must be the same.");
    }
}

inline void validateTwistTensor(bool &hasPerSampleTwist, nvcv::DataType &twistDtype, int numImages,
                                const nvcv::Optional<nvcv::TensorDataStridedCuda> &twistData)
{
    if (!twistData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "The twist argument must be cuda-accessible, pitch-linear tensor");
    }

    twistDtype = twistData->dtype();

    if (twistDtype.numChannels() != 1 && twistDtype.numChannels() != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The twist transformation must be a 3x4 matrix");
    }

    int  rank               = twistData->rank();
    bool hasBakedInChannels = twistDtype.numChannels() > 1;
    int  numDataDims        = rank + hasBakedInChannels;
    hasPerSampleTwist       = numDataDims == 3;

    if (!hasPerSampleTwist && numDataDims != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The twist argument must be 2D or 3D tensor");
    }

    int numCols, numRows;
    if (hasPerSampleTwist)
    {
        if (numImages != twistData->shape(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "The twist must be 2D matrix or 3D tensor where the outermost dimenstion matches "
                                  "the input batch size");
        }
        numRows = twistData->shape(1);
        numCols = hasBakedInChannels ? 4 : twistData->shape(2);
    }
    else
    {
        numRows = twistData->shape(0);
        numCols = hasBakedInChannels ? 4 : twistData->shape(1);
    }

    if (numRows != 3 || numCols != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The twist must matrix must be 3x4");
    }
}

} // anonymous namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

ColorTwist::ColorTwist() {}

// Operator --------------------------------------------------------------------

// Tensor input variant
void ColorTwist::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                            const nvcv::Tensor &twist) const
{
    int            numSamples;
    int            numChannels;
    nvcv::DataType srcDstDtype;
    auto           srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    validateSrcDstTensors(numSamples, numChannels, srcDstDtype, srcData, dstData);

    bool           hasPerSampleTwist;
    nvcv::DataType twistDtype;
    auto           twistData = twist.exportData<nvcv::TensorDataStridedCuda>();
    validateTwistTensor(hasPerSampleTwist, twistDtype, numSamples, twistData);

    RunSrcTypeSwitch(numChannels, srcDstDtype, twistDtype,
                     [&](auto srcDummy, auto twistDummy)
                     {
                         using SrcDstT = decltype(srcDummy);
                         using TwistT  = decltype(twistDummy);
                         RunColorTwist<SrcDstT, TwistT>(stream, *srcData, *dstData, *twistData, hasPerSampleTwist);
                     });
}

// VarShape input variant
void ColorTwist::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                            const nvcv::ImageBatchVarShape &dst, const nvcv::Tensor &twist) const
{
    int            numSamples;
    int            numChannels;
    nvcv::DataType srcDstDtype;
    auto           srcData = src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    auto           dstData = dst.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    validateSrcDstVarBatch(numSamples, numChannels, srcDstDtype, srcData, dstData);

    bool           hasPerSampleTwist;
    nvcv::DataType twistDtype;
    auto           twistData = twist.exportData<nvcv::TensorDataStridedCuda>();
    validateTwistTensor(hasPerSampleTwist, twistDtype, numSamples, twistData);

    RunSrcTypeSwitch(numChannels, srcDstDtype, twistDtype,
                     [&](auto srcDummy, auto twistDummy)
                     {
                         using SrcDstT = decltype(srcDummy);
                         using TwistT  = decltype(twistDummy);
                         RunColorTwist<SrcDstT, TwistT>(stream, *srcData, *dstData, *twistData, hasPerSampleTwist);
                     });
}

} // namespace cvcuda::priv

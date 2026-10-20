/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

#include "Nvtx.hpp"
#include "OpBilateralFilter.hpp"
#include "TensorAccessValidation.hpp"

#include "BilateralFilterCommon.cuh"

#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/BorderVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/BorderWrap.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cstdint>
#include <type_traits>

namespace cvcuda::priv {

namespace cuda = nvcv::cuda;

namespace {

using namespace bilateral;

// The plain filter drives the range weight from the source itself, so it never loads a guide image
// and the guide arm of the shared tile compiles away.
constexpr bool kHasGuide = false;

// With no guide to load there is register budget left for the interior shortcut, which skips the
// border coordinate fixup on taps already inside the image. The tensor planar path is the one
// exception, below.
constexpr bool kFastInterior = true;

// The planar window is walked row-major here and column-major in the joint filter; each keeps the
// order its kernels have always used.
constexpr bool kRowOuter = true;

// =============================================================================
// Kernels
// =============================================================================

template<class SrcWrapper, class DstWrapper>
__global__ void BilateralFilterKernel(SrcWrapper src, DstWrapper dst, FilterParams params, int rows, int columns)
{
    using T = typename DstWrapper::ValueType;

    const int colIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    // float3 is the one instantiation that skips the tile-wide shortcut: its four float3
    // accumulators leave no register budget for the extra live flag. Its taps still take the
    // per-tap in-range shortcut inside ReadTap.
    const bool windowInside
        = !std::is_same_v<T, float3> && IsWindowInside(colIdx, rowIdx, columns, rows, params.radius);

    PackedTile<kHasGuide, kFastInterior>(src, src, dst, blockIdx.z, colIdx, rowIdx, rows, columns, params,
                                         windowInside);
}

template<typename T, NVCVBorderType B, int CHANNELS>
__global__ void BilateralFilterPlanarKernel(StridedPlaneView<T> src, StridedPlaneWriteView<T> dst, FilterParams params,
                                            int rows, int columns)
{
    const int colIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    // 4-channel float is this path's register-pressure ceiling -- four centres times four channels
    // of accumulator -- so it drops the interior shortcut and lets the border wrap fix up every tap,
    // the same trade the interleaved kernel makes for float3.
    constexpr bool kPlanarFastInterior = !(std::is_same_v<T, float> && CHANNELS == 4);

    const auto srcSample = src.AtSample(blockIdx.z);

    PlanarTile<T, CHANNELS, B, kHasGuide, kPlanarFastInterior, kRowOuter>(
        srcSample, srcSample, dst.AtSample(blockIdx.z), colIdx, rowIdx, rows, columns, params);
}

template<class SrcWrapper, class DstWrapper>
__global__ void BilateralFilterVarShapeKernel(SrcWrapper src, DstWrapper dst, VarShapeFilterParams varParams)
{
    const int batchIdx = blockIdx.z;
    const int rows     = dst.height(batchIdx);
    const int columns  = dst.width(batchIdx);

    const FilterParams params = MakeFilterParams(varParams, batchIdx);

    const int colIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    PackedTile<kHasGuide, kFastInterior>(src, src, dst, batchIdx, colIdx, rowIdx, rows, columns, params,
                                         IsWindowInside(colIdx, rowIdx, columns, rows, params.radius));
}

template<NVCVBorderType B, int CHANNELS, class SrcWrapper, class DstWrapper>
__global__ __launch_bounds__(64, CHANNELS == 4 ? 12 : 16) void BilateralFilterVarShapePlanarKernel(
    SrcWrapper src, DstWrapper dst, VarShapeFilterParams varParams)
{
    using T = typename DstWrapper::ValueType;

    const int batchIdx = blockIdx.z;
    const int rows     = dst.height(batchIdx);
    const int columns  = dst.width(batchIdx);

    const FilterParams params = MakeFilterParams(varParams, batchIdx);

    const int colIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    const auto srcView = MakePlaneArrayView<T, CHANNELS>(src, batchIdx);
    const auto dstView = MakePlaneArrayWriteView<T, CHANNELS>(dst, batchIdx);

    if constexpr (B == NVCV_BORDER_REFLECT && CHANNELS == 3)
    {
        if (params.radius == 2)
        {
            PlanarTile<T, CHANNELS, B, kHasGuide, kFastInterior, kRowOuter, 2>(srcView, srcView, dstView, colIdx,
                                                                               rowIdx, rows, columns, params);
            return;
        }
    }

    PlanarTile<T, CHANNELS, B, kHasGuide, kFastInterior, kRowOuter>(srcView, srcView, dstView, colIdx, rowIdx, rows,
                                                                    columns, params);
}

// =============================================================================
// Launch
// =============================================================================

template<typename T, NVCVBorderType B>
void LaunchTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData, int batch,
                  int rows, int columns, const FilterParams &params, cudaStream_t stream)
{
    const dim3 block = RowTileBlock();
    const dim3 grid  = TileGrid(block, columns, rows, batch);

    auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(0.f));
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    BilateralFilterKernel<<<grid, block, 0, stream>>>(src, dst, params, rows, columns);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B, int CHANNELS>
void LaunchTensorPlanar(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                        const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                        const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int batch, int rows, int columns,
                        const FilterParams &params, cudaStream_t stream)
{
    const dim3 block = RowTileBlock();
    const dim3 grid  = TileGrid(block, columns, rows, batch);

    BilateralFilterPlanarKernel<T, B, CHANNELS><<<grid, block, 0, stream>>>(
        MakePlaneView<T>(inData, inAccess), MakePlaneWriteView<T>(outData, outAccess), params, rows, columns);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B>
void LaunchVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                    const nvcv::ImageBatchVarShapeDataStridedCuda &outData, int batch,
                    const VarShapeFilterParams &varParams, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData);
    cuda::ImageBatchVarShapeWrap<T>      dst(outData);

    const nvcv::Size2D maxSize = outData.maxSize();
    const dim3         block   = RowTileBlock();
    const dim3         grid    = TileGrid(block, maxSize.w, maxSize.h, batch);

    BilateralFilterVarShapeKernel<<<grid, block, 0, stream>>>(src, dst, varParams);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B, int CHANNELS>
void LaunchVarShapePlanar(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                          const nvcv::ImageBatchVarShapeDataStridedCuda &outData, int batch,
                          const VarShapeFilterParams &varParams, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData);
    cuda::ImageBatchVarShapeWrap<T>      dst(outData);

    const nvcv::Size2D maxSize = outData.maxSize();
    const dim3         block   = RowTileBlock();
    const dim3         grid    = TileGrid(block, maxSize.w, maxSize.h, batch);

    BilateralFilterVarShapePlanarKernel<B, CHANNELS><<<grid, block, 0, stream>>>(src, dst, varParams);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// =============================================================================
// Validation and dispatch
// =============================================================================

inline void RunTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                      int diameter, float sigmaColor, float sigmaSpace, NVCVBorderType borderMode, cudaStream_t stream)
{
    const LayoutCode inLayout  = ClassifyLayout(inData.layout());
    const LayoutCode outLayout = ClassifyLayout(outData.layout());

    ValidateMatchingDataType(inData.dtype(), outData.dtype(), "Input");
    ValidateMatchingLayout(inLayout, outLayout, "Input");

    const bool isPlanar = IsPlanarLayout(inLayout);

    ValidateBorderMode(borderMode);

    const DataTypeCode dataType = ClassifyDataType(outData.dtype());
    ValidateSupportedDataType(dataType);

    const FilterParams params = MakeFilterParams(diameter, sigmaColor, sigmaSpace);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be an image-shaped tensor");
    }
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be an image-shaped tensor");
    }

    ValidateSameImageShape(*inAccess, *outAccess, "Input");

    const int batch    = static_cast<int>(inAccess->numSamples());
    const int channels = inAccess->numChannels();
    const int rows     = inAccess->numRows();
    const int columns  = inAccess->numCols();

    ValidateChannelCount(channels, isPlanar, "BilateralFilter");
    RequireInt32Addressable(*inAccess);

    DispatchTile(
        borderMode, dataType, channels, isPlanar,
        [&](auto borderTag, auto pixelTag)
        {
            LaunchTensor<decltype(pixelTag), decltype(borderTag)::value>(inData, outData, batch, rows, columns, params,
                                                                         stream);
        },
        [&](auto borderTag, auto baseTag, auto channelTag)
        {
            LaunchTensorPlanar<decltype(baseTag), decltype(borderTag)::value, decltype(channelTag)::value>(
                inData, outData, *inAccess, *outAccess, batch, rows, columns, params, stream);
        });
}

inline void RunVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                        const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                        const nvcv::TensorDataStridedCuda             &diameterData,
                        const nvcv::TensorDataStridedCuda             &sigmaColorData,
                        const nvcv::TensorDataStridedCuda &sigmaSpaceData, NVCVBorderType borderMode,
                        cudaStream_t stream)
{
    if (!inData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in the input varshape must all have the same format");
    }

    const LayoutCode inLayout  = ClassifyLayout(inData);
    const LayoutCode outLayout = ClassifyLayout(outData);

    ValidateMatchingLayout(inLayout, outLayout, "Input");

    const bool isPlanar = IsPlanarLayout(inLayout);

    ValidateMatchingFormat(inData.uniqueFormat(), outData.uniqueFormat(), "Input");
    ValidateBorderMode(borderMode);

    const DataTypeCode dataType = ClassifyDataType(outData.uniqueFormat());
    ValidateSupportedDataType(dataType);

    ValidateFilterParamTensors(diameterData, sigmaColorData, sigmaSpaceData);
    ValidateMatchingImageCount(inData.numImages(), outData.numImages(), "Input");

    const int batch    = inData.numImages();
    const int channels = inData.uniqueFormat().numChannels();

    ValidateChannelCount(channels, isPlanar, "BilateralFilter");

    const VarShapeFilterParams varParams{cuda::Tensor1DWrap<int>(diameterData),
                                         cuda::Tensor1DWrap<float>(sigmaColorData),
                                         cuda::Tensor1DWrap<float>(sigmaSpaceData)};

    DispatchTile(
        borderMode, dataType, channels, isPlanar,
        [&](auto borderTag, auto pixelTag)
        { LaunchVarShape<decltype(pixelTag), decltype(borderTag)::value>(inData, outData, batch, varParams, stream); },
        [&](auto borderTag, auto baseTag, auto channelTag)
        {
            LaunchVarShapePlanar<decltype(baseTag), decltype(borderTag)::value, decltype(channelTag)::value>(
                inData, outData, batch, varParams, stream);
        });
}

} // namespace

BilateralFilter::BilateralFilter() = default;

void BilateralFilter::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, int diameter,
                                 float sigmaColor, float sigmaSpace, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BilateralFilter::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    RunTensor(*inData, *outData, diameter, sigmaColor, sigmaSpace, borderMode, stream);
}

void BilateralFilter::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                 const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &diameter,
                                 const nvcv::Tensor &sigmaColor, const nvcv::Tensor &sigmaSpace,
                                 NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BilateralFilter::operator()[ImageBatchVarShape]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be device-accessible, varshape image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-accessible,  varshape image batch");
    }

    auto diameterData = diameter.exportData<nvcv::TensorDataStridedCuda>();
    if (diameterData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Diameter must be device-accessible, pitch-linear tensor");
    }

    auto sigmaColorData = sigmaColor.exportData<nvcv::TensorDataStridedCuda>();
    if (sigmaColorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "sigmaColor must be device-accessible, pitch-linear tensor");
    }

    auto sigmaSpaceData = sigmaSpace.exportData<nvcv::TensorDataStridedCuda>();
    if (sigmaSpaceData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "sigmaSpace must be device-accessible, pitch-linear tensor");
    }

    // The kernel derives its interior window from the destination extent and takes the fast interior
    // path unconditionally, so a larger output image would sample past its source. Read the sizes
    // off the handles: the strided batch data keeps its image list in device memory.
    if (in.numImages() == out.numImages())
    {
        cvcuda::priv::RequireSameVarShapeImageSizes(in, out);
    }

    RunVarShape(*inData, *outData, *diameterData, *sigmaColorData, *sigmaSpaceData, borderMode, stream);
}

} // namespace cvcuda::priv

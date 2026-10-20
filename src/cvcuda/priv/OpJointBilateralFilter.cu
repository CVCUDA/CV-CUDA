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
#include "OpJointBilateralFilter.hpp"
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

// The range weight comes from the guide image, which is the whole difference from the plain filter:
// the accumulated value still comes from the source.
constexpr bool kHasGuide = true;

// Every tap already costs two wide loads. The interior shortcut would add live state on top of an
// accumulator set that is twice the plain filter's, so taps go through the border wrap unconditionally.
constexpr bool kFastInterior = false;

// The planar window is walked column-major here and row-major in the plain filter; each keeps the
// order its kernels have always used.
constexpr bool kRowOuter = false;

// A 2x2 output tile per thread, 64 threads either way. The row-shaped tile keeps stores contiguous;
// the square one covers a squarer region, so neighbouring threads' windows overlap far more. That
// pays exactly when a tap is expensive, which for the joint filter is a 4-byte base type carrying
// more than one element -- two 8-to-16-byte loads per tap against one for the guide-free filter.
template<typename T>
inline dim3 PackedTileBlock()
{
    using BT                  = cuda::BaseType<T>;
    constexpr int  kElements  = cuda::NumElements<T>;
    constexpr bool kRowShaped = !(sizeof(BT) == 4 && kElements > 1);

    return kRowShaped ? RowTileBlock() : dim3{8, 8};
}

// =============================================================================
// Kernels
// =============================================================================

template<class SrcWrapper, class DstWrapper>
__global__ void JointBilateralFilterKernel(SrcWrapper src, SrcWrapper guide, DstWrapper dst, FilterParams params,
                                           int rows, int columns)
{
    const int colIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    PackedTile<kHasGuide, kFastInterior>(src, guide, dst, blockIdx.z, colIdx, rowIdx, rows, columns, params, false);
}

template<typename T, NVCVBorderType B, int CHANNELS>
__global__ void JointBilateralFilterPlanarKernel(StridedPlaneView<T> src, StridedPlaneView<T> guide,
                                                 StridedPlaneWriteView<T> dst, FilterParams params, int rows,
                                                 int columns)
{
    const int colIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    PlanarTile<T, CHANNELS, B, kHasGuide, kFastInterior, kRowOuter, 0, true>(
        src.AtSample(blockIdx.z), guide.AtSample(blockIdx.z), dst.AtSample(blockIdx.z), colIdx, rowIdx, rows, columns,
        params);
}

struct SharedPlanarTileView
{
    const float *base;

    inline __device__ float Read(int channel, int row, int column) const
    {
        return base[channel * 720 + row * 36 + column];
    }
};

template<typename T>
struct SharedPlanarTileOutput
{
    StridedPlaneWriteView<T> destination;
    int64_t                  columnOrigin;
    int64_t                  rowOrigin;
    int                      rows;
    int                      columns;

    inline __device__ void Write(int channel, int row, int column, T value) const
    {
        const int64_t globalColumn = columnOrigin + column - 2;
        const int64_t globalRow    = rowOrigin + row - 2;
        if (globalColumn >= 0 && globalColumn < columns && globalRow >= 0 && globalRow < rows)
        {
            destination.Write(channel, static_cast<int>(globalRow), static_cast<int>(globalColumn), value);
        }
    }
};

template<typename T>
__global__ void __launch_bounds__(128)
    JointBilateralFilterSharedPlanarKernel(StridedPlaneView<T> src, StridedPlaneView<T> guide,
                                           StridedPlaneWriteView<T> dst, FilterParams params, int rows, int columns)
{
    static_assert(std::is_same_v<T, float>);
    __shared__ float   tile[4320];
    __shared__ int64_t rowOffset[2][20];
    __shared__ int64_t columnOffset[2][36];

    const int64_t columnOrigin = static_cast<int64_t>(blockIdx.x) * 32;
    const int64_t rowOrigin    = static_cast<int64_t>(blockIdx.y) * 16;
    const int     thread       = threadIdx.y * 32 + threadIdx.x;
    const auto    sourceSample = src.AtSample(blockIdx.z);
    const auto    guideSample  = guide.AtSample(blockIdx.z);

    if (thread < 36)
    {
        const int64_t column
            = cuda::GetIndexWithBorder<NVCV_BORDER_REFLECT>(columnOrigin - 2 + thread, static_cast<int64_t>(columns));
        columnOffset[0][thread] = column * sourceSample.colStride;
        columnOffset[1][thread] = column * guideSample.colStride;
    }
    else if (thread < 56)
    {
        const int     localRow = thread - 36;
        const int64_t row
            = cuda::GetIndexWithBorder<NVCV_BORDER_REFLECT>(rowOrigin - 2 + localRow, static_cast<int64_t>(rows));
        rowOffset[0][localRow] = row * sourceSample.rowStride;
        rowOffset[1][localRow] = row * guideSample.rowStride;
    }
    __syncthreads();

    for (int element = thread; element < 720; element += 128)
    {
        const int     row          = element / 36;
        const int     column       = element % 36;
        const int64_t sourceOffset = rowOffset[0][row] + columnOffset[0][column];
        const int64_t guideOffset  = rowOffset[1][row] + columnOffset[1][column];
#pragma unroll
        for (int channel = 0; channel < 3; ++channel)
        {
            tile[channel * 720 + element] = static_cast<float>(
                *reinterpret_cast<const T *>(sourceSample.base + channel * sourceSample.chStride + sourceOffset));
            tile[(3 + channel) * 720 + element] = static_cast<float>(
                *reinterpret_cast<const T *>(guideSample.base + channel * guideSample.chStride + guideOffset));
        }
    }
    __syncthreads();

    if (columnOrigin + threadIdx.x >= columns || rowOrigin + threadIdx.y * 4 >= rows)
    {
        return;
    }

    const SharedPlanarTileView      sourceTile{tile};
    const SharedPlanarTileView      guideTile{tile + 3 * 720};
    const SharedPlanarTileOutput<T> output{dst.AtSample(blockIdx.z), columnOrigin, rowOrigin, rows, columns};
    PlanarTile<T, 3, NVCV_BORDER_REFLECT, true, true, false, 2, true, 1>(sourceTile, guideTile, output, 2 + threadIdx.x,
                                                                         2 + threadIdx.y * 4, 20, 36, params);
}

template<class SrcWrapper, class DstWrapper>
__global__ void JointBilateralFilterVarShapeKernel(SrcWrapper src, SrcWrapper guide, DstWrapper dst,
                                                   VarShapeFilterParams varParams)
{
    const int batchIdx = blockIdx.z;
    const int rows     = dst.height(batchIdx);
    const int columns  = dst.width(batchIdx);

    const FilterParams params = MakeFilterParams(varParams, batchIdx);

    const int colIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    PackedTile<kHasGuide, kFastInterior>(src, guide, dst, batchIdx, colIdx, rowIdx, rows, columns, params, false);
}

template<NVCVBorderType B, int CHANNELS, class SrcWrapper, class DstWrapper>
__global__ void JointBilateralFilterVarShapePlanarKernel(SrcWrapper src, SrcWrapper guide, DstWrapper dst,
                                                         VarShapeFilterParams varParams)
{
    using T = typename DstWrapper::ValueType;

    const int batchIdx = blockIdx.z;
    const int rows     = dst.height(batchIdx);
    const int columns  = dst.width(batchIdx);

    const FilterParams params = MakeFilterParams(varParams, batchIdx);

    const int colIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    PlanarTile<T, CHANNELS, B, kHasGuide, kFastInterior, kRowOuter>(
        MakePlaneArrayView<T, CHANNELS>(src, batchIdx), MakePlaneArrayView<T, CHANNELS>(guide, batchIdx),
        MakePlaneArrayWriteView<T, CHANNELS>(dst, batchIdx), colIdx, rowIdx, rows, columns, params);
}

// =============================================================================
// Launch
// =============================================================================

template<typename T, NVCVBorderType B>
void LaunchTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &inColorData,
                  const nvcv::TensorDataStridedCuda &outData, int batch, int rows, int columns,
                  const FilterParams &params, cudaStream_t stream)
{
    const bool useWidePackedBlock = std::is_same_v<T, float3> && B == NVCV_BORDER_REFLECT && params.radius == 2
                                 && inData.layout() == nvcv::TENSOR_NHWC;
    const dim3 block = useWidePackedBlock ? dim3{16, 8} : PackedTileBlock<T>();
    const dim3 grid  = TileGrid(block, columns, rows, batch);

    auto src   = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(0.f));
    auto guide = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inColorData, cuda::SetAll<T>(0.f));
    auto dst   = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    JointBilateralFilterKernel<<<grid, block, 0, stream>>>(src, guide, dst, params, rows, columns);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B, int CHANNELS>
void LaunchTensorPlanar(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &inColorData,
                        const nvcv::TensorDataStridedCuda              &outData,
                        const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                        const nvcv::TensorDataAccessStridedImagePlanar &inColorAccess,
                        const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int batch, int rows, int columns,
                        const FilterParams &params, cudaStream_t stream)
{
    if constexpr (std::is_same_v<T, float> && CHANNELS == 3 && B == NVCV_BORDER_REFLECT)
    {
        if (params.radius == 2 && rows > 0 && columns > 0)
        {
            const dim3 sharedBlock{32, 4};
            const dim3 sharedGrid{static_cast<unsigned>((columns - 1) / 32 + 1),
                                  static_cast<unsigned>((rows - 1) / 16 + 1), static_cast<unsigned>(batch)};
            JointBilateralFilterSharedPlanarKernel<T><<<sharedGrid, sharedBlock, 0, stream>>>(
                MakePlaneView<T>(inData, inAccess), MakePlaneView<T>(inColorData, inColorAccess),
                MakePlaneWriteView<T>(outData, outAccess), params, rows, columns);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }
    }
    const bool useTallBlock
        = std::is_same_v<T, float> && CHANNELS == 3 && B == NVCV_BORDER_REFLECT && params.radius == 2;
    const dim3 block = useTallBlock ? dim3{32, 4} : RowTileBlock();
    const dim3 grid  = TileGrid(block, columns, rows, batch);

    JointBilateralFilterPlanarKernel<T, B, CHANNELS>
        <<<grid, block, 0, stream>>>(MakePlaneView<T>(inData, inAccess), MakePlaneView<T>(inColorData, inColorAccess),
                                     MakePlaneWriteView<T>(outData, outAccess), params, rows, columns);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B>
void LaunchVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                    const nvcv::ImageBatchVarShapeDataStridedCuda &inColorData,
                    const nvcv::ImageBatchVarShapeDataStridedCuda &outData, int batch,
                    const VarShapeFilterParams &varParams, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData);
    cuda::BorderVarShapeWrap<const T, B> guide(inColorData);
    cuda::ImageBatchVarShapeWrap<T>      dst(outData);

    const nvcv::Size2D maxSize = outData.maxSize();
    const dim3         block   = PackedTileBlock<T>();
    const dim3         grid    = TileGrid(block, maxSize.w, maxSize.h, batch);

    JointBilateralFilterVarShapeKernel<<<grid, block, 0, stream>>>(src, guide, dst, varParams);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B, int CHANNELS>
void LaunchVarShapePlanar(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                          const nvcv::ImageBatchVarShapeDataStridedCuda &inColorData,
                          const nvcv::ImageBatchVarShapeDataStridedCuda &outData, int batch,
                          const VarShapeFilterParams &varParams, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData);
    cuda::BorderVarShapeWrap<const T, B> guide(inColorData);
    cuda::ImageBatchVarShapeWrap<T>      dst(outData);

    const nvcv::Size2D maxSize = outData.maxSize();
    const dim3         block   = RowTileBlock();
    const dim3         grid    = TileGrid(block, maxSize.w, maxSize.h, batch);

    JointBilateralFilterVarShapePlanarKernel<B, CHANNELS><<<grid, block, 0, stream>>>(src, guide, dst, varParams);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// =============================================================================
// Validation and dispatch
// =============================================================================

inline void RunTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &inColorData,
                      const nvcv::TensorDataStridedCuda &outData, int diameter, float sigmaColor, float sigmaSpace,
                      NVCVBorderType borderMode, cudaStream_t stream)
{
    const LayoutCode inLayout      = ClassifyLayout(inData.layout());
    const LayoutCode inColorLayout = ClassifyLayout(inColorData.layout());
    const LayoutCode outLayout     = ClassifyLayout(outData.layout());

    ValidateMatchingDataType(inData.dtype(), outData.dtype(), "Input");
    ValidateMatchingDataType(inColorData.dtype(), outData.dtype(), "InputColor");
    ValidateMatchingLayout(inLayout, outLayout, "Input");
    ValidateMatchingLayout(inColorLayout, outLayout, "InputColor");

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
    auto inColorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inColorData);
    if (!inColorAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "InputColor must be an image-shaped tensor");
    }
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be an image-shaped tensor");
    }

    // Covered by OpJointBilateralFilter_Negative.rejects_mismatched_output_shape and
    // .rejects_mismatched_color_shape.
    ValidateSameImageShape(*inAccess, *outAccess, "Input");
    ValidateSameImageShape(*inColorAccess, *outAccess, "InputColor");

    const int batch    = static_cast<int>(inAccess->numSamples());
    const int channels = inAccess->numChannels();
    const int rows     = inAccess->numRows();
    const int columns  = inAccess->numCols();

    ValidateChannelCount(channels, isPlanar, "JointBilateralFilter");
    RequireInt32Addressable(*inAccess);

    DispatchTile(
        borderMode, dataType, channels, isPlanar,
        [&](auto borderTag, auto pixelTag)
        {
            LaunchTensor<decltype(pixelTag), decltype(borderTag)::value>(inData, inColorData, outData, batch, rows,
                                                                         columns, params, stream);
        },
        [&](auto borderTag, auto baseTag, auto channelTag)
        {
            LaunchTensorPlanar<decltype(baseTag), decltype(borderTag)::value, decltype(channelTag)::value>(
                inData, inColorData, outData, *inAccess, *inColorAccess, *outAccess, batch, rows, columns, params,
                stream);
        });
}

inline void RunVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                        const nvcv::ImageBatchVarShapeDataStridedCuda &inColorData,
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

    if (!inColorData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in the inputColor varshape must all have the same format");
    }

    const LayoutCode inLayout      = ClassifyLayout(inData);
    const LayoutCode inColorLayout = ClassifyLayout(inColorData);
    const LayoutCode outLayout     = ClassifyLayout(outData);

    ValidateMatchingLayout(inLayout, outLayout, "Input");
    ValidateMatchingLayout(inColorLayout, outLayout, "InputColor");

    const bool isPlanar = IsPlanarLayout(inLayout);

    ValidateMatchingFormat(inData.uniqueFormat(), outData.uniqueFormat(), "Input");
    ValidateMatchingFormat(inColorData.uniqueFormat(), outData.uniqueFormat(), "InputColor");
    ValidateBorderMode(borderMode);

    const DataTypeCode dataType = ClassifyDataType(outData.uniqueFormat());
    ValidateSupportedDataType(dataType);

    ValidateFilterParamTensors(diameterData, sigmaColorData, sigmaSpaceData);
    ValidateMatchingImageCount(inData.numImages(), outData.numImages(), "Input");
    // Covered by OpJointBilateralFilter_Negative.varshape_rejects_mismatched_color_image_count.
    ValidateMatchingImageCount(inColorData.numImages(), outData.numImages(), "InputColor");

    const int batch    = inData.numImages();
    const int channels = inData.uniqueFormat().numChannels();

    ValidateChannelCount(channels, isPlanar, "JointBilateralFilter");

    const VarShapeFilterParams varParams{cuda::Tensor1DWrap<int>(diameterData),
                                         cuda::Tensor1DWrap<float>(sigmaColorData),
                                         cuda::Tensor1DWrap<float>(sigmaSpaceData)};

    DispatchTile(
        borderMode, dataType, channels, isPlanar,
        [&](auto borderTag, auto pixelTag)
        {
            LaunchVarShape<decltype(pixelTag), decltype(borderTag)::value>(inData, inColorData, outData, batch,
                                                                           varParams, stream);
        },
        [&](auto borderTag, auto baseTag, auto channelTag)
        {
            LaunchVarShapePlanar<decltype(baseTag), decltype(borderTag)::value, decltype(channelTag)::value>(
                inData, inColorData, outData, batch, varParams, stream);
        });
}

} // namespace

JointBilateralFilter::JointBilateralFilter() = default;

void JointBilateralFilter::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &inColor,
                                      const nvcv::Tensor &out, int diameter, float sigmaColor, float sigmaSpace,
                                      NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::JointBilateralFilter::operator()[Tensor]");
    auto inData      = RequireCudaTensor(in, "Input");
    auto inColorData = RequireCudaTensor(inColor, "InputColor");
    auto outData     = RequireCudaTensor(out, "Output");

    RunTensor(*inData, *inColorData, *outData, diameter, sigmaColor, sigmaSpace, borderMode, stream);
}

void JointBilateralFilter::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                      const nvcv::ImageBatchVarShape &inColor, const nvcv::ImageBatchVarShape &out,
                                      const nvcv::Tensor &diameter, const nvcv::Tensor &sigmaColor,
                                      const nvcv::Tensor &sigmaSpace, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::JointBilateralFilter::operator()[ImageBatchVarShape]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "in must be a device-accessible, varshape image batch");
    }

    auto inColorData = inColor.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inColorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "inColor must be device-accessible, varshape image batch");
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

    RunVarShape(*inData, *inColorData, *outData, *diameterData, *sigmaColorData, *sigmaSpaceData, borderMode, stream);
}

} // namespace cvcuda::priv

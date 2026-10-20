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

/**
 * @file BilateralFilterCommon.cuh
 *
 * @brief Window math, validation and dispatch shared by BilateralFilter and JointBilateralFilter.
 */

#ifndef CVCUDA_PRIV_BILATERAL_FILTER_COMMON_CUH
#define CVCUDA_PRIV_BILATERAL_FILTER_COMMON_CUH

#include "LegacyFormatClassify.hpp"
#include "TensorAccessValidation.hpp"

#include "PixelLayoutCommon.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/BorderWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/BorderType.h>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Math.hpp>

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace cvcuda::priv::bilateral {

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

// =============================================================================
// Input classification
//
// The two operators run the same gates in the same order on the same values; only the number of
// images they are applied to differs, so the gates live here and the sequences stay in each
// operator's translation unit, where their order is observable.
// =============================================================================

// Element classification by data kind plus bits-per-channel, which is what the legacy
// GetLegacyDataType classified on. A packed type therefore lands on the same entry as its scalar
// sibling (TYPE_3U8 classifies as kU8), and any (kind, width) pair without an entry here is
// rejected before the operator's own supported-type list is consulted.
enum class DataTypeCode
{
    kU8,
    kS8,
    kU16,
    kS16,
    kS32,
    kF16,
    kF32,
    kF64,
};

inline DataTypeCode ClassifyDataType(nvcv::DataType dtype)
{
    auto bpc = dtype.bitsPerChannel();
    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    switch (dtype.dataKind())
    {
    case nvcv::DataKind::UNSIGNED:
        if (bpc[0] == 8)
        {
            return DataTypeCode::kU8;
        }
        if (bpc[0] == 16)
        {
            return DataTypeCode::kU16;
        }
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ",
                              bpc[0]);

    case nvcv::DataKind::SIGNED:
        if (bpc[0] == 8)
        {
            return DataTypeCode::kS8;
        }
        if (bpc[0] == 16)
        {
            return DataTypeCode::kS16;
        }
        if (bpc[0] == 32)
        {
            return DataTypeCode::kS32;
        }
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc[0]);

    case nvcv::DataKind::FLOAT:
        if (bpc[0] == 64)
        {
            return DataTypeCode::kF64;
        }
        if (bpc[0] == 32)
        {
            return DataTypeCode::kF32;
        }
        if (bpc[0] == 16)
        {
            return DataTypeCode::kF16;
        }
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc[0]);

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only floating-point, signed integer and unsigned integer data kinds are supported ");
    }
}

using cvcuda::priv::legacy_format::RequireUniformPlaneDataType;
using cvcuda::priv::legacy_format::RequireUniformVarShapeFormat;

inline DataTypeCode ClassifyDataType(nvcv::ImageFormat fmt)
{
    RequireUniformPlaneDataType(fmt);

    return ClassifyDataType(fmt.planeDataType(0));
}

inline bool IsSupportedDataType(DataTypeCode code)
{
    return code == DataTypeCode::kU8 || code == DataTypeCode::kU16 || code == DataTypeCode::kS16
        || code == DataTypeCode::kS32 || code == DataTypeCode::kF32 || code == DataTypeCode::kF16;
}

// Image layouts the operators accept; anything else is rejected while classifying. The numbering is
// pinned because it reaches the user through the "%d" in the layout-mismatch rejections.
enum class LayoutCode : int
{
    kNCHW = 0,
    kNHWC = 1,
    kCHW  = 2,
    kHWC  = 3,
};

inline bool IsPlanarLayout(LayoutCode layout)
{
    return layout == LayoutCode::kNCHW || layout == LayoutCode::kCHW;
}

inline LayoutCode ClassifyLayout(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW)
    {
        return LayoutCode::kNCHW;
    }
    if (layout == nvcv::TENSOR_CHW)
    {
        return LayoutCode::kCHW;
    }
    if (layout == nvcv::TENSOR_NHWC)
    {
        return LayoutCode::kNHWC;
    }
    if (layout == nvcv::TENSOR_HWC)
    {
        return LayoutCode::kHWC;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
}

inline LayoutCode ClassifyLayout(const nvcv::ImageBatchVarShapeDataStridedCuda &batch)
{
    nvcv::ImageFormat fmt = RequireUniformVarShapeFormat(batch);

    if (fmt.numPlanes() >= 2)
    {
        return batch.numImages() >= 2 ? LayoutCode::kNCHW : LayoutCode::kCHW;
    }

    return batch.numImages() >= 2 ? LayoutCode::kNHWC : LayoutCode::kHWC;
}

using cvcuda::priv::ValidateBorderMode;

// The kernels take their extent from the input and index the output with it, so a differing output
// shape addresses memory the output does not own.
inline void ValidateSameImageShape(const nvcv::TensorDataAccessStridedImagePlanar &input,
                                   const nvcv::TensorDataAccessStridedImagePlanar &output, const char *inputName)
{
    if (input.numSamples() != output.numSamples() || input.numChannels() != output.numChannels()
        || input.numRows() != output.numRows() || input.numCols() != output.numCols())
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT, "%s/output shape is different %dx%dx%dx%d/%dx%dx%dx%d", inputName,
            static_cast<int>(input.numSamples()), input.numChannels(), input.numRows(), input.numCols(),
            static_cast<int>(output.numSamples()), output.numChannels(), output.numRows(), output.numCols());
    }
}

inline void ValidateSupportedDataType(DataTypeCode code)
{
    if (!IsSupportedDataType(code))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "[Error] Invalid DataType %d",
                              static_cast<int>(code));
    }
}

inline void ValidateMatchingDataType(nvcv::DataType input, nvcv::DataType output, const char *inputName)
{
    if (input != output)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s and Output formats must be same input format = %lu output format = %lu", inputName,
                              static_cast<unsigned long>(static_cast<NVCVDataType>(input)),
                              static_cast<unsigned long>(static_cast<NVCVDataType>(output)));
    }
}

inline void ValidateMatchingFormat(nvcv::ImageFormat input, nvcv::ImageFormat output, const char *inputName)
{
    if (input != output)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s and Output formats must be same input format = %lu output format = %lu", inputName,
                              static_cast<unsigned long>(input.cvalue()), static_cast<unsigned long>(output.cvalue()));
    }
}

inline void ValidateMatchingLayout(LayoutCode input, LayoutCode output, const char *inputName)
{
    if (input != output)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s data format (%d) and output data format (%d) must be the same.", inputName,
                              static_cast<int>(input), static_cast<int>(output));
    }
}

inline void ValidateMatchingImageCount(int input, int output, const char *inputName)
{
    if (input != output)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s and Output data must have the same number of images (%d != %d)", inputName, input,
                              output);
    }
}

// Interleaved pixels are a 1- to 4-element vector; a planar image additionally needs one plane per
// channel, and there is no 2-plane packing to filter.
inline void ValidateChannelCount(int channels, bool isPlanar, const char *opName)
{
    if (channels > 4 || channels < 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number ch = %d", channels);
    }
    if (isPlanar && channels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Planar %s does not support 2-channel images",
                              opName);
    }
}

using cvcuda::priv::RequireImageAccess;
using cvcuda::priv::RequireInt32Addressable;

// The var-shape paths read diameter as int and both sigmas as float straight out of device tensors,
// so a mismatched element type would reinterpret the bits rather than convert them.
inline void ValidateParamTensorType(const nvcv::TensorDataStridedCuda &data, DataTypeCode expected, const char *name)
{
    const DataTypeCode code = ClassifyDataType(data.dtype());
    if (code != expected)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "[Error] Invalid %s DataType %d", name,
                              static_cast<int>(code));
    }
}

inline void ValidateFilterParamTensors(const nvcv::TensorDataStridedCuda &diameter,
                                       const nvcv::TensorDataStridedCuda &sigmaColor,
                                       const nvcv::TensorDataStridedCuda &sigmaSpace)
{
    ValidateParamTensorType(diameter, DataTypeCode::kS32, "diameterData");
    ValidateParamTensorType(sigmaColor, DataTypeCode::kF32, "sigmaColorData");
    ValidateParamTensorType(sigmaSpace, DataTypeCode::kF32, "sigmaSpaceData");
}

// =============================================================================
// Dispatch
//
// The legacy translation units reached their kernels through a funcs[5][8][4] table of function
// pointers whose two null rows (8-bit signed, 64-bit float) were exactly the types the gate above
// rejects. These switches reproduce the reachable set and nothing wider.
// =============================================================================

using cvcuda::priv::DispatchBorderMode;

// Hands `cb` a zero-valued tag of the element base type.
template<class Cb>
inline void DispatchByBaseType(DataTypeCode code, const Cb &cb)
{
    switch (code)
    {
    case DataTypeCode::kU8:
        return cb(static_cast<unsigned char>(0));
    case DataTypeCode::kU16:
        return cb(static_cast<unsigned short>(0));
    case DataTypeCode::kS16:
        return cb(static_cast<short>(0));
    case DataTypeCode::kS32:
        return cb(0);
    case DataTypeCode::kF32:
        return cb(0.0f);
    case DataTypeCode::kF16:
        return cb(__half{});
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "[Error] Invalid DataType");
    }
}

// Interleaved pixels are one vector of `channels` elements. Channel count 1 is the bare scalar, not
// MakeType<BT, 1>, matching the legacy table's uchar/__half entries.
template<class BT, class Cb>
inline void DispatchByChannels(int channels, const Cb &cb)
{
    switch (channels)
    {
    case 1:
        return cb(BT{});
    case 2:
        return cb(cuda::MakeType<BT, 2>{});
    case 3:
        return cb(cuda::MakeType<BT, 3>{});
    case 4:
        return cb(cuda::MakeType<BT, 4>{});
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number ch = %d", channels);
    }
}

template<class Cb>
inline void DispatchByPixelType(DataTypeCode code, int channels, const Cb &cb)
{
    DispatchByBaseType(code, [&](auto baseTag) { DispatchByChannels<decltype(baseTag)>(channels, cb); });
}

// Planar pixels are scalars, so the channel count is a kernel template argument instead. Two
// channels have no planar instantiation: ValidateChannelCount rejects them ahead of this.
template<class Cb>
inline void DispatchByPlanarChannels(int channels, const Cb &cb)
{
    switch (channels)
    {
    case 1:
        return cb(std::integral_constant<int, 1>{});
    case 3:
        return cb(std::integral_constant<int, 3>{});
    case 4:
        return cb(std::integral_constant<int, 4>{});
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid planar channel number ch = %d", channels);
    }
}

template<class Cb>
inline void DispatchByPlanarType(DataTypeCode code, int channels, const Cb &cb)
{
    DispatchByBaseType(code, [&](auto baseTag)
                       { DispatchByPlanarChannels(channels, [&](auto channelTag) { cb(baseTag, channelTag); }); });
}

// Selects one of the four kernel families. Dispatch order carries no observable behaviour -- unlike
// validation order, which is why the sequences stay in each operator's translation unit and this
// does not.
template<class PackedCb, class PlanarCb>
inline void DispatchTile(NVCVBorderType borderMode, DataTypeCode dataType, int channels, bool isPlanar,
                         const PackedCb &packedCb, const PlanarCb &planarCb)
{
    DispatchBorderMode(
        borderMode,
        [&](auto borderTag)
        {
            if (isPlanar)
            {
                DispatchByPlanarType(dataType, channels,
                                     [&](auto baseTag, auto channelTag) { planarCb(borderTag, baseTag, channelTag); });
            }
            else
            {
                DispatchByPixelType(dataType, channels, [&](auto pixelTag) { packedCb(borderTag, pixelTag); });
            }
        });
}

// =============================================================================
// Filter parameters and launch geometry
// =============================================================================

struct FilterParams
{
    int   radius;
    int   squaredRadius;
    float colorCoefficient;
    float spaceCoefficient;
};

// The var-shape paths read diameter and both sigmas from device-resident tensors, so this derivation
// has to run per image inside the kernel; the tensor paths run it once on the host. Both sides
// evaluate the same correctly-rounded float multiplies and divides, so the coefficients agree bit
// for bit either way.
inline __host__ __device__ FilterParams MakeFilterParams(int diameter, float sigmaColor, float sigmaSpace)
{
    if (sigmaColor <= 0)
    {
        sigmaColor = 1;
    }
    if (sigmaSpace <= 0)
    {
        sigmaSpace = 1;
    }

    int radius;
    if (diameter <= 0)
    {
        radius = std::roundf(sigmaSpace * 1.5f);
    }
    else
    {
        radius = diameter / 2;
    }
    if (radius < 1)
    {
        radius = 1;
    }
    assert(radius < 10000);

    return FilterParams{radius, radius * radius, -1 / (2 * sigmaColor * sigmaColor),
                        -1 / (2 * sigmaSpace * sigmaSpace)};
}

// The var-shape kernels take all three per-image parameters as one argument so the four kernel and
// four launcher signatures do not each carry the same trio.
struct VarShapeFilterParams
{
    cuda::Tensor1DWrap<int>   diameter;
    cuda::Tensor1DWrap<float> sigmaColor;
    cuda::Tensor1DWrap<float> sigmaSpace;
};

inline __device__ FilterParams MakeFilterParams(const VarShapeFilterParams &params, int sample)
{
    return MakeFilterParams(params.diameter[sample], params.sigmaColor[sample], params.sigmaSpace[sample]);
}

// 64 threads laid out along the row, so the four stores of neighbouring threads stay contiguous.
inline dim3 RowTileBlock()
{
    return dim3{32, 2};
}

// Every thread owns a 2x2 output tile, so the grid covers twice the block extent in each direction.
// Both divisors are powers of two, where DivUp agrees exactly with ceil((float)a/b).
inline dim3 TileGrid(dim3 block, int columns, int rows, int batch)
{
    return dim3(util::DivUp(columns, static_cast<int>(block.x) * 2), util::DivUp(rows, static_cast<int>(block.y) * 2),
                batch);
}

// =============================================================================
// Device-side window math
// =============================================================================

namespace detail {

template<class W, class = void>
struct HasTensorWrap : std::false_type
{
};

template<class W>
struct HasTensorWrap<W, std::void_t<decltype(std::declval<const W &>().tensorWrap())>> : std::true_type
{
};

} // namespace detail

// The accessor a border wrap defers to once the coordinate is known to be in range.
template<class SrcWrapper>
inline __device__ decltype(auto) UnborderedWrap(const SrcWrapper &src)
{
    if constexpr (detail::HasTensorWrap<SrcWrapper>::value)
    {
        return (src.tensorWrap());
    }
    else
    {
        return (src.imageBatchWrap());
    }
}

// Reading through a border wrap costs a coordinate fixup on every tap. When the whole window sits
// inside the image, or when this particular tap does, the wrapped accessor can be indexed directly.
// kFastInterior is a per-operator policy: false collapses this to the plain wrapped read, with no
// branch left behind.
template<bool kFastInterior, class SrcWrapper>
inline __device__ typename SrcWrapper::ValueType ReadTap(const SrcWrapper &src, int3 coord, int rows, int columns,
                                                         bool windowInside)
{
    if constexpr (kFastInterior && SrcWrapper::kBorderType != NVCV_BORDER_CONSTANT)
    {
        if (windowInside || (!cuda::IsOutside(coord.x, columns) && !cuda::IsOutside(coord.y, rows)))
        {
            return UnborderedWrap(src)[coord];
        }
    }
    return src[coord];
}

// Planar counterpart of ReadTap: maps one window coordinate into the image and reports whether the
// tap contributes at all, which only a constant border can deny.
template<NVCVBorderType B, bool kFastInterior>
inline __device__ bool MapBorderCoordinate(int &x, int &y, int columns, int rows, bool windowInside)
{
    if constexpr (B == NVCV_BORDER_CONSTANT)
    {
        return !cuda::IsOutside(x, columns) && !cuda::IsOutside(y, rows);
    }
    else
    {
        if constexpr (kFastInterior)
        {
            if (windowInside || (!cuda::IsOutside(x, columns) && !cuda::IsOutside(y, rows)))
            {
                return true;
            }
        }
        x = cuda::GetIndexWithBorder<B>(x, columns);
        y = cuda::GetIndexWithBorder<B>(y, rows);
        return true;
    }
}

// L1 norm driving the range weight, one arm per channel count. The element indices are template
// arguments rather than a loop counter so the value never needs an address, which a runtime
// GetElement would force.
template<class WorkType>
inline __device__ float Norm1(WorkType a)
{
    constexpr int kElements = cuda::NumElements<WorkType>;

    float sum = cuda::abs(cuda::GetElement<0>(a));
    if constexpr (kElements >= 2)
    {
        sum += cuda::abs(cuda::GetElement<1>(a));
    }
    if constexpr (kElements >= 3)
    {
        sum += cuda::abs(cuda::GetElement<2>(a));
    }
    if constexpr (kElements >= 4)
    {
        sum += cuda::abs(cuda::GetElement<3>(a));
    }
    return sum;
}

// True when the whole 2x2 tile's window lies inside the image, so no tap needs border handling.
inline __device__ bool IsWindowInside(int colIdx, int rowIdx, int columns, int rows, int radius)
{
    return colIdx >= radius && rowIdx >= radius && colIdx + radius + 1 < columns && rowIdx + radius + 1 < rows;
}

/**
 * @brief Filters the 2x2 interleaved output tile anchored at (colIdx, rowIdx).
 *
 * Grouping four neighbouring centres lets a tap be loaded once and folded into every centre it is
 * within radius of, which is the whole point of the 2x2 shape.
 *
 * @tparam kHasGuide      the range weight comes from @p guide instead of @p src (the joint filter).
 *                        When false @p guide is never read and callers pass @p src for it.
 * @tparam kFastInterior  enable the unbordered-read shortcut on source taps.
 */
template<bool kHasGuide, bool kFastInterior, class SrcWrapper, class DstWrapper>
inline __device__ void PackedTile(const SrcWrapper &src, const SrcWrapper &guide, const DstWrapper &dst, int batchIdx,
                                  int colIdx, int rowIdx, int rows, int columns, const FilterParams &params,
                                  bool windowInside = false)
{
    using T        = typename DstWrapper::ValueType;
    using WorkType = cuda::ConvertBaseTypeTo<float, T>;

    const int3 coord[4] = {
        {    colIdx,     rowIdx, batchIdx},
        {colIdx + 1,     rowIdx, batchIdx},
        {    colIdx, rowIdx + 1, batchIdx},
        {colIdx + 1, rowIdx + 1, batchIdx},
    };

    WorkType center[4];
    WorkType numerator[4];
    float    denominator[4];

#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        if constexpr (kHasGuide)
        {
            center[p] = cuda::StaticCast<float>(guide[coord[p]]);
        }
        else
        {
            center[p] = cuda::StaticCast<float>(ReadTap<kFastInterior>(src, coord[p], rows, columns, windowInside));
        }
        numerator[p]   = cuda::SetAll<WorkType>(0);
        denominator[p] = 0;
    }

    for (int c = colIdx - params.radius; c < colIdx + params.radius + 2; c++)
    {
        for (int r = rowIdx - params.radius; r < rowIdx + params.radius + 2; r++)
        {
            const int dx0 = c - colIdx;
            const int dy0 = r - rowIdx;
            const int dx1 = c - (colIdx + 1);
            const int dy1 = r - (rowIdx + 1);

            const int squaredDistance[4]
                = {dx0 * dx0 + dy0 * dy0, dx1 * dx1 + dy0 * dy0, dx0 * dx0 + dy1 * dy1, dx1 * dx1 + dy1 * dy1};

            if (!(squaredDistance[0] <= params.squaredRadius || squaredDistance[1] <= params.squaredRadius
                  || squaredDistance[2] <= params.squaredRadius || squaredDistance[3] <= params.squaredRadius))
            {
                continue;
            }

            const int3 tap  = {c, r, batchIdx};
            WorkType   curr = cuda::StaticCast<float>(ReadTap<kFastInterior>(src, tap, rows, columns, windowInside));

            WorkType currRange;
            if constexpr (kHasGuide)
            {
                currRange = cuda::StaticCast<float>(guide[tap]);
            }
            else
            {
                currRange = curr;
            }

#pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                if (squaredDistance[p] <= params.squaredRadius)
                {
                    const float eSpace      = squaredDistance[p] * params.spaceCoefficient;
                    const float oneNormSize = Norm1(currRange - center[p]);
                    const float eColor      = oneNormSize * oneNormSize * params.colorCoefficient;
                    const float weight      = cuda::exp(eSpace + eColor);

                    denominator[p] += weight;
                    numerator[p] = numerator[p] + (weight * curr);
                }
            }
        }
    }

#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        if (coord[p].x < columns && coord[p].y < rows)
        {
            dst[coord[p]] = cuda::SaturateCast<T>(numerator[p] / denominator[p]);
        }
    }
}

/**
 * @brief Planar counterpart of PackedTile.
 *
 * @p src, @p guide and @p dst are plane readers/writers (see StridedPlaneView and PlaneArrayView),
 * already bound to the sample this thread works on.
 *
 * @tparam kRowOuter  walk the window row-major rather than column-major. Each operator keeps the
 *                    order its kernels have always used: it sets both the reuse distance between
 *                    consecutive taps and the order the float accumulators are summed in.
 */
template<typename T, int CHANNELS, NVCVBorderType B, bool kHasGuide, bool kFastInterior, bool kRowOuter,
         int FIXED_RADIUS = 0, bool kCoordinateLoop = false, int OUTPUT_COLUMNS = 2, class Reader, class Writer>
inline __device__ void PlanarTile(const Reader &src, const Reader &guide, const Writer &dst, int colIdx, int rowIdx,
                                  int rows, int columns, const FilterParams &params)
{
    constexpr bool kFixedRadius      = FIXED_RADIUS > 0;
    const int      tileRadius        = kFixedRadius ? FIXED_RADIUS : params.radius;
    const int      tileSquaredRadius = kFixedRadius ? FIXED_RADIUS * FIXED_RADIUS : params.squaredRadius;

    static_assert(OUTPUT_COLUMNS == 1 || OUTPUT_COLUMNS == 2);
    constexpr int OUTPUT_ROWS = 4 / OUTPUT_COLUMNS;
    const int  x[4] = {colIdx, colIdx + 1 % OUTPUT_COLUMNS, colIdx + 2 % OUTPUT_COLUMNS, colIdx + 3 % OUTPUT_COLUMNS};
    const int  y[4] = {rowIdx, rowIdx + 1 / OUTPUT_COLUMNS, rowIdx + 2 / OUTPUT_COLUMNS, rowIdx + 3 / OUTPUT_COLUMNS};
    const bool windowInside
        = kFastInterior
       && (OUTPUT_COLUMNS == 2 ? IsWindowInside(colIdx, rowIdx, columns, rows, tileRadius)
                               : colIdx >= tileRadius && rowIdx >= tileRadius && colIdx + tileRadius < columns
                                     && rowIdx + tileRadius + OUTPUT_ROWS - 1 < rows);

    bool valid[4];
    if constexpr (OUTPUT_COLUMNS == 2)
    {
        valid[0] = colIdx < columns && rowIdx < rows;
        valid[1] = colIdx + 1 < columns && rowIdx < rows;
        valid[2] = colIdx < columns && rowIdx + 1 < rows;
        valid[3] = colIdx + 1 < columns && rowIdx + 1 < rows;
    }
    else
    {
        valid[0] = x[0] < columns && y[0] < rows;
        valid[1] = x[1] < columns && y[1] < rows;
        valid[2] = x[2] < columns && y[2] < rows;
        valid[3] = x[3] < columns && y[3] < rows;
    }

    if (!(valid[0] || valid[1] || valid[2] || valid[3]))
    {
        return;
    }

    float center[4][CHANNELS]    = {};
    float numerator[4][CHANNELS] = {};
    float denominator[4]         = {};

#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        if (valid[p])
        {
#pragma unroll
            for (int ch = 0; ch < CHANNELS; ++ch)
            {
                if constexpr (kHasGuide)
                {
                    center[p][ch] = guide.Read(ch, y[p], x[p]);
                }
                else
                {
                    center[p][ch] = src.Read(ch, y[p], x[p]);
                }
            }
        }
    }

    // kRowOuter picks which axis the outer index drives; it is a compile-time constant, so what is
    // left is the same nest the operator's kernels have always run.
    using LoopIndex = std::conditional_t<kCoordinateLoop, uint32_t, int>;
    const LoopIndex outerSpan
        = LoopIndex{2} * static_cast<LoopIndex>(tileRadius) + LoopIndex{kRowOuter ? OUTPUT_ROWS : OUTPUT_COLUMNS};
    const LoopIndex innerSpan
        = LoopIndex{2} * static_cast<LoopIndex>(tileRadius) + LoopIndex{kRowOuter ? OUTPUT_COLUMNS : OUTPUT_ROWS};
    const LoopIndex outerBegin
        = kCoordinateLoop ? static_cast<LoopIndex>(kRowOuter ? rowIdx : colIdx) - static_cast<LoopIndex>(tileRadius)
                          : LoopIndex{0};
    const LoopIndex innerBegin
        = kCoordinateLoop ? static_cast<LoopIndex>(kRowOuter ? colIdx : rowIdx) - static_cast<LoopIndex>(tileRadius)
                          : LoopIndex{0};
    const LoopIndex outerEnd = outerBegin + outerSpan;
    const LoopIndex innerEnd = innerBegin + innerSpan;
#pragma unroll
    for (LoopIndex outer = outerBegin, outerTrip = 0; kCoordinateLoop ? outerTrip < outerSpan : outer < outerEnd;
         outer++, outerTrip++)
    {
#pragma unroll
        for (LoopIndex inner = innerBegin, innerTrip = 0; kCoordinateLoop ? innerTrip < innerSpan : inner < innerEnd;
             inner++, innerTrip++)
        {
            int c;
            int r;
            if constexpr (kCoordinateLoop)
            {
                const uint32_t column = kRowOuter ? inner : outer;
                const uint32_t row    = kRowOuter ? outer : inner;
                c = column <= INT_MAX ? static_cast<int>(column) : -1 - static_cast<int>(UINT_MAX - column);
                r = row <= INT_MAX ? static_cast<int>(row) : -1 - static_cast<int>(UINT_MAX - row);
            }
            else
            {
                c = colIdx - tileRadius + (kRowOuter ? inner : outer);
                r = rowIdx - tileRadius + (kRowOuter ? outer : inner);
            }

            const int dx0 = c - colIdx;
            const int dy0 = r - rowIdx;
            const int dx1 = c - (colIdx + 1);
            const int dy1 = r - (rowIdx + 1);

            const int squaredDistance[4] = {
                dx0 * dx0 + dy0 * dy0, OUTPUT_COLUMNS == 2 ? dx1 * dx1 + dy0 * dy0 : dx0 * dx0 + (dy0 - 1) * (dy0 - 1),
                OUTPUT_COLUMNS == 2 ? dx0 * dx0 + dy1 * dy1 : dx0 * dx0 + (dy0 - 2) * (dy0 - 2),
                OUTPUT_COLUMNS == 2 ? dx1 * dx1 + dy1 * dy1 : dx0 * dx0 + (dy0 - 3) * (dy0 - 3)};

            if (!(squaredDistance[0] <= tileSquaredRadius || squaredDistance[1] <= tileSquaredRadius
                  || squaredDistance[2] <= tileSquaredRadius || squaredDistance[3] <= tileSquaredRadius))
            {
                continue;
            }

            int srcX = c;
            int srcY = r;

            float curr[CHANNELS]      = {};
            float currRange[CHANNELS] = {};
            if (MapBorderCoordinate<B, kFastInterior>(srcX, srcY, columns, rows, windowInside))
            {
#pragma unroll
                for (int ch = 0; ch < CHANNELS; ++ch)
                {
                    curr[ch] = src.Read(ch, srcY, srcX);
                    if constexpr (kHasGuide)
                    {
                        currRange[ch] = guide.Read(ch, srcY, srcX);
                    }
                    else
                    {
                        currRange[ch] = curr[ch];
                    }
                }
            }

#pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                if (valid[p] && squaredDistance[p] <= tileSquaredRadius)
                {
                    float oneNormSize = 0.f;
#pragma unroll
                    for (int ch = 0; ch < CHANNELS; ++ch)
                    {
                        oneNormSize += cuda::abs(currRange[ch] - center[p][ch]);
                    }

                    const float eSpace = squaredDistance[p] * params.spaceCoefficient;
                    const float eColor = oneNormSize * oneNormSize * params.colorCoefficient;
                    const float weight = cuda::exp(eSpace + eColor);

                    denominator[p] += weight;
#pragma unroll
                    for (int ch = 0; ch < CHANNELS; ++ch)
                    {
                        numerator[p][ch] += weight * curr[ch];
                    }
                }
            }
        }
    }

#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        if (valid[p])
        {
#pragma unroll
            for (int ch = 0; ch < CHANNELS; ++ch)
            {
                dst.Write(ch, y[p], x[p], cuda::SaturateCast<T>(numerator[p][ch] / denominator[p]));
            }
        }
    }
}

// =============================================================================
// Planar plane accessors
//
// The tile walks one channel plane at a time and revisits that plane for every tap in the window,
// so these hoist the per-plane addressing out of the tap loop: the var-shape pair resolves each
// plane descriptor once per thread instead of once per tap, and the tensor pair folds the sample
// offset in once. Both keep the stride widths the wraps use.
// =============================================================================

// Tensor planar: one base pointer for the whole batch; AtSample() binds it to the sample a block
// works on, which is only known on the device.
template<typename T>
struct StridedPlaneView
{
    const NVCVByte *base;
    int64_t         sampleStride;
    int64_t         chStride;
    int64_t         rowStride;
    int64_t         colStride;

    inline __device__ StridedPlaneView AtSample(int sample) const
    {
        return {base + sample * sampleStride, sampleStride, chStride, rowStride, colStride};
    }

    inline __device__ float Read(int ch, int y, int x) const
    {
        return static_cast<float>(*reinterpret_cast<const T *>(base + ch * chStride + y * rowStride + x * colStride));
    }
};

template<typename T>
struct StridedPlaneWriteView
{
    NVCVByte *base;
    int64_t   sampleStride;
    int64_t   chStride;
    int64_t   rowStride;
    int64_t   colStride;

    inline __device__ StridedPlaneWriteView AtSample(int sample) const
    {
        return {base + sample * sampleStride, sampleStride, chStride, rowStride, colStride};
    }

    inline __device__ void Write(int ch, int y, int x, T value) const
    {
        *reinterpret_cast<T *>(base + ch * chStride + y * rowStride + x * colStride) = value;
    }
};

// Var-shape planar: planes are separate allocations, so each carries its own base and row stride.
template<typename T, int CHANNELS>
struct PlaneArrayView
{
    const NVCVByte *base[CHANNELS];
    int             rowStride[CHANNELS];

    inline __device__ float Read(int ch, int y, int x) const
    {
        return static_cast<float>(*reinterpret_cast<const T *>(base[ch] + y * rowStride[ch] + x * sizeof(T)));
    }
};

template<typename T, int CHANNELS>
struct PlaneArrayWriteView
{
    NVCVByte *base[CHANNELS];
    int       rowStride[CHANNELS];

    inline __device__ void Write(int ch, int y, int x, T value) const
    {
        *reinterpret_cast<T *>(base[ch] + y * rowStride[ch] + x * sizeof(T)) = value;
    }
};

template<typename T>
inline StridedPlaneView<T> MakePlaneView(const nvcv::TensorDataStridedCuda              &data,
                                         const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    return {reinterpret_cast<const NVCVByte *>(data.basePtr()), access.sampleStride(), access.chStride(),
            access.rowStride(), access.colStride()};
}

template<typename T>
inline StridedPlaneWriteView<T> MakePlaneWriteView(const nvcv::TensorDataStridedCuda              &data,
                                                   const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    return {reinterpret_cast<NVCVByte *>(data.basePtr()), access.sampleStride(), access.chStride(), access.rowStride(),
            access.colStride()};
}

template<typename T, int CHANNELS, class SrcWrapper>
inline __device__ PlaneArrayView<T, CHANNELS> MakePlaneArrayView(const SrcWrapper &src, int sample)
{
    PlaneArrayView<T, CHANNELS> view{};
#pragma unroll
    for (int ch = 0; ch < CHANNELS; ++ch)
    {
        const NVCVImagePlaneStrided plane = UnborderedWrap(src).plane(sample, ch);

        view.base[ch]      = plane.basePtr;
        view.rowStride[ch] = plane.rowStride;
    }
    return view;
}

template<typename T, int CHANNELS, class DstWrapper>
inline __device__ PlaneArrayWriteView<T, CHANNELS> MakePlaneArrayWriteView(const DstWrapper &dst, int sample)
{
    PlaneArrayWriteView<T, CHANNELS> view{};
#pragma unroll
    for (int ch = 0; ch < CHANNELS; ++ch)
    {
        const NVCVImagePlaneStrided plane = dst.plane(sample, ch);

        view.base[ch]      = plane.basePtr;
        view.rowStride[ch] = plane.rowStride;
    }
    return view;
}

} // namespace cvcuda::priv::bilateral

#endif // CVCUDA_PRIV_BILATERAL_FILTER_COMMON_CUH

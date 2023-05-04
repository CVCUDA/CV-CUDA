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

#include "OpRemap.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/cuda/DropCast.hpp>
#include <nvcv/cuda/InterpolationVarShapeWrap.hpp>
#include <nvcv/cuda/InterpolationWrap.hpp>
#include <nvcv/cuda/MathOps.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <util/Math.hpp>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// Remap parameters ------------------------------------------------------------

constexpr NVCVBorderType kMapBorderType = NVCV_BORDER_REPLICATE;

struct NVCVRemapParams
{
    float2 srcScale, mapScale, valScale, srcOffset;
    float  dstOffset;
};

inline NVCVRemapParams __host__ __device__ GetRemapParams(const int2 &srcSize, const int2 &dstSize, const int2 &mapSize,
                                                          bool alignCorners, NVCVRemapMapValueType mapValueType)
{
    // To avoid floating-point issues, instead of normalizing coordinates by dividing them by destination size, it
    // is better to compute the {source, map, map value} scale by dividing its size by destination size and use it
    // to scale the {source, map} coordinates accordingly.  The map value affects the source coordinates, and the
    // source offset is used to shift its coordinate position depending on the map value type.
    NVCVRemapParams params;

    switch (mapValueType)
    {
    case NVCV_REMAP_ABSOLUTE:
        params.srcScale  = float2{0.f, 0.f};
        params.mapScale  = cuda::StaticCast<float>(mapSize) / dstSize;
        params.valScale  = float2{1.f, 1.f};
        params.srcOffset = float2{0.f, 0.f};
        params.dstOffset = 0.f;
        break;
    case NVCV_REMAP_ABSOLUTE_NORMALIZED:
        params.srcScale  = float2{0.f, 0.f};
        params.mapScale  = cuda::StaticCast<float>(mapSize) / dstSize;
        params.valScale  = (srcSize - (alignCorners ? 1.f : 0.f)) / 2.f;
        params.srcOffset = params.valScale - (alignCorners ? 0.f : .5f);
        params.dstOffset = 0.f;
        break;
    case NVCV_REMAP_RELATIVE_NORMALIZED:
        params.srcScale  = cuda::StaticCast<float>(srcSize) / dstSize;
        params.mapScale  = (mapSize - 1.f) / dstSize;
        params.valScale  = srcSize - 1.f;
        params.dstOffset = alignCorners ? 0.f : .5f;
        params.srcOffset = params.srcScale * params.dstOffset - params.dstOffset;
        break;
    default:
        assert(false && "wrong map value type");
        break;
    }

    return params;
}

// Do remap kernel -------------------------------------------------------------

template<class SrcWrapper, class DstWrapper, class MapWrapper>
inline void __device__ DoRemap(SrcWrapper src, DstWrapper dst, MapWrapper map, const int2 &dstSize,
                               const int &mapNumSamples, const NVCVRemapParams &params)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (dstCoord.x >= dstSize.x || dstCoord.y >= dstSize.y)
    {
        return;
    }

    float3 mapCoord{0.f, 0.f, (mapNumSamples == 1 ? 0.f : static_cast<float>(dstCoord.z))};

    // The map is accessed at destination coordinate, with destination offset and scaled by map scale.  The
    // resulting map coordinate is interpolated in the map, given by the map interpolation type.

    mapCoord.x = (dstCoord.x + params.dstOffset) * params.mapScale.x;
    mapCoord.y = (dstCoord.y + params.dstOffset) * params.mapScale.y;

    float2 mapValue = map[mapCoord];

    float3 srcCoord{0.f, 0.f, static_cast<float>(dstCoord.z)};

    // The source is accessed at destination coordinate scaled by source scale, plus the map value that is either a
    // relative distance from destination or an absolute position at source (either normalized or not), multiplied
    // by value scale and offset by source offset.  The result of the map value scaled must be rounded to get an
    // absolute position regardless of source interpolation.  The source interpolation type only affects the source
    // scaling and offset values.

    srcCoord.x = dstCoord.x * params.srcScale.x + mapValue.x * params.valScale.x + params.srcOffset.x;
    srcCoord.y = dstCoord.y * params.srcScale.y + mapValue.y * params.valScale.y + params.srcOffset.y;

    dst[dstCoord] = src[srcCoord];
}

// Remap with tensors kernel ---------------------------------------------------

template<class SrcWrapper, class DstWrapper, class MapWrapper>
__global__ void Remap(SrcWrapper src, DstWrapper dst, MapWrapper map, int2 dstSize, int mapNumSamples,
                      NVCVRemapParams params)
{
    DoRemap(src, dst, map, dstSize, mapNumSamples, params);
}

// Remap with varshape kernel --------------------------------------------------

template<class SrcWrapper, class DstWrapper, class MapWrapper>
__global__ void Remap(SrcWrapper src, DstWrapper dst, MapWrapper map, int2 mapSize, int mapNumSamples,
                      bool alignCorners, NVCVRemapMapValueType mapValueType)
{
    int z = blockIdx.z;

    int2 dstSize{dst.width(z), dst.height(z)};
    int2 srcSize{src.borderWrap().imageBatchWrap().width(z), src.borderWrap().imageBatchWrap().height(z)};

    NVCVRemapParams params = GetRemapParams(srcSize, dstSize, mapSize, alignCorners, mapValueType);

    DoRemap(src, dst, map, dstSize, mapNumSamples, params);
}

// Host run remap functions ----------------------------------------------------

template<typename T, NVCVBorderType B, NVCVInterpolationType MI, NVCVInterpolationType SI, class DataStridedCuda>
void RunRemap(cudaStream_t stream, const DataStridedCuda &srcData, const DataStridedCuda &dstData,
              const nvcv::TensorDataStridedCuda &mapData, NVCVRemapMapValueType mapValueType, bool alignCorners,
              const T &borderValue)
{
    auto mapAccess     = nvcv::TensorDataAccessStridedImagePlanar::Create(mapData);
    int2 mapSize       = cuda::StaticCast<int>(long2{mapAccess->numCols(), mapAccess->numRows()});
    int  mapNumSamples = mapAccess->numSamples();

    dim3 block(32, 4, 1);

    auto map = cuda::CreateInterpolationWrapNHW<const float2, kMapBorderType, MI>(mapData);

    if constexpr (std::is_same_v<DataStridedCuda, nvcv::TensorDataStridedCuda>)
    {
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);

        int2 srcSize = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});
        int2 dstSize = cuda::StaticCast<int>(long2{dstAccess->numCols(), dstAccess->numRows()});

        NVCVRemapParams params = GetRemapParams(srcSize, dstSize, mapSize, alignCorners, mapValueType);

        dim3 grid(util::DivUp(dstSize.x, block.x), util::DivUp(dstSize.y, block.y), dstAccess->numSamples());

        auto src = cuda::CreateInterpolationWrapNHW<const T, B, SI>(srcData, borderValue);
        auto dst = cuda::CreateTensorWrapNHW<T>(dstData);

        Remap<<<grid, block, 0, stream>>>(src, dst, map, dstSize, mapNumSamples, params);
    }
    else
    {
        static_assert(std::is_same_v<DataStridedCuda, nvcv::ImageBatchVarShapeDataStridedCuda>);

        int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};

        dim3 grid(util::DivUp(dstMaxSize.x, block.x), util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);

        cuda::InterpolationVarShapeWrap<const T, B, SI> src(srcData, borderValue);
        cuda::ImageBatchVarShapeWrap<T>                 dst(dstData);

        Remap<<<grid, block, 0, stream>>>(src, dst, map, mapSize, mapNumSamples, alignCorners, mapValueType);
    }
}

template<typename T, NVCVBorderType B, NVCVInterpolationType MI, class DataStridedCuda>
void RunRemap(cudaStream_t stream, const DataStridedCuda &srcData, const DataStridedCuda &dstData,
              const nvcv::TensorDataStridedCuda &mapData, NVCVInterpolationType srcInterp,
              NVCVRemapMapValueType mapValueType, bool alignCorners, const T &borderValue)
{
#define NVCV_RUN_REMAP(INTERP_TYPE)                                                                                  \
    case NVCV_INTERP_##INTERP_TYPE:                                                                                  \
        RunRemap<T, B, MI, NVCV_INTERP_##INTERP_TYPE>(stream, srcData, dstData, mapData, mapValueType, alignCorners, \
                                                      borderValue);                                                  \
        break

    switch (srcInterp)
    {
        NVCV_RUN_REMAP(NEAREST);
        NVCV_RUN_REMAP(LINEAR);
        NVCV_RUN_REMAP(CUBIC);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input interpolation type");
    }

#undef NVCV_RUN_REMAP
}

template<typename T, NVCVBorderType B, class DataStridedCuda>
void RunRemap(cudaStream_t stream, const DataStridedCuda &srcData, const DataStridedCuda &dstData,
              const nvcv::TensorDataStridedCuda &mapData, NVCVInterpolationType srcInterp,
              NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType, bool alignCorners,
              const T &borderValue)
{
#define NVCV_RUN_REMAP(INTERP_TYPE)                                                                           \
    case NVCV_INTERP_##INTERP_TYPE:                                                                           \
        RunRemap<T, B, NVCV_INTERP_##INTERP_TYPE>(stream, srcData, dstData, mapData, srcInterp, mapValueType, \
                                                  alignCorners, borderValue);                                 \
        break

    switch (mapInterp)
    {
        NVCV_RUN_REMAP(NEAREST);
        NVCV_RUN_REMAP(LINEAR);
        NVCV_RUN_REMAP(CUBIC);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid map interpolation type");
    }

#undef NVCV_RUN_REMAP
}

template<typename T, class DataStridedCuda>
void RunRemap(cudaStream_t stream, const DataStridedCuda &srcData, const DataStridedCuda &dstData,
              const nvcv::TensorDataStridedCuda &mapData, NVCVInterpolationType srcInterp,
              NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType, bool alignCorners,
              NVCVBorderType border, const float4 &borderValue)
{
    const T bvalue = cuda::DropCast<cuda::NumElements<T>>(cuda::StaticCast<cuda::BaseType<T>>(borderValue));

#define NVCV_RUN_REMAP(BORDER_TYPE)                                                                                   \
    case NVCV_BORDER_##BORDER_TYPE:                                                                                   \
        RunRemap<T, NVCV_BORDER_##BORDER_TYPE>(stream, srcData, dstData, mapData, srcInterp, mapInterp, mapValueType, \
                                               alignCorners, bvalue);                                                 \
        break

    switch (border)
    {
        NVCV_RUN_REMAP(CONSTANT);
        NVCV_RUN_REMAP(REPLICATE);
        NVCV_RUN_REMAP(REFLECT);
        NVCV_RUN_REMAP(WRAP);
        NVCV_RUN_REMAP(REFLECT101);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid border type");
    }

#undef NVCV_RUN_REMAP
}

template<class DataStridedCuda>
inline void RunRemap(cudaStream_t stream, const DataStridedCuda &srcData, const DataStridedCuda &dstData,
                     const nvcv::TensorDataStridedCuda &mapData, NVCVInterpolationType srcInterp,
                     NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType, bool alignCorners,
                     NVCVBorderType border, const float4 &borderValue, nvcv::DataType dataType, int numChannels = 1)
{
    // When this function is called with tensors, the data type may contain the channels baked in or the number of
    // channels is in the tensor shape; when it is called with varshape, the data type always contain the channels
    // as each image only stores size, there is no shape information with number of channels baked in.

    // clang-format off

#define NVCV_RUN_REMAP(BT, DT, T)                                                          \
    ((dataType == nvcv::TYPE_##BT && numChannels == cuda::NumElements<T>) ||               \
     (dataType == nvcv::TYPE_##DT && numChannels == 1))                                    \
        RunRemap<T>(stream, srcData, dstData, mapData, srcInterp, mapInterp, mapValueType, \
                    alignCorners, border, borderValue)

    if NVCV_RUN_REMAP(U8, U8, uchar1);
    else if NVCV_RUN_REMAP(U8, 3U8, uchar3);
    else if NVCV_RUN_REMAP(U8, 4U8, uchar4);
    else if NVCV_RUN_REMAP(F32, F32, float1);
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid data type in input/output");
    }

#undef NVCV_RUN_REMAP

    // clang-format on
}

} // anonymous namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

Remap::Remap() {}

// Tensor operator -------------------------------------------------------------

void Remap::operator()(cudaStream_t stream, nvcv::ITensor &src, nvcv::ITensor &dst, nvcv::ITensor &map,
                       NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp,
                       NVCVRemapMapValueType mapValueType, bool alignCorners, NVCVBorderType border,
                       float4 borderValue) const
{
    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto mapData = map.exportData<nvcv::TensorDataStridedCuda>();
    if (!mapData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Remap map input must be cuda-accessible, pitch-linear tensor");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    auto mapAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*mapData);
    NVCV_ASSERT(srcAccess && dstAccess && mapAccess);

    if (srcAccess->numChannels() != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }

    if (srcAccess->numSamples() != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    if (mapAccess->numSamples() != srcAccess->numSamples() && mapAccess->numSamples() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Map must have 1 or N samples equal to input");
    }

    if (!((mapData->dtype() == nvcv::TYPE_2F32 && mapAccess->numChannels() == 1)
          || (mapData->dtype() == nvcv::TYPE_F32 && mapAccess->numChannels() == 2)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Remap map input must have 2F32 data type");
    }

    if (srcData->dtype() != dstData->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output data type are different");
    }

    if (srcData->layout() != nvcv::TENSOR_HWC && srcData->layout() != nvcv::TENSOR_NHWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC layout");
    }

    if (dstData->layout() != nvcv::TENSOR_HWC && dstData->layout() != nvcv::TENSOR_NHWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must have (N)HWC layout");
    }

    RunRemap(stream, *srcData, *dstData, *mapData, srcInterp, mapInterp, mapValueType, alignCorners, border,
             borderValue, dstData->dtype(), dstAccess->numChannels());
}

// VarShape operator -----------------------------------------------------------

void Remap::operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &src, nvcv::IImageBatchVarShape &dst,
                       nvcv::ITensor &map, NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp,
                       NVCVRemapMapValueType mapValueType, bool alignCorners, NVCVBorderType border,
                       float4 borderValue) const
{
    auto srcData = src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto dstData = dst.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto mapData = map.exportData<nvcv::TensorDataStridedCuda>();
    if (!mapData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Remap map input must be cuda-accessible, pitch-linear tensor");
    }

    auto mapAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*mapData);
    NVCV_ASSERT(mapAccess);

    if (srcData->numImages() != dstData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    if (mapAccess->numSamples() != srcData->numImages() && mapAccess->numSamples() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Map must have 1 or N samples equal to input");
    }

    if (!((mapData->dtype() == nvcv::TYPE_2F32 && mapAccess->numChannels() == 1)
          || (mapData->dtype() == nvcv::TYPE_F32 && mapAccess->numChannels() == 2)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Remap map input must have 2F32 data type");
    }

    if (srcData->uniqueFormat() != dstData->uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output formats are different");
    }

    if (srcData->uniqueFormat().numPlanes() > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image batches must have (N)HWC layout");
    }

    RunRemap(stream, *srcData, *dstData, *mapData, srcInterp, mapInterp, mapValueType, alignCorners, border,
             borderValue, dstData->uniqueFormat().planeDataType(0));
}

} // namespace cvcuda::priv

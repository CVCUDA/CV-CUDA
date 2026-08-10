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

#include "Nvtx.hpp"
#include "OpRemap.hpp"
#include "PlanarTensorView.hpp"

#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/InterpolationVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/InterpolationWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/Math.hpp>

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

// Compute the (x, y) source-sampling coordinate for an output pixel (x, y) of image mapSample.
// The map is accessed at the destination coordinate (offset and scaled by map scale) and
// interpolated; the source is then accessed at the destination coordinate scaled by source scale,
// plus the map value (relative distance or absolute position, normalized or not) times the value
// scale, offset by the source offset. Shared by the interleaved and planar paths so the math has a
// single definition.
template<class MapWrapper>
inline float2 __device__ ComputeRemapSrcCoord(int x, int y, const MapWrapper &map, int mapSample,
                                              const NVCVRemapParams &params)
{
    float3 mapCoord{(x + params.dstOffset) * params.mapScale.x, (y + params.dstOffset) * params.mapScale.y,
                    static_cast<float>(mapSample)};

    float2 mapValue = map[mapCoord];

    return float2{x * params.srcScale.x + mapValue.x * params.valScale.x + params.srcOffset.x,
                  y * params.srcScale.y + mapValue.y * params.valScale.y + params.srcOffset.y};
}

// Number of output pixels each thread processes along x for the latency-bound
// (NEAREST/LINEAR) interpolation path. The per-pixel work is a serial map-read ->
// coordinate -> source-gather dependency chain; issuing several independent chains
// per thread overlaps their latency. Threads stay x-adjacent within each strided
// step, so writes remain coalesced. The CUBIC path is compute-bound at full
// occupancy, where the extra per-thread state only hurts, so it uses NIX == 1.
constexpr int kRemapNIX = 4;

template<int NIX, class SrcWrapper, class DstWrapper, class MapWrapper>
inline void __device__ DoRemap(SrcWrapper src, DstWrapper dst, MapWrapper map, const int2 &dstSize,
                               const int &mapNumSamples, const NVCVRemapParams &params, int channels = 1)
{
    // For planar (NCHW/CHW) tensors the C channel planes are flattened into grid-z as N*C
    // single-channel samples, so the per-(x,y) map sample is the original image index
    // z / channels. For interleaved data channels == 1, so this is just z.
    if constexpr (NIX == 1)
    {
        // Compute-bound (CUBIC) path: one pixel per thread, unchanged from the
        // original kernel so the compute-bound case takes no x-unroll overhead.
        int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

        if (dstCoord.x >= dstSize.x || dstCoord.y >= dstSize.y)
        {
            return;
        }

        int    mapSample = (mapNumSamples == 1) ? 0 : dstCoord.z / channels;
        float2 sc        = ComputeRemapSrcCoord(dstCoord.x, dstCoord.y, map, mapSample, params);

        dst[dstCoord] = src[float3{sc.x, sc.y, static_cast<float>(dstCoord.z)}];
    }
    else
    {
        // Latency-bound (NEAREST/LINEAR) path: NIX independent chains per thread.
        int baseX = (blockIdx.x * blockDim.x) * NIX + threadIdx.x;
        int y     = blockIdx.y * blockDim.y + threadIdx.y;
        int z     = blockIdx.z * blockDim.z + threadIdx.z;

        if (y >= dstSize.y)
        {
            return;
        }

        int mapSample = (mapNumSamples == 1) ? 0 : z / channels;

#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            int x = baseX + i * blockDim.x;
            if (x >= dstSize.x)
            {
                continue;
            }

            float2 sc = ComputeRemapSrcCoord(x, y, map, mapSample, params);

            dst[int3{x, y, z}] = src[float3{sc.x, sc.y, static_cast<float>(z)}];
        }
    }
}

// Remap with tensors kernel ---------------------------------------------------

template<int NIX, class SrcWrapper, class DstWrapper, class MapWrapper>
__global__ void Remap(SrcWrapper src, DstWrapper dst, MapWrapper map, int2 dstSize, int mapNumSamples,
                      NVCVRemapParams params, int channels = 1)
{
    DoRemap<NIX>(src, dst, map, dstSize, mapNumSamples, params, channels);
}

// Remap with planar (NCHW) tensor kernel --------------------------------------
//
// grid-z runs over images (not images*channels). The per-pixel source coordinate
// (the map lookup + interpolation) is computed once and reused for every channel
// plane, so the map traffic and coordinate math are amortized across channels
// instead of repeated per plane. The (N,C,H,W) tensor is addressed through the
// flattened (N*C,H,W) single-channel view, so plane p of image i is sample
// i*channels + p. Output is identical to the per-plane path (the coordinate does
// not depend on the plane).
template<int NIX, class SrcWrapper, class DstWrapper, class MapWrapper>
__global__ void RemapPlanarTensor(SrcWrapper src, DstWrapper dst, MapWrapper map, int2 dstSize, int mapNumSamples,
                                  NVCVRemapParams params, int channels)
{
    int baseX = (blockIdx.x * blockDim.x) * NIX + threadIdx.x;
    int y     = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= dstSize.y)
    {
        return;
    }

    int image     = blockIdx.z;
    int mapSample = (mapNumSamples == 1) ? 0 : image;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        int x = baseX + i * blockDim.x;
        if (x >= dstSize.x)
        {
            continue;
        }

        float2 sc = ComputeRemapSrcCoord(x, y, map, mapSample, params);

        for (int plane = 0; plane < channels; ++plane)
        {
            int z = image * channels + plane;

            dst[int3{x, y, z}] = src[float3{sc.x, sc.y, static_cast<float>(z)}];
        }
    }
}

// Remap with varshape kernel --------------------------------------------------

template<int NIX, class SrcWrapper, class DstWrapper, class MapWrapper>
__global__ void Remap(SrcWrapper src, DstWrapper dst, MapWrapper map, int2 mapSize, int mapNumSamples,
                      bool alignCorners, NVCVRemapMapValueType mapValueType)
{
    int z = blockIdx.z;

    int2 dstSize{dst.width(z), dst.height(z)};
    int2 srcSize{src.borderWrap().imageBatchWrap().width(z), src.borderWrap().imageBatchWrap().height(z)};

    NVCVRemapParams params = GetRemapParams(srcSize, dstSize, mapSize, alignCorners, mapValueType);

    DoRemap<NIX>(src, dst, map, dstSize, mapNumSamples, params);
}

// Remap with planar (CHW) varshape kernel -------------------------------------
//
// Each planar image has `channels` single-channel planes; grid-z runs over numImages.
// The per-pixel source coordinate (map lookup + interpolation) is computed once per
// (image, x, y) and reused for every plane, amortizing the map traffic and coordinate
// math across channels. Each plane is sampled via the 4D (sample, plane, y, x)
// coordinate the var-shape wraps accept. Remap treats channels independently, so this
// matches the interleaved result plane-for-plane.
template<int NIX, class SrcWrapper, class DstWrapper, class MapWrapper>
__global__ void RemapPlanarVarShape(SrcWrapper src, DstWrapper dst, MapWrapper map, int2 mapSize, int mapNumSamples,
                                    bool alignCorners, NVCVRemapMapValueType mapValueType, int channels)
{
    int image = blockIdx.z;

    int baseX = (blockIdx.x * blockDim.x) * NIX + threadIdx.x;
    int y     = blockIdx.y * blockDim.y + threadIdx.y;

    int2 dstSize{dst.width(image), dst.height(image)};
    if (y >= dstSize.y)
    {
        return;
    }

    int2 srcSize{src.borderWrap().imageBatchWrap().width(image), src.borderWrap().imageBatchWrap().height(image)};

    NVCVRemapParams params = GetRemapParams(srcSize, dstSize, mapSize, alignCorners, mapValueType);

    int mapSample = (mapNumSamples == 1) ? 0 : image;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        int x = baseX + i * blockDim.x;
        if (x >= dstSize.x)
        {
            continue;
        }

        float2 sc = ComputeRemapSrcCoord(x, y, map, mapSample, params);

        for (int plane = 0; plane < channels; ++plane)
        {
            *dst.ptr(image, plane, y, x)
                = src[float4{sc.x, sc.y, static_cast<float>(plane), static_cast<float>(image)}];
        }
    }
}

// Host run remap functions ----------------------------------------------------

template<typename T, NVCVBorderType B, NVCVInterpolationType SI, class DataStridedCuda, typename MapWrapper>
void RunRemap(cudaStream_t stream, const DataStridedCuda &srcData, const DataStridedCuda &dstData,
              const MapWrapper &mapWrap, NVCVRemapMapValueType mapValueType, bool alignCorners, const T &borderValue,
              int2 mapSize, int mapNumSamples, int channels = 1)
{
    dim3 block(32, 4, 1);

    // CUBIC interpolation is compute-bound at full occupancy; the extra per-thread
    // x-unrolling only adds register/loop overhead there, so disable it (NIX == 1).
    // NEAREST/LINEAR are latency-bound and benefit from the overlap.
    constexpr int NIX = (SI == NVCV_INTERP_CUBIC) ? 1 : kRemapNIX;

    if constexpr (std::is_same_v<DataStridedCuda, nvcv::TensorDataStridedCuda>)
    {
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);

        int2 srcSize = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});
        int2 dstSize = cuda::StaticCast<int>(long2{dstAccess->numCols(), dstAccess->numRows()});

        NVCVRemapParams params = GetRemapParams(srcSize, dstSize, mapSize, alignCorners, mapValueType);

        dim3 grid(util::DivUp(dstSize.x, block.x * NIX), util::DivUp(dstSize.y, block.y), dstAccess->numSamples());

        int64_t srcMaxStride = srcAccess->sampleStride() * srcAccess->numSamples();
        int64_t dstMaxStride = dstAccess->sampleStride() * dstAccess->numSamples();
        if (std::max(srcMaxStride, dstMaxStride) <= cuda::TypeTraits<int32_t>::max)
        {
            auto src = cuda::CreateInterpolationWrapNHW<const T, B, SI, int32_t>(srcData, borderValue);
            auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(dstData);

            if (channels > 1)
            {
                dim3 planarGrid(util::DivUp(dstSize.x, block.x * NIX), util::DivUp(dstSize.y, block.y),
                                dstAccess->numSamples() / channels);
                RemapPlanarTensor<NIX>
                    <<<planarGrid, block, 0, stream>>>(src, dst, mapWrap, dstSize, mapNumSamples, params, channels);
            }
            else
            {
                Remap<NIX><<<grid, block, 0, stream>>>(src, dst, mapWrap, dstSize, mapNumSamples, params, channels);
            }
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }
    }
    else
    {
        static_assert(std::is_same_v<DataStridedCuda, nvcv::ImageBatchVarShapeDataStridedCuda>);

        int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};

        cuda::InterpolationVarShapeWrap<const T, B, SI> src(srcData, borderValue);
        cuda::ImageBatchVarShapeWrap<T>                 dst(dstData);

        // A valid multi-plane format has at most four channels in total, so a four-channel first plane
        // cannot have another plane. Avoid instantiating that unreachable planar uchar4 kernel family.
        if constexpr (cuda::NumElements<T> < 4)
        {
            if (channels > 1)
            {
                // Planar (CHW) var-shape: grid-z spans numImages; each thread amortizes the source
                // coordinate across all channel planes via 4D var-shape coordinates.
                dim3 grid(util::DivUp(dstMaxSize.x, block.x * NIX), util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);
                RemapPlanarVarShape<NIX><<<grid, block, 0, stream>>>(src, dst, mapWrap, mapSize, mapNumSamples,
                                                                     alignCorners, mapValueType, channels);
                return;
            }
        }

        dim3 grid(util::DivUp(dstMaxSize.x, block.x * NIX), util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);
        Remap<NIX><<<grid, block, 0, stream>>>(src, dst, mapWrap, mapSize, mapNumSamples, alignCorners, mapValueType);
    }
}

template<typename T, NVCVBorderType B, NVCVInterpolationType MI, NVCVInterpolationType SI, class DataStridedCuda>
void RunRemap(cudaStream_t stream, const DataStridedCuda &srcData, const DataStridedCuda &dstData,
              const nvcv::TensorDataStridedCuda &mapData, NVCVRemapMapValueType mapValueType, bool alignCorners,
              const T &borderValue, int channels = 1)
{
    auto mapAccess     = nvcv::TensorDataAccessStridedImagePlanar::Create(mapData);
    int2 mapSize       = cuda::StaticCast<int>(long2{mapAccess->numCols(), mapAccess->numRows()});
    int  mapNumSamples = mapAccess->numSamples();

    if (mapAccess->sampleStride() * mapAccess->numSamples() <= cuda::TypeTraits<int32_t>::max)
    {
        auto map = cuda::CreateInterpolationWrapNHW<const float2, kMapBorderType, MI, int32_t>(mapData);
        RunRemap<T, B, SI>(stream, srcData, dstData, map, mapValueType, alignCorners, borderValue, mapSize,
                           mapNumSamples, channels);
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Map size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }
}

template<typename T, NVCVBorderType B, NVCVInterpolationType MI, class DataStridedCuda>
void RunRemap(cudaStream_t stream, const DataStridedCuda &srcData, const DataStridedCuda &dstData,
              const nvcv::TensorDataStridedCuda &mapData, NVCVInterpolationType srcInterp,
              NVCVRemapMapValueType mapValueType, bool alignCorners, const T &borderValue, int channels = 1)
{
#define NVCV_RUN_REMAP(INTERP_TYPE)                                                                                  \
    case NVCV_INTERP_##INTERP_TYPE:                                                                                  \
        RunRemap<T, B, MI, NVCV_INTERP_##INTERP_TYPE>(stream, srcData, dstData, mapData, mapValueType, alignCorners, \
                                                      borderValue, channels);                                        \
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
              const T &borderValue, int channels = 1)
{
#define NVCV_RUN_REMAP(INTERP_TYPE)                                                                           \
    case NVCV_INTERP_##INTERP_TYPE:                                                                           \
        RunRemap<T, B, NVCV_INTERP_##INTERP_TYPE>(stream, srcData, dstData, mapData, srcInterp, mapValueType, \
                                                  alignCorners, borderValue, channels);                       \
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
              NVCVBorderType border, const float4 &borderValue, int channels = 1)
{
    const T bvalue = cuda::DropCast<cuda::NumElements<T>>(cuda::StaticCast<cuda::BaseType<T>>(borderValue));

#define NVCV_RUN_REMAP(BORDER_TYPE)                                                                                   \
    case NVCV_BORDER_##BORDER_TYPE:                                                                                   \
        RunRemap<T, NVCV_BORDER_##BORDER_TYPE>(stream, srcData, dstData, mapData, srcInterp, mapInterp, mapValueType, \
                                               alignCorners, bvalue, channels);                                       \
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
                     NVCVBorderType border, const float4 &borderValue, nvcv::DataType dataType, int numChannels = 1,
                     int mapSampleChannels = 1)
{
    // When this function is called with tensors, the data type may contain the channels baked in or the number of
    // channels is in the tensor shape; when it is called with varshape, the data type always contain the channels
    // as each image only stores size, there is no shape information with number of channels baked in.
    //
    // numChannels selects the element type T (e.g. uchar3 vs single-channel uchar1 for a flattened planar view).
    // mapSampleChannels is the planar channel count used to map a flattened grid-z plane back to its image index
    // for the per-sample map lookup (1 for interleaved/var-shape, C for the planar single-channel tensor view).

    // clang-format off

#define NVCV_RUN_REMAP(BT, DT, T)                                                          \
    ((dataType == nvcv::TYPE_##BT && numChannels == cuda::NumElements<T>) ||               \
     (dataType == nvcv::TYPE_##DT && numChannels == 1))                                    \
        RunRemap<T>(stream, srcData, dstData, mapData, srcInterp, mapInterp, mapValueType, \
                    alignCorners, border, borderValue, mapSampleChannels)

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

// The planar path flattens the C channel planes into single-channel samples that all share one
// border scalar (borderValue.x). The interleaved path applies borderValue componentwise, so for
// NVCV_BORDER_CONSTANT the two only agree when the first `channels` border components are equal.
// Reject the non-uniform case rather than silently producing a different result for out-of-bounds
// samples (other border modes ignore borderValue, so they are always fine).
inline void RequireUniformConstantBorderForPlanar(NVCVBorderType border, const float4 &borderValue, int channels)
{
    if (border != NVCV_BORDER_CONSTANT)
    {
        return;
    }
    const float *bv = &borderValue.x;
    for (int c = 1; c < channels; ++c)
    {
        if (bv[c] != bv[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar remap with NVCV_BORDER_CONSTANT requires a uniform border value across "
                                  "channels");
        }
    }
}

} // anonymous namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

Remap::Remap() {}

// Tensor operator -------------------------------------------------------------

void Remap::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, const nvcv::Tensor &map,
                       NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp,
                       NVCVRemapMapValueType mapValueType, bool alignCorners, NVCVBorderType border,
                       float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Remap::operator()[Tensor]");
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

    const bool srcPlanar      = (srcData->layout() == nvcv::TENSOR_NCHW || srcData->layout() == nvcv::TENSOR_CHW);
    const bool dstPlanar      = (dstData->layout() == nvcv::TENSOR_NCHW || dstData->layout() == nvcv::TENSOR_CHW);
    const bool srcInterleaved = (srcData->layout() == nvcv::TENSOR_HWC || srcData->layout() == nvcv::TENSOR_NHWC);

    if (!(srcInterleaved || srcPlanar))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC or (N)CHW layout");
    }
    if (srcPlanar != dstPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must share the same layout family ((N)HWC or (N)CHW)");
    }

    if (srcPlanar)
    {
        // Remap treats channels independently: flatten (N,C,H,W) into N*C single-channel samples and
        // reuse the interleaved single-channel kernel, looking the per-sample map up at plane/channels
        // (see PlanarTensorView.hpp / DoRemap). The flattened plane count is the kernel grid-z, which
        // is capped at CUDA's 65535 limit.
        const int     channels    = srcAccess->numChannels();
        const int64_t planarBatch = static_cast<int64_t>(channels) * srcAccess->numSamples();
        if (planarBatch > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar remap requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
        }
        RequireUniformConstantBorderForPlanar(border, borderValue, channels);

        auto srcView = PlanarAsSingleChannelView(*srcData, *srcAccess);
        auto dstView = PlanarAsSingleChannelView(*dstData, *dstAccess);
        RunRemap(stream, srcView, dstView, *mapData, srcInterp, mapInterp, mapValueType, alignCorners, border,
                 borderValue, dstData->dtype(), /*numChannels=*/1, /*mapSampleChannels=*/channels);
        return;
    }

    RunRemap(stream, *srcData, *dstData, *mapData, srcInterp, mapInterp, mapValueType, alignCorners, border,
             borderValue, dstData->dtype(), dstAccess->numChannels());
}

// VarShape operator -----------------------------------------------------------

void Remap::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst,
                       const nvcv::Tensor &map, NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp,
                       NVCVRemapMapValueType mapValueType, bool alignCorners, NVCVBorderType border,
                       float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Remap::operator()[ImageBatchVarShape]");
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

    const int numPlanes = srcData->uniqueFormat().numPlanes();

    if (numPlanes > 1)
    {
        // Planar (CHW) var-shape: each image stores numPlanes single-channel planes; remap each
        // plane independently with the single-channel element type. grid-z spans
        // numImages*numPlanes, capped at CUDA's 65535 limit.
        if (static_cast<int64_t>(srcData->numImages()) * numPlanes > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar remap requires numImages * numChannels <= 65535 (CUDA grid-z limit)");
        }
        RequireUniformConstantBorderForPlanar(border, borderValue, numPlanes);

        RunRemap(stream, *srcData, *dstData, *mapData, srcInterp, mapInterp, mapValueType, alignCorners, border,
                 borderValue, srcData->uniqueFormat().planeDataType(0), /*numChannels=*/1,
                 /*mapSampleChannels=*/numPlanes);
        return;
    }

    RunRemap(stream, *srcData, *dstData, *mapData, srcInterp, mapInterp, mapValueType, alignCorners, border,
             borderValue, dstData->uniqueFormat().planeDataType(0));
}

} // namespace cvcuda::priv

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
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

#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpNormalize.hpp"

#include "ElementStreamCommon.cuh"

#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cvcuda/OpNormalize.h> // for CVCUDA_NORMALIZE_SCALE_IS_STDDEV
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace cuda        = nvcv::cuda;
namespace elem_stream = cvcuda::priv::elem_stream;
namespace util        = nvcv::util;

namespace {

// Kernels and leaf launchers keep their pre-migration lowerCamel names: this operator's optimization
// commits cite those symbols verbatim from profiler output.

using uchar  = unsigned char;
using schar  = signed char;
using ushort = unsigned short;

struct DataShape
{
    int N;
    int C;
    int H;
    int W;
};

inline bool operator==(const DataShape &a, const DataShape &b)
{
    return a.N == b.N && a.C == b.C && a.H == b.H && a.W == b.W;
}

inline DataShape GetDataShape(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    return DataShape{static_cast<int>(access.numSamples()), access.numChannels(), access.numRows(), access.numCols()};
}

constexpr int64_t kMaxGridZ = 65535;

struct NormalizeParams
{
    float        globalScale;
    float        shift;
    float        epsilon;
    bool         isStdDev;
    cudaStream_t stream;
};

using Access = nvcv::TensorDataAccessStridedImagePlanar;

struct TensorArgs
{
    const nvcv::TensorDataStridedCuda &in;
    const nvcv::TensorDataStridedCuda &base;
    const nvcv::TensorDataStridedCuda &scale;
    const nvcv::TensorDataStridedCuda &out;
    const Access                      &inAccess;
    const Access                      &baseAccess;
    const Access                      &scaleAccess;
    const Access                      &outAccess;
    NormalizeParams                    p;
};

struct ScalarArgs
{
    const nvcv::TensorDataStridedCuda &in;
    float4                             base;
    float4                             scale;
    int                                baseCount;
    int                                scaleCount;
    const nvcv::TensorDataStridedCuda &out;
    const Access                      &inAccess;
    const Access                      &outAccess;
    NormalizeParams                    p;
};

struct VarShapeArgs
{
    const nvcv::ImageBatchVarShapeDataStridedCuda &in;
    const nvcv::TensorDataStridedCuda             &base;
    const nvcv::TensorDataStridedCuda             &scale;
    const Access                                  &baseAccess;
    const Access                                  &scaleAccess;
    const nvcv::ImageBatchVarShapeDataStridedCuda &out;
    int                                            channels;
    NormalizeParams                                p;
};

// The kernels index every wrap with int32_t offsets.
inline void RequireStridesFitInt32(std::initializer_list<const Access *> accesses, const char *what)
{
    for (const Access *a : accesses)
    {
        if (a->sampleStride() * a->numSamples() > cuda::TypeTraits<int32_t>::max)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s size exceeds %d. Tensor is too large.",
                                  what, cuda::TypeTraits<int32_t>::max);
        }
    }
}

inline dim3 NormGrid(int width, int height, unsigned int z, dim3 block, int colsPerThread = 1)
{
    return dim3(util::DivUp(width, static_cast<int>(block.x) * colsPerThread),
                util::DivUp(height, static_cast<int>(block.y)), z);
}

// The classifiers throw on unnamed layouts/types before the operator's own checks, so call order
// sets the rejection status.
using cvcuda::priv::legacy_format::ClassifyDataFormat;
using cvcuda::priv::legacy_format::ClassifyDataType;
using cvcuda::priv::legacy_format::DataFormatCode;
using cvcuda::priv::legacy_format::DataTypeCode;
using cvcuda::priv::legacy_format::IsPlanar;

// A single channel keeps the plain scalar type (uchar, not uchar1), matching the element types the
// legacy dispatch tables named.
template<typename T, int NC>
using PixelType = cuda::MakeType<T, NC == 1 ? 0 : NC>;

// ---------------------------------------------------------------------------------------------------
// Shared per-element device math.
// ---------------------------------------------------------------------------------------------------

// Per-axis broadcast read index: 0 wherever the parameter extent is 1, else the data index.
__device__ __forceinline__ int4 PlanarBroadcastIndex(int4 param_size, int batch, int channel, int y, int x)
{
    return int4{param_size.x == 1 ? 0 : batch, param_size.y == 1 ? 0 : channel, param_size.z == 1 ? 0 : y,
                param_size.w == 1 ? 0 : x};
}

// The multiplier CVCUDA_NORMALIZE_SCALE_IS_STDDEV substitutes for the raw scale. S may be compound
// (float3 / float4) on the multi-channel interleaved paths, so the result type is deduced.
template<typename S>
__device__ __forceinline__ auto InvStdDevMul(S scale, float epsilon)
{
    return 1.0f / cuda::sqrt(scale * scale + epsilon);
}

// The interleaved kernels promote F64 implicitly through their wrapped pixel types; the helpers
// below take a plain scalar, so a hardcoded float would make NCHW disagree with NHWC. The
// multiplier stays float on both, which is what keeps them agreeing rather than merely both wide.
template<typename SrcT>
using NormalizeWorkType = std::conditional_t<std::is_same_v<cuda::BaseType<SrcT>, double>, double, float>;

template<typename OutT, typename SrcT>
__device__ __forceinline__ OutT ApplyPlanarNormalize(SrcT src_val, float base_val, float scale_val, float global_scale,
                                                     float global_shift, bool is_stddev, float epsilon)
{
    const float mul = is_stddev ? InvStdDevMul(scale_val, epsilon) : scale_val;
    return cuda::SaturateCast<OutT>((static_cast<NormalizeWorkType<SrcT>>(src_val) - base_val) * mul * global_scale
                                    + global_shift);
}

// SaturateCast<OutT> is the identity for float, so the F32 kernels evaluate exactly the float
// expression they spelled out before this was factored out.
template<typename OutT, typename RawT>
__device__ __forceinline__ OutT ApplyNormalize(RawT raw, float baseV, float mul, float global_scale, float global_shift)
{
    return cuda::SaturateCast<OutT>((static_cast<NormalizeWorkType<RawT>>(raw) - baseV) * mul * global_scale
                                    + global_shift);
}

// Per-thread broadcast context for the interleaved vectorized inverse-std-dev kernels: the batch and
// row axes resolve once per thread, leaving only the column axis per element.
template<typename BaseWrapper, typename ScaleWrapper>
struct NormInvStdDevCtx
{
    BaseWrapper  base;
    ScaleWrapper scale;
    bool         base_bcast_x;
    bool         scale_bcast_x;
    int          base_y;
    int          base_batch_idx;
    int          scale_y;
    int          scale_batch_idx;
    float        global_scale;
    float        global_shift;
    float        epsilon;
};

template<typename BaseWrapper, typename ScaleWrapper>
__device__ __forceinline__ NormInvStdDevCtx<BaseWrapper, ScaleWrapper> MakeNormInvStdDevCtx(
    BaseWrapper base, ScaleWrapper scale, int3 base_size, int3 scale_size, int src_y, int batch_idx, float global_scale,
    float global_shift, float epsilon)
{
    return {base,
            scale,
            base_size.x == 1,
            scale_size.x == 1,
            base_size.y == 1 ? 0 : src_y,
            base_size.z == 1 ? 0 : batch_idx,
            scale_size.y == 1 ? 0 : src_y,
            scale_size.z == 1 ? 0 : batch_idx,
            global_scale,
            global_shift,
            epsilon};
}

template<typename T, typename BaseWrapper, typename ScaleWrapper>
__device__ __forceinline__ T NormInvStdDevOne(T raw, int x, const NormInvStdDevCtx<BaseWrapper, ScaleWrapper> &c)
{
    const int   base_x  = c.base_bcast_x ? 0 : x;
    const int   scale_x = c.scale_bcast_x ? 0 : x;
    const float mul     = InvStdDevMul(*c.scale.ptr(c.scale_batch_idx, c.scale_y, scale_x), c.epsilon);
    return ApplyNormalize<T>(raw, *c.base.ptr(c.base_batch_idx, c.base_y, base_x), mul, c.global_scale, c.global_shift);
}

template<typename BaseWrapper, typename ScaleWrapper>
__device__ __forceinline__ void ReadBroadcastBaseAndMul(const NormInvStdDevCtx<BaseWrapper, ScaleWrapper> &c,
                                                        float &baseV, float &mul)
{
    baseV = *c.base.ptr(c.base_batch_idx, c.base_y, 0);
    mul   = InvStdDevMul(*c.scale.ptr(c.scale_batch_idx, c.scale_y, 0), c.epsilon);
}

// ---------------------------------------------------------------------------------------------------
// ILP depth ("NGROUP"): each thread owns that many independent vector groups and issues all of their
// loads before any compute. Every scalar Normalize kernel moves one element per thread, leaving the
// narrow paths latency- rather than bandwidth-bound; a wider request raises bytes-per-request but not
// the number of in-flight requests, so it is issuing several independent loads first that raises
// memory-level parallelism at fixed occupancy. Each depth is measured separately, because the
// register cost of a group -- and so the depth at which occupancy starts to fall -- differs per path.
// ---------------------------------------------------------------------------------------------------

constexpr int kNormalizePlanarILPNGroup    = 2; // 1-byte planar tensor (uchar4 groups)
constexpr int kNormalizePlanarF32ILPNGroup = 2; // F32 planar tensor (float4 groups, 16 B live per group)
constexpr int kNormalizeILPNGroup          = 4; // single-channel 8-bit interleaved tensor (uint4 groups)
constexpr int kNormalizeF32ILPNGroup       = 4; // single-channel F32 interleaved tensor (float4 groups)
constexpr int kNormVarShapeILPNGroup       = 4; // single-channel 1-byte interleaved var-shape
constexpr int kNormVarShapeF32ILPNGroup    = 4; // single-channel F32 interleaved var-shape
constexpr int kNormVarShapePlanarILPNGroup = 4; // spatially-broadcast 1-byte planar var-shape

// Not an ILP depth: the hoist kernels hold no group array, they reuse one multiplier across this many
// strided pixels, which is what cuts the redundant sqrt on the issue-bound multi-channel 1-byte paths.
constexpr int kNormalizeHoistNIX = 4;

// Vectorized (4 columns/thread) planar body for the var-shape planar kernels. The tensor planar
// kernel keeps its own loop rather than calling this: its caller guards stride alignment up front, so
// it must not pay the per-group probe below, which exists because a var-shape image's row pitch is
// not known to the launcher.
//
// Known divergence preserved from the pre-migration kernel: this saturates to the *input* type T
// while the scalar normPlanarKernel saturates to the output type. They agree for every pair the
// var-shape planar guard admits except signed-8-bit in / unsigned-8-bit out.
template<typename T, typename SrcWrap, typename DstWrap, typename ParamWrap>
__device__ __forceinline__ void NormalizePlanarVec4(const SrcWrap &src, const DstWrap &dst, const ParamWrap &base,
                                                    const ParamWrap &scale, int batch, int channel, int src_y, int cx,
                                                    int width, int4 base_size, int4 scale_size, float global_scale,
                                                    float global_shift, bool is_stddev, float epsilon)
{
    using Vec4 = cuda::MakeType<T, 4>;

    // Every NVCV row pitch is 4-byte aligned, but a user-wrapped buffer can pick one that is a
    // multiple of 4 yet not the 16 a float4 access needs.
    const bool vecAligned = sizeof(T) == 1
                         || (reinterpret_cast<uintptr_t>(src.ptr(batch, channel, src_y, cx)) % sizeof(Vec4) == 0
                             && reinterpret_cast<uintptr_t>(dst.ptr(batch, channel, src_y, cx)) % sizeof(Vec4) == 0);

    const auto applyAt = [&](T raw, int x) -> T
    {
        const int4 b = PlanarBroadcastIndex(base_size, batch, channel, src_y, x);
        const int4 s = PlanarBroadcastIndex(scale_size, batch, channel, src_y, x);
        return ApplyPlanarNormalize<T>(raw, *base.ptr(b.x, b.y, b.z, b.w), *scale.ptr(s.x, s.y, s.z, s.w), global_scale,
                                       global_shift, is_stddev, epsilon);
    };

    if (cx + 4 <= width && vecAligned)
    {
        const Vec4 in4 = *reinterpret_cast<const Vec4 *>(src.ptr(batch, channel, src_y, cx));
        Vec4       out4;
        if (base_size.w == 1 && scale_size.w == 1)
        {
            const int4  b      = PlanarBroadcastIndex(base_size, batch, channel, src_y, 0);
            const int4  s      = PlanarBroadcastIndex(scale_size, batch, channel, src_y, 0);
            const float baseV  = *base.ptr(b.x, b.y, b.z, b.w);
            const float scaleV = *scale.ptr(s.x, s.y, s.z, s.w);
            const float mul    = is_stddev ? InvStdDevMul(scaleV, epsilon) : scaleV;

            out4.x = ApplyNormalize<T>(in4.x, baseV, mul, global_scale, global_shift);
            out4.y = ApplyNormalize<T>(in4.y, baseV, mul, global_scale, global_shift);
            out4.z = ApplyNormalize<T>(in4.z, baseV, mul, global_scale, global_shift);
            out4.w = ApplyNormalize<T>(in4.w, baseV, mul, global_scale, global_shift);
        }
        else
        {
            out4.x = applyAt(static_cast<T>(in4.x), cx + 0);
            out4.y = applyAt(static_cast<T>(in4.y), cx + 1);
            out4.z = applyAt(static_cast<T>(in4.z), cx + 2);
            out4.w = applyAt(static_cast<T>(in4.w), cx + 3);
        }
        *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, src_y, cx)) = out4;
    }
    else
    {
        // Bound to this group's columns so threads do not overlap; a partial group clamps to width.
        const int xEnd = cx + 4 < width ? cx + 4 : width;
        for (int x = cx; x < xEnd; ++x)
        {
            *dst.ptr(batch, channel, src_y, x) = applyAt(*src.ptr(batch, channel, src_y, x), x);
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// Tensor path: planar (NCHW / CHW) kernels.
// ---------------------------------------------------------------------------------------------------

// base/scale are template wrapper types rather than a fixed Tensor4DWrap<float> so the same kernel
// serves the parameter-tensor path and the by-value path; both only need ptr(n, c, y, x).
template<typename T, typename BaseWrap, typename ScaleWrap>
__global__ void normalizePlanarKernel(const cuda::Tensor4DWrap<T, int32_t> src, const BaseWrap base,
                                      const ScaleWrap scale, cuda::Tensor4DWrap<T, int32_t> dst, int4 inout_size,
                                      int4 base_size, int4 scale_size, float global_scale, float global_shift,
                                      float epsilon, bool is_stddev)
{
    const int src_x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / inout_size.y;
    const int channel = nc % inout_size.y;

    if (src_x >= inout_size.w || src_y >= inout_size.z)
        return;

    const int4 b = PlanarBroadcastIndex(base_size, batch, channel, src_y, src_x);
    const int4 s = PlanarBroadcastIndex(scale_size, batch, channel, src_y, src_x);

    *dst.ptr(batch, channel, src_y, src_x)
        = ApplyPlanarNormalize<T>(*src.ptr(batch, channel, src_y, src_x), *base.ptr(b.x, b.y, b.z, b.w),
                                  *scale.ptr(s.x, s.y, s.z, s.w), global_scale, global_shift, is_stddev, epsilon);
}

// Requires Vec4-aligned rows/planes/samples and base pointers, caller-guarded for F16.
template<int NGROUP, typename T, typename BaseWrap, typename ScaleWrap>
__global__ void normalizePlanarVec4Kernel(const cuda::Tensor4DWrap<T, int32_t> src, const BaseWrap base,
                                          const ScaleWrap scale, cuda::Tensor4DWrap<T, int32_t> dst, int4 inout_size,
                                          int4 base_size, int4 scale_size, float global_scale, float global_shift,
                                          float epsilon, bool is_stddev)
{
    const int g0      = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int src_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / inout_size.y;
    const int channel = nc % inout_size.y;

    const int cx0 = g0 * 4;
    if (cx0 >= inout_size.w || src_y >= inout_size.z)
        return;

    const int width = inout_size.w;

    using Vec4 = cuda::MakeType<T, 4>;

    int  cx[NGROUP];
    bool full[NGROUP];
    Vec4 in4[NGROUP];
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        cx[i]   = (g0 + i * blockDim.x) * 4;
        full[i] = cx[i] + 4 <= width;
        if (full[i])
            in4[i] = *reinterpret_cast<const Vec4 *>(src.ptr(batch, channel, src_y, cx[i]));
    }

    if (base_size.w == 1 && scale_size.w == 1)
    {
        const int4  b      = PlanarBroadcastIndex(base_size, batch, channel, src_y, 0);
        const int4  s      = PlanarBroadcastIndex(scale_size, batch, channel, src_y, 0);
        const float baseV  = *base.ptr(b.x, b.y, b.z, b.w);
        const float scaleV = *scale.ptr(s.x, s.y, s.z, s.w);
        const float mul    = is_stddev ? InvStdDevMul(scaleV, epsilon) : scaleV;
        auto        apply  = [&](cuda::BaseType<Vec4> raw) -> T
        {
            return ApplyNormalize<T>(raw, baseV, mul, global_scale, global_shift);
        };

#pragma unroll
        for (int i = 0; i < NGROUP; ++i)
        {
            if (full[i])
            {
                Vec4 out4;
                out4.x                                                           = apply(in4[i].x);
                out4.y                                                           = apply(in4[i].y);
                out4.z                                                           = apply(in4[i].z);
                out4.w                                                           = apply(in4[i].w);
                *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, src_y, cx[i])) = out4;
            }
            else if (cx[i] < width)
            {
                for (int x = cx[i]; x < width; ++x)
                {
                    *dst.ptr(batch, channel, src_y, x) = apply(*src.ptr(batch, channel, src_y, x));
                }
            }
        }
    }
    else
    {
        auto applyAt = [&](cuda::BaseType<Vec4> raw, int x) -> T
        {
            const int4 b = PlanarBroadcastIndex(base_size, batch, channel, src_y, x);
            const int4 s = PlanarBroadcastIndex(scale_size, batch, channel, src_y, x);
            return ApplyPlanarNormalize<T>(static_cast<T>(raw), *base.ptr(b.x, b.y, b.z, b.w),
                                           *scale.ptr(s.x, s.y, s.z, s.w), global_scale, global_shift, is_stddev,
                                           epsilon);
        };

#pragma unroll
        for (int i = 0; i < NGROUP; ++i)
        {
            if (full[i])
            {
                const cuda::BaseType<Vec4> raw[4] = {in4[i].x, in4[i].y, in4[i].z, in4[i].w};
                Vec4                       out4;
                out4.x                                                           = applyAt(raw[0], cx[i] + 0);
                out4.y                                                           = applyAt(raw[1], cx[i] + 1);
                out4.z                                                           = applyAt(raw[2], cx[i] + 2);
                out4.w                                                           = applyAt(raw[3], cx[i] + 3);
                *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, src_y, cx[i])) = out4;
            }
            else if (cx[i] < width)
            {
                for (int x = cx[i]; x < width; ++x)
                {
                    *dst.ptr(batch, channel, src_y, x) = applyAt(*src.ptr(batch, channel, src_y, x), x);
                }
            }
        }
    }
}

inline int4 PlanarExtents(const Access &access)
{
    const DataShape s = GetDataShape(access);
    return int4{s.N, s.C, s.H, s.W};
}

inline unsigned int PlanarGridZ(int numSamples, int numChannels)
{
    const int64_t planes = static_cast<int64_t>(numSamples) * numChannels;
    if (planes > kMaxGridZ)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar normalize launch exceeds CUDA grid.z limit: N*C=%ld", static_cast<long>(planes));
    }
    return static_cast<unsigned int>(planes);
}

bool IsCurrentDeviceL4NameOnly(); // Avoid a missing-declarations warning in nvcc's host compilation.

// Query only the name: a per-submission bulk properties query is costly on the measured L4 host.
bool IsCurrentDeviceL4NameOnly()
{
    int device;
    NVCV_CHECK_THROW(cudaGetDevice(&device));
    void                           *getDeviceEntry = nullptr, *getNameEntry = nullptr;
    cudaDriverEntryPointQueryResult deviceStatus, nameStatus;
#if CUDART_VERSION >= 13000
    NVCV_CHECK_THROW(
        cudaGetDriverEntryPointByVersion("cuDeviceGet", &getDeviceEntry, 2000, cudaEnableDefault, &deviceStatus));
    NVCV_CHECK_THROW(
        cudaGetDriverEntryPointByVersion("cuDeviceGetName", &getNameEntry, 2000, cudaEnableDefault, &nameStatus));
#else
    NVCV_CHECK_THROW(cudaGetDriverEntryPoint("cuDeviceGet", &getDeviceEntry, cudaEnableDefault, &deviceStatus));
    NVCV_CHECK_THROW(cudaGetDriverEntryPoint("cuDeviceGetName", &getNameEntry, cudaEnableDefault, &nameStatus));
#endif
    if (deviceStatus != cudaDriverEntryPointSuccess || nameStatus != cudaDriverEntryPointSuccess
        || getDeviceEntry == nullptr || getNameEntry == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CUDA device identity entry points are unavailable");
    }
    CUdevice driverDevice;
    char     name[256];
    if (reinterpret_cast<PFN_cuDeviceGet_v2000>(getDeviceEntry)(&driverDevice, device) != CUDA_SUCCESS
        || reinterpret_cast<PFN_cuDeviceGetName_v2000>(getNameEntry)(name, sizeof(name), driverDevice) != CUDA_SUCCESS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "CUDA device identity query failed");
    }
    return std::strcmp(name, "NVIDIA L4") == 0;
}

struct PlanarConstParamWrap;

// base/scale are generic wrappers so the parameter-tensor and by-value callers share every launch
// decision; only the wrapper type differs.
template<typename T, typename BaseWrap, typename ScaleWrap>
void LaunchPlanar(const cuda::Tensor4DWrap<T, int32_t> &srcWrap, const cuda::Tensor4DWrap<T, int32_t> &dstWrap,
                  const BaseWrap &baseWrap, const ScaleWrap &scaleWrap, const Access &inAccess, const Access &outAccess,
                  int4 base_size, int4 scale_size, const NormalizeParams &p)
{
    const int4         inout_size = PlanarExtents(inAccess);
    const unsigned int planes     = PlanarGridZ(inout_size.x, inout_size.y);

    dim3 block(32, 8);

    constexpr bool kIsByte = sizeof(T) == 1;
    constexpr bool kIsF32  = std::is_same_v<T, float>;
    constexpr bool kIsF16  = std::is_same_v<T, __half>;

    if constexpr (kIsByte || kIsF32 || kIsF16)
    {
        constexpr int64_t kAlign  = sizeof(cuda::MakeType<T, 4>);
        constexpr int     kNGroup = kIsF16 ? 2 : (kIsByte ? kNormalizePlanarILPNGroup : kNormalizePlanarF32ILPNGroup);

        const bool aligned = inAccess.rowStride() % kAlign == 0 && inAccess.chStride() % kAlign == 0
                          && inAccess.sampleStride() % kAlign == 0 && outAccess.rowStride() % kAlign == 0
                          && outAccess.chStride() % kAlign == 0 && outAccess.sampleStride() % kAlign == 0
                          && (!kIsF16
                              || (reinterpret_cast<uintptr_t>(inAccess.sampleData(0)) % kAlign == 0
                                  && reinterpret_cast<uintptr_t>(outAccess.sampleData(0)) % kAlign == 0));
        bool useVec4 = aligned;
        if constexpr (kIsF16)
        {
            const bool spatialBroadcast
                = base_size.z == 1 && base_size.w == 1 && scale_size.z == 1 && scale_size.w == 1;
            if (aligned && spatialBroadcast
                && (!p.isStdDev
                    || std::is_same_v<BaseWrap, PlanarConstParamWrap>)&&elem_stream::DeviceComputeCapability()
                       == 89)
            {
                // Native L4 measurements favor the scalar kernel for these spatially invariant modes.
                useVec4 = !IsCurrentDeviceL4NameOnly();
            }
        }
        if (useVec4)
        {
            dim3 vgrid = NormGrid(inout_size.w, inout_size.z, planes, block, 4 * kNGroup);
            normalizePlanarVec4Kernel<kNGroup, T>
                <<<vgrid, block, 0, p.stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size,
                                                scale_size, p.globalScale, p.shift, p.epsilon, p.isStdDev);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }
    }

    dim3 grid = NormGrid(inout_size.w, inout_size.z, planes, block);
    normalizePlanarKernel<T><<<grid, block, 0, p.stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size,
                                                           scale_size, p.globalScale, p.shift, p.epsilon, p.isStdDev);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void normalizePlanar(const TensorArgs &a)
{
    RequireStridesFitInt32({&a.inAccess, &a.outAccess, &a.baseAccess, &a.scaleAccess}, "Input, output, base, or scale");

    LaunchPlanar<T>(cuda::CreateTensorWrapNCHW<T, int32_t>(a.in), cuda::CreateTensorWrapNCHW<T, int32_t>(a.out),
                    cuda::CreateTensorWrapNCHW<float, int32_t>(a.base),
                    cuda::CreateTensorWrapNCHW<float, int32_t>(a.scale), a.inAccess, a.outAccess,
                    PlanarExtents(a.baseAccess), PlanarExtents(a.scaleAccess), a.p);
}

// ---------------------------------------------------------------------------------------------------
// Tensor path: interleaved (NHWC / HWC) kernels.
// ---------------------------------------------------------------------------------------------------

template<typename input_type, typename base_type, typename scale_type>
__global__ void normalizeKernel(const input_type src, const base_type base, const scale_type scale, input_type dst,
                                int2 inout_size, int3 base_size, int3 scale_size, float global_scale,
                                float global_shift)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    if (src_x >= inout_size.x || src_y >= inout_size.y)
        return;

    const int base_x         = base_size.x == 1 ? 0 : src_x;
    const int base_y         = base_size.y == 1 ? 0 : src_y;
    const int base_batch_idx = base_size.z == 1 ? 0 : batch_idx;

    const int scale_x         = scale_size.x == 1 ? 0 : src_x;
    const int scale_y         = scale_size.y == 1 ? 0 : src_y;
    const int scale_batch_idx = scale_size.z == 1 ? 0 : batch_idx;

    using input_value_type = typename input_type::ValueType;

    *dst.ptr(batch_idx, src_y, src_x) = cuda::SaturateCast<input_value_type>(
        (*src.ptr(batch_idx, src_y, src_x) - *base.ptr(base_batch_idx, base_y, base_x))
            * (*scale.ptr(scale_batch_idx, scale_y, scale_x)) * global_scale
        + global_shift);
}

template<typename input_type, typename base_type, typename scale_type>
__global__ void normalizeInvStdDevKernel(const input_type src, const base_type base, const scale_type scale,
                                         input_type dst, int2 inout_size, int3 base_size, int3 scale_size,
                                         float global_scale, float global_shift, float epsilon)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    if (src_x >= inout_size.x || src_y >= inout_size.y)
        return;

    const int base_x         = base_size.x == 1 ? 0 : src_x;
    const int base_y         = base_size.y == 1 ? 0 : src_y;
    const int base_batch_idx = base_size.z == 1 ? 0 : batch_idx;

    const int scale_x         = scale_size.x == 1 ? 0 : src_x;
    const int scale_y         = scale_size.y == 1 ? 0 : src_y;
    const int scale_batch_idx = scale_size.z == 1 ? 0 : batch_idx;

    using input_value_type = typename input_type::ValueType;
    using scale_value_type = typename scale_type::ValueType;

    scale_value_type mul = InvStdDevMul(*scale.ptr(scale_batch_idx, scale_y, scale_x), epsilon);

    *dst.ptr(batch_idx, src_y, src_x) = cuda::SaturateCast<input_value_type>(
        (*src.ptr(batch_idx, src_y, src_x) - *base.ptr(base_batch_idx, base_y, base_x)) * mul * global_scale
        + global_shift);
}

// The scalar kernel recomputes 1 / sqrt(scale^2 + eps) per pixel; for multi-channel interleaved input
// that is several sqrt + reciprocal per pixel, all redundant when scale is spatially broadcast, which
// makes those paths issue-bound (~83% issue-slot utilization on Ampere) rather than memory-bound.
template<int NIX, typename input_type, typename base_type, typename scale_type>
__global__ void normalizeInvStdDevHoistKernel(const input_type src, const base_type base, const scale_type scale,
                                              input_type dst, int2 inout_size, int3 base_size, int3 scale_size,
                                              float global_scale, float global_shift, float epsilon)
{
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    if (src_y >= inout_size.y)
        return;

    const int base_batch_idx  = base_size.z == 1 ? 0 : batch_idx;
    const int scale_batch_idx = scale_size.z == 1 ? 0 : batch_idx;

    using input_value_type = typename input_type::ValueType;
    using base_value_type  = typename base_type::ValueType;
    using scale_value_type = typename scale_type::ValueType;

    const base_value_type  b   = *base.ptr(base_batch_idx, 0, 0);
    const scale_value_type mul = InvStdDevMul(*scale.ptr(scale_batch_idx, 0, 0), epsilon);

    const int x0 = blockIdx.x * blockDim.x * NIX + threadIdx.x;
#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int src_x = x0 + i * blockDim.x;
        if (src_x < inout_size.x)
        {
            *dst.ptr(batch_idx, src_y, src_x) = cuda::SaturateCast<input_value_type>(
                (*src.ptr(batch_idx, src_y, src_x) - b) * mul * global_scale + global_shift);
        }
    }
}

// Requires sizeof(Vec4)-aligned row/sample strides, caller-guarded.
template<int NGROUP, typename T, typename SrcWrapper, typename BaseWrapper, typename ScaleWrapper, typename DstWrapper>
__global__ void normalizeInvStdDevVec4Kernel(SrcWrapper src, BaseWrapper base, ScaleWrapper scale, DstWrapper dst,
                                             int2 inout_size, int3 base_size, int3 scale_size, float global_scale,
                                             float global_shift, float epsilon)
{
    using Vec4 = cuda::MakeType<T, 4>;

    const int base_cx   = (blockIdx.x * blockDim.x + threadIdx.x) * (4 * NGROUP);
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    if (base_cx >= inout_size.x || src_y >= inout_size.y)
        return;

    const auto ctx = MakeNormInvStdDevCtx(base, scale, base_size, scale_size, src_y, batch_idx, global_scale,
                                          global_shift, epsilon);

    if (base_cx + 4 * NGROUP <= inout_size.x)
    {
        Vec4 v[NGROUP];
#pragma unroll
        for (int g = 0; g < NGROUP; ++g)
        {
            v[g] = *reinterpret_cast<const Vec4 *>(src.ptr(batch_idx, src_y, base_cx + g * 4));
        }

        Vec4 o[NGROUP];
        if (base_size.x == 1 && scale_size.x == 1)
        {
            float baseV, mul;
            ReadBroadcastBaseAndMul(ctx, baseV, mul);
#pragma unroll
            for (int g = 0; g < NGROUP; ++g)
            {
                o[g].x = ApplyNormalize<T>(v[g].x, baseV, mul, global_scale, global_shift);
                o[g].y = ApplyNormalize<T>(v[g].y, baseV, mul, global_scale, global_shift);
                o[g].z = ApplyNormalize<T>(v[g].z, baseV, mul, global_scale, global_shift);
                o[g].w = ApplyNormalize<T>(v[g].w, baseV, mul, global_scale, global_shift);
            }
        }
        else
        {
#pragma unroll
            for (int g = 0; g < NGROUP; ++g)
            {
                const int cx = base_cx + g * 4;
                o[g].x       = NormInvStdDevOne<T>(v[g].x, cx + 0, ctx);
                o[g].y       = NormInvStdDevOne<T>(v[g].y, cx + 1, ctx);
                o[g].z       = NormInvStdDevOne<T>(v[g].z, cx + 2, ctx);
                o[g].w       = NormInvStdDevOne<T>(v[g].w, cx + 3, ctx);
            }
        }

#pragma unroll
        for (int g = 0; g < NGROUP; ++g)
        {
            *reinterpret_cast<Vec4 *>(dst.ptr(batch_idx, src_y, base_cx + g * 4)) = o[g];
        }
    }
    else
    {
        for (int x = base_cx; x < inout_size.x; ++x)
        {
            *dst.ptr(batch_idx, src_y, x) = NormInvStdDevOne<T>(*src.ptr(batch_idx, src_y, x), x, ctx);
        }
    }
}

// Requires 16-byte-aligned row/sample strides, caller-guarded.
template<int NGROUP, typename SrcWrapper, typename BaseWrapper, typename ScaleWrapper, typename DstWrapper>
__global__ void normalizeInvStdDevU8VecILPKernel(SrcWrapper src, BaseWrapper base, ScaleWrapper scale, DstWrapper dst,
                                                 int2 inout_size, int3 base_size, int3 scale_size, float global_scale,
                                                 float global_shift, float epsilon)
{
    const int base_cx   = (blockIdx.x * blockDim.x + threadIdx.x) * (16 * NGROUP);
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    if (base_cx >= inout_size.x || src_y >= inout_size.y)
        return;

    const auto ctx = MakeNormInvStdDevCtx(base, scale, base_size, scale_size, src_y, batch_idx, global_scale,
                                          global_shift, epsilon);

    if (base_cx + 16 * NGROUP <= inout_size.x)
    {
        uint4 v[NGROUP];
#pragma unroll
        for (int g = 0; g < NGROUP; ++g)
        {
            v[g] = *reinterpret_cast<const uint4 *>(src.ptr(batch_idx, src_y, base_cx + g * 16));
        }

        uchar o[NGROUP][16];
        if (base_size.x == 1 && scale_size.x == 1)
        {
            float baseV, mul;
            ReadBroadcastBaseAndMul(ctx, baseV, mul);
#pragma unroll
            for (int g = 0; g < NGROUP; ++g)
            {
                const uchar *b = reinterpret_cast<const uchar *>(&v[g]);
#pragma unroll
                for (int i = 0; i < 16; ++i)
                {
                    o[g][i] = ApplyNormalize<uchar>(b[i], baseV, mul, global_scale, global_shift);
                }
            }
        }
        else
        {
#pragma unroll
            for (int g = 0; g < NGROUP; ++g)
            {
                const uchar *b = reinterpret_cast<const uchar *>(&v[g]);
#pragma unroll
                for (int i = 0; i < 16; ++i)
                {
                    o[g][i] = NormInvStdDevOne<uchar>(b[i], base_cx + g * 16 + i, ctx);
                }
            }
        }

#pragma unroll
        for (int g = 0; g < NGROUP; ++g)
        {
            *reinterpret_cast<uint4 *>(dst.ptr(batch_idx, src_y, base_cx + g * 16))
                = *reinterpret_cast<const uint4 *>(o[g]);
        }
    }
    else
    {
        for (int x = base_cx; x < inout_size.x; ++x)
        {
            *dst.ptr(batch_idx, src_y, x) = NormInvStdDevOne<uchar>(*src.ptr(batch_idx, src_y, x), x, ctx);
        }
    }
}

// The base/scale extents the interleaved kernels broadcast against, as (W, H, N).
inline int3 ParamExtents(const Access &access)
{
    return int3{access.numCols(), access.numRows(), static_cast<int>(access.numSamples())};
}

template<typename WrapInput, typename WrapOutput, typename BaseWrap, typename ScaleWrap>
void LaunchInterleaved(WrapInput srcWrap, WrapOutput dstWrap, BaseWrap baseWrap, ScaleWrap scaleWrap,
                       DataShape input_shape, int3 base_size, int3 scale_size, const NormalizeParams &p)
{
    dim3 block(32, 8);
    int2 inout_size = {input_shape.W, input_shape.H};

    if (p.isStdDev)
    {
        // Multi-byte elements are already memory-bound, so recomputing the multiplier per pixel costs
        // nothing there. By-value parameters are spatially broadcast by construction.
        using pixel_type = typename WrapInput::ValueType;
        if constexpr (sizeof(cuda::BaseType<pixel_type>) == 1)
        {
            const bool spatialBroadcast
                = base_size.x == 1 && base_size.y == 1 && scale_size.x == 1 && scale_size.y == 1;
            if (spatialBroadcast)
            {
                constexpr int NIX   = kNormalizeHoistNIX;
                dim3          hgrid = NormGrid(input_shape.W, input_shape.H, input_shape.N, block, NIX);
                normalizeInvStdDevHoistKernel<NIX><<<hgrid, block, 0, p.stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap,
                                                                                  inout_size, base_size, scale_size,
                                                                                  p.globalScale, p.shift, p.epsilon);
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        dim3 grid = NormGrid(input_shape.W, input_shape.H, input_shape.N, block);
        normalizeInvStdDevKernel<<<grid, block, 0, p.stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size,
                                                               base_size, scale_size, p.globalScale, p.shift,
                                                               p.epsilon);
    }
    else
    {
        dim3 grid = NormGrid(input_shape.W, input_shape.H, input_shape.N, block);
        normalizeKernel<<<grid, block, 0, p.stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size,
                                                      scale_size, p.globalScale, p.shift);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

// base/scale are read as the input's work type when they carry one value per channel, and as a
// broadcast scalar float otherwise.
template<typename WorkT, class F>
void DispatchParamKinds(bool basePerChannel, bool scalePerChannel, F &&f)
{
    if (basePerChannel && scalePerChannel)
        f(WorkT{}, WorkT{});
    else if (basePerChannel)
        f(WorkT{}, float{});
    else if (scalePerChannel)
        f(float{}, WorkT{});
    else
        f(float{}, float{});
}

template<typename input_type>
void normalize(const TensorArgs &a)
{
    RequireStridesFitInt32({&a.inAccess, &a.outAccess}, "Input or output");

    auto srcWrap = cuda::CreateTensorWrapNHW<input_type, int32_t>(a.in);
    auto dstWrap = cuda::CreateTensorWrapNHW<input_type, int32_t>(a.out);

    const DataShape shape      = GetDataShape(a.inAccess);
    const int3      base_size  = ParamExtents(a.baseAccess);
    const int3      scale_size = ParamExtents(a.scaleAccess);

    using work_type = cuda::ConvertBaseTypeTo<float, input_type>;

    DispatchParamKinds<work_type>(a.baseAccess.numChannels() != 1, a.scaleAccess.numChannels() != 1,
                                  [&](auto baseVal, auto scaleVal)
                                  {
                                      LaunchInterleaved(srcWrap, dstWrap,
                                                        cuda::CreateTensorWrapNHW<decltype(baseVal), int32_t>(a.base),
                                                        cuda::CreateTensorWrapNHW<decltype(scaleVal), int32_t>(a.scale),
                                                        shape, base_size, scale_size, a.p);
                                  });
}

// VEC is the columns a single transfer covers: 16 for the uint4 kernel, 4 for the 4-wide ones.
template<typename T, int NGROUP, int VEC>
void normalizeInvStdDevVec(const TensorArgs &a)
{
    RequireStridesFitInt32({&a.inAccess, &a.outAccess}, "Input or output");

    const DataShape input_shape = GetDataShape(a.inAccess);

    auto srcWrap   = cuda::CreateTensorWrapNHW<T, int32_t>(a.in);
    auto dstWrap   = cuda::CreateTensorWrapNHW<T, int32_t>(a.out);
    auto baseWrap  = cuda::CreateTensorWrapNHW<float, int32_t>(a.base);
    auto scaleWrap = cuda::CreateTensorWrapNHW<float, int32_t>(a.scale);

    int2 inout_size = {input_shape.W, input_shape.H};
    int3 base_size  = ParamExtents(a.baseAccess);
    int3 scale_size = ParamExtents(a.scaleAccess);

    dim3 block(32, 8);
    dim3 grid = NormGrid(input_shape.W, input_shape.H, input_shape.N, block, VEC * NGROUP);

    if constexpr (VEC == 16)
    {
        normalizeInvStdDevU8VecILPKernel<NGROUP>
            <<<grid, block, 0, a.p.stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size, scale_size,
                                             a.p.globalScale, a.p.shift, a.p.epsilon);
    }
    else
    {
        normalizeInvStdDevVec4Kernel<NGROUP, T>
            <<<grid, block, 0, a.p.stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size, scale_size,
                                             a.p.globalScale, a.p.shift, a.p.epsilon);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------
// Tensor path: interleaved element stream.
//
// The pixel-shaped kernels above access one pixel per thread, so their transaction width is the pixel
// width: an RGB8 pixel is three 1-byte accesses and an RGBf32 pixel three 4-byte ones. The stream
// moves 16 bytes per access regardless, which for a three-channel image means a carrier is not a whole
// number of pixels -- 16 bytes is 5 1/3 RGB8 pixels -- so the carrier has to know which channel its
// first element belongs to. That is what the shared header's Position and RotateLanes are for: the
// per-channel (base, multiplier) set is resolved once per thread and then rotated by the carrier's
// phase, leaving every element's channel index a compile-time constant inside the unrolled group.
//
// The per-element expression is ApplyNormalize on the same operands with the same association as
// normalizeKernel and normalizeInvStdDevKernel, so the output is bit-identical. Applies only when
// base and scale do not vary spatially, which is what lets them be resolved once per thread; the
// pixel-shaped kernels keep every other case.
// ---------------------------------------------------------------------------------------------------

// A 1-byte carrier holds 16 elements, each needing an integer/float conversion pair; a 4-byte one
// holds 4 needing none, so the narrower element type saturates its memory-level parallelism sooner.
template<typename T>
inline constexpr int kNormalizeStreamGroups = sizeof(T) == 1 ? 4 : 8;

template<typename T, int C>
struct NormalizeStreamCarrier
{
    static constexpr bool kPositionAware = true;
    static constexpr int  kElems         = elem_stream::kCarrierBytes / static_cast<int>(sizeof(T));

    using type = uint4;

    // Pointers rather than values: a per-channel parameter lives in device memory. `stride` is 0 when
    // one value serves every channel, which is how a scalar base or scale broadcasts.
    struct Param
    {
        const float *base;
        const float *scale;
        int          baseStride;
        int          scaleStride;
        int64_t      baseSampleStride;
        int64_t      scaleSampleStride;
        float        globalScale, shift, epsilon;
        bool         isStdDev;
    };

    struct Context
    {
        float base[C];
        float mul[C];
        float globalScale, shift;
    };

    inline static __device__ Context MakeContext(const Param &p, elem_stream::Position pos)
    {
        Context c;
#pragma unroll
        for (int k = 0; k < C; ++k)
        {
            const float s = p.scale[pos.sample * p.scaleSampleStride + k * p.scaleStride];
            c.base[k]     = p.base[pos.sample * p.baseSampleStride + k * p.baseStride];
            c.mul[k]      = p.isStdDev ? InvStdDevMul(s, p.epsilon) : s;
        }
        c.globalScale = p.globalScale;
        c.shift       = p.shift;
        return c;
    }

    union Group
    {
        type wide;
        T    elem[kElems];
    };

    inline static __device__ type Apply(type v, const Context &c, elem_stream::Position pos)
    {
        const int phase = elem_stream::ChannelPhase<kElems, C>(pos.carrier);

        float base[C], mul[C];
        elem_stream::RotateLanes<C>(c.base, phase, base);
        elem_stream::RotateLanes<C>(c.mul, phase, mul);

        Group in, out;
        in.wide = v;
#pragma unroll
        for (int e = 0; e < kElems; ++e)
        {
            out.elem[e] = ApplyNormalize<T>(in.elem[e], base[e % C], mul[e % C], c.globalScale, c.shift);
        }
        return out.wide;
    }

    // At most one carrier per run is partial, so this stays a runtime loop indexed by the unrotated
    // set rather than replicating the group body for it.
    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, const Context &c,
                                       elem_stream::Position pos)
    {
        const int phase = elem_stream::ChannelPhase<kElems, C>(pos.carrier);
        const T  *se    = reinterpret_cast<const T *>(s);
        T        *de    = reinterpret_cast<T *>(d);
        for (int e = 0; e < bytes / static_cast<int>(sizeof(T)); ++e)
        {
            const int k = (phase + e) % C;
            de[e]       = ApplyNormalize<T>(se[e], c.base[k], c.mul[k], c.globalScale, c.shift);
        }
    }
};

template<typename T, int C>
bool TryNormalizeStreamAs(const TensorArgs &a, int baseStride, int scaleStride)
{
    // The stream only pays where a pixel does not already fill a native vector access. Measured on an
    // A100 against the pixel-shaped kernels: 1.36x for RGB8 and 1.13x for RGBA8, but 0.99x for both
    // RGBf32 and RGBAf32, where a 4-byte element already coalesces and the carrier's channel rotation
    // is pure overhead. Same boundary, and same reason, as the CvtColor stream's element-width limit.
    if constexpr (sizeof(T) > 2)
    {
        return false;
    }
    else
    {
        using Carrier = NormalizeStreamCarrier<T, C>;

        typename Carrier::Param param{
            reinterpret_cast<const float *>(a.base.basePtr()),
            reinterpret_cast<const float *>(a.scale.basePtr()),
            baseStride,
            scaleStride,
            a.baseAccess.numSamples() == 1 ? 0 : a.baseAccess.sampleStride() / static_cast<int64_t>(sizeof(float)),
            a.scaleAccess.numSamples() == 1 ? 0 : a.scaleAccess.sampleStride() / static_cast<int64_t>(sizeof(float)),
            a.p.globalScale,
            a.p.shift,
            a.p.epsilon,
            a.p.isStdDev};

        constexpr int64_t pixelBytes = C * sizeof(T);
        elem_stream::Plan plan;
        if (!elem_stream::MakePlanBytes(a.inAccess, a.outAccess, pixelBytes, pixelBytes, false, plan))
        {
            return false;
        }
        if (param.baseSampleStride || param.scaleSampleStride)
        {
            const int64_t samples = a.inAccess.numSamples();
            if (samples > elem_stream::kMaxGridYZ)
            {
                return false;
            }
            // Each sample needs its own parameter context, even when image storage is dense.
            if (plan.numSamples == 1 && samples > 1)
            {
                plan.runBytes /= samples;
                plan.dstRunBytes /= samples;
                plan.numSamples      = static_cast<int>(samples);
                plan.srcSampleStride = a.inAccess.sampleStride();
                plan.dstSampleStride = a.outAccess.sampleStride();
            }
        }
        if (!elem_stream::IsPlanAligned(a.in.basePtr(), a.out.basePtr(), plan, alignof(uint4), alignof(uint4)))
        {
            return false;
        }
        elem_stream::LaunchPlan<kNormalizeStreamGroups<T>, Carrier>(a.p.stream, a.in.basePtr(), a.out.basePtr(), plan,
                                                                    param);
        return true;
    }
}

// ---------------------------------------------------------------------------------------------------
// Tensor-free (by-value scalar / per-channel parameter) paths.
//
// base/scale are host constants passed by value, so the caller avoids allocating and uploading
// parameter tensors. The wraps below adapt a register-resident value to the .ptr()/ValueType
// interface the kernels already consume, so with param extents of 1 the output is bit-identical to
// the tensor path fed the same values. The single-channel U8/F32 vectorized ILP kernels are selected
// only by the parameter-tensor entry point and are not wired in here.
// ---------------------------------------------------------------------------------------------------

template<typename T>
struct ConstantValueWrap
{
    using ValueType = T;

    T value;

    inline const __host__ __device__ T *ptr(int, int, int) const
    {
        return &value;
    }
};

template<typename VecT>
inline VecT MakeWorkVec(float4 v)
{
    if constexpr (cuda::IsCompound<VecT>)
    {
        return cuda::DropCast<cuda::NumElements<VecT>>(v);
    }
    else
    {
        return v.x;
    }
}

template<typename input_type>
void normalizeScalar(const ScalarArgs &a)
{
    RequireStridesFitInt32({&a.inAccess, &a.outAccess}, "Input or output");

    auto srcWrap = cuda::CreateTensorWrapNHW<input_type, int32_t>(a.in);
    auto dstWrap = cuda::CreateTensorWrapNHW<input_type, int32_t>(a.out);

    using work_type = cuda::ConvertBaseTypeTo<float, input_type>;

    const DataShape shape = GetDataShape(a.inAccess);
    const int3      ones  = {1, 1, 1};

    DispatchParamKinds<work_type>(
        a.baseCount != 1, a.scaleCount != 1,
        [&](auto baseVal, auto scaleVal)
        {
            LaunchInterleaved(srcWrap, dstWrap,
                              ConstantValueWrap<decltype(baseVal)>{MakeWorkVec<decltype(baseVal)>(a.base)},
                              ConstantValueWrap<decltype(scaleVal)>{MakeWorkVec<decltype(scaleVal)>(a.scale)}, shape,
                              ones, ones, a.p);
        });
}

// The planar kernels read one scalar base/scale per (batch, channel), where the channel is a grid
// dimension -- unlike the interleaved kernels, whose base/scale vector lanes ARE the channels -- so
// this wrap must be channel-indexed rather than a single broadcast value.
struct PlanarConstParamWrap
{
    using ValueType = float;

    float data[4];

    inline const __host__ __device__ float *ptr(int, int channel, int, int) const
    {
        return &data[channel];
    }
};

template<typename input_type>
void normalizeScalarPlanar(const ScalarArgs &a)
{
    RequireStridesFitInt32({&a.inAccess, &a.outAccess}, "Input or output");

    auto srcWrap = cuda::CreateTensorWrapNCHW<input_type, int32_t>(a.in);
    auto dstWrap = cuda::CreateTensorWrapNCHW<input_type, int32_t>(a.out);

    PlanarConstParamWrap baseWrap{
        {a.base.x, a.base.y, a.base.z, a.base.w}
    };
    PlanarConstParamWrap scaleWrap{
        {a.scale.x, a.scale.y, a.scale.z, a.scale.w}
    };

    LaunchPlanar<input_type>(srcWrap, dstWrap, baseWrap, scaleWrap, a.inAccess, a.outAccess, int4{1, a.baseCount, 1, 1},
                             int4{1, a.scaleCount, 1, 1}, a.p);
}

// ---------------------------------------------------------------------------------------------------
// ImageBatchVarShape path.
// ---------------------------------------------------------------------------------------------------

template<typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normKernel(const cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<out_T> dst,
                           const scale_type *scale, const base_type *base, float global_scale, float global_shift)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    T out                             = *src.ptr(batch_idx, dst_y, dst_x);
    *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<out_T>((out - *base) * *scale * global_scale + global_shift);
}

template<typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normInvStdDevKernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                    cuda::ImageBatchVarShapeWrap<out_T> dst, const scale_type *scale,
                                    const base_type *base, float global_scale, float global_shift, float epsilon)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    scale_type mul = InvStdDevMul(*scale, epsilon);

    T out                             = *src.ptr(batch_idx, dst_y, dst_x);
    *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<out_T>((out - *base) * mul * global_scale + global_shift);
}

// Relies on NVCV's image row-pitch alignment (>= sizeof(Vec4)).
template<int NGROUP, typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normInvStdDevVec4Kernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                        cuda::ImageBatchVarShapeWrap<out_T> dst, const scale_type *scale,
                                        const base_type *base, float global_scale, float global_shift, float epsilon)
{
    // 1-byte samples move as unsigned bytes whatever their signedness; F32 moves as float4.
    using VecBase = std::conditional_t<sizeof(T) == 1, uchar, float>;
    using Vec4    = cuda::MakeType<VecBase, 4>;

    const int g0        = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int width = dst.width(batch_idx);
    const int cx0   = g0 * 4;
    if (cx0 >= width || dst_y >= dst.height(batch_idx))
        return;

    const scale_type mul = InvStdDevMul(*scale, epsilon);
    const base_type  b   = *base;

    auto apply = [&](VecBase raw) -> out_T
    {
        return cuda::SaturateCast<out_T>((raw - b) * mul * global_scale + global_shift);
    };

    int  cx[NGROUP];
    bool full[NGROUP];
    Vec4 in4[NGROUP];
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        cx[i]   = (g0 + i * blockDim.x) * 4;
        full[i] = cx[i] + 4 <= width;
        if (full[i])
            in4[i] = *reinterpret_cast<const Vec4 *>(src.ptr(batch_idx, dst_y, cx[i]));
    }

#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        if (full[i])
        {
            Vec4 out4;
            out4.x                                                      = apply(in4[i].x);
            out4.y                                                      = apply(in4[i].y);
            out4.z                                                      = apply(in4[i].z);
            out4.w                                                      = apply(in4[i].w);
            *reinterpret_cast<Vec4 *>(dst.ptr(batch_idx, dst_y, cx[i])) = out4;
        }
        else if (cx[i] < width)
        {
            for (int x = cx[i]; x < width; ++x)
            {
                *dst.ptr(batch_idx, dst_y, x) = apply(*src.ptr(batch_idx, dst_y, x));
            }
        }
    }
}

template<int NIX, typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normInvStdDevHoistKernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                         cuda::ImageBatchVarShapeWrap<out_T> dst, const scale_type *scale,
                                         const base_type *base, float global_scale, float global_shift, float epsilon)
{
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int width = dst.width(batch_idx);
    if (dst_y >= dst.height(batch_idx))
        return;

    const base_type  b   = *base;
    const scale_type mul = InvStdDevMul(*scale, epsilon);

    const int x0 = blockIdx.x * blockDim.x * NIX + threadIdx.x;
#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int dst_x = x0 + i * blockDim.x;
        if (dst_x < width)
        {
            *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<out_T>(
                (*src.ptr(batch_idx, dst_y, dst_x) - b) * mul * global_scale + global_shift);
        }
    }
}

template<typename T, typename out_T>
__global__ void normPlanarKernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                 cuda::ImageBatchVarShapeWrap<out_T> dst, const cuda::Tensor4DWrap<float, int32_t> base,
                                 const cuda::Tensor4DWrap<float, int32_t> scale, int num_channels, int4 base_size,
                                 int4 scale_size, float global_scale, float global_shift, float epsilon, bool is_stddev)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc        = blockIdx.z;
    const int batch_idx = nc / num_channels;
    const int channel   = nc % num_channels;

    if (dst_x >= dst.width(batch_idx, channel) || dst_y >= dst.height(batch_idx, channel))
        return;

    const int4 b = PlanarBroadcastIndex(base_size, batch_idx, channel, dst_y, dst_x);
    const int4 s = PlanarBroadcastIndex(scale_size, batch_idx, channel, dst_y, dst_x);

    *dst.ptr(batch_idx, channel, dst_y, dst_x)
        = ApplyPlanarNormalize<out_T>(*src.ptr(batch_idx, channel, dst_y, dst_x), *base.ptr(b.x, b.y, b.z, b.w),
                                      *scale.ptr(s.x, s.y, s.z, s.w), global_scale, global_shift, is_stddev, epsilon);
}

template<typename T, typename out_T>
__global__ void normPlanarVec4Kernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                     cuda::ImageBatchVarShapeWrap<out_T>         dst,
                                     const cuda::Tensor4DWrap<float, int32_t>    base,
                                     const cuda::Tensor4DWrap<float, int32_t> scale, int num_channels, int4 base_size,
                                     int4 scale_size, float global_scale, float global_shift, float epsilon,
                                     bool is_stddev)
{
    const int cx        = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc        = blockIdx.z;
    const int batch_idx = nc / num_channels;
    const int channel   = nc % num_channels;

    const int width = dst.width(batch_idx, channel);
    if (cx >= width || dst_y >= dst.height(batch_idx, channel))
        return;

    NormalizePlanarVec4<T>(src, dst, base, scale, batch_idx, channel, dst_y, cx, width, base_size, scale_size,
                           global_scale, global_shift, is_stddev, epsilon);
}

// normPlanarVec4Kernel moves only 4 bytes per thread, so its per-thread cost -- above all the
// ImageBatchVarShapeWrap::ptr metadata lookup -- is amortized over 3x less data than the interleaved
// path, leaving it co-limited by compute (~76% memory and ~76% compute SOL). Resolving base/scale and
// the plane row pointer once per thread and striding across NGROUP groups fixes that.
//
// Preserved from the pre-migration kernel: the row is read through uchar4 whatever T's signedness is,
// so a signed 1-byte plane is widened as unsigned here.
template<int NGROUP, typename T, typename out_T>
__global__ void normPlanarVec4HoistKernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                          cuda::ImageBatchVarShapeWrap<out_T>         dst,
                                          const cuda::Tensor4DWrap<float, int32_t>    base,
                                          const cuda::Tensor4DWrap<float, int32_t> scale, int num_channels,
                                          int4 base_size, int4 scale_size, float global_scale, float global_shift,
                                          float epsilon, bool is_stddev)
{
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc        = blockIdx.z;
    const int batch_idx = nc / num_channels;
    const int channel   = nc % num_channels;

    const int width = dst.width(batch_idx, channel);
    if (dst_y >= dst.height(batch_idx, channel))
        return;

    // base/scale are broadcast across H and W here (caller-guarded).
    const int4  b      = PlanarBroadcastIndex(base_size, batch_idx, channel, 0, 0);
    const int4  s      = PlanarBroadcastIndex(scale_size, batch_idx, channel, 0, 0);
    const float baseV  = *base.ptr(b.x, b.y, b.z, b.w);
    const float scaleV = *scale.ptr(s.x, s.y, s.z, s.w);
    const float mul    = is_stddev ? InvStdDevMul(scaleV, epsilon) : scaleV;

    const T *const rowSrc = src.ptr(batch_idx, channel, dst_y, 0);
    out_T *const   rowDst = dst.ptr(batch_idx, channel, dst_y, 0);

    const int g0 = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        const int cx = (g0 + i * blockDim.x) * 4;
        if (cx + 4 <= width)
        {
            const uchar4 in4 = *reinterpret_cast<const uchar4 *>(rowSrc + cx);
            uchar4       out4;
            out4.x = ApplyNormalize<out_T>(in4.x, baseV, mul, global_scale, global_shift);
            out4.y = ApplyNormalize<out_T>(in4.y, baseV, mul, global_scale, global_shift);
            out4.z = ApplyNormalize<out_T>(in4.z, baseV, mul, global_scale, global_shift);
            out4.w = ApplyNormalize<out_T>(in4.w, baseV, mul, global_scale, global_shift);
            *reinterpret_cast<uchar4 *>(rowDst + cx) = out4;
        }
        else if (cx < width)
        {
            for (int x = cx; x < width; ++x)
            {
                rowDst[x] = ApplyNormalize<out_T>(rowSrc[x], baseV, mul, global_scale, global_shift);
            }
        }
    }
}

template<typename T, typename out_T, typename base_type, typename scale_type>
void LaunchInterleavedVarShape(const VarShapeArgs &a, const base_type *base, const scale_type *scale)
{
    const int max_width  = a.in.maxSize().w;
    const int max_height = a.in.maxSize().h;
    const int batch      = a.in.numImages();

    dim3 block(32, 8, 1);

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(a.in);
    cuda::ImageBatchVarShapeWrap<out_T>   dst_ptr(a.out);

    if (a.p.isStdDev)
    {
        // The element size implies one channel here, so base/scale are scalar.
        constexpr bool kIsByte = sizeof(T) == 1 && sizeof(out_T) == 1;
        constexpr bool kIsF32  = std::is_same_v<T, float> && std::is_same_v<out_T, float>;

        if constexpr (kIsByte || kIsF32)
        {
            constexpr int NGROUP = kIsByte ? kNormVarShapeILPNGroup : kNormVarShapeF32ILPNGroup;
            dim3          vgrid  = NormGrid(max_width, max_height, batch, block, 4 * NGROUP);
            normInvStdDevVec4Kernel<NGROUP, T, out_T><<<vgrid, block, 0, a.p.stream>>>(
                src_ptr, dst_ptr, scale, base, a.p.globalScale, a.p.shift, a.p.epsilon);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }

        // Multi-channel 1-byte interleaved input is issue-bound on the redundant per-pixel sqrt;
        // float3 keeps the scalar kernel since it is already memory-bound.
        if constexpr (sizeof(cuda::BaseType<T>) == 1)
        {
            constexpr int NIX   = kNormalizeHoistNIX;
            dim3          hgrid = NormGrid(max_width, max_height, batch, block, NIX);
            normInvStdDevHoistKernel<NIX, T, out_T><<<hgrid, block, 0, a.p.stream>>>(
                src_ptr, dst_ptr, scale, base, a.p.globalScale, a.p.shift, a.p.epsilon);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }

        dim3 grid = NormGrid(max_width, max_height, batch, block);
        normInvStdDevKernel<T, out_T>
            <<<grid, block, 0, a.p.stream>>>(src_ptr, dst_ptr, scale, base, a.p.globalScale, a.p.shift, a.p.epsilon);
    }
    else
    {
        dim3 grid = NormGrid(max_width, max_height, batch, block);
        normKernel<T, out_T><<<grid, block, 0, a.p.stream>>>(src_ptr, dst_ptr, scale, base, a.p.globalScale, a.p.shift);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, typename out_T>
void norm(const VarShapeArgs &a)
{
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const void *basePtr  = a.baseAccess.sampleData(0);
    const void *scalePtr = a.scaleAccess.sampleData(0);

    DispatchParamKinds<work_type>(a.baseAccess.numChannels() != 1, a.scaleAccess.numChannels() != 1,
                                  [&](auto baseVal, auto scaleVal)
                                  {
                                      LaunchInterleavedVarShape<T, out_T>(
                                          a, reinterpret_cast<const decltype(baseVal) *>(basePtr),
                                          reinterpret_cast<const decltype(scaleVal) *>(scalePtr));
                                  });
}

template<typename T, typename out_T>
void normPlanar(const VarShapeArgs &a)
{
    RequireStridesFitInt32({&a.baseAccess, &a.scaleAccess}, "Base or scale");

    const auto baseWrap   = cuda::CreateTensorWrapNCHW<float, int32_t>(a.base);
    const auto scaleWrap  = cuda::CreateTensorWrapNCHW<float, int32_t>(a.scale);
    const int4 base_size  = PlanarExtents(a.baseAccess);
    const int4 scale_size = PlanarExtents(a.scaleAccess);

    const int          max_width  = a.in.maxSize().w;
    const int          max_height = a.in.maxSize().h;
    const unsigned int gridZ      = PlanarGridZ(a.in.numImages(), a.channels);

    dim3 block(32, 8, 1);

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(a.in);
    cuda::ImageBatchVarShapeWrap<out_T>   dst_ptr(a.out);

    constexpr bool kIsByte = sizeof(T) == 1 && sizeof(out_T) == 1;
    constexpr bool kIsF32  = std::is_same_v<T, float> && std::is_same_v<out_T, float>;
    constexpr bool kIsF16  = std::is_same_v<T, __half> && std::is_same_v<out_T, __half>;

    if constexpr (kIsByte)
    {
        const bool spatialBroadcast = base_size.z == 1 && base_size.w == 1 && scale_size.z == 1 && scale_size.w == 1;
        if (spatialBroadcast)
        {
            constexpr int NGROUP = kNormVarShapePlanarILPNGroup;
            dim3          hgrid  = NormGrid(max_width, max_height, gridZ, block, 4 * NGROUP);
            normPlanarVec4HoistKernel<NGROUP, T, out_T>
                <<<hgrid, block, 0, a.p.stream>>>(src_ptr, dst_ptr, baseWrap, scaleWrap, a.channels, base_size,
                                                  scale_size, a.p.globalScale, a.p.shift, a.p.epsilon, a.p.isStdDev);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }
    }

    // Plain vectorization only -- loads-first ILP regressed the 1-byte planar var-shape path.
    if constexpr (kIsByte || kIsF32 || kIsF16)
    {
        dim3 vgrid = NormGrid(max_width, max_height, gridZ, block, 4);
        normPlanarVec4Kernel<T, out_T>
            <<<vgrid, block, 0, a.p.stream>>>(src_ptr, dst_ptr, baseWrap, scaleWrap, a.channels, base_size, scale_size,
                                              a.p.globalScale, a.p.shift, a.p.epsilon, a.p.isStdDev);
        NVCV_CHECK_THROW(cudaGetLastError());
        return;
    }

    dim3 grid = NormGrid(max_width, max_height, gridZ, block);
    normPlanarKernel<T, out_T><<<grid, block, 0, a.p.stream>>>(src_ptr, dst_ptr, baseWrap, scaleWrap, a.channels,
                                                               base_size, scale_size, a.p.globalScale, a.p.shift,
                                                               a.p.epsilon, a.p.isStdDev);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------
// Dispatch: element type (and, on the interleaved paths, channel count) -> kernel instantiation.
// ---------------------------------------------------------------------------------------------------

template<int N>
using IntTag = std::integral_constant<int, N>;

template<class F>
void DispatchElemType(DataTypeCode type, F &&f)
{
    switch (type)
    {
    case DataTypeCode::kU8:
        f(uchar{});
        return;
    case DataTypeCode::kS8:
        f(schar{});
        return;
    case DataTypeCode::kU16:
        f(ushort{});
        return;
    case DataTypeCode::kS16:
        f(short{});
        return;
    case DataTypeCode::kS32:
        f(int{});
        return;
    case DataTypeCode::kF32:
        f(float{});
        return;
    case DataTypeCode::kF16:
        f(__half{});
        return;
    case DataTypeCode::kF64:
        f(double{});
        return;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
}

template<class F>
void DispatchVarShapeOutType(DataTypeCode type, F &&f)
{
    switch (type)
    {
    case DataTypeCode::kU8:
        f(uchar{});
        return;
    case DataTypeCode::kF32:
        f(float{});
        return;
    case DataTypeCode::kF16:
        f(__half{});
        return;
    case DataTypeCode::kF64:
        f(double{});
        return;
    default:
        break;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Output DataType");
}

// AllowTwo is false on the Tensor paths, where two-channel input is rejected up front, so no
// two-channel pixel type is instantiated there.
template<bool AllowTwo, class F>
void DispatchChannels(int channels, F &&f)
{
    switch (channels)
    {
    case 1:
        f(IntTag<1>{});
        return;
    case 2:
        if constexpr (AllowTwo)
        {
            f(IntTag<2>{});
            return;
        }
        break;
    case 3:
        f(IntTag<3>{});
        return;
    case 4:
        f(IntTag<4>{});
        return;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
}

// False when the stream does not apply and the caller must keep the pixel-shaped kernels.
inline bool TryNormalizeStream(const TensorArgs &a, DataTypeCode elemType, int channels)
{
    // Spatially varying parameters cannot be resolved once per thread.
    const int3 base_size  = ParamExtents(a.baseAccess);
    const int3 scale_size = ParamExtents(a.scaleAccess);
    if (base_size.x != 1 || base_size.y != 1)
    {
        return false;
    }
    if (scale_size.x != 1 || scale_size.y != 1)
    {
        return false;
    }

    // A per-channel parameter is read as a flat float array from its base pointer, so its channels
    // must be adjacent; a single-channel parameter broadcasts and needs no stride.
    constexpr int64_t kFloat     = sizeof(float);
    const bool        basePerCh  = a.baseAccess.numChannels() != 1;
    const bool        scalePerCh = a.scaleAccess.numChannels() != 1;
    if ((basePerCh && a.baseAccess.chStride() != kFloat) || (scalePerCh && a.scaleAccess.chStride() != kFloat))
    {
        return false;
    }
    if (a.base.dtype() != nvcv::TYPE_F32 || a.scale.dtype() != nvcv::TYPE_F32)
    {
        return false;
    }

    const int baseStride  = basePerCh ? 1 : 0;
    const int scaleStride = scalePerCh ? 1 : 0;

    bool launched = false;
    DispatchElemType(
        elemType,
        [&](auto elem)
        {
            using T = decltype(elem);
            DispatchChannels<false>(
                channels, [&](auto ncTag)
                { launched = TryNormalizeStreamAs<T, decltype(ncTag)::value>(a, baseStride, scaleStride); });
        });
    return launched;
}

// ---------------------------------------------------------------------------------------------------
// Validation and operator bodies.
// ---------------------------------------------------------------------------------------------------

// Every parameter extent must either match the input extent or be 1 (broadcast over that axis).
inline bool CheckParamShape(const DataShape &input_shape, const DataShape &param_shape)
{
    return (param_shape.N == input_shape.N || param_shape.N == 1)
        && (param_shape.C == input_shape.C || param_shape.C == 1)
        && (param_shape.H == input_shape.H || param_shape.H == 1)
        && (param_shape.W == input_shape.W || param_shape.W == 1);
}

// NHWC/HWC parameters are still allowed when the channel axis is 1, which both layouts express
// identically.
inline void RequirePlanarParamLayout(const nvcv::TensorDataStridedCuda &param, const Access &access, const char *name)
{
    if (!IsPlanar(ClassifyDataFormat(param.layout())) && access.numChannels() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s layout is not compatible with planar input; use NCHW/CHW (or a scalar layout)", name);
    }
}

inline Access RequireImageAccess(const nvcv::TensorDataStridedCuda &data, const char *name)
{
    auto access = Access::Create(data);
    if (!access)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s layout", name);
    }
    return *access;
}

// Whole formats are compared, so an NHWC input with an HWC output is a format mismatch even though
// both are interleaved.
inline bool RequireMatchingLayout(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                  const char *message)
{
    const DataFormatCode inLayout = ClassifyDataFormat(inData.layout());
    if (inLayout != ClassifyDataFormat(outData.layout()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", message);
    }
    return IsPlanar(inLayout);
}

inline void RequireChannelCount(int channels)
{
    if (channels > 4 || channels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }
}

template<class Data, class Src, class... Args>
auto ExportOrThrow(const Src &src, const char *message, Args &&...args)
{
    auto data = src.template exportData<Data>(std::forward<Args>(args)...);
    if (data == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", message);
    }
    return data;
}

inline void RunNormalize(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                         const nvcv::TensorDataStridedCuda &baseData, const nvcv::TensorDataStridedCuda &scaleData,
                         const nvcv::TensorDataStridedCuda &outData, float globalScale, float shift, float epsilon,
                         uint32_t flags)
{
    const bool isPlanar = RequireMatchingLayout(
        inData, outData, "Input and output must have the same layout (kNHWC, kHWC, kNCHW or kCHW)");

    const auto inAccess    = RequireImageAccess(inData, "input");
    const auto baseAccess  = RequireImageAccess(baseData, "base");
    const auto scaleAccess = RequireImageAccess(scaleData, "scale");
    const auto outAccess   = RequireImageAccess(outData, "output");

    if (isPlanar)
    {
        RequirePlanarParamLayout(baseData, baseAccess, "base");
        RequirePlanarParamLayout(scaleData, scaleAccess, "scale");
    }

    const DataTypeCode elemType    = ClassifyDataType(inData.dtype());
    const DataShape    input_shape = GetDataShape(inAccess);
    const int          channels    = input_shape.C;

    RequireChannelCount(channels);

    auto requireParamShape = [&](const char *name, const DataShape &shape)
    {
        if (!CheckParamShape(input_shape, shape))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid %s shape (N = %d, C = %d, H = %d, W = %d) for input shape "
                                  "(N = %d, C = %d, H = %d, W = %d); each dimension must either match the input "
                                  "or be 1",
                                  name, shape.N, shape.C, shape.H, shape.W, input_shape.N, input_shape.C, input_shape.H,
                                  input_shape.W);
        }
    };
    requireParamShape("base", GetDataShape(baseAccess));
    requireParamShape("scale", GetDataShape(scaleAccess));

    const NormalizeParams params{globalScale, shift, epsilon, (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV) != 0, stream};
    const TensorArgs args{inData, baseData, scaleData, outData, inAccess, baseAccess, scaleAccess, outAccess, params};

    if (isPlanar)
    {
        // Each channel is a separate plane of scalars, so one specialization per element type serves
        // every supported channel count.
        DispatchElemType(elemType, [&](auto elem) { normalizePlanar<decltype(elem)>(args); });
        return;
    }

    // For 8-bit, prefer the 16-wide uint4 path when the strides allow it, to maximize bytes in flight
    // per request; else the 4-wide path; else the scalar kernel. F32 has the float4 path only.
    if (params.isStdDev && channels == 1 && (elemType == DataTypeCode::kU8 || elemType == DataTypeCode::kF32))
    {
        const bool aligned16 = inAccess.rowStride() % 16 == 0 && inAccess.sampleStride() % 16 == 0
                            && outAccess.rowStride() % 16 == 0 && outAccess.sampleStride() % 16 == 0;

        if (elemType == DataTypeCode::kF32)
        {
            if (aligned16)
            {
                normalizeInvStdDevVec<float, kNormalizeF32ILPNGroup, 4>(args);
                return;
            }
        }
        else if (aligned16)
        {
            normalizeInvStdDevVec<uchar, kNormalizeILPNGroup, 16>(args);
            return;
        }
        else if (inAccess.rowStride() % 4 == 0 && inAccess.sampleStride() % 4 == 0 && outAccess.rowStride() % 4 == 0
                 && outAccess.sampleStride() % 4 == 0)
        {
            normalizeInvStdDevVec<uchar, 1, 4>(args);
            return;
        }
    }

    // Tried after the measured single-channel arms above, which keep their own tuned launch shapes.
    if (TryNormalizeStream(args, elemType, channels))
    {
        return;
    }

    DispatchElemType(elemType,
                     [&](auto elem)
                     {
                         DispatchChannels<false>(
                             channels,
                             [&](auto ncTag) { normalize<PixelType<decltype(elem), decltype(ncTag)::value>>(args); });
                     });
}

inline void RunNormalizeScalar(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData, float4 base,
                               float4 scale, int baseCount, int scaleCount, const nvcv::TensorDataStridedCuda &outData,
                               float globalScale, float shift, float epsilon, uint32_t flags)
{
    const bool isPlanar
        = RequireMatchingLayout(inData, outData,
                                "By-value (scalar/per-channel) normalize requires matching input and output layouts "
                                "(kNHWC, kHWC, kNCHW or kCHW)");

    const auto inAccess  = RequireImageAccess(inData, "input");
    const auto outAccess = RequireImageAccess(outData, "output");

    const DataTypeCode elemType    = ClassifyDataType(inData.dtype());
    const DataShape    input_shape = GetDataShape(inAccess);
    const int          channels    = input_shape.C;

    RequireChannelCount(channels);

    // The kernels bounds-check against the input extents only, so a mismatched output would be
    // written out of bounds (or with the wrong element type) through its wrap.
    if (elemType != ClassifyDataType(outData.dtype()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "DataType of input and output must be equal");
    }

    const DataShape output_shape = GetDataShape(outAccess);
    if (!(input_shape == output_shape))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Shape of input and output must be equal, but got (N = %d, C = %d, H = %d, W = %d) "
                              "and (N = %d, C = %d, H = %d, W = %d)",
                              input_shape.N, input_shape.C, input_shape.H, input_shape.W, output_shape.N,
                              output_shape.C, output_shape.H, output_shape.W);
    }

    auto requireParamCount = [&](const char *name, int count)
    {
        if (count != 1 && count != channels)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid %s value count %d for %d-channel input; must be 1 (broadcast) or the "
                                  "channel count",
                                  name, count, channels);
        }
    };
    requireParamCount("base", baseCount);
    requireParamCount("scale", scaleCount);

    const int64_t launchPlanes = static_cast<int64_t>(input_shape.N) * (isPlanar ? channels : 1);
    if (launchPlanes > kMaxGridZ)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "By-value normalize launch exceeds CUDA grid.z limit: %ld",
                              static_cast<long>(launchPlanes));
    }

    const NormalizeParams params{globalScale, shift, epsilon, (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV) != 0, stream};
    const ScalarArgs      args{inData, base, scale, baseCount, scaleCount, outData, inAccess, outAccess, params};

    if (isPlanar)
    {
        DispatchElemType(elemType, [&](auto elem) { normalizeScalarPlanar<decltype(elem)>(args); });
        return;
    }

    DispatchElemType(elemType,
                     [&](auto elem)
                     {
                         DispatchChannels<false>(
                             channels, [&](auto ncTag)
                             { normalizeScalar<PixelType<decltype(elem), decltype(ncTag)::value>>(args); });
                     });
}

inline void RunNormalizeVarShape(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                 const nvcv::TensorDataStridedCuda             &baseData,
                                 const nvcv::TensorDataStridedCuda             &scaleData,
                                 const nvcv::ImageBatchVarShapeDataStridedCuda &outData, float globalScale, float shift,
                                 float epsilon, uint32_t flags)
{
    if (!inData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in the input batch must all have the same format");
    }

    if (!outData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in the output batch must all have the same format");
    }

    const DataFormatCode inLayout  = ClassifyDataFormat(inData);
    const DataFormatCode outLayout = ClassifyDataFormat(outData);
    if (inLayout != outLayout)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same layout (kNHWC, kHWC, kNCHW or kCHW)");
    }

    const bool isPlanar = IsPlanar(inLayout);

    const DataTypeCode inType = ClassifyDataType(inData.uniqueFormat());

    // The var-shape path converts the output dtype independently of the input dtype, but only to
    // these types.
    const DataTypeCode outType = ClassifyDataType(outData.uniqueFormat());
    if (outType != DataTypeCode::kU8 && outType != DataTypeCode::kF32 && outType != DataTypeCode::kF16
        && outType != DataTypeCode::kF64)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid Output DataType: only 8-bit unsigned, 16-bit float, 32-bit float and 64-bit "
                              "float outputs are supported");
    }

    const int channels = inData.uniqueFormat().numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    const auto baseAccess  = RequireImageAccess(baseData, "base");
    const auto scaleAccess = RequireImageAccess(scaleData, "scale");

    if (isPlanar)
    {
        RequirePlanarParamLayout(baseData, baseAccess, "base");
        RequirePlanarParamLayout(scaleData, scaleAccess, "scale");

        // The planar var-shape kernels read one value per (batch, channel) from a batch-invariant
        // parameter tensor, so only [1,1,1,1] and [1,C,1,1] are representable.
        auto requireParamShape = [&](const char *name, const Access &access)
        {
            if (!(access.numSamples() == 1 && (access.numChannels() == 1 || access.numChannels() == channels)
                  && access.numRows() == 1 && access.numCols() == 1))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Invalid planar %s shape; expected scalar [1,1,1,1] or per-channel [1,C,1,1]",
                                      name);
            }
        };
        requireParamShape("base", baseAccess);
        requireParamShape("scale", scaleAccess);
    }

    const NormalizeParams params{globalScale, shift, epsilon, (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV) != 0, stream};
    const VarShapeArgs    args{inData, baseData, scaleData, baseAccess, scaleAccess, outData, channels, params};

    DispatchElemType(inType,
                     [&](auto inElem)
                     {
                         using T = decltype(inElem);
                         DispatchVarShapeOutType(outType,
                                                 [&](auto outElem)
                                                 {
                                                     using OutT = decltype(outElem);
                                                     if (isPlanar)
                                                     {
                                                         normPlanar<T, OutT>(args);
                                                         return;
                                                     }
                                                     DispatchChannels<true>(
                                                         channels,
                                                         [&](auto ncTag)
                                                         {
                                                             constexpr int NC = decltype(ncTag)::value;
                                                             norm<PixelType<T, NC>, PixelType<OutT, NC>>(args);
                                                         });
                                                 });
                     });
}

} // namespace

namespace cvcuda::priv {

void Normalize::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &base,
                           const nvcv::Tensor &scale, const nvcv::Tensor &out, const float global_scale,
                           const float shift, const float epsilon, const uint32_t flags) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Normalize::operator()[Tensor]");

    auto inData = ExportOrThrow<nvcv::TensorDataStridedCuda>(in, "Input must be cuda-accessible, pitch-linear tensor");
    auto baseData
        = ExportOrThrow<nvcv::TensorDataStridedCuda>(base, "Input base must be cuda-accessible, pitch-linear tensor");
    auto scaleData
        = ExportOrThrow<nvcv::TensorDataStridedCuda>(scale, "Input scale must be cuda-accessible, pitch-linear tensor");
    auto outData
        = ExportOrThrow<nvcv::TensorDataStridedCuda>(out, "Output must be cuda-accessible, pitch-linear tensor");

    RunNormalize(stream, *inData, *baseData, *scaleData, *outData, global_scale, shift, epsilon, flags);
}

void Normalize::operator()(cudaStream_t stream, const nvcv::Tensor &in, const float4 base, const float4 scale,
                           const int baseCount, const int scaleCount, const nvcv::Tensor &out, const float global_scale,
                           const float shift, const float epsilon, const uint32_t flags) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Normalize::operator()[Tensor scalar]");

    auto inData = ExportOrThrow<nvcv::TensorDataStridedCuda>(in, "Input must be cuda-accessible, pitch-linear tensor");
    auto outData
        = ExportOrThrow<nvcv::TensorDataStridedCuda>(out, "Output must be cuda-accessible, pitch-linear tensor");

    RunNormalizeScalar(stream, *inData, base, scale, baseCount, scaleCount, *outData, global_scale, shift, epsilon,
                       flags);
}

void Normalize::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &base,
                           const nvcv::Tensor &scale, const nvcv::ImageBatchVarShape &out, const float global_scale,
                           const float shift, const float epsilon, const uint32_t flags) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Normalize::operator()[ImageBatchVarShape]");

    auto inData = ExportOrThrow<nvcv::ImageBatchVarShapeDataStridedCuda>(
        in, "Input must be cuda-accessible, varshape pitch-linear image batch", stream);
    auto baseData
        = ExportOrThrow<nvcv::TensorDataStridedCuda>(base, "Input base must be cuda-accessible, pitch-linear tensor");
    auto scaleData
        = ExportOrThrow<nvcv::TensorDataStridedCuda>(scale, "Input scale must be cuda-accessible, pitch-linear tensor");
    auto outData = ExportOrThrow<nvcv::ImageBatchVarShapeDataStridedCuda>(
        out, "Output must be cuda-accessible, varshape pitch-linear image batch", stream);

    RunNormalizeVarShape(stream, *inData, *baseData, *scaleData, *outData, global_scale, shift, epsilon, flags);
}

} // namespace cvcuda::priv

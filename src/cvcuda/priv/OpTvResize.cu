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

#include "Nvtx.hpp"
#include "OpTvResize.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <vector>

namespace cvcuda::priv {
namespace {

static void Require(bool condition, const char *message)
{
    if (!condition)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, message);
    }
}

// CUDA pointer differences and the workspace allocator both use signed 64-bit sizes.
constexpr uint64_t kMaxSize = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

static uint64_t CheckedMul(uint64_t a, uint64_t b)
{
    Require(a == 0 || b <= kMaxSize / a, "Size arithmetic overflow");
    return a * b;
}

static uint64_t CheckedAdd(uint64_t a, uint64_t b)
{
    Require(a <= kMaxSize && b <= kMaxSize - a, "Size arithmetic overflow");
    return a + b;
}

static bool IsNearest(NVCVInterpolationType interpolation)
{
    return interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_NEAREST_FLOOR;
}

static void ValidateMode(NVCVInterpolationType interpolation)
{
    Require(IsNearest(interpolation) || interpolation == NVCV_INTERP_LINEAR || interpolation == NVCV_INTERP_CUBIC,
            "Unsupported interpolation mode");
}

static void ValidateCount(int64_t count)
{
    Require(count >= 0 && count <= 65535, "Batch size must be in [0, 65535]");
}

static int PositiveExtent(int64_t extent)
{
    Require(extent > 0 && extent <= std::numeric_limits<int32_t>::max(), "Invalid spatial or channel extent");
    return static_cast<int>(extent);
}

static uint64_t Elements(int n, int h, int w, int c)
{
    return CheckedMul(CheckedMul(CheckedMul(n, h), w), c);
}

// Coefficient storage holds 32 taps per output: at most 2 * ratio + 1 linear or 4 * ratio + 1 cubic taps.
static bool CanUseAACoefficients(int sh, int sw, int dh, int dw, bool cubic)
{
    int64_t ratio = cubic ? 7 : 15;
    return sh >= 8 && sw >= 8 && dh >= 16 && dw >= 16 && sh != dh && sw != dw && dh <= 4096 && dw <= 4096
        && static_cast<int64_t>(sw) <= ratio * dw && static_cast<int64_t>(sh) <= ratio * dh;
}

static size_t ScratchSize(int n, int sh, int sw, int dh, int dw, int c, NVCVInterpolationType interpolation, bool aa)
{
    ValidateCount(n);
    PositiveExtent(sh);
    PositiveExtent(sw);
    PositiveExtent(dh);
    PositiveExtent(dw);
    PositiveExtent(c);
    Elements(n, sh, sw, c);
    Elements(n, dh, dw, c);
    if (n == 0 || IsNearest(interpolation) || (sh == dh && sw == dw))
    {
        return 0;
    }
    size_t bytes = CheckedMul(Elements(n, sh, dw, c), sizeof(float));
    if (aa && CanUseAACoefficients(sh, sw, dh, dw, interpolation == NVCV_INTERP_CUBIC))
    {
        bytes = CheckedAdd(bytes, (static_cast<uint64_t>(dh) + dw) * 34 * sizeof(float));
    }
    return bytes;
}

static WorkspaceRequirements Requirements(size_t bytes)
{
    WorkspaceRequirements result{};
    result.hostMem.alignment   = 1;
    result.pinnedMem.alignment = 1;
    result.cudaMem.alignment   = alignof(float);
    result.cudaMem.size        = bytes;
    return result;
}

static void ValidateWorkspace(const Workspace &workspace, size_t bytes)
{
    if (bytes != 0)
    {
        Require(workspace.cudaMem.data != nullptr && workspace.cudaMem.req.size >= bytes,
                "Insufficient CUDA workspace");
        Require(reinterpret_cast<uintptr_t>(workspace.cudaMem.data) % alignof(float) == 0
                    && workspace.cudaMem.req.alignment >= alignof(float)
                    && (workspace.cudaMem.req.alignment & (workspace.cudaMem.req.alignment - 1)) == 0,
                "Misaligned CUDA workspace");
        Require(bytes <= std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(workspace.cudaMem.data),
                "Workspace address overflow");
    }
}

static void BeginWorkspace(cudaStream_t stream, const Workspace &workspace, size_t bytes)
{
    if (bytes != 0 && workspace.cudaMem.ready)
    {
        NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, workspace.cudaMem.ready, 0));
    }
}

static void EndWorkspace(cudaStream_t stream, const Workspace &workspace, size_t bytes)
{
    if (bytes != 0 && workspace.cudaMem.ready)
    {
        NVCV_CHECK_THROW(cudaEventRecord(workspace.cudaMem.ready, stream));
    }
}

static int ScalarBytes(nvcv::DataType dtype)
{
    Require(dtype == nvcv::TYPE_U8 || dtype == nvcv::TYPE_U16 || dtype == nvcv::TYPE_S16 || dtype == nvcv::TYPE_F16
                || dtype == nvcv::TYPE_F32,
            "Supported scalar types are U8, U16, S16, F16, F32");
    return dtype.strideBytes();
}

struct View
{
    unsigned char *data;
    int            n, h, w, c;
    uint64_t       sn, sy, sx, sc;

    __device__ unsigned char *ptr(int sample, int y, int x, int channel) const
    {
        return data + sample * sn + y * sy + x * sx + channel * sc;
    }
};

struct ViewPair
{
    View src;
    View dst;
};

constexpr int kBatchPacketCapacity = 32;

struct BatchPacket
{
    ViewPair images[kBatchPacketCapacity];
};

static_assert(sizeof(BatchPacket) <= 4096 - 16);

struct NearestImage
{
    unsigned char *source;
    unsigned char *destination;
    uint64_t       sourcePitch;
    uint64_t       destinationPitch;
    int            sourceHeight, sourceWidth, destinationHeight, destinationWidth;
};

struct NearestBatchPacket
{
    NearestImage images[kBatchPacketCapacity];
    int          channels;
};

static_assert(sizeof(NearestBatchPacket) <= 4096 - 16);

static void ValidateAddress(const View &view, uint64_t span, int alignment)
{
    if (view.n != 0)
    {
        Require(view.data != nullptr, "Null image storage");
        Require(reinterpret_cast<uintptr_t>(view.data) % alignment == 0, "Misaligned image storage");
        Require(span <= std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(view.data),
                "Image address overflow");
    }
}

static View TensorView(const nvcv::TensorDataStridedCuda &data)
{
    auto layout  = data.layout();
    bool planar  = layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
    bool batched = layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_NHWC;
    Require(planar || layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC, "Unsupported tensor layout");
    Require(data.rank() == (batched ? 4 : 3), "Invalid tensor rank");
    int     base    = batched ? 1 : 0;
    int     hi      = base + (planar ? 1 : 0);
    int     wi      = hi + 1;
    int     ci      = base + (planar ? 0 : 2);
    int64_t samples = batched ? data.shape(0) : 1;
    ValidateCount(samples);
    int h = PositiveExtent(data.shape(hi));
    int w = PositiveExtent(data.shape(wi));
    int c = PositiveExtent(data.shape(ci));
    Require(planar ? samples * c <= 65535 : c <= 4, "Unsupported tensor channel count");
    int      bytes = ScalarBytes(data.dtype());
    uint64_t span  = bytes;
    for (int axis = data.rank() - 1; axis >= 0; --axis)
    {
        int64_t stride = data.stride(axis);
        Require(stride > 0 && stride % bytes == 0, "Tensor strides must be positive and scalar-aligned");
        int64_t extent = data.shape(axis);
        if (extent > 1)
        {
            Require(static_cast<uint64_t>(stride) >= span, "Overlapping tensor elements");
            span = CheckedAdd(span, CheckedMul(extent - 1, stride));
        }
    }
    View view{reinterpret_cast<unsigned char *>(data.basePtr()),
              static_cast<int>(samples),
              h,
              w,
              c,
              batched ? static_cast<uint64_t>(data.stride(0)) : 0,
              static_cast<uint64_t>(data.stride(hi)),
              static_cast<uint64_t>(data.stride(wi)),
              static_cast<uint64_t>(data.stride(ci))};
    ValidateAddress(view, span, bytes);
    return view;
}

static nvcv::DataType ImageType(nvcv::ImageFormat format)
{
    int c = format.numChannels();
    Require(format.numPlanes() == 1 && (c == 1 || c == 3 || c == 4),
            "VarShape requires a single interleaved plane with 1, 3 or 4 channels");
    auto packed = format.planeDataType(0);
    auto scalar = packed.channelType(0);
    int  bytes  = ScalarBytes(scalar);
    Require(packed.numChannels() == c && format.planePixelStrideBytes(0) == c * bytes,
            "Packed image channels are unsupported");
    for (int i = 1; i < c; ++i)
    {
        Require(packed.channelType(i) == scalar, "Image channels must have equal scalar type");
    }
    return scalar;
}

static View ImageView(const nvcv::Image &image, nvcv::ImageFormat format, int bytes)
{
    Require(image.format() == format, "VarShape images must have one uniform matching format");
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    Require(data && data->numPlanes() == 1, "Images must be CUDA-accessible and pitch-linear");
    const auto &plane    = data->plane(0);
    int         h        = PositiveExtent(plane.height);
    int         w        = PositiveExtent(plane.width);
    int         c        = format.numChannels();
    uint64_t    rowBytes = CheckedMul(CheckedMul(w, c), bytes);
    Require(plane.rowStride > 0 && plane.rowStride % bytes == 0 && static_cast<uint64_t>(plane.rowStride) >= rowBytes,
            "Invalid image row stride");
    uint64_t span = CheckedAdd(CheckedMul(h - 1, plane.rowStride), rowBytes);
    View     view{reinterpret_cast<unsigned char *>(plane.basePtr),
              1,
              h,
              w,
              c,
              0,
              static_cast<uint64_t>(plane.rowStride),
              static_cast<uint64_t>(c * bytes),
              static_cast<uint64_t>(bytes)};
    ValidateAddress(view, span, bytes);
    return view;
}

static __device__ int NearestIndex(int d, int source, int destination, bool exact)
{
    float   scale    = __fdiv_rn(static_cast<float>(source), static_cast<float>(destination));
    float   position = __fmul_rn(__fadd_rn(static_cast<float>(d), exact ? 0.5f : 0.0f), scale);
    // Convert through int64 so rounding near INT32_MAX cannot overflow the conversion.
    int64_t index = static_cast<int64_t>(floorf(position));
    return static_cast<int>(index < source ? index : source - 1);
}

constexpr int kGatherWidth = 32, kGatherRows = 8, kGatherSamples = 4;

// One thread per destination pixel: source indices are computed once and reused across channels and samples.
template<class Bits>
__device__ void GatherPixels(const View &src, const View &dst, int n0, int n1, bool exact, bool identity)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= static_cast<unsigned>(dst.w))
    {
        return;
    }
    int sx = identity ? x : NearestIndex(x, src.w, dst.w, exact);
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < static_cast<unsigned>(dst.h);
         y += gridDim.y * blockDim.y)
    {
        int                  sy = identity ? y : NearestIndex(y, src.h, dst.h, exact);
        const unsigned char *s  = src.ptr(n0, sy, sx, 0);
        unsigned char       *d  = dst.ptr(n0, y, x, 0);
        for (int n = n0; n < n1; ++n, s += src.sn, d += dst.sn)
        {
            for (int c = 0; c < dst.c; ++c)
            {
                *reinterpret_cast<Bits *>(d + c * dst.sc) = *reinterpret_cast<const Bits *>(s + c * src.sc);
            }
        }
    }
}

template<class Bits>
__global__ void Gather(View src, View dst, bool exact, bool identity)
{
    int n0 = blockIdx.z * kGatherSamples;
    GatherPixels<Bits>(src, dst, n0, min(n0 + kGatherSamples, dst.n), exact, identity);
}

template<class Bits>
__global__ void GatherBatch(const __grid_constant__ NearestBatchPacket packet, bool exact)
{
    const auto &image = packet.images[blockIdx.z];
    int         c     = packet.channels;
    uint64_t    sx    = static_cast<uint64_t>(c) * sizeof(Bits);
    View        src{image.source, 1, image.sourceHeight, image.sourceWidth, c, 0, image.sourcePitch, sx, sizeof(Bits)};
    View dst{image.destination, 1, image.destinationHeight, image.destinationWidth, c, 0, image.destinationPitch, sx,
             sizeof(Bits)};
    GatherPixels<Bits>(src, dst, 0, 1, exact, src.h == dst.h && src.w == dst.w);
}

static dim3 GatherGrid(int h, int w, int z)
{
    uint64_t x = (static_cast<uint64_t>(w) + kGatherWidth - 1) / kGatherWidth;
    uint64_t y = (static_cast<uint64_t>(h) + kGatherRows - 1) / kGatherRows;
    return dim3(static_cast<unsigned>(x), static_cast<unsigned>(std::min<uint64_t>(y, 65535)), z);
}

static __device__ float FilterWeight(float distance, bool cubic, float a)
{
    float t = fabsf(distance);
    if (!cubic)
    {
        return fmaxf(0.0f, 1.0f - t);
    }
    if (t < 1.0f)
    {
        return ((a + 2.0f) * t - (a + 3.0f)) * t * t + 1.0f;
    }
    if (t < 2.0f)
    {
        return ((a * t - 5.0f * a) * t + 8.0f * a) * t - 4.0f * a;
    }
    return 0.0f;
}

struct Taps
{
    int64_t first, end, anchor;
    float   fraction, dilation;
    bool    cubic, aa;

    template<bool Narrow = false>
    __device__ float weight(int64_t i) const
    {
        // Subtract integer coordinates before conversion so adjacent taps remain distinct above 2^24.
        float distance;
        if constexpr (Narrow)
        {
            distance = static_cast<float>(static_cast<int32_t>(i - anchor)) - fraction;
        }
        else
        {
            distance = static_cast<float>(i - anchor) - fraction;
        }
        return FilterWeight(distance / dilation, cubic, aa ? -0.5f : -0.75f);
    }

    // weight() with the FMA contraction ptxas picks in the single-output generic kernels, spelled out so the
    // multi-output generic kernels, which the compiler contracts differently, stay bit-identical.
    __device__ float exactWeight(int64_t i) const
    {
        float t = fabsf((static_cast<float>(i - anchor) - fraction) / dilation);
        if (!cubic)
        {
            return fmaxf(0.0f, 1.0f - t);
        }
        float a = aa ? -0.5f : -0.75f;
        if (t < 1.0f)
        {
            return __fmaf_rn(t, __fmul_rn(t, __fmaf_rn(a + 2.0f, t, -(a + 3.0f))), 1.0f);
        }
        if (t < 2.0f)
        {
            return __fmaf_rn(t, __fmaf_rn(t, __fmaf_rn(a, t, -(5.0f * a)), 8.0f * a), -(4.0f * a));
        }
        return 0.0f;
    }
};

static __device__ Taps MakeTaps(int destinationIndex, int sourceSize, int destinationSize, bool cubic, bool aa)
{
    float scale    = __fdiv_rn(static_cast<float>(sourceSize), static_cast<float>(destinationSize));
    float center   = __fadd_rn(__fmul_rn(__fadd_rn(static_cast<float>(destinationIndex), 0.5f), scale), -0.5f);
    float dilation = aa ? fmaxf(1.0f, scale) : 1.0f;
    if (!aa && !cubic)
    {
        center = fmaxf(0.0f, center);
    }
    int64_t anchor   = static_cast<int64_t>(floorf(center));
    float   fraction = center - static_cast<float>(anchor);
    if (aa)
    {
        // Float32 rounding can move the center beyond the last source pixel and erase all surviving weights.
        if (anchor >= sourceSize)
        {
            anchor   = sourceSize - 1;
            fraction = 0.0f;
        }
        float   support = (cubic ? 2.0f : 1.0f) * dilation;
        int64_t first   = anchor + static_cast<int64_t>(ceilf(fraction - support));
        int64_t end     = anchor + static_cast<int64_t>(floorf(fraction + support)) + 1;
        return {first < 0 ? 0 : first, end > sourceSize ? sourceSize : end, anchor, fraction, dilation, cubic, aa};
    }
    int64_t first = anchor - (cubic ? 1 : 0);
    return {first, first + (cubic ? 4 : 2), anchor, fraction, 1.0f, cubic, aa};
}

static __device__ int ClampIndex(int64_t i, int size)
{
    return static_cast<int>(i < 0 ? 0 : (i >= size ? size - 1 : i));
}

template<class T>
__device__ float ToFloat(T value)
{
    return static_cast<float>(value);
}

template<>
__device__ float ToFloat<__half>(__half value)
{
    return __half2float(value);
}

template<class T>
__device__ T StoreValue(float value);

template<>
__device__ uint8_t StoreValue<uint8_t>(float value)
{
    return static_cast<uint8_t>(__float2int_rn(fminf(255.0f, fmaxf(0.0f, value))));
}

template<>
__device__ uint16_t StoreValue<uint16_t>(float value)
{
    return static_cast<uint16_t>(__float2int_rn(fminf(65535.0f, fmaxf(0.0f, value))));
}

template<>
__device__ int16_t StoreValue<int16_t>(float value)
{
    return static_cast<int16_t>(__float2int_rn(fminf(32767.0f, fmaxf(-32768.0f, value))));
}

template<>
__device__ __half StoreValue<__half>(float value)
{
    return __float2half_rn(value);
}

template<>
__device__ float StoreValue<float>(float value)
{
    return value;
}

// One row per thread; the multi-row AA form below loses on tiny outputs and its exact weights cost extra registers.
template<class T>
__global__ void HorizontalScalar(View src, int dstWidth, float *scratch, uint64_t count, bool cubic, bool aa)
{
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    {
        uint64_t rest = index;
        int      c    = rest % src.c;
        rest /= src.c;
        int x = rest % dstWidth;
        rest /= dstWidth;
        int   y     = rest % src.h;
        int   n     = rest / src.h;
        float value = 0.0f;
        if (src.w == dstWidth)
        {
            value = ToFloat(*reinterpret_cast<const T *>(src.ptr(n, y, x, c)));
        }
        else
        {
            Taps  taps = MakeTaps(x, src.w, dstWidth, cubic, aa);
            float sum  = 0.0f;
            if (aa)
            {
                for (int64_t i = taps.first; i < taps.end; ++i)
                {
                    sum += taps.weight(i);
                }
            }
            for (int64_t i = taps.first; i < taps.end; ++i)
            {
                float weight = taps.weight(i);
                if (aa)
                {
                    weight /= sum;
                }
                value += weight * ToFloat(*reinterpret_cast<const T *>(src.ptr(n, y, ClampIndex(i, src.w), c)));
            }
        }
        scratch[index] = value;
    }
}

// AA recomputes every weight per output, so AA threads reuse each weight across Rows source rows.
template<class T, int Rows>
__global__ void Horizontal(View src, int dstWidth, float *scratch, uint64_t count, bool cubic, bool aa)
{
    int groups = (src.h - 1) / Rows + 1; // Not (h + Rows - 1): h may be INT_MAX.
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    {
        uint64_t rest = index;
        int      c    = rest % src.c;
        rest /= src.c;
        int x = rest % dstWidth;
        rest /= dstWidth;
        int   y0 = static_cast<int>(rest % groups) * Rows;
        int   n  = rest / groups;
        float value[Rows]{};
        if (src.w == dstWidth)
        {
#pragma unroll
            for (int r = 0; r < Rows; ++r)
            {
                if (y0 + r < src.h)
                {
                    value[r] = ToFloat(*reinterpret_cast<const T *>(src.ptr(n, y0 + r, x, c)));
                }
            }
        }
        else
        {
            Taps  taps = MakeTaps(x, src.w, dstWidth, cubic, aa);
            float sum  = 0.0f;
            if (aa)
            {
                for (int64_t i = taps.first; i < taps.end; ++i)
                {
                    sum += taps.exactWeight(i);
                }
            }
            for (int64_t i = taps.first; i < taps.end; ++i)
            {
                float weight = taps.exactWeight(i);
                if (aa)
                {
                    weight /= sum;
                }
                int xi = ClampIndex(i, src.w);
#pragma unroll
                for (int r = 0; r < Rows; ++r)
                {
                    if (y0 + r < src.h)
                    {
                        value[r] = __fmaf_rn(weight, ToFloat(*reinterpret_cast<const T *>(src.ptr(n, y0 + r, xi, c))),
                                             value[r]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r)
        {
            if (y0 + r < src.h)
            {
                scratch[((static_cast<uint64_t>(n) * src.h + y0 + r) * dstWidth + x) * src.c + c] = value[r];
            }
        }
    }
}

template<int Channels, bool Vector>
__device__ __forceinline__ void LoadPixel(const uint8_t *pixel, float (&value)[Channels])
{
    if constexpr (Vector)
    {
        uchar4 packed = *reinterpret_cast<const uchar4 *>(pixel);
        value[0]      = ToFloat(packed.x);
        value[1]      = ToFloat(packed.y);
        value[2]      = ToFloat(packed.z);
        value[3]      = ToFloat(packed.w);
    }
    else
    {
#pragma unroll
        for (int c = 0; c < Channels; ++c) value[c] = ToFloat(pixel[c]);
    }
}

template<int Channels, bool Vector>
__device__ __forceinline__ void StorePixel(uint8_t *pixel, const float (&value)[Channels])
{
    if constexpr (Vector)
    {
        *reinterpret_cast<uchar4 *>(pixel) = make_uchar4(StoreValue<uint8_t>(value[0]), StoreValue<uint8_t>(value[1]),
                                                         StoreValue<uint8_t>(value[2]), StoreValue<uint8_t>(value[3]));
    }
    else
    {
#pragma unroll
        for (int c = 0; c < Channels; ++c) pixel[c] = StoreValue<uint8_t>(value[c]);
    }
}

// Callers guarantee w * Channels fits in int, so column byte offsets stay 32-bit.
template<int Channels, bool Vector>
__device__ __forceinline__ void CubicInterleavedValues(const View &src, const View &dst, const uint8_t *source,
                                                       uint8_t *destination)
{
    int x = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= dst.w)
    {
        return;
    }

    Taps     tapsX = MakeTaps(x, src.w, dst.w, true, false);
    float    weightsX[4];
    uint32_t offsetsX[4];
#pragma unroll
    for (int tap = 0; tap < 4; ++tap)
    {
        weightsX[tap] = tapsX.weight(tapsX.first + tap);
        offsetsX[tap] = static_cast<uint32_t>(ClampIndex(tapsX.first + tap, src.w)) * Channels;
    }
    destination += static_cast<uint32_t>(x) * Channels;
    for (int64_t y64 = blockIdx.y; y64 < dst.h; y64 += gridDim.y)
    {
        int   y     = static_cast<int>(y64);
        Taps  tapsY = MakeTaps(y, src.h, dst.h, true, false);
        float value[Channels]{};
#pragma unroll
        for (int tapY = 0; tapY < 4; ++tapY)
        {
            int64_t        iy = tapsY.first + tapY;
            const uint8_t *in = source + ClampIndex(iy, src.h) * src.sy;
            float          row[Channels]{};
#pragma unroll
            for (int tap = 0; tap < 4; ++tap)
            {
                float pixel[Channels];
                LoadPixel<Channels, Vector>(in + offsetsX[tap], pixel);
#pragma unroll
                for (int c = 0; c < Channels; ++c)
                {
                    row[c] += weightsX[tap] * pixel[c];
                }
            }
            float weight = tapsY.weight(iy);
#pragma unroll
            for (int c = 0; c < Channels; ++c)
            {
                value[c] += weight * row[c];
            }
        }
        StorePixel<Channels, Vector>(destination + y * dst.sy, value);
    }
}

template<int Channels, bool Vector>
__global__ __launch_bounds__(256) void CubicInterleavedU8(View src, View dst)
{
    CubicInterleavedValues<Channels, Vector>(src, dst, src.data + blockIdx.z * src.sn, dst.data + blockIdx.z * dst.sn);
}

template<int Channels, bool Vector>
__global__ __launch_bounds__(256) void CubicInterleavedBatch(const __grid_constant__ BatchPacket packet)
{
    const auto &image = packet.images[blockIdx.z];
    CubicInterleavedValues<Channels, Vector>(image.src, image.dst, image.src.data, image.dst.data);
}

template<class T, int Channels, bool Precompute = false>
__global__ void CubicInterleavedT(View src, View dst)
{
    int x = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= dst.w)
    {
        return;
    }

    Taps  tapsX = MakeTaps(x, src.w, dst.w, true, false);
    float weightsX[4];
    if constexpr (Precompute)
    {
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) weightsX[tap] = tapsX.weight(tapsX.first + tap);
    }
    for (int64_t y64 = blockIdx.y; y64 < dst.h; y64 += gridDim.y)
    {
        int   y     = static_cast<int>(y64);
        Taps  tapsY = MakeTaps(y, src.h, dst.h, true, false);
        float value[Channels]{};
        for (int64_t iy = tapsY.first; iy < tapsY.end; ++iy)
        {
            float row[Channels]{};
            for (int64_t ix = tapsX.first; ix < tapsX.end; ++ix)
            {
                float weight = Precompute ? weightsX[ix - tapsX.first] : tapsX.weight(ix);
#pragma unroll
                for (int c = 0; c < Channels; ++c)
                {
                    row[c] += weight
                            * ToFloat(*reinterpret_cast<const T *>(
                                src.ptr(blockIdx.z, ClampIndex(iy, src.h), ClampIndex(ix, src.w), c)));
                }
            }
            float weight = tapsY.weight(iy);
#pragma unroll
            for (int c = 0; c < Channels; ++c)
            {
                value[c] += weight * row[c];
            }
        }
#pragma unroll
        for (int c = 0; c < Channels; ++c)
        {
            *reinterpret_cast<T *>(dst.ptr(blockIdx.z, y, x, c)) = StoreValue<T>(value[c]);
        }
    }
}

template<class T, int Channels>
__device__ __forceinline__ void AccumulateBatchChannelsT(float (&row)[Channels], float weight, View src, int64_t iy,
                                                         int64_t ix)
{
#pragma unroll
    for (int channel = 0; channel < Channels; ++channel)
    {
        row[channel]
            += weight
             * ToFloat(*reinterpret_cast<const T *>(src.ptr(0, ClampIndex(iy, src.h), ClampIndex(ix, src.w), channel)));
    }
}

template<class T, int Channels, bool Precompute>
__device__ void CubicInterleavedBatchValuesT(View src, View dst)
{
    int x = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= dst.w)
    {
        return;
    }

    Taps  tapsX = MakeTaps(x, src.w, dst.w, true, false);
    float weightsX[4];
    if constexpr (Precompute)
    {
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) weightsX[tap] = tapsX.weight(tapsX.first + tap);
    }
    for (int64_t y64 = blockIdx.y; y64 < dst.h; y64 += gridDim.y)
    {
        int   y     = static_cast<int>(y64);
        Taps  tapsY = MakeTaps(y, src.h, dst.h, true, false);
        float value[Channels]{};
        for (int64_t iy = tapsY.first; iy < tapsY.end; ++iy)
        {
            float row[Channels]{};
            for (int64_t ix = tapsX.first; ix < tapsX.end; ++ix)
            {
                float weight = Precompute ? weightsX[ix - tapsX.first] : tapsX.weight(ix);
                AccumulateBatchChannelsT<T>(row, weight, src, iy, ix);
            }
            float weight = tapsY.weight(iy);
#pragma unroll
            for (int c = 0; c < Channels; ++c)
            {
                value[c] += weight * row[c];
            }
        }
#pragma unroll
        for (int c = 0; c < Channels; ++c)
        {
            *reinterpret_cast<T *>(dst.ptr(0, y, x, c)) = StoreValue<T>(value[c]);
        }
    }
}

template<class T, int Channels, bool Precompute>
__global__ void CubicInterleavedBatchT(const __grid_constant__ BatchPacket packet)
{
    const auto &image = packet.images[blockIdx.z];
    CubicInterleavedBatchValuesT<T, Channels, Precompute>(image.src, image.dst);
}

// AA recomputes every weight per output, so AA threads reuse each weight across Items scalars of one output row;
// each warp covers 32 * Items consecutive scalars with lane stride 32 to keep accesses coalesced.
template<class T, int Items>
__global__ void Vertical(const float *scratch, int srcHeight, View dst, uint64_t count, bool cubic, bool aa)
{
    uint64_t      rowStride = static_cast<uint64_t>(dst.w) * dst.c;
    constexpr int kLanes    = Items == 1 ? 1 : 32;
    uint64_t      chunks    = (rowStride + kLanes * Items - 1) / (kLanes * Items);
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    {
        uint64_t chunk  = index / kLanes;
        uint64_t base   = (chunk % chunks) * kLanes * Items + index % kLanes;
        uint64_t rest   = chunk / chunks;
        int      y      = rest % dst.h;
        int      n      = rest / dst.h;
        uint64_t offset = static_cast<uint64_t>(n) * srcHeight * rowStride + base;
        float    value[Items]{};
        if (srcHeight == dst.h)
        {
#pragma unroll
            for (int k = 0; k < Items; ++k)
            {
                if (base + k * kLanes < rowStride)
                {
                    value[k] = scratch[offset + k * kLanes + y * rowStride];
                }
            }
        }
        else
        {
            Taps  taps = MakeTaps(y, srcHeight, dst.h, cubic, aa);
            float sum  = 0.0f;
            if (aa)
            {
                for (int64_t i = taps.first; i < taps.end; ++i)
                {
                    sum += (Items > 1 ? taps.exactWeight(i) : taps.weight(i));
                }
            }
            for (int64_t i = taps.first; i < taps.end; ++i)
            {
                float weight = (Items > 1 ? taps.exactWeight(i) : taps.weight(i));
                if (aa)
                {
                    weight /= sum;
                }
                const float *row = scratch + offset + ClampIndex(i, srcHeight) * rowStride;
#pragma unroll
                for (int k = 0; k < Items; ++k)
                {
                    if (base + k * kLanes < rowStride)
                    {
                        value[k] = __fmaf_rn(weight, row[k * kLanes], value[k]);
                    }
                }
            }
        }
#pragma unroll
        for (int k = 0; k < Items; ++k)
        {
            uint64_t scalar = base + k * kLanes;
            if (scalar < rowStride)
            {
                *reinterpret_cast<T *>(dst.ptr(n, y, static_cast<int>(scalar / dst.c), scalar % dst.c))
                    = StoreValue<T>(value[k]);
            }
        }
    }
}

static int Blocks(uint64_t count)
{
    return static_cast<int>(std::min<uint64_t>((count + 255) / 256, 65535));
}

struct AxisCoefficients
{
    float *weights;
    int   *first;
    int   *count;
    int    size;
};

static AxisCoefficients Coefficients(float *storage, int size)
{
    auto *first = reinterpret_cast<int *>(storage + size * 32);
    return {storage, first, first + size, size};
}

template<bool Cubic>
__global__ void PrepareAACoefficients(View src, View dst, AxisCoefficients horizontal, AxisCoefficients vertical)
{
    int              coordinate   = blockIdx.x * blockDim.x + threadIdx.x;
    bool             isHorizontal = coordinate < dst.w;
    AxisCoefficients axis         = isHorizontal ? horizontal : vertical;
    int              index        = isHorizontal ? coordinate : coordinate - dst.w;
    if (index >= axis.size)
    {
        return;
    }
    Taps  taps = MakeTaps(index, isHorizontal ? src.w : src.h, axis.size, Cubic, true);
    float sum  = 0.0f;
    for (int64_t tap = taps.first; tap < taps.end; ++tap)
    {
        sum += taps.weight<true>(tap);
    }
    axis.first[index] = static_cast<int>(taps.first);
    axis.count[index] = static_cast<int>(taps.end - taps.first);
    for (int tap = 0; tap < axis.count[index]; ++tap)
    {
        axis.weights[tap * axis.size + index] = taps.weight<true>(taps.first + tap) / sum;
    }
}

template<class T, bool Planar>
__global__ void HorizontalAA(View src, int dstWidth, float *scratch, AxisCoefficients coefficients)
{
    constexpr int kColumns = 32;
    constexpr int kRows    = 8;
    int           column   = blockIdx.x * kColumns + threadIdx.x;
    int           row      = blockIdx.y * (kRows * 4) + threadIdx.y;
    int           channels = Planar ? 1 : src.c;
    int           rowSize  = dstWidth * channels;
    if (column >= rowSize || row >= src.h)
    {
        return;
    }
    int                  sample     = Planar ? blockIdx.z / src.c : blockIdx.z;
    int                  channel    = Planar ? blockIdx.z % src.c : column % channels;
    int                  coordinate = column / channels;
    const unsigned char *input      = src.ptr(sample, row, coefficients.first[coordinate], channel);
    uint64_t             offset     = (static_cast<uint64_t>(blockIdx.z) * src.h + row) * rowSize + column;
    float                value[kRows]{};
    if constexpr (Planar)
    {
#pragma unroll 4
        for (int tap = 0; tap < coefficients.count[coordinate]; ++tap)
        {
            float weight = coefficients.weights[tap * dstWidth + coordinate];
#pragma unroll
            for (int item = 0; item < kRows; ++item)
            {
                if (row + item * 4 < src.h)
                {
                    value[item]
                        += weight * ToFloat(*reinterpret_cast<const T *>(input + item * 4 * src.sy + tap * sizeof(T)));
                }
            }
        }
    }
    else
    {
        // Stepping the input pointer avoids per-tap runtime pixel-stride multiplies, which spilled NHWC to 144 registers.
        uint64_t     step    = src.sx;
        uint64_t     pitch   = 4 * src.sy;
        int          rows    = min(kRows, (src.h - row + 3) / 4);
        int          count   = coefficients.count[coordinate];
        const float *weights = coefficients.weights + coordinate;
        for (int tap = 0; tap < count; ++tap, input += step, weights += dstWidth)
        {
            float weight = *weights;
            if (rows == kRows)
            {
#pragma unroll
                for (int item = 0; item < kRows; ++item)
                {
                    value[item] += weight * ToFloat(*reinterpret_cast<const T *>(input + item * pitch));
                }
            }
            else
            {
#pragma unroll
                for (int item = 0; item < kRows; ++item)
                {
                    if (item < rows)
                    {
                        value[item] += weight * ToFloat(*reinterpret_cast<const T *>(input + item * pitch));
                    }
                }
            }
        }
    }
#pragma unroll
    for (int item = 0; item < kRows; ++item)
    {
        if (row + item * 4 < src.h)
        {
            scratch[offset + static_cast<uint64_t>(item * 4) * rowSize] = value[item];
        }
    }
}

template<class T>
struct alignas(sizeof(T) * 4) PixelVector
{
    T values[4];
};

template<class T, bool Planar, bool Packed>
__global__ void VerticalAA(const float *scratch, int srcHeight, View dst, AxisCoefficients coefficients)
{
    constexpr int kColumns = 4;
    int           row      = blockIdx.y;
    uint64_t      column = static_cast<uint64_t>(blockIdx.x) * (128 * kColumns) + threadIdx.x * (Packed ? kColumns : 1);
    int           channels = Planar ? 1 : dst.c;
    int           rowSize  = dst.w * channels;
    if (column >= static_cast<uint64_t>(rowSize))
    {
        return;
    }
    uint64_t offset = (static_cast<uint64_t>(blockIdx.z) * srcHeight + coefficients.first[row]) * rowSize + column;
    float    value[kColumns]{};
    for (int tap = 0; tap < coefficients.count[row]; ++tap)
    {
        float weight = coefficients.weights[tap * dst.h + row];
        if constexpr (Packed)
        {
            float4 pixels = *reinterpret_cast<const float4 *>(scratch + offset + static_cast<uint64_t>(tap) * rowSize);
            value[0] += weight * pixels.x;
            value[1] += weight * pixels.y;
            value[2] += weight * pixels.z;
            value[3] += weight * pixels.w;
        }
        else
        {
#pragma unroll
            for (int item = 0; item < kColumns; ++item)
            {
                if (column + item * 128 < static_cast<uint64_t>(rowSize))
                {
                    value[item] += weight * scratch[offset + static_cast<uint64_t>(tap) * rowSize + item * 128];
                }
            }
        }
    }
    int sample = Planar ? blockIdx.z / dst.c : blockIdx.z;
    if constexpr (Packed)
    {
        PixelVector<T> pixels;
#pragma unroll
        for (int item = 0; item < kColumns; ++item)
        {
            pixels.values[item] = StoreValue<T>(value[item]);
        }
        int channel = Planar ? blockIdx.z % dst.c : column % channels;
        *reinterpret_cast<PixelVector<T> *>(dst.ptr(sample, row, static_cast<int>(column / channels), channel))
            = pixels;
    }
    else
    {
#pragma unroll
        for (int item = 0; item < kColumns; ++item)
        {
            uint64_t scalar = column + item * 128;
            if (scalar < static_cast<uint64_t>(rowSize))
            {
                int channel = Planar ? blockIdx.z % dst.c : scalar % channels;
                *reinterpret_cast<T *>(dst.ptr(sample, row, static_cast<int>(scalar / channels), channel))
                    = StoreValue<T>(value[item]);
            }
        }
    }
}

constexpr int kFusedColumns   = 128;
constexpr int kFusedRowGroups = 2;
constexpr int kFusedBytes     = 64 * 1024;

// Single-pass interleaved AA: each block filters the source rows its output tile needs horizontally into fp32 shared
// memory, then vertically. Per-output tap order and FMAs match HorizontalAA/VerticalAA, so results are identical
// without the fp32 scratch round trip.
template<class T>
__global__ void FusedAA(View src, View dst, AxisCoefficients horizontal, AxisCoefficients vertical, int tileRows)
{
    extern __shared__ float rows[];
    constexpr int           kItems  = 4;
    int                     rowSize = dst.w * src.c;
    int                     column  = blockIdx.x * kFusedColumns + threadIdx.x;
    int                     oy0     = blockIdx.y * tileRows;
    int                     oy1     = min(oy0 + tileRows, dst.h);
    int                     sy0     = vertical.first[oy0];
    int                     span    = vertical.first[oy1 - 1] + vertical.count[oy1 - 1] - sy0;
    int                     channel = column % src.c;
    int                     x       = column / src.c;
    bool                    active  = column < rowSize;
    if (active)
    {
        int                  count  = horizontal.count[x];
        const float         *weight = horizontal.weights + x;
        const unsigned char *input  = src.ptr(blockIdx.z, sy0, horizontal.first[x], channel);
        uint64_t             step   = kFusedRowGroups * src.sy;
        for (int r = threadIdx.y; r < span; r += kFusedRowGroups * kItems)
        {
            float                value[kItems]{};
            const unsigned char *pixel = input + r * src.sy;
            // Full row groups drop the per-item guard, which otherwise forces an address rebuild per predicated load.
            if (r + (kItems - 1) * kFusedRowGroups < span)
            {
                for (int tap = 0; tap < count; ++tap, pixel += src.sx)
                {
                    float w = weight[tap * dst.w];
#pragma unroll
                    for (int item = 0; item < kItems; ++item)
                    {
                        value[item]
                            = __fmaf_rn(w, ToFloat(*reinterpret_cast<const T *>(pixel + item * step)), value[item]);
                    }
                }
            }
            else
            {
                for (int tap = 0; tap < count; ++tap, pixel += src.sx)
                {
                    float w = weight[tap * dst.w];
#pragma unroll
                    for (int item = 0; item < kItems; ++item)
                    {
                        if (r + item * kFusedRowGroups < span)
                        {
                            value[item]
                                = __fmaf_rn(w, ToFloat(*reinterpret_cast<const T *>(pixel + item * step)), value[item]);
                        }
                    }
                }
            }
#pragma unroll
            for (int item = 0; item < kItems; ++item)
            {
                int row = r + item * kFusedRowGroups;
                if (row < span)
                {
                    rows[row * kFusedColumns + threadIdx.x] = value[item];
                }
            }
        }
    }
    __syncthreads();
    if (!active)
    {
        return;
    }
    for (int y = oy0 + threadIdx.y; y < oy1; y += kFusedRowGroups)
    {
        const float *tile  = rows + (vertical.first[y] - sy0) * kFusedColumns + threadIdx.x;
        int          count = vertical.count[y];
        float        value = 0.0f;
        for (int tap = 0; tap < count; ++tap)
        {
            value = __fmaf_rn(vertical.weights[tap * dst.h + y], tile[tap * kFusedColumns], value);
        }
        *reinterpret_cast<T *>(dst.ptr(blockIdx.z, y, x, channel)) = StoreValue<T>(value);
    }
}

// Conservative bound on the source rows spanned by `outputs` consecutive AA outputs (see MakeTaps).
static int FusedSpan(int outputs, int sourceSize, int destinationSize, bool cubic)
{
    double scale   = static_cast<double>(sourceSize) / destinationSize;
    double support = (cubic ? 2.0 : 1.0) * std::max(1.0, scale);
    return static_cast<int>(std::min<double>(sourceSize, std::ceil((outputs - 1) * scale + 2 * support) + 8));
}

static int MultiprocessorCount()
{
    constexpr int           kDevices = 64;
    static std::atomic<int> counts[kDevices]{};
    int                     device = 0;
    NVCV_CHECK_THROW(cudaGetDevice(&device));
    int count = device < kDevices ? counts[device].load(std::memory_order_relaxed) : 0;
    if (count == 0)
    {
        NVCV_CHECK_THROW(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        if (device < kDevices)
            counts[device].store(count, std::memory_order_relaxed);
    }
    return count;
}

// Integrated GPUs only: skipping the scratch round trip wins where DRAM is shared LPDDR (Jetson Orin), while on
// discrete A100/H100 the fused kernel's per-block serialization loses to the two-pass path under locked clocks.
static bool IntegratedDevice()
{
    constexpr int           kDevices = 64;
    static std::atomic<int> flags[kDevices]{};
    int                     device = 0;
    NVCV_CHECK_THROW(cudaGetDevice(&device));
    int flag = device < kDevices ? flags[device].load(std::memory_order_relaxed) : 0;
    if (flag == 0)
    {
        int integrated = 0;
        NVCV_CHECK_THROW(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device));
        flag = integrated ? 1 : 2;
        if (device < kDevices)
            flags[device].store(flag, std::memory_order_relaxed);
    }
    return flag == 1;
}

// Interleaved linear or cubic AA with horizontal ratio <= 3.5 and enough blocks to hide the per-block serialization
// between the two passes.
template<class T, bool Planar>
static bool TryFusedAA(cudaStream_t stream, View src, View dst, AxisCoefficients horizontal, AxisCoefficients vertical,
                       bool cubic, int planes)
{
    if (Planar || !IntegratedDevice() || 2 * static_cast<int64_t>(src.w) > 7 * static_cast<int64_t>(dst.w))
    {
        return false;
    }
    auto span = [&](int tile)
    {
        return FusedSpan(tile, src.h, dst.h, cubic) * kFusedColumns * sizeof(float);
    };
    int tileRows = 32;
    while (tileRows > 1 && span(tileRows) > kFusedBytes)
    {
        tileRows /= 2;
    }
    // Small batches: shorter tiles until the grid covers 8 blocks per SM, trading halo rows for occupancy.
    int      sms     = MultiprocessorCount();
    int      columns = (dst.w * src.c + kFusedColumns - 1) / kFusedColumns;
    uint64_t target  = 8ull * sms;
    while (tileRows > 8 && static_cast<uint64_t>(columns) * ((dst.h + tileRows - 1) / tileRows) * planes < target)
    {
        tileRows /= 2;
    }
    size_t bytes = span(tileRows);
    if (bytes > kFusedBytes)
    {
        return false;
    }
    // The opt-in is per device context, so it cannot be cached process-wide.
    if (bytes > 48 * 1024
        && cudaFuncSetAttribute(FusedAA<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedBytes) != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    dim3 grid(columns, (dst.h + tileRows - 1) / tileRows, planes);
    if (static_cast<uint64_t>(grid.x) * grid.y * grid.z < 2ull * sms)
    {
        return false;
    }
    FusedAA<T><<<grid, dim3(kFusedColumns, kFusedRowGroups), bytes, stream>>>(src, dst, horizontal, vertical, tileRows);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

template<class T, bool Planar>
__global__ void VerticalAAPairedPacked(const float *scratch, int srcHeight, View dst, AxisCoefficients coefficients)
{
    constexpr int kColumns = 4;
    int           row0     = blockIdx.y * 2;
    int           row1     = row0 + 1;
    uint64_t      column   = static_cast<uint64_t>(blockIdx.x) * (128 * kColumns) + threadIdx.x * kColumns;
    int           channels = Planar ? 1 : dst.c;
    int           rowSize  = dst.w * channels;
    if (column >= static_cast<uint64_t>(rowSize))
    {
        return;
    }

    int first0 = coefficients.first[row0];
    int end0   = first0 + coefficients.count[row0];
    int first1 = row1 < dst.h ? coefficients.first[row1] : end0;
    int end1   = row1 < dst.h ? first1 + coefficients.count[row1] : end0;
    int first  = min(first0, first1);
    int end    = max(end0, end1);

    float    value0[kColumns]{};
    float    value1[kColumns]{};
    uint64_t offset = (static_cast<uint64_t>(blockIdx.z) * srcHeight + first) * rowSize + column;
    for (int tap = first; tap < end; ++tap, offset += rowSize)
    {
        float4 pixels = *reinterpret_cast<const float4 *>(scratch + offset);
        if (tap >= first0 && tap < end0)
        {
            float weight = coefficients.weights[(tap - first0) * dst.h + row0];
            value0[0] += weight * pixels.x;
            value0[1] += weight * pixels.y;
            value0[2] += weight * pixels.z;
            value0[3] += weight * pixels.w;
        }
        if (tap >= first1 && tap < end1)
        {
            float weight = coefficients.weights[(tap - first1) * dst.h + row1];
            value1[0] += weight * pixels.x;
            value1[1] += weight * pixels.y;
            value1[2] += weight * pixels.z;
            value1[3] += weight * pixels.w;
        }
    }

    int            sample  = Planar ? blockIdx.z / dst.c : blockIdx.z;
    int            channel = Planar ? blockIdx.z % dst.c : column % channels;
    PixelVector<T> pixels0;
#pragma unroll
    for (int item = 0; item < kColumns; ++item)
    {
        pixels0.values[item] = StoreValue<T>(value0[item]);
    }
    *reinterpret_cast<PixelVector<T> *>(dst.ptr(sample, row0, static_cast<int>(column / channels), channel)) = pixels0;
    if (row1 < dst.h)
    {
        PixelVector<T> pixels1;
#pragma unroll
        for (int item = 0; item < kColumns; ++item)
        {
            pixels1.values[item] = StoreValue<T>(value1[item]);
        }
        *reinterpret_cast<PixelVector<T> *>(dst.ptr(sample, row1, static_cast<int>(column / channels), channel))
            = pixels1;
    }
}

template<class T, bool Planar>
static void LaunchAA(cudaStream_t stream, View src, View dst, float *scratch, bool cubic)
{
    auto horizontal = Coefficients(scratch + Elements(src.n, src.h, dst.w, src.c), dst.w);
    auto vertical   = Coefficients(reinterpret_cast<float *>(horizontal.count + dst.w), dst.h);
    if (cubic)
    {
        PrepareAACoefficients<true><<<(dst.w + dst.h + 255) / 256, 256, 0, stream>>>(src, dst, horizontal, vertical);
    }
    else
    {
        PrepareAACoefficients<false><<<(dst.w + dst.h + 255) / 256, 256, 0, stream>>>(src, dst, horizontal, vertical);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
    int rowSize = dst.w * (Planar ? 1 : src.c);
    int planes  = src.n * (Planar ? src.c : 1);
    if (TryFusedAA<T, Planar>(stream, src, dst, horizontal, vertical, cubic, planes))
    {
        return;
    }
    dim3 horizontalGrid((static_cast<uint64_t>(rowSize) + 31) / 32, (src.h + 31) / 32, planes);
    HorizontalAA<T, Planar><<<horizontalGrid, dim3(32, 4), 0, stream>>>(src, dst.w, scratch, horizontal);
    NVCV_CHECK_THROW(cudaGetLastError());
    dim3 verticalGrid((static_cast<uint64_t>(rowSize) + 511) / 512, dst.h, planes);
    bool contiguous = Planar ? dst.sx == sizeof(T) : dst.sc == sizeof(T) && dst.sx == dst.c * sizeof(T);
    if (contiguous && reinterpret_cast<uintptr_t>(scratch) % alignof(float4) == 0 && rowSize % 4 == 0
        && reinterpret_cast<uintptr_t>(dst.data) % alignof(PixelVector<T>) == 0 && dst.sy % alignof(PixelVector<T>) == 0
        && dst.sn % alignof(PixelVector<T>) == 0 && (!Planar || dst.sc % alignof(PixelVector<T>) == 0))
    {
        if (src.h > dst.h && dst.h > 1)
        {
            verticalGrid.y = (dst.h + 1) / 2;
            VerticalAAPairedPacked<T, Planar><<<verticalGrid, 128, 0, stream>>>(scratch, src.h, dst, vertical);
        }
        else
        {
            VerticalAA<T, Planar, true><<<verticalGrid, 128, 0, stream>>>(scratch, src.h, dst, vertical);
        }
    }
    else
    {
        VerticalAA<T, Planar, false><<<verticalGrid, 128, 0, stream>>>(scratch, src.h, dst, vertical);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

static int CubicThreads(int width)
{
    if (width <= 32)
        return 32;
    if (width <= 64)
        return 64;
    if (width <= 128)
        return 128;
    return 256;
}

static bool NarrowRows(const View &view)
{
    return int64_t{view.w} * view.c <= std::numeric_limits<int>::max();
}

static bool VectorPixels(const View &view)
{
    return view.c == 4 && reinterpret_cast<uintptr_t>(view.data) % 4 == 0 && view.sy % 4 == 0 && view.sn % 4 == 0;
}

static void LaunchCubicInterleavedU8(cudaStream_t stream, View src, View dst)
{
    int  threads = CubicThreads(dst.w);
    int  xBlocks = static_cast<int>((static_cast<uint64_t>(dst.w) + threads - 1) / threads);
    dim3 grid(xBlocks,
              std::min<int64_t>(dst.h, std::max<int64_t>(1, 2048 / std::max<int64_t>(1, int64_t{xBlocks} * dst.n))),
              src.n);
    if (src.c == 1)
        CubicInterleavedU8<1, false><<<grid, threads, 0, stream>>>(src, dst);
    else if (src.c == 2)
        CubicInterleavedU8<2, false><<<grid, threads, 0, stream>>>(src, dst);
    else if (src.c == 3)
        CubicInterleavedU8<3, false><<<grid, threads, 0, stream>>>(src, dst);
    else if (VectorPixels(src) && VectorPixels(dst))
        CubicInterleavedU8<4, true><<<grid, threads, 0, stream>>>(src, dst);
    else
        CubicInterleavedU8<4, false><<<grid, threads, 0, stream>>>(src, dst);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<class T>
static void LaunchCubicInterleavedT(cudaStream_t stream, View src, View dst)
{
    int  threads = CubicThreads(dst.w);
    int  xBlocks = static_cast<int>((static_cast<uint64_t>(dst.w) + threads - 1) / threads);
    dim3 grid(xBlocks,
              std::min<int64_t>(dst.h, std::max<int64_t>(1, 2048 / std::max<int64_t>(1, int64_t{xBlocks} * dst.n))),
              src.n);
    bool reuse = int64_t{dst.h} * dst.w >= 720 * 1280;
    if (src.c == 3)
    {
        if (reuse || (src.h < dst.h && src.w < dst.w))
            CubicInterleavedT<T, 3, true><<<grid, threads, 0, stream>>>(src, dst);
        else
            CubicInterleavedT<T, 3><<<grid, threads, 0, stream>>>(src, dst);
    }
    else
    {
        if (reuse)
            CubicInterleavedT<T, 4, true><<<grid, threads, 0, stream>>>(src, dst);
        else
            CubicInterleavedT<T, 4><<<grid, threads, 0, stream>>>(src, dst);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<class T>
static bool LaunchCubicBatchT(cudaStream_t stream, const std::vector<ViewPair> &images)
{
    if (images[0].src.c < 3 || (std::is_same_v<T, float> && images[0].src.c == 4))
    {
        return false;
    }
    for (size_t first = 0; first < images.size(); first += kBatchPacketCapacity)
    {
        int         count = static_cast<int>(std::min<size_t>(kBatchPacketCapacity, images.size() - first));
        BatchPacket packet{};
        int         width  = 0;
        int         height = 0;
        for (int sample = 0; sample < count; ++sample)
        {
            packet.images[sample]   = images[first + sample];
            const auto &destination = packet.images[sample].dst;
            width                   = std::max(width, destination.w);
            height                  = std::max(height, destination.h);
        }
        {
            int  threads = CubicThreads(width);
            int  columns = static_cast<int>((static_cast<uint64_t>(width) + threads - 1) / threads);
            dim3 grid(columns, std::min<int64_t>(height, std::max<int64_t>(1, 2048 / (int64_t{columns} * count))),
                      count);
            int  channels = images[first].src.c;
            bool reuse    = int64_t{height} * width >= 720 * 1280;
            if (channels == 3)
            {
                if (reuse)
                    CubicInterleavedBatchT<T, 3, true><<<grid, threads, 0, stream>>>(packet);
                else
                    CubicInterleavedBatchT<T, 3, false><<<grid, threads, 0, stream>>>(packet);
            }
            else
            {
                if (reuse)
                    CubicInterleavedBatchT<T, 4, true><<<grid, threads, 0, stream>>>(packet);
                else
                    CubicInterleavedBatchT<T, 4, false><<<grid, threads, 0, stream>>>(packet);
            }
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    return true;
}

static bool LaunchBatch(cudaStream_t stream, const std::vector<ViewPair> &images, nvcv::DataType dtype,
                        NVCVInterpolationType interpolation, bool antialias)
{
    bool nearest = IsNearest(interpolation);
    if (images.size() < 2 || (!nearest && (interpolation != NVCV_INTERP_CUBIC || antialias)))
    {
        return false;
    }
    if (!nearest
        && std::any_of(images.begin(), images.end(),
                       [](const ViewPair &image)
                       {
                           return image.src.h == image.dst.h || image.src.w == image.dst.w || !NarrowRows(image.src)
                               || !NarrowRows(image.dst);
                       }))
    {
        return false;
    }
    if (nearest)
    {
        for (size_t first = 0; first < images.size(); first += kBatchPacketCapacity)
        {
            int                count = static_cast<int>(std::min<size_t>(kBatchPacketCapacity, images.size() - first));
            NearestBatchPacket packet{};
            packet.channels = images[first].src.c;
            int height = 0, width = 0;
            for (int sample = 0; sample < count; ++sample)
            {
                const auto &image     = images[first + sample];
                packet.images[sample] = {image.src.data, image.dst.data, image.src.sy, image.dst.sy,
                                         image.src.h,    image.src.w,    image.dst.h,  image.dst.w};
                height                = std::max(height, image.dst.h);
                width                 = std::max(width, image.dst.w);
            }
            dim3 grid = GatherGrid(height, width, count);
            dim3 block(kGatherWidth, kGatherRows);
            bool exact       = interpolation == NVCV_INTERP_NEAREST;
            int  scalarBytes = ScalarBytes(dtype);
            if (scalarBytes == 1)
                GatherBatch<uint8_t><<<grid, block, 0, stream>>>(packet, exact);
            else if (scalarBytes == 2)
                GatherBatch<uint16_t><<<grid, block, 0, stream>>>(packet, exact);
            else
                GatherBatch<uint32_t><<<grid, block, 0, stream>>>(packet, exact);
            NVCV_CHECK_THROW(cudaGetLastError());
        }
        return true;
    }
    if (dtype == nvcv::TYPE_U16)
        return LaunchCubicBatchT<uint16_t>(stream, images);
    if (dtype == nvcv::TYPE_S16)
        return LaunchCubicBatchT<int16_t>(stream, images);
    if (dtype == nvcv::TYPE_F16)
        return LaunchCubicBatchT<__half>(stream, images);
    if (dtype == nvcv::TYPE_F32)
        return LaunchCubicBatchT<float>(stream, images);
    for (size_t first = 0; first < images.size(); first += kBatchPacketCapacity)
    {
        int         count = static_cast<int>(std::min<size_t>(kBatchPacketCapacity, images.size() - first));
        BatchPacket packet{};
        int         width  = 0;
        int         height = 0;
        for (int sample = 0; sample < count; ++sample)
        {
            packet.images[sample]   = images[first + sample];
            const auto &destination = packet.images[sample].dst;
            width                   = std::max(width, destination.w);
            height                  = std::max(height, destination.h);
        }
        {
            int  threads = CubicThreads(width);
            int  columns = static_cast<int>((static_cast<uint64_t>(width) + threads - 1) / threads);
            dim3 grid(columns, std::min<int64_t>(height, std::max<int64_t>(1, 2048 / (int64_t{columns} * count))),
                      count);
            int  channels = images[first].src.c;
            bool vector   = true;
            for (int sample = 0; sample < count; ++sample)
                vector = vector && VectorPixels(packet.images[sample].src) && VectorPixels(packet.images[sample].dst);
            if (channels == 1)
                CubicInterleavedBatch<1, false><<<grid, threads, 0, stream>>>(packet);
            else if (channels == 3)
                CubicInterleavedBatch<3, false><<<grid, threads, 0, stream>>>(packet);
            else if (vector)
                CubicInterleavedBatch<4, true><<<grid, threads, 0, stream>>>(packet);
            else
                CubicInterleavedBatch<4, false><<<grid, threads, 0, stream>>>(packet);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    return true;
}

template<class T, bool Cubic>
__device__ float FixedHorizontal(View src, int sample, int row, int column, int channel, bool identity,
                                 const int (&indices)[Cubic ? 4 : 2], const float (&weights)[Cubic ? 4 : 2])
{
    if (identity)
    {
        return ToFloat(*reinterpret_cast<const T *>(src.ptr(sample, row, column, channel)));
    }
    float value = 0.0f;
#pragma unroll
    for (int tap = 0; tap < (Cubic ? 4 : 2); ++tap)
    {
        value += weights[tap] * ToFloat(*reinterpret_cast<const T *>(src.ptr(sample, row, indices[tap], channel)));
    }
    return value;
}

template<class T, bool Cubic, bool Planar>
__global__ void FilterNonAA(View src, View dst)
{
    uint64_t scalar   = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int      channels = Planar ? 1 : dst.c;
    if (scalar >= static_cast<uint64_t>(dst.w) * channels)
    {
        return;
    }
    int   sample  = Planar ? blockIdx.z / dst.c : blockIdx.z;
    int   channel = Planar ? blockIdx.z % dst.c : scalar % channels;
    int   column  = scalar / channels;
    int   indices[Cubic ? 4 : 2];
    float weights[Cubic ? 4 : 2];
    bool  identityWidth = src.w == dst.w;
    if (!identityWidth)
    {
        Taps horizontal = MakeTaps(column, src.w, dst.w, Cubic, false);
#pragma unroll
        for (int tap = 0; tap < (Cubic ? 4 : 2); ++tap)
        {
            indices[tap] = ClampIndex(horizontal.first + tap, src.w);
            weights[tap] = horizontal.weight(horizontal.first + tap);
        }
    }
    for (int64_t row = blockIdx.y; row < dst.h; row += gridDim.y)
    {
        float value = 0.0f;
        if (src.h == dst.h)
        {
            value = FixedHorizontal<T, Cubic>(src, sample, row, column, channel, identityWidth, indices, weights);
        }
        else
        {
            Taps vertical = MakeTaps(row, src.h, dst.h, Cubic, false);
#pragma unroll
            for (int tap = 0; tap < (Cubic ? 4 : 2); ++tap)
            {
                float horizontal = FixedHorizontal<T, Cubic>(src, sample, ClampIndex(vertical.first + tap, src.h),
                                                             column, channel, identityWidth, indices, weights);
                value += vertical.weight(vertical.first + tap) * horizontal;
            }
        }
        *reinterpret_cast<T *>(dst.ptr(sample, row, column, channel)) = StoreValue<T>(value);
    }
}

template<class T, int V>
struct alignas(sizeof(T) * V) Pack
{
    T values[V];
};

// Planar resize of both axes: V adjacent outputs per thread share one packed store and 32-bit row offsets; each
// output uses the exact FilterNonAA arithmetic (same taps, ascending horizontal-then-vertical fp32 sums).
template<class T, bool Cubic, int V>
__global__ void FilterPlanarNonAA(View src, View dst, bool packed)
{
    constexpr int K      = Cubic ? 4 : 2;
    int           column = (blockIdx.x * blockDim.x + threadIdx.x) * V;
    if (column >= dst.w)
    {
        return;
    }
    int   sample  = blockIdx.z / dst.c;
    int   channel = blockIdx.z % dst.c;
    int   offsets[V][K];
    float weights[V][K];
#pragma unroll
    for (int v = 0; v < V; ++v)
    {
        Taps horizontal = MakeTaps(min(column + v, dst.w - 1), src.w, dst.w, Cubic, false);
#pragma unroll
        for (int tap = 0; tap < K; ++tap)
        {
            offsets[v][tap] = ClampIndex(horizontal.first + tap, src.w) * static_cast<int>(sizeof(T));
            weights[v][tap] = horizontal.weight(horizontal.first + tap);
        }
    }
    const unsigned char *in    = src.data + sample * src.sn + channel * src.sc;
    unsigned char       *out   = dst.data + sample * dst.sn + channel * dst.sc + column * sizeof(T);
    bool                 whole = packed && column + V <= dst.w;
    for (int row = blockIdx.y; row < dst.h; row += gridDim.y)
    {
        Taps  vertical = MakeTaps(row, src.h, dst.h, Cubic, false);
        float value[V];
#pragma unroll
        for (int v = 0; v < V; ++v)
        {
            value[v] = 0.0f;
        }
#pragma unroll
        for (int tap = 0; tap < K; ++tap)
        {
            const unsigned char *line   = in + ClampIndex(vertical.first + tap, src.h) * src.sy;
            float                weight = vertical.weight(vertical.first + tap);
#pragma unroll
            for (int v = 0; v < V; ++v)
            {
                float horizontal = 0.0f;
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
                    horizontal += weights[v][k] * ToFloat(*reinterpret_cast<const T *>(line + offsets[v][k]));
                }
                value[v] += weight * horizontal;
            }
        }
        unsigned char *pixel = out + row * dst.sy;
        if (whole)
        {
            Pack<T, V> pack;
#pragma unroll
            for (int v = 0; v < V; ++v)
            {
                pack.values[v] = StoreValue<T>(value[v]);
            }
            *reinterpret_cast<Pack<T, V> *>(pixel) = pack;
        }
        else
        {
#pragma unroll
            for (int v = 0; v < V; ++v)
            {
                if (column + v < dst.w)
                {
                    reinterpret_cast<T *>(pixel)[v] = StoreValue<T>(value[v]);
                }
            }
        }
    }
}

template<int Channels, bool Vector>
__device__ __forceinline__ void LinearRow(const uint8_t *in, bool identity, const uint32_t (&offsets)[2],
                                          const float (&weights)[2], float (&value)[Channels])
{
    float first[Channels];
    LoadPixel<Channels, Vector>(in + offsets[0], first);
    if (identity)
    {
#pragma unroll
        for (int c = 0; c < Channels; ++c) value[c] = first[c];
        return;
    }
    float second[Channels];
    LoadPixel<Channels, Vector>(in + offsets[1], second);
#pragma unroll
    for (int c = 0; c < Channels; ++c)
    {
        float h = 0.0f;
        h += weights[0] * first[c];
        h += weights[1] * second[c];
        value[c] = h;
    }
}

// Callers guarantee w * Channels fits in int, so column byte offsets stay 32-bit.
template<int Channels, bool Vector>
__global__ __launch_bounds__(256) void LinearInterleavedU8(View src, View dst)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= dst.w)
        return;

    const uint8_t *source      = src.data + blockIdx.z * src.sn;
    uint8_t       *destination = dst.data + blockIdx.z * dst.sn + static_cast<uint32_t>(column) * Channels;
    uint32_t       offsets[2]  = {static_cast<uint32_t>(column) * Channels, 0};
    float          weights[2];
    bool           identityWidth = src.w == dst.w;
    if (!identityWidth)
    {
        Taps horizontal = MakeTaps(column, src.w, dst.w, false, false);
#pragma unroll
        for (int tap = 0; tap < 2; ++tap)
        {
            offsets[tap] = static_cast<uint32_t>(ClampIndex(horizontal.first + tap, src.w)) * Channels;
            weights[tap] = horizontal.weight(horizontal.first + tap);
        }
    }
    for (int64_t row = blockIdx.y; row < dst.h; row += gridDim.y)
    {
        float value[Channels];
        if (src.h == dst.h)
        {
            LinearRow<Channels, Vector>(source + row * src.sy, identityWidth, offsets, weights, value);
        }
        else
        {
            Taps vertical = MakeTaps(row, src.h, dst.h, false, false);
#pragma unroll
            for (int c = 0; c < Channels; ++c) value[c] = 0.0f;
#pragma unroll
            for (int tap = 0; tap < 2; ++tap)
            {
                float h[Channels];
                LinearRow<Channels, Vector>(source + ClampIndex(vertical.first + tap, src.h) * src.sy, identityWidth,
                                            offsets, weights, h);
                float weight = vertical.weight(vertical.first + tap);
#pragma unroll
                for (int c = 0; c < Channels; ++c) value[c] += weight * h[c];
            }
        }
        StorePixel<Channels, Vector>(destination + row * dst.sy, value);
    }
}

template<class T, int Channels>
__global__ void LinearInterleavedT(View src, View dst)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= dst.w)
        return;

    int   indices[2];
    float weights[2];
    bool  identityWidth = src.w == dst.w;
    if (!identityWidth)
    {
        Taps horizontal = MakeTaps(column, src.w, dst.w, false, false);
#pragma unroll
        for (int tap = 0; tap < 2; ++tap)
        {
            indices[tap] = ClampIndex(horizontal.first + tap, src.w);
            weights[tap] = horizontal.weight(horizontal.first + tap);
        }
    }
    for (int64_t row = blockIdx.y; row < dst.h; row += gridDim.y)
    {
        int   verticalIndices[2];
        float verticalWeights[2];
        if (src.h != dst.h)
        {
            Taps vertical = MakeTaps(row, src.h, dst.h, false, false);
#pragma unroll
            for (int tap = 0; tap < 2; ++tap)
            {
                verticalIndices[tap] = ClampIndex(vertical.first + tap, src.h);
                verticalWeights[tap] = vertical.weight(vertical.first + tap);
            }
        }
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel)
        {
            float value = 0.0f;
            if (src.h == dst.h)
            {
                value
                    = FixedHorizontal<T, false>(src, blockIdx.z, row, column, channel, identityWidth, indices, weights);
            }
            else
            {
#pragma unroll
                for (int tap = 0; tap < 2; ++tap)
                {
                    float h = FixedHorizontal<T, false>(src, blockIdx.z, verticalIndices[tap], column, channel,
                                                        identityWidth, indices, weights);
                    value += verticalWeights[tap] * h;
                }
            }
            *reinterpret_cast<T *>(dst.ptr(blockIdx.z, row, column, channel)) = StoreValue<T>(value);
        }
    }
}

template<class T, bool Planar>
static bool LaunchNonAA(cudaStream_t stream, View src, View dst, bool cubic)
{
    uint64_t planes  = static_cast<uint64_t>(src.n) * (Planar ? src.c : 1);
    uint64_t rowSize = static_cast<uint64_t>(dst.w) * (Planar ? 1 : dst.c);
    if (planes > 65535 || rowSize > std::numeric_limits<int>::max())
    {
        return false;
    }
    // Scaling both planar axes packs V outputs per thread; one-axis identities keep the direct-read path.
    constexpr int V = sizeof(T) >= 4 ? 2 : 4 / static_cast<int>(sizeof(T));
    // Fewer than 4 planes cannot fill the GPU with V-fold fewer threads.
    bool          vectored = Planar && planes >= 4 && src.w != dst.w && src.h != dst.h
                 && static_cast<uint64_t>(src.w) * sizeof(T) <= static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (vectored)
    {
        rowSize = (rowSize + V - 1) / V;
    }
    int      threads      = rowSize <= 32 ? 32 : (rowSize <= 64 ? 64 : (rowSize <= 128 ? 128 : 256));
    int      columns      = (rowSize + threads - 1) / threads;
    uint64_t totalBlocks  = static_cast<uint64_t>(columns) * planes;
    uint64_t targetBlocks = 2048;
    if constexpr (std::is_same_v<T, uint8_t> && Planar)
        if (cubic && src.h > dst.h && src.w > dst.w)
            targetBlocks = 4096; // Protects NCHW uchar3 cubic 1080p->720p and 4K->1080p downscales (3-15 planes).
    // Large batched/planar launches starve the SMs at 2048 blocks; per-image VarShape fallbacks (planes < 16) do not.
    if (planes >= 16 && static_cast<uint64_t>(dst.h) >= (8192 + totalBlocks - 1) / totalBlocks)
        targetBlocks = 8192;
    dim3 grid(columns, std::min<uint64_t>(dst.h, std::max<uint64_t>(1, targetBlocks / totalBlocks)), planes);
    if (vectored)
    {
        uint64_t alignment = sizeof(Pack<T, V>);
        bool     packed    = reinterpret_cast<uintptr_t>(dst.data) % alignment == 0 && dst.sy % alignment == 0
                   && dst.sn % alignment == 0 && dst.sc % alignment == 0;
        if (cubic)
            FilterPlanarNonAA<T, true, V><<<grid, threads, 0, stream>>>(src, dst, packed);
        else
            FilterPlanarNonAA<T, false, V><<<grid, threads, 0, stream>>>(src, dst, packed);
    }
    else if (cubic)
    {
        FilterNonAA<T, true, Planar><<<grid, threads, 0, stream>>>(src, dst);
    }
    else
    {
        FilterNonAA<T, false, Planar><<<grid, threads, 0, stream>>>(src, dst);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

template<class T>
void Filter(cudaStream_t stream, View src, View dst, float *scratch, NVCVInterpolationType interpolation, bool aa)
{
    if (!aa)
    {
        bool planar   = src.sx == sizeof(T) && dst.sx == sizeof(T);
        bool launched = planar ? LaunchNonAA<T, true>(stream, src, dst, interpolation == NVCV_INTERP_CUBIC)
                               : LaunchNonAA<T, false>(stream, src, dst, interpolation == NVCV_INTERP_CUBIC);
        if (launched)
        {
            return;
        }
    }
    if (aa && CanUseAACoefficients(src.h, src.w, dst.h, dst.w, interpolation == NVCV_INTERP_CUBIC))
    {
        if (src.sx == sizeof(T) && dst.sx == sizeof(T))
        {
            LaunchAA<T, true>(stream, src, dst, scratch, interpolation == NVCV_INTERP_CUBIC);
        }
        else
        {
            LaunchAA<T, false>(stream, src, dst, scratch, interpolation == NVCV_INTERP_CUBIC);
        }
        return;
    }
    uint64_t      intermediate = Elements(src.n, src.h, dst.w, src.c);
    uint64_t      output       = Elements(dst.n, dst.h, dst.w, dst.c);
    bool          cubic        = interpolation == NVCV_INTERP_CUBIC;
    // Eight-row AA threads need half a block per SM to beat one row per thread; tiny outputs (e.g. 1x1) fall short.
    constexpr int kRows     = 8;
    uint64_t      rowGroups = Elements(src.n, (src.h - 1) / kRows + 1, dst.w, src.c);
    // Planar 4-byte scalars and planar 2-byte linear read too few taps per row for the reuse to beat the lost parallelism.
    bool          planarSrc = src.sx == sizeof(T);
    bool          rowReuse  = !(planarSrc && (sizeof(T) == 4 || (sizeof(T) == 2 && !cubic)));
    if (aa && rowGroups >= 128ull * MultiprocessorCount() && rowReuse)
    {
        Horizontal<T, kRows><<<Blocks(rowGroups), 256, 0, stream>>>(src, dst.w, scratch, rowGroups, cubic, aa);
    }
    else
    {
        HorizontalScalar<T><<<Blocks(intermediate), 256, 0, stream>>>(src, dst.w, scratch, intermediate, cubic, aa);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
    if (aa && output >= 512)
    {
        constexpr int kItems = 4;
        uint64_t      chunks = (static_cast<uint64_t>(dst.w) * dst.c + 32 * kItems - 1) / (32 * kItems);
        uint64_t      count  = CheckedMul(Elements(dst.n, dst.h, 1, 1), chunks * 32);
        Vertical<T, kItems><<<Blocks(count), 256, 0, stream>>>(scratch, src.h, dst, count, cubic, aa);
    }
    else
    {
        Vertical<T, 1><<<Blocks(output), 256, 0, stream>>>(scratch, src.h, dst, output, cubic, aa);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

// Non-U8 per-pixel kernels share taps across channels; they only gain with 3+ channels, and F32 C4 cubic is already
// bandwidth-bound per scalar.
template<class T>
void FilterInterleaved(cudaStream_t stream, View src, View dst, float *scratch, NVCVInterpolationType interpolation,
                       bool aa)
{
    bool packed = !aa && src.sc == sizeof(T) && dst.sc == sizeof(T) && src.sx == src.c * sizeof(T)
               && dst.sx == dst.c * sizeof(T) && src.c <= 4 && src.n <= 65535 && src.c >= 3;
    // F32 linear loads a full scalar per tap, so sharing taps across channels saves too little to beat the per-scalar
    // kernel's parallelism on H100.
    if (packed && interpolation == NVCV_INTERP_LINEAR
        && !std::is_same_v<T, float> && static_cast<uint64_t>(dst.w) * dst.h >= 256 * 256)
    {
        uint64_t columns = (static_cast<uint64_t>(dst.w) + 255) / 256;
        dim3     grid(columns, std::min<uint64_t>(dst.h, std::max<uint64_t>(1, 2048 / (columns * src.n))), src.n);
        if (src.c == 3)
            LinearInterleavedT<T, 3><<<grid, 256, 0, stream>>>(src, dst);
        else
            LinearInterleavedT<T, 4><<<grid, 256, 0, stream>>>(src, dst);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else if (packed && interpolation == NVCV_INTERP_CUBIC && src.h != dst.h && src.w != dst.w
             && !(std::is_same_v<T, float> && src.c == 4))
    {
        LaunchCubicInterleavedT<T>(stream, src, dst);
    }
    else
    {
        Filter<T>(stream, src, dst, scratch, interpolation, aa);
    }
}

static void Run(cudaStream_t stream, View src, View dst, nvcv::DataType dtype, const Workspace &workspace,
                NVCVInterpolationType interpolation, bool aa)
{
    if (src.n == 0)
    {
        return;
    }
    bool identity = src.h == dst.h && src.w == dst.w;
    if (identity || IsNearest(interpolation))
    {
        dim3 grid = GatherGrid(dst.h, dst.w, (dst.n + kGatherSamples - 1) / kGatherSamples);
        dim3 block(kGatherWidth, kGatherRows);
        bool exact = interpolation == NVCV_INTERP_NEAREST;
        int  bytes = ScalarBytes(dtype);
        if (bytes == 1)
        {
            Gather<uint8_t><<<grid, block, 0, stream>>>(src, dst, exact, identity);
        }
        else if (bytes == 2)
        {
            Gather<uint16_t><<<grid, block, 0, stream>>>(src, dst, exact, identity);
        }
        else
        {
            Gather<uint32_t><<<grid, block, 0, stream>>>(src, dst, exact, identity);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
        return;
    }
    auto *scratch = static_cast<float *>(workspace.cudaMem.data);
    if (dtype == nvcv::TYPE_U8)
    {
        if (interpolation == NVCV_INTERP_LINEAR && !aa && src.sc == 1 && dst.sc == 1
            && src.sx == static_cast<uint64_t>(src.c) && dst.sx == static_cast<uint64_t>(dst.c)
            && (src.c == 3 || src.c == 4) && static_cast<uint64_t>(dst.w) * dst.h >= 256 * 256 && NarrowRows(src)
            && NarrowRows(dst))
        {
            uint64_t columns = (static_cast<uint64_t>(dst.w) + 255) / 256;
            dim3     grid(columns, std::min<uint64_t>(dst.h, std::max<uint64_t>(1, 2048 / (columns * src.n))), src.n);
            if (src.c == 3)
                LinearInterleavedU8<3, false><<<grid, 256, 0, stream>>>(src, dst);
            else if (VectorPixels(src) && VectorPixels(dst))
                LinearInterleavedU8<4, true><<<grid, 256, 0, stream>>>(src, dst);
            else
                LinearInterleavedU8<4, false><<<grid, 256, 0, stream>>>(src, dst);
            NVCV_CHECK_THROW(cudaGetLastError());
        }
        else if (interpolation == NVCV_INTERP_CUBIC && !aa && src.h != dst.h && src.w != dst.w && src.sc == 1
                 && dst.sc == 1 && src.sx == static_cast<uint64_t>(src.c) && dst.sx == static_cast<uint64_t>(dst.c)
                 && src.c <= 4 && NarrowRows(src) && NarrowRows(dst))
        {
            LaunchCubicInterleavedU8(stream, src, dst);
        }
        else
        {
            Filter<uint8_t>(stream, src, dst, scratch, interpolation, aa);
        }
    }
    else if (dtype == nvcv::TYPE_U16)
    {
        FilterInterleaved<uint16_t>(stream, src, dst, scratch, interpolation, aa);
    }
    else if (dtype == nvcv::TYPE_S16)
    {
        FilterInterleaved<int16_t>(stream, src, dst, scratch, interpolation, aa);
    }
    else if (dtype == nvcv::TYPE_F16)
    {
        FilterInterleaved<__half>(stream, src, dst, scratch, interpolation, aa);
    }
    else
    {
        FilterInterleaved<float>(stream, src, dst, scratch, interpolation, aa);
    }
}

} // namespace

WorkspaceRequirements TvResize::getWorkspaceRequirements(int batchSize, int32_t srcHeight, int32_t srcWidth,
                                                         int32_t dstHeight, int32_t dstWidth, int32_t numChannels,
                                                         NVCVInterpolationType interpolation, bool antialias) const
{
    ValidateMode(interpolation);
    return Requirements(
        ScratchSize(batchSize, srcHeight, srcWidth, dstHeight, dstWidth, numChannels, interpolation, antialias));
}

WorkspaceRequirements TvResize::getWorkspaceRequirements(int batchSize, const int32_t *srcHeights,
                                                         const int32_t *srcWidths, const int32_t *dstHeights,
                                                         const int32_t *dstWidths, int32_t numChannels,
                                                         NVCVInterpolationType interpolation, bool antialias) const
{
    ValidateMode(interpolation);
    ValidateCount(batchSize);
    PositiveExtent(numChannels);
    Require(numChannels == 1 || numChannels == 3 || numChannels == 4, "VarShape requires 1, 3 or 4 channels");
    Require(srcHeights && srcWidths && dstHeights && dstWidths, "Extent arrays must not be null");
    size_t bytes = 0;
    for (int n = 0; n < batchSize; ++n)
    {
        bytes = std::max(bytes, ScratchSize(1, srcHeights[n], srcWidths[n], dstHeights[n], dstWidths[n], numChannels,
                                            interpolation, antialias));
    }
    return Requirements(bytes);
}

void TvResize::operator()(cudaStream_t stream, const Workspace &workspace, const nvcv::Tensor &in,
                          const nvcv::Tensor &out, NVCVInterpolationType interpolation, bool antialias) const
{
    CVCUDA_NVTX_RANGE("cvcuda::TvResize::operator()[Tensor]");
    ValidateMode(interpolation);
    auto inData  = in.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    Require(inData && outData, "Tensors must be CUDA-accessible and pitch-linear");
    Require(inData->dtype() == outData->dtype(), "Input and output scalar types must match");
    Require(inData->layout() == outData->layout(), "Input and output layouts must match");
    View src = TensorView(*inData);
    View dst = TensorView(*outData);
    Require(src.n == dst.n && src.c == dst.c, "Input and output sample and channel counts must match");
    size_t bytes = ScratchSize(src.n, src.h, src.w, dst.h, dst.w, src.c, interpolation, antialias);
    ValidateWorkspace(workspace, bytes);
    BeginWorkspace(stream, workspace, bytes);
    Run(stream, src, dst, inData->dtype(), workspace, interpolation, antialias);
    EndWorkspace(stream, workspace, bytes);
}

void TvResize::operator()(cudaStream_t stream, const Workspace &workspace, const nvcv::ImageBatchVarShape &in,
                          const nvcv::ImageBatchVarShape &out, NVCVInterpolationType interpolation,
                          bool antialias) const
{
    CVCUDA_NVTX_RANGE("cvcuda::TvResize::operator()[ImageBatchVarShape]");
    ValidateMode(interpolation);
    int count = in.numImages();
    ValidateCount(count);
    ValidateCount(out.numImages());
    Require(count == out.numImages(), "Input and output image counts must match");
    if (count == 0)
    {
        return;
    }
    auto format = in.uniqueFormat();
    Require(format != nvcv::FMT_NONE && out.uniqueFormat() == format, "VarShape formats must be uniform and matching");
    auto dtype       = ImageType(format);
    int  scalarBytes = ScalarBytes(dtype);

    std::vector<ViewPair> images;
    images.reserve(count);
    size_t bytes = 0;
    // Export individual host descriptors: batch export may enqueue a descriptor upload before validation.
    for (int n = 0; n < count; ++n)
    {
        View src = ImageView(in[n], format, scalarBytes);
        View dst = ImageView(out[n], format, scalarBytes);
        bytes    = std::max(bytes, ScratchSize(1, src.h, src.w, dst.h, dst.w, src.c, interpolation, antialias));
        images.push_back({src, dst});
    }
    ValidateWorkspace(workspace, bytes);
    BeginWorkspace(stream, workspace, bytes);
    if (!LaunchBatch(stream, images, dtype, interpolation, antialias))
    {
        for (const auto &image : images)
        {
            Run(stream, image.src, image.dst, dtype, workspace, interpolation, antialias);
        }
    }
    EndWorkspace(stream, workspace, bytes);
}

} // namespace cvcuda::priv

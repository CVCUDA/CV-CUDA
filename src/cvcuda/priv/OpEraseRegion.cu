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
#include "OpErase.hpp"

#include <cuda_fp16.h>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/util/CheckError.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cvcuda::priv {

namespace {

struct ImageTensorDesc
{
    const unsigned char *src;
    unsigned char       *dst;
    int64_t              shape[4];
    int64_t              srcStride[4];
    int64_t              dstStride[4];
};

struct ValueTensorDesc
{
    const unsigned char *base;
    int64_t              stride[4];
};

struct Region
{
    int64_t top;
    int64_t left;
    int64_t height;
    int64_t width;
};

template<typename DstT, typename ValueT>
__device__ __forceinline__ DstT ConvertValue(ValueT value)
{
    return static_cast<DstT>(value);
}

template<>
__device__ __forceinline__ __half ConvertValue<__half, float>(float value)
{
    return __float2half_rn(value);
}

__device__ __forceinline__ int32_t TorchFloatToS32(float value)
{
    if (isnan(value))
    {
        return 0;
    }
    if (value >= 0x1p31f)
    {
        return 0x7fffffff;
    }
    if (value <= -0x1p31f)
    {
        return -0x7fffffff - 1;
    }
    return static_cast<int32_t>(value);
}

__device__ __forceinline__ uint32_t TorchFloatToU32(float value)
{
    if (isnan(value) || value <= 0.0f)
    {
        return 0;
    }
    if (value >= 0x1p32f)
    {
        return 0xffffffffU;
    }
    return static_cast<uint32_t>(value);
}

__device__ __forceinline__ int64_t TorchFloatToS64(float value)
{
    if (isnan(value) || value <= -0x1p63f)
    {
        return -0x7fffffffffffffffLL - 1;
    }
    if (value >= 0x1p63f)
    {
        return 0x7fffffffffffffffLL;
    }
    return static_cast<int64_t>(value);
}

__device__ __forceinline__ uint64_t TorchFloatToU64(float value)
{
    if (isnan(value))
    {
        return uint64_t{1} << 63;
    }
    if (value <= 0.0f)
    {
        return 0;
    }
    if (value >= 0x1p64f)
    {
        return 0xffffffffffffffffULL;
    }
    return static_cast<uint64_t>(value);
}

template<>
__device__ __forceinline__ uint8_t ConvertValue<uint8_t, float>(float value)
{
    return static_cast<uint8_t>(TorchFloatToS64(value));
}

template<>
__device__ __forceinline__ int8_t ConvertValue<int8_t, float>(float value)
{
    return static_cast<int8_t>(TorchFloatToS32(value));
}

template<>
__device__ __forceinline__ uint16_t ConvertValue<uint16_t, float>(float value)
{
    return static_cast<uint16_t>(TorchFloatToU32(value));
}

template<>
__device__ __forceinline__ int16_t ConvertValue<int16_t, float>(float value)
{
    return static_cast<int16_t>(TorchFloatToS32(value));
}

template<>
__device__ __forceinline__ uint32_t ConvertValue<uint32_t, float>(float value)
{
    return TorchFloatToU32(value);
}

template<>
__device__ __forceinline__ int32_t ConvertValue<int32_t, float>(float value)
{
    return TorchFloatToS32(value);
}

template<>
__device__ __forceinline__ uint64_t ConvertValue<uint64_t, float>(float value)
{
    return TorchFloatToU64(value);
}

template<>
__device__ __forceinline__ int64_t ConvertValue<int64_t, float>(float value)
{
    return TorchFloatToS64(value);
}

template<typename T>
__device__ __forceinline__ T LoadAt(const unsigned char *base, int64_t offset)
{
    return *reinterpret_cast<const T *>(base + offset);
}

template<typename T>
__device__ __forceinline__ void StoreAt(unsigned char *base, int64_t offset, T value)
{
    *reinterpret_cast<T *>(base + offset) = value;
}

__device__ __forceinline__ int64_t ImageOffset(const int64_t *stride, int64_t n, int64_t c, int64_t y, int64_t x)
{
    return n * stride[0] + c * stride[1] + y * stride[2] + x * stride[3];
}

__device__ __forceinline__ int64_t ValueOffset(const ValueTensorDesc &value, int64_t n, int64_t c, int64_t y, int64_t x)
{
    return n * value.stride[0] + c * value.stride[1] + y * value.stride[2] + x * value.stride[3];
}

template<typename DstT, typename ValueT>
__global__ void EraseRegionOutOfPlace(ImageTensorDesc image, ValueTensorDesc value, Region region, int64_t total)
{
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int64_t       remaining = index;
        const int64_t x         = remaining % image.shape[3];
        remaining /= image.shape[3];
        const int64_t y = remaining % image.shape[2];
        remaining /= image.shape[2];
        const int64_t c = remaining % image.shape[1];
        const int64_t n = remaining / image.shape[1];

        const int64_t dstOffset = ImageOffset(image.dstStride, n, c, y, x);
        if (y >= region.top && y < region.top + region.height && x >= region.left && x < region.left + region.width)
        {
            const int64_t valueOffset = ValueOffset(value, n, c, y - region.top, x - region.left);
            StoreAt<DstT>(image.dst, dstOffset, ConvertValue<DstT>(LoadAt<ValueT>(value.base, valueOffset)));
        }
        else
        {
            const int64_t srcOffset = ImageOffset(image.srcStride, n, c, y, x);
            StoreAt<DstT>(image.dst, dstOffset, LoadAt<DstT>(image.src, srcOffset));
        }
    }
}

// Region fill. The sample and the region row come straight from the grid, so no region extent
// is ever a divisor: the predecessor kernel recovered all four coordinates from a flat index
// with three 64-bit divisions and three 64-bit modulos per element, and profiled at 0.52% of
// DRAM peak against 85% SM throughput -- it was issue-bound on that addressing, not on memory.
//
// `Broadcast` is the shape torchvision reaches with a scalar or per-channel fill value (value
// strides along the region's y and x are zero). It hoists the float-to-destination conversion
// -- which for the integer types is torchvision's saturating 64-bit cast -- out of the element
// loop entirely, so each thread converts at most four values instead of one per store.
//
// The `values[4]` array is only ever indexed by a fully unrolled loop counter, so every index
// is a compile-time constant and the array stays in registers; a runtime index here would spill
// the whole array to local memory.
template<typename DstT, typename ValueT, int Depth, bool Broadcast>
__global__ void EraseRegionFill(unsigned char *__restrict__ dst, int64_t dstS0, int64_t dstS1, int64_t dstS2,
                                int64_t dstS3, ValueTensorDesc value, int top, int left, int height, int width,
                                int channels, int samples)
{
    for (int n = blockIdx.z; n < samples; n += gridDim.z)
        for (int y = blockIdx.y; y < height; y += gridDim.y)
        {
            const int64_t dRow = n * dstS0 + static_cast<int64_t>(top + y) * dstS2 + static_cast<int64_t>(left) * dstS3;
            const int64_t vRow
                = n * value.stride[0] + (Broadcast ? int64_t{0} : static_cast<int64_t>(y) * value.stride[2]);

            DstT values[4];
            if constexpr (Broadcast)
            {
                // Unrolled so every index into `values` is a compile-time constant; a runtime index
                // would push the array to local memory.
#pragma unroll
                for (int c = 0; c < 4; ++c)
                {
                    if (c < channels)
                    {
                        values[c] = ConvertValue<DstT>(LoadAt<ValueT>(value.base, vRow + c * value.stride[1]));
                    }
                }
            }

            const int x0 = blockIdx.x * (blockDim.x * Depth) + threadIdx.x;
#pragma unroll
            for (int d = 0; d < Depth; ++d)
            {
                const int x = x0 + d * blockDim.x;
                if (x >= width)
                {
                    continue;
                }
                const int64_t dOff = dRow + static_cast<int64_t>(x) * dstS3;
                if constexpr (Broadcast)
                {
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        if (c < channels)
                        {
                            StoreAt<DstT>(dst, dOff + c * dstS1, values[c]);
                        }
                    }
                }
                else
                {
                    // Unrolled over the maximum channel count rather than the runtime one: a
                    // runtime trip count leaves the loop rolled and costs more than the predicated
                    // stores it saves (measured on the single-channel rows, where it dominates).
                    const int64_t vOff = vRow + static_cast<int64_t>(x) * value.stride[3];
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        if (c < channels)
                        {
                            StoreAt<DstT>(dst, dOff + c * dstS1,
                                          ConvertValue<DstT>(LoadAt<ValueT>(value.base, vOff + c * value.stride[1])));
                        }
                    }
                }
            }
        }
}

// Byte-described copy of the untouched image. The destination is described as `nY * nZ`
// contiguous runs of `runBytes`, so the widest carrier that divides every base, stride and run
// length moves the payload; no pixel, channel or region arithmetic survives into the kernel.
template<typename Carrier, int Depth>
__global__ void EraseCopyRuns(const unsigned char *__restrict__ src, unsigned char *__restrict__ dst, int64_t srcY,
                              int64_t dstY, int64_t srcZ, int64_t dstZ, int64_t carriers)
{
    const int64_t  sBase = blockIdx.y * srcY + blockIdx.z * srcZ;
    const int64_t  dBase = blockIdx.y * dstY + blockIdx.z * dstZ;
    const Carrier *s     = reinterpret_cast<const Carrier *>(src + sBase);
    Carrier       *d     = reinterpret_cast<Carrier *>(dst + dBase);

    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x * Depth + threadIdx.x;
#pragma unroll
    for (int k = 0; k < Depth; ++k)
    {
        const int64_t index = start + static_cast<int64_t>(k) * blockDim.x;
        if (index < carriers)
        {
            d[index] = s[index];
        }
    }
}

static int64_t SaturatingAdd(int64_t a, int64_t b)
{
    if (b > 0 && a > std::numeric_limits<int64_t>::max() - b)
    {
        return std::numeric_limits<int64_t>::max();
    }
    if (b < 0 && a < std::numeric_limits<int64_t>::min() - b)
    {
        return std::numeric_limits<int64_t>::min();
    }
    return a + b;
}

static int64_t NormalizeSliceIndex(int64_t index, int64_t size)
{
    if (index < 0)
    {
        if (index < -size)
        {
            return 0;
        }
        return index + size;
    }
    return std::min(index, size);
}

static Region NormalizeRegion(int64_t i, int64_t j, int64_t h, int64_t w, int64_t imageHeight, int64_t imageWidth)
{
    const int64_t top    = NormalizeSliceIndex(i, imageHeight);
    const int64_t bottom = NormalizeSliceIndex(SaturatingAdd(i, h), imageHeight);
    const int64_t left   = NormalizeSliceIndex(j, imageWidth);
    const int64_t right  = NormalizeSliceIndex(SaturatingAdd(j, w), imageWidth);

    return Region{top, left, std::max<int64_t>(bottom - top, 0), std::max<int64_t>(right - left, 0)};
}

static bool IsSupportedScalarType(nvcv::DataType dtype)
{
    return dtype == nvcv::TYPE_U8 || dtype == nvcv::TYPE_S8 || dtype == nvcv::TYPE_U16 || dtype == nvcv::TYPE_S16
        || dtype == nvcv::TYPE_U32 || dtype == nvcv::TYPE_S32 || dtype == nvcv::TYPE_U64 || dtype == nvcv::TYPE_S64
        || dtype == nvcv::TYPE_F16 || dtype == nvcv::TYPE_F32 || dtype == nvcv::TYPE_F64;
}

static ImageTensorDesc MakeImageDesc(const nvcv::TensorDataStridedCuda &input,
                                     const nvcv::TensorDataStridedCuda &output)
{
    ImageTensorDesc desc{};
    desc.src = reinterpret_cast<const unsigned char *>(input.basePtr());
    desc.dst = reinterpret_cast<unsigned char *>(output.basePtr());

    const nvcv::TensorLayout layout = input.layout();
    if (layout == nvcv::TENSOR_NHWC)
    {
        desc.shape[0]     = input.shape(0);
        desc.shape[1]     = input.shape(3);
        desc.shape[2]     = input.shape(1);
        desc.shape[3]     = input.shape(2);
        desc.srcStride[0] = input.stride(0);
        desc.srcStride[1] = input.stride(3);
        desc.srcStride[2] = input.stride(1);
        desc.srcStride[3] = input.stride(2);
        desc.dstStride[0] = output.stride(0);
        desc.dstStride[1] = output.stride(3);
        desc.dstStride[2] = output.stride(1);
        desc.dstStride[3] = output.stride(2);
    }
    else if (layout == nvcv::TENSOR_HWC)
    {
        desc.shape[0]     = 1;
        desc.shape[1]     = input.shape(2);
        desc.shape[2]     = input.shape(0);
        desc.shape[3]     = input.shape(1);
        desc.srcStride[0] = 0;
        desc.srcStride[1] = input.stride(2);
        desc.srcStride[2] = input.stride(0);
        desc.srcStride[3] = input.stride(1);
        desc.dstStride[0] = 0;
        desc.dstStride[1] = output.stride(2);
        desc.dstStride[2] = output.stride(0);
        desc.dstStride[3] = output.stride(1);
    }
    else if (layout == nvcv::TENSOR_NCHW)
    {
        desc.shape[0] = input.shape(0);
        desc.shape[1] = input.shape(1);
        desc.shape[2] = input.shape(2);
        desc.shape[3] = input.shape(3);
        for (int k = 0; k < 4; ++k)
        {
            desc.srcStride[k] = input.stride(k);
            desc.dstStride[k] = output.stride(k);
        }
    }
    else if (layout == nvcv::TENSOR_CHW)
    {
        desc.shape[0]     = 1;
        desc.shape[1]     = input.shape(0);
        desc.shape[2]     = input.shape(1);
        desc.shape[3]     = input.shape(2);
        desc.srcStride[0] = 0;
        desc.srcStride[1] = input.stride(0);
        desc.srcStride[2] = input.stride(1);
        desc.srcStride[3] = input.stride(2);
        desc.dstStride[0] = 0;
        desc.dstStride[1] = output.stride(0);
        desc.dstStride[2] = output.stride(1);
        desc.dstStride[3] = output.stride(2);
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Erase region supports only NHWC, HWC, NCHW, and CHW layouts");
    }

    return desc;
}

static ValueTensorDesc MakeValueDesc(const nvcv::TensorDataStridedCuda &values, const ImageTensorDesc &image,
                                     const Region &region)
{
    constexpr int targetRank = 4;
    if (values.rank() > targetRank)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Erase region value rank exceeds the logical image rank");
    }

    const int64_t   targetShape[4] = {image.shape[0], image.shape[1], region.height, region.width};
    ValueTensorDesc desc{};
    desc.base = reinterpret_cast<const unsigned char *>(values.basePtr());

    const int rank    = values.rank();
    const int leading = targetRank - rank;
    for (int valueDim = 0; valueDim < rank; ++valueDim)
    {
        const int     targetDim  = leading + valueDim;
        const int     logicalDim = targetDim;
        const int64_t extent     = values.shape(valueDim);
        if (extent != 1 && extent != targetShape[logicalDim])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Erase region value is not broadcastable to the selected image region");
        }
        desc.stride[logicalDim] = extent == 1 ? 0 : values.stride(valueDim);
    }
    return desc;
}

static int64_t CheckedProduct(int64_t a, int64_t b, int64_t c, int64_t d)
{
    int64_t result = 1;
    for (int64_t value : {a, b, c, d})
    {
        if (value == 0)
        {
            return 0;
        }
        if (value < 0 || result > std::numeric_limits<int64_t>::max() / value)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Erase region tensor size overflows int64_t");
        }
        result *= value;
    }
    return result;
}

// Description of the source/destination byte layout as `nY * nZ` contiguous runs of `runBytes`.
// Built by sorting the logical axes into physical (ascending-stride) order and merging every
// adjacent pair that is packed in both tensors, so a densely allocated tensor of any layout
// collapses to a single run and a row-padded one to a single outer level.
struct CopyPlan
{
    int64_t runBytes;
    int64_t srcY, dstY, srcZ, dstZ;
    int32_t nY, nZ;
    int     carrierBytes;
};

constexpr int kEraseCopyBlock = 256;
constexpr int kEraseCopyDepth = 4;
constexpr int kEraseFillBlock = 128;
constexpr int kEraseFillDepth = 4;

static bool BuildCopyPlan(const ImageTensorDesc &image, int64_t elemSize, CopyPlan &plan)
{
    struct Axis
    {
        int64_t extent, srcStride, dstStride;
    };

    Axis axes[4];
    int  count = 0;
    for (int k = 0; k < 4; ++k)
    {
        if (image.shape[k] > 1)
        {
            axes[count++] = Axis{image.shape[k], image.srcStride[k], image.dstStride[k]};
        }
    }
    // Insertion sort rather than std::sort: there are at most four axes, and std::sort's
    // __final_insertion_sort carries a `last - first > 16` branch that GCC cannot prove dead
    // against an Axis[4], so it reports a -Warray-bounds subscript at offset 16 and
    // -Werror all-warnings makes that fatal. A hand-rolled sort of four elements is also
    // strictly less code than the generic one it replaces.
    for (int i = 1; i < count; ++i)
    {
        const Axis key = axes[i];
        int        j   = i - 1;
        while (j >= 0 && key.srcStride < axes[j].srcStride)
        {
            axes[j + 1] = axes[j];
            --j;
        }
        axes[j + 1] = key;
    }

    int64_t runBytes = elemSize;
    int     axis     = 0;
    for (; axis < count && axes[axis].srcStride == runBytes && axes[axis].dstStride == runBytes; ++axis)
    {
        runBytes *= axes[axis].extent;
    }

    Axis outer[4];
    int  outerCount = 0;
    for (; axis < count; ++axis)
    {
        if (outerCount > 0 && axes[axis].srcStride == outer[outerCount - 1].srcStride * outer[outerCount - 1].extent
            && axes[axis].dstStride == outer[outerCount - 1].dstStride * outer[outerCount - 1].extent)
        {
            outer[outerCount - 1].extent *= axes[axis].extent;
        }
        else
        {
            outer[outerCount++] = axes[axis];
        }
    }
    if (outerCount > 2)
    {
        return false;
    }

    // Tested on the int64_t extents, before they are narrowed into the plan. Narrowing first
    // would let an extent past 2^31 wrap to a small positive value, pass this gate, and launch
    // a grid that silently covers a fraction of the image instead of declining to the fallback.
    for (int i = 0; i < outerCount; ++i)
    {
        if (outer[i].extent > 65535)
        {
            return false;
        }
    }

    plan          = CopyPlan{runBytes, 0, 0, 0, 0, 1, 1, 1};
    plan.runBytes = runBytes;
    if (outerCount > 0)
    {
        plan.nY   = static_cast<int32_t>(outer[0].extent);
        plan.srcY = outer[0].srcStride;
        plan.dstY = outer[0].dstStride;
    }
    if (outerCount > 1)
    {
        plan.nZ   = static_cast<int32_t>(outer[1].extent);
        plan.srcZ = outer[1].srcStride;
        plan.dstZ = outer[1].dstStride;
    }

    // Widest power-of-two carrier that divides every base pointer, every run length and every
    // outer stride. Everything is naturally aligned for tensors from the standard allocator, so
    // the 16-byte arm is the common case; the narrower arms keep exotic strides correct.
    const uintptr_t bits = reinterpret_cast<uintptr_t>(image.src) | reinterpret_cast<uintptr_t>(image.dst)
                         | static_cast<uintptr_t>(plan.runBytes) | static_cast<uintptr_t>(plan.srcY)
                         | static_cast<uintptr_t>(plan.dstY) | static_cast<uintptr_t>(plan.srcZ)
                         | static_cast<uintptr_t>(plan.dstZ);
    plan.carrierBytes = 16;
    while (plan.carrierBytes > 1 && (bits & (static_cast<uintptr_t>(plan.carrierBytes) - 1)) != 0)
    {
        plan.carrierBytes /= 2;
    }
    return true;
}

template<typename Carrier>
static void LaunchCopyRuns(cudaStream_t stream, const CopyPlan &plan, const ImageTensorDesc &image)
{
    const int64_t carriers = plan.runBytes / static_cast<int64_t>(sizeof(Carrier));
    const dim3    block(kEraseCopyBlock);
    const dim3    grid(
           static_cast<unsigned>(std::max<int64_t>(
            (carriers + int64_t(kEraseCopyBlock) * kEraseCopyDepth - 1) / (kEraseCopyBlock * kEraseCopyDepth), 1)),
           static_cast<unsigned>(plan.nY), static_cast<unsigned>(plan.nZ));
    EraseCopyRuns<Carrier, kEraseCopyDepth>
        <<<grid, block, 0, stream>>>(image.src, image.dst, plan.srcY, plan.dstY, plan.srcZ, plan.dstZ, carriers);
    NVCV_CHECK_THROW(cudaGetLastError());
}

static void LaunchCopy(cudaStream_t stream, const CopyPlan &plan, const ImageTensorDesc &image)
{
    switch (plan.carrierBytes)
    {
    case 16:
        LaunchCopyRuns<uint4>(stream, plan, image);
        break;
    case 8:
        LaunchCopyRuns<uint2>(stream, plan, image);
        break;
    case 4:
        LaunchCopyRuns<uint32_t>(stream, plan, image);
        break;
    case 2:
        LaunchCopyRuns<uint16_t>(stream, plan, image);
        break;
    default:
        LaunchCopyRuns<uint8_t>(stream, plan, image);
        break;
    }
}

template<typename DstT, typename ValueT>
void LaunchEraseRegion(cudaStream_t stream, const ImageTensorDesc &image, const ValueTensorDesc &value,
                       const Region &region, bool inplace, int64_t elemSize)
{
    const int64_t imageTotal = CheckedProduct(image.shape[0], image.shape[1], image.shape[2], image.shape[3]);
    if (imageTotal == 0)
    {
        return;
    }

    // EraseRegionFill takes the region and the image's sample/channel counts as `int`, and forms
    // `top + y` in `int` before widening. The public entry point accepts int64_t coordinates, so
    // an extent past INT_MAX would truncate here rather than being rejected. The fallback kernel
    // addresses entirely in int64_t, so hand those geometries to it instead of narrowing.
    const auto fitsInInt = [](int64_t v)
    {
        return v >= 0 && v <= std::numeric_limits<int>::max();
    };
    const bool fillFitsInInt = fitsInInt(region.top) && fitsInInt(region.left) && fitsInInt(region.height)
                            && fitsInInt(region.width) && fitsInInt(region.top + region.height)
                            && fitsInInt(image.shape[0]) && fitsInInt(image.shape[1]);

    CopyPlan plan{};
    if (!fillFitsInInt || (!inplace && !BuildCopyPlan(image, elemSize, plan)))
    {
        // Geometry the run description cannot express (more than two outer levels after
        // merging, or an outer extent past the grid limit), or a region the fill kernel's int
        // parameters cannot carry. Fall back to the general whole-image predicated kernel,
        // which is correct for any strides and addresses in int64_t throughout.
        constexpr int kBlockSize   = 256;
        const int64_t neededBlocks = (imageTotal + kBlockSize - 1) / kBlockSize;
        const int     blocks       = static_cast<int>(std::min<int64_t>(neededBlocks, 65535));
        EraseRegionOutOfPlace<DstT, ValueT><<<blocks, kBlockSize, 0, stream>>>(image, value, region, imageTotal);
        NVCV_CHECK_THROW(cudaGetLastError());
        return;
    }

    if (!inplace)
    {
        LaunchCopy(stream, plan, image);
    }

    if (region.height == 0 || region.width == 0)
    {
        return;
    }

    // A value tensor that does not vary across the region (a scalar or a per-channel fill, which
    // is what torchvision's own erase passes) lets the conversion be hoisted per thread.
    const bool broadcast = value.stride[2] == 0 && value.stride[3] == 0;

    const dim3 block(kEraseFillBlock, 1, 1);
    const dim3 grid(static_cast<unsigned>((region.width + int64_t{kEraseFillBlock} * kEraseFillDepth - 1)
                                          / (int64_t{kEraseFillBlock} * kEraseFillDepth)),
                    static_cast<unsigned>(std::min<int64_t>(region.height, 65535)),
                    static_cast<unsigned>(std::min<int64_t>(image.shape[0], 65535)));

    const auto launch = [&](auto broadcastTag)
    {
        EraseRegionFill<DstT, ValueT, kEraseFillDepth, decltype(broadcastTag)::value><<<grid, block, 0, stream>>>(
            image.dst, image.dstStride[0], image.dstStride[1], image.dstStride[2], image.dstStride[3], value,
            static_cast<int>(region.top), static_cast<int>(region.left), static_cast<int>(region.height),
            static_cast<int>(region.width), static_cast<int>(image.shape[1]), static_cast<int>(image.shape[0]));
    };
    if (broadcast)
    {
        launch(std::true_type{});
    }
    else
    {
        launch(std::false_type{});
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename DstT>
void DispatchValueType(cudaStream_t stream, const ImageTensorDesc &image, const ValueTensorDesc &value,
                       const Region &region, bool inplace, bool floatValue)
{
    if (floatValue)
    {
        LaunchEraseRegion<DstT, float>(stream, image, value, region, inplace, sizeof(DstT));
    }
    else
    {
        LaunchEraseRegion<DstT, DstT>(stream, image, value, region, inplace, sizeof(DstT));
    }
}

static void DispatchImageType(cudaStream_t stream, nvcv::DataType dtype, const ImageTensorDesc &image,
                              const ValueTensorDesc &value, const Region &region, bool inplace, bool floatValue)
{
    if (dtype == nvcv::TYPE_U8)
        DispatchValueType<uint8_t>(stream, image, value, region, inplace, floatValue);
    else if (dtype == nvcv::TYPE_S8)
        DispatchValueType<int8_t>(stream, image, value, region, inplace, floatValue);
    else if (dtype == nvcv::TYPE_U16)
        DispatchValueType<uint16_t>(stream, image, value, region, inplace, floatValue);
    else if (dtype == nvcv::TYPE_S16)
        DispatchValueType<int16_t>(stream, image, value, region, inplace, floatValue);
    else if (dtype == nvcv::TYPE_U32)
        DispatchValueType<uint32_t>(stream, image, value, region, inplace, floatValue);
    else if (dtype == nvcv::TYPE_S32)
        DispatchValueType<int32_t>(stream, image, value, region, inplace, floatValue);
    else if (dtype == nvcv::TYPE_U64)
        DispatchValueType<uint64_t>(stream, image, value, region, inplace, floatValue);
    else if (dtype == nvcv::TYPE_S64)
        DispatchValueType<int64_t>(stream, image, value, region, inplace, floatValue);
    else if (dtype == nvcv::TYPE_F16)
        DispatchValueType<__half>(stream, image, value, region, inplace, floatValue);
    else if (dtype == nvcv::TYPE_F32)
        DispatchValueType<float>(stream, image, value, region, inplace, floatValue);
    else if (dtype == nvcv::TYPE_F64)
        DispatchValueType<double>(stream, image, value, region, inplace, floatValue);
    else
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported Erase region dtype");
}

} // namespace

void Erase::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, int64_t i, int64_t j,
                       int64_t h, int64_t w, const nvcv::Tensor &values) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Erase::operator()[Region]");

    auto input  = in.exportData<nvcv::TensorDataStridedCuda>();
    auto output = out.exportData<nvcv::TensorDataStridedCuda>();
    auto value  = values.exportData<nvcv::TensorDataStridedCuda>();
    if (input == nullptr || output == nullptr || value == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Erase region inputs must be CUDA-accessible pitch-linear tensors");
    }
    if (input->rank() != output->rank() || input->layout() != output->layout() || input->dtype() != output->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Erase region input and output rank, layout, and dtype must match");
    }
    for (int dim = 0; dim < input->rank(); ++dim)
    {
        if (input->shape(dim) != output->shape(dim))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Erase region input and output shapes must match");
        }
    }
    if (!IsSupportedScalarType(input->dtype()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Erase region supports only real scalar CV-CUDA dtypes");
    }
    if (value->dtype() != input->dtype() && value->dtype() != nvcv::TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Erase region values must match the image dtype or use float32");
    }

    ImageTensorDesc image = MakeImageDesc(*input, *output);
    if (image.shape[1] < 1 || image.shape[1] > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Erase region supports image channel counts from 1 through 4");
    }

    Region          region    = NormalizeRegion(i, j, h, w, image.shape[2], image.shape[3]);
    ValueTensorDesc valueDesc = MakeValueDesc(*value, image, region);
    const bool      inplace   = in.handle() == out.handle();
    DispatchImageType(stream, input->dtype(), image, valueDesc, region, inplace, value->dtype() == nvcv::TYPE_F32);
}

} // namespace cvcuda::priv

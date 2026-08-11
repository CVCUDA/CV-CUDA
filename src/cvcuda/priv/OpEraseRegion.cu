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

template<typename DstT, typename ValueT>
__global__ void EraseRegionInPlace(ImageTensorDesc image, ValueTensorDesc value, Region region, int64_t total)
{
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int64_t       remaining = index;
        const int64_t x         = remaining % region.width;
        remaining /= region.width;
        const int64_t y = remaining % region.height;
        remaining /= region.height;
        const int64_t c = remaining % image.shape[1];
        const int64_t n = remaining / image.shape[1];

        const int64_t dstOffset   = ImageOffset(image.dstStride, n, c, y + region.top, x + region.left);
        const int64_t valueOffset = ValueOffset(value, n, c, y, x);
        StoreAt<DstT>(image.dst, dstOffset, ConvertValue<DstT>(LoadAt<ValueT>(value.base, valueOffset)));
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

template<typename DstT, typename ValueT>
void LaunchEraseRegion(cudaStream_t stream, const ImageTensorDesc &image, const ValueTensorDesc &value,
                       const Region &region, bool inplace)
{
    constexpr int kBlockSize = 256;
    const int64_t total      = inplace ? CheckedProduct(image.shape[0], image.shape[1], region.height, region.width)
                                       : CheckedProduct(image.shape[0], image.shape[1], image.shape[2], image.shape[3]);
    if (total == 0)
    {
        return;
    }

    const int64_t neededBlocks = (total + kBlockSize - 1) / kBlockSize;
    const int     blocks       = static_cast<int>(std::min<int64_t>(neededBlocks, 65535));
    if (inplace)
    {
        EraseRegionInPlace<DstT, ValueT><<<blocks, kBlockSize, 0, stream>>>(image, value, region, total);
    }
    else
    {
        EraseRegionOutOfPlace<DstT, ValueT><<<blocks, kBlockSize, 0, stream>>>(image, value, region, total);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename DstT>
void DispatchValueType(cudaStream_t stream, const ImageTensorDesc &image, const ValueTensorDesc &value,
                       const Region &region, bool inplace, bool floatValue)
{
    if (floatValue)
    {
        LaunchEraseRegion<DstT, float>(stream, image, value, region, inplace);
    }
    else
    {
        LaunchEraseRegion<DstT, DstT>(stream, image, value, region, inplace);
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

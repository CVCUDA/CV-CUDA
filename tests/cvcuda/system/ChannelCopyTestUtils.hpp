/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_TEST_CHANNEL_COPY_UTILS_HPP
#define CVCUDA_TEST_CHANNEL_COPY_UTILS_HPP

#include "Definitions.hpp"

#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/util/CheckError.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <tuple>
#include <vector>

namespace channeltest {

inline nvcv::TensorLayout OtherLayout(nvcv::TensorLayout layout)
{
    if (layout == nvcv::TENSOR_HWC)
        return nvcv::TENSOR_CHW;
    if (layout == nvcv::TENSOR_CHW)
        return nvcv::TENSOR_HWC;
    return layout == nvcv::TENSOR_NHWC ? nvcv::TENSOR_NCHW : nvcv::TENSOR_NHWC;
}

inline nvcv::TensorShape Shape(nvcv::TensorLayout layout, int samples, int rows, int columns, int channels)
{
    std::array<int64_t, 4> shape{};
    for (int dimension = 0; dimension < layout.rank(); ++dimension)
    {
        switch (layout[dimension])
        {
        case 'N':
            shape[dimension] = samples;
            break;
        case 'H':
            shape[dimension] = rows;
            break;
        case 'W':
            shape[dimension] = columns;
            break;
        case 'C':
            shape[dimension] = channels;
            break;
        default:
            break;
        }
    }
    return nvcv::TensorShape(shape.data(), layout.rank(), layout);
}

class Stream
{
public:
    Stream()
    {
        NVCV_CHECK_THROW(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking));
    }

    ~Stream()
    {
        cudaStreamDestroy(handle);
    }

    Stream(const Stream &)                 = delete;
    Stream      &operator=(const Stream &) = delete;
    cudaStream_t handle{};
};

class Buffer
{
public:
    Buffer(nvcv::TensorLayout layout, nvcv::DataType dtype, int samples, int rows, int columns, int channels,
           int padding = 32, size_t baseOffset = 64, uint8_t poison = 0xA5, int samplePadding = 0)
        : shape(Shape(layout, samples, rows, columns, channels))
        , bytes(dtype.strideBytes())
        , baseOffset(baseOffset)
        , poison(poison)
    {
        int64_t stride = bytes;
        for (int dimension = shape.rank() - 1; dimension >= 0; --dimension)
        {
            buffer.strides[dimension] = stride;
            stride *= shape[dimension];
            if (layout[dimension] == 'W' || layout[dimension] == 'H')
                stride += padding;
            if (dimension == 1 && shape.rank() == 4)
                stride += samplePadding;
        }
        host.assign(stride + baseOffset + 64, poison);
        NVCVByte *allocation = nullptr;
        NVCV_CHECK_THROW(cudaMalloc(reinterpret_cast<void **>(&allocation), host.size()));
        storage        = std::shared_ptr<NVCVByte>(allocation, [](NVCVByte *pointer) { cudaFree(pointer); });
        buffer.basePtr = allocation + baseOffset;
        tensor         = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(shape, dtype, buffer));
    }

    void CompactSingletonStrides()
    {
        for (int dimension = 0; dimension < shape.rank(); ++dimension)
            if (shape[dimension] == 1)
                buffer.strides[dimension] = bytes;
        tensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(shape, tensor.dtype(), buffer));
    }

    nvcv::Tensor View(size_t byteOffset) const
    {
        auto view = buffer;
        view.basePtr += byteOffset;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(shape, tensor.dtype(), view));
    }

    size_t Offset(int sample, int row, int column, int channel) const
    {
        size_t offset = baseOffset;
        for (int dimension = 0; dimension < shape.rank(); ++dimension)
        {
            int coordinate = 0;
            switch (shape.layout()[dimension])
            {
            case 'N':
                coordinate = sample;
                break;
            case 'H':
                coordinate = row;
                break;
            case 'W':
                coordinate = column;
                break;
            case 'C':
                coordinate = channel;
                break;
            default:
                break;
            }
            offset += coordinate * buffer.strides[dimension];
        }
        return offset;
    }

    void Put(const std::vector<uint8_t> &pixels, int samples, int rows, int columns, int channels,
             size_t byteOffset = 0)
    {
        size_t source = 0;
        for (int64_t sampleRow = 0; sampleRow < static_cast<int64_t>(samples) * rows; ++sampleRow)
        {
            const auto sample = static_cast<int>(sampleRow / rows);
            const auto row    = static_cast<int>(sampleRow % rows);
            for (int column = 0; column < columns; ++column)
                for (int channel = 0; channel < channels; ++channel)
                {
                    std::copy_n(pixels.begin() + source, bytes,
                                host.begin() + byteOffset + Offset(sample, row, column, channel));
                    source += bytes;
                }
        }
    }

    void Upload(cudaStream_t stream) const
    {
        NVCV_CHECK_THROW(cudaMemcpyAsync(storage.get(), host.data(), host.size(), cudaMemcpyHostToDevice, stream));
    }

    void Clear(cudaStream_t stream) const
    {
        NVCV_CHECK_THROW(cudaMemsetAsync(storage.get(), poison, host.size(), stream));
    }

    std::vector<uint8_t> Download() const
    {
        std::vector<uint8_t> result(host.size());
        NVCV_CHECK_THROW(cudaMemcpy(result.data(), storage.get(), result.size(), cudaMemcpyDeviceToHost));
        return result;
    }

    nvcv::TensorShape                   shape;
    int                                 bytes;
    size_t                              baseOffset;
    uint8_t                             poison;
    nvcv::TensorDataStridedCuda::Buffer buffer{};
    std::vector<uint8_t>                host;
    std::shared_ptr<NVCVByte>           storage;
    nvcv::Tensor                        tensor{nullptr};
};

inline std::vector<uint8_t> Patterns(size_t elements, int bytes, int seed = 0)
{
    const std::array<uint64_t, 16> patterns = {0,
                                               UINT64_MAX,
                                               0x8000000000000000ULL,
                                               0x7ff0000000000000ULL,
                                               0xfff0000000000000ULL,
                                               0x7ff8000000001234ULL,
                                               0x7ff0000000000042ULL,
                                               0x80000000ULL,
                                               0x7f800000ULL,
                                               0xff800000ULL,
                                               0x7fc01234ULL,
                                               0x7f800042ULL,
                                               0x8000ULL,
                                               0x7c00ULL,
                                               0xfc00ULL,
                                               0x7e42ULL};
    std::vector<uint8_t>           result(elements * bytes);
    for (size_t element = 0; element < elements; ++element)
    {
        const uint64_t value = (element % 3 == 0) ? patterns[(element + seed) % patterns.size()]
                                                  : (element + seed + 1) * 0x9e3779b97f4a7c15ULL;
        std::memcpy(result.data() + element * bytes, &value, bytes);
    }
    return result;
}

inline nvcv::ImageFormat ImageFormat(nvcv::DataType dtype, int channels, bool planar)
{
    if (planar && channels == 4 && dtype.strideBytes() == 1)
        return nvcv::FMT_RGBA8p.dataKind(dtype.dataKind());
    using Packing                                        = nvcv::Packing;
    const std::array<std::array<Packing, 4>, 4> packings = {
        {
         {Packing::X8, Packing::X8_Y8, Packing::X8_Y8_Z8, Packing::X8_Y8_Z8_W8},
         {Packing::X16, Packing::X16_Y16, Packing::X16_Y16_Z16, Packing::X16_Y16_Z16_W16},
         {Packing::X32, Packing::X32_Y32, Packing::X32_Y32_Z32, Packing::X32_Y32_Z32_W32},
         {Packing::X64, Packing::X64_Y64, Packing::X64_Y64_Z64, Packing::X64_Y64_Z64_W64},
         }
    };
    int width;
    switch (dtype.strideBytes())
    {
    case 8:
        width = 3;
        break;
    case 4:
        width = 2;
        break;
    default:
        width = dtype.strideBytes() - 1;
        break;
    }
    const std::array<nvcv::Swizzle, 4> swizzles
        = {nvcv::Swizzle::S_X000, nvcv::Swizzle::S_XY00, nvcv::Swizzle::S_XYZ0, nvcv::Swizzle::S_XYZW};
    std::array<Packing, 4> planes{Packing::NONE, Packing::NONE, Packing::NONE, Packing::NONE};
    for (int plane = 0; plane < (planar ? channels : 1); ++plane)
        planes[plane] = packings[width][planar ? 0 : channels - 1];
    return {nvcv::MemLayout::PITCH_LINEAR,
            dtype.dataKind(),
            swizzles[channels - 1],
            planes[0],
            planes[1],
            planes[2],
            planes[3]};
}

inline nvcv::ImageBatchVarShape Batch(std::initializer_list<nvcv::Image> images)
{
    nvcv::ImageBatchVarShape batch(std::max(1, static_cast<int>(images.size())));
    for (const auto &image : images) batch.pushBack(image);
    return batch;
}

class ImageBuffer
{
public:
    ImageBuffer(nvcv::ImageFormat format, int width, int height, int salt = 0, uint8_t poison = 0xA5)
        : format(format)
    {
        buffer.numPlanes = format.numPlanes();
        for (int plane = 0; plane < buffer.numPlanes; ++plane)
        {
            auto      &metadata  = buffer.planes[plane];
            const auto size      = format.planeSize({width, height}, plane);
            const int  alignment = std::max(1, format.planeDataType(plane).alignment());
            offsets.push_back((3 + salt + plane) * alignment);
            metadata.width     = size.w;
            metadata.height    = size.h;
            const int rowBytes = std::max((size.w * format.planeBitsPerPixel(plane) + 7) / 8,
                                          size.w * format.planePixelStrideBytes(plane));
            metadata.rowStride = ((rowBytes + alignment - 1) / alignment + 1 + salt + plane) * alignment;
            host.emplace_back(offsets.back() + metadata.rowStride * size.h + 64, poison);
            NVCVByte *allocation = nullptr;
            NVCV_CHECK_THROW(cudaMalloc(reinterpret_cast<void **>(&allocation), host.back().size()));
            storage.emplace_back(allocation, [](NVCVByte *pointer) { cudaFree(pointer); });
            metadata.basePtr = allocation + offsets.back();
        }
        image = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda(format, buffer));
    }

    size_t Offset(int row, int column, int channel) const
    {
        const int plane = buffer.numPlanes == 1 ? 0 : channel;
        const int bytes = format.bitsPerChannel()[0] / 8;
        return offsets[plane] + row * buffer.planes[plane].rowStride + column * format.planePixelStrideBytes(plane)
             + (buffer.numPlanes == 1 ? channel * bytes : 0);
    }

    void Put(const std::vector<uint8_t> &pixels, size_t byteOffset = 0)
    {
        const int bytes  = format.bitsPerChannel()[0] / 8;
        size_t    source = 0;
        for (int row = 0; row < buffer.planes[0].height; ++row)
            for (int column = 0; column < buffer.planes[0].width; ++column)
                for (int channel = 0; channel < format.numChannels(); ++channel)
                {
                    const int plane = buffer.numPlanes == 1 ? 0 : channel;
                    std::copy_n(pixels.begin() + source, bytes,
                                host[plane].begin() + byteOffset + Offset(row, column, channel));
                    source += bytes;
                }
    }

    void Upload(cudaStream_t stream) const
    {
        for (size_t plane = 0; plane < storage.size(); ++plane)
            NVCV_CHECK_THROW(cudaMemcpyAsync(storage[plane].get(), host[plane].data(), host[plane].size(),
                                             cudaMemcpyHostToDevice, stream));
    }

    std::vector<std::vector<uint8_t>> Download() const
    {
        auto result = host;
        for (size_t plane = 0; plane < storage.size(); ++plane)
            NVCV_CHECK_THROW(
                cudaMemcpy(result[plane].data(), storage[plane].get(), result[plane].size(), cudaMemcpyDeviceToHost));
        return result;
    }

    std::vector<uint8_t> Pixels() const
    {
        const auto           data  = Download();
        const int            bytes = format.bitsPerChannel()[0] / 8;
        std::vector<uint8_t> result;
        for (int row = 0; row < buffer.planes[0].height; ++row)
            for (int column = 0; column < buffer.planes[0].width; ++column)
                for (int channel = 0; channel < format.numChannels(); ++channel)
                {
                    const int  plane = buffer.numPlanes == 1 ? 0 : channel;
                    const auto start = data[plane].begin() + Offset(row, column, channel);
                    result.insert(result.end(), start, start + bytes);
                }
        return result;
    }

    nvcv::Image View(size_t byteOffset = 0, int rowDelta = 0) const
    {
        auto view = buffer;
        for (int plane = 0; plane < view.numPlanes; ++plane)
        {
            view.planes[plane].basePtr += byteOffset;
            view.planes[plane].rowStride += rowDelta;
        }
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda(format, view));
    }

    nvcv::ImageFormat                      format;
    nvcv::ImageDataStridedCuda::Buffer     buffer{};
    std::vector<size_t>                    offsets;
    std::vector<std::vector<uint8_t>>      host;
    std::vector<std::shared_ptr<NVCVByte>> storage;
    nvcv::Image                            image{nullptr};
};

inline std::vector<nvcv::ImageFormat> InvalidImageFormats()
{
    using Packing                       = nvcv::Packing;
    using Swizzle                       = nvcv::Swizzle;
    const auto                   layout = nvcv::MemLayout::PITCH_LINEAR;
    const auto                   kind   = nvcv::DataKind::UNSIGNED;
    const nvcv::ExtraChannelInfo extra{1, 8, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_U};
    const nvcv::ImageFormat withExtra(layout, kind, Swizzle::S_XYZ0, Packing::X8_Y8_Z8, Packing::NONE, Packing::NONE,
                                      Packing::NONE, nvcv::AlphaType::ASSOCIATED, &extra);
    return {
        nvcv::FMT_NV12,
        nvcv::FMT_YUYV,
        withExtra,
        {layout,                   kind, Swizzle::S_X000,                                  Packing::X1},
        {layout,                   kind, Swizzle::S_XYZ0,                              Packing::X5Y6Z5},
        {layout,                   kind, Swizzle::S_XY00, static_cast<Packing>(NVCV_PACKING_X32_Y24b8)},
        {layout,                   kind, Swizzle::S_XY00,                                  Packing::X8,Packing::X16},
        {layout,                   kind, Swizzle::S_XYZ0,                               Packing::X8_Y8,                                            Packing::X8},
        {layout, nvcv::DataKind::SIGNED, Swizzle::S_X000,                                 Packing::X64     },
        {layout,                   kind, Swizzle::S_X000,                                 Packing::X64          }
    };
}

inline void CheckCudaArrayRejected()
{
    cudaArray_t allocation = nullptr;
    const auto  descriptor = cudaCreateChannelDesc<unsigned char>();
    NVCV_CHECK_THROW(cudaMallocArray(&allocation, &descriptor, 7, 3));
    std::shared_ptr<cudaArray> storage(allocation, [](cudaArray_t pointer) { cudaFreeArray(pointer); });
    NVCVImageData              data{};
    data.format                     = NVCV_IMAGE_FORMAT_U8;
    data.bufferType                 = NVCV_IMAGE_BUFFER_CUDA_ARRAY;
    data.buffer.cudaarray.numPlanes = 1;
    data.buffer.cudaarray.planes[0] = allocation;
    NVCVImageHandle image           = nullptr;
    EXPECT_EQ(nvcvImageWrapDataConstruct(&data, nullptr, nullptr, &image), NVCV_ERROR_INVALID_ARGUMENT);
    if (image)
        nvcvImageDecRef(image, nullptr);
}

} // namespace channeltest

#endif

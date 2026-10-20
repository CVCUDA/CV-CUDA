/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ChannelCopy.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/util/CheckError.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace cvcuda::priv {
namespace {

struct ChannelTensor
{
    unsigned char *base;
    int64_t        sampleStride;
    int64_t        rowStride;
    int64_t        columnStride;
    int64_t        channelStride;
    int64_t        samples;
    int64_t        rows;
    int64_t        columns;
    int64_t        channels;
    uintptr_t      end;
};

struct ChannelArguments
{
    ChannelTensor image;
    ChannelTensor bands[4];
};

static void Require(bool valid, const char *message)
{
    if (!valid)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", message);
}

static bool SupportedScalar(nvcv::DataType dtype)
{
    return dtype == nvcv::TYPE_U8 || dtype == nvcv::TYPE_S8 || dtype == nvcv::TYPE_U16 || dtype == nvcv::TYPE_S16
        || dtype == nvcv::TYPE_U32 || dtype == nvcv::TYPE_S32 || dtype == nvcv::TYPE_F16 || dtype == nvcv::TYPE_F32
        || dtype == nvcv::TYPE_F64;
}

static ChannelTensor Describe(const nvcv::Tensor &tensor)
{
    const auto layout = tensor.layout();
    Require(layout == nvcv::TENSOR_HWC || layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_CHW
                || layout == nvcv::TENSOR_NCHW,
            "Expected HWC, NHWC, CHW or NCHW layout");
    const auto dtype = tensor.dtype();
    Require(SupportedScalar(dtype), "Unsupported scalar dtype");
    const auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
    Require(data && data->basePtr(), "Expected CUDA-accessible pitch-linear tensor storage");
    const auto shape = tensor.shape();
    const int  bytes = dtype.strideBytes();
    int64_t    span  = bytes;
    for (int dimension = tensor.rank() - 1; dimension >= 0; --dimension)
    {
        Require(shape[dimension] > 0, "Tensor dimensions must be positive");
        const auto stride = data->stride(dimension);
        Require(stride > 0 && stride % bytes == 0 && (shape[dimension] == 1 || stride >= span),
                "Expected positive, aligned, non-overlapping tensor strides");
        Require(shape[dimension] - 1 <= (std::numeric_limits<int64_t>::max() - span) / stride,
                "Tensor byte span overflows");
        span += (shape[dimension] - 1) * stride;
    }
    const auto base = reinterpret_cast<uintptr_t>(data->basePtr());
    Require(base % bytes == 0 && static_cast<uint64_t>(span) <= std::numeric_limits<uintptr_t>::max() - base,
            "Unaligned or overflowing tensor address");
    const bool batched = tensor.rank() == 4;
    const bool planar  = layout == nvcv::TENSOR_CHW || layout == nvcv::TENSOR_NCHW;
    const int  row     = (batched ? 1 : 0) + (planar ? 1 : 0);
    const int  channel = planar ? (batched ? 1 : 0) : tensor.rank() - 1;
    return {reinterpret_cast<unsigned char *>(data->basePtr()),
            batched ? data->stride(0) : 0,
            data->stride(row),
            data->stride(row + 1),
            data->stride(channel),
            batched ? shape[0] : 1,
            shape[row],
            shape[row + 1],
            shape[channel],
            base + span};
}

static bool Overlaps(const ChannelTensor &first, const ChannelTensor &second)
{
    return reinterpret_cast<uintptr_t>(first.base) < second.end && reinterpret_cast<uintptr_t>(second.base) < first.end;
}

template<typename Bits, bool Split, int Channels>
__global__ void ChannelCopyKernel(ChannelArguments args, int64_t pixels)
{
    for (int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pixel < pixels;
         pixel += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int64_t column = pixel % args.image.columns;
        const int64_t row    = pixel / args.image.columns % args.image.rows;
        const int64_t sample = pixel / args.image.columns / args.image.rows;
        auto         *image  = args.image.base + sample * args.image.sampleStride + row * args.image.rowStride
                    + column * args.image.columnStride;
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel)
        {
            const auto &band   = args.bands[channel];
            auto *bandElement  = reinterpret_cast<Bits *>(band.base + sample * band.sampleStride + row * band.rowStride
                                                         + column * band.columnStride);
            auto *imageElement = reinterpret_cast<Bits *>(image + channel * args.image.channelStride);
            if constexpr (Split)
                *bandElement = *imageElement;
            else
                *imageElement = *bandElement;
        }
    }
}

__device__ __forceinline__ uint3 SplitWords(uint32_t first, uint32_t second, uint32_t third)
{
    return {__byte_perm(__byte_perm(first, second, 0x0630), third, 0x5210),
            __byte_perm(__byte_perm(first, second, 0x0741), third, 0x6210),
            __byte_perm(__byte_perm(first, second, 0x0052), third, 0x7410)};
}

__device__ __forceinline__ void SplitPacked3Row(const uint8_t *image, uint8_t *band0, uint8_t *band1, uint8_t *band2,
                                                int64_t pixels)
{
    for (int64_t pixel = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8; pixel < pixels;
         pixel += static_cast<int64_t>(blockDim.x) * gridDim.x * 8)
    {
        if (pixels - pixel >= 8)
        {
            const auto *input                         = reinterpret_cast<const uint2 *>(image + 3 * pixel);
            const uint2 first                         = input[0];
            const uint2 second                        = input[1];
            const uint2 third                         = input[2];
            const uint3 low                           = SplitWords(first.x, first.y, second.x);
            const uint3 high                          = SplitWords(second.y, third.x, third.y);
            *reinterpret_cast<uint2 *>(band0 + pixel) = {low.x, high.x};
            *reinterpret_cast<uint2 *>(band1 + pixel) = {low.y, high.y};
            *reinterpret_cast<uint2 *>(band2 + pixel) = {low.z, high.z};
        }
        else
        {
            for (; pixel < pixels; ++pixel)
            {
                band0[pixel] = image[3 * pixel];
                band1[pixel] = image[3 * pixel + 1];
                band2[pixel] = image[3 * pixel + 2];
            }
        }
    }
}

__global__ void SplitPacked3Kernel(const uint8_t *image, uint8_t *band0, uint8_t *band1, uint8_t *band2, int64_t pixels)
{
    SplitPacked3Row(image, band0, band1, band2, pixels);
}

__device__ __forceinline__ void MergePacked3Row(const uint8_t *band0, const uint8_t *band1, const uint8_t *band2,
                                                uint8_t *image, int64_t pixels)
{
    const auto *words0 = reinterpret_cast<const uint32_t *>(band0);
    const auto *words1 = reinterpret_cast<const uint32_t *>(band1);
    const auto *words2 = reinterpret_cast<const uint32_t *>(band2);
    auto       *out    = reinterpret_cast<uint32_t *>(image);
    for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; group < pixels / 4;
         group += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const uint32_t first  = words0[group];
        const uint32_t second = words1[group];
        const uint32_t third  = words2[group];
        out[3 * group]        = __byte_perm(__byte_perm(first, second, 0x5140), third, 0x2410);
        out[3 * group + 1]    = __byte_perm(__byte_perm(first, second, 0x0625), third, 0x2150);
        out[3 * group + 2]    = __byte_perm(__byte_perm(first, second, 0x0073), third, 0x7106);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (int64_t pixel = pixels / 4 * 4; pixel < pixels; ++pixel)
        {
            image[3 * pixel]     = band0[pixel];
            image[3 * pixel + 1] = band1[pixel];
            image[3 * pixel + 2] = band2[pixel];
        }
}

__global__ void MergePacked3Kernel(const uint8_t *band0, const uint8_t *band1, const uint8_t *band2, uint8_t *image,
                                   int64_t pixels)
{
    MergePacked3Row(band0, band1, band2, image, pixels);
}

template<int WordsPerLane>
__device__ __forceinline__ void MergeWarp3Row(const uint8_t *band0, const uint8_t *band1, const uint8_t *band2,
                                              uint8_t *image, int64_t pixels)
{
    const int     lane       = threadIdx.x % 32;
    const int     warp       = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int     warpStride = gridDim.x * (blockDim.x / 32);
    const auto   *words0     = reinterpret_cast<const uint32_t *>(band0);
    const auto   *words1     = reinterpret_cast<const uint32_t *>(band1);
    const auto   *words2     = reinterpret_cast<const uint32_t *>(band2);
    auto         *out        = reinterpret_cast<uint32_t *>(image);
    constexpr int tilePixels = WordsPerLane * 128;
    for (int64_t tile = warp; tile < pixels / tilePixels; tile += warpStride)
    {
        uint32_t first[WordsPerLane], second[WordsPerLane], third[WordsPerLane];
#pragma unroll
        for (int word = 0; word < WordsPerLane; ++word)
        {
            const int64_t index = (tile * WordsPerLane + word) * 32 + lane;
            first[word]         = words0[index];
            second[word]        = words1[index];
            third[word]         = words2[index];
        }
#pragma unroll
        for (int phase = 0; phase < WordsPerLane * 3; ++phase)
        {
            const int      index      = phase * 32 + lane;
            const int      sourceLane = index / 3 % 32;
            const int      pattern    = index % 3;
            const uint32_t value0     = __shfl_sync(0xffffffff, first[phase / 3], sourceLane);
            const uint32_t value1     = __shfl_sync(0xffffffff, second[phase / 3], sourceLane);
            const uint32_t value2     = __shfl_sync(0xffffffff, third[phase / 3], sourceLane);
            const uint32_t selector0  = pattern == 0 ? 0x5140 : (pattern == 1 ? 0x0625 : 0x0073);
            const uint32_t selector1  = pattern == 0 ? 0x2410 : (pattern == 1 ? 0x2150 : 0x7106);
            out[(tile * WordsPerLane * 3 + phase) * 32 + lane]
                = __byte_perm(__byte_perm(value0, value1, selector0), value2, selector1);
        }
    }
    if (warp == 0)
        for (int64_t pixel = pixels / tilePixels * tilePixels + lane; pixel < pixels; pixel += 32)
        {
            image[3 * pixel]     = band0[pixel];
            image[3 * pixel + 1] = band1[pixel];
            image[3 * pixel + 2] = band2[pixel];
        }
}

template<int WordsPerLane>
__global__ void MergeWarp3Kernel(const uint8_t *band0, const uint8_t *band1, const uint8_t *band2, uint8_t *image,
                                 int64_t pixels)
{
    MergeWarp3Row<WordsPerLane>(band0, band1, band2, image, pixels);
}

static bool IsPacked(const ChannelTensor &tensor, int channels)
{
    return (channels == 1 || tensor.channelStride == 1) && (tensor.columns == 1 || tensor.columnStride == channels)
        && (tensor.rows == 1 || tensor.rowStride == tensor.columns * channels)
        && (tensor.samples == 1 || tensor.sampleStride == tensor.columns * channels * tensor.rows);
}

static bool LaunchPacked3(cudaStream_t stream, const ChannelArguments &args, bool split)
{
    if (!IsPacked(args.image, 3))
        return false;
    uintptr_t addresses = reinterpret_cast<uintptr_t>(args.image.base);
    for (int channel = 0; channel < 3; ++channel)
    {
        const auto &band = args.bands[channel];
        if (!IsPacked(band, 1))
            return false;
        addresses |= reinterpret_cast<uintptr_t>(band.base);
    }
    const int64_t pixels = args.image.samples * args.image.rows * args.image.columns;
    if (split)
    {
        if ((addresses & 7) != 0)
            return false;
        const int blocks = static_cast<int>(std::min<int64_t>(65535, (pixels + 2047) / 2048));
        SplitPacked3Kernel<<<blocks, 256, 0, stream>>>(args.image.base, args.bands[0].base, args.bands[1].base,
                                                       args.bands[2].base, pixels);
    }
    else
    {
        if ((addresses & 3) != 0 || pixels < 128)
            return false;
        if (pixels < 1048576)
        {
            const int blocks = static_cast<int>((pixels / 4 + 255) / 256);
            MergePacked3Kernel<<<blocks, 256, 0, stream>>>(args.bands[0].base, args.bands[1].base, args.bands[2].base,
                                                           args.image.base, pixels);
        }
        else if (pixels < 6000000)
        {
            const int blocks = static_cast<int>(std::min<int64_t>(792, (pixels / 128 + 7) / 8));
            MergeWarp3Kernel<1><<<blocks, 256, 0, stream>>>(args.bands[0].base, args.bands[1].base, args.bands[2].base,
                                                            args.image.base, pixels);
        }
        else
        {
            const int blocks = static_cast<int>(std::min<int64_t>(792, (pixels / 512 + 7) / 8));
            MergeWarp3Kernel<4><<<blocks, 256, 0, stream>>>(args.bands[0].base, args.bands[1].base, args.bands[2].base,
                                                            args.image.base, pixels);
        }
    }
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

struct PlanarCopy
{
    const unsigned char *source;
    unsigned char       *destination;
    int64_t              sourceSampleStride;
    int64_t              destinationSampleStride;
    int64_t              sourceRowStride;
    int64_t              destinationRowStride;
};

struct PlanarArguments
{
    PlanarCopy copies[4];
    int64_t    rowBytes;
    int64_t    rows;
    int64_t    lines;
};

__device__ __forceinline__ void CopyPlanarRow(const unsigned char *source, unsigned char *destination, int64_t rowBytes)
{
    const int64_t head    = min(rowBytes, static_cast<int64_t>((0 - reinterpret_cast<uintptr_t>(destination)) & 15));
    const int64_t vectors = (rowBytes - head) / 16;
    if (blockIdx.x == 0)
    {
        if (threadIdx.x < head)
            destination[threadIdx.x] = source[threadIdx.x];
        const int64_t tailBytes = (rowBytes - head) % 16;
        if (threadIdx.x < tailBytes)
        {
            const int64_t tail = rowBytes - tailBytes + threadIdx.x;
            destination[tail]  = source[tail];
        }
    }
    for (int64_t vector = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; vector < vectors;
         vector += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        reinterpret_cast<uint4 *>(destination + head)[vector] = reinterpret_cast<const uint4 *>(source + head)[vector];
    }
}

__global__ void PlanarCopyKernel(PlanarArguments args)
{
    const auto &copy = args.copies[blockIdx.z];
    for (uint64_t line = blockIdx.y; line < static_cast<uint64_t>(args.lines); line += gridDim.y)
    {
        const int64_t sample = line / args.rows;
        const int64_t row    = line % args.rows;
        const auto   *source = copy.source + sample * copy.sourceSampleStride + row * copy.sourceRowStride;
        auto *destination = copy.destination + sample * copy.destinationSampleStride + row * copy.destinationRowStride;
        CopyPlanarRow(source, destination, args.rowBytes);
    }
}

static bool LaunchPlanar(cudaStream_t stream, const ChannelArguments &args, bool split, int bytes)
{
    if (args.image.columns > 1 && args.image.columnStride != bytes)
        return false;
    PlanarArguments planar{};
    planar.rowBytes = args.image.columns * bytes;
    bool collapse   = args.image.rows == 1 || args.image.rowStride == planar.rowBytes;
    for (int channel = 0; channel < args.image.channels; ++channel)
    {
        const auto &band = args.bands[channel];
        if (band.columns > 1 && band.columnStride != bytes)
            return false;
        auto *image = args.image.base + channel * args.image.channelStride;
        if (((reinterpret_cast<uintptr_t>(image) ^ reinterpret_cast<uintptr_t>(band.base)) & 15) != 0
            || (args.image.rows > 1 && ((args.image.rowStride ^ band.rowStride) & 15) != 0)
            || (args.image.samples > 1 && ((args.image.sampleStride ^ band.sampleStride) & 15) != 0))
            return false;
        collapse = collapse && (band.rows == 1 || band.rowStride == planar.rowBytes);
        planar.copies[channel]
            = split
                ? PlanarCopy{image,         band.base, args.image.sampleStride, band.sampleStride, args.image.rowStride,
                             band.rowStride}
                : PlanarCopy{band.base,           image, band.sampleStride, args.image.sampleStride, band.rowStride,
                             args.image.rowStride};
    }
    planar.rows = collapse ? 1 : args.image.rows;
    if (collapse)
        planar.rowBytes *= args.image.rows;
    if (planar.rowBytes < 512)
        return false;
    planar.lines       = args.image.samples * planar.rows;
    const int  threads = static_cast<int>(std::min<int64_t>(256, ((planar.rowBytes - 1) / 512 + 1) * 32));
    const dim3 grid(static_cast<unsigned>(std::min<int64_t>(256, (planar.rowBytes - 1) / (threads * 16) + 1)),
                    static_cast<unsigned>(std::min<int64_t>(1024, planar.lines)),
                    static_cast<unsigned>(args.image.channels));
    PlanarCopyKernel<<<grid, threads, 0, stream>>>(planar);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

template<typename Bits, bool Split>
void Launch(cudaStream_t stream, const ChannelArguments &args)
{
    const int64_t pixels = args.image.samples * args.image.rows * args.image.columns;
    const int     blocks = static_cast<int>(std::min<int64_t>(65535, pixels / 256 + (pixels % 256 != 0)));
    switch (args.image.channels)
    {
    case 1:
        ChannelCopyKernel<Bits, Split, 1><<<blocks, 256, 0, stream>>>(args, pixels);
        break;
    case 2:
        ChannelCopyKernel<Bits, Split, 2><<<blocks, 256, 0, stream>>>(args, pixels);
        break;
    case 3:
        ChannelCopyKernel<Bits, Split, 3><<<blocks, 256, 0, stream>>>(args, pixels);
        break;
    case 4:
        ChannelCopyKernel<Bits, Split, 4><<<blocks, 256, 0, stream>>>(args, pixels);
        break;
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

struct VarShapeArguments
{
    const NVCVImageBufferStrided *image;
    const NVCVImageBufferStrided *bands[4];
    int                           samples;
    int                           channels;
    int                           bytes;
    bool                          planar;
    bool                          split;
};

template<typename Bits, int Channels>
__device__ __forceinline__ void CopyInterleavedRow(unsigned char *image, unsigned char *const *bands, int64_t width,
                                                   bool split)
{
    auto *pixels = reinterpret_cast<Bits *>(image);
    if (split)
    {
        for (int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pixel < width;
             pixel += static_cast<int64_t>(blockDim.x) * gridDim.x)
        {
#pragma unroll
            for (int channel = 0; channel < Channels; ++channel)
                reinterpret_cast<Bits *>(bands[channel])[pixel] = pixels[pixel * Channels + channel];
        }
        return;
    }
    const int     lane  = threadIdx.x % 32;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const int64_t step  = static_cast<int64_t>(gridDim.x) * (blockDim.x / 32);
    for (int64_t tile = first; tile < (width + 31) / 32; tile += step)
    {
        const int64_t pixel = tile * 32 + lane;
        Bits          values[Channels];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel)
            values[channel] = pixel < width ? reinterpret_cast<const Bits *>(bands[channel])[pixel] : Bits{};
#pragma unroll
        for (int phase = 0; phase < Channels; ++phase)
        {
            const int position      = phase * 32 + lane;
            const int sourceLane    = position / Channels;
            const int sourceChannel = position % Channels;
            Bits      selected{};
#pragma unroll
            for (int channel = 0; channel < Channels; ++channel)
            {
                Bits value;
                if constexpr (sizeof(Bits) == 8)
                    value = static_cast<Bits>(
                        __shfl_sync(0xffffffff, static_cast<unsigned long long>(values[channel]), sourceLane));
                else
                    value = static_cast<Bits>(
                        __shfl_sync(0xffffffff, static_cast<unsigned int>(values[channel]), sourceLane));
                if (sourceChannel == channel)
                    selected = value;
            }
            const int64_t output = tile * 32 * Channels + position;
            if (output < width * Channels)
                pixels[output] = selected;
        }
    }
}

template<typename Bits>
__device__ __forceinline__ void CopyVarShapeRow(const VarShapeArguments &args, int sample, int row)
{
    const auto   &image = args.image[sample];
    const int64_t width = image.planes[0].width;
    if (args.planar || args.channels == 1)
    {
        for (int channel = 0; channel < args.channels; ++channel)
        {
            const auto *batch     = channel == 0 ? args.bands[0]
                                  : channel == 1 ? args.bands[1]
                                  : channel == 2 ? args.bands[2]
                                                 : args.bands[3];
            const auto &plane     = image.planes[channel];
            const auto &bandPlane = batch[sample].planes[0];
            auto       *band      = reinterpret_cast<unsigned char *>(bandPlane.basePtr)
                       + static_cast<int64_t>(row) * bandPlane.rowStride;
            auto *pixels
                = reinterpret_cast<unsigned char *>(plane.basePtr) + static_cast<int64_t>(row) * plane.rowStride;
            auto *source      = args.split ? pixels : band;
            auto *destination = args.split ? band : pixels;
            if (((reinterpret_cast<uintptr_t>(source) ^ reinterpret_cast<uintptr_t>(destination)) & 15) == 0)
                CopyPlanarRow(source, destination, width * sizeof(Bits));
            else
                for (int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pixel < width;
                     pixel += static_cast<int64_t>(blockDim.x) * gridDim.x)
                    reinterpret_cast<Bits *>(destination)[pixel] = reinterpret_cast<const Bits *>(source)[pixel];
        }
        return;
    }
    unsigned char *bands[4];
#pragma unroll
    for (int channel = 0; channel < 4; ++channel)
        if (channel < args.channels)
        {
            const auto &plane = args.bands[channel][sample].planes[0];
            bands[channel]
                = reinterpret_cast<unsigned char *>(plane.basePtr) + static_cast<int64_t>(row) * plane.rowStride;
        }
    auto *pixels = reinterpret_cast<unsigned char *>(image.planes[0].basePtr)
                 + static_cast<int64_t>(row) * image.planes[0].rowStride;
    if constexpr (sizeof(Bits) == 1)
    {
        if (args.channels == 3)
        {
            const uintptr_t addresses = reinterpret_cast<uintptr_t>(pixels) | reinterpret_cast<uintptr_t>(bands[0])
                                      | reinterpret_cast<uintptr_t>(bands[1]) | reinterpret_cast<uintptr_t>(bands[2]);
            if (args.split && (addresses & 7) == 0)
            {
                SplitPacked3Row(pixels, bands[0], bands[1], bands[2], width);
                return;
            }
            if (!args.split && (addresses & 3) == 0)
            {
                if (width >= 128)
                    MergeWarp3Row<1>(bands[0], bands[1], bands[2], pixels, width);
                else
                    MergePacked3Row(bands[0], bands[1], bands[2], pixels, width);
                return;
            }
        }
    }
    switch (args.channels)
    {
    case 2:
        CopyInterleavedRow<Bits, 2>(pixels, bands, width, args.split);
        break;
    case 3:
        CopyInterleavedRow<Bits, 3>(pixels, bands, width, args.split);
        break;
    case 4:
        CopyInterleavedRow<Bits, 4>(pixels, bands, width, args.split);
        break;
    }
}

__global__ void VarShapeChannelCopyKernel(VarShapeArguments args)
{
    for (int64_t sample = blockIdx.z; sample < args.samples; sample += gridDim.z)
        for (int64_t row = blockIdx.y; row < args.image[sample].planes[0].height; row += gridDim.y)
        {
            switch (args.bytes)
            {
            case 1:
                CopyVarShapeRow<uint8_t>(args, sample, row);
                break;
            case 2:
                CopyVarShapeRow<uint16_t>(args, sample, row);
                break;
            case 4:
                CopyVarShapeRow<uint32_t>(args, sample, row);
                break;
            case 8:
                CopyVarShapeRow<uint64_t>(args, sample, row);
                break;
            }
        }
}

struct ImageSpan
{
    uintptr_t begin;
    uintptr_t end;
    bool      write;
};

static nvcv::DataType ImageType(nvcv::ImageFormat format)
{
    Require(format != nvcv::FMT_NONE && format.memLayout() == nvcv::MemLayout::PITCH_LINEAR,
            "Images must have a uniform pitch-linear format");
    Require(format.chromaSubsampling() == nvcv::ChromaSubsampling::CSS_444, "Subsampled image formats are unsupported");
    const int channels = format.numChannels();
    const int planes   = format.numPlanes();
    Require(channels >= 1 && channels <= 4 && (planes == 1 || planes == channels),
            "Expected one interleaved plane or one plane per channel");
    NVCVExtraChannelInfo extra{};
    format.extraChannelInfo(&extra);
    Require(extra.numChannels == 0, "Extra image channels are unsupported");
    const auto dtype = format.planeDataType(0).channelType(0);
    Require(SupportedScalar(dtype), "Unsupported scalar dtype");
    for (int plane = 0; plane < planes; ++plane)
    {
        const int  count  = planes == 1 ? channels : 1;
        const auto packed = format.planeDataType(plane);
        Require(packed.numChannels() == count && format.planeBitsPerPixel(plane) == count * dtype.strideBytes() * 8
                    && format.planePixelStrideBytes(plane) == count * dtype.strideBytes(),
                "Image channels must be contiguous whole scalar elements");
        for (int channel = 0; channel < count; ++channel)
            Require(packed.channelType(channel) == dtype, "Image channels must have equal scalar types");
    }
    return dtype;
}

static void ValidateImage(const nvcv::Image &image, nvcv::ImageFormat format, int bytes, bool write,
                          std::vector<ImageSpan> &spans)
{
    Require(image.format() == format, "Image batches must have a uniform format");
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    Require(data && data->numPlanes() == format.numPlanes(), "Images must be CUDA-accessible and pitch-linear");
    const auto size = image.size();
    Require(size.w > 0 && size.h > 0, "Image dimensions must be positive");
    for (int plane = 0; plane < data->numPlanes(); ++plane)
    {
        const auto    &buffer   = data->plane(plane);
        const uint64_t rowBytes = static_cast<uint64_t>(size.w) * format.planeNumChannels(plane) * bytes;
        Require(buffer.width == size.w && buffer.height == size.h && format.planeSize(size, plane) == size,
                "Subsampled image planes are unsupported");
        Require(buffer.basePtr && reinterpret_cast<uintptr_t>(buffer.basePtr) % bytes == 0 && buffer.rowStride > 0
                    && buffer.rowStride % bytes == 0 && static_cast<uint64_t>(buffer.rowStride) >= rowBytes,
                "Expected positive, scalar-aligned image rows");
        const uintptr_t base = reinterpret_cast<uintptr_t>(buffer.basePtr);
        const uint64_t  span = static_cast<uint64_t>(size.h - 1) * buffer.rowStride + rowBytes;
        Require(span <= std::numeric_limits<uintptr_t>::max() - base, "Image byte span overflows");
        spans.push_back({base, base + span, write});
    }
}

static void ValidateImageSpans(std::vector<ImageSpan> &spans)
{
    std::sort(spans.begin(), spans.end(),
              [](const ImageSpan &first, const ImageSpan &second) { return first.begin < second.begin; });
    uintptr_t readEnd  = 0;
    uintptr_t writeEnd = 0;
    for (const auto &span : spans)
    {
        Require(span.begin >= writeEnd && (!span.write || span.begin >= readEnd),
                "Destination spans must not overlap any source or destination, including other images");
        if (span.write)
            writeEnd = std::max(writeEnd, span.end);
        else
            readEnd = std::max(readEnd, span.end);
    }
}

} // namespace

void CopyChannels(cudaStream_t stream, const nvcv::Tensor &image, const nvcv::TensorBatch &bands, bool split)
{
    ChannelArguments args{};
    args.image = Describe(image);
    Require(args.image.channels >= 1 && args.image.channels <= 4, "Expected one to four image channels");
    Require(bands.numTensors() == args.image.channels, "Band count must equal image channel count");
    Require(bands.dtype() == image.dtype() && bands.rank() == image.rank(), "Band dtype and rank must match image");
    bool packed  = image.dtype() == nvcv::TYPE_U8 && image.layout() == nvcv::TENSOR_NHWC && args.image.channels == 3;
    int  channel = 0;
    for (const auto &tensor : bands)
    {
        auto &band = args.bands[channel];
        band       = Describe(tensor);
        packed     = packed && tensor.layout() == nvcv::TENSOR_NHWC;
        Require(band.channels == 1 && band.samples == args.image.samples && band.rows == args.image.rows
                    && band.columns == args.image.columns,
                "Expected single-channel bands with matching sample count, height and width");
        Require(!Overlaps(args.image, band), "Input and output byte spans must not overlap");
        if (split)
        {
            for (int previous = 0; previous < channel; ++previous)
                Require(!Overlaps(args.bands[previous], band), "Split destination byte spans must not overlap");
        }
        ++channel;
    }
    if (packed && LaunchPacked3(stream, args, split))
        return;
    if ((image.layout() == nvcv::TENSOR_CHW || image.layout() == nvcv::TENSOR_NCHW || args.image.channels == 1)
        && LaunchPlanar(stream, args, split, image.dtype().strideBytes()))
        return;
    switch (image.dtype().strideBytes())
    {
    case 1:
        split ? Launch<uint8_t, true>(stream, args) : Launch<uint8_t, false>(stream, args);
        break;
    case 2:
        split ? Launch<uint16_t, true>(stream, args) : Launch<uint16_t, false>(stream, args);
        break;
    case 4:
        split ? Launch<uint32_t, true>(stream, args) : Launch<uint32_t, false>(stream, args);
        break;
    case 8:
        split ? Launch<uint64_t, true>(stream, args) : Launch<uint64_t, false>(stream, args);
        break;
    }
}

void CopyChannels(cudaStream_t stream, const nvcv::ImageBatchVarShape &image, const nvcv::ImageBatchVarShape *bands,
                  int32_t count, bool split)
{
    Require(bands && count >= 1 && count <= 4, "Expected one to four band batches");
    const int samples = image.numImages();
    Require(samples > 0, "Image batches must not be empty");
    const auto format = image.uniqueFormat();
    const auto dtype  = ImageType(format);
    const int  bytes  = dtype.strideBytes();
    Require(count == format.numChannels(), "Band count must equal the image channel count");
    nvcv::ImageFormat bandFormats[4];
    for (int channel = 0; channel < count; ++channel)
    {
        Require(bands[channel].numImages() == samples, "Image and band batch sizes must match");
        bandFormats[channel] = bands[channel].uniqueFormat();
        Require(ImageType(bandFormats[channel]) == dtype && bandFormats[channel].numChannels() == 1,
                "Expected single-channel bands of matching scalar type");
    }
    std::vector<ImageSpan> spans;
    spans.reserve(static_cast<size_t>(samples) * count * 2);
    for (int sample = 0; sample < samples; ++sample)
    {
        const auto frame = image[sample];
        ValidateImage(frame, format, bytes, !split, spans);
        for (int channel = 0; channel < count; ++channel)
        {
            const auto band = bands[channel][sample];
            Require(band.size() == frame.size(), "Corresponding image and band dimensions must match");
            ValidateImage(band, bandFormats[channel], bytes, split, spans);
        }
    }
    ValidateImageSpans(spans);
    const auto imageData = image.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    Require(imageData && imageData->imageList(), "Expected CUDA image batch descriptors");
    VarShapeArguments args{};
    args.image    = imageData->imageList();
    args.samples  = samples;
    args.channels = count;
    args.bytes    = bytes;
    args.planar   = format.numPlanes() > 1;
    args.split    = split;
    for (int channel = 0; channel < count; ++channel)
    {
        const auto data = bands[channel].exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
        Require(data && data->imageList(), "Expected CUDA band batch descriptors");
        args.bands[channel] = data->imageList();
    }
    const auto size = image.maxSize();
    dim3       grid(static_cast<unsigned>(std::min<int64_t>(128, (static_cast<int64_t>(size.w) * bytes + 4095) / 4096)),
                    std::min(size.h, 256), std::min(samples, 65535));
    VarShapeChannelCopyKernel<<<grid, 256, 0, stream>>>(args);
    NVCV_CHECK_THROW(cudaGetLastError());
}

} // namespace cvcuda::priv

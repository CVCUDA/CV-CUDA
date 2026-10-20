// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Definitions.hpp"

#include <common/DeviceBuffer.hpp>
#include <cuda_fp16.h>
#include <cvcuda/OpWarpPerspective.hpp>
#include <cvcuda/priv/WarpPerspectiveAccess.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct AlignmentCase
{
    const char           *name;
    int                   inputOffset;
    int                   outputOffset;
    int                   inputRowPadding;
    int                   outputRowPadding;
    int                   inputSamplePadding;
    int                   outputSamplePadding;
    int                   batches;
    int                   channels;
    bool                  planar;
    bool                  unbatched;
    nvcv::DataType        dataType      = nvcv::TYPE_U8;
    bool                  fractional    = false;
    NVCVInterpolationType interpolation = NVCV_INTERP_CUBIC;
};

const std::array<float, 4> kBorderValues{8.f, 40.f, 72.f, 104.f};

float PatternValue(int batch, int channel, int row, int column)
{
    return static_cast<float>(16 + 32 * batch + 32 * channel + 16 * row + 8 * column);
}

struct AlignmentStorage
{
    int                  width;
    int                  height;
    AlignmentCase        params;
    int                  elementSize;
    int64_t              offset;
    int64_t              rowStride;
    int64_t              planeStride;
    int64_t              sampleStride;
    std::vector<uint8_t> bytes;

    AlignmentStorage(int imageWidth, int imageHeight, const AlignmentCase &specification, bool output, uint8_t sentinel)
        : width(imageWidth)
        , height(imageHeight)
        , params(specification)
        , elementSize(params.dataType.strideBytes())
        , offset(output ? params.outputOffset : params.inputOffset)
        , rowStride(width * (params.planar ? 1 : params.channels) * elementSize
                    + (output ? params.outputRowPadding : params.inputRowPadding))
        , planeStride(height * rowStride)
        , sampleStride(planeStride * (params.planar ? params.channels : 1)
                       + (output ? params.outputSamplePadding : params.inputSamplePadding))
        , bytes(offset + params.batches * sampleStride + 13, sentinel)
    {
    }

    void Store(int batch, int channel, int row, int column, float value)
    {
        auto *destination = bytes.data() + offset + batch * sampleStride + row * rowStride
                          + (params.planar ? channel * planeStride + column * elementSize
                                           : (column * params.channels + channel) * elementSize);
        if (params.dataType == nvcv::TYPE_F32)
        {
            std::memcpy(destination, &value, sizeof(value));
        }
        else if (params.dataType == nvcv::TYPE_F16)
        {
            const __half halfValue = __float2half(value);
            std::memcpy(destination, &halfValue, sizeof(halfValue));
        }
        else
        {
            *destination = static_cast<uint8_t>(std::rint(value));
        }
    }

    float PixelValue(int batch, int channel, int row, int column, bool warped) const
    {
        constexpr std::array<std::array<uint8_t, 4>, 2> pixels{
            {{0x99, 0xd8, 0xa4, 0x36}, {0, 127, 255, 64}}
        };
        if (!params.fractional)
        {
            return pixels[batch][channel];
        }
        if (!warped)
        {
            return PatternValue(batch, channel, row, column);
        }
        const auto read = [this, batch, channel](int sourceRow, int sourceColumn)
        {
            return sourceRow < 0 || sourceRow >= height || sourceColumn < 0 || sourceColumn >= width
                     ? kBorderValues[channel]
                     : PatternValue(batch, channel, sourceRow, sourceColumn);
        };
        if (params.interpolation == NVCV_INTERP_NEAREST)
        {
            return read(row + 1, column - 1);
        }
        return (3.f * read(row, column - 1) + read(row, column) + 9.f * read(row + 1, column - 1)
                + 3.f * read(row + 1, column))
             / 16.f;
    }

    void FillPixels(bool warped = false)
    {
        for (int batch = 0; batch < params.batches; ++batch)
            for (int pixel = 0; pixel < width * height; ++pixel)
                for (int channel = 0; channel < params.channels; ++channel)
                    Store(batch, channel, pixel / width, pixel % width,
                          PixelValue(batch, channel, pixel / width, pixel % width, warped));
    }

    nvcv::Tensor Wrap(NVCVByte *allocation) const
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr = allocation + offset;
        int dimension  = 0;
        if (!params.unbatched)
            buffer.strides[dimension++] = sampleStride;
        if (params.planar)
            buffer.strides[dimension++] = planeStride;
        buffer.strides[dimension++] = rowStride;
        if (!params.planar)
            buffer.strides[dimension++] = params.channels * elementSize;
        buffer.strides[dimension] = elementSize;
        const auto shape = params.planar
                             ? (params.unbatched ? nvcv::TensorShape{{params.channels, height, width}, "CHW"}
                                                 : nvcv::TensorShape{{params.batches, params.channels, height, width}, "NCHW"})
                             : (params.unbatched ? nvcv::TensorShape{{height, width, params.channels}, "HWC"}
                                                 : nvcv::TensorShape{{params.batches, height, width, params.channels}, "NHWC"});
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, params.dataType, buffer},
                                    nvcv::TensorDataCleanupCallback{[allocation](const nvcv::TensorData &)
                                                                    {
                                                                        EXPECT_EQ(cudaSuccess, cudaFree(allocation));
                                                                    }});
    }

    nvcv::Image WrapImage(NVCVByte *allocation, int batch) const
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes = 1;
        buffer.planes[0] = {width, height, static_cast<int32_t>(rowStride), allocation + offset + batch * sampleStride};
        const auto format = params.dataType == nvcv::TYPE_F32 ? nvcv::FMT_RGBAf32
                          : params.dataType == nvcv::TYPE_F16 ? nvcv::FMT_RGBAf16
                          : params.channels == 3              ? nvcv::FMT_RGB8
                                                              : nvcv::FMT_RGBA8;
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, buffer});
    }
};

class OpWarpPerspectiveAlignment : public ::testing::TestWithParam<AlignmentCase>
{
};

TEST_P(OpWarpPerspectiveAlignment, exact_pixels_with_storage_canaries)
{
    const AlignmentCase params  = GetParam();
    const bool          strided = params.inputRowPadding || params.outputRowPadding || params.inputSamplePadding
                      || params.outputSamplePadding || params.fractional;
    AlignmentStorage input(strided ? 4 : 1, strided ? 3 : 1, params, false, 0xa5);
    AlignmentStorage expected(4, 3, params, true, 0x5a);
    input.FillPixels();
    expected.FillPixels(true);
    nvcv::test::StreamGuard streamOwner;
    const auto              stream = streamOwner.get();
    ASSERT_NE(nullptr, stream);
    NVCVByte *inputAllocation  = nullptr;
    NVCVByte *outputAllocation = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&inputAllocation), input.bytes.size()));
    auto inputTensor = input.Wrap(inputAllocation);
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&outputAllocation), expected.bytes.size()));
    auto outputTensor = expected.Wrap(outputAllocation);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(inputAllocation, input.bytes.data(), input.bytes.size(), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(outputAllocation, 0x5a, expected.bytes.size(), stream));
    const NVCVPerspectiveTransform matrix{
        1.f, 0.f, params.fractional ? -0.75f : 0.f, 0.f, 1.f, params.fractional ? 0.75f : 0.f, 0.f, 0.f, 1.f};
    cvcuda::WarpPerspective operation(0);
    ASSERT_NO_THROW(operation(stream, inputTensor, outputTensor, matrix, params.interpolation | NVCV_WARP_INVERSE_MAP,
                              params.fractional ? NVCV_BORDER_CONSTANT : NVCV_BORDER_REFLECT,
                              {kBorderValues[0], kBorderValues[1], kBorderValues[2], kBorderValues[3]}));
    std::vector<uint8_t> actual(expected.bytes.size());
    std::vector<uint8_t> inputAfter(input.bytes.size());
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(actual.data(), outputAllocation, actual.size(), cudaMemcpyDeviceToHost, stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(inputAfter.data(), inputAllocation, inputAfter.size(), cudaMemcpyDeviceToHost, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(input.bytes, inputAfter);
    EXPECT_EQ(expected.bytes, actual);
}

class OpWarpPerspectiveVarShapeAlignment : public ::testing::TestWithParam<AlignmentCase>
{
};

TEST_P(OpWarpPerspectiveVarShapeAlignment, exact_pixels_with_storage_canaries)
{
    const AlignmentCase params  = GetParam();
    const bool          strided = params.inputRowPadding || params.outputRowPadding || params.inputSamplePadding
                      || params.outputSamplePadding || params.fractional;
    AlignmentStorage input(strided ? 4 : 1, strided ? 3 : 1, params, false, 0xa5);
    AlignmentStorage expected(4, 3, params, true, 0x5a);
    input.FillPixels();
    expected.FillPixels(true);
    nvcv::test::StreamGuard streamOwner;
    const auto              stream = streamOwner.get();
    ASSERT_NE(nullptr, stream);
    NVCVByte *inputAllocation  = nullptr;
    NVCVByte *outputAllocation = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&inputAllocation), input.bytes.size()));
    auto inputOwner = input.Wrap(inputAllocation);
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&outputAllocation), expected.bytes.size()));
    auto                     outputOwner = expected.Wrap(outputAllocation);
    nvcv::ImageBatchVarShape inputs(params.batches);
    nvcv::ImageBatchVarShape outputs(params.batches);
    for (int batch = 0; batch < params.batches; ++batch)
    {
        inputs.pushBack(input.WrapImage(inputAllocation, batch));
        outputs.pushBack(expected.WrapImage(outputAllocation, batch));
    }
    nvcv::Tensor matrices(
        nvcv::TensorShape{
            {params.batches, 9},
            "NW"
    },
        nvcv::TYPE_F32);
    const auto                  matrixData = matrices.exportData<nvcv::TensorDataStridedCuda>();
    const float                 shiftX     = params.fractional ? -0.75f : 0.f;
    const float                 shiftY     = params.fractional ? 0.75f : 0.f;
    const std::array<float, 18> matrix{1.f, 0.f, shiftX, 0.f, 1.f, shiftY, 0.f, 0.f, 1.f,
                                       1.f, 0.f, shiftX, 0.f, 1.f, shiftY, 0.f, 0.f, 1.f};
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy2DAsync(matrixData->basePtr(), matrixData->stride(0), matrix.data(), 9 * sizeof(float),
                                9 * sizeof(float), params.batches, cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(inputAllocation, input.bytes.data(), input.bytes.size(), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(outputAllocation, 0x5a, expected.bytes.size(), stream));
    cvcuda::WarpPerspective operation(params.batches);
    ASSERT_NO_THROW(operation(stream, inputs, outputs, matrices, params.interpolation | NVCV_WARP_INVERSE_MAP,
                              params.fractional ? NVCV_BORDER_CONSTANT : NVCV_BORDER_REFLECT,
                              {kBorderValues[0], kBorderValues[1], kBorderValues[2], kBorderValues[3]}));
    std::vector<uint8_t> actual(expected.bytes.size());
    std::vector<uint8_t> inputAfter(input.bytes.size());
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(actual.data(), outputAllocation, actual.size(), cudaMemcpyDeviceToHost, stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(inputAfter.data(), inputAllocation, inputAfter.size(), cudaMemcpyDeviceToHost, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(input.bytes, inputAfter);
    EXPECT_EQ(expected.bytes, actual);
}

TEST(OpWarpPerspectiveAlignmentAccess, vector_alignment_includes_used_row_and_sample_strides)
{
    using cvcuda::priv::perspective::PixelAligned;
    alignas(16) std::array<uint8_t, 128> storage{};
    EXPECT_TRUE(PixelAligned<uchar4>(storage.data(), 16, 48, 3, 2));
    EXPECT_FALSE(PixelAligned<uchar4>(storage.data() + 1, 16, 48, 3, 2));
    EXPECT_FALSE(PixelAligned<uchar4>(storage.data(), 17, 52, 3, 2));
    EXPECT_FALSE(PixelAligned<uchar4>(storage.data(), 16, 49, 3, 2));
    EXPECT_TRUE(PixelAligned<uchar4>(storage.data(), 17, 49, 1, 1));
    EXPECT_TRUE(PixelAligned<uchar3>(storage.data() + 1, 17, 49, 3, 2));
    EXPECT_FALSE(PixelAligned<float4>(storage.data() + 4, 64, 128, 2, 2));
}

const std::array<AlignmentCase, 17> kAlignmentCases{
    {
     {"BothBases", 1, 1, 0, 0, 0, 0, 1, 4, false, false},
     {"InputBase", 1, 0, 0, 0, 0, 0, 1, 4, false, false},
     {"OutputBase", 0, 1, 0, 0, 0, 0, 1, 4, false, false},
     {"InputRows", 0, 0, 1, 0, 0, 0, 1, 4, false, false},
     {"OutputRows", 0, 0, 0, 1, 0, 0, 1, 4, false, false},
     {"InputSamples", 0, 0, 0, 0, 1, 0, 2, 4, false, false},
     {"OutputSamples", 0, 0, 0, 0, 0, 1, 2, 4, false, false},
     {"HWC", 1, 1, 0, 0, 0, 0, 1, 4, false, true},
     {"NCHW", 1, 1, 0, 0, 0, 0, 1, 4, true, false},
     {"CHW", 1, 1, 0, 0, 0, 0, 1, 4, true, true},
     {"RGB", 1, 1, 0, 0, 0, 0, 1, 3, false, false},
     {"F32InputBase", 4, 0, 0, 0, 0, 0, 1, 4, false, false, nvcv::TYPE_F32},
     {"F32OutputStrides", 0, 0, 0, 4, 0, 4, 2, 4, false, false, nvcv::TYPE_F32},
     {"F16ScalarStorage", 2, 2, 2, 2, 2, 2, 2, 4, false, false, nvcv::TYPE_F16},
     {"U8FractionalBorder", 1, 1, 1, 1, 1, 1, 2, 4, false, false, nvcv::TYPE_U8, true},
     {"F32FractionalBorder", 4, 4, 4, 4, 4, 4, 2, 4, false, false, nvcv::TYPE_F32, true},
     {"F16FractionalBorder", 2, 2, 2, 2, 2, 2, 2, 4, false, false, nvcv::TYPE_F16, true},
     }
};

std::vector<AlignmentCase> AlignmentCases(bool varShape, NVCVInterpolationType interpolation)
{
    std::vector<AlignmentCase> cases;
    for (auto params : kAlignmentCases)
    {
        if ((varShape && (params.planar || params.unbatched))
            || (interpolation == NVCV_INTERP_CUBIC && params.fractional)
            || (interpolation == NVCV_INTERP_NEAREST && !params.fractional))
        {
            continue;
        }
        params.interpolation = interpolation;
        cases.push_back(params);
    }
    return cases;
}

std::string AlignmentCaseName(const ::testing::TestParamInfo<AlignmentCase> &info)
{
    return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(ValidCubicStorage, OpWarpPerspectiveAlignment,
                         ::testing::ValuesIn(AlignmentCases(false, NVCV_INTERP_CUBIC)), AlignmentCaseName);

INSTANTIATE_TEST_SUITE_P(ValidCubicImageStorage, OpWarpPerspectiveVarShapeAlignment,
                         ::testing::ValuesIn(AlignmentCases(true, NVCV_INTERP_CUBIC)), AlignmentCaseName);

INSTANTIATE_TEST_SUITE_P(ValidLinearStorage, OpWarpPerspectiveAlignment,
                         ::testing::ValuesIn(AlignmentCases(false, NVCV_INTERP_LINEAR)), AlignmentCaseName);

INSTANTIATE_TEST_SUITE_P(ValidLinearImageStorage, OpWarpPerspectiveVarShapeAlignment,
                         ::testing::ValuesIn(AlignmentCases(true, NVCV_INTERP_LINEAR)), AlignmentCaseName);

INSTANTIATE_TEST_SUITE_P(ValidNearestStorage, OpWarpPerspectiveAlignment,
                         ::testing::ValuesIn(AlignmentCases(false, NVCV_INTERP_NEAREST)), AlignmentCaseName);

INSTANTIATE_TEST_SUITE_P(ValidNearestImageStorage, OpWarpPerspectiveVarShapeAlignment,
                         ::testing::ValuesIn(AlignmentCases(true, NVCV_INTERP_NEAREST)), AlignmentCaseName);

} // namespace

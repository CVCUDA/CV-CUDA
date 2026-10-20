// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Definitions.hpp"
#include "WarpPerspectiveCubicGold.hpp"

#include <common/DeviceBuffer.hpp>
#include <cuda_runtime.h>
#include <cvcuda/OpWarpPerspective.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <vector>

namespace {

struct CubicCase
{
    const char *name;
    int         width;
    int         outputWidth;
    int         channels;
    bool        planar;
    int         phase;
    bool        aligned = false;
};

const std::array<CubicCase, 16> kCubicCases{
    {
     {"RGBASingleton", 1, 4, 4, false, 0},
     {"RGBAIntegerOffset", 7, 10, 4, false, 0},
     {"RGBAQuarterOffset", 7, 10, 4, false, 1},
     {"RGBAHalfOffset", 7, 10, 4, false, 2},
     {"RGBAThreeQuarterOffset", 7, 10, 4, false, 3},
     {"RGBAAlignedControl", 7, 10, 4, false, 1, true},
     {"RGBPackedControl", 7, 10, 3, false, 1},
     {"PlanarPaddedPlanes", 7, 10, 4, true, 1},
     {"RGBATail31", 31, 31, 4, false, 1},
     {"RGBATail32", 32, 32, 4, false, 1},
     {"RGBATail33", 33, 33, 4, false, 1},
     {"RGBATail127", 127, 127, 4, false, 1},
     {"RGBATail128", 128, 128, 4, false, 1},
     {"RGBATail129", 129, 129, 4, false, 1},
     {"RGBATwoColumns", 2, 5, 4, false, 2},
     {"PlanarTail129", 129, 129, 4, true, 2},
     }
};

struct HostBuffer
{
    int                  width;
    int                  height;
    int                  channels;
    bool                 planar;
    int64_t              baseOffset;
    int64_t              rowStride;
    int64_t              planeStride;
    int64_t              sampleStride;
    std::vector<uint8_t> bytes;

    HostBuffer(int imageWidth, int imageHeight, int channelCount, int batchSize, bool isPlanar, int offset, bool padded,
               uint8_t sentinel)
        : width(imageWidth)
        , height(imageHeight)
        , channels(channelCount)
        , planar(isPlanar)
        , baseOffset(offset)
        , rowStride(width * (planar ? 1 : channels) + (padded ? 5 : 0))
        , planeStride(rowStride * height + (padded ? 7 : 0))
        , sampleStride(planeStride * (planar ? channels : 1) + (padded ? 11 : 0))
        , bytes(baseOffset + batchSize * sampleStride + 13, sentinel)
    {
    }

    size_t Offset(int batch, int channel, int row, int column) const
    {
        return baseOffset + batch * sampleStride + row * rowStride
             + (planar ? channel * planeStride + column : column * channels + channel);
    }

    nvcv::Tensor Wrap(NVCVByte *allocation, int batchSize) const
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = allocation + baseOffset;
        buffer.strides[0] = sampleStride;
        buffer.strides[1] = planar ? planeStride : rowStride;
        buffer.strides[2] = planar ? rowStride : channels;
        buffer.strides[3] = 1;
        const auto shape = planar ? nvcv::TensorShape{{batchSize, channels, height, width}, "NCHW"}
                                  : nvcv::TensorShape{{batchSize, height, width, channels}, "NHWC"};
        auto cleanup = [allocation](const nvcv::TensorData &)
        {
            EXPECT_EQ(cudaSuccess, cudaFree(allocation));
        };
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, buffer},
                                    nvcv::TensorDataCleanupCallback{cleanup});
    }
};

class OpWarpPerspectiveCubicIndependent : public ::testing::TestWithParam<CubicCase>
{
};

TEST_P(OpWarpPerspectiveCubicIndependent, ExactQuarterReflectWithStorageCanaries)
{
    const CubicCase params    = GetParam();
    const int       batchSize = params.phase % 2 ? 3 : 1;
    const int       height    = params.phase ? params.phase + 5 : 1;
    const int       offset    = params.aligned ? 0 : params.phase + 1;
    const bool      padded    = !params.aligned && params.phase != 0;
    HostBuffer      input(params.width, height, params.channels, batchSize, params.planar, offset, padded, 0xA5);
    HostBuffer      expected(params.outputWidth, height + 2, params.channels, batchSize, params.planar, offset, padded,
                             0x5A);
    const std::array<int, 4> shiftsX{0, -3, 2, 4 * params.width + 3};
    const std::array<int, 4> shiftsY{0, 3, -6, -4 * height - 1};
    const int                shiftX = shiftsX[params.phase];
    const int                shiftY = shiftsY[params.phase];
    for (int sampleChannel = 0; sampleChannel < batchSize * params.channels; ++sampleChannel)
        for (int row = 0; row < height; ++row)
            for (int column = 0; column < params.width; ++column)
                input.bytes[input.Offset(sampleChannel / params.channels, sampleChannel % params.channels, row, column)]
                    = static_cast<uint8_t>(sampleChannel * 97 + row * 53 + column * 29 + column * column * 7
                                           + row * column * 11);

    for (int sampleChannel = 0; sampleChannel < batchSize * params.channels; ++sampleChannel)
    {
        const int batch   = sampleChannel / params.channels;
        const int channel = sampleChannel % params.channels;
        for (int row = 0; row < expected.height; ++row)
            for (int column = 0; column < expected.width; ++column)
                expected.bytes[expected.Offset(batch, channel, row, column)]
                    = nvcv::test::perspective::ExactQuarterCubicReflect(
                        4 * column + shiftX, 4 * row + shiftY, input.width, input.height,
                        [&input, batch, channel](int sourceRow, int sourceColumn)
                        { return input.bytes[input.Offset(batch, channel, sourceRow, sourceColumn)]; });
    }

    nvcv::test::StreamGuard stream;
    ASSERT_NE(nullptr, stream.get());
    NVCVByte *inputAllocation = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&inputAllocation), input.bytes.size()));
    nvcv::Tensor inputTensor      = input.Wrap(inputAllocation, batchSize);
    NVCVByte    *outputAllocation = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&outputAllocation), expected.bytes.size()));
    nvcv::Tensor outputTensor = expected.Wrap(outputAllocation, batchSize);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inputAllocation, input.bytes.data(), input.bytes.size(),
                                           cudaMemcpyHostToDevice, stream.get()));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(outputAllocation, 0x5A, expected.bytes.size(), stream.get()));
    const bool                     inverseMap = params.phase % 2 == 0;
    const float                    direction  = inverseMap ? 1.f : -1.f;
    const NVCVPerspectiveTransform matrix{1.f, 0.f, direction * shiftX / 4.f, 0.f, 1.f, direction * shiftY / 4.f, 0.f,
                                          0.f, 1.f};
    cvcuda::WarpPerspective        operation(0);
    ASSERT_NO_THROW(operation(stream.get(), inputTensor, outputTensor, matrix,
                              NVCV_INTERP_CUBIC | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0), NVCV_BORDER_REFLECT,
                              {0.f, 0.f, 0.f, 0.f}));
    std::vector<uint8_t> actual(expected.bytes.size());
    std::vector<uint8_t> inputAfter(input.bytes.size());
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(actual.data(), outputAllocation, actual.size(), cudaMemcpyDeviceToHost, stream.get()));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inputAfter.data(), inputAllocation, inputAfter.size(),
                                           cudaMemcpyDeviceToHost, stream.get()));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.get()));
    EXPECT_EQ(input.bytes, inputAfter);
    EXPECT_EQ(expected.bytes, actual);
}

INSTANTIATE_TEST_SUITE_P(IndependentGold, OpWarpPerspectiveCubicIndependent, ::testing::ValuesIn(kCubicCases),
                         [](const ::testing::TestParamInfo<CubicCase> &info) { return info.param.name; });

} // namespace

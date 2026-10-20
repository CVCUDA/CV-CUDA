/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ChannelCopyTestUtils.hpp"

#include <cvcuda/OpSplit.hpp>

namespace {

std::vector<uint8_t> SplitCpuGold(const std::vector<uint8_t> &source, int channels, int channel, int bytes)
{
    std::vector<uint8_t> result(source.size() / channels);
    for (size_t pixel = 0; pixel < result.size() / bytes; ++pixel)
        std::copy_n(source.begin() + (pixel * channels + channel) * bytes, bytes, result.begin() + pixel * bytes);
    return result;
}

NVCV_TEST_SUITE_P(
    OpSplit, nvcv::test::Combine(nvcv::test::ValueList<nvcv::DataType>{nvcv::TYPE_U8, nvcv::TYPE_S8, nvcv::TYPE_U16,
                                                                       nvcv::TYPE_S16, nvcv::TYPE_U32, nvcv::TYPE_S32,
                                                                       nvcv::TYPE_F16, nvcv::TYPE_F32, nvcv::TYPE_F64},
                                 nvcv::test::ValueList<int>{1, 2, 3, 4},
                                 nvcv::test::ValueList<nvcv::TensorLayout>{nvcv::TENSOR_HWC, nvcv::TENSOR_NHWC,
                                                                           nvcv::TENSOR_CHW, nvcv::TENSOR_NCHW},
                                 nvcv::test::ValueList<bool>{false, true}));

TEST_P(OpSplit, matches_interleaved_cpu_gold_and_preserves_padding)
{
    const auto &[dtype, channels, layout, crossLayout] = GetParam();
    const auto          bandLayout                     = crossLayout ? channeltest::OtherLayout(layout) : layout;
    const int           samples                        = layout.rank() == 4 ? 3 : 1;
    channeltest::Stream stream;
    cvcuda::Split       op;
    for (const int width : {1, 17, 257})
    {
        channeltest::Buffer image(layout, dtype, samples, 3, width, channels);
        if (width == 1)
            image.CompactSingletonStrides();
        const auto pixels = channeltest::Patterns(samples * 3 * width * channels, image.bytes);
        image.Put(pixels, samples, 3, width, channels);
        image.Upload(stream.handle);
        std::vector<channeltest::Buffer> bands;
        nvcv::TensorBatch                outputs(channels);
        for (int channel = 0; channel < channels; ++channel)
        {
            bands.emplace_back(bandLayout, dtype, samples, 3, width, 1, 64 + 8 * channel);
            outputs.pushBack(bands.back().tensor);
            bands.back().Clear(stream.handle);
            bands.back().Put(SplitCpuGold(pixels, channels, channel, image.bytes), samples, 3, width, 1);
        }
        op(stream.handle, image.tensor, outputs);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
        EXPECT_EQ(image.Download(), image.host);
    }
}

NVCV_TEST_SUITE_P(OpSplitPacked,
                  nvcv::test::Combine(nvcv::test::ValueList<int>{1,       7,       8,       9,      15,      16,
                                                                 17,      127,     128,     129,    511,     512,
                                                                 513,     2047,    2048,    2049,   1048575, 1048576,
                                                                 1048577, 5999999, 6000000, 6000001},
                                      nvcv::test::ValueList<int>{0, 1, 4, 8}));

void CheckPackedSplit(int width, int offset)
{
    channeltest::Stream stream;
    cvcuda::Split       op;
    channeltest::Buffer image(nvcv::TENSOR_NHWC, nvcv::TYPE_U8, 1, 1, width, 3, 0, 64 + offset, 0xC3);
    image.CompactSingletonStrides();
    const auto pixels = channeltest::Patterns(width * 3, 1);
    image.Put(pixels, 1, 1, width, 3);
    image.Upload(stream.handle);
    nvcv::TensorBatch                outputs(3);
    std::vector<channeltest::Buffer> bands;
    for (int channel = 0; channel < 3; ++channel)
    {
        bands.emplace_back(nvcv::TENSOR_NHWC, nvcv::TYPE_U8, 1, 1, width, 1, 0, 64 + (channel + 2) * offset,
                           0x31 + channel * 0x11);
        bands.back().CompactSingletonStrides();
        bands.back().Clear(stream.handle);
        bands.back().Put(SplitCpuGold(pixels, 3, channel, 1), 1, 1, width, 1);
        outputs.pushBack(bands.back().tensor);
    }
    op(stream.handle, image.tensor, outputs);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    EXPECT_EQ(image.Download(), image.host);
    for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
}

TEST_P(OpSplitPacked, vector_boundaries_and_independent_alignment)
{
    const auto &[width, offset] = GetParam();
    CheckPackedSplit(width, offset);
}

TEST(OpSplitTensor, packed_grid_stride_second_iteration)
{
    CheckPackedSplit(65535 * 256 * 8 + 1, 0);
}

NVCV_TEST_SUITE_P(OpSplitStrided,
                  nvcv::test::Combine(nvcv::test::ValueList<nvcv::DataType>{nvcv::TYPE_U8, nvcv::TYPE_U16,
                                                                            nvcv::TYPE_U32, nvcv::TYPE_F64},
                                      nvcv::test::ValueList<int>{1, 2, 3, 4},
                                      nvcv::test::ValueList<nvcv::TensorLayout>{nvcv::TENSOR_HWC, nvcv::TENSOR_NHWC,
                                                                                nvcv::TENSOR_CHW, nvcv::TENSOR_NCHW}));

TEST_P(OpSplitStrided, independent_band_strides_and_scalar_aligned_bases)
{
    const auto &[dtype, channels, layout] = GetParam();

    const int samples = layout.rank() == 4 ? 3 : 1;
    const int bytes   = dtype.strideBytes();

    channeltest::Stream stream;
    cvcuda::Split       op;
    for (const auto &[width, crossLayout] : {
             std::tuple{ 17, false},
             std::tuple{257,  true}
    })
    {
        SCOPED_TRACE(width);
        const auto          bandLayout = crossLayout ? channeltest::OtherLayout(layout) : layout;
        channeltest::Buffer image(layout, dtype, samples, 3, width, channels, 3 * bytes, 64 + bytes);
        const auto          pixels = channeltest::Patterns(samples * 3 * width * channels, bytes);
        image.Put(pixels, samples, 3, width, channels);
        image.Upload(stream.handle);
        std::vector<channeltest::Buffer> bands;
        nvcv::TensorBatch                outputs(channels);
        for (int channel = 0; channel < channels; ++channel)
        {
            bands.emplace_back(bandLayout, dtype, samples, 3, width, 1, (5 + 3 * channel) * bytes,
                               64 + (2 * channel + 1) * bytes);
            bands.back().Clear(stream.handle);
            bands.back().Put(SplitCpuGold(pixels, channels, channel, bytes), samples, 3, width, 1);
            outputs.pushBack(bands.back().tensor);
        }
        op(stream.handle, image.tensor, outputs);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        EXPECT_EQ(image.Download(), image.host);
        for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
    }
}

NVCV_TEST_SUITE_P(OpSplitPlanar, nvcv::test::Combine(
                                     nvcv::test::ValueList<nvcv::DataType>{
                                         nvcv::TYPE_U8, nvcv::TYPE_S8, nvcv::TYPE_U16, nvcv::TYPE_S16, nvcv::TYPE_U32,
                                         nvcv::TYPE_S32, nvcv::TYPE_F16, nvcv::TYPE_F32, nvcv::TYPE_F64},
                                     nvcv::test::ValueList<int>{1, 2, 3, 4},
                                     nvcv::test::ValueList<nvcv::TensorLayout>{nvcv::TENSOR_CHW, nvcv::TENSOR_NCHW}));

void CheckPlanarSplit(nvcv::DataType dtype, int channels, nvcv::TensorLayout layout, int samples, int rows, int width,
                      int padding, int offset, bool badAlignment, bool spaced, int badStride = 0, bool dense = false)
{
    const int           bytes = dtype.strideBytes();
    channeltest::Stream stream;
    cvcuda::Split       op;
    channeltest::Buffer image(layout, dtype, samples, rows, width, channels, padding * bytes, 64 + offset * bytes, 0xC3,
                              32);
    const auto          pixels = channeltest::Patterns(static_cast<size_t>(samples) * rows * width * channels, bytes);
    image.Put(pixels, samples, rows, width, channels);
    image.Upload(stream.handle);
    nvcv::TensorBatch                outputs(channels);
    std::vector<channeltest::Buffer> bands;
    for (int channel = 0; channel < channels; ++channel)
    {
        const auto bandLayout = channeltest::OtherLayout(layout);
        const int  step       = spaced && channel == channels - 1 ? 2 : 1;
        bands.emplace_back(
            bandLayout, dtype, samples, rows, width * step, 1,
            (padding + (dense ? 0 : channel * 16) + (badStride == 1 && channel == channels - 1 ? 1 : 0)) * bytes,
            64 + (offset + (badAlignment && channel == channels - 1 ? 1 : 0)) * bytes
                + (dense ? (channel * rows * width * bytes) % 16 : 0),
            0x31 + channel * 0x11,
            16 * (channel + 1) + (badStride == 2 && channel == channels - 1 ? bytes : 0)
                + (dense ? ((channels - 1) * rows * width * bytes) % 16 : 0));
        auto &band = bands.back();
        if (step == 2)
        {
            band.shape                                 = channeltest::Shape(bandLayout, samples, rows, width, 1);
            band.buffer.strides[band.shape.rank() - 2] = 2 * bytes;
            band.buffer.strides[band.shape.rank() - 1] = 2 * bytes;
            band.tensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(band.shape, dtype, band.buffer));
        }
        band.Clear(stream.handle);
        band.Put(SplitCpuGold(pixels, channels, channel, bytes), samples, rows, width, 1);
        outputs.pushBack(band.tensor);
    }
    op(stream.handle, image.tensor, outputs);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    EXPECT_EQ(image.Download(), image.host);
    for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
}

TEST_P(OpSplitPlanar, byte_vectors_independent_strides_and_fallbacks)
{
    const auto &[dtype, channels, layout] = GetParam();
    for (const auto &[rows, width, padding, offset, badAlignment, spaced] : {
             std::tuple{3,   16,  0, 0, false, false},
             {3,   17, 15, 0, false, false},
             {2,  257,  0, 1, false, false},
             {1, 4097,  0, 1, false, false},
             {3,  512, 16, 0,  true, false},
             {3,  512, 16, 0, false,  true},
             {1,    1, 16, 0, false, false},
             {3,  512, 16, 1, false, false},
             {1,  511,  0, 0, false, false},
             {1,  512,  0, 0, false, false},
             {1,  513,  0, 0, false, false}
    })
    {
        SCOPED_TRACE(::testing::PrintToString(std::make_tuple(rows, width, padding, offset, badAlignment, spaced)));
        CheckPlanarSplit(dtype, channels, layout, layout.rank() == 4 ? 3 : 1, rows, width, padding, offset,
                         badAlignment, spaced);
    }
    for (int badStride : {1, 2})
        CheckPlanarSplit(dtype, channels, layout, layout.rank() == 4 ? 3 : 1, 3, 512, 16, 0, false, false, badStride);
    for (const auto &[rows, width] : {
             std::tuple{17, 31},
             {17, 32},
             {17, 33},
             {33, 23},
             {33, 24},
             {33, 25}
    })
        CheckPlanarSplit(dtype, channels, layout, layout.rank() == 4 ? 3 : 1, rows, width, 0, 1, false, false, 0, true);
}

TEST(OpSplitPlanarTensor, bounded_grid_multiple_iterations)
{
    CheckPlanarSplit(nvcv::TYPE_U8, 1, nvcv::TENSOR_NCHW, 2, 1, 256 * 256 * 16 + 17, 0, 0, false, false);
    CheckPlanarSplit(nvcv::TYPE_F32, 3, nvcv::TENSOR_NCHW, 1025, 1, 128, 0, 0, false, false);
}

TEST(OpSplit_Negative, rejects_null_and_wrong_band_count)
{
    cvcuda::Split     op;
    nvcv::Tensor      image(channeltest::Shape(nvcv::TENSOR_HWC, 1, 3, 7, 3), nvcv::TYPE_U8);
    nvcv::TensorBatch bands(4);
    EXPECT_EQ(cvcudaSplitCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaSplitSubmit(nullptr, nullptr, image.handle(), bands.handle()), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaSplitSubmit(op.handle(), nullptr, nullptr, bands.handle()), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaSplitSubmit(op.handle(), nullptr, image.handle(), nullptr), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaSplitSubmit(op.handle(), nullptr, image.handle(), bands.handle()), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpSplitTensor, image_format_constructor_channels)
{
    channeltest::Stream stream;
    cvcuda::Split       op;
    for (const auto format : {nvcv::FMT_U8, nvcv::FMT_2F32, nvcv::FMT_RGB8, nvcv::FMT_RGBA8})
    {
        const int    channels = format.numChannels();
        const auto   dtype    = format.planeDataType(0).channelType(0);
        const int    bytes    = dtype.strideBytes();
        nvcv::Tensor image(1, {7, 2}, format);
        const auto   data = image.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(data);
        ASSERT_EQ(image.layout(), nvcv::TENSOR_NHWC);
        ASSERT_EQ(image.dtype(), dtype);
        const auto           pixels   = channeltest::Patterns(2 * 7 * channels, bytes);
        const size_t         rowBytes = 7 * channels * bytes;
        std::vector<uint8_t> source(data->stride(0), 0xA5);
        for (int row = 0; row < 2; ++row)
            std::copy_n(pixels.begin() + row * rowBytes, rowBytes, source.begin() + row * data->stride(1));
        NVCV_CHECK_THROW(
            cudaMemcpyAsync(data->basePtr(), source.data(), source.size(), cudaMemcpyHostToDevice, stream.handle));
        nvcv::TensorBatch                outputs(channels);
        std::vector<channeltest::Buffer> bands;
        for (int channel = 0; channel < channels; ++channel)
        {
            bands.emplace_back(nvcv::TENSOR_NHWC, dtype, 1, 2, 7, 1);
            bands.back().Clear(stream.handle);
            bands.back().Put(SplitCpuGold(pixels, channels, channel, bytes), 1, 2, 7, 1);
            outputs.pushBack(bands.back().tensor);
        }
        op(stream.handle, image, outputs);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
        std::vector<uint8_t> actual(source.size());
        NVCV_CHECK_THROW(cudaMemcpy(actual.data(), data->basePtr(), actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_EQ(actual, source);
    }
}

TEST(OpSplit_Negative, rejects_dtype_layout_channels_and_shape_mismatches)
{
    cvcuda::Split     op;
    nvcv::Tensor      band(channeltest::Shape(nvcv::TENSOR_HWC, 1, 3, 7, 1), nvcv::TYPE_U8);
    nvcv::TensorBatch bands(1);
    bands.pushBack(band);
    const std::vector<nvcv::Tensor> inputs
        = {nvcv::Tensor(channeltest::Shape(nvcv::TensorLayout{"WHC"}, 1, 3, 7, 1), nvcv::TYPE_U8),
           nvcv::Tensor(channeltest::Shape(nvcv::TENSOR_HWC, 1, 3, 7, 5), nvcv::TYPE_U8),
           nvcv::Tensor(channeltest::Shape(nvcv::TENSOR_HWC, 1, 3, 7, 1), nvcv::TYPE_U16),
           nvcv::Tensor(channeltest::Shape(nvcv::TENSOR_HWC, 1, 4, 7, 1), nvcv::TYPE_U8),
           nvcv::Tensor(channeltest::Shape(nvcv::TENSOR_NHWC, 1, 3, 7, 1), nvcv::TYPE_U8)};
    for (const auto &input : inputs)
        EXPECT_EQ(cvcudaSplitSubmit(op.handle(), nullptr, input.handle(), bands.handle()), NVCV_ERROR_INVALID_ARGUMENT);
    nvcv::Tensor unsupportedInput(band.shape(), nvcv::TYPE_S64);
    nvcv::Tensor unsupportedBand(band.shape(), nvcv::TYPE_S64);
    bands.clear();
    bands.pushBack(unsupportedBand);
    EXPECT_EQ(cvcudaSplitSubmit(op.handle(), nullptr, unsupportedInput.handle(), bands.handle()),
              NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpSplit_Negative, rejects_aliasing_and_multichannel_destinations_before_writing)
{
    channeltest::Stream stream;
    cvcuda::Split       op;
    channeltest::Buffer image(nvcv::TENSOR_HWC, nvcv::TYPE_U8, 1, 3, 7, 2);
    channeltest::Buffer band(nvcv::TENSOR_HWC, nvcv::TYPE_U8, 1, 3, 7, 1);
    image.Put(channeltest::Patterns(3 * 7 * 2, 1), 1, 3, 7, 2);
    image.Upload(stream.handle);
    band.Upload(stream.handle);
    nvcv::TensorBatch outputs(2);
    outputs.pushBack(band.tensor);
    outputs.pushBack(band.tensor);
    EXPECT_EQ(cvcudaSplitSubmit(op.handle(), stream.handle, image.tensor.handle(), outputs.handle()),
              NVCV_ERROR_INVALID_ARGUMENT);
    outputs.clear();
    outputs.pushBack(band.tensor);
    EXPECT_EQ(cvcudaSplitSubmit(op.handle(), stream.handle, band.tensor.handle(), outputs.handle()),
              NVCV_ERROR_INVALID_ARGUMENT);
    outputs.pushBack(image.tensor);
    EXPECT_EQ(cvcudaSplitSubmit(op.handle(), stream.handle, image.tensor.handle(), outputs.handle()),
              NVCV_ERROR_INVALID_ARGUMENT);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    EXPECT_EQ(image.Download(), image.host);
    EXPECT_EQ(band.Download(), band.host);
}

TEST(OpSplitTensor, wrapped_output_output_span_aliasing)
{
    channeltest::Stream stream;
    cvcuda::Split       op;
    for (const size_t byteOffset : {1, 2, 36, 64})
    {
        SCOPED_TRACE(byteOffset);
        channeltest::Buffer image(nvcv::TENSOR_HWC, nvcv::TYPE_U8, 1, 2, 2, 2);
        channeltest::Buffer band(nvcv::TENSOR_HWC, nvcv::TYPE_U8, 1, 2, 2, 1);
        const auto          pixels = channeltest::Patterns(8, 1);
        image.Put(pixels, 1, 2, 2, 2);
        image.Upload(stream.handle);
        band.Upload(stream.handle);
        const auto        destination = band.View(byteOffset);
        nvcv::TensorBatch outputs(2);
        outputs.pushBack(band.tensor);
        outputs.pushBack(destination);
        ASSERT_NE(band.tensor.handle(), destination.handle());
        EXPECT_EQ(cvcudaSplitSubmit(op.handle(), stream.handle, image.tensor.handle(), outputs.handle()),
                  byteOffset >= 36 ? NVCV_SUCCESS : NVCV_ERROR_INVALID_ARGUMENT);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        if (byteOffset >= 36)
        {
            band.Put(SplitCpuGold(pixels, 2, 0, 1), 1, 2, 2, 1);
            band.Put(SplitCpuGold(pixels, 2, 1, 1), 1, 2, 2, 1, byteOffset);
        }
        EXPECT_EQ(image.Download(), image.host);
        EXPECT_EQ(band.Download(), band.host);
    }
}

NVCV_TEST_SUITE_P(
    OpSplitVarShape,
    nvcv::test::Combine(nvcv::test::ValueList<nvcv::DataType>{nvcv::TYPE_U8, nvcv::TYPE_S8, nvcv::TYPE_U16,
                                                              nvcv::TYPE_S16, nvcv::TYPE_U32, nvcv::TYPE_S32,
                                                              nvcv::TYPE_F16, nvcv::TYPE_F32, nvcv::TYPE_F64},
                        nvcv::test::ValueList<int>{1, 2, 3, 4}, nvcv::test::ValueList<bool>{false, true}));

TEST_P(OpSplitVarShape, varshape_correct_output_matches_interleaved_and_tensor_cpu_gold)
{
    const auto &[dtype, channels, planar] = GetParam();
    channeltest::Stream               stream;
    cvcuda::Split                     op;
    const std::array<nvcv::Size2D, 8> sizes = {
        {{1, 1}, {7, 5}, {8, 3}, {9, 7}, {17, 3}, {257, 5}, {513, 1}, {33, 259}}
    };
    std::vector<channeltest::ImageBuffer> images;
    std::vector<channeltest::ImageBuffer> parityImages;
    std::vector<channeltest::ImageBuffer> bands;
    std::vector<channeltest::ImageBuffer> parityBands;
    nvcv::ImageBatchVarShape              input(sizes.size());
    nvcv::ImageBatchVarShape              parityInput(sizes.size());
    std::vector<nvcv::ImageBatchVarShape> outputs;
    std::vector<nvcv::ImageBatchVarShape> parityOutputs;
    for (int channel = 0; channel < channels; ++channel)
    {
        outputs.emplace_back(sizes.size());
        parityOutputs.emplace_back(sizes.size());
    }
    for (size_t index = 0; index < sizes.size(); ++index)
    {
        const auto [width, height] = sizes[index];
        const auto pixels
            = channeltest::Patterns(width * height * channels, dtype.strideBytes(), static_cast<int>(index) * 29);
        images.emplace_back(channeltest::ImageFormat(dtype, channels, planar), width, height, index);
        parityImages.emplace_back(channeltest::ImageFormat(dtype, channels, !planar), width, height, index + 3);
        images.back().Put(pixels);
        parityImages.back().Put(pixels);
        images.back().Upload(stream.handle);
        parityImages.back().Upload(stream.handle);
        input.pushBack(images.back().image);
        parityInput.pushBack(parityImages.back().image);
        for (int channel = 0; channel < channels; ++channel)
        {
            bands.emplace_back(channeltest::ImageFormat(dtype, 1, false), width, height, index + channel + 5, 0xD7);
            parityBands.emplace_back(channeltest::ImageFormat(dtype, 1, false), width, height, index + channel + 9,
                                     0xE9);
            bands.back().Upload(stream.handle);
            parityBands.back().Upload(stream.handle);
            const auto gold = SplitCpuGold(pixels, channels, channel, dtype.strideBytes());
            bands.back().Put(gold);
            parityBands.back().Put(gold);
            outputs[channel].pushBack(bands.back().image);
            parityOutputs[channel].pushBack(parityBands.back().image);
        }
        if (planar)
        {
            continue;
        }
        for (const auto layout : {nvcv::TENSOR_HWC, nvcv::TENSOR_CHW, nvcv::TENSOR_NHWC, nvcv::TENSOR_NCHW})
        {
            channeltest::Buffer tensor(layout, dtype, 1, height, width, channels);
            tensor.Put(pixels, 1, height, width, channels);
            tensor.Upload(stream.handle);
            std::vector<channeltest::Buffer> tensorBands;
            nvcv::TensorBatch                tensorOutputs(channels);
            for (int channel = 0; channel < channels; ++channel)
            {
                tensorBands.emplace_back(layout, dtype, 1, height, width, 1);
                tensorBands.back().Clear(stream.handle);
                tensorBands.back().Put(SplitCpuGold(pixels, channels, channel, dtype.strideBytes()), 1, height, width,
                                       1);
                tensorOutputs.pushBack(tensorBands.back().tensor);
            }
            op(stream.handle, tensor.tensor, tensorOutputs);
            NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
            for (const auto &band : tensorBands) EXPECT_EQ(band.Download(), band.host);
        }
    }
    op(stream.handle, input, outputs.data(), channels);
    op(stream.handle, parityInput, parityOutputs.data(), channels);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    for (size_t index = 0; index < bands.size(); ++index)
    {
        EXPECT_EQ(bands[index].Download(), bands[index].host);
        EXPECT_EQ(parityBands[index].Download(), parityBands[index].host);
        EXPECT_EQ(bands[index].Pixels(), parityBands[index].Pixels());
    }
    for (const auto &image : images) EXPECT_EQ(image.Download(), image.host);
    for (const auto &image : parityImages) EXPECT_EQ(image.Download(), image.host);
}

TEST(OpSplitVarShape, repeated_read_only_images_and_physical_bgr_order)
{
    channeltest::Stream stream;
    cvcuda::Split       op;
    for (const auto format : {nvcv::FMT_BGR8, nvcv::FMT_BGR8p, nvcv::FMT_BGRA8})
    {
        const int                channels = format.numChannels();
        channeltest::ImageBuffer source(format, 17, 3);
        const auto               pixels = channeltest::Patterns(17 * 3 * channels, 1);
        source.Put(pixels);
        source.Upload(stream.handle);
        auto                                  input = channeltest::Batch({source.image, source.View()});
        std::vector<channeltest::ImageBuffer> storage;
        std::vector<nvcv::ImageBatchVarShape> outputs;
        for (int channel = 0; channel < channels; ++channel)
        {
            outputs.emplace_back(2);
            for (int index = 0; index < 2; ++index)
            {
                storage.emplace_back(channel % 2 == 0 ? nvcv::FMT_U8 : nvcv::FMT_Y8, 17, 3, channel + index);
                storage.back().Upload(stream.handle);
                storage.back().Put(SplitCpuGold(pixels, channels, channel, 1));
                outputs.back().pushBack(storage.back().image);
            }
        }
        op(stream.handle, input, outputs.data(), channels);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        for (const auto &band : storage) EXPECT_EQ(band.Download(), band.host);
        EXPECT_EQ(source.Download(), source.host);
    }
}

TEST(OpSplit_Negative, varshape_validates_all_metadata_before_writing)
{
    channeltest::Stream                             stream;
    cvcuda::Split                                   op;
    channeltest::ImageBuffer                        image(channeltest::ImageFormat(nvcv::TYPE_U8, 2, false), 7, 3);
    channeltest::ImageBuffer                        first(nvcv::FMT_U8, 7, 3);
    channeltest::ImageBuffer                        second(nvcv::FMT_U8, 7, 3);
    channeltest::ImageBuffer                        third(nvcv::FMT_U8, 7, 3);
    channeltest::ImageBuffer                        fourth(nvcv::FMT_U8, 7, 3);
    channeltest::ImageBuffer                        wrongSize(nvcv::FMT_U8, 9, 3);
    channeltest::ImageBuffer                        wrongType(nvcv::FMT_S8, 7, 3);
    channeltest::ImageBuffer                        wrongFormat(nvcv::FMT_Y8, 7, 3);
    channeltest::ImageBuffer                        planar(channeltest::ImageFormat(nvcv::TYPE_U8, 2, true), 7, 3);
    const std::array<channeltest::ImageBuffer *, 9> storage
        = {&image, &first, &second, &third, &fourth, &wrongSize, &wrongType, &wrongFormat, &planar};
    image.Put(channeltest::Patterns(42, 1));
    for (const auto buffer : storage) buffer->Upload(stream.handle);
    auto input        = channeltest::Batch({image.image, image.image});
    auto good         = channeltest::Batch({first.image, second.image});
    auto otherGood    = channeltest::Batch({third.image, fourth.image});
    auto empty        = channeltest::Batch({});
    auto shortBatch   = channeltest::Batch({first.image});
    auto sizes        = channeltest::Batch({third.image, wrongSize.image});
    auto types        = channeltest::Batch({third.image, wrongType.image});
    auto formats      = channeltest::Batch({third.image, wrongFormat.image});
    auto multiChannel = channeltest::Batch({planar.image, planar.image});
    auto check        = [&](const nvcv::ImageBatchVarShape &source, const NVCVImageBatchHandle *outputs, int count)
    {
        EXPECT_EQ(cvcudaSplitVarShapeSubmit(op.handle(), stream.handle, source.handle(), outputs, count),
                  NVCV_ERROR_INVALID_ARGUMENT);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        for (const auto buffer : storage) EXPECT_EQ(buffer->Download(), buffer->host);
    };
    std::array<NVCVImageBatchHandle, 5> handles
        = {good.handle(), otherGood.handle(), good.handle(), good.handle(), good.handle()};
    for (const int count : {-1, 0, 1, 3, 5}) check(input, handles.data(), count);
    check(input, nullptr, 2);
    check(empty, handles.data(), 2);
    check(channeltest::Batch({image.image, planar.image}), handles.data(), 2);
    for (const auto &bad : {empty, shortBatch, sizes, types, formats, multiChannel})
    {
        handles[1] = bad.handle();
        check(input, handles.data(), 2);
    }
    handles[1] = nullptr;
    check(input, handles.data(), 2);
    handles[1] = otherGood.handle();
    EXPECT_EQ(cvcudaSplitVarShapeSubmit(op.handle(), stream.handle, nullptr, handles.data(), 2),
              NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaSplitVarShapeSubmit(nullptr, stream.handle, input.handle(), handles.data(), 2),
              NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpSplit_Negative, varshape_unsupported_image_storage)
{
    channeltest::Stream stream;
    cvcuda::Split       op;
    for (const auto format : channeltest::InvalidImageFormats())
    {
        SCOPED_TRACE(format);
        channeltest::ImageBuffer source(format, 8, 6, 0, 0xC3);
        source.Upload(stream.handle);
        auto                                  input = channeltest::Batch({source.image});
        std::vector<channeltest::ImageBuffer> storage;
        std::vector<nvcv::ImageBatchVarShape> outputs;
        std::vector<NVCVImageBatchHandle>     handles;
        for (int channel = 0; channel < format.numChannels(); ++channel)
        {
            storage.emplace_back(nvcv::FMT_U8, 8, 6, channel);
            storage.back().Upload(stream.handle);
            outputs.push_back(channeltest::Batch({storage.back().image}));
            handles.push_back(outputs.back().handle());
        }
        EXPECT_EQ(
            cvcudaSplitVarShapeSubmit(op.handle(), stream.handle, input.handle(), handles.data(), format.numChannels()),
            NVCV_ERROR_INVALID_ARGUMENT);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        EXPECT_EQ(source.Download(), source.host);
        for (const auto &band : storage) EXPECT_EQ(band.Download(), band.host);
    }
}

TEST(OpSplit_Negative, varshape_cross_batch_cross_index_write_overlap)
{
    channeltest::Stream                   stream;
    cvcuda::Split                         op;
    channeltest::ImageBuffer              source(channeltest::ImageFormat(nvcv::TYPE_U8, 2, true), 7, 3);
    std::vector<channeltest::ImageBuffer> storage;
    source.Put(channeltest::Patterns(42, 1));
    for (int index = 0; index < 4; ++index)
    {
        storage.emplace_back(nvcv::FMT_U8, 7, 3);
        storage.back().Upload(stream.handle);
    }
    source.Upload(stream.handle);
    auto                                      input   = channeltest::Batch({source.image, source.image});
    auto                                      first   = channeltest::Batch({storage[0].image, storage[1].image});
    auto                                      second  = channeltest::Batch({storage[2].image, storage[0].View(1)});
    const std::array<NVCVImageBatchHandle, 2> handles = {first.handle(), second.handle()};
    EXPECT_EQ(cvcudaSplitVarShapeSubmit(op.handle(), stream.handle, input.handle(), handles.data(), 2),
              NVCV_ERROR_INVALID_ARGUMENT);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    EXPECT_EQ(source.Download(), source.host);
    for (const auto &band : storage) EXPECT_EQ(band.Download(), band.host);
}

TEST(OpSplitVarShape, overlapping_read_only_planes_are_allowed)
{
    channeltest::Stream      stream;
    cvcuda::Split            op;
    channeltest::ImageBuffer source(channeltest::ImageFormat(nvcv::TYPE_F32, 2, true), 17, 3);
    channeltest::ImageBuffer first(nvcv::FMT_F32, 17, 3);
    channeltest::ImageBuffer second(nvcv::FMT_F32, 17, 3);
    const auto               pixels = channeltest::Patterns(102, 4);
    source.Put(pixels);
    source.Upload(stream.handle);
    first.Upload(stream.handle);
    second.Upload(stream.handle);
    auto metadata      = source.buffer;
    metadata.planes[1] = metadata.planes[0];
    auto wrapped       = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda(source.format, metadata));
    auto input         = channeltest::Batch({wrapped});
    std::array<nvcv::ImageBatchVarShape, 2> outputs
        = {channeltest::Batch({first.image}), channeltest::Batch({second.image})};
    const auto gold = SplitCpuGold(pixels, 2, 0, 4);
    first.Put(gold);
    second.Put(gold);
    op(stream.handle, input, outputs.data(), outputs.size());
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    EXPECT_EQ(first.Download(), first.host);
    EXPECT_EQ(second.Download(), second.host);
    EXPECT_EQ(source.Download(), source.host);
}

TEST(OpSplit_Negative, varshape_checks_every_planar_plane_before_writes)
{
    channeltest::Stream      stream;
    cvcuda::Split            op;
    channeltest::ImageBuffer source(channeltest::ImageFormat(nvcv::TYPE_F64, 2, true), 7, 3);
    channeltest::ImageBuffer first(nvcv::FMT_F64, 7, 3);
    channeltest::ImageBuffer second(nvcv::FMT_F64, 7, 3);
    source.Put(channeltest::Patterns(42, 8));
    for (const auto buffer : {&source, &first, &second}) buffer->Upload(stream.handle);
    auto                                      firstBatch  = channeltest::Batch({first.image});
    auto                                      secondBatch = channeltest::Batch({second.image});
    const std::array<NVCVImageBatchHandle, 2> handles     = {firstBatch.handle(), secondBatch.handle()};
    for (int invalid = 0; invalid < 5; ++invalid)
    {
        SCOPED_TRACE(invalid);
        auto metadata = source.buffer;
        if (invalid == 0)
            --metadata.planes[1].width;
        if (invalid == 1)
            --metadata.planes[1].height;
        if (invalid == 2)
            ++metadata.planes[1].basePtr;
        if (invalid == 3)
            ++metadata.planes[1].rowStride;
        if (invalid == 4)
            --metadata.numPlanes;
        auto wrapped = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda(source.format, metadata));
        auto input   = channeltest::Batch({wrapped});
        EXPECT_EQ(cvcudaSplitVarShapeSubmit(op.handle(), stream.handle, input.handle(), handles.data(), 2),
                  NVCV_ERROR_INVALID_ARGUMENT);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        for (const auto buffer : {&source, &first, &second}) EXPECT_EQ(buffer->Download(), buffer->host);
    }
}

TEST(OpSplit_Negative, varshape_cuda_array_rejected_by_image_container)
{
    channeltest::CheckCudaArrayRejected();
}

TEST(OpSplit_Negative, varshape_rejects_1x1_subsampling_even_when_plane_sizes_match)
{
    const nvcv::ImageFormat format(nvcv::CSPEC_BT601, nvcv::ChromaSubsampling::CSS_420, nvcv::MemLayout::PITCH_LINEAR,
                                   nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XYZ0, nvcv::Packing::X8,
                                   nvcv::Packing::X8, nvcv::Packing::X8);
    ASSERT_EQ(format.numPlanes(), 3);
    ASSERT_EQ(format.chromaSubsampling(), nvcv::ChromaSubsampling::CSS_420);
    for (int plane = 0; plane < 3; ++plane) ASSERT_EQ(format.planeSize({1, 1}, plane), nvcv::Size2D(1, 1));
    channeltest::Stream      stream;
    cvcuda::Split            op;
    channeltest::ImageBuffer source(format, 1, 1);
    source.Put({0x11, 0x42, 0x79});
    source.Upload(stream.handle);
    auto                                  input = channeltest::Batch({source.image});
    std::vector<channeltest::ImageBuffer> storage;
    std::vector<nvcv::ImageBatchVarShape> outputs;
    std::vector<NVCVImageBatchHandle>     handles;
    for (int channel = 0; channel < 3; ++channel)
    {
        storage.emplace_back(nvcv::FMT_U8, 1, 1, channel);
        storage.back().Upload(stream.handle);
        outputs.push_back(channeltest::Batch({storage.back().image}));
        handles.push_back(outputs.back().handle());
    }
    EXPECT_EQ(cvcudaSplitVarShapeSubmit(op.handle(), stream.handle, input.handle(), handles.data(), 3),
              NVCV_ERROR_INVALID_ARGUMENT);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    EXPECT_EQ(source.Download(), source.host);
    for (const auto &band : storage) EXPECT_EQ(band.Download(), band.host);
}

NVCV_TEST_SUITE_P(OpSplitVarShapePacked,
                  nvcv::test::Combine(nvcv::test::ValueList<nvcv::DataType>{nvcv::TYPE_U8, nvcv::TYPE_S8},
                                      nvcv::test::ValueList<int>{3, 4, 5, 7, 8, 9, 127, 128, 129, 4097, 8193}));

TEST_P(OpSplitVarShapePacked, aligned_c3_single_row_matches_cpu_gold_and_preserves_canaries)
{
    const auto &[dtype, width] = GetParam();
    channeltest::Stream      stream;
    cvcuda::Split            op;
    channeltest::ImageBuffer image(channeltest::ImageFormat(dtype, 3, false), width, 1, 5, 0xC3);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(image.buffer.planes[0].basePtr) % 8, 0u);
    const auto pixels = channeltest::Patterns(width * 3, 1);
    image.Put(pixels);
    image.Upload(stream.handle);
    auto                                  input = channeltest::Batch({image.image});
    std::vector<channeltest::ImageBuffer> bands;
    std::vector<nvcv::ImageBatchVarShape> outputs;
    bands.reserve(3);
    for (int channel = 0; channel < 3; ++channel)
    {
        bands.emplace_back(channeltest::ImageFormat(dtype, 1, false), width, 1, 5, 0x31 + channel * 0x11);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(bands.back().buffer.planes[0].basePtr) % 8, 0u);
        bands.back().Upload(stream.handle);
        bands.back().Put(SplitCpuGold(pixels, 3, channel, 1));
        outputs.push_back(channeltest::Batch({bands.back().image}));
    }
    op(stream.handle, input, outputs.data(), static_cast<int32_t>(outputs.size()));
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    EXPECT_EQ(image.Download(), image.host);
    for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
}

} // namespace

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ChannelCopyTestUtils.hpp"

#include <cvcuda/OpMerge.hpp>
#include <cvcuda/OpSplit.hpp>

namespace {

std::vector<uint8_t> MergeCpuGold(const std::vector<std::vector<uint8_t>> &bands, int bytes)
{
    std::vector<uint8_t> result(bands.front().size() * bands.size());
    for (size_t pixel = 0; pixel < bands.front().size() / bytes; ++pixel)
        for (size_t channel = 0; channel < bands.size(); ++channel)
            std::copy_n(bands[channel].begin() + pixel * bytes, bytes,
                        result.begin() + (pixel * bands.size() + channel) * bytes);
    return result;
}

NVCV_TEST_SUITE_P(
    OpMerge, nvcv::test::Combine(nvcv::test::ValueList<nvcv::DataType>{nvcv::TYPE_U8, nvcv::TYPE_S8, nvcv::TYPE_U16,
                                                                       nvcv::TYPE_S16, nvcv::TYPE_U32, nvcv::TYPE_S32,
                                                                       nvcv::TYPE_F16, nvcv::TYPE_F32, nvcv::TYPE_F64},
                                 nvcv::test::ValueList<int>{1, 2, 3, 4},
                                 nvcv::test::ValueList<nvcv::TensorLayout>{nvcv::TENSOR_HWC, nvcv::TENSOR_NHWC,
                                                                           nvcv::TENSOR_CHW, nvcv::TENSOR_NCHW},
                                 nvcv::test::ValueList<bool>{false, true}));

TEST_P(OpMerge, matches_interleaved_cpu_gold_and_preserves_padding)
{
    const auto &[dtype, channels, layout, crossLayout] = GetParam();
    const auto          imageLayout                    = crossLayout ? channeltest::OtherLayout(layout) : layout;
    const int           samples                        = layout.rank() == 4 ? 3 : 1;
    channeltest::Stream stream;
    cvcuda::Merge       op;
    for (const int width : {1, 17, 257})
    {
        channeltest::Buffer image(imageLayout, dtype, samples, 3, width, channels);
        if (width == 1)
            image.CompactSingletonStrides();
        image.Clear(stream.handle);
        std::vector<channeltest::Buffer>  bands;
        std::vector<std::vector<uint8_t>> pixels;
        nvcv::TensorBatch                 inputs(channels);
        for (int channel = 0; channel < channels; ++channel)
        {
            pixels.push_back(channeltest::Patterns(samples * 3 * width, image.bytes, channel * 17));
            bands.emplace_back(layout, dtype, samples, 3, width, 1, 64 + 8 * channel);
            bands.back().Put(pixels.back(), samples, 3, width, 1);
            bands.back().Upload(stream.handle);
            inputs.pushBack(bands.back().tensor);
        }
        image.Put(MergeCpuGold(pixels, image.bytes), samples, 3, width, channels);
        op(stream.handle, inputs, image.tensor);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        EXPECT_EQ(image.Download(), image.host);
        for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
    }
}

NVCV_TEST_SUITE_P(OpMergePacked, nvcv::test::Combine(nvcv::test::ValueList<int>{1, 3, 4, 5, 127, 128, 129, 511, 512,
                                                                                513, 2047, 2048, 2049, 1048575, 1048576,
                                                                                1048577, 5999999, 6000000, 6000001},
                                                     nvcv::test::ValueList<int>{0, 1, 4, 8}));

TEST_P(OpMergePacked, vector_boundaries_alignment_and_repeated_sources)
{
    const auto &[width, offset] = GetParam();
    channeltest::Stream stream;
    cvcuda::Merge       op;
    channeltest::Buffer image(nvcv::TENSOR_NHWC, nvcv::TYPE_U8, 1, 1, width, 3, 0, 64 + offset, 0xE7);
    image.CompactSingletonStrides();
    std::vector<channeltest::Buffer>  bands;
    std::vector<std::vector<uint8_t>> pixels;
    for (int channel = 0; channel < 3; ++channel)
    {
        pixels.push_back(channeltest::Patterns(width, 1, channel * 17));
        bands.emplace_back(nvcv::TENSOR_NHWC, nvcv::TYPE_U8, 1, 1, width, 1, 0, 64 + (channel + 2) * offset,
                           0x31 + channel * 0x11);
        bands.back().CompactSingletonStrides();
        bands.back().Put(pixels.back(), 1, 1, width, 1);
        bands.back().Upload(stream.handle);
    }
    for (const bool repeated : {false, true})
    {
        SCOPED_TRACE(repeated);
        image.Clear(stream.handle);
        nvcv::TensorBatch inputs(3);
        for (int channel = 0; channel < 3; ++channel) inputs.pushBack(bands[repeated ? 0 : channel].tensor);
        const auto expected = repeated ? std::vector<std::vector<uint8_t>>(3, pixels.front()) : pixels;
        image.Put(MergeCpuGold(expected, 1), 1, 1, width, 3);
        op(stream.handle, inputs, image.tensor);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        EXPECT_EQ(image.Download(), image.host);
        for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
    }
}

NVCV_TEST_SUITE_P(OpMergeStrided,
                  nvcv::test::Combine(nvcv::test::ValueList<nvcv::DataType>{nvcv::TYPE_U8, nvcv::TYPE_U16,
                                                                            nvcv::TYPE_U32, nvcv::TYPE_F64},
                                      nvcv::test::ValueList<int>{1, 2, 3, 4},
                                      nvcv::test::ValueList<nvcv::TensorLayout>{nvcv::TENSOR_HWC, nvcv::TENSOR_NHWC,
                                                                                nvcv::TENSOR_CHW, nvcv::TENSOR_NCHW}));

TEST_P(OpMergeStrided, independent_band_strides_and_scalar_aligned_bases)
{
    const auto &[dtype, channels, layout] = GetParam();

    const int samples = layout.rank() == 4 ? 3 : 1;
    const int bytes   = dtype.strideBytes();

    channeltest::Stream stream;
    cvcuda::Merge       op;
    for (const auto &[width, crossLayout] : {
             std::tuple{ 17, false},
             std::tuple{257,  true}
    })
    {
        SCOPED_TRACE(width);
        const auto          bandLayout = crossLayout ? channeltest::OtherLayout(layout) : layout;
        channeltest::Buffer image(layout, dtype, samples, 3, width, channels, 3 * bytes, 64 + bytes);
        image.Clear(stream.handle);
        std::vector<channeltest::Buffer>  bands;
        std::vector<std::vector<uint8_t>> pixels;
        nvcv::TensorBatch                 inputs(channels);
        for (int channel = 0; channel < channels; ++channel)
        {
            pixels.push_back(channeltest::Patterns(samples * 3 * width, bytes, channel * 17));
            bands.emplace_back(bandLayout, dtype, samples, 3, width, 1, (5 + 3 * channel) * bytes,
                               64 + (2 * channel + 1) * bytes);
            bands.back().Put(pixels.back(), samples, 3, width, 1);
            bands.back().Upload(stream.handle);
            inputs.pushBack(bands.back().tensor);
        }
        image.Put(MergeCpuGold(pixels, bytes), samples, 3, width, channels);
        op(stream.handle, inputs, image.tensor);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        EXPECT_EQ(image.Download(), image.host);
        for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
    }
}

NVCV_TEST_SUITE_P(OpMergePlanar, nvcv::test::Combine(
                                     nvcv::test::ValueList<nvcv::DataType>{
                                         nvcv::TYPE_U8, nvcv::TYPE_S8, nvcv::TYPE_U16, nvcv::TYPE_S16, nvcv::TYPE_U32,
                                         nvcv::TYPE_S32, nvcv::TYPE_F16, nvcv::TYPE_F32, nvcv::TYPE_F64},
                                     nvcv::test::ValueList<int>{1, 2, 3, 4},
                                     nvcv::test::ValueList<nvcv::TensorLayout>{nvcv::TENSOR_CHW, nvcv::TENSOR_NCHW}));

void CheckPlanarMerge(nvcv::DataType dtype, int channels, nvcv::TensorLayout layout, int samples, int rows, int width,
                      int padding, int offset, bool badAlignment, bool spaced, int badStride = 0, bool dense = false)
{
    const int           bytes = dtype.strideBytes();
    channeltest::Stream stream;
    cvcuda::Merge       op;
    channeltest::Buffer image(layout, dtype, samples, rows, width, channels, padding * bytes, 64 + offset * bytes, 0xC3,
                              32);
    image.Clear(stream.handle);
    nvcv::TensorBatch                 inputs(channels);
    std::vector<channeltest::Buffer>  bands;
    std::vector<std::vector<uint8_t>> pixels;
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
        pixels.push_back(channeltest::Patterns(static_cast<size_t>(samples) * rows * width, bytes, channel * 17));
        band.Put(pixels.back(), samples, rows, width, 1);
        band.Upload(stream.handle);
        inputs.pushBack(band.tensor);
    }
    image.Put(MergeCpuGold(pixels, bytes), samples, rows, width, channels);
    op(stream.handle, inputs, image.tensor);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    EXPECT_EQ(image.Download(), image.host);
    for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
}

TEST_P(OpMergePlanar, byte_vectors_independent_strides_and_fallbacks)
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
        CheckPlanarMerge(dtype, channels, layout, layout.rank() == 4 ? 3 : 1, rows, width, padding, offset,
                         badAlignment, spaced);
    }
    for (int badStride : {1, 2})
        CheckPlanarMerge(dtype, channels, layout, layout.rank() == 4 ? 3 : 1, 3, 512, 16, 0, false, false, badStride);
    for (const auto &[rows, width] : {
             std::tuple{17, 31},
             {17, 32},
             {17, 33},
             {33, 23},
             {33, 24},
             {33, 25}
    })
        CheckPlanarMerge(dtype, channels, layout, layout.rank() == 4 ? 3 : 1, rows, width, 0, 1, false, false, 0, true);
}

TEST(OpMergePlanarTensor, bounded_grid_multiple_iterations)
{
    CheckPlanarMerge(nvcv::TYPE_U8, 1, nvcv::TENSOR_NCHW, 2, 1, 256 * 256 * 16 + 17, 0, 0, false, false);
    CheckPlanarMerge(nvcv::TYPE_F32, 3, nvcv::TENSOR_NCHW, 1025, 1, 128, 0, 0, false, false);
}

void CheckC1UnusedSingletonStrides(bool splitDirection)
{
    channeltest::Stream stream;
    cvcuda::Split       split;
    cvcuda::Merge       merge;
    SCOPED_TRACE(splitDirection ? "Split" : "Merge");
    for (const auto layout : {nvcv::TENSOR_HWC, nvcv::TENSOR_CHW, nvcv::TENSOR_NHWC, nvcv::TENSOR_NCHW})
        for (const auto dtype : {nvcv::TYPE_U8, nvcv::TYPE_F64})
        {
            channeltest::Buffer source(layout, dtype, 1, 1, 3, 1);
            channeltest::Buffer destination(channeltest::OtherLayout(layout), dtype, 1, 1, 3, 1);
            source.CompactSingletonStrides();
            destination.CompactSingletonStrides();
            const auto pixels = channeltest::Patterns(3, source.bytes);
            source.Put(pixels, 1, 1, 3, 1);
            source.Upload(stream.handle);
            destination.Clear(stream.handle);
            destination.Put(pixels, 1, 1, 3, 1);
            nvcv::TensorBatch bands(1);
            bands.pushBack(splitDirection ? destination.tensor : source.tensor);
            ASSERT_EQ(
                splitDirection
                    ? cvcudaSplitSubmit(split.handle(), stream.handle, source.tensor.handle(), bands.handle())
                    : cvcudaMergeSubmit(merge.handle(), stream.handle, bands.handle(), destination.tensor.handle()),
                NVCV_SUCCESS);
            NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
            EXPECT_EQ(source.Download(), source.host);
            EXPECT_EQ(destination.Download(), destination.host);
        }
}

TEST(OpSplitTensor, accepts_unused_singleton_strides)
{
    CheckC1UnusedSingletonStrides(true);
}

TEST(OpMergeTensor, accepts_unused_singleton_strides)
{
    CheckC1UnusedSingletonStrides(false);
}

TEST(OpMergeTensor, round_trip_and_repeated_bands)
{
    channeltest::Stream stream;
    channeltest::Buffer source(nvcv::TENSOR_NHWC, nvcv::TYPE_U8, 2, 5, 7, 3);
    channeltest::Buffer result(nvcv::TENSOR_NHWC, nvcv::TYPE_U8, 2, 5, 7, 3);
    const auto          pixels = channeltest::Patterns(2 * 5 * 7 * 3, 1);
    source.Put(pixels, 2, 5, 7, 3);
    source.Upload(stream.handle);
    result.Upload(stream.handle);
    std::vector<channeltest::Buffer> bands;
    nvcv::TensorBatch                collection(3);
    for (int channel = 0; channel < 3; ++channel)
    {
        bands.emplace_back(nvcv::TENSOR_NCHW, nvcv::TYPE_U8, 2, 5, 7, 1);
        collection.pushBack(bands.back().tensor);
    }
    cvcuda::Split split;
    cvcuda::Merge merge;
    split(stream.handle, source.tensor, collection);
    merge(stream.handle, collection, result.tensor);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    result.Put(pixels, 2, 5, 7, 3);
    EXPECT_EQ(result.Download(), result.host);
    const auto gray = channeltest::Patterns(2 * 5 * 7, 1);
    bands.front().Put(gray, 2, 5, 7, 1);
    bands.front().Upload(stream.handle);
    collection.clear();
    for (int channel = 0; channel < 3; ++channel) collection.pushBack(bands.front().tensor);
    merge(stream.handle, collection, result.tensor);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    result.Put(MergeCpuGold({gray, gray, gray}, 1), 2, 5, 7, 3);
    EXPECT_EQ(result.Download(), result.host);
}

TEST(OpMerge_Negative, rejects_null_and_wrong_band_count)
{
    cvcuda::Merge     op;
    nvcv::Tensor      image(channeltest::Shape(nvcv::TENSOR_HWC, 1, 3, 7, 3), nvcv::TYPE_U8);
    nvcv::TensorBatch bands(4);
    EXPECT_EQ(cvcudaMergeCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaMergeSubmit(nullptr, nullptr, bands.handle(), image.handle()), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaMergeSubmit(op.handle(), nullptr, nullptr, image.handle()), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaMergeSubmit(op.handle(), nullptr, bands.handle(), nullptr), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaMergeSubmit(op.handle(), nullptr, bands.handle(), image.handle()), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpMergeTensor, repeated_bands_and_graph_capture)
{
    channeltest::Stream stream;
    cvcuda::Merge       op;
    for (const auto format : {nvcv::FMT_U8, nvcv::FMT_2F32, nvcv::FMT_RGB8, nvcv::FMT_RGBA8})
    {
        const int           channels = format.numChannels();
        const auto          dtype    = format.planeDataType(0).channelType(0);
        channeltest::Buffer image(nvcv::TENSOR_NHWC, dtype, 1, 2, 7, channels);
        channeltest::Buffer band(nvcv::TENSOR_NHWC, dtype, 1, 2, 7, 1);
        const auto          pixels = channeltest::Patterns(2 * 7, band.bytes);
        band.Put(pixels, 1, 2, 7, 1);
        band.Upload(stream.handle);
        image.Clear(stream.handle);
        image.Put(MergeCpuGold(std::vector<std::vector<uint8_t>>(channels, pixels), band.bytes), 1, 2, 7, channels);
        nvcv::TensorBatch inputs(channels);
        for (int channel = 0; channel < channels; ++channel) inputs.pushBack(band.tensor);
        cudaGraph_t     graph      = nullptr;
        cudaGraphExec_t executable = nullptr;
        NVCV_CHECK_THROW(cudaStreamBeginCapture(stream.handle, cudaStreamCaptureModeThreadLocal));
        op(stream.handle, inputs, image.tensor);
        NVCV_CHECK_THROW(cudaStreamEndCapture(stream.handle, &graph));
        NVCV_CHECK_THROW(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        NVCV_CHECK_THROW(cudaGraphLaunch(executable, stream.handle));
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        NVCV_CHECK_THROW(cudaGraphExecDestroy(executable));
        NVCV_CHECK_THROW(cudaGraphDestroy(graph));
        EXPECT_EQ(image.Download(), image.host);
    }
}

TEST(OpMerge_Negative, rejects_dtype_layout_channels_and_shape_mismatches)
{
    cvcuda::Merge     op;
    nvcv::Tensor      band(channeltest::Shape(nvcv::TENSOR_HWC, 1, 3, 7, 1), nvcv::TYPE_U8);
    nvcv::TensorBatch bands(1);
    bands.pushBack(band);
    const std::vector<nvcv::Tensor> outputs
        = {nvcv::Tensor(channeltest::Shape(nvcv::TensorLayout{"WHC"}, 1, 3, 7, 1), nvcv::TYPE_U8),
           nvcv::Tensor(channeltest::Shape(nvcv::TENSOR_HWC, 1, 3, 7, 5), nvcv::TYPE_U8),
           nvcv::Tensor(channeltest::Shape(nvcv::TENSOR_HWC, 1, 3, 7, 1), nvcv::TYPE_U16),
           nvcv::Tensor(channeltest::Shape(nvcv::TENSOR_HWC, 1, 4, 7, 1), nvcv::TYPE_U8),
           nvcv::Tensor(channeltest::Shape(nvcv::TENSOR_NHWC, 1, 3, 7, 1), nvcv::TYPE_U8)};
    for (const auto &output : outputs)
        EXPECT_EQ(cvcudaMergeSubmit(op.handle(), nullptr, bands.handle(), output.handle()),
                  NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaMergeSubmit(op.handle(), nullptr, bands.handle(), band.handle()), NVCV_ERROR_INVALID_ARGUMENT);
    nvcv::Tensor unsupportedBand(band.shape(), nvcv::TYPE_S64);
    nvcv::Tensor unsupportedOutput(band.shape(), nvcv::TYPE_S64);
    bands.clear();
    bands.pushBack(unsupportedBand);
    EXPECT_EQ(cvcudaMergeSubmit(op.handle(), nullptr, bands.handle(), unsupportedOutput.handle()),
              NVCV_ERROR_INVALID_ARGUMENT);
}

void CheckC1WrappedInputOutputSpanAliasing(bool splitDirection)
{
    channeltest::Stream stream;
    cvcuda::Split       split;
    cvcuda::Merge       merge;
    SCOPED_TRACE(splitDirection ? "Split" : "Merge");
    for (const size_t byteOffset : {1, 2, 36, 64})
    {
        SCOPED_TRACE(byteOffset);
        channeltest::Buffer source(nvcv::TENSOR_HWC, nvcv::TYPE_U8, 1, 2, 2, 1);
        const auto          pixels = channeltest::Patterns(4, 1);
        source.Put(pixels, 1, 2, 2, 1);
        source.Upload(stream.handle);
        const auto        destination = source.View(byteOffset);
        nvcv::TensorBatch bands(1);
        bands.pushBack(splitDirection ? destination : source.tensor);
        ASSERT_NE(source.tensor.handle(), destination.handle());
        EXPECT_EQ(splitDirection
                      ? cvcudaSplitSubmit(split.handle(), stream.handle, source.tensor.handle(), bands.handle())
                      : cvcudaMergeSubmit(merge.handle(), stream.handle, bands.handle(), destination.handle()),
                  byteOffset >= 36 ? NVCV_SUCCESS : NVCV_ERROR_INVALID_ARGUMENT);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        if (byteOffset >= 36)
            source.Put(pixels, 1, 2, 2, 1, byteOffset);
        EXPECT_EQ(source.Download(), source.host);
    }
}

TEST(OpSplitTensor, wrapped_input_output_span_aliasing)
{
    CheckC1WrappedInputOutputSpanAliasing(true);
}

TEST(OpMergeTensor, wrapped_input_output_span_aliasing)
{
    CheckC1WrappedInputOutputSpanAliasing(false);
}

void CheckC1InvalidSingletonMetadata(bool splitDirection)
{
    channeltest::Stream stream;
    cvcuda::Split       split;
    cvcuda::Merge       merge;
    SCOPED_TRACE(splitDirection ? "Split" : "Merge");
    channeltest::Buffer source(nvcv::TENSOR_CHW, nvcv::TYPE_F64, 1, 2, 2, 1);
    channeltest::Buffer destination(nvcv::TENSOR_CHW, nvcv::TYPE_F64, 1, 2, 2, 1);
    source.Put(channeltest::Patterns(4, source.bytes), 1, 2, 2, 1);
    source.Upload(stream.handle);
    destination.Upload(stream.handle);
    const std::array<std::tuple<int64_t, int>, 4> invalidMetadata = {
        {{0, 0}, {-8, 0}, {1, 0}, {8, 1}}
    };
    for (const auto &[singletonStride, baseOffset] : invalidMetadata)
        for (const bool invalidSource : {false, true})
        {
            SCOPED_TRACE(testing::Message() << "stride=" << singletonStride << ", baseOffset=" << baseOffset
                                            << ", invalidSource=" << invalidSource);
            const auto &target        = invalidSource ? source : destination;
            auto        invalidBuffer = target.buffer;
            invalidBuffer.strides[0]  = singletonStride;
            invalidBuffer.basePtr += baseOffset;
            const auto invalidTensor
                = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(target.shape, nvcv::TYPE_F64, invalidBuffer));
            const auto       &input  = invalidSource ? invalidTensor : source.tensor;
            const auto       &output = invalidSource ? destination.tensor : invalidTensor;
            nvcv::TensorBatch bands(1);
            bands.pushBack(splitDirection ? output : input);
            EXPECT_EQ(splitDirection
                          ? cvcudaSplitSubmit(split.handle(), stream.handle, input.handle(), bands.handle())
                          : cvcudaMergeSubmit(merge.handle(), stream.handle, bands.handle(), output.handle()),
                      NVCV_ERROR_INVALID_ARGUMENT);
            NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
            EXPECT_EQ(source.Download(), source.host);
            EXPECT_EQ(destination.Download(), destination.host);
        }
}

TEST(OpSplit_Negative, rejects_invalid_singleton_strides_and_unaligned_bases)
{
    CheckC1InvalidSingletonMetadata(true);
}

TEST(OpMerge_Negative, rejects_invalid_singleton_strides_and_unaligned_bases)
{
    CheckC1InvalidSingletonMetadata(false);
}

void CheckMergeTensorCpuGold(cudaStream_t stream, const cvcuda::Merge &op, nvcv::DataType dtype, int width, int height,
                             const std::vector<std::vector<uint8_t>> &pixels, const std::vector<uint8_t> &gold)
{
    const auto channels = static_cast<int>(pixels.size());
    for (const auto layout : {nvcv::TENSOR_HWC, nvcv::TENSOR_CHW, nvcv::TENSOR_NHWC, nvcv::TENSOR_NCHW})
    {
        channeltest::Buffer tensor(layout, dtype, 1, height, width, channels);
        tensor.Clear(stream);
        tensor.Put(gold, 1, height, width, channels);
        std::vector<channeltest::Buffer> tensorBands;
        nvcv::TensorBatch                tensorInputs(channels);
        for (int channel = 0; channel < channels; ++channel)
        {
            tensorBands.emplace_back(layout, dtype, 1, height, width, 1);
            tensorBands.back().Put(pixels[channel], 1, height, width, 1);
            tensorBands.back().Upload(stream);
            tensorInputs.pushBack(tensorBands.back().tensor);
        }
        op(stream, tensorInputs, tensor.tensor);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream));
        EXPECT_EQ(tensor.Download(), tensor.host);
    }
}

NVCV_TEST_SUITE_P(
    OpMergeVarShape,
    nvcv::test::Combine(nvcv::test::ValueList<nvcv::DataType>{nvcv::TYPE_U8, nvcv::TYPE_S8, nvcv::TYPE_U16,
                                                              nvcv::TYPE_S16, nvcv::TYPE_U32, nvcv::TYPE_S32,
                                                              nvcv::TYPE_F16, nvcv::TYPE_F32, nvcv::TYPE_F64},
                        nvcv::test::ValueList<int>{1, 2, 3, 4}, nvcv::test::ValueList<bool>{false, true}));

TEST_P(OpMergeVarShape, varshape_correct_output_matches_interleaved_and_tensor_cpu_gold)
{
    const auto &[dtype, channels, planar] = GetParam();
    channeltest::Stream               stream;
    cvcuda::Merge                     op;
    const std::array<nvcv::Size2D, 8> sizes = {
        {{1, 1}, {7, 5}, {8, 3}, {9, 7}, {17, 3}, {257, 5}, {513, 1}, {33, 259}}
    };
    for (const bool repeated : {false, true})
    {
        SCOPED_TRACE(repeated);
        std::vector<channeltest::ImageBuffer> images;
        std::vector<channeltest::ImageBuffer> parityImages;
        std::vector<channeltest::ImageBuffer> bands;
        nvcv::ImageBatchVarShape              output(sizes.size());
        nvcv::ImageBatchVarShape              parityOutput(sizes.size());
        std::vector<nvcv::ImageBatchVarShape> inputs;
        for (int channel = 0; channel < channels; ++channel) inputs.emplace_back(sizes.size());
        for (size_t index = 0; index < sizes.size(); ++index)
        {
            const auto [width, height] = sizes[index];
            std::vector<std::vector<uint8_t>> pixels;
            const size_t                      start = bands.size();
            for (int channel = 0; channel < channels; ++channel)
            {
                pixels.push_back(channeltest::Patterns(width * height, dtype.strideBytes(),
                                                       static_cast<int>(index) * 29 + channel * 17));
                bands.emplace_back(channeltest::ImageFormat(dtype, 1, false), width, height, index + channel + 5);
                bands.back().Put(pixels.back());
                bands.back().Upload(stream.handle);
                inputs[channel].pushBack(bands[repeated ? start : bands.size() - 1].image);
            }
            if (repeated)
                pixels.assign(channels, pixels.front());
            const auto gold = MergeCpuGold(pixels, dtype.strideBytes());
            images.emplace_back(channeltest::ImageFormat(dtype, channels, planar), width, height, index, 0xC7);
            parityImages.emplace_back(channeltest::ImageFormat(dtype, channels, !planar), width, height, index + 3,
                                      0xE9);
            images.back().Upload(stream.handle);
            parityImages.back().Upload(stream.handle);
            images.back().Put(gold);
            parityImages.back().Put(gold);
            output.pushBack(images.back().image);
            parityOutput.pushBack(parityImages.back().image);
            if (!planar)
                CheckMergeTensorCpuGold(stream.handle, op, dtype, width, height, pixels, gold);
        }
        op(stream.handle, inputs.data(), channels, output);
        op(stream.handle, inputs.data(), channels, parityOutput);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        for (size_t index = 0; index < images.size(); ++index)
        {
            EXPECT_EQ(images[index].Download(), images[index].host);
            EXPECT_EQ(parityImages[index].Download(), parityImages[index].host);
            EXPECT_EQ(images[index].Pixels(), parityImages[index].Pixels());
        }
        for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
    }
}

TEST(OpMergeVarShape, repeated_read_only_images_batches_and_physical_bgr_order)
{
    channeltest::Stream      stream;
    cvcuda::Merge            op;
    channeltest::ImageBuffer band(nvcv::FMT_U8, 17, 3);
    const auto               pixels = channeltest::Patterns(51, 1);
    band.Put(pixels);
    band.Upload(stream.handle);
    auto       input       = channeltest::Batch({band.image, band.View()});
    const auto taggedImage = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda(nvcv::FMT_Y8, band.buffer));
    auto       taggedInput = channeltest::Batch({taggedImage, taggedImage});
    for (const auto format : {nvcv::FMT_BGR8, nvcv::FMT_BGR8p, nvcv::FMT_BGRA8})
    {
        const int                             channels = format.numChannels();
        std::vector<nvcv::ImageBatchVarShape> inputs(channels, input);
        for (int channel = 1; channel < channels; channel += 2) inputs[channel] = taggedInput;
        std::vector<channeltest::ImageBuffer> images;
        nvcv::ImageBatchVarShape              output(2);
        for (int index = 0; index < 2; ++index)
        {
            images.emplace_back(format, 17, 3, index);
            images.back().Upload(stream.handle);
            images.back().Put(MergeCpuGold(std::vector<std::vector<uint8_t>>(channels, pixels), 1));
            output.pushBack(images.back().image);
        }
        op(stream.handle, inputs.data(), channels, output);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        for (const auto &image : images) EXPECT_EQ(image.Download(), image.host);
        EXPECT_EQ(band.Download(), band.host);
    }
}

TEST(OpMerge_Negative, varshape_validates_all_metadata_before_writing)
{
    channeltest::Stream                             stream;
    cvcuda::Merge                                   op;
    channeltest::ImageBuffer                        first(channeltest::ImageFormat(nvcv::TYPE_U8, 2, false), 7, 3);
    channeltest::ImageBuffer                        second(channeltest::ImageFormat(nvcv::TYPE_U8, 2, false), 7, 3);
    channeltest::ImageBuffer                        band(nvcv::FMT_U8, 7, 3);
    channeltest::ImageBuffer                        wrongSize(nvcv::FMT_U8, 9, 3);
    channeltest::ImageBuffer                        wrongType(nvcv::FMT_S8, 7, 3);
    channeltest::ImageBuffer                        wrongFormat(nvcv::FMT_Y8, 7, 3);
    channeltest::ImageBuffer                        planar(channeltest::ImageFormat(nvcv::TYPE_U8, 2, true), 7, 3);
    const std::array<channeltest::ImageBuffer *, 7> storage
        = {&first, &second, &band, &wrongSize, &wrongType, &wrongFormat, &planar};
    band.Put(channeltest::Patterns(21, 1));
    for (const auto buffer : storage) buffer->Upload(stream.handle);
    auto output       = channeltest::Batch({first.image, second.image});
    auto good         = channeltest::Batch({band.image, band.image});
    auto empty        = channeltest::Batch({});
    auto shortBatch   = channeltest::Batch({band.image});
    auto sizes        = channeltest::Batch({band.image, wrongSize.image});
    auto types        = channeltest::Batch({band.image, wrongType.image});
    auto formats      = channeltest::Batch({band.image, wrongFormat.image});
    auto multiChannel = channeltest::Batch({planar.image, planar.image});
    auto check        = [&](const NVCVImageBatchHandle *inputs, int count, const nvcv::ImageBatchVarShape &destination)
    {
        EXPECT_EQ(cvcudaMergeVarShapeSubmit(op.handle(), stream.handle, inputs, count, destination.handle()),
                  NVCV_ERROR_INVALID_ARGUMENT);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        for (const auto buffer : storage) EXPECT_EQ(buffer->Download(), buffer->host);
    };
    std::array<NVCVImageBatchHandle, 5> handles
        = {good.handle(), good.handle(), good.handle(), good.handle(), good.handle()};
    for (const int count : {-1, 0, 1, 3, 5}) check(handles.data(), count, output);
    check(nullptr, 2, output);
    check(handles.data(), 2, empty);
    check(handles.data(), 2, channeltest::Batch({first.image, planar.image}));
    for (const auto &bad : {empty, shortBatch, sizes, types, formats, multiChannel})
    {
        handles[1] = bad.handle();
        check(handles.data(), 2, output);
    }
    handles[1] = nullptr;
    check(handles.data(), 2, output);
    handles[1] = good.handle();
    EXPECT_EQ(cvcudaMergeVarShapeSubmit(op.handle(), stream.handle, handles.data(), 2, nullptr),
              NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaMergeVarShapeSubmit(nullptr, stream.handle, handles.data(), 2, output.handle()),
              NVCV_ERROR_INVALID_ARGUMENT);
}

void CheckC1VarShapeAliasesAndAlignment(bool splitDirection)
{
    channeltest::Stream stream;
    cvcuda::Split       split;
    cvcuda::Merge       merge;
    SCOPED_TRACE(splitDirection ? "Split" : "Merge");
    auto submit = [&](const nvcv::ImageBatchVarShape &input, const nvcv::ImageBatchVarShape &output)
    {
        if (splitDirection)
        {
            const auto handle = output.handle();
            return cvcudaSplitVarShapeSubmit(split.handle(), stream.handle, input.handle(), &handle, 1);
        }
        const auto handle = input.handle();
        return cvcudaMergeVarShapeSubmit(merge.handle(), stream.handle, &handle, 1, output.handle());
    };
    for (const auto dtype : {nvcv::TYPE_U8, nvcv::TYPE_F64})
    {
        channeltest::ImageBuffer source(channeltest::ImageFormat(dtype, 1, false), 7, 3);
        channeltest::ImageBuffer otherSource(channeltest::ImageFormat(dtype, 1, false), 7, 3);
        channeltest::ImageBuffer destination(channeltest::ImageFormat(dtype, 1, false), 7, 3);
        source.Put(channeltest::Patterns(21, dtype.strideBytes()));
        source.Upload(stream.handle);
        otherSource.Upload(stream.handle);
        destination.Upload(stream.handle);
        auto input = channeltest::Batch({source.image, otherSource.image});
        for (const auto &output : {channeltest::Batch({otherSource.View(dtype.strideBytes()), destination.image}),
                                   channeltest::Batch({destination.image, destination.View()}),
                                   channeltest::Batch({destination.image, destination.View(dtype.strideBytes())})})
        {
            EXPECT_EQ(submit(input, output), NVCV_ERROR_INVALID_ARGUMENT);
            NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
            EXPECT_EQ(source.Download(), source.host);
            EXPECT_EQ(otherSource.Download(), otherSource.host);
            EXPECT_EQ(destination.Download(), destination.host);
        }
    }
    for (const bool badSource : {false, true})
        for (const bool badPitch : {false, true})
        {
            channeltest::ImageBuffer source(nvcv::FMT_F64, 7, 3);
            channeltest::ImageBuffer destination(nvcv::FMT_F64, 7, 3);
            source.Put(channeltest::Patterns(21, 8));
            source.Upload(stream.handle);
            destination.Upload(stream.handle);
            auto invalid = (badSource ? source : destination).View(badPitch ? 0 : 1, badPitch ? 1 : 0);
            auto input   = channeltest::Batch({badSource ? invalid : source.image});
            auto output  = channeltest::Batch({badSource ? destination.image : invalid});
            EXPECT_EQ(submit(input, output), NVCV_ERROR_INVALID_ARGUMENT);
            NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
            EXPECT_EQ(source.Download(), source.host);
            EXPECT_EQ(destination.Download(), destination.host);
        }
}

TEST(OpSplit_Negative, varshape_rejects_global_aliases_and_misalignment_before_writing)
{
    CheckC1VarShapeAliasesAndAlignment(true);
}

TEST(OpMerge_Negative, varshape_rejects_global_aliases_and_misalignment_before_writing)
{
    CheckC1VarShapeAliasesAndAlignment(false);
}

TEST(OpMerge_Negative, varshape_unsupported_image_storage)
{
    channeltest::Stream stream;
    cvcuda::Merge       op;
    for (const auto format : channeltest::InvalidImageFormats())
    {
        SCOPED_TRACE(format);
        channeltest::ImageBuffer destination(format, 8, 6);
        channeltest::ImageBuffer band(nvcv::FMT_U8, 8, 6, 0, 0xC3);
        destination.Upload(stream.handle);
        band.Upload(stream.handle);
        auto                                    output = channeltest::Batch({destination.image});
        auto                                    input  = channeltest::Batch({band.image});
        const std::vector<NVCVImageBatchHandle> handles(format.numChannels(), input.handle());
        EXPECT_EQ(cvcudaMergeVarShapeSubmit(op.handle(), stream.handle, handles.data(), format.numChannels(),
                                            output.handle()),
                  NVCV_ERROR_INVALID_ARGUMENT);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        EXPECT_EQ(destination.Download(), destination.host);
        EXPECT_EQ(band.Download(), band.host);
    }
}

TEST(OpMerge_Negative, varshape_checks_every_planar_plane_and_overlap_before_writes)
{
    channeltest::Stream      stream;
    cvcuda::Merge            op;
    channeltest::ImageBuffer destination(channeltest::ImageFormat(nvcv::TYPE_F64, 2, true), 7, 3);
    channeltest::ImageBuffer band(nvcv::FMT_F64, 7, 3);
    band.Put(channeltest::Patterns(21, 8));
    destination.Upload(stream.handle);
    band.Upload(stream.handle);
    auto                                      input   = channeltest::Batch({band.image});
    const std::array<NVCVImageBatchHandle, 2> handles = {input.handle(), input.handle()};
    for (int invalid = 0; invalid < 7; ++invalid)
    {
        SCOPED_TRACE(invalid);
        auto metadata = destination.buffer;
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
        if (invalid == 5)
            metadata.planes[1] = metadata.planes[0];
        if (invalid == 6)
            metadata.planes[1] = band.buffer.planes[0];
        auto wrapped = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda(destination.format, metadata));
        auto output  = channeltest::Batch({wrapped});
        EXPECT_EQ(cvcudaMergeVarShapeSubmit(op.handle(), stream.handle, handles.data(), 2, output.handle()),
                  NVCV_ERROR_INVALID_ARGUMENT);
        NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
        EXPECT_EQ(destination.Download(), destination.host);
        EXPECT_EQ(band.Download(), band.host);
    }
}

void CheckC1DisjointVarShapeViews(bool splitDirection)
{
    channeltest::Stream stream;
    cvcuda::Split       split;
    cvcuda::Merge       merge;
    SCOPED_TRACE(splitDirection ? "Split" : "Merge");
    channeltest::ImageBuffer source(nvcv::FMT_U8, 7, 3);
    const auto               pixels = channeltest::Patterns(21, 1);
    source.Put(pixels);
    source.Upload(stream.handle);
    const size_t offset = source.buffer.planes[0].rowStride * 2 + 7;
    auto         input  = channeltest::Batch({source.image});
    auto         output = channeltest::Batch({source.View(offset)});
    if (splitDirection)
        split(stream.handle, input, &output, 1);
    else
        merge(stream.handle, &input, 1, output);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    source.Put(pixels, offset);
    EXPECT_EQ(source.Download(), source.host);
}

TEST(OpSplitVarShape, disjoint_views_of_one_allocation_are_allowed)
{
    CheckC1DisjointVarShapeViews(true);
}

TEST(OpMergeVarShape, disjoint_views_of_one_allocation_are_allowed)
{
    CheckC1DisjointVarShapeViews(false);
}

TEST(OpMerge_Negative, varshape_cuda_array_rejected_by_image_container)
{
    channeltest::CheckCudaArrayRejected();
}

TEST(OpMerge_Negative, varshape_rejects_1x1_subsampling_even_when_plane_sizes_match)
{
    const nvcv::ImageFormat format(nvcv::CSPEC_BT601, nvcv::ChromaSubsampling::CSS_420, nvcv::MemLayout::PITCH_LINEAR,
                                   nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XYZ0, nvcv::Packing::X8,
                                   nvcv::Packing::X8, nvcv::Packing::X8);
    ASSERT_EQ(format.numPlanes(), 3);
    ASSERT_EQ(format.chromaSubsampling(), nvcv::ChromaSubsampling::CSS_420);
    for (int plane = 0; plane < 3; ++plane) ASSERT_EQ(format.planeSize({1, 1}, plane), nvcv::Size2D(1, 1));
    channeltest::Stream      stream;
    cvcuda::Merge            op;
    channeltest::ImageBuffer destination(format, 1, 1);
    channeltest::ImageBuffer band(nvcv::FMT_U8, 1, 1);
    band.Put({0x39});
    destination.Upload(stream.handle);
    band.Upload(stream.handle);
    auto                                      output  = channeltest::Batch({destination.image});
    auto                                      input   = channeltest::Batch({band.image});
    const std::array<NVCVImageBatchHandle, 3> handles = {input.handle(), input.handle(), input.handle()};
    EXPECT_EQ(cvcudaMergeVarShapeSubmit(op.handle(), stream.handle, handles.data(), 3, output.handle()),
              NVCV_ERROR_INVALID_ARGUMENT);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    EXPECT_EQ(destination.Download(), destination.host);
    EXPECT_EQ(band.Download(), band.host);
}

NVCV_TEST_SUITE_P(OpMergeVarShapePacked,
                  nvcv::test::Combine(nvcv::test::ValueList<nvcv::DataType>{nvcv::TYPE_U8, nvcv::TYPE_S8},
                                      nvcv::test::ValueList<int>{3, 4, 5, 7, 8, 9, 127, 128, 129, 4097, 8193}));

TEST_P(OpMergeVarShapePacked, aligned_c3_single_row_matches_cpu_gold_and_preserves_canaries)
{
    const auto &[dtype, width] = GetParam();
    channeltest::Stream      stream;
    cvcuda::Merge            op;
    channeltest::ImageBuffer image(channeltest::ImageFormat(dtype, 3, false), width, 1, 5, 0xE7);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(image.buffer.planes[0].basePtr) % 8, 0u);
    image.Upload(stream.handle);
    auto                                  output = channeltest::Batch({image.image});
    std::vector<channeltest::ImageBuffer> bands;
    std::vector<std::vector<uint8_t>>     pixels;
    std::vector<nvcv::ImageBatchVarShape> inputs;
    bands.reserve(3);
    for (int channel = 0; channel < 3; ++channel)
    {
        pixels.push_back(channeltest::Patterns(width, 1, channel * 17));
        bands.emplace_back(channeltest::ImageFormat(dtype, 1, false), width, 1, 5, 0x31 + channel * 0x11);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(bands.back().buffer.planes[0].basePtr) % 8, 0u);
        bands.back().Put(pixels.back());
        bands.back().Upload(stream.handle);
        inputs.push_back(channeltest::Batch({bands.back().image}));
    }
    image.Put(MergeCpuGold(pixels, 1));
    op(stream.handle, inputs.data(), static_cast<int32_t>(inputs.size()), output);
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream.handle));
    EXPECT_EQ(image.Download(), image.host);
    for (const auto &band : bands) EXPECT_EQ(band.Download(), band.host);
}

} // namespace

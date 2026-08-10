/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Definitions.hpp"
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpChannelReorder.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <type_traits>

namespace test = nvcv::test;

class ChannelReorderOrdersException : public nvcv::Exception
{
public:
    explicit ChannelReorderOrdersException(const char *message)
        : nvcv::Exception(nvcv::Status::ERROR_INTERNAL, message)
    {
    }
};

class TestOpChannelReorder : public ::testing::Test
{
protected:
    TestOpChannelReorder() = default;

    ~TestOpChannelReorder() override = default;

    void SetUp() override
    {
        // clang-format off
        orders() = nvcv::Tensor(
            {
                {1, 4},
                "NC"
            },
            nvcv::TYPE_S32);
        // clang-format on
    }

    void pushDefaultImages()
    {
        input().pushBack(nvcv::Image{
            nvcv::Size2D{4, 2},
            nvcv::FMT_RGBA8
        });
        output().pushBack(nvcv::Image{
            nvcv::Size2D{4, 2},
            nvcv::FMT_RGBA8
        });
    }

    nvcv::ImageBatchVarShape &input()
    {
        return m_in;
    }

    nvcv::ImageBatchVarShape &output()
    {
        return m_out;
    }

    nvcv::Tensor &orders()
    {
        return m_inOrders;
    }

    cvcuda::ChannelReorder &channelReorder()
    {
        return m_chReorder;
    }

private:
    nvcv::ImageBatchVarShape m_in{nvcv::ImageBatchVarShape(2)};
    nvcv::ImageBatchVarShape m_out{nvcv::ImageBatchVarShape(2)};
    nvcv::Tensor             m_inOrders;
    cvcuda::ChannelReorder   m_chReorder;
};

static nvcv::Tensor MakeChannelReorderOrders(int numImages, const std::vector<int> &order)
{
    const auto   channels = static_cast<int>(order.size());
    nvcv::Tensor orders(
        {
            {numImages, channels},
            "NC"
    },
        nvcv::TYPE_S32);

    auto orderData = orders.exportData<nvcv::TensorDataStridedCuda>();
    if (!orderData)
    {
        throw ChannelReorderOrdersException("Failed to export ChannelReorder orders tensor");
    }

    for (int i = 0; i < numImages; ++i)
    {
        auto *row = orderData->basePtr() + i * orderData->stride(0);
        if (cudaSuccess != cudaMemcpy(row, order.data(), channels * sizeof(int), cudaMemcpyHostToDevice))
        {
            throw ChannelReorderOrdersException("Failed to upload ChannelReorder orders tensor");
        }
    }

    return orders;
}

static nvcv::Tensor MakeChannelReorderOrders(int numImages, int channels)
{
    std::vector<int> order(channels);
    for (int c = 0; c < channels; ++c)
    {
        order[c] = c;
    }

    if (channels >= 3)
    {
        order[0] = 2;
        order[1] = -1;
        order[2] = 1;
    }
    if (channels == 4)
    {
        order[3] = 3;
    }

    return MakeChannelReorderOrders(numImages, order);
}

static std::vector<uint8_t> ReferenceChannelReorder(const std::vector<uint8_t> &src, int numPixels, int numSrcChannels,
                                                    const std::vector<int> &order)
{
    std::vector<uint8_t> dst(static_cast<size_t>(numPixels) * order.size());
    for (int pixel = 0; pixel < numPixels; ++pixel)
    {
        for (size_t outChannel = 0; outChannel < order.size(); ++outChannel)
        {
            const int inChannel = order[outChannel];
            dst[static_cast<size_t>(pixel) * order.size() + outChannel]
                = inChannel < 0 ? 0 : src[static_cast<size_t>(pixel) * numSrcChannels + inChannel];
        }
    }
    return dst;
}

static void RunChannelReorderOrdersGold(nvcv::ImageFormat format, const std::vector<int> &order)
{
    const int                         channels = format.numChannels();
    const std::array<nvcv::Size2D, 2> sizes    = {
           {{5, 3}, {3, 4}}
    };
    ASSERT_EQ(static_cast<size_t>(channels), order.size());

    std::vector<nvcv::Image>          srcImages;
    std::vector<nvcv::Image>          dstImages;
    std::vector<std::vector<uint8_t>> srcValues(sizes.size());
    for (size_t image = 0; image < sizes.size(); ++image)
    {
        srcImages.emplace_back(sizes[image], format);
        dstImages.emplace_back(sizes[image], format);

        auto &values = srcValues[image];
        values.resize(static_cast<size_t>(sizes[image].w) * sizes[image].h * channels);
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] = static_cast<uint8_t>((i * 17 + image * 29 + 3) & 0xff);
        }

        auto srcData = srcImages.back().exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(srcData);
        const size_t rowBytes = static_cast<size_t>(sizes[image].w) * channels;
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, values.data(),
                                            rowBytes, rowBytes, sizes[image].h, cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape srcBatch(sizes.size());
    nvcv::ImageBatchVarShape dstBatch(sizes.size());
    srcBatch.pushBack(srcImages.begin(), srcImages.end());
    dstBatch.pushBack(dstImages.begin(), dstImages.end());
    nvcv::Tensor orders = MakeChannelReorderOrders(sizes.size(), order);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::ChannelReorder op;
    EXPECT_NO_THROW(op(stream, srcBatch, dstBatch, orders));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (size_t image = 0; image < sizes.size(); ++image)
    {
        auto dstData = dstImages[image].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(dstData);
        const size_t         rowBytes = static_cast<size_t>(sizes[image].w) * order.size();
        std::vector<uint8_t> got(rowBytes * sizes[image].h);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(got.data(), rowBytes, dstData->plane(0).basePtr, dstData->plane(0).rowStride, rowBytes,
                               sizes[image].h, cudaMemcpyDeviceToHost));

        const auto gold = ReferenceChannelReorder(srcValues[image], sizes[image].w * sizes[image].h, channels, order);
        EXPECT_EQ(gold, got);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpChannelReorderVarShape, varshape_correct_output_rgb)
{
    RunChannelReorderOrdersGold(nvcv::FMT_RGB8, {2, -1, 1});
}

TEST(OpChannelReorderVarShape, varshape_correct_output_rgba)
{
    RunChannelReorderOrdersGold(nvcv::FMT_RGBA8, {2, -1, 1, 3});
}

static nvcv::TensorShape MakeChannelReorderTensorShape(nvcv::TensorLayout layout, int numSamples, int width, int height,
                                                       int channels)
{
    if (layout == nvcv::TENSOR_NHWC)
    {
        return nvcv::TensorShape{
            {numSamples, height, width, channels},
            layout
        };
    }
    if (layout == nvcv::TENSOR_NCHW)
    {
        return nvcv::TensorShape{
            {numSamples, channels, height, width},
            layout
        };
    }
    if (layout == nvcv::TENSOR_HWC)
    {
        return nvcv::TensorShape{
            {height, width, channels},
            layout
        };
    }
    return nvcv::TensorShape{
        {channels, height, width},
        layout
    };
}

static std::vector<int32_t> MakeChannelReorderTensorOrder(int channels)
{
    std::vector<int32_t> order(channels);
    std::iota(order.rbegin(), order.rend(), 0);
    if (channels >= 3)
    {
        order[1] = -1;
        order[2] = order[0]; // repeated source channel is part of the native contract.
    }
    return order;
}

static size_t ChannelReorderTensorIndex(bool planar, int pixel, int channel, int pixelCount, int channels)
{
    if (planar)
    {
        return static_cast<size_t>(channel) * pixelCount + pixel;
    }
    return static_cast<size_t>(pixel) * channels + channel;
}

template<typename T>
static std::vector<T> ReferenceChannelReorderTensor(const std::vector<T> &input, const std::vector<int32_t> &order,
                                                    int width, int height, bool planar)
{
    const auto     channels   = static_cast<int>(order.size());
    const int      pixelCount = width * height;
    std::vector<T> expected(input.size());
    for (int pixel = 0; pixel < pixelCount; ++pixel)
    {
        for (int outputChannel = 0; outputChannel < channels; ++outputChannel)
        {
            const int  sourceChannel = order[outputChannel];
            const int  inputChannel  = std::max(sourceChannel, 0);
            const auto outputIndex   = ChannelReorderTensorIndex(planar, pixel, outputChannel, pixelCount, channels);
            const auto inputIndex    = ChannelReorderTensorIndex(planar, pixel, inputChannel, pixelCount, channels);
            expected[outputIndex]    = sourceChannel < 0 ? T{} : input[inputIndex];
        }
    }
    return expected;
}

template<typename T>
static std::vector<std::vector<T>> PrepareChannelReorderTensorSamples(nvcv::Tensor               &src,
                                                                      const std::vector<int32_t> &order, int samples,
                                                                      int width, int height, bool planar)
{
    const auto                  channels = static_cast<int>(order.size());
    std::vector<std::vector<T>> gold(samples);
    for (int sample = 0; sample < samples; ++sample)
    {
        std::vector<T> input(static_cast<size_t>(width) * height * channels);
        std::ranges::generate(
            input, [index = size_t{0}, sample]() mutable { return static_cast<T>(index++ * 13 + sample * 17 + 1); });
        nvcv::util::SetImageTensorFromVector<T>(src.exportData(), input, sample);
        gold[sample] = ReferenceChannelReorderTensor(input, order, width, height, planar);
    }
    return gold;
}

template<typename T>
static void RunChannelReorderTensorCase(nvcv::DataType dtype, nvcv::TensorLayout layout, int channels)
{
    constexpr int numSamples = 2;
    constexpr int width      = 7;
    constexpr int height     = 5;
    const bool    batched    = layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_NCHW;
    const bool    planar     = layout == nvcv::TENSOR_CHW || layout == nvcv::TENSOR_NCHW;

    const auto   shape = MakeChannelReorderTensorShape(layout, numSamples, width, height, channels);
    nvcv::Tensor src(shape, dtype);
    nvcv::Tensor dst(shape, dtype);
    const int    samples = batched ? numSamples : 1;
    const auto   order   = MakeChannelReorderTensorOrder(channels);
    const auto   gold    = PrepareChannelReorderTensorSamples<T>(src, order, samples, width, height, planar);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::ChannelReorder op;
    op(stream, src, dst, order.data(), static_cast<int32_t>(order.size()));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int n = 0; n < samples; ++n)
    {
        std::vector<T> got;
        nvcv::util::GetImageVectorFromTensor<T>(dst.exportData(), n, got);
        ASSERT_EQ(gold[n].size() * sizeof(T), got.size() * sizeof(T));
        EXPECT_EQ(0, std::memcmp(gold[n].data(), got.data(), got.size() * sizeof(T)));
    }
}

template<typename T>
static void RunChannelReorderTensorCases(nvcv::DataType dtype)
{
    const std::array<nvcv::TensorLayout, 4> layouts{nvcv::TENSOR_HWC, nvcv::TENSOR_NHWC, nvcv::TENSOR_CHW,
                                                    nvcv::TENSOR_NCHW};
    for (nvcv::TensorLayout layout : layouts)
    {
        const bool planar = layout == nvcv::TENSOR_CHW || layout == nvcv::TENSOR_NCHW;
        for (int channels = 1; channels <= 4; ++channels)
        {
            if (planar && channels == 2)
            {
                continue;
            }
            RunChannelReorderTensorCase<T>(dtype, layout, channels);
        }
    }
}

TEST(OpChannelReorderTensor, correct_output_all_declared_types_layouts_channels)
{
    RunChannelReorderTensorCases<uint8_t>(nvcv::TYPE_U8);
    RunChannelReorderTensorCases<uint16_t>(nvcv::TYPE_U16);
    RunChannelReorderTensorCases<int16_t>(nvcv::TYPE_S16);
    RunChannelReorderTensorCases<int32_t>(nvcv::TYPE_S32);
    RunChannelReorderTensorCases<float>(nvcv::TYPE_F32);
}

TEST(OpChannelReorderTensor, preserves_float_bit_patterns)
{
    nvcv::Tensor src(
        {
            {1, 1, 2, 4},
            "NHWC"
    },
        nvcv::TYPE_F32);
    nvcv::Tensor dst(src.shape(), src.dtype());

    const std::array<uint32_t, 8> inputBits{0x7FC01234, 0x80000000, 0x7F800000, 0xFF800000,
                                            0x00000001, 0x3F800000, 0xBF800000, 0x7FA00001};
    std::vector<float>            input(inputBits.size());
    std::memcpy(input.data(), inputBits.data(), sizeof(inputBits));
    nvcv::util::SetImageTensorFromVector<float>(src.exportData(), input, 0);

    const std::array<int32_t, 4> order{3, 2, 1, 0};
    cvcuda::ChannelReorder       op;
    EXPECT_NO_THROW(op(nullptr, src, dst, order.data(), static_cast<int32_t>(order.size())));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float> got;
    nvcv::util::GetImageVectorFromTensor<float>(dst.exportData(), 0, got);
    std::array<uint32_t, 8> gotBits{};
    ASSERT_EQ(gotBits.size(), got.size());
    std::memcpy(gotBits.data(), got.data(), sizeof(gotBits));
    const std::array<uint32_t, 8> expected{inputBits[3], inputBits[2], inputBits[1], inputBits[0],
                                           inputBits[7], inputBits[6], inputBits[5], inputBits[4]};
    EXPECT_EQ(expected, gotBits);
}

TEST(OpChannelReorderTensor_Negative, invalid_order_and_alias_are_rejected)
{
    nvcv::Tensor src(
        {
            {1, 5, 7, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor dst(
        {
            {1, 5, 7, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);
    cvcuda::ChannelReorder op;
    std::array<int32_t, 3> valid{2, 1, 0};
    std::array<int32_t, 3> outOfRange{3, 1, 0};

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaChannelReorderSubmit(op.handle(), nullptr, src.handle(), dst.handle(), nullptr, 3));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaChannelReorderSubmit(op.handle(), nullptr, src.handle(), dst.handle(), valid.data(), -1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaChannelReorderSubmit(op.handle(), nullptr, src.handle(), dst.handle(), outOfRange.data(), 3));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaChannelReorderSubmit(op.handle(), nullptr, src.handle(), src.handle(), valid.data(), 3));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

static void RunChannelReorderPlanarParityCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int width,
                                              int height, int numImages)
{
    cvcuda::ChannelReorder op;
    nvcv::Tensor           orders = MakeChannelReorderOrders(numImages, planarFmt.numChannels());

    nvcv::test::planar::RunVarShapeParity(planarFmt, interleavedFmt, width, height, width, height, numImages,
                                          [&op, &orders](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                                         const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                          { op(stream, src, dst, orders); });
}

TEST_F(TestOpChannelReorder, smoke_test_works)
{
    // Let's set up input and output images
    std::array<nvcv::Image, 2> inImages = {
        nvcv::Image{nvcv::Size2D{4, 2}, nvcv::FMT_RGBA8},
        nvcv::Image{nvcv::Size2D{4, 2}, nvcv::FMT_RGBA8}
    };

    std::array<nvcv::Image, 2> outImages = {
        nvcv::Image{nvcv::Size2D{4, 2}, nvcv::FMT_BGRA8},
        nvcv::Image{nvcv::Size2D{4, 2}, nvcv::FMT_RGBA8}
    };

    input()  = nvcv::ImageBatchVarShape(2);
    output() = nvcv::ImageBatchVarShape(2);

    // Create the input and output varshapes
    input().pushBack(inImages[0]);
    input().pushBack(inImages[1]);

    output().pushBack(outImages[0]);
    output().pushBack(outImages[1]);

    // Populate input images
    std::vector<uchar4> inImageValues0;
    auto                inImageData0 = inImages[0].exportData<nvcv::ImageDataStrided>();
    inImageValues0.resize(inImageData0->plane(0).rowStride / sizeof(uchar4) * inImageData0->size().h);
    inImageValues0[0] = {1, 2, 3, 7};
    inImageValues0[1] = {7, 3, 2, 9};
    nvcv::util::SetTensorFromVector<uchar4>(nvcv::TensorWrapImage(inImages[0]).exportData(), inImageValues0, -1);

    std::vector<uchar4> inImageValues1;
    auto                inImageData1 = inImages[1].exportData<nvcv::ImageDataStrided>();
    inImageValues1.resize(inImageData1->plane(0).rowStride / sizeof(uchar4) * inImageData1->size().h);
    inImageValues1[0] = {3, 2, 1, 4};
    inImageValues1[1] = {1, 3, 10, 28};
    nvcv::util::SetTensorFromVector<uchar4>(nvcv::TensorWrapImage(inImages[1]).exportData(), inImageValues1, -1);

    // Populate the order tensor
    // clang-format off
    orders() = nvcv::Tensor(
        {
            {2, 4},
            "NC"
        },
        nvcv::TYPE_S32);
    // clang-format on

    auto              inOrderData = orders().exportData<nvcv::TensorDataStrided>();
    std::vector<int4> inOrderValues(inOrderData->stride(0) / sizeof(int4));

    // N==0
    inOrderValues[0] = {2, -1, 1, 3};
    nvcv::util::SetTensorFromVector<int4>(orders().exportData(), inOrderValues, 0);

    // N=1
    inOrderValues[0] = {3, 2, 1, -1};
    nvcv::util::SetTensorFromVector<int4>(orders().exportData(), inOrderValues, 1);

    // Execute operation
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    channelReorder()(stream, input(), output(), orders());

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Fetch results
    std::vector<uchar4> outImageValues;

    // First image uses the first channel order.
    nvcv::util::GetVectorFromTensor<uchar4>(nvcv::TensorWrapImage(outImages[0]).exportData(), 0, outImageValues);
    EXPECT_EQ(make_uchar4(3, 0, 2, 7), outImageValues[0]);
    EXPECT_EQ(make_uchar4(2, 0, 3, 9), outImageValues[1]);

    // Second image uses the second channel order.
    outImageValues.clear();
    nvcv::util::GetVectorFromTensor<uchar4>(nvcv::TensorWrapImage(outImages[1]).exportData(), 0, outImageValues);
    EXPECT_EQ(make_uchar4(4, 1, 2, 0), outImageValues[0]);
    EXPECT_EQ(make_uchar4(28, 10, 3, 0), outImageValues[1]);
}

TEST_F(TestOpChannelReorder, smoke_test_expands_output_channels)
{
    nvcv::Image inImage{
        nvcv::Size2D{2, 1},
        nvcv::FMT_RGB8
    };
    nvcv::Image outImage{
        nvcv::Size2D{2, 1},
        nvcv::FMT_RGBA8
    };

    input()  = nvcv::ImageBatchVarShape(1);
    output() = nvcv::ImageBatchVarShape(1);
    input().pushBack(inImage);
    output().pushBack(outImage);

    std::vector<uint8_t> inValues    = {1, 2, 3, 4, 5, 6};
    auto                 inImageData = inImage.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inImageData->plane(0).basePtr, inImageData->plane(0).rowStride, inValues.data(),
                                        2 * 3, 2 * 3, 1, cudaMemcpyHostToDevice));

    std::vector<uchar4> outValues(2);
    std::ranges::fill(outValues, make_uchar4(99, 99, 99, 99));
    auto outImageData = outImage.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy2D(outImageData->plane(0).basePtr, outImageData->plane(0).rowStride, outValues.data(),
                           2 * sizeof(uchar4), 2 * sizeof(uchar4), 1, cudaMemcpyHostToDevice));

    orders() = MakeChannelReorderOrders(1, {2, -1, 1, -1});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    channelReorder()(stream, input(), output(), orders());

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    outValues.clear();
    outValues.resize(2);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy2D(outValues.data(), 2 * sizeof(uchar4), outImageData->plane(0).basePtr,
                           outImageData->plane(0).rowStride, 2 * sizeof(uchar4), 1, cudaMemcpyDeviceToHost));
    EXPECT_EQ(make_uchar4(3, 0, 2, 0), outValues[0]);
    EXPECT_EQ(make_uchar4(6, 0, 5, 0), outValues[1]);
}

TEST_F(TestOpChannelReorder, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaChannelReorderCreate(nullptr));
}

TEST_F(TestOpChannelReorder, infer_different_samples)
{
    input().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_RGBA8
    });
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_void_samples)
{
    EXPECT_EQ(NVCV_SUCCESS, nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_input_dataType)
{
    input().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_RGBAf16
    });
    output().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_RGBA8
    });

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_output_dataType)
{
    input().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_RGBA8
    });
    output().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_RGBAf16
    });

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_order_rank)
{
    pushDefaultImages();

    // clang-format off
    orders() = nvcv::Tensor(
        {
            {1, 4, 4},
            "NHW"
        },
        nvcv::TYPE_S32);
    // clang-format on

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_order_dataType)
{
    pushDefaultImages();

    // clang-format off
    orders() = nvcv::Tensor(
        {
            {1, 4},
            "NC"
        },
        nvcv::TYPE_F32);
    // clang-format on

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_order_first_label)
{
    pushDefaultImages();

    // clang-format off
    orders() = nvcv::Tensor(
        {
            {4, 1},
            "CN"
        },
        nvcv::TYPE_S32);
    // clang-format on

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_order_num_samples)
{
    pushDefaultImages();

    // clang-format off
    orders() = nvcv::Tensor(
        {
            {2, 4},
            "NC"
        },
        nvcv::TYPE_S32);
    // clang-format on

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_order_num_channels)
{
    pushDefaultImages();

    // clang-format off
    orders() = nvcv::Tensor(
        {
            {1, 5},
            "NC"
        },
        nvcv::TYPE_S32);
    // clang-format on

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_order_small_num_channels)
{
    pushDefaultImages();

    // clang-format off
    orders() = nvcv::Tensor(
        {
            {1, 3},
            "NC"
        },
        nvcv::TYPE_S32);
    // clang-format on

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

// Parameters: width, height, numImages, planar format, interleaved format
// clang-format off
NVCV_TEST_SUITE_P(OpChannelReorderPlanar,
    test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {37, 29, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {31, 23, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    {35, 27, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    {33, 25, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

// clang-format on

TEST_P(OpChannelReorderPlanar, varshape_matches_interleaved)
{
    RunChannelReorderPlanarParityCase(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                      GetParamValue<2>());
}

TEST_F(TestOpChannelReorder, infer_invalid_input_output_layout_mismatch_planar_to_interleaved)
{
    input().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_BGRA8p
    });
    output().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_BGRA8
    });

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_input_output_layout_mismatch_interleaved_to_planar)
{
    input().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_BGRA8
    });
    output().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_BGRA8p
    });

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_input_different_channels)
{
    pushDefaultImages();
    input().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_BGR8
    });
    output().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_BGR8
    });

    // clang-format off
    orders() = nvcv::Tensor(
        {
            {2, 4},
            "NC"
        },
        nvcv::TYPE_S32);
    // clang-format on

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

TEST_F(TestOpChannelReorder, infer_invalid_input_different_format)
{
    pushDefaultImages();
    input().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_BGRAf32
    });
    output().pushBack(nvcv::Image{
        nvcv::Size2D{4, 2},
        nvcv::FMT_BGRAf32
    });

    // clang-format off
    orders() = nvcv::Tensor(
        {
            {2, 4},
            "NC"
        },
        nvcv::TYPE_S32);
    // clang-format on

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([this] { channelReorder()(nullptr, input(), output(), orders()); }));
}

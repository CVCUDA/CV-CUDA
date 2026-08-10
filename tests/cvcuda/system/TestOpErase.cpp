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
#include <cuda_fp16.h>
#include <cvcuda/OpErase.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <deque>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <vector>

static int ScaledSize(int value, double scale)
{
    return static_cast<int>(value * scale);
}

static void ClearU8Image(const nvcv::Image &image)
{
    const auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    assert(data->numPlanes() == 1);

    int width     = data->plane(0).width;
    int height    = data->plane(0).height;
    int rowStride = width * nvcv::FMT_U8.planePixelStrideBytes(0);

    EXPECT_EQ(cudaSuccess, cudaMemset2D(data->plane(0).basePtr, rowStride, 0, rowStride, height));
}

static void UploadU8Image(const nvcv::Image &image, const std::vector<uint8_t> &host)
{
    const auto data     = image.exportData<nvcv::ImageDataStridedCuda>();
    const int  width    = image.size().w;
    const int  height   = image.size().h;
    const int  channels = image.format().numChannels();

    ASSERT_NE(data, nullptr);
    ASSERT_EQ(host.size(), static_cast<size_t>(width * height * channels));

    if (data->numPlanes() == 1)
    {
        const int rowBytes = width * channels;
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(data->plane(0).basePtr, data->plane(0).rowStride, host.data(), rowBytes,
                                            rowBytes, height, cudaMemcpyHostToDevice));
        return;
    }

    ASSERT_EQ(data->numPlanes(), channels);
    const auto planes     = nvcv::test::planar::DeinterleaveToPlanes(host, width, height, channels, 1);
    const int  planeBytes = width * height;
    for (int c = 0; c < channels; ++c)
    {
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(data->plane(c).basePtr, data->plane(c).rowStride, planes.data() + c * planeBytes, width,
                               width, height, cudaMemcpyHostToDevice));
    }
}

static void DownloadU8Image(const nvcv::Image &image, std::vector<uint8_t> &host)
{
    const auto data     = image.exportData<nvcv::ImageDataStridedCuda>();
    const int  width    = image.size().w;
    const int  height   = image.size().h;
    const int  channels = image.format().numChannels();

    ASSERT_NE(data, nullptr);

    if (data->numPlanes() == 1)
    {
        const int rowBytes = width * channels;
        host.resize(height * rowBytes);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(host.data(), rowBytes, data->plane(0).basePtr, data->plane(0).rowStride,
                                            rowBytes, height, cudaMemcpyDeviceToHost));
        return;
    }

    ASSERT_EQ(data->numPlanes(), channels);
    std::vector<uint8_t> planes(width * height * channels);
    const int            planeBytes = width * height;
    for (int c = 0; c < channels; ++c)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(planes.data() + c * planeBytes, width, data->plane(c).basePtr,
                                            data->plane(c).rowStride, width, height, cudaMemcpyDeviceToHost));
    }
    host = nvcv::test::planar::InterleaveFromPlanes(planes, width, height, channels, 1);
}

static std::vector<uint8_t> MakeNonzeroU8Image(int width, int height, int channels, int sample)
{
    std::vector<uint8_t> image(width * height * channels);
    for (size_t i = 0; i < image.size(); ++i)
    {
        image[i] = static_cast<uint8_t>(1 + (i * 29 + static_cast<size_t>(sample) * 47) % 251);
    }
    return image;
}

struct EraseParamTensors
{
    const nvcv::Tensor &anchor;
    const nvcv::Tensor &erasing;
    const nvcv::Tensor &values;
    const nvcv::Tensor &imgIdx;
};

struct EraseParamValues
{
    const std::vector<int2>  &anchor;
    const std::vector<int3>  &erasing;
    const std::vector<float> &values;
    const std::vector<int>   &imgIdx;
};

static void ApplyErasePixelGold(std::vector<uint8_t> &image, size_t pixelOffset, int channels, int channelMask,
                                const float *values)
{
    for (int c = 0; c < channels; ++c)
    {
        if ((channelMask & (1 << c)) != 0)
        {
            image[pixelOffset + c] = static_cast<uint8_t>(values[c]);
        }
    }
}

static void ApplyEraseGold(std::vector<std::vector<uint8_t>> &images, const std::vector<nvcv::Size2D> &sizes,
                           int channels, const EraseParamValues &params)
{
    ASSERT_EQ(images.size(), sizes.size());
    ASSERT_EQ(params.anchor.size(), params.erasing.size());
    ASSERT_EQ(params.anchor.size(), params.imgIdx.size());
    ASSERT_EQ(params.anchor.size() * static_cast<size_t>(channels), params.values.size());

    for (size_t area = 0; area < params.anchor.size(); ++area)
    {
        const int imageIndex = params.imgIdx[area];
        ASSERT_GE(imageIndex, 0);
        ASSERT_LT(static_cast<size_t>(imageIndex), sizes.size());

        const int  width   = sizes[imageIndex].w;
        const int  height  = sizes[imageIndex].h;
        const int2 anchor  = params.anchor[area];
        const int3 erasing = params.erasing[area];

        ASSERT_EQ(images[imageIndex].size(), static_cast<size_t>(width * height * channels));
        ASSERT_GE(anchor.x, 0);
        ASSERT_GE(anchor.y, 0);

        for (int y = 0; y < erasing.y && anchor.y + y < height; ++y)
        {
            for (int x = 0; x < erasing.x && anchor.x + x < width; ++x)
            {
                const auto pixelOffset = static_cast<size_t>(((anchor.y + y) * width + anchor.x + x) * channels);
                ApplyErasePixelGold(images[imageIndex], pixelOffset, channels, erasing.z,
                                    &params.values[area * channels]);
            }
        }
    }
}

static void CopyEraseParams(cudaStream_t stream, const EraseParamTensors &tensors, const EraseParamValues &values)
{
    auto anchorData  = tensors.anchor.exportData<nvcv::TensorDataStridedCuda>();
    auto erasingData = tensors.erasing.exportData<nvcv::TensorDataStridedCuda>();
    auto valuesData  = tensors.values.exportData<nvcv::TensorDataStridedCuda>();
    auto imgIdxData  = tensors.imgIdx.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, anchorData);
    ASSERT_NE(nullptr, erasingData);
    ASSERT_NE(nullptr, valuesData);
    ASSERT_NE(nullptr, imgIdxData);

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(anchorData->basePtr(), values.anchor.data(),
                                           values.anchor.size() * sizeof(int2), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(erasingData->basePtr(), values.erasing.data(),
                                           values.erasing.size() * sizeof(int3), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(valuesData->basePtr(), values.values.data(),
                                           values.values.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgIdxData->basePtr(), values.imgIdx.data(),
                                           values.imgIdx.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
}

template<typename Input>
static void RunEraseWithParams(cudaStream_t stream, const Input &src, const Input &dst, int channels,
                               const EraseParamValues &params)
{
    const auto   numErasingArea = static_cast<int>(params.anchor.size());
    nvcv::Tensor anchor({{numErasingArea}, "N"}, nvcv::TYPE_2S32);
    nvcv::Tensor erasing({{numErasingArea}, "N"}, nvcv::TYPE_3S32);
    nvcv::Tensor values({{numErasingArea * channels}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor imgIdx({{numErasingArea}, "N"}, nvcv::TYPE_S32);

    ASSERT_NO_FATAL_FAILURE(CopyEraseParams(stream, {anchor, erasing, values, imgIdx}, params));

    cvcuda::Erase op(numErasingArea);
    EXPECT_NO_THROW(op(stream, src, dst, anchor, erasing, values, imgIdx, false, 0));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
}

static EraseParamValues MakePlanarParityEraseParams(int channels, int numImages, std::vector<int2> &anchor,
                                                    std::vector<int3> &erasing, std::vector<float> &values,
                                                    std::vector<int> &imgIdx)
{
    constexpr int numErasingArea = 2;

    anchor = {
        { 1, 1},
        {12, 7}
    };
    erasing = {
        {5, 4,       (1 << channels) - 1},
        {4, 3, channels == 4 ? 0xA : 0x5}
    };
    imgIdx = {0, numImages > 1 ? 1 : 0};

    values.resize(numErasingArea * channels);
    for (int area = 0; area < numErasingArea; ++area)
    {
        for (int c = 0; c < channels; ++c)
        {
            values[area * channels + c] = static_cast<float>(13 + area * 17 + c * 3);
        }
    }

    return {anchor, erasing, values, imgIdx};
}

struct PlanarParityEraseRunner
{
    static constexpr int kNumErasingArea = 2;

    nvcv::Tensor anchor{
        {{kNumErasingArea}, "N"},
        nvcv::TYPE_2S32
    };
    nvcv::Tensor erasing{
        {{kNumErasingArea}, "N"},
        nvcv::TYPE_3S32
    };
    nvcv::Tensor values;
    nvcv::Tensor imgIdx{
        {{kNumErasingArea}, "N"},
        nvcv::TYPE_S32
    };

    std::vector<int2>  anchorVec;
    std::vector<int3>  erasingVec;
    std::vector<float> valuesVec;
    std::vector<int>   imgIdxVec;
    EraseParamValues   params;
    cvcuda::Erase      op{kNumErasingArea};
    bool               random;

    PlanarParityEraseRunner(int channels, int numImages, bool random = false)
        : values({{kNumErasingArea * channels}, "N"}, nvcv::TYPE_F32)
        , params(MakePlanarParityEraseParams(channels, numImages, anchorVec, erasingVec, valuesVec, imgIdxVec))
        , random(random)
    {
    }

    void operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
    {
        submit(stream, src, dst);
    }

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst,
                    nvcv::ImageFormat)
    {
        submit(stream, src, dst);
    }

private:
    template<typename Input, typename Output>
    void submit(cudaStream_t stream, const Input &src, const Output &dst)
    {
        CopyEraseParams(stream, {anchor, erasing, values, imgIdx}, params);
        op(stream, src, dst, anchor, erasing, values, imgIdx, random, 17);
    }
};

static void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                                      int numImages)
{
    PlanarParityEraseRunner runner(planarFmt.numChannels(), numImages);
    nvcv::test::planar::RunTensorParity(planarFmt, interleavedFmt, w, h, w, h, numImages, runner);
}

static void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                                        int numImages)
{
    PlanarParityEraseRunner runner(planarFmt.numChannels(), numImages);
    nvcv::test::planar::RunVarShapeParity(planarFmt, interleavedFmt, w, h, w, h, numImages, runner);
}

NVCV_TEST_SUITE_P(OpErase, nvcv::test::ValueList<int, bool, bool>{
  // N, random, isInplace
                               {1, false, false},
                               {2, false, false},
                               {1,  true, false},
                               {2,  true, false},
                               {1, false,  true},
                               {2, false,  true},
                               {1,  true,  true},
                               {2,  true,  true},
});

TEST_P(OpErase, correct_output)
{
    int          N                    = GetParamValue<0>();
    bool         random               = GetParamValue<1>();
    bool         isInplace            = GetParamValue<2>();
    int          max_num_erasing_area = 2;
    unsigned int seed                 = 0;

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor        imgIn   = nvcv::util::CreateTensor(N, 640, 480, nvcv::FMT_U8);
    nvcv::Tensor        _imgOut = nvcv::util::CreateTensor(N, 640, 480, nvcv::FMT_U8);
    const nvcv::Tensor &imgOut  = isInplace ? imgIn : _imgOut;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(imgIn.exportData());
    ASSERT_TRUE(inAccess);

    ASSERT_EQ(N, inAccess->numSamples());

    // setup the buffer
    EXPECT_EQ(cudaSuccess, cudaMemset2D(inAccess->planeData(0), inAccess->rowStride(), 0,
                                        inAccess->numCols() * inAccess->colStride(), inAccess->numRows()));

    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, outData);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    int64_t outSampleStride = outAccess->sampleStride();

    if (outData->rank() == 3)
    {
        outSampleStride = outAccess->numRows() * outAccess->rowStride();
    }

    int64_t outBufferSize = outSampleStride * outAccess->numSamples();

    // Set output buffer to dummy value
    if (!isInplace)
    {
        EXPECT_EQ(cudaSuccess, cudaMemset(outAccess->sampleData(0), 0xFA, outBufferSize));
    }

    //parameters
    int          num_erasing_area = 2;
    nvcv::Tensor anchor({{num_erasing_area}, "N"}, nvcv::TYPE_2S32);
    nvcv::Tensor erasing({{num_erasing_area}, "N"}, nvcv::TYPE_3S32);
    nvcv::Tensor values({{num_erasing_area}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor imgIdx({{num_erasing_area}, "N"}, nvcv::TYPE_S32);

    auto anchorData  = anchor.exportData<nvcv::TensorDataStridedCuda>();
    auto erasingData = erasing.exportData<nvcv::TensorDataStridedCuda>();
    auto valuesData  = values.exportData<nvcv::TensorDataStridedCuda>();
    auto imgIdxData  = imgIdx.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, anchorData);
    ASSERT_NE(nullptr, erasingData);
    ASSERT_NE(nullptr, valuesData);
    ASSERT_NE(nullptr, imgIdxData);

    std::vector<int2>  anchorVec(num_erasing_area);
    std::vector<int3>  erasingVec(num_erasing_area);
    std::vector<int>   imgIdxVec(num_erasing_area);
    std::vector<float> valuesVec(num_erasing_area);

    anchorVec[0].x  = 0;
    anchorVec[0].y  = 0;
    erasingVec[0].x = 10;
    erasingVec[0].y = 10;
    erasingVec[0].z = 0x1;
    imgIdxVec[0]    = 0;
    valuesVec[0]    = 1.f;

    anchorVec[1].x  = 10;
    anchorVec[1].y  = 10;
    erasingVec[1].x = 20;
    erasingVec[1].y = 20;
    erasingVec[1].z = 0x1;
    imgIdxVec[1]    = 0;
    valuesVec[1]    = 1.f;

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(anchorData->basePtr(), anchorVec.data(), anchorVec.size() * sizeof(int2),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(erasingData->basePtr(), erasingVec.data(), erasingVec.size() * sizeof(int3),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgIdxData->basePtr(), imgIdxVec.data(), imgIdxVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(valuesData->basePtr(), valuesVec.data(), valuesVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));

    // Call operator
    cvcuda::Erase eraseOp(max_num_erasing_area);
    EXPECT_NO_THROW(eraseOp(stream, imgIn, imgOut, anchor, erasing, values, imgIdx, random, seed));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> test(outBufferSize, 0xA);

    //Check data
    if (!random)
    {
        EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), outData->basePtr(), outBufferSize, cudaMemcpyDeviceToHost));

        EXPECT_EQ(test[0], 1);
        EXPECT_EQ(test[9], 1);
        EXPECT_EQ(test[10], 0);
        EXPECT_EQ(test[9 * 640], 1);
        EXPECT_EQ(test[9 * 640 + 9], 1);
        EXPECT_EQ(test[9 * 640 + 10], 0);
        EXPECT_EQ(test[10 * 640], 0);
        EXPECT_EQ(test[10 * 640 + 10], 1);
    }
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), outData->basePtr(), outBufferSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpErase, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    std::deque<bool> isInplaces{true, false};
    std::deque<bool> isRandoms{true, false};
    for (bool isInplace : isInplaces)
    {
        for (bool random : isRandoms) // NOSONAR
        {
            std::vector<nvcv::Image> imgSrc;
            std::vector<nvcv::Image> imgDst;
            imgSrc.emplace_back(nvcv::Size2D{640, 480}, nvcv::FMT_U8);
            imgDst.emplace_back(nvcv::Size2D{640, 480}, nvcv::FMT_U8);

            nvcv::ImageBatchVarShape        batchSrc(1);
            nvcv::ImageBatchVarShape        _batchDst(1);
            const nvcv::ImageBatchVarShape &batchDst = isInplace ? batchSrc : _batchDst;
            batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
            _batchDst.pushBack(imgDst.begin(), imgDst.end());

            ClearU8Image(imgSrc[0]);

            if (!isInplace)
            {
                ClearU8Image(imgDst[0]);
            }

            //parameters
            int          num_erasing_area = 2;
            nvcv::Tensor anchor({{num_erasing_area}, "N"}, nvcv::TYPE_2S32);
            nvcv::Tensor erasing({{num_erasing_area}, "N"}, nvcv::TYPE_3S32);
            nvcv::Tensor values({{num_erasing_area}, "N"}, nvcv::TYPE_F32);
            nvcv::Tensor imgIdx({{num_erasing_area}, "N"}, nvcv::TYPE_S32);

            auto anchorData  = anchor.exportData<nvcv::TensorDataStridedCuda>();
            auto erasingData = erasing.exportData<nvcv::TensorDataStridedCuda>();
            auto valuesData  = values.exportData<nvcv::TensorDataStridedCuda>();
            auto imgIdxData  = imgIdx.exportData<nvcv::TensorDataStridedCuda>();

            ASSERT_NE(nullptr, anchorData);
            ASSERT_NE(nullptr, erasingData);
            ASSERT_NE(nullptr, valuesData);
            ASSERT_NE(nullptr, imgIdxData);

            std::vector<int2>  anchorVec(num_erasing_area);
            std::vector<int3>  erasingVec(num_erasing_area);
            std::vector<int>   imgIdxVec(num_erasing_area);
            std::vector<float> valuesVec(num_erasing_area);

            anchorVec[0].x  = 0;
            anchorVec[0].y  = 0;
            erasingVec[0].x = 10;
            erasingVec[0].y = 10;
            erasingVec[0].z = 0x1;
            imgIdxVec[0]    = 0;
            valuesVec[0]    = 1.f;

            anchorVec[1].x  = 10;
            anchorVec[1].y  = 10;
            erasingVec[1].x = 20;
            erasingVec[1].y = 20;
            erasingVec[1].z = 0x1;
            imgIdxVec[1]    = 0;
            valuesVec[1]    = 1.f;

            // Copy vectors to the GPU
            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(anchorData->basePtr(), anchorVec.data(),
                                                   anchorVec.size() * sizeof(int2), cudaMemcpyHostToDevice, stream));
            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(erasingData->basePtr(), erasingVec.data(),
                                                   erasingVec.size() * sizeof(int3), cudaMemcpyHostToDevice, stream));
            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgIdxData->basePtr(), imgIdxVec.data(),
                                                   imgIdxVec.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(valuesData->basePtr(), valuesVec.data(),
                                                   valuesVec.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

            // Call operator
            unsigned int  seed                 = 0;
            int           max_num_erasing_area = 2;
            cvcuda::Erase eraseOp(max_num_erasing_area);
            EXPECT_NO_THROW(eraseOp(stream, batchSrc, batchDst, anchor, erasing, values, imgIdx, random, seed));

            EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

            const auto dstData = isInplace ? imgSrc[0].exportData<nvcv::ImageDataStridedCuda>()
                                           : imgDst[0].exportData<nvcv::ImageDataStridedCuda>();
            assert(dstData->numPlanes() == 1);

            int dstWidth  = dstData->plane(0).width;
            int dstHeight = dstData->plane(0).height;

            int dstRowStride = dstWidth * nvcv::FMT_U8.planePixelStrideBytes(0);

            std::vector<uint8_t> test(dstHeight * dstRowStride, 0xFF);

            // Copy output data to Host
            if (!random)
            {
                ASSERT_EQ(cudaSuccess,
                          cudaMemcpy2D(test.data(), dstRowStride, dstData->plane(0).basePtr,
                                       dstData->plane(0).rowStride, dstRowStride, dstHeight, cudaMemcpyDeviceToHost));

                EXPECT_EQ(test[0], 1);
                EXPECT_EQ(test[9], 1);
                EXPECT_EQ(test[10], 0);
                EXPECT_EQ(test[9 * 640], 1);
                EXPECT_EQ(test[9 * 640 + 9], 1);
                EXPECT_EQ(test[9 * 640 + 10], 0);
                EXPECT_EQ(test[10 * 640], 0);
                EXPECT_EQ(test[10 * 640 + 10], 1);
            }
        }
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

NVCV_TEST_SUITE_P(OpEraseVarShapeCopyGold, nvcv::test::ValueList<nvcv::ImageFormat, bool>{
  // format, row bytes are a multiple of 16
                                               { nvcv::FMT_RGB8,  true},
                                               { nvcv::FMT_RGB8, false},
                                               {nvcv::FMT_RGB8p,  true},
                                               {nvcv::FMT_RGB8p, false},
});

TEST_P(OpEraseVarShapeCopyGold, full_nonzero_output_matches_independent_gold)
{
    const nvcv::ImageFormat format         = GetParamValue<0>();
    const bool              vectorCopyRows = GetParamValue<1>();
    const int               channels       = format.numChannels();
    const std::vector<nvcv::Size2D> sizes  = vectorCopyRows ? std::vector<nvcv::Size2D>{{16, 9}, {32, 7}}
                                                             : std::vector<nvcv::Size2D>{{13, 9}, {19, 7}};

    const int rowChannels = format.numPlanes() == 1 ? channels : 1;
    for (const nvcv::Size2D &size : sizes)
    {
        EXPECT_EQ(vectorCopyRows, (size.w * rowChannels) % static_cast<int>(sizeof(uint4)) == 0);
    }

    std::vector<nvcv::Image>          srcImages;
    std::vector<nvcv::Image>          dstImages;
    std::vector<std::vector<uint8_t>> srcHost;
    std::vector<std::vector<uint8_t>> gold;
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        srcImages.emplace_back(sizes[i], format);
        dstImages.emplace_back(sizes[i], format);
        srcHost.emplace_back(MakeNonzeroU8Image(sizes[i].w, sizes[i].h, channels, static_cast<int>(i)));
        UploadU8Image(srcImages.back(), srcHost.back());
        UploadU8Image(dstImages.back(), std::vector<uint8_t>(srcHost.back().size(), 0xA5));
    }
    gold = srcHost;

    nvcv::ImageBatchVarShape batchSrc(static_cast<int32_t>(sizes.size()));
    nvcv::ImageBatchVarShape batchDst(static_cast<int32_t>(sizes.size()));
    batchSrc.pushBack(srcImages.begin(), srcImages.end());
    batchDst.pushBack(dstImages.begin(), dstImages.end());

    std::vector<int2> anchorVec{
        {             1,              1},
        {sizes[0].w - 2, sizes[0].h - 2},
        {             0,              0},
        {sizes[1].w - 3,              1},
    };
    std::vector<int3> erasingVec{
        {6, 4, 0x7},
        {5, 4, 0x5},
        {4, 3, 0x2},
        {6, 5, 0x7},
    };
    std::vector<int>   imgIdxVec{0, 0, 1, 1};
    std::vector<float> valuesVec(anchorVec.size() * channels);
    for (size_t area = 0; area < anchorVec.size(); ++area)
    {
        for (int c = 0; c < channels; ++c)
        {
            valuesVec[area * channels + c] = static_cast<float>(17 + area * 37 + c * 9);
        }
    }

    const EraseParamValues params{anchorVec, erasingVec, valuesVec, imgIdxVec};
    ApplyEraseGold(gold, sizes, channels, params);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_NO_FATAL_FAILURE(RunEraseWithParams(stream, batchSrc, batchDst, channels, params));

    for (size_t i = 0; i < dstImages.size(); ++i)
    {
        SCOPED_TRACE(i);
        std::vector<uint8_t> got;
        DownloadU8Image(dstImages[i], got);
        EXPECT_EQ(gold[i], got);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpEraseVarShapeCopyGold, asymmetric_pitches_copy_visible_bytes_only)
{
    constexpr int    width          = 5;
    constexpr int    height         = 3;
    constexpr int    channels       = 3;
    constexpr size_t rowBytes       = width * channels;
    constexpr size_t srcRowStride   = 32;
    constexpr size_t dstRowStride   = 16;
    constexpr size_t dstGuardBytes  = 32;
    constexpr size_t srcBufferBytes = srcRowStride * height;
    constexpr size_t dstBufferBytes = dstRowStride * height + dstGuardBytes;

    NVCVByte *srcAllocation{};
    NVCVByte *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), srcBufferBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), dstBufferBytes));

    {
        nvcv::ImageDataStridedCuda::Buffer srcBuffer{};
        srcBuffer.numPlanes           = 1;
        srcBuffer.planes[0].width     = width;
        srcBuffer.planes[0].height    = height;
        srcBuffer.planes[0].rowStride = srcRowStride;
        srcBuffer.planes[0].basePtr   = srcAllocation;

        auto dstBuffer                = srcBuffer;
        dstBuffer.planes[0].rowStride = dstRowStride;
        dstBuffer.planes[0].basePtr   = dstAllocation;

        nvcv::Image srcImage = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8, srcBuffer});
        nvcv::Image dstImage = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGB8, dstBuffer});

        std::vector<uint8_t> srcHost(rowBytes * height);
        for (size_t i = 0; i < srcHost.size(); ++i)
        {
            srcHost[i] = static_cast<uint8_t>(i + 1);
        }

        ASSERT_EQ(cudaSuccess, cudaMemset(srcAllocation, 0xCC, srcBufferBytes));
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcAllocation, srcRowStride, srcHost.data(), rowBytes, rowBytes, height,
                                            cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemset(dstAllocation, 0xA5, dstBufferBytes));

        nvcv::ImageBatchVarShape batchSrc(1);
        nvcv::ImageBatchVarShape batchDst(1);
        batchSrc.pushBack(srcImage);
        batchDst.pushBack(dstImage);

        std::vector<int2> anchorVec{
            {0, 0}
        };
        std::vector<int3> erasingVec{
            {0, 0, 0x7}
        };
        std::vector<float> valuesVec{0, 0, 0};
        std::vector<int>   imgIdxVec{0};

        nvcv::Tensor anchor({{1}, "N"}, nvcv::TYPE_2S32);
        nvcv::Tensor erasing({{1}, "N"}, nvcv::TYPE_3S32);
        nvcv::Tensor values({{channels}, "N"}, nvcv::TYPE_F32);
        nvcv::Tensor imgIdx({{1}, "N"}, nvcv::TYPE_S32);

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
        CopyEraseParams(stream, {anchor, erasing, values, imgIdx}, {anchorVec, erasingVec, valuesVec, imgIdxVec});

        cvcuda::Erase op(1);
        EXPECT_NO_THROW(op(stream, batchSrc, batchDst, anchor, erasing, values, imgIdx, false, 0));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        std::vector<uint8_t> got(dstBufferBytes);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(got.data(), dstAllocation, dstBufferBytes, cudaMemcpyDeviceToHost));
        for (size_t y = 0; y < height; ++y)
        {
            EXPECT_TRUE(std::equal(srcHost.begin() + y * rowBytes, srcHost.begin() + (y + 1) * rowBytes,
                                   got.begin() + y * dstRowStride));
            EXPECT_EQ(0xA5, got[y * dstRowStride + rowBytes]);
        }
        EXPECT_TRUE(
            std::all_of(got.begin() + dstRowStride * height, got.end(), [](uint8_t value) { return value == 0xA5; }));

        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }

    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

TEST(OpEraseTensorCopyGold, tight_and_padded_planar_output_match_independent_gold)
{
    constexpr int numImages = 2;
    constexpr int channels  = 3;
    constexpr int width     = 16;
    constexpr int height    = 9;

    auto runCase = [&](bool padded)
    {
        SCOPED_TRACE(padded ? "padded fallback" : "tight bulk copy");

        const int64_t rowStride     = width + (padded ? 1 : 0);
        const int64_t channelStride = rowStride * height;
        const int64_t sampleStride  = channelStride * channels;
        const size_t  bufferBytes   = sampleStride * numImages;

        NVCVByte *srcAllocation{};
        NVCVByte *dstAllocation{};
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), bufferBytes));
        cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&dstAllocation), bufferBytes);
        if (status != cudaSuccess)
        {
            cudaFree(srcAllocation);
        }
        ASSERT_EQ(cudaSuccess, status);

        nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
        srcBuffer.basePtr    = srcAllocation;
        srcBuffer.strides[0] = sampleStride;
        srcBuffer.strides[1] = channelStride;
        srcBuffer.strides[2] = rowStride;
        srcBuffer.strides[3] = 1;
        auto dstBuffer       = srcBuffer;
        dstBuffer.basePtr    = dstAllocation;

        nvcv::TensorShape shape{
            {numImages, channels, height, width},
            "NCHW"
        };
        nvcv::Tensor src
            = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, srcBuffer},
                                   nvcv::TensorDataCleanupCallback{[srcAllocation](const nvcv::TensorData &)
                                                                   {
                                                                       cudaFree(srcAllocation);
                                                                   }});
        nvcv::Tensor dst
            = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, dstBuffer},
                                   nvcv::TensorDataCleanupCallback{[dstAllocation](const nvcv::TensorData &)
                                                                   {
                                                                       cudaFree(dstAllocation);
                                                                   }});

        auto srcData   = src.exportData<nvcv::TensorDataStridedCuda>();
        auto dstData   = dst.exportData<nvcv::TensorDataStridedCuda>();
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
        ASSERT_TRUE(srcAccess && dstAccess);
        ASSERT_EQ(rowStride, srcAccess->rowStride());
        ASSERT_EQ(channelStride, srcAccess->chStride());
        ASSERT_EQ(sampleStride, srcAccess->sampleStride());
        ASSERT_EQ(rowStride, dstAccess->rowStride());
        ASSERT_EQ(channelStride, dstAccess->chStride());
        ASSERT_EQ(sampleStride, dstAccess->sampleStride());
        EXPECT_EQ(padded, rowStride != width);

        std::vector<std::vector<uint8_t>> srcHost;
        for (int image = 0; image < numImages; ++image)
        {
            srcHost.emplace_back(MakeNonzeroU8Image(width, height, channels, image));
            const auto planes = nvcv::test::planar::DeinterleaveToPlanes(srcHost.back(), width, height, channels, 1);
            for (int c = 0; c < channels; ++c)
            {
                ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcAccess->sampleData(image) + c * srcAccess->chStride(),
                                                    srcAccess->rowStride(), planes.data() + c * width * height, width,
                                                    width, height, cudaMemcpyHostToDevice));
            }
        }
        ASSERT_EQ(cudaSuccess, cudaMemset(dstAllocation, 0xA5, bufferBytes));

        std::vector<int2> anchorVec{
            { 1, 1},
            {13, 7},
        };
        std::vector<int3> erasingVec{
            {6, 4, 0x7},
            {6, 5, 0x5},
        };
        std::vector<float> valuesVec{
            17, 29, 41, 53, 65, 77,
        };
        std::vector<int> imgIdxVec{0, 1};

        std::vector<std::vector<uint8_t>> gold = srcHost;
        const std::vector<nvcv::Size2D>   sizes(numImages, {width, height});
        const EraseParamValues            params{anchorVec, erasingVec, valuesVec, imgIdxVec};
        ApplyEraseGold(gold, sizes, channels, params);

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
        ASSERT_NO_FATAL_FAILURE(RunEraseWithParams(stream, src, dst, channels, params));

        for (int image = 0; image < numImages; ++image)
        {
            std::vector<uint8_t> planes(width * height * channels);
            for (int c = 0; c < channels; ++c)
            {
                ASSERT_EQ(cudaSuccess, cudaMemcpy2D(planes.data() + c * width * height, width,
                                                    dstAccess->sampleData(image) + c * dstAccess->chStride(),
                                                    dstAccess->rowStride(), width, height, cudaMemcpyDeviceToHost));
            }
            const auto got = nvcv::test::planar::InterleaveFromPlanes(planes, width, height, channels, 1);
            EXPECT_EQ(gold[image], got);
        }

        if (padded)
        {
            std::vector<uint8_t> raw(bufferBytes);
            ASSERT_EQ(cudaSuccess, cudaMemcpy(raw.data(), dstAllocation, bufferBytes, cudaMemcpyDeviceToHost));
            for (int plane = 0; plane < numImages * channels; ++plane)
            {
                for (int y = 0; y < height; ++y)
                {
                    const size_t paddingOffset = plane * channelStride + y * rowStride + width;
                    EXPECT_EQ(0xA5, raw[paddingOffset]);
                }
            }
        }

        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    };

    ASSERT_NO_FATAL_FAILURE(runCase(false));
    ASSERT_NO_FATAL_FAILURE(runCase(true));
}

NVCV_TEST_SUITE_P(OpErasePlanar, nvcv::test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
  // W, H, N, planar format, interleaved format
                                     {23, 17, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                                     {29, 19, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
                                     {21, 15, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                                     {25, 13, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

TEST_P(OpErasePlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>());
}

NVCV_TEST_SUITE_P(OpErasePlanarVarShape, nvcv::test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
  // W, H, N, planar format, interleaved format
                                             {23, 17, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                                             {29, 19, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
                                             {21, 15, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                                             {25, 13, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

TEST_P(OpErasePlanarVarShape, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>());
}

TEST(OpErasePlanarRandom, varshape_matches_interleaved)
{
    PlanarParityEraseRunner runner(3, 2, true);
    nvcv::test::planar::RunVarShapeParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 23, 17, 23, 17, 2, runner);
}

namespace {

struct EraseRegionTypeCase
{
    nvcv::DataType dtype;
    int            kind;
};

constexpr int kEraseRegionWidth  = 5;
constexpr int kEraseRegionHeight = 4;
constexpr int kEraseRegionBatch  = 2;

nvcv::Tensor MakeEraseRegionImage(std::string_view layout, int channels, nvcv::DataType dtype)
{
    if (layout == "NHWC")
        return nvcv::Tensor(
            {
                {kEraseRegionBatch, kEraseRegionHeight, kEraseRegionWidth, channels},
                "NHWC"
        },
            dtype);
    if (layout == "NCHW")
        return nvcv::Tensor(
            {
                {kEraseRegionBatch, channels, kEraseRegionHeight, kEraseRegionWidth},
                "NCHW"
        },
            dtype);
    if (layout == "HWC")
        return nvcv::Tensor(
            {
                {kEraseRegionHeight, kEraseRegionWidth, channels},
                "HWC"
        },
            dtype);
    return nvcv::Tensor(
        {
            {channels, kEraseRegionHeight, kEraseRegionWidth},
            "CHW"
    },
        dtype);
}

int EraseRegionStorageIndex(std::string_view layout, int channels, int c, int y, int x)
{
    if (layout == "NHWC" || layout == "HWC")
        return (y * kEraseRegionWidth + x) * channels + c;
    return (c * kEraseRegionHeight + y) * kEraseRegionWidth + x;
}

uint16_t EraseRegionHalfBits(float value)
{
    return __half_raw(__float2half(value)).x;
}

template<typename T>
T EraseRegionHostValue(float value, bool isHalf)
{
    if constexpr (std::is_same_v<T, uint16_t>)
        return isHalf ? EraseRegionHalfBits(value) : static_cast<uint16_t>(value);
    else
        return static_cast<T>(value);
}

template<typename T>
std::vector<T> MakeEraseRegionSource(int channels, int sample, bool isHalf)
{
    std::vector<T> source(channels * kEraseRegionHeight * kEraseRegionWidth);
    size_t         index = 0;
    for (T &element : source)
    {
        element = EraseRegionHostValue<T>(static_cast<float>((index + sample * 7) % 23), isHalf);
        ++index;
    }
    return source;
}

template<typename T>
void FillEraseRegionExpected(std::vector<T> &expected, std::string_view layout, int channels, bool isHalf)
{
    const T erasedValue = EraseRegionHostValue<T>(7.0f, isHalf);
    for (int c = 0; c < channels; ++c)
        for (int y = 1; y < 3; ++y)
            for (int x = 1; x < 4; ++x) expected[EraseRegionStorageIndex(layout, channels, c, y, x)] = erasedValue;
}

template<typename T>
void RunEraseRegionMatrixCase(cudaStream_t stream, cvcuda::Erase &op, nvcv::DataType dtype, bool isHalf,
                              std::string_view layout, int channels, bool inplace, bool floatValue)
{
    nvcv::Tensor input  = MakeEraseRegionImage(layout, channels, dtype);
    nvcv::Tensor output = MakeEraseRegionImage(layout, channels, dtype);
    nvcv::Tensor values({{1}, "W"}, floatValue ? nvcv::TYPE_F32 : dtype);

    const int samples = layout.size() == 4 ? kEraseRegionBatch : 1;
    for (int sample = 0; sample < samples; ++sample)
    {
        std::vector<T> source = MakeEraseRegionSource<T>(channels, sample, isHalf);
        nvcv::util::SetImageTensorFromVector<T>(input.exportData(), source, sample);
    }

    if (floatValue)
    {
        std::vector<float> hostValue{7.0f};
        ASSERT_EQ(cudaSuccess, cudaMemcpy(values.exportData<nvcv::TensorDataStridedCuda>()->basePtr(), hostValue.data(),
                                          sizeof(float), cudaMemcpyHostToDevice));
    }
    else
    {
        std::vector<T> hostValue{EraseRegionHostValue<T>(7.0f, isHalf)};
        ASSERT_EQ(cudaSuccess, cudaMemcpy(values.exportData<nvcv::TensorDataStridedCuda>()->basePtr(), hostValue.data(),
                                          sizeof(T), cudaMemcpyHostToDevice));
    }

    const nvcv::Tensor &destination = inplace ? input : output;
    op(stream, input, destination, 1, 1, 2, 3, values);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int sample = 0; sample < samples; ++sample)
    {
        std::vector<T> expected = MakeEraseRegionSource<T>(channels, sample, isHalf);
        FillEraseRegionExpected(expected, layout, channels, isHalf);

        std::vector<T> actual;
        nvcv::util::GetImageVectorFromTensor<T>(destination.exportData(), sample, actual);
        EXPECT_EQ(expected, actual);
    }
}

void DispatchEraseRegionMatrixCase(cudaStream_t stream, cvcuda::Erase &op, const EraseRegionTypeCase &typeCase,
                                   std::string_view layout, int channels, bool inplace, bool floatValue)
{
    switch (typeCase.kind)
    {
    case 0:
        RunEraseRegionMatrixCase<uint8_t>(stream, op, typeCase.dtype, false, layout, channels, inplace, floatValue);
        break;
    case 1:
        RunEraseRegionMatrixCase<int8_t>(stream, op, typeCase.dtype, false, layout, channels, inplace, floatValue);
        break;
    case 2:
        RunEraseRegionMatrixCase<uint16_t>(stream, op, typeCase.dtype, false, layout, channels, inplace, floatValue);
        break;
    case 3:
        RunEraseRegionMatrixCase<int16_t>(stream, op, typeCase.dtype, false, layout, channels, inplace, floatValue);
        break;
    case 4:
        RunEraseRegionMatrixCase<uint32_t>(stream, op, typeCase.dtype, false, layout, channels, inplace, floatValue);
        break;
    case 5:
        RunEraseRegionMatrixCase<int32_t>(stream, op, typeCase.dtype, false, layout, channels, inplace, floatValue);
        break;
    case 6:
        RunEraseRegionMatrixCase<uint64_t>(stream, op, typeCase.dtype, false, layout, channels, inplace, floatValue);
        break;
    case 7:
        RunEraseRegionMatrixCase<int64_t>(stream, op, typeCase.dtype, false, layout, channels, inplace, floatValue);
        break;
    case 8:
        RunEraseRegionMatrixCase<uint16_t>(stream, op, typeCase.dtype, true, layout, channels, inplace, floatValue);
        break;
    case 9:
        RunEraseRegionMatrixCase<float>(stream, op, typeCase.dtype, false, layout, channels, inplace, floatValue);
        break;
    default:
        RunEraseRegionMatrixCase<double>(stream, op, typeCase.dtype, false, layout, channels, inplace, floatValue);
        break;
    }
}

void RunEraseRegionValueModes(cudaStream_t stream, cvcuda::Erase &op, const EraseRegionTypeCase &typeCase,
                              std::string_view layout, int channels)
{
    for (bool inplace : {false, true})
    {
        DispatchEraseRegionMatrixCase(stream, op, typeCase, layout, channels, inplace, false);
        if (typeCase.dtype != nvcv::TYPE_F32)
            DispatchEraseRegionMatrixCase(stream, op, typeCase, layout, channels, inplace, true);
    }
}

void FillEraseRegionBroadcastExpected(std::vector<float> &expected, const std::vector<float> &values, int channels)
{
    constexpr int start  = 2;
    constexpr int height = 2;
    constexpr int width  = 3;

    for (int c = 0; c < channels; ++c)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                expected[(c * kEraseRegionHeight + y + start) * kEraseRegionWidth + x + start]
                    = values[(c * height + y) * width + x];
}

} // namespace

TEST(OpEraseRegion, all_scalar_dtypes_layouts_channels_and_inplace)
{
    const std::array<EraseRegionTypeCase, 11> types{
        {
         {nvcv::TYPE_U8, 0},
         {nvcv::TYPE_S8, 1},
         {nvcv::TYPE_U16, 2},
         {nvcv::TYPE_S16, 3},
         {nvcv::TYPE_U32, 4},
         {nvcv::TYPE_S32, 5},
         {nvcv::TYPE_U64, 6},
         {nvcv::TYPE_S64, 7},
         {nvcv::TYPE_F16, 8},
         {nvcv::TYPE_F32, 9},
         {nvcv::TYPE_F64, 10},
         }
    };
    constexpr std::array<std::string_view, 4> layouts{
        {"NHWC", "HWC", "NCHW", "CHW"}
    };

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::Erase op(0);

    for (const auto &typeCase : types)
        for (const auto &layout : layouts)
            for (int channels = 1; channels <= 4; ++channels)
                RunEraseRegionValueModes(stream, op, typeCase, layout, channels);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpEraseRegion, broadcast_and_python_slice_semantics_are_bit_exact)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int channels = 3;
    nvcv::Tensor  input(
         {
             {2, channels, 4, 5},
             "NCHW"
    },
         nvcv::TYPE_F32);
    nvcv::Tensor output(
        {
            {2, channels, 4, 5},
            "NCHW"
    },
        nvcv::TYPE_F32);
    nvcv::Tensor values(
        {
            {channels, 2, 3},
            "CHW"
    },
        nvcv::TYPE_F32);

    std::vector<float> source(channels * 4 * 5, -1.0f);
    nvcv::util::SetImageTensorFromVector<float>(input.exportData(), source, 0);
    nvcv::util::SetImageTensorFromVector<float>(input.exportData(), source, 1);

    std::vector<float> hostValues(channels * 2 * 3);
    for (size_t index = 0; index < hostValues.size(); ++index) hostValues[index] = static_cast<float>(index) + 0.25f;
    nvcv::util::SetImageTensorFromVector<float>(values.exportData(), hostValues, 0);

    cvcuda::Erase op(0);
    op(stream, input, output, -2, -3, 10, 10, values);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int sample = 0; sample < 2; ++sample)
    {
        std::vector<float> actual;
        nvcv::util::GetImageVectorFromTensor<float>(output.exportData(), sample, actual);
        std::vector<float> expected = source;
        FillEraseRegionBroadcastExpected(expected, hostValues, channels);
        EXPECT_EQ(expected, actual);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpEraseRegion, strided_values_are_bit_exact)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int channels      = 3;
    constexpr int valueHeight   = 2;
    constexpr int valueWidth    = 3;
    constexpr int valueRowPitch = 5;
    constexpr int valueChPitch  = valueHeight * valueRowPitch;

    NVCVByte *valueAllocation{};
    ASSERT_EQ(cudaSuccess,
              cudaMalloc(reinterpret_cast<void **>(&valueAllocation), channels * valueChPitch * sizeof(float)));

    nvcv::TensorDataStridedCuda::Buffer valueBuffer{};
    valueBuffer.basePtr    = valueAllocation;
    valueBuffer.strides[0] = valueChPitch * sizeof(float);
    valueBuffer.strides[1] = valueRowPitch * sizeof(float);
    valueBuffer.strides[2] = sizeof(float);
    nvcv::Tensor values    = nvcv::TensorWrapData(
           nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{channels, valueHeight, valueWidth}, "CHW"},
            nvcv::TYPE_F32, valueBuffer
    },
           nvcv::TensorDataCleanupCallback{[valueAllocation](const nvcv::TensorData &)
                                        {
                                            cudaFree(valueAllocation);
                                        }});

    std::vector<float> hostValues(channels * valueChPitch, -123.0f);
    for (int c = 0; c < channels; ++c)
        for (int y = 0; y < valueHeight; ++y)
            for (int x = 0; x < valueWidth; ++x)
                hostValues[c * valueChPitch + y * valueRowPitch + x]
                    = static_cast<float>(c * 20 + y * valueWidth + x) + 0.25f;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(valueAllocation, hostValues.data(), hostValues.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));

    nvcv::Tensor input(
        {
            {1, channels, 4, 5},
            "NCHW"
    },
        nvcv::TYPE_F32);
    nvcv::Tensor output(
        {
            {1, channels, 4, 5},
            "NCHW"
    },
        nvcv::TYPE_F32);
    std::vector<float> source(channels * 4 * 5, -1.0f);
    nvcv::util::SetImageTensorFromVector<float>(input.exportData(), source, 0);

    cvcuda::Erase op(0);
    op(stream, input, output, 1, 1, valueHeight, valueWidth, values);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<float> expected = source;
    for (int c = 0; c < channels; ++c)
        for (int y = 0; y < valueHeight; ++y)
            for (int x = 0; x < valueWidth; ++x)
                expected[(c * 4 + y + 1) * 5 + x + 1] = hostValues[c * valueChPitch + y * valueRowPitch + x];

    std::vector<float> actual;
    nvcv::util::GetImageVectorFromTensor<float>(output.exportData(), 0, actual);
    EXPECT_EQ(expected, actual);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpEraseRegion, float32_to_uint8_matches_torch_assignment_cast)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor input(
        {
            {1, 1, 1, 6},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor output(
        {
            {1, 1, 1, 6},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor values({{6}, "W"}, nvcv::TYPE_F32);
    ASSERT_EQ(cudaSuccess, cudaMemset(input.exportData<nvcv::TensorDataStridedCuda>()->basePtr(), 0, 6));
    std::vector<float> hostValues{-1.9f, 1.9f, 255.9f, 256.1f, 4294967296.0f, 1.0e20f};
    ASSERT_EQ(cudaSuccess, cudaMemcpy(values.exportData<nvcv::TensorDataStridedCuda>()->basePtr(), hostValues.data(),
                                      hostValues.size() * sizeof(float), cudaMemcpyHostToDevice));

    cvcuda::Erase op(0);
    op(stream, input, output, 0, 0, 1, 6, values);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> actual;
    nvcv::util::GetImageVectorFromTensor<uint8_t>(output.exportData(), 0, actual);
    EXPECT_EQ((std::vector<uint8_t>{255, 1, 255, 0, 0, 255}), actual);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpEraseRegion_Negative, rejects_unsupported_matrix_and_broadcast_complement)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::Erase op(0);

    nvcv::Tensor validInput(
        {
            {1, 4, 5, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor validOutput(
        {
            {1, 4, 5, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor scalarValue({{1}, "W"}, nvcv::TYPE_U8);

    auto expectInvalid
        = [&op, stream](const nvcv::Tensor &input, const nvcv::Tensor &output, const nvcv::Tensor &values)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, stream, &input, &output, &values]
                                                                 { op(stream, input, output, 1, 1, 2, 3, values); }));
    };

    nvcv::Tensor packedInput(
        {
            {1, 4, 5, 3},
            "NHWC"
    },
        nvcv::TYPE_2U8);
    nvcv::Tensor packedOutput(
        {
            {1, 4, 5, 3},
            "NHWC"
    },
        nvcv::TYPE_2U8);
    nvcv::Tensor packedValue({{1}, "W"}, nvcv::TYPE_2U8);
    expectInvalid(packedInput, packedOutput, packedValue);

    nvcv::Tensor fiveChannelInput(
        {
            {1, 4, 5, 5},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor fiveChannelOutput(
        {
            {1, 4, 5, 5},
            "NHWC"
    },
        nvcv::TYPE_U8);
    expectInvalid(fiveChannelInput, fiveChannelOutput, scalarValue);

    nvcv::Tensor wrongDtypeValue({{1}, "W"}, nvcv::TYPE_S16);
    expectInvalid(validInput, validOutput, wrongDtypeValue);

    nvcv::Tensor wrongBroadcastValue(
        {
            {4, 2, 3},
            "CHW"
    },
        nvcv::TYPE_U8);
    expectInvalid(validInput, validOutput, wrongBroadcastValue);

    nvcv::Tensor wrongLayoutOutput(
        {
            {1, 3, 4, 5},
            "NCHW"
    },
        nvcv::TYPE_U8);
    expectInvalid(validInput, wrongLayoutOutput, scalarValue);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpErase_Negative, nvcv::test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, std::string, nvcv::DataType, std::string, nvcv::DataType, std::string, nvcv::DataType, std::string, nvcv::DataType, int>
{
    //   in_format, out_format, anchor_layout, anchor_datatype, erasingData_layout, erasingData_datatype, imgIdxData_layout, imgIdxData_datatype, valuesData_layout, valuesData_type, num_erasing_area
    { nvcv::FMT_RGB8p, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // layout mismatch
    { nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid in layout
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8p, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // layout mismatch
    { nvcv::FMT_RGB8, nvcv::FMT_RGBf32, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // different datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2F32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid anchor datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "NHW", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid anchor dim
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 3}, // Invalid num of erasing area 3 (> max)
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3F32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid erasing datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "NHW", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid erasing dim
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_F32, "N", nvcv::TYPE_F32, 2}, // invalid imgIdx datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "NHW", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid imgIdx datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_S32, 2}, // invalid values datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "NHW", nvcv::TYPE_F32, 2}, // invalid values datatype
});

// clang-format on

TEST(OpErase_Negative, create_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaEraseCreate(nullptr, 1));
}

TEST(OpErase_Negative, create_negative_area)
{
    NVCVOperatorHandle handle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaEraseCreate(&handle, -1));
}

TEST(OpErase_Negative, rejectsInvalidTensorAreaParameters)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int numErasingArea = 1;
    cvcuda::Erase eraseOp(numErasingArea);

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 8, 8, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 8, 8, nvcv::FMT_RGB8);

    auto expectInvalid = [&](int2 anchorValue, int imgIdxValue, int valuesLen)
    {
        nvcv::Tensor anchor({{numErasingArea}, "N"}, nvcv::TYPE_2S32);
        nvcv::Tensor erasing({{numErasingArea}, "N"}, nvcv::TYPE_3S32);
        nvcv::Tensor values({{valuesLen}, "N"}, nvcv::TYPE_F32);
        nvcv::Tensor imgIdx({{numErasingArea}, "N"}, nvcv::TYPE_S32);

        std::vector<int2> anchorVec{anchorValue};
        std::vector<int3> erasingVec{
            {0, 0, 0x7}
        };
        std::vector<float> valuesVec(valuesLen, 1.f);
        std::vector<int>   imgIdxVec{imgIdxValue};

        CopyEraseParams(stream, {anchor, erasing, values, imgIdx}, {anchorVec, erasingVec, valuesVec, imgIdxVec});

        EXPECT_EQ(
            NVCV_ERROR_INVALID_ARGUMENT,
            nvcv::ProtectCall([&] { eraseOp(stream, imgIn, imgOut, anchor, erasing, values, imgIdx, false, 0); }));
    };

    expectInvalid({-1, 0}, 0, 3);
    expectInvalid({0, -1}, 0, 3);
    expectInvalid({0, 0}, 1, 3);
    expectInvalid({0, 0}, 0, 2);

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpErase_Negative, rejectsInvalidVarShapeAreaParameters)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int numErasingArea = 1;
    cvcuda::Erase eraseOp(numErasingArea);

    nvcv::ImageBatchVarShape batchSrc(1);
    nvcv::ImageBatchVarShape batchDst(1);
    batchSrc.pushBack(nvcv::Image{
        nvcv::Size2D{8, 8},
        nvcv::FMT_RGB8
    });
    batchDst.pushBack(nvcv::Image{
        nvcv::Size2D{8, 8},
        nvcv::FMT_RGB8
    });

    auto expectInvalid = [&](int2 anchorValue, int imgIdxValue, int valuesLen)
    {
        nvcv::Tensor anchor({{numErasingArea}, "N"}, nvcv::TYPE_2S32);
        nvcv::Tensor erasing({{numErasingArea}, "N"}, nvcv::TYPE_3S32);
        nvcv::Tensor values({{valuesLen}, "N"}, nvcv::TYPE_F32);
        nvcv::Tensor imgIdx({{numErasingArea}, "N"}, nvcv::TYPE_S32);

        std::vector<int2> anchorVec{anchorValue};
        std::vector<int3> erasingVec{
            {0, 0, 0x7}
        };
        std::vector<float> valuesVec(valuesLen, 1.f);
        std::vector<int>   imgIdxVec{imgIdxValue};

        CopyEraseParams(stream, {anchor, erasing, values, imgIdx}, {anchorVec, erasingVec, valuesVec, imgIdxVec});

        EXPECT_EQ(
            NVCV_ERROR_INVALID_ARGUMENT,
            nvcv::ProtectCall([&] { eraseOp(stream, batchSrc, batchDst, anchor, erasing, values, imgIdx, false, 0); }));
    };

    expectInvalid({-1, 0}, 0, 3);
    expectInvalid({0, -1}, 0, 3);
    expectInvalid({0, 0}, 1, 3);
    expectInvalid({0, 0}, 0, 2);

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpErase_Negative, varshape_batch_count_mismatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int numErasingArea = 1;
    cvcuda::Erase eraseOp(2);

    nvcv::ImageBatchVarShape batchSrc(2);
    nvcv::ImageBatchVarShape batchDst(2);
    batchSrc.pushBack(nvcv::Image{
        nvcv::Size2D{8, 8},
        nvcv::FMT_RGB8
    });
    batchDst.pushBack(nvcv::Image{
        nvcv::Size2D{8, 8},
        nvcv::FMT_RGB8
    });
    batchDst.pushBack(nvcv::Image{
        nvcv::Size2D{8, 8},
        nvcv::FMT_RGB8
    });

    nvcv::Tensor anchor({{numErasingArea}, "N"}, nvcv::TYPE_2S32);
    nvcv::Tensor erasing({{numErasingArea}, "N"}, nvcv::TYPE_3S32);
    nvcv::Tensor values({{numErasingArea * 3}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor imgIdx({{numErasingArea}, "N"}, nvcv::TYPE_S32);

    std::vector<int2> anchorVec{
        {0, 0}
    };
    std::vector<int3> erasingVec{
        {1, 1, 0x7}
    };
    std::vector<float> valuesVec{1.f, 1.f, 1.f};
    std::vector<int>   imgIdxVec{0};
    CopyEraseParams(stream, {anchor, erasing, values, imgIdx}, {anchorVec, erasingVec, valuesVec, imgIdxVec});

    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&] { eraseOp(stream, batchSrc, batchDst, anchor, erasing, values, imgIdx, false, 0); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpErase_Negative, infer_negative_parameter)
{
    nvcv::ImageFormat inputFmt             = GetParamValue<0>();
    nvcv::ImageFormat outputFmt            = GetParamValue<1>();
    std::string       anchor_layout        = GetParamValue<2>();
    nvcv::DataType    anchor_datatype      = GetParamValue<3>();
    std::string       erasingData_layout   = GetParamValue<4>();
    nvcv::DataType    erasingData_datatype = GetParamValue<5>();
    std::string       imgIdxData_layout    = GetParamValue<6>();
    nvcv::DataType    imgIdxData_datatype  = GetParamValue<7>();
    std::string       valuesData_layout    = GetParamValue<8>();
    nvcv::DataType    valuesData_datatype  = GetParamValue<9>();
    int               num_erasing_area     = GetParamValue<10>();

    int          max_num_erasing_area = 2;
    unsigned int seed                 = 0;

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 24, 24, inputFmt);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 24, 24, outputFmt);

    //parameters
    nvcv::TensorShape anchorShape = anchor_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, anchor_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, anchor_layout.c_str()};
    nvcv::TensorShape erasingShape = erasingData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, erasingData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, erasingData_layout.c_str()};
    nvcv::TensorShape imgIdxShape = imgIdxData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, imgIdxData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, imgIdxData_layout.c_str()};
    nvcv::TensorShape valuesShape = valuesData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, valuesData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, valuesData_layout.c_str()};
    nvcv::Tensor anchor(anchorShape, anchor_datatype);
    nvcv::Tensor erasing(erasingShape, erasingData_datatype);
    nvcv::Tensor values(valuesShape, valuesData_datatype);
    nvcv::Tensor imgIdx(imgIdxShape, imgIdxData_datatype);

    // Call operator
    cvcuda::Erase eraseOp(max_num_erasing_area);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&eraseOp, &imgIn, &imgOut, &anchor, &erasing, &values, &imgIdx, &seed]
                                { eraseOp(nullptr, imgIn, imgOut, anchor, erasing, values, imgIdx, false, seed); }));
}

TEST_P(OpErase_Negative, varshape_infer_negative_parameter)
{
    nvcv::ImageFormat inputFmt             = GetParamValue<0>();
    nvcv::ImageFormat outputFmt            = GetParamValue<1>();
    std::string       anchor_layout        = GetParamValue<2>();
    nvcv::DataType    anchor_datatype      = GetParamValue<3>();
    std::string       erasingData_layout   = GetParamValue<4>();
    nvcv::DataType    erasingData_datatype = GetParamValue<5>();
    std::string       imgIdxData_layout    = GetParamValue<6>();
    nvcv::DataType    imgIdxData_datatype  = GetParamValue<7>();
    std::string       valuesData_layout    = GetParamValue<8>();
    nvcv::DataType    valuesData_datatype  = GetParamValue<9>();
    int               num_erasing_area     = GetParamValue<10>();

    int          max_num_erasing_area = 2;
    unsigned int seed                 = 0;

    nvcv::ImageBatchVarShape batchSrc(1);
    nvcv::ImageBatchVarShape batchDst(1);
    batchSrc.pushBack(nvcv::Image{
        nvcv::Size2D{32, 32},
        inputFmt
    });
    batchDst.pushBack(nvcv::Image{
        nvcv::Size2D{32, 32},
        outputFmt
    });

    //parameters
    nvcv::TensorShape anchorShape = anchor_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, anchor_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, anchor_layout.c_str()};
    nvcv::TensorShape erasingShape = erasingData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, erasingData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, erasingData_layout.c_str()};
    nvcv::TensorShape imgIdxShape = imgIdxData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, imgIdxData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, imgIdxData_layout.c_str()};
    nvcv::TensorShape valuesShape = valuesData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, valuesData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, valuesData_layout.c_str()};
    nvcv::Tensor anchor(anchorShape, anchor_datatype);
    nvcv::Tensor erasing(erasingShape, erasingData_datatype);
    nvcv::Tensor values(valuesShape, valuesData_datatype);
    nvcv::Tensor imgIdx(imgIdxShape, imgIdxData_datatype);

    // Call operator
    cvcuda::Erase eraseOp(max_num_erasing_area);
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&eraseOp, &batchSrc, &batchDst, &anchor, &erasing, &values, &imgIdx, &seed]
                          { eraseOp(nullptr, batchSrc, batchDst, anchor, erasing, values, imgIdx, false, seed); }));
}

TEST(OpErase_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmt            = nvcv::FMT_RGB8;
    const int         numberOfImages = 5;
    unsigned int      seed           = 0;

    int srcWidthBase  = 4;
    int srcHeightBase = 4;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_RGBA8,             fmt},
        {            fmt, nvcv::FMT_RGBA8}
    };
    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
    {
        // Create input and output
        std::default_random_engine    randEng;
        std::uniform_int_distribution rndSrcWidth(ScaledSize(srcWidthBase, 0.8), ScaledSize(srcWidthBase, 1.1));
        std::uniform_int_distribution rndSrcHeight(ScaledSize(srcHeightBase, 0.8), ScaledSize(srcHeightBase, 1.1));

        int          num_erasing_area = 2;
        nvcv::Tensor anchor({{num_erasing_area}, "N"}, nvcv::TYPE_2S32);
        nvcv::Tensor erasing({{num_erasing_area}, "N"}, nvcv::TYPE_3S32);
        nvcv::Tensor values({{num_erasing_area}, "N"}, nvcv::TYPE_F32);
        nvcv::Tensor imgIdx({{num_erasing_area}, "N"}, nvcv::TYPE_S32);

        std::vector<nvcv::Image> imgSrc;

        std::vector<nvcv::Image> imgDst;

        for (int i = 0; i < numberOfImages - 1; ++i)
        {
            int tmpWidth  = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
            int tmpHeight = i == 0 ? srcHeightBase : rndSrcHeight(randEng);

            imgSrc.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);
            imgDst.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchDst(numberOfImages);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Generate test result
        cvcuda::Erase eraseOp(num_erasing_area);
        EXPECT_EQ(
            NVCV_ERROR_INVALID_ARGUMENT,
            nvcv::ProtectCall([&eraseOp, &batchSrc, &batchDst, &anchor, &erasing, &values, &imgIdx, &seed]
                              { eraseOp(nullptr, batchSrc, batchDst, anchor, erasing, values, imgIdx, false, seed); }));
    }

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

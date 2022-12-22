/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ConvUtils.hpp"
#include "Definitions.hpp"

#include <common/ValueTests.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/TypeTraits.hpp>

#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

#define VEC_EXPECT_NEAR(vec1, vec2, delta)                              \
    ASSERT_EQ(vec1.size(), vec2.size());                                \
    for (std::size_t idx = 0; idx < vec1.size(); ++idx)                 \
    {                                                                   \
        EXPECT_NEAR(vec1[idx], vec2[idx], delta) << "At index " << idx; \
    }

#define NVCV_IMAGE_FORMAT_Y16   NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, UNSIGNED, X000, X16)
#define NVCV_IMAGE_FORMAT_BGR16 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, ZYX1, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_RGB16 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_YUV8  NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, UNSIGNED, XYZ1, X8_Y8_Z8)
#define NVCV_IMAGE_FORMAT_NV21  NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601, 420, PL, UNSIGNED, XZY0, X8, X8_Y8)

// clang-format off

NVCV_TEST_SUITE_P(OpCvtColor,
test::ValueList<int, int, int, NVCVImageFormat, NVCVImageFormat, NVCVColorConversionCode, NVCVColorConversionCode, double>
{
    //  W,   H,  N,               inputFormat,            outputFormat,                in2outCode,               out2inCode, maxDiff
    { 176, 113,  1,    NVCV_IMAGE_FORMAT_BGR8,   NVCV_IMAGE_FORMAT_BGRA8,     NVCV_COLOR_BGR2BGRA,      NVCV_COLOR_BGRA2BGR,   0.0},
    { 336, 432,  2,    NVCV_IMAGE_FORMAT_RGB8,   NVCV_IMAGE_FORMAT_RGBA8,     NVCV_COLOR_RGB2RGBA,      NVCV_COLOR_RGBA2RGB,   0.0},
    {  77, 212,  3,    NVCV_IMAGE_FORMAT_BGR8,   NVCV_IMAGE_FORMAT_RGBA8,     NVCV_COLOR_BGR2RGBA,      NVCV_COLOR_RGBA2BGR,   0.0},
    {  33,  55,  4,    NVCV_IMAGE_FORMAT_RGB8,   NVCV_IMAGE_FORMAT_BGRA8,     NVCV_COLOR_RGB2BGRA,      NVCV_COLOR_BGRA2RGB,   0.0},
    { 123, 321,  5,   NVCV_IMAGE_FORMAT_RGBA8,   NVCV_IMAGE_FORMAT_BGRA8,    NVCV_COLOR_RGBA2BGRA,     NVCV_COLOR_BGRA2RGBA,   0.0},
    {  23,  21, 63,      NVCV_IMAGE_FORMAT_Y8,    NVCV_IMAGE_FORMAT_BGR8,     NVCV_COLOR_GRAY2BGR,      NVCV_COLOR_BGR2GRAY,   0.0},
    { 402, 202,  5,      NVCV_IMAGE_FORMAT_Y8,    NVCV_IMAGE_FORMAT_RGB8,     NVCV_COLOR_GRAY2RGB,      NVCV_COLOR_RGB2GRAY,   0.0},
    {  32,  21,  4,     NVCV_IMAGE_FORMAT_Y16,   NVCV_IMAGE_FORMAT_BGR16,     NVCV_COLOR_GRAY2BGR,      NVCV_COLOR_BGR2GRAY,   0.0},
    {  54,  66,  5,     NVCV_IMAGE_FORMAT_Y16,   NVCV_IMAGE_FORMAT_RGB16,     NVCV_COLOR_GRAY2RGB,      NVCV_COLOR_RGB2GRAY,   0.0},
    { 129,  61,  4,  NVCV_IMAGE_FORMAT_BGRf32, NVCV_IMAGE_FORMAT_BGRAf32,     NVCV_COLOR_BGR2BGRA,      NVCV_COLOR_BGRA2BGR,   0.0},
    {  63,  31,  3,  NVCV_IMAGE_FORMAT_RGBf32, NVCV_IMAGE_FORMAT_RGBAf32,     NVCV_COLOR_RGB2RGBA,      NVCV_COLOR_RGBA2RGB,   0.0},
    {  42, 111,  2,  NVCV_IMAGE_FORMAT_BGRf32, NVCV_IMAGE_FORMAT_RGBAf32,     NVCV_COLOR_BGR2RGBA,      NVCV_COLOR_RGBA2BGR,   0.0},
    {  21,  72,  2,  NVCV_IMAGE_FORMAT_RGBf32, NVCV_IMAGE_FORMAT_BGRAf32,     NVCV_COLOR_RGB2BGRA,      NVCV_COLOR_BGRA2RGB,   0.0},
    {  23,  31,  3, NVCV_IMAGE_FORMAT_RGBAf32, NVCV_IMAGE_FORMAT_BGRAf32,    NVCV_COLOR_RGBA2BGRA,     NVCV_COLOR_BGRA2RGBA,   0.0},
    // Codes 9 to 39 are not implemented
    {  55, 257,  4,    NVCV_IMAGE_FORMAT_BGR8,   NVCV_IMAGE_FORMAT_HSV8,       NVCV_COLOR_BGR2HSV,       NVCV_COLOR_HSV2BGR,   5.0},
    { 366,  14,  5,    NVCV_IMAGE_FORMAT_RGB8,   NVCV_IMAGE_FORMAT_HSV8,       NVCV_COLOR_RGB2HSV,       NVCV_COLOR_HSV2RGB,   5.0},
    // Codes 42 to 53 and 56 to 65 and 68 to 69 are not implemented
    { 112, 157,  4,    NVCV_IMAGE_FORMAT_BGR8,   NVCV_IMAGE_FORMAT_HSV8,  NVCV_COLOR_BGR2HSV_FULL,  NVCV_COLOR_HSV2BGR_FULL,   8.0},
    { 333,  13,  3,    NVCV_IMAGE_FORMAT_RGB8,   NVCV_IMAGE_FORMAT_HSV8,  NVCV_COLOR_RGB2HSV_FULL,  NVCV_COLOR_HSV2RGB_FULL,   8.0},
    // Codes 72 to 81 are not implemented
    { 133,  22,  2,    NVCV_IMAGE_FORMAT_YUV8,   NVCV_IMAGE_FORMAT_BGR8,       NVCV_COLOR_YUV2BGR,       NVCV_COLOR_BGR2YUV, 128.0},
    { 123,  21,  3,    NVCV_IMAGE_FORMAT_YUV8,   NVCV_IMAGE_FORMAT_RGB8,       NVCV_COLOR_YUV2RGB,       NVCV_COLOR_RGB2YUV, 128.0},
    // Codes 86 to 89 are not implemented
    // Codes 90 to 147 dealing with subsampled planes (NV12, etc. formats) are postponed (see comment below)
    //     Codes 109, 110, 113, 114 dealing with VYUY format are not implemented
    //     Codes 125, 126 dealing alpha premultiplication are not implemented
    //     Codes 135 to 139 dealing edge-aware demosaicing are not implemented
/*
    // NV12, ... makes tensors raise an error:
    // "NVCV_ERROR_NOT_IMPLEMENTED: Batch image format must not have subsampled planes, but it is: X"
    { 120,  20,  2,    NVCV_IMAGE_FORMAT_NV12,   NVCV_IMAGE_FORMAT_RGB8,  NVCV_COLOR_YUV2RGB_NV12,  NVCV_COLOR_RGB2YUV_NV12, 128.0},
    { 100,  40,  3,    NVCV_IMAGE_FORMAT_NV12,   NVCV_IMAGE_FORMAT_BGR8,  NVCV_COLOR_YUV2BGR_NV12,  NVCV_COLOR_BGR2YUV_NV12, 128.0},
    {  80, 120,  4,    NVCV_IMAGE_FORMAT_NV12,  NVCV_IMAGE_FORMAT_RGBA8, NVCV_COLOR_YUV2RGBA_NV12, NVCV_COLOR_RGBA2YUV_NV12, 128.0},
    {  60,  60,  5,    NVCV_IMAGE_FORMAT_NV12,  NVCV_IMAGE_FORMAT_BGRA8, NVCV_COLOR_YUV2BGRA_NV12, NVCV_COLOR_BGRA2YUV_NV12, 128.0},
    { 140,  80,  6,    NVCV_IMAGE_FORMAT_NV21,   NVCV_IMAGE_FORMAT_RGB8,  NVCV_COLOR_YUV2RGB_NV21,  NVCV_COLOR_RGB2YUV_NV21, 128.0},
    { 160,  60,  5,    NVCV_IMAGE_FORMAT_NV21,   NVCV_IMAGE_FORMAT_BGR8,  NVCV_COLOR_YUV2BGR_NV21,  NVCV_COLOR_BGR2YUV_NV21, 128.0},
    {  60, 100,  4,    NVCV_IMAGE_FORMAT_NV21,  NVCV_IMAGE_FORMAT_RGBA8, NVCV_COLOR_YUV2RGBA_NV21, NVCV_COLOR_RGBA2YUV_NV21, 128.0},
    {  80,  80,  3,    NVCV_IMAGE_FORMAT_NV21,  NVCV_IMAGE_FORMAT_BGRA8, NVCV_COLOR_YUV2BGRA_NV21, NVCV_COLOR_BGRA2YUV_NV21, 128.0},
    { 120,  40,  2,    NVCV_IMAGE_FORMAT_UYVY,   NVCV_IMAGE_FORMAT_RGB8,  NVCV_COLOR_YUV2RGB_UYVY,       NVCV_COLOR_RGB2YUV, 128.0},
    { 120,  40,  2,    NVCV_IMAGE_FORMAT_YUYV,   NVCV_IMAGE_FORMAT_RGB8,  NVCV_COLOR_YUV2RGB_YUYV,       NVCV_COLOR_RGB2YUV, 128.0},
*/
    // Code 148 is not implemented
});

#undef NVCV_IMAGE_FORMAT_Y16
#undef NVCV_IMAGE_FORMAT_BGR16
#undef NVCV_IMAGE_FORMAT_RGB16
#undef NVCV_IMAGE_FORMAT_YUV8
#undef NVCV_IMAGE_FORMAT_NV21

// clang-format on

TEST_P(OpCvtColor, correct_output)
{
    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat srcFormat{GetParamValue<3>()};
    nvcv::ImageFormat dstFormat{GetParamValue<4>()};

    NVCVColorConversionCode src2dstCode{GetParamValue<5>()};
    NVCVColorConversionCode dst2srcCode{GetParamValue<6>()};

    double maxDiff{GetParamValue<7>()};

    nvcv::Tensor srcTensor(batches, {width, height}, srcFormat);
    nvcv::Tensor dstTensor(batches, {width, height}, dstFormat);

    const auto *srcData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(srcTensor.exportData());
    const auto *dstData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(dstTensor.exportData());

    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);

    long srcBufSize = srcData->stride(0) * srcData->shape(0);

    std::vector<uint8_t> srcVec(srcBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::generate(srcVec.begin(), srcVec.end(), [&]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), srcBufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // run operator
    cvcuda::CvtColor cvtColorOp;

    EXPECT_NO_THROW(cvtColorOp(stream, srcTensor, dstTensor, src2dstCode));

    EXPECT_NO_THROW(cvtColorOp(stream, dstTensor, srcTensor, dst2srcCode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> testVec(srcBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), srcData->basePtr(), srcBufSize, cudaMemcpyDeviceToHost));

    VEC_EXPECT_NEAR(testVec, srcVec, maxDiff);
}

TEST_P(OpCvtColor, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat srcFormat{GetParamValue<3>()};
    nvcv::ImageFormat dstFormat{GetParamValue<4>()};

    NVCVColorConversionCode src2dstCode{GetParamValue<5>()};
    NVCVColorConversionCode dst2srcCode{GetParamValue<6>()};

    double maxDiff{GetParamValue<7>()};

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, srcFormat));

        int srcRowStride   = imgSrc[i]->size().w * srcFormat.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i]->size().h * srcRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        auto *imgData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        ASSERT_NE(imgData, nullptr);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i]->size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<std::unique_ptr<nvcv::Image>> imgDst;

    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(std::make_unique<nvcv::Image>(imgSrc[i]->size(), dstFormat));
    }

    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // run operator
    cvcuda::CvtColor cvtColorOp;

    EXPECT_NO_THROW(cvtColorOp(stream, batchSrc, batchDst, src2dstCode));

    EXPECT_NO_THROW(cvtColorOp(stream, batchDst, batchSrc, dst2srcCode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto *imgData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        ASSERT_NE(imgData, nullptr);
        ASSERT_EQ(imgData->numPlanes(), 1);

        std::vector<uint8_t> testVec(imgSrc[i]->size().h * srcVecRowStride[i]);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), srcVecRowStride[i], imgData->plane(0).basePtr,
                                            imgData->plane(0).rowStride, srcVecRowStride[i], imgSrc[i]->size().h,
                                            cudaMemcpyDeviceToHost));

        VEC_EXPECT_NEAR(testVec, srcVec[i], maxDiff);
    }
}

#undef VEC_EXPECT_NEAR

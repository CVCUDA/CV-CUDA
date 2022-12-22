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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpMorphology.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/TypeTraits.hpp>

#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

using uchar = unsigned char;

// checks pixels only in the logical image region.
template<class T>
static bool imageRegionValuesSame(test::TensorImageData &a, test::TensorImageData &b)
{
    int minWidth  = a.size().w > b.size().w ? b.size().w : a.size().w;
    int minHeight = a.size().h > b.size().h ? b.size().h : a.size().h;

    if (a.bytesPerC() != b.bytesPerC() || a.imageCHW() != b.imageCHW() || a.numC() != b.numC())
        return false;

    for (int x = 0; x < minWidth; ++x)
        for (int y = 0; y < minHeight; ++y)
            for (int c = 0; c < a.numC(); ++c)
                if (*a.item<T>(x, y, c) != *b.item<T>(x, y, c))
                    return false;

    return true;
}

template<class T, size_t rows, size_t cols>
void SetTensorToTestVector(const uchar inputVals[rows][cols], int width, int height, nvcv::Tensor &tensor, int sample)
{
    test::TensorImageData data(tensor.exportData(), sample);

    for (int x = 0; x < width; ++x)
        for (int y = 0; y < height; ++y)
            for (int c = 0; c < data.numC(); ++c) *data.item<T>(x, y, c) = (T)inputVals[y][x];

    EXPECT_NO_THROW(test::SetTensorFromVector<T>(tensor.exportData(), data.getVector(), sample));
}

template<class T, size_t rows, size_t cols>
bool MatchTensorToTestVector(const uchar checkVals[rows][cols], int width, int height, nvcv::Tensor &Tensor, int sample)
{
    test::TensorImageData data(Tensor.exportData(), sample);
    for (int x = 0; x < width; ++x)
        for (int y = 0; y < height; ++y)
            for (int c = 0; c < data.numC(); ++c)
                if (*data.item<T>(x, y, c) != (T)checkVals[y][x])
                {
                    return false;
                }

    return true;
}

template<class T, size_t rows, size_t cols>
void checkTestVectors(cudaStream_t &stream, nvcv::Tensor &inTensor, nvcv::Tensor &outTensor,
                      const uchar input[rows][cols], const uchar output[rows][cols], int width, int height,
                      const nvcv::Size2D &maskSize, const int2 &anchor, int iteration, NVCVMorphologyType type,
                      NVCVBorderType borderMode, int batches)
{
    for (int i = 0; i < batches; ++i)
    {
        SetTensorToTestVector<uchar, rows, cols>(input, width, height, inTensor, i);
    }

    cvcuda::Morphology morphOp(0);
    morphOp(stream, inTensor, outTensor, type, maskSize, anchor, iteration, borderMode);

    if (cudaSuccess != cudaStreamSynchronize(stream))
        throw std::runtime_error("Cuda Sync failed");

    for (int i = 0; i < batches; ++i)
    {
        if (MatchTensorToTestVector<uchar, rows, cols>(output, width, height, outTensor, i) != true)
        {
            throw std::runtime_error("Op returned unexpected result");
        }
    }
}

TEST(OpMorphology, morph_check_dilate_kernel)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 5;
    constexpr int height  = 5;
    int           batches = 3;

    nvcv::ImageFormat  format{NVCV_IMAGE_FORMAT_U8};
    nvcv::Tensor       inTensor(batches, {width, height}, format);
    nvcv::Tensor       outTensor(batches, {width, height}, format);
    int2               anchor(-1, -1);
    nvcv::Size2D       maskSize(3, 3);
    int                iteration  = 1;
    NVCVMorphologyType type       = NVCVMorphologyType::NVCV_DILATE;
    NVCVBorderType     borderMode = NVCVBorderType::NVCV_BORDER_CONSTANT;

    {
        // clang-format off
        uchar inImg[height][width] ={
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    };

        uchar expImg[height][width] ={
                        {0,0,0,0,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,0,0,0,0}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
    }

    // iteration = 2
    {
        // clang-format off
        iteration = 2;
        uchar inImg[height][width] ={
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    };

        uchar expImg[height][width] ={
                        {1,1,1,1,1},
                        {1,1,1,1,1},
                        {1,1,1,1,1},
                        {1,1,1,1,1},
                        {1,1,1,1,1}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
        iteration = 1;
    }

    {
        // overlap
        // clang-format off
        uchar inImg[height][width] ={
                        {1,0,0,0,2},
                        {0,0,0,0,0},
                        {0,0,5,0,0},
                        {0,0,0,0,0},
                        {4,0,0,0,3}
                    };

        uchar expImg[height][width] ={
                        {1,1,0,2,2},
                        {1,5,5,5,2},
                        {0,5,5,5,0},
                        {4,5,5,5,3},
                        {4,4,0,3,3}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
    }

    {
        // mask
        // clang-format off
        maskSize.w = 1;
        maskSize.h = 2;
        uchar inImg[height][width] ={
                        {1,0,0,0,2},
                        {0,0,0,0,0},
                        {0,0,5,0,0},
                        {0,0,0,0,0},
                        {4,0,0,0,3}
                    };

        uchar expImg[height][width] ={
                        {1,0,0,0,2},
                        {1,0,0,0,2},
                        {0,0,5,0,0},
                        {0,0,5,0,0},
                        {4,0,0,0,3}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
        maskSize.w = 3;
        maskSize.h = 3;
    }

    // anchor
    {
        // clang-format off
        anchor.x = 0;
        anchor.y = 0;

        uchar inImg[height][width] ={
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    };

        uchar expImg[height][width]  ={
                        {1,1,1,0,0},
                        {1,1,1,0,0},
                        {1,1,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
        anchor.x = -1;
        anchor.y = -1;
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpMorphology, morph_check_erode_kernel)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 5;
    constexpr int height  = 5;
    int           batches = 3;

    nvcv::ImageFormat  format{NVCV_IMAGE_FORMAT_U8};
    nvcv::Tensor       inTensor(batches, {width, height}, format);
    nvcv::Tensor       outTensor(batches, {width, height}, format);
    int2               anchor(-1, -1);
    nvcv::Size2D       maskSize(3, 3);
    int                iteration  = 1;
    NVCVMorphologyType type       = NVCVMorphologyType::NVCV_ERODE;
    NVCVBorderType     borderMode = NVCVBorderType::NVCV_BORDER_CONSTANT;

    {
        // clang-format off
        uchar inImg[height][width] ={
                        {0,0,0,0,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,0,0,0,0}
                    };

        uchar expImg[height][width] ={
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpMorphology, morph_check_dilate_kernel_even)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 6;
    constexpr int height  = 6;
    int           batches = 3;

    nvcv::ImageFormat  format{NVCV_IMAGE_FORMAT_U8};
    nvcv::Tensor       inTensor(batches, {width, height}, format);
    nvcv::Tensor       outTensor(batches, {width, height}, format);
    int2               anchor(-1, -1);
    nvcv::Size2D       maskSize(3, 3);
    int                iteration  = 1;
    NVCVMorphologyType type       = NVCVMorphologyType::NVCV_DILATE;
    NVCVBorderType     borderMode = NVCVBorderType::NVCV_BORDER_CONSTANT;

    {
        // clang-format off
         uchar inImg[height][width] ={
                        {1,0,0,0,0,2},
                        {0,0,0,0,0,0},
                        {0,0,5,0,0,0},
                        {0,0,0,0,0,0},
                        {0,0,0,0,0,0},
                        {4,0,0,0,0,3}
                    };

        uchar expImg[height][width] ={
                        {1,1,0,0,2,2},
                        {1,5,5,5,2,2},
                        {0,5,5,5,0,0},
                        {0,5,5,5,0,0},
                        {4,4,0,0,3,3},
                        {4,4,0,0,3,3}
                    };

        EXPECT_NO_THROW((checkTestVectors<uchar,width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,anchor,iteration, type, borderMode, batches)));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpMorphology, test::ValueList<int, int, int, NVCVImageFormat, int, int, NVCVBorderType, NVCVMorphologyType>
{
    // width, height, batches,                    format,  maskWidth, maskHeight,            borderMode, morphType
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_U8,          2,         2,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_RGBAf32,     3,         3,   NVCV_BORDER_CONSTANT, NVCV_DILATE},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,         3,   NVCV_BORDER_CONSTANT, NVCV_DILATE},
    {    125,     35,       1,      NVCV_IMAGE_FORMAT_RGBA8,       3,         3,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {     52,     45,       1,      NVCV_IMAGE_FORMAT_U16,         3,         3,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {    325,     45,       3,      NVCV_IMAGE_FORMAT_RGB8,        3,         3,   NVCV_BORDER_CONSTANT, NVCV_DILATE},
    {     25,     45,       1,      NVCV_IMAGE_FORMAT_U8,          3,         3,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,         3,   NVCV_BORDER_CONSTANT, NVCV_DILATE},

});

// clang-format on

TEST_P(OpMorphology, morph_noop)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                width      = GetParamValue<0>();
    int                height     = GetParamValue<1>();
    int                batches    = GetParamValue<2>();
    NVCVBorderType     borderMode = GetParamValue<6>();
    NVCVMorphologyType morphType  = GetParamValue<7>();

    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_U8};

    nvcv::Tensor inTensor(batches, {width, height}, format);
    nvcv::Tensor outTensor(batches, {width, height}, format);

    EXPECT_NO_THROW(test::SetTensorToRandomValue<uint8_t>(inTensor.exportData(), 0, 0xFF));
    EXPECT_NO_THROW(test::SetTensorTo<uint8_t>(outTensor.exportData(), 0));

    cvcuda::Morphology morphOp(0);
    int2               anchor(0, 0);

    nvcv::Size2D maskSize(1, 1);
    int          iteration = 0;
    EXPECT_NO_THROW(morphOp(stream, inTensor, outTensor, morphType, maskSize, anchor, iteration, borderMode));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batches; ++i)
    {
        test::TensorImageData cvTensorIn(inTensor.exportData());
        test::TensorImageData cvTensorOut(outTensor.exportData());
        EXPECT_TRUE(imageRegionValuesSame<uint8_t>(cvTensorIn, cvTensorOut));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpMorphology, morph_random)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    nvcv::Size2D maskSize;
    maskSize.w                    = GetParamValue<4>();
    maskSize.h                    = GetParamValue<5>();
    NVCVBorderType     borderMode = GetParamValue<6>();
    NVCVMorphologyType morphType  = GetParamValue<7>();

    int  iteration = 1;
    int3 shape{width, height, batches};

    nvcv::Tensor inTensor(batches, {width, height}, format);
    nvcv::Tensor outTensor(batches, {width, height}, format);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(inTensor.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(outTensor.exportData());

    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    long3 inStrides{inData->stride(0), inData->stride(1), inData->stride(2)};
    long3 outStrides{outData->stride(0), outData->stride(1), outData->stride(2)};

    long inBufSize  = inStrides.x * inData->shape(0);
    long outBufSize = outStrides.x * outData->shape(0);

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::generate(inVec.begin(), inVec.end(), [&]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::Morphology morphOp(0);
    int2               anchor(-1, -1);

    EXPECT_NO_THROW(morphOp(stream, inTensor, outTensor, morphType, maskSize, anchor, iteration, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    int2 kernelAnchor{maskSize.w / 2, maskSize.h / 2};
    test::Morph(goldVec, outStrides, inVec, inStrides, shape, format, maskSize, kernelAnchor, borderMode, morphType);

    EXPECT_EQ(testVec, goldVec);
}

// clang-format off
NVCV_TEST_SUITE_P(OpMorphologyVarShape, test::ValueList<int, int, int, NVCVImageFormat, int, int, NVCVBorderType, NVCVMorphologyType>
{
    // width, height, batches,                    format,  maskWidth, maskHeight,            borderMode, morphType
    {      5,      5,       5,      NVCV_IMAGE_FORMAT_U8,          3,        3,    NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_RGBAf32,     3,         3,   NVCV_BORDER_CONSTANT, NVCV_DILATE},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          2,         2,   NVCV_BORDER_CONSTANT, NVCV_DILATE},
    {    125,     35,       1,      NVCV_IMAGE_FORMAT_RGBA8,       3,         3,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {     52,     45,       1,      NVCV_IMAGE_FORMAT_U16,         1,         2,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {    325,     45,       3,      NVCV_IMAGE_FORMAT_RGB8,        3,         4,   NVCV_BORDER_CONSTANT, NVCV_DILATE},
    {     25,     45,       4,      NVCV_IMAGE_FORMAT_U8,          3,         3,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          -1,       -1,   NVCV_BORDER_CONSTANT, NVCV_DILATE}

});

// clang-format on

TEST_P(OpMorphologyVarShape, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int maskSizeX = GetParamValue<4>();
    int maskSizeY = GetParamValue<5>();

    int                anchorX    = -1;
    int                anchorY    = -1;
    int                iteration  = 1;
    NVCVBorderType     borderMode = GetParamValue<6>();
    NVCVMorphologyType morphType  = GetParamValue<7>();

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    //setup the input images
    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format));

        int srcRowStride   = imgSrc[i]->size().w * format.planePixelStrideBytes(0);
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
        imgDst.emplace_back(std::make_unique<nvcv::Image>(imgSrc[i]->size(), imgSrc[i]->format()));
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create kernel mask size tensor
    nvcv::Tensor maskTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto *dev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(maskTensor.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{maskSizeX, maskSizeY});

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create Anchor tensor
    nvcv::Tensor anchorTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto *dev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(anchorTensor.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{anchorX, anchorY});

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator set the max batches
    cvcuda::Morphology morphOp(batches);

    EXPECT_NO_THROW(morphOp(stream, batchSrc, batchDst, morphType, maskTensor, anchorTensor, iteration, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto *dstData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgDst[i]->exportData());
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowStride = srcVecRowStride[i];

        int3  shape{srcData->plane(0).width, srcData->plane(0).height, 1};
        long3 pitches{shape.y * dstRowStride, dstRowStride, format.planePixelStrideBytes(0)};

        std::vector<uint8_t> testVec(shape.y * pitches.y);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        // Generate gold result

        if (maskSizeX == -1 || maskSizeY == -1)
        {
            maskSizeX = 3;
            maskSizeY = 3;
        }
        nvcv::Size2D         maskSize{maskSizeX, maskSizeY};
        int2                 kernelAnchor{maskSize.w / 2, maskSize.h / 2};
        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        //generate gold result
        test::Morph(goldVec, pitches, srcVec[i], pitches, shape, format, maskSize, kernelAnchor, borderMode, morphType);

        EXPECT_EQ(testVec, goldVec);
    }
}

TEST_P(OpMorphologyVarShape, varshape_noop)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int maskSizeX = GetParamValue<4>();
    int maskSizeY = GetParamValue<5>();

    int                anchorX    = -1;
    int                anchorY    = -1;
    int                iteration  = 0; // this will bypass and do a copy
    NVCVBorderType     borderMode = GetParamValue<6>();
    NVCVMorphologyType morphType  = GetParamValue<7>();

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    //setup the input images
    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format));

        int srcRowStride   = imgSrc[i]->size().w * format.planePixelStrideBytes(0);
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
        imgDst.emplace_back(std::make_unique<nvcv::Image>(imgSrc[i]->size(), imgSrc[i]->format()));
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create kernel mask size tensor
    nvcv::Tensor maskTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto *dev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(maskTensor.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{maskSizeX, maskSizeY});

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create Anchor tensor
    nvcv::Tensor anchorTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto *dev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(anchorTensor.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{anchorX, anchorY});

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator set the max batches
    cvcuda::Morphology morphOp(batches);

    EXPECT_NO_THROW(morphOp(stream, batchSrc, batchDst, morphType, maskTensor, anchorTensor, iteration, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto *dstData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgDst[i]->exportData());
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowStride = srcVecRowStride[i];

        int3  shape{srcData->plane(0).width, srcData->plane(0).height, 1};
        long3 pitches{shape.y * dstRowStride, dstRowStride, format.planePixelStrideBytes(0)};

        std::vector<uint8_t> testVec(shape.y * pitches.y);
        std::vector<uint8_t> goldVec(shape.y * pitches.y); // should be the same as source with iteration == 0

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(goldVec.data(), srcVecRowStride[i], srcData->plane(0).basePtr,
                               srcData->plane(0).rowStride, srcVecRowStride[i], shape.y, cudaMemcpyDeviceToHost));

        EXPECT_EQ(testVec, goldVec);
    }
}

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

#include "Definitions.hpp"

#include <common/ValueTests.hpp>
#include <cvcuda/OpRotate.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <cmath>
#include <random>

namespace test = nvcv::test;
namespace t    = ::testing;

#define PI 3.1415926535897932384626433832795

// #define DBG_ROTATE 1

static void compute_warpAffine(const double angle, const double xShift, const double yShift, double *aCoeffs)
{
    aCoeffs[0] = cos(angle * PI / 180);
    aCoeffs[1] = sin(angle * PI / 180);
    aCoeffs[2] = xShift;
    aCoeffs[3] = -sin(angle * PI / 180);
    aCoeffs[4] = cos(angle * PI / 180);
    aCoeffs[5] = yShift;
}

static void compute_center_shift(const int center_x, const int center_y, const double angle, double &xShift,
                                 double &yShift)
{
    xShift = (1 - cos(angle * PI / 180)) * center_x - sin(angle * PI / 180) * center_y;
    yShift = sin(angle * PI / 180) * center_x + (1 - cos(angle * PI / 180)) * center_y;
}

static void assignCustomValuesInSrc(std::vector<uint8_t> &srcVec, int srcWidth, int srcHeight, int srcVecRowStride)
{
    int initialValue = 1;
    int pixelBytes   = static_cast<int>(srcVecRowStride / srcWidth);
    for (int i = 0; i < srcHeight; i++)
    {
        for (int j = 0; j < srcVecRowStride; j = j + pixelBytes)
        {
            for (int k = 0; k < pixelBytes; k++)
            {
                srcVec[i * srcVecRowStride + j + k] = initialValue;
            }
            initialValue++;
        }
    }

#if DBG_ROTATE
    std::cout << "\nPrint input " << std::endl;

    for (int i = 0; i < srcHeight; i++)
    {
        for (int j = 0; j < srcVecRowStride; j++)
        {
            std::cout << static_cast<int>(srcVec[i * srcVecRowStride + j]) << ",";
        }
        std::cout << std::endl;
    }
#endif
}

template<typename T>
static void Rotate(std::vector<T> &hDst, int dstRowStride, nvcv::Size2D dstSize, const std::vector<T> &hSrc,
                   int srcRowStride, nvcv::Size2D srcSize, nvcv::ImageFormat fmt, const double angleDeg,
                   const double2 shift, NVCVInterpolationType interpolation)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    T       *dstPtr = hDst.data();
    const T *srcPtr = hSrc.data();

    // calculate coefficients
    double d_aCoeffs[6];
    compute_warpAffine(angleDeg, shift.x, shift.y, d_aCoeffs);

    int width  = dstSize.w;
    int height = dstSize.h;

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            if (interpolation == NVCV_INTERP_LINEAR)
            {
                const double dst_x_shift = dst_x - d_aCoeffs[2];
                const double dst_y_shift = dst_y - d_aCoeffs[5];
                float        src_x       = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
                float        src_y       = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

                if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
                {
                    const int x1 = src_x > 0 ? std::floor(src_x) : std::rint(src_x);
                    const int y1 = src_y > 0 ? std::floor(src_y) : std::rint(src_y);

                    const int x2      = x1 + 1;
                    const int y2      = y1 + 1;
                    const int x2_read = std::min(x2, width - 1);
                    const int y2_read = std::min(y2, height - 1);

                    for (int k = 0; k < elementsPerPixel; k++)
                    {
                        float out = 0.;

                        T src_reg = srcPtr[y1 * srcRowStride + x1 * elementsPerPixel + k];
                        out       = out + src_reg * ((x2 - src_x) * (y2 - src_y));

                        src_reg = srcPtr[y1 * srcRowStride + x2_read * elementsPerPixel + k];
                        out     = out + src_reg * ((src_x - x1) * (y2 - src_y));

                        src_reg = srcPtr[y2_read * srcRowStride + x1 * elementsPerPixel + k];
                        out     = out + src_reg * ((x2 - src_x) * (src_y - y1));

                        src_reg = srcPtr[y2_read * srcRowStride + x2_read * elementsPerPixel + k];
                        out     = out + src_reg * ((src_x - x1) * (src_y - y1));

                        out = std::rint(out);
                        dstPtr[dst_y * dstRowStride + dst_x * elementsPerPixel + k]
                            = out < 0 ? 0 : (out > 255 ? 255 : out);
                    }
                }
            }
            else if (interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_CUBIC)
            {
                /*
                    Use this for NVCV_INTERP_CUBIC interpolation only for angles - {90, 180}
                */

                const double dst_x_shift = dst_x - d_aCoeffs[2];
                const double dst_y_shift = dst_y - d_aCoeffs[5];

                float src_x = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
                float src_y = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

                if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
                {
                    const int x1 = std::min(static_cast<int>(src_x + 0.5), width - 1);
                    const int y1 = std::min(static_cast<int>(src_y + 0.5), height - 1);

                    for (int k = 0; k < elementsPerPixel; k++)
                    {
                        dstPtr[dst_y * dstRowStride + dst_x * elementsPerPixel + k]
                            = srcPtr[y1 * srcRowStride + x1 * elementsPerPixel + k];
                    }
                }
            }
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpRotate, test::ValueList<int, int, int, int, NVCVInterpolationType, int, double>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,         interpolation, numberImages, angle
    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           1,     90},
    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           4,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            1,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            4,     90},

    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           1,     45},
    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           4,     45},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            1,     45},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            4,     45},

    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             1,     90},
    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             4,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             1,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             4,     90},

    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             1,     180},
    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             4,     180},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             1,     180},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             4,     180},
});

// clang-format on

TEST_P(OpRotate, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    NVCVInterpolationType interpolation = GetParamValue<4>();

    int numberOfImages = GetParamValue<5>();

    double angleDeg = GetParamValue<6>();
    double shiftX   = -1;
    double shiftY   = -1;

    const nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    // Generate input
    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);

    const auto *srcData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgSrc.exportData());

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    int                               srcVecRowStride = srcWidth * fmt.planePixelStrideBytes(0);

    for (int i = 0; i < numberOfImages; ++i)
    {
        srcVec[i].resize(srcHeight * srcVecRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return 0; });

        // Assign custom values in input vector
        assignCustomValuesInSrc(srcVec[i], srcWidth, srcHeight, srcVecRowStride);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(), srcVecRowStride,
                               srcVecRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, fmt);

    // Compute shiftX, shiftY using center
    int center_x = (srcWidth - 1) / 2, center_y = (srcHeight - 1) / 2;
    compute_center_shift(center_x, center_y, angleDeg, shiftX, shiftY);

    cvcuda::Rotate RotateOp(0);
    double2        shift = {shiftX, shiftY};
    EXPECT_NO_THROW(RotateOp(stream, imgSrc, imgDst, angleDeg, shift, interpolation));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    const auto *dstData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgDst.exportData());
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowStride = dstWidth * fmt.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(dstHeight * dstVecRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowStride, dstAccess->sampleData(i), dstAccess->rowStride(),
                               dstVecRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstVecRowStride);
        std::generate(goldVec.begin(), goldVec.end(), [&]() { return 0; });

        // Generate gold result
        Rotate<uint8_t>(goldVec, dstVecRowStride, {dstWidth, dstHeight}, srcVec[i], srcVecRowStride,
                        {srcWidth, srcHeight}, fmt, angleDeg, shift, interpolation);

#if DBG_ROTATE
        std::cout << "\nPrint golden output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowStride; j++)
            {
                std::cout << static_cast<int>(goldVec[k * dstVecRowStride + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint rotated output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowStride; j++)
            {
                std::cout << static_cast<int>(testVec[k * dstVecRowStride + j]) << ",";
            }
            std::cout << std::endl;
        }
#endif

        EXPECT_EQ(goldVec, testVec);
    }
}

TEST_P(OpRotate, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidthBase  = GetParamValue<0>();
    int srcHeightBase = GetParamValue<1>();

    NVCVInterpolationType interpolation = GetParamValue<4>();

    int numberOfImages = GetParamValue<5>();

    double angleDegBase = GetParamValue<6>();

    const nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    // Create input and output
    std::default_random_engine         randEng;
    std::uniform_int_distribution<int> rndSrcWidth(srcWidthBase * 0.8, srcWidthBase * 1.1);
    std::uniform_int_distribution<int> rndSrcHeight(srcHeightBase * 0.8, srcHeightBase * 1.1);
    std::uniform_int_distribution<int> rndAngle(0, 360);

    nvcv::Tensor angleDegTensor(nvcv::TensorShape({numberOfImages}, "N"), nvcv::TYPE_F64);
    const auto  *angleDegTensorData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(angleDegTensor.exportData());
    ASSERT_NE(nullptr, angleDegTensorData);

    nvcv::Tensor shiftTensor(nvcv::TensorShape({numberOfImages, 2}, nvcv::TensorLayout::NW), nvcv::TYPE_F64);
    const auto  *shiftTensorData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(shiftTensor.exportData());
    ASSERT_NE(nullptr, shiftTensorData);

    auto shiftTensorDataAccess = nvcv::TensorDataAccessStrided::Create(*shiftTensorData);
    ASSERT_TRUE(shiftTensorDataAccess);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc, imgDst;
    std::vector<double>                       angleDegVecs;
    std::vector<double2>                      shiftVecs;

    for (int i = 0; i < numberOfImages; ++i)
    {
        int tmpWidth  = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
        int tmpHeight = i == 0 ? srcHeightBase : rndSrcHeight(randEng);

        imgSrc.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{tmpWidth, tmpHeight}, fmt));

        imgDst.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{tmpWidth, tmpHeight}, fmt));

        double2 shift    = {-1, -1};
        double  angleDeg = i == 0 ? angleDegBase : rndAngle(randEng);
        if (i != 0 && interpolation == NVCV_INTERP_CUBIC)
        {
            // Use the computed angle as rand int and
            // then compute index to pick one of the angles
            std::vector<double> tmpAngleValues = {90, 180, 270};
            size_t              indexToChoose  = static_cast<size_t>(angleDeg) % tmpAngleValues.size();
            angleDeg                           = tmpAngleValues[indexToChoose];
        }

        // Compute shiftX, shiftY using center
        int center_x = (tmpWidth - 1) / 2, center_y = (tmpHeight - 1) / 2;
        compute_center_shift(center_x, center_y, angleDeg, shift.x, shift.y);

        angleDegVecs.push_back(angleDeg);
        shiftVecs.push_back(shift);
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(angleDegTensorData->basePtr(), angleDegVecs.data(),
                                           angleDegVecs.size() * sizeof(double), cudaMemcpyHostToDevice, stream));

    ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(shiftTensorDataAccess->sampleData(0),
                                             shiftTensorDataAccess->sampleStride(), shiftVecs.data(), sizeof(double2),
                                             sizeof(double2), numberOfImages, cudaMemcpyHostToDevice, stream));

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    std::vector<int>                  srcVecRowStride(numberOfImages);

    // Populate input
    for (int i = 0; i < numberOfImages; ++i)
    {
        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return 0; });

        // Assign custom values in input vector
        assignCustomValuesInSrc(srcVec[i], srcWidth, srcHeight, srcRowStride);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(), srcRowStride,
                               srcRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    cvcuda::Rotate rotateOp(numberOfImages);
    EXPECT_NO_THROW(rotateOp(stream, batchSrc, batchDst, angleDegTensor, shiftTensor, interpolation));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        assert(srcData->numPlanes() == 1);
        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        const auto *dstData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgDst[i]->exportData());
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * fmt.planePixelStrideBytes(0);
        int srcRowStride = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dstHeight * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstRowStride);
        std::generate(goldVec.begin(), goldVec.end(), [&]() { return 0; });

        // Generate gold result
        Rotate<uint8_t>(goldVec, dstRowStride, {dstWidth, dstHeight}, srcVec[i], srcRowStride, {srcWidth, srcHeight},
                        fmt, angleDegVecs[i], shiftVecs[i], interpolation);

        EXPECT_EQ(goldVec, testVec);
    }
}

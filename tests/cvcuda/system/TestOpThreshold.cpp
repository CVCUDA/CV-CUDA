/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpThreshold.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <util/TensorDataUtils.hpp>

#include <cmath>
#include <iostream>
#include <random>

static double getThreshVal_Otsu(std::vector<uint8_t> &src)
{
    int              N = 256;
    std::vector<int> h(N, 0);
    int              i;

    int size = src.size();
    for (i = 0; i < size; i++) h[src[i]]++;

    double mu = 0, scale = 1. / size;
    for (i = 0; i < N; i++)
    {
        mu += i * (double)h[i];
    }

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for (i = 0; i < N; i++)
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i] * scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
            continue;

        mu1   = (mu1 + i * p_i) / q1;
        mu2   = (mu - q1 * mu1) / q2;
        sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        if (sigma > max_sigma)
        {
            max_sigma = sigma;
            max_val   = i;
        }
    }
    return max_val;
}

static double getThreshVal_Triangle(std::vector<uint8_t> &src)
{
    int              N = 256;
    std::vector<int> h(N, 0);
    int              i, j;

    int size = src.size();
    for (i = 0; i < size; i++) h[src[i]]++;

    int  left_bound = 0, right_bound = 0, max_ind = 0, max = 0;
    int  temp;
    bool isflipped = false;

    for (i = 0; i < N; i++)
    {
        if (h[i] > 0)
        {
            left_bound = i;
            break;
        }
    }
    if (left_bound > 0)
        left_bound--;

    for (i = N - 1; i > 0; i--)
    {
        if (h[i] > 0)
        {
            right_bound = i;
            break;
        }
    }
    if (right_bound < N - 1)
        right_bound++;

    for (i = 0; i < N; i++)
    {
        if (h[i] > max)
        {
            max     = h[i];
            max_ind = i;
        }
    }

    if (max_ind - left_bound < right_bound - max_ind)
    {
        isflipped = true;
        i = 0, j = N - 1;
        while (i < j)
        {
            temp = h[i];
            h[i] = h[j];
            h[j] = temp;
            i++;
            j--;
        }
        left_bound = N - 1 - right_bound;
        max_ind    = N - 1 - max_ind;
    }

    double thresh = left_bound;
    double a, b, dist = 0, tempdist;

    a = max;
    b = left_bound - max_ind;
    for (i = left_bound + 1; i <= max_ind; i++)
    {
        tempdist = a * i + b * h[i];
        if (tempdist > dist)
        {
            dist   = tempdist;
            thresh = i;
        }
    }
    thresh--;

    if (isflipped)
        thresh = N - 1 - thresh;

    return thresh;
}

//test for uint8
template<typename T>
static void Threshold(std::vector<T> &src, std::vector<T> &dst, double thresh, double maxval, uint32_t type)
{
    int automatic_thresh = (type & ~NVCV_THRESH_MASK);
    type &= NVCV_THRESH_MASK;

    if (automatic_thresh == (NVCV_THRESH_OTSU | NVCV_THRESH_TRIANGLE))
        return;
    if (automatic_thresh == NVCV_THRESH_OTSU)
        thresh = getThreshVal_Otsu(src);
    else if (automatic_thresh == NVCV_THRESH_TRIANGLE)
        thresh = getThreshVal_Triangle(src);

    int ithresh = floor(thresh);
    thresh      = ithresh;
    int imaxval = round(maxval);
    if (type == NVCV_THRESH_TRUNC)
        imaxval = ithresh;
    imaxval = (uint8_t)((unsigned)imaxval <= UCHAR_MAX ? imaxval : imaxval > 0 ? UCHAR_MAX : 0);

    if (ithresh < 0 || ithresh >= 255)
    {
        if (type == NVCV_THRESH_BINARY || type == NVCV_THRESH_BINARY_INV
            || ((type == NVCV_THRESH_TRUNC || type == NVCV_THRESH_TOZERO_INV) && ithresh < 0)
            || (type == NVCV_THRESH_TOZERO && ithresh >= 255))
        {
            int v = type == NVCV_THRESH_BINARY     ? (ithresh >= 255 ? 0 : imaxval)
                  : type == NVCV_THRESH_BINARY_INV ? (ithresh >= 255 ? imaxval : 0)
                                                   : 0;
            std::fill(dst.begin(), dst.end(), v);
        }
        else
            dst.assign(src.begin(), src.end());
        return;
    }
    thresh = ithresh;
    maxval = imaxval;

    int size = src.size();
    switch (type)
    {
    case NVCV_THRESH_BINARY:
        for (int i = 0; i < size; i++) dst[i] = src[i] > thresh ? maxval : 0;
        break;
    case NVCV_THRESH_BINARY_INV:
        for (int i = 0; i < size; i++) dst[i] = src[i] <= thresh ? maxval : 0;
        break;
    case NVCV_THRESH_TRUNC:
        for (int i = 0; i < size; i++) dst[i] = std::min(src[i], (uint8_t)thresh);
        break;
    case NVCV_THRESH_TOZERO:
        for (int i = 0; i < size; i++) dst[i] = src[i] > thresh ? src[i] : 0;
        break;
    case NVCV_THRESH_TOZERO_INV:
        for (int i = 0; i < size; i++) dst[i] = src[i] <= thresh ? src[i] : 0;
        break;
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpThreshold, nvcv::test::ValueList<int, int, int, uint32_t, double, double>
{
    //batch,    height,     width,                                                type,         thresh,       maxval
    {     1,       480,       360,                                  NVCV_THRESH_BINARY,            100,         255},
    {     5,       100,       100,                              NVCV_THRESH_BINARY_INV,            100,         255},
    {     4,       100,       101,                                   NVCV_THRESH_TRUNC,            100,         255},
    {     3,       360,       480,                                  NVCV_THRESH_TOZERO,            100,         255},
    {     2,       100,       101,                              NVCV_THRESH_TOZERO_INV,            100,         255},
    {     1,       800,       600,                 NVCV_THRESH_OTSU|NVCV_THRESH_BINARY,            100,         255},
    {     3,       600,       1000,        NVCV_THRESH_TRIANGLE|NVCV_THRESH_BINARY_INV,            100,         255},
});

// clang-format on

TEST_P(OpThreshold, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int      batch  = GetParamValue<0>();
    int      height = GetParamValue<1>();
    int      width  = GetParamValue<2>();
    uint32_t type   = GetParamValue<3>();
    double   thresh = GetParamValue<4>();
    double   maxval = GetParamValue<5>();

    nvcv::ImageFormat fmt    = nvcv::FMT_U8;
    nvcv::Tensor      imgIn  = nvcv::util::CreateTensor(batch, width, height, fmt);
    nvcv::Tensor      imgOut = nvcv::util::CreateTensor(batch, width, height, fmt);

    auto inData = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, inData);
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);
    ASSERT_EQ(batch, inAccess->numSamples());

    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, outData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);
    ASSERT_EQ(batch, outAccess->numSamples());

    int64_t outSampleStride = outAccess->sampleStride();

    if (outData->rank() == 3)
    {
        outSampleStride = outAccess->numRows() * outAccess->rowStride();
    }

    int64_t outBufferSize = outSampleStride * outAccess->numSamples();

    // Set output buffer to dummy value
    EXPECT_EQ(cudaSuccess, cudaMemset(outAccess->sampleData(0), 0xFA, outBufferSize));

    //parameters
    nvcv::Tensor threshval({{batch}, "N"}, nvcv::TYPE_F64);
    nvcv::Tensor maxvalval({{batch}, "N"}, nvcv::TYPE_F64);

    auto threshData = threshval.exportData<nvcv::TensorDataStridedCuda>();
    auto maxvalData = maxvalval.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, threshData);
    ASSERT_NE(nullptr, maxvalData);

    std::vector<double> threshVec(batch, thresh);
    std::vector<double> maxvalVec(batch, maxval);

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(threshData->basePtr(), threshVec.data(), threshVec.size() * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(maxvalData->basePtr(), maxvalVec.data(), maxvalVec.size() * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));

    //Generate input
    std::vector<std::vector<uint8_t>> srcVec(batch);
    std::default_random_engine        randEng;
    int                               rowStride = width * fmt.planePixelStrideBytes(0);

    for (int i = 0; i < batch; i++)
    {
        std::uniform_int_distribution<uint8_t> rand(0, 255);
        srcVec[i].resize(height * rowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(i), inAccess->rowStride(), srcVec[i].data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));
    }

    // Call operator
    int               maxBatch = 5;
    cvcuda::Threshold thresholdOp(type, maxBatch);
    EXPECT_NO_THROW(thresholdOp(stream, imgIn, imgOut, threshval, maxvalval));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(height * rowStride);
        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowStride, outAccess->sampleData(i), outAccess->rowStride(),
                                            rowStride, height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(height * rowStride);
        Threshold(srcVec[i], goldVec, thresh, maxval, type);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpThreshold, varshape_correct_shape)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int      batch  = GetParamValue<0>();
    int      height = GetParamValue<1>();
    int      width  = GetParamValue<2>();
    uint32_t type   = GetParamValue<3>();
    double   thresh = GetParamValue<4>();
    double   maxval = GetParamValue<5>();

    nvcv::ImageFormat                  fmt = nvcv::FMT_U8;
    // Create input and output
    std::default_random_engine         randEng;
    std::uniform_int_distribution<int> rndWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> rndHeight(height * 0.8, height * 1.1);

    std::vector<nvcv::Image> imgSrc, imgDst;
    for (int i = 0; i < batch; ++i)
    {
        int rw = rndWidth(randEng);
        int rh = rndHeight(randEng);
        imgSrc.emplace_back(nvcv::Size2D{rw, rh}, fmt);
        imgDst.emplace_back(nvcv::Size2D{rw, rh}, fmt);
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    //parameters
    nvcv::Tensor threshval({{batch}, "N"}, nvcv::TYPE_F64);
    nvcv::Tensor maxvalval({{batch}, "N"}, nvcv::TYPE_F64);

    auto threshData = threshval.exportData<nvcv::TensorDataStridedCuda>();
    auto maxvalData = maxvalval.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, threshData);
    ASSERT_NE(nullptr, maxvalData);

    std::vector<double> threshVec(batch, thresh);
    std::vector<double> maxvalVec(batch, maxval);

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(threshData->basePtr(), threshVec.data(), threshVec.size() * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(maxvalData->basePtr(), maxvalVec.data(), maxvalVec.size() * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));

    //Generate input
    std::vector<std::vector<uint8_t>> srcVec(batch);

    for (int i = 0; i < batch; i++)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(),
                                            srcRowStride, srcRowStride, srcHeight, cudaMemcpyHostToDevice));
    }

    // Call operator
    int               maxBatch = 5;
    cvcuda::Threshold thresholdOp(type, maxBatch);
    EXPECT_NO_THROW(thresholdOp(stream, batchSrc, batchDst, threshval, maxvalval));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dstHeight * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstRowStride);
        Threshold(srcVec[i], goldVec, thresh, maxval, type);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

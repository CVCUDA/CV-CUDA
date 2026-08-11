/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpThreshold.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(static_cast<double>(size) * scale);
}

static double getThreshVal_Otsu(std::vector<uint8_t> &src)
{
    int              N = 256;
    std::vector<int> h(N, 0);
    int              i;

    auto size = static_cast<int>(src.size());
    for (i = 0; i < size; i++) h[src[i]]++;

    double mu    = 0;
    double scale = 1. / size;
    for (i = 0; i < N; i++)
    {
        mu += i * (double)h[i];
    }

    mu *= scale;
    double mu1       = 0;
    double q1        = 0;
    double max_sigma = 0;
    double max_val   = 0;

    for (i = 0; i < N; i++)
    {
        double p_i;
        double q2;
        double mu2;
        double sigma;

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
    int              i;
    int              j;

    auto size = static_cast<int>(src.size());
    for (i = 0; i < size; i++) h[src[i]]++;

    int  left_bound  = 0;
    int  right_bound = 0;
    int  max_ind     = 0;
    int  max         = 0;
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
        i         = 0;
        j         = N - 1;
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
    double a;
    double b;
    double dist = 0;
    double tempdist;

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

namespace {
//test for uint8
template<typename T>
bool ResolveAutomaticThreshold(std::vector<T> &src, double &thresh, int automatic_thresh)
{
    if (automatic_thresh == (NVCV_THRESH_OTSU | NVCV_THRESH_TRIANGLE))
    {
        return false;
    }

    if (automatic_thresh == NVCV_THRESH_OTSU)
    {
        thresh = getThreshVal_Otsu(src);
    }
    else if (automatic_thresh == NVCV_THRESH_TRIANGLE)
    {
        thresh = getThreshVal_Triangle(src);
    }

    return true;
}

uint8_t ClampMaxThreshold(int maxval)
{
    return static_cast<uint8_t>(std::clamp(maxval, 0, UCHAR_MAX));
}

bool ShouldFillOutOfRangeThreshold(uint32_t type, int ithresh)
{
    return type == NVCV_THRESH_BINARY || type == NVCV_THRESH_BINARY_INV
        || ((type == NVCV_THRESH_TRUNC || type == NVCV_THRESH_TOZERO_INV) && ithresh < 0)
        || (type == NVCV_THRESH_TOZERO && ithresh >= 255);
}

uint8_t OutOfRangeThresholdValue(uint32_t type, int ithresh, uint8_t maxval)
{
    if (type == NVCV_THRESH_BINARY)
    {
        return ithresh >= 255 ? 0 : maxval;
    }

    if (type == NVCV_THRESH_BINARY_INV)
    {
        return ithresh >= 255 ? maxval : 0;
    }

    return 0;
}

template<typename T>
void ApplyOutOfRangeThreshold(const std::vector<T> &src, std::vector<T> &dst, uint32_t type, int ithresh,
                              uint8_t maxval)
{
    if (ShouldFillOutOfRangeThreshold(type, ithresh))
    {
        std::ranges::fill(dst, static_cast<T>(OutOfRangeThresholdValue(type, ithresh, maxval)));
    }
    else
    {
        dst.assign(src.begin(), src.end());
    }
}

template<typename T>
void ApplyThreshold(std::vector<T> &src, std::vector<T> &dst, uint32_t type, int ithresh, uint8_t maxval)
{
    auto size = static_cast<int>(src.size());
    switch (type)
    {
    case NVCV_THRESH_BINARY:
        for (int i = 0; i < size; i++) dst[i] = src[i] > ithresh ? maxval : 0;
        break;
    case NVCV_THRESH_BINARY_INV:
        for (int i = 0; i < size; i++) dst[i] = src[i] <= ithresh ? maxval : 0;
        break;
    case NVCV_THRESH_TRUNC:
        for (int i = 0; i < size; i++) dst[i] = std::min(src[i], static_cast<uint8_t>(ithresh));
        break;
    case NVCV_THRESH_TOZERO:
        for (int i = 0; i < size; i++) dst[i] = src[i] > ithresh ? src[i] : 0;
        break;
    case NVCV_THRESH_TOZERO_INV:
        for (int i = 0; i < size; i++) dst[i] = src[i] <= ithresh ? src[i] : 0;
        break;
    default:
        break;
    }
}

template<typename T>
void Threshold(std::vector<T> &src, std::vector<T> &dst, double thresh, double maxval, uint32_t type)
{
    int automatic_thresh = (type & ~NVCV_THRESH_MASK);
    type &= NVCV_THRESH_MASK;

    if (!ResolveAutomaticThreshold(src, thresh, automatic_thresh))
    {
        return;
    }

    auto ithresh = static_cast<int>(floor(thresh));
    auto imaxval = static_cast<int>(round(maxval));
    if (type == NVCV_THRESH_TRUNC)
    {
        imaxval = ithresh;
    }
    uint8_t clampedMaxVal = ClampMaxThreshold(imaxval);

    if (ithresh < 0 || ithresh >= 255)
    {
        ApplyOutOfRangeThreshold(src, dst, type, ithresh, clampedMaxVal);
        return;
    }

    ApplyThreshold(src, dst, type, ithresh, clampedMaxVal);
}

// test for double
template<>
void Threshold(std::vector<double> &src, std::vector<double> &dst, double thresh, double maxval, uint32_t type)
{
    int automatic_thresh = (type & ~NVCV_THRESH_MASK);
    type &= NVCV_THRESH_MASK;

    if (automatic_thresh == (NVCV_THRESH_OTSU | NVCV_THRESH_TRIANGLE) || automatic_thresh == NVCV_THRESH_OTSU
        || automatic_thresh == NVCV_THRESH_TRIANGLE)
        return;
    dst.assign(src.begin(), src.end());

    auto size = static_cast<int>(src.size());
    switch (type)
    {
    case NVCV_THRESH_BINARY:
        for (int i = 0; i < size; i++) dst[i] = src[i] > thresh ? maxval : 0;
        break;
    case NVCV_THRESH_BINARY_INV:
        for (int i = 0; i < size; i++) dst[i] = src[i] <= thresh ? maxval : 0;
        break;
    case NVCV_THRESH_TRUNC:
        for (int i = 0; i < size; i++) dst[i] = std::min(src[i], thresh);
        break;
    case NVCV_THRESH_TOZERO:
        for (int i = 0; i < size; i++) dst[i] = src[i] > thresh ? src[i] : 0;
        break;
    case NVCV_THRESH_TOZERO_INV:
        for (int i = 0; i < size; i++) dst[i] = src[i] <= thresh ? src[i] : 0;
        break;
    default:
        break;
    }
}

void ThresholdWrapper(std::vector<uint8_t> &src, std::vector<uint8_t> &dst, double thresh, double maxval, uint32_t type,
                      NVCVDataType nvcvDataType)
{
    if (nvcvDataType == NVCV_DATA_TYPE_F64)
    {
        std::vector<double> src_tmp(src.size() / sizeof(double));
        std::vector<double> dst_tmp(dst.size() / sizeof(double));
        size_t              copySize = src.size();
        memcpy(static_cast<void *>(src_tmp.data()), static_cast<void *>(src.data()), copySize);
        memcpy(static_cast<void *>(dst_tmp.data()), static_cast<void *>(dst.data()), copySize);
        Threshold(src_tmp, dst_tmp, thresh, maxval, type);
        memcpy(static_cast<void *>(dst.data()), static_cast<void *>(dst_tmp.data()), copySize);
    }
    else
    {
        Threshold(src, dst, thresh, maxval, type);
    }
}

template<typename T>
void myGenerate( // NOSONAR: std::span is C++20.
    T *src, std::size_t size, std::default_random_engine &randEng)
{
    std::uniform_int_distribution rand(0u, 255u);
    for (std::size_t idx = 0; idx < size; ++idx)
    {
        src[idx] = static_cast<T>(rand(randEng));
    }
}

template<>
void myGenerate( // NOSONAR: std::span is C++20.
    double *src, std::size_t size, std::default_random_engine &randEng)
{
    std::uniform_real_distribution rand(0., 1.);
    for (std::size_t idx = 0; idx < size; ++idx)
    {
        src[idx] = rand(randEng);
    }
}

nvcv::Tensor MakeThresholdParam(int numImages, double value)
{
    return nvcv::test::planar::MakePerImageTensor<double>(numImages, nvcv::TYPE_F64, value);
}

template<typename T>
T ThresholdGoldValue(T input, double thresh, double maxval, uint32_t type)
{
    T typedThresh;
    T typedMaxval;
    if constexpr (std::is_floating_point_v<T>)
    {
        typedThresh = static_cast<T>(thresh);
        typedMaxval = static_cast<T>(maxval);
    }
    else
    {
        typedThresh = static_cast<T>(static_cast<int>(std::floor(thresh)));
        typedMaxval = static_cast<T>(static_cast<int>(std::round(maxval)));
    }

    switch (type & NVCV_THRESH_MASK)
    {
    case NVCV_THRESH_BINARY:
        return input > typedThresh ? typedMaxval : T{};
    case NVCV_THRESH_BINARY_INV:
        return input > typedThresh ? T{} : typedMaxval;
    case NVCV_THRESH_TRUNC:
        return input > typedThresh ? typedThresh : input;
    case NVCV_THRESH_TOZERO:
        return input > typedThresh ? input : T{};
    default:
        return input > typedThresh ? T{} : input;
    }
}

template<typename T>
T ThresholdPackInput(int sample, int index)
{
    int value = (sample * 11 + index * 7) % 19;
    if constexpr (std::is_signed_v<T> || std::is_floating_point_v<T>)
        value -= 7;
    return static_cast<T>(value);
}

template<typename T>
void ThresholdGold(std::vector<T> &src, std::vector<T> &dst, double thresh, double maxval, uint32_t type)
{
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        Threshold(src, dst, thresh, maxval, type);
    }
    else
    {
        std::ranges::transform(src, dst.begin(),
                               [=](T input) { return ThresholdGoldValue(input, thresh, maxval, type); });
    }
}

template<typename T>
void RunTensorUnalignedPackCase(nvcv::DataType dtype, uint32_t type, int width)
{
    constexpr int    numImages    = 2;
    constexpr int    height       = 2;
    constexpr size_t guardBytes   = 32;
    constexpr double thresh       = 3.5;
    constexpr double maxval       = 9.25;
    const size_t     offset       = alignof(T);
    const int        rowBytes     = width * sizeof(T);
    const int        rowStride    = rowBytes + 16;
    const int        sampleStride = rowStride * height + 16;
    const size_t     bytes        = offset + sampleStride * numImages + guardBytes;
    const size_t     packBytes    = sizeof(T) == sizeof(double) ? 2 * sizeof(T) : 4 * sizeof(T);

    NVCVByte *srcAllocation{};
    NVCVByte *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), bytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), bytes));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    {
        auto wrapTensor = [&](NVCVByte *allocation)
        {
            nvcv::TensorDataStridedCuda::Buffer buffer{};
            buffer.basePtr    = allocation + offset;
            buffer.strides[0] = sampleStride;
            buffer.strides[1] = rowStride;
            buffer.strides[2] = sizeof(T);
            buffer.strides[3] = sizeof(T);
            return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{numImages, height, width, 1}, "NHWC"},
                dtype, buffer
            });
        };

        nvcv::Tensor src = wrapTensor(srcAllocation);
        nvcv::Tensor dst = wrapTensor(dstAllocation);
        EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(srcAllocation + offset) % alignof(T));
        EXPECT_NE(0u, reinterpret_cast<uintptr_t>(srcAllocation + offset) % packBytes);
        EXPECT_NE(0u, reinterpret_cast<uintptr_t>(dstAllocation + offset) % packBytes);

        std::vector<uint8_t> expected(bytes, 0xD7);
        ASSERT_EQ(cudaSuccess, cudaMemset(srcAllocation, 0xA5, bytes));
        ASSERT_EQ(cudaSuccess, cudaMemset(dstAllocation, 0xD7, bytes));

        for (int b = 0; b < numImages; ++b)
        {
            std::vector<T> srcVisible(width * height);
            size_t         index = 0;
            std::ranges::generate(srcVisible,
                                  [&]
                                  {
                                      const T value = ThresholdPackInput<T>(b, static_cast<int>(index));
                                      ++index;
                                      return value;
                                  });

            std::vector<T> dstVisible(srcVisible.size());
            ThresholdGold(srcVisible, dstVisible, thresh, maxval, type);
            for (int y = 0; y < height; ++y)
            {
                std::memcpy(expected.data() + offset + b * sampleStride + y * rowStride, dstVisible.data() + y * width,
                            rowBytes);
            }

            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcAllocation + offset + b * sampleStride, rowStride, srcVisible.data(),
                                                rowBytes, rowBytes, height, cudaMemcpyHostToDevice));
        }

        nvcv::Tensor      threshval = MakeThresholdParam(numImages, thresh);
        nvcv::Tensor      maxvalval = MakeThresholdParam(numImages, maxval);
        cvcuda::Threshold thresholdOp(type, numImages);
        EXPECT_NO_THROW(thresholdOp(stream, src, dst, threshval, maxvalval));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        std::vector<uint8_t> got(bytes);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(got.data(), dstAllocation, bytes, cudaMemcpyDeviceToHost));
        EXPECT_EQ(expected, got);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    EXPECT_EQ(cudaSuccess, cudaFree(srcAllocation));
    EXPECT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

template<typename T>
void RunVarShapeUnalignedPackCase(nvcv::ImageFormat fmt, uint32_t type)
{
    constexpr int    numImages  = 2;
    constexpr int    width      = 9;
    constexpr int    height     = 2;
    constexpr size_t guardBytes = 32;
    constexpr double thresh     = 3.5;
    constexpr double maxval     = 9.25;
    const size_t     offset     = alignof(T);
    const int        rowBytes   = width * sizeof(T);
    const int        rowStride  = rowBytes + 16;
    const size_t     bytes      = offset + static_cast<size_t>(rowStride) * height + guardBytes;
    const size_t     packBytes  = sizeof(T) == sizeof(double) ? 2 * sizeof(T) : 4 * sizeof(T);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::array<NVCVByte *, numImages>           srcAllocations{};
    std::array<NVCVByte *, numImages>           dstAllocations{};
    std::array<std::vector<uint8_t>, numImages> expected;
    {
        std::vector<nvcv::Image> srcImages;
        std::vector<nvcv::Image> dstImages;
        for (int b = 0; b < numImages; ++b)
        {
            ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocations[b]), bytes));
            ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocations[b]), bytes));

            nvcv::ImageDataStridedCuda::Buffer srcBuffer{};
            srcBuffer.numPlanes           = 1;
            srcBuffer.planes[0].width     = width;
            srcBuffer.planes[0].height    = height;
            srcBuffer.planes[0].rowStride = rowStride;
            srcBuffer.planes[0].basePtr   = srcAllocations[b] + offset;
            auto dstBuffer                = srcBuffer;
            dstBuffer.planes[0].basePtr   = dstAllocations[b] + offset;
            EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(srcBuffer.planes[0].basePtr) % alignof(T));
            EXPECT_NE(0u, reinterpret_cast<uintptr_t>(srcBuffer.planes[0].basePtr) % packBytes);
            EXPECT_NE(0u, reinterpret_cast<uintptr_t>(dstBuffer.planes[0].basePtr) % packBytes);

            srcImages.emplace_back(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{fmt, srcBuffer}));
            dstImages.emplace_back(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{fmt, dstBuffer}));

            std::vector<T> srcVisible(width * height);
            size_t         index = 0;
            std::ranges::generate(srcVisible,
                                  [&]
                                  {
                                      const T value = ThresholdPackInput<T>(b, static_cast<int>(index));
                                      ++index;
                                      return value;
                                  });
            std::vector<T> dstVisible(srcVisible.size());
            ThresholdGold(srcVisible, dstVisible, thresh, maxval, type);

            expected[b].assign(bytes, 0xD7);
            for (int y = 0; y < height; ++y)
            {
                std::memcpy(expected[b].data() + offset + y * rowStride, dstVisible.data() + y * width, rowBytes);
            }
            ASSERT_EQ(cudaSuccess, cudaMemset(srcAllocations[b], 0xA5, bytes));
            ASSERT_EQ(cudaSuccess, cudaMemset(dstAllocations[b], 0xD7, bytes));
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcBuffer.planes[0].basePtr, rowStride, srcVisible.data(), rowBytes,
                                                rowBytes, height, cudaMemcpyHostToDevice));
        }

        nvcv::ImageBatchVarShape srcBatch(numImages);
        nvcv::ImageBatchVarShape dstBatch(numImages);
        srcBatch.pushBack(srcImages.begin(), srcImages.end());
        dstBatch.pushBack(dstImages.begin(), dstImages.end());
        nvcv::Tensor      threshval = MakeThresholdParam(numImages, thresh);
        nvcv::Tensor      maxvalval = MakeThresholdParam(numImages, maxval);
        cvcuda::Threshold thresholdOp(type, numImages);
        EXPECT_NO_THROW(thresholdOp(stream, srcBatch, dstBatch, threshval, maxvalval));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        for (int b = 0; b < numImages; ++b)
        {
            std::vector<uint8_t> got(bytes);
            ASSERT_EQ(cudaSuccess, cudaMemcpy(got.data(), dstAllocations[b], bytes, cudaMemcpyDeviceToHost));
            EXPECT_EQ(expected[b], got);
        }
    }

    for (int b = 0; b < numImages; ++b)
    {
        EXPECT_EQ(cudaSuccess, cudaFree(srcAllocations[b]));
        EXPECT_EQ(cudaSuccess, cudaFree(dstAllocations[b]));
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

struct U8BinaryPackCase
{
    const char *name;
    int         width;
    int         height;
    int         rowStride;
    size_t      srcOffset;
    size_t      dstOffset;
    double      thresh;
    double      maxval;
};

struct U8BinaryPackStorage
{
    NVCVByte            *srcAllocation{};
    NVCVByte            *dstAllocation{};
    size_t               srcBytes{};
    size_t               dstBytes{};
    std::vector<uint8_t> expected;
};

void RunU8PackCases(nvcv::ImageFormat fmt, uint32_t type, const std::vector<U8BinaryPackCase> &cases)
{
    constexpr size_t guardBytes = 32;
    const int        pixelBytes = fmt.planePixelStrideBytes(0);
    const auto       numImages  = static_cast<int>(cases.size());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<U8BinaryPackStorage> storage(numImages);
    {
        std::vector<nvcv::Image> srcImages;
        std::vector<nvcv::Image> dstImages;
        srcImages.reserve(numImages);
        dstImages.reserve(numImages);

        for (int b = 0; b < numImages; ++b)
        {
            const U8BinaryPackCase &testCase = cases[b];
            SCOPED_TRACE(testCase.name);

            const size_t rowBytes = static_cast<size_t>(testCase.width) * pixelBytes;
            ASSERT_GE(static_cast<size_t>(testCase.rowStride), rowBytes);

            U8BinaryPackStorage &data = storage[b];
            data.srcBytes = testCase.srcOffset + static_cast<size_t>(testCase.rowStride) * testCase.height + guardBytes;
            data.dstBytes = testCase.dstOffset + static_cast<size_t>(testCase.rowStride) * testCase.height + guardBytes;
            ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&data.srcAllocation), data.srcBytes));
            ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&data.dstAllocation), data.dstBytes));

            nvcv::ImageDataStridedCuda::Buffer srcBuffer{};
            srcBuffer.numPlanes           = 1;
            srcBuffer.planes[0].width     = testCase.width;
            srcBuffer.planes[0].height    = testCase.height;
            srcBuffer.planes[0].rowStride = testCase.rowStride;
            srcBuffer.planes[0].basePtr   = data.srcAllocation + testCase.srcOffset;

            auto dstBuffer              = srcBuffer;
            dstBuffer.planes[0].basePtr = data.dstAllocation + testCase.dstOffset;

            if (testCase.srcOffset == 0)
            {
                EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(srcBuffer.planes[0].basePtr) & 15u);
                EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(dstBuffer.planes[0].basePtr) & 15u);
            }
            else
            {
                EXPECT_NE(0u, reinterpret_cast<uintptr_t>(srcBuffer.planes[0].basePtr) & 15u);
                EXPECT_NE(0u, reinterpret_cast<uintptr_t>(dstBuffer.planes[0].basePtr) & 15u);
            }

            srcImages.emplace_back(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{fmt, srcBuffer}));
            dstImages.emplace_back(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{fmt, dstBuffer}));

            std::vector<uint8_t> srcVisible(rowBytes * testCase.height);
            data.expected.assign(data.dstBytes, 0xD7);

            for (int y = 0; y < testCase.height; ++y)
            {
                for (size_t x = 0; x < rowBytes; ++x)
                {
                    const auto input = static_cast<uint8_t>(b * 43 + y * 29 + x * 17 + 11);
                    srcVisible[static_cast<size_t>(y) * rowBytes + x] = input;
                }
            }

            std::vector<uint8_t> dstVisible(srcVisible.size());
            Threshold(srcVisible, dstVisible, testCase.thresh, testCase.maxval, type);
            for (int y = 0; y < testCase.height; ++y)
            {
                std::copy_n(dstVisible.data() + static_cast<size_t>(y) * rowBytes, rowBytes,
                            data.expected.data() + testCase.dstOffset + static_cast<size_t>(y) * testCase.rowStride);
            }

            ASSERT_EQ(cudaSuccess, cudaMemset(data.srcAllocation, 0xA5, data.srcBytes));
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcBuffer.planes[0].basePtr, testCase.rowStride, srcVisible.data(),
                                                rowBytes, rowBytes, testCase.height, cudaMemcpyHostToDevice));
            ASSERT_EQ(cudaSuccess, cudaMemset(data.dstAllocation, 0xD7, data.dstBytes));
        }

        nvcv::ImageBatchVarShape srcBatch(numImages);
        nvcv::ImageBatchVarShape dstBatch(numImages);
        srcBatch.pushBack(srcImages.begin(), srcImages.end());
        dstBatch.pushBack(dstImages.begin(), dstImages.end());

        nvcv::Tensor threshval({{numImages}, "N"}, nvcv::TYPE_F64);
        nvcv::Tensor maxvalval({{numImages}, "N"}, nvcv::TYPE_F64);
        auto         threshData = threshval.exportData<nvcv::TensorDataStridedCuda>();
        auto         maxvalData = maxvalval.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nullptr, threshData);
        ASSERT_NE(nullptr, maxvalData);

        std::vector<double> threshVec;
        std::vector<double> maxvalVec;
        threshVec.reserve(numImages);
        maxvalVec.reserve(numImages);
        for (const U8BinaryPackCase &testCase : cases)
        {
            threshVec.push_back(testCase.thresh);
            maxvalVec.push_back(testCase.maxval);
        }

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(threshData->basePtr(), threshVec.data(), numImages * sizeof(double),
                                               cudaMemcpyHostToDevice, stream));
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(maxvalData->basePtr(), maxvalVec.data(), numImages * sizeof(double),
                                               cudaMemcpyHostToDevice, stream));

        cvcuda::Threshold thresholdOp(type, numImages);
        EXPECT_NO_THROW(thresholdOp(stream, srcBatch, dstBatch, threshval, maxvalval));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        for (int b = 0; b < numImages; ++b)
        {
            SCOPED_TRACE(cases[b].name);
            std::vector<uint8_t> got(storage[b].dstBytes);
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy(got.data(), storage[b].dstAllocation, storage[b].dstBytes, cudaMemcpyDeviceToHost));
            EXPECT_EQ(storage[b].expected, got);
        }
    }

    for (U8BinaryPackStorage &data : storage)
    {
        EXPECT_EQ(cudaSuccess, cudaFree(data.srcAllocation));
        EXPECT_EQ(cudaSuccess, cudaFree(data.dstAllocation));
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
} // namespace

TEST(OpThreshold, varshape_u8_binary_aligned_pack_tail_broadcast_exact)
{
    RunU8PackCases(nvcv::FMT_U8, NVCV_THRESH_BINARY,
                   {
                       {     "aligned_full_pack", 32, 2, 48, 0, 0, 100, 211},
                       {   "aligned_scalar_tail", 19, 2, 32, 0, 0, 100, 211},
                       { "aligned_broadcast_max", 32, 2, 48, 0, 0,  -1, 173},
                       {"aligned_broadcast_zero", 19, 2, 32, 0, 0, 256, 173},
    });

    RunU8PackCases(nvcv::FMT_RGB8, NVCV_THRESH_BINARY,
                   {
                       {"rgb_aligned_pack_tail", 7, 2, 32, 0, 0, 100, 211},
    });
}

TEST(OpThreshold, varshape_u8_binary_unaligned_pack_tail_broadcast_exact)
{
    RunU8PackCases(nvcv::FMT_U8, NVCV_THRESH_BINARY,
                   {
                       {          "unaligned_full_pack", 32, 2, 48, 1, 1, 100, 211},
                       {        "unaligned_scalar_tail", 19, 2, 32, 1, 1, 100, 211},
                       {      "unaligned_broadcast_max", 32, 2, 48, 1, 1,  -1, 173},
                       {"unaligned_broadcast_zero_tail", 19, 2, 32, 1, 1, 256, 173},
    });
}

TEST(OpThreshold, varshape_rgb8_binary_unaligned_pack_tail_exact)
{
    RunU8PackCases(nvcv::FMT_RGB8, NVCV_THRESH_BINARY,
                   {
                       {"rgb_unaligned_pack_tail", 7, 2, 32, 1, 1, 100, 211},
    });
}

TEST(OpThreshold, varshape_u8_binary_inv_unaligned_pack_tail_broadcast_exact)
{
    RunU8PackCases(nvcv::FMT_U8, NVCV_THRESH_BINARY_INV,
                   {
                       {"binary_inv_in_range", 19, 2, 32, 1, 1, 100, 211},
                       {   "binary_inv_below", 19, 2, 32, 1, 1,  -1, 173},
                       {   "binary_inv_above", 19, 2, 32, 1, 1, 256, 173},
    });
}

TEST(OpThreshold, varshape_u8_trunc_unaligned_pack_tail_broadcast_exact)
{
    RunU8PackCases(nvcv::FMT_U8, NVCV_THRESH_TRUNC,
                   {
                       {"trunc_in_range", 19, 2, 32, 1, 1, 100, 211},
                       {   "trunc_below", 19, 2, 32, 1, 1,  -1, 173},
                       {   "trunc_above", 19, 2, 32, 1, 1, 256, 173},
    });
}

TEST(OpThreshold, varshape_u8_tozero_unaligned_pack_tail_broadcast_exact)
{
    RunU8PackCases(nvcv::FMT_U8, NVCV_THRESH_TOZERO,
                   {
                       {"tozero_in_range", 19, 2, 32, 1, 1, 100, 211},
                       {   "tozero_below", 19, 2, 32, 1, 1,  -1, 173},
                       {   "tozero_above", 19, 2, 32, 1, 1, 256, 173},
    });
}

TEST(OpThreshold, varshape_u8_tozero_inv_unaligned_pack_tail_broadcast_exact)
{
    RunU8PackCases(nvcv::FMT_U8, NVCV_THRESH_TOZERO_INV,
                   {
                       {"tozero_inv_in_range", 19, 2, 32, 1, 1, 100, 211},
                       {   "tozero_inv_below", 19, 2, 32, 1, 1,  -1, 173},
                       {   "tozero_inv_above", 19, 2, 32, 1, 1, 256, 173},
    });
}

namespace {
enum class UnalignedContainer
{
    Tensor,
    VarShape,
};

enum class UnalignedDataType
{
    U16,
    S16,
    F32,
    F64,
};

using UnalignedNonU8Param = std::tuple<UnalignedContainer, UnalignedDataType, uint32_t>;

const char *ThresholdModeName(uint32_t type)
{
    switch (type)
    {
    case NVCV_THRESH_BINARY:
        return "Binary";
    case NVCV_THRESH_BINARY_INV:
        return "BinaryInv";
    case NVCV_THRESH_TRUNC:
        return "Trunc";
    case NVCV_THRESH_TOZERO:
        return "ToZero";
    default:
        return "ToZeroInv";
    }
}

std::string UnalignedNonU8Name(const testing::TestParamInfo<UnalignedNonU8Param> &info)
{
    const auto [container, dtype, type] = info.param;
    const char *containerName           = container == UnalignedContainer::Tensor ? "Tensor" : "VarShape";
    const char *dtypeName;
    switch (dtype)
    {
    case UnalignedDataType::U16:
        dtypeName = "U16";
        break;
    case UnalignedDataType::S16:
        dtypeName = "S16";
        break;
    case UnalignedDataType::F32:
        dtypeName = "F32";
        break;
    default:
        dtypeName = "F64";
        break;
    }
    return std::string(containerName) + dtypeName + ThresholdModeName(type);
}

class OpThresholdUnalignedNonU8 : public testing::TestWithParam<UnalignedNonU8Param>
{
};

TEST_P(OpThresholdUnalignedNonU8, exact_gold)
{
    const auto [container, dtype, type] = GetParam();
    switch (dtype)
    {
    case UnalignedDataType::U16:
        if (container == UnalignedContainer::Tensor)
            RunTensorUnalignedPackCase<uint16_t>(nvcv::TYPE_U16, type, 8);
        else
            RunVarShapeUnalignedPackCase<uint16_t>(nvcv::FMT_U16, type);
        break;
    case UnalignedDataType::S16:
        if (container == UnalignedContainer::Tensor)
            RunTensorUnalignedPackCase<int16_t>(nvcv::TYPE_S16, type, 8);
        else
            RunVarShapeUnalignedPackCase<int16_t>(nvcv::FMT_S16, type);
        break;
    case UnalignedDataType::F32:
        if (container == UnalignedContainer::Tensor)
            RunTensorUnalignedPackCase<float>(nvcv::TYPE_F32, type, 8);
        else
            RunVarShapeUnalignedPackCase<float>(nvcv::FMT_F32, type);
        break;
    default:
        if (container == UnalignedContainer::Tensor)
            RunTensorUnalignedPackCase<double>(nvcv::TYPE_F64, type, 8);
        else
            RunVarShapeUnalignedPackCase<double>(nvcv::FMT_F64, type);
        break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    All, OpThresholdUnalignedNonU8,
    testing::Values(UnalignedNonU8Param{UnalignedContainer::Tensor, UnalignedDataType::U16, NVCV_THRESH_BINARY},
                    UnalignedNonU8Param{UnalignedContainer::VarShape, UnalignedDataType::U16, NVCV_THRESH_BINARY_INV},
                    UnalignedNonU8Param{UnalignedContainer::Tensor, UnalignedDataType::S16, NVCV_THRESH_TRUNC},
                    UnalignedNonU8Param{UnalignedContainer::VarShape, UnalignedDataType::S16, NVCV_THRESH_TOZERO},
                    UnalignedNonU8Param{UnalignedContainer::Tensor, UnalignedDataType::F32, NVCV_THRESH_TOZERO_INV},
                    UnalignedNonU8Param{UnalignedContainer::VarShape, UnalignedDataType::F32, NVCV_THRESH_BINARY},
                    UnalignedNonU8Param{UnalignedContainer::Tensor, UnalignedDataType::F64, NVCV_THRESH_BINARY_INV},
                    UnalignedNonU8Param{UnalignedContainer::VarShape, UnalignedDataType::F64, NVCV_THRESH_TRUNC}),
    UnalignedNonU8Name);

enum class AutomaticMode
{
    Otsu,
    Triangle,
};

using UnalignedAutomaticParam = std::tuple<UnalignedContainer, AutomaticMode>;

std::string UnalignedAutomaticName(const testing::TestParamInfo<UnalignedAutomaticParam> &info)
{
    const auto [container, mode] = info.param;
    return std::string(container == UnalignedContainer::Tensor ? "Tensor" : "VarShape")
         + (mode == AutomaticMode::Otsu ? "Otsu" : "Triangle");
}

class OpThresholdUnalignedAutomatic : public testing::TestWithParam<UnalignedAutomaticParam>
{
};

TEST_P(OpThresholdUnalignedAutomatic, exact_gold)
{
    const auto [container, mode] = GetParam();
    const uint32_t type          = mode == AutomaticMode::Otsu ? NVCV_THRESH_OTSU | NVCV_THRESH_BINARY
                                                               : NVCV_THRESH_TRIANGLE | NVCV_THRESH_BINARY_INV;
    if (container == UnalignedContainer::Tensor)
    {
        RunTensorUnalignedPackCase<uint8_t>(nvcv::TYPE_U8, type, 32);
    }
    else if (mode == AutomaticMode::Otsu)
    {
        RunU8PackCases(nvcv::FMT_U8, type,
                       {
                           {"otsu_unaligned_full_pack", 32, 2, 48, 1, 1, 100, 211},
                           {     "otsu_unaligned_tail", 19, 2, 32, 1, 1, 100, 211},
        });
    }
    else
    {
        RunU8PackCases(nvcv::FMT_U8, type,
                       {
                           {"triangle_unaligned_full_pack", 32, 2, 48, 1, 1, 100, 211},
                           {     "triangle_unaligned_tail", 19, 2, 32, 1, 1, 100, 211},
        });
    }
}

INSTANTIATE_TEST_SUITE_P(All, OpThresholdUnalignedAutomatic,
                         testing::Combine(testing::Values(UnalignedContainer::Tensor, UnalignedContainer::VarShape),
                                          testing::Values(AutomaticMode::Otsu, AutomaticMode::Triangle)),
                         UnalignedAutomaticName);
} // namespace

TEST(OpThreshold, tensor_u8_binary_unaligned_pack_exact)
{
    RunTensorUnalignedPackCase<uint8_t>(nvcv::TYPE_U8, NVCV_THRESH_BINARY, 32);
}

// clang-format off
NVCV_TEST_SUITE_P(OpThreshold, nvcv::test::ValueList<int, int, int, uint32_t, double, double, nvcv::ImageFormat>
{
    //batch,    height,     width,                                                type,         thresh,      maxval,      format,
    {     1,       480,       360,                                  NVCV_THRESH_BINARY,            100,         255, nvcv::FMT_U8},
    {     1,       102,       102,                                  NVCV_THRESH_BINARY,            100,         255, nvcv::FMT_U8},
    {     1,         3,         3,                                  NVCV_THRESH_BINARY,            100,         255, nvcv::FMT_U8},
    {     1,         3,         3,                                  NVCV_THRESH_BINARY,             -1,         255, nvcv::FMT_U8},
    {     1,       480,       360,                                  NVCV_THRESH_BINARY,             -1,         255, nvcv::FMT_U8},
    {     1,       480,       360,                                  NVCV_THRESH_BINARY,            256,         255, nvcv::FMT_U8},
    {     1,         9,         9,                                  NVCV_THRESH_BINARY,            0.5,         255, nvcv::FMT_F64},
    {     1,       480,       360,                                  NVCV_THRESH_BINARY,            0.5,         255, nvcv::FMT_F64},
    {     5,       100,       100,                              NVCV_THRESH_BINARY_INV,            100,         255, nvcv::FMT_U8},
    {     5,       100,       100,                              NVCV_THRESH_BINARY_INV,             -1,         255, nvcv::FMT_U8},
    {     5,       100,       100,                              NVCV_THRESH_BINARY_INV,            256,         255, nvcv::FMT_U8},
    {     5,       100,       100,                              NVCV_THRESH_BINARY_INV,            0.5,         255, nvcv::FMT_F64},
    {     4,       100,       101,                                   NVCV_THRESH_TRUNC,            100,         255, nvcv::FMT_U8},
    {     4,       100,       101,                                   NVCV_THRESH_TRUNC,             -1,         255, nvcv::FMT_U8},
    {     4,       100,       101,                                   NVCV_THRESH_TRUNC,            256,         255, nvcv::FMT_U8},
    {     4,       100,       101,                                   NVCV_THRESH_TRUNC,            0.5,         255, nvcv::FMT_F64},
    {     3,       360,       480,                                  NVCV_THRESH_TOZERO,            100,         255, nvcv::FMT_U8},
    {     3,       360,       480,                                  NVCV_THRESH_TOZERO,             -1,         255, nvcv::FMT_U8},
    {     3,       360,       480,                                  NVCV_THRESH_TOZERO,            256,         255, nvcv::FMT_U8},
    {     3,       360,       480,                                  NVCV_THRESH_TOZERO,            0.5,         255, nvcv::FMT_F64},
    {     2,       100,       101,                              NVCV_THRESH_TOZERO_INV,            100,         255, nvcv::FMT_U8},
    {     1,         3,         3,                              NVCV_THRESH_TOZERO_INV,            100,         255, nvcv::FMT_U8},
    {     1,         3,         3,                              NVCV_THRESH_TOZERO_INV,             -1,         255, nvcv::FMT_U8},
    {     2,       100,       101,                              NVCV_THRESH_TOZERO_INV,             -1,         255, nvcv::FMT_U8},
    {     2,       100,       101,                              NVCV_THRESH_TOZERO_INV,            256,         255, nvcv::FMT_U8},
    {     2,       100,       101,                              NVCV_THRESH_TOZERO_INV,            0.5,         255, nvcv::FMT_F64},
    {     1,         9,         9,                              NVCV_THRESH_TOZERO_INV,            0.5,         255, nvcv::FMT_F64},
    {     1,       800,       600,                 NVCV_THRESH_OTSU|NVCV_THRESH_BINARY,            100,         255, nvcv::FMT_U8},
    {     3,       600,       1000,        NVCV_THRESH_TRIANGLE|NVCV_THRESH_BINARY_INV,            100,         255, nvcv::FMT_U8},
});

// clang-format on

TEST_P(OpThreshold, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               batch  = GetParamValue<0>();
    int               height = GetParamValue<1>();
    int               width  = GetParamValue<2>();
    uint32_t          type   = GetParamValue<3>();
    double            thresh = GetParamValue<4>();
    double            maxval = GetParamValue<5>();
    nvcv::ImageFormat fmt    = GetParamValue<6>();

    NVCVDataType nvcvDataType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(fmt), 0, &nvcvDataType));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(batch, width, height, fmt);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(batch, width, height, fmt);

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
        srcVec[i].resize(height * rowStride);
        if (nvcvDataType == NVCV_DATA_TYPE_F64)
        {
            myGenerate(reinterpret_cast<double *>(srcVec[i].data()), srcVec[i].size() / sizeof(double), randEng);
        }
        else
        {
            myGenerate(srcVec[i].data(), srcVec[i].size(), randEng);
        }
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
        ThresholdWrapper(srcVec[i], goldVec, thresh, maxval, type, nvcvDataType);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpThreshold, varshape_correct_shape)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               batch  = GetParamValue<0>();
    int               height = GetParamValue<1>();
    int               width  = GetParamValue<2>();
    uint32_t          type   = GetParamValue<3>();
    double            thresh = GetParamValue<4>();
    double            maxval = GetParamValue<5>();
    nvcv::ImageFormat fmt    = GetParamValue<6>();

    NVCVDataType nvcvDataType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(fmt), 0, &nvcvDataType));

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution rndHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
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

        srcVec[i].resize(srcHeight * srcRowStride);
        if (nvcvDataType == NVCV_DATA_TYPE_F64)
        {
            myGenerate(reinterpret_cast<double *>(srcVec[i].data()), srcVec[i].size() / sizeof(double), randEng);
        }
        else
        {
            myGenerate(srcVec[i].data(), srcVec[i].size(), randEng);
        }

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
        ThresholdWrapper(srcVec[i], goldVec, thresh, maxval, type, nvcvDataType);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpThresholdPlanar, nvcv::test::ValueList<int, int, int, uint32_t, double, double, nvcv::ImageFormat, nvcv::ImageFormat>
{
    // width, height, batch,                      type, thresh, maxval,             planar format,      interleaved format
    {    64,     48,     2,        NVCV_THRESH_BINARY,  100.0,  255.0,          nvcv::FMT_RGB8p,          nvcv::FMT_RGB8},
    {    57,     41,     1,    NVCV_THRESH_BINARY_INV,   96.0,  201.0,          nvcv::FMT_RGB8p,          nvcv::FMT_RGB8},
    {    53,     39,     1,         NVCV_THRESH_TRUNC,  100.0,  255.0,          nvcv::FMT_RGB8p,          nvcv::FMT_RGB8},
    {    51,     37,     1,        NVCV_THRESH_TOZERO,  100.0,  255.0,          nvcv::FMT_RGB8p,          nvcv::FMT_RGB8},
    {    49,     35,     1,    NVCV_THRESH_TOZERO_INV,  100.0,  255.0,          nvcv::FMT_RGB8p,          nvcv::FMT_RGB8},
    {    47,     33,     1,    NVCV_THRESH_BINARY_INV,   -1.0,  201.0,          nvcv::FMT_RGB8p,          nvcv::FMT_RGB8},
    {    45,     31,     1,         NVCV_THRESH_TRUNC,  300.0,  255.0,          nvcv::FMT_RGB8p,          nvcv::FMT_RGB8},
    {    43,     29,     1,        NVCV_THRESH_TOZERO,   -1.0,  255.0,          nvcv::FMT_RGB8p,          nvcv::FMT_RGB8},
    {    41,     27,     1,    NVCV_THRESH_TOZERO_INV,  300.0,  255.0,          nvcv::FMT_RGB8p,          nvcv::FMT_RGB8},
    {    39,     25,     1,        NVCV_THRESH_BINARY,    0.5,    1.0,       nvcv::FMT_RGBf32p,       nvcv::FMT_RGBf32},
    {    37,     23,     1,    NVCV_THRESH_BINARY_INV,    0.5,    1.0,       nvcv::FMT_RGBf32p,       nvcv::FMT_RGBf32},
    {    35,     31,     2,          NVCV_THRESH_TRUNC,    0.5,    1.0,       nvcv::FMT_RGBf32p,       nvcv::FMT_RGBf32},
    {    31,     21,     1,         NVCV_THRESH_TOZERO,    0.5,    1.0,      nvcv::FMT_RGBAf32p,      nvcv::FMT_RGBAf32},
    {    33,     29,     1,     NVCV_THRESH_TOZERO_INV,    0.5,    1.0,      nvcv::FMT_RGBAf32p,      nvcv::FMT_RGBAf32},
});

// clang-format on

TEST_P(OpThresholdPlanar, tensor_matches_interleaved)
{
    const int               width          = GetParamValue<0>();
    const int               height         = GetParamValue<1>();
    const int               numImages      = GetParamValue<2>();
    const uint32_t          type           = GetParamValue<3>();
    const double            thresh         = GetParamValue<4>();
    const double            maxval         = GetParamValue<5>();
    const nvcv::ImageFormat planarFmt      = GetParamValue<6>();
    const nvcv::ImageFormat interleavedFmt = GetParamValue<7>();

    nvcv::test::planar::RunTensorParity(planarFmt, interleavedFmt, width, height, width, height, numImages,
                                        [numImages, type, thresh, maxval](cudaStream_t s, const nvcv::Tensor &src,
                                                                          const nvcv::Tensor &dst, nvcv::ImageFormat)
                                        {
                                            auto threshval = MakeThresholdParam(numImages, thresh);
                                            auto maxvalval = MakeThresholdParam(numImages, maxval);

                                            cvcuda::Threshold thresholdOp(type, numImages);
                                            EXPECT_NO_THROW(thresholdOp(s, src, dst, threshval, maxvalval));
                                        });
}

TEST_P(OpThresholdPlanar, varshape_matches_interleaved)
{
    const int               width          = GetParamValue<0>();
    const int               height         = GetParamValue<1>();
    const int               numImages      = GetParamValue<2>();
    const uint32_t          type           = GetParamValue<3>();
    const double            thresh         = GetParamValue<4>();
    const double            maxval         = GetParamValue<5>();
    const nvcv::ImageFormat planarFmt      = GetParamValue<6>();
    const nvcv::ImageFormat interleavedFmt = GetParamValue<7>();

    nvcv::test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, width, height, width, height, numImages,
        [numImages, type, thresh, maxval](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                          const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            auto threshval = MakeThresholdParam(numImages, thresh);
            auto maxvalval = MakeThresholdParam(numImages, maxval);

            cvcuda::Threshold thresholdOp(type, numImages);
            EXPECT_NO_THROW(thresholdOp(s, src, dst, threshval, maxvalval));
        });
}

TEST(OpThresholdPlanar, tensor_rejects_two_channel)
{
    nvcv::test::planar::ExpectPlanarTensorRejected({1, 2, 16, 16}, {1, 2, 16, 16},
                                                   [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst)
                                                   {
                                                       auto threshval = MakeThresholdParam(1, 100.0);
                                                       auto maxvalval = MakeThresholdParam(1, 255.0);

                                                       cvcuda::Threshold thresholdOp(NVCV_THRESH_BINARY, 1);
                                                       thresholdOp(s, src, dst, threshval, maxvalval);
                                                   });
}

// clang-format off
NVCV_TEST_SUITE_P(OpThreshold_Negative, nvcv::test::ValueList<int, int, int, uint32_t, double, double, std::string, std::string, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::DataType, nvcv::DataType>
{
    //batch,    height,     width,                                                     type,         thresh,      maxval,      inFormat,    outFormat,  threshDataType,     maxvalType
    {     1,       224,       224,                                       NVCV_THRESH_BINARY,            100,         255, "N", "N", nvcv::FMT_F16, nvcv::FMT_F16, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                                       NVCV_THRESH_BINARY,            100,         255, "N", "N", nvcv::FMT_U8,  nvcv::FMT_U16, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                                       NVCV_THRESH_BINARY,            100,         255, "N", "N", nvcv::FMT_U8,  nvcv::FMT_U8,  nvcv::TYPE_F32, nvcv::TYPE_F64},
    {     1,       224,       224,                                       NVCV_THRESH_BINARY,            100,         255, "N", "N", nvcv::FMT_U8,  nvcv::FMT_U8,  nvcv::TYPE_F64, nvcv::TYPE_F32},
    {     1,       224,       224, NVCV_THRESH_TRIANGLE|NVCV_THRESH_OTSU|NVCV_THRESH_BINARY,            100,         255, "N", "N", nvcv::FMT_U8,  nvcv::FMT_U8,  nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                     NVCV_THRESH_TRUNC|NVCV_THRESH_BINARY,            100,         255, "N", "N", nvcv::FMT_U8,  nvcv::FMT_U8,  nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                NVCV_THRESH_BINARY_INV|NVCV_THRESH_TRUNC,            100,         255, "N", "N", nvcv::FMT_U8,  nvcv::FMT_U8,  nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,               NVCV_THRESH_BINARY_INV|NVCV_THRESH_TOZERO,            100,         255, "N", "N", nvcv::FMT_U8,  nvcv::FMT_U8,  nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                    NVCV_THRESH_TRUNC|NVCV_THRESH_TOZERO,            100,         255, "N", "N", nvcv::FMT_U8,  nvcv::FMT_U8,  nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                      NVCV_THRESH_OTSU|NVCV_THRESH_BINARY,            100,         255, "N", "N", nvcv::FMT_U16, nvcv::FMT_U16, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                      NVCV_THRESH_OTSU|NVCV_THRESH_BINARY,            100,         255, "N", "N", nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                  NVCV_THRESH_TRIANGLE|NVCV_THRESH_BINARY,            100,         255, "N", "N", nvcv::FMT_U16, nvcv::FMT_U16, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                  NVCV_THRESH_TRIANGLE|NVCV_THRESH_BINARY,            100,         255, "N", "N", nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                                       NVCV_THRESH_BINARY,            100,         255, "NW", "N", nvcv::FMT_U8,  nvcv::FMT_U8, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {     1,       224,       224,                                       NVCV_THRESH_BINARY,            100,         255, "N", "NW", nvcv::FMT_U8,  nvcv::FMT_U8, nvcv::TYPE_F64, nvcv::TYPE_F64},
});

// clang-format on

TEST(OpThreshold_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaThresholdCreate(nullptr, NVCV_THRESH_BINARY, 5));
}

TEST(OpThreshold_Negative, create_with_negative_maxBatchSize)
{
    NVCVOperatorHandle thresholdHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaThresholdCreate(&thresholdHandle, NVCV_THRESH_BINARY, -1));
}

static void ExpectAutoThresholdTensorBatchExceedsMaxBatch(uint32_t type)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int batch    = 2;
    constexpr int maxBatch = 1;
    constexpr int width    = 4;
    constexpr int height   = 4;

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);

    nvcv::Tensor threshval({{batch}, "N"}, nvcv::TYPE_F64);
    nvcv::Tensor maxvalval({{batch}, "N"}, nvcv::TYPE_F64);

    auto threshData = threshval.exportData<nvcv::TensorDataStridedCuda>();
    auto maxvalData = maxvalval.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, threshData);
    ASSERT_NE(nullptr, maxvalData);

    std::vector<double> threshVec(batch, 100);
    std::vector<double> maxvalVec(batch, 255);

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(threshData->basePtr(), threshVec.data(), threshVec.size() * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(maxvalData->basePtr(), maxvalVec.data(), maxvalVec.size() * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));

    cvcuda::Threshold thresholdOp(type, maxBatch);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&thresholdOp, &stream, &imgIn, &imgOut, &threshval, &maxvalval]
                                { thresholdOp(stream, imgIn, imgOut, threshval, maxvalval); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void ExpectAutoThresholdVarShapeBatchExceedsMaxBatch(uint32_t type)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int batch    = 2;
    constexpr int maxBatch = 1;
    constexpr int width    = 4;
    constexpr int height   = 4;

    std::vector<nvcv::Image> imgSrc;

    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batch; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_U8);
        imgDst.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_U8);
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor threshval({{batch}, "N"}, nvcv::TYPE_F64);
    nvcv::Tensor maxvalval({{batch}, "N"}, nvcv::TYPE_F64);

    auto threshData = threshval.exportData<nvcv::TensorDataStridedCuda>();
    auto maxvalData = maxvalval.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, threshData);
    ASSERT_NE(nullptr, maxvalData);

    std::vector<double> threshVec(batch, 100);
    std::vector<double> maxvalVec(batch, 255);

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(threshData->basePtr(), threshVec.data(), threshVec.size() * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(maxvalData->basePtr(), maxvalVec.data(), maxvalVec.size() * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));

    cvcuda::Threshold thresholdOp(type, maxBatch);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&thresholdOp, &stream, &batchSrc, &batchDst, &threshval, &maxvalval]
                                { thresholdOp(stream, batchSrc, batchDst, threshval, maxvalval); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpThreshold_Negative, tensor_otsu_batch_exceeds_maxBatch)
{
    ExpectAutoThresholdTensorBatchExceedsMaxBatch(NVCV_THRESH_OTSU | NVCV_THRESH_BINARY);
}

TEST(OpThreshold_Negative, tensor_triangle_batch_exceeds_maxBatch)
{
    ExpectAutoThresholdTensorBatchExceedsMaxBatch(NVCV_THRESH_TRIANGLE | NVCV_THRESH_BINARY);
}

TEST(OpThreshold_Negative, varshape_otsu_batch_exceeds_maxBatch)
{
    ExpectAutoThresholdVarShapeBatchExceedsMaxBatch(NVCV_THRESH_OTSU | NVCV_THRESH_BINARY);
}

TEST(OpThreshold_Negative, varshape_triangle_batch_exceeds_maxBatch)
{
    ExpectAutoThresholdVarShapeBatchExceedsMaxBatch(NVCV_THRESH_TRIANGLE | NVCV_THRESH_BINARY);
}

TEST_P(OpThreshold_Negative, invalid_inputs)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               batch          = GetParamValue<0>();
    int               height         = GetParamValue<1>();
    int               width          = GetParamValue<2>();
    uint32_t          type           = GetParamValue<3>();
    double            thresh         = GetParamValue<4>();
    double            maxval         = GetParamValue<5>();
    std::string       threshLayout   = GetParamValue<6>();
    std::string       maxvalLayout   = GetParamValue<7>();
    nvcv::ImageFormat inFormat       = GetParamValue<8>();
    nvcv::ImageFormat outFormat      = GetParamValue<9>();
    nvcv::DataType    threshDataType = GetParamValue<10>();
    nvcv::DataType    maxvalDataType = GetParamValue<11>();

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(batch, width, height, inFormat);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(batch, width, height, outFormat);

    //parameters
    nvcv::TensorShape threshShape = threshLayout.size() == 1 ? nvcv::TensorShape({batch}, threshLayout.c_str())
                                                             : nvcv::TensorShape({batch, 1}, threshLayout.c_str());
    nvcv::TensorShape maxvalShape = maxvalLayout.size() == 1 ? nvcv::TensorShape({batch}, maxvalLayout.c_str())
                                                             : nvcv::TensorShape({batch, 1}, maxvalLayout.c_str());
    nvcv::Tensor      threshval(threshShape, threshDataType);
    nvcv::Tensor      maxvalval(maxvalShape, maxvalDataType);

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

    // Call operator
    int               maxBatch = 5;
    cvcuda::Threshold thresholdOp(type, maxBatch);
    EXPECT_ANY_THROW(thresholdOp(stream, imgIn, imgOut, threshval, maxvalval));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpThreshold_Negative, varshape_invalid_inputs)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               batch          = GetParamValue<0>();
    int               height         = GetParamValue<1>();
    int               width          = GetParamValue<2>();
    uint32_t          type           = GetParamValue<3>();
    double            thresh         = GetParamValue<4>();
    double            maxval         = GetParamValue<5>();
    std::string       threshLayout   = GetParamValue<6>();
    std::string       maxvalLayout   = GetParamValue<7>();
    nvcv::ImageFormat inFormat       = GetParamValue<8>();
    nvcv::ImageFormat outFormat      = GetParamValue<9>();
    nvcv::DataType    threshDataType = GetParamValue<10>();
    nvcv::DataType    maxvalDataType = GetParamValue<11>();

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution rndHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batch; ++i)
    {
        int rw = rndWidth(randEng);
        int rh = rndHeight(randEng);
        imgSrc.emplace_back(nvcv::Size2D{rw, rh}, inFormat);
        imgDst.emplace_back(nvcv::Size2D{rw, rh}, outFormat);
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    //parameters
    nvcv::TensorShape threshShape = threshLayout.size() == 1 ? nvcv::TensorShape({batch}, threshLayout.c_str())
                                                             : nvcv::TensorShape({batch, 1}, threshLayout.c_str());
    nvcv::TensorShape maxvalShape = maxvalLayout.size() == 1 ? nvcv::TensorShape({batch}, maxvalLayout.c_str())
                                                             : nvcv::TensorShape({batch, 1}, maxvalLayout.c_str());
    nvcv::Tensor      threshval(threshShape, threshDataType);
    nvcv::Tensor      maxvalval(maxvalShape, maxvalDataType);

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
    // Call operator
    int               maxBatch = 5;
    cvcuda::Threshold thresholdOp(type, maxBatch);
    EXPECT_ANY_THROW(thresholdOp(stream, batchSrc, batchDst, threshval, maxvalval));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpThreshold, otsu_corner_cases)
{
    int      batch  = 3;
    int      height = 255;
    int      width  = 255;
    uint32_t type   = NVCV_THRESH_OTSU | NVCV_THRESH_BINARY;

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    double            thresh = 100;
    double            maxval = 255;
    nvcv::ImageFormat fmt    = nvcv::FMT_U8;

    NVCVDataType nvcvDataType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(fmt), 0, &nvcvDataType));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(batch, width, height, fmt);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(batch, width, height, fmt);

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
        srcVec[i].resize(height * rowStride);
    }

    for (size_t j = 0; j < srcVec[0].size(); ++j)
    {
        srcVec[0][j] = (j % 2 == 0) ? 50 : 200;
    }
    for (size_t j = 0; j < srcVec[1].size(); ++j)
    {
        srcVec[1][j] = static_cast<uint8_t>(j % 256);
    }
    for (size_t j = 0; j < srcVec[2].size(); ++j)
    {
        srcVec[2][j] = static_cast<uint8_t>((j % 4) * 64);
    }

    for (int i = 0; i < batch; i++)
    {
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
        ThresholdWrapper(srcVec[i], goldVec, thresh, maxval, type, nvcvDataType);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpThreshold, otsu_corner_cases_varshape)
{
    int      batch  = 3;
    int      height = 255;
    int      width  = 255;
    uint32_t type   = NVCV_THRESH_OTSU | NVCV_THRESH_BINARY;

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    double            thresh = 100;
    double            maxval = 255;
    nvcv::ImageFormat fmt    = nvcv::FMT_U8;

    NVCVDataType nvcvDataType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(fmt), 0, &nvcvDataType));

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution rndHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<nvcv::Image> imgDst;
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

        srcVec[i].resize(srcHeight * srcRowStride);
    }

    for (size_t j = 0; j < srcVec[0].size(); ++j)
    {
        srcVec[0][j] = (j % 2 == 0) ? 50 : 200;
    }

    for (size_t j = 0; j < srcVec[1].size(); ++j)
    {
        srcVec[1][j] = static_cast<uint8_t>(j % 256);
    }

    for (size_t j = 0; j < srcVec[2].size(); ++j)
    {
        srcVec[2][j] = static_cast<uint8_t>((j % 4) * 64);
    }

    for (int i = 0; i < batch; i++)
    {
        const auto srcData   = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        int        srcWidth  = srcData->plane(0).width;
        int        srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);
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
        ThresholdWrapper(srcVec[i], goldVec, thresh, maxval, type, nvcvDataType);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpThreshold, triangle_corner_cases)
{
    const int height = 256;
    const int width  = 256;
    const int batch  = 3;

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    double            thresh = 100;
    double            maxval = 255;
    nvcv::ImageFormat fmt    = nvcv::FMT_U8;

    NVCVDataType nvcvDataType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(fmt), 0, &nvcvDataType));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(batch, width, height, fmt);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(batch, width, height, fmt);

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

    // Generate input
    std::vector<std::vector<uint8_t>> srcVec(batch);
    std::default_random_engine        randEng;
    int                               rowStride = width * fmt.planePixelStrideBytes(0);

    for (size_t i = 0; i < batch; i++)
    {
        srcVec[i].resize(height * rowStride);
    }

    for (size_t j = 0; j < srcVec[0].size(); j++)
    {
        srcVec[0][j] = static_cast<uint8_t>((j % 191) + 10);
    }

    for (size_t j = 0; j < srcVec[1].size(); j++)
    {
        srcVec[1][j] = static_cast<uint8_t>(j % 201);
    }

    for (size_t j = 0; j < srcVec[2].size(); j++)
    {
        if (j % 2 == 0)
        {
            srcVec[2][j] = 64;
        }
        else
        {
            srcVec[2][j] = 192;
        }
    }
    for (int j = 0; j < 1000; j++)
    {
        int idx        = j * 3 % srcVec[2].size();
        srcVec[2][idx] = 128;
    }

    for (int i = 0; i < batch; i++)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(i), inAccess->rowStride(), srcVec[i].data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));
    }

    // Call operator
    int               maxBatch = 5;
    cvcuda::Threshold thresholdOp(NVCV_THRESH_TRIANGLE | NVCV_THRESH_BINARY, maxBatch);
    thresholdOp(stream, imgIn, imgOut, threshval, maxvalval);

    // Verify the results
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(height * rowStride);
        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowStride, outAccess->sampleData(i), outAccess->rowStride(),
                                            rowStride, height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(height * rowStride);
        ThresholdWrapper(srcVec[i], goldVec, thresh, maxval, NVCV_THRESH_TRIANGLE | NVCV_THRESH_BINARY, nvcvDataType);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

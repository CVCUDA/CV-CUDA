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

#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>
#include <vector>

namespace test  = nvcv::test;
namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

using uchar = unsigned char;

template<typename T>
inline T &ValueAt(std::vector<uint8_t> &vec, const long4_16a &pitches, int b, int y, int x, int c,
                  nvcv::TensorLayout layout)
{
    if (layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC)
    {
        return *reinterpret_cast<T *>(&vec[b * pitches.x + y * pitches.y + x * pitches.z + c * pitches.w]);
    }
    else if (layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW)
    {
        return *reinterpret_cast<T *>(&vec[b * pitches.x + c * pitches.y + y * pitches.z + x * pitches.w]);
    }
    return *reinterpret_cast<T *>(&vec[0]);
}

struct ReformatRefData
{
    std::vector<uint8_t> &hDst;
    long4_16a             dstStrides;
    nvcv::TensorLayout    dstLayout;
    std::vector<uint8_t> &hSrc;
    long4_16a             srcStrides;
    nvcv::TensorLayout    srcLayout;
    int                   numChannels;
};

template<typename T>
inline void ReformatPixel(ReformatRefData &ref, int b, int y, int x)
{
    for (int c = 0; c < ref.numChannels; ++c)
    {
        ValueAt<T>(ref.hDst, ref.dstStrides, b, y, x, c, ref.dstLayout)
            = ValueAt<T>(ref.hSrc, ref.srcStrides, b, y, x, c, ref.srcLayout);
    }
}

template<typename T>
inline void Reformat(std::vector<uint8_t> &hDst, const long4_16a &dstStrides, nvcv::TensorLayout dstLayout,
                     std::vector<uint8_t> &hSrc, const long4_16a &srcStrides, nvcv::TensorLayout srcLayout,
                     int numBatches, int numRows, int numCols, int numChannels)
{
    ReformatRefData ref{hDst, dstStrides, dstLayout, hSrc, srcStrides, srcLayout, numChannels};

    for (int b = 0; b < numBatches; ++b)
    {
        for (int y = 0; y < numRows; ++y)
        {
            for (int x = 0; x < numCols; ++x)
            {
                ReformatPixel<T>(ref, b, y, x);
            }
        }
    }
}

#define NVCV_TEST_ROW(WIDTH, HEIGHT, BATCHES, INFORMAT, OUTFORMAT, VALUETYPE)                              \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<BATCHES>, ttype::Value<INFORMAT>, \
                 ttype::Value<OUTFORMAT>, VALUETYPE>

NVCV_TYPED_TEST_SUITE(
    OpReformat, ttype::Types<NVCV_TEST_ROW(176, 113, 1, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8, uchar),
                             NVCV_TEST_ROW(23, 43, 23, NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_RGB8p, uchar),
                             NVCV_TEST_ROW(7, 4, 7, NVCV_IMAGE_FORMAT_RGBf32, NVCV_IMAGE_FORMAT_RGBf32, float),
                             NVCV_TEST_ROW(3, 2, 1, NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_RGB8, uchar),
                             NVCV_TEST_ROW(2, 3, 1, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8p, uchar),
                             NVCV_TEST_ROW(56, 49, 2, NVCV_IMAGE_FORMAT_RGBA8p, NVCV_IMAGE_FORMAT_RGBA8, uchar),
                             NVCV_TEST_ROW(56, 49, 3, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8p, uchar),
                             NVCV_TEST_ROW(31, 30, 3, NVCV_IMAGE_FORMAT_RGBAf32, NVCV_IMAGE_FORMAT_RGBAf32p, float),
                             NVCV_TEST_ROW(30, 31, 3, NVCV_IMAGE_FORMAT_RGBf32p, NVCV_IMAGE_FORMAT_RGBf32, float)>);

#undef NVCV_TEST_ROW

TYPED_TEST(OpReformat, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = ttype::GetValue<TypeParam, 0>;
    int height  = ttype::GetValue<TypeParam, 1>;
    int batches = ttype::GetValue<TypeParam, 2>;

    nvcv::ImageFormat inFormat{ttype::GetValue<TypeParam, 3>};
    nvcv::ImageFormat outFormat{ttype::GetValue<TypeParam, 4>};

    using ValueType = ttype::GetType<TypeParam, 5>;

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, inFormat);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, outFormat);

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    ASSERT_EQ(inAccess->numChannels(), outAccess->numChannels());
    ASSERT_EQ(inData->rank(), outData->rank());
    ASSERT_TRUE(inData->rank() == 3 || inData->rank() == 4);

    long4_16a inStrides;
    long4_16a outStrides;

    if (inData->rank() == 3)
    {
        long inNumPlanes  = inData->layout() == nvcv::TENSOR_CHW ? inAccess->numChannels() : 1;
        long outNumPlanes = outData->layout() == nvcv::TENSOR_CHW ? outAccess->numChannels() : 1;

        inStrides.x  = inAccess->numRows() * inAccess->rowStride() * inNumPlanes;
        outStrides.x = outAccess->numRows() * outAccess->rowStride() * outNumPlanes;

        inStrides.y = inData->stride(0);
        inStrides.z = inData->stride(1);
        inStrides.w = inData->stride(2);

        outStrides.y = outData->stride(0);
        outStrides.z = outData->stride(1);
        outStrides.w = outData->stride(2);
    }
    else
    {
        inStrides  = long4_16a{inData->stride(0), inData->stride(1), inData->stride(2), inData->stride(3)};
        outStrides = long4_16a{outData->stride(0), outData->stride(1), outData->stride(2), outData->stride(3)};
    }

    auto numBatches  = static_cast<int>(inAccess->numSamples());
    int  numRows     = inAccess->numRows();
    int  numCols     = inAccess->numCols();
    int  numChannels = inAccess->numChannels();

    long inBufSize  = inStrides.x * inAccess->numSamples();
    long outBufSize = outStrides.x * outAccess->numSamples();

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::ranges::generate(inVec, [&rand, &randEng]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::Reformat reformatOp;

    EXPECT_NO_THROW(reformatOp(stream, inTensor, outTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    Reformat<ValueType>(goldVec, outStrides, outData->layout(), inVec, inStrides, inData->layout(), numBatches, numRows,
                        numCols, numChannels);

    EXPECT_EQ(testVec, goldVec);
}

static nvcv::Tensor WrapU8C1Tensor(NVCVByte *basePtr, nvcv::TensorLayout layout, int batches, int height, int width,
                                   int rowStride, int sampleStride)
{
    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.basePtr    = basePtr;
    buffer.strides[0] = sampleStride;

    if (layout == nvcv::TENSOR_NHWC)
    {
        buffer.strides[1] = rowStride;
        buffer.strides[2] = 1;
        buffer.strides[3] = 1;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{batches, height, width, 1}, "NHWC"},
            nvcv::TYPE_U8, buffer
        });
    }

    buffer.strides[1] = rowStride * height;
    buffer.strides[2] = rowStride;
    buffer.strides[3] = 1;
    return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{batches, 1, height, width}, "NCHW"},
        nvcv::TYPE_U8, buffer
    });
}

static void RunU8C1PaddedStrideCase(nvcv::TensorLayout srcLayout, nvcv::TensorLayout dstLayout, int batches,
                                    int height = 5, int width = 37)
{
    const int         srcRowStride    = width + 11;
    const int         dstRowStride    = width + 27;
    const int         srcSampleStride = srcRowStride * height + 16;
    const int         dstSampleStride = dstRowStride * height + 32;
    constexpr uint8_t srcPadding      = 0xA5;
    constexpr uint8_t dstPadding      = 0xD7;

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, batches * srcSampleStride));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, batches * dstSampleStride));

    std::vector<uint8_t> src(batches * srcSampleStride, srcPadding);
    std::vector<uint8_t> dst(batches * dstSampleStride, dstPadding);
    std::vector<uint8_t> expected = dst;
    for (int b = 0; b < batches; ++b)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const auto value                                = static_cast<uint8_t>(b * 73 + y * 19 + x * 7 + 11);
                src[b * srcSampleStride + y * srcRowStride + x] = value;
                expected[b * dstSampleStride + y * dstRowStride + x] = value;
            }
        }
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, src.data(), src.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dst.data(), dst.size(), cudaMemcpyHostToDevice));

    nvcv::Tensor srcTensor = WrapU8C1Tensor(srcPtr, srcLayout, batches, height, width, srcRowStride, srcSampleStride);
    nvcv::Tensor dstTensor = WrapU8C1Tensor(dstPtr, dstLayout, batches, height, width, dstRowStride, dstSampleStride);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::Reformat op;
    EXPECT_NO_THROW(op(stream, srcTensor, dstTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> got(expected.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(got.data(), dstPtr, got.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, got);

    std::vector<uint8_t> gotSrc(src.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(gotSrc.data(), srcPtr, gotSrc.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(src, gotSrc);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

TEST(OpReformat, u8_c1_padded_strides_preserve_canaries)
{
    RunU8C1PaddedStrideCase(nvcv::TENSOR_NCHW, nvcv::TENSOR_NHWC, 1);
    RunU8C1PaddedStrideCase(nvcv::TENSOR_NHWC, nvcv::TENSOR_NCHW, 1);
    RunU8C1PaddedStrideCase(nvcv::TENSOR_NCHW, nvcv::TENSOR_NHWC, 3);
    RunU8C1PaddedStrideCase(nvcv::TENSOR_NHWC, nvcv::TENSOR_NCHW, 3);
}

TEST(OpReformat, u8_c1_large_image_pitched_copy_preserves_canaries)
{
    constexpr int batches = 8;
    constexpr int height  = 900;
    constexpr int width   = 1600;

    RunU8C1PaddedStrideCase(nvcv::TENSOR_NCHW, nvcv::TENSOR_NHWC, batches, height, width);
    RunU8C1PaddedStrideCase(nvcv::TENSOR_NHWC, nvcv::TENSOR_NCHW, batches, height, width);
}

struct U8ReformatSpec
{
    nvcv::TensorLayout layout;
    int                batches;
    int                channels;
    int                height;
    int                width;
    int                rowStride;
    int                sampleStride;
};

static bool IsInterleaved(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC;
}

static bool HasBatch(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_NCHW;
}

static U8ReformatSpec MakeU8ReformatSpec(nvcv::TensorLayout layout, int batches, int channels, int height, int width,
                                         int rowPadding = 0, int samplePadding = 0)
{
    const int rowStride    = width * (IsInterleaved(layout) ? channels : 1) + rowPadding;
    const int sampleStride = rowStride * height * (layout == nvcv::TENSOR_NCHW ? channels : 1) + samplePadding;
    return {layout, batches, channels, height, width, rowStride, sampleStride};
}

static size_t U8StorageSize(const U8ReformatSpec &spec)
{
    if (HasBatch(spec.layout))
    {
        return static_cast<size_t>(spec.batches) * spec.sampleStride;
    }
    return static_cast<size_t>(spec.rowStride) * spec.height * (spec.layout == nvcv::TENSOR_CHW ? spec.channels : 1);
}

static size_t U8Offset(const U8ReformatSpec &spec, int b, int y, int x, int c)
{
    if (spec.layout == nvcv::TENSOR_NHWC)
    {
        return static_cast<size_t>(b) * spec.sampleStride + static_cast<size_t>(y) * spec.rowStride
             + static_cast<size_t>(x) * spec.channels + c;
    }
    if (spec.layout == nvcv::TENSOR_NCHW)
    {
        return static_cast<size_t>(b) * spec.sampleStride + static_cast<size_t>(c) * spec.rowStride * spec.height
             + static_cast<size_t>(y) * spec.rowStride + x;
    }
    if (spec.layout == nvcv::TENSOR_HWC)
    {
        return static_cast<size_t>(y) * spec.rowStride + static_cast<size_t>(x) * spec.channels + c;
    }
    return static_cast<size_t>(c) * spec.rowStride * spec.height + static_cast<size_t>(y) * spec.rowStride + x;
}

static nvcv::Tensor WrapU8Tensor(NVCVByte *basePtr, const U8ReformatSpec &spec)
{
    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.basePtr = basePtr;

    if (spec.layout == nvcv::TENSOR_NHWC)
    {
        buffer.strides[0] = spec.sampleStride;
        buffer.strides[1] = spec.rowStride;
        buffer.strides[2] = spec.channels;
        buffer.strides[3] = 1;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{spec.batches, spec.height, spec.width, spec.channels}, "NHWC"},
            nvcv::TYPE_U8, buffer
        });
    }
    if (spec.layout == nvcv::TENSOR_NCHW)
    {
        buffer.strides[0] = spec.sampleStride;
        buffer.strides[1] = spec.rowStride * spec.height;
        buffer.strides[2] = spec.rowStride;
        buffer.strides[3] = 1;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{spec.batches, spec.channels, spec.height, spec.width}, "NCHW"},
            nvcv::TYPE_U8, buffer
        });
    }
    if (spec.layout == nvcv::TENSOR_HWC)
    {
        buffer.strides[0] = spec.rowStride;
        buffer.strides[1] = spec.channels;
        buffer.strides[2] = 1;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{spec.height, spec.width, spec.channels}, "HWC"},
            nvcv::TYPE_U8, buffer
        });
    }

    buffer.strides[0] = spec.rowStride * spec.height;
    buffer.strides[1] = spec.rowStride;
    buffer.strides[2] = 1;
    return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{spec.channels, spec.height, spec.width}, "CHW"},
        nvcv::TYPE_U8, buffer
    });
}

static void SetU8ReformatPixel(std::vector<uint8_t> &src, std::vector<uint8_t> &expected, const U8ReformatSpec &srcSpec,
                               const U8ReformatSpec &dstSpec, int b, int y, int x)
{
    for (int c = 0; c < srcSpec.channels; ++c)
    {
        const auto value                        = static_cast<uint8_t>(b * 73 + y * 19 + x * 7 + c * 41 + 11);
        src[U8Offset(srcSpec, b, y, x, c)]      = value;
        expected[U8Offset(dstSpec, b, y, x, c)] = value;
    }
}

static void RunU8BitExactCase(nvcv::TensorLayout srcLayout, nvcv::TensorLayout dstLayout, int batches, int channels,
                              int height, int width)
{
    ASSERT_EQ(HasBatch(srcLayout), HasBatch(dstLayout));
    if (!HasBatch(srcLayout))
    {
        ASSERT_EQ(batches, 1);
    }

    const U8ReformatSpec srcSpec = MakeU8ReformatSpec(srcLayout, batches, channels, height, width);
    const U8ReformatSpec dstSpec = MakeU8ReformatSpec(dstLayout, batches, channels, height, width);

    std::vector<uint8_t> src(U8StorageSize(srcSpec), 0xA5);
    std::vector<uint8_t> dst(U8StorageSize(dstSpec), 0xD7);
    std::vector<uint8_t> expected = dst;
    for (int b = 0; b < batches; ++b)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                SetU8ReformatPixel(src, expected, srcSpec, dstSpec, b, y, x);
            }
        }
    }

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, src.size()));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, dst.size()));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, src.data(), src.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dst.data(), dst.size(), cudaMemcpyHostToDevice));

    nvcv::Tensor srcTensor = WrapU8Tensor(srcPtr, srcSpec);
    nvcv::Tensor dstTensor = WrapU8Tensor(dstPtr, dstSpec);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::Reformat op;
    EXPECT_NO_THROW(op(stream, srcTensor, dstTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> got(expected.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(got.data(), dstPtr, got.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, got);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

TEST(OpReformat, u8_c2_odd_tail_bit_exact)
{
    RunU8BitExactCase(nvcv::TENSOR_NCHW, nvcv::TENSOR_NHWC, 3, 2, 19, 131);
    RunU8BitExactCase(nvcv::TENSOR_NHWC, nvcv::TENSOR_NCHW, 3, 2, 19, 131);
}

TEST(OpReformat, u8_c1_contiguous_rank4_copy)
{
    RunU8BitExactCase(nvcv::TENSOR_NCHW, nvcv::TENSOR_NHWC, 3, 1, 5, 37);
    RunU8BitExactCase(nvcv::TENSOR_NHWC, nvcv::TENSOR_NCHW, 3, 1, 5, 37);
}

TEST(OpReformat, u8_c1_contiguous_width_256_bit_exact)
{
    RunU8BitExactCase(nvcv::TENSOR_NCHW, nvcv::TENSOR_NHWC, 1, 1, 5, 256);
    RunU8BitExactCase(nvcv::TENSOR_NHWC, nvcv::TENSOR_NCHW, 1, 1, 5, 256);
}

TEST(OpReformat, u8_c1_contiguous_rank3_copy)
{
    RunU8BitExactCase(nvcv::TENSOR_CHW, nvcv::TENSOR_HWC, 1, 1, 5, 37);
    RunU8BitExactCase(nvcv::TENSOR_HWC, nvcv::TENSOR_CHW, 1, 1, 5, 37);
}

TEST(OpReformat_Negative, mismatched_extents)
{
    struct ExtentCase
    {
        const char *name;
        int         inSamples;
        int         inChannels;
        int         inRows;
        int         inCols;
        int         outSamples;
        int         outChannels;
        int         outRows;
        int         outCols;
    };

    const std::vector<ExtentCase> testCases = {
        { "samples", 1, 1, 5, 37, 2, 1, 5, 37},
        {"channels", 1, 4, 5, 37, 1, 3, 5, 37},
        {    "rows", 1, 1, 4, 37, 1, 1, 5, 37},
        {    "cols", 1, 1, 5, 36, 1, 1, 5, 37},
    };

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Reformat reformatOp;
    for (const ExtentCase &testCase : testCases)
    {
        SCOPED_TRACE(testCase.name);

        nvcv::Tensor inTensor(
            nvcv::TensorShape{
                {testCase.inSamples, testCase.inChannels, testCase.inRows, testCase.inCols},
                "NCHW"
        },
            nvcv::TYPE_U8);
        nvcv::Tensor outTensor(
            nvcv::TensorShape{
                {testCase.outSamples, testCase.outRows, testCase.outCols, testCase.outChannels},
                "NHWC"
        },
            nvcv::TYPE_U8);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&reformatOp, &stream, &inTensor, &outTensor]
                                                                 { reformatOp(stream, inTensor, outTensor); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpReformat_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int, int>{
    // inFmt, outFmt, inputBatches, outputBatches
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 2, 1},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 1, 2},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 6, 3},
    {nvcv::FMT_RGB8p, nvcv::FMT_RGBf32, 1, 1},
    {nvcv::FMT_RGBf32, nvcv::FMT_RGB8p, 1, 1},
    {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16p, 1, 1}
});

// clang-format on

TEST_P(OpReformat_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inputFmt      = GetParamValue<0>();
    nvcv::ImageFormat outputFmt     = GetParamValue<1>();
    int               inputBatches  = GetParamValue<2>();
    int               outputBatches = GetParamValue<3>();

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(inputBatches, 24, 24, inputFmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(outputBatches, 24, 24, outputFmt);

    // run operator
    cvcuda::Reformat reformatOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&reformatOp, &stream, &inTensor, &outTensor]
                                                             { reformatOp(stream, inTensor, outTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpReformat_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaReformatCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}

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

#include "ConvUtils.hpp"
#include "Definitions.hpp"
#include "HalfTestUtils.hpp"
#include "PlanarParityUtils.hpp"

#include <common/ValueTests.hpp>
#include <cuda_fp16.h>
#include <cvcuda/OpConv2D.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cstring>
#include <random>

namespace cuda = nvcv::cuda;
namespace test = nvcv::test;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

static nvcv::ImageBatchVarShape MakeConv2DKernelBatch(const std::vector<nvcv::Size2D> &kernelSizes,
                                                      std::vector<std::vector<float>> *hostKernels = nullptr)
{
    const auto               numImages = static_cast<int>(kernelSizes.size());
    std::vector<nvcv::Image> kernelImages;
    kernelImages.reserve(numImages);
    if (hostKernels)
        hostKernels->resize(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        const nvcv::Size2D kernelSize = kernelSizes[i];
        kernelImages.emplace_back(kernelSize, nvcv::FMT_F32);

        std::vector<float> kernel(static_cast<size_t>(kernelSize.w) * kernelSize.h);
        for (size_t k = 0; k < kernel.size(); ++k)
        {
            const int coeff = static_cast<int>((k + static_cast<size_t>(i)) % 7) - 3;
            kernel[k]       = static_cast<float>(coeff) * 0.03125f;
        }
        if (hostKernels)
            (*hostKernels)[i] = kernel;

        auto data = kernelImages.back().exportData<nvcv::ImageDataStridedCuda>();
        EXPECT_NE(data, nvcv::NullOpt);
        if (data)
        {
            const size_t rowBytes = static_cast<size_t>(kernelSize.w) * sizeof(float);
            EXPECT_EQ(cudaSuccess, cudaMemcpy2D(data->plane(0).basePtr, data->plane(0).rowStride, kernel.data(),
                                                rowBytes, rowBytes, kernelSize.h, cudaMemcpyHostToDevice));
        }
    }

    nvcv::ImageBatchVarShape batchKernel(numImages);
    batchKernel.pushBack(kernelImages.begin(), kernelImages.end());
    return batchKernel;
}

// clang-format off

NVCV_TEST_SUITE_P(OpConv2D, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, bool>
{
    // width, height, numImages, kernelWidth, kernelHeight, anchorX, anchorY, borderMode, mixedKernelSizes
    {     32,     33,         1,           3,            3,      -1,      -1,   NVCV_BORDER_CONSTANT, false},
    {    123,    144,         2,           5,            5,      -1,      -1,   NVCV_BORDER_CONSTANT, false},
    {     66,     99,         3,           7,            7,       5,       5,   NVCV_BORDER_CONSTANT, false},
    {     13,     12,        13,           5,            5,       4,       4,       NVCV_BORDER_WRAP, false},
    {      4,      3,         4,           3,            3,       1,       1,  NVCV_BORDER_REPLICATE, false},
    {     44,     55,         5,           3,            3,      -1,      -1,    NVCV_BORDER_REFLECT, false},
    {    244,    155,         6,           5,            5,      -1,      -1, NVCV_BORDER_REFLECT101, false},
    {     64,     48,         2,           7,            7,      -1,      -1,   NVCV_BORDER_CONSTANT, true}
});

// clang-format on

TEST_P(OpConv2D, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int  width            = GetParamValue<0>();
    int  height           = GetParamValue<1>();
    int  numImages        = GetParamValue<2>();
    int  kernelWidth      = GetParamValue<3>();
    int  kernelHeight     = GetParamValue<4>();
    int  kernelAnchorX    = GetParamValue<5>();
    int  kernelAnchorY    = GetParamValue<6>();
    bool mixedKernelSizes = GetParamValue<8>();

    NVCVBorderType borderMode = GetParamValue<7>();

    nvcv::ImageFormat imageFormat  = nvcv::FMT_RGBA8;
    nvcv::ImageFormat kernelFormat = nvcv::FMT_F32;

    nvcv::Size2D              kernelSize{kernelWidth, kernelHeight};
    std::vector<nvcv::Size2D> kernelSizes(numImages, kernelSize);
    if (mixedKernelSizes)
        kernelSizes.back() = nvcv::Size2D{3, 3};
    int2 kernelAnchor{kernelAnchorX, kernelAnchorY};

    float4 borderValue = cuda::SetAll<float4>(0);

    // Create input varshape

    std::default_random_engine rng;

    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    std::vector<int>                  srcVecRowStride(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, imageFormat);

        int srcRowStride   = imgSrc[i].size().w * imageFormat.numChannels();
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        std::ranges::generate(srcVec[i], [&udist, &rng]() { return udist(rng); });

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape

    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create kernel varshape

    std::vector<nvcv::Image>        kernel;
    std::vector<std::vector<float>> kernelVec(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        kernel.emplace_back(kernelSizes[i], kernelFormat);

        int rowStride = kernel[i].size().w * sizeof(float);

        std::uniform_real_distribution<float> udist(0.f, 1.f);

        kernelVec[i].resize(kernel[i].size().h * kernel[i].size().w);

        std::ranges::generate(kernelVec[i], [&udist, &rng]() { return udist(rng); });

        auto data = kernel[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(data, nvcv::NullOpt);

        // Copy kernel data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(data->plane(0).basePtr, data->plane(0).rowStride, kernelVec[i].data(), rowStride,
                                    rowStride, kernel[i].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchKernel(numImages);
    batchKernel.pushBack(kernel.begin(), kernel.end());

    // Create kernel anchor tensor

    nvcv::Tensor kernelAnchorTensor({{numImages}, "N"}, nvcv::TYPE_2S32);

    {
        auto dev = kernelAnchorTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(numImages, kernelAnchor);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Generate test result

    cvcuda::Conv2D conv2dOp;
    EXPECT_NO_THROW(conv2dOp(stream, batchSrc, batchDst, batchKernel, kernelAnchorTensor, borderMode));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowStride = srcVecRowStride[i];

        int3  shape{srcData->plane(0).width, srcData->plane(0).height, 1};
        long3 pitches{shape.y * dstRowStride, dstRowStride, 4};

        std::vector<uint8_t> testVec(shape.y * pitches.y);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        // Generate gold result
        int2 imageKernelAnchor = kernelAnchor;
        test::Convolve(goldVec, pitches, srcVec[i], pitches, shape, imageFormat, kernelVec[i], kernelSizes[i],
                       imageKernelAnchor, borderMode, borderValue);

        EXPECT_EQ(testVec, goldVec);
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpConv2DUncommonWindow,
                  test::ValueList<int, int, int, int, int, int, bool, nvcv::ImageFormat, nvcv::ImageFormat, int, int>{
    // width, height, kernelWidth, kernelHeight, anchorX, anchorY, mixedKernelSizes, planarFmt, interleavedFmt, width2, height2
    {129, 17, 3, 5, -1, -1, false, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 129, 17},
    {129, 17, 5, 3,  0,  2, true, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 129, 17},
    {131, 19, 9, 9, -1, -1, false, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 131, 19},
    {131, 19, 9, 9,  8,  0, true, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 131, 19},
    {  2,  1, 3, 5,  0,  4, false, nvcv::FMT_RGB8p, nvcv::FMT_RGB8,   2,  1},
    {  3,  2, 9, 9,  0,  8, true, nvcv::FMT_RGB8p, nvcv::FMT_RGB8,   3,  2},
    {129, 17, 3, 5, -1, -1, true, nvcv::FMT_RGB8p, nvcv::FMT_RGB8,  67,  9},
    {131, 19, 9, 9,  8,  0, true, nvcv::FMT_RGB8p, nvcv::FMT_RGB8,  33,  2},
    {129, 17, 3, 5, -1, -1, true, nvcv::FMT_U8, nvcv::FMT_U8,  67,  9},
    {131, 19, 9, 9, -1, -1, true, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8,  33,  2},
    {129, 17, 3, 5, -1, -1, true, nvcv::FMT_F32, nvcv::FMT_F32,  67,  9},
    {129, 17, 3, 5, -1, -1, true, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32,  67,  9},
    {131, 19, 9, 9, -1, -1, true, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32,  33,  2},
    {129, 17, 3, 5, -1, -1, true, nvcv::FMT_U16, nvcv::FMT_U16,  67,  9},
    {129, 17, 3, 5, -1, -1, true, nvcv::FMT_RGBf16p.dataKind(nvcv::DataKind::UNSIGNED), nvcv::FMT_RGBf16.dataKind(nvcv::DataKind::UNSIGNED),  67,  9},
    {131, 19, 9, 9, -1, -1, true, nvcv::FMT_RGBAf16p.dataKind(nvcv::DataKind::UNSIGNED), nvcv::FMT_RGBAf16.dataKind(nvcv::DataKind::UNSIGNED),  33,  2},
    {129, 17, 3, 5, -1, -1, true, nvcv::FMT_S16, nvcv::FMT_S16,  67,  9},
    {129, 17, 3, 5, -1, -1, true, nvcv::FMT_RGBf16p.dataKind(nvcv::DataKind::SIGNED), nvcv::FMT_RGBf16.dataKind(nvcv::DataKind::SIGNED),  67,  9},
    {131, 19, 9, 9, -1, -1, true, nvcv::FMT_RGBAf16p.dataKind(nvcv::DataKind::SIGNED), nvcv::FMT_RGBAf16.dataKind(nvcv::DataKind::SIGNED),  33,  2},
    {129, 17, 3, 5, -1, -1, true, nvcv::FMT_S32, nvcv::FMT_S32,  67,  9},
    {129, 17, 3, 5, -1, -1, true, nvcv::FMT_RGBf32p.dataKind(nvcv::DataKind::SIGNED), nvcv::FMT_RGBf32.dataKind(nvcv::DataKind::SIGNED),  67,  9},
    {131, 19, 9, 9, -1, -1, true, nvcv::FMT_RGBAf32p.dataKind(nvcv::DataKind::SIGNED), nvcv::FMT_RGBAf32.dataKind(nvcv::DataKind::SIGNED),  33,  2},
});

// clang-format on

TEST_P(OpConv2DUncommonWindow, varshape_correct_output_and_matches_interleaved)
{
    const int          width  = GetParamValue<0>();
    const int          height = GetParamValue<1>();
    const nvcv::Size2D kernelSize{GetParamValue<2>(), GetParamValue<3>()};
    const int2         kernelAnchor{GetParamValue<4>(), GetParamValue<5>()};
    constexpr int      numImages = 2;

    const auto planarFmt      = GetParamValue<7>();
    const auto interleavedFmt = GetParamValue<8>();
    const int  elemSize       = planarFmt.planePixelStrideBytes(0);
    const auto dtype          = planarFmt.planeDataType(0);

    const std::vector<nvcv::Size2D> sizes{
        {             width,              height},
        {GetParamValue<9>(), GetParamValue<10>()}
    };

    std::vector<nvcv::Size2D> kernelSizes(numImages, kernelSize);
    if (GetParamValue<6>())
        kernelSizes.back() = nvcv::Size2D{3, 3};
    const std::vector<int2> anchors{
        kernelAnchor, int2{2, 0}
    };
    auto anchorTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, kernelAnchor);
    test::planar::UploadTensorValues(anchorTensor, anchors);

    std::vector<std::vector<float>> hostKernels;
    auto                            kernelBatch = MakeConv2DKernelBatch(kernelSizes, &hostKernels);
    cvcuda::Conv2D                  op;

    for (NVCVBorderType border :
         {NVCV_BORDER_CONSTANT, NVCV_BORDER_REPLICATE, NVCV_BORDER_REFLECT, NVCV_BORDER_WRAP, NVCV_BORDER_REFLECT101})
    {
        SCOPED_TRACE(border);
        std::vector<std::vector<uint8_t>> gold(numImages);
        size_t                            comparedImages = 0;
        auto makeInput = [&elemSize, &dtype, &gold, &anchors, &interleavedFmt, &hostKernels, &kernelSizes, &border](
                             int sample, int w, int h, int channels)
        {
            std::vector<uint8_t> input(static_cast<size_t>(w) * h * channels * elemSize);
            const size_t         seed = static_cast<size_t>(sample) * 101 + 7;
            if (dtype == nvcv::TYPE_F32)
            {
                for (size_t k = 0; k < input.size() / sizeof(float); ++k)
                {
                    const float value = static_cast<float>(static_cast<int>((k * 31 + seed) % 251) - 125) * 0.03125f;
                    std::memcpy(input.data() + k * sizeof(float), &value, sizeof(value));
                }
            }
            else
            {
                test::planar::FillDeterministicValues(input, seed, dtype);
            }
            gold[sample].resize(input.size());
            const long3 pitches{static_cast<long>(input.size()), w * channels * elemSize, channels * elemSize};
            int2        anchor = anchors[sample];
            // Bounded integer/binary-fraction inputs and coefficients keep all 81-tap
            // sums exactly representable in FP32, including with device FMA contraction.
            test::Convolve(gold[sample], pitches, input, pitches, int3{w, h, 1}, interleavedFmt, hostKernels[sample],
                           kernelSizes[sample], anchor, border, float4{0, 0, 0, 0});
            return input;
        };
        auto compare
            = [&comparedImages, &gold](const std::vector<uint8_t> &interleaved, const std::vector<uint8_t> &planar)
        {
            ASSERT_LT(comparedImages, gold.size());
            EXPECT_EQ(interleaved, gold[comparedImages]);
            EXPECT_EQ(planar, gold[comparedImages]);
            ++comparedImages;
        };
        test::planar::RunVarShapeParity(
            planarFmt, interleavedFmt, sizes, sizes,
            [&op, &kernelBatch, &anchorTensor, &border](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                                        const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
            { EXPECT_NO_THROW(op(stream, src, dst, kernelBatch, anchorTensor, border)); },
            makeInput, compare);
        EXPECT_EQ(comparedImages, gold.size());
    }
}

// =============================================================================
// F16 (half) support (var-shape only, like every Conv2D path)
//
// F16 rides the same float-accumulating kernels as every other dtype — the kernel batch itself
// stays F32 (MakeConv2DKernelBatch and the public contract are unchanged) — so the gold runs the
// FP32 CPU reference (test::Convolve with an F32 format) on the widened half input. Bit-exactness
// against that gold is not applicable: the reference keeps the accumulator unrounded while the
// kernel narrows it once to half at the store. Per the HalfTestUtils.hpp policy the bound is
// kUlps = 4 half-ULPs (n-tap convolution with non-negative random weights: one store rounding
// plus device FMA-contraction drift across the taps).
// =============================================================================

// clang-format off

NVCV_TEST_SUITE_P(OpConv2DF16, test::ValueList<int, int, int, NVCVImageFormat, int, int, int, int, NVCVBorderType>
{
    // width, height, numImages,                    format, kernelW, kernelH, anchorX, anchorY,           borderMode
    {    123,    144,         2,  NVCV_IMAGE_FORMAT_RGBf16,       5,       5,      -1,      -1,   NVCV_BORDER_CONSTANT},
    {     66,     99,         3, NVCV_IMAGE_FORMAT_RGBAf16,       7,       7,       5,       5,  NVCV_BORDER_REPLICATE},
    {     33,     28,         4,     NVCV_IMAGE_FORMAT_F16,       3,       3,       1,       1,    NVCV_BORDER_REFLECT},
    {     64,     48,         2,  NVCV_IMAGE_FORMAT_RGBf16,       3,       3,      -1,      -1,       NVCV_BORDER_WRAP},
    {     44,     55,         3,     NVCV_IMAGE_FORMAT_F16,       5,       5,      -1,      -1, NVCV_BORDER_REFLECT101},
    {    129,     17,         2,     NVCV_IMAGE_FORMAT_F16,       3,       5,      -1,      -1,   NVCV_BORDER_CONSTANT},
    {    129,     17,         2,  NVCV_IMAGE_FORMAT_RGBf16,       3,       5,       0,       4,  NVCV_BORDER_REPLICATE},
    {    131,     19,         2, NVCV_IMAGE_FORMAT_RGBAf16,       9,       9,       8,       0, NVCV_BORDER_REFLECT101},
});

// clang-format on

TEST_P(OpConv2DF16, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width     = GetParamValue<0>();
    int height    = GetParamValue<1>();
    int numImages = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int kernelWidth   = GetParamValue<4>();
    int kernelHeight  = GetParamValue<5>();
    int kernelAnchorX = GetParamValue<6>();
    int kernelAnchorY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    nvcv::ImageFormat kernelFormat = nvcv::FMT_F32; // the kernel batch stays F32 (see suite comment)

    nvcv::Size2D kernelSize{kernelWidth, kernelHeight};
    int2         kernelAnchor{kernelAnchorX, kernelAnchorY};

    float4 borderValue = cuda::SetAll<float4>(0);

    // Create input varshape with [0, 1] values already quantized to half, so the FP32 gold
    // consumes exactly the values the kernel reads.
    std::default_random_engine rng; // NOSONAR: deterministic test data, not security-sensitive.

    std::uniform_int_distribution         udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution         udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));
    std::uniform_real_distribution<float> udist(0.f, 1.f);

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    std::vector<int>                  srcVecRowStride(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format);

        int srcRowStride   = imgSrc[i].size().w * format.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        test::FillRandomHalfBytes(srcVec[i], rng, 0.f, 1.f);

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create F32 kernel varshape with random non-negative weights (like the F32 test above)
    std::vector<nvcv::Image>        kernel;
    std::vector<std::vector<float>> kernelVec(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        kernel.emplace_back(kernelSize, kernelFormat);

        int rowStride = kernel[i].size().w * sizeof(float);

        kernelVec[i].resize(kernel[i].size().h * kernel[i].size().w);

        std::ranges::generate(kernelVec[i], [&udist, &rng]() { return udist(rng); });

        auto data = kernel[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(data, nvcv::NullOpt);

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(data->plane(0).basePtr, data->plane(0).rowStride, kernelVec[i].data(), rowStride,
                                    rowStride, kernel[i].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchKernel(numImages);
    batchKernel.pushBack(kernel.begin(), kernel.end());

    auto kernelAnchorTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, kernelAnchor);

    cvcuda::Conv2D conv2dOp;
    EXPECT_NO_THROW(conv2dOp(stream, batchSrc, batchDst, batchKernel, kernelAnchorTensor, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowStride = srcVecRowStride[i];

        int3  shape{srcData->plane(0).width, srcData->plane(0).height, 1};
        long3 pitches{shape.y * dstRowStride, dstRowStride, format.planePixelStrideBytes(0)};

        std::vector<uint8_t> testVec(shape.y * pitches.y);

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        // FP32 gold: widen the half input into a float buffer with doubled pitches (__half is 2
        // bytes, float is 4) and run the shared F32 CPU reference on it.
        std::vector<float>   srcFloats = test::HalfBytesToFloat(srcVec[i]);
        std::vector<uint8_t> srcVecF(srcVec[i].size() * 2);
        std::memcpy(srcVecF.data(), srcFloats.data(), srcVecF.size());

        long3 pitchesF{pitches.x * 2, pitches.y * 2, pitches.z * 2};

        std::vector<uint8_t> goldVecF(shape.y * pitchesF.y);

        int2 imageKernelAnchor = kernelAnchor;
        test::Convolve(goldVecF, pitchesF, srcVecF, pitchesF, shape, test::EquivalentFloatFormat(format), kernelVec[i],
                       kernelSize, imageKernelAnchor, borderMode, borderValue);

        // 4 half-ULPs: n-tap FP32-accumulated convolution with one half store rounding (see suite
        // comment). Both buffers are packed (no row padding), so they compare whole.
        std::vector<float> goldFloats(goldVecF.size() / sizeof(float));
        std::memcpy(goldFloats.data(), goldVecF.data(), goldVecF.size());
        test::ExpectNearHalfUlps(goldFloats, test::HalfBytesToFloat(testVec), 4.f);
    }
}

static void RunConv2DPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int width,
                                              int height, nvcv::Size2D kernelSize, int2 kernelAnchor,
                                              NVCVBorderType borderMode, int numImages, bool mixedKernelSizes)
{
    std::vector<nvcv::Size2D> kernelSizes(numImages, kernelSize);
    if (mixedKernelSizes)
        kernelSizes.back() = nvcv::Size2D{3, 3};
    auto kernelBatch = MakeConv2DKernelBatch(kernelSizes);
    auto kernelAnchorTensor
        = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{kernelAnchor.x, kernelAnchor.y});

    cvcuda::Conv2D conv2dOp;

    test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, width, height, width, height, numImages,
        [&conv2dOp, &kernelBatch, &kernelAnchorTensor, borderMode](
            cudaStream_t stream, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst,
            nvcv::ImageFormat)
        { EXPECT_NO_THROW(conv2dOp(stream, src, dst, kernelBatch, kernelAnchorTensor, borderMode)); });
}

// Parameters: width, height, kernelWidth, kernelHeight, anchorX, anchorY, borderMode, numImages,
//             planarFmt, interleavedFmt, mixedKernelSizes
// clang-format off
NVCV_TEST_SUITE_P(OpConv2DPlanar,
                  test::ValueList<int, int, int, int, int, int, NVCVBorderType, int, nvcv::ImageFormat,
                                  nvcv::ImageFormat, bool>{
    { 32, 24, 3, 3, -1, -1,   NVCV_BORDER_CONSTANT, 2,  nvcv::FMT_RGB8p,    nvcv::FMT_RGB8, false},
    { 29, 27, 5, 3,  2,  1,  NVCV_BORDER_REPLICATE, 1, nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8, false},
    { 17, 15, 3, 5, -1, -1, NVCV_BORDER_REFLECT101, 2, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, false},
    { 65, 37, 7, 7, -1, -1,   NVCV_BORDER_CONSTANT, 2,  nvcv::FMT_RGB8p,    nvcv::FMT_RGB8, true},
    // F16 parity is bit-exact like the other dtypes: both layouts run the same float-accumulating
    // taps in the same order, so the single half store rounding matches.
    { 32, 24, 3, 3, -1, -1,   NVCV_BORDER_CONSTANT, 2, nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16, false},
    {129, 17, 3, 5, -1, -1,  NVCV_BORDER_REPLICATE, 2, nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16, true},
    {131, 19, 9, 9, -1, -1, NVCV_BORDER_REFLECT101, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16, true},
});

// clang-format on

TEST_P(OpConv2DPlanar, varshape_matches_interleaved)
{
    RunConv2DPlanarParityVarShapeCase(GetParamValue<8>(), GetParamValue<9>(), GetParamValue<0>(), GetParamValue<1>(),
                                      nvcv::Size2D{GetParamValue<2>(), GetParamValue<3>()},
                                      int2{GetParamValue<4>(), GetParamValue<5>()}, GetParamValue<6>(),
                                      GetParamValue<7>(), GetParamValue<10>());
}

// Planar var-shape tiling flattens image and plane into grid.z. Keep the image count below the
// public grid-z guard while making the tiled launch one past the CUDA limit so the untiled fallback
// is selected.
TEST(OpConv2DPlanar, varshape_planar_grid_z_fallback)
{
    constexpr int    kNumImages    = 16384;
    constexpr int    kChannels     = 4;
    constexpr size_t kImageBytes   = kChannels;
    constexpr size_t kBufferBytes  = static_cast<size_t>(kNumImages) * kImageBytes;
    constexpr int    kKernelWidth  = 3;
    constexpr int    kKernelHeight = 3;

    uint8_t *srcAllocation{};
    uint8_t *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), kBufferBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), kBufferBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(srcAllocation, 0, kBufferBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstAllocation, 0xA5, kBufferBytes));

    auto wrapImage = [=](uint8_t *base, int imageIndex)
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes = kChannels;
        for (int plane = 0; plane < kChannels; ++plane)
        {
            buffer.planes[plane].width     = 1;
            buffer.planes[plane].height    = 1;
            buffer.planes[plane].rowStride = 1;
            buffer.planes[plane].basePtr   = reinterpret_cast<NVCVByte *>(base + imageIndex * kImageBytes + plane);
        }
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_RGBA8p, buffer});
    };

    std::vector<nvcv::Image> srcImages;
    std::vector<nvcv::Image> dstImages;
    srcImages.reserve(kNumImages);
    dstImages.reserve(kNumImages);
    for (int imageIndex = 0; imageIndex < kNumImages; ++imageIndex)
    {
        srcImages.emplace_back(wrapImage(srcAllocation, imageIndex));
        dstImages.emplace_back(wrapImage(dstAllocation, imageIndex));
    }

    nvcv::ImageBatchVarShape batchSrc(kNumImages);
    nvcv::ImageBatchVarShape batchDst(kNumImages);
    batchSrc.pushBack(srcImages.begin(), srcImages.end());
    batchDst.pushBack(dstImages.begin(), dstImages.end());

    nvcv::Image kernelImage({kKernelWidth, kKernelHeight}, nvcv::FMT_F32);
    auto        kernelData = kernelImage.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(kernelData, nvcv::NullOpt);
    ASSERT_EQ(cudaSuccess, cudaMemset2D(kernelData->plane(0).basePtr, kernelData->plane(0).rowStride, 0,
                                        kernelData->plane(0).rowStride, kernelData->plane(0).height));

    nvcv::ImageBatchVarShape batchKernel(kNumImages);
    for (int imageIndex = 0; imageIndex < kNumImages; ++imageIndex)
    {
        batchKernel.pushBack(kernelImage);
    }

    auto kernelAnchor = test::planar::MakePerImageTensor(kNumImages, nvcv::TYPE_2S32, int2{-1, -1});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Conv2D conv2dOp;
    EXPECT_NO_THROW(conv2dOp(stream, batchSrc, batchDst, batchKernel, kernelAnchor, NVCV_BORDER_CONSTANT));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> dstHost(kBufferBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstHost.data(), dstAllocation, kBufferBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(dstHost, std::vector<uint8_t>(kBufferBytes, 0));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

NVCV_TEST_SUITE_P(OpConv2D_Negative, test::PlanarConvolutionNegativeParams());

TEST_P(OpConv2D_Negative, varshape_op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width        = 32;
    int height       = 32;
    int numImages    = 2;
    int kernelWidth  = 3;
    int kernelHeight = 3;

    nvcv::ImageFormat inputFmt   = GetParamValue<0>();
    nvcv::ImageFormat outputFmt  = GetParamValue<1>();
    NVCVBorderType    borderMode = GetParamValue<2>();

    nvcv::ImageFormat kernelFormat = nvcv::FMT_F32;

    nvcv::Size2D kernelSize{kernelWidth, kernelHeight};

    // Create input varshape

    std::default_random_engine rng;

    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inputFmt);
        imgDst.emplace_back(imgSrc[i].size(), outputFmt);
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create kernel varshape

    std::vector<nvcv::Image> kernel;

    for (int i = 0; i < numImages; ++i)
    {
        kernel.emplace_back(kernelSize, kernelFormat);
    }

    nvcv::ImageBatchVarShape batchKernel(numImages);
    batchKernel.pushBack(kernel.begin(), kernel.end());

    // Create kernel anchor tensor

    nvcv::Tensor kernelAnchorTensor({{numImages}, "N"}, nvcv::TYPE_2S32);

    // Generate test result

    cvcuda::Conv2D conv2dOp;
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&conv2dOp, &stream, &batchSrc, &batchDst, &batchKernel, &kernelAnchorTensor, &borderMode]
                          { conv2dOp(stream, batchSrc, batchDst, batchKernel, kernelAnchorTensor, borderMode); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpConv2D_Negative, varshape_rejects_mismatched_image_counts_and_kernel_format)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Image srcImage({8, 8}, nvcv::FMT_RGB8);
    nvcv::Image dstImage({8, 8}, nvcv::FMT_RGB8);

    nvcv::ImageBatchVarShape batchSrc(1);
    batchSrc.pushBack(srcImage);
    nvcv::ImageBatchVarShape batchDst(1);
    batchDst.pushBack(dstImage);

    nvcv::ImageBatchVarShape mismatchedKernel(2);
    nvcv::Image              validKernel({3, 3}, nvcv::FMT_F32);
    mismatchedKernel.pushBack(validKernel);
    mismatchedKernel.pushBack(validKernel);

    nvcv::Tensor   kernelAnchor({{1}, "N"}, nvcv::TYPE_2S32);
    cvcuda::Conv2D conv2dOp;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&] { conv2dOp(stream, batchSrc, batchDst, mismatchedKernel, kernelAnchor, NVCV_BORDER_CONSTANT); }));

    nvcv::ImageBatchVarShape invalidKernel(1);
    nvcv::Image              nonF32Kernel({3, 3}, nvcv::FMT_U8);
    invalidKernel.pushBack(nonF32Kernel);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&] { conv2dOp(stream, batchSrc, batchDst, invalidKernel, kernelAnchor, NVCV_BORDER_CONSTANT); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpConv2D_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_U8,          fmt},
        {         fmt, nvcv::FMT_U8}
    };

    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
    {
        int width        = 32;
        int height       = 32;
        int numImages    = 2;
        int kernelWidth  = 3;
        int kernelHeight = 3;

        NVCVBorderType borderMode = NVCV_BORDER_CONSTANT;

        nvcv::ImageFormat kernelFormat = nvcv::FMT_F32;

        nvcv::Size2D kernelSize{kernelWidth, kernelHeight};

        // Create input varshape

        std::default_random_engine rng;

        std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
        std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;
        for (int i = 0; i < numImages - 1; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);
            imgDst.emplace_back(imgSrc[i].size(), fmt);
        }
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(numImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        nvcv::ImageBatchVarShape batchDst(numImages);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Create kernel varshape

        std::vector<nvcv::Image> kernel;

        for (int i = 0; i < numImages; ++i)
        {
            kernel.emplace_back(kernelSize, kernelFormat);
        }

        nvcv::ImageBatchVarShape batchKernel(numImages);
        batchKernel.pushBack(kernel.begin(), kernel.end());

        // Create kernel anchor tensor

        nvcv::Tensor kernelAnchorTensor({{numImages}, "N"}, nvcv::TYPE_2S32);

        // Generate test result

        cvcuda::Conv2D conv2dOp;
        EXPECT_EQ(
            NVCV_ERROR_INVALID_ARGUMENT,
            nvcv::ProtectCall([&conv2dOp, &stream, &batchSrc, &batchDst, &batchKernel, &kernelAnchorTensor, &borderMode]
                              { conv2dOp(stream, batchSrc, batchDst, batchKernel, kernelAnchorTensor, borderMode); }));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Equal batch maxima do not imply that the corresponding images have equal sizes.
NVCV_TEST_SUITE_P(OpConv2D_SizeMismatch, test::ValueList<int, int, int, int, int>{
                                             {24, 20, 24, 20, 2},
                                             {24, 20, 16, 12, 2},
                                             {16, 12, 24, 20, 1}
});

TEST_P(OpConv2D_SizeMismatch, rejects_unpaired_output_images)
{
    for (auto format : {nvcv::FMT_U8, nvcv::FMT_RGB8p})
    {
        SCOPED_TRACE(format);
        nvcv::ImageBatchVarShape input(2);
        nvcv::ImageBatchVarShape output(2);
        input.pushBack(nvcv::Image(nvcv::Size2D{16, 12}, format));
        input.pushBack(nvcv::Image(nvcv::Size2D{24, 20}, format));
        output.pushBack(nvcv::Image(nvcv::Size2D{GetParamValue<0>(), GetParamValue<1>()}, format));
        if (GetParamValue<4>() == 2)
            output.pushBack(nvcv::Image(nvcv::Size2D{GetParamValue<2>(), GetParamValue<3>()}, format));
        auto kernels = MakeConv2DKernelBatch({
            {3, 3},
            {3, 3}
        });
        auto anchors = test::planar::MakePerImageTensor(2, nvcv::TYPE_2S32, int2{-1, -1});
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        cvcuda::Conv2D op;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { op(nullptr, input, output, kernels, anchors, NVCV_BORDER_CONSTANT); }));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    }
}

TEST(OpConv2D_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaConv2DCreate(nullptr));
}

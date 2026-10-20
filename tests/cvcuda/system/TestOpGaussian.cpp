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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cuda_fp16.h>
#include <cvcuda/OpGaussian.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp> // for round
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cstring>
#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

namespace {

// builds Gaussian and its per-image parameter tensors for a var-shape negative case
inline void InvokeGaussianVarShapeNegative(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                           const nvcv::ImageBatchVarShape &dst, int maxBatches,
                                           NVCVBorderType borderMode)
{
    const nvcv::Size2D kernelSize(3, 3);
    const int          numImages = src.numImages();
    auto               kernelSizeTensor
        = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{kernelSize.w, kernelSize.h});
    auto             sigmaTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2F64, double2{0.5, 0.5});
    cvcuda::Gaussian op(kernelSize, maxBatches);
    op(stream, src, dst, kernelSizeTensor, sigmaTensor, borderMode);
}

} // namespace

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

// clang-format off

NVCV_TEST_SUITE_P(OpGaussian, test::ValueList<int, int, int, NVCVImageFormat, int, int, double, double, NVCVBorderType>
{
    // width, height, batches,                    format, ksizeX, ksizeY, sigmaX, sigmaY,           borderMode
    {    176,    113,       1,      NVCV_IMAGE_FORMAT_U8,      3,      3,    0.5,    0.5, NVCV_BORDER_CONSTANT},
    {    123,     66,       2,      NVCV_IMAGE_FORMAT_U8,      5,      5,   0.75,   0.75, NVCV_BORDER_CONSTANT},
    {     65,     17,       2,    NVCV_IMAGE_FORMAT_RGB8,      7,      7,    0.8,    0.8, NVCV_BORDER_CONSTANT},
    {    123,     33,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,    1.0,    1.0, NVCV_BORDER_WRAP},
    {    111,     33,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,    1.0,   -1.0, NVCV_BORDER_WRAP},
    {     42,     53,       4,   NVCV_IMAGE_FORMAT_RGBA8,      7,      7,    0.4,    0.4, NVCV_BORDER_REPLICATE},
    {     13,     42,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,    0.9,    0.9, NVCV_BORDER_REFLECT},
    {     62,    111,       4,   NVCV_IMAGE_FORMAT_RGBA8,      9,      9,    0.8,    0.8, NVCV_BORDER_REFLECT101},
    {    128,    128,       1,      NVCV_IMAGE_FORMAT_U8,     -1,     -1,    0.5,    0.5, NVCV_BORDER_CONSTANT}
});

// clang-format on

TEST_P(OpGaussian, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int    ksizeX = GetParamValue<4>();
    int    ksizeY = GetParamValue<5>();
    double sigmaX = GetParamValue<6>();
    double sigmaY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    int3 shape{width, height, batches};

    double2 sigma{sigmaX, sigmaY};
    int2    kernelAnchor{-1, -1};

    int newKsizeX = ksizeX;
    int newKsizeY = ksizeY;

    // auto detection of kernel size from sigma
    bool is8U = format.planeDataType(0) == nvcv::TYPE_U8;
    if (ksizeX <= 0 && sigmaX > 0)
    {
        newKsizeX = nvcv::cuda::round<int>(sigmaX * (is8U ? 3 : 4) * 2 + 1) | 1;
    }
    if (ksizeY <= 0 && sigmaY > 0)
    {
        newKsizeY = nvcv::cuda::round<int>(sigmaY * (is8U ? 3 : 4) * 2 + 1) | 1;
    }

    nvcv::Size2D kernelSize(ksizeX, ksizeY);
    nvcv::Size2D newKernelSize(newKsizeX, newKsizeY);

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    long3 inStrides{inAccess->sampleStride(), inAccess->rowStride(), inAccess->colStride()};
    long3 outStrides{outAccess->sampleStride(), outAccess->rowStride(), outAccess->colStride()};

    if (inData->rank() == 3)
    {
        inStrides.x  = inAccess->numRows() * inAccess->rowStride();
        outStrides.x = outAccess->numRows() * outAccess->rowStride();
    }

    long inBufSize  = inStrides.x * inAccess->numSamples();
    long outBufSize = outStrides.x * outAccess->numSamples();

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine    randEng(0); // NOSONAR: deterministic test data, not security-sensitive.
    std::uniform_int_distribution rand(0u, 255u);

    std::ranges::generate(inVec, [&rand, &randEng]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::Gaussian gaussianOp(newKernelSize, 1);

    EXPECT_NO_THROW(gaussianOp(stream, inTensor, outTensor, kernelSize, sigma, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    std::vector<float> kernel = test::ComputeGaussianKernel(newKernelSize, sigma);

    test::Convolve(goldVec, outStrides, inVec, inStrides, shape, format, kernel, newKernelSize, kernelAnchor,
                   borderMode, borderValue);

    EXPECT_EQ(testVec, goldVec);
}

NVCV_TEST_SUITE_P(OpGaussianF32Runtime, test::ValueList<NVCVImageFormat, int, int>{
                                            {    NVCV_IMAGE_FORMAT_F32, 21, 21},
                                            { NVCV_IMAGE_FORMAT_RGBf32, 11,  7},
                                            {NVCV_IMAGE_FORMAT_RGBAf32, 17, 17}
});

TEST_P(OpGaussianF32Runtime, matches_independent_cpu_reference)
{
    const nvcv::ImageFormat format{GetParamValue<0>()};
    const nvcv::Size2D      window{GetParamValue<1>(), GetParamValue<2>()};
    const double2           sigma{1.7, 1.2};
    constexpr int           width  = 67;
    constexpr int           height = 35;
    auto                    input  = nvcv::util::CreateTensor(1, width, height, format);
    auto                    output = nvcv::util::CreateTensor(1, width, height, format);
    nvcv::util::SetTensorToRandomValue<float>(input.exportData(), 0.f, 1.f);
    nvcv::util::TensorImageData source(input.exportData());

    cvcuda::Gaussian op(window, 1);
    op(nullptr, input, output, window, sigma, NVCV_BORDER_REFLECT101);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    nvcv::util::TensorImageData actual(output.exportData());
    const long3          srcStrides{height * source.rowStride(), source.rowStride(), format.planePixelStrideBytes(0)};
    const long3          dstStrides{height * actual.rowStride(), actual.rowStride(), format.planePixelStrideBytes(0)};
    std::vector<uint8_t> gold(actual.getVector().size());
    int2                 anchor{-1, -1};
    test::Convolve(gold, dstStrides, source.getVector(), srcStrides, int3{width, height, 1}, format,
                   test::ComputeGaussianKernel(window, sigma), window, anchor, NVCV_BORDER_REFLECT101,
                   float4{0, 0, 0, 0});
    // Independent host/device coefficient generation and FMA can differ; inputs/outputs lie in [0,1].
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < format.numChannels(); ++c)
            {
                float expected;
                std::memcpy(&expected, gold.data() + y * dstStrides.y + x * dstStrides.z + c * sizeof(float),
                            sizeof(float));
                ASSERT_NEAR(*actual.item<float>(x, y, c), expected, 1e-5f) << "at " << x << "," << y << "," << c;
            }
}

// The coefficient window is generated once per (kernelSize, sigma) and reused by later submits on
// the same operator. Every other test here submits once against a fresh operator, so nothing else
// exercises reuse at all -- and reuse is exactly what can serve a stale window. The first test below
// pins that: it fails if the resident window is never evicted. The second covers the same path
// across two streams; see the note on its limits.
namespace {

// One 3-channel u8 image, a window small enough to keep the test quick, and the gold convolution
// the operator has to reproduce on every submit.
struct GaussianReuseFixture
{
    static constexpr int    kWidth  = 129;
    static constexpr int    kHeight = 67;
    static constexpr double kSigma  = 1.5;

    nvcv::ImageFormat format    = nvcv::FMT_RGB8;
    nvcv::Size2D      kernelSze = {7, 7};
    double2           sigma{kSigma, kSigma};
    NVCVBorderType    border = NVCV_BORDER_REPLICATE;

    nvcv::Tensor         in  = nvcv::util::CreateTensor(1, kWidth, kHeight, nvcv::FMT_RGB8);
    nvcv::Tensor         out = nvcv::util::CreateTensor(1, kWidth, kHeight, nvcv::FMT_RGB8);
    std::vector<uint8_t> inVec;
    std::vector<uint8_t> goldVec;
    long3                strides{};
    long                 bufSize = 0;

    void Init()
    {
        auto inData  = in.exportData<nvcv::TensorDataStridedCuda>();
        auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(inData, nullptr);
        ASSERT_NE(outData, nullptr);

        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
        ASSERT_TRUE(inAccess);

        strides = long3{inAccess->sampleStride(), inAccess->rowStride(), inAccess->colStride()};
        // A single-sample tensor comes back rank 3, where the sample stride is not the buffer
        // pitch; the parameterized tests above make the same adjustment.
        if (inData->rank() == 3)
        {
            strides.x = inAccess->numRows() * inAccess->rowStride();
        }
        bufSize = strides.x * inAccess->numSamples();

        inVec.resize(bufSize);
        std::default_random_engine    randEng(0); // NOSONAR: deterministic test data.
        std::uniform_int_distribution rand(0u, 255u);
        std::ranges::generate(inVec, [&rand, &randEng]() { return rand(randEng); });
        ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), bufSize, cudaMemcpyHostToDevice));

        goldVec.resize(bufSize);
        std::vector<float> kernel = test::ComputeGaussianKernel(kernelSze, sigma);
        int2               anchor{-1, -1}; // Convolve normalizes the anchor in place, so it needs an lvalue.
        test::Convolve(goldVec, strides, inVec, strides, int3{kWidth, kHeight, 1}, format, kernel, kernelSze, anchor,
                       border, float4{0.f, 0.f, 0.f, 0.f});
    }

    void ExpectOutputMatchesGold(const char *what) const
    {
        ExpectMatchesGold(out, goldVec, what);
    }

    // The reference for an arbitrary sigma, so a submit that evicts the resident window can be
    // checked against its own parameters rather than only against the default ones.
    std::vector<uint8_t> GoldFor(double2 otherSigma) const
    {
        std::vector<uint8_t> gold(bufSize);
        std::vector<float>   kernel = test::ComputeGaussianKernel(kernelSze, otherSigma);
        int2                 anchor{-1, -1};
        test::Convolve(gold, strides, inVec, strides, int3{kWidth, kHeight, 1}, format, kernel, kernelSze, anchor,
                       border, float4{0.f, 0.f, 0.f, 0.f});
        return gold;
    }

    void ExpectMatchesGold(const nvcv::Tensor &tensor, const std::vector<uint8_t> &gold, const char *what) const
    {
        auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(data, nullptr);
        auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
        ASSERT_TRUE(access);
        constexpr int        rowBytes = kWidth * 3;
        std::vector<uint8_t> testVec(kHeight * rowBytes);
        // Row padding is not output image data and need not be initialized by the operator.
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowBytes, data->basePtr(), access->rowStride(), rowBytes,
                                            kHeight, cudaMemcpyDeviceToHost));
        for (int y = 0; y < kHeight; ++y)
        {
            for (int x = 0; x < rowBytes; ++x)
            {
                // CPU/GPU float accumulation can straddle a U8 rounding boundary.
                EXPECT_NEAR(testVec[y * rowBytes + x], gold[y * strides.y + x], 1)
                    << what << " at x=" << x / 3 << ", y=" << y << ", channel=" << x % 3;
            }
        }
    }
};

} // namespace

TEST(OpGaussian, reuses_the_coefficient_window_across_submits)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    GaussianReuseFixture f;
    ASSERT_NO_FATAL_FAILURE(f.Init());

    cvcuda::Gaussian op(f.kernelSze, 1);

    // First submit generates the window; the second must reuse it and still be correct.
    EXPECT_NO_THROW(op(stream, f.in, f.out, f.kernelSze, f.sigma, f.border));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_NO_FATAL_FAILURE(f.ExpectOutputMatchesGold("first submit"));

    ASSERT_EQ(cudaSuccess, cudaMemset(f.out.exportData<nvcv::TensorDataStridedCuda>()->basePtr(), 0, f.bufSize));
    EXPECT_NO_THROW(op(stream, f.in, f.out, f.kernelSze, f.sigma, f.border));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_NO_FATAL_FAILURE(f.ExpectOutputMatchesGold("reused window"));

    // A different sigma has to evict the resident window rather than silently reuse it, and the
    // original parameters have to regenerate correctly afterwards.
    const double2 otherSigma{GaussianReuseFixture::kSigma * 2, GaussianReuseFixture::kSigma * 2};
    EXPECT_NO_THROW(op(stream, f.in, f.out, f.kernelSze, otherSigma, f.border));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    // Checked against otherSigma's own reference: a cache that keyed on kernel size alone would
    // reuse the first window here and still satisfy every other assertion in this test.
    ASSERT_NO_FATAL_FAILURE(f.ExpectMatchesGold(f.out, f.GoldFor(otherSigma), "evicting submit"));

    EXPECT_NO_THROW(op(stream, f.in, f.out, f.kernelSze, f.sigma, f.border));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_NO_FATAL_FAILURE(f.ExpectOutputMatchesGold("regenerated after eviction"));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGaussian, orders_a_reusing_submit_behind_the_producing_stream)
{
    cudaStream_t producer;
    cudaStream_t consumer;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&producer));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&consumer));

    GaussianReuseFixture f;
    ASSERT_NO_FATAL_FAILURE(f.Init());

    cvcuda::Gaussian op(f.kernelSze, 1);

    // Queue both without synchronizing in between, so the consumer finds the window resident and
    // skips the generation.
    //
    // This does not detect a missing wait on its own: with the event removed it still passes, because
    // the generator is small enough to retire before the consumer's filter reaches the buffer. It is
    // a smoke test for the cross-stream reuse path -- that a second stream gets a complete, correct
    // window rather than a stale or absent one -- not a race detector. The ordering itself is
    // argued from the event, not from this test.
    // One destination per stream: sharing one would have both submits writing the same bytes
    // concurrently, so a correct result would prove nothing about the window they read.
    nvcv::Tensor consumerOut
        = nvcv::util::CreateTensor(1, GaussianReuseFixture::kWidth, GaussianReuseFixture::kHeight, nvcv::FMT_RGB8);

    EXPECT_NO_THROW(op(producer, f.in, f.out, f.kernelSze, f.sigma, f.border));
    EXPECT_NO_THROW(op(consumer, f.in, consumerOut, f.kernelSze, f.sigma, f.border));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(producer));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(consumer));
    ASSERT_NO_FATAL_FAILURE(f.ExpectOutputMatchesGold("cross-stream reuse (producer)"));
    ASSERT_NO_FATAL_FAILURE(f.ExpectMatchesGold(consumerOut, f.goldVec, "cross-stream reuse (consumer)"));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(producer));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(consumer));
}

TEST(OpGaussian, orders_an_evicting_submit_behind_every_outstanding_reader)
{
    cudaStream_t reader;
    cudaStream_t evictor;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&reader));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&evictor));

    GaussianReuseFixture f;
    ASSERT_NO_FATAL_FAILURE(f.Init());

    cvcuda::Gaussian op(f.kernelSze, 1);

    const double2 otherSigma{GaussianReuseFixture::kSigma * 2, GaussianReuseFixture::kSigma * 2};
    nvcv::Tensor  evictorOut
        = nvcv::util::CreateTensor(1, GaussianReuseFixture::kWidth, GaussianReuseFixture::kHeight, nvcv::FMT_RGB8);

    // Queue a reader, then, without synchronizing, a submit whose different sigma evicts the window
    // the reader is still consuming. The evicting regenerate has to wait for the reader's filter,
    // not merely for whichever stream produced the window.
    //
    // Like the cross-stream test above, this is not a race detector -- the kernels are small enough
    // that it can pass with the ordering removed. It exists to execute the eviction-with-outstanding
    // -reader path and to pin that both submits get the window their own parameters ask for.
    EXPECT_NO_THROW(op(reader, f.in, f.out, f.kernelSze, f.sigma, f.border));
    EXPECT_NO_THROW(op(evictor, f.in, evictorOut, f.kernelSze, otherSigma, f.border));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(reader));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(evictor));

    ASSERT_NO_FATAL_FAILURE(f.ExpectOutputMatchesGold("reader keeps its window"));
    ASSERT_NO_FATAL_FAILURE(f.ExpectMatchesGold(evictorOut, f.GoldFor(otherSigma), "evictor gets its own window"));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(reader));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(evictor));
}

TEST_P(OpGaussian, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int    ksizeX = GetParamValue<4>();
    int    ksizeY = GetParamValue<5>();
    double sigmaX = GetParamValue<6>();
    double sigmaY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    double2 sigma{sigmaX, sigmaY};
    int2    kernelAnchor{-1, -1};

    int newKsizeX = ksizeX;
    int newKsizeY = ksizeY;

    // auto detection of kernel size from sigma
    bool is8U = format.planeDataType(0) == nvcv::TYPE_U8;
    if (ksizeX <= 0 && sigmaX > 0)
    {
        newKsizeX = nvcv::cuda::round<int>(sigmaX * (is8U ? 3 : 4) * 2 + 1) | 1;
    }
    if (ksizeY <= 0 && sigmaY > 0)
    {
        newKsizeY = nvcv::cuda::round<int>(sigmaY * (is8U ? 3 : 4) * 2 + 1) | 1;
    }

    nvcv::Size2D kernelSize(newKsizeX, newKsizeY);

    // Create input varshape
    std::default_random_engine    rng; // NOSONAR: deterministic test data, not security-sensitive.
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format);

        int srcRowStride   = imgSrc[i].size().w * format.planePixelStrideBytes(0);
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

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create kernel size tensor
    nvcv::Tensor kernelSizeTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = kernelSizeTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{ksizeX, ksizeY});

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create sigma tensor
    nvcv::Tensor sigmaTensor({{batches}, "N"}, nvcv::TYPE_2F64);
    {
        auto dev = sigmaTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<double2> vec(batches, sigma);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(double2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::Gaussian gaussianOp(kernelSize, batches);

    EXPECT_NO_THROW(gaussianOp(stream, batchSrc, batchDst, kernelSizeTensor, sigmaTensor, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
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

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        // Generate gold result
        std::vector<float> kernel = test::ComputeGaussianKernel(kernelSize, sigma);

        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        test::Convolve(goldVec, pitches, srcVec[i], pitches, shape, format, kernel, kernelSize, kernelAnchor,
                       borderMode, borderValue);

        EXPECT_EQ(testVec, goldVec);
    }
}

// =============================================================================
// F16 (half) support
//
// F16 rides the same float-accumulating kernels as every other dtype, so the gold runs the FP32
// CPU reference (test::Convolve with an F32 format) on the widened half input. Bit-exactness
// against that gold is not applicable: the reference keeps the accumulator unrounded while the
// kernel narrows it once to half at the store. Per the HalfTestUtils.hpp policy the bound is
// kUlps = 4 half-ULPs (n-tap filter: one store rounding plus device FMA-contraction drift across
// the taps, which the half-ULP scale comfortably absorbs).
// =============================================================================

// clang-format off

NVCV_TEST_SUITE_P(OpGaussianF16, test::ValueList<int, int, int, NVCVImageFormat, int, int, double, double, NVCVBorderType>
{
    // width, height, batches,                    format, ksizeX, ksizeY, sigmaX, sigmaY,           borderMode
    {    176,    113,       1,     NVCV_IMAGE_FORMAT_F16,      3,      3,    0.5,    0.5, NVCV_BORDER_CONSTANT},
    {    123,     66,       2,  NVCV_IMAGE_FORMAT_RGBf16,      5,      5,   0.75,   0.75, NVCV_BORDER_REPLICATE},
    {     42,     53,       4, NVCV_IMAGE_FORMAT_RGBAf16,      7,      7,    0.4,    0.4, NVCV_BORDER_REFLECT},
    {     62,     33,       3,  NVCV_IMAGE_FORMAT_RGBf16,      3,      3,    1.0,   -1.0, NVCV_BORDER_WRAP},
    {    128,     64,       2,     NVCV_IMAGE_FORMAT_F16,     -1,     -1,    0.5,    0.5, NVCV_BORDER_REFLECT101},
});

// clang-format on

TEST_P(OpGaussianF16, tensor_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int    ksizeX = GetParamValue<4>();
    int    ksizeY = GetParamValue<5>();
    double sigmaX = GetParamValue<6>();
    double sigmaY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    int3 shape{width, height, batches};

    double2 sigma{sigmaX, sigmaY};
    int2    kernelAnchor{-1, -1};

    // auto detection of kernel size from sigma (F16 uses the non-U8 4*sigma rule)
    int newKsizeX = ksizeX;
    int newKsizeY = ksizeY;
    if (ksizeX <= 0 && sigmaX > 0)
    {
        newKsizeX = nvcv::cuda::round<int>(sigmaX * 4 * 2 + 1) | 1;
    }
    if (ksizeY <= 0 && sigmaY > 0)
    {
        newKsizeY = nvcv::cuda::round<int>(sigmaY * 4 * 2 + 1) | 1;
    }

    nvcv::Size2D kernelSize(ksizeX, ksizeY);
    nvcv::Size2D newKernelSize(newKsizeX, newKsizeY);

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    long3 inStrides{inAccess->sampleStride(), inAccess->rowStride(), inAccess->colStride()};
    long3 outStrides{outAccess->sampleStride(), outAccess->rowStride(), outAccess->colStride()};

    if (inData->rank() == 3)
    {
        inStrides.x  = inAccess->numRows() * inAccess->rowStride();
        outStrides.x = outAccess->numRows() * outAccess->rowStride();
    }

    long inBufSize  = inStrides.x * inAccess->numSamples();
    long outBufSize = outStrides.x * outAccess->numSamples();

    // [0, 1] input already quantized to half, so the FP32 gold consumes exactly the values the
    // kernel reads.
    std::vector<uint8_t>       inVec(inBufSize);
    std::default_random_engine randEng(0); // NOSONAR: deterministic test data, not security-sensitive.
    test::FillRandomHalfBytes(inVec, randEng, 0.f, 1.f);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    cvcuda::Gaussian gaussianOp(newKernelSize, batches);

    EXPECT_NO_THROW(gaussianOp(stream, inTensor, outTensor, kernelSize, sigma, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> testVec(outBufSize);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // FP32 gold: widen the half input into a float buffer with doubled strides (__half is 2
    // bytes, float is 4) and run the shared F32 CPU reference on it.
    std::vector<float>   inFloats = test::HalfBytesToFloat(inVec);
    std::vector<uint8_t> inVecF(inBufSize * 2);
    std::memcpy(inVecF.data(), inFloats.data(), inVecF.size());

    long3 inStridesF{inStrides.x * 2, inStrides.y * 2, inStrides.z * 2};
    long3 outStridesF{outStrides.x * 2, outStrides.y * 2, outStrides.z * 2};

    std::vector<float> kernel = test::ComputeGaussianKernel(newKernelSize, sigma);

    std::vector<uint8_t> goldVecF(outBufSize * 2);
    test::Convolve(goldVecF, outStridesF, inVecF, inStridesF, shape, test::EquivalentFloatFormat(format), kernel,
                   newKernelSize, kernelAnchor, borderMode, borderValue);

    // Gather per-element (skipping row padding) and compare within 4 half-ULPs (rationale above).
    const int           channels = format.numChannels();
    std::vector<float>  goldVals;
    std::vector<__half> testVals;
    for (int b = 0; b < batches; ++b)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int xc = 0; xc < width * channels; ++xc)
            {
                const int x = xc / channels;
                const int c = xc % channels;
                goldVals.push_back(*reinterpret_cast<const float *>(
                    &goldVecF[b * outStridesF.x + y * outStridesF.y + x * outStridesF.z + c * sizeof(float)]));
                testVals.push_back(*reinterpret_cast<const __half *>(
                    &testVec[b * outStrides.x + y * outStrides.y + x * outStrides.z + c * sizeof(__half)]));
            }
        }
    }
    test::ExpectNearHalfUlps(goldVals, testVals, 4.f);
}

TEST_P(OpGaussianF16, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int    ksizeX = GetParamValue<4>();
    int    ksizeY = GetParamValue<5>();
    double sigmaX = GetParamValue<6>();
    double sigmaY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    double2 sigma{sigmaX, sigmaY};
    int2    kernelAnchor{-1, -1};

    // auto detection of kernel size from sigma (F16 uses the non-U8 4*sigma rule)
    int newKsizeX = ksizeX;
    int newKsizeY = ksizeY;
    if (ksizeX <= 0 && sigmaX > 0)
    {
        newKsizeX = nvcv::cuda::round<int>(sigmaX * 4 * 2 + 1) | 1;
    }
    if (ksizeY <= 0 && sigmaY > 0)
    {
        newKsizeY = nvcv::cuda::round<int>(sigmaY * 4 * 2 + 1) | 1;
    }

    nvcv::Size2D kernelSize(newKsizeX, newKsizeY);

    // Create input varshape with [0, 1] values already quantized to half
    std::default_random_engine    rng; // NOSONAR: deterministic test data, not security-sensitive.
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    for (int i = 0; i < batches; ++i)
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

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    auto kernelSizeTensor = test::planar::MakePerImageTensor(batches, nvcv::TYPE_2S32, int2{ksizeX, ksizeY});
    auto sigmaTensor      = test::planar::MakePerImageTensor(batches, nvcv::TYPE_2F64, double2{sigma.x, sigma.y});

    cvcuda::Gaussian gaussianOp(kernelSize, batches);

    EXPECT_NO_THROW(gaussianOp(stream, batchSrc, batchDst, kernelSizeTensor, sigmaTensor, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < batches; ++i)
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

        // FP32 gold on the widened half input (doubled strides), as in the tensor test above.
        std::vector<float>   srcFloats = test::HalfBytesToFloat(srcVec[i]);
        std::vector<uint8_t> srcVecF(srcVec[i].size() * 2);
        std::memcpy(srcVecF.data(), srcFloats.data(), srcVecF.size());

        long3 pitchesF{pitches.x * 2, pitches.y * 2, pitches.z * 2};

        std::vector<float> kernel = test::ComputeGaussianKernel(kernelSize, sigma);

        std::vector<uint8_t> goldVecF(shape.y * pitchesF.y);
        test::Convolve(goldVecF, pitchesF, srcVecF, pitchesF, shape, test::EquivalentFloatFormat(format), kernel,
                       kernelSize, kernelAnchor, borderMode, borderValue);

        // 4 half-ULPs: n-tap FP32-accumulated filter with one half store rounding (see suite
        // comment). Both buffers are packed (no row padding), so they compare whole.
        std::vector<float> goldFloats(goldVecF.size() / sizeof(float));
        std::memcpy(goldFloats.data(), goldVecF.data(), goldVecF.size());
        test::ExpectNearHalfUlps(goldFloats, test::HalfBytesToFloat(testVec), 4.f);
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpGaussianF16Alignment, test::ValueList<int, int, bool>{
    // width, height, planar; cover a four-half load and both fixed output-tile widths.
    {  1,  1, false}, {  3,  2, true}, {  4,  3, false}, {  5, 17, true},
    { 31, 17, false}, { 32, 17, true}, { 33, 17, false}, { 65, 19, true},
});

// clang-format on

namespace {

template<bool VarShape>
auto WrapGaussianF16Alignment(nvcv::Byte *base, int pitch, nvcv::Size2D size, bool planar)
{
    const int planes = planar ? 3 : 1;
    if constexpr (VarShape)
    {
        nvcv::ImageBatchVarShape batch(2);
        for (int image = 0; image < 2; ++image)
        {
            nvcv::ImageDataStridedCuda::Buffer buffer{};
            buffer.numPlanes = planes;
            for (int p = 0; p < planes; ++p)
            {
                buffer.planes[p].basePtr   = reinterpret_cast<NVCVByte *>(base + (image * planes + p) * size.h * pitch);
                buffer.planes[p].width     = size.w - 2 * image;
                buffer.planes[p].height    = size.h - 2 * image;
                buffer.planes[p].rowStride = pitch;
            }
            batch.pushBack(
                nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{planar ? nvcv::FMT_RGBf16p : nvcv::FMT_F16, buffer}));
        }
        return batch;
    }
    else
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = reinterpret_cast<NVCVByte *>(base);
        buffer.strides[0] = planes * size.h * pitch;
        buffer.strides[1] = planar ? size.h * pitch : pitch;
        buffer.strides[2] = planar ? pitch : sizeof(__half);
        buffer.strides[3] = sizeof(__half);
        const nvcv::TensorShape shape = planar
            ? nvcv::TensorShape{{2, planes, size.h, size.w}, "NCHW"}
            : nvcv::TensorShape{{2, size.h, size.w, 1}, "NHWC"};
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_F16, buffer});
    }
}

void ExpectGaussianF16AlignmentValues(uint8_t *output, long3 outputStrides, const uint8_t *gold, long3 goldStrides,
                                      int3 shape)
{
    std::vector<float> values;
    std::vector<float> expected;
    for (int pixel = 0; pixel < shape.x * shape.y * shape.z; ++pixel)
    {
        const int     image  = pixel / (shape.x * shape.y);
        const int     y      = pixel / shape.x % shape.y;
        const int     x      = pixel % shape.x;
        const int64_t offset = image * outputStrides.x + y * outputStrides.y + x * static_cast<int64_t>(sizeof(__half));
        __half        value;
        float         reference;
        std::memcpy(&value, output + offset, sizeof(value));
        std::memcpy(&reference, gold + image * goldStrides.x + y * goldStrides.y + x * sizeof(float),
                    sizeof(reference));
        values.push_back(__half2float(value));
        expected.push_back(reference);
        std::memset(output + offset, 0xA5, sizeof(value));
    }
    test::ExpectNearHalfUlps(expected, values, 4.f);
}

template<bool VarShape>
void CheckGaussianF16AlignmentCase(cvcuda::Gaussian &op, int width, int height, bool planar, int residue,
                                   int kernelWidth, NVCVBorderType border)
{
    SCOPED_TRACE(residue);
    SCOPED_TRACE(kernelWidth);
    SCOPED_TRACE(border);
    const int                         planes  = planar ? 3 : 1;
    const int                         batches = 2;
    const int                         images  = batches * planes;
    const std::array<nvcv::Size2D, 2> sizes   = {
          {{width, height}, {VarShape ? width - 2 : width, VarShape ? height - 2 : height}}
    };
    const long3 goldStrides{height * width * 4L, width * 4L, 4};

    // Include both signs, signed zero, subnormals and finite extremes. Each plane has one sign,
    // avoiding cancellation near zero in the existing four-half-ULP FP32-reference policy.
    const std::array<uint16_t, 7> patterns = {0x0000, 0x0001, 0x03ff, 0x0400, 0x3555, 0x3c00, 0x7bff};
    std::vector<uint8_t>          inputGold(images * goldStrides.x);
    std::vector<uint16_t>         inputBits(images * width * height);
    for (int index = 0; index < images * width * height; ++index)
    {
        const int      image = index / (width * height);
        const int      pixel = index % (width * height);
        const uint16_t bits  = patterns[(pixel * 3 + image) % patterns.size()] | ((image % 2) ? 0x8000 : 0);
        const float    wide  = __half2float(__ushort_as_half(bits));
        inputBits[index]     = bits;
        std::memcpy(inputGold.data() + index * sizeof(float), &wide, sizeof(wide));
    }

    const int inOffset   = 2 * residue;
    const int outOffset  = 2 * ((residue + 1) % 4);
    const int inPitch    = 2 * (width + 1 + residue);
    const int outPitch   = 2 * (width + 3 + residue);
    const int inBytes    = inOffset + images * height * inPitch + 16;
    const int outBytes   = outOffset + images * height * outPitch + 16;
    auto      inStorage  = nvcv::util::CreateTensor(1, inBytes, 1, nvcv::FMT_U8);
    auto      outStorage = nvcv::util::CreateTensor(1, outBytes, 1, nvcv::FMT_U8);
    auto      inData     = inStorage.exportData<nvcv::TensorDataStridedCuda>();
    auto      outData    = outStorage.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);
    auto src = WrapGaussianF16Alignment<VarShape>(inData->basePtr() + inOffset, inPitch, sizes[0], planar);
    auto dst = WrapGaussianF16Alignment<VarShape>(outData->basePtr() + outOffset, outPitch, sizes[0], planar);
    std::vector<uint8_t> input(inBytes, 0xC3);
    for (int row = 0; row < images * height; ++row)
    {
        std::memcpy(input.data() + inOffset + row * inPitch, inputBits.data() + row * width, width * sizeof(__half));
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), input.data(), inBytes, cudaMemcpyHostToDevice));

    const nvcv::Size2D      kernelSize{kernelWidth, kernelWidth};
    const double2           sigma{0.5, 0.5};
    const auto              kernel = test::ComputeGaussianKernel(kernelSize, sigma);
    const std::vector<int2> windows{
        {    kernelWidth, kernelWidth},
        {kernelWidth - 2, kernelWidth}
    };
    const std::vector<double2> sigmas{
        sigma,
        {0.9, 0.7}
    };
    nvcv::Tensor windowTensor;
    nvcv::Tensor sigmaTensor;
    if constexpr (VarShape)
    {
        windowTensor = nvcv::Tensor({{batches}, "N"}, nvcv::TYPE_2S32);
        sigmaTensor  = nvcv::Tensor({{batches}, "N"}, nvcv::TYPE_2F64);
        test::planar::UploadTensorValues(windowTensor, windows);
        test::planar::UploadTensorValues(sigmaTensor, sigmas);
    }
    ASSERT_EQ(cudaSuccess, cudaMemset(outData->basePtr(), 0xA5, outBytes));
    if constexpr (VarShape)
        ASSERT_NO_THROW(op(nullptr, src, dst, windowTensor, sigmaTensor, border));
    else
        ASSERT_NO_THROW(op(nullptr, src, dst, kernelSize, sigma, border));
    std::vector<uint8_t> output(outBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), outData->basePtr(), outBytes, cudaMemcpyDeviceToHost));
    for (int batch = 0; batch < (VarShape ? batches : 1); ++batch)
    {
        const int                  count = VarShape ? planes : images;
        const auto                 first = inputGold.begin() + batch * planes * goldStrides.x;
        const std::vector<uint8_t> batchInput(first, first + count * goldStrides.x);
        std::vector<uint8_t>       gold(batchInput.size());
        const nvcv::Size2D         window  = VarShape ? nvcv::Size2D{windows[batch].x, windows[batch].y} : kernelSize;
        const auto                 weights = VarShape ? test::ComputeGaussianKernel(window, sigmas[batch]) : kernel;
        int2                       anchor{-1, -1};
        const int3                 shape{sizes[batch].w, sizes[batch].h, count};
        test::Convolve(gold, goldStrides, batchInput, goldStrides, shape, nvcv::FMT_F32, weights, window, anchor,
                       border, float4{0, 0, 0, 0});
        ExpectGaussianF16AlignmentValues(output.data() + outOffset + batch * planes * height * outPitch,
                                         long3{height * outPitch, outPitch, 2}, gold.data(), goldStrides, shape);
    }
    EXPECT_EQ(output, std::vector<uint8_t>(outBytes, 0xA5));
    std::vector<uint8_t> sourceAfter(inBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(sourceAfter.data(), inData->basePtr(), inBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(sourceAfter, input);
}

template<bool VarShape = false>
void CheckGaussianF16Alignment(int width, int height, bool planar)
{
    for (int residue = 0; residue < 4; ++residue)
    {
        cvcuda::Gaussian op({7, 7}, 2);
        for (int kernelWidth : {3, 5, 7})
        {
            // VarShape dispatch uses the constructor maximum, not the per-image windows.
            if constexpr (VarShape)
                op = cvcuda::Gaussian({kernelWidth, kernelWidth}, 2);
            for (NVCVBorderType border : {NVCV_BORDER_CONSTANT, NVCV_BORDER_REPLICATE, NVCV_BORDER_REFLECT,
                                          NVCV_BORDER_WRAP, NVCV_BORDER_REFLECT101})
            {
                CheckGaussianF16AlignmentCase<VarShape>(op, width, height, planar, residue, kernelWidth, border);
            }
        }
    }
}

} // namespace

TEST_P(OpGaussianF16Alignment, tensor_matches_float_gold_and_preserves_padding)
{
    CheckGaussianF16Alignment(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>());
}

NVCV_TEST_SUITE_P(OpGaussianF16AlignmentVarShape, test::ValueList<bool>{false, true});

TEST_P(OpGaussianF16AlignmentVarShape, matches_float_gold_and_preserves_padding)
{
    CheckGaussianF16Alignment<true>(65, 19, GetParam());
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// Gaussian filters each channel independently, so a planar input is filtered plane-by-plane and
// must produce exactly the same pixels as the interleaved path. These tests feed identical uint8
// data through cvcuda::Gaussian in both layouts and require the (re-interleaved) planar output to
// match the interleaved output bit-for-bit.
// =============================================================================

// Parameters: width, height, kernelWidth, kernelHeight, sigmaX, sigmaY, borderMode, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpGaussianPlanar,
                  test::ValueList<int, int, int, int, double, double, NVCVBorderType, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    { 64, 48, 3, 3, 0.5, 0.5, NVCV_BORDER_CONSTANT,  2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8},
    { 67, 51, 5, 3, 0.9, 0.7,   NVCV_BORDER_REFLECT, 1,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8},
    { 50, 40, 7, 7, 1.2, 1.2, NVCV_BORDER_REPLICATE, 2, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    { 64, 48, 9, 5, 1.4, 0.8, NVCV_BORDER_REFLECT101, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 32, 28, 3, 5, 0.6, 1.1,      NVCV_BORDER_WRAP, 1, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    { 33, 29, 67, 67, 14.0, 14.0,  NVCV_BORDER_CONSTANT, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 33, 29, 75, 75, 16.0, 16.0, NVCV_BORDER_REPLICATE, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 33, 29, -1, -1, 15.0, 15.0,  NVCV_BORDER_CONSTANT, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 33, 29, 95, 95, 18.0, 18.0,   NVCV_BORDER_REFLECT, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 33, 29, -1, -1, 16.0, 16.0, NVCV_BORDER_REPLICATE, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    // F16 parity is bit-exact like the other dtypes: both layouts run the same float-accumulating
    // kernel with identical tap order, so the single half store rounding matches.
    { 64, 48, 3, 3, 0.5, 0.5, NVCV_BORDER_CONSTANT,  2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
    { 50, 40, 7, 7, 1.2, 1.2, NVCV_BORDER_REPLICATE, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},

    // Runtime-sized taps use X4 for RGB/planar float and X2 for interleaved RGBA float.
    { 63, 47, 11, 11, 1.7, 1.7,       NVCV_BORDER_WRAP, 1, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
    { 63, 47, 17, 17, 2.5, 2.5,  NVCV_BORDER_REPLICATE, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    { 66, 34, 11, 11, 1.7, 1.7, NVCV_BORDER_REFLECT101, 1, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
});

// clang-format on

TEST_P(OpGaussianPlanar, tensor_matches_interleaved)
{
    nvcv::Size2D kernelSize{GetParamValue<2>(), GetParamValue<3>()};
    double2      sigma{GetParamValue<4>(), GetParamValue<5>()};
    nvcv::Size2D maxKernelSize = kernelSize;
    if (maxKernelSize.w <= 0)
        maxKernelSize.w = nvcv::cuda::round<int>(sigma.x * 3 * 2 + 1) | 1;
    if (maxKernelSize.h <= 0)
        maxKernelSize.h = nvcv::cuda::round<int>(sigma.y * 3 * 2 + 1) | 1;
    NVCVBorderType borderMode = GetParamValue<6>();
    int            numImages  = GetParamValue<7>();

    test::planar::RunTensorParity(
        GetParamValue<8>(), GetParamValue<9>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), numImages,
        [kernelSize, maxKernelSize, sigma, borderMode, numImages](cudaStream_t s, const nvcv::Tensor &src,
                                                                  const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::Gaussian op(maxKernelSize, numImages);
            EXPECT_NO_THROW(op(s, src, dst, kernelSize, sigma, borderMode));
        });
}

TEST_P(OpGaussianPlanar, varshape_matches_interleaved)
{
    nvcv::Size2D kernelSize{GetParamValue<2>(), GetParamValue<3>()};
    double2      sigma{GetParamValue<4>(), GetParamValue<5>()};
    nvcv::Size2D maxKernelSize = kernelSize;
    if (maxKernelSize.w <= 0)
        maxKernelSize.w = nvcv::cuda::round<int>(sigma.x * 3 * 2 + 1) | 1;
    if (maxKernelSize.h <= 0)
        maxKernelSize.h = nvcv::cuda::round<int>(sigma.y * 3 * 2 + 1) | 1;
    NVCVBorderType borderMode = GetParamValue<6>();
    int            numImages  = GetParamValue<7>();

    auto kernelSizeTensor
        = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{kernelSize.w, kernelSize.h});
    auto sigmaTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2F64, sigma);

    test::planar::RunVarShapeParity(
        GetParamValue<8>(), GetParamValue<9>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), numImages,
        [&kernelSizeTensor, &sigmaTensor, maxKernelSize, borderMode, numImages](
            cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::Gaussian op(maxKernelSize, numImages);
            EXPECT_NO_THROW(op(s, src, dst, kernelSizeTensor, sigmaTensor, borderMode));
        });
}

TEST(OpGaussianPlanar, varshape_specialized_and_dynamic_kernel_paths)
{
    const auto runCase = [](nvcv::Size2D maxKernelSize, nvcv::Size2D kernelSize)
    {
        auto kernelSizeTensor = test::planar::MakePerImageTensor(1, nvcv::TYPE_2S32, int2{kernelSize.w, kernelSize.h});
        auto sigmaTensor      = test::planar::MakePerImageTensor(1, nvcv::TYPE_2F64, double2{1.0, 1.0});

        test::planar::RunVarShapeParity(
            nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 47, 35, 47, 35, 1,
            [&kernelSizeTensor, &sigmaTensor, maxKernelSize](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                                             const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
            {
                cvcuda::Gaussian op(maxKernelSize, 1);
                EXPECT_NO_THROW(op(stream, src, dst, kernelSizeTensor, sigmaTensor, NVCV_BORDER_REPLICATE));
            });
    };

    runCase({7, 7}, {7, 7});
    runCase({17, 17}, {9, 9});
}

TEST(OpGaussianPlanar, varshape_f16_dynamic_kernel_fallback_matches_interleaved)
{
    constexpr int      numImages = 2;
    const nvcv::Size2D maxKernelSize{17, 17};
    const nvcv::Size2D kernelSize{9, 9};
    auto               kernelSizeTensor
        = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{kernelSize.w, kernelSize.h});
    auto sigmaTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2F64, double2{1.0, 1.0});
    const std::vector<nvcv::Size2D> sizes{
        {35, 27},
        {67, 51}
    };

    test::planar::RunVarShapeParity(
        nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16, sizes, sizes,
        [&kernelSizeTensor, &sigmaTensor, maxKernelSize](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                                         const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::Gaussian op(maxKernelSize, numImages);
            EXPECT_NO_THROW(op(stream, src, dst, kernelSizeTensor, sigmaTensor, NVCV_BORDER_REPLICATE));
        });
}

TEST(OpGaussian, dynamic_planar_dispatch_and_cross_stream_window_eviction)
{
    constexpr int      numImages = 2;
    const nvcv::Size2D maxKernelSize{17, 17};
    const nvcv::Size2D kernelSize{9, 9};
    auto               kernelSizeTensor
        = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{kernelSize.w, kernelSize.h});
    auto sigmaTensor = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2F64, double2{1.0, 1.0});
    const std::vector<nvcv::Size2D> sizes{
        {35, 27},
        {67, 51}
    };

    test::planar::RunVarShapeParity(
        nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, sizes, sizes,
        [&kernelSizeTensor, &sigmaTensor, maxKernelSize](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                                         const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::Gaussian op(maxKernelSize, numImages);
            EXPECT_NO_THROW(op(stream, src, dst, kernelSizeTensor, sigmaTensor, NVCV_BORDER_REPLICATE));
        });

    cudaStream_t reader;
    cudaStream_t evictor;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&reader));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&evictor));

    GaussianReuseFixture f;
    f.kernelSze = nvcv::Size2D{9, 9};
    ASSERT_NO_FATAL_FAILURE(f.Init());

    cvcuda::Gaussian op(f.kernelSze, 1);
    const double2    otherSigma{GaussianReuseFixture::kSigma * 2, GaussianReuseFixture::kSigma * 2};
    nvcv::Tensor     evictorOut
        = nvcv::util::CreateTensor(1, GaussianReuseFixture::kWidth, GaussianReuseFixture::kHeight, nvcv::FMT_RGB8);

    EXPECT_NO_THROW(op(reader, f.in, f.out, f.kernelSze, f.sigma, f.border));
    EXPECT_NO_THROW(op(evictor, f.in, evictorOut, f.kernelSze, f.sigma, f.border));
    EXPECT_NO_THROW(op(evictor, f.in, evictorOut, f.kernelSze, otherSigma, f.border));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(reader));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(evictor));
    ASSERT_NO_FATAL_FAILURE(f.ExpectOutputMatchesGold("dynamic reader keeps its window"));
    ASSERT_NO_FATAL_FAILURE(f.ExpectMatchesGold(evictorOut, f.GoldFor(otherSigma), "evictor gets its own window"));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(reader));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(evictor));
}

static auto OpGaussianNegativeParams()
{
    nvcv::test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, int, int, double, double, NVCVBorderType>
        params{
            {NVCV_ERROR_INVALID_ARGUMENT,    nvcv::FMT_U8,   nvcv::FMT_U16, 3, 3, 0.5, 0.5,
             NVCV_BORDER_CONSTANT}, // data type is different
            {NVCV_ERROR_INVALID_ARGUMENT,  nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 3, 3, 0.5, 0.5,
             NVCV_BORDER_CONSTANT}, // data format is different
            {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, 3, 3, 0.5, 0.5,
             NVCV_BORDER_CONSTANT}, // data format is different
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 3, 3, 0.5, 0.5,
                        static_cast<NVCVBorderType>(255));
#endif
    // unsupported data type (64-bit float; F16 is now valid)
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F64, nvcv::FMT_F64, 3, 3, 0.5, 0.5,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 4, 3, 0.5, 0.5, NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 3, 4, 0.5, 0.5, NVCV_BORDER_CONSTANT);
    return params;
}

NVCV_TEST_SUITE_P(OpGaussian_Negative, OpGaussianNegativeParams());

TEST_P(OpGaussian_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVStatus        expectedReturnCode = GetParamValue<0>();
    nvcv::ImageFormat inputFmt           = GetParamValue<1>();
    nvcv::ImageFormat outputFmt          = GetParamValue<2>();
    int               ksizeX             = GetParamValue<3>();
    int               ksizeY             = GetParamValue<4>();
    double            sigmaX             = GetParamValue<5>();
    double            sigmaY             = GetParamValue<6>();
    NVCVBorderType    borderMode         = GetParamValue<7>();

    int width   = 24;
    int height  = 24;
    int batches = 1;

    double2      sigma{sigmaX, sigmaY};
    nvcv::Size2D kernelSize(ksizeX, ksizeY);

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, inputFmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, outputFmt);

    // run operator
    cvcuda::Gaussian gaussianOp({11, 11}, 1);

    EXPECT_EQ(expectedReturnCode,
              nvcv::ProtectCall([&gaussianOp, &stream, &inTensor, &outTensor, &kernelSize, &sigma, &borderMode]
                                { gaussianOp(stream, inTensor, outTensor, kernelSize, sigma, borderMode); }));
}

// Same matrix as test::PlanarFilterVarShapeNegativeParams(), except the unsupported-dtype row is
// F64: Gaussian now accepts F16, while the shared list keeps RGBf16 for the filter ops that
// still reject it.
static auto OpGaussianVarShapeNegativeParams()
{
    test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType, int, int> params{
        { nvcv::FMT_RGB8, nvcv::FMT_RGB8p, NVCV_BORDER_CONSTANT, 3,  3},
        {nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, 3,  3},
        {  nvcv::FMT_F64,   nvcv::FMT_F64, NVCV_BORDER_CONSTANT, 3,  3},
        { nvcv::FMT_RGB8,  nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, 3, -1},
        { nvcv::FMT_RGB8,  nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, 5,  3},
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(nvcv::FMT_RGB8, nvcv::FMT_RGB8, static_cast<NVCVBorderType>(255), 3, 3);
#endif
    return params;
}

NVCV_TEST_SUITE_P(OpGaussianVarshape_Negative, OpGaussianVarShapeNegativeParams());

TEST_P(OpGaussianVarshape_Negative, op)
{
    test::planar::ExpectVarShapeUniformFormatRejected(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<3>(),
                                                      GetParamValue<4>(), GetParamValue<2>(),
                                                      InvokeGaussianVarShapeNegative);
}

TEST(OpGaussianVarshape_Negative, varshape_hasDifferentFormat)
{
    test::planar::ExpectVarShapeMixedFormatRejected(InvokeGaussianVarShapeNegative);
}

// Regression: the var-shape channel guard read `channels > 4 || (isPlanar && channels == 2)`, so an
// interleaved 2-channel batch passed validation -- unlike the Tensor overload and unlike Conv2D,
// which both rejected `channels == 2` outright, and unlike the public header, which documents
// channels [1, 3, 4]. The accepted batch then indexed the [dtype][channels] dispatch table at column
// 1, which is null in every row because no 2-component vector type is ever instantiated, and called
// through that null pointer; the NVCV_ASSERT meant to catch it compiles away in release builds. This
// test submits exactly such a batch, so before the fix it faults instead of reporting
// NVCV_ERROR_INVALID_ARGUMENT.
TEST(OpGaussianVarshape_Negative, rejects_two_channel_interleaved_batch)
{
    test::planar::ExpectVarShapeUniformFormatRejected(nvcv::FMT_2F32, nvcv::FMT_2F32, 2, 2, NVCV_BORDER_CONSTANT,
                                                      InvokeGaussianVarShapeNegative);
}

// Hardening: the two var-shape dispatches size their grids from opposite batches -- interleaved from
// the input's maxSize, planar from the output's -- so a per-image extent mismatch left part of the
// output unwritten on one path and blurred the image against the border on the other. Neither the
// image-count, format, nor per-image-parameter guards covered it.
TEST(OpGaussianVarshape_Negative, rejects_varshape_per_image_size_mismatch)
{
    constexpr int kBatch = 2;

    nvcv::ImageBatchVarShape src(kBatch);
    nvcv::ImageBatchVarShape dst(kBatch);
    src.pushBack(nvcv::Image(nvcv::Size2D{16, 12}, nvcv::FMT_U8));
    src.pushBack(nvcv::Image(nvcv::Size2D{16, 12}, nvcv::FMT_U8));
    dst.pushBack(nvcv::Image(nvcv::Size2D{16, 12}, nvcv::FMT_U8));
    dst.pushBack(nvcv::Image(nvcv::Size2D{24, 20}, nvcv::FMT_U8));

    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&] { InvokeGaussianVarShapeNegative(nullptr, src, dst, kBatch, NVCV_BORDER_CONSTANT); }));
}

TEST(OpGaussian_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaGaussianCreate(nullptr, 224, 224, 10), NVCV_ERROR_INVALID_ARGUMENT);
}

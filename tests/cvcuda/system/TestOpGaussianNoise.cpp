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

#include "GaussianNoiseUtils.cuh"

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpGaussianNoise.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#define NVCV_IMAGE_FORMAT_2U8 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XY00, ASSOCIATED, X8_Y8)

namespace {
template<typename Submit>
void SubmitGaussianNoiseRepeatedly(int calls, Submit &&submit)
{
    for (int call = 0; call < calls; ++call)
    {
        submit();
    }
}

inline int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

template<typename T>
T cast(float value)
{
    static_assert(std::is_integral_v<T>);

    double rounded = static_cast<double>(value) + (value >= 0 ? 0.5 : -0.5);
    if (rounded < static_cast<double>(std::numeric_limits<T>::lowest()))
    {
        return std::numeric_limits<T>::lowest();
    }
    if (rounded > static_cast<double>(std::numeric_limits<T>::max()))
    {
        return std::numeric_limits<T>::max();
    }
    return static_cast<T>(rounded);
}

template<>
float cast(float value)
{
    return nvcv::cuda::clamp(nvcv::cuda::StaticCast<float>(value), 0.f, 1.f);
}

template<typename T>
void GaussianNoise(std::vector<T> &src, std::vector<T> &dst, float mu, float sigma, int batch, bool per_channel,
                   int channels, int call_index = 0)
{
    auto mem_size = static_cast<int>(src.size());
    if (!per_channel)
        mem_size /= channels;
    std::vector<float> rand_h(mem_size);
    get_random(rand_h.data(), per_channel, batch, mem_size, channels, call_index);
    const float *rand = rand_h.data();

    auto img_size = static_cast<int>(src.size() / channels);
    for (int i = 0; i < img_size; i++)
    {
        if (per_channel)
        {
            for (int ch = 0; ch < channels; ch++)
            {
                float delta            = mu + rand[i * channels + ch] * sigma;
                dst[i * channels + ch] = cast<T>(static_cast<float>(src[i * channels + ch]) + delta);
            }
        }
        else
        {
            float delta = mu + rand[i] * sigma;
            for (int ch = 0; ch < channels; ++ch)
            {
                dst[i * channels + ch] = cast<T>(static_cast<float>(src[i * channels + ch]) + delta);
            }
        }
    }
}

nvcv::Tensor MakeGaussianNoiseParam(int batch, float value)
{
    nvcv::Tensor tensor({{batch}, "N"}, nvcv::TYPE_F32);
    nvcv::test::planar::UploadTensorValues(tensor, std::vector<float>(batch, value));
    return tensor;
}

std::vector<uint8_t> MakeGaussianNoiseInput(nvcv::ImageFormat fmt, int width, int height, int sample)
{
    const int    channels = fmt.numChannels();
    const int    elemSize = fmt.planePixelStrideBytes(0) / channels;
    const size_t count    = static_cast<size_t>(width) * height * channels;

    std::vector<uint8_t> bytes(count * elemSize);
    if (fmt.dataKind() == nvcv::DataKind::FLOAT)
    {
        if (elemSize != static_cast<int>(sizeof(float)))
        {
            throw std::invalid_argument("MakeGaussianNoiseInput supports only 32-bit float formats");
        }
        std::vector<float> values(count);
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] = static_cast<float>(((i + 1) * 37 + sample * 17) % 1024) / 1023.f;
        }
        std::memcpy(bytes.data(), values.data(), values.size() * sizeof(float));
    }
    else
    {
        for (size_t i = 0; i < bytes.size(); ++i)
        {
            bytes[i] = static_cast<uint8_t>(((i + 1) * 37 + sample * 17) % 256);
        }
    }
    return bytes;
}

void RunGaussianNoiseTensorPlanarParity(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int width,
                                        int height, int batch, float mu, float sigma, bool per_channel)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int channels  = planarFmt.numChannels();
    const int elemSize  = planarFmt.planePixelStrideBytes(0);
    const int rowStride = width * channels * elemSize;

    nvcv::Tensor srcI = nvcv::util::CreateTensor(batch, width, height, interleavedFmt);
    nvcv::Tensor dstI = nvcv::util::CreateTensor(batch, width, height, interleavedFmt);
    nvcv::Tensor srcP = nvcv::util::CreateTensor(batch, width, height, planarFmt);
    nvcv::Tensor dstP = nvcv::util::CreateTensor(batch, width, height, planarFmt);

    auto srcIData = srcI.exportData<nvcv::TensorDataStridedCuda>();
    auto dstIData = dstI.exportData<nvcv::TensorDataStridedCuda>();
    auto srcPData = srcP.exportData<nvcv::TensorDataStridedCuda>();
    auto dstPData = dstP.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcIData && dstIData && srcPData && dstPData);

    auto srcIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcIData);
    auto dstIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstIData);
    auto srcPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcPData);
    auto dstPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstPData);
    ASSERT_TRUE(srcIAcc && dstIAcc && srcPAcc && dstPAcc);

    for (int i = 0; i < batch; ++i)
    {
        auto hwc = MakeGaussianNoiseInput(interleavedFmt, width, height, i + 17);
        nvcv::test::planar::UploadInterleavedSample(*srcIAcc, i, hwc, width, height, rowStride);
        nvcv::test::planar::UploadPlanarSample(
            *srcPAcc, i, nvcv::test::planar::DeinterleaveToPlanes(hwc, width, height, channels, elemSize), width,
            height, channels, elemSize);
    }

    nvcv::Tensor muval    = MakeGaussianNoiseParam(batch, mu);
    nvcv::Tensor sigmaval = MakeGaussianNoiseParam(batch, sigma);

    unsigned long long    seed = 12345;
    cvcuda::GaussianNoise interleavedOp(batch);
    cvcuda::GaussianNoise planarOp(batch);
    EXPECT_NO_THROW(interleavedOp(stream, srcI, dstI, muval, sigmaval, per_channel, seed));
    EXPECT_NO_THROW(planarOp(stream, srcP, dstP, muval, sigmaval, per_channel, seed));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; ++i)
    {
        SCOPED_TRACE(i);
        auto gpuInter    = nvcv::test::planar::DownloadInterleavedSample(*dstIAcc, i, width, height, rowStride);
        auto planesOut   = nvcv::test::planar::DownloadPlanarSample(*dstPAcc, i, width, height, channels, elemSize);
        auto planarInter = nvcv::test::planar::InterleaveFromPlanes(planesOut, width, height, channels, elemSize);
        EXPECT_EQ(gpuInter, planarInter);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

void RunGaussianNoiseVarShapePlanarParity(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int width,
                                          int height, int batch, float mu, float sigma, bool per_channel)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int channels = planarFmt.numChannels();
    const int elemSize = planarFmt.planePixelStrideBytes(0);

    std::vector<nvcv::Image> srcI;
    std::vector<nvcv::Image> dstI;
    std::vector<nvcv::Image> srcP;
    std::vector<nvcv::Image> dstP;
    std::vector<int>         widths;
    std::vector<int>         heights;
    for (int i = 0; i < batch; ++i)
    {
        widths.push_back(width + i % 3);
        heights.push_back(height + i % 2);
        srcI.emplace_back(nvcv::Size2D{widths.back(), heights.back()}, interleavedFmt);
        dstI.emplace_back(nvcv::Size2D{widths.back(), heights.back()}, interleavedFmt);
        srcP.emplace_back(nvcv::Size2D{widths.back(), heights.back()}, planarFmt);
        dstP.emplace_back(nvcv::Size2D{widths.back(), heights.back()}, planarFmt);
    }

    nvcv::ImageBatchVarShape batchSrcI(batch);
    nvcv::ImageBatchVarShape batchDstI(batch);
    nvcv::ImageBatchVarShape batchSrcP(batch);
    nvcv::ImageBatchVarShape batchDstP(batch);
    batchSrcI.pushBack(srcI.begin(), srcI.end());
    batchDstI.pushBack(dstI.begin(), dstI.end());
    batchSrcP.pushBack(srcP.begin(), srcP.end());
    batchDstP.pushBack(dstP.begin(), dstP.end());

    for (int i = 0; i < batch; ++i)
    {
        const int rowStride = widths[i] * channels * elemSize;
        auto      hwc       = MakeGaussianNoiseInput(interleavedFmt, widths[i], heights[i], i + 31);

        auto idata = srcI[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(idata->plane(0).basePtr, idata->plane(0).rowStride, hwc.data(), rowStride,
                                            rowStride, heights[i], cudaMemcpyHostToDevice));

        auto planes = nvcv::test::planar::DeinterleaveToPlanes(hwc, widths[i], heights[i], channels, elemSize);
        auto pdata  = srcP[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(pdata->numPlanes(), channels);
        const int planeBytes = widths[i] * heights[i] * elemSize;
        for (int c = 0; c < channels; ++c)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(pdata->plane(c).basePtr, pdata->plane(c).rowStride, planes.data() + c * planeBytes,
                                   widths[i] * elemSize, widths[i] * elemSize, heights[i], cudaMemcpyHostToDevice));
        }
    }

    nvcv::Tensor muval    = MakeGaussianNoiseParam(batch, mu);
    nvcv::Tensor sigmaval = MakeGaussianNoiseParam(batch, sigma);

    unsigned long long    seed = 12345;
    cvcuda::GaussianNoise interleavedOp(batch);
    cvcuda::GaussianNoise planarOp(batch);
    EXPECT_NO_THROW(interleavedOp(stream, batchSrcI, batchDstI, muval, sigmaval, per_channel, seed));
    EXPECT_NO_THROW(planarOp(stream, batchSrcP, batchDstP, muval, sigmaval, per_channel, seed));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; ++i)
    {
        SCOPED_TRACE(i);
        const int rowStride  = widths[i] * channels * elemSize;
        const int planeBytes = widths[i] * heights[i] * elemSize;

        std::vector<uint8_t> gpuInter(heights[i] * rowStride);
        auto                 idata = dstI[i].exportData<nvcv::ImageDataStridedCuda>();
        EXPECT_EQ(cudaSuccess, cudaMemcpy2D(gpuInter.data(), rowStride, idata->plane(0).basePtr,
                                            idata->plane(0).rowStride, rowStride, heights[i], cudaMemcpyDeviceToHost));

        std::vector<uint8_t> planesOut(widths[i] * heights[i] * channels * elemSize);
        auto                 pdata = dstP[i].exportData<nvcv::ImageDataStridedCuda>();
        for (int c = 0; c < channels; ++c)
        {
            EXPECT_EQ(cudaSuccess, cudaMemcpy2D(planesOut.data() + c * planeBytes, widths[i] * elemSize,
                                                pdata->plane(c).basePtr, pdata->plane(c).rowStride,
                                                widths[i] * elemSize, heights[i], cudaMemcpyDeviceToHost));
        }
        auto planarInter
            = nvcv::test::planar::InterleaveFromPlanes(planesOut, widths[i], heights[i], channels, elemSize);
        EXPECT_EQ(gpuInter, planarInter);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
} // namespace

// clang-format off
NVCV_TEST_SUITE_P(OpGaussianNoise, nvcv::test::ValueList<int, int, int, float, float, bool>
{
    //batch,    height,     width,      mu,       sigma,     per_channel
    {     1,       480,       360,       0,       0.005,       false  },
    {     4,       100,       101,       0,       0.008,        true  },
    {     3,       360,       480,       0,       0.004,       false  },
    {     1,       800,       600,       0,       0.006,        true  },
});

// clang-format on

template<typename datatype>
static void tensor_correct_output_test(int batch, int height, int width, float mu, float sigma, bool per_channel,
                                       nvcv::ImageFormat fmt, int calls = 1)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

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
    nvcv::Tensor muval({{batch}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor sigmaval({{batch}, "N"}, nvcv::TYPE_F32);

    auto muData    = muval.exportData<nvcv::TensorDataStridedCuda>();
    auto sigmaData = sigmaval.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, muData);
    ASSERT_NE(nullptr, sigmaData);

    std::vector<float> muVec(batch, mu);
    std::vector<float> sigmaVec(batch, sigma);

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(muData->basePtr(), muVec.data(), muVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(sigmaData->basePtr(), sigmaVec.data(), sigmaVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));

    //Generate input
    std::vector<std::vector<datatype>> srcVec(batch);
    std::default_random_engine         randEng;
    int                                rowStride = width * fmt.planePixelStrideBytes(0);

    for (int i = 0; i < batch; i++)
    {
        if constexpr (std::is_integral_v<datatype>)
        {
            constexpr int64_t                      minValue = std::is_signed_v<datatype> ? -1000 : 0;
            constexpr int64_t                      maxValue = std::is_same_v<datatype, uint8_t> ? 255 : 2000;
            std::uniform_int_distribution<int64_t> rand(minValue, maxValue);
            srcVec[i].resize(height * rowStride / sizeof(datatype));
            std::ranges::generate(srcVec[i], [&rand, &randEng]() { return static_cast<datatype>(rand(randEng)); });
        }
        else
        {
            std::uniform_real_distribution<float> rand(0.f, 1.f);
            srcVec[i].resize(height * rowStride / sizeof(datatype));
            std::ranges::generate(srcVec[i], [&rand, &randEng]() { return rand(randEng); });
        }
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(i), inAccess->rowStride(), srcVec[i].data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));
    }

    // Call operator
    int                   maxBatch = 4;
    unsigned long long    seed     = 12345;
    cvcuda::GaussianNoise GaussianNoiseOp(maxBatch);
    SubmitGaussianNoiseRepeatedly(
        calls, [&GaussianNoiseOp, &stream, &imgIn, &imgOut, &muval, &sigmaval, &per_channel, &seed]
        { EXPECT_NO_THROW(GaussianNoiseOp(stream, imgIn, imgOut, muval, sigmaval, per_channel, seed)); });

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        std::vector<datatype> testVec(height * rowStride / sizeof(datatype));
        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowStride, outAccess->sampleData(i), outAccess->rowStride(),
                                            rowStride, height, cudaMemcpyDeviceToHost));

        std::vector<datatype> goldVec(height * rowStride / sizeof(datatype));
        GaussianNoise<datatype>(srcVec[i], goldVec, mu, sigma, i, per_channel, fmt.numChannels(), calls - 1);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpGaussianNoise, tensor_correct_output)
{
    int   batch       = GetParamValue<0>();
    int   height      = GetParamValue<1>();
    int   width       = GetParamValue<2>();
    float mu          = GetParamValue<3>();
    float sigma       = GetParamValue<4>();
    bool  per_channel = GetParamValue<5>();
    tensor_correct_output_test<uint8_t>(batch, height, width, mu, sigma, per_channel, nvcv::FMT_RGB8);
}

TEST_P(OpGaussianNoise, tensor_correct_output_float)
{
    int   batch       = GetParamValue<0>();
    int   height      = GetParamValue<1>();
    int   width       = GetParamValue<2>();
    float mu          = GetParamValue<3>();
    float sigma       = GetParamValue<4>();
    bool  per_channel = GetParamValue<5>();
    tensor_correct_output_test<float>(batch, height, width, mu, sigma, per_channel, nvcv::FMT_RGBf32);
}

TEST(OpGaussianNoise, tensor_repeated_call_advances_rng_state)
{
    // Saturation makes the integer oracle depend on the random sign, not host/device rounding at half-integers.
    tensor_correct_output_test<uint8_t>(1, 480, 360, 0.f, 1e20f, true, nvcv::FMT_RGB8, 2);
    tensor_correct_output_test<float>(1, 480, 360, 0.f, 0.005f, false, nvcv::FMT_RGBf32, 2);
}

TEST(OpGaussianNoise, tensor_variable_batch_preserves_rng_state_across_streams)
{
    constexpr int batch1    = 1;
    constexpr int batch2    = 2;
    constexpr int width     = 263;
    constexpr int height    = 257;
    constexpr int rowStride = width * 3;

    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaEvent_t  firstSubmitDone;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream1));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream2));
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&firstSubmitDone, cudaEventDisableTiming));

    nvcv::Tensor src1     = nvcv::util::CreateTensor(batch1, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor dst1     = nvcv::util::CreateTensor(batch1, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor src2     = nvcv::util::CreateTensor(batch2, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor dst2     = nvcv::util::CreateTensor(batch2, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor ref0Dst  = nvcv::util::CreateTensor(batch1, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor ref12Dst = nvcv::util::CreateTensor(batch2, width, height, nvcv::FMT_RGB8);

    auto src1Data  = src1.exportData<nvcv::TensorDataStridedCuda>();
    auto src2Data  = src2.exportData<nvcv::TensorDataStridedCuda>();
    auto dst2Data  = dst2.exportData<nvcv::TensorDataStridedCuda>();
    auto ref0Data  = ref0Dst.exportData<nvcv::TensorDataStridedCuda>();
    auto ref12Data = ref12Dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(src1Data && src2Data && dst2Data && ref0Data && ref12Data);
    auto src1Access  = nvcv::TensorDataAccessStridedImagePlanar::Create(*src1Data);
    auto src2Access  = nvcv::TensorDataAccessStridedImagePlanar::Create(*src2Data);
    auto dst2Access  = nvcv::TensorDataAccessStridedImagePlanar::Create(*dst2Data);
    auto ref0Access  = nvcv::TensorDataAccessStridedImagePlanar::Create(*ref0Data);
    auto ref12Access = nvcv::TensorDataAccessStridedImagePlanar::Create(*ref12Data);
    ASSERT_TRUE(src1Access && src2Access && dst2Access && ref0Access && ref12Access);

    auto sample0 = MakeGaussianNoiseInput(nvcv::FMT_RGB8, width, height, 101);
    auto sample1 = MakeGaussianNoiseInput(nvcv::FMT_RGB8, width, height, 202);
    nvcv::test::planar::UploadInterleavedSample(*src1Access, 0, sample0, width, height, rowStride);
    nvcv::test::planar::UploadInterleavedSample(*src2Access, 0, sample0, width, height, rowStride);
    nvcv::test::planar::UploadInterleavedSample(*src2Access, 1, sample1, width, height, rowStride);

    nvcv::Tensor   mu1    = MakeGaussianNoiseParam(batch1, 0.f);
    nvcv::Tensor   sigma1 = MakeGaussianNoiseParam(batch1, 1e20f);
    nvcv::Tensor   mu2    = MakeGaussianNoiseParam(batch2, 0.f);
    nvcv::Tensor   sigma2 = MakeGaussianNoiseParam(batch2, 1e20f);
    constexpr auto seed   = 12345ULL;
    constexpr bool perCh  = true;

    cvcuda::GaussianNoise op(2);
    cvcuda::GaussianNoise ref0(1);
    cvcuda::GaussianNoise ref12(2);

    EXPECT_NO_THROW(op(stream1, src1, dst1, mu1, sigma1, perCh, seed));
    ASSERT_EQ(cudaSuccess, cudaEventRecord(firstSubmitDone, stream1));
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(stream2, firstSubmitDone));
    EXPECT_NO_THROW(op(stream2, src2, dst2, mu2, sigma2, perCh, seed));

    EXPECT_NO_THROW(ref0(stream2, src1, ref0Dst, mu1, sigma1, perCh, seed));
    EXPECT_NO_THROW(ref0(stream2, src1, ref0Dst, mu1, sigma1, perCh, seed));
    EXPECT_NO_THROW(ref12(stream2, src2, ref12Dst, mu2, sigma2, perCh, seed));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream2));

    EXPECT_EQ(nvcv::test::planar::DownloadInterleavedSample(*dst2Access, 0, width, height, rowStride),
              nvcv::test::planar::DownloadInterleavedSample(*ref0Access, 0, width, height, rowStride));
    EXPECT_EQ(nvcv::test::planar::DownloadInterleavedSample(*dst2Access, 1, width, height, rowStride),
              nvcv::test::planar::DownloadInterleavedSample(*ref12Access, 1, width, height, rowStride));

    EXPECT_EQ(cudaSuccess, cudaEventDestroy(firstSubmitDone));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream1));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream2));
}

TEST(OpGaussianNoise, tensor_correct_output_supported_integer_types_and_channels)
{
    const nvcv::ImageFormat fmt2U8{NVCV_IMAGE_FORMAT_2U8};

    tensor_correct_output_test<uint8_t>(2, 17, 19, 3.f, 25.f, false, nvcv::FMT_U8);
    tensor_correct_output_test<uint8_t>(2, 19, 17, 3.f, 25.f, false, fmt2U8);
    tensor_correct_output_test<uint8_t>(2, 17, 19, 3.f, 25.f, true, fmt2U8);
    tensor_correct_output_test<uint8_t>(2, 19, 17, 3.f, 25.f, false, nvcv::FMT_RGBA8);
    tensor_correct_output_test<uint16_t>(2, 17, 19, 3.f, 25.f, false, nvcv::FMT_U16);
    tensor_correct_output_test<uint16_t>(2, 19, 17, 3.f, 25.f, true, nvcv::FMT_U16);
    tensor_correct_output_test<int16_t>(2, 17, 19, 3.f, 25.f, false, nvcv::FMT_S16);
    tensor_correct_output_test<int16_t>(2, 19, 17, 3.f, 25.f, true, nvcv::FMT_S16);
    tensor_correct_output_test<int32_t>(2, 17, 19, 3.f, 25.f, false, nvcv::FMT_S32);
    tensor_correct_output_test<int32_t>(2, 19, 17, 3.f, 25.f, true, nvcv::FMT_S32);
    tensor_correct_output_test<float>(2, 19, 17, 0.f, 0.05f, true, nvcv::FMT_RGBAf32);
}

template<typename datatype>
static void varshape_correct_output_test(int batch, int height, int width, float mu, float sigma, bool per_channel,
                                         nvcv::ImageFormat fmt, int calls = 1)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

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
    nvcv::Tensor muval({{batch}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor sigmaval({{batch}, "N"}, nvcv::TYPE_F32);

    auto muData    = muval.exportData<nvcv::TensorDataStridedCuda>();
    auto sigmaData = sigmaval.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, muData);
    ASSERT_NE(nullptr, sigmaData);

    std::vector<float> muVec(batch, mu);
    std::vector<float> sigmaVec(batch, sigma);

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(muData->basePtr(), muVec.data(), muVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(sigmaData->basePtr(), sigmaVec.data(), sigmaVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));

    //Generate input
    std::vector<std::vector<datatype>> srcVec(batch);

    for (int i = 0; i < batch; i++)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        if constexpr (std::is_integral_v<datatype>)
        {
            constexpr int64_t                      minValue = std::is_signed_v<datatype> ? -1000 : 0;
            constexpr int64_t                      maxValue = std::is_same_v<datatype, uint8_t> ? 255 : 2000;
            std::uniform_int_distribution<int64_t> rand(minValue, maxValue);
            srcVec[i].resize(srcHeight * srcRowStride / sizeof(datatype));
            std::ranges::generate(srcVec[i], [&rand, &randEng]() { return static_cast<datatype>(rand(randEng)); });
        }
        else
        {
            std::uniform_real_distribution<float> rand(0.f, 1.f);
            srcVec[i].resize(srcHeight * srcRowStride / sizeof(datatype));
            std::ranges::generate(srcVec[i], [&rand, &randEng]() { return rand(randEng); });
        }

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(),
                                            srcRowStride, srcRowStride, srcHeight, cudaMemcpyHostToDevice));
    }

    // Call operator
    int                   maxBatch = 4;
    unsigned long long    seed     = 12345;
    cvcuda::GaussianNoise GaussianNoiseOp(maxBatch);
    SubmitGaussianNoiseRepeatedly(
        calls, [&GaussianNoiseOp, &stream, &batchSrc, &batchDst, &muval, &sigmaval, &per_channel, &seed]
        { EXPECT_NO_THROW(GaussianNoiseOp(stream, batchSrc, batchDst, muval, sigmaval, per_channel, seed)); });

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<datatype> testVec(dstHeight * dstRowStride / sizeof(datatype));

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<datatype> goldVec(dstHeight * dstRowStride / sizeof(datatype));
        GaussianNoise<datatype>(srcVec[i], goldVec, mu, sigma, i, per_channel, fmt.numChannels(), calls - 1);
        EXPECT_EQ(goldVec, testVec);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpGaussianNoise, varshape_correct_shape)
{
    int   batch       = GetParamValue<0>();
    int   height      = GetParamValue<1>();
    int   width       = GetParamValue<2>();
    float mu          = GetParamValue<3>();
    float sigma       = GetParamValue<4>();
    bool  per_channel = GetParamValue<5>();

    varshape_correct_output_test<uint8_t>(batch, height, width, mu, sigma, per_channel, nvcv::FMT_RGB8);
}

TEST_P(OpGaussianNoise, varshape_correct_shape_float)
{
    int   batch       = GetParamValue<0>();
    int   height      = GetParamValue<1>();
    int   width       = GetParamValue<2>();
    float mu          = GetParamValue<3>();
    float sigma       = GetParamValue<4>();
    bool  per_channel = GetParamValue<5>();

    varshape_correct_output_test<float>(batch, height, width, mu, sigma, per_channel, nvcv::FMT_RGBf32);
}

TEST(OpGaussianNoise, varshape_repeated_call_advances_rng_state)
{
    // Saturation makes the integer oracle depend on the random sign, not host/device rounding at half-integers.
    varshape_correct_output_test<uint8_t>(2, 480, 360, 0.f, 1e20f, true, nvcv::FMT_RGB8, 2);
    varshape_correct_output_test<float>(2, 480, 360, 0.f, 0.005f, false, nvcv::FMT_RGBf32, 2);
}

TEST(OpGaussianNoise, varshape_correct_output_supported_integer_types_and_channels)
{
    const nvcv::ImageFormat fmt2U8{NVCV_IMAGE_FORMAT_2U8};

    varshape_correct_output_test<uint8_t>(2, 17, 19, 3.f, 25.f, false, nvcv::FMT_U8);
    varshape_correct_output_test<uint8_t>(2, 19, 17, 3.f, 25.f, false, fmt2U8);
    varshape_correct_output_test<uint8_t>(2, 17, 19, 3.f, 25.f, true, fmt2U8);
    varshape_correct_output_test<uint8_t>(2, 19, 17, 3.f, 25.f, false, nvcv::FMT_RGBA8);
    varshape_correct_output_test<uint16_t>(2, 17, 19, 3.f, 25.f, false, nvcv::FMT_U16);
    varshape_correct_output_test<uint16_t>(2, 19, 17, 3.f, 25.f, true, nvcv::FMT_U16);
    varshape_correct_output_test<int16_t>(2, 17, 19, 3.f, 25.f, false, nvcv::FMT_S16);
    varshape_correct_output_test<int16_t>(2, 19, 17, 3.f, 25.f, true, nvcv::FMT_S16);
    varshape_correct_output_test<int32_t>(2, 17, 19, 3.f, 25.f, false, nvcv::FMT_S32);
    varshape_correct_output_test<int32_t>(2, 19, 17, 3.f, 25.f, true, nvcv::FMT_S32);
    varshape_correct_output_test<float>(2, 19, 17, 0.f, 0.05f, true, nvcv::FMT_RGBAf32);
}

// clang-format off
NVCV_TEST_SUITE_P(OpGaussianNoisePlanar, nvcv::test::ValueList<int, int, int, float, float, bool, nvcv::ImageFormat, nvcv::ImageFormat>
{
    // batch, height, width, mu, sigma, per_channel, planar format, interleaved format
    {    2,     48,    64, 0.f, 0.005f,       false,     nvcv::FMT_RGB8p,     nvcv::FMT_RGB8},
    {    1,     47,    65, 0.f, 0.008f,        true,     nvcv::FMT_RGB8p,     nvcv::FMT_RGB8},
    {    2,     29,    33, 0.f, 0.004f,       false, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    {    1,     27,    31, 0.f, 0.006f,        true, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    {    1,    257,   263, 3.f,   25.f,         true,     nvcv::FMT_RGB8p,     nvcv::FMT_RGB8},
    {    1,    263,   257, 0.f,  0.05f,        false, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

// clang-format on

TEST_P(OpGaussianNoisePlanar, tensor_matches_interleaved)
{
    int               batch          = GetParamValue<0>();
    int               height         = GetParamValue<1>();
    int               width          = GetParamValue<2>();
    float             mu             = GetParamValue<3>();
    float             sigma          = GetParamValue<4>();
    bool              per_channel    = GetParamValue<5>();
    nvcv::ImageFormat planarFmt      = GetParamValue<6>();
    nvcv::ImageFormat interleavedFmt = GetParamValue<7>();

    RunGaussianNoiseTensorPlanarParity(planarFmt, interleavedFmt, width, height, batch, mu, sigma, per_channel);
}

TEST_P(OpGaussianNoisePlanar, varshape_matches_interleaved)
{
    int               batch          = GetParamValue<0>();
    int               height         = GetParamValue<1>();
    int               width          = GetParamValue<2>();
    float             mu             = GetParamValue<3>();
    float             sigma          = GetParamValue<4>();
    bool              per_channel    = GetParamValue<5>();
    nvcv::ImageFormat planarFmt      = GetParamValue<6>();
    nvcv::ImageFormat interleavedFmt = GetParamValue<7>();

    RunGaussianNoiseVarShapePlanarParity(planarFmt, interleavedFmt, width, height, batch, mu, sigma, per_channel);
}

TEST(OpGaussianNoisePlanar, tensor_rejects_two_channel)
{
    constexpr int         batch    = 1;
    nvcv::Tensor          muval    = MakeGaussianNoiseParam(batch, 0.f);
    nvcv::Tensor          sigmaval = MakeGaussianNoiseParam(batch, 0.005f);
    cvcuda::GaussianNoise op(batch);

    nvcv::test::planar::ExpectPlanarTensorRejected(
        {batch, 2, 16, 16}, {batch, 2, 16, 16},
        [&op, &muval, &sigmaval](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
        { op(stream, src, dst, muval, sigmaval, true, 12345); });
}

TEST(OpGaussianNoisePlanar, varshape_rejects_two_channel)
{
    constexpr int           batch    = 1;
    nvcv::Tensor            muval    = MakeGaussianNoiseParam(batch, 0.f);
    nvcv::Tensor            sigmaval = MakeGaussianNoiseParam(batch, 0.005f);
    cvcuda::GaussianNoise   op(batch);
    const nvcv::ImageFormat twoChannelPlanar{NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XY00, ASSOCIATED, X8, X8)};

    nvcv::test::planar::ExpectVarShapeRejected(
        {twoChannelPlanar}, {twoChannelPlanar},
        [&op, &muval, &sigmaval](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                 const nvcv::ImageBatchVarShape &dst)
        { op(stream, src, dst, muval, sigmaval, true, 12345); });
}

TEST(OpGaussianNoiseScalar, uint8_matches_torchvision_clip_and_wrap_semantics)
{
    constexpr int width     = 2;
    constexpr int height    = 1;
    constexpr int rowStride = width * 3;
    nvcv::Tensor  src       = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor  clipped   = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor  wrapped   = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGB8);

    auto srcData     = src.exportData<nvcv::TensorDataStridedCuda>();
    auto clippedData = clipped.exportData<nvcv::TensorDataStridedCuda>();
    auto wrappedData = wrapped.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && clippedData && wrappedData);
    auto srcAccess     = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto clippedAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*clippedData);
    auto wrappedAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*wrappedData);
    ASSERT_TRUE(srcAccess && clippedAccess && wrappedAccess);

    std::vector<uint8_t> input{250, 1, 127, 0, 245, 255};
    nvcv::test::planar::UploadInterleavedSample(*srcAccess, 0, input, width, height, rowStride);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::GaussianNoise op(1);
    op(stream, src, clipped, 10.75f, 0.f, true, 12345, true, true);
    op(stream, src, wrapped, 10.75f, 0.f, true, 12345, true, false);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    EXPECT_EQ((std::vector<uint8_t>{255, 11, 137, 10, 255, 255}),
              nvcv::test::planar::DownloadInterleavedSample(*clippedAccess, 0, width, height, rowStride));
    EXPECT_EQ((std::vector<uint8_t>{4, 11, 137, 10, 255, 9}),
              nvcv::test::planar::DownloadInterleavedSample(*wrappedAccess, 0, width, height, rowStride));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGaussianNoiseScalar, float_clip_is_optional)
{
    constexpr int width     = 1;
    constexpr int height    = 1;
    constexpr int rowStride = 3 * sizeof(float);
    nvcv::Tensor  src       = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor  clipped   = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor  raw       = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);

    auto srcData     = src.exportData<nvcv::TensorDataStridedCuda>();
    auto clippedData = clipped.exportData<nvcv::TensorDataStridedCuda>();
    auto rawData     = raw.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && clippedData && rawData);
    auto srcAccess     = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto clippedAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*clippedData);
    auto rawAccess     = nvcv::TensorDataAccessStridedImagePlanar::Create(*rawData);
    ASSERT_TRUE(srcAccess && clippedAccess && rawAccess);

    std::vector<float>   input{0.9f, -0.2f, 0.5f};
    std::vector<uint8_t> inputBytes(input.size() * sizeof(float));
    std::memcpy(inputBytes.data(), input.data(), inputBytes.size());
    nvcv::test::planar::UploadInterleavedSample(*srcAccess, 0, inputBytes, width, height, rowStride);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::GaussianNoise op(1);
    op(stream, src, clipped, 0.3f, 0.f, true, 12345, true, true);
    op(stream, src, raw, 0.3f, 0.f, true, 12345, true, false);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto clippedBytes = nvcv::test::planar::DownloadInterleavedSample(*clippedAccess, 0, width, height, rowStride);
    auto rawBytes     = nvcv::test::planar::DownloadInterleavedSample(*rawAccess, 0, width, height, rowStride);
    std::vector<float> clippedValues(input.size());
    std::vector<float> rawValues(input.size());
    std::memcpy(clippedValues.data(), clippedBytes.data(), clippedBytes.size());
    std::memcpy(rawValues.data(), rawBytes.data(), rawBytes.size());

    EXPECT_FLOAT_EQ(1.f, clippedValues[0]);
    EXPECT_FLOAT_EQ(0.1f, clippedValues[1]);
    EXPECT_FLOAT_EQ(0.8f, clippedValues[2]);
    EXPECT_FLOAT_EQ(1.2f, rawValues[0]);
    EXPECT_FLOAT_EQ(0.1f, rawValues[1]);
    EXPECT_FLOAT_EQ(0.8f, rawValues[2]);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGaussianNoiseScalar, explicit_seed_reseeds_while_implicit_seed_advances)
{
    constexpr int width     = 37;
    constexpr int height    = 29;
    constexpr int rowStride = width * 3 * sizeof(float);
    nvcv::Tensor  src       = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor  first     = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor  repeated  = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor  advanced  = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);

    auto srcData      = src.exportData<nvcv::TensorDataStridedCuda>();
    auto firstData    = first.exportData<nvcv::TensorDataStridedCuda>();
    auto repeatedData = repeated.exportData<nvcv::TensorDataStridedCuda>();
    auto advancedData = advanced.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && firstData && repeatedData && advancedData);
    auto srcAccess      = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto firstAccess    = nvcv::TensorDataAccessStridedImagePlanar::Create(*firstData);
    auto repeatedAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*repeatedData);
    auto advancedAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*advancedData);
    ASSERT_TRUE(srcAccess && firstAccess && repeatedAccess && advancedAccess);

    auto input = MakeGaussianNoiseInput(nvcv::FMT_RGBf32, width, height, 7);
    nvcv::test::planar::UploadInterleavedSample(*srcAccess, 0, input, width, height, rowStride);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::GaussianNoise op(1);
    op(stream, src, first, 0.f, 0.05f, true, 98765, true, true);
    op(stream, src, repeated, 0.f, 0.05f, true, 98765, true, true);
    op(stream, src, advanced, 0.f, 0.05f, true, 98765, false, true);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto firstBytes    = nvcv::test::planar::DownloadInterleavedSample(*firstAccess, 0, width, height, rowStride);
    auto repeatedBytes = nvcv::test::planar::DownloadInterleavedSample(*repeatedAccess, 0, width, height, rowStride);
    auto advancedBytes = nvcv::test::planar::DownloadInterleavedSample(*advancedAccess, 0, width, height, rowStride);
    EXPECT_EQ(firstBytes, repeatedBytes);
    EXPECT_NE(repeatedBytes, advancedBytes);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGaussianNoiseScalar, implicit_seed_continues_after_explicit_seed)
{
    constexpr int width      = 37;
    constexpr int height     = 29;
    constexpr int rowStride  = width * 3 * sizeof(float);
    nvcv::Tensor  src        = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor  primedA    = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor  primedB    = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor  continuedA = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor  continuedB = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);

    auto srcData        = src.exportData<nvcv::TensorDataStridedCuda>();
    auto continuedAData = continuedA.exportData<nvcv::TensorDataStridedCuda>();
    auto continuedBData = continuedB.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && continuedAData && continuedBData);
    auto srcAccess        = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto continuedAAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*continuedAData);
    auto continuedBAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*continuedBData);
    ASSERT_TRUE(srcAccess && continuedAAccess && continuedBAccess);

    auto input = MakeGaussianNoiseInput(nvcv::FMT_RGBf32, width, height, 7);
    nvcv::test::planar::UploadInterleavedSample(*srcAccess, 0, input, width, height, rowStride);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::GaussianNoise implicitSeedOp(1);
    cvcuda::GaussianNoise explicitSeedOp(1);
    implicitSeedOp(stream, src, primedA, 0.f, 0.05f, true, 98765, true, true);
    implicitSeedOp(stream, src, continuedA, 0.f, 0.05f, true, 0, false, true);
    explicitSeedOp(stream, src, primedB, 0.f, 0.05f, true, 98765, true, true);
    explicitSeedOp(stream, src, continuedB, 0.f, 0.05f, true, 98765, false, true);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    EXPECT_EQ(nvcv::test::planar::DownloadInterleavedSample(*continuedAAccess, 0, width, height, rowStride),
              nvcv::test::planar::DownloadInterleavedSample(*continuedBAccess, 0, width, height, rowStride));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGaussianNoiseScalar, planar_matches_interleaved)
{
    cvcuda::GaussianNoise op(2);
    nvcv::test::planar::RunTensorParity(
        nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 31, 27, 31, 27, 2,
        [&op](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        { op(stream, src, dst, 3.f, 25.f, true, 12345, true, true); });
}

TEST(OpGaussianNoiseScalar_Negative, rejects_negative_sigma_and_unsupported_dtype)
{
    nvcv::Tensor          srcU8  = nvcv::util::CreateTensor(1, 4, 4, nvcv::FMT_RGB8);
    nvcv::Tensor          dstU8  = nvcv::util::CreateTensor(1, 4, 4, nvcv::FMT_RGB8);
    nvcv::Tensor          srcF16 = nvcv::util::CreateTensor(1, 4, 4, nvcv::FMT_RGBf16);
    nvcv::Tensor          dstF16 = nvcv::util::CreateTensor(1, 4, 4, nvcv::FMT_RGBf16);
    cvcuda::GaussianNoise op(1);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &srcU8, &dstU8] { op(nullptr, srcU8, dstU8, 0.f, -1.f, true, 0, true, true); }));
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&op, &srcF16, &dstF16] { op(nullptr, srcF16, dstF16, 0.f, 1.f, true, 0, true, true); }));
}

TEST(OpGaussianNoise_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaGaussianNoiseCreate(nullptr, 10));
}

TEST(OpGaussianNoise_Negative, create_with_negative_batch)
{
    NVCVOperatorHandle opHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaGaussianNoiseCreate(&opHandle, -1));
}

TEST(OpGaussianNoise_Negative, invalid_mu_sigma_layout)
{
    nvcv::Tensor imgIn(
        {
            {24, 24, 2},
            "HWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor imgOut(
        {
            {24, 24, 2},
            "HWC"
    },
        nvcv::TYPE_U8);

    //parameters
    nvcv::Tensor muval({{2}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor sigmaval({{2}, "N"}, nvcv::TYPE_F32);

    // invalid mu parameters
    nvcv::Tensor invalidMuval(
        {
            {2, 2, 2},
            "HWC"
    },
        nvcv::TYPE_F32);
    nvcv::Tensor invalidSigmaval(
        {
            {2, 2, 2},
            "HWC"
    },
        nvcv::TYPE_F32);

    // Call operator
    int                   maxBatch = 4;
    unsigned long long    seed     = 12345;
    cvcuda::GaussianNoise GaussianNoiseOp(maxBatch);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&GaussianNoiseOp, &imgIn, &imgOut, &invalidMuval, &sigmaval, &seed]
                                { GaussianNoiseOp(nullptr, imgIn, imgOut, invalidMuval, sigmaval, false, seed); }));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&GaussianNoiseOp, &imgIn, &imgOut, &muval, &invalidSigmaval, &seed]
                                { GaussianNoiseOp(nullptr, imgIn, imgOut, muval, invalidSigmaval, false, seed); }));
}

static void FillGaussianNoiseParams(const nvcv::Tensor &mu, const nvcv::Tensor &sigma, int batch, cudaStream_t stream)
{
    auto muData    = mu.exportData<nvcv::TensorDataStridedCuda>();
    auto sigmaData = sigma.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, muData);
    ASSERT_NE(nullptr, sigmaData);

    std::vector<float> muVec(batch, 0.f);
    std::vector<float> sigmaVec(batch, 0.005f);

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(muData->basePtr(), muVec.data(), muVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(sigmaData->basePtr(), sigmaVec.data(), sigmaVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
}

TEST(OpGaussianNoise_Negative, tensor_batch_exceeds_maxBatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int batch    = 2;
    constexpr int maxBatch = 1;

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(batch, 4, 4, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(batch, 4, 4, nvcv::FMT_RGB8);
    nvcv::Tensor muval({{batch}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor sigmaval({{batch}, "N"}, nvcv::TYPE_F32);

    FillGaussianNoiseParams(muval, sigmaval, batch, stream);

    cvcuda::GaussianNoise gaussianNoiseOp(maxBatch);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&gaussianNoiseOp, &stream, &imgIn, &imgOut, &muval, &sigmaval]
                                { gaussianNoiseOp(stream, imgIn, imgOut, muval, sigmaval, false, 12345); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGaussianNoiseVarShape_Negative, varshape_batch_exceeds_maxBatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int batch    = 2;
    constexpr int maxBatch = 1;

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batch; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{4, 4}, nvcv::FMT_RGB8);
        imgDst.emplace_back(nvcv::Size2D{4, 4}, nvcv::FMT_RGB8);
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor muval({{batch}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor sigmaval({{batch}, "N"}, nvcv::TYPE_F32);

    FillGaussianNoiseParams(muval, sigmaval, batch, stream);

    cvcuda::GaussianNoise gaussianNoiseOp(maxBatch);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&gaussianNoiseOp, &stream, &batchSrc, &batchDst, &muval, &sigmaval]
                                { gaussianNoiseOp(stream, batchSrc, batchDst, muval, sigmaval, false, 12345); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpGaussianNoise_Negative, nvcv::test::ValueList<std::string, nvcv::DataType, std::string, nvcv::DataType, std::string, nvcv::DataType, std::string, nvcv::DataType>
{
    //   in_layout,        in_data_type,   out_layout,     out_data_type,     mu_layout,         mu_data_type,    sigma_layout,    sigma_data_type,    expected_return_status
    {        "CHW",       nvcv::TYPE_U8,        "HWC",     nvcv::TYPE_U8,           "N",       nvcv::TYPE_F32,             "N",     nvcv::TYPE_F32},
    {        "HWC",       nvcv::TYPE_U8,        "CHW",     nvcv::TYPE_U8,           "N",       nvcv::TYPE_F32,             "N",     nvcv::TYPE_F32},
    {        "HWC",      nvcv::TYPE_F64,        "HWC",     nvcv::TYPE_U8,           "N",       nvcv::TYPE_F32,             "N",     nvcv::TYPE_F32},
    {        "HWC",       nvcv::TYPE_U8,        "HWC",    nvcv::TYPE_F64,           "N",       nvcv::TYPE_F32,             "N",     nvcv::TYPE_F32},
    {        "HWC",      nvcv::TYPE_U32,        "HWC",     nvcv::TYPE_U8,           "N",       nvcv::TYPE_F32,             "N",     nvcv::TYPE_F32},
    {        "HWC",       nvcv::TYPE_U8,        "HWC",    nvcv::TYPE_U32,           "N",       nvcv::TYPE_F32,             "N",     nvcv::TYPE_F32},
    {        "HWC",       nvcv::TYPE_U8,        "HWC",    nvcv::TYPE_U16,           "N",       nvcv::TYPE_F32,             "N",     nvcv::TYPE_F32},
    {        "HWC",       nvcv::TYPE_U8,        "HWC",     nvcv::TYPE_U8,           "N",       nvcv::TYPE_F64,             "N",     nvcv::TYPE_F32},
    {        "HWC",       nvcv::TYPE_U8,        "HWC",     nvcv::TYPE_U8,           "N",       nvcv::TYPE_F32,             "N",     nvcv::TYPE_F64},
});

NVCV_TEST_SUITE_P(OpGaussianNoiseVarShape_Negative, nvcv::test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, std::string, nvcv::DataType, std::string, nvcv::DataType>
{
    // inFmt, outFmt, mu_layout, mu_data_type, sigma_layout, sigma_data_type
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, "N", nvcv::TYPE_F32, "N", nvcv::TYPE_F32},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, "N", nvcv::TYPE_F32, "N", nvcv::TYPE_F32},
    {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, "N", nvcv::TYPE_F32, "N", nvcv::TYPE_F32},
    {nvcv::FMT_RGB8, nvcv::FMT_RGBf32, "N", nvcv::TYPE_F32, "N", nvcv::TYPE_F32},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_F64, "N", nvcv::TYPE_F32},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_F32, "N", nvcv::TYPE_F64},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, "NW", nvcv::TYPE_F32, "N", nvcv::TYPE_F32},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_F32, "NW", nvcv::TYPE_F32},
});

// clang-format on

TEST_P(OpGaussianNoise_Negative, op)
{
    std::string    in_layout       = GetParamValue<0>();
    nvcv::DataType in_data_type    = GetParamValue<1>();
    std::string    out_layout      = GetParamValue<2>();
    nvcv::DataType out_data_type   = GetParamValue<3>();
    std::string    mu_layout       = GetParamValue<4>();
    nvcv::DataType mu_data_type    = GetParamValue<5>();
    std::string    sigma_layout    = GetParamValue<6>();
    nvcv::DataType sigma_data_type = GetParamValue<7>();

    nvcv::Tensor imgIn(
        {
            {24, 24, 2},
            in_layout.c_str()
    },
        in_data_type);
    nvcv::Tensor imgOut(
        {
            {24, 24, 2},
            out_layout.c_str()
    },
        out_data_type);

    //parameters
    nvcv::Tensor muval({{2}, mu_layout.c_str()}, mu_data_type);
    nvcv::Tensor sigmaval({{2}, sigma_layout.c_str()}, sigma_data_type);

    // Call operator
    int                   maxBatch = 4;
    unsigned long long    seed     = 12345;
    cvcuda::GaussianNoise GaussianNoiseOp(maxBatch);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&GaussianNoiseOp, &imgIn, &imgOut, &muval, &sigmaval, &seed]
                                { GaussianNoiseOp(nullptr, imgIn, imgOut, muval, sigmaval, false, seed); }));
}

TEST_P(OpGaussianNoiseVarShape_Negative, op)
{
    nvcv::ImageFormat inFmt           = GetParamValue<0>();
    nvcv::ImageFormat outFmt          = GetParamValue<1>();
    std::string       mu_layout       = GetParamValue<2>();
    nvcv::DataType    mu_data_type    = GetParamValue<3>();
    std::string       sigma_layout    = GetParamValue<4>();
    nvcv::DataType    sigma_data_type = GetParamValue<5>();

    int width  = 24;
    int height = 24;
    int batch  = 3;

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

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
        imgSrc.emplace_back(nvcv::Size2D{rw, rh}, inFmt);
        imgDst.emplace_back(nvcv::Size2D{rw, rh}, outFmt);
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    //parameters
    nvcv::TensorShape muValShape    = mu_layout.size() == 1 ? nvcv::TensorShape({2}, mu_layout.c_str())
                                                            : nvcv::TensorShape({2, 1}, mu_layout.c_str());
    nvcv::TensorShape sigmaValShape = sigma_layout.size() == 1 ? nvcv::TensorShape({2}, sigma_layout.c_str())
                                                               : nvcv::TensorShape({2, 1}, sigma_layout.c_str());
    nvcv::Tensor      muVal(muValShape, mu_data_type);
    nvcv::Tensor      sigmaVal(sigmaValShape, sigma_data_type);

    // Call operator
    int                   maxBatch = 4;
    unsigned long long    seed     = 12345;
    cvcuda::GaussianNoise GaussianNoiseOp(maxBatch);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&GaussianNoiseOp, &stream, &batchSrc, &batchDst, &muVal, &sigmaVal, &seed]
                                { GaussianNoiseOp(stream, batchSrc, batchDst, muVal, sigmaVal, false, seed); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

#undef NVCV_IMAGE_FORMAT_2U8

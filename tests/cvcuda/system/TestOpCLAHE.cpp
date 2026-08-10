/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpCLAHE.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace test = nvcv::test;
namespace util = nvcv::util;

namespace {

constexpr int kBins = 256;
using Histogram     = std::array<unsigned int, kBins>;

int Reflect101(int p, int len)
{
    if (len <= 1)
    {
        return 0;
    }
    while (p >= len)
    {
        p = 2 * len - p - 2;
    }
    return p;
}

Histogram BuildHistogram(const std::vector<uint8_t> &src, int width, int height, int x0, int y0, int tileW, int tileH)
{
    Histogram hist{};
    for (int yy = 0; yy < tileH; ++yy)
    {
        for (int xx = 0; xx < tileW; ++xx)
        {
            const int sx = Reflect101(x0 + xx, width);
            const int sy = Reflect101(y0 + yy, height);
            hist[src[sy * width + sx]]++;
        }
    }
    return hist;
}

void ClipHistogram(Histogram &hist, double clipLimit, int tileArea)
{
    const unsigned int clipCount = std::max((unsigned int)(clipLimit * (double)tileArea / (double)kBins), 1U);
    unsigned int       excess    = 0;
    for (auto &v : hist)
    {
        if (v > clipCount)
        {
            excess += v - clipCount;
            v = clipCount;
        }
    }

    constexpr unsigned int kBinsU = kBins;
    const unsigned int     redist = excess / kBinsU;
    const unsigned int     rem    = excess % kBinsU;
    for (auto &v : hist)
    {
        v += redist;
    }

    const unsigned int step = rem > 0U ? std::max(kBinsU / rem, 1U) : 1U;
    for (unsigned int i = 0; i < rem; ++i)
    {
        const unsigned int idx = i * step;
        if (idx < kBinsU)
        {
            hist[idx]++;
        }
    }
}

void StoreLut(std::vector<uint8_t> &luts, const Histogram &hist, int tileIndex, int tileArea)
{
    unsigned int cdf = 0;
    for (int i = 0; i < kBins; ++i)
    {
        cdf += hist[i];
        unsigned int lut            = (cdf * 255U + (unsigned int)(tileArea / 2)) / (unsigned int)std::max(tileArea, 1);
        lut                         = std::min(lut, 255U);
        luts[tileIndex * kBins + i] = (uint8_t)lut;
    }
}

std::vector<uint8_t> RefCLAHE(const std::vector<uint8_t> &src, int width, int height, double clipLimit, int tilesX,
                              int tilesY)
{
    const int padW     = (tilesX - (width % tilesX)) % tilesX;
    const int padH     = (tilesY - (height % tilesY)) % tilesY;
    const int extW     = width + padW;
    const int extH     = height + padH;
    const int tileW    = extW / tilesX;
    const int tileH    = extH / tilesY;
    const int tileArea = tileW * tileH;
    const int numTiles = tilesX * tilesY;

    std::vector<uint8_t> luts(numTiles * kBins, 0);
    for (int ty = 0; ty < tilesY; ++ty)
    {
        for (int tx = 0; tx < tilesX; ++tx)
        {
            Histogram hist      = BuildHistogram(src, width, height, tx * tileW, ty * tileH, tileW, tileH);
            const int tileIndex = ty * tilesX + tx;
            ClipHistogram(hist, clipLimit, tileArea);
            StoreLut(luts, hist, tileIndex, tileArea);
        }
    }

    std::vector<uint8_t> out(width * height);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float gx = (float)x / (float)tileW - 0.5f;
            float gy = (float)y / (float)tileH - 0.5f;
            gx       = std::min(std::max(gx, 0.0f), (float)(tilesX - 1));
            gy       = std::min(std::max(gy, 0.0f), (float)(tilesY - 1));

            const auto  tx0 = static_cast<int>(std::floor(gx));
            const auto  ty0 = static_cast<int>(std::floor(gy));
            const int   tx1 = std::min(tx0 + 1, tilesX - 1);
            const int   ty1 = std::min(ty0 + 1, tilesY - 1);
            const float fx  = gx - static_cast<float>(tx0);
            const float fy  = gy - static_cast<float>(ty0);

            const int v     = src[y * width + x];
            const int idx00 = (ty0 * tilesX + tx0) * kBins + v;
            const int idx10 = (ty0 * tilesX + tx1) * kBins + v;
            const int idx01 = (ty1 * tilesX + tx0) * kBins + v;
            const int idx11 = (ty1 * tilesX + tx1) * kBins + v;

            const float w00 = (1.0f - fx) * (1.0f - fy);
            const float w10 = fx * (1.0f - fy);
            const float w01 = (1.0f - fx) * fy;
            const float w11 = fx * fy;

            const float value  = w00 * luts[idx00] + w10 * luts[idx10] + w01 * luts[idx01] + w11 * luts[idx11];
            out[y * width + x] = (uint8_t)(value + 0.5f);
        }
    }

    return out;
}

void ExpectNearVec(const std::vector<uint8_t> &got, const std::vector<uint8_t> &gold, int maxAbsDiff)
{
    ASSERT_EQ(got.size(), gold.size());
    for (size_t i = 0; i < got.size(); ++i)
    {
        const int diff = std::abs((int)got[i] - (int)gold[i]);
        EXPECT_LE(diff, maxAbsDiff);
    }
}

void RunTensorLayoutParity(const char *interleavedLayout, const char *planarLayout, int width, int height, int batches,
                           double clip, int tilesX, int tilesY)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const bool hasBatch = planarLayout[0] == 'N';
    const int  samples  = hasBatch ? batches : 1;

    nvcv::Tensor srcI(hasBatch ? nvcv::TensorShape{{batches, height, width, 1}, interleavedLayout}
                               : nvcv::TensorShape{{height, width, 1}, interleavedLayout},
                      nvcv::TYPE_U8);
    nvcv::Tensor dstI(hasBatch ? nvcv::TensorShape{{batches, height, width, 1}, interleavedLayout}
                               : nvcv::TensorShape{{height, width, 1}, interleavedLayout},
                      nvcv::TYPE_U8);
    nvcv::Tensor srcP(hasBatch ? nvcv::TensorShape{{batches, 1, height, width}, planarLayout}
                               : nvcv::TensorShape{{1, height, width}, planarLayout},
                      nvcv::TYPE_U8);
    nvcv::Tensor dstP(hasBatch ? nvcv::TensorShape{{batches, 1, height, width}, planarLayout}
                               : nvcv::TensorShape{{1, height, width}, planarLayout},
                      nvcv::TYPE_U8);

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

    for (int i = 0; i < samples; ++i)
    {
        std::vector<uint8_t> src(width * height);
        test::planar::FillDeterministicValues(src, static_cast<size_t>(i) * 101 + 17, nvcv::TYPE_U8);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcIAcc->sampleData(i), srcIAcc->rowStride(), src.data(), width, width,
                                            height, cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcPAcc->sampleData(i), srcPAcc->rowStride(), src.data(), width, width,
                                            height, cudaMemcpyHostToDevice));
    }

    cvcuda::CLAHE op(samples, tilesX, tilesY);
    EXPECT_NO_THROW(op(stream, srcI, dstI, clip));
    EXPECT_NO_THROW(op(stream, srcP, dstP, clip));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < samples; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<uint8_t> gotI(width * height);
        std::vector<uint8_t> gotP(width * height);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(gotI.data(), width, dstIAcc->sampleData(i), dstIAcc->rowStride(), width,
                                            height, cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(gotP.data(), width, dstPAcc->sampleData(i), dstPAcc->rowStride(), width,
                                            height, cudaMemcpyDeviceToHost));
        EXPECT_EQ(gotI, gotP);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

} // namespace

// clang-format off
NVCV_TEST_SUITE_P(OpCLAHE, test::ValueList<int, int, int, double, int, int>
{
    // width, height, batches, clip, tilesX, tilesY
    {    17,     19,       1,  40.0,      8,      8},
    {    40,     24,       2,   2.0,      8,      8},
    {   101,     67,       2,  40.0,      8,      8},
    {   320,    240,       3,   2.0,      7,      9},
    {   976,     32,       1,   2.0,      8,      8},
});

// clang-format on

TEST_P(OpCLAHE, tensor_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    width   = GetParamValue<0>();
    const int    height  = GetParamValue<1>();
    const int    batches = GetParamValue<2>();
    const double clip    = GetParamValue<3>();
    const int    tilesX  = GetParamValue<4>();
    const int    tilesY  = GetParamValue<5>();

    nvcv::Tensor in  = util::CreateTensor(batches, width, height, nvcv::FMT_U8);
    nvcv::Tensor out = util::CreateTensor(batches, width, height, nvcv::FMT_U8);

    std::default_random_engine    rng(0);
    std::uniform_int_distribution rand(0, 255);
    for (int i = 0; i < batches; ++i) // NOSONAR
    {
        std::vector<uint8_t> src(width * height);
        std::ranges::generate(src, [&rand, &rng] { return (uint8_t)rand(rng); });
        EXPECT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(in.exportData(), src, i));
    }

    cvcuda::CLAHE op(batches, tilesX, tilesY);
    EXPECT_NO_THROW(op(stream, in, out, clip));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batches; ++i) // NOSONAR
    {
        std::vector<uint8_t> src;
        std::vector<uint8_t> got;
        EXPECT_NO_THROW(util::GetImageVectorFromTensor(in.exportData(), i, src));
        EXPECT_NO_THROW(util::GetImageVectorFromTensor(out.exportData(), i, got));
        const auto gold = RefCLAHE(src, width, height, clip, tilesX, tilesY);
        // GPU FMA contraction in the bilinear interpolation can differ from the CPU reference by one LSB.
        ExpectNearVec(got, gold, 1);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCLAHE, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    batches = 4;
    const double clip    = 40.0;
    const int    tilesX  = 8;
    const int    tilesY  = 8;

    std::vector<nvcv::Image>          srcImages;
    std::vector<nvcv::Image>          dstImages;
    std::vector<std::vector<uint8_t>> srcVec(batches);

    std::default_random_engine    rng(0);
    std::uniform_int_distribution randVal(0, 255);
    std::uniform_int_distribution randW(31, 97);
    std::uniform_int_distribution randH(29, 111);

    for (int i = 0; i < batches; ++i)
    {
        nvcv::Size2D size{randW(rng), randH(rng)};
        srcImages.emplace_back(size, nvcv::FMT_U8);
        dstImages.emplace_back(size, nvcv::FMT_U8);

        srcVec[i].resize(size.w * size.h);
        std::ranges::generate(srcVec[i], [&randVal, &rng] { return (uint8_t)randVal(rng); });

        auto srcData = srcImages.back().exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(srcData, nvcv::NullOpt);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(), size.w,
                                    size.w, size.h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape srcBatch(batches);

    nvcv::ImageBatchVarShape dstBatch(batches);
    srcBatch.pushBack(srcImages.begin(), srcImages.end());
    dstBatch.pushBack(dstImages.begin(), dstImages.end());

    cvcuda::CLAHE op(batches, tilesX, tilesY);
    EXPECT_NO_THROW(op(stream, srcBatch, dstBatch, clip));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batches; ++i)
    {
        auto dstData = dstImages[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(dstData, nvcv::NullOpt);

        const int            width  = dstImages[i].size().w;
        const int            height = dstImages[i].size().h;
        std::vector<uint8_t> got(width * height);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(got.data(), width, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                                            width, height, cudaMemcpyDeviceToHost));

        const auto gold = RefCLAHE(srcVec[i], width, height, clip, tilesX, tilesY);
        // GPU FMA contraction in the bilinear interpolation can differ from the CPU reference by one LSB.
        ExpectNearVec(got, gold, 1);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCLAHEPlanar, tensor_nchw_matches_interleaved)
{
    RunTensorLayoutParity("NHWC", "NCHW", 101, 67, 2, 2.0, 7, 9);
}

TEST(OpCLAHEPlanar, tensor_chw_matches_interleaved)
{
    RunTensorLayoutParity("HWC", "CHW", 37, 29, 1, 40.0, 8, 8);
}

TEST(OpCLAHEPlanar, varshape_single_plane_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        nvcv::FMT_U8, nvcv::FMT_U8, 53, 47, 53, 47, 3,
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::CLAHE op(src.numImages(), 7, 9);
            EXPECT_NO_THROW(op(s, src, dst, 2.0));
        });
}

TEST(OpCLAHE_Negative, invalid_input_format_or_type)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::CLAHE op(2, 8, 8);

    {
        nvcv::Tensor in  = util::CreateTensor(2, 32, 32, nvcv::FMT_RGB8);
        nvcv::Tensor out = util::CreateTensor(2, 32, 32, nvcv::FMT_RGB8);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&op, &stream, &in, &out] { op(stream, in, out, 40.0); }));
    }
    {
        nvcv::Tensor in  = util::CreateTensor(2, 32, 32, nvcv::FMT_F16);
        nvcv::Tensor out = util::CreateTensor(2, 32, 32, nvcv::FMT_F16);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&op, &stream, &in, &out] { op(stream, in, out, 40.0); }));
    }
    {
        nvcv::Tensor in  = util::CreateTensor(2, 32, 32, nvcv::FMT_U8);
        nvcv::Tensor out = util::CreateTensor(2, 32, 32, nvcv::FMT_U8);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&op, &stream, &in, &out] { op(stream, in, out, 0.0); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCLAHE_Negative, create_invalid_arguments)
{
    NVCVOperatorHandle handle;
    EXPECT_EQ(cvcudaCLAHECreate(nullptr, 1, 8, 8), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaCLAHECreate(&handle, 0, 8, 8), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaCLAHECreate(&handle, 1, 0, 8), NVCV_ERROR_INVALID_ARGUMENT);
}

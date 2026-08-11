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
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpNormalize.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/TensorShape.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>

namespace test = nvcv::test;
namespace t    = ::testing;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

struct NormalizeRefData
{
    std::vector<uint8_t>       &hDst;
    int                         dstRowStride;
    const std::vector<uint8_t> &hSrc;
    int                         srcRowStride;
    nvcv::Size2D                size;
    nvcv::ImageFormat           fmt;
    const std::vector<float>   &hBase;
    int                         baseRowStride;
    nvcv::Size2D                baseSize;
    nvcv::ImageFormat           baseFormat;
    const std::vector<float>   &hScale;
    int                         scaleRowStride;
    nvcv::Size2D                scaleSize;
    nvcv::ImageFormat           scaleFormat;
    float                       globalScale;
    float                       globalShift;
    float                       epsilon;
    uint32_t                    flags;
};

static int BroadcastIndex(int index, int size)
{
    return size == 1 ? 0 : index;
}

static float NormalizeMultiplier(const NormalizeRefData &ref, int y, int x, int channel)
{
    using FT = float;

    const int si = BroadcastIndex(y, ref.scaleSize.h);
    const int sj = BroadcastIndex(x, ref.scaleSize.w);
    const int sk = BroadcastIndex(channel, ref.scaleFormat.numChannels());

    FT scale = ref.hScale.at(si * ref.scaleRowStride + sj * ref.scaleFormat.numChannels() + sk);
    if (ref.flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
    {
        return FT{1} / std::sqrt(scale * scale + ref.epsilon);
    }

    return scale;
}

static uint8_t SaturateToU8(float value)
{
    if (value < 0)
    {
        return 0;
    }

    if (value > 255)
    {
        return 255;
    }

    return static_cast<uint8_t>(value);
}

static uint8_t NormalizeValue(const NormalizeRefData &ref, int y, int x, int channel)
{
    using FT = float;

    const int bi = BroadcastIndex(y, ref.baseSize.h);
    const int bj = BroadcastIndex(x, ref.baseSize.w);
    const int bk = BroadcastIndex(channel, ref.baseFormat.numChannels());

    const FT src  = ref.hSrc.at(y * ref.srcRowStride + x * ref.fmt.numChannels() + channel);
    const FT base = ref.hBase.at(bi * ref.baseRowStride + bj * ref.baseFormat.numChannels() + bk);
    const FT mul  = NormalizeMultiplier(ref, y, x, channel);

    return SaturateToU8(std::rint((src - base) * mul * ref.globalScale + ref.globalShift));
}

static void Normalize(NormalizeRefData ref)
{
    for (int i = 0; i < ref.size.h; i++)
    {
        for (int j = 0; j < ref.size.w; j++)
        {
            for (int k = 0; k < ref.fmt.numChannels(); k++)
            {
                ref.hDst.at(i * ref.dstRowStride + j * ref.fmt.numChannels() + k) = NormalizeValue(ref, i, j, k);
            }
        }
    }
}

static constexpr uint32_t normalScale   = 0;
static constexpr uint32_t scaleIsStdDev = CVCUDA_NORMALIZE_SCALE_IS_STDDEV;

template<typename T>
static T ReferenceNormalizeCast(float value)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return static_cast<T>(value);
    }
    else
    {
        float rounded = std::rint(value);
        float clamped = std::min(std::max(rounded, 0.f), 255.f);
        return static_cast<T>(clamped);
    }
}

template<typename T>
static T MakePlanarInputValue(int sample, int y, int x, int channel)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return static_cast<T>(10.f + static_cast<float>(sample) * 0.25f + static_cast<float>(y) * 0.5f
                              + static_cast<float>(x) * 0.125f + static_cast<float>(channel));
    }
    else
    {
        return static_cast<T>((sample * 17 + y * 5 + x * 3 + channel * 11) % 211);
    }
}

template<typename T>
static std::vector<T> MakePlanarHostImage(int sample, int width, int height, int channels)
{
    std::vector<T> image(height * width * channels);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                image[y * width * channels + x * channels + c] = MakePlanarInputValue<T>(sample, y, x, c);
            }
        }
    }
    return image;
}

static void FillPlanarParamTensor(nvcv::Tensor &tensor, const std::vector<float> &values)
{
    auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, data);
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
    ASSERT_TRUE(access);
    ASSERT_EQ(static_cast<int>(values.size()), access->numChannels());

    for (int c = 0; c < access->numChannels(); ++c)
    {
        nvcv::Byte *p = access->sampleData(0) + c * access->chStride();
        ASSERT_EQ(cudaSuccess, cudaMemcpy(p, &values[c], sizeof(float), cudaMemcpyHostToDevice));
    }
}

// Deinterleave an HWC host image into per-channel planes and upload each plane to the device.
// planeDst(c) returns the {device pointer, row stride in bytes} for channel c's plane, so the same
// code serves both planar tensors (channel stride) and multi-plane images (per-plane base ptr).
template<typename T, typename PlaneDstFn>
static void UploadDeinterleavedPlanes(const std::vector<T> &hostImage, int width, int height, int channels,
                                      PlaneDstFn planeDst)
{
    std::vector<T> plane(static_cast<size_t>(height) * width);
    for (int c = 0; c < channels; ++c)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                plane[y * width + x] = hostImage[y * width * channels + x * channels + c];
            }
        }

        auto [ptr, rowStride] = planeDst(c);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(ptr, rowStride, plane.data(), width * sizeof(T), width * sizeof(T), height,
                                            cudaMemcpyHostToDevice));
    }
}

// Inverse of UploadDeinterleavedPlanes: download per-channel planes and interleave into HWC.
template<typename T, typename PlaneSrcFn>
static std::vector<T> DownloadInterleavedPlanes(int width, int height, int channels, PlaneSrcFn planeSrc)
{
    std::vector<T> image(static_cast<size_t>(height) * width * channels);
    std::vector<T> plane(static_cast<size_t>(height) * width);
    for (int c = 0; c < channels; ++c)
    {
        auto [ptr, rowStride] = planeSrc(c);
        EXPECT_EQ(cudaSuccess, cudaMemcpy2D(plane.data(), width * sizeof(T), ptr, rowStride, width * sizeof(T), height,
                                            cudaMemcpyDeviceToHost));
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                image[y * width * channels + x * channels + c] = plane[y * width + x];
            }
        }
    }
    return image;
}

template<typename T>
static void UploadPlanarTensorSample(nvcv::TensorDataAccessStridedImagePlanar &access, int sample,
                                     const std::vector<T> &image, int width, int height, int channels)
{
    UploadDeinterleavedPlanes<T>(
        image, width, height, channels,
        [&access, sample](int c)
        { return std::make_pair(access.sampleData(sample) + c * access.chStride(), access.rowStride()); });
}

template<typename T>
static void UploadPlanarImage(nvcv::Image &image, const std::vector<T> &hostImage)
{
    auto imgData = image.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(imgData, nvcv::NullOpt);

    UploadDeinterleavedPlanes<T>(hostImage, image.size().w, image.size().h, image.format().numChannels(),
                                 [&imgData](int c)
                                 { return std::make_pair(imgData->plane(c).basePtr, imgData->plane(c).rowStride); });
}

template<typename T>
static std::vector<T> DownloadPlanarTensorSample(nvcv::TensorDataAccessStridedImagePlanar &access, int sample,
                                                 int width, int height, int channels)
{
    return DownloadInterleavedPlanes<T>(
        width, height, channels,
        [&access, sample](int c)
        { return std::make_pair(access.sampleData(sample) + c * access.chStride(), access.rowStride()); });
}

template<typename T>
static std::vector<T> DownloadPlanarImage(nvcv::Image &image)
{
    auto imgData = image.exportData<nvcv::ImageDataStridedCuda>();
    EXPECT_NE(imgData, nvcv::NullOpt);

    return DownloadInterleavedPlanes<T>(
        image.size().w, image.size().h, image.format().numChannels(),
        [&imgData](int c) { return std::make_pair(imgData->plane(c).basePtr, imgData->plane(c).rowStride); });
}

template<typename T>
static std::vector<T> MakeNormalizeGold(const std::vector<T> &src, int width, int height, int channels,
                                        const std::vector<float> &base, const std::vector<float> &scale,
                                        float globalScale, float globalShift, uint32_t flags = 0, float epsilon = 0.f)
{
    std::vector<T> gold(src.size());
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                int         idx   = y * width * channels + x * channels + c;
                const float mul   = (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
                                      ? 1.f / std::sqrt(scale[c] * scale[c] + epsilon)
                                      : scale[c];
                float       value = (static_cast<float>(src[idx]) - base[c]) * mul * globalScale + globalShift;
                gold[idx]         = ReferenceNormalizeCast<T>(value);
            }
        }
    }
    return gold;
}

template<typename T>
static void ExpectPlanarVectorsNear(const std::vector<T> &gold, const std::vector<T> &test)
{
    ASSERT_EQ(gold.size(), test.size());
    if constexpr (std::is_floating_point_v<T>)
    {
        for (size_t i = 0; i < gold.size(); ++i)
        {
            EXPECT_NEAR(gold[i], test[i], 1e-4f) << "at flat index " << i;
        }
    }
    else
    {
        EXPECT_EQ(gold, test);
    }
}

// Create a 1x1 planar param (base or scale) tensor of the given format, fill `channels` random
// floats in [0, hi], upload one float per channel via the tensor's channel stride, and return the
// host values for use in the reference computation.
static std::vector<float> MakeAndUploadRandomParam(nvcv::Tensor &param, nvcv::ImageFormat fmt, int channels, float hi,
                                                   std::default_random_engine &rng)
{
    param = nvcv::util::CreateTensor(1, 1, 1, fmt);

    auto data = param.exportData<nvcv::TensorDataStridedCuda>();
    EXPECT_NE(nullptr, data);
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
    EXPECT_TRUE(access);

    std::vector<float>                    values(channels);
    std::uniform_real_distribution<float> udist(0.f, hi);
    std::ranges::generate(values, [&]() { return udist(rng); });

    for (int c = 0; c < channels; ++c)
    {
        nvcv::Byte *p = access->sampleData(0) + c * access->chStride();
        EXPECT_EQ(cudaSuccess, cudaMemcpy(p, &values[c], sizeof(float), cudaMemcpyHostToDevice));
    }
    return values;
}

static std::vector<float> MakeAndUploadSpatialPlanarParam(nvcv::Tensor &param, int width, int height, int channels,
                                                          float hi)
{
    param = nvcv::Tensor(
        {
            {1, channels, height, width},
            "NCHW"
    },
        nvcv::TYPE_F32);

    std::vector<float> values(static_cast<size_t>(width) * height * channels);
    const float        valueScale = hi / static_cast<float>(values.size() + 1);
    std::ranges::generate(
        values, [index = size_t{0}, valueScale]() mutable { return static_cast<float>(++index) * valueScale; });

    auto data = param.exportData<nvcv::TensorDataStridedCuda>();
    EXPECT_NE(nullptr, data);
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
    EXPECT_TRUE(access);
    if (access)
        UploadPlanarTensorSample<float>(*access, 0, values, width, height, channels);
    return values;
}

// Compute the interleaved reference for one planar sample via the shared Normalize() gold and
// compare it against the operator's (re-interleaved) output. base/scale are 1x1 spatially; their
// channel layout is described by {base,scale}HelperFmt.
static void ExpectPlanarNormalizeMatchesGold(const std::vector<uint8_t> &testVec, const std::vector<uint8_t> &srcVec,
                                             int rowStride, int width, int height, nvcv::ImageFormat fmt,
                                             const std::vector<float> &baseVec, int baseRowStride,
                                             nvcv::Size2D baseSize, nvcv::ImageFormat baseHelperFmt,
                                             const std::vector<float> &scaleVec, int scaleRowStride,
                                             nvcv::Size2D scaleSize, nvcv::ImageFormat scaleHelperFmt,
                                             float globalScale, float globalShift, float epsilon, uint32_t flags)
{
    std::vector<uint8_t> goldVec(static_cast<size_t>(height) * rowStride);
    Normalize({
        goldVec,
        rowStride,
        srcVec,
        rowStride,
        {width, height},
        fmt,
        baseVec,
        baseRowStride,
        baseSize,
        baseHelperFmt,
        scaleVec,
        scaleRowStride,
        scaleSize,
        scaleHelperFmt,
        globalScale,
        globalShift,
        epsilon,
        flags
    });
    EXPECT_EQ(goldVec, testVec);
}

// Deterministic per-channel base/scale (base = c+1, scale = 0.25*(c+1)) plus the global scale/shift
// used by the fixed-value planar correctness cases.
template<typename T>
static void MakeSequentialPlanarParams(int channels, std::vector<float> &base, std::vector<float> &scale,
                                       float &globalScale, float &globalShift)
{
    base.resize(channels);
    scale.resize(channels);
    for (int c = 0; c < channels; ++c)
    {
        base[c]  = static_cast<float>(c + 1);
        scale[c] = 0.25f * static_cast<float>(c + 1);
    }
    globalScale = 1.5f;
    globalShift = std::is_floating_point_v<T> ? -0.75f : 3.f;
}

template<typename T>
static void RunPlanarTensorNormalizeCase(nvcv::ImageFormat fmt, int width, int height, int numImages)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int channels = fmt.numChannels();

    std::vector<float> base;
    std::vector<float> scale;
    float              globalScale;
    float              globalShift;
    MakeSequentialPlanarParams<T>(channels, base, scale, globalScale, globalShift);

    nvcv::Tensor imgSrc(numImages, {width, height}, fmt);
    nvcv::Tensor imgDst(numImages, {width, height}, fmt);
    nvcv::Tensor imgBase(
        {
            {1, channels, 1, 1},
            "NCHW"
    },
        nvcv::TYPE_F32);
    nvcv::Tensor imgScale(
        {
            {1, channels, 1, 1},
            "NCHW"
    },
        nvcv::TYPE_F32);

    ASSERT_NO_FATAL_FAILURE(FillPlanarParamTensor(imgBase, base));
    ASSERT_NO_FATAL_FAILURE(FillPlanarParamTensor(imgScale, scale));

    auto srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<T>> srcVec(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        srcVec[i] = MakePlanarHostImage<T>(i, width, height, channels);
        ASSERT_NO_FATAL_FAILURE(UploadPlanarTensorSample(*srcAccess, i, srcVec[i], width, height, channels));
    }

    cvcuda::Normalize normalizeOp;
    EXPECT_NO_THROW(normalizeOp(stream, imgSrc, imgBase, imgScale, imgDst, globalScale, globalShift, 0.f, normalScale));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<T> test = DownloadPlanarTensorSample<T>(*dstAccess, i, width, height, channels);
        std::vector<T> gold
            = MakeNormalizeGold<T>(srcVec[i], width, height, channels, base, scale, globalScale, globalShift);
        ExpectPlanarVectorsNear(gold, test);
    }
}

template<typename T>
static void RunPlanarVarShapeNormalizeCase(nvcv::ImageFormat fmt, int width, int height, int numImages)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int channels = fmt.numChannels();

    std::vector<float> base;
    std::vector<float> scale;
    float              globalScale;
    float              globalShift;
    MakeSequentialPlanarParams<T>(channels, base, scale, globalScale, globalShift);

    std::vector<nvcv::Image>    imgSrc;
    std::vector<nvcv::Image>    imgDst;
    std::vector<std::vector<T>> srcVec(numImages);
    std::vector<nvcv::Size2D>   sizes(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        sizes[i] = nvcv::Size2D{width + i * 3, height + i * 2};
        imgSrc.emplace_back(sizes[i], fmt);
        imgDst.emplace_back(sizes[i], fmt);
        srcVec[i] = MakePlanarHostImage<T>(i, sizes[i].w, sizes[i].h, channels);
        ASSERT_NO_FATAL_FAILURE(UploadPlanarImage(imgSrc[i], srcVec[i]));
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor imgBase(
        {
            {1, channels, 1, 1},
            "NCHW"
    },
        nvcv::TYPE_F32);
    nvcv::Tensor imgScale(
        {
            {1, channels, 1, 1},
            "NCHW"
    },
        nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(FillPlanarParamTensor(imgBase, base));
    ASSERT_NO_FATAL_FAILURE(FillPlanarParamTensor(imgScale, scale));

    cvcuda::Normalize normalizeOp;
    EXPECT_NO_THROW(
        normalizeOp(stream, batchSrc, imgBase, imgScale, batchDst, globalScale, globalShift, 0.f, normalScale));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<T> test = DownloadPlanarImage<T>(imgDst[i]);
        std::vector<T> gold
            = MakeNormalizeGold<T>(srcVec[i], sizes[i].w, sizes[i].h, channels, base, scale, globalScale, globalShift);
        ExpectPlanarVectorsNear(gold, test);
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpNormalize, test::ValueList<int, int, int, bool, bool, uint32_t, float, float, float>
{
    // width, height, numImages, scalarBase, scalarScale,         flags, globalScale, globalShift, epsilon,
    {     32,     33,         1,       true,        true,   normalScale,         0.f,         0.f,     0.f, },
    {     66,     55,         1,       true,        true,   normalScale,         1.f,         0.f,     0.f, },
    {    122,    212,         2,       true,        true,   normalScale,      1.234f,      43.21f,     0.f, },
    {    211,    102,         3,      false,       false,   normalScale,        1.1f,        0.1f,     0.f, },
    {     21,     12,         5,       true,        true, scaleIsStdDev,        1.2f,        0.2f,     0.f, },
    {     22,     12,         5,      false,        true, scaleIsStdDev,        1.3f,        0.3f,     0.f, },
    {     55,     23,         5,       true,       false, scaleIsStdDev,        1.2f,        0.2f,     0.f, },
    {     72,     88,         5,      false,       false, scaleIsStdDev,        1.2f,        0.2f,     0.f, },
    {     63,     32,         7,      false,        true,   normalScale,        1.3f,        0.3f,     0.f, },
    {     22,     13,         9,       true,       false,   normalScale,        1.4f,        0.4f,     0.f, },
    {     55,     33,         2,       true,       false, scaleIsStdDev,        2.1f,        1.1f,   1.23f, },
    {    444,    222,         4,       true,       false, scaleIsStdDev,        2.2f,        2.2f,   12.3f, }
});

// clang-format on

TEST_P(OpNormalize, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int      width       = GetParamValue<0>();
    int      height      = GetParamValue<1>();
    int      numImages   = GetParamValue<2>();
    bool     scalarBase  = GetParamValue<3>();
    bool     scalarScale = GetParamValue<4>();
    uint32_t flags       = GetParamValue<5>();
    float    globalScale = GetParamValue<6>();
    float    globalShift = GetParamValue<7>();
    float    epsilon     = GetParamValue<8>();

    int baseWidth      = (scalarBase ? 1 : width);
    int scaleWidth     = (scalarScale ? 1 : width);
    int baseHeight     = (scalarBase ? 1 : height);
    int scaleHeight    = (scalarScale ? 1 : height);
    int baseNumImages  = (scalarBase ? 1 : numImages);
    int scaleNumImages = (scalarScale ? 1 : numImages);

    nvcv::ImageFormat baseFormat  = (scalarBase ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);
    nvcv::ImageFormat scaleFormat = (scalarScale ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);

    nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    std::default_random_engine rng;

    // Create input tensor
    nvcv::Tensor imgSrc  = nvcv::util::CreateTensor(numImages, width, height, fmt);
    auto         srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    int                               srcVecRowStride = width * fmt.numChannels();
    for (int i = 0; i < numImages; ++i)
    {
        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(height * srcVecRowStride);
        std::ranges::generate(srcVec[i], [&udist, &rng]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(), srcVecRowStride,
                               srcVecRowStride, // vec has no padding
                               height, cudaMemcpyHostToDevice));
    }

    // Create base tensor
    nvcv::Tensor imgBase(baseNumImages, {baseWidth, baseHeight}, baseFormat);
    auto         baseData = imgBase.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, baseData);
    auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
    ASSERT_TRUE(baseAccess);

    std::vector<std::vector<float>> baseVec(baseNumImages);
    int                             baseVecRowStride = baseWidth * baseFormat.numChannels();
    for (int i = 0; i < baseNumImages; ++i)
    {
        std::uniform_real_distribution<float> udist(0, 255.f);

        baseVec[i].resize(baseHeight * baseVecRowStride);
        std::ranges::generate(baseVec[i], [&udist, &rng]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(baseAccess->sampleData(i), baseAccess->rowStride(), baseVec[i].data(),
                                            baseVecRowStride * sizeof(float),
                                            baseVecRowStride * sizeof(float), // vec has no padding
                                            baseHeight, cudaMemcpyHostToDevice));
    }

    // Create scale tensor
    nvcv::Tensor imgScale(scaleNumImages, {scaleWidth, scaleHeight}, scaleFormat);
    auto         scaleData = imgScale.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, scaleData);
    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*scaleData);
    ASSERT_TRUE(scaleAccess);

    std::vector<std::vector<float>> scaleVec(scaleNumImages);
    assert(scaleFormat.numPlanes() == 1);
    int scaleVecRowStride = scaleWidth * scaleFormat.numChannels();
    for (int i = 0; i < scaleNumImages; ++i)
    {
        std::uniform_real_distribution<float> udist(0, 1.f);

        scaleVec[i].resize(scaleHeight * scaleVecRowStride);
        std::ranges::generate(scaleVec[i], [&udist, &rng]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(scaleAccess->sampleData(i), scaleAccess->rowStride(), scaleVec[i].data(),
                                            scaleVecRowStride * sizeof(float),
                                            scaleVecRowStride * sizeof(float), // vec has no padding
                                            scaleHeight, cudaMemcpyHostToDevice));
    }

    // Create dest tensor
    nvcv::Tensor imgDst = nvcv::util::CreateTensor(numImages, width, height, fmt);

    // Generate test result
    cvcuda::Normalize normalizeOp;
    EXPECT_NO_THROW(normalizeOp(stream, imgSrc, imgBase, imgScale, imgDst, globalScale, globalShift, epsilon, flags));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowStride = width * fmt.numChannels();
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(height * dstVecRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowStride, dstAccess->sampleData(i), dstAccess->rowStride(),
                               dstVecRowStride, // vec has no padding
                               height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(height * dstVecRowStride);

        int bi = baseNumImages == 1 ? 0 : i;
        int si = scaleNumImages == 1 ? 0 : i;

        // Generate gold result
        Normalize({
            goldVec,
            dstVecRowStride,
            srcVec[i],
            srcVecRowStride,
            {     width,      height},
            fmt,
            baseVec[bi],
            baseVecRowStride,
            { baseWidth,  baseHeight},
            baseFormat,
            scaleVec[si],
            scaleVecRowStride,
            {scaleWidth, scaleHeight},
            scaleFormat,
            globalScale,
            globalShift,
            epsilon,
            flags
        });

        EXPECT_EQ(goldVec, testVec);
    }
}

// Shared fixed parameters for the inverse-std-dev vectorized/hoisted coverage cases below.
static constexpr float kStdDevGlobalScale = 1.7f;
static constexpr float kStdDevGlobalShift = 3.5f;

// Deterministic host fills shared by the inverse-std-dev tensor and var-shape helpers. The SAME
// host data feeds both the device upload and the gold computation, preserving bit-exactness; using
// fixed index-based formulas (instead of a PRNG) keeps the values reproducible without flagging the
// cpp:S2245 security hotspot.
//   src bytes in [0,255]; base in [0,255); scale in [0.1, 2.0] (strictly positive).
static void FillDeterministicSrc(std::vector<uint8_t> &dst)
{
    for (size_t k = 0; k < dst.size(); ++k)
    {
        dst[k] = static_cast<uint8_t>((k * 37 + 11) & 0xFF);
    }
}

static void FillDeterministicBaseScale(std::vector<float> &base, std::vector<float> &scale)
{
    for (size_t k = 0; k < base.size(); ++k)
    {
        base[k] = static_cast<float>((k * 53) % 256);
    }
    for (size_t k = 0; k < scale.size(); ++k)
    {
        scale[k] = 0.1f + static_cast<float>((k * 7) % 20) * 0.1f;
    }
}

// Uploads a single-sample F32 param tensor (base or scale) from host data via a contiguous copy.
static void UploadParamTensor(const nvcv::Tensor &tensor, const std::vector<float> &host)
{
    auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, data);
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
    ASSERT_TRUE(access);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(access->sampleData(0), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
}

// Builds the gold image for sample i against the shared reference Normalize() and EXPECT_EQ-compares
// it to the device result. Shared by the tensor and var-shape inverse-std-dev helpers.
static void ExpectGoldEquals(std::vector<uint8_t> &testVec, const std::vector<uint8_t> &srcVec, int srcRowStride,
                             nvcv::Size2D size, nvcv::ImageFormat fmt, const std::vector<float> &baseVec,
                             const std::vector<float> &scaleVec, int paramRowStride, nvcv::Size2D paramSize,
                             nvcv::ImageFormat paramFmt, float epsilon)
{
    std::vector<uint8_t> goldVec(testVec.size());
    Normalize({goldVec, srcRowStride, srcVec, srcRowStride, size, fmt, baseVec, paramRowStride, paramSize, paramFmt,
               scaleVec, paramRowStride, paramSize, paramFmt, kStdDevGlobalScale, kStdDevGlobalShift, epsilon,
               CVCUDA_NORMALIZE_SCALE_IS_STDDEV});

    EXPECT_EQ(goldVec, testVec);
}

// Inverse-std-dev base/scale broadcast modes exercised by the consolidated tensor/var-shape helpers:
//   Scalar     -> 1 x 1 x 1 x 1 F32 param, broadcast over every sample/pixel/channel
//   PerPixel   -> N x H x W x 1 F32 param, one base/scale per src pixel (single-channel src)
//   PerChannel -> 1 x 1 x 1 x C param (RGBf32/RGBAf32), broadcast over the spatial extent
enum class ParamMode
{
    Scalar,
    PerPixel,
    PerChannel
};

// The F32 param image-format that pairs with a given mode and channel count.
static nvcv::ImageFormat ParamFormatFor(ParamMode mode, int channels)
{
    if (mode == ParamMode::PerChannel)
    {
        return channels == 3 ? nvcv::FMT_RGBf32 : nvcv::FMT_RGBAf32;
    }
    return nvcv::FMT_F32;
}

// Builds, fills (deterministically) and uploads the base/scale param tensors for a given mode. The
// per-sample host data is returned in baseVec/scaleVec so the gold can reuse the exact same values:
//   Scalar/PerChannel -> one host vector (index 0); PerPixel -> one host vector per sample.
// paramW/paramH/paramRowStride report the per-sample param geometry the gold loop needs.
static void MakeStdDevParam(ParamMode mode, nvcv::ImageFormat paramFmt, int channels, int width, int height,
                            int numImages, nvcv::Tensor &imgBase, nvcv::Tensor &imgScale,
                            std::vector<std::vector<float>> &baseVec, std::vector<std::vector<float>> &scaleVec,
                            int &paramW, int &paramH, int &paramRowStride)
{
    if (mode == ParamMode::PerChannel)
    {
        imgBase = nvcv::Tensor(
            {
                {1, 1, 1, channels},
                nvcv::TENSOR_NHWC
        },
            paramFmt.planeDataType(0));
        imgScale = nvcv::Tensor(
            {
                {1, 1, 1, channels},
                nvcv::TENSOR_NHWC
        },
            paramFmt.planeDataType(0));
        baseVec.assign(1, std::vector<float>(channels));
        scaleVec.assign(1, std::vector<float>(channels));
        FillDeterministicBaseScale(baseVec[0], scaleVec[0]);
        UploadParamTensor(imgBase, baseVec[0]);
        UploadParamTensor(imgScale, scaleVec[0]);
        paramW = paramH = paramRowStride = 1; // unused by the PerChannel gold (paramRowStride==channels passed there)
        return;
    }

    const int paramN = (mode == ParamMode::PerPixel) ? numImages : 1;
    paramW           = (mode == ParamMode::PerPixel) ? width : 1;
    paramH           = (mode == ParamMode::PerPixel) ? height : 1;
    paramRowStride   = paramW;

    imgBase  = nvcv::Tensor(paramN, {paramW, paramH}, paramFmt);
    imgScale = nvcv::Tensor(paramN, {paramW, paramH}, paramFmt);
    baseVec.assign(paramN, {});
    scaleVec.assign(paramN, {});

    auto baseData  = imgBase.exportData<nvcv::TensorDataStridedCuda>();
    auto scaleData = imgScale.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, baseData);
    ASSERT_NE(nullptr, scaleData);
    auto baseAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*scaleData);
    ASSERT_TRUE(baseAccess);
    ASSERT_TRUE(scaleAccess);
    for (int i = 0; i < paramN; ++i)
    {
        baseVec[i].resize(static_cast<size_t>(paramH) * paramRowStride);
        scaleVec[i].resize(static_cast<size_t>(paramH) * paramRowStride);
        FillDeterministicBaseScale(baseVec[i], scaleVec[i]);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(baseAccess->sampleData(i), baseAccess->rowStride(), baseVec[i].data(),
                                            paramRowStride * sizeof(float), paramRowStride * sizeof(float), paramH,
                                            cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(scaleAccess->sampleData(i), scaleAccess->rowStride(), scaleVec[i].data(),
                                            paramRowStride * sizeof(float), paramRowStride * sizeof(float), paramH,
                                            cudaMemcpyHostToDevice));
    }
}

// Consolidated inverse-std-dev tensor coverage (channels==1 U8 with Scalar/PerPixel param, or
// interleaved RGB8/RGBA8 with PerChannel param). The SAME deterministic host data feeds both the
// device upload and the shared gold, preserving bit-exactness. Exercises the vectorized
// single-channel uchar4 kernel and the hoisted NIX-pixels-per-thread interleaved kernel across the
// vector body (W aligned) and the scalar/per-pixel tail (W unaligned), plus every broadcast branch.
static void RunTensorStdDevCase(nvcv::ImageFormat fmt, int width, int height, int numImages, ParamMode mode)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const float             epsilon  = (mode == ParamMode::PerChannel) ? 0.234f : 0.123f;
    const int               channels = fmt.numChannels();
    const nvcv::ImageFormat paramFmt = ParamFormatFor(mode, channels);

    nvcv::Tensor imgSrc  = nvcv::util::CreateTensor(numImages, width, height, fmt);
    auto         srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    const int                         srcRowStride = width * channels;
    for (int i = 0; i < numImages; ++i)
    {
        srcVec[i].resize(static_cast<size_t>(height) * srcRowStride);
        FillDeterministicSrc(srcVec[i]);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(),
                                            srcRowStride, srcRowStride, height, cudaMemcpyHostToDevice));
    }

    nvcv::Tensor                    imgBase;
    nvcv::Tensor                    imgScale;
    std::vector<std::vector<float>> baseVec;
    std::vector<std::vector<float>> scaleVec;
    int                             paramW         = 1;
    int                             paramH         = 1;
    int                             paramRowStride = 1;
    MakeStdDevParam(mode, paramFmt, channels, width, height, numImages, imgBase, imgScale, baseVec, scaleVec, paramW,
                    paramH, paramRowStride);

    nvcv::Tensor imgDst = nvcv::util::CreateTensor(numImages, width, height, fmt);

    cvcuda::Normalize op;
    EXPECT_NO_THROW(op(stream, imgSrc, imgBase, imgScale, imgDst, kStdDevGlobalScale, kStdDevGlobalShift, epsilon,
                       CVCUDA_NORMALIZE_SCALE_IS_STDDEV));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    const int dstRowStride = srcRowStride;
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<uint8_t> testVec(static_cast<size_t>(height) * dstRowStride);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), dstRowStride, dstAccess->sampleData(i),
                                            dstAccess->rowStride(), dstRowStride, height, cudaMemcpyDeviceToHost));

        if (mode == ParamMode::PerChannel)
        {
            ExpectGoldEquals(testVec, srcVec[i], dstRowStride, {width, height}, fmt, baseVec[0], scaleVec[0], channels,
                             {1, 1}, paramFmt, epsilon);
        }
        else
        {
            const auto bi = (mode == ParamMode::PerPixel) ? i : 0;
            ExpectGoldEquals(testVec, srcVec[i], dstRowStride, {width, height}, fmt, baseVec[bi], scaleVec[bi],
                             paramRowStride, {paramW, paramH}, paramFmt, epsilon);
        }
    }
}

TEST(OpNormalize, tensor_u8_single_channel_stddev_vectorized)
{
    RunTensorStdDevCase(nvcv::FMT_U8, 1920, 4, 2, ParamMode::Scalar);   // W%4==0, vector body, scalar base/scale
    RunTensorStdDevCase(nvcv::FMT_U8, 1920, 3, 1, ParamMode::PerPixel); // W%4==0, per-pixel base/scale
    RunTensorStdDevCase(nvcv::FMT_U8, 23, 5, 2, ParamMode::Scalar);     // W%4!=0, scalar-tail branch
    RunTensorStdDevCase(nvcv::FMT_U8, 21, 7, 1, ParamMode::PerPixel);   // W%4!=0 tail + per-pixel base/scale
}

static void RunTensorU8SingleChannelStdDevVec4Stride(ParamMode mode)
{
    constexpr int   width        = 22;
    constexpr int   height       = 3;
    constexpr int   rowStride    = 24; // 4-byte aligned but deliberately not 16-byte aligned.
    constexpr int   sampleStride = rowStride * height;
    constexpr float epsilon      = 0.123f;

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, sampleStride));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, sampleStride));

    auto wrapTensor = [=](NVCVByte *ptr)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr;
        buffer.strides[0] = sampleStride;
        buffer.strides[1] = rowStride;
        buffer.strides[2] = 1;
        buffer.strides[3] = 1;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, height, width, 1}, "NHWC"},
            nvcv::TYPE_U8, buffer
        });
    };

    nvcv::Tensor src = wrapTensor(srcPtr);
    nvcv::Tensor dst = wrapTensor(dstPtr);

    std::vector<uint8_t> srcVec(width * height);
    FillDeterministicSrc(srcVec);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy2D(srcPtr, rowStride, srcVec.data(), width, width, height, cudaMemcpyHostToDevice));

    nvcv::Tensor                    base;
    nvcv::Tensor                    scale;
    std::vector<std::vector<float>> baseVec;
    std::vector<std::vector<float>> scaleVec;
    int                             paramW;
    int                             paramH;
    int                             paramRowStride;
    MakeStdDevParam(mode, nvcv::FMT_F32, 1, width, height, 1, base, scale, baseVec, scaleVec, paramW, paramH,
                    paramRowStride);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::Normalize op;
    EXPECT_NO_THROW(op(stream, src, base, scale, dst, kStdDevGlobalScale, kStdDevGlobalShift, epsilon,
                       CVCUDA_NORMALIZE_SCALE_IS_STDDEV));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> testVec(width * height);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy2D(testVec.data(), width, dstPtr, rowStride, width, height, cudaMemcpyDeviceToHost));
    ExpectGoldEquals(testVec, srcVec, width, {width, height}, nvcv::FMT_U8, baseVec[0], scaleVec[0], paramRowStride,
                     {paramW, paramH}, nvcv::FMT_F32, epsilon);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

TEST(OpNormalize, tensor_u8_single_channel_stddev_vec4_stride)
{
    RunTensorU8SingleChannelStdDevVec4Stride(ParamMode::PerPixel);
}

TEST(OpNormalize, tensor_u8_single_channel_stddev_vec4_scalar_broadcast)
{
    RunTensorU8SingleChannelStdDevVec4Stride(ParamMode::Scalar);
}

// Bit-exact coverage for the vectorized single-channel F32 (NHWC) inverse-std-dev path. The reference
// is the device scalar kernel itself (run on the same data via the 3-channel F32 dispatch, which is
// untouched and uses the per-element normalizeInvStdDevKernel): the spec's bit-exact contract is that
// the vectorized path's output is byte-for-byte identical to the scalar kernel's, not to a host re-
// derivation (device FMA contraction differs from a naive host expression). Single-channel input is
// compared per element against one channel of a 3-channel run fed the identical per-pixel values, so
// any difference is purely the vectorization. Widths that are and are not a multiple of 16 (the float4
// ILP block) cover both the vector body and the scalar tail.
static void RunTensorF32SingleChannelStdDevCase(int width, int height, int numImages)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const float epsilon  = 0.123f;
    const float baseVal  = 2.5f;
    const float scaleVal = 0.8f;

    // Deterministic per-pixel source values, shared by the single-channel (vectorized) run and the
    // 3-channel (scalar reference) run.
    std::vector<std::vector<float>> srcVec(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        srcVec[i].resize(static_cast<size_t>(height) * width);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                srcVec[i][y * width + x] = 7.0f + static_cast<float>(i) * 1.5f + static_cast<float>(y) * 0.25f
                                         + static_cast<float>(x) * 0.5f;
            }
        }
    }

    cvcuda::Normalize op;

    // Run 1: single-channel F32 -> vectorized float4 ILP path.
    std::vector<std::vector<float>> vecOut(numImages);
    {
        nvcv::Tensor imgSrc = nvcv::util::CreateTensor(numImages, width, height, nvcv::FMT_F32);
        nvcv::Tensor imgDst = nvcv::util::CreateTensor(numImages, width, height, nvcv::FMT_F32);
        auto sAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*imgSrc.exportData<nvcv::TensorDataStridedCuda>());
        auto dAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*imgDst.exportData<nvcv::TensorDataStridedCuda>());
        ASSERT_TRUE(sAcc && dAcc);
        for (int i = 0; i < numImages; ++i)
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(sAcc->sampleData(i), sAcc->rowStride(), srcVec[i].data(), width * sizeof(float),
                                   width * sizeof(float), height, cudaMemcpyHostToDevice));

        nvcv::Tensor imgBase(1, {1, 1}, nvcv::FMT_F32);
        nvcv::Tensor imgScale(1, {1, 1}, nvcv::FMT_F32);
        UploadParamTensor(imgBase, {baseVal});
        UploadParamTensor(imgScale, {scaleVal});

        EXPECT_NO_THROW(op(stream, imgSrc, imgBase, imgScale, imgDst, kStdDevGlobalScale, kStdDevGlobalShift, epsilon,
                           CVCUDA_NORMALIZE_SCALE_IS_STDDEV));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        for (int i = 0; i < numImages; ++i)
        {
            vecOut[i].resize(static_cast<size_t>(height) * width);
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(vecOut[i].data(), width * sizeof(float), dAcc->sampleData(i), dAcc->rowStride(),
                                   width * sizeof(float), height, cudaMemcpyDeviceToHost));
        }
    }

    // Run 2: 3-channel F32 with the same value replicated to every channel -> scalar reference kernel.
    std::vector<std::vector<float>> refOut(numImages);
    {
        nvcv::Tensor imgSrc = nvcv::util::CreateTensor(numImages, width, height, nvcv::FMT_RGBf32);
        nvcv::Tensor imgDst = nvcv::util::CreateTensor(numImages, width, height, nvcv::FMT_RGBf32);
        auto sAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*imgSrc.exportData<nvcv::TensorDataStridedCuda>());
        auto dAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*imgDst.exportData<nvcv::TensorDataStridedCuda>());
        ASSERT_TRUE(sAcc && dAcc);
        for (int i = 0; i < numImages; ++i)
        {
            std::vector<float> interleaved(static_cast<size_t>(height) * width * 3);
            for (int p = 0; p < height * width; ++p)
                interleaved[p * 3 + 0] = interleaved[p * 3 + 1] = interleaved[p * 3 + 2] = srcVec[i][p];
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(sAcc->sampleData(i), sAcc->rowStride(), interleaved.data(),
                                                width * 3 * sizeof(float), width * 3 * sizeof(float), height,
                                                cudaMemcpyHostToDevice));
        }

        // Scalar (1x1x1x1) base/scale broadcasts over the channels too, so each channel sees identical
        // base/scale -- matching the single-channel run's per-element inputs exactly.
        nvcv::Tensor imgBase(1, {1, 1}, nvcv::FMT_F32);
        nvcv::Tensor imgScale(1, {1, 1}, nvcv::FMT_F32);
        UploadParamTensor(imgBase, {baseVal});
        UploadParamTensor(imgScale, {scaleVal});

        EXPECT_NO_THROW(op(stream, imgSrc, imgBase, imgScale, imgDst, kStdDevGlobalScale, kStdDevGlobalShift, epsilon,
                           CVCUDA_NORMALIZE_SCALE_IS_STDDEV));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        for (int i = 0; i < numImages; ++i)
        {
            std::vector<float> interleaved(static_cast<size_t>(height) * width * 3);
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(interleaved.data(), width * 3 * sizeof(float), dAcc->sampleData(i),
                                   dAcc->rowStride(), width * 3 * sizeof(float), height, cudaMemcpyDeviceToHost));
            refOut[i].resize(static_cast<size_t>(height) * width);
            for (int p = 0; p < height * width; ++p) refOut[i][p] = interleaved[p * 3 + 0];
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        // Byte-for-byte identical to the scalar kernel's output.
        EXPECT_EQ(0, std::memcmp(refOut[i].data(), vecOut[i].data(), refOut[i].size() * sizeof(float)));
    }
}

TEST(OpNormalize, tensor_f32_single_channel_stddev_vectorized)
{
    RunTensorF32SingleChannelStdDevCase(1920, 4, 2); // W%16==0, float4 vector body
    RunTensorF32SingleChannelStdDevCase(67, 5, 2);   // W%16!=0, vector body + scalar tail
    RunTensorF32SingleChannelStdDevCase(9, 3, 1);    // small width, mostly tail
}

// Single-channel F32 var-shape inverse-std-dev: drives the vectorized float4 ILP var-shape path and
// byte-compares it against the untouched scalar var-shape kernel. The reference run uses a 3-channel
// F32 var-shape batch (replicating each value to all 3 channels) with a scalar (1x1) base/scale, which
// broadcasts identically across channels -- so each channel's per-element inputs match the
// single-channel run exactly, and the 3-channel batch never takes the single-channel vec path.
// Per-image widths cover a full float4-group body (W%4==0) and a sub-group tail (W%4!=0).
static void RunVarShapeF32SingleChannelStdDevCase(const std::vector<nvcv::Size2D> &sizes)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const float epsilon   = 0.123f;
    const float baseVal   = 2.5f;
    const float scaleVal  = 0.8f;
    const auto  numImages = static_cast<int>(sizes.size());

    // Deterministic per-pixel source values, shared by the single-channel (vectorized) run and the
    // 3-channel (scalar reference) run.
    std::vector<std::vector<float>> srcVec(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        srcVec[i].resize(static_cast<size_t>(sizes[i].h) * sizes[i].w);
        for (int y = 0; y < sizes[i].h; ++y)
            for (int x = 0; x < sizes[i].w; ++x)
                srcVec[i][y * sizes[i].w + x] = 7.0f + static_cast<float>(i) * 1.5f + static_cast<float>(y) * 0.25f
                                              + static_cast<float>(x) * 0.5f;
    }

    cvcuda::Normalize op;

    nvcv::Tensor imgBase(1, {1, 1}, nvcv::FMT_F32);
    nvcv::Tensor imgScale(1, {1, 1}, nvcv::FMT_F32);
    UploadParamTensor(imgBase, {baseVal});
    UploadParamTensor(imgScale, {scaleVal});

    // Run 1: single-channel F32 var-shape -> vectorized float4 ILP path.
    std::vector<std::vector<float>> vecOut(numImages);
    {
        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;
        for (int i = 0; i < numImages; ++i)
        {
            imgSrc.emplace_back(sizes[i], nvcv::FMT_F32);
            imgDst.emplace_back(sizes[i], nvcv::FMT_F32);
            auto sData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(sData, nvcv::NullOpt);
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(sData->plane(0).basePtr, sData->plane(0).rowStride, srcVec[i].data(),
                                                sizes[i].w * sizeof(float), sizes[i].w * sizeof(float), sizes[i].h,
                                                cudaMemcpyHostToDevice));
        }
        nvcv::ImageBatchVarShape batchSrc(numImages);
        nvcv::ImageBatchVarShape batchDst(numImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        EXPECT_NO_THROW(op(stream, batchSrc, imgBase, imgScale, batchDst, kStdDevGlobalScale, kStdDevGlobalShift,
                           epsilon, CVCUDA_NORMALIZE_SCALE_IS_STDDEV));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        for (int i = 0; i < numImages; ++i)
        {
            auto dData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(dData, nvcv::NullOpt);
            vecOut[i].resize(static_cast<size_t>(sizes[i].h) * sizes[i].w);
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(vecOut[i].data(), sizes[i].w * sizeof(float), dData->plane(0).basePtr,
                                                dData->plane(0).rowStride, sizes[i].w * sizeof(float), sizes[i].h,
                                                cudaMemcpyDeviceToHost));
        }
    }

    // Run 2: 3-channel F32 var-shape with the same value replicated to every channel -> scalar kernel.
    std::vector<std::vector<float>> refOut(numImages);
    {
        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;
        for (int i = 0; i < numImages; ++i)
        {
            imgSrc.emplace_back(sizes[i], nvcv::FMT_RGBf32);
            imgDst.emplace_back(sizes[i], nvcv::FMT_RGBf32);
            std::vector<float> interleaved(static_cast<size_t>(sizes[i].h) * sizes[i].w * 3);
            for (int p = 0; p < sizes[i].h * sizes[i].w; ++p)
                interleaved[p * 3 + 0] = interleaved[p * 3 + 1] = interleaved[p * 3 + 2] = srcVec[i][p];
            auto sData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(sData, nvcv::NullOpt);
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(sData->plane(0).basePtr, sData->plane(0).rowStride, interleaved.data(),
                                                sizes[i].w * 3 * sizeof(float), sizes[i].w * 3 * sizeof(float),
                                                sizes[i].h, cudaMemcpyHostToDevice));
        }
        nvcv::ImageBatchVarShape batchSrc(numImages);
        nvcv::ImageBatchVarShape batchDst(numImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        EXPECT_NO_THROW(op(stream, batchSrc, imgBase, imgScale, batchDst, kStdDevGlobalScale, kStdDevGlobalShift,
                           epsilon, CVCUDA_NORMALIZE_SCALE_IS_STDDEV));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        for (int i = 0; i < numImages; ++i)
        {
            auto dData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(dData, nvcv::NullOpt);
            std::vector<float> interleaved(static_cast<size_t>(sizes[i].h) * sizes[i].w * 3);
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(interleaved.data(), sizes[i].w * 3 * sizeof(float),
                                                dData->plane(0).basePtr, dData->plane(0).rowStride,
                                                sizes[i].w * 3 * sizeof(float), sizes[i].h, cudaMemcpyDeviceToHost));
            refOut[i].resize(static_cast<size_t>(sizes[i].h) * sizes[i].w);
            for (int p = 0; p < sizes[i].h * sizes[i].w; ++p) refOut[i][p] = interleaved[p * 3 + 0];
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        // Byte-for-byte identical to the scalar var-shape kernel's output.
        EXPECT_EQ(0, std::memcmp(refOut[i].data(), vecOut[i].data(), refOut[i].size() * sizeof(float)));
    }
}

TEST(OpNormalize, varshape_f32_single_channel_stddev_vectorized)
{
    RunVarShapeF32SingleChannelStdDevCase({
        {1920, 4}, // W%4==0, full float4-group body
        {1923, 3}  // W%4!=0, body + scalar tail
    });
    RunVarShapeF32SingleChannelStdDevCase({
        {9, 3}, // small width, mostly tail
        {8, 5}  // exactly two float4 groups
    });
}

TEST(OpNormalize, tensor_interleaved_per_channel_stddev_hoist)
{
    RunTensorStdDevCase(nvcv::FMT_RGB8, 1920, 4, 2, ParamMode::PerChannel);  // uchar3, NIX vector body
    RunTensorStdDevCase(nvcv::FMT_RGB8, 257, 5, 1, ParamMode::PerChannel);   // uchar3, NIX tail
    RunTensorStdDevCase(nvcv::FMT_RGBA8, 1920, 4, 2, ParamMode::PerChannel); // uchar4, NIX vector body
    RunTensorStdDevCase(nvcv::FMT_RGBA8, 259, 3, 1, ParamMode::PerChannel);  // uchar4, NIX tail
}

// Consolidated inverse-std-dev var-shape coverage: single-channel U8 with Scalar param, or
// interleaved RGB8/RGBA8 with PerChannel param. Per-image widths differ so both the vectorized
// uchar4 body / hoisted NIX body and the W-unaligned tail are covered. Each image is compared
// against the shared gold bit-for-bit.
static void RunVarShapeStdDevCase(nvcv::ImageFormat fmt, const std::vector<nvcv::Size2D> &sizes, ParamMode mode)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const float             epsilon   = (mode == ParamMode::PerChannel) ? 0.234f : 0.123f;
    const int               channels  = fmt.numChannels();
    const nvcv::ImageFormat paramFmt  = ParamFormatFor(mode, channels);
    const auto              numImages = static_cast<int>(sizes.size());

    std::vector<nvcv::Image>          imgSrc;
    std::vector<std::vector<uint8_t>> srcVec(numImages);
    std::vector<int>                  srcRowStride(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(sizes[i], fmt);
        srcRowStride[i] = sizes[i].w * channels;

        srcVec[i].resize(static_cast<size_t>(sizes[i].h) * srcRowStride[i]);
        FillDeterministicSrc(srcVec[i]);

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                            srcRowStride[i], srcRowStride[i], sizes[i].h, cudaMemcpyHostToDevice));
    }
    nvcv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Scalar param keeps a single host value (single-channel src); PerChannel broadcasts 1x1x1xC.
    nvcv::Tensor                    imgBase;
    nvcv::Tensor                    imgScale;
    std::vector<std::vector<float>> baseVec;
    std::vector<std::vector<float>> scaleVec;
    if (mode == ParamMode::PerChannel)
    {
        int paramW         = 1;
        int paramH         = 1;
        int paramRowStride = 1;
        MakeStdDevParam(mode, paramFmt, channels, 0, 0, numImages, imgBase, imgScale, baseVec, scaleVec, paramW, paramH,
                        paramRowStride);
    }
    else
    {
        imgBase  = nvcv::Tensor(1, {1, 1}, paramFmt);
        imgScale = nvcv::Tensor(1, {1, 1}, paramFmt);
        baseVec.assign(1, std::vector<float>(1));
        scaleVec.assign(1, std::vector<float>(1));
        FillDeterministicBaseScale(baseVec[0], scaleVec[0]);
        UploadParamTensor(imgBase, baseVec[0]);
        UploadParamTensor(imgScale, scaleVec[0]);
    }

    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgDst.emplace_back(sizes[i], fmt);
    }
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::Normalize op;
    EXPECT_NO_THROW(op(stream, batchSrc, imgBase, imgScale, batchDst, kStdDevGlobalScale, kStdDevGlobalShift, epsilon,
                       CVCUDA_NORMALIZE_SCALE_IS_STDDEV));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    const int          paramChannels = (mode == ParamMode::PerChannel) ? channels : 0;
    const nvcv::Size2D paramSz       = {1, 1};
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(dstData, nvcv::NullOpt);

        const int            dstRowStride = srcRowStride[i];
        std::vector<uint8_t> testVec(static_cast<size_t>(sizes[i].h) * dstRowStride);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, sizes[i].h, cudaMemcpyDeviceToHost));

        ExpectGoldEquals(testVec, srcVec[i], dstRowStride, {sizes[i].w, sizes[i].h}, fmt, baseVec[0], scaleVec[0],
                         paramChannels, paramSz, paramFmt, epsilon);
    }
}

TEST(OpNormalize, varshape_u8_single_channel_stddev_vectorized)
{
    RunVarShapeStdDevCase(nvcv::FMT_U8,
                          {
                              {1920, 4},
                              {1923, 3}
    },
                          ParamMode::Scalar); // vector body + W%4 tail across differing per-image widths
    RunVarShapeStdDevCase(nvcv::FMT_U8,
                          {
                              {22, 5},
                              {21, 7},
                              {20, 3}
    },
                          ParamMode::Scalar); // small tail widths
}

TEST(OpNormalize, varshape_interleaved_per_channel_stddev_hoist)
{
    RunVarShapeStdDevCase(nvcv::FMT_RGB8,
                          {
                              {1920, 4},
                              {1923, 3}
    },
                          ParamMode::PerChannel); // uchar3, NIX body + tail
    RunVarShapeStdDevCase(nvcv::FMT_RGBA8,
                          {
                              {1920, 4},
                              {1925, 3}
    },
                          ParamMode::PerChannel); // uchar4, NIX body + tail
}

// Distinct per-channel (or scalar-broadcast) base/scale for the float reference cases. PerChannel
// fills C distinct values; Scalar fills one value and replicates it across channels so the host gold
// (MakeNormalizeGold, indexed per channel) matches the operator's broadcast exactly.
static void MakeFloatRefParams(ParamMode mode, nvcv::ImageFormat paramFmt, int channels, nvcv::Tensor &imgBase,
                               nvcv::Tensor &imgScale, std::vector<float> &goldBase, std::vector<float> &goldScale)
{
    if (mode == ParamMode::PerChannel)
    {
        imgBase = nvcv::Tensor(
            {
                {1, 1, 1, channels},
                nvcv::TENSOR_NHWC
        },
            paramFmt.planeDataType(0));
        imgScale = nvcv::Tensor(
            {
                {1, 1, 1, channels},
                nvcv::TENSOR_NHWC
        },
            paramFmt.planeDataType(0));
        goldBase.resize(channels);
        goldScale.resize(channels);
        FillDeterministicBaseScale(goldBase, goldScale);
        UploadParamTensor(imgBase, goldBase);
        UploadParamTensor(imgScale, goldScale);
    }
    else
    {
        imgBase  = nvcv::Tensor(1, {1, 1}, paramFmt);
        imgScale = nvcv::Tensor(1, {1, 1}, paramFmt);
        std::vector<float> b(1);
        std::vector<float> s(1);
        FillDeterministicBaseScale(b, s);
        UploadParamTensor(imgBase, b);
        UploadParamTensor(imgScale, s);
        goldBase.assign(channels, b[0]);
        goldScale.assign(channels, s[0]);
    }
}

// Independent math-correctness backstop for the FLOAT paths (single-channel F32 and interleaved
// float3/float4). The reference is the host C++ MakeNormalizeGold(), NOT the device scalar kernel, so
// EXPECT_NEAR(1e-4) is used: device FMA contraction differs from a naive host expression bit-for-bit.
// This is the only independent reference that exercises distinct per-channel base/scale on the
// interleaved float kernels; it complements the bit-identity memcmp-vs-scalar checks (which feed every
// channel the same value). Covers Scalar and PerChannel params, plain (normalScale) and inverse-std-
// dev (scaleIsStdDev) flags, and widths that do and do not fill the float4 vector body (scalar tail).
static void RunTensorFloatRefCase(nvcv::ImageFormat fmt, int width, int height, int numImages, ParamMode mode,
                                  uint32_t flags)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const float             epsilon  = 0.234f;
    const int               channels = fmt.numChannels();
    const nvcv::ImageFormat paramFmt = ParamFormatFor(mode, channels);
    const int               rowElems = width * channels;

    nvcv::Tensor imgSrc  = nvcv::util::CreateTensor(numImages, width, height, fmt);
    auto         srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<float>> srcVec(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        srcVec[i] = MakePlanarHostImage<float>(i, width, height, channels);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(),
                               rowElems * sizeof(float), rowElems * sizeof(float), height, cudaMemcpyHostToDevice));
    }

    nvcv::Tensor       imgBase;
    nvcv::Tensor       imgScale;
    std::vector<float> goldBase;
    std::vector<float> goldScale;
    MakeFloatRefParams(mode, paramFmt, channels, imgBase, imgScale, goldBase, goldScale);

    nvcv::Tensor imgDst = nvcv::util::CreateTensor(numImages, width, height, fmt);

    cvcuda::Normalize op;
    EXPECT_NO_THROW(
        op(stream, imgSrc, imgBase, imgScale, imgDst, kStdDevGlobalScale, kStdDevGlobalShift, epsilon, flags));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<float> testVec(static_cast<size_t>(height) * rowElems);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), rowElems * sizeof(float), dstAccess->sampleData(i),
                               dstAccess->rowStride(), rowElems * sizeof(float), height, cudaMemcpyDeviceToHost));
        std::vector<float> gold = MakeNormalizeGold<float>(srcVec[i], width, height, channels, goldBase, goldScale,
                                                           kStdDevGlobalScale, kStdDevGlobalShift, flags, epsilon);
        ExpectPlanarVectorsNear(gold, testVec);
    }
}

// Var-shape twin of RunTensorFloatRefCase: per-image widths differ so both the float4 vector body and
// the W-unaligned tail are exercised; each image is compared against the independent host gold.
static void RunVarShapeFloatRefCase(nvcv::ImageFormat fmt, const std::vector<nvcv::Size2D> &sizes, ParamMode mode,
                                    uint32_t flags)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const float             epsilon   = 0.234f;
    const int               channels  = fmt.numChannels();
    const nvcv::ImageFormat paramFmt  = ParamFormatFor(mode, channels);
    const auto              numImages = static_cast<int>(sizes.size());

    std::vector<nvcv::Image>        imgSrc;
    std::vector<std::vector<float>> srcVec(numImages);
    std::vector<int>                rowElems(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(sizes[i], fmt);
        rowElems[i] = sizes[i].w * channels;
        srcVec[i]   = MakePlanarHostImage<float>(i, sizes[i].w, sizes[i].h, channels);

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                            rowElems[i] * sizeof(float), rowElems[i] * sizeof(float), sizes[i].h,
                                            cudaMemcpyHostToDevice));
    }
    nvcv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::Tensor       imgBase;
    nvcv::Tensor       imgScale;
    std::vector<float> goldBase;
    std::vector<float> goldScale;
    MakeFloatRefParams(mode, paramFmt, channels, imgBase, imgScale, goldBase, goldScale);

    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i) imgDst.emplace_back(sizes[i], fmt);
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::Normalize op;
    EXPECT_NO_THROW(
        op(stream, batchSrc, imgBase, imgScale, batchDst, kStdDevGlobalScale, kStdDevGlobalShift, epsilon, flags));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(dstData, nvcv::NullOpt);
        std::vector<float> testVec(static_cast<size_t>(sizes[i].h) * rowElems[i]);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowElems[i] * sizeof(float), dstData->plane(0).basePtr,
                                            dstData->plane(0).rowStride, rowElems[i] * sizeof(float), sizes[i].h,
                                            cudaMemcpyDeviceToHost));
        std::vector<float> gold
            = MakeNormalizeGold<float>(srcVec[i], sizes[i].w, sizes[i].h, channels, goldBase, goldScale,
                                       kStdDevGlobalScale, kStdDevGlobalShift, flags, epsilon);
        ExpectPlanarVectorsNear(gold, testVec);
    }
}

TEST(OpNormalize, tensor_f32_single_channel_ref)
{
    for (uint32_t flags : {normalScale, scaleIsStdDev})
    {
        RunTensorFloatRefCase(nvcv::FMT_F32, 1920, 4, 2, ParamMode::Scalar, flags); // W%16==0, float4 ILP body
        RunTensorFloatRefCase(nvcv::FMT_F32, 21, 7, 1, ParamMode::Scalar, flags);   // W%16!=0, scalar tail
    }
}

TEST(OpNormalize, tensor_interleaved_float_ref)
{
    for (uint32_t flags : {normalScale, scaleIsStdDev})
        for (nvcv::ImageFormat fmt : {nvcv::FMT_RGBf32, nvcv::FMT_RGBAf32})
            for (ParamMode mode : {ParamMode::Scalar, ParamMode::PerChannel})
            {
                RunTensorFloatRefCase(fmt, 1920, 4, 2, mode, flags); // vector body
                RunTensorFloatRefCase(fmt, 257, 5, 1, mode, flags);  // unaligned tail
            }
}

TEST(OpNormalize, varshape_f32_single_channel_ref)
{
    for (uint32_t flags : {normalScale, scaleIsStdDev})
        RunVarShapeFloatRefCase(nvcv::FMT_F32,
                                {
                                    {1920, 4},
                                    {  21, 7}
        },
                                ParamMode::Scalar, flags);
}

TEST(OpNormalize, varshape_interleaved_float_ref)
{
    for (uint32_t flags : {normalScale, scaleIsStdDev})
        for (nvcv::ImageFormat fmt : {nvcv::FMT_RGBf32, nvcv::FMT_RGBAf32})
            for (ParamMode mode : {ParamMode::Scalar, ParamMode::PerChannel})
                RunVarShapeFloatRefCase(fmt,
                                        {
                                            {1920, 4},
                                            { 257, 3}
                },
                                        mode, flags);
}

TEST_P(OpNormalize, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int      width       = GetParamValue<0>();
    int      height      = GetParamValue<1>();
    int      numImages   = GetParamValue<2>();
    bool     scalarBase  = GetParamValue<3>();
    bool     scalarScale = GetParamValue<4>();
    uint32_t flags       = GetParamValue<5>();
    float    globalScale = GetParamValue<6>();
    float    globalShift = GetParamValue<7>();
    float    epsilon     = GetParamValue<8>();

    nvcv::ImageFormat baseFormat  = (scalarBase ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);
    nvcv::ImageFormat scaleFormat = (scalarScale ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);

    nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    std::default_random_engine rng;

    // Create input varshape

    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    std::vector<int>                  srcVecRowStride(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);

        int srcRowStride   = imgSrc[i].size().w * fmt.numChannels();
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        std::ranges::generate(srcVec[i], [&udist, &rng]() { return udist(rng); });

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(), srcRowStride,
                               srcRowStride, // vec has no padding
                               imgSrc[i].size().h, cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create base tensor
    nvcv::Tensor imgBase(
        {
            {1, 1, 1, baseFormat.numChannels()},
            nvcv::TENSOR_NHWC
    },
        baseFormat.planeDataType(0));
    std::vector<float> baseVec(baseFormat.numChannels());
    {
        auto baseData = imgBase.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nullptr, baseData);
        auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
        ASSERT_TRUE(baseAccess);

        std::uniform_real_distribution<float> baseDist(0, 255.f);
        std::ranges::generate(baseVec, [&baseDist, &rng]() { return baseDist(rng); });

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(baseAccess->sampleData(0), baseAccess->rowStride(), baseVec.data(),
                                            baseVec.size() * sizeof(float),
                                            baseVec.size() * sizeof(float), // vec has no padding
                                            1, cudaMemcpyHostToDevice));
    }

    // Create scale tensor
    nvcv::Tensor imgScale(
        {
            {1, 1, 1, scaleFormat.numChannels()},
            nvcv::TENSOR_NHWC
    },
        scaleFormat.planeDataType(0));
    std::vector<float> scaleVec(scaleFormat.numChannels());
    {
        auto scaleData = imgScale.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nullptr, scaleData);
        auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*scaleData);
        ASSERT_TRUE(scaleAccess);

        std::uniform_real_distribution<float> scaleDist(0, 1.f);
        std::ranges::generate(scaleVec, [&scaleDist, &rng]() { return scaleDist(rng); });

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(scaleAccess->sampleData(0), scaleAccess->rowStride(), scaleVec.data(),
                                            scaleVec.size() * sizeof(float),
                                            scaleVec.size() * sizeof(float), // vec has no padding
                                            1, cudaMemcpyHostToDevice));
    }

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Generate test result
    cvcuda::Normalize normalizeOp;
    EXPECT_NO_THROW(
        normalizeOp(stream, batchSrc, imgBase, imgScale, batchDst, globalScale, globalShift, epsilon, flags));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);
        int sampleWidth  = srcData->plane(0).width;
        int sampleHeight = srcData->plane(0).height;

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int dstRowStride = srcVecRowStride[i];

        std::vector<uint8_t> testVec(sampleHeight * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               sampleHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(sampleHeight * dstRowStride);

        // Generate gold result
        Normalize({
            goldVec,
            dstRowStride,
            srcVec[i],
            srcVecRowStride[i],
            {sampleWidth, sampleHeight},
            fmt,
            baseVec,
            0,
            {          1,            1},
            baseFormat,
            scaleVec,
            0,
            {          1,            1},
            scaleFormat,
            globalScale,
            globalShift,
            epsilon,
            flags
        });

        EXPECT_THAT(testVec, t::ElementsAreArray(goldVec));
    }
}

// Shared body for the planar param-sweep correctness tests. Builds random interleaved input for
// numImages samples, runs Normalize with scalar-or-per-channel base/scale, and compares each
// re-interleaved output against the shared gold. varShape selects the image-batch path; otherwise a
// single NCHW tensor with numImages samples is used.
static void RunPlanarParamSweepCase(int width, int height, int numImages, bool scalarBase, bool scalarScale,
                                    uint32_t flags, float globalScale, float globalShift, float epsilon, bool varShape,
                                    bool spatialParams = false)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const bool baseScalar  = scalarBase && !spatialParams;
    const bool scaleScalar = scalarScale && !spatialParams;

    nvcv::ImageFormat fmt            = nvcv::FMT_RGBA8p;
    nvcv::ImageFormat baseFormat     = (baseScalar ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32p);
    nvcv::ImageFormat scaleFormat    = (scaleScalar ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32p);
    nvcv::ImageFormat baseHelperFmt  = (baseScalar ? nvcv::ImageFormat{NVCV_IMAGE_FORMAT_F32} : nvcv::FMT_RGBAf32);
    nvcv::ImageFormat scaleHelperFmt = (scaleScalar ? nvcv::ImageFormat{NVCV_IMAGE_FORMAT_F32} : nvcv::FMT_RGBAf32);
    const int         numChannels    = fmt.numChannels();
    const int         baseChannels   = baseFormat.numChannels();
    const int         scaleChannels  = scaleFormat.numChannels();

    std::default_random_engine rng;

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    std::vector<int>                  rowStride(numImages);
    std::vector<int>                  sampleW(numImages);
    std::vector<int>                  sampleH(numImages);

    // Source/destination containers; only the variant selected by varShape is populated.
    nvcv::Tensor             tensorSrc;
    nvcv::Tensor             tensorDst;
    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    nvcv::ImageBatchVarShape batchSrc(numImages);
    nvcv::ImageBatchVarShape batchDst(numImages);

    auto fillSrc = [&](int i, int w, int h)
    {
        sampleW[i]   = w;
        sampleH[i]   = h;
        rowStride[i] = w * numChannels;
        srcVec[i].resize(static_cast<size_t>(h) * rowStride[i]);
        std::uniform_int_distribution<uint8_t> udist(0, 255);
        std::ranges::generate(srcVec[i], [&]() { return udist(rng); });
    };

    if (varShape)
    {
        std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
        std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));
        for (int i = 0; i < numImages; ++i)
        {
            nvcv::Size2D size{udistWidth(rng), udistHeight(rng)};
            imgSrc.emplace_back(size, fmt);
            imgDst.emplace_back(size, fmt);
            fillSrc(i, size.w, size.h);
            ASSERT_NO_FATAL_FAILURE(UploadPlanarImage(imgSrc[i], srcVec[i]));
        }
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        batchDst.pushBack(imgDst.begin(), imgDst.end());
    }
    else
    {
        tensorSrc = nvcv::util::CreateTensor(numImages, width, height, fmt);
        tensorDst = nvcv::util::CreateTensor(numImages, width, height, fmt);

        auto srcData = tensorSrc.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nullptr, srcData);
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
        ASSERT_TRUE(srcAccess);
        ASSERT_TRUE(srcData->layout() == nvcv::TENSOR_NCHW || srcData->layout() == nvcv::TENSOR_CHW);

        for (int i = 0; i < numImages; ++i)
        {
            fillSrc(i, width, height);
            ASSERT_NO_FATAL_FAILURE(UploadPlanarTensorSample(*srcAccess, i, srcVec[i], width, height, numChannels));
        }
    }

    // base/scale: scalar [1,1,1,1] or per-channel [1,C,1,1]; broadcasting is handled in the kernel.
    nvcv::Tensor       imgBase;
    nvcv::Tensor       imgScale;
    std::vector<float> baseVec;
    std::vector<float> scaleVec;
    if (spatialParams)
    {
        ASSERT_FALSE(varShape);
        baseVec  = MakeAndUploadSpatialPlanarParam(imgBase, width, height, numChannels, 255.f);
        scaleVec = MakeAndUploadSpatialPlanarParam(imgScale, width, height, numChannels, 1.f);
    }
    else
    {
        baseVec  = MakeAndUploadRandomParam(imgBase, baseFormat, baseChannels, 255.f, rng);
        scaleVec = MakeAndUploadRandomParam(imgScale, scaleFormat, scaleChannels, 1.f, rng);
    }

    cvcuda::Normalize normalizeOp;
    if (varShape)
    {
        EXPECT_NO_THROW(
            normalizeOp(stream, batchSrc, imgBase, imgScale, batchDst, globalScale, globalShift, epsilon, flags));
    }
    else
    {
        EXPECT_NO_THROW(
            normalizeOp(stream, tensorSrc, imgBase, imgScale, tensorDst, globalScale, globalShift, epsilon, flags));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // For the tensor path, hold the output data/access alive for the compare loop.
    nvcv::Optional<nvcv::TensorDataStridedCuda>              dstData;
    nvcv::Optional<nvcv::TensorDataAccessStridedImagePlanar> dstAccess;
    if (!varShape)
    {
        dstData = tensorDst.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(dstData);
        dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
        ASSERT_TRUE(dstAccess);
    }

    // Param geometry is identical for every sample, so compute it once outside the compare loop.
    const nvcv::Size2D baseSize       = spatialParams ? nvcv::Size2D{width, height} : nvcv::Size2D{1, 1};
    const nvcv::Size2D scaleSize      = spatialParams ? nvcv::Size2D{width, height} : nvcv::Size2D{1, 1};
    const int          baseRowStride  = spatialParams ? width * numChannels : baseChannels;
    const int          scaleRowStride = spatialParams ? width * numChannels : scaleChannels;

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<uint8_t> testVec
            = varShape ? DownloadPlanarImage<uint8_t>(imgDst[i])
                       : DownloadPlanarTensorSample<uint8_t>(*dstAccess, i, sampleW[i], sampleH[i], numChannels);
        ASSERT_NO_FATAL_FAILURE(
            ExpectPlanarNormalizeMatchesGold(testVec, srcVec[i], rowStride[i], sampleW[i], sampleH[i], fmt, baseVec,
                                             baseRowStride, baseSize, baseHelperFmt, scaleVec, scaleRowStride,
                                             scaleSize, scaleHelperFmt, globalScale, globalShift, epsilon, flags));
    }
}

TEST_P(OpNormalize, tensor_planar_correct_output)
{
    ASSERT_NO_FATAL_FAILURE(RunPlanarParamSweepCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                                    GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>(),
                                                    GetParamValue<6>(), GetParamValue<7>(), GetParamValue<8>(),
                                                    /*varShape=*/false));
}

TEST_P(OpNormalize, varshape_planar_correct_output)
{
    ASSERT_NO_FATAL_FAILURE(RunPlanarParamSweepCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                                    GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>(),
                                                    GetParamValue<6>(), GetParamValue<7>(), GetParamValue<8>(),
                                                    /*varShape=*/true));
}

TEST(OpNormalize, tensor_planar_spatial_params_vec4_correct_output)
{
    RunPlanarParamSweepCase(65, 3, 1, false, false, normalScale, 1.25f, 0.5f, 0.f,
                            /*varShape=*/false, /*spatialParams=*/true);
}

TEST(OpNormalize, tensor_planar_rgb8_three_channel_correct_output)
{
    RunPlanarTensorNormalizeCase<uint8_t>(nvcv::FMT_RGB8p, 13, 11, 2);
}

TEST(OpNormalize, varshape_planar_rgb8_three_channel_correct_output)
{
    RunPlanarVarShapeNormalizeCase<uint8_t>(nvcv::FMT_RGB8p, 13, 11, 2);
}

TEST(OpNormalize, tensor_planar_f32_correct_output)
{
    RunPlanarTensorNormalizeCase<float>(nvcv::FMT_RGBAf32p, 9, 7, 2);
}

TEST(OpNormalize, varshape_planar_f32_correct_output)
{
    RunPlanarVarShapeNormalizeCase<float>(nvcv::FMT_RGBAf32p, 9, 7, 2);
}

TEST(OpNormalize, tensor_planar_s16_correct_output)
{
    const nvcv::ImageFormat fmt{nvcv::ColorModel::RGB,  nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                nvcv::DataKind::SIGNED, nvcv::Swizzle::S_XYZ0, nvcv::Packing::X16,
                                nvcv::Packing::X16,     nvcv::Packing::X16};
    RunPlanarTensorNormalizeCase<int16_t>(fmt, 13, 11, 2);
}

TEST(OpNormalize, tensor_planar_s8_preserves_signed_values)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int           width    = 8;
    constexpr int           height   = 1;
    constexpr int           channels = 3;
    const nvcv::TensorShape shape({1, channels, height, width}, nvcv::TENSOR_NCHW);
    nvcv::Tensor            src(shape, nvcv::TYPE_S8);
    nvcv::Tensor            dst(shape, nvcv::TYPE_S8);
    nvcv::Tensor            base(nvcv::TensorShape({1, channels, 1, 1}, nvcv::TENSOR_NCHW), nvcv::TYPE_F32);
    nvcv::Tensor            scale(nvcv::TensorShape({1, channels, 1, 1}, nvcv::TENSOR_NCHW), nvcv::TYPE_F32);

    ASSERT_NO_FATAL_FAILURE(FillPlanarParamTensor(base, {0.f, 0.f, 0.f}));
    ASSERT_NO_FATAL_FAILURE(FillPlanarParamTensor(scale, {1.f, 1.f, 1.f}));

    const std::vector<int8_t> input{-128, -127, -126, -96, -95, -94, -64, -63, -62, -1,  0,   1,
                                    0,    1,    2,    31,  32,  33,  63,  64,  65,  125, 126, 127};
    auto                      srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);
    ASSERT_EQ(0, srcAccess->rowStride() % 4);
    ASSERT_EQ(0, srcAccess->chStride() % 4);
    ASSERT_EQ(0, srcAccess->sampleStride() % 4);
    ASSERT_NO_FATAL_FAILURE(UploadPlanarTensorSample(*srcAccess, 0, input, width, height, channels));

    cvcuda::Normalize op;
    EXPECT_NO_THROW(op(stream, src, base, scale, dst, 1.f, 0.f, 0.f, normalScale));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);
    EXPECT_EQ(input, DownloadPlanarTensorSample<int8_t>(*dstAccess, 0, width, height, channels));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// -------------------------------------------------------------------------------------------------
// Tensor-free (by-value list / float4 base & scale) path.
//
// The by-value overload must produce output byte-identical to the tensor-based overload fed the same
// base/scale values -- the tensor path is the oracle (no external reference needed). The explicit cases
// below cover every dtype and each independent layout, channel, flag, parameter-mode, and global-profile
// value without multiplying unrelated axes.
// -------------------------------------------------------------------------------------------------
static float4 MakeFloat4(const std::vector<float> &vals)
{
    float4 v{0.f, 0.f, 0.f, 0.f};
    auto  *p = reinterpret_cast<float *>(&v);
    for (size_t i = 0; i < vals.size() && i < 4; ++i)
    {
        p[i] = vals[i];
    }
    return v;
}

static bool IsScalarPlanarLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

static nvcv::TensorShape MakeScalarTensorShape(nvcv::TensorLayout layout, int numImages, int channels, int height,
                                               int width)
{
    if (layout == nvcv::TENSOR_NHWC)
    {
        return nvcv::TensorShape({numImages, height, width, channels}, layout);
    }
    if (layout == nvcv::TENSOR_NCHW)
    {
        return nvcv::TensorShape({numImages, channels, height, width}, layout);
    }
    return nvcv::TensorShape({channels, height, width}, layout);
}

static std::vector<float> MakeScalarParamValues(int count, float initialValue, float increment)
{
    std::vector<float> values(count);
    for (int i = 0; i < count; ++i)
    {
        values[i] = initialValue + increment * static_cast<float>(i);
    }
    return values;
}

template<typename T>
static void UploadScalarTensorSample(nvcv::TensorDataAccessStridedImagePlanar &access, int sample,
                                     const std::vector<T> &values, int width, int height, int channels, bool isPlanar)
{
    if (isPlanar)
    {
        ASSERT_NO_FATAL_FAILURE(UploadPlanarTensorSample<T>(access, sample, values, width, height, channels));
        return;
    }

    const size_t rowBytes = static_cast<size_t>(width) * channels * sizeof(T);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(access.sampleData(sample), access.rowStride(), values.data(), rowBytes,
                                        rowBytes, height, cudaMemcpyHostToDevice));
}

template<typename T>
static std::vector<T> DownloadScalarTensorSample(nvcv::TensorDataAccessStridedImagePlanar &access, int sample,
                                                 int width, int height, int channels, bool isPlanar)
{
    if (isPlanar)
    {
        return DownloadPlanarTensorSample<T>(access, sample, width, height, channels);
    }

    const size_t   rowBytes = static_cast<size_t>(width) * channels * sizeof(T);
    std::vector<T> values(static_cast<size_t>(height) * width * channels);
    EXPECT_EQ(cudaSuccess, cudaMemcpy2D(values.data(), rowBytes, access.sampleData(sample), access.rowStride(),
                                        rowBytes, height, cudaMemcpyDeviceToHost));
    return values;
}

static void UploadScalarParamTensor(nvcv::Tensor &tensor, const std::vector<float> &values, bool isPlanar)
{
    if (isPlanar)
    {
        ASSERT_NO_FATAL_FAILURE(FillPlanarParamTensor(tensor, values));
        return;
    }
    ASSERT_NO_FATAL_FAILURE(UploadParamTensor(tensor, values));
}

template<typename T>
static void RunScalarBitIdentityCase(nvcv::DataType dtype, nvcv::TensorLayout layout, int channels, uint32_t flags,
                                     bool scalarBase, bool scalarScale, float gscale, float gshift, float eps)
{
    const bool isPlanar  = IsScalarPlanarLayout(layout);
    const int  numImages = layout == nvcv::TENSOR_CHW ? 1 : 2;
    const int  width     = 13;
    const int  height    = 9;
    SCOPED_TRACE(testing::Message() << "layout=" << layout << " C=" << channels << " flags=" << flags
                                    << " scalarBase=" << scalarBase << " scalarScale=" << scalarScale
                                    << " gscale=" << gscale << " gshift=" << gshift << " eps=" << eps);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const auto   shape = MakeScalarTensorShape(layout, numImages, channels, height, width);
    nvcv::Tensor src(shape, dtype);
    nvcv::Tensor dstRef(shape, dtype);
    nvcv::Tensor dstNew(shape, dtype);

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<T>> srcVec(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        srcVec[i] = MakePlanarHostImage<T>(i, width, height, channels);
        ASSERT_NO_FATAL_FAILURE(
            UploadScalarTensorSample<T>(*srcAccess, i, srcVec[i], width, height, channels, isPlanar));
    }

    const int baseCount  = scalarBase ? 1 : channels;
    const int scaleCount = scalarScale ? 1 : channels;
    auto      baseVals   = MakeScalarParamValues(baseCount, 0.40f, 0.05f);
    auto      scaleVals  = MakeScalarParamValues(scaleCount, 0.20f, 0.03f);

    const auto   paramLayout = isPlanar ? nvcv::TENSOR_NCHW : nvcv::TENSOR_NHWC;
    const auto   baseShape   = isPlanar ? nvcv::TensorShape({1, baseCount, 1, 1}, paramLayout)
                                        : nvcv::TensorShape({1, 1, 1, baseCount}, paramLayout);
    const auto   scaleShape  = isPlanar ? nvcv::TensorShape({1, scaleCount, 1, 1}, paramLayout)
                                        : nvcv::TensorShape({1, 1, 1, scaleCount}, paramLayout);
    nvcv::Tensor baseT(baseShape, nvcv::TYPE_F32);
    nvcv::Tensor scaleT(scaleShape, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadScalarParamTensor(baseT, baseVals, isPlanar));
    ASSERT_NO_FATAL_FAILURE(UploadScalarParamTensor(scaleT, scaleVals, isPlanar));

    cvcuda::Normalize op;
    ASSERT_NO_THROW(op(stream, src, baseT, scaleT, dstRef, gscale, gshift, eps, flags));
    ASSERT_NO_THROW(op(stream, src, MakeFloat4(baseVals), MakeFloat4(scaleVals), baseCount, scaleCount, dstNew, gscale,
                       gshift, eps, flags));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto refData = dstRef.exportData<nvcv::TensorDataStridedCuda>();
    auto newData = dstNew.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, refData);
    ASSERT_NE(nullptr, newData);
    auto refAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*refData);
    auto newAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*newData);
    ASSERT_TRUE(refAccess);
    ASSERT_TRUE(newAccess);

    for (int i = 0; i < numImages; ++i)
    {
        auto refVec = DownloadScalarTensorSample<T>(*refAccess, i, width, height, channels, isPlanar);
        auto newVec = DownloadScalarTensorSample<T>(*newAccess, i, width, height, channels, isPlanar);
        EXPECT_EQ(refVec, newVec) << "sample " << i;
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

using ScalarIdentityProfile = std::tuple<float, float, float>;
using ScalarIdentityParams  = std::tuple<nvcv::TensorLayout, int, uint32_t, bool, bool, ScalarIdentityProfile>;

static const ScalarIdentityProfile identityProfile{1.f, 0.f, 0.f};
static const ScalarIdentityProfile adjustedProfile{2.f, 5.f, 1e-4f};

class OpNormalizeScalarIdentity : public testing::TestWithParam<ScalarIdentityParams>
{
};

TEST_P(OpNormalizeScalarIdentity, matches_tensor_path)
{
    const auto &[layout, channels, flags, scalarBase, scalarScale, profile] = GetParam();
    const auto &[globalScale, globalShift, epsilon]                         = profile;

    RunScalarBitIdentityCase<uint8_t>(nvcv::TYPE_U8, layout, channels, flags, scalarBase, scalarScale, globalScale,
                                      globalShift, epsilon);
    RunScalarBitIdentityCase<int8_t>(nvcv::TYPE_S8, layout, channels, flags, scalarBase, scalarScale, globalScale,
                                     globalShift, epsilon);
    RunScalarBitIdentityCase<uint16_t>(nvcv::TYPE_U16, layout, channels, flags, scalarBase, scalarScale, globalScale,
                                       globalShift, epsilon);
    RunScalarBitIdentityCase<int16_t>(nvcv::TYPE_S16, layout, channels, flags, scalarBase, scalarScale, globalScale,
                                      globalShift, epsilon);
    RunScalarBitIdentityCase<int32_t>(nvcv::TYPE_S32, layout, channels, flags, scalarBase, scalarScale, globalScale,
                                      globalShift, epsilon);
    RunScalarBitIdentityCase<float>(nvcv::TYPE_F32, layout, channels, flags, scalarBase, scalarScale, globalScale,
                                    globalShift, epsilon);
}

INSTANTIATE_TEST_SUITE_P(
    AxisCovering, OpNormalizeScalarIdentity,
    testing::Values(ScalarIdentityParams{nvcv::TENSOR_NHWC, 1, normalScale, false, false, identityProfile},
                    ScalarIdentityParams{nvcv::TENSOR_NCHW, 3, scaleIsStdDev, true, true, adjustedProfile},
                    ScalarIdentityParams{nvcv::TENSOR_CHW, 4, normalScale, true, false, adjustedProfile}));

// Equivalent-layout parity (COV-PARITY / TST-7): the by-value path on native planar (NCHW) input must
// match the by-value path on interleaved (NHWC) input for the same logical pixels and per-channel
// base/scale -- i.e. no reformat is needed to get the planar result.
TEST(OpNormalizeScalar, planar_matches_interleaved)
{
    const int numImages = 2;
    const int width     = 13;
    const int height    = 9;
    const int channels  = 3;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor srcHWC(nvcv::TensorShape({numImages, height, width, channels}, nvcv::TENSOR_NHWC), nvcv::TYPE_F32);
    nvcv::Tensor dstHWC(nvcv::TensorShape({numImages, height, width, channels}, nvcv::TENSOR_NHWC), nvcv::TYPE_F32);
    nvcv::Tensor srcCHW(nvcv::TensorShape({numImages, channels, height, width}, nvcv::TENSOR_NCHW), nvcv::TYPE_F32);
    nvcv::Tensor dstCHW(nvcv::TensorShape({numImages, channels, height, width}, nvcv::TENSOR_NCHW), nvcv::TYPE_F32);

    auto hwcData = srcHWC.exportData<nvcv::TensorDataStridedCuda>();
    auto chwData = srcCHW.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, hwcData);
    ASSERT_NE(nullptr, chwData);
    auto hwcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*hwcData);
    auto chwAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*chwData);
    ASSERT_TRUE(hwcAccess);
    ASSERT_TRUE(chwAccess);

    // Same logical HWC-ordered images uploaded interleaved (NHWC) and deinterleaved (NCHW).
    std::vector<std::vector<float>> srcVec(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        srcVec[i] = MakePlanarHostImage<float>(i, width, height, channels);
        ASSERT_NO_FATAL_FAILURE(
            UploadScalarTensorSample<float>(*hwcAccess, i, srcVec[i], width, height, channels, false));
    }
    for (int i = 0; i < numImages; ++i)
    {
        ASSERT_NO_FATAL_FAILURE(
            UploadScalarTensorSample<float>(*chwAccess, i, srcVec[i], width, height, channels, true));
    }

    const std::vector<float> baseVals{0.40f, 0.45f, 0.50f};
    const std::vector<float> scaleVals{0.20f, 0.23f, 0.26f};

    cvcuda::Normalize op;
    ASSERT_NO_THROW(
        op(stream, srcHWC, MakeFloat4(baseVals), MakeFloat4(scaleVals), channels, channels, dstHWC, 2.f, 5.f, 0.f, 0));
    ASSERT_NO_THROW(
        op(stream, srcCHW, MakeFloat4(baseVals), MakeFloat4(scaleVals), channels, channels, dstCHW, 2.f, 5.f, 0.f, 0));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto dHWCData = dstHWC.exportData<nvcv::TensorDataStridedCuda>();
    auto dCHWData = dstCHW.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dHWCData);
    ASSERT_NE(nullptr, dCHWData);
    auto dHWCAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dHWCData);
    auto dCHWAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dCHWData);
    ASSERT_TRUE(dHWCAccess);
    ASSERT_TRUE(dCHWAccess);

    for (int i = 0; i < numImages; ++i)
    {
        auto outHWC = DownloadScalarTensorSample<float>(*dHWCAccess, i, width, height, channels, false);
        auto outCHW = DownloadScalarTensorSample<float>(*dCHWAccess, i, width, height, channels, true);
        EXPECT_EQ(outHWC, outCHW) << "sample " << i;
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpNormalizeScalar_Negative, rejects_invalid_param_count)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const nvcv::TensorShape shape({2, 9, 13, 3}, nvcv::TENSOR_NHWC); // 3-channel input
    auto                    src = nvcv::Tensor(shape, nvcv::TYPE_F32);
    auto                    dst = nvcv::Tensor(shape, nvcv::TYPE_F32);
    cvcuda::Normalize       op;
    float4                  v{0.5f, 0.5f, 0.5f, 0.f};

    // A count that is neither 1 (broadcast) nor the channel count (3) is rejected.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&]() { op(stream, src, v, v, 2, 3, dst, 1.f, 0.f, 0.f, 0); }));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&]() { op(stream, src, v, v, 3, 4, dst, 1.f, 0.f, 0.f, 0); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpNormalizeScalar_Negative, rejects_batch_exceeding_grid_z)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const nvcv::TensorShape shape({65536, 1, 1, 1}, nvcv::TENSOR_NHWC);
    nvcv::Tensor            src(shape, nvcv::TYPE_U8);
    nvcv::Tensor            dst(shape, nvcv::TYPE_U8);
    cvcuda::Normalize       op;
    float4                  base{0.f, 0.f, 0.f, 0.f};
    float4                  scale{1.f, 0.f, 0.f, 0.f};

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&]() { op(stream, src, base, scale, 1, 1, dst, 1.f, 0.f, 0.f, normalScale); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpNormalizeScalar_Negative, rejects_unsupported_channels)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // 2-channel input is not a supported channel count.
    const nvcv::TensorShape shape({2, 9, 13, 2}, nvcv::TENSOR_NHWC);
    auto                    src = nvcv::Tensor(shape, nvcv::TYPE_F32);
    auto                    dst = nvcv::Tensor(shape, nvcv::TYPE_F32);
    cvcuda::Normalize       op;
    float4                  v{0.5f, 0.5f, 0.f, 0.f};

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&]() { op(stream, src, v, v, 2, 2, dst, 1.f, 0.f, 0.f, 0); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpNormalizeScalar_Negative, rejects_mismatched_output_dtype)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // The header contract requires output dtype == input dtype; the kernels dispatch on the input
    // dtype and would write the wrong element type through the output wrap.
    const nvcv::TensorShape shape({2, 9, 13, 3}, nvcv::TENSOR_NHWC);
    auto                    src = nvcv::Tensor(shape, nvcv::TYPE_F32);
    auto                    dst = nvcv::Tensor(shape, nvcv::TYPE_S32);
    cvcuda::Normalize       op;
    float4                  v{0.5f, 0.5f, 0.5f, 0.f};

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&]() { op(stream, src, v, v, 3, 3, dst, 1.f, 0.f, 0.f, 0); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpNormalizeScalar_Negative, rejects_mismatched_output_shape)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // The header contract requires output extents == input extents; the kernels bounds-check
    // against the input extents only, so a smaller output would be written out of bounds.
    cvcuda::Normalize op;
    float4            v{0.5f, 0.5f, 0.5f, 0.f};

    nvcv::Tensor src(nvcv::TensorShape({2, 9, 13, 3}, nvcv::TENSOR_NHWC), nvcv::TYPE_F32);
    nvcv::Tensor dstSmallH(nvcv::TensorShape({2, 4, 13, 3}, nvcv::TENSOR_NHWC), nvcv::TYPE_F32);
    nvcv::Tensor dstSmallN(nvcv::TensorShape({1, 9, 13, 3}, nvcv::TENSOR_NHWC), nvcv::TYPE_F32);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&]() { op(stream, src, v, v, 3, 3, dstSmallH, 1.f, 0.f, 0.f, 0); }));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&]() { op(stream, src, v, v, 3, 3, dstSmallN, 1.f, 0.f, 0.f, 0); }));

    // Planar path validates the same contract.
    nvcv::Tensor srcP(nvcv::TensorShape({2, 3, 9, 13}, nvcv::TENSOR_NCHW), nvcv::TYPE_F32);
    nvcv::Tensor dstPSmallW(nvcv::TensorShape({2, 3, 9, 7}, nvcv::TENSOR_NCHW), nvcv::TYPE_F32);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&]() { op(stream, srcP, v, v, 3, 3, dstPSmallW, 1.f, 0.f, 0.f, 0); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpNormalize_Negative, tensor_planar_invalid_param_shape)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width     = 13;
    constexpr int height    = 11;
    constexpr int numImages = 2;
    constexpr int channels  = 3;

    nvcv::Tensor imgSrc(numImages, {width, height}, nvcv::FMT_RGB8p);
    nvcv::Tensor imgDst(numImages, {width, height}, nvcv::FMT_RGB8p);
    nvcv::Tensor imgBase(
        {
            {numImages + 1, channels, 1, 1},
            "NCHW"
    },
        nvcv::TYPE_F32);
    nvcv::Tensor imgScale(
        {
            {1, channels, 1, 1},
            "NCHW"
    },
        nvcv::TYPE_F32);

    cvcuda::Normalize normalizeOp;
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&] { normalizeOp(stream, imgSrc, imgBase, imgScale, imgDst, 1.f, 0.f, 0.f, normalScale); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpNormalize_Negative, varshape_planar_invalid_param_shape)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width     = 13;
    constexpr int height    = 11;
    constexpr int numImages = 2;
    constexpr int channels  = 3;

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{width + i, height + i}, nvcv::FMT_RGB8p);
        imgDst.emplace_back(imgSrc.back().size(), nvcv::FMT_RGB8p);
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::Normalize normalizeOp;

    auto expectInvalidArgument = [&](nvcv::Tensor &imgBase, nvcv::Tensor &imgScale)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall(
                      [&] { normalizeOp(stream, batchSrc, imgBase, imgScale, batchDst, 1.f, 0.f, 0.f, normalScale); }));
    };

    {
        nvcv::Tensor imgBase(
            {
                {numImages, channels, 1, 1},
                "NCHW"
        },
            nvcv::TYPE_F32);
        nvcv::Tensor imgScale(
            {
                {1, channels, 1, 1},
                "NCHW"
        },
            nvcv::TYPE_F32);

        expectInvalidArgument(imgBase, imgScale);
    }

    {
        nvcv::Tensor imgBase(
            {
                {1, channels, 1, 1},
                "NCHW"
        },
            nvcv::TYPE_F32);
        nvcv::Tensor imgScale(
            {
                {numImages, channels, 1, 1},
                "NCHW"
        },
            nvcv::TYPE_F32);

        expectInvalidArgument(imgBase, imgScale);
    }

    {
        nvcv::Tensor imgBase(
            {
                {1, 1, 1, channels},
                "NHWC"
        },
            nvcv::TYPE_F32);
        nvcv::Tensor imgScale(
            {
                {1, channels, 1, 1},
                "NCHW"
        },
            nvcv::TYPE_F32);

        expectInvalidArgument(imgBase, imgScale);
    }

    {
        nvcv::Tensor imgBase(
            {
                {1, channels, 1, 1},
                "NCHW"
        },
            nvcv::TYPE_F32);
        nvcv::Tensor imgScale(
            {
                {1, 1, 1, channels},
                "NHWC"
        },
            nvcv::TYPE_F32);

        expectInvalidArgument(imgBase, imgScale);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpNormalize_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, bool>{
    // inFmt, outFmt, isVarShapeDifferentFormatTest
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, false},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, true},
    {nvcv::FMT_2F32, nvcv::FMT_2F32, false},
    {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, false},
    {nvcv::FMT_U16, nvcv::FMT_U16, false},
});

// clang-format on

TEST_P(OpNormalize_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inFmt  = GetParamValue<0>();
    nvcv::ImageFormat outFmt = GetParamValue<1>();
    if (bool isVarShapeDifferentFormatTest = GetParamValue<2>(); isVarShapeDifferentFormatTest)
    {
        GTEST_SKIP() << "Skip varshape different format test for tensor test";
    }
    if (inFmt == nvcv::FMT_U16 && outFmt == nvcv::FMT_U16)
    {
        GTEST_SKIP() << "Skip U16 test for tensor test";
    }

    int width     = 24;
    int height    = 24;
    int numImages = 2;

    uint32_t flags       = normalScale;
    float    globalScale = 0.f;
    float    globalShift = 0.f;
    float    epsilon     = 0.f;

    int baseWidth      = 1;
    int scaleWidth     = 1;
    int baseHeight     = 1;
    int scaleHeight    = 1;
    int baseNumImages  = 1;
    int scaleNumImages = 1;

    nvcv::ImageFormat baseFormat  = nvcv::FMT_F32;
    nvcv::ImageFormat scaleFormat = nvcv::FMT_F32;

    // Create input and output tensors
    nvcv::Tensor imgSrc = nvcv::util::CreateTensor(numImages, width, height, inFmt);
    nvcv::Tensor imgBase(baseNumImages, {baseWidth, baseHeight}, baseFormat);
    nvcv::Tensor imgScale(scaleNumImages, {scaleWidth, scaleHeight}, scaleFormat);
    nvcv::Tensor imgDst = nvcv::util::CreateTensor(numImages, width, height, outFmt);

    // Generate test result
    cvcuda::Normalize normalizeOp;
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall(
            [&normalizeOp, &stream, &imgSrc, &imgBase, &imgScale, &imgDst, &globalScale, &globalShift, &epsilon, &flags]
            { normalizeOp(stream, imgSrc, imgBase, imgScale, imgDst, globalScale, globalShift, epsilon, flags); }));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpNormalize_Negative, varshape_op)
{
    nvcv::ImageFormat inFmt                         = GetParamValue<0>();
    nvcv::ImageFormat outFmt                        = GetParamValue<1>();
    bool              isVarShapeDifferentFormatTest = GetParamValue<2>();

    if (inFmt == nvcv::FMT_2F32 && outFmt == nvcv::FMT_2F32)
    {
        GTEST_SKIP() << "2-channel interleaved varshape normalize is supported";
    }

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<std::pair<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {isVarShapeDifferentFormatTest ? nvcv::FMT_U16 : inFmt,                                                 outFmt},
        {                                                inFmt, isVarShapeDifferentFormatTest ? nvcv::FMT_U16 : outFmt}
    };

    // check
    if (isVarShapeDifferentFormatTest)
    {
        ASSERT_NE(testSet[0].first, inFmt);
        ASSERT_EQ(testSet[0].second, outFmt);
        ASSERT_EQ(testSet[1].first, inFmt);
        ASSERT_NE(testSet[1].second, outFmt);
    }
    else
    {
        testSet.pop_back();
        ASSERT_EQ(testSet[0].first, inFmt);
        ASSERT_EQ(testSet[0].second, outFmt);
    }

    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
    {
        int width     = 24;
        int height    = 24;
        int numImages = 5;

        uint32_t flags       = normalScale;
        float    globalScale = 0.f;
        float    globalShift = 0.f;
        float    epsilon     = 0.f;

        nvcv::ImageFormat baseFormat  = nvcv::FMT_F32;
        nvcv::ImageFormat scaleFormat = nvcv::FMT_F32;

        std::default_random_engine rng;

        // Create input and output varshape

        std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
        std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;

        for (int i = 0; i < numImages - 1; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inFmt);
            imgDst.emplace_back(imgSrc[i].size(), outFmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(numImages);
        nvcv::ImageBatchVarShape batchDst(numImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Create base tensor
        nvcv::Tensor imgBase(
            {
                {1, 1, 1, baseFormat.numChannels()},
                nvcv::TENSOR_NHWC
        },
            baseFormat.planeDataType(0));

        // Create scale tensor
        nvcv::Tensor imgScale(
            {
                {1, 1, 1, scaleFormat.numChannels()},
                nvcv::TENSOR_NHWC
        },
            scaleFormat.planeDataType(0));

        // Generate test result
        cvcuda::Normalize normalizeOp;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                                   [&normalizeOp, &stream, &batchSrc, &imgBase, &imgScale, &batchDst,
                                                    &globalScale, &globalShift, &epsilon, &flags] {
                                                       normalizeOp(stream, batchSrc, imgBase, imgScale, batchDst,
                                                                   globalScale, globalShift, epsilon, flags);
                                                   }));

        // Get test data back
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpNormalize_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaNormalizeCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}

// =============================================================================
// Planar (NCHW/CHW) parity: Normalize treats channels independently, so the planar layout must
// produce exactly the same pixels as the interleaved path. These tests feed identical data in both
// layouts through cvcuda::Normalize and require the (re-interleaved) planar output to match the
// interleaved output bit-for-bit. Scalar (single-channel F32) base/scale are layout-agnostic, so any
// divergence is a real planar-vs-interleaved bug, not a parameter-layout artifact.
// =============================================================================

namespace {

// 1x1x1 single-channel F32 scalar param (base or scale), broadcast to every channel by Normalize.
nvcv::Tensor MakeScalarNormalizeParam(float value)
{
    nvcv::Tensor t    = nvcv::util::CreateTensor(1, 1, 1, nvcv::FMT_F32);
    auto         data = t.exportData<nvcv::TensorDataStridedCuda>();
    EXPECT_NE(nullptr, data);
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
    EXPECT_TRUE(access);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(access->sampleData(0), &value, sizeof(float), cudaMemcpyHostToDevice));
    return t;
}

// Normalize identical data in interleaved and planar tensor layout; outputs must match bit-for-bit.
// The shared scaffolding (upload/run/download/compare) lives in PlanarParityUtils.hpp; here we only
// bind the Normalize call with scalar base/scale (the same params for both layouts).
void RunNormalizePlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                                        int numImages)
{
    nvcv::Tensor base  = MakeScalarNormalizeParam(10.f);
    nvcv::Tensor scale = MakeScalarNormalizeParam(2.f);
    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, w, h, w, h, numImages,
        [&base, &scale](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::Normalize op;
            EXPECT_NO_THROW(op(s, src, base, scale, dst, 1.5f, 3.f, 0.f, 0));
        });
}

// Var-shape counterpart of RunNormalizePlanarParityTensorCase.
void RunNormalizePlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                                          int numImages)
{
    nvcv::Tensor base  = MakeScalarNormalizeParam(10.f);
    nvcv::Tensor scale = MakeScalarNormalizeParam(2.f);
    test::planar::RunVarShapeParity(planarFmt, interleavedFmt, w, h, w, h, numImages,
                                    [&base, &scale](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                                    const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::Normalize op;
                                        EXPECT_NO_THROW(op(s, src, base, scale, dst, 1.5f, 3.f, 0.f, 0));
                                    });
}

} // namespace

// Parameters: width, height, numImages, planarFmt, interleavedFmt
NVCV_TEST_SUITE_P(OpNormalizePlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
  // RGB8 (3 channel uint8).
                      {64, 48, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      {33, 17, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
 // RGBA8 (4 channel uint8).
                      {64, 48, 2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
 // Float planar (3 and 4 channel) -- exercises the float kernel path bit-exactly.
                      {64, 48, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                      {50, 40, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                      {64, 48, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

TEST_P(OpNormalizePlanar, tensor_matches_interleaved)
{
    RunNormalizePlanarParityTensorCase(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                       GetParamValue<2>());
}

TEST_P(OpNormalizePlanar, varshape_matches_interleaved)
{
    RunNormalizePlanarParityVarShapeCase(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                         GetParamValue<2>());
}

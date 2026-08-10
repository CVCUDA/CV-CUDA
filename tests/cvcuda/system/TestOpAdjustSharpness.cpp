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
#include <cvcuda/OpAdjustSharpness.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace test = nvcv::test;

namespace {

// Unsigned 16-bit RGB/RGBA formats are not predefined by NVCV.
#define NVCV_IMAGE_FORMAT_RGB16U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_RGBA16U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X16_Y16_Z16_W16)
#define NVCV_IMAGE_FORMAT_RGB16Up \
    NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, UNSIGNED, XYZ0, ASSOCIATED, X16, X16, X16)
#define NVCV_IMAGE_FORMAT_RGBA16Up \
    NVCV_DETAIL_MAKE_COLOR_FMT4(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X16, X16, X16, X16)

// Blend clamp bound per base type: dtype max for unsigned integers, 1.0 for float. Mirrors
// SharpnessBound<BT>() in the kernel and torchvision's _max_value.
template<typename BT>
float SharpnessRefBound()
{
    if constexpr (std::is_floating_point_v<BT>)
    {
        return 1.0f;
    }
    else
    {
        return static_cast<float>(std::numeric_limits<BT>::max());
    }
}

// Independent CPU gold for a single interior pixel/channel. This is a deliberate reimplementation of
// the documented oracle (torchvision adjust_sharpness), computed independently of the kernel. It uses
// the SAME fixed accumulation order and explicit `fmaf`/`rintf`, so the correctly-rounded IEEE-754
// result is bit-identical to the device kernel (AdjustSharpnessScalar in OpAdjustSharpness.cu) and
// the bit-exact EXPECT_EQ below is a real regression check. `c11` is the center (original) pixel.
template<typename BT>
BT AdjustSharpnessRefScalar(float c00, float c01, float c02, float c10, float c11, float c12, float c20, float c21,
                            float c22, float oneMinusFactor, float bound)
{
    constexpr float kEdge   = 1.0f / 13.0f;
    constexpr float kCenter = 5.0f / 13.0f;

    float blur = c11 * kCenter;
    blur       = std::fmaf(c00, kEdge, blur);
    blur       = std::fmaf(c01, kEdge, blur);
    blur       = std::fmaf(c02, kEdge, blur);
    blur       = std::fmaf(c10, kEdge, blur);
    blur       = std::fmaf(c12, kEdge, blur);
    blur       = std::fmaf(c20, kEdge, blur);
    blur       = std::fmaf(c21, kEdge, blur);
    blur       = std::fmaf(c22, kEdge, blur);

    if constexpr (!std::is_floating_point_v<BT>)
    {
        blur = std::rintf(blur);
    }

    const float out     = std::fmaf(oneMinusFactor, blur - c11, c11);
    const float clamped = std::fminf(std::fmaxf(out, 0.0f), bound);
    return static_cast<BT>(clamped);
}

// Byte-offset of element (n, y, x, c) in an interleaved (N)HWC buffer described by `strides`
// {sampleStride, rowStride, colStride} in bytes. Channel c sits `c*sizeof(BT)` into the pixel.
template<typename BT>
long ElemOffset(long3 strides, int n, int y, int x, int c)
{
    return n * strides.x + y * strides.y + x * strides.z + static_cast<long>(c) * static_cast<long>(sizeof(BT));
}

template<typename BT>
BT ReadElem(const std::vector<uint8_t> &buf, long3 strides, int n, int y, int x, int c)
{
    BT v;
    std::memcpy(&v, &buf[ElemOffset<BT>(strides, n, y, x, c)], sizeof(BT));
    return v;
}

template<typename BT>
void WriteElem(std::vector<uint8_t> &buf, long3 strides, int n, int y, int x, int c, BT v)
{
    std::memcpy(&buf[ElemOffset<BT>(strides, n, y, x, c)], &v, sizeof(BT));
}

// Fill an interleaved buffer with per-element typed random data: float in [0, 1] (matching a
// normalized float image, keeping the blend finite), integer in [0, dtype-max].
template<typename BT>
void FillTypedRandom(std::vector<uint8_t> &buf, long3 strides, int width, int height, int batches, int channels,
                     unsigned seed)
{
    std::mt19937 rng(seed);
    for (int i = 0; i < batches * height * width; ++i)
    {
        const int n = i / (height * width);
        const int y = i / width % height;
        const int x = i % width;
        for (int c = 0; c < channels; ++c)
        {
            BT v;
            if constexpr (std::is_floating_point_v<BT>)
            {
                std::uniform_real_distribution dist(0.0f, 1.0f);
                v = static_cast<BT>(dist(rng));
            }
            else
            {
                std::uniform_int_distribution dist(0, static_cast<int>(std::numeric_limits<BT>::max()));
                v = static_cast<BT>(dist(rng));
            }
            WriteElem<BT>(buf, strides, n, y, x, c, v);
        }
    }
}

// Full-buffer CPU gold: interior pixels are blended, the 1-pixel border is copied unchanged, and any
// image with a dimension < 3 is copied in full (every pixel is a border pixel).
template<typename BT>
void AdjustSharpnessGold(std::vector<uint8_t> &dst, const std::vector<uint8_t> &src, long3 strides, int width,
                         int height, int batches, int channels, float factor)
{
    const float oneMinusFactor = 1.0f - factor;
    const float bound          = SharpnessRefBound<BT>();
    for (int i = 0; i < batches * height * width; ++i)
    {
        const int  n      = i / (height * width);
        const int  y      = i / width % height;
        const int  x      = i % width;
        const bool border = (x == 0 || y == 0 || x == width - 1 || y == height - 1);
        for (int c = 0; c < channels; ++c)
        {
            if (border)
            {
                WriteElem<BT>(dst, strides, n, y, x, c, ReadElem<BT>(src, strides, n, y, x, c));
                continue;
            }
            const auto c00 = static_cast<float>(ReadElem<BT>(src, strides, n, y - 1, x - 1, c));
            const auto c01 = static_cast<float>(ReadElem<BT>(src, strides, n, y - 1, x, c));
            const auto c02 = static_cast<float>(ReadElem<BT>(src, strides, n, y - 1, x + 1, c));
            const auto c10 = static_cast<float>(ReadElem<BT>(src, strides, n, y, x - 1, c));
            const auto c11 = static_cast<float>(ReadElem<BT>(src, strides, n, y, x, c));
            const auto c12 = static_cast<float>(ReadElem<BT>(src, strides, n, y, x + 1, c));
            const auto c20 = static_cast<float>(ReadElem<BT>(src, strides, n, y + 1, x - 1, c));
            const auto c21 = static_cast<float>(ReadElem<BT>(src, strides, n, y + 1, x, c));
            const auto c22 = static_cast<float>(ReadElem<BT>(src, strides, n, y + 1, x + 1, c));
            WriteElem<BT>(
                dst, strides, n, y, x, c,
                AdjustSharpnessRefScalar<BT>(c00, c01, c02, c10, c11, c12, c20, c21, c22, oneMinusFactor, bound));
        }
    }
}

// Run the tensor path for one dtype and assert bit-exact equality with the CPU gold.
template<typename BT>
void RunCorrectness(int width, int height, int batches, nvcv::ImageFormat format, float factor)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int channels = format.numChannels();

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(inAccess && outAccess);

    long3 strides{inAccess->sampleStride(), inAccess->rowStride(), inAccess->colStride()};
    if (inData->rank() == 3) // HWC: no sample dimension, one image
    {
        strides.x = inAccess->numRows() * inAccess->rowStride();
    }
    const long bufSize = strides.x * batches;

    std::vector<uint8_t> inVec(bufSize, 0);
    std::vector<uint8_t> goldVec(bufSize, 0);
    std::vector<uint8_t> testVec(bufSize, 0);

    FillTypedRandom<BT>(inVec, strides, width, height, batches, channels, /*seed=*/7u);

    // Zero-fill the device output so any inter-row/-pixel padding compares equal to the zero-filled
    // gold (the kernel writes only valid pixels).
    ASSERT_EQ(cudaSuccess, cudaMemset(outData->basePtr(), 0, bufSize));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), bufSize, cudaMemcpyHostToDevice));

    cvcuda::AdjustSharpness op;
    EXPECT_NO_THROW(op(stream, inTensor, outTensor, factor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), bufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    AdjustSharpnessGold<BT>(goldVec, inVec, strides, width, height, batches, channels, factor);

    EXPECT_EQ(testVec, goldVec);
}

// Dispatch a fixed-format test case to the matching base type.
void RunCorrectnessDispatch(int width, int height, int batches, nvcv::ImageFormat format, float factor)
{
    const nvcv::DataType dtype = format.planeDataType(0);
    if (dtype == nvcv::TYPE_U8 || dtype == nvcv::TYPE_3U8 || dtype == nvcv::TYPE_4U8)
    {
        RunCorrectness<uint8_t>(width, height, batches, format, factor);
    }
    else if (dtype == nvcv::TYPE_U16 || dtype == nvcv::TYPE_3U16 || dtype == nvcv::TYPE_4U16)
    {
        RunCorrectness<uint16_t>(width, height, batches, format, factor);
    }
    else if (dtype == nvcv::TYPE_F32 || dtype == nvcv::TYPE_3F32 || dtype == nvcv::TYPE_4F32)
    {
        RunCorrectness<float>(width, height, batches, format, factor);
    }
    else
    {
        FAIL() << "Unhandled format in RunCorrectnessDispatch";
    }
}

// Create in/out tensors with the given formats and require the op to reject them.
void ExpectTensorRejected(nvcv::ImageFormat inFmt, nvcv::ImageFormat outFmt, float factor = 1.0f, int width = 24,
                          int height = 24)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(1, width, height, inFmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(1, width, height, outFmt);

    cvcuda::AdjustSharpness op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, inTensor, outTensor, factor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

} // namespace

// Tensor correctness over the declared dtype x channel matrix, plus factor sweep and small-image
// (border-only) passthrough. Bit-exact vs an independent CPU gold.
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSharpness, test::ValueList<int, int, int, nvcv::ImageFormat, float>
{
    //  width, height, batch,             format,   factor
    {      66,     55,     1,      nvcv::FMT_U8,     2.0f}, // u8  / 1ch, sharpen
    {      35,     33,     1,      nvcv::FMT_U8,     0.0f}, // u8  / 1ch, fully smoothed
    {     123,     67,     3,    nvcv::FMT_RGB8,     0.5f}, // u8  / 3ch
    {      42,     53,     4,   nvcv::FMT_RGBA8,     2.0f}, // u8  / 4ch
    {      40,     40,     2,    nvcv::FMT_RGB8,     1.0f}, // u8  / 3ch, identity
    {      80,     40,     2,     nvcv::FMT_U16,     1.5f}, // u16 / 1ch
    {      51,     49,     1,     nvcv::FMT_U16,     0.0f}, // u16 / 1ch, fully smoothed
    {      47,     39,     2, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB16U}, 0.5f}, // u16 / 3ch
    {      31,     37,     1, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA16U}, 2.0f}, // u16 / 4ch
    {      17,     19,     1,     nvcv::FMT_F32,     2.0f}, // f32 / 1ch
    {     101,     33,     2,  nvcv::FMT_RGBf32,     0.5f}, // f32 / 3ch
    {      64,     48,     3, nvcv::FMT_RGBAf32,     2.0f}, // f32 / 4ch
    {      23,     21,     1,  nvcv::FMT_RGBf32,     1.0f}, // f32 / 3ch, identity
    {       3,      3,     1,    nvcv::FMT_RGB8,     0.5f}, // minimal interior (single interior pixel)
    {       2,     10,     1,    nvcv::FMT_RGB8,     0.0f}, // width 2 -> all border, passthrough
    {      10,      2,     2,      nvcv::FMT_U8,     0.0f}, // height 2 -> all border, passthrough
    {       1,      1,     1,     nvcv::FMT_F32,     2.0f}, // 1x1 -> passthrough
});

// clang-format on
TEST_P(OpAdjustSharpness, correct_output)
{
    RunCorrectnessDispatch(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                           nvcv::ImageFormat{GetParamValue<3>()}, GetParamValue<4>());
}

// VarShape correctness: bit-exact vs the CPU gold across the declared dtype x channel matrix. Each
// image has its own size, so the gold is computed per image from its own strided buffer.
template<typename BT>
void RunVarShapeCorrectness(int batches, nvcv::ImageFormat format, float factor)
{
    const int chans = format.numChannels();

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::mt19937                  rng(11);
    std::uniform_int_distribution udW(60, 90);
    std::uniform_int_distribution udH(50, 70);

    std::vector<nvcv::Image>          imgSrc;
    std::vector<nvcv::Image>          imgDst;
    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  rowStride(batches);
    std::vector<int2>                 sizes(batches);

    for (int i = 0; i < batches; ++i)
    {
        const int w = udW(rng);
        const int h = udH(rng);
        sizes[i]    = int2{w, h};
        imgSrc.emplace_back(nvcv::Size2D{w, h}, format);
        imgDst.emplace_back(nvcv::Size2D{w, h}, format);

        rowStride[i] = w * format.planePixelStrideBytes(0);
        srcVec[i].resize(static_cast<size_t>(h) * rowStride[i], 0);
        long3 str{static_cast<long>(h) * rowStride[i], rowStride[i], format.planePixelStrideBytes(0)};
        FillTypedRandom<BT>(srcVec[i], str, w, h, 1, chans, 100u + i);

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    rowStride[i], rowStride[i], h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    nvcv::ImageBatchVarShape batchDst(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::AdjustSharpness op;
    EXPECT_NO_THROW(op(stream, batchSrc, batchDst, factor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);
        const int w = sizes[i].x;
        const int h = sizes[i].y;
        long3     str{static_cast<long>(h) * rowStride[i], rowStride[i], format.planePixelStrideBytes(0)};

        std::vector<uint8_t> testVec(static_cast<size_t>(h) * rowStride[i], 0);
        std::vector<uint8_t> goldVec(static_cast<size_t>(h) * rowStride[i], 0);

        auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowStride[i], dstData->plane(0).basePtr,
                                            dstData->plane(0).rowStride, rowStride[i], h, cudaMemcpyDeviceToHost));

        AdjustSharpnessGold<BT>(goldVec, srcVec[i], str, w, h, 1, chans, factor);
        EXPECT_EQ(testVec, goldVec);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void RunVarShapeCorrectnessDispatch(int batches, nvcv::ImageFormat format, float factor)
{
    const nvcv::DataType dtype = format.planeDataType(0);
    if (dtype == nvcv::TYPE_U8 || dtype == nvcv::TYPE_3U8 || dtype == nvcv::TYPE_4U8)
    {
        RunVarShapeCorrectness<uint8_t>(batches, format, factor);
    }
    else if (dtype == nvcv::TYPE_U16 || dtype == nvcv::TYPE_3U16 || dtype == nvcv::TYPE_4U16)
    {
        RunVarShapeCorrectness<uint16_t>(batches, format, factor);
    }
    else if (dtype == nvcv::TYPE_F32 || dtype == nvcv::TYPE_3F32 || dtype == nvcv::TYPE_4F32)
    {
        RunVarShapeCorrectness<float>(batches, format, factor);
    }
    else
    {
        FAIL() << "Unhandled format in RunVarShapeCorrectnessDispatch";
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSharpnessVarShape, test::ValueList<int, nvcv::ImageFormat, float>{
    {2, nvcv::FMT_U8,                                            2.0f},
    {3, nvcv::FMT_RGB8,                                         0.5f},
    {2, nvcv::FMT_RGBA8,                                        1.5f},
    {2, nvcv::FMT_U16,                                          0.5f},
    {3, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB16U},             2.0f},
    {2, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA16U},            1.5f},
    {2, nvcv::FMT_F32,                                          2.0f},
    {3, nvcv::FMT_RGBf32,                                       0.5f},
    {2, nvcv::FMT_RGBAf32,                                      1.5f},
});

// clang-format on

TEST_P(OpAdjustSharpnessVarShape, correct_output)
{
    RunVarShapeCorrectnessDispatch(GetParamValue<0>(), nvcv::ImageFormat{GetParamValue<1>()}, GetParamValue<2>());
}

// Planar == interleaved parity (the op treats channels independently). ---------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSharpnessPlanar,
                  test::ValueList<int, int, int, float, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2, 2.0f,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 67,  51, 1, 0.5f,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 64,  48, 1, 2.0f,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    { 73,  61, 2, 0.5f, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB16Up},
                         nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB16U}},
    { 58,  46, 1, 2.0f, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA16Up},
                         nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA16U}},
    { 50,  40, 2, 1.5f,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    {100,  80, 2, 0.5f, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

// clang-format on
TEST_P(OpAdjustSharpnessPlanar, tensor_matches_interleaved)
{
    const float factor = GetParamValue<3>();
    test::planar::RunTensorParity(
        GetParamValue<4>(), GetParamValue<5>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [factor](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::AdjustSharpness op;
            EXPECT_NO_THROW(op(s, src, dst, factor));
        });
}

TEST_P(OpAdjustSharpnessPlanar, varshape_matches_interleaved)
{
    if (GetParamValue<4>() == nvcv::FMT_RGBA8p)
    {
        GTEST_SKIP() << "uchar4 planar var-shape unsupported by the image API";
    }
    const float factor = GetParamValue<3>();
    test::planar::RunVarShapeParity(GetParamValue<4>(), GetParamValue<5>(), GetParamValue<0>(), GetParamValue<1>(),
                                    GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                    [factor](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                             const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::AdjustSharpness op;
                                        EXPECT_NO_THROW(op(s, src, dst, factor));
                                    });
}

// Negative tests: the complement of the support matrix must be rejected. -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSharpness_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_F16,    nvcv::FMT_F16   }, // unsupported dtype (16-bit float)
    {nvcv::FMT_S16,    nvcv::FMT_S16   }, // unsupported dtype (signed 16-bit)
    {nvcv::FMT_RGB8,   nvcv::FMT_RGB8p }, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,   nvcv::FMT_RGBf32}, // input/output data type mismatch
});

// clang-format on
TEST_P(OpAdjustSharpness_Negative, rejects_unsupported)
{
    ExpectTensorRejected(GetParamValue<0>(), GetParamValue<1>());
}

TEST(OpAdjustSharpness_Negative, rejects_two_channel)
{
    ExpectTensorRejected(nvcv::FMT_2F32, nvcv::FMT_2F32, 1.0f, 16, 16);
}

TEST(OpAdjustSharpness_Negative, rejects_negative_factor)
{
    ExpectTensorRejected(nvcv::FMT_RGB8, nvcv::FMT_RGB8, -0.1f);
}

TEST(OpAdjustSharpness_Negative, varshape_rejects_negative_factor)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Image              srcImage({16, 16}, nvcv::FMT_RGB8);
    nvcv::Image              dstImage({16, 16}, nvcv::FMT_RGB8);
    nvcv::ImageBatchVarShape srcBatch(1);
    nvcv::ImageBatchVarShape dstBatch(1);
    srcBatch.pushBack(srcImage);
    dstBatch.pushBack(dstImage);

    cvcuda::AdjustSharpness op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, stream, &srcBatch, &dstBatch] { op(stream, srcBatch, dstBatch, -0.1f); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustSharpness_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaAdjustSharpnessCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}

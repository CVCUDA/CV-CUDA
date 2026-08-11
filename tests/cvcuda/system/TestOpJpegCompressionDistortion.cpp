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

#include "ElementwiseOpHarness.hpp" // ExpectRejected, RunTensorCorrectBuffer
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpJpegCompressionDistortion.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace test = nvcv::test;
namespace ew   = nvcv::test::elementwise;

namespace {

// Independent CPU gold -----------------------------------------------------------------------
//
// Reimplements the operator's documented pipeline (DALI's JpegCompressionDistortion fixed to
// 4:2:0): full-range JFIF YCbCr with uint8 intermediate storage, chroma from the 2x2 RGB box
// average, level shift, per-8x8-block forward DCT (rows then columns), Annex-K quantization
// scaled by the libjpeg quality mapping, inverse DCT (columns then rows), and nearest-neighbor
// chroma upsampling. Every multiply-add uses the same canonical fmaf order as the kernel; all
// other operations (add/sub, lone multiply, 1/x, roundf, round-to-nearest-even saturating cast)
// are correctly rounded, so the gold matches the GPU output bit-exactly. Constants are
// redeclared here rather than shared with the implementation header.

constexpr float kA    = 1.387039845322148f;  // sqrt(2) * cos(    pi / 16)
constexpr float kB    = 1.306562964876377f;  // sqrt(2) * cos(    pi /  8)
constexpr float kC    = 1.175875602419359f;  // sqrt(2) * cos(3 * pi / 16)
constexpr float kD    = 0.785694958387102f;  // sqrt(2) * cos(5 * pi / 16)
constexpr float kE    = 0.541196100146197f;  // sqrt(2) * cos(3 * pi /  8)
constexpr float kF    = 0.275899379282943f;  // sqrt(2) * cos(7 * pi / 16)
constexpr float kNorm = 0.3535533905932737f; // 1 / sqrt(8)

constexpr std::array<uint8_t, 64> kLumaQuantBase = {
    16, 11, 10, 16, 24,  40,  51,  61,  //
    12, 12, 14, 19, 26,  58,  60,  55,  //
    14, 13, 16, 24, 40,  57,  69,  56,  //
    14, 17, 22, 29, 51,  87,  80,  62,  //
    18, 22, 37, 56, 68,  109, 103, 77,  //
    24, 35, 55, 64, 81,  104, 113, 92,  //
    49, 64, 78, 87, 103, 121, 120, 101, //
    72, 92, 95, 98, 112, 100, 103, 99,  //
};

constexpr std::array<uint8_t, 64> kChromaQuantBase = {
    17, 18, 24, 47, 99, 99, 99, 99, //
    18, 21, 26, 66, 99, 99, 99, 99, //
    24, 26, 56, 99, 99, 99, 99, 99, //
    47, 66, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
};

struct RgbU8
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

// Round-to-nearest-even saturating cast (the CPU equivalent of the device's cvt.rni.sat.u8.f32).
uint8_t GoldSatCastU8(float v)
{
    const float r = std::rint(v); // ties-to-even under the default rounding mode
    return static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, r)));
}

float GoldDot2(float c0, float v0, float c1, float v1)
{
    return std::fmaf(c0, v0, c1 * v1);
}

float GoldDot3(float c0, float v0, float c1, float v1, float c2, float v2)
{
    return std::fmaf(c0, v0, GoldDot2(c1, v1, c2, v2));
}

float GoldDot4(float c0, float v0, float c1, float v1, float c2, float v2, float c3, float v3)
{
    return std::fmaf(c0, v0, GoldDot3(c1, v1, c2, v2, c3, v3));
}

float GoldDot3Bias(float c0, float v0, float c1, float v1, float c2, float v2, float bias)
{
    return std::fmaf(c0, v0, std::fmaf(c1, v1, std::fmaf(c2, v2, bias)));
}

float GoldQuantScale(int quality)
{
    const int q = std::clamp(quality, 1, 100);
    return q < 50 ? 50.0f / static_cast<float>(q) : 2.0f - static_cast<float>(2 * q) / 100.0f;
}

// Half-away-from-zero rounding (DALI host-build / libjpeg semantics); ties matter at q=75.
float GoldQuantEntry(float scale, uint8_t base)
{
    const float entry = std::roundf(scale * static_cast<float>(base));
    return std::clamp(entry, 1.0f, 255.0f);
}

float GoldQuantize(float value, float q)
{
    return q * std::roundf(value * (1.0f / q));
}

uint8_t GoldRgbToY(RgbU8 p)
{
    return GoldSatCastU8(GoldDot3(0.299f, p.r, 0.587f, p.g, 0.114f, p.b));
}

uint8_t GoldRgbToCb(RgbU8 p)
{
    return GoldSatCastU8(GoldDot3Bias(-0.16873589f, p.r, -0.33126411f, p.g, 0.5f, p.b, 128.0f));
}

uint8_t GoldRgbToCr(RgbU8 p)
{
    return GoldSatCastU8(GoldDot3Bias(0.5f, p.r, -0.41868759f, p.g, -0.08131241f, p.b, 128.0f));
}

RgbU8 GoldYCbCrToRgb(uint8_t y, uint8_t cb, uint8_t cr)
{
    const auto  ys = static_cast<float>(y);
    const float tb = static_cast<float>(cb) - 128.0f;
    const float tr = static_cast<float>(cr) - 128.0f;
    return RgbU8{GoldSatCastU8(std::fmaf(1.402f, tr, ys)),
                 GoldSatCastU8(std::fmaf(-0.714136285f, tr, std::fmaf(-0.344136285f, tb, ys))),
                 GoldSatCastU8(std::fmaf(1.772f, tb, ys))};
}

RgbU8 GoldAvg4(RgbU8 p00, RgbU8 p01, RgbU8 p10, RgbU8 p11)
{
    return RgbU8{GoldSatCastU8(static_cast<float>(p00.r + p01.r + p10.r + p11.r) * 0.25f),
                 GoldSatCastU8(static_cast<float>(p00.g + p01.g + p10.g + p11.g) * 0.25f),
                 GoldSatCastU8(static_cast<float>(p00.b + p01.b + p10.b + p11.b) * 0.25f)};
}

void GoldFwdDct8(float *data, int stride)
{
    float x0 = data[0 * stride];
    float x1 = data[1 * stride];
    float x2 = data[2 * stride];
    float x3 = data[3 * stride];
    float x4 = data[4 * stride];
    float x5 = data[5 * stride];
    float x6 = data[6 * stride];
    float x7 = data[7 * stride];

    const float tmp0 = x0 + x7;
    const float tmp1 = x1 + x6;
    const float tmp2 = x2 + x5;
    const float tmp3 = x3 + x4;

    const float tmp4 = x0 - x7;
    const float tmp5 = x6 - x1;
    const float tmp6 = x2 - x5;
    const float tmp7 = x4 - x3;

    const float tmp8  = tmp0 + tmp3;
    const float tmp9  = tmp0 - tmp3;
    const float tmp10 = tmp1 + tmp2;
    const float tmp11 = tmp1 - tmp2;

    x0 = kNorm * (tmp8 + tmp10);
    x2 = kNorm * GoldDot2(kB, tmp9, kE, tmp11);
    x4 = kNorm * (tmp8 - tmp10);
    x6 = kNorm * GoldDot2(kE, tmp9, -kB, tmp11);

    x1 = kNorm * GoldDot4(kA, tmp4, -kC, tmp5, kD, tmp6, -kF, tmp7);
    x3 = kNorm * GoldDot4(kC, tmp4, kF, tmp5, -kA, tmp6, kD, tmp7);
    x5 = kNorm * GoldDot4(kD, tmp4, kA, tmp5, kF, tmp6, -kC, tmp7);
    x7 = kNorm * GoldDot4(kF, tmp4, kD, tmp5, kC, tmp6, kA, tmp7);

    data[0 * stride] = x0;
    data[1 * stride] = x1;
    data[2 * stride] = x2;
    data[3 * stride] = x3;
    data[4 * stride] = x4;
    data[5 * stride] = x5;
    data[6 * stride] = x6;
    data[7 * stride] = x7;
}

void GoldInvDct8(float *data, int stride)
{
    float x0 = data[0 * stride];
    float x1 = data[1 * stride];
    float x2 = data[2 * stride];
    float x3 = data[3 * stride];
    float x4 = data[4 * stride];
    float x5 = data[5 * stride];
    float x6 = data[6 * stride];
    float x7 = data[7 * stride];

    const float tmp0 = x0 + x4;
    const float tmp1 = GoldDot2(kB, x2, kE, x6);

    const float tmp2 = tmp0 + tmp1;
    const float tmp3 = tmp0 - tmp1;
    const float tmp4 = GoldDot4(kF, x7, kA, x1, kC, x3, kD, x5);
    const float tmp5 = GoldDot4(kA, x7, -kF, x1, kD, x3, -kC, x5);

    const float tmp6 = x0 - x4;
    const float tmp7 = GoldDot2(kE, x2, -kB, x6);

    const float tmp8  = tmp6 + tmp7;
    const float tmp9  = tmp6 - tmp7;
    const float tmp10 = GoldDot4(kC, x1, -kD, x7, -kF, x3, -kA, x5);
    const float tmp11 = GoldDot4(kD, x1, kC, x7, -kA, x3, kF, x5);

    x0 = kNorm * (tmp2 + tmp4);
    x7 = kNorm * (tmp2 - tmp4);
    x4 = kNorm * (tmp3 + tmp5);
    x3 = kNorm * (tmp3 - tmp5);

    x1 = kNorm * (tmp8 + tmp10);
    x5 = kNorm * (tmp9 - tmp11);
    x2 = kNorm * (tmp9 + tmp11);
    x6 = kNorm * (tmp8 - tmp10);

    data[0 * stride] = x0;
    data[1 * stride] = x1;
    data[2 * stride] = x2;
    data[3 * stride] = x3;
    data[4 * stride] = x4;
    data[5 * stride] = x5;
    data[6 * stride] = x6;
    data[7 * stride] = x7;
}

// Forward DCT (rows then columns), quantization, inverse DCT (columns then rows) on one 8x8
// block at blk within a plane of row pitch planeW.
void GoldDctQuantIdctBlock(float *blk, int planeW, const std::array<float, 64> &table)
{
    for (int r = 0; r < 8; ++r)
    {
        GoldFwdDct8(blk + static_cast<size_t>(r) * planeW, 1);
    }
    for (int c = 0; c < 8; ++c)
    {
        GoldFwdDct8(blk + c, planeW);
    }
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            float &v = blk[static_cast<size_t>(i) * planeW + j];
            v        = GoldQuantize(v, table[static_cast<size_t>(i) * 8 + j]);
        }
    }
    for (int c = 0; c < 8; ++c)
    {
        GoldInvDct8(blk + c, planeW);
    }
    for (int r = 0; r < 8; ++r)
    {
        GoldInvDct8(blk + static_cast<size_t>(r) * planeW, 1);
    }
}

// Applies the block round trip over every 8x8 block of a padded plane whose dimensions are
// multiples of 8.
void GoldDctQuantIdctPlane(std::vector<float> &plane, int planeW, int planeH, const std::array<float, 64> &table)
{
    for (int by = 0; by < planeH / 8; ++by)
    {
        for (int bx = 0; bx < planeW / 8; ++bx)
        {
            GoldDctQuantIdctBlock(&plane[static_cast<size_t>(by) * 8 * planeW + static_cast<size_t>(bx) * 8], planeW,
                                  table);
        }
    }
}

inline int PadTo8(int v)
{
    return (v + 7) / 8 * 8;
}

// Full-pipeline gold on one interleaved HWC (or single-channel HW) image.
std::vector<uint8_t> JpegDistortionGold(const std::vector<uint8_t> &in, int w, int h, int channels, int quality)
{
    const float           scale = GoldQuantScale(quality);
    std::array<float, 64> lumaTable{};
    std::array<float, 64> chromaTable{};
    for (int i = 0; i < 64; ++i)
    {
        lumaTable[i]   = GoldQuantEntry(scale, kLumaQuantBase[i]);
        chromaTable[i] = GoldQuantEntry(scale, kChromaQuantBase[i]);
    }

    std::vector<uint8_t> out(in.size());

    if (channels == 1)
    {
        const int          planeW = PadTo8(w);
        const int          planeH = PadTo8(h);
        std::vector<float> plane(static_cast<size_t>(planeW) * planeH);
        for (int y = 0; y < planeH; ++y)
        {
            for (int x = 0; x < planeW; ++x)
            {
                const uint8_t v = in[static_cast<size_t>(std::min(y, h - 1)) * w + std::min(x, w - 1)];
                plane[static_cast<size_t>(y) * planeW + x] = static_cast<float>(v) - 128.0f;
            }
        }
        GoldDctQuantIdctPlane(plane, planeW, planeH, lumaTable);
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                out[static_cast<size_t>(y) * w + x]
                    = GoldSatCastU8(plane[static_cast<size_t>(y) * planeW + x] + 128.0f);
            }
        }
        return out;
    }

    // 4:2:0 color path: the luma plane covers twice the padded chroma grid, so partially-in-image
    // luma blocks see the same edge-replicated content the kernel loads.
    const int chromaW = PadTo8((w + 1) / 2);
    const int chromaH = PadTo8((h + 1) / 2);
    const int lumaW   = 2 * chromaW;
    const int lumaH   = 2 * chromaH;

    const auto pixel = [&in, w, h](int y, int x)
    {
        const size_t ofs = (static_cast<size_t>(std::min(y, h - 1)) * w + std::min(x, w - 1)) * 3;
        return RgbU8{in[ofs], in[ofs + 1], in[ofs + 2]};
    };

    std::vector<float> luma(static_cast<size_t>(lumaW) * lumaH);
    for (int y = 0; y < lumaH; ++y)
    {
        for (int x = 0; x < lumaW; ++x)
        {
            luma[static_cast<size_t>(y) * lumaW + x] = static_cast<float>(GoldRgbToY(pixel(y, x))) - 128.0f;
        }
    }

    std::vector<float> cb(static_cast<size_t>(chromaW) * chromaH);
    std::vector<float> cr(cb.size());
    for (int cy = 0; cy < chromaH; ++cy)
    {
        for (int cx = 0; cx < chromaW; ++cx)
        {
            const RgbU8 avg = GoldAvg4(pixel(2 * cy, 2 * cx), pixel(2 * cy, 2 * cx + 1), pixel(2 * cy + 1, 2 * cx),
                                       pixel(2 * cy + 1, 2 * cx + 1));
            cb[static_cast<size_t>(cy) * chromaW + cx] = static_cast<float>(GoldRgbToCb(avg)) - 128.0f;
            cr[static_cast<size_t>(cy) * chromaW + cx] = static_cast<float>(GoldRgbToCr(avg)) - 128.0f;
        }
    }

    GoldDctQuantIdctPlane(luma, lumaW, lumaH, lumaTable);
    GoldDctQuantIdctPlane(cb, chromaW, chromaH, chromaTable);
    GoldDctQuantIdctPlane(cr, chromaW, chromaH, chromaTable);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            const size_t  cofs = static_cast<size_t>(y / 2) * chromaW + x / 2;
            const uint8_t y8   = GoldSatCastU8(luma[static_cast<size_t>(y) * lumaW + x] + 128.0f);
            const uint8_t cb8  = GoldSatCastU8(cb[cofs] + 128.0f);
            const uint8_t cr8  = GoldSatCastU8(cr[cofs] + 128.0f);
            const RgbU8   rgb  = GoldYCbCrToRgb(y8, cb8, cr8);

            const size_t ofs = (static_cast<size_t>(y) * w + x) * 3;
            out[ofs]         = rgb.r;
            out[ofs + 1]     = rgb.g;
            out[ofs + 2]     = rgb.b;
        }
    }
    return out;
}

// Bind the operator with a fixed scalar quality for the harness / parity / negative helpers.
auto invokeQuality(int quality)
{
    return [quality](cudaStream_t s, const auto &in, auto &out)
    {
        cvcuda::JpegCompressionDistortion op;
        op(s, in, out, quality);
    };
}

} // namespace

// Tensor correctness over the declared dtype x channel matrix + edge factors --------------------
// clang-format off
NVCV_TEST_SUITE_P(OpJpegCompressionDistortion, test::ValueList<int, int, int, nvcv::ImageFormat, int>
{
    //   width, height, batch,          format,  quality
    {       64,     32,     2,   nvcv::FMT_RGB8,      50}, // block-aligned batch
    {      123,     67,     2,   nvcv::FMT_RGB8,      50}, // odd size (edge-replicated blocks)
    {      123,     67,     1,   nvcv::FMT_RGB8,       1}, // strongest distortion
    {      123,     67,     1,   nvcv::FMT_RGB8,     100}, // weakest distortion
    {       17,     19,     1,   nvcv::FMT_RGB8,      10}, // small odd size
    {        9,      7,     1,   nvcv::FMT_RGB8,      95}, // sub-block image
    {        8,      8,     1,   nvcv::FMT_RGB8,      50}, // single block
    {      123,     67,     2,   nvcv::FMT_RGB8,      75}, // exact-tie table scale (0.5, half-away)
    {       64,     32,     2,   nvcv::FMT_U8,        50}, // grayscale, block-aligned
    {       64,     32,     1,   nvcv::FMT_U8,        75}, // grayscale, exact-tie table scale
    {      123,     67,     1,   nvcv::FMT_U8,        10}, // grayscale, odd size
    {       17,     19,     1,   nvcv::FMT_U8,       100}, // grayscale, small odd size
    {        9,      7,     1,   nvcv::FMT_U8,         1}, // grayscale, sub-block image
});

// clang-format on
TEST_P(OpJpegCompressionDistortion, correct_output)
{
    const int               w       = GetParamValue<0>();
    const int               h       = GetParamValue<1>();
    const int               b       = GetParamValue<2>();
    const nvcv::ImageFormat fmt     = GetParamValue<3>();
    const int               quality = GetParamValue<4>();

    ew::RunTensorCorrectBuffer<uint8_t>(
        w, h, b, fmt,
        [w, h, quality](const std::vector<uint8_t> &in, int channels)
        { return JpegDistortionGold(in, w, h, channels, quality); },
        invokeQuality(quality));
}

// Per-image quality tensor: every sample must be distorted with its own quality value; values
// outside [1, 100] are clamped on the device (DALI semantics).
TEST(OpJpegCompressionDistortion, per_image_quality_tensor)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int w = 40;
    constexpr int h = 25;

    const std::vector<int32_t> qualities{5, 50, 75, 95, 0, 200}; // 0 / 200 exercise the device clamp
    const std::vector<int>     goldQualities{5, 50, 75, 95, 1, 100};
    const auto                 batch = static_cast<int>(qualities.size());

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batch, w, h, nvcv::FMT_RGB8);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batch, w, h, nvcv::FMT_RGB8);
    auto         inData    = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto         outData   = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    for (int s = 0; s < batch; ++s)
    {
        std::vector<uint8_t> in(static_cast<size_t>(w) * h * 3);
        ew::FillDeterministicValues(in, static_cast<size_t>(s));
        nvcv::util::SetImageTensorFromVector<uint8_t>(*inData, in, s);
    }

    nvcv::Tensor qualityTensor({{batch}, "N"}, nvcv::TYPE_S32);
    test::planar::UploadTensorValues(qualityTensor, qualities);

    cvcuda::JpegCompressionDistortion op;
    ASSERT_NO_THROW(op(stream, inTensor, outTensor, qualityTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int s = 0; s < batch; ++s)
    {
        SCOPED_TRACE(s);
        std::vector<uint8_t> in;
        std::vector<uint8_t> got;
        nvcv::util::GetImageVectorFromTensor<uint8_t>(*inData, s, in);
        nvcv::util::GetImageVectorFromTensor<uint8_t>(*outData, s, got);
        EXPECT_EQ(JpegDistortionGold(in, w, h, 3, goldQualities[s]), got);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// VarShape correctness: per-image sizes with both a scalar quality and a per-image tensor.
// clang-format off
NVCV_TEST_SUITE_P(OpJpegCompressionDistortionVarShape, test::ValueList<nvcv::ImageFormat, bool>
{
    {nvcv::FMT_RGB8,  false}, // scalar quality
    {nvcv::FMT_RGB8,  true }, // per-image quality tensor
    {nvcv::FMT_U8,    false}, // grayscale, scalar quality
    {nvcv::FMT_U8,    true }, // grayscale, per-image quality tensor
});

// clang-format on
TEST_P(OpJpegCompressionDistortionVarShape, varshape_correct_output)
{
    const nvcv::ImageFormat    fmt       = GetParamValue<0>();
    const bool                 perImage  = GetParamValue<1>();
    const int                  channels  = fmt.numChannels();
    constexpr std::array       widths    = {23, 57, 89};
    constexpr std::array       heights   = {31, 71, 43};
    const auto                 numImages = static_cast<int>(widths.size());
    const std::vector<int32_t> qualities{15, 75, 85};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image>          srcImgs;
    std::vector<nvcv::Image>          dstImgs;
    std::vector<std::vector<uint8_t>> inputs(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        srcImgs.emplace_back(nvcv::Size2D{widths[i], heights[i]}, fmt);
        dstImgs.emplace_back(nvcv::Size2D{widths[i], heights[i]}, fmt);

        const size_t rowBytes = static_cast<size_t>(widths[i]) * channels;
        inputs[i].resize(rowBytes * heights[i]);
        ew::FillDeterministicValues(inputs[i], static_cast<size_t>(i));

        auto idata = srcImgs[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(idata, nullptr);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(idata->plane(0).basePtr, idata->plane(0).rowStride, inputs[i].data(),
                                            rowBytes, rowBytes, heights[i], cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape src(numImages);
    nvcv::ImageBatchVarShape dst(numImages);
    src.pushBack(srcImgs.begin(), srcImgs.end());
    dst.pushBack(dstImgs.begin(), dstImgs.end());

    cvcuda::JpegCompressionDistortion op;
    if (perImage)
    {
        nvcv::Tensor qualityTensor({{numImages}, "N"}, nvcv::TYPE_S32);
        test::planar::UploadTensorValues(qualityTensor, qualities);
        ASSERT_NO_THROW(op(stream, src, dst, qualityTensor));
    }
    else
    {
        ASSERT_NO_THROW(op(stream, src, dst, 50));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        const size_t         rowBytes = static_cast<size_t>(widths[i]) * channels;
        std::vector<uint8_t> got(rowBytes * heights[i]);
        auto                 odata = dstImgs[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(odata, nullptr);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(got.data(), rowBytes, odata->plane(0).basePtr, odata->plane(0).rowStride,
                                            rowBytes, heights[i], cudaMemcpyDeviceToHost));

        const int quality = perImage ? qualities[i] : 50;
        EXPECT_EQ(JpegDistortionGold(inputs[i], widths[i], heights[i], channels, quality), got);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpJpegCompressionDistortion, zero_extent_tensors_are_noops)
{
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {0, 3, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeQuality(50));
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {1, 0, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeQuality(50));
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {1, 3, 0, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeQuality(50));
}

TEST(OpJpegCompressionDistortion, empty_matching_varshape_is_noop)
{
    ew::ExpectEmptyVarShapeNoop(invokeQuality(50));
}

// Planar == interleaved parity ------------------------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpJpegCompressionDistortionPlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    { 48,  32, 2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
});

// clang-format on
TEST_P(OpJpegCompressionDistortionPlanar, tensor_matches_interleaved)
{
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeQuality(50)(s, src, dst)); });
}

TEST_P(OpJpegCompressionDistortionPlanar, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invokeQuality(50)(s, src, dst)); });
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpJpegCompressionDistortion_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_U16,      nvcv::FMT_U16   }, // unsupported dtype (16-bit unsigned)
    {nvcv::FMT_F32,      nvcv::FMT_F32   }, // unsupported dtype (32-bit float)
    {nvcv::FMT_S8,       nvcv::FMT_S8    }, // unsupported dtype (8-bit signed)
    {nvcv::FMT_RGBA8,    nvcv::FMT_RGBA8 }, // unsupported channel count (4)
    {nvcv::FMT_RGB8,     nvcv::FMT_RGB8p }, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,     nvcv::FMT_RGBf32}, // input/output data type mismatch
});

// clang-format on
TEST_P(OpJpegCompressionDistortion_Negative, rejects_unsupported)
{
    ew::ExpectRejected(GetParamValue<0>(), GetParamValue<1>(), invokeQuality(50));
}

TEST(OpJpegCompressionDistortion_Negative, rejects_two_channel)
{
    // 2-channel uint8 (no predefined FMT_2U8) so the rejection isolates the channel count, not the dtype.
    constexpr nvcv::ImageFormat fmt2U8{NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XY00, ASSOCIATED, X8_Y8)};
    ew::ExpectRejected(fmt2U8, fmt2U8, invokeQuality(50), 16, 16);
}

TEST(OpJpegCompressionDistortion_Negative, rejects_too_tall_image)
{
    // 1x1048576 grayscale: the luma grid needs DivUp(1048576, 16) = 65536 rows, past the CUDA grid.y limit.
    ew::ExpectRejected(nvcv::FMT_U8, nvcv::FMT_U8, invokeQuality(50), 1, 1048576);
}

TEST(OpJpegCompressionDistortion_Negative, rejects_non_rgb_varshape_formats)
{
    ew::ExpectVarShapeRejected(nvcv::FMT_YUV8, invokeQuality(50));
    ew::ExpectVarShapeRejected(nvcv::FMT_YUV8p, invokeQuality(50));
    ew::ExpectVarShapeRejected(nvcv::FMT_BGR8, invokeQuality(50));
    ew::ExpectVarShapeRejected(nvcv::FMT_NV12, invokeQuality(50));
}

TEST(OpJpegCompressionDistortion_Negative, rejects_vector_tensor_dtype)
{
    nvcv::Tensor src(
        nvcv::TensorShape{
            {1, 2, 4, 1},
            "NHWC"
    },
        nvcv::TYPE_3U8);
    nvcv::Tensor dst(
        nvcv::TensorShape{
            {1, 2, 4, 1},
            "NHWC"
    },
        nvcv::TYPE_3U8);
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { cvcuda::JpegCompressionDistortion{}(stream, src, dst, 50); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpJpegCompressionDistortion_Negative, rejects_out_of_range_scalar_quality)
{
    ew::ExpectRejected(nvcv::FMT_RGB8, nvcv::FMT_RGB8, invokeQuality(0));
    ew::ExpectRejected(nvcv::FMT_RGB8, nvcv::FMT_RGB8, invokeQuality(101));
}

TEST(OpJpegCompressionDistortion_Negative, rejects_bad_quality_tensor)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int batch     = 2;
    nvcv::Tensor  inTensor  = nvcv::util::CreateTensor(batch, 16, 16, nvcv::FMT_RGB8);
    nvcv::Tensor  outTensor = nvcv::util::CreateTensor(batch, 16, 16, nvcv::FMT_RGB8);

    cvcuda::JpegCompressionDistortion op;

    nvcv::Tensor wrongDtype({{batch}, "N"}, nvcv::TYPE_F32);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, inTensor, outTensor, wrongDtype); }));

    nvcv::Tensor wrongRank(
        {
            {batch, 1},
            "NC"
    },
        nvcv::TYPE_S32);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, inTensor, outTensor, wrongRank); }));

    nvcv::Tensor wrongLength({{batch + 1}, "N"}, nvcv::TYPE_S32);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, inTensor, outTensor, wrongLength); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpJpegCompressionDistortion_Negative, rejects_mismatched_varshape_image_sizes)
{
    std::vector<nvcv::Image> srcImages{
        nvcv::Image{{17, 19}, nvcv::FMT_RGB8},
        nvcv::Image{{31, 37}, nvcv::FMT_RGB8}
    };
    std::vector<nvcv::Image> dstImages{
        nvcv::Image{{17, 19}, nvcv::FMT_RGB8},
        nvcv::Image{{23, 29}, nvcv::FMT_RGB8}
    };

    nvcv::ImageBatchVarShape src(2);
    nvcv::ImageBatchVarShape dst(2);
    src.pushBack(srcImages.begin(), srcImages.end());
    dst.pushBack(dstImages.begin(), dstImages.end());

    cvcuda::JpegCompressionDistortion op;
    try
    {
        op(nullptr, src, dst, 50);
        FAIL() << "Expected mismatched image sizes to be rejected";
    }
    catch (const nvcv::Exception &e)
    {
        EXPECT_EQ(nvcv::Status::ERROR_INVALID_ARGUMENT, e.code());
        EXPECT_STREQ("Input and output image 1 sizes must match: input is 31x37, output is 23x29", e.msg());
    }
}

TEST(OpJpegCompressionDistortion_Negative, create_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaJpegCompressionDistortionCreate(nullptr));
}

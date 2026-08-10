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

#include "ConvUtils.hpp"
#include "Definitions.hpp"
#include "PlanarParityUtils.hpp"

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpSIFT.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <bitset>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// ----------------------- Basic utility definitions ---------------------------

namespace cuda = nvcv::cuda;
namespace test = nvcv::test;
namespace type = nvcv::test::type;
namespace util = nvcv::util;

using VT = uint8_t; // value type, SIFT only accepts U8
using WT = float;   // work type, SIFT internal work type is F32

using RawPyramidType = std::vector<std::vector<std::vector<uint8_t>>>;
using RawBufferType  = std::vector<uint8_t>;

// Define a descriptor type that (1) compares equal if its Hamming distance to another descriptor is 99% close and
// (2) prints out as a hexadecimal string of 128 bytes.
struct DescriptorType
{
    bool operator==(const DescriptorType &other) const
    {
        int hammingDist = 0;
        for (int i = 0; i < static_cast<int>(data.size()); i++)
        {
            hammingDist += std::bitset<8>(data[i] ^ other.data[i]).count(); // NOSONAR: SIFT descriptors are bytes.
        }
        return (hammingDist < static_cast<int>((data.size() * sizeof(uint8_t) * 8) / 100));
    }

    friend std::ostream &operator<<(std::ostream &out, const DescriptorType &desc)
    {
        std::ios_base::fmtflags f{out.flags()};
        out << "0x";
        for (uint8_t byte : desc.data)
        {
            out << std::hex << std::setfill('0') << std::setw(2) // NOSONAR: std::format is C++20.
                << static_cast<unsigned int>(byte);
        }
        out.flags(f);
        return out;
    }

    std::array<uint8_t, 128> data;
};

constexpr nvcv::ImageFormat kInFormat{nvcv::FMT_U8};
constexpr nvcv::ImageFormat kWorkFormat{nvcv::FMT_F32};
constexpr NVCVBorderType    kBorderGauss{NVCV_BORDER_REFLECT101};
constexpr float4            kBorderValues{0.f, 0.f, 0.f, 0.f};
constexpr int               kImageBorder{5};
constexpr float2            kScale{0.f, 0.f};
constexpr float             kMinSigma{0.01f};
constexpr float             kPrevSigma{0.5f};
constexpr int2              kMaxKernelSize{59, 59};

constexpr int   kMaxInterpolationSteps = 5;
constexpr float kOrientationSigma      = 1.5f;
constexpr float kOrientationRadius     = 3 * kOrientationSigma;
constexpr float kHistogramPeakRatio    = 0.8f;
constexpr int   kHistogramBins         = 36;
constexpr int   kDescHistBins          = 8;
constexpr int   kDescOriRadius         = 3;
constexpr int   kDescWidth             = 4;
constexpr float kDescWidthToRadius     = static_cast<float>(M_SQRT2) * static_cast<float>(kDescWidth + 1) * .5f;
constexpr int   kDescMaxRadius         = 51;
constexpr float kDescWeightScale       = -1.f / (kDescWidth * kDescWidth * .5f);
constexpr float kDescHistPeakRatio     = .2f;
constexpr int   kDescHistSize          = (kDescWidth + 2) * (kDescWidth + 2) * (kDescHistBins + 2);

using DescriptorHistogram  = std::array<float, kDescHistSize>;
using OrientationHistogram = std::array<float, kHistogramBins>;

static nvcv::Tensor CreateSIFTInputTensor(int3 inShape, std::string_view layout)
{
    if (layout == "HWC")
    {
        EXPECT_EQ(inShape.z, 1);
        return nvcv::Tensor(
            {
                {inShape.y, inShape.x, 1},
                "HWC"
        },
            nvcv::TYPE_U8);
    }
    if (layout == "CHW")
    {
        EXPECT_EQ(inShape.z, 1);
        return nvcv::Tensor(
            {
                {1, inShape.y, inShape.x},
                "CHW"
        },
            nvcv::TYPE_U8);
    }
    if (layout == "NCHW")
    {
        return nvcv::Tensor(
            {
                {inShape.z, 1, inShape.y, inShape.x},
                "NCHW"
        },
            nvcv::TYPE_U8);
    }

    if (layout == "NHWC")
    {
        return nvcv::Tensor(
            {
                {inShape.z, inShape.y, inShape.x, 1},
                "NHWC"
        },
            nvcv::TYPE_U8);
    }

    throw std::invalid_argument("Unsupported SIFT input layout: " + std::string(layout));
}

static std::default_random_engine &Rng()
{
    static std::default_random_engine rng(0); // seed 0 to fix pseudo-randomness
    return rng;
}

// --------------------- Gold (reference) definitions --------------------------

inline int GoldNumberOfOctaves(int width, int height)
{
    return static_cast<int>(std::floor(std::log2(std::min(width, height)))) - 2;
}

inline nvcv::Size2D GoldKernelSize(float sigma)
{
    int ksize = std::min(static_cast<int>(std::round(sigma * 8.f + 1.f)) | 1, kMaxKernelSize.x);

    return nvcv::Size2D{ksize, ksize};
}

inline void GoldGaussianSigmas(std::vector<float> &layerSigmas, float initSigma, int numOctaveLayers)
{
    layerSigmas[0] = initSigma;

    float k = std::pow(2.f, 1.f / static_cast<float>(numOctaveLayers));
    float prevSigma;
    float totalSigma;

    for (int i = 1; i < numOctaveLayers + 3; i++)
    {
        prevSigma  = std::pow(k, static_cast<float>(i - 1)) * initSigma;
        totalSigma = k * prevSigma;

        layerSigmas[i] = std::sqrt(totalSigma * totalSigma - prevSigma * prevSigma);
    }
}

template<typename T, typename U>
inline void GoldCopyPixel(RawBufferType &dstVec, const long3 &dstStrides, const RawBufferType &srcVec,
                          const long3 &srcStrides, const long3 &srcShape, const float2 &srcScale, long x, long y,
                          long z)
{
    float2 srcCoord;

    if (srcScale.x >= 1.f)
    {
        srcCoord.x = static_cast<float>(z) * srcScale.x;
        srcCoord.y = static_cast<float>(y) * srcScale.y;

        util::ValueAt<T>(dstVec, dstStrides, long3{x, y, z})
            = test::GoldInterp<NVCV_INTERP_NEAREST, NVCV_BORDER_REPLICATE>(
                srcVec, srcStrides, int2{static_cast<int>(srcShape.z), static_cast<int>(srcShape.y)}, U{}, kScale,
                srcCoord, static_cast<int>(x));
    }
    else
    {
        srcCoord.x = (static_cast<float>(z) + .5f) * srcScale.x - .5f;
        srcCoord.y = (static_cast<float>(y) + .5f) * srcScale.y - .5f;

        util::ValueAt<T>(dstVec, dstStrides, long3{x, y, z})
            = test::GoldInterp<NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE>(
                srcVec, srcStrides, int2{static_cast<int>(srcShape.z), static_cast<int>(srcShape.y)}, U{}, kScale,
                srcCoord, static_cast<int>(x));
    }
}

template<typename T, typename U>
inline void GoldCopy(RawBufferType &dstVec, const long3 &dstStrides, const long3 &dstShape, const RawBufferType &srcVec,
                     const long3 &srcStrides, const long3 &srcShape)
{
    ASSERT_EQ(dstShape.x, srcShape.x);

    float2 srcScale{(float)srcShape.z / dstShape.z, (float)srcShape.y / dstShape.y};

    for (long x = 0; x < dstShape.x; x++)
    {
        for (long y = 0; y < dstShape.y; y++)
        {
            for (long z = 0; z < dstShape.z; z++)
            {
                GoldCopyPixel<T, U>(dstVec, dstStrides, srcVec, srcStrides, srcShape, srcScale, x, y, z);
            }
        }
    }
}

template<typename T>
inline void GoldSubtract(RawBufferType &dstVec, const RawBufferType &aVec, const RawBufferType &bVec,
                         const long3 &strides, const long3 &shape)
{
    for (long x = 0; x < shape.x; x++)
    {
        for (long y = 0; y < shape.y; y++)
        {
            for (long z = 0; z < shape.z; z++)
            {
                util::ValueAt<T>(dstVec, strides, long3{x, y, z})
                    = util::ValueAt<T>(bVec, strides, long3{x, y, z}) - util::ValueAt<T>(aVec, strides, long3{x, y, z});
            }
        }
    }
}

inline long3 GoldStrides(const long3 &shape)
{
    const auto elemStride = static_cast<long>(sizeof(WT));
    return long3{shape.y * shape.z * elemStride, shape.z * elemStride, elemStride};
}

inline int3 ConvolveShape(const long3 &shape)
{
    return int3{static_cast<int>(shape.z), static_cast<int>(shape.y),
                static_cast<int>(shape.x)}; // test::Convolve expects shape as WHN int3 instead of NHW long3
}

inline void GoldGeneratePyramids(RawPyramidType &dstGaussianPyramid, RawPyramidType &dstDoGPyramid, long3 &baseStrides,
                                 long3 &baseShape, int &numOctaves, const RawBufferType &srcVec,
                                 const long3 &srcStrides, const long3 &srcShape, bool expandInput, float initSigma,
                                 int numOctaveLayers)
{
    baseShape = expandInput ? long3{srcShape.x, srcShape.y * 2, srcShape.z * 2} : srcShape;

    numOctaves = GoldNumberOfOctaves(static_cast<int>(baseShape.z), static_cast<int>(baseShape.y));

    baseStrides = GoldStrides(baseShape);

    RawBufferType srcBase(baseShape.x * baseStrides.x);

    GoldCopy<WT, VT>(srcBase, baseStrides, baseShape, srcVec, srcStrides, srcShape);

    std::vector<float> layerSigmas(numOctaveLayers + 3);

    GoldGaussianSigmas(layerSigmas, initSigma, numOctaveLayers);

    RawBufferType srcGaussBase(baseShape.x * baseStrides.x);

    int   srcScale = expandInput ? 4 : 1;
    float sigma    = layerSigmas[0];

    sigma = std::sqrt(std::max(sigma * sigma - kPrevSigma * kPrevSigma * static_cast<float>(srcScale), kMinSigma));

    double2 sigma2{sigma, sigma};

    int2               kernelAnchor{-1, -1};
    nvcv::Size2D       kernelSize = GoldKernelSize(sigma);
    std::vector<float> kernel     = test::ComputeGaussianKernel(kernelSize, sigma2);

    long3 currShape   = baseShape;
    long3 currStrides = baseStrides;
    long3 prevShape   = currShape;
    long3 prevStrides = currStrides;

    test::Convolve(srcGaussBase, baseStrides, srcBase, baseStrides, ConvolveShape(currShape), kWorkFormat, kernel,
                   kernelSize, kernelAnchor, kBorderGauss, kBorderValues);

    dstGaussianPyramid.resize(numOctaves);
    dstDoGPyramid.resize(numOctaves);

    for (int octave = 0; octave < numOctaves; octave++)
    {
        dstGaussianPyramid[octave].resize(numOctaveLayers + 3);
        dstDoGPyramid[octave].resize(numOctaveLayers + 2);

        dstGaussianPyramid[octave][0].resize(currShape.x * currStrides.x);

        if (octave > 0)
            GoldCopy<WT, WT>(dstGaussianPyramid[octave][0], currStrides, currShape,
                             dstGaussianPyramid[octave - 1][numOctaveLayers], prevStrides, prevShape);
        else
            GoldCopy<WT, WT>(dstGaussianPyramid[octave][0], currStrides, currShape, srcGaussBase, prevStrides,
                             prevShape);

        for (int layer = 1; layer < numOctaveLayers + 3; layer++)
        {
            dstGaussianPyramid[octave][layer].resize(currShape.x * currStrides.x);

            sigma        = layerSigmas[layer];
            sigma2       = double2{sigma, sigma};
            kernelSize   = GoldKernelSize(sigma);
            kernel       = test::ComputeGaussianKernel(kernelSize, sigma2);
            kernelAnchor = int2{-1, -1};

            test::Convolve(dstGaussianPyramid[octave][layer], currStrides, dstGaussianPyramid[octave][layer - 1],
                           currStrides, ConvolveShape(currShape), kWorkFormat, kernel, kernelSize, kernelAnchor,
                           kBorderGauss, kBorderValues);
        }

        for (int layer = 0; layer < numOctaveLayers + 2; layer++)
        {
            dstDoGPyramid[octave][layer].resize(currShape.x * currStrides.x);

            GoldSubtract<WT>(dstDoGPyramid[octave][layer], dstGaussianPyramid[octave][layer],
                             dstGaussianPyramid[octave][layer + 1], currStrides, currShape);
        }

        prevShape   = currShape;
        prevStrides = currStrides;
        currShape.y /= 2;
        currShape.z /= 2;
        currStrides = GoldStrides(currShape);
    }
}

inline int GoldWrapBin(int bin, int numBins)
{
    if (bin >= numBins)
    {
        return bin - numBins;
    }
    if (bin < 0)
    {
        return bin + numBins;
    }
    return bin;
}

inline float GoldWrapAngle(float angle)
{
    if (angle < 0.f)
    {
        return angle + 360.f;
    }
    if (angle >= 360.f)
    {
        return angle - 360.f;
    }
    return angle;
}

inline int GoldDescriptorHistIndex(int row, int col, int bin)
{
    return ((row + 1) * (kDescWidth + 2) + (col + 1)) * (kDescHistBins + 2) + bin;
}

template<typename GaussValue>
inline void GoldAccumulateDescriptorSample(DescriptorHistogram &histogram, const GaussValue &gaussVal, float angle,
                                           float cos_a, float sin_a, const long3 &currShape, int r, int c, int i, int j)
{
    if (r + i <= 0 || r + i >= currShape.y - 1 || c + j <= 0 || c + j >= currShape.z - 1)
    {
        return;
    }

    auto  r_rot  = static_cast<float>(j) * sin_a + static_cast<float>(i) * cos_a;
    auto  c_rot  = static_cast<float>(j) * cos_a - static_cast<float>(i) * sin_a;
    float weight = r_rot * r_rot + c_rot * c_rot;

    r_rot += kDescWidth / 2 - .5f;
    c_rot += kDescWidth / 2 - .5f;

    if (r_rot <= -1 || r_rot >= kDescWidth || c_rot <= -1 || c_rot >= kDescWidth)
    {
        return;
    }

    auto iHist = static_cast<int>(std::floor(r_rot));
    auto jHist = static_cast<int>(std::floor(c_rot));

    r_rot -= static_cast<float>(iHist);
    c_rot -= static_cast<float>(jHist);

    float dx = gaussVal(r + i + 0, c + j + 1) - gaussVal(r + i + 0, c + j - 1);
    float dy = gaussVal(r + i - 1, c + j + 0) - gaussVal(r + i + 1, c + j + 0);

    float o_rot = (GoldWrapAngle(std::atan2(dy, dx) * 180.f / static_cast<float>(M_PI)) - angle)
                * static_cast<float>(kDescHistBins) / 360.f;
    auto bin = static_cast<int>(std::floor(o_rot));

    o_rot -= static_cast<float>(bin);
    bin = GoldWrapBin(bin, kDescHistBins);

    weight = std::exp2f(weight * kDescWeightScale);

    float magnitude = std::sqrt(dx * dx + dy * dy) * weight;

    float v_r1     = magnitude * r_rot;
    float v_r0     = magnitude - v_r1;
    float v_rc11   = v_r1 * c_rot;
    float v_rc10   = v_r1 - v_rc11;
    float v_rc01   = v_r0 * c_rot;
    float v_rc00   = v_r0 - v_rc01;
    float v_rco111 = v_rc11 * o_rot;
    float v_rco110 = v_rc11 - v_rco111;
    float v_rco101 = v_rc10 * o_rot;
    float v_rco100 = v_rc10 - v_rco101;
    float v_rco011 = v_rc01 * o_rot;
    float v_rco010 = v_rc01 - v_rco011;
    float v_rco001 = v_rc00 * o_rot;
    float v_rco000 = v_rc00 - v_rco001;

    int idx = GoldDescriptorHistIndex(iHist, jHist, bin);

    histogram[idx] += v_rco000;
    histogram[idx + 1] += v_rco001;
    histogram[idx + (kDescHistBins + 2)] += v_rco010;
    histogram[idx + (kDescHistBins + 3)] += v_rco011;
    histogram[idx + (kDescWidth + 2) * (kDescHistBins + 2)] += v_rco100;
    histogram[idx + (kDescWidth + 2) * (kDescHistBins + 2) + 1] += v_rco101;
    histogram[idx + (kDescWidth + 3) * (kDescHistBins + 2)] += v_rco110;
    histogram[idx + (kDescWidth + 3) * (kDescHistBins + 2) + 1] += v_rco111;
}

inline float GoldDescriptorNorm(DescriptorHistogram &histogram)
{
    float norm = 0.f;

    for (int i = 0; i < kDescWidth; i++)
    {
        for (int j = 0; j < kDescWidth; j++)
        {
            int histIdx = GoldDescriptorHistIndex(i, j, 0);

            histogram[histIdx] += histogram[histIdx + kDescHistBins];
            histogram[histIdx + 1] += histogram[histIdx + kDescHistBins + 1];

            for (int bin = 0; bin < kDescHistBins; bin++)
            {
                float magnitude = histogram[histIdx + bin];
                norm += magnitude * magnitude;
            }
        }
    }

    return norm;
}

inline float GoldClampDescriptorHistogram(DescriptorHistogram &histogram, float histMax)
{
    float norm = 0.f;

    for (int i = 0; i < kDescWidth; i++)
    {
        for (int j = 0; j < kDescWidth; j++)
        {
            int histIdx = GoldDescriptorHistIndex(i, j, 0);

            for (int bin = 0; bin < kDescHistBins; bin++)
            {
                float magnitude = std::min(histogram[histIdx + bin], histMax);

                norm += magnitude * magnitude;
                histogram[histIdx + bin] = magnitude;
            }
        }
    }

    return norm;
}

inline void GoldWriteDescriptor(DescriptorType &descriptor, DescriptorHistogram &histogram, float norm)
{
    for (int i = 0; i < kDescWidth; i++)
    {
        for (int j = 0; j < kDescWidth; j++)
        {
            int histIdx = GoldDescriptorHistIndex(i, j, 0);

            for (int bin = 0; bin < kDescHistBins; bin++)
            {
                float magnitude = histogram[histIdx + bin];
                int   descIdx   = (i * kDescWidth + j) * kDescHistBins + bin;

                descriptor.data[descIdx] = cuda::SaturateCast<uint8_t>(magnitude * norm);
            }
        }
    }
}

inline void GoldComputeDescriptor(DescriptorType &descriptor, float angle, float featRadius,
                                  const RawPyramidType &srcGaussianPyramid, const long3 &currStrides,
                                  const long3 &currShape, int octave, int layer, int currBatch, int r, int c)
{
    float cos_a = std::cos(static_cast<float>(angle * M_PI / 180.f));
    float sin_a = std::sin(static_cast<float>(angle * M_PI / 180.f));

    float histWidth = kDescOriRadius * featRadius;

    int radius = cuda::round<int>(histWidth * kDescWidthToRadius);

    if (radius > kDescMaxRadius)
    {
        radius    = kDescMaxRadius;
        histWidth = kDescMaxRadius / kDescWidthToRadius;
    }

    cos_a /= histWidth;
    sin_a /= histWidth;

    auto gaussVal = [&srcGaussianPyramid, &currStrides, &octave, &layer, &currBatch](int row, int col)
    {
        return util::ValueAt<WT>(srcGaussianPyramid[octave][layer], currStrides, long3{currBatch, row, col});
    };

    DescriptorHistogram histogram{};

    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            GoldAccumulateDescriptorSample(histogram, gaussVal, angle, cos_a, sin_a, currShape, r, c, i, j);
        }
    }

    float norm    = GoldDescriptorNorm(histogram);
    float histMax = std::sqrt(norm) * kDescHistPeakRatio;

    norm = GoldClampDescriptorHistogram(histogram, histMax);
    norm = 512 / std::max(std::sqrt(norm), 1e-5f);

    GoldWriteDescriptor(descriptor, histogram, norm);
}

inline void GoldComputeHistogram(OrientationHistogram &histogram, float featRadius,
                                 const RawPyramidType &srcGaussianPyramid, const long3 &currStrides,
                                 const long3 &currShape, int octave, int layer, int currBatch, int r, int c)
{
    std::vector<float> tempHistogram(kHistogramBins + 4, 0.f);

    auto radius = static_cast<int>(std::round(featRadius * kOrientationRadius));

    float weightScale = -1.f / (2.f * (featRadius * kOrientationSigma) * (featRadius * kOrientationSigma));

    auto gaussVal = [&srcGaussianPyramid, &currStrides, &octave, &layer, &currBatch](int row, int col)
    {
        return util::ValueAt<WT>(srcGaussianPyramid[octave][layer], currStrides, long3{currBatch, row, col});
    };

    for (int i = -radius; i <= radius; i++)
    {
        if (r + i <= 0 || r + i >= currShape.y - 1)
        {
            continue;
        }

        for (int j = -radius; j <= radius; j++)
        {
            if (c + j <= 0 || c + j >= currShape.z - 1)
            {
                continue;
            }

            float dx = gaussVal(r + i + 0, c + j + 1) - gaussVal(r + i + 0, c + j - 1);
            float dy = gaussVal(r + i - 1, c + j + 0) - gaussVal(r + i + 1, c + j + 0);

            float angle     = std::atan2(dy, dx) * 180.f / static_cast<float>(M_PI);
            float weight    = std::exp2f(static_cast<float>(i * i + j * j) * weightScale);
            float magnitude = std::sqrt(dx * dx + dy * dy);

            int bin = GoldWrapBin(static_cast<int>(std::round(angle * static_cast<float>(kHistogramBins) / 360.f)),
                                  kHistogramBins);

            tempHistogram[2 + bin] += weight * magnitude;
        }
    }

    tempHistogram[0] = tempHistogram[2 + kHistogramBins - 2];
    tempHistogram[1] = tempHistogram[2 + kHistogramBins - 1];

    tempHistogram[2 + kHistogramBins + 0] = tempHistogram[2 + 0];
    tempHistogram[2 + kHistogramBins + 1] = tempHistogram[2 + 1];

    for (int i = 0; i < kHistogramBins; i++)
    {
        histogram[i] = (tempHistogram[2 + i - 2] + tempHistogram[2 + i + 2]) * 1.f / 16
                     + (tempHistogram[2 + i - 1] + tempHistogram[2 + i + 1]) * 4.f / 16
                     + (tempHistogram[2 + i + 0]) * 6.f / 16;
    }
}

inline void GoldAddFeatureOrientation(RawBufferType &featCoords, const long2 &featCoordsStrides,
                                      RawBufferType &featMetadata, const long2 &featMetadataStrides,
                                      RawBufferType &featDescriptors, const long2 &featDescriptorsStrides,
                                      int maxCapacity, RawBufferType &numFeatures, const long1 &numFeaturesStrides,
                                      const RawPyramidType &srcGaussianPyramid, const long3 &currStrides,
                                      const long3 &currShape, int octave, int l, int currBatch, int r, int c,
                                      const float4 &keypoint, float3 metadata, float featRadius, float descAngle)
{
    DescriptorType descriptor;
    GoldComputeDescriptor(descriptor, descAngle, featRadius, srcGaussianPyramid, currStrides, currShape, octave, l,
                          currBatch, r, c);

    int &featIdx = util::ValueAt<int>(numFeatures, numFeaturesStrides, long1{currBatch});

    if (featIdx < maxCapacity)
    {
        util::ValueAt<float4>(featCoords, featCoordsStrides, long2{currBatch, featIdx})                   = keypoint;
        util::ValueAt<float3>(featMetadata, featMetadataStrides, long2{currBatch, featIdx})               = metadata;
        util::ValueAt<DescriptorType>(featDescriptors, featDescriptorsStrides, long2{currBatch, featIdx}) = descriptor;
    }

    featIdx += 1;
}

struct GoldFeatureLocation
{
    int layer;
    int row;
    int col;

    WT                              value{};
    cuda::math::Vector<float, 3>    derivative{};
    cuda::math::Vector<float, 3>    offset{};
    cuda::math::Matrix<float, 3, 3> hessian{};
};

template<typename DogValue>
inline void GoldComputeFeatureSystem(GoldFeatureLocation &feature, const DogValue &dogVal)
{
    constexpr float kImageScale = 1.f / cuda::TypeTraits<VT>::max; // source images data type scale
    constexpr float kDScale1    = kImageScale * .5f;               // first derivative scale
    constexpr float kDScale2    = kImageScale;                     // second derivative scale
    constexpr float kDScaleC    = kImageScale * .25f;              // cross derivative scale

    const int l = feature.layer;
    const int r = feature.row;
    const int c = feature.col;

    auto &dD = feature.derivative;
    auto &H  = feature.hessian;

    // clang-format off
    dD[0] = (dogVal(l + 0, r + 0, c + 1) - dogVal(l + 0, r + 0, c - 1)) * kDScale1;
    dD[1] = (dogVal(l + 0, r + 1, c + 0) - dogVal(l + 0, r - 1, c + 0)) * kDScale1;
    dD[2] = (dogVal(l + 1, r + 0, c + 0) - dogVal(l - 1, r + 0, c + 0)) * kDScale1;

    feature.value = dogVal(l, r, c);

    H[0][0] = (dogVal(l + 0, r + 0, c + 1) + dogVal(l + 0, r + 0, c - 1) - 2 * feature.value) * kDScale2;
    H[1][1] = (dogVal(l + 0, r + 1, c + 0) + dogVal(l + 0, r - 1, c + 0) - 2 * feature.value) * kDScale2;
    H[2][2] = (dogVal(l + 1, r + 0, c + 0) + dogVal(l - 1, r + 0, c + 0) - 2 * feature.value) * kDScale2;

    H[0][1] = H[1][0] = (dogVal(l + 0, r + 1, c + 1) - dogVal(l + 0, r + 1, c - 1) -
                         dogVal(l + 0, r - 1, c + 1) + dogVal(l + 0, r - 1, c - 1)) * kDScaleC;
    H[0][2] = H[2][0] = (dogVal(l + 1, r + 0, c + 1) - dogVal(l + 1, r + 0, c - 1) -
                         dogVal(l - 1, r + 0, c + 1) + dogVal(l - 1, r + 0, c - 1)) * kDScaleC;
    H[1][2] = H[2][1] = (dogVal(l + 1, r + 1, c + 0) - dogVal(l + 1, r - 1, c + 0) -
                         dogVal(l - 1, r + 1, c + 0) + dogVal(l - 1, r - 1, c + 0)) * kDScaleC;
    // clang-format on
}

inline bool GoldOffsetIsSmall(const cuda::math::Vector<float, 3> &offset)
{
    return std::abs(offset[2]) < 0.5f && std::abs(offset[1]) < 0.5f && std::abs(offset[0]) < 0.5f;
}

inline bool GoldOffsetIsInRange(const cuda::math::Vector<float, 3> &offset)
{
    constexpr float kMaxStep = static_cast<float>(std::numeric_limits<int>::max()) / 3.f;
    return std::abs(offset[2]) <= kMaxStep && std::abs(offset[1]) <= kMaxStep && std::abs(offset[0]) <= kMaxStep;
}

inline void GoldApplyOffset(GoldFeatureLocation &feature)
{
    feature.col += static_cast<int>(std::round(feature.offset[0]));
    feature.row += static_cast<int>(std::round(feature.offset[1]));
    feature.layer += static_cast<int>(std::round(feature.offset[2]));
}

inline bool GoldLocationIsInRange(const GoldFeatureLocation &feature, const long3 &currShape, int numOctaveLayers)
{
    return feature.layer >= 1 && feature.layer <= numOctaveLayers && feature.col >= kImageBorder
        && feature.col < currShape.z - kImageBorder && feature.row >= kImageBorder
        && feature.row < currShape.y - kImageBorder;
}

template<typename DogValue>
inline bool GoldLocalizeFeature(GoldFeatureLocation &feature, const DogValue &dogVal, const long3 &currShape,
                                int numOctaveLayers)
{
    for (int i = 0; i < kMaxInterpolationSteps; i++)
    {
        GoldComputeFeatureSystem(feature, dogVal);

        feature.offset = feature.derivative;
        if (!cuda::math::solve_inplace(feature.hessian, feature.offset))
        {
            return false;
        }

        feature.offset = -feature.offset;
        if (GoldOffsetIsSmall(feature.offset))
        {
            return true;
        }

        if (!GoldOffsetIsInRange(feature.offset))
        {
            return false;
        }

        GoldApplyOffset(feature);
        if (!GoldLocationIsInRange(feature, currShape, numOctaveLayers))
        {
            return false;
        }
    }

    return false;
}

inline bool GoldFeaturePassesResponse(float3 &metadata, const GoldFeatureLocation &feature, int numOctaveLayers,
                                      float contrastThreshold, float edgeThreshold)
{
    constexpr float kImageScale = 1.f / cuda::TypeTraits<VT>::max; // source images data type scale

    const auto &dD = feature.derivative;
    const auto &H  = feature.hessian;

    metadata.y = std::abs(feature.value * kImageScale + cuda::math::dot(dD, feature.offset) * .5f);
    if (metadata.y * static_cast<float>(numOctaveLayers) < contrastThreshold)
    {
        return false;
    }

    float trace       = H[0][0] + H[1][1];
    float determinant = H[0][0] * H[1][1] - H[0][1] * H[1][0];
    return determinant > 0
        && trace * trace * edgeThreshold < (edgeThreshold + 1.f) * (edgeThreshold + 1.f) * determinant;
}

inline float4 GoldMakeKeypoint(const GoldFeatureLocation &feature, int currOctave)
{
    float4 keypoint;
    keypoint.x = (static_cast<float>(feature.col) + feature.offset[0]) * std::pow(2.f, static_cast<float>(currOctave));
    keypoint.y = (static_cast<float>(feature.row) + feature.offset[1]) * std::pow(2.f, static_cast<float>(currOctave));
    keypoint.w = static_cast<float>(feature.layer) + feature.offset[2];
    keypoint.z = static_cast<float>(currOctave);
    return keypoint;
}

inline void GoldAddFeatures(RawBufferType &featCoords, const long2 &featCoordsStrides, RawBufferType &featMetadata,
                            const long2 &featMetadataStrides, RawBufferType &featDescriptors,
                            const long2 &featDescriptorsStrides, int maxCapacity, RawBufferType &numFeatures,
                            const long1 &numFeaturesStrides, const RawPyramidType &srcGaussianPyramid,
                            const RawPyramidType &srcDoGPyramid, const long3 &currStrides, const long3 &currShape,
                            int octave, int firstOctave, int numOctaveLayers, float contrastThreshold,
                            float edgeThreshold, float initSigma, int l, int currBatch, int r, int c)
{
    auto dogVal = [&srcDoGPyramid, &currStrides, &octave, &currBatch](int layer, int row, int col)
    {
        return util::ValueAt<WT>(srcDoGPyramid[octave][layer], currStrides, long3{currBatch, row, col});
    };

    GoldFeatureLocation feature{l, r, c};
    if (!GoldLocalizeFeature(feature, dogVal, currShape, numOctaveLayers))
    {
        return;
    }

    float3 metadata;

    if (!GoldFeaturePassesResponse(metadata, feature, numOctaveLayers, contrastThreshold, edgeThreshold))
    {
        return;
    }

    int currOctave = octave + firstOctave;

    float4 keypoint   = GoldMakeKeypoint(feature, currOctave);
    float  featRadius = initSigma * std::pow(2.f, keypoint.w / static_cast<float>(numOctaveLayers));

    OrientationHistogram hist{};

    GoldComputeHistogram(hist, featRadius, srcGaussianPyramid, currStrides, currShape, octave, feature.layer, currBatch,
                         feature.row, feature.col);

    metadata.z = featRadius * 2.f * std::pow(2.f, static_cast<float>(currOctave));

    float histPeak = hist[0];

    for (int i = 1; i < kHistogramBins; ++i)
    {
        histPeak = std::max(histPeak, hist[i]);
    }

    histPeak *= kHistogramPeakRatio;

    for (int i = 0; i < kHistogramBins; ++i)
    {
        int prologue = i > 0 ? i - 1 : kHistogramBins - 1;
        int epilogue = i < kHistogramBins - 1 ? i + 1 : 0;

        if (hist[i] <= hist[prologue] || hist[i] <= hist[epilogue] || hist[i] < histPeak)
        {
            continue;
        }

        auto bin = static_cast<float>(i)
                 + .5f * (hist[prologue] - hist[epilogue]) / (hist[prologue] - 2.f * hist[i] + hist[epilogue]);

        if (bin < 0)
        {
            bin += kHistogramBins;
        }
        else if (bin >= kHistogramBins)
        {
            bin -= kHistogramBins;
        }

        metadata.x = 360.f - (360.f / kHistogramBins) * bin;

        if (cuda::abs(metadata.x - 360.f) < 1e-5)
            metadata.x = 0.f;

        ASSERT_TRUE(metadata.x >= 0.f && metadata.x <= 360.f);

        float descAngle = 360.f - metadata.x;

        if (cuda::abs(descAngle - 360.f) < 1e-5)
            descAngle = 0.f;

        ASSERT_TRUE(descAngle >= 0.f && descAngle <= 360.f);

        GoldAddFeatureOrientation(featCoords, featCoordsStrides, featMetadata, featMetadataStrides, featDescriptors,
                                  featDescriptorsStrides, maxCapacity, numFeatures, numFeaturesStrides,
                                  srcGaussianPyramid, currStrides, currShape, octave, feature.layer, currBatch,
                                  feature.row, feature.col, keypoint, metadata, featRadius, descAngle);
    }
}

template<typename DogValue>
inline bool GoldIsDoGExtremum(const DogValue &dogVal, int octave, int layer, long batch, long row, long col, WT val)
{
    for (int i = 0; i < 27; ++i)
    {
        int dl = i / 9 - 1;
        int dr = (i / 3) % 3 - 1;
        int dc = i % 3 - 1;

        if (dl == 0 && dr == 0 && dc == 0)
        {
            continue;
        }

        WT neighbor = dogVal(octave, layer + dl, batch, row + dr, col + dc);
        if ((val > 0 && val < neighbor) || (val < 0 && val > neighbor))
        {
            return false;
        }
    }

    return true;
}

inline void GoldFindExtrema(RawBufferType &featCoords, const long2 &featCoordsStrides, const long2 &featCoordsShape,
                            RawBufferType &featMetadata, const long2 &featMetadataStrides,
                            RawBufferType &featDescriptors, const long2 &featDescriptorsStrides,
                            const long2 &featMetadataShape, RawBufferType &numFeatures, const long1 &numFeaturesStrides,
                            const long1 &numFeaturesShape, const RawPyramidType &srcGaussianPyramid,
                            const RawPyramidType &srcDoGPyramid, const long3 &baseStrides, const long3 &baseShape,
                            int firstOctave, int numOctaves, int numOctaveLayers, float contrastThreshold,
                            float edgeThreshold, float initSigma)
{
    auto threshold
        = static_cast<int>(std::floor(.5f * contrastThreshold / static_cast<float>(numOctaveLayers) * 255.f));

    long3 currShape   = baseShape;
    long3 currStrides = baseStrides;

    auto dogVal = [&currStrides, &srcDoGPyramid](int octave, int layer, long batch, long row, long col)
    {
        return util::ValueAt<WT>(srcDoGPyramid[octave][layer], currStrides, long3{batch, row, col});
    };

    auto maxCapacity = static_cast<int>(featCoordsShape.y);

    ASSERT_TRUE(featCoordsShape.x == currShape.x && featMetadataShape.x == currShape.x
                && featCoordsShape.y == featMetadataShape.y && numFeaturesShape.x == currShape.x);

    auto addFeatureIfExtremum
        = [&featCoords, &featCoordsStrides, &featDescriptors, &featDescriptorsStrides, &featMetadata,
           &featMetadataStrides, &numFeatures, &numFeaturesStrides, &srcDoGPyramid, &srcGaussianPyramid,
           contrastThreshold, edgeThreshold, firstOctave, initSigma, maxCapacity, numOctaveLayers, threshold,
           &currShape, &currStrides, &dogVal](int o, int l, long b, int r, int c)
    {
        if (WT val = dogVal(o, l, b, r, c);
            std::abs(val) <= static_cast<WT>(threshold) || !GoldIsDoGExtremum(dogVal, o, l, b, r, c, val))
        {
            return;
        }

        GoldAddFeatures(featCoords, featCoordsStrides, featMetadata, featMetadataStrides, featDescriptors,
                        featDescriptorsStrides, maxCapacity, numFeatures, numFeaturesStrides, srcGaussianPyramid,
                        srcDoGPyramid, currStrides, currShape, o, firstOctave, numOctaveLayers, contrastThreshold,
                        edgeThreshold, initSigma, l, static_cast<int>(b), r, c);
    };

    for (int o = 0; o < numOctaves; o++)
    {
        for (int l = 1; l <= numOctaveLayers; l++)
        {
            const long width        = currShape.z - 2 * kImageBorder;
            const long height       = currShape.y - 2 * kImageBorder;
            const long numPositions = currShape.x * height * width;

            for (long p = 0; p < numPositions; ++p)
            {
                long rest = p;
                int  c    = kImageBorder + static_cast<int>(rest % width);
                rest /= width;
                int r = kImageBorder + static_cast<int>(rest % height);
                rest /= height;

                addFeatureIfExtremum(o, l, rest, r, c);
            } // for each batch image, row, and column
        }     // for each layer

        currShape.y /= 2;
        currShape.z /= 2;
        currStrides = GoldStrides(currShape);
    } // for each octave
}

// Struct to hold SIFT results
struct SIFTResults
{
    using TupleType = std::tuple<float4, float3, DescriptorType>; // float4 coordinates, float3 metadata, descriptor

    std::vector<std::vector<TupleType>> testFeatures;
    std::vector<std::vector<TupleType>> goldFeatures;

    std::vector<int> testNumFeatures;
    std::vector<int> goldNumFeatures;
};

// Gold (CPU reference) computation of SIFT
inline void GoldSIFT(SIFTResults &outResults, const nvcv::Tensor &featCoords, const nvcv::Tensor &featMetadata,
                     const nvcv::Tensor &featDescriptors, const nvcv::Tensor &numFeatures, float initSigma,
                     bool expandInput, int numOctaveLayers, long capacity, float contrastThreshold, float edgeThreshold,
                     const RawBufferType &srcVec, const long3 &srcStrides, const long3 &srcShape)
{
    auto featCoordsData      = featCoords.exportData<nvcv::TensorDataStridedCuda>();
    auto featMetadataData    = featMetadata.exportData<nvcv::TensorDataStridedCuda>();
    auto featDescriptorsData = featDescriptors.exportData<nvcv::TensorDataStridedCuda>();
    auto numFeaturesData     = numFeatures.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(featCoordsData && featMetadataData && featDescriptorsData && numFeaturesData);

    ASSERT_TRUE(featCoordsData->rank() == 2 && featMetadataData->rank() == 2 && featDescriptorsData->rank() == 3
                && numFeaturesData->rank() == 1);

    long2 featCoordsShape   = {featCoordsData->shape(0), featCoordsData->shape(1)};
    long2 featMetadataShape = {featMetadataData->shape(0), featMetadataData->shape(1)};
    long3 featDescriptorsShape
        = {featDescriptorsData->shape(0), featDescriptorsData->shape(1), featDescriptorsData->shape(2)};
    long1 numFeaturesShape = {numFeaturesData->shape(0)};

    ASSERT_TRUE((featCoordsShape == long2{srcShape.x, capacity} && featMetadataShape == long2{srcShape.x, capacity}
                 && featDescriptorsShape == long3{srcShape.x, capacity, 128} && numFeaturesShape == long1{srcShape.x}));

    long2 featCoordsStrides   = {featCoordsData->stride(0), featCoordsData->stride(1)};
    long2 featMetadataStrides = {featMetadataData->stride(0), featMetadataData->stride(1)};
    long3 featDescriptorsStrides3
        = {featDescriptorsData->stride(0), featDescriptorsData->stride(1), featDescriptorsData->stride(2)};
    long2 featDescriptorsStrides = {featDescriptorsStrides3.x, featDescriptorsStrides3.y};
    long1 numFeaturesStrides     = {numFeaturesData->stride(0)};

    ASSERT_TRUE(featCoordsStrides.y == sizeof(float4) && featMetadataStrides.y == sizeof(float3)
                && featDescriptorsStrides3.z == sizeof(uint8_t) && featDescriptorsStrides.y == 128 * sizeof(uint8_t)
                && numFeaturesStrides.x == sizeof(int));

    long featCoordsBufSize      = featCoordsShape.x * featCoordsStrides.x;
    long featMetadataBufSize    = featMetadataShape.x * featMetadataStrides.x;
    long featDescriptorsBufSize = featDescriptorsShape.x * featDescriptorsStrides.x;
    long numFeaturesBufSize     = numFeaturesShape.x * numFeaturesStrides.x;

    RawBufferType testFeatCoordsBuf(featCoordsBufSize);
    RawBufferType testFeatMetadataBuf(featMetadataBufSize);
    RawBufferType testFeatDescriptorsBuf(featDescriptorsBufSize);
    RawBufferType testNumFeaturesBuf(numFeaturesBufSize);

#define NVCV_TEST_CUDA_COPY(FROM, TO, SIZE) \
    ASSERT_EQ(cudaSuccess, cudaMemcpy(TO.data(), FROM->basePtr(), SIZE, cudaMemcpyDeviceToHost))

    NVCV_TEST_CUDA_COPY(featCoordsData, testFeatCoordsBuf, featCoordsBufSize);
    NVCV_TEST_CUDA_COPY(featMetadataData, testFeatMetadataBuf, featMetadataBufSize);
    NVCV_TEST_CUDA_COPY(featDescriptorsData, testFeatDescriptorsBuf, featDescriptorsBufSize);
    NVCV_TEST_CUDA_COPY(numFeaturesData, testNumFeaturesBuf, numFeaturesBufSize);

#undef NVCV_TEST_CUDA_COPY

    RawPyramidType pyrGaussian;
    RawPyramidType pyrDoG;
    long3          baseShape;
    long3          baseStrides;

    int numOctaves;

    GoldGeneratePyramids(pyrGaussian, pyrDoG, baseStrides, baseShape, numOctaves, srcVec, srcStrides, srcShape,
                         expandInput, initSigma, numOctaveLayers);

    RawBufferType goldFeatCoordsBuf(featCoordsBufSize);
    RawBufferType goldFeatMetadataBuf(featMetadataBufSize);
    RawBufferType goldFeatDescriptorsBuf(featDescriptorsBufSize);
    RawBufferType goldNumFeaturesBuf(numFeaturesBufSize);

    int firstOctave = expandInput ? -1 : 0;

    GoldFindExtrema(goldFeatCoordsBuf, featCoordsStrides, featCoordsShape, goldFeatMetadataBuf, featMetadataStrides,
                    goldFeatDescriptorsBuf, featDescriptorsStrides, featMetadataShape, goldNumFeaturesBuf,
                    numFeaturesStrides, numFeaturesShape, pyrGaussian, pyrDoG, baseStrides, baseShape, firstOctave,
                    numOctaves, numOctaveLayers, contrastThreshold, edgeThreshold, initSigma);

    outResults.testFeatures.resize(srcShape.x);
    outResults.goldFeatures.resize(srcShape.x);

    outResults.testNumFeatures.resize(srcShape.x);
    outResults.goldNumFeatures.resize(srcShape.x);

    auto featureLower = [](const SIFTResults::TupleType &f1, const SIFTResults::TupleType &f2)
    {
        return std::make_tuple(std::get<0>(f1).z, std::get<0>(f1).w, std::get<0>(f1).y, std::get<0>(f1).x,
                               std::get<1>(f1).z, std::get<1>(f1).y, std::get<1>(f1).x)
             < std::make_tuple(std::get<0>(f2).z, std::get<0>(f2).w, std::get<0>(f2).y, std::get<0>(f2).x,
                               std::get<1>(f2).z, std::get<1>(f2).y, std::get<1>(f2).x);
    };

    for (int x = 0; x < srcShape.x; x++)
    {
        outResults.testNumFeatures[x] = util::ValueAt<int>(testNumFeaturesBuf, numFeaturesStrides, long1{x});
        outResults.goldNumFeatures[x] = util::ValueAt<int>(goldNumFeaturesBuf, numFeaturesStrides, long1{x});

        int testMaxFeatures = std::min((int)capacity, outResults.testNumFeatures[x]);
        int goldMaxFeatures = std::min((int)capacity, outResults.goldNumFeatures[x]);

        outResults.testFeatures[x].resize(testMaxFeatures);
        outResults.goldFeatures[x].resize(goldMaxFeatures);

        // To proper compare gold against test: rounding to 3 binary places due to a lot of FLOPs during SIFT

        for (int y = 0; y < testMaxFeatures; y++)
        {
            outResults.testFeatures[x][y]
                = {cuda::round(8 * util::ValueAt<float4>(testFeatCoordsBuf, featCoordsStrides, long2{x, y})) / 8,
                   cuda::round(8 * util::ValueAt<float3>(testFeatMetadataBuf, featMetadataStrides, long2{x, y})) / 8,
                   util::ValueAt<DescriptorType>(testFeatDescriptorsBuf, featDescriptorsStrides, long2{x, y})};
        }
        for (int y = 0; y < goldMaxFeatures; y++)
        {
            outResults.goldFeatures[x][y]
                = {cuda::round(8 * util::ValueAt<float4>(goldFeatCoordsBuf, featCoordsStrides, long2{x, y})) / 8,
                   cuda::round(8 * util::ValueAt<float3>(goldFeatMetadataBuf, featMetadataStrides, long2{x, y})) / 8,
                   util::ValueAt<DescriptorType>(goldFeatDescriptorsBuf, featDescriptorsStrides, long2{x, y})};
        }

        // Need to sort both CPU and CUDA results due to extrema interpolation in add features

        std::ranges::sort(outResults.testFeatures[x], featureLower);
        std::ranges::sort(outResults.goldFeatures[x], featureLower);
    }
}

// ----------------------------- Start tests -----------------------------------

// clang-format off

#define NVCV_SHAPE(w, h, n) (int3{w, h, n})

#define NVCV_TEST_ROW(InShape, MaxFeatures, NumOctaveLayers, ContrastTh, EdgeTh, InitSigma, ExpandInput)        \
    type::Types<type::Value<InShape>, type::Value<MaxFeatures>, type::Value<NumOctaveLayers>,                   \
                type::Value<ContrastTh>, type::Value<EdgeTh>, type::Value<InitSigma>, type::Value<ExpandInput>>

NVCV_TYPED_TEST_SUITE(OpSIFT, type::Types<
    NVCV_TEST_ROW(NVCV_SHAPE(23, 17, 3), 55, 2, 0.01f, 20.f, .5f, false),
    NVCV_TEST_ROW(NVCV_SHAPE(32, 32, 1), 5, 8, .42f, 8.f, 1.f, true),
    NVCV_TEST_ROW(NVCV_SHAPE(43, 53, 2), 444, 5, 0.21f, 3.f, .75f, false),
    NVCV_TEST_ROW(NVCV_SHAPE(56, 22, 5), 88, 1, 0.009f, 12.f, .6f, true),
    NVCV_TEST_ROW(NVCV_SHAPE(96, 21, 4), 600, 4, 0.05f, 9.f, .8f, false),
    NVCV_TEST_ROW(NVCV_SHAPE(16, 16, 1), 66, 3, 0.02f, 13.f, .7f, true),
    NVCV_TEST_ROW(NVCV_SHAPE(13, 20, 2), 2222, 6, 0.13f, 4.f, .55f, false),
    NVCV_TEST_ROW(NVCV_SHAPE(39, 38, 3), 44, 7, 0.04f, 10.f, 1.6f, true),
    NVCV_TEST_ROW(NVCV_SHAPE(24, 42, 2), 1111, 9, 0.14f, 4.f, 1.1f, false),
    NVCV_TEST_ROW(NVCV_SHAPE(77, 65, 2), 33, 13, 0.33f, 23.5f, .9f, false)
>);

// clang-format on

static void RunSIFTCorrectOutput(int3 inShape, long capacity, int numOctaveLayers, float contrastThreshold,
                                 float edgeThreshold, float initSigma, bool expandInput, std::string_view layout)
{
    NVCVSIFTFlagType flags = expandInput ? NVCV_SIFT_USE_EXPANDED_INPUT : NVCV_SIFT_USE_ORIGINAL_INPUT;

    // Increasing inShape and numOctaveLayers to test bigger maxShape and maxOctaveLayers
    int3 maxShape        = (inShape + 3) * (expandInput ? 2 : 1);
    int  maxOctaveLayers = numOctaveLayers + 1;

    nvcv::Tensor src = CreateSIFTInputTensor(inShape, layout);

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    long3 srcShape{srcAccess->numSamples(), srcAccess->numRows(), srcAccess->numCols()};
    long3 srcStrides{srcAccess->sampleStride(), srcAccess->rowStride(), srcAccess->colStride()};

    // While inShape is WHN, srcShape is NHW to match srcStrides
    ASSERT_TRUE(inShape.z == srcShape.x && inShape.y == srcShape.y && inShape.x == srcShape.z);

    // CHW/HWC tensors have no explicit N dimension, so compute the
    // per-sample stride from height and row stride before using the accessor.
    srcStrides.x = (srcData->rank() == 3) ? srcShape.y * srcStrides.y : srcStrides.x;

    long srcBufSize = srcStrides.x * srcShape.x;

    RawBufferType srcVec(srcBufSize);

    // clang-format off

    std::uniform_int_distribution<VT> rg(0, 255);

    for (long x = 0; x < srcShape.x; ++x)
        for (long y = 0; y < srcShape.y; ++y)
            for (long z = 0; z < srcShape.z; ++z)
                util::ValueAt<VT>(srcVec, srcStrides, long3{x, y, z}) = rg(Rng());

    nvcv::Tensor featCoords({{srcShape.x, capacity}, "NM"}, nvcv::TYPE_4F32);
    nvcv::Tensor featMetadata({{srcShape.x, capacity}, "NM"}, nvcv::TYPE_3F32);
    nvcv::Tensor featDescriptors({{srcShape.x, capacity, 128}, "NMD"}, nvcv::TYPE_U8);
    nvcv::Tensor numFeatures({{srcShape.x}, "N"}, nvcv::TYPE_S32);

    // clang-format on

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), srcBufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::SIFT op(maxShape, maxOctaveLayers);

    EXPECT_NO_THROW(op(stream, src, featCoords, featMetadata, featDescriptors, numFeatures, numOctaveLayers,
                       contrastThreshold, edgeThreshold, initSigma, flags));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    SIFTResults results;

    GoldSIFT(results, featCoords, featMetadata, featDescriptors, numFeatures, initSigma, expandInput, numOctaveLayers,
             capacity, contrastThreshold, edgeThreshold, srcVec, srcStrides, srcShape);

    EXPECT_EQ(results.testNumFeatures, results.goldNumFeatures);
    EXPECT_EQ(results.testFeatures, results.goldFeatures);
}

TYPED_TEST(OpSIFT, correct_output)
{
    int3  inShape           = type::GetValue<TypeParam, 0>;
    long  capacity          = type::GetValue<TypeParam, 1>;
    int   numOctaveLayers   = type::GetValue<TypeParam, 2>;
    float contrastThreshold = type::GetValue<TypeParam, 3>;
    float edgeThreshold     = type::GetValue<TypeParam, 4>;
    float initSigma         = type::GetValue<TypeParam, 5>;
    bool  expandInput       = type::GetValue<TypeParam, 6>;

    RunSIFTCorrectOutput(inShape, capacity, numOctaveLayers, contrastThreshold, edgeThreshold, initSigma, expandInput,
                         "NHWC");
}

TEST(OpSIFT, planar_correct_output)
{
    RunSIFTCorrectOutput(int3{32, 32, 2}, 64, 3, 0.02f, 12.f, 1.0f, false, "NCHW");
    RunSIFTCorrectOutput(int3{32, 32, 1}, 64, 3, 0.02f, 12.f, 1.0f, true, "CHW");
}

static void RunSIFTPlanarParity(int3 inShape, std::string_view interleavedLayout, std::string_view planarLayout)
{
    constexpr long  capacity          = 256;
    constexpr int   numOctaveLayers   = 3;
    constexpr float contrastThreshold = 0.02f;
    constexpr float edgeThreshold     = 12.f;
    constexpr float initSigma         = 1.f;

    nvcv::Tensor srcInterleaved = CreateSIFTInputTensor(inShape, interleavedLayout);
    nvcv::Tensor srcPlanar      = CreateSIFTInputTensor(inShape, planarLayout);

    auto srcInterleavedData = srcInterleaved.exportData<nvcv::TensorDataStridedCuda>();
    auto srcPlanarData      = srcPlanar.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcInterleavedData && srcPlanarData);

    auto srcInterleavedAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcInterleavedData);
    auto srcPlanarAccess      = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcPlanarData);
    ASSERT_TRUE(srcInterleavedAccess && srcPlanarAccess);

    for (int sample = 0; sample < inShape.z; ++sample)
    {
        std::vector<uint8_t> input(inShape.x * inShape.y);
        test::planar::FillDeterministicValues(input, static_cast<size_t>(sample) * 101 + 1, nvcv::TYPE_U8);
        test::planar::UploadInterleavedSample(*srcInterleavedAccess, sample, input, inShape.x, inShape.y, inShape.x);
        test::planar::UploadPlanarSample(*srcPlanarAccess, sample, input, inShape.x, inShape.y, 1, sizeof(uint8_t));
    }

    nvcv::Tensor featCoordsInterleaved(
        {
            {inShape.z, capacity},
            "NM"
    },
        nvcv::TYPE_4F32);
    nvcv::Tensor featMetadataInterleaved(
        {
            {inShape.z, capacity},
            "NM"
    },
        nvcv::TYPE_3F32);
    nvcv::Tensor featDescriptorsInterleaved(
        {
            {inShape.z, capacity, 128},
            "NMD"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor numFeaturesInterleaved({{inShape.z}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor featCoordsPlanar(
        {
            {inShape.z, capacity},
            "NM"
    },
        nvcv::TYPE_4F32);
    nvcv::Tensor featMetadataPlanar(
        {
            {inShape.z, capacity},
            "NM"
    },
        nvcv::TYPE_3F32);
    nvcv::Tensor featDescriptorsPlanar(
        {
            {inShape.z, capacity, 128},
            "NMD"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor numFeaturesPlanar({{inShape.z}, "N"}, nvcv::TYPE_S32);

    std::array<nvcv::Tensor *, 8> outputs{
        &featCoordsInterleaved, &featMetadataInterleaved, &featDescriptorsInterleaved, &numFeaturesInterleaved,
        &featCoordsPlanar,      &featMetadataPlanar,      &featDescriptorsPlanar,      &numFeaturesPlanar,
    };

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    for (const nvcv::Tensor *output : outputs)
    {
        auto data = output->exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(data);
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(data->basePtr(), 0, data->shape(0) * data->stride(0), stream));
    }

    cvcuda::SIFT op(inShape + 3, numOctaveLayers + 1);
    EXPECT_NO_THROW(op(stream, srcInterleaved, featCoordsInterleaved, featMetadataInterleaved,
                       featDescriptorsInterleaved, numFeaturesInterleaved, numOctaveLayers, contrastThreshold,
                       edgeThreshold, initSigma, NVCV_SIFT_USE_ORIGINAL_INPUT));
    EXPECT_NO_THROW(op(stream, srcPlanar, featCoordsPlanar, featMetadataPlanar, featDescriptorsPlanar,
                       numFeaturesPlanar, numOctaveLayers, contrastThreshold, edgeThreshold, initSigma,
                       NVCV_SIFT_USE_ORIGINAL_INPUT));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto expectExactTensor = [](const nvcv::Tensor &interleaved, const nvcv::Tensor &planar)
    {
        auto interleavedData = interleaved.exportData<nvcv::TensorDataStridedCuda>();
        auto planarData      = planar.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(interleavedData && planarData);

        const long numBytes = interleavedData->shape(0) * interleavedData->stride(0);
        ASSERT_EQ(numBytes, planarData->shape(0) * planarData->stride(0));

        RawBufferType interleavedBuffer(numBytes);
        RawBufferType planarBuffer(numBytes);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(interleavedBuffer.data(), interleavedData->basePtr(), numBytes, cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(planarBuffer.data(), planarData->basePtr(), numBytes, cudaMemcpyDeviceToHost));
        EXPECT_EQ(interleavedBuffer, planarBuffer);
    };

    expectExactTensor(numFeaturesInterleaved, numFeaturesPlanar);
    expectExactTensor(featCoordsInterleaved, featCoordsPlanar);
    expectExactTensor(featMetadataInterleaved, featMetadataPlanar);
    expectExactTensor(featDescriptorsInterleaved, featDescriptorsPlanar);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpSIFTPlanar, tensor_nchw_matches_interleaved)
{
    RunSIFTPlanarParity(int3{67, 53, 2}, "NHWC", "NCHW");
}

TEST(OpSIFTPlanar, tensor_chw_matches_interleaved)
{
    RunSIFTPlanarParity(int3{43, 37, 1}, "HWC", "CHW");
}

TEST(OpSIFT, no_linear_system_solution)
{
    int3 inShape = {64, 64, 1};

    nvcv::Tensor src = nvcv::util::CreateTensor(inShape.z, inShape.x, inShape.y, kInFormat);

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData);

    std::vector<uint8_t> flatImage(64 * 64, 128);
    flatImage[32 * 64 + 32] = 130;
    flatImage[32 * 64 + 31] = 126;
    flatImage[31 * 64 + 32] = 126;

    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(srcData->basePtr(), flatImage.data(), 64 * 64 * sizeof(uint8_t), cudaMemcpyHostToDevice));

    nvcv::Tensor featCoords(
        {
            {1, 100},
            "NM"
    },
        nvcv::TYPE_4F32);
    nvcv::Tensor featMetadata(
        {
            {1, 100},
            "NM"
    },
        nvcv::TYPE_3F32);
    nvcv::Tensor featDescriptors(
        {
            {1, 100, 128},
            "NMD"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor numFeatures({{1}, "N"}, nvcv::TYPE_S32);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::SIFT op(inShape, 3);

    EXPECT_NO_THROW(op(stream, src, featCoords, featMetadata, featDescriptors, numFeatures, 3, 0.001f, 5.0f, 0.5f,
                       NVCV_SIFT_USE_ORIGINAL_INPUT));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpSIFT_Negative, test::ValueList<nvcv::ImageFormat, int, int, int, float, float, float, int, int, int, nvcv::DataType, int, int, nvcv::DataType, int, int, int, nvcv::DataType, int, nvcv::DataType>{
    // Negative cases vary image format, shape, SIFT thresholds, output tensors, and feature counts.
    // invalid input
    {  nvcv::FMT_RGB8p , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_F32   , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_RGB8  , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_U8    , 1 , 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    // invalid parameters
    {  nvcv::FMT_U8    , 32, 32, 8   , -0.5f    , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , -0.01f           , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , -20.f        , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 0              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    // invalid featCoords
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_F32 , 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 7, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    // invalid featMetadata
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_F32 , 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 7, 55, nvcv::TYPE_3F32, 7, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 56, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    // invalid featDescriptorsData
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_S8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 127, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 7, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
    {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 56, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_S32},
   // invalid numFeaturesData
   {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 8, nvcv::TYPE_F32},
   {  nvcv::FMT_U8    , 32, 32, 8   , 0.5f     , 0.01f            , 20.f         , 2              , 8, 55, nvcv::TYPE_4F32, 8, 55, nvcv::TYPE_3F32, 8, 55, 128, nvcv::TYPE_U8, 7, nvcv::TYPE_S32}
});

// clang-format on

TEST(OpSIFT_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaSIFTCreate(nullptr, int3{32, 32, 8}, 2), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST_P(OpSIFT_Negative, invalid_parameters)
{
    nvcv::ImageFormat inFmt = GetParamValue<0>();
    int3              inShape{GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>()};
    float             initSigma         = GetParamValue<4>();
    float             contrastThreshold = GetParamValue<5>();
    float             edgeThreshold     = GetParamValue<6>();
    int               numOctaveLayers   = GetParamValue<7>();
    int2              featCoordsShape{GetParamValue<8>(), GetParamValue<9>()};
    nvcv::DataType    featCoordsDataType = GetParamValue<10>();
    int2              featMetadataShape{GetParamValue<11>(), GetParamValue<12>()};
    nvcv::DataType    featMetadataDataType = GetParamValue<13>();
    int3              featDescriptorsShape{GetParamValue<14>(), GetParamValue<15>(), GetParamValue<16>()};
    nvcv::DataType    featDescriptorsDataType = GetParamValue<17>();
    int               numFeaturesShape        = GetParamValue<18>();
    nvcv::DataType    numFeaturesDataType     = GetParamValue<19>();

    NVCVSIFTFlagType flags = NVCV_SIFT_USE_ORIGINAL_INPUT;

    int3 maxShape        = (inShape + 3);
    int  maxOctaveLayers = numOctaveLayers + 1;

    nvcv::Tensor src = nvcv::util::CreateTensor(inShape.z, inShape.x, inShape.y, inFmt);

    nvcv::Tensor featCoords(
        {
            {featCoordsShape.x, featCoordsShape.y},
            "NM"
    },
        featCoordsDataType);
    nvcv::Tensor featMetadata(
        {
            {featMetadataShape.x, featMetadataShape.y},
            "NM"
    },
        featMetadataDataType);
    nvcv::Tensor featDescriptors(
        {
            {featDescriptorsShape.x, featDescriptorsShape.y, featDescriptorsShape.z},
            "NMD"
    },
        featDescriptorsDataType);
    nvcv::Tensor numFeatures({{numFeaturesShape}, "N"}, numFeaturesDataType);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::SIFT op(maxShape, maxOctaveLayers);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&op, &stream, &src, &featCoords, &featMetadata, &featDescriptors, &numFeatures, &numOctaveLayers,
                   &contrastThreshold, &edgeThreshold, &initSigma, &flags]
                  {
                      op(stream, src, featCoords, featMetadata, featDescriptors, numFeatures, numOctaveLayers,
                         contrastThreshold, edgeThreshold, initSigma, flags);
                  }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpSIFT_Negative, create_invalid_shape)
{
    NVCVOperatorHandle handle;
    EXPECT_EQ(cvcudaSIFTCreate(&handle, int3{1, 8, 8}, 2), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaSIFTCreate(&handle, int3{8, 1, 8}, 2), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaSIFTCreate(&handle, int3{8, 8, 0}, 2), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaSIFTCreate(&handle, int3{8, 8, 65536}, 2), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaSIFTCreate(&handle, int3{8, 8, 8}, 0), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(cvcudaSIFTCreate(&handle, int3{8, 8, 8}, 17), NVCV_ERROR_INVALID_ARGUMENT);
}

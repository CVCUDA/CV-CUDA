/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/InterpUtils.hpp>
#include <common/TypedTests.hpp>
#include <cvcuda/OpSIFT.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/MathOps.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/cuda/math/LinAlg.hpp>
#include <util/TensorDataUtils.hpp>

#include <array>
#include <bitset>
#include <random>
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
            hammingDist += std::bitset<8>(data[i] ^ other.data[i]).count();
        }
        return (hammingDist < static_cast<int>((data.size() * sizeof(uint8_t) * 8) / 100));
    }

    friend std::ostream &operator<<(std::ostream &out, const DescriptorType &desc)
    {
        std::ios_base::fmtflags f{out.flags()};
        out << "0x";
        for (int i = 0; i < (int)desc.data.size(); i++)
        {
            out << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(desc.data[i]);
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
constexpr float kDescWidthToRadius     = M_SQRT2 * (kDescWidth + 1) * .5f;
constexpr int   kDescMaxRadius         = 51;
constexpr float kDescWeightScale       = -1.f / (kDescWidth * kDescWidth * .5f);
constexpr float kDescHistPeakRatio     = .2f;

static std::default_random_engine g_rng(0); // seed 0 to fix pseudo-randomness

// --------------------- Gold (reference) definitions --------------------------

inline int GoldNumberOfOctaves(int width, int height)
{
    return std::floor(std::log2(std::min(width, height))) - 2;
}

inline nvcv::Size2D GoldKernelSize(float sigma)
{
    int ksize = std::min((int)std::round(sigma * 8 + 1) | 1, kMaxKernelSize.x);

    return nvcv::Size2D{ksize, ksize};
}

inline void GoldGaussianSigmas(std::vector<float> &layerSigmas, float initSigma, int numOctaveLayers)
{
    layerSigmas[0] = initSigma;

    float k = std::pow(2.0, 1.0 / numOctaveLayers);
    float prevSigma, totalSigma;

    for (int i = 1; i < numOctaveLayers + 3; i++)
    {
        prevSigma  = std::pow(k, i - 1) * initSigma;
        totalSigma = k * prevSigma;

        layerSigmas[i] = std::sqrt(totalSigma * totalSigma - prevSigma * prevSigma);
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
                float2 srcCoord;

                if (srcScale.x >= 1.f)
                {
                    srcCoord.x = z * srcScale.x;
                    srcCoord.y = y * srcScale.y;

                    util::ValueAt<T>(dstVec, dstStrides, long3{x, y, z})
                        = test::GoldInterp<NVCV_INTERP_NEAREST, NVCV_BORDER_REPLICATE>(
                            srcVec, srcStrides, int2{(int)srcShape.z, (int)srcShape.y}, U{}, kScale, srcCoord, x);
                }
                else
                {
                    srcCoord.x = (z + .5f) * srcScale.x - .5f;
                    srcCoord.y = (y + .5f) * srcScale.y - .5f;

                    util::ValueAt<T>(dstVec, dstStrides, long3{x, y, z})
                        = test::GoldInterp<NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE>(
                            srcVec, srcStrides, int2{(int)srcShape.z, (int)srcShape.y}, U{}, kScale, srcCoord, x);
                }
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
    return long3(shape.y * shape.z * sizeof(WT), shape.z * sizeof(WT), sizeof(WT));
}

inline int3 ConvolveShape(const long3 &shape)
{
    return int3(shape.z, shape.y, shape.x); // test::Convolve expects shape as WHN int3 instead of NHW long3
}

inline void GoldGeneratePyramids(RawPyramidType &dstGaussianPyramid, RawPyramidType &dstDoGPyramid, long3 &baseStrides,
                                 long3 &baseShape, int &numOctaves, const RawBufferType &srcVec,
                                 const long3 &srcStrides, const long3 &srcShape, bool expandInput, float initSigma,
                                 int numOctaveLayers)
{
    baseShape = expandInput ? long3{srcShape.x, srcShape.y * 2, srcShape.z * 2} : srcShape;

    numOctaves = GoldNumberOfOctaves(baseShape.z, baseShape.y);

    baseStrides = GoldStrides(baseShape);

    RawBufferType srcBase(baseShape.x * baseStrides.x);

    GoldCopy<WT, VT>(srcBase, baseStrides, baseShape, srcVec, srcStrides, srcShape);

    std::vector<float> layerSigmas(numOctaveLayers + 3);

    GoldGaussianSigmas(layerSigmas, initSigma, numOctaveLayers);

    RawBufferType srcGaussBase(baseShape.x * baseStrides.x);

    int   srcScale = expandInput ? 4 : 1;
    float sigma    = layerSigmas[0];

    sigma = std::sqrt(std::max(sigma * sigma - kPrevSigma * kPrevSigma * srcScale, kMinSigma));

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

    float histogram[(kDescWidth + 2) * (kDescWidth + 2) * (kDescHistBins + 2)] = {0.f};

    float magnitude;

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

            float r_rot = j * sin_a + i * cos_a;
            float c_rot = j * cos_a - i * sin_a;

            float weight = r_rot * r_rot + c_rot * c_rot;

            r_rot += kDescWidth / 2 - .5f;
            c_rot += kDescWidth / 2 - .5f;

            if (r_rot <= -1 || r_rot >= kDescWidth || c_rot <= -1 || c_rot >= kDescWidth)
            {
                continue;
            }

            int iHist = std::floor(r_rot);
            int jHist = std::floor(c_rot);

            r_rot -= iHist;
            c_rot -= jHist;

            float dx = gaussVal(r + i + 0, c + j + 1) - gaussVal(r + i + 0, c + j - 1);
            float dy = gaussVal(r + i - 1, c + j + 0) - gaussVal(r + i + 1, c + j + 0);

            float o_rot = std::atan2(dy, dx) * 180.f / M_PI;

            if (o_rot < 0.f)
                o_rot += 360.f;
            if (o_rot >= 360.f)
                o_rot -= 360.f;

            o_rot = (o_rot - angle) * kDescHistBins / 360.f;

            int bin = std::floor(o_rot);

            o_rot -= bin;

            if (bin < 0)
                bin += kDescHistBins;
            if (bin >= kDescHistBins)
                bin -= kDescHistBins;

            weight = std::exp2f(weight * kDescWeightScale);

            magnitude = std::sqrt(dx * dx + dy * dy) * weight;

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

            int idx = ((iHist + 1) * (kDescWidth + 2) + (jHist + 1)) * (kDescHistBins + 2) + bin;

            histogram[idx] += v_rco000;
            histogram[idx + 1] += v_rco001;
            histogram[idx + (kDescHistBins + 2)] += v_rco010;
            histogram[idx + (kDescHistBins + 3)] += v_rco011;
            histogram[idx + (kDescWidth + 2) * (kDescHistBins + 2)] += v_rco100;
            histogram[idx + (kDescWidth + 2) * (kDescHistBins + 2) + 1] += v_rco101;
            histogram[idx + (kDescWidth + 3) * (kDescHistBins + 2)] += v_rco110;
            histogram[idx + (kDescWidth + 3) * (kDescHistBins + 2) + 1] += v_rco111;
        }
    }

    float norm = 0.f;

    for (int i = 0; i < kDescWidth; i++)
    {
        for (int j = 0; j < kDescWidth; j++)
        {
            int histIdx = ((i + 1) * (kDescWidth + 2) + (j + 1)) * (kDescHistBins + 2);

            histogram[histIdx] += histogram[histIdx + kDescHistBins];
            histogram[histIdx + 1] += histogram[histIdx + kDescHistBins + 1];

            for (int bin = 0; bin < kDescHistBins; bin++)
            {
                magnitude = histogram[histIdx + bin];

                norm += magnitude * magnitude;
            }
        }
    }

    float histMax = std::sqrt(norm) * kDescHistPeakRatio;

    norm = 0.f;

    for (int i = 0; i < kDescWidth; i++)
    {
        for (int j = 0; j < kDescWidth; j++)
        {
            int histIdx = ((i + 1) * (kDescWidth + 2) + (j + 1)) * (kDescHistBins + 2);

            for (int bin = 0; bin < kDescHistBins; bin++)
            {
                magnitude = histogram[histIdx + bin];

                magnitude = std::min(magnitude, histMax);

                norm += magnitude * magnitude;

                histogram[histIdx + bin] = magnitude;
            }
        }
    }

    norm = 512 / std::max(std::sqrt(norm), 1e-5f);

    for (int i = 0; i < kDescWidth; i++)
    {
        for (int j = 0; j < kDescWidth; j++)
        {
            int histIdx = ((i + 1) * (kDescWidth + 2) + (j + 1)) * (kDescHistBins + 2);

            for (int bin = 0; bin < kDescHistBins; bin++)
            {
                magnitude = histogram[histIdx + bin];

                int descIdx = (i * kDescWidth + j) * kDescHistBins + bin;

                descriptor.data[descIdx] = cuda::SaturateCast<uint8_t>(magnitude * norm);
            }
        }
    }
}

inline void GoldComputeHistogram(float (&histogram)[kHistogramBins], float featRadius,
                                 const RawPyramidType &srcGaussianPyramid, const long3 &currStrides,
                                 const long3 &currShape, int octave, int layer, int currBatch, int r, int c)
{
    std::vector<float> tempHistogram(kHistogramBins + 4, 0.f);

    int radius = std::round(featRadius * kOrientationRadius);

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

            float angle     = std::atan2(dy, dx) * 180.f / M_PI;
            float weight    = std::exp2f((i * i + j * j) * weightScale);
            float magnitude = std::sqrt(dx * dx + dy * dy);

            int bin = std::round(angle * kHistogramBins / 360.f);

            bin = (bin >= kHistogramBins ? bin - kHistogramBins : (bin < 0 ? bin + kHistogramBins : bin));

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

inline void GoldAddFeatures(RawBufferType &featCoords, const long2 &featCoordsStrides, RawBufferType &featMetadata,
                            const long2 &featMetadataStrides, RawBufferType &featDescriptors,
                            const long2 &featDescriptorsStrides, int maxCapacity, RawBufferType &numFeatures,
                            const long1 &numFeaturesStrides, const RawPyramidType &srcGaussianPyramid,
                            const RawPyramidType &srcDoGPyramid, const long3 &currStrides, const long3 &currShape,
                            int octave, int firstOctave, int numOctaveLayers, float contrastThreshold,
                            float edgeThreshold, float initSigma, int l, int currBatch, int r, int c)
{
    constexpr float kImageScale = 1.f / cuda::TypeTraits<VT>::max; // source images data type scale
    constexpr float kDScale1    = kImageScale * .5f;               // first derivative scale
    constexpr float kDScale2    = kImageScale;                     // second derivative scale
    constexpr float kDScaleC    = kImageScale * .25f;              // cross derivative scale

    float                           cv;      // central value
    cuda::math::Vector<float, 3>    dD, sol; // derivative distances and solver solution
    cuda::math::Matrix<float, 3, 3> H;       // Hessian matrix

    auto dogVal = [&srcDoGPyramid, &currStrides, &octave, &currBatch](int layer, int row, int col)
    {
        return util::ValueAt<WT>(srcDoGPyramid[octave][layer], currStrides, long3{currBatch, row, col});
    };

    bool converged = false;

    for (int i = 0; i < kMaxInterpolationSteps; i++)
    {
        // clang-format off
        dD[0] = (dogVal(l + 0, r + 0, c + 1) - dogVal(l + 0, r + 0, c - 1)) * kDScale1;
        dD[1] = (dogVal(l + 0, r + 1, c + 0) - dogVal(l + 0, r - 1, c + 0)) * kDScale1;
        dD[2] = (dogVal(l + 1, r + 0, c + 0) - dogVal(l - 1, r + 0, c + 0)) * kDScale1;

        cv = dogVal(l, r, c);

        H[0][0] = (dogVal(l + 0, r + 0, c + 1) + dogVal(l + 0, r + 0, c - 1) - 2 * cv) * kDScale2;
        H[1][1] = (dogVal(l + 0, r + 1, c + 0) + dogVal(l + 0, r - 1, c + 0) - 2 * cv) * kDScale2;
        H[2][2] = (dogVal(l + 1, r + 0, c + 0) + dogVal(l - 1, r + 0, c + 0) - 2 * cv) * kDScale2;

        H[0][1] = H[1][0] = (dogVal(l + 0, r + 1, c + 1) - dogVal(l + 0, r + 1, c - 1) -
                             dogVal(l + 0, r - 1, c + 1) + dogVal(l + 0, r - 1, c - 1)) * kDScaleC;
        H[0][2] = H[2][0] = (dogVal(l + 1, r + 0, c + 1) - dogVal(l + 1, r + 0, c - 1) -
                             dogVal(l - 1, r + 0, c + 1) + dogVal(l - 1, r + 0, c - 1)) * kDScaleC;
        H[1][2] = H[2][1] = (dogVal(l + 1, r + 1, c + 0) - dogVal(l + 1, r - 1, c + 0) -
                             dogVal(l - 1, r + 1, c + 0) + dogVal(l - 1, r - 1, c + 0)) * kDScaleC;
        // clang-format on

        sol = dD;

        if (!cuda::math::solve_inplace(H, sol))
        {
            return;
        }

        sol = -sol;

        if (std::abs(sol[2]) < 0.5f && std::abs(sol[1]) < 0.5f && std::abs(sol[0]) < 0.5f)
        {
            converged = true;
            break;
        }

        if (std::abs(sol[2]) > std::numeric_limits<int>::max() / 3.f
            || std::abs(sol[1]) > std::numeric_limits<int>::max() / 3.f
            || std::abs(sol[0]) > std::numeric_limits<int>::max() / 3.f)
        {
            return;
        }

        c += std::round(sol[0]);
        r += std::round(sol[1]);
        l += std::round(sol[2]);

        if (l < 1 || l > numOctaveLayers || c < kImageBorder || c >= currShape.z - kImageBorder || r < kImageBorder
            || r >= currShape.y - kImageBorder)
        {
            return;
        }
    }

    if (!converged)
    {
        return;
    }

    float3 metadata;

    metadata.y = std::abs(cv * kImageScale + cuda::math::dot(dD, sol) * .5f);

    if (metadata.y * numOctaveLayers < contrastThreshold)
    {
        return;
    }

    float trace       = H[0][0] + H[1][1];
    float determinant = H[0][0] * H[1][1] - H[0][1] * H[1][0];

    if (determinant <= 0 || trace * trace * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * determinant)
    {
        return;
    }

    int currOctave = octave + firstOctave;

    float4 keypoint;

    keypoint.x = (c + sol[0]) * std::pow(2, currOctave);
    keypoint.y = (r + sol[1]) * std::pow(2, currOctave);
    keypoint.w = (l + sol[2]);
    keypoint.z = currOctave;

    float featRadius = initSigma * std::pow(2, keypoint.w / numOctaveLayers);

    float hist[kHistogramBins];

    GoldComputeHistogram(hist, featRadius, srcGaussianPyramid, currStrides, currShape, octave, l, currBatch, r, c);

    metadata.z = featRadius * 2.f * std::pow(2, currOctave);

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

        if (hist[i] > hist[prologue] && hist[i] > hist[epilogue] && hist[i] >= histPeak)
        {
            float bin = i + .5f * (hist[prologue] - hist[epilogue]) / (hist[prologue] - 2 * hist[i] + hist[epilogue]);

            bin = (bin < 0 ? kHistogramBins + bin : (bin >= kHistogramBins ? bin - kHistogramBins : bin));

            metadata.x = 360.f - (360.f / kHistogramBins) * bin;

            if (cuda::abs(metadata.x - 360.f) < 1e-5)
                metadata.x = 0.f;

            ASSERT_TRUE(metadata.x >= 0.f && metadata.x <= 360.f);

            float descAngle = 360.f - metadata.x;

            if (cuda::abs(descAngle - 360.f) < 1e-5)
                descAngle = 0.f;

            ASSERT_TRUE(descAngle >= 0.f && descAngle <= 360.f);

            DescriptorType descriptor;
            GoldComputeDescriptor(descriptor, descAngle, featRadius, srcGaussianPyramid, currStrides, currShape, octave,
                                  l, currBatch, r, c);

            int &featIdx = util::ValueAt<int>(numFeatures, numFeaturesStrides, long1{currBatch});

            if (featIdx < maxCapacity)
            {
                util::ValueAt<float4>(featCoords, featCoordsStrides, long2{currBatch, featIdx})     = keypoint;
                util::ValueAt<float3>(featMetadata, featMetadataStrides, long2{currBatch, featIdx}) = metadata;
                util::ValueAt<DescriptorType>(featDescriptors, featDescriptorsStrides, long2{currBatch, featIdx})
                    = descriptor;
            }

            featIdx += 1;
        }
    }
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
    int threshold = std::floor(.5f * contrastThreshold / numOctaveLayers * 255);

    long3 currShape   = baseShape;
    long3 currStrides = baseStrides;

    auto dogVal = [&currStrides, &srcDoGPyramid](int octave, int layer, long batch, long row, long col)
    {
        return util::ValueAt<WT>(srcDoGPyramid[octave][layer], currStrides, long3{batch, row, col});
    };

    long maxCapacity = featCoordsShape.y;

    ASSERT_TRUE(featCoordsShape.x == currShape.x && featMetadataShape.x == currShape.x
                && featCoordsShape.y == featMetadataShape.y && numFeaturesShape.x == currShape.x);

    for (int o = 0; o < numOctaves; o++)
    {
        for (int l = 1; l <= numOctaveLayers; l++)
        {
            for (long b = 0; b < currShape.x; b++)
            {
                for (long r = kImageBorder; r < currShape.y - kImageBorder; r++)
                {
                    for (long c = kImageBorder; c < currShape.z - kImageBorder; c++)
                    {
                        WT val = dogVal(o, l, b, r, c);

                        // clang-format off
                        if (std::abs(val) > threshold &&
                            ((val > 0 &&
                              val >= dogVal(o, l, b, r + 0, c - 1) && val >= dogVal(o, l, b, r + 0, c + 1) &&
                              val >= dogVal(o, l, b, r - 1, c - 1) && val >= dogVal(o, l, b, r - 1, c + 0) &&
                              val >= dogVal(o, l, b, r - 1, c + 1) && val >= dogVal(o, l, b, r + 1, c - 1) &&
                              val >= dogVal(o, l, b, r + 1, c + 0) && val >= dogVal(o, l, b, r + 1, c + 1) &&
                              val >= dogVal(o, l + 1, b, r + 0, c + 0) &&
                              val >= dogVal(o, l + 1, b, r + 0, c - 1) && val >= dogVal(o, l + 1, b, r + 0, c + 1) &&
                              val >= dogVal(o, l + 1, b, r - 1, c - 1) && val >= dogVal(o, l + 1, b, r - 1, c + 0) &&
                              val >= dogVal(o, l + 1, b, r - 1, c + 1) && val >= dogVal(o, l + 1, b, r + 1, c - 1) &&
                              val >= dogVal(o, l + 1, b, r + 1, c + 0) && val >= dogVal(o, l + 1, b, r + 1, c + 1) &&
                              val >= dogVal(o, l - 1, b, r + 0, c + 0) &&
                              val >= dogVal(o, l - 1, b, r + 0, c - 1) && val >= dogVal(o, l - 1, b, r + 0, c + 1) &&
                              val >= dogVal(o, l - 1, b, r - 1, c - 1) && val >= dogVal(o, l - 1, b, r - 1, c + 0) &&
                              val >= dogVal(o, l - 1, b, r - 1, c + 1) && val >= dogVal(o, l - 1, b, r + 1, c - 1) &&
                              val >= dogVal(o, l - 1, b, r + 1, c + 0) && val >= dogVal(o, l - 1, b, r + 1, c + 1)) ||
                             (val < 0 &&
                              val <= dogVal(o, l, b, r + 0, c - 1) && val <= dogVal(o, l, b, r + 0, c + 1) &&
                              val <= dogVal(o, l, b, r - 1, c - 1) && val <= dogVal(o, l, b, r - 1, c + 0) &&
                              val <= dogVal(o, l, b, r - 1, c + 1) && val <= dogVal(o, l, b, r + 1, c - 1) &&
                              val <= dogVal(o, l, b, r + 1, c + 0) && val <= dogVal(o, l, b, r + 1, c + 1) &&
                              val <= dogVal(o, l + 1, b, r + 0, c + 0) &&
                              val <= dogVal(o, l + 1, b, r + 0, c - 1) && val <= dogVal(o, l + 1, b, r + 0, c + 1) &&
                              val <= dogVal(o, l + 1, b, r - 1, c - 1) && val <= dogVal(o, l + 1, b, r - 1, c + 0) &&
                              val <= dogVal(o, l + 1, b, r - 1, c + 1) && val <= dogVal(o, l + 1, b, r + 1, c - 1) &&
                              val <= dogVal(o, l + 1, b, r + 1, c + 0) && val <= dogVal(o, l + 1, b, r + 1, c + 1) &&
                              val <= dogVal(o, l - 1, b, r + 0, c + 0) &&
                              val <= dogVal(o, l - 1, b, r + 0, c - 1) && val <= dogVal(o, l - 1, b, r + 0, c + 1) &&
                              val <= dogVal(o, l - 1, b, r - 1, c - 1) && val <= dogVal(o, l - 1, b, r - 1, c + 0) &&
                              val <= dogVal(o, l - 1, b, r - 1, c + 1) && val <= dogVal(o, l - 1, b, r + 1, c - 1) &&
                              val <= dogVal(o, l - 1, b, r + 1, c + 0) && val <= dogVal(o, l - 1, b, r + 1, c + 1))))
                        {
                            // clang-format on

                            GoldAddFeatures(featCoords, featCoordsStrides, featMetadata, featMetadataStrides,
                                            featDescriptors, featDescriptorsStrides, maxCapacity, numFeatures,
                                            numFeaturesStrides, srcGaussianPyramid, srcDoGPyramid, currStrides,
                                            currShape, o, firstOctave, numOctaveLayers, contrastThreshold,
                                            edgeThreshold, initSigma, l, b, r, c);
                        }
                    } // for each column
                }     // for each row
            }         // for each batch image
        }             // for each layer

        currShape.y /= 2;
        currShape.z /= 2;
        currStrides = GoldStrides(currShape);
    } // for each octave
}

// Struct to hold SIFT results
struct SIFTResults
{
    using TupleType = std::tuple<float4, float3, DescriptorType>; // float4 coordinates, float3 metadata, descriptor

    std::vector<std::vector<TupleType>> testFeatures, goldFeatures;

    std::vector<int> testNumFeatures, goldNumFeatures;
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

    RawPyramidType pyrGaussian, pyrDoG;
    long3          baseShape, baseStrides;

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

        std::sort(outResults.testFeatures[x].begin(), outResults.testFeatures[x].end(), featureLower);
        std::sort(outResults.goldFeatures[x].begin(), outResults.goldFeatures[x].end(), featureLower);
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

TYPED_TEST(OpSIFT, correct_output)
{
    int3  inShape           = type::GetValue<TypeParam, 0>;
    long  capacity          = type::GetValue<TypeParam, 1>;
    int   numOctaveLayers   = type::GetValue<TypeParam, 2>;
    float contrastThreshold = type::GetValue<TypeParam, 3>;
    float edgeThreshold     = type::GetValue<TypeParam, 4>;
    float initSigma         = type::GetValue<TypeParam, 5>;
    bool  expandInput       = type::GetValue<TypeParam, 6>;

    NVCVSIFTFlagType flags = expandInput ? NVCV_SIFT_USE_EXPANDED_INPUT : NVCV_SIFT_USE_ORIGINAL_INPUT;

    // Increasing inShape and numOctaveLayers to test bigger maxShape and maxOctaveLayers
    int3 maxShape        = (inShape + 3) * (expandInput ? 2 : 1);
    int  maxOctaveLayers = numOctaveLayers + 1;

    nvcv::Tensor src = nvcv::util::CreateTensor(inShape.z, inShape.x, inShape.y, kInFormat);

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    long3 srcShape{srcAccess->numSamples(), srcAccess->numRows(), srcAccess->numCols()};
    long3 srcStrides{srcAccess->sampleStride(), srcAccess->rowStride(), srcAccess->colStride()};

    // While inShape is WHN, srcShape is NHW to match srcStrides
    ASSERT_TRUE(inShape.z == srcShape.x && inShape.y == srcShape.y && inShape.x == srcShape.z);

    srcStrides.x = (srcData->rank() == 3) ? srcShape.y * srcStrides.y : srcStrides.x;

    long srcBufSize = srcStrides.x * srcShape.x;

    RawBufferType srcVec(srcBufSize);

    // clang-format off

    std::uniform_int_distribution<VT> rg(0, 255);

    for (long x = 0; x < srcShape.x; ++x)
        for (long y = 0; y < srcShape.y; ++y)
            for (long z = 0; z < srcShape.z; ++z)
                util::ValueAt<VT>(srcVec, srcStrides, long3{x, y, z}) = rg(g_rng);

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

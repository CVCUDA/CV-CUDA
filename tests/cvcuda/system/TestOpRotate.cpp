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

#include <common/ValueTests.hpp>
#include <cvcuda/OpRotate.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cmath>
#include <random>

namespace t    = ::testing;
namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

constexpr double PI = 3.1415926535897932384626433832795; // NOSONAR: std::numbers::pi is C++20.

// #define DBG_ROTATE 1

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(static_cast<double>(size) * scale);
}

static void compute_warpAffine(const double angle, const double xShift, const double yShift, double *aCoeffs)
{
    aCoeffs[0] = cos(angle * PI / 180);
    aCoeffs[1] = sin(angle * PI / 180);
    aCoeffs[2] = xShift;
    aCoeffs[3] = -sin(angle * PI / 180);
    aCoeffs[4] = cos(angle * PI / 180);
    aCoeffs[5] = yShift;
}

static void compute_center_shift(const int center_x, const int center_y, const double angle, double &xShift,
                                 double &yShift)
{
    xShift = (1 - cos(angle * PI / 180)) * static_cast<double>(center_x)
           - sin(angle * PI / 180) * static_cast<double>(center_y);
    yShift = sin(angle * PI / 180) * static_cast<double>(center_x)
           + (1 - cos(angle * PI / 180)) * static_cast<double>(center_y);
}

static void assignCustomValuesInSrc(std::vector<uint8_t> &srcVec, int srcWidth, int srcHeight, int srcVecRowStride)
{
    int  initialValue = 1;
    auto pixelBytes   = srcVecRowStride / srcWidth;
    for (int i = 0; i < srcHeight; i++)
    {
        for (int j = 0; j < srcVecRowStride; j = j + pixelBytes)
        {
            for (int k = 0; k < pixelBytes; k++)
            {
                srcVec[i * srcVecRowStride + j + k] = static_cast<uint8_t>(initialValue);
            }
            initialValue++;
        }
    }

#if DBG_ROTATE
    std::cout << "\nPrint input " << std::endl;

    for (int i = 0; i < srcHeight; i++)
    {
        for (int j = 0; j < srcVecRowStride; j++)
        {
            std::cout << static_cast<int>(srcVec[i * srcVecRowStride + j]) << ",";
        }
        std::cout << std::endl;
    }
#endif
}

static bool IsInsideSource(float src_x, float src_y, int width, int height)
{
    return src_x > -0.5f && src_x < static_cast<float>(width) && src_y > -0.5f && src_y < static_cast<float>(height);
}

template<typename T>
static void StoreLinearPixel(T *dstPtr, int dstBase, const T *srcPtr, int srcRowStride, int elementsPerPixel,
                             float src_x, float src_y, int width, int height)
{
    if (!IsInsideSource(src_x, src_y, width, height))
    {
        return;
    }

    const int x1 = cuda::round<cuda::RoundMode::DOWN, int>(src_x);
    const int y1 = cuda::round<cuda::RoundMode::DOWN, int>(src_y);

    const int x2      = x1 + 1;
    const int y2      = y1 + 1;
    const int x1_read = std::max(x1, 0);
    const int y1_read = std::max(y1, 0);
    const int x2_read = std::min(x2, width - 1);
    const int y2_read = std::min(y2, height - 1);

    for (int k = 0; k < elementsPerPixel; k++)
    {
        float out = 0.;

        T src_reg = srcPtr[y1_read * srcRowStride + x1_read * elementsPerPixel + k];
        out       = out + src_reg * ((static_cast<float>(x2) - src_x) * (static_cast<float>(y2) - src_y));

        src_reg = srcPtr[y1_read * srcRowStride + x2_read * elementsPerPixel + k];
        out     = out + src_reg * ((src_x - static_cast<float>(x1)) * (static_cast<float>(y2) - src_y));

        src_reg = srcPtr[y2_read * srcRowStride + x1_read * elementsPerPixel + k];
        out     = out + src_reg * ((static_cast<float>(x2) - src_x) * (src_y - static_cast<float>(y1)));

        src_reg = srcPtr[y2_read * srcRowStride + x2_read * elementsPerPixel + k];
        out     = out + src_reg * ((src_x - static_cast<float>(x1)) * (src_y - static_cast<float>(y1)));

        dstPtr[dstBase + k] = cuda::SaturateCast<T>(out);
    }
}

template<typename T>
static void StoreNearestPixel(T *dstPtr, int dstBase, const T *srcPtr, int srcRowStride, int elementsPerPixel,
                              float src_x, float src_y, int width, int height)
{
    if (!IsInsideSource(src_x, src_y, width, height))
    {
        return;
    }

    const int x1 = std::min(cuda::round<cuda::RoundMode::DOWN, int>(src_x + .5f), width - 1);
    const int y1 = std::min(cuda::round<cuda::RoundMode::DOWN, int>(src_y + .5f), height - 1);

    for (int k = 0; k < elementsPerPixel; k++)
    {
        dstPtr[dstBase + k] = srcPtr[y1 * srcRowStride + x1 * elementsPerPixel + k];
    }
}

template<typename T>
static void Rotate(std::vector<T> &hDst, int dstRowStride, nvcv::Size2D dstSize, const std::vector<T> &hSrc,
                   int srcRowStride, nvcv::Size2D, nvcv::ImageFormat fmt, const double angleDeg, const double2 shift,
                   NVCVInterpolationType interpolation)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    T       *dstPtr = hDst.data();
    const T *srcPtr = hSrc.data();

    // calculate coefficients
    std::array<double, 6> d_aCoeffs;
    compute_warpAffine(angleDeg, shift.x, shift.y, d_aCoeffs.data());

    int width  = dstSize.w;
    int height = dstSize.h;

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            const double dst_x_shift = dst_x - d_aCoeffs[2];
            const double dst_y_shift = dst_y - d_aCoeffs[5];

            auto src_x = static_cast<float>(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
            auto src_y = static_cast<float>(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

            if (interpolation == NVCV_INTERP_LINEAR)
            {
                StoreLinearPixel(dstPtr, dst_y * dstRowStride + dst_x * elementsPerPixel, srcPtr, srcRowStride,
                                 elementsPerPixel, src_x, src_y, width, height);
            }
            else if (interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_CUBIC)
            {
                /*
                    Use this for NVCV_INTERP_CUBIC interpolation only for angles - {90, 180}
                */
                StoreNearestPixel(dstPtr, dst_y * dstRowStride + dst_x * elementsPerPixel, srcPtr, srcRowStride,
                                  elementsPerPixel, src_x, src_y, width, height);
            }
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpRotate, test::ValueList<int, int, int, int, NVCVInterpolationType, int, double>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,         interpolation, numberImages, angle
    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           1,     90},
    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           4,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            1,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            4,     90},

    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           1,     45},
    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           4,     45},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            1,     45},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            4,     45},

    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             1,     90},
    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             4,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             1,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             4,     90},

    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             1,     180},
    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             4,     180},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             1,     180},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             4,     180},
});

// clang-format on

TEST_P(OpRotate, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    NVCVInterpolationType interpolation = GetParamValue<4>();

    int numberOfImages = GetParamValue<5>();

    double angleDeg = GetParamValue<6>();
    double shiftX   = -1;
    double shiftY   = -1;

    const nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    // Generate input
    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);

    auto srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    int                               srcVecRowStride = srcWidth * fmt.planePixelStrideBytes(0);

    for (int i = 0; i < numberOfImages; ++i)
    {
        srcVec[i].resize(srcHeight * srcVecRowStride);
        std::ranges::generate(srcVec[i], []() { return 0; });

        // Assign custom values in input vector
        assignCustomValuesInSrc(srcVec[i], srcWidth, srcHeight, srcVecRowStride);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(), srcVecRowStride,
                               srcVecRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, fmt);

    // Compute shiftX, shiftY using center
    int center_x = (srcWidth - 1) / 2;
    int center_y = (srcHeight - 1) / 2;
    compute_center_shift(center_x, center_y, angleDeg, shiftX, shiftY);

    cvcuda::Rotate RotateOp(0);
    double2        shift = {shiftX, shiftY};
    EXPECT_NO_THROW(RotateOp(stream, imgSrc, imgDst, angleDeg, shift, interpolation));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowStride = dstWidth * fmt.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(dstHeight * dstVecRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowStride, dstAccess->sampleData(i), dstAccess->rowStride(),
                               dstVecRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstVecRowStride);
        std::ranges::generate(goldVec, []() { return 0; });

        // Generate gold result
        Rotate<uint8_t>(goldVec, dstVecRowStride, {dstWidth, dstHeight}, srcVec[i], srcVecRowStride,
                        {srcWidth, srcHeight}, fmt, angleDeg, shift, interpolation);

#if DBG_ROTATE
        std::cout << "\nPrint golden output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowStride; j++)
            {
                std::cout << static_cast<int>(goldVec[k * dstVecRowStride + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint rotated output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowStride; j++)
            {
                std::cout << static_cast<int>(testVec[k * dstVecRowStride + j]) << ",";
            }
            std::cout << std::endl;
        }
#endif

        EXPECT_EQ(goldVec, testVec);
    }
}

TEST_P(OpRotate, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidthBase  = GetParamValue<0>();
    int srcHeightBase = GetParamValue<1>();

    NVCVInterpolationType interpolation = GetParamValue<4>();

    int numberOfImages = GetParamValue<5>();

    double angleDegBase = GetParamValue<6>();

    const nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndSrcWidth(ScaledSize(srcWidthBase, 0.8), ScaledSize(srcWidthBase, 1.1));
    std::uniform_int_distribution rndSrcHeight(ScaledSize(srcHeightBase, 0.8), ScaledSize(srcHeightBase, 1.1));
    std::uniform_int_distribution rndAngle(0, 360);

    nvcv::Tensor angleDegTensor(nvcv::TensorShape({numberOfImages}, "N"), nvcv::TYPE_F64);
    auto         angleDegTensorData = angleDegTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nvcv::NullOpt, angleDegTensorData);

    nvcv::Tensor shiftTensor(nvcv::TensorShape({numberOfImages, 2}, nvcv::TENSOR_NW), nvcv::TYPE_F64);
    auto         shiftTensorData = shiftTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nvcv::NullOpt, shiftTensorData);

    auto shiftTensorDataAccess = nvcv::TensorDataAccessStrided::Create(*shiftTensorData);
    ASSERT_TRUE(shiftTensorDataAccess);

    std::vector<nvcv::Image> imgSrc;

    std::vector<nvcv::Image> imgDst;
    std::vector<double>      angleDegVecs;
    std::vector<double2>     shiftVecs;

    for (int i = 0; i < numberOfImages; ++i)
    {
        int tmpWidth  = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
        int tmpHeight = i == 0 ? srcHeightBase : rndSrcHeight(randEng);

        imgSrc.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);

        imgDst.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);

        double2 shift    = {-1, -1};
        double  angleDeg = i == 0 ? angleDegBase : rndAngle(randEng);
        if (i != 0 && interpolation == NVCV_INTERP_CUBIC)
        {
            // Use the computed angle as rand int and
            // then compute index to pick one of the angles
            std::vector<double> tmpAngleValues = {90, 180, 270};
            size_t              indexToChoose  = static_cast<size_t>(angleDeg) % tmpAngleValues.size();
            angleDeg                           = tmpAngleValues[indexToChoose];
        }

        // Compute shiftX, shiftY using center
        int center_x = (tmpWidth - 1) / 2;
        int center_y = (tmpHeight - 1) / 2;
        compute_center_shift(center_x, center_y, angleDeg, shift.x, shift.y);

        angleDegVecs.push_back(angleDeg);
        shiftVecs.push_back(shift);
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(angleDegTensorData->basePtr(), angleDegVecs.data(),
                                           angleDegVecs.size() * sizeof(double), cudaMemcpyHostToDevice, stream));

    ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(shiftTensorDataAccess->sampleData(0),
                                             shiftTensorDataAccess->sampleStride(), shiftVecs.data(), sizeof(double2),
                                             sizeof(double2), numberOfImages, cudaMemcpyHostToDevice, stream));

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    std::vector<int>                  srcVecRowStride(numberOfImages);

    // Populate input
    for (int i = 0; i < numberOfImages; ++i)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcRowStride);
        std::ranges::generate(srcVec[i], []() { return 0; });

        // Assign custom values in input vector
        assignCustomValuesInSrc(srcVec[i], srcWidth, srcHeight, srcRowStride);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(), srcRowStride,
                               srcRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    cvcuda::Rotate rotateOp(numberOfImages);
    EXPECT_NO_THROW(rotateOp(stream, batchSrc, batchDst, angleDegTensor, shiftTensor, interpolation));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);
        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * fmt.planePixelStrideBytes(0);
        int srcRowStride = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dstHeight * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstRowStride);
        std::ranges::generate(goldVec, []() { return 0; });

        // Generate gold result
        Rotate<uint8_t>(goldVec, dstRowStride, {dstWidth, dstHeight}, srcVec[i], srcRowStride, {srcWidth, srcHeight},
                        fmt, angleDegVecs[i], shiftVecs[i], interpolation);

        EXPECT_EQ(goldVec, testVec);
    }
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// Rotate maps each output pixel to a source pixel with the same per-image affine coefficients
// regardless of the channel, so a planar input is rotated plane-by-plane and must produce exactly the
// same pixels as the interleaved path. These tests feed identical data in both layouts through
// cvcuda::Rotate and require the (re-interleaved) planar output to match the interleaved output
// bit-for-bit, for every dtype and interpolation type.
//
// Unlike Resize/Flip, Rotate leaves out-of-bounds destination pixels unwritten (the source maps
// outside the image under a replicate border guard), so the destination is zero-filled before each
// run; a constant byte value is layout-invariant, keeping uncovered regions equal across layouts.
// =============================================================================

namespace {

// Zero every plane of every sample on the stream.
void ZeroTensor(const nvcv::Tensor &t, cudaStream_t stream)
{
    auto data = t.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(data, nvcv::NullOpt);
    auto acc = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
    ASSERT_TRUE(acc);
    for (int n = 0; n < acc->numSamples(); ++n)
    {
        for (int p = 0; p < acc->numPlanes(); ++p)
        {
            ASSERT_EQ(cudaSuccess, cudaMemset2DAsync(acc->planeData(p, acc->sampleData(n)), acc->rowStride(), 0,
                                                     acc->rowStride(), acc->numRows(), stream));
        }
    }
}

// Zero every plane of every image in a var-shape batch on the stream.
void ZeroVarShapeBatch(const nvcv::ImageBatchVarShape &batch, cudaStream_t stream)
{
    for (int i = 0; i < batch.numImages(); ++i)
    {
        auto data = batch[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(data, nvcv::NullOpt);
        for (int p = 0; p < data->numPlanes(); ++p)
        {
            const auto &plane = data->plane(p);
            ASSERT_EQ(cudaSuccess,
                      cudaMemsetAsync(plane.basePtr, 0, static_cast<size_t>(plane.rowStride) * plane.height, stream));
        }
    }
}

// Rotate identical data in interleaved and planar tensor layout; outputs must match bit-for-bit. The
// shared scaffolding (upload/run/download/compare) lives in PlanarParityUtils.hpp; here we only bind
// the Rotate call (with a center-shift so the rotation pivots about the image center, like the other
// Rotate tests).
void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                               NVCVInterpolationType interp, double angleDeg, int numImages)
{
    double shiftX = 0;
    double shiftY = 0;
    compute_center_shift((w - 1) / 2, (h - 1) / 2, angleDeg, shiftX, shiftY);
    const double2 shift = {shiftX, shiftY};

    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, w, h, w, h, numImages,
        [angleDeg, shift, interp](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            ZeroTensor(dst, s);
            cvcuda::Rotate op(0);
            EXPECT_NO_THROW(op(s, src, dst, angleDeg, shift, interp));
        });
}

// Var-shape counterpart of RunPlanarParityTensorCase. Var-shape Rotate takes per-image angle/shift
// tensors; upload them once (synchronously, so they are ready before the op runs on the parity
// helper's stream). All images share the same transform here.
void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                                 NVCVInterpolationType interp, double angleDeg, int numImages)
{
    double shiftX = 0;
    double shiftY = 0;
    compute_center_shift((w - 1) / 2, (h - 1) / 2, angleDeg, shiftX, shiftY);

    nvcv::Tensor angleDegTensor(nvcv::TensorShape({numImages}, "N"), nvcv::TYPE_F64);
    nvcv::Tensor shiftTensor(nvcv::TensorShape({numImages, 2}, nvcv::TENSOR_NW), nvcv::TYPE_F64);
    {
        std::vector<double>  angles(numImages, angleDeg);
        std::vector<double2> shifts(numImages, double2{shiftX, shiftY});

        auto angleData = angleDegTensor.exportData<nvcv::TensorDataStridedCuda>();
        auto shiftData = shiftTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(angleData, nvcv::NullOpt);
        ASSERT_NE(shiftData, nvcv::NullOpt);
        auto shiftAcc = nvcv::TensorDataAccessStrided::Create(*shiftData);
        ASSERT_TRUE(shiftAcc);

        ASSERT_EQ(cudaSuccess, cudaMemcpy(angleData->basePtr(), angles.data(), angles.size() * sizeof(double),
                                          cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(shiftAcc->sampleData(0), shiftAcc->sampleStride(), shifts.data(),
                                            sizeof(double2), sizeof(double2), numImages, cudaMemcpyHostToDevice));
    }

    test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, w, h, w, h, numImages,
        [numImages, &angleDegTensor, &shiftTensor, interp](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                                           const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            ZeroVarShapeBatch(dst, s);
            cvcuda::Rotate op(numImages);
            EXPECT_NO_THROW(op(s, src, dst, angleDegTensor, shiftTensor, interp));
        });
}

} // namespace

// Parameters: width, height, interpolation, angle (deg), numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpRotatePlanar,
                  test::ValueList<int, int, NVCVInterpolationType, double, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, NVCV_INTERP_NEAREST,  90, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, nearest
    {123,  66,  NVCV_INTERP_LINEAR,  45, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, linear, fractional
    { 64,  48,   NVCV_INTERP_CUBIC,  30, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, cubic, fractional
    { 50,  40, NVCV_INTERP_NEAREST,  90, 2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8}, // RGBA8, nearest
    {100,  80,  NVCV_INTERP_LINEAR,  60, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8}, // RGBA8, linear, fractional
    { 64,  48,   NVCV_INTERP_CUBIC,  45, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32}, // float, cubic, fractional
    { 72,  56,  NVCV_INTERP_LINEAR, 120, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32}, // float, linear
});

// clang-format on

TEST_P(OpRotatePlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<5>(), GetParamValue<6>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>());
}

TEST_P(OpRotatePlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<5>(), GetParamValue<6>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>());
}

// clang-format off
NVCV_TEST_SUITE_P(OpRotate_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, NVCVInterpolationType>{
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, NVCV_INTERP_LANCZOS},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, NVCV_INTERP_NEAREST}, // data format is different (interleaved in, planar out)
    {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, NVCV_INTERP_NEAREST},
});

NVCV_TEST_SUITE_P(OpRotateVarshape_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int, int, NVCVInterpolationType, nvcv::DataType, nvcv::DataType>{
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 2, 5, NVCV_INTERP_LANCZOS, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 6, 5, NVCV_INTERP_NEAREST, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 2, -1, NVCV_INTERP_NEAREST, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 2, 5, NVCV_INTERP_NEAREST, nvcv::TYPE_F64, nvcv::TYPE_F64}, // mismatched layout
    {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, 2, 5, NVCV_INTERP_NEAREST, nvcv::TYPE_F64, nvcv::TYPE_F64},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 2, 5, NVCV_INTERP_NEAREST, nvcv::TYPE_F32, nvcv::TYPE_F64},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 2, 5, NVCV_INTERP_NEAREST, nvcv::TYPE_F64, nvcv::TYPE_F32},
});

// clang-format on

TEST_P(OpRotate_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat     inputFmt      = GetParamValue<0>();
    nvcv::ImageFormat     outputFmt     = GetParamValue<1>();
    NVCVInterpolationType interpolation = GetParamValue<2>();

    // Generate input
    nvcv::Tensor imgSrc(2, {4, 4}, inputFmt);
    // Generate test result
    nvcv::Tensor imgDst(2, {4, 4}, outputFmt);

    cvcuda::Rotate RotateOp(0);
    double         angleDeg = 90;
    double2        shift    = {-1, -1};
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&RotateOp, &stream, &imgSrc, &imgDst, &angleDeg, &shift, &interpolation]
                                { RotateOp(stream, imgSrc, imgDst, angleDeg, shift, interpolation); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpRotateVarshape_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat     inputFmt             = GetParamValue<0>();
    nvcv::ImageFormat     outputFmt            = GetParamValue<1>();
    const int             numberOfImages       = GetParamValue<2>();
    const int             maxVarShapeBatchSize = GetParamValue<3>();
    NVCVInterpolationType interpolation        = GetParamValue<4>();
    nvcv::DataType        angleDataType        = GetParamValue<5>();
    nvcv::DataType        shiftDataType        = GetParamValue<6>();

    int srcWidthBase  = 4;
    int srcHeightBase = 4;

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndSrcWidth(ScaledSize(srcWidthBase, 0.8), ScaledSize(srcWidthBase, 1.1));
    std::uniform_int_distribution rndSrcHeight(ScaledSize(srcHeightBase, 0.8), ScaledSize(srcHeightBase, 1.1));

    nvcv::Tensor angleDegTensor(nvcv::TensorShape({numberOfImages}, "N"), angleDataType);
    nvcv::Tensor shiftTensor(nvcv::TensorShape({numberOfImages, 2}, nvcv::TENSOR_NW), shiftDataType);

    std::vector<nvcv::Image> imgSrc;

    std::vector<nvcv::Image> imgDst;

    for (int i = 0; i < numberOfImages; ++i)
    {
        int tmpWidth  = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
        int tmpHeight = i == 0 ? srcHeightBase : rndSrcHeight(randEng);

        imgSrc.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, inputFmt);
        imgDst.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, outputFmt);
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Generate test result
    cvcuda::Rotate rotateOp(maxVarShapeBatchSize);
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&rotateOp, &stream, &batchSrc, &batchDst, &angleDegTensor, &shiftTensor, &interpolation]
                          { rotateOp(stream, batchSrc, batchDst, angleDegTensor, shiftTensor, interpolation); }));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpRotate_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat     fmt            = nvcv::FMT_RGB8;
    const int             numberOfImages = 5;
    NVCVInterpolationType interpolation  = NVCV_INTERP_NEAREST;

    int srcWidthBase  = 4;
    int srcHeightBase = 4;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_RGBA8,             fmt},
        {            fmt, nvcv::FMT_RGBA8}
    };
    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
    {
        // Create input and output
        std::default_random_engine    randEng;
        std::uniform_int_distribution rndSrcWidth(ScaledSize(srcWidthBase, 0.8), ScaledSize(srcWidthBase, 1.1));
        std::uniform_int_distribution rndSrcHeight(ScaledSize(srcHeightBase, 0.8), ScaledSize(srcHeightBase, 1.1));

        nvcv::Tensor angleDegTensor(nvcv::TensorShape({numberOfImages}, "N"), nvcv::TYPE_F64);
        nvcv::Tensor shiftTensor(nvcv::TensorShape({numberOfImages, 2}, nvcv::TENSOR_NW), nvcv::TYPE_F64);

        std::vector<nvcv::Image> imgSrc;

        std::vector<nvcv::Image> imgDst;

        for (int i = 0; i < numberOfImages - 1; ++i)
        {
            int tmpWidth  = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
            int tmpHeight = i == 0 ? srcHeightBase : rndSrcHeight(randEng);

            imgSrc.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);
            imgDst.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchDst(numberOfImages);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        cvcuda::Rotate rotateOp(numberOfImages);
        EXPECT_EQ(
            NVCV_ERROR_INVALID_ARGUMENT,
            nvcv::ProtectCall([&rotateOp, &stream, &batchSrc, &batchDst, &angleDegTensor, &shiftTensor, &interpolation]
                              { rotateOp(stream, batchSrc, batchDst, angleDegTensor, shiftTensor, interpolation); }));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpRotate_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaRotateCreate(nullptr, 2), NVCV_ERROR_INVALID_ARGUMENT);
}

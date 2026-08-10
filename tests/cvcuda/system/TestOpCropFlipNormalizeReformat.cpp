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

#include <common/BorderUtils.hpp>
#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpCropFlipNormalizeReformat.hpp>
#include <cvcuda/OpNormalize.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace gt    = ::testing;
namespace test  = nvcv::test;
namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

static int ScaledSize(int value, double scale)
{
    return static_cast<int>(value * scale);
}

static int2 FlippedCropCoord(int j, int i, int flip_code, const NVCVRectI &cropRect)
{
    int2 coord{j, i};

    if (flip_code == 1)
    {
        coord.x = cropRect.width - 1 - j;
    }
    else if (flip_code == 0)
    {
        coord.y = cropRect.height - 1 - i;
    }
    else if (flip_code == -1)
    {
        coord.x = cropRect.width - 1 - j;
        coord.y = cropRect.height - 1 - i;
    }

    coord.x += cropRect.x;
    coord.y += cropRect.y;

    return coord;
}

static float NormalizeScaleValue(const std::vector<float> &hScale, int scaleRowStride, nvcv::ImageFormat scaleFormat,
                                 int si, int sj, int sk, const float epsilon, const uint32_t flags)
{
    float scale = hScale.at(si * scaleRowStride + sj * scaleFormat.numChannels() + sk);

    if (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
    {
        scale = float{1} / std::sqrt(scale * scale + epsilon);
    }

    return scale;
}

template<typename T_Dst>
static void WriteCropDestination(std::vector<uint8_t> &hDst, long4_16a outStrides, bool dst_planar, int j, int i, int k,
                                 T_Dst value)
{
    if (dst_planar)
    {
        test::ValueAt<T_Dst>(hDst, outStrides, int4{j, i, k, 0}) = value;
    }
    else
    {
        test::ValueAt<T_Dst>(hDst, outStrides, int4{k, j, i, 0}) = value;
    }
}

template<typename T_Dst>
static void ClearCropDestination(std::vector<uint8_t> &hDst, long4_16a outStrides, bool dst_planar, int numChannels,
                                 int j, int i)
{
    for (int k = 0; k < numChannels; k++)
    {
        WriteCropDestination<T_Dst>(hDst, outStrides, dst_planar, j, i, k, 0);
    }
}

template<typename T_Src, NVCVBorderType B>
static T_Src ReadCropSource(const std::vector<uint8_t> &hSrc, const long4_16a &inStrides, bool src_planar,
                            nvcv::Size2D src_size, const float borderValue, int2 coord, int k)
{
    int2 size{src_size.w, src_size.h};
    auto typedBorderValue = cuda::SaturateCast<T_Src>(borderValue);

    if (src_planar)
    {
        return test::ValueAt<B, T_Src>(hSrc, long4_16a{inStrides.x, inStrides.z, inStrides.w, inStrides.y}, size,
                                       typedBorderValue, int4{k, coord.x, coord.y, 0});
    }

    return test::ValueAt<B, T_Src>(hSrc, long4_16a{inStrides.x, inStrides.y, inStrides.z, inStrides.w}, size,
                                   typedBorderValue, int4{k, coord.x, coord.y, 0});
}

template<typename T_Src, typename T_Dst, NVCVBorderType B>
static void NormalizeCropPixel(std::vector<uint8_t> &hDst, long4_16a outStrides, const std::vector<uint8_t> &hSrc,
                               long4_16a inStrides, nvcv::Size2D src_size, bool src_planar, bool dst_planar,
                               nvcv::ImageFormat fmt, const float borderValue, int2 coord,
                               const std::vector<float> &hBase, int baseRowStride, int bi, int bj,
                               nvcv::ImageFormat baseFormat, const std::vector<float> &hScale, int scaleRowStride,
                               int si, int sj, nvcv::ImageFormat scaleFormat, const float globalScale,
                               const float globalShift, const float epsilon, const uint32_t flags, int j, int i)
{
    for (int k = 0; k < fmt.numChannels(); k++)
    {
        const int bk   = (baseFormat.numChannels() == 1 ? 0 : k);
        const int sk   = (scaleFormat.numChannels() == 1 ? 0 : k);
        float     mul  = NormalizeScaleValue(hScale, scaleRowStride, scaleFormat, si, sj, sk, epsilon, flags);
        float     base = hBase.at(bi * baseRowStride + bj * baseFormat.numChannels() + bk);
        T_Src     out  = ReadCropSource<T_Src, B>(hSrc, inStrides, src_planar, src_size, borderValue, coord, k);

        WriteCropDestination<T_Dst>(
            hDst, outStrides, dst_planar, j, i, k,
            cuda::SaturateCast<T_Dst>((static_cast<float>(out) - base) * mul * globalScale + globalShift));
    }
}

template<typename T_Src, typename T_Dst, NVCVBorderType B>
static void CropFlipNormalizeReformatRow(std::vector<uint8_t> &hDst, long4_16a outStrides,
                                         const std::vector<uint8_t> &hSrc, long4_16a inStrides, nvcv::Size2D src_size,
                                         nvcv::Size2D dst_size, nvcv::ImageFormat fmt, bool src_planar, bool dst_planar,
                                         const float borderValue, int flip_code, const NVCVRectI &cropRect,
                                         const std::vector<float> &hBase, int baseRowStride, nvcv::Size2D baseSize,
                                         nvcv::ImageFormat baseFormat, const std::vector<float> &hScale,
                                         int scaleRowStride, nvcv::Size2D scaleSize, nvcv::ImageFormat scaleFormat,
                                         const float globalScale, const float globalShift, const float epsilon,
                                         const uint32_t flags, int i)
{
    const int bi = baseSize.h == 1 ? 0 : i;
    const int si = scaleSize.h == 1 ? 0 : i;

    for (int j = 0; j < dst_size.w; j++)
    {
        if (i >= cropRect.height || j >= cropRect.width)
        {
            ClearCropDestination<T_Dst>(hDst, outStrides, dst_planar, fmt.numChannels(), j, i);
            continue;
        }

        const int bj    = baseSize.w == 1 ? 0 : j;
        const int sj    = scaleSize.w == 1 ? 0 : j;
        int2      coord = FlippedCropCoord(j, i, flip_code, cropRect);

        NormalizeCropPixel<T_Src, T_Dst, B>(hDst, outStrides, hSrc, inStrides, src_size, src_planar, dst_planar, fmt,
                                            borderValue, coord, hBase, baseRowStride, bi, bj, baseFormat, hScale,
                                            scaleRowStride, si, sj, scaleFormat, globalScale, globalShift, epsilon,
                                            flags, j, i);
    }
}

template<typename T_Src, typename T_Dst, NVCVBorderType B>
static void CropFlipNormalizeReformat(std::vector<uint8_t> &hDst, int, long4_16a outStrides, std::vector<uint8_t> &hSrc,
                                      int, long4_16a inStrides, nvcv::Size2D src_size, nvcv::Size2D dst_size,
                                      nvcv::ImageFormat fmt, nvcv::ImageFormat dst_fmt, const float borderValue,
                                      int flip_code, const NVCVRectI &cropRect, const std::vector<float> &hBase,
                                      int baseRowStride, nvcv::Size2D baseSize, nvcv::ImageFormat baseFormat,
                                      const std::vector<float> &hScale, int scaleRowStride, nvcv::Size2D scaleSize,
                                      nvcv::ImageFormat scaleFormat, const float globalScale, const float globalShift,
                                      const float epsilon, const uint32_t flags)
{
    bool src_planar = fmt.numPlanes() > 1;
    bool dst_planar = dst_fmt.numPlanes() > 1;

    for (int i = 0; i < dst_size.h; i++)
    {
        CropFlipNormalizeReformatRow<T_Src, T_Dst, B>(
            hDst, outStrides, hSrc, inStrides, src_size, dst_size, fmt, src_planar, dst_planar, borderValue, flip_code,
            cropRect, hBase, baseRowStride, baseSize, baseFormat, hScale, scaleRowStride, scaleSize, scaleFormat,
            globalScale, globalShift, epsilon, flags, i);
    }
}

constexpr uint32_t normalScale   = 0;
constexpr uint32_t scaleIsStdDev = CVCUDA_NORMALIZE_SCALE_IS_STDDEV;

template<typename T_Src, typename T_Dst, NVCVBorderType B>
void testCropFlipNormalizeReformatPad(int width, int height, int numImages, bool scalarBase, bool scalarScale,
                                      uint32_t flags, float globalScale, float globalShift, float epsilon,
                                      nvcv::ImageFormat fmt, nvcv::ImageFormat dst_fmt, NVCVBorderType borderMode)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    float borderValue = 1.f;

    nvcv::ImageFormat baseFormat  = (scalarBase ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);
    nvcv::ImageFormat scaleFormat = (scalarScale ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);
    int               numChannels = dst_fmt.numChannels();

    std::default_random_engine rng;

    // Create input varshape
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    std::vector<int>                  srcVecRowStride(numImages);

    int src_planes = fmt.numPlanes();
    int dst_planes = dst_fmt.numPlanes();

    int max_out_width  = 0;
    int max_out_height = 0;

    // create source images
    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);
        max_out_width  = std::max(max_out_width, imgSrc[i].size().w);
        max_out_height = std::max(max_out_height, imgSrc[i].size().h);

        int srcRowStride   = imgSrc[i].size().w * fmt.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> pixelDist(0, 255);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride * src_planes);

        for (int j = 0; j < imgSrc[i].size().h * imgSrc[i].size().w * numChannels; ++j)
        {
            reinterpret_cast<T_Src *>(srcVec[i].data())[j] = static_cast<T_Src>(pixelDist(rng));
        }

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(), srcRowStride,
                               srcRowStride, // vec has no padding
                               imgSrc[i].size().h * src_planes, cudaMemcpyHostToDevice));
    }

    // Create batch
    nvcv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create flip code tensor
    std::uniform_int_distribution udistflip(-1, 2);
    std::vector<int>              flip_vec(numImages);
    nvcv::Tensor                  flipCode({{numImages}, "N"}, nvcv::TYPE_S32);
    auto                          dev = flipCode.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);
    for (int i = 0; i < numImages; ++i)
    {
        flip_vec[i] = udistflip(rng);
    }
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(dev->basePtr(), flip_vec.data(), flip_vec.size() * sizeof(int), cudaMemcpyHostToDevice));

    // create crop param
    nvcv::Tensor cropRect(
        {
            {numImages, 1, 1, 4},
            nvcv::TENSOR_NHWC
    },
        nvcv::TYPE_S32);

    auto cropRectData = cropRect.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, cropRectData);
    auto cropRectAccess = nvcv::TensorDataAccessStridedImage::Create(*cropRectData);
    ASSERT_TRUE(cropRectAccess);
    std::vector<int> cropVec;
    for (int i = 0; i < numImages; i++)
    {
        std::uniform_int_distribution x_dist(-3, std::min(10, imgSrc[i].size().w / 10) + 1);
        std::uniform_int_distribution y_dist(-3, std::min(10, imgSrc[i].size().h / 10) + 1);
        std::uniform_int_distribution w_dist(std::max((int)(imgSrc[i].size().w * 0.8), imgSrc[i].size().w - 10),
                                             imgSrc[i].size().w - 1);
        std::uniform_int_distribution h_dist(std::max((int)(imgSrc[i].size().h * 0.8), imgSrc[i].size().h - 10),
                                             imgSrc[i].size().h - 1);

        std::vector<int> cropVecTmp = {x_dist(rng), y_dist(rng), w_dist(rng), h_dist(rng)};

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(cropRectAccess->sampleData(i), cropRectAccess->rowStride(),
                                            cropVecTmp.data(), cropVecTmp.size() * sizeof(int),
                                            cropVecTmp.size() * sizeof(int), // vec has no padding
                                            1, cudaMemcpyHostToDevice));
        cropVec.insert(cropVec.end(), cropVecTmp.begin(), cropVecTmp.end());
    }

    // Create base tensor
    nvcv::Tensor imgBase(
        {
            {1, 1, 1, baseFormat.numChannels()},
            nvcv::TENSOR_NHWC
    },
        baseFormat.planeDataType(0));
    std::vector<float> baseVec(baseFormat.numChannels());
    auto               baseData = imgBase.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, baseData);
    auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
    ASSERT_TRUE(baseAccess);

    std::uniform_real_distribution<float> baseDist(0, 255.f);
    std::ranges::generate(baseVec, [&baseDist, &rng]() { return baseDist(rng); });

    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(baseAccess->sampleData(0), baseAccess->rowStride(), baseVec.data(),
                                        baseVec.size() * sizeof(float),
                                        baseVec.size() * sizeof(float), // vec has no padding
                                        1, cudaMemcpyHostToDevice));

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

    // Create output tensor
    nvcv::Tensor imgDstTensor(numImages, {max_out_width, max_out_height}, dst_fmt);

    // Generate test result
    cvcuda::CropFlipNormalizeReformat CropFlipNormalizeReformatOp;

    EXPECT_NO_THROW(CropFlipNormalizeReformatOp(stream, batchSrc, imgDstTensor, cropRect, borderMode, borderValue,
                                                flipCode, imgBase, imgScale, globalScale, globalShift, epsilon, flags));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    int src_plane_ch_stride
        = fmt.numPlanes() == 1 ? (fmt.planePixelStrideBytes(0) / fmt.numChannels()) : fmt.planePixelStrideBytes(0);
    int dst_plane_ch_stride = dst_fmt.numPlanes() == 1 ? (dst_fmt.planePixelStrideBytes(0) / dst_fmt.numChannels())
                                                       : dst_fmt.planePixelStrideBytes(0);

    // Check test data against gold
    auto dstTensorData = imgDstTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstTensorData);
    auto dstTensorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstTensorData);

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        int src_width  = imgSrc[i].size().w;
        int src_height = imgSrc[i].size().h;

        int dst_width  = max_out_width;
        int dst_height = max_out_height;

        int dstRowStride = dst_width * dst_fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dst_height * dstRowStride * dst_planes);

        // Copy output data to Host

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), dstRowStride,
                                            dstTensorData->basePtr() + dstTensorAccess->sampleStride() * i,
                                            dstTensorAccess->rowStride(),
                                            dstRowStride, // vec has no padding
                                            dst_height * dst_planes, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dst_height * dstRowStride * dst_planes);

        NVCVRectI cropRectValue = {cropVec[4 * i], cropVec[4 * i + 1], cropVec[4 * i + 2], cropVec[4 * i + 3]};

        long4_16a inStrides;
        long4_16a outStrides;

        bool src_planar = fmt.numPlanes() > 1;
        bool dst_planar = dst_fmt.numPlanes() > 1;

        if (src_planar)
        {
            inStrides.x = src_width * src_height * numChannels * src_plane_ch_stride;
            inStrides.y = src_width * src_height * src_plane_ch_stride;
            inStrides.z = src_width * src_plane_ch_stride;
            inStrides.w = src_plane_ch_stride;
        }
        else
        {
            inStrides.x = src_width * src_height * numChannels * src_plane_ch_stride;
            inStrides.y = src_width * numChannels * src_plane_ch_stride;
            inStrides.z = numChannels * src_plane_ch_stride;
            inStrides.w = src_plane_ch_stride;
        }

        if (dst_planar)
        {
            outStrides.x = dst_width * dst_height * numChannels * dst_plane_ch_stride;
            outStrides.y = dst_width * dst_height * dst_plane_ch_stride;
            outStrides.z = dst_width * dst_plane_ch_stride;
            outStrides.w = dst_plane_ch_stride;
        }
        else
        {
            outStrides.x = dst_width * dst_height * numChannels * dst_plane_ch_stride;
            outStrides.y = dst_width * numChannels * dst_plane_ch_stride;
            outStrides.z = numChannels * dst_plane_ch_stride;
            outStrides.w = dst_plane_ch_stride;
        }

        // Generate gold result
        CropFlipNormalizeReformat<T_Src, T_Dst, B>(
            goldVec, dstRowStride, outStrides, srcVec[i], srcVecRowStride[i], inStrides, {src_width, src_height},
            {dst_width, dst_height}, fmt, dst_fmt, borderValue, flip_vec[i], cropRectValue, baseVec, 0, {1, 1},
            baseFormat, scaleVec, 0, {1, 1}, scaleFormat, globalScale, globalShift, epsilon, flags);

        // Compare test and gold with correct type
        std::vector<T_Dst> testVecTyped(dst_height * dst_width * numChannels);
        std::vector<T_Dst> goldVecTyped(dst_height * dst_width * numChannels);
        auto              *testData = reinterpret_cast<T_Dst *>(testVec.data());
        auto              *goldData = reinterpret_cast<T_Dst *>(goldVec.data());
        std::copy_n(testData, testVecTyped.size(), testVecTyped.begin());
        std::copy_n(goldData, goldVecTyped.size(), goldVecTyped.begin());

        VEC_EXPECT_NEAR(goldVecTyped, testVecTyped, 1e-4);
    }
}

#define NVCV_TEST_ROW(WIDTH, HEIGHT, IMAGES, SCALAR_BASE, SCALAR_SCALE, FLAGS, GLOBAL_SCALE, GLOBAL_SHIFT, EPS, \
                      SRC_FMT, DST_FMT, BORDERTYPE, SRC_TYPE, DST_TYPE)                                         \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<IMAGES>, ttype::Value<SCALAR_BASE>,    \
                 ttype::Value<SCALAR_SCALE>, ttype::Value<FLAGS>, ttype::Value<GLOBAL_SCALE>,                   \
                 ttype::Value<GLOBAL_SHIFT>, ttype::Value<EPS>, ttype::Value<SRC_FMT>, ttype::Value<DST_FMT>,   \
                 ttype::Value<BORDERTYPE>, SRC_TYPE, DST_TYPE>

NVCV_TYPED_TEST_SUITE(
    OpCropFlipNormalizeReformat,
    ttype::Types<NVCV_TEST_ROW(9, 13, 10, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_RGB8p,
                               NVCV_IMAGE_FORMAT_RGB8p, NVCV_BORDER_CONSTANT, uint8_t, uint8_t),
                 NVCV_TEST_ROW(9, 13, 10, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_RGB8p,
                               NVCV_IMAGE_FORMAT_RGB8, NVCV_BORDER_CONSTANT, uint8_t, uint8_t),
                 NVCV_TEST_ROW(9, 13, 10, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_RGB8,
                               NVCV_IMAGE_FORMAT_RGB8p, NVCV_BORDER_CONSTANT, uint8_t, uint8_t),
                 NVCV_TEST_ROW(9, 13, 10, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_RGBA8,
                               NVCV_IMAGE_FORMAT_RGBA8, NVCV_BORDER_CONSTANT, uint8_t, uint8_t),
                 NVCV_TEST_ROW(9, 13, 10, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_RGBA8p,
                               NVCV_IMAGE_FORMAT_RGBA8, NVCV_BORDER_CONSTANT, uint8_t, uint8_t),
                 NVCV_TEST_ROW(9, 13, 10, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_RGBA8,
                               NVCV_IMAGE_FORMAT_RGBA8, NVCV_BORDER_REFLECT, uint8_t, uint8_t),
                 NVCV_TEST_ROW(9, 13, 10, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_RGBA8,
                               NVCV_IMAGE_FORMAT_RGBA8, NVCV_BORDER_REFLECT101, uint8_t, uint8_t),
                 NVCV_TEST_ROW(10, 10, 2, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_RGB8,
                               NVCV_IMAGE_FORMAT_RGB8, NVCV_BORDER_WRAP, uint8_t, uint8_t),
                 NVCV_TEST_ROW(15, 15, 3, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_U8,
                               NVCV_IMAGE_FORMAT_U8, NVCV_BORDER_REPLICATE, uint8_t, uint8_t),
                 NVCV_TEST_ROW(9, 13, 10, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_RGBA8p,
                               NVCV_IMAGE_FORMAT_RGBAf32, NVCV_BORDER_CONSTANT, uint8_t, float),
                 NVCV_TEST_ROW(15, 15, 10, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_F32,
                               NVCV_IMAGE_FORMAT_F32, NVCV_BORDER_REPLICATE, float, float),
                 NVCV_TEST_ROW(15, 15, 3, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_U8,
                               NVCV_IMAGE_FORMAT_U8, NVCV_BORDER_REPLICATE, uint8_t, uint8_t),
                 NVCV_TEST_ROW(15, 15, 2, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_S32,
                               NVCV_IMAGE_FORMAT_F32, NVCV_BORDER_REPLICATE, int, float),
                 NVCV_TEST_ROW(15, 15, 2, true, true, normalScale, 1.f, 0.f, 0.f, NVCV_IMAGE_FORMAT_S32,
                               NVCV_IMAGE_FORMAT_S32, NVCV_BORDER_REPLICATE, int, int),
                 NVCV_TEST_ROW(51, 53, 10, true, false, normalScale, 1.234f, 33.21f, 0.f, NVCV_IMAGE_FORMAT_RGBf32,
                               NVCV_IMAGE_FORMAT_RGBf32, NVCV_BORDER_REPLICATE, float, float),
                 NVCV_TEST_ROW(51, 53, 10, true, true, scaleIsStdDev, 1.1f, 0.3f, 1.23f, NVCV_IMAGE_FORMAT_RGBAf32,
                               NVCV_IMAGE_FORMAT_RGBAf32p, NVCV_BORDER_REPLICATE, float, float),
                 NVCV_TEST_ROW(51, 53, 10, false, false, normalScale, 1.f, 12.3f, 0.f, NVCV_IMAGE_FORMAT_RGBAf32p,
                               NVCV_IMAGE_FORMAT_RGBAf32, NVCV_BORDER_REPLICATE, float, float),
                 NVCV_TEST_ROW(51, 53, 10, false, true, scaleIsStdDev, 1.1f, 0.3f, 1.23f, NVCV_IMAGE_FORMAT_RGBAf32,
                               NVCV_IMAGE_FORMAT_RGBAf32p, NVCV_BORDER_REPLICATE, float, float)>);
#undef NVCV_TEST_ROW

TYPED_TEST(OpCropFlipNormalizeReformat, correct_output)
{
    int               width       = ttype::GetValue<TypeParam, 0>;
    int               height      = ttype::GetValue<TypeParam, 1>;
    int               numImages   = ttype::GetValue<TypeParam, 2>;
    bool              scalarBase  = ttype::GetValue<TypeParam, 3>;
    bool              scalarScale = ttype::GetValue<TypeParam, 4>;
    uint32_t          flags       = ttype::GetValue<TypeParam, 5>;
    float             globalScale = ttype::GetValue<TypeParam, 6>;
    float             globalShift = ttype::GetValue<TypeParam, 7>;
    float             epsilon     = ttype::GetValue<TypeParam, 8>;
    nvcv::ImageFormat fmt         = nvcv::ImageFormat(ttype::GetValue<TypeParam, 9>);
    nvcv::ImageFormat dst_fmt     = nvcv::ImageFormat(ttype::GetValue<TypeParam, 10>);
    constexpr auto    borderMode  = ttype::GetValue<TypeParam, 11>;

    using InType  = typename ttype::GetType<TypeParam, 12>;
    using OutType = typename ttype::GetType<TypeParam, 13>;

    testCropFlipNormalizeReformatPad<InType, OutType, borderMode>(width, height, numImages, scalarBase, scalarScale,
                                                                  flags, globalScale, globalShift, epsilon, fmt,
                                                                  dst_fmt, borderMode);
}

TEST(OpCropFlipNormalizeReformatPlanar, varshape_matches_interleaved)
{
    constexpr int width     = 19;
    constexpr int height    = 13;
    constexpr int numImages = 2;
    constexpr int channels  = 3;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> srcInterleaved;
    std::vector<nvcv::Image> srcPlanar;
    for (int i = 0; i < numImages; ++i)
    {
        srcInterleaved.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
        srcPlanar.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8p);

        std::vector<uint8_t> hwc(width * height * channels);
        nvcv::test::planar::FillDeterministicValues(hwc, i * 101 + 17, nvcv::TYPE_U8);
        auto interleavedData = srcInterleaved.back().exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(interleavedData);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(interleavedData->plane(0).basePtr, interleavedData->plane(0).rowStride, hwc.data(),
                               width * channels, width * channels, height, cudaMemcpyHostToDevice));

        auto planes     = nvcv::test::planar::DeinterleaveToPlanes(hwc, width, height, channels, 1);
        auto planarData = srcPlanar.back().exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(planarData);
        for (int c = 0; c < channels; ++c)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(planarData->plane(c).basePtr, planarData->plane(c).rowStride,
                                   planes.data() + c * width * height, width, width, height, cudaMemcpyHostToDevice));
        }
    }

    nvcv::ImageBatchVarShape batchInterleaved(numImages);
    nvcv::ImageBatchVarShape batchPlanar(numImages);
    batchInterleaved.pushBack(srcInterleaved.begin(), srcInterleaved.end());
    batchPlanar.pushBack(srcPlanar.begin(), srcPlanar.end());

    nvcv::Tensor dstInterleaved(numImages, {width, height}, nvcv::FMT_RGB8);
    nvcv::Tensor dstPlanar(numImages, {width, height}, nvcv::FMT_RGB8p);
    nvcv::Tensor cropRect(
        {
            {numImages, 1, 1, 4},
            nvcv::TENSOR_NHWC
    },
        nvcv::TYPE_S32);
    auto cropData   = cropRect.exportData<nvcv::TensorDataStridedCuda>();
    auto cropAccess = nvcv::TensorDataAccessStridedImage::Create(*cropData);
    ASSERT_TRUE(cropAccess);
    const std::array<int, 4> fullCrop{0, 0, width, height};
    for (int i = 0; i < numImages; ++i)
    {
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(cropAccess->sampleData(i), fullCrop.data(), sizeof(fullCrop), cudaMemcpyHostToDevice));
    }

    nvcv::Tensor flipCode = nvcv::test::planar::MakePerImageTensor<int>(numImages, nvcv::TYPE_S32, 0);
    nvcv::Tensor base(
        {
            {1, 1, 1, 1},
            nvcv::TENSOR_NHWC
    },
        nvcv::TYPE_F32);
    nvcv::Tensor scale(
        {
            {1, 1, 1, 1},
            nvcv::TENSOR_NHWC
    },
        nvcv::TYPE_F32);
    nvcv::test::planar::UploadTensorValues(base, std::vector<float>{0.f});
    nvcv::test::planar::UploadTensorValues(scale, std::vector<float>{1.f});

    cvcuda::CropFlipNormalizeReformat op;
    EXPECT_NO_THROW(op(stream, batchInterleaved, dstInterleaved, cropRect, NVCV_BORDER_REPLICATE, 0.f, flipCode, base,
                       scale, 1.f, 0.f, 0.f, 0));
    EXPECT_NO_THROW(op(stream, batchPlanar, dstPlanar, cropRect, NVCV_BORDER_REPLICATE, 0.f, flipCode, base, scale, 1.f,
                       0.f, 0.f, 0));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto interleavedData   = dstInterleaved.exportData<nvcv::TensorDataStridedCuda>();
    auto planarData        = dstPlanar.exportData<nvcv::TensorDataStridedCuda>();
    auto interleavedAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*interleavedData);
    auto planarAccess      = nvcv::TensorDataAccessStridedImagePlanar::Create(*planarData);
    ASSERT_TRUE(interleavedAccess && planarAccess);
    for (int i = 0; i < numImages; ++i)
    {
        auto gpuInter
            = nvcv::test::planar::DownloadInterleavedSample(*interleavedAccess, i, width, height, width * channels);
        auto planesOut = nvcv::test::planar::DownloadPlanarSample(*planarAccess, i, width, height, channels, 1);
        EXPECT_EQ(gpuInter, nvcv::test::planar::InterleaveFromPlanes(planesOut, width, height, channels, 1));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCropFlipNormalizeReformat_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaCropFlipNormalizeReformatCreate(nullptr));
}

TEST(OpCropFlipNormalizeReformat_Negative, invalid_base_channels)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat        fmt = nvcv::FMT_RGB8; // 3 channels
    nvcv::Image              imgSrc(nvcv::Size2D{10, 10}, fmt);
    nvcv::ImageBatchVarShape batchSrc(1);
    batchSrc.pushBack(imgSrc);
    nvcv::Tensor imgDstTensor(1, {10, 10}, fmt);
    nvcv::Tensor flipCode({{1}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor cropRect(
        {
            {1, 1, 1, 4},
            nvcv::TENSOR_NHWC
    },
        nvcv::TYPE_S32);

    nvcv::Tensor imgBase(
        {
            {1, 1, 1, 3},
            nvcv::TENSOR_NHWC
    },
        nvcv::TYPE_F32); // 3 channels
    nvcv::Tensor imgBaseInvalid(
        {
            {1, 1, 1, 2},
            nvcv::TENSOR_NHWC
    },
        nvcv::TYPE_F32); // 2 channels < 3

    nvcv::Tensor imgScale(
        {
            {1, 1, 1, 3},
            nvcv::TENSOR_NHWC
    },
        nvcv::TYPE_F32); // 3 channels
    nvcv::Tensor imgScaleInvalid(
        {
            {1, 1, 1, 2},
            nvcv::TENSOR_NHWC
    },
        nvcv::TYPE_F32); // 2 channels < 3

    cvcuda::CropFlipNormalizeReformat op;
    EXPECT_THROW(op(stream, batchSrc, imgDstTensor, cropRect, NVCV_BORDER_CONSTANT, 0.0f, flipCode, imgBaseInvalid,
                    imgScale, 1.0f, 0.0f, 0.0f, 0),
                 nvcv::Exception);

    EXPECT_THROW(op(stream, batchSrc, imgDstTensor, cropRect, NVCV_BORDER_CONSTANT, 0.0f, flipCode, imgBase,
                    imgScaleInvalid, 1.0f, 0.0f, 0.0f, 0),
                 nvcv::Exception);

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

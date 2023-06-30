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

#include "Definitions.hpp"

#include <common/BorderUtils.hpp>
#include <common/InterpUtils.hpp>
#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpCropFlipNormalizeReformat.hpp>
#include <cvcuda/OpNormalize.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/SaturateCast.hpp>
#include <util/TensorDataUtils.hpp>

#include <cmath>
#include <iostream>
#include <random>

namespace gt    = ::testing;
namespace test  = nvcv::test;
namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

template<typename T_Src, typename T_Dst, NVCVBorderType B>
static void CropFlipNormalizeReformat(std::vector<uint8_t> &hDst, int dstRowStride, long4 outStrides,
                                      std::vector<uint8_t> &hSrc, int srcRowStride, long4 inStrides,
                                      nvcv::Size2D src_size, nvcv::Size2D dst_size, nvcv::ImageFormat fmt,
                                      nvcv::ImageFormat dst_fmt, const float borderValue, int flip_code,
                                      const NVCVRectI &cropRect, const std::vector<float> &hBase, int baseRowStride,
                                      nvcv::Size2D baseSize, nvcv::ImageFormat baseFormat,
                                      const std::vector<float> &hScale, int scaleRowStride, nvcv::Size2D scaleSize,
                                      nvcv::ImageFormat scaleFormat, const float globalScale, const float globalShift,
                                      const float epsilon, const uint32_t flags)
{
    bool src_planar = fmt.numPlanes() > 1;
    bool dst_planar = dst_fmt.numPlanes() > 1;

    for (int i = 0; i < dst_size.h; i++)
    {
        const int bi = baseSize.h == 1 ? 0 : i;
        const int si = scaleSize.h == 1 ? 0 : i;

        for (int j = 0; j < dst_size.w; j++)
        {
            const int bj = baseSize.w == 1 ? 0 : j;
            const int sj = scaleSize.w == 1 ? 0 : j;
            if (i >= cropRect.height || j >= cropRect.width)
            {
                for (int k = 0; k < fmt.numChannels(); k++)
                {
                    if (dst_planar)
                    {
                        test::ValueAt<T_Dst>(hDst, outStrides, int4{j, i, k, 0}) = 0;
                    }
                    else
                    {
                        test::ValueAt<T_Dst>(hDst, outStrides, int4{k, j, i, 0}) = 0;
                    }
                }
                continue;
            }
            for (int k = 0; k < fmt.numChannels(); k++)
            {
                const int bk = (baseFormat.numChannels() == 1 ? 0 : k);
                const int sk = (scaleFormat.numChannels() == 1 ? 0 : k);

                float mul;

                if (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
                {
                    float s = hScale.at(si * scaleRowStride + sj * scaleFormat.numChannels() + sk);
                    float x = s * s + epsilon;
                    mul     = float{1} / std::sqrt(x);
                }
                else
                {
                    mul = hScale.at(si * scaleRowStride + sj * scaleFormat.numChannels() + sk);
                }

                float base = hBase.at(bi * baseRowStride + bj * baseFormat.numChannels() + bk);

                int2 coord{j, i}, size{src_size.w, src_size.h};

                if (flip_code == 1)
                {
                    coord.x = cropRect.width - 1 - j + cropRect.x;
                    coord.y = i + cropRect.y;
                }
                else if (flip_code == 0)
                {
                    coord.x = j + cropRect.x;
                    coord.y = cropRect.height - 1 - i + cropRect.y;
                }
                else if (flip_code == -1)
                {
                    coord.x = cropRect.width - 1 - j + cropRect.x;
                    coord.y = cropRect.height - 1 - i + cropRect.y;
                }
                else
                {
                    coord.x = j + cropRect.x;
                    coord.y = i + cropRect.y;
                }

                T_Src out = 0;
                if (src_planar)
                {
                    out = test::ValueAt<B, T_Src>(hSrc, long4{inStrides.x, inStrides.z, inStrides.w, inStrides.y}, size,
                                                  borderValue, int4{k, coord.x, coord.y, 0});
                }
                else
                {
                    out = test::ValueAt<B, T_Src>(hSrc, long4{inStrides.x, inStrides.y, inStrides.z, inStrides.w}, size,
                                                  borderValue, int4{k, coord.x, coord.y, 0});
                }

                if (dst_planar)
                {
                    test::ValueAt<T_Dst>(hDst, outStrides, int4{j, i, k, 0})
                        = cuda::SaturateCast<T_Dst>((out - base) * mul * globalScale + globalShift);
                }
                else
                {
                    test::ValueAt<T_Dst>(hDst, outStrides, int4{k, j, i, 0})
                        = cuda::SaturateCast<T_Dst>((out - base) * mul * globalScale + globalShift);
                }
            }
        }
    }
}

constexpr uint32_t normalScale   = 0;
constexpr uint32_t scaleIsStdDev = CVCUDA_NORMALIZE_SCALE_IS_STDDEV;

template<typename T_Src, typename T_Dst, NVCVBorderType B>
const void testCropFlipNormalizeReformatPad(int width, int height, int numImages, bool scalarBase, bool scalarScale,
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
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

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

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride * src_planes);

        for (int j = 0; j < imgSrc[i].size().h * imgSrc[i].size().w * numChannels; ++j)
        {
            reinterpret_cast<T_Src *>(srcVec[i].data())[j] = static_cast<T_Src>(udist(rng));
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
    std::uniform_int_distribution<int> udistflip(-1, 2);
    std::vector<int>                   flip_vec(numImages);
    nvcv::Tensor                       flipCode({{numImages}, "N"}, nvcv::TYPE_S32);
    auto                               dev = flipCode.exportData<nvcv::TensorDataStridedCuda>();
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
        std::uniform_int_distribution<int> x_dist(-3, std::min(10, imgSrc[i].size().w / 10) + 1);
        std::uniform_int_distribution<int> y_dist(-3, std::min(10, imgSrc[i].size().h / 10) + 1);
        std::uniform_int_distribution<int> w_dist(std::max((int)(imgSrc[i].size().w * 0.8), imgSrc[i].size().w - 10),
                                                  imgSrc[i].size().w - 1);
        std::uniform_int_distribution<int> h_dist(std::max((int)(imgSrc[i].size().h * 0.8), imgSrc[i].size().h - 10),
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

    std::uniform_real_distribution<float> udist(0, 255.f);
    generate(baseVec.begin(), baseVec.end(), [&]() { return udist(rng); });

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

        std::uniform_real_distribution<float> udist(0, 1.f);
        generate(scaleVec.begin(), scaleVec.end(), [&]() { return udist(rng); });

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

        NVCVRectI cropRect = {cropVec[4 * i], cropVec[4 * i + 1], cropVec[4 * i + 2], cropVec[4 * i + 3]};

        long4 inStrides;
        long4 outStrides;

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
            {dst_width, dst_height}, fmt, dst_fmt, borderValue, flip_vec[i], cropRect, baseVec, 0, {1, 1}, baseFormat,
            scaleVec, 0, {1, 1}, scaleFormat, globalScale, globalShift, epsilon, flags);

        // Compare test and gold with correct type
        std::vector<T_Dst> testVecTyped(dst_height * dst_width * numChannels);
        std::vector<T_Dst> goldVecTyped(dst_height * dst_width * numChannels);
        for (size_t j = 0; j < testVecTyped.size(); ++j)
        {
            testVecTyped[j] = reinterpret_cast<T_Dst *>(testVec.data())[j];
            goldVecTyped[j] = reinterpret_cast<T_Dst *>(goldVec.data())[j];
        }

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

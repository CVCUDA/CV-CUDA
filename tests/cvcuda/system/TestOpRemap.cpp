/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpRemap.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>
#include <vector>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

static std::default_random_engine g_rng(std::random_device{}());

template<typename T>
using uniform_distribution
    = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

template<NVCVInterpolationType SI, NVCVInterpolationType MI, NVCVBorderType SB, typename ValueType>
void Remap(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, const std::vector<uint8_t> &map,
           const long3 &srcStrides, const long3 &dstStrides, const long3 &mapStrides, const int3 &srcShape,
           const int3 &dstShape, const int3 &mapShape, NVCVRemapMapValueType mapValueType, bool alignCorners,
           const float4 &borderValue)
{
    using BT = cuda::BaseType<ValueType>;

    constexpr NVCVBorderType MB = NVCV_BORDER_REPLICATE;

    const int2   mapSize{mapShape.x, mapShape.y}, srcSize{srcShape.x, srcShape.y}, dstSize{dstShape.x, dstShape.y};
    const float2 mapAS{0.f, 0.f}, srcAS{0.f, 0.f}, mapBV{0.f, 0.f}; // map area scale and border values are not used

    const ValueType srcBV = cuda::DropCast<cuda::NumElements<ValueType>>(cuda::StaticCast<BT>(borderValue));

    float2 srcScale, mapScale, valScale, srcOffset;
    float  dstOffset;

    if (mapValueType == NVCV_REMAP_ABSOLUTE)
    {
        srcScale  = float2{0.f, 0.f};
        mapScale  = cuda::StaticCast<float>(mapSize) / dstSize;
        valScale  = float2{1.f, 1.f};
        srcOffset = float2{0.f, 0.f};
        dstOffset = 0.f;
    }
    else if (mapValueType == NVCV_REMAP_ABSOLUTE_NORMALIZED)
    {
        srcScale  = float2{0.f, 0.f};
        mapScale  = cuda::StaticCast<float>(mapSize) / dstSize;
        valScale  = (srcSize - (alignCorners ? 1.f : 0.f)) / 2.f;
        srcOffset = valScale - (alignCorners ? 0.f : .5f);
        dstOffset = 0.f;
    }
    else
    {
        ASSERT_EQ(mapValueType, NVCV_REMAP_RELATIVE_NORMALIZED);

        srcScale  = cuda::StaticCast<float>(srcSize) / dstSize;
        mapScale  = (mapSize - 1.f) / dstSize;
        valScale  = srcSize - 1.f;
        dstOffset = alignCorners ? 0.f : .5f;
        srcOffset = srcScale * dstOffset - dstOffset;
    }

    ASSERT_EQ(srcShape.z, dstShape.z);
    ASSERT_TRUE(mapShape.z == 1 || mapShape.z == dstShape.z);

    for (int z = 0; z < dstShape.z; ++z)
    {
        int mapZ = mapShape.z == 1 ? 0 : z;

        for (int y = 0; y < dstShape.y; ++y)
        {
            for (int x = 0; x < dstShape.x; ++x)
            {
                int3 dstCoord{x, y, z};

                float2 fdstCoord{static_cast<float>(x), static_cast<float>(y)};

                float2 mapCoord = (fdstCoord + dstOffset) * mapScale;

                float2 mapValue = test::GoldInterp<MI, MB>(map, mapStrides, mapSize, mapBV, mapAS, mapCoord, mapZ);

                float2 srcCoord = fdstCoord * srcScale + mapValue * valScale + srcOffset;

                test::ValueAt<ValueType>(dst, dstStrides, dstCoord)
                    = test::GoldInterp<SI, SB>(src, srcStrides, srcSize, srcBV, srcAS, srcCoord, z);
            }
        }
    }
}

// clang-format off

#define NVCV_SHAPE(w, h, n) (int3{w, h, n})

#define NVCV_TEST_ROW(SrcShape, DstShape, MapShape, ValueType, ImgFormat, AlignCorners, SrcInterp, MapInterp, \
                      MapValueType, BorderType, BorderValue)                                                  \
    ttype::Types<ttype::Value<SrcShape>, ttype::Value<DstShape>, ttype::Value<MapShape>, ValueType,           \
                 ttype::Value<ImgFormat>, ttype::Value<AlignCorners>, ttype::Value<SrcInterp>,                \
                 ttype::Value<MapInterp>, ttype::Value<MapValueType>, ttype::Value<BorderType>,               \
                 ttype::Value<BorderValue>>

NVCV_TYPED_TEST_SUITE(
    OpRemap, ttype::Types<
    NVCV_TEST_ROW(NVCV_SHAPE(42, 42, 1), NVCV_SHAPE(42, 42, 1), NVCV_SHAPE(2, 2, 1), float1, NVCV_IMAGE_FORMAT_F32,
                  false, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, NVCV_REMAP_RELATIVE_NORMALIZED, NVCV_BORDER_CONSTANT, 123.f),
    NVCV_TEST_ROW(NVCV_SHAPE(42, 42, 1), NVCV_SHAPE(42, 42, 1), NVCV_SHAPE(42, 42, 1), float1, NVCV_IMAGE_FORMAT_F32,
                  true, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, NVCV_REMAP_ABSOLUTE, NVCV_BORDER_CONSTANT, 72.f),
    NVCV_TEST_ROW(NVCV_SHAPE(41, 31, 2), NVCV_SHAPE(41, 31, 2), NVCV_SHAPE(2, 2, 1), uchar3, NVCV_IMAGE_FORMAT_RGB8,
                  false, NVCV_INTERP_NEAREST, NVCV_INTERP_LINEAR, NVCV_REMAP_ABSOLUTE_NORMALIZED, NVCV_BORDER_REPLICATE, 0.f),
    NVCV_TEST_ROW(NVCV_SHAPE(44, 33, 2), NVCV_SHAPE(44, 33, 2), NVCV_SHAPE(1, 1, 2), uchar3, NVCV_IMAGE_FORMAT_RGB8,
                  false, NVCV_INTERP_NEAREST, NVCV_INTERP_LINEAR, NVCV_REMAP_RELATIVE_NORMALIZED, NVCV_BORDER_REPLICATE, 0.f),
    NVCV_TEST_ROW(NVCV_SHAPE(44, 32, 2), NVCV_SHAPE(44, 32, 2), NVCV_SHAPE(4, 3, 2), uchar3, NVCV_IMAGE_FORMAT_RGB8,
                  true, NVCV_INTERP_NEAREST, NVCV_INTERP_LINEAR, NVCV_REMAP_ABSOLUTE, NVCV_BORDER_REPLICATE, 0.f),
    NVCV_TEST_ROW(NVCV_SHAPE(23, 13, 3), NVCV_SHAPE(25, 27, 3), NVCV_SHAPE(23, 13, 3), uchar4, NVCV_IMAGE_FORMAT_RGBA8,
                  false, NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR, NVCV_REMAP_ABSOLUTE_NORMALIZED, NVCV_BORDER_REFLECT, 0.f),
    NVCV_TEST_ROW(NVCV_SHAPE(23, 13, 3), NVCV_SHAPE(25, 27, 3), NVCV_SHAPE(25, 27, 3), uchar4, NVCV_IMAGE_FORMAT_RGBA8,
                  true, NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR, NVCV_REMAP_ABSOLUTE_NORMALIZED, NVCV_BORDER_REFLECT, 0.f),
    NVCV_TEST_ROW(NVCV_SHAPE(51, 62, 2), NVCV_SHAPE(12, 38, 2), NVCV_SHAPE(24, 76, 2), uchar1, NVCV_IMAGE_FORMAT_U8,
                  false, NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_REMAP_ABSOLUTE, NVCV_BORDER_WRAP, 0.f),
    NVCV_TEST_ROW(NVCV_SHAPE(51, 62, 2), NVCV_SHAPE(12, 38, 2), NVCV_SHAPE(24, 76, 1), uchar1, NVCV_IMAGE_FORMAT_U8,
                  false, NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_REMAP_ABSOLUTE, NVCV_BORDER_WRAP, 0.f),
    NVCV_TEST_ROW(NVCV_SHAPE(51, 62, 2), NVCV_SHAPE(12, 38, 2), NVCV_SHAPE(12, 38, 2), uchar1, NVCV_IMAGE_FORMAT_U8,
                  true, NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_REMAP_ABSOLUTE, NVCV_BORDER_WRAP, 0.f),
    NVCV_TEST_ROW(NVCV_SHAPE(16, 17, 3), NVCV_SHAPE(18, 12, 3), NVCV_SHAPE(18, 12, 3), uchar1, NVCV_IMAGE_FORMAT_Y8,
                  false, NVCV_INTERP_CUBIC, NVCV_INTERP_CUBIC, NVCV_REMAP_ABSOLUTE, NVCV_BORDER_REFLECT101, 0.f),
    NVCV_TEST_ROW(NVCV_SHAPE(16, 17, 3), NVCV_SHAPE(18, 12, 3), NVCV_SHAPE(36, 24, 3), uchar1, NVCV_IMAGE_FORMAT_Y8,
                  true, NVCV_INTERP_CUBIC, NVCV_INTERP_CUBIC, NVCV_REMAP_RELATIVE_NORMALIZED, NVCV_BORDER_REFLECT101, 0.f)
>);

// clang-format on

TYPED_TEST(OpRemap, correct_output)
{
    const int3 srcShape = ttype::GetValue<TypeParam, 0>;
    const int3 dstShape = ttype::GetValue<TypeParam, 1>;
    const int3 mapShape = ttype::GetValue<TypeParam, 2>;

    using ValueType = ttype::GetType<TypeParam, 3>;
    using BT        = cuda::BaseType<ValueType>;

    const nvcv::ImageFormat imgFormat{ttype::GetValue<TypeParam, 4>};

    const bool kAlignCorners = ttype::GetValue<TypeParam, 5>;

    constexpr NVCVInterpolationType kSrcInterp = ttype::GetValue<TypeParam, 6>;
    constexpr NVCVInterpolationType kMapInterp = ttype::GetValue<TypeParam, 7>;

    const NVCVRemapMapValueType kMapValueType = ttype::GetValue<TypeParam, 8>;

    constexpr NVCVBorderType kBorderType = ttype::GetValue<TypeParam, 9>;

    const float4 borderValue = nvcv::cuda::SetAll<float4>(ttype::GetValue<TypeParam, 10>);

    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(srcShape.z, srcShape.x, srcShape.y, imgFormat);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(dstShape.z, dstShape.x, dstShape.y, imgFormat);
    nvcv::Tensor mapTensor = nvcv::util::CreateTensor(mapShape.z, mapShape.x, mapShape.y, nvcv::FMT_2F32);

    auto srcData = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dstTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto mapData = mapTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData && mapData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    auto mapAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*mapData);
    ASSERT_TRUE(srcAccess && dstAccess && mapAccess);

    long3 srcStrides{srcAccess->sampleStride(), srcAccess->rowStride(), srcAccess->colStride()};
    long3 dstStrides{dstAccess->sampleStride(), dstAccess->rowStride(), dstAccess->colStride()};
    long3 mapStrides{mapAccess->sampleStride(), mapAccess->rowStride(), mapAccess->colStride()};

    srcStrides.x = (srcData->rank() == 3) ? srcAccess->numRows() * srcAccess->rowStride() : srcStrides.x;
    dstStrides.x = (dstData->rank() == 3) ? dstAccess->numRows() * dstAccess->rowStride() : dstStrides.x;
    mapStrides.x = (mapData->rank() == 3) ? mapAccess->numRows() * mapAccess->rowStride() : mapStrides.x;

    size_t srcBufSize = srcStrides.x * srcAccess->numSamples();
    size_t dstBufSize = dstStrides.x * dstAccess->numSamples();
    size_t mapBufSize = mapStrides.x * mapAccess->numSamples();

    std::vector<uint8_t> srcVec(srcBufSize, uint8_t{0});
    std::vector<uint8_t> dstVec(dstBufSize, uint8_t{0});
    std::vector<uint8_t> mapVec(mapBufSize, uint8_t{0});
    std::vector<uint8_t> refVec(dstBufSize, uint8_t{0});

    uniform_distribution<BT> rand(BT{0}, std::is_integral_v<BT> ? cuda::TypeTraits<BT>::max : BT{1});

    for (int z = 0; z < srcShape.z; ++z)
        for (int y = 0; y < srcShape.y; ++y)
            for (int x = 0; x < srcShape.x; ++x)
                for (int k = 0; k < cuda::NumElements<ValueType>; ++k)
                    cuda::GetElement(test::ValueAt<ValueType>(srcVec, srcStrides, int3{x, y, z}), k) = rand(g_rng);

    std::uniform_real_distribution<float> randf(-1.f, 1.f);

    for (int z = 0; z < mapShape.z; ++z)
        for (int y = 0; y < mapShape.y; ++y)
            for (int x = 0; x < mapShape.x; ++x)
                test::ValueAt<float2>(mapVec, mapStrides, int3{x, y, z}) = float2{randf(g_rng), randf(g_rng)};

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), srcBufSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(mapData->basePtr(), mapVec.data(), mapBufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Remap op;
    EXPECT_NO_THROW(op(stream, srcTensor, dstTensor, mapTensor, kSrcInterp, kMapInterp, kMapValueType, kAlignCorners,
                       kBorderType, borderValue));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstVec.data(), dstData->basePtr(), dstBufSize, cudaMemcpyDeviceToHost));

    Remap<kSrcInterp, kMapInterp, kBorderType, ValueType>(srcVec, refVec, mapVec, srcStrides, dstStrides, mapStrides,
                                                          srcShape, dstShape, mapShape, kMapValueType, kAlignCorners,
                                                          borderValue);

    VEC_EXPECT_NEAR(dstVec, refVec, 1);
}

TYPED_TEST(OpRemap, varshape_correct_output)
{
    const int3 srcShape = ttype::GetValue<TypeParam, 0>;
    const int3 dstShape = ttype::GetValue<TypeParam, 1>;
    const int3 mapShape = ttype::GetValue<TypeParam, 2>;

    using ValueType = ttype::GetType<TypeParam, 3>;
    using BT        = cuda::BaseType<ValueType>;

    const nvcv::ImageFormat imgFormat{ttype::GetValue<TypeParam, 4>};

    const bool kAlignCorners = ttype::GetValue<TypeParam, 5>;

    constexpr NVCVInterpolationType kSrcInterp = ttype::GetValue<TypeParam, 6>;
    constexpr NVCVInterpolationType kMapInterp = ttype::GetValue<TypeParam, 7>;

    const NVCVRemapMapValueType kMapValueType = ttype::GetValue<TypeParam, 8>;

    constexpr NVCVBorderType kBorderType = ttype::GetValue<TypeParam, 9>;

    const float4 borderValue = nvcv::cuda::SetAll<float4>(ttype::GetValue<TypeParam, 10>);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(srcShape.z);

    std::uniform_int_distribution<int> srcRandW(srcShape.x * 0.8, srcShape.x * 1.2);
    std::uniform_int_distribution<int> srcRandH(srcShape.y * 0.8, srcShape.y * 1.2);

    uniform_distribution<BT> rand(BT{0}, std::is_integral_v<BT> ? cuda::TypeTraits<BT>::max : BT{1});

    ASSERT_EQ(sizeof(ValueType), imgFormat.planePixelStrideBytes(0));

    for (int z = 0; z < srcShape.z; ++z)
    {
        imgSrc.emplace_back(nvcv::Size2D{srcRandW(g_rng), srcRandH(g_rng)}, imgFormat);

        auto imgData = imgSrc[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        int   srcRowStride = imgData->plane(0).rowStride;
        long2 srcStrides   = long2{srcRowStride, sizeof(ValueType)};

        srcVec[z].resize(srcRowStride * imgSrc[z].size().h);

        for (int y = 0; y < imgSrc[z].size().h; ++y)
            for (int x = 0; x < imgSrc[z].size().w; ++x)
                for (int k = 0; k < cuda::NumElements<ValueType>; ++k)
                    cuda::GetElement(test::ValueAt<ValueType>(srcVec[z], srcStrides, int2{x, y}), k) = rand(g_rng);

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(imgData->plane(0).basePtr, srcRowStride, srcVec[z].data(), srcRowStride,
                                            srcRowStride, imgSrc[z].size().h, cudaMemcpyHostToDevice));
    }

    std::uniform_int_distribution<int> dstRandW(dstShape.x * 0.8, dstShape.x * 1.2);
    std::uniform_int_distribution<int> dstRandH(dstShape.y * 0.8, dstShape.y * 1.2);

    nvcv::ImageBatchVarShape batchSrc(srcShape.z);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    std::vector<nvcv::Image> imgDst;
    for (int z = 0; z < dstShape.z; ++z)
    {
        imgDst.emplace_back(nvcv::Size2D{dstRandW(g_rng), dstRandH(g_rng)}, imgFormat);
    }
    nvcv::ImageBatchVarShape batchDst(dstShape.z);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor mapTensor = nvcv::util::CreateTensor(mapShape.z, mapShape.x, mapShape.y, nvcv::FMT_2F32);

    auto mapData = mapTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(mapData);

    auto mapAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*mapData);
    ASSERT_TRUE(mapAccess);

    long3 mapStrides = long3{mapAccess->sampleStride(), mapAccess->rowStride(), mapAccess->colStride()};

    mapStrides.x = (mapData->rank() == 3) ? mapAccess->numRows() * mapAccess->rowStride() : mapStrides.x;

    size_t mapBufSize = mapStrides.x * mapAccess->numSamples();

    std::vector<uint8_t> mapVec(mapBufSize, uint8_t{0});

    std::uniform_real_distribution<float> randf(-1.f, 1.f);

    for (int z = 0; z < mapShape.z; ++z)
        for (int y = 0; y < mapShape.y; ++y)
            for (int x = 0; x < mapShape.x; ++x)
                test::ValueAt<float2>(mapVec, mapStrides, int3{x, y, z}) = float2{randf(g_rng), randf(g_rng)};

    ASSERT_EQ(cudaSuccess, cudaMemcpy(mapData->basePtr(), mapVec.data(), mapBufSize, cudaMemcpyHostToDevice));

    cvcuda::Remap op;
    EXPECT_NO_THROW(op(stream, batchSrc, batchDst, mapTensor, kSrcInterp, kMapInterp, kMapValueType, kAlignCorners,
                       kBorderType, borderValue));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int z = 0; z < srcShape.z; ++z)
    {
        SCOPED_TRACE(z);

        const auto srcData = imgSrc[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = imgDst[z].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        long3 srcStrides{0, srcData->plane(0).rowStride, sizeof(ValueType)};
        long3 dstStrides{0, dstData->plane(0).rowStride, sizeof(ValueType)};

        int3 srcShape2{srcData->plane(0).width, srcData->plane(0).height, 1};
        int3 dstShape2{dstData->plane(0).width, dstData->plane(0).height, 1};
        int3 mapShape2{mapShape.x, mapShape.y, 1};

        std::vector<uint8_t> dstVec(dstShape2.y * dstStrides.y);
        std::vector<uint8_t> refVec(dstShape2.y * dstStrides.y);

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(dstVec.data(), dstStrides.y, dstData->plane(0).basePtr, dstStrides.y,
                                            dstStrides.y, dstShape2.y, cudaMemcpyDeviceToHost));

        int  mapZ    = mapShape.z == 1 ? 0 : z;
        auto mapZbeg = mapVec.begin() + mapZ * mapStrides.x;
        auto mapZend = mapZ == mapShape.z - 1 ? mapVec.end() : mapVec.begin() + (mapZ + 1) * mapStrides.x;

        std::vector<uint8_t> mapVec2(mapZbeg, mapZend);

        Remap<kSrcInterp, kMapInterp, kBorderType, ValueType>(srcVec[z], refVec, mapVec2, srcStrides, dstStrides,
                                                              mapStrides, srcShape2, dstShape2, mapShape2,
                                                              kMapValueType, kAlignCorners, borderValue);

        VEC_EXPECT_NEAR(dstVec, refVec, 1);
    }
}

#define NVCV_IMAGE_FORMAT_INVALID_MAP NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, FLOAT, XYZW, ASSOCIATED, X32_Y32, X32_Y32)

// clang-format off
NVCV_TEST_SUITE_P(OpRemap_Negative, test::ValueList<int, int, int, int, int, int, int, int, int, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, NVCVInterpolationType, NVCVInterpolationType, NVCVBorderType>{
    // inputShape, dstShape, mapShape, inputFormat, outputFormat, mapFormat, inputInterp, mapInterp, borderValue
    {42, 42, 1, 42, 42, 1, 2, 2, 1, nvcv::FMT_RGB8, nvcv::FMT_U8, nvcv::FMT_2F32, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT},
    {42, 42, 2, 42, 42, 1, 2, 2, 1, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_2F32, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT},
    {42, 42, 2, 42, 42, 2, 2, 2, 5, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_2F32, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT},
    {42, 42, 1, 42, 42, 1, 2, 2, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, nvcv::FMT_2F32, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT},
    {42, 42, 1, 42, 42, 1, 2, 2, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, nvcv::FMT_2F32, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT},
    {42, 42, 1, 42, 42, 1, 2, 2, 1, nvcv::FMT_RGB8, nvcv::FMT_RGBf32, nvcv::FMT_2F32, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT},
    {42, 42, 1, 42, 42, 1, 2, 2, 1, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_RGBf32, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT},
    {42, 42, 1, 42, 42, 1, 2, 2, 1, nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, nvcv::FMT_2F32, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT},
#ifndef ENABLE_SANITIZER
    {42, 42, 1, 42, 42, 1, 2, 2, 1, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_2F32, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST, static_cast<NVCVBorderType>(255)},
    {42, 42, 1, 42, 42, 1, 2, 2, 1, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_2F32, NVCV_INTERP_NEAREST, static_cast<NVCVInterpolationType>(255), NVCV_BORDER_CONSTANT},
    {42, 42, 1, 42, 42, 1, 2, 2, 1, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_2F32, static_cast<NVCVInterpolationType>(255), NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT},
#endif
});

NVCV_TEST_SUITE_P(OpRemapVarshape_Negative, test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat>{
    // inputNumImages, outputNumImages, mapNumSamples, inputFormat, outputFormat, mapFormat
    {2, 1, 1, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_2F32},
    {1, 1, 2, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_2F32},
    {1, 1, 1, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_RGBf32},
    {1, 1, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, nvcv::FMT_2F32},
    {1, 1, 1, nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, nvcv::FMT_2F32},
});

// clang-format on

TEST_P(OpRemap_Negative, op)
{
    const int3 srcShape = {GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>()};
    const int3 dstShape = {GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>()};
    const int3 mapShape = {GetParamValue<6>(), GetParamValue<7>(), GetParamValue<8>()};

    const nvcv::ImageFormat inputFmt  = GetParamValue<9>();
    const nvcv::ImageFormat outputFmt = GetParamValue<10>();
    const nvcv::ImageFormat mapFmt    = GetParamValue<11>();

    const NVCVRemapMapValueType kMapValueType = NVCV_REMAP_RELATIVE_NORMALIZED;
    const NVCVInterpolationType kSrcInterp    = GetParamValue<12>();
    const NVCVInterpolationType kMapInterp    = GetParamValue<13>();
    const NVCVBorderType        kBorderType   = GetParamValue<14>();

    const float4 borderValue = nvcv::cuda::SetAll<float4>(0.f);

    const bool kAlignCorners = false;

    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(srcShape.z, srcShape.x, srcShape.y, inputFmt);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(dstShape.z, dstShape.x, dstShape.y, outputFmt);
    nvcv::Tensor mapTensor = nvcv::util::CreateTensor(mapShape.z, mapShape.x, mapShape.y, mapFmt);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Remap op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&]
                                               {
                                                   op(stream, srcTensor, dstTensor, mapTensor, kSrcInterp, kMapInterp,
                                                      kMapValueType, kAlignCorners, kBorderType, borderValue);
                                               }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpRemapVarshape_Negative, op)
{
    const int inputNumImages  = GetParamValue<0>();
    const int outputNumImages = GetParamValue<1>();
    const int mapNumSamples   = GetParamValue<2>();

    const nvcv::ImageFormat inputFmt  = GetParamValue<3>();
    const nvcv::ImageFormat outputFmt = GetParamValue<4>();
    const nvcv::ImageFormat mapFmt    = GetParamValue<5>();

    const int3 srcShape = {24, 24, inputNumImages};
    const int3 dstShape = {24, 24, outputNumImages};
    const int3 mapShape = {24, 24, mapNumSamples};

    const bool kAlignCorners = false;

    constexpr NVCVInterpolationType kSrcInterp    = NVCV_INTERP_NEAREST;
    constexpr NVCVInterpolationType kMapInterp    = NVCV_INTERP_NEAREST;
    const NVCVRemapMapValueType     kMapValueType = NVCV_REMAP_RELATIVE_NORMALIZED;
    constexpr NVCVBorderType        kBorderType   = NVCV_BORDER_CONSTANT;

    const float4 borderValue = nvcv::cuda::SetAll<float4>(0.f);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> imgSrc;

    std::uniform_int_distribution<int> srcRandW(srcShape.x * 0.8, srcShape.x * 1.2);
    std::uniform_int_distribution<int> srcRandH(srcShape.y * 0.8, srcShape.y * 1.2);

    for (int z = 0; z < srcShape.z; ++z)
    {
        imgSrc.emplace_back(nvcv::Size2D{srcRandW(g_rng), srcRandH(g_rng)}, inputFmt);
    }

    std::uniform_int_distribution<int> dstRandW(dstShape.x * 0.8, dstShape.x * 1.2);
    std::uniform_int_distribution<int> dstRandH(dstShape.y * 0.8, dstShape.y * 1.2);

    nvcv::ImageBatchVarShape batchSrc(srcShape.z);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    std::vector<nvcv::Image> imgDst;
    for (int z = 0; z < dstShape.z; ++z)
    {
        imgDst.emplace_back(nvcv::Size2D{dstRandW(g_rng), dstRandH(g_rng)}, outputFmt);
    }
    nvcv::ImageBatchVarShape batchDst(dstShape.z);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor mapTensor = nvcv::util::CreateTensor(mapShape.z, mapShape.x, mapShape.y, mapFmt);

    cvcuda::Remap op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&]
                                               {
                                                   op(stream, batchSrc, batchDst, mapTensor, kSrcInterp, kMapInterp,
                                                      kMapValueType, kAlignCorners, kBorderType, borderValue);
                                               }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpRemap_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat     fmt          = nvcv::FMT_RGB8;
    NVCVInterpolationType srcInterp    = NVCV_INTERP_LINEAR;
    NVCVInterpolationType mapInterp    = NVCV_INTERP_LINEAR;
    NVCVBorderType        borderType   = NVCV_BORDER_CONSTANT;
    float4                borderValue  = nvcv::cuda::SetAll<float4>(0.f);
    bool                  alignCorners = false;
    NVCVRemapMapValueType mapValueType = NVCV_REMAP_RELATIVE_NORMALIZED;

    int numberOfImages = 3;
    int srcWidthBase   = 42;
    int srcHeightBase  = 42;
    int dstWidthBase   = 42;
    int dstHeightBase  = 42;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_RGBA8,             fmt},
        {            fmt, nvcv::FMT_RGBA8}
    };

    for (auto testCase : testSet)
    {
        nvcv::ImageFormat inputFmtExtra  = std::get<0>(testCase);
        nvcv::ImageFormat outputFmtExtra = std::get<1>(testCase);

        // Create input and output
        std::default_random_engine         randEng;
        std::uniform_int_distribution<int> rndSrcWidth(srcWidthBase * 0.8, srcWidthBase * 1.1);
        std::uniform_int_distribution<int> rndSrcHeight(srcHeightBase * 0.8, srcHeightBase * 1.1);
        std::uniform_int_distribution<int> rndDstWidth(dstWidthBase * 0.8, dstWidthBase * 1.1);
        std::uniform_int_distribution<int> rndDstHeight(dstHeightBase * 0.8, dstHeightBase * 1.1);

        std::vector<nvcv::Image>  imgSrc, imgDst;
        std::vector<nvcv::Size2D> srcSizes, dstSizes;

        // Create n-1 images with standard format
        for (int i = 0; i < numberOfImages - 1; ++i)
        {
            int tmpSrcWidth  = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
            int tmpSrcHeight = i == 0 ? srcHeightBase : rndSrcHeight(randEng);
            int tmpDstWidth  = i == 0 ? dstWidthBase : rndDstWidth(randEng);
            int tmpDstHeight = i == 0 ? dstHeightBase : rndDstHeight(randEng);

            imgSrc.emplace_back(nvcv::Size2D{tmpSrcWidth, tmpSrcHeight}, fmt);
            imgDst.emplace_back(nvcv::Size2D{tmpDstWidth, tmpDstHeight}, fmt);
            srcSizes.emplace_back(imgSrc.back().size());
            dstSizes.emplace_back(imgDst.back().size());
        }

        // Add the last image with different format
        imgSrc.emplace_back(nvcv::Size2D{srcWidthBase, srcHeightBase}, inputFmtExtra);
        imgDst.emplace_back(nvcv::Size2D{dstWidthBase, dstHeightBase}, outputFmtExtra);
        srcSizes.emplace_back(imgSrc.back().size());
        dstSizes.emplace_back(imgDst.back().size());

        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchDst(numberOfImages);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Create map tensor with fixed size
        nvcv::Tensor mapTensor = nvcv::util::CreateTensor(2, 2, 1, nvcv::FMT_2F32);

        // Generate test result
        cvcuda::Remap remapOp;

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                                   [&]
                                                   {
                                                       remapOp(stream, batchSrc, batchDst, mapTensor, srcInterp,
                                                               mapInterp, mapValueType, alignCorners, borderType,
                                                               borderValue);
                                                   }));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOpRemap_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaRemapCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}

#undef NVCV_IMAGE_FORMAT_INVALID_MAP

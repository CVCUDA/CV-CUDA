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

#include <common/HashUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/Allocator.hpp>

#include <array>
#include <cstddef>
#include <list>
#include <random>
#include <vector>

#include <nvcv/Fwd.hpp>

namespace t    = ::testing;
namespace test = nvcv::test;

class TensorImageTests
    : public t::TestWithParam<std::tuple<test::Param<"numImages", int>, test::Param<"width", int>,
                                         test::Param<"height", int>, test::Param<"format", nvcv::ImageFormat>,
                                         test::Param<"shape", nvcv::TensorShape>, test::Param<"dtype", nvcv::DataType>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, TensorImageTests,
    test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::TensorShape, nvcv::DataType>
    {
        {53, 32, 16, nvcv::FMT_RGBA8p, nvcv::TensorShape{{53, 4, 16, 32},nvcv::TENSOR_NCHW} , nvcv::TYPE_U8},
        {14, 64, 18, nvcv::FMT_RGB8, nvcv::TensorShape{{14, 18, 64, 3},nvcv::TENSOR_NHWC}, nvcv::TYPE_U8}
    }
);

// clang-format on

TEST_P(TensorImageTests, smoke_create)
{
    const int               PARAM_NUM_IMAGES = ::nvcv::test::ParamValue(std::get<0>(GetParam()));
    const int               PARAM_WIDTH      = ::nvcv::test::ParamValue(std::get<1>(GetParam()));
    const int               PARAM_HEIGHT     = ::nvcv::test::ParamValue(std::get<2>(GetParam()));
    const nvcv::ImageFormat PARAM_FORMAT     = ::nvcv::test::ParamValue(std::get<3>(GetParam()));
    const nvcv::TensorShape GOLD_SHAPE       = ::nvcv::test::ParamValue(std::get<4>(GetParam()));
    const nvcv::DataType    GOLD_DTYPE       = ::nvcv::test::ParamValue(std::get<5>(GetParam()));
    const int               GOLD_RANK        = 4;

    nvcv::Tensor tensor(PARAM_NUM_IMAGES, {PARAM_WIDTH, PARAM_HEIGHT}, PARAM_FORMAT);

    EXPECT_EQ(GOLD_DTYPE, tensor.dtype());
    EXPECT_EQ(GOLD_SHAPE, tensor.shape());
    EXPECT_EQ(GOLD_RANK, tensor.rank());
    EXPECT_EQ(GOLD_SHAPE.layout(), tensor.layout());
    ASSERT_NE(nullptr, tensor.handle());

    {
        auto data = tensor.exportData();
        ASSERT_EQ(tensor.dtype(), data.dtype());

        auto devdata = data.cast<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nvcv::NullOpt, devdata);

        EXPECT_EQ(GOLD_RANK, devdata->rank());
        ASSERT_EQ(GOLD_SHAPE, devdata->shape());
        ASSERT_EQ(GOLD_SHAPE.layout(), devdata->layout());
        ASSERT_EQ(GOLD_DTYPE, devdata->dtype());

        auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*devdata);
        ASSERT_TRUE(access);

        EXPECT_EQ(access->sampleStride(), devdata->stride(0));
        EXPECT_EQ(access->planeStride(), access->infoLayout().isChannelFirst() ? devdata->stride(1) : 0);
        EXPECT_EQ(access->numSamples(), devdata->shape(0));

        // Write data to each plane
        for (int i = 0; i < access->numSamples(); ++i)
        {
            nvcv::Byte *sampleBuffer = access->sampleData(i);
            for (int p = 0; p < access->numPlanes(); ++p)
            {
                nvcv::Byte *planeBuffer = access->planeData(p, sampleBuffer);

                ASSERT_EQ(cudaSuccess, cudaMemset2D(planeBuffer, access->rowStride(), i * 3 + p * 7,
                                                    access->numCols() * access->colStride(), access->numRows()))
                    << "Image #" << i << ", plane #" << p;
            }
        }

        // Check if no overwrites
        for (int i = 0; i < access->numSamples(); ++i)
        {
            nvcv::Byte *sampleBuffer = access->sampleData(i);
            for (int p = 1; p < access->numPlanes(); ++p)
            {
                const nvcv::Byte *planeBuffer = access->planeData(p, sampleBuffer);

                // enough for one plane
                std::vector<uint8_t> buf(access->numCols() * access->colStride() * access->numRows());

                ASSERT_EQ(cudaSuccess, cudaMemcpy2D(&buf[0], access->numCols() * access->colStride(), planeBuffer,
                                                    access->rowStride(), access->numCols() * access->colStride(),
                                                    access->numRows(), cudaMemcpyDeviceToHost))
                    << "Image #" << i << ", plane #" << p;

                ASSERT_TRUE(
                    all_of(buf.begin(), buf.end(), [gold = (uint8_t)(i * 3 + p * 7)](uint8_t v) { return v == gold; }))
                    << "Image #" << i << ", plane #" << p;
            }
        }
    }
}

class TensorTests
    : public t::TestWithParam<std::tuple<test::Param<"shape", nvcv::TensorShape>, test::Param<"dtype", nvcv::DataType>,
                                         test::Param<"strides", std::vector<int64_t>>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, TensorTests,
    test::ValueList<nvcv::TensorShape, nvcv::DataType, std::vector<int64_t>>
    {
        {nvcv::TensorShape{{53, 4, 16, 17},nvcv::TENSOR_NCHW}, nvcv::TYPE_U8, {4*16*32,16*32,32,1}},
        {nvcv::TensorShape{{53, 17, 16, 3},nvcv::TENSOR_NHWC}, nvcv::TYPE_U8, {17*64,64,3,1}},
        {nvcv::TensorShape{{4, 16, 17},nvcv::TENSOR_CHW}, nvcv::TYPE_U8, {16*32,32,1}},
        {nvcv::TensorShape{{17, 16, 3},nvcv::TENSOR_HWC}, nvcv::TYPE_U8, {64,3,1}},
        {nvcv::TensorShape{{32, 3, 16, 16},nvcv::TENSOR_NCHW}, nvcv::TYPE_F16, {3*16*32,16*32,32,2}},
        {nvcv::TensorShape{{16, 24, 24, 4},nvcv::TENSOR_NHWC}, nvcv::TYPE_F16, {24*24*4*2,24*4*2,4*2,2}}
    }
);

// clang-format on

TEST_P(TensorTests, smoke_create)
{
    const nvcv::TensorShape    PARAM_SHAPE = ::nvcv::test::ParamValue(std::get<0>(GetParam()));
    const nvcv::DataType       PARAM_DTYPE = ::nvcv::test::ParamValue(std::get<1>(GetParam()));
    const std::vector<int64_t> GOLD_SHAPE  = ::nvcv::test::ParamValue(std::get<2>(GetParam()));

    nvcv::Tensor tensor(PARAM_SHAPE, PARAM_DTYPE);

    EXPECT_EQ(PARAM_DTYPE, tensor.dtype());
    EXPECT_EQ(PARAM_SHAPE, tensor.shape());
    ASSERT_NE(nullptr, tensor.handle());

    {
        auto data = tensor.exportData();

        auto devdata = data.cast<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nvcv::NullOpt, devdata);

        const int64_t *strides = devdata->cdata().buffer.strided.strides;
        EXPECT_THAT(std::vector<int64_t>(strides, strides + data.rank()), t::ElementsAreArray(GOLD_SHAPE));
    }
}

TEST(TensorTests, smoke_create_allocator)
{
    int64_t setBufLen   = 0;
    int32_t setBufAlign = 0;

    // clang-format off
    nvcv::CustomAllocator myAlloc
    {
        nvcv::CustomCudaMemAllocator
        {
            [&setBufLen, &setBufAlign](int64_t size, int32_t bufAlign)
            {
                setBufLen = size;
                setBufAlign = bufAlign;

                NVCVMemoryBuffer ptr = nullptr;
                cudaMalloc(&ptr, size);
                return ptr;
            },
            [](NVCVMemoryBuffer ptr, int64_t, int32_t)
            {
                cudaFree(ptr);
            }
        }
    };
    // clang-format on

    nvcv::Tensor tensor(5, {163, 117}, nvcv::FMT_RGBA8, nvcv::MemAlignment{}.rowAddr(1).baseAddr(32),
                        myAlloc); // packed rows
    EXPECT_EQ(32, setBufAlign);

    auto devdata = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nvcv::NullOpt, devdata);

    EXPECT_EQ(1, devdata->stride(3));
    EXPECT_EQ(4 * 1, devdata->stride(2));
    EXPECT_EQ(163 * 4 * 1, devdata->stride(1));
    EXPECT_EQ(117 * 163 * 4 * 1, devdata->stride(0));
}

TEST(Tensor, smoke_cast)
{
    NVCVTensorHandle       handle;
    NVCVTensorRequirements reqs;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorCalcRequirementsForImages(5, 163, 117, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, &reqs));
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorConstruct(&reqs, nullptr, &handle));
    int ref;
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorRefCount(handle, &ref));
    EXPECT_EQ(ref, 1);

    auto         h = handle;
    nvcv::Tensor tensor(std::move(handle));

    EXPECT_EQ(h, tensor.handle());
    ASSERT_EQ(4, tensor.rank());
    EXPECT_EQ(4, tensor.shape()[3]);
    EXPECT_EQ(163, tensor.shape()[2]);
    EXPECT_EQ(117, tensor.shape()[1]);
    EXPECT_EQ(5, tensor.shape()[0]);
    EXPECT_EQ(nvcv::TYPE_U8, tensor.dtype());

    ref = tensor.reset();
    EXPECT_EQ(ref, 0);
}

TEST(Tensor, dec_ref_returns_new_reference_count)
{
    NVCVTensorRequirements reqs;
    NVCVTensorHandle       handle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorCalcRequirementsForImages(1, 8, 8, NVCV_IMAGE_FORMAT_U8, 0, 0, &reqs));
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorConstruct(&reqs, nullptr, &handle));
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorIncRef(handle, nullptr));

    int newRefCount = 0;
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorDecRef(handle, &newRefCount));
    EXPECT_EQ(1, newRefCount);
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorDecRef(handle, nullptr));
}

TEST(Tensor, smoke_user_pointer)
{
    nvcv::Tensor tensor(3, {163, 117}, nvcv::FMT_RGBA8);
    EXPECT_EQ(nullptr, tensor.userPointer());

    NVCVUserPointer userPtr;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorGetUserPointer(tensor.handle(), &userPtr));
    ASSERT_EQ(nullptr, userPtr);

    tensor.setUserPointer(reinterpret_cast<NVCVUserPointer>(0x123));
    EXPECT_EQ(reinterpret_cast<NVCVUserPointer>(0x123), tensor.userPointer());

    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorGetUserPointer(tensor.handle(), &userPtr));
    ASSERT_EQ(reinterpret_cast<NVCVUserPointer>(0x123), userPtr);

    tensor.setUserPointer(nullptr);
    EXPECT_EQ(nullptr, tensor.userPointer());

    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorGetUserPointer(tensor.handle(), &userPtr));
    ASSERT_EQ(nullptr, userPtr);
}

TEST(Tensor, valid_get_allocator)
{
    int                    tmp = 1;
    NVCVTensorHandle       tensorHandle;
    NVCVTensorRequirements reqs;
    auto                   alloc = reinterpret_cast<NVCVAllocatorHandle>(&tmp);
    EXPECT_NE(alloc, nullptr);

    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorCalcRequirementsForImages(1, 224, 224, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, &reqs));
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorConstruct(&reqs, nullptr, &tensorHandle));

    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorGetAllocator(tensorHandle, &alloc));
    EXPECT_EQ(alloc, nullptr);

    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorDecRef(tensorHandle, nullptr));
}

TEST(Tensor, layout_ne_op)
{
    NVCVTensorLayout lLayout = NVCV_TENSOR_NHWC;
    NVCVTensorLayout rLayout = NVCV_TENSOR_NCHW;
    EXPECT_TRUE(lLayout != rLayout);
}

TEST(TensorWrapData, valid_get_allocator)
{
    int                    tmp = 1;
    NVCVTensorHandle       tensorHandle;
    NVCVTensorHandle       tensorWrapHandle;
    NVCVTensorData         tensorData;
    NVCVTensorRequirements reqs;
    auto                   alloc = reinterpret_cast<NVCVAllocatorHandle>(&tmp);
    EXPECT_NE(alloc, nullptr);

    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorCalcRequirementsForImages(1, 224, 224, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, &reqs));
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorConstruct(&reqs, nullptr, &tensorHandle));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorExportData(tensorHandle, &tensorData));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorWrapDataConstruct(&tensorData, nullptr, nullptr, &tensorWrapHandle));

    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorGetAllocator(tensorWrapHandle, &alloc));
    EXPECT_EQ(alloc, nullptr);

    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorDecRef(tensorHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorDecRef(tensorWrapHandle, nullptr));
}

TEST(TensorWrapData, smoke_create)
{
    auto           fmt        = nvcv::ImageFormat(nvcv::ColorModel::RGB, nvcv::CSPEC_BT601_ER, nvcv::MemLayout::PL,
                                                  nvcv::DataKind::FLOAT, nvcv::Swizzle::S_XY00, nvcv::Packing::X16, nvcv::Packing::X16);
    nvcv::DataType GOLD_DTYPE = fmt.planeDataType(0);

    nvcv::Tensor origTensor(5, {173, 79}, fmt, nvcv::MemAlignment{}.rowAddr(1).baseAddr(32)); // packed rows

    auto tdata = origTensor.exportData<nvcv::TensorDataStridedCuda>();

    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*tdata);
    ASSERT_TRUE(access);

    EXPECT_EQ(nvcv::TENSOR_NCHW, tdata->layout());
    EXPECT_EQ(5, access->numSamples());
    EXPECT_EQ(173, access->numCols());
    EXPECT_EQ(79, access->numRows());
    EXPECT_EQ(2, access->numChannels());

    EXPECT_EQ(5, tdata->shape()[0]);
    EXPECT_EQ(173, tdata->shape()[3]);
    EXPECT_EQ(79, tdata->shape()[2]);
    EXPECT_EQ(2, tdata->shape()[1]);
    EXPECT_EQ(4, tdata->rank());

    EXPECT_EQ(2, tdata->stride(3));
    EXPECT_EQ(173 * 2, tdata->stride(2));

    auto tensor = nvcv::TensorWrapData(*tdata);

    ASSERT_NE(nullptr, tensor.handle());

    EXPECT_EQ(tdata->shape(), tensor.shape());
    EXPECT_EQ(tdata->layout(), tensor.layout());
    EXPECT_EQ(tdata->rank(), tensor.rank());
    EXPECT_EQ(GOLD_DTYPE, tensor.dtype());

    auto devdata = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nvcv::NullOpt, devdata);

    auto accessRef = nvcv::TensorDataAccessStridedImagePlanar::Create(*devdata);
    ASSERT_TRUE(access);

    EXPECT_EQ(tdata->dtype(), devdata->dtype());
    EXPECT_EQ(tdata->shape(), devdata->shape());
    EXPECT_EQ(tdata->rank(), devdata->rank());

    EXPECT_EQ(tdata->basePtr(), devdata->basePtr());

    auto *mem = tdata->basePtr();

    EXPECT_LE(mem + access->sampleStride() * 4, accessRef->sampleData(4));
    EXPECT_LE(mem + access->sampleStride() * 3, accessRef->sampleData(3));

    EXPECT_LE(mem + access->sampleStride() * 4, accessRef->sampleData(4, accessRef->planeData(0)));
    EXPECT_LE(mem + access->sampleStride() * 4 + access->planeStride() * 1,
              accessRef->sampleData(4, accessRef->planeData(1)));

    EXPECT_LE(mem + access->sampleStride() * 3, accessRef->sampleData(3, accessRef->planeData(0)));
    EXPECT_LE(mem + access->sampleStride() * 3 + access->planeStride() * 1,
              accessRef->sampleData(3, accessRef->planeData(1)));
}

class TensorWrapImageTests
    : public t::TestWithParam<
          std::tuple<test::Param<"size", nvcv::Size2D>, test::Param<"format", nvcv::ImageFormat>,
                     test::Param<"gold_shape", nvcv::TensorShape>, test::Param<"dtype", nvcv::DataType>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, TensorWrapImageTests,
    test::ValueList<nvcv::Size2D, nvcv::ImageFormat, nvcv::TensorShape, nvcv::DataType>
    {
        {{61,23}, nvcv::FMT_RGBA8p, nvcv::TensorShape{{1,4,23,61},nvcv::TENSOR_NCHW}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_RGBA8, nvcv::TensorShape{{1,23,61,4},nvcv::TENSOR_NHWC}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_RGB8, nvcv::TensorShape{{1,23,61,3},nvcv::TENSOR_NHWC}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_RGB8p, nvcv::TensorShape{{1,3,23,61},nvcv::TENSOR_NCHW}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_F32, nvcv::TensorShape{{1,1,23,61},nvcv::TENSOR_NCHW}, nvcv::TYPE_F32},
        {{61,23}, nvcv::FMT_2F32, nvcv::TensorShape{{1,23,61,2},nvcv::TENSOR_NHWC}, nvcv::TYPE_F32},
        {{61,23}, nvcv::FMT_F16, nvcv::TensorShape{{1,1,23,61},nvcv::TENSOR_NCHW}, nvcv::TYPE_F16},
        {{61,23}, nvcv::FMT_2F16, nvcv::TensorShape{{1,23,61,2},nvcv::TENSOR_NHWC}, nvcv::TYPE_F16}
    }
);

// clang-format on

TEST_P(TensorWrapImageTests, smoke_create)
{
    const nvcv::Size2D      PARAM_SIZE   = ::nvcv::test::ParamValue(std::get<0>(GetParam()));
    const nvcv::ImageFormat PARAM_FORMAT = ::nvcv::test::ParamValue(std::get<1>(GetParam()));
    const nvcv::TensorShape GOLD_SHAPE   = ::nvcv::test::ParamValue(std::get<2>(GetParam()));
    const nvcv::DataType    GOLD_DTYPE   = ::nvcv::test::ParamValue(std::get<3>(GetParam()));

    nvcv::Image img(PARAM_SIZE, PARAM_FORMAT);

    auto tensor = nvcv::TensorWrapImage(img);

    EXPECT_EQ(GOLD_SHAPE, tensor.shape());
    EXPECT_EQ(GOLD_DTYPE, tensor.dtype());

    auto imgData    = img.exportData<nvcv::ImageDataStridedCuda>();
    auto tensorData = tensor.exportData<nvcv::TensorDataStridedCuda>();

    EXPECT_TRUE(imgData);

    auto tensorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*tensorData);
    EXPECT_TRUE(tensorAccess);

    EXPECT_EQ(imgData->plane(0).basePtr, reinterpret_cast<NVCVByte *>(tensorData->basePtr()));

    for (int p = 0; p < imgData->numPlanes(); ++p)
    {
        EXPECT_EQ(imgData->plane(p).basePtr, reinterpret_cast<NVCVByte *>(tensorAccess->planeData(p)));
        EXPECT_EQ(imgData->plane(p).rowStride, tensorAccess->rowStride());
        EXPECT_EQ(img.format().planePixelStrideBytes(p), tensorAccess->colStride());
    }
}

class TensorWrapParamTests
    : public t::TestWithParam<
          std::tuple<test::Param<"shape", nvcv::TensorShape>, test::Param<"strides", std::vector<int>>,
                     test::Param<"dtype", nvcv::DataType>, test::Param<"goldStatus", NVCVStatus>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Positive, TensorWrapParamTests,
                              test::ValueList<nvcv::TensorShape, std::vector<int>, nvcv::DataType>
{
  {{       {2},  "C"},     {4}, nvcv::TYPE_F32},
  {{       {1},  "W"},     {1}, nvcv::TYPE_U8},
  {{   {10, 5}, "HW"},  {5, 1}, nvcv::TYPE_U8},
  {{ {1,10, 5}, "NHW"}, {1,5, 1}, nvcv::TYPE_U8},
  {{ {3,1, 5}, "NHW"}, {5*4,1,4}, nvcv::TYPE_F32},
  {{ {3,1, 1}, "NHW"}, {5*4,1,1}, nvcv::TYPE_F32},
  { {{10, 5,3}, "HWC"}, {5*3*4, 3*4,4}, nvcv::TYPE_F32},
  {{ {3,1, 5}, "NHW"}, {5*2,1,2}, nvcv::TYPE_F16},
  {{ {3,1, 1}, "NHW"}, {5*2,1,1}, nvcv::TYPE_F16},
  { {{10, 5,3}, "HWC"}, {5*3*2, 3*2,2}, nvcv::TYPE_F16},
} * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative, TensorWrapParamTests,
                              test::ValueList<nvcv::TensorShape, std::vector<int>, nvcv::DataType>
{
  {{       {2},  "C"},     {3}, nvcv::TYPE_F32},
  {{       {2},  "C"},     {5}, nvcv::TYPE_F32},
  {{   {10, 5}, "HW"},  {1, 10}, nvcv::TYPE_U8}, // WH order
  {{ {10, 5,3}, "HWC"}, {4,10*4,10*4*5}, nvcv::TYPE_F32}, // CWH order
  {{ {1,10, 5}, "NHW"}, {1,5*4-1,4}, nvcv::TYPE_F32},
  {{ {3,1, 5}, "NHW"}, {5*4-1,1,4}, nvcv::TYPE_F32},
} * NVCV_ERROR_INVALID_ARGUMENT);

// clang-format on

TEST_P(TensorWrapParamTests, smoke_create)
{
    const nvcv::TensorShape PARAM_TSHAPE  = ::nvcv::test::ParamValue(std::get<0>(GetParam()));
    const std::vector<int>  PARAM_STRIDES = ::nvcv::test::ParamValue(std::get<1>(GetParam()));
    const nvcv::DataType    PARAM_DTYPE   = ::nvcv::test::ParamValue(std::get<2>(GetParam()));
    const NVCVStatus        GOLD_STATUS   = ::nvcv::test::ParamValue(std::get<3>(GetParam()));

    NVCVTensorBufferStrided buf = {};
    for (size_t i = 0; i < PARAM_STRIDES.size(); ++i)
    {
        buf.strides[i] = PARAM_STRIDES[i];
    }
    // we're not accessing the memory in any way, let's set it to something not null
    buf.basePtr = (NVCVByte *)0xDEADBEEF;

    nvcv::Tensor tensor;
    NVCV_ASSERT_STATUS(GOLD_STATUS,
                       tensor = TensorWrapData(nvcv::TensorDataStridedCuda(PARAM_TSHAPE, PARAM_DTYPE, buf)));

    if (GOLD_STATUS == NVCV_SUCCESS)
    {
        auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nullptr, data);

        EXPECT_EQ(buf.basePtr, (NVCVByte *)data->basePtr());
        EXPECT_EQ(PARAM_TSHAPE.size(), data->rank());
        EXPECT_EQ(PARAM_TSHAPE, data->shape());
        for (int i = 0; i < data->rank(); ++i)
        {
            EXPECT_EQ(PARAM_STRIDES[i], data->stride(i)) << "stride #" << i;
        }
    }
}

class TensorTests_Negative : public ::testing::Test
{
public:
    TensorTests_Negative() = default;

    ~TensorTests_Negative() override = default;

    void SetUp() override
    {
        ASSERT_EQ(NVCV_SUCCESS, nvcvTensorCalcRequirementsForImages(1, 224, 224, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, &reqs));
        ASSERT_EQ(NVCV_SUCCESS, nvcvTensorConstruct(&reqs, nullptr, &handle));
    }

    void TearDown() override
    {
        nvcv::Tensor tensor(std::move(handle));
    }

    NVCVTensorHandle       handle;
    NVCVTensorRequirements reqs;
};

TEST_F(TensorTests_Negative, invalid_parameter_TensorCalcRequirementsForImages)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorCalcRequirementsForImages(-1, 224, 224, NVCV_IMAGE_FORMAT_RGBA8, 0,
                                                                               0, &reqs)); // invalid numImages
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorCalcRequirementsForImages(5, -1, 224, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, &reqs)); // invalid width
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorCalcRequirementsForImages(5, 224, -1, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, &reqs)); // invalid height
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorCalcRequirementsForImages(5, 224, 224, NVCV_IMAGE_FORMAT_NONE, 0, 0, &reqs)); // invalid format
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorCalcRequirementsForImages(5, 224, 224, NVCV_IMAGE_FORMAT_RGBA8, 3,
                                                                               0, &reqs)); // invalid baseAddrAlignment
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorCalcRequirementsForImages(5, 224, 224, NVCV_IMAGE_FORMAT_RGBA8, 0,
                                                                               3, &reqs)); // invalid rowAddrAlignment
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorCalcRequirementsForImages(5, 224, 224, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, nullptr)); // null reqs
    EXPECT_EQ(NVCV_ERROR_NOT_IMPLEMENTED,
              nvcvTensorCalcRequirementsForImages(
                  5, 224, 224, NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, BL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8_Z8_W8),
                  0, 0, &reqs)); // BL layout
    EXPECT_EQ(NVCV_ERROR_NOT_IMPLEMENTED,
              nvcvTensorCalcRequirementsForImages(5, 224, 224, NVCV_IMAGE_FORMAT_UYVY, 0, 0,
                                                  &reqs)); // Not implemented subsampled planes (422)
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorCalcRequirementsForImages(5, 224, 224, NVCV_IMAGE_FORMAT_NV24, 0,
                                                                               0, &reqs)); // semi-planar image format
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcvTensorCalcRequirementsForImages(
            5, 224, 224, NVCV_DETAIL_MAKE_COLOR_FMT4(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X8, X8, X8, X32),
            0, 0, &reqs)); // planes of image format don't have the same packing
}

TEST_F(TensorTests_Negative, invalid_parameter_TensorCalcRequirements)
{
    std::array<int64_t, 2> valid_wh = {224, 224};
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorCalcRequirements(2, valid_wh.data(), NVCV_DATA_TYPE_NONE, NVCV_TENSOR_LAYOUT_MAKE("HW"), 0, 0,
                                         &reqs)); // invalid dtype
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorCalcRequirements(3, valid_wh.data(), NVCV_DATA_TYPE_U8, NVCV_TENSOR_LAYOUT_MAKE("HW"), 0, 0,
                                         &reqs)); // mismatch rank
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorCalcRequirements(-1, valid_wh.data(), NVCV_DATA_TYPE_U8, NVCV_TENSOR_LAYOUT_MAKE(""), 0, 0,
                                         &reqs)); // invalid rank
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorCalcRequirements(2, valid_wh.data(), NVCV_DATA_TYPE_U8, NVCV_TENSOR_NONE, 3, 0,
                                         &reqs)); // invalid baseAddrAlignment
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorCalcRequirements(2, valid_wh.data(), NVCV_DATA_TYPE_U8, NVCV_TENSOR_NONE, 0, 3,
                                         &reqs)); // invalid rowAddrAlignment
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorCalcRequirements(2, valid_wh.data(), NVCV_DATA_TYPE_U8, NVCV_TENSOR_NONE, 0, 0,
                                         nullptr)); // null reqs
}

TEST_F(TensorTests_Negative, invalid_parameter_TensorConstruct)
{
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorCalcRequirementsForImages(1, 224, 224, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, &reqs));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorConstruct(nullptr, nullptr, &handle)); // null reqs
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorConstruct(&reqs, nullptr, nullptr));   // null handle
}

TEST_F(TensorTests_Negative, invalid_parameter_TensorWrapDataConstruct)
{
    NVCVTensorData           tensorData;
    NVCVTensorBufferStrided &tensorStrided = tensorData.buffer.strided;
    tensorData.bufferType                  = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    tensorData.layout                      = NVCV_TENSOR_NHWC;
    tensorData.rank                        = 4;
    tensorData.shape[0]                    = 1;
    tensorData.shape[1]                    = 224;
    tensorData.shape[2]                    = 224;
    tensorData.shape[3]                    = 3;
    tensorData.dtype                       = NVCV_DATA_TYPE_F32;
    tensorStrided.strides[3]               = nvcv::FMT_RGBf32.planePixelStrideBytes(0) / nvcv::FMT_RGBf32.numChannels();
    tensorStrided.strides[2]               = nvcv::FMT_RGBf32.planePixelStrideBytes(0);
    tensorStrided.strides[1]               = 224 * nvcv::FMT_RGBf32.planePixelStrideBytes(0);
    tensorStrided.strides[0]               = tensorStrided.strides[1] * tensorData.shape[1];

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorWrapDataConstruct(nullptr, nullptr, nullptr, &handle)); // null tensorData
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorWrapDataConstruct(&tensorData, nullptr, nullptr, nullptr)); // null handle

    tensorData.bufferType = NVCV_TENSOR_BUFFER_NONE;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorWrapDataConstruct(&tensorData, nullptr, nullptr, &handle)); // unsupported buffer type
}

TEST_F(TensorTests_Negative, invalid_parameter_TensorGetLayout)
{
    NVCVTensorLayout layout;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorGetLayout(nullptr, &layout)); // null handle
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorGetLayout(handle, nullptr));  // null layout
}

TEST_F(TensorTests_Negative, invalid_parameter_TensorExportData)
{
    NVCVTensorData data;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorExportData(nullptr, &data));  // null handle
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorExportData(handle, nullptr)); // null data
}

TEST_F(TensorTests_Negative, invalid_parameter_TensorGetShape)
{
    int32_t                                   rank  = NVCV_TENSOR_MAX_RANK;
    std::array<int64_t, NVCV_TENSOR_MAX_RANK> shape = {};

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorGetShape(nullptr, &rank, shape.data()));  // null handle
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorGetShape(handle, nullptr, shape.data())); // null rank
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorGetShape(handle, &rank, nullptr));        // null shape
}

TEST_F(TensorTests_Negative, invalid_parameter_TensorGetUserPointer)
{
    NVCVUserPointer userPtr;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorGetUserPointer(nullptr, &userPtr)); // null handle
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorGetUserPointer(handle, nullptr));   // null rank
}

TEST_F(TensorTests_Negative, invalid_parameter_TensorReshape)
{
    std::array<int64_t, 3> new_shape = {4, 224, 224};
    NVCVTensorHandle       outHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorReshape(nullptr, 3, new_shape.data(), NVCV_TENSOR_CHW, &outHandle)); // null handle
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorReshape(handle, 0, new_shape.data(), NVCV_TENSOR_CHW, &outHandle)); // invalid rank
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorReshape(handle, NVCV_TENSOR_MAX_RANK + 1, new_shape.data(),
                                                             NVCV_TENSOR_CHW, &outHandle)); // invalid rank 2
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorReshape(handle, 3, new_shape.data(), NVCV_TENSOR_HW, &outHandle)); // mismatch layout
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorReshape(handle, 3, new_shape.data(), NVCV_TENSOR_CHW, nullptr)); // null out handle
}

TEST(TensorTests, reshape_contiguous_tensor)
{
    nvcv::TensorShape shape{
        {2, 3, 4},
        nvcv::TENSOR_HWC
    };
    nvcv::Tensor           tensor(shape, nvcv::TYPE_U8, nvcv::MemAlignment{}.rowAddr(1).baseAddr(32));
    std::array<int64_t, 2> newShape{6, 4};
    NVCVTensorHandle       reshapedHandle{};

    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorReshape(tensor.handle(), 2, newShape.data(), NVCV_TENSOR_HW, &reshapedHandle));
    nvcv::Tensor reshaped(std::move(reshapedHandle));

    EXPECT_EQ((nvcv::TensorShape{
                  {6, 4},
                  nvcv::TENSOR_HW
    }),
              reshaped.shape());
}

TEST(TensorTests, reshape_rejects_mismatched_volume)
{
    nvcv::TensorShape shape{
        {2, 3, 4},
        nvcv::TENSOR_HWC
    };
    nvcv::Tensor           tensor(shape, nvcv::TYPE_U8);
    std::array<int64_t, 2> newShape{5, 5};
    NVCVTensorHandle       reshapedHandle{};

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorReshape(tensor.handle(), 2, newShape.data(), NVCV_TENSOR_HW, &reshapedHandle));
}

TEST(TensorTests, reshape_rejects_shape_crossing_padded_stride)
{
    nvcv::TensorShape shape{
        {2, 3},
        nvcv::TENSOR_HW
    };
    nvcv::Tensor           tensor(shape, nvcv::TYPE_U8);
    std::array<int64_t, 1> newShape{6};
    NVCVTensorHandle       reshapedHandle{};

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorReshape(tensor.handle(), 1, newShape.data(), NVCV_TENSOR_W, &reshapedHandle));
}

TEST(TensorTests, reshape_rank_one_tensor)
{
    NVCVTensorData tensorData{};
    tensorData.bufferType                = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    tensorData.rank                      = 1;
    tensorData.shape[0]                  = 6;
    tensorData.dtype                     = NVCV_DATA_TYPE_U8;
    tensorData.layout                    = NVCV_TENSOR_NONE;
    tensorData.buffer.strided.basePtr    = reinterpret_cast<NVCVByte *>(1);
    tensorData.buffer.strided.strides[0] = 1;

    NVCVTensorHandle tensorHandle{};
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorWrapDataConstruct(&tensorData, nullptr, nullptr, &tensorHandle));
    nvcv::Tensor tensor(std::move(tensorHandle));

    std::array<int64_t, 1> newShape{6};
    NVCVTensorHandle       reshapedHandle{};

    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorReshape(tensor.handle(), 1, newShape.data(), NVCV_TENSOR_NONE, &reshapedHandle));
    nvcv::Tensor reshaped(std::move(reshapedHandle));

    auto reshapedData = reshaped.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(reshapedData);
    EXPECT_EQ(1, reshapedData->rank());
    EXPECT_EQ(6, reshapedData->shape(0));
    EXPECT_EQ(1, reshapedData->stride(0));
}

TEST_F(TensorTests_Negative, invalid_parameter_TensorShapePermute)
{
    NVCVTensorLayout     srcLayout = NVCV_TENSOR_NHWC;
    std::vector<int64_t> srcShape{16, 61, 23, 3};
    NVCVTensorLayout     dstLayout = NVCV_TENSOR_NCHW;
    std::vector<int64_t> outShape(srcShape.size());

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorShapePermute(srcLayout, nullptr, dstLayout, outShape.data())); // null srcShape
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvTensorShapePermute(srcLayout, srcShape.data(), dstLayout, nullptr)); // null outShape
}

TEST_F(TensorTests_Negative, invalid_out_get_allocator)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorGetAllocator(handle, nullptr));
}

class TensorPermuteTests
    : public t::TestWithParam<
          std::tuple<test::Param<"srcLayout", NVCVTensorLayout>, test::Param<"srcShape", std::vector<int64_t>>,
                     test::Param<"dstLayout", NVCVTensorLayout>, test::Param<"goldShape", std::vector<int64_t>>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, TensorPermuteTests,
    test::ValueList<NVCVTensorLayout, std::vector<int64_t>, NVCVTensorLayout, std::vector<int64_t>>
    {
        {NVCV_TENSOR_NHWC, {16, 61, 23, 3}, NVCV_TENSOR_NCHW, {16, 3, 61, 23}},
        {NVCV_TENSOR_CHW, {3, 61, 23}, NVCV_TENSOR_HWC, {61, 23, 3}},
        {NVCV_TENSOR_CFDHW, {3, 2, 6, 61, 23}, NVCV_TENSOR_FDHWC, {2, 6, 61, 23, 3}},
        {NVCV_TENSOR_CHW, {3, 61, 23}, NVCV_TENSOR_HW, {61, 23}},
        {NVCV_TENSOR_HWC, {61, 23, 3}, NVCV_TENSOR_HW, {61, 23}}
    }
);

// clang-format on

TEST_P(TensorPermuteTests, smoke)
{
    NVCVTensorLayout           srcLayout = ::nvcv::test::ParamValue(std::get<0>(GetParam()));
    std::vector<int64_t>       srcShape  = ::nvcv::test::ParamValue(std::get<1>(GetParam()));
    NVCVTensorLayout           dstLayout = ::nvcv::test::ParamValue(std::get<2>(GetParam()));
    const std::vector<int64_t> goldShape = ::nvcv::test::ParamValue(std::get<3>(GetParam()));

    std::vector<int64_t> outShape(goldShape.size());
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorShapePermute(srcLayout, srcShape.data(), dstLayout, outShape.data()));
    EXPECT_EQ(outShape, goldShape);
}

TEST(TensorTests_Negative_, invalidWrapImageConstruct_invalid_mem_layout)
{
    NVCVTensorHandle handle;
    nvcv::Image      img(nvcv::Size2D{24, 24}, nvcv::FMT_UYVY);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorWrapImageConstruct(img.handle(), &handle));
}

TEST(TensorTests_Negative_, invalidWrapImageConstruct_diff_dtype)
{
#define NVCV_IMAGE_FORMAT_DIFF_DTYPE \
    NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X16)

    NVCVTensorHandle handle;
    nvcv::Image      img(nvcv::Size2D{24, 24}, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_DIFF_DTYPE});

#undef NVCV_IMAGE_FORMAT_DIFF_DTYPE

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorWrapImageConstruct(img.handle(), &handle));
}

TEST(TensorTests_Negative_, invalidWrapImageConstruct_invalid_params)
{
    NVCVTensorHandle handle;
    nvcv::Image      img(nvcv::Size2D{24, 24}, nvcv::FMT_U8);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorWrapImageConstruct(nullptr, &handle));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorWrapImageConstruct(img.handle(), nullptr));
}

TEST(TensorTests_Negative_, invalid_getDataType)
{
    nvcv::Tensor tensor(
        nvcv::TensorShape{
            {4, 16, 17},
            nvcv::TENSOR_CHW
    },
        nvcv::TYPE_U8);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorGetDataType(tensor.handle(), nullptr));
}

// ---------------------------------------------------------------------------
// Stride-alignment validation tests
//
// Non-packed strides must be multiples of the element alignment so every
// element in every row/sample is naturally aligned on the device.
// ---------------------------------------------------------------------------

// Helper: build a TensorWrapData from raw components and return the status.
static NVCVStatus MakeStridedTensor(const nvcv::TensorShape &shape, nvcv::DataType dtype,
                                    std::initializer_list<int64_t> strides)
{
    NVCVTensorBufferStrided buf = {};
    int                     i   = 0;
    for (int64_t s : strides) buf.strides[i++] = s;
    // basePtr: 16-byte-aligned pointer (no memory is accessed, just validated)
    alignas(16) static std::array<std::byte, 1> dummy;
    buf.basePtr = reinterpret_cast<NVCVByte *>(dummy.data());

    NVCVTensorHandle h      = nullptr;
    auto             td     = nvcv::TensorDataStridedCuda(shape, dtype, buf);
    NVCVStatus       status = nvcvTensorWrapDataConstruct(&td.cdata(), nullptr, nullptr, &h);
    if (h)
        nvcvTensorDecRef(h, nullptr);
    return status;
}

// ----- Negative: misaligned non-packed strides are rejected -----

// Exact reproduction of the reported bug: U16 NHWC {2,2,8,1}, sample stride 33 (odd).
TEST(TensorStrideAlignment_Negative, u16_nhwc_odd_sample_stride)
{
    // stride[0]=33 is not 2-aligned; should fail
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, MakeStridedTensor(
                                               nvcv::TensorShape{
                                                   {2, 2, 8, 1},
                                                   "NHWC"
    },
                                               nvcv::TYPE_U16, {33, 16, 2, 2}));
}

// U16 NHWC: row stride misaligned (non-packed H dim).
TEST(TensorStrideAlignment_Negative, u16_nhwc_odd_row_stride)
{
    // stride[1]=17 is not 2-aligned for U16
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, MakeStridedTensor(
                                               nvcv::TensorShape{
                                                   {2, 4, 8, 1},
                                                   "NHWC"
    },
                                               nvcv::TYPE_U16, {136, 17, 2, 2}));
}

// U32 NCHW: sample stride not 4-aligned.
TEST(TensorStrideAlignment_Negative, u32_nchw_misaligned_sample_stride)
{
    // shape {3,1,4,8}, NCHW: only W is packed. C is skipped because shape[1] == 1.
    // minPitch(N)=128. stride[0]=130 is >= minPitch and not 4-aligned.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, MakeStridedTensor(
                                               nvcv::TensorShape{
                                                   {3, 1, 4, 8},
                                                   "NCHW"
    },
                                               nvcv::TYPE_U32, {130, 32, 32, 4}));
}

// F32 NHW: sample stride not 4-aligned.
TEST(TensorStrideAlignment_Negative, f32_nhw_misaligned_sample_stride)
{
    // minPitch(N)=128. stride[0]=130 is >= minPitch and not 4-aligned.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, MakeStridedTensor(
                                               nvcv::TensorShape{
                                                   {3, 4, 8},
                                                   "NHW"
    },
                                               nvcv::TYPE_F32, {130, 32, 4}));
}

// F16 NHW: sample stride not 2-aligned.
TEST(TensorStrideAlignment_Negative, f16_nhw_misaligned_sample_stride)
{
    // minPitch(N)=64. stride[0]=65 is >= minPitch and not 2-aligned.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, MakeStridedTensor(
                                               nvcv::TensorShape{
                                                   {3, 4, 8},
                                                   "NHW"
    },
                                               nvcv::TYPE_F16, {65, 16, 2}));
}

// F16 NHW: row stride not 2-aligned.
TEST(TensorStrideAlignment_Negative, f16_nhw_misaligned_row_stride)
{
    // stride[1]=17 is not 2-aligned
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, MakeStridedTensor(
                                               nvcv::TensorShape{
                                                   {3, 4, 8},
                                                   "NHW"
    },
                                               nvcv::TYPE_F16, {68, 17, 2}));
}

// NHWC float4 (4F32): sample stride not 4-aligned.
TEST(TensorStrideAlignment_Negative, float4_nhwc_misaligned_sample_stride)
{
    // TYPE_4F32 has strideBytes=16, alignment=4; minPitch(N)=128 and stride[0]=130 is not 4-aligned.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, MakeStridedTensor(
                                               nvcv::TensorShape{
                                                   {2, 2, 4, 4},
                                                   "NHWC"
    },
                                               nvcv::TYPE_4F32, {130, 64, 16, 4}));
}

// RGBA8 (4U8): alignment=1, but check that a stride of 1 is still rejected when minPitch fails.
// (Existing minPitch logic; alignment check must not mask it.)
TEST(TensorStrideAlignment_Negative, rgba8_nhwc_row_stride_below_minpitch)
{
    // A three-byte row cannot hold four four-byte pixels.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, MakeStridedTensor(
                                               nvcv::TensorShape{
                                                   {2, 4, 4, 4},
                                                   "NHWC"
    },
                                               nvcv::TYPE_4U8, {48, 3, 16, 4}));
}

// ----- Boundary: stride exactly at alignment boundary is accepted -----

// U16 NHWC: sample stride exactly 2-aligned (should succeed).
TEST(TensorStrideAlignment_Positive, u16_nhwc_aligned_sample_stride_exact)
{
    EXPECT_EQ(NVCV_SUCCESS, MakeStridedTensor(
                                nvcv::TensorShape{
                                    {2, 2, 8, 1},
                                    "NHWC"
    },
                                nvcv::TYPE_U16, {32, 16, 2, 2}));
}

// U16 NHWC: sample stride 34 (next even number above minPitch=32), should succeed.
TEST(TensorStrideAlignment_Positive, u16_nhwc_aligned_sample_stride_padded)
{
    EXPECT_EQ(NVCV_SUCCESS, MakeStridedTensor(
                                nvcv::TensorShape{
                                    {2, 2, 8, 1},
                                    "NHWC"
    },
                                nvcv::TYPE_U16, {34, 16, 2, 2}));
}

// ----- Positive: legitimate tensors must still be accepted -----

// RGB8 (3U8, uchar3): alignment=1, so any positive stride works for non-packed dims,
// including odd ones.  For HWC rank=3, firstPacked=1: W (dim 1) and C (dim 2) are packed;
// H (dim 0) is non-packed — any stride >= minPitch is accepted regardless of parity.
TEST(TensorStrideAlignment_Positive, rgb8_hwc_odd_row_stride_accepted)
{
    // shape {4,5,3} HWC: C stride=3 (packed), W stride=9 (packed, 3*3=9).
    // minPitch(H) = 9*5 = 45.  H stride=45 is odd but 1-aligned — accepted.
    EXPECT_EQ(NVCV_SUCCESS, MakeStridedTensor(
                                nvcv::TensorShape{
                                    {4, 5, 3},
                                    "HWC"
    },
                                nvcv::TYPE_3U8, {47, 9, 3}));
}

// NHWC RGB8: batch and row non-packed dims both with odd strides — alignment=1, accepted.
TEST(TensorStrideAlignment_Positive, rgb8_nhwc_odd_strides_accepted)
{
    // shape {3,4,5,3}: C stride=3, W stride=9 (packed).
    // minPitch(H) = 9*5 = 45.  H stride=45 is odd, 1-aligned ✓.
    // minPitch(N) = 45*4 = 180.  N stride=181 is odd, 1-aligned ✓.
    EXPECT_EQ(NVCV_SUCCESS, MakeStridedTensor(
                                nvcv::TensorShape{
                                    {3, 4, 5, 3},
                                    "NHWC"
    },
                                nvcv::TYPE_3U8, {181, 45, 9, 3}));
}

// RGBA8 (4U8): alignment=1, odd row and sample strides accepted.
TEST(TensorStrideAlignment_Positive, rgba8_nhwc_odd_strides_accepted)
{
    // shape {2,3,4,4}: C stride=4, W stride=16 (packed, 4*4=16).
    // minPitch(H) = 16*4 = 64.  H stride=65 is odd, 1-aligned ✓.
    // minPitch(N) = 65*3 = 195.  N stride=195 is odd, 1-aligned ✓.
    EXPECT_EQ(NVCV_SUCCESS, MakeStridedTensor(
                                nvcv::TensorShape{
                                    {2, 3, 4, 4},
                                    "NHWC"
    },
                                nvcv::TYPE_4U8, {195, 65, 16, 4}));
}

// U8 (1-byte): alignment=1 so odd strides are fine for non-packed dims.
TEST(TensorStrideAlignment_Positive, u8_nhw_odd_strides_accepted)
{
    // W stride=1 (packed).  H stride=9 (≥8, 1-aligned).  N stride=37 (≥9*4=36, 1-aligned).
    EXPECT_EQ(NVCV_SUCCESS, MakeStridedTensor(
                                nvcv::TensorShape{
                                    {3, 4, 8},
                                    "NHW"
    },
                                nvcv::TYPE_U8, {37, 9, 1}));
}

// F32 NHWC: all strides 4-aligned, accepted.
TEST(TensorStrideAlignment_Positive, f32_nhwc_aligned_strides)
{
    EXPECT_EQ(NVCV_SUCCESS, MakeStridedTensor(
                                nvcv::TensorShape{
                                    {2, 3, 4, 1},
                                    "NHWC"
    },
                                nvcv::TYPE_F32, {48, 16, 4, 4}));
}

// F16 NHWC: all strides 2-aligned, accepted.
TEST(TensorStrideAlignment_Positive, f16_nhwc_aligned_strides)
{
    EXPECT_EQ(NVCV_SUCCESS, MakeStridedTensor(
                                nvcv::TensorShape{
                                    {2, 3, 4, 1},
                                    "NHWC"
    },
                                nvcv::TYPE_F16, {24, 8, 2, 2}));
}

// NCHW F32: sample and channel strides 4-aligned, accepted.
TEST(TensorStrideAlignment_Positive, f32_nchw_aligned_strides)
{
    // shape {2,3,4,8} NCHW: W is packed (stride=4), H/C/N non-packed.
    // H stride=32 (≥8*4=32), C stride=128 (≥32*4=128), N stride=384 (≥128*3=384). All 4-aligned.
    EXPECT_EQ(NVCV_SUCCESS, MakeStridedTensor(
                                nvcv::TensorShape{
                                    {2, 3, 4, 8},
                                    "NCHW"
    },
                                nvcv::TYPE_F32, {384, 128, 32, 4}));
}

// Rank-1: packed only, no non-packed dims — always accepted.
TEST(TensorStrideAlignment_Positive, rank1_no_nonpacked_dims)
{
    EXPECT_EQ(NVCV_SUCCESS, MakeStridedTensor(nvcv::TensorShape{{8}, "W"}, nvcv::TYPE_U16, {2}));
}

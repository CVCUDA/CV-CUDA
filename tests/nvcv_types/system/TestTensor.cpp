/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <list>
#include <random>

#include <nvcv/Fwd.hpp>

namespace t    = ::testing;
namespace test = nvcv::test;

class TensorTests
    : public t::TestWithParam<std::tuple<test::Param<"numImages", int>, test::Param<"width", int>,
                                         test::Param<"height", int>, test::Param<"format", nvcv::ImageFormat>,
                                         test::Param<"shape", nvcv::TensorShape>, test::Param<"dtype", nvcv::DataType>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, TensorTests,
    test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::TensorShape, nvcv::DataType>
    {
        {53, 32, 16, nvcv::FMT_RGBA8p, nvcv::TensorShape{{53, 4, 16, 32},nvcv::TensorLayout::NCHW} , nvcv::TYPE_U8},
        {14, 64, 18, nvcv::FMT_RGB8, nvcv::TensorShape{{14, 18, 64, 3},nvcv::TensorLayout::NHWC}, nvcv::TYPE_U8}
    }
);

// clang-format on

TEST_P(TensorTests, wip_create)
{
    const int               PARAM_NUM_IMAGES = std::get<0>(GetParam());
    const int               PARAM_WIDTH      = std::get<1>(GetParam());
    const int               PARAM_HEIGHT     = std::get<2>(GetParam());
    const nvcv::ImageFormat PARAM_FORMAT     = std::get<3>(GetParam());
    const nvcv::TensorShape GOLD_SHAPE       = std::get<4>(GetParam());
    const nvcv::DataType    GOLD_DTYPE       = std::get<5>(GetParam());
    const int               GOLD_RANK        = 4;

    nvcv::Tensor tensor(PARAM_NUM_IMAGES, {PARAM_WIDTH, PARAM_HEIGHT}, PARAM_FORMAT);

    EXPECT_EQ(GOLD_DTYPE, tensor.dtype());
    EXPECT_EQ(GOLD_SHAPE, tensor.shape());
    EXPECT_EQ(GOLD_RANK, tensor.rank());
    EXPECT_EQ(GOLD_SHAPE.layout(), tensor.layout());
    ASSERT_NE(nullptr, tensor.handle());

    {
        const nvcv::ITensorData *data = tensor.exportData();
        ASSERT_NE(nullptr, data);

        ASSERT_EQ(tensor.dtype(), data->dtype());

        auto *devdata = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(data);
        ASSERT_NE(nullptr, devdata);

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
                nvcv::Byte *planeBuffer = access->planeData(p, sampleBuffer);

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

TEST(TensorTests, wip_create_allocator)
{
    ;

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

                void *ptr = nullptr;
                cudaMalloc(&ptr, size);
                return ptr;
            },
            [](void *ptr, int64_t bufLen, int32_t bufAlign)
            {
                cudaFree(ptr);
            }
        }
    };
    // clang-format on

    nvcv::Tensor tensor(5, {163, 117}, nvcv::FMT_RGBA8, nvcv::MemAlignment{}.rowAddr(1).baseAddr(32),
                        &myAlloc); // packed rows
    EXPECT_EQ(32, setBufAlign);

    const nvcv::ITensorData *data = tensor.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(data);
    ASSERT_NE(nullptr, devdata);

    EXPECT_EQ(1, devdata->stride(3));
    EXPECT_EQ(4 * 1, devdata->stride(2));
    EXPECT_EQ(163 * 4 * 1, devdata->stride(1));
    EXPECT_EQ(117 * 163 * 4 * 1, devdata->stride(0));
}

TEST(Tensor, wip_cast)
{
    nvcv::Tensor tensor(3, {163, 117}, nvcv::FMT_RGBA8);

    EXPECT_EQ(&tensor, nvcv::StaticCast<nvcv::Tensor *>(tensor.handle()));
    EXPECT_EQ(&tensor, &nvcv::StaticCast<nvcv::Tensor>(tensor.handle()));

    EXPECT_EQ(&tensor, nvcv::StaticCast<nvcv::ITensor *>(tensor.handle()));
    EXPECT_EQ(&tensor, &nvcv::StaticCast<nvcv::ITensor>(tensor.handle()));

    EXPECT_EQ(&tensor, nvcv::DynamicCast<nvcv::Tensor *>(tensor.handle()));
    EXPECT_EQ(&tensor, &nvcv::DynamicCast<nvcv::Tensor>(tensor.handle()));

    EXPECT_EQ(&tensor, nvcv::DynamicCast<nvcv::ITensor *>(tensor.handle()));
    EXPECT_EQ(&tensor, &nvcv::DynamicCast<nvcv::ITensor>(tensor.handle()));

    EXPECT_EQ(nullptr, nvcv::DynamicCast<nvcv::TensorWrapData *>(tensor.handle()));

    EXPECT_EQ(nullptr, nvcv::StaticCast<nvcv::ITensor *>(nullptr));
    EXPECT_EQ(nullptr, nvcv::DynamicCast<nvcv::ITensor *>(nullptr));
    EXPECT_THROW(nvcv::DynamicCast<nvcv::ITensor>(nullptr), std::bad_cast);

    // Now when we create the object via C API

    NVCVTensorHandle       handle;
    NVCVTensorRequirements reqs;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorCalcRequirementsForImages(5, 163, 117, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, &reqs));
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorConstruct(&reqs, nullptr, &handle));

    uintptr_t max = 512;

    EXPECT_GE(max, sizeof(nvcv::detail::WrapHandle<nvcv::ITensor>)) << "Must be big enough for the WrapHandle";

    void *cxxPtr = &max;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorGetUserPointer((NVCVTensorHandle)(((uintptr_t)handle) | 1), &cxxPtr));
    ASSERT_NE(&max, cxxPtr) << "Pointer must have been changed";

    // Buffer too big, bail.
    max    = 513;
    cxxPtr = &max;
    ASSERT_EQ(NVCV_ERROR_INTERNAL, nvcvTensorGetUserPointer((NVCVTensorHandle)(((uintptr_t)handle) | 1), &cxxPtr))
        << "Required WrapHandle buffer storage should have been too big";

    nvcv::ITensor *ptensor = nvcv::StaticCast<nvcv::ITensor *>(handle);
    ASSERT_NE(nullptr, ptensor);
    EXPECT_EQ(handle, ptensor->handle());
    ASSERT_EQ(4, ptensor->rank());
    EXPECT_EQ(4, ptensor->shape()[3]);
    EXPECT_EQ(163, ptensor->shape()[2]);
    EXPECT_EQ(117, ptensor->shape()[1]);
    EXPECT_EQ(5, ptensor->shape()[0]);
    EXPECT_EQ(nvcv::TYPE_U8, ptensor->dtype());

    EXPECT_EQ(ptensor, nvcv::DynamicCast<nvcv::ITensor *>(handle));
    EXPECT_EQ(nullptr, nvcv::DynamicCast<nvcv::Tensor *>(handle));

    nvcvTensorDestroy(handle);
}

TEST(Tensor, wip_user_pointer)
{
    nvcv::Tensor tensor(3, {163, 117}, nvcv::FMT_RGBA8);
    EXPECT_EQ(nullptr, tensor.userPointer());

    void *cxxPtr;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorGetUserPointer((NVCVTensorHandle)(((uintptr_t)tensor.handle()) | 1), &cxxPtr));
    ASSERT_EQ(&tensor, cxxPtr) << "cxx object pointer must always be associated with the corresponding handle";

    tensor.setUserPointer((void *)0x123);
    EXPECT_EQ((void *)0x123, tensor.userPointer());

    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorGetUserPointer((NVCVTensorHandle)(((uintptr_t)tensor.handle()) | 1), &cxxPtr));
    ASSERT_EQ(&tensor, cxxPtr) << "cxx object pointer must always be associated with the corresponding handle";

    tensor.setUserPointer(nullptr);
    EXPECT_EQ(nullptr, tensor.userPointer());

    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorGetUserPointer((NVCVTensorHandle)(((uintptr_t)tensor.handle()) | 1), &cxxPtr));
    ASSERT_EQ(&tensor, cxxPtr) << "cxx object pointer must always be associated with the corresponding handle";
}

TEST(TensorWrapData, wip_create)
{
    nvcv::ImageFormat fmt
        = nvcv::ImageFormat(nvcv::ColorModel::RGB, nvcv::CSPEC_BT601_ER, nvcv::MemLayout::PL, nvcv::DataKind::FLOAT,
                            nvcv::Swizzle::S_XY00, nvcv::Packing::X16, nvcv::Packing::X16);
    nvcv::DataType GOLD_DTYPE = fmt.planeDataType(0);

    nvcv::Tensor origTensor(5, {173, 79}, fmt, nvcv::MemAlignment{}.rowAddr(1).baseAddr(32)); // packed rows

    auto *tdata = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(origTensor.exportData());

    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*tdata);
    ASSERT_TRUE(access);

    EXPECT_EQ(nvcv::TensorLayout::NCHW, tdata->layout());
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

    nvcv::TensorWrapData tensor{*tdata};

    ASSERT_NE(nullptr, tensor.handle());

    EXPECT_EQ(tdata->shape(), tensor.shape());
    EXPECT_EQ(tdata->layout(), tensor.layout());
    EXPECT_EQ(tdata->rank(), tensor.rank());
    EXPECT_EQ(GOLD_DTYPE, tensor.dtype());

    const nvcv::ITensorData *data = tensor.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(data);
    ASSERT_NE(nullptr, devdata);

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
        {{61,23}, nvcv::FMT_RGBA8p, nvcv::TensorShape{{1,4,23,61},nvcv::TensorLayout::NCHW}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_RGBA8, nvcv::TensorShape{{1,23,61,4},nvcv::TensorLayout::NHWC}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_RGB8, nvcv::TensorShape{{1,23,61,3},nvcv::TensorLayout::NHWC}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_RGB8p, nvcv::TensorShape{{1,3,23,61},nvcv::TensorLayout::NCHW}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_F32, nvcv::TensorShape{{1,1,23,61},nvcv::TensorLayout::NCHW}, nvcv::TYPE_F32},
        {{61,23}, nvcv::FMT_2F32, nvcv::TensorShape{{1,23,61,2},nvcv::TensorLayout::NHWC}, nvcv::TYPE_F32},
    }
);

// clang-format on

TEST_P(TensorWrapImageTests, wip_create)
{
    const nvcv::Size2D      PARAM_SIZE   = std::get<0>(GetParam());
    const nvcv::ImageFormat PARAM_FORMAT = std::get<1>(GetParam());
    const nvcv::TensorShape GOLD_SHAPE   = std::get<2>(GetParam());
    const nvcv::DataType    GOLD_DTYPE   = std::get<3>(GetParam());

    nvcv::Image img(PARAM_SIZE, PARAM_FORMAT);

    nvcv::TensorWrapImage tensor(img);

    EXPECT_EQ(GOLD_SHAPE, tensor.shape());
    EXPECT_EQ(GOLD_DTYPE, tensor.dtype());

    auto *imgData    = dynamic_cast<const nvcv::IImageDataStridedCuda *>(img.exportData());
    auto *tensorData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(tensor.exportData());

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

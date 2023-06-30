/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "DeviceTensorWrap.hpp" // to test in device

#include <common/HashUtils.hpp>      // for NVCV_INSTANTIATE_TEST_SUITE_P, etc.
#include <common/TypedTests.hpp>     // for NVCV_TYPED_TEST_SUITE, etc.
#include <common/ValueTests.hpp>     // for StringLiteral
#include <nvcv/Image.hpp>            // for Image, etc.
#include <nvcv/Tensor.hpp>           // for Tensor, etc.
#include <nvcv/TensorDataAccess.hpp> // for TensorDataAccessStridedImagePlanar, etc.
#include <nvcv/cuda/MathOps.hpp>     // for operator == to allow EXPECT_EQ

#include <limits>

namespace t     = ::testing;
namespace test  = nvcv::test;
namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

// -------------------------- Testing Tensor1DWrap -----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor1DWrapTest, ttype::Types<
    ttype::Types<ttype::Value<Array<int, 2>{-5, 1}>>,
    ttype::Types<ttype::Value<Array<short3, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<Array<float1, 4>{
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<Array<uchar4, 3>{
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor1DWrapTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::Tensor1DWrap<const ValueType> wrap(input.data());

    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr())>);
    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr())>>);

    EXPECT_EQ(wrap.ptr(), input.data());

    for (int i = 0; i < InputType::kShapes[0]; ++i)
    {
        EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(i))>);
        EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(i))>>);

        EXPECT_EQ(wrap.ptr(i), &input[i]);

        int1 c1{i};

        EXPECT_TRUE(std::is_reference_v<decltype(wrap[c1])>);
        EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(wrap[c1])>>);

        EXPECT_EQ(wrap[c1], input[i]);
    }
}

TYPED_TEST(Tensor1DWrapTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensorWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor1DWrapCopyTest, ttype::Types<
    ttype::Types<ttype::Value<Array<int, 2>{}>, ttype::Value<Array<int, 2>{
       -5, 1}>>,
    ttype::Types<ttype::Value<Array<short3, 2>{}>, ttype::Value<Array<short3, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<Array<float1, 4>{}>, ttype::Value<Array<float1, 4>{
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<Array<uchar4, 3>{}>, ttype::Value<Array<uchar4, 3>{
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor1DWrapCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::Tensor1DWrap<ValueType> wrap(test.data());

    ASSERT_EQ(InputType::kShapes[0], decltype(gold)::kShapes[0]);

    for (int i = 0; i < InputType::kShapes[0]; ++i)
    {
        int1 c{i};

        wrap[c] = gold[i];
    }

    EXPECT_EQ(test, gold);
}

// The death tests below are to be run in debug mode only

#ifndef NDEBUG

TEST(Tensor1DWrapWrongCompileStrideDeathTest, it_dies)
{
    using DataType = uint8_t;
    nvcv::DataType dt{NVCV_DATA_TYPE_U8};

    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.strides[0] = 2;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(123);

    auto tdata = nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{1}, "J"},
        dt, buf
    };
    auto tensor = nvcv::TensorWrapData(tdata);

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nvcv::NullOpt);

    EXPECT_DEATH({ cuda::Tensor1DWrap<DataType> wrap(*dev); }, "");
}

#endif

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor1DWrapTensorTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_DATA_TYPE_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_DATA_TYPE_U8>>,
    ttype::Types<const short2, ttype::Value<NVCV_DATA_TYPE_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_DATA_TYPE_3U8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_DATA_TYPE_4U8>>,
    ttype::Types<float3, ttype::Value<NVCV_DATA_TYPE_3F32>>,
    ttype::Types<const float4, ttype::Value<NVCV_DATA_TYPE_4F32>>
>);

// clang-format on

TYPED_TEST(Tensor1DWrapTensorTest, correct_with_tensor)
{
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto dataType   = ttype::GetValue<TypeParam, 1>;

    nvcv::Tensor tensor({{123}, "N"}, nvcv::DataType{dataType});

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::Tensor1DWrap<ValueType> wrap(*dev);

    const ValueType *ptr0 = reinterpret_cast<const ValueType *>(dev->basePtr());
    const ValueType *ptr1
        = reinterpret_cast<const ValueType *>(reinterpret_cast<const uint8_t *>(dev->basePtr()) + dev->stride(0));

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
}

TYPED_TEST(Tensor1DWrapTensorTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto dataType   = ttype::GetValue<TypeParam, 1>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor tensor({{123}, "N"}, nvcv::DataType{dataType});

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nvcv::NullOpt);

    cuda::Tensor1DWrap<ValueType> wrap(*dev);

    int1 size{static_cast<int>(dev->shape(0))};

    DeviceSetOnes(wrap, size, stream);

    std::size_t sizeBytes = dev->shape(0) * dev->stride(0);

    std::vector<uint8_t> test(sizeBytes);
    std::vector<uint8_t> gold(sizeBytes);

    for (int i = 0; i < size.x; i++)
    {
        *reinterpret_cast<ValueType *>(&gold[i * dev->stride(0)]) = cuda::SetAll<ValueType>(1);
    }

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dev->basePtr(), sizeBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test, gold);
}

// -------------------------- Testing Tensor2DWrap -----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor2DWrapTest, ttype::Types<
    ttype::Types<ttype::Value<PackedImage<int, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<PackedImage<short3, 1, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<PackedImage<float1, 2, 4>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedImage<uchar4, 3, 3>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor2DWrapTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::Tensor2DWrap<const ValueType> wrap(input.data(), InputType::kStrides[0]);

    auto strides = wrap.strides();
    EXPECT_EQ(strides[0], InputType::kStrides[0]);

    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr())>);
    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr())>>);

    EXPECT_EQ(wrap.ptr(), input.data());

    for (int y = 0; y < InputType::kShapes[0]; ++y)
    {
        EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(y))>);
        EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(y))>>);

        EXPECT_EQ(wrap.ptr(y), &input[y * InputType::kShapes[1]]);

        for (int x = 0; x < InputType::kShapes[1]; ++x)
        {
            EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(y, x))>);
            EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(y, x))>>);

            EXPECT_EQ(wrap.ptr(y, x), &input[y * InputType::kShapes[1] + x]);

            int2 c2{x, y};

            EXPECT_TRUE(std::is_reference_v<decltype(wrap[c2])>);
            EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(wrap[c2])>>);

            EXPECT_EQ(wrap[c2], input[y * InputType::kShapes[1] + x]);
        }
    }
}

TYPED_TEST(Tensor2DWrapTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensorWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor2DWrapCopyTest, ttype::Types<
    ttype::Types<ttype::Value<PackedImage<int, 2, 2>{}>, ttype::Value<PackedImage<int, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<PackedImage<short3, 1, 2>{}>, ttype::Value<PackedImage<short3, 1, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<PackedImage<float1, 2, 4>{}>, ttype::Value<PackedImage<float1, 2, 4>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedImage<uchar4, 3, 3>{}>, ttype::Value<PackedImage<uchar4, 3, 3>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor2DWrapCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::Tensor2DWrap<ValueType> wrap(test.data(), InputType::kStrides[0]);

    for (int i = 0; i < 2; ++i)
    {
        ASSERT_EQ(InputType::kShapes[i], decltype(gold)::kShapes[i]);
    }

    for (int y = 0; y < InputType::kShapes[0]; ++y)
    {
        for (int x = 0; x < InputType::kShapes[1]; ++x)
        {
            int2 c{x, y};

            wrap[c] = gold[y * InputType::kShapes[1] + x];
        }
    }

    EXPECT_EQ(test, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor2DWrapImageWrapTest, ttype::Types<
    ttype::Types<ttype::Value<NVCV_IMAGE_FORMAT_S32>, ttype::Value<PackedImage<int, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<NVCV_IMAGE_FORMAT_2S16>, ttype::Value<PackedImage<short2, 1, 2>{
        short2{-12, 2}, short2{5678, -2345}}>>,
    ttype::Types<ttype::Value<NVCV_IMAGE_FORMAT_F32>, ttype::Value<PackedImage<float1, 2, 4>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<NVCV_IMAGE_FORMAT_RGBA8>, ttype::Value<PackedImage<uchar4, 3, 3>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor2DWrapImageWrapTest, correct_with_image_wrap)
{
    auto imgFormat = ttype::GetValue<TypeParam, 0>;
    auto input     = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    nvcv::ImageDataStridedCuda::Buffer buf;
    buf.numPlanes           = 1;
    buf.planes[0].width     = InputType::kShapes[1];
    buf.planes[0].height    = InputType::kShapes[0];
    buf.planes[0].rowStride = InputType::kStrides[0];
    buf.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(input.data());

    auto img = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::ImageFormat{imgFormat}, buf});

    auto dev = img.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(dev, nvcv::NullOpt);

    cuda::Tensor2DWrap<ValueType> wrap(*dev);

    auto strides = wrap.strides();
    EXPECT_EQ(strides[0], InputType::kStrides[0]);

    for (int y = 0; y < InputType::kShapes[0]; ++y)
    {
        for (int x = 0; x < InputType::kShapes[1]; ++x)
        {
            int2 c2{x, y};

            EXPECT_EQ(wrap[c2], *reinterpret_cast<ValueType *>(
                                    (dev->plane(0).basePtr + y * dev->plane(0).rowStride + x * sizeof(ValueType))));
        }
    }
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor2DWrapImageTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_IMAGE_FORMAT_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_IMAGE_FORMAT_Y8>>,
    ttype::Types<const short2, ttype::Value<NVCV_IMAGE_FORMAT_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_IMAGE_FORMAT_RGB8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_IMAGE_FORMAT_RGBA8>>,
    ttype::Types<float3, ttype::Value<NVCV_IMAGE_FORMAT_RGBf32>>,
    ttype::Types<const float4, ttype::Value<NVCV_IMAGE_FORMAT_RGBAf32>>
>);

// clang-format on

TYPED_TEST(Tensor2DWrapImageTest, correct_with_image)
{
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nvcv::Image img({213, 211}, nvcv::ImageFormat{imgFormat});

    const auto dev = img.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(dev, nvcv::NullOpt);

    cuda::Tensor2DWrap<ValueType> wrap(*dev);

    auto strides = wrap.strides();
    EXPECT_EQ(strides[0], dev->plane(0).rowStride);

    const ValueType *ptr0 = reinterpret_cast<const ValueType *>(dev->plane(0).basePtr);
    const ValueType *ptr1 = reinterpret_cast<const ValueType *>(dev->plane(0).basePtr + dev->plane(0).rowStride);

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
}

TYPED_TEST(Tensor2DWrapImageTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Image img({357, 642}, nvcv::ImageFormat{imgFormat});

    int width  = img.size().w;
    int height = img.size().h;

    const auto dev = img.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(dev, nvcv::NullOpt);

    cuda::Tensor2DWrap<ValueType> wrap(*dev);

    DeviceSetOnes(wrap, int2{width, height}, stream);

    std::size_t sizeBytes = height * dev->plane(0).rowStride;

    std::vector<uint8_t> test(sizeBytes);
    std::vector<uint8_t> gold(sizeBytes);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            *reinterpret_cast<ValueType *>(&gold[i * dev->plane(0).rowStride + j * sizeof(ValueType)])
                = cuda::SetAll<ValueType>(1);
        }
    }

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dev->plane(0).basePtr, sizeBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test, gold);
}

// --------------------------- Testing Tensor3DWrap ----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor3DWrapTest, ttype::Types<
    ttype::Types<ttype::Value<PackedTensor3D<int, 1, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<PackedTensor3D<short3, 2, 2, 1>{
        short3{-12, 2, -34}, short3{5678, -2345, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}}>>,
    ttype::Types<ttype::Value<PackedTensor3D<float1, 2, 2, 2>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedTensor3D<uchar4, 3, 3, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor3DWrapTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::Tensor3DWrap<const ValueType> wrap(input.data(), InputType::kStrides[0], InputType::kStrides[1]);

    auto strides = wrap.strides();
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_EQ(strides[i], InputType::kStrides[i]);
    }

    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr())>);
    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr())>>);

    EXPECT_EQ(wrap.ptr(), input.data());

    for (int b = 0; b < InputType::kShapes[0]; ++b)
    {
        EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b))>);
        EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b))>>);

        EXPECT_EQ(wrap.ptr(b), &input[b * InputType::kShapes[2] * InputType::kShapes[1]]);

        for (int y = 0; y < InputType::kShapes[1]; ++y)
        {
            EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b, y))>);
            EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b, y))>>);

            EXPECT_EQ(wrap.ptr(b, y),
                      &input[b * InputType::kShapes[2] * InputType::kShapes[1] + y * InputType::kShapes[2]]);

            for (int x = 0; x < InputType::kShapes[2]; ++x)
            {
                EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b, y, x))>);
                EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b, y, x))>>);

                EXPECT_EQ(wrap.ptr(b, y, x),
                          &input[b * InputType::kShapes[2] * InputType::kShapes[1] + y * InputType::kShapes[2] + x]);

                int3 c3{x, y, b};

                EXPECT_TRUE(std::is_reference_v<decltype(wrap[c3])>);
                EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(wrap[c3])>>);

                EXPECT_EQ(wrap[c3],
                          input[b * InputType::kShapes[2] * InputType::kShapes[1] + y * InputType::kShapes[2] + x]);
            }
        }
    }
}

TYPED_TEST(Tensor3DWrapTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensorWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor3DWrapCopyTest, ttype::Types<
    ttype::Types<ttype::Value<PackedTensor3D<int, 1, 2, 2>{}>, ttype::Value<PackedTensor3D<int, 1, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<PackedTensor3D<short3, 2, 2, 1>{}>, ttype::Value<PackedTensor3D<short3, 2, 2, 1>{
        short3{-12, 2, -34}, short3{5678, -2345, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}}>>,
    ttype::Types<ttype::Value<PackedTensor3D<float1, 2, 2, 2>{}>, ttype::Value<PackedTensor3D<float1, 2, 2, 2>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedTensor3D<uchar4, 3, 3, 1>{}>, ttype::Value<PackedTensor3D<uchar4, 3, 3, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor3DWrapCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::Tensor3DWrap<ValueType> wrap(test.data(), InputType::kStrides[0], InputType::kStrides[1]);

    for (int i = 0; i < 3; ++i)
    {
        ASSERT_EQ(InputType::kShapes[i], decltype(gold)::kShapes[i]);
    }

    for (int b = 0; b < InputType::kShapes[0]; ++b)
    {
        for (int y = 0; y < InputType::kShapes[1]; ++y)
        {
            for (int x = 0; x < InputType::kShapes[2]; ++x)
            {
                int3 c{x, y, b};

                wrap[c] = gold[b * InputType::kShapes[1] * InputType::kShapes[2] + y * InputType::kShapes[2] + x];
            }
        }
    }

    EXPECT_EQ(test, gold);
}

// The death tests below are to be run in debug mode only

#ifndef NDEBUG

TEST(Tensor3DWrapBigPitchDeathTest, it_dies)
{
    using DataType = uint8_t;
    int64_t height = 2;
    int64_t width  = std::numeric_limits<int>::max();

    nvcv::DataType                      dt{NVCV_DATA_TYPE_U8};
    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.strides[2] = sizeof(DataType);
    buf.strides[1] = width * buf.strides[2];
    buf.strides[0] = height * buf.strides[1];
    buf.basePtr    = reinterpret_cast<NVCVByte *>(123);

    auto tensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{1, height, width}, "NHW"},
        dt, buf
    });

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    EXPECT_DEATH({ cuda::Tensor3DWrap<DataType> wrap(*dev); }, "");
}

#endif

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor3DWrapTensorWrapTest, ttype::Types<
    ttype::Types<ttype::Value<NVCV_DATA_TYPE_S32>, ttype::Value<PackedTensor3D<int, 1, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<NVCV_DATA_TYPE_2S16>, ttype::Value<PackedTensor3D<short2, 2, 2, 1>{
        short2{-12, 2}, short2{5678, -2345},
        short2{123, 0}, short2{-9876, 4321}}>>,
    ttype::Types<ttype::Value<NVCV_DATA_TYPE_F32>, ttype::Value<PackedTensor3D<float1, 2, 2, 2>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<NVCV_DATA_TYPE_4U8>, ttype::Value<PackedTensor3D<uchar4, 3, 3, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor3DWrapTensorWrapTest, correct_with_tensor_wrap)
{
    auto dataType = ttype::GetValue<TypeParam, 0>;
    auto input    = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    int N = InputType::kShapes[0];
    int H = InputType::kShapes[1];
    int W = InputType::kShapes[2];

    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.strides[0] = InputType::kStrides[0];
    buf.strides[1] = InputType::kStrides[1];
    buf.strides[2] = InputType::kStrides[2];
    buf.basePtr    = reinterpret_cast<NVCVByte *>(input.data());

    auto tensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{N, H, W}, "NHW"},
        nvcv::DataType{ dataType      },
        buf
    });

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::Tensor3DWrap<ValueType> wrap(*dev);

    auto strides = wrap.strides();
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_EQ(strides[i], InputType::kStrides[i]);
    }

    for (int b = 0; b < N; ++b)
    {
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                int3 c3{x, y, b};

                EXPECT_EQ(wrap[c3], *reinterpret_cast<ValueType *>(dev->basePtr() + b * dev->stride(0)
                                                                   + y * dev->stride(1) + x * dev->stride(2)));
            }
        }
    }
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor3DWrapTensorTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_IMAGE_FORMAT_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_IMAGE_FORMAT_Y8>>,
    ttype::Types<const short2, ttype::Value<NVCV_IMAGE_FORMAT_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_IMAGE_FORMAT_RGB8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_IMAGE_FORMAT_RGBA8>>,
    ttype::Types<float3, ttype::Value<NVCV_IMAGE_FORMAT_RGBf32>>,
    ttype::Types<const float4, ttype::Value<NVCV_IMAGE_FORMAT_RGBAf32>>
>);

// clang-format on

TYPED_TEST(Tensor3DWrapTensorTest, correct_with_tensor)
{
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nvcv::Tensor tensor(3, {213, 211}, nvcv::ImageFormat{imgFormat});

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::Tensor3DWrap<ValueType> wrap(*dev);

    auto strides = wrap.strides();
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_EQ(strides[i], dev->stride(i));
    }

    const ValueType *ptr0  = reinterpret_cast<const ValueType *>(dev->basePtr());
    const ValueType *ptr1  = reinterpret_cast<const ValueType *>(dev->basePtr() + dev->stride(0));
    const ValueType *ptr12 = reinterpret_cast<const ValueType *>(dev->basePtr() + dev->stride(0) + 2 * dev->stride(1));

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
    EXPECT_EQ(wrap.ptr(1, 2), ptr12);
}

TYPED_TEST(Tensor3DWrapTensorTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nvcv::Tensor tensor(5, {234, 567}, nvcv::ImageFormat{imgFormat});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nvcv::NullOpt);

    cuda::Tensor3DWrap<ValueType> wrap(*dev);

    int3 size{static_cast<int>(dev->shape(2)), static_cast<int>(dev->shape(1)), static_cast<int>(dev->shape(0))};

    DeviceSetOnes(wrap, size, stream);

    auto strides = wrap.strides();
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_EQ(strides[i], dev->stride(i));
    }

    std::size_t sizeBytes = dev->shape(0) * dev->stride(0);

    std::vector<uint8_t> test(sizeBytes);
    std::vector<uint8_t> gold(sizeBytes);

    for (int b = 0; b < dev->shape(0); b++)
    {
        for (int i = 0; i < dev->shape(1); i++)
        {
            for (int j = 0; j < dev->shape(2); j++)
            {
                *reinterpret_cast<ValueType *>(&gold[b * dev->stride(0) + i * dev->stride(1) + j * dev->stride(2)])
                    = cuda::SetAll<ValueType>(1);
            }
        }
    }

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dev->basePtr(), sizeBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test, gold);
}

// --------------------------- Testing Tensor4DWrap ----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor4DWrapTest, ttype::Types<
    ttype::Types<ttype::Value<PackedTensor4D<int, 1, 2, 2, 2>{
        2, 3, 4, 5
       -5, 1, 6, 7}>>,
    ttype::Types<ttype::Value<PackedTensor4D<short3, 2, 2, 1, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}, short3{-1, -2, -3}, short3{-567, 234, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}, short3{1, 2, 3}, short3{-56, 23, 1}}>>,
    ttype::Types<ttype::Value<PackedTensor4D<float1, 2, 2, 2, 1>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedTensor4D<uchar4, 3, 3, 1, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor4DWrapTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::Tensor4DWrap<const ValueType> wrap(input.data(), InputType::kStrides[0], InputType::kStrides[1],
                                             InputType::kStrides[2]);

    auto strides = wrap.strides();
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(strides[i], InputType::kStrides[i]);
    }

    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr())>);
    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr())>>);

    EXPECT_EQ(wrap.ptr(), input.data());

    for (int b = 0; b < InputType::kShapes[0]; ++b)
    {
        EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b))>);
        EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b))>>);

        EXPECT_EQ(wrap.ptr(b), &input[b * InputType::kShapes[1] * InputType::kShapes[2] * InputType::kShapes[3]]);

        for (int y = 0; y < InputType::kShapes[1]; ++y)
        {
            EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b, y))>);
            EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b, y))>>);

            EXPECT_EQ(wrap.ptr(b, y), &input[b * InputType::kShapes[1] * InputType::kShapes[2] * InputType::kShapes[3]
                                             + y * InputType::kShapes[2] * InputType::kShapes[3]]);

            for (int x = 0; x < InputType::kShapes[2]; ++x)
            {
                EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b, y, x))>);
                EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b, y, x))>>);

                EXPECT_EQ(wrap.ptr(b, y, x),
                          &input[b * InputType::kShapes[1] * InputType::kShapes[2] * InputType::kShapes[3]
                                 + y * InputType::kShapes[2] * InputType::kShapes[3] + x * InputType::kShapes[3]]);

                for (int k = 0; k < InputType::kShapes[3]; ++k)
                {
                    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b, y, x, k))>);
                    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b, y, x, k))>>);

                    EXPECT_EQ(
                        wrap.ptr(b, y, x, k),
                        &input[b * InputType::kShapes[1] * InputType::kShapes[2] * InputType::kShapes[3]
                               + y * InputType::kShapes[2] * InputType::kShapes[3] + x * InputType::kShapes[3] + k]);

                    int4 c4{k, x, y, b};

                    EXPECT_TRUE(std::is_reference_v<decltype(wrap[c4])>);
                    EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(wrap[c4])>>);

                    EXPECT_EQ(
                        wrap[c4],
                        input[b * InputType::kShapes[1] * InputType::kShapes[2] * InputType::kShapes[3]
                              + y * InputType::kShapes[2] * InputType::kShapes[3] + x * InputType::kShapes[3] + k]);
                }
            }
        }
    }
}

TYPED_TEST(Tensor4DWrapTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensorWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor4DWrapCopyTest, ttype::Types<
    ttype::Types<ttype::Value<PackedTensor4D<int, 1, 2, 2, 2>{}>, ttype::Value<PackedTensor4D<int, 1, 2, 2, 2>{
        2, 3, 4, 5,
       -5, 1, 6, 7}>>,
    ttype::Types<ttype::Value<PackedTensor4D<short3, 2, 2, 1, 2>{}>, ttype::Value<PackedTensor4D<short3, 2, 2, 1, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}, short3{-1, -2, -3}, short3{-567, 234, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}, short3{1, 2, 3}, short3{-56, 23, 1}}>>,
    ttype::Types<ttype::Value<PackedTensor4D<float1, 2, 2, 2, 1>{}>, ttype::Value<PackedTensor4D<float1, 2, 2, 2, 1>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedTensor4D<uchar4, 3, 3, 1, 1>{}>, ttype::Value<PackedTensor4D<uchar4, 3, 3, 1, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor4DWrapCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::Tensor4DWrap<ValueType> wrap(test.data(), InputType::kStrides[0], InputType::kStrides[1],
                                       InputType::kStrides[2]);

    for (int i = 0; i < 4; ++i)
    {
        ASSERT_EQ(InputType::kShapes[i], decltype(gold)::kShapes[i]);
    }

    for (int b = 0; b < InputType::kShapes[0]; ++b)
    {
        for (int y = 0; y < InputType::kShapes[1]; ++y)
        {
            for (int x = 0; x < InputType::kShapes[2]; ++x)
            {
                for (int k = 0; k < InputType::kShapes[3]; ++k)
                {
                    int4 c{k, x, y, b};

                    wrap[c] = gold[b * InputType::kShapes[1] * InputType::kShapes[2] * InputType::kShapes[3]
                                   + y * InputType::kShapes[2] * InputType::kShapes[3] + x * InputType::kShapes[3] + k];
                }
            }
        }
    }

    EXPECT_EQ(test, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor4DWrapTensorWrapTest, ttype::Types<
    ttype::Types<ttype::Value<NVCV_DATA_TYPE_S32>, ttype::Value<PackedTensor4D<int, 1, 2, 2, 2>{
        2, 3, 4, 5,
       -5, 1, 6, 7}>>,
    ttype::Types<ttype::Value<NVCV_DATA_TYPE_2S16>, ttype::Value<PackedTensor4D<short2, 2, 2, 1, 2>{
        short2{-12, 2}, short2{5678, -2345}, short2{123, -321}, short2{-567, 234},
        short2{123, 0}, short2{-9876, 4321}, short2{12, -32}, short2{567, -234}}>>,
    ttype::Types<ttype::Value<NVCV_DATA_TYPE_F32>, ttype::Value<PackedTensor4D<float1, 2, 2, 2, 1>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<NVCV_DATA_TYPE_4U8>, ttype::Value<PackedTensor4D<uchar4, 3, 3, 1, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor4DWrapTensorWrapTest, correct_with_tensor_wrap)
{
    auto dataType = ttype::GetValue<TypeParam, 0>;
    auto input    = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    int N = InputType::kShapes[0];
    int H = InputType::kShapes[1];
    int W = InputType::kShapes[2];
    int C = InputType::kShapes[3];

    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.strides[0] = InputType::kStrides[0];
    buf.strides[1] = InputType::kStrides[1];
    buf.strides[2] = InputType::kStrides[2];
    buf.strides[3] = InputType::kStrides[3];
    buf.basePtr    = reinterpret_cast<NVCVByte *>(input.data());

    auto tensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{N, H, W, C}, "NHWC"},
        nvcv::DataType{    dataType       },
        buf
    });

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::Tensor4DWrap<ValueType> wrap(*dev);

    auto strides = wrap.strides();
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(strides[i], InputType::kStrides[i]);
    }

    for (int b = 0; b < N; ++b)
    {
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                for (int k = 0; k < C; ++k)
                {
                    int4 c4{k, x, y, b};

                    EXPECT_EQ(wrap[c4],
                              *reinterpret_cast<ValueType *>(dev->basePtr() + b * dev->stride(0) + y * dev->stride(1)
                                                             + x * dev->stride(2) + k * dev->stride(3)));
                }
            }
        }
    }
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor4DWrapTensorTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_DATA_TYPE_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_DATA_TYPE_U8>>,
    ttype::Types<const short2, ttype::Value<NVCV_DATA_TYPE_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_DATA_TYPE_3U8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_DATA_TYPE_4U8>>,
    ttype::Types<float3, ttype::Value<NVCV_DATA_TYPE_3F32>>,
    ttype::Types<const float4, ttype::Value<NVCV_DATA_TYPE_4F32>>
>);

// clang-format on

TYPED_TEST(Tensor4DWrapTensorTest, correct_with_tensor)
{
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto dataType   = ttype::GetValue<TypeParam, 1>;

    nvcv::Tensor tensor(
        {
            {3, 213, 211, 4},
            "NHWC"
    },
        nvcv::DataType{dataType});

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::Tensor4DWrap<ValueType> wrap(*dev);

    auto strides = wrap.strides();
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(strides[i], dev->stride(i));
    }

    const ValueType *ptr0   = reinterpret_cast<const ValueType *>(dev->basePtr());
    const ValueType *ptr1   = reinterpret_cast<const ValueType *>(dev->basePtr() + dev->stride(0));
    const ValueType *ptr12  = reinterpret_cast<const ValueType *>(dev->basePtr() + dev->stride(0) + 2 * dev->stride(1));
    const ValueType *ptr123 = reinterpret_cast<const ValueType *>(dev->basePtr() + dev->stride(0) + 2 * dev->stride(1)
                                                                  + 3 * dev->stride(2));

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
    EXPECT_EQ(wrap.ptr(1, 2), ptr12);
    EXPECT_EQ(wrap.ptr(1, 2, 3), ptr123);
}

TYPED_TEST(Tensor4DWrapTensorTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto dataType   = ttype::GetValue<TypeParam, 1>;

    nvcv::Tensor tensor(
        {
            {3, 213, 211, 4},
            "NHWC"
    },
        nvcv::DataType{dataType});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nvcv::NullOpt);

    cuda::Tensor4DWrap<ValueType> wrap(*dev);

    int4 size{static_cast<int>(dev->shape(2)), static_cast<int>(dev->shape(1)), static_cast<int>(dev->shape(0)),
              static_cast<int>(dev->shape(3))};

    DeviceSetOnes(wrap, size, stream);

    auto strides = wrap.strides();
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(strides[i], dev->stride(i));
    }

    size_t sizeBytes = dev->shape(0) * dev->stride(0);

    std::vector<uint8_t> test(sizeBytes);
    std::vector<uint8_t> gold(sizeBytes);

    for (int b = 0; b < dev->shape(0); b++)
    {
        for (int i = 0; i < dev->shape(1); i++)
        {
            for (int j = 0; j < dev->shape(2); j++)
            {
                for (int k = 0; k < dev->shape(3); k++)
                {
                    *reinterpret_cast<ValueType *>(
                        &gold[b * dev->stride(0) + i * dev->stride(1) + j * dev->stride(2) + k * dev->stride(3)])
                        = cuda::SetAll<ValueType>(1);
                }
            }
        }
    }

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dev->basePtr(), sizeBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test, gold);
}

// --------------------------- Testing TensorNDWrap ----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    TensorNDWrapTest, ttype::Types<
    ttype::Types<ttype::Value<Array<int, 11>{}>>,
    ttype::Types<ttype::Value<PackedImage<int, 2, 13>{}>>,
    ttype::Types<ttype::Value<PackedTensor3D<int, 2, 3, 7>{}>>,
    ttype::Types<ttype::Value<PackedTensor4D<int, 3, 2, 4, 5>{}>>
>);

// clang-format on

TYPED_TEST(TensorNDWrapTest, correct_dimensionality)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);

    using TW = cuda::TensorNDWrap<const typename InputType::value_type, InputType::kNumDim>;

    EXPECT_EQ(TW::kNumDimensions, InputType::kNumDim);
}

// ----------------------- Testing CreateTensorWrapNHWx ------------------------

class CreateTensorWrapNHWxTests
    : public t::TestWithParam<std::tuple<test::Param<"shape", nvcv::TensorShape>, test::Param<"dtype", nvcv::DataType>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, CreateTensorWrapNHWxTests,
    test::ValueList<nvcv::TensorShape, nvcv::DataType>
    {
        {nvcv::TensorShape{{7, 3, 3}, nvcv::TENSOR_HWC}, nvcv::TYPE_U8},
        {nvcv::TensorShape{{5, 3, 2, 4}, nvcv::TENSOR_NHWC}, nvcv::TYPE_U8}
    }
);

// clang-format on

TEST_P(CreateTensorWrapNHWxTests, correct_properties_in_nhw)
{
    auto tensorShape    = std::get<0>(GetParam());
    auto tensorDataType = std::get<1>(GetParam());

    nvcv::Tensor tensor(tensorShape, tensorDataType);

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nvcv::NullOpt);

    auto wrap = cuda::CreateTensorWrapNHW<const unsigned char>(*dev);

    using TW = decltype(wrap);

    EXPECT_EQ(TW::kNumDimensions, 3);
    EXPECT_EQ(TW::kVariableStrides, 2);
    EXPECT_EQ(TW::kConstantStrides, 1);

    EXPECT_EQ(wrap.ptr(), reinterpret_cast<unsigned char *>(dev->basePtr()));

    auto tensorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dev);
    NVCV_ASSERT(tensorAccess);

    EXPECT_EQ(wrap.strides()[0], tensorAccess->sampleStride());
    EXPECT_EQ(wrap.strides()[1], tensorAccess->rowStride());
}

TEST_P(CreateTensorWrapNHWxTests, correct_properties_in_nhwc)
{
    auto tensorShape    = std::get<0>(GetParam());
    auto tensorDataType = std::get<1>(GetParam());

    nvcv::Tensor tensor(tensorShape, tensorDataType);

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nvcv::NullOpt);

    auto wrap = cuda::CreateTensorWrapNHWC<const unsigned char>(*dev);

    using TW = decltype(wrap);

    EXPECT_EQ(TW::kNumDimensions, 4);
    EXPECT_EQ(TW::kVariableStrides, 3);
    EXPECT_EQ(TW::kConstantStrides, 1);

    EXPECT_EQ(wrap.ptr(), reinterpret_cast<unsigned char *>(dev->basePtr()));

    auto tensorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dev);
    NVCV_ASSERT(tensorAccess);

    EXPECT_EQ(wrap.strides()[0], tensorAccess->sampleStride());
    EXPECT_EQ(wrap.strides()[1], tensorAccess->rowStride());
    EXPECT_EQ(wrap.strides()[2], tensorAccess->colStride());
}

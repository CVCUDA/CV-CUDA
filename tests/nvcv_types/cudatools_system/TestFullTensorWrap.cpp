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

#include "DeviceBorderWrap.hpp"     // to test in device
#include "DeviceFullTensorWrap.hpp" // to test in device

#include <common/BorderUtils.hpp>    // for test::IsInside, etc.
#include <common/TypedTests.hpp>     // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/Tensor.hpp>           // for Tensor, etc.
#include <nvcv/TensorDataAccess.hpp> // for TensorDataAccessStridedImagePlanar, etc.
#include <nvcv/cuda/BorderWrap.hpp>  // for BorderWrap, etc.
#include <nvcv/cuda/MathOps.hpp>     // for operator == to allow EXPECT_EQ

#include <limits>
#include <random>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

// ------------------------ Testing FullTensorWrap 1D --------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap1DTest, ttype::Types<
    ttype::Types<ttype::Value<Array<int, 2>{-5, 1}>>,
    ttype::Types<ttype::Value<Array<short3, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<Array<float1, 4>{
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<Array<uchar4, 3>{
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(FullTensorWrap1DTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::FullTensorWrap<const ValueType, 1> wrap(input.data(), InputType::kStrides, InputType::kShapes);

    EXPECT_EQ(wrap.strides()[0], InputType::kStrides[0]);
    EXPECT_EQ(wrap.shapes()[0], InputType::kShapes[0]);

    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr())>);
    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr())>>);

    EXPECT_EQ(wrap.ptr(), input.data());

    for (int i = 0; i < InputType::kShapes[0]; ++i)
    {
        EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(i))>);
        EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(i))>>);

        EXPECT_EQ(wrap.ptr(i), &input[i]);

        int c1{i};

        EXPECT_TRUE(std::is_reference_v<decltype(wrap[c1])>);
        EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(wrap[c1])>>);

        EXPECT_EQ(wrap[c1], input[i]);
    }
}

TYPED_TEST(FullTensorWrap1DTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseFullTensorWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap1DCopyTest, ttype::Types<
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

TYPED_TEST(FullTensorWrap1DCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::FullTensorWrap<ValueType, 1> wrap(test.data(), InputType::kStrides, InputType::kShapes);

    ASSERT_EQ(InputType::kShapes[0], decltype(gold)::kShapes[0]);

    for (int i = 0; i < InputType::kShapes[0]; ++i)
    {
        int c{i};

        wrap[c] = gold[i];
    }

    EXPECT_EQ(test, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap1DTensorTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_DATA_TYPE_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_DATA_TYPE_U8>>,
    ttype::Types<const short2, ttype::Value<NVCV_DATA_TYPE_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_DATA_TYPE_3U8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_DATA_TYPE_4U8>>,
    ttype::Types<float3, ttype::Value<NVCV_DATA_TYPE_3F32>>,
    ttype::Types<const float4, ttype::Value<NVCV_DATA_TYPE_4F32>>
>);

// clang-format on

TYPED_TEST(FullTensorWrap1DTensorTest, correct_with_tensor)
{
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto dataType   = ttype::GetValue<TypeParam, 1>;

    nvcv::Tensor tensor({{123}, "N"}, nvcv::DataType{dataType});

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::FullTensorWrap<ValueType, 1> wrap(*dev);

    EXPECT_EQ(wrap.strides()[0], dev->stride(0));
    EXPECT_EQ(wrap.shapes()[0], dev->shape(0));

    const ValueType *ptr0 = reinterpret_cast<const ValueType *>(dev->basePtr());
    const ValueType *ptr1
        = reinterpret_cast<const ValueType *>(reinterpret_cast<const uint8_t *>(dev->basePtr()) + dev->stride(0));

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
}

TYPED_TEST(FullTensorWrap1DTensorTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto dataType   = ttype::GetValue<TypeParam, 1>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor tensor({{123}, "N"}, nvcv::DataType{dataType});

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::FullTensorWrap<ValueType, 1> wrap(*dev);

    DeviceSetOnes(wrap, stream);

    EXPECT_EQ(wrap.strides()[0], dev->stride(0));
    EXPECT_EQ(wrap.shapes()[0], dev->shape(0));

    std::size_t sizeBytes = dev->shape(0) * dev->stride(0);

    std::vector<uint8_t> test(sizeBytes);
    std::vector<uint8_t> gold(sizeBytes);

    for (int i = 0; i < dev->shape(0); i++)
    {
        *reinterpret_cast<ValueType *>(&gold[i * dev->stride(0)]) = cuda::SetAll<ValueType>(1);
    }

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dev->basePtr(), sizeBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test, gold);
}

// ------------------------ Testing FullTensorWrap 2D --------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap2DTest, ttype::Types<
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

TYPED_TEST(FullTensorWrap2DTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::FullTensorWrap<const ValueType, 2> wrap(input.data(), InputType::kStrides, InputType::kShapes);

    for (int i = 0; i < 2; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], InputType::kStrides[i]);
        EXPECT_EQ(wrap.shapes()[i], InputType::kShapes[i]);
    }

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

TYPED_TEST(FullTensorWrap2DTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensorWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap2DCopyTest, ttype::Types<
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

TYPED_TEST(FullTensorWrap2DCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::FullTensorWrap<ValueType, 2> wrap(test.data(), InputType::kStrides, InputType::kShapes);

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
    FullTensorWrap2DTensorWrapTest, ttype::Types<
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

TYPED_TEST(FullTensorWrap2DTensorWrapTest, correct_with_tensor_wrap)
{
    auto dataType = ttype::GetValue<TypeParam, 0>;
    auto input    = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    int H = InputType::kShapes[0];
    int W = InputType::kShapes[1];

    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.strides[0] = InputType::kStrides[0];
    buf.strides[1] = InputType::kStrides[1];
    buf.basePtr    = reinterpret_cast<NVCVByte *>(input.data());

    auto tensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{  {H, W}, "HW"},
        nvcv::DataType{dataType     },
        buf
    });

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::FullTensorWrap<ValueType, 2> wrap(*dev);

    for (int i = 0; i < 2; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], InputType::kStrides[i]);
        EXPECT_EQ(wrap.shapes()[i], InputType::kShapes[i]);
    }

    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            int2 c2{x, y};

            EXPECT_EQ(wrap[c2],
                      *reinterpret_cast<ValueType *>(dev->basePtr() + y * dev->stride(0) + x * dev->stride(1)));
        }
    }
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap2DTensorTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_IMAGE_FORMAT_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_IMAGE_FORMAT_Y8>>,
    ttype::Types<const short2, ttype::Value<NVCV_IMAGE_FORMAT_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_IMAGE_FORMAT_RGB8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_IMAGE_FORMAT_RGBA8>>,
    ttype::Types<float3, ttype::Value<NVCV_IMAGE_FORMAT_RGBf32>>,
    ttype::Types<const float4, ttype::Value<NVCV_IMAGE_FORMAT_RGBAf32>>
>);

// clang-format on

TYPED_TEST(FullTensorWrap2DTensorTest, correct_with_tensor)
{
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nvcv::Tensor tensor(
        nvcv::TensorShape{
            {211, 213},
            "HW"
    },
        nvcv::DataType{imgFormat});

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::FullTensorWrap<ValueType, 2> wrap(*dev);

    for (int i = 0; i < 2; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], dev->stride(i));
        EXPECT_EQ(wrap.shapes()[i], dev->shape(i));
    }

    const ValueType *ptr0  = reinterpret_cast<const ValueType *>(dev->basePtr());
    const ValueType *ptr1  = reinterpret_cast<const ValueType *>(dev->basePtr() + dev->stride(0));
    const ValueType *ptr12 = reinterpret_cast<const ValueType *>(dev->basePtr() + dev->stride(0) + 2 * dev->stride(1));

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
    EXPECT_EQ(wrap.ptr(1, 2), ptr12);
}

TYPED_TEST(FullTensorWrap2DTensorTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nvcv::Tensor tensor(
        nvcv::TensorShape{
            {567, 234},
            "HW"
    },
        nvcv::DataType{imgFormat});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::FullTensorWrap<ValueType, 2> wrap(*dev);

    DeviceSetOnes(wrap, stream);

    for (int i = 0; i < 2; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], dev->stride(i));
        EXPECT_EQ(wrap.shapes()[i], dev->shape(i));
    }

    std::size_t sizeBytes = dev->shape(0) * dev->stride(0);

    std::vector<uint8_t> test(sizeBytes);
    std::vector<uint8_t> gold(sizeBytes);

    for (int i = 0; i < dev->shape(0); i++)
    {
        for (int j = 0; j < dev->shape(1); j++)
        {
            *reinterpret_cast<ValueType *>(&gold[i * dev->stride(0) + j * dev->stride(1)]) = cuda::SetAll<ValueType>(1);
        }
    }

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dev->basePtr(), sizeBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test, gold);
}

// ------------------------ Testing FullTensorWrap 3D --------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap3DTest, ttype::Types<
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

TYPED_TEST(FullTensorWrap3DTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::FullTensorWrap<const ValueType, 3> wrap(input.data(), InputType::kStrides, InputType::kShapes);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], InputType::kStrides[i]);
        EXPECT_EQ(wrap.shapes()[i], InputType::kShapes[i]);
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

TYPED_TEST(FullTensorWrap3DTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensorWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap3DCopyTest, ttype::Types<
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

TYPED_TEST(FullTensorWrap3DCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::FullTensorWrap<ValueType, 3> wrap(test.data(), InputType::kStrides, InputType::kShapes);

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

TEST(FullTensorWrap3DBigPitchDeathTest, it_dies)
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

    using TensorWrapper = cuda::FullTensorWrap<DataType, 3>;

    EXPECT_DEATH({ TensorWrapper wrap(*dev); }, "");
}

TEST(FullTensorWrap3DBigShapeDeathTest, it_dies)
{
    using DataType = uint8_t;
    int64_t height = 2;
    int64_t width  = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;

    nvcv::DataType                      dt{NVCV_DATA_TYPE_U8};
    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.strides[2] = sizeof(DataType);
    buf.strides[1] = 12 * buf.strides[2];
    buf.strides[0] = height * buf.strides[1];
    buf.basePtr    = reinterpret_cast<NVCVByte *>(123);

    auto tensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{1, height, width}, "NHW"},
        dt, buf
    });

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    using TensorWrapper = cuda::FullTensorWrap<DataType, 3>;

    EXPECT_DEATH({ TensorWrapper wrap(*dev); }, "");
}

#endif

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap3DTensorWrapTest, ttype::Types<
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

TYPED_TEST(FullTensorWrap3DTensorWrapTest, correct_with_tensor_wrap)
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

    cuda::FullTensorWrap<ValueType, 3> wrap(*dev);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], InputType::kStrides[i]);
        EXPECT_EQ(wrap.shapes()[i], InputType::kShapes[i]);
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
    FullTensorWrap3DTensorTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_IMAGE_FORMAT_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_IMAGE_FORMAT_Y8>>,
    ttype::Types<const short2, ttype::Value<NVCV_IMAGE_FORMAT_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_IMAGE_FORMAT_RGB8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_IMAGE_FORMAT_RGBA8>>,
    ttype::Types<float3, ttype::Value<NVCV_IMAGE_FORMAT_RGBf32>>,
    ttype::Types<const float4, ttype::Value<NVCV_IMAGE_FORMAT_RGBAf32>>
>);

// clang-format on

TYPED_TEST(FullTensorWrap3DTensorTest, correct_with_tensor)
{
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nvcv::Tensor tensor(3, {213, 211}, nvcv::ImageFormat{imgFormat});

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::FullTensorWrap<ValueType, 3> wrap(*dev);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], dev->stride(i));
        EXPECT_EQ(wrap.shapes()[i], dev->shape(i));
    }

    const ValueType *ptr0  = reinterpret_cast<const ValueType *>(dev->basePtr());
    const ValueType *ptr1  = reinterpret_cast<const ValueType *>(dev->basePtr() + dev->stride(0));
    const ValueType *ptr12 = reinterpret_cast<const ValueType *>(dev->basePtr() + dev->stride(0) + 2 * dev->stride(1));

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
    EXPECT_EQ(wrap.ptr(1, 2), ptr12);
}

TYPED_TEST(FullTensorWrap3DTensorTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nvcv::Tensor tensor(5, {234, 567}, nvcv::ImageFormat{imgFormat});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);

    cuda::FullTensorWrap<ValueType, 3> wrap(*dev);

    DeviceSetOnes(wrap, stream);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], dev->stride(i));
        EXPECT_EQ(wrap.shapes()[i], dev->shape(i));
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

// ------------------------ Testing FullTensorWrap 4D --------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap4DTest, ttype::Types<
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

TYPED_TEST(FullTensorWrap4DTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::FullTensorWrap<const ValueType, 4> wrap(input.data(), InputType::kStrides, InputType::kShapes);

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], InputType::kStrides[i]);
        EXPECT_EQ(wrap.shapes()[i], InputType::kShapes[i]);
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

TYPED_TEST(FullTensorWrap4DTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensorWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    FullTensorWrap4DCopyTest, ttype::Types<
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

TYPED_TEST(FullTensorWrap4DCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::FullTensorWrap<ValueType, 4> wrap(test.data(), InputType::kStrides, InputType::kShapes);

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
    FullTensorWrap4DTensorWrapTest, ttype::Types<
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

TYPED_TEST(FullTensorWrap4DTensorWrapTest, correct_with_tensor_wrap)
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

    cuda::FullTensorWrap<ValueType, 4> wrap(*dev);

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], InputType::kStrides[i]);
        EXPECT_EQ(wrap.shapes()[i], InputType::kShapes[i]);
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
    FullTensorWrap4DTensorTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_DATA_TYPE_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_DATA_TYPE_U8>>,
    ttype::Types<const short2, ttype::Value<NVCV_DATA_TYPE_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_DATA_TYPE_3U8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_DATA_TYPE_4U8>>,
    ttype::Types<float3, ttype::Value<NVCV_DATA_TYPE_3F32>>,
    ttype::Types<const float4, ttype::Value<NVCV_DATA_TYPE_4F32>>
>);

// clang-format on

TYPED_TEST(FullTensorWrap4DTensorTest, correct_with_tensor)
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

    cuda::FullTensorWrap<ValueType, 4> wrap(*dev);

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], dev->stride(i));
        EXPECT_EQ(wrap.shapes()[i], dev->shape(i));
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

TYPED_TEST(FullTensorWrap4DTensorTest, it_works_in_device)
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
    ASSERT_NE(dev, nullptr);

    cuda::FullTensorWrap<ValueType, 4> wrap(*dev);

    DeviceSetOnes(wrap, stream);

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(wrap.strides()[i], dev->stride(i));
        EXPECT_EQ(wrap.shapes()[i], dev->shape(i));
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

// ---------------- Testing BorderWrap with FullTensorWrap 3D ------------------

// Shortcuts to easy write each test case

constexpr auto U8      = NVCV_IMAGE_FORMAT_U8;
constexpr auto S16     = NVCV_IMAGE_FORMAT_S16;
constexpr auto _2S16   = NVCV_IMAGE_FORMAT_2S16;
constexpr auto F32     = NVCV_IMAGE_FORMAT_F32;
constexpr auto RGB8    = NVCV_IMAGE_FORMAT_RGB8;
constexpr auto RGBA8   = NVCV_IMAGE_FORMAT_RGBA8;
constexpr auto RGBf32  = NVCV_IMAGE_FORMAT_RGBf32;
constexpr auto RGBAf32 = NVCV_IMAGE_FORMAT_RGBAf32;

#define NVCV_TEST_ROW(WIDTH, HEIGHT, BATCHES, BORDERSIZE, FORMAT, VALUETYPE, BORDERTYPE)                     \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<BATCHES>, ttype::Value<BORDERSIZE>, \
                 ttype::Value<FORMAT>, VALUETYPE, ttype::Value<BORDERTYPE>>

NVCV_TYPED_TEST_SUITE(BorderWrapFullTensorWrap3DTest,
                      ttype::Types<NVCV_TEST_ROW(22, 33, 1, 0, RGBA8, uchar4, NVCV_BORDER_CONSTANT),
                                   NVCV_TEST_ROW(33, 22, 3, 3, _2S16, short2, NVCV_BORDER_CONSTANT),
                                   NVCV_TEST_ROW(11, 44, 5, 55, RGBAf32, float4, NVCV_BORDER_REPLICATE),
                                   NVCV_TEST_ROW(122, 6, 9, 7, RGBf32, float3, NVCV_BORDER_WRAP),
                                   NVCV_TEST_ROW(66, 163, 6, 9, RGB8, uchar3, NVCV_BORDER_REFLECT),
                                   NVCV_TEST_ROW(199, 99, 4, 19, S16, short1, NVCV_BORDER_REFLECT101)>);

TYPED_TEST(BorderWrapFullTensorWrap3DTest, correct_fill)
{
    int width      = ttype::GetValue<TypeParam, 0>;
    int height     = ttype::GetValue<TypeParam, 1>;
    int batches    = ttype::GetValue<TypeParam, 2>;
    int borderSize = ttype::GetValue<TypeParam, 3>;

    nvcv::ImageFormat format{ttype::GetValue<TypeParam, 4>};

    using ValueType            = ttype::GetType<TypeParam, 5>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 6>;
    using DimType              = int3;

    ValueType borderValue = cuda::SetAll<ValueType>(123);

    int2 bSize{borderSize, borderSize};

    nvcv::Tensor srcTensor(batches, {width, height}, format);
    nvcv::Tensor dstTensor(batches, {width + borderSize * 2, height + borderSize * 2}, format);

    auto srcDev = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstDev = dstTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(srcDev, nullptr);
    ASSERT_NE(dstDev, nullptr);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcDev);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstDev);

    ASSERT_TRUE(srcAccess);
    ASSERT_TRUE(dstAccess);

    DimType srcSize, dstSize;

    srcSize.x = srcAccess->numCols();
    dstSize.x = dstAccess->numCols();

    srcSize.y = srcAccess->numRows();
    dstSize.y = dstAccess->numRows();

    srcSize.z = srcAccess->numSamples();
    dstSize.z = dstAccess->numSamples();

    int srcSizeBytes = srcDev->stride(0) * srcSize.z;
    int dstSizeBytes = dstDev->stride(0) * dstSize.z;

    std::vector<uint8_t> srcVec(srcSizeBytes);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcDev->basePtr(), srcVec.data(), srcVec.size(), cudaMemcpyHostToDevice));

    using SrcWrapper = cuda::FullTensorWrap<const ValueType, 3>;
    using DstWrapper = cuda::FullTensorWrap<ValueType, 3>;

    SrcWrapper srcWrap(*srcDev);
    DstWrapper dstWrap(*dstDev);

    cuda::BorderWrap<SrcWrapper, kBorderType, false, true, true> srcBorderWrap(srcWrap, borderValue);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    DeviceRunFillBorder(dstWrap, srcBorderWrap, dstSize, bSize, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> test(dstSizeBytes);
    std::vector<uint8_t> gold(dstSizeBytes);

    // Get test fill border
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dstDev->basePtr(), test.size(), cudaMemcpyDeviceToHost));

    // Run gold fill border
    for (int z = 0; z < dstSize.z; ++z)
    {
        int2 srcCoord;

        for (int y = 0; y < dstSize.y; ++y)
        {
            srcCoord.y = y - borderSize;

            for (int x = 0; x < dstSize.x; ++x)
            {
                srcCoord.x = x - borderSize;

                bool isInside = test::IsInside(srcCoord, {width, height}, kBorderType);

                *reinterpret_cast<ValueType *>(
                    &gold[z * dstDev->stride(0) + y * dstDev->stride(1) + x * dstDev->stride(2)])
                    = isInside
                        ? *reinterpret_cast<ValueType *>(&srcVec[z * srcDev->stride(0) + srcCoord.y * srcDev->stride(1)
                                                                 + srcCoord.x * srcDev->stride(2)])
                        : borderValue;
            }
        }
    }

    EXPECT_EQ(test, gold);
}

// ---------------- Testing BorderWrap with FullTensorWrap 4D ------------------

NVCV_TYPED_TEST_SUITE(BorderWrapFullTensorWrap4DTest,
                      ttype::Types<NVCV_TEST_ROW(33, 22, 3, 3, F32, float1, NVCV_BORDER_CONSTANT),
                                   NVCV_TEST_ROW(23, 13, 11, 33, _2S16, short2, NVCV_BORDER_REPLICATE),
                                   NVCV_TEST_ROW(77, 15, 8, 16, RGB8, uchar3, NVCV_BORDER_WRAP),
                                   NVCV_TEST_ROW(144, 33, 8, 99, U8, uchar1, NVCV_BORDER_REFLECT),
                                   NVCV_TEST_ROW(199, 99, 6, 200, RGBA8, uchar4, NVCV_BORDER_REFLECT101)>);

#undef NVCV_TEST_ROW

TYPED_TEST(BorderWrapFullTensorWrap4DTest, correct_fill)
{
    int width      = ttype::GetValue<TypeParam, 0>;
    int height     = ttype::GetValue<TypeParam, 1>;
    int batches    = ttype::GetValue<TypeParam, 2>;
    int borderSize = ttype::GetValue<TypeParam, 3>;

    nvcv::ImageFormat format{ttype::GetValue<TypeParam, 4>};

    using ValueType            = ttype::GetType<TypeParam, 5>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 6>;
    using DimType              = int4;
    using ChannelType          = cuda::BaseType<ValueType>;

    ChannelType chBorderValue = 123;
    ValueType   borderValue   = cuda::SetAll<ValueType>(chBorderValue);

    int2 bSize{borderSize, borderSize};

    nvcv::Tensor srcTensor(batches, {width, height}, format);
    nvcv::Tensor dstTensor(batches, {width + borderSize * 2, height + borderSize * 2}, format);

    auto srcDev = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstDev = dstTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(srcDev, nullptr);
    ASSERT_NE(dstDev, nullptr);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcDev);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstDev);

    ASSERT_TRUE(srcAccess);
    ASSERT_TRUE(dstAccess);

    DimType srcSize, dstSize;

    srcSize.x = srcAccess->numCols();
    dstSize.x = dstAccess->numCols();

    srcSize.y = srcAccess->numRows();
    dstSize.y = dstAccess->numRows();

    srcSize.z = srcAccess->numSamples();
    dstSize.z = dstAccess->numSamples();

    srcSize.w = srcAccess->numChannels();
    dstSize.w = dstAccess->numChannels();

    int srcSizeBytes = srcDev->stride(0) * srcSize.z;
    int dstSizeBytes = dstDev->stride(0) * dstSize.z;

    std::vector<uint8_t> srcVec(srcSizeBytes);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcDev->basePtr(), srcVec.data(), srcVec.size(), cudaMemcpyHostToDevice));

    using SrcWrapper = cuda::FullTensorWrap<const ValueType, 4>;
    using DstWrapper = cuda::FullTensorWrap<ValueType, 4>;

    SrcWrapper srcWrap(*srcDev);
    DstWrapper dstWrap(*dstDev);

    cuda::BorderWrap<SrcWrapper, kBorderType, false, true, true, false> srcBorderWrap(srcWrap, borderValue);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    DeviceRunFillBorder(dstWrap, srcBorderWrap, dstSize, bSize, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> test(dstSizeBytes);
    std::vector<uint8_t> gold(dstSizeBytes);

    // Get test fill border
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dstDev->basePtr(), test.size(), cudaMemcpyDeviceToHost));

    // Run gold fill border
    for (int z = 0; z < dstSize.z; ++z)
    {
        int2 srcCoord;

        for (int y = 0; y < dstSize.y; ++y)
        {
            srcCoord.y = y - borderSize;

            for (int x = 0; x < dstSize.x; ++x)
            {
                srcCoord.x = x - borderSize;

                bool isInside = test::IsInside(srcCoord, {width, height}, kBorderType);

                for (int w = 0; w < dstSize.w; ++w)
                {
                    *reinterpret_cast<ChannelType *>(&gold[z * dstDev->stride(0) + y * dstDev->stride(1)
                                                           + x * dstDev->stride(2) + w * dstDev->stride(3)])
                        = isInside ? *reinterpret_cast<ChannelType *>(
                              &srcVec[z * srcDev->stride(0) + srcCoord.y * srcDev->stride(1)
                                      + srcCoord.x * srcDev->stride(2) + w * srcDev->stride(3)])
                                   : chBorderValue;
                }
            }
        }
    }

    EXPECT_EQ(test, gold);
}

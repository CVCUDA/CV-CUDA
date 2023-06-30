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

#include "Definitions.hpp"

#include <common/TypedTests.hpp>
#include <nvcv/Config.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Allocator.hpp>

namespace t     = ::testing;
namespace ttest = nvcv::test::type;

template<class T>
T CreateObj()
{
    if constexpr (std::is_same_v<nvcv::Image, T>)
    {
        return nvcv::Image(nvcv::Size2D{64, 32}, nvcv::FMT_RGBA8);
    }
    else if constexpr (std::is_same_v<nvcv::ImageBatch, T>)
    {
        return nvcv::ImageBatchVarShape(32);
    }
    else if constexpr (std::is_same_v<nvcv::Allocator, T>)
    {
        return nvcv::CustomAllocator<>();
    }
    else if constexpr (std::is_same_v<nvcv::Tensor, T>)
    {
        return nvcv::Tensor(nvcv::TensorShape({32, 12, 4}, nvcv::TENSOR_NONE), nvcv::TYPE_U8);
    }
    else
    {
        static_assert(sizeof(T) != 0 && "Invalid core object type");
    }
}

template<class T>
void SetMaxCount(int32_t maxCount)
{
    if constexpr (std::is_same_v<nvcv::Image, T>)
    {
        nvcv::cfg::SetMaxImageCount(maxCount);
    }
    else if constexpr (std::is_same_v<nvcv::ImageBatch, T>)
    {
        nvcv::cfg::SetMaxImageBatchCount(maxCount);
    }
    else if constexpr (std::is_same_v<nvcv::Allocator, T>)
    {
        nvcv::cfg::SetMaxAllocatorCount(maxCount);
    }
    else if constexpr (std::is_same_v<nvcv::Tensor, T>)
    {
        nvcv::cfg::SetMaxTensorCount(maxCount);
    }
    else
    {
        static_assert(sizeof(T) != 0 && "Invalid core object type");
    }
}

using AllCoreTypes = ttest::Types<nvcv::Image, nvcv::ImageBatch, nvcv::Tensor, nvcv::Allocator>;

template<class T>
class ConfigTests : public ::testing::Test
{
public:
    ~ConfigTests()
    {
        // Make sure we set the handle manager back to dynamic allocation.
        EXPECT_NO_THROW(SetMaxCount<T>(-1));
    }
};

NVCV_TYPED_TEST_SUITE_F(ConfigTests, AllCoreTypes);

TYPED_TEST(ConfigTests, set_max_obj_count_works)
{
    std::vector<TypeParam> objs;

    ASSERT_NO_THROW(SetMaxCount<TypeParam>(5));

    for (int i = 0; i < 5; ++i)
    {
        ASSERT_NO_THROW(objs.emplace_back(CreateObj<TypeParam>()));
    }

    NVCV_ASSERT_STATUS(NVCV_ERROR_OUT_OF_MEMORY, CreateObj<TypeParam>());

    objs.pop_back();
    ASSERT_NO_THROW(objs.emplace_back(CreateObj<TypeParam>()));
    NVCV_ASSERT_STATUS(NVCV_ERROR_OUT_OF_MEMORY, CreateObj<TypeParam>());
}

TYPED_TEST(ConfigTests, cant_change_limits_when_objects_are_alive)
{
    TypeParam obj = CreateObj<TypeParam>();

    NVCV_ASSERT_STATUS(NVCV_ERROR_INVALID_OPERATION, SetMaxCount<TypeParam>(5));

    obj.reset();

    ASSERT_NO_THROW(SetMaxCount<TypeParam>(5));
}

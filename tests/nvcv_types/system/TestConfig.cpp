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

#include <common/TypedTests.hpp>
#include <nvcv/Config.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>

namespace t     = ::testing;
namespace ttest = nvcv::test::type;

template<class T>
std::unique_ptr<T> CreateObj()
{
    if constexpr (std::is_same_v<nvcv::IImage, T>)
    {
        return std::make_unique<nvcv::Image>(nvcv::Size2D{64, 32}, nvcv::FMT_RGBA8);
    }
    else if constexpr (std::is_same_v<nvcv::IImageBatch, T>)
    {
        return std::make_unique<nvcv::ImageBatchVarShape>(32);
    }
    else if constexpr (std::is_same_v<nvcv::IAllocator, T>)
    {
        return std::make_unique<nvcv::CustomAllocator<>>();
    }
    else if constexpr (std::is_same_v<nvcv::ITensor, T>)
    {
        return std::make_unique<nvcv::Tensor>(nvcv::TensorShape({32, 12, 4}, nvcv::TensorLayout::NONE), nvcv::TYPE_U8);
    }
    else
    {
        static_assert(sizeof(T) != 0 && "Invalid core object type");
    }
}

template<class T>
void SetMaxCount(int32_t maxCount)
{
    if constexpr (std::is_same_v<nvcv::IImage, T>)
    {
        nvcv::cfg::SetMaxImageCount(maxCount);
    }
    else if constexpr (std::is_same_v<nvcv::IImageBatch, T>)
    {
        nvcv::cfg::SetMaxImageBatchCount(maxCount);
    }
    else if constexpr (std::is_same_v<nvcv::IAllocator, T>)
    {
        nvcv::cfg::SetMaxAllocatorCount(maxCount);
    }
    else if constexpr (std::is_same_v<nvcv::ITensor, T>)
    {
        nvcv::cfg::SetMaxTensorCount(maxCount);
    }
    else
    {
        static_assert(sizeof(T) != 0 && "Invalid core object type");
    }
}

using AllCoreTypes = ttest::Types<nvcv::IImage, nvcv::IImageBatch, nvcv::ITensor, nvcv::IAllocator>;
NVCV_TYPED_TEST_SUITE(ConfigTests, AllCoreTypes);

TYPED_TEST(ConfigTests, set_max_obj_count_works)
{
    std::vector<std::unique_ptr<TypeParam>> objs;

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
    std::unique_ptr<TypeParam> obj = CreateObj<TypeParam>();

    NVCV_ASSERT_STATUS(NVCV_ERROR_INVALID_OPERATION, SetMaxCount<TypeParam>(5));

    obj.reset();

    ASSERT_NO_THROW(SetMaxCount<TypeParam>(5));
}

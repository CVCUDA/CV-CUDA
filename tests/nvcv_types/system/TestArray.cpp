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

#include <common/HashUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Array.hpp>
#include <nvcv/ArrayDataAccess.hpp>
#include <nvcv/alloc/Allocator.hpp>

#include <list>
#include <random>
#include <vector>

#include <nvcv/Fwd.hpp>

namespace t    = ::testing;
namespace test = nvcv::test;

// clang-format off
namespace std {

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec)
{
    out << '{';
    std::string sep = "";
    for (auto value: vec)
    {
        out << sep << value;
        sep = ",";
    }
    return out << '}';
}

} // namespace std

// clang-format on

static std::vector<int>              g_testCapacities    = {1, 2, 1024, 1048576};
static std::vector<nvcv::DataType>   g_testDataTypes     = {nvcv::TYPE_U8, nvcv::TYPE_F32, nvcv::TYPE_4F64};
static std::vector<NVCVResourceType> g_testResourceTypes = {NVCV_RESOURCE_MEM_CUDA, NVCV_RESOURCE_MEM_HOST};

class ArrayTests : public t::TestWithParam<std::tuple<int, nvcv::DataType, NVCVResourceType>>
{
public:
    struct PrintToStringParamName
    {
        template<typename ParamType>
        std::string operator()(const testing::TestParamInfo<ParamType> &info) const
        {
            std::string result = "";

            auto paramPack = info.param;
            auto capacity  = std::get<0>(paramPack);
            auto dtype     = std::get<1>(paramPack);
            auto target    = std::get<2>(paramPack);

            result += "mem" + std::string(nvcvResourceTypeGetName(target)).substr(18) + "__";
            result += "dtype" + std::string(nvcvDataTypeGetName(dtype)).substr(15) + "__";
            result += "cap" + std::to_string(capacity);

            return result;
        }
    };
};

TEST_P(ArrayTests, smoke_c_create)
{
    auto paramPack = GetParam();

    auto capacity = std::get<0>(paramPack);
    auto dtype    = std::get<1>(paramPack);
    auto target   = std::get<2>(paramPack);

    NVCVArrayHandle       handle;
    NVCVArrayRequirements reqs;
    ASSERT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(capacity, dtype, 0, &reqs));
    ASSERT_EQ(NVCV_SUCCESS, nvcvArrayConstructWithTarget(&reqs, nullptr, target, &handle));

    int ref;
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayRefCount(handle, &ref));
    EXPECT_EQ(ref, 1);

    auto        h = handle;
    nvcv::Array array(std::move(handle));

    EXPECT_EQ(h, array.handle());
    ASSERT_EQ(1, array.rank());
    EXPECT_EQ(capacity, array.capacity());
    EXPECT_EQ(dtype, array.dtype());

    ref = array.reset();
    EXPECT_EQ(ref, 0);
}

TEST_P(ArrayTests, smoke_cxx_create)
{
    auto paramPack = GetParam();

    auto capacity = std::get<0>(paramPack);
    auto dtype    = std::get<1>(paramPack);
    auto target   = std::get<2>(paramPack);

    nvcv::Array *pArray = nullptr;

    ASSERT_NO_THROW(pArray = new nvcv::Array(capacity, dtype, 0, target));

    EXPECT_EQ(pArray->target(), target);
    EXPECT_EQ(pArray->dtype(), dtype);
    EXPECT_EQ(pArray->capacity(), capacity);
    EXPECT_EQ(pArray->length(), 0);

    auto data = pArray->exportData<nvcv::ArrayData>();
    ASSERT_TRUE(data);
    EXPECT_NE(data->basePtr(), nullptr);

    EXPECT_NO_THROW(delete pArray);

    pArray = nullptr;
}

INSTANTIATE_TEST_SUITE_P(_, ArrayTests,
                         t::Combine(t::ValuesIn(g_testCapacities), t::ValuesIn(g_testDataTypes),
                                    t::ValuesIn(g_testResourceTypes)),
                         ArrayTests::PrintToStringParamName());

class ArrayWrapTests : public ArrayTests
{
};

TEST_P(ArrayWrapTests, smoke_create)
{
    auto paramPack = GetParam();

    auto capacity = std::get<0>(paramPack);
    auto dtype    = std::get<1>(paramPack);
    auto target   = std::get<2>(paramPack);

    nvcv::Array baseArray(capacity, dtype, dtype.alignment(), target);

    auto data = baseArray.exportData<nvcv::ArrayData>();
    ASSERT_TRUE(data);
    auto access = nvcv::ArrayDataAccess::Create(*data);
    ASSERT_TRUE(access);

    EXPECT_EQ(data->basePtr(), access->ptr());
    EXPECT_EQ(data->length(), access->length());
    EXPECT_EQ(data->kind(), access->kind());
    EXPECT_EQ(data->stride(), access->stride());

    auto array = nvcv::ArrayWrapData(*data);
    ASSERT_NE(array.handle(), nullptr);
}

INSTANTIATE_TEST_SUITE_P(_, ArrayWrapTests,
                         t::Combine(t::ValuesIn(g_testCapacities), t::ValuesIn(g_testDataTypes),
                                    t::ValuesIn(g_testResourceTypes)),
                         ArrayWrapTests::PrintToStringParamName());

TEST(ArrayTests, smoke_create_allocator)
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

    nvcv::Array array(1024, nvcv::TYPE_4U64, 0, NVCV_RESOURCE_MEM_CUDA, myAlloc);
    EXPECT_EQ(32, setBufAlign);

    auto data = array.exportData<nvcv::ArrayDataCuda>();
    ASSERT_TRUE(data);

    EXPECT_EQ(32, data->stride());
}

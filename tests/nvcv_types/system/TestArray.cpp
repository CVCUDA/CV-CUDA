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

#include <common/HashUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Array.hpp>
#include <nvcv/ArrayDataAccess.hpp>
#include <nvcv/alloc/Allocator.hpp>

#include <cstddef>
#include <list>
#include <random>
#include <vector>

#include <nvcv/Fwd.hpp>

namespace t    = ::testing;
namespace test = nvcv::test;

namespace {

const std::vector<int>              g_testCapacities = {1, 2, 1024, 1048576};
const std::vector<nvcv::DataType>   g_testDataTypes  = {nvcv::TYPE_U8, nvcv::TYPE_F16, nvcv::TYPE_F32, nvcv::TYPE_4F64};
const std::vector<NVCVResourceType> g_testResourceTypes = {NVCV_RESOURCE_MEM_CUDA, NVCV_RESOURCE_MEM_HOST};

} // namespace

class ArrayTests : public t::TestWithParam<std::tuple<int, nvcv::DataType, NVCVResourceType>>
{
public:
    struct PrintToStringParamName
    {
        template<typename ParamType>
        std::string operator()(const testing::TestParamInfo<ParamType> &info) const
        {
            std::string result = "";

            auto [capacity, dtype, target] = info.param;

            result += "mem" + std::string(nvcvResourceTypeGetName(target)).substr(18) + "__";
            result += "dtype" + std::string(nvcvDataTypeGetName(static_cast<NVCVDataType>(dtype))).substr(15) + "__";
            result += "cap" + std::to_string(capacity);

            return result;
        }
    };
};

TEST_P(ArrayTests, smoke_c_create)
{
    const auto &[capacity, dtype, target] = GetParam();

    NVCVArrayHandle       handle;
    NVCVArrayRequirements reqs;
    ASSERT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(capacity, static_cast<NVCVDataType>(dtype), 0, &reqs));
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
    const auto &[capacity, dtype, target] = GetParam();

    nvcv::Array array;

    ASSERT_NO_THROW(array = nvcv::Array(capacity, dtype, 0, target));

    EXPECT_EQ(array.target(), target);
    EXPECT_EQ(array.dtype(), dtype);
    EXPECT_EQ(array.capacity(), capacity);
    EXPECT_EQ(array.length(), 0);

    auto data = array.exportData<nvcv::ArrayData>();
    ASSERT_TRUE(data);
    EXPECT_NE(data->basePtr(), nullptr);
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
    const auto &[capacity, dtype, target] = GetParam();

    nvcv::Array baseArray(capacity, dtype, dtype.alignment(), target);

    auto data = baseArray.exportData<nvcv::ArrayData>();
    ASSERT_TRUE(data);
    auto access = nvcv::ArrayDataAccess::Create(*data);
    ASSERT_TRUE(access);

    EXPECT_EQ(data->basePtr(), access->ptr());
    EXPECT_EQ(data->length(), access->length());
    EXPECT_EQ(data->kind(), access->kind());
    EXPECT_EQ(data->stride(), access->stride());
    EXPECT_EQ(data->rank(), 1);

    auto array = nvcv::ArrayWrapData(*data);
    ASSERT_NE(array.handle(), nullptr);
    EXPECT_EQ(array.rank(), 1);
    EXPECT_EQ(array.capacity(), capacity);
    EXPECT_EQ(array.length(), data->length());
    EXPECT_EQ(array.dtype(), data->dtype());
    EXPECT_EQ(array.target(), baseArray.target());

    auto arrayData = array.exportData<nvcv::ArrayData>();
    ASSERT_TRUE(arrayData);
    auto arrayAccess = nvcv::ArrayDataAccess::Create(*arrayData);
    ASSERT_TRUE(arrayAccess);

    EXPECT_EQ(arrayData->basePtr(), arrayAccess->ptr());
    EXPECT_EQ(arrayData->length(), arrayAccess->length());
    EXPECT_EQ(arrayData->kind(), arrayAccess->kind());
    EXPECT_EQ(arrayData->stride(), arrayAccess->stride());
    EXPECT_EQ(arrayData->rank(), 1);
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
                return static_cast<std::byte *>(ptr);
            },
            [](auto ptr, int64_t, int32_t)
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

TEST(ArrayTests, invalid_outputs_calcReq)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvArrayCalcRequirementsWithTarget(16, NVCV_DATA_TYPE_U8, 0, NVCV_RESOURCE_MEM_HOST, nullptr));
}

TEST(ArrayTests, invalid_alignment_calcReq_with_target)
{
    NVCVArrayRequirements req;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvArrayCalcRequirementsWithTarget(16, NVCV_DATA_TYPE_U8, 7, NVCV_RESOURCE_MEM_HOST, &req));
#ifndef ENABLE_SANITIZER
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvArrayCalcRequirementsWithTarget(16, NVCV_DATA_TYPE_U8, 0, static_cast<NVCVResourceType>(255), &req));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayCalcRequirementsWithTarget(
                                               16, NVCV_DATA_TYPE_U8, 128, static_cast<NVCVResourceType>(255), &req));
#endif
}

TEST(ArrayTests, invalid_input_construct)
{
    NVCVArrayRequirements req;
    NVCVArrayHandle       arrayHandle;
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayConstruct(nullptr, nullptr, &arrayHandle));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayConstruct(&req, nullptr, nullptr));
}

TEST(ArrayTests, valid_construct)
{
    NVCVArrayRequirements req;
    NVCVArrayHandle       arrayHandle;
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, valid_construct_with_target)
{
    NVCVArrayRequirements req;
    NVCVArrayHandle       arrayHandle;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvArrayCalcRequirementsWithTarget(16, NVCV_DATA_TYPE_U8, 0, NVCV_RESOURCE_MEM_HOST_PINNED, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstructWithTarget(&req, nullptr, NVCV_RESOURCE_MEM_HOST_PINNED, &arrayHandle));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, invalid_construct_with_target)
{
    NVCVArrayRequirements req;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvArrayCalcRequirementsWithTarget(16, NVCV_DATA_TYPE_U8, 0, NVCV_RESOURCE_MEM_HOST_PINNED, &req));
#ifndef ENABLE_SANITIZER
    NVCVArrayHandle arrayHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvArrayConstructWithTarget(&req, nullptr, static_cast<NVCVResourceType>(255), &arrayHandle));
#endif
}

TEST(ArrayTests, mismatch_construct_with_target)
{
    NVCVArrayRequirements req;
    NVCVArrayHandle       arrayHandle;
    int64_t               capacity = -1;
    int64_t               length   = -1;
    NVCVResourceType      target;
    NVCVDataType          dType;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvArrayCalcRequirementsWithTarget(16, NVCV_DATA_TYPE_U8, 0, NVCV_RESOURCE_MEM_CUDA, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstructWithTarget(&req, nullptr, NVCV_RESOURCE_MEM_HOST, &arrayHandle));

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetCapacity(arrayHandle, &capacity));
    EXPECT_EQ(16, capacity);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetTarget(arrayHandle, &target));
    EXPECT_EQ(NVCV_RESOURCE_MEM_HOST, target);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetLength(arrayHandle, &length));
    EXPECT_EQ(0, length);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetDataType(arrayHandle, &dType));
    EXPECT_EQ(NVCV_DATA_TYPE_U8, dType);
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, invalid_req_construct_with_target)
{
    NVCVArrayHandle arrayHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvArrayConstructWithTarget(nullptr, nullptr, NVCV_RESOURCE_MEM_HOST, &arrayHandle));
}

TEST(ArrayTests, invalid_handle_construct_with_target)
{
    NVCVArrayRequirements req;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvArrayCalcRequirementsWithTarget(16, NVCV_DATA_TYPE_U8, 0, NVCV_RESOURCE_MEM_HOST, &req));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvArrayConstructWithTarget(&req, nullptr, NVCV_RESOURCE_MEM_HOST, nullptr));
}

TEST(ArrayTests, invalid_data_wrap_data_construct)
{
    NVCVArrayHandle arrayHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayWrapDataConstruct(nullptr, nullptr, nullptr, &arrayHandle));
}

TEST(ArrayTests, invalid_handle_wrap_data_construct)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;
    NVCVArrayData         arrayData;
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayExportData(arrayHandle, &arrayData));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayWrapDataConstruct(&arrayData, nullptr, nullptr, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

void arrayDataCleanUpFunc(void *ctx, const NVCVArrayData *data);

void arrayDataCleanUpFunc(void *ctx, const NVCVArrayData *data)
{
    (void)ctx;
    (void)data;
}

TEST(ArrayTests, valid_handle_wrap_data_construct)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayHandle       arrayHandle2;
    NVCVArrayRequirements req;
    NVCVArrayData         arrayData;
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayExportData(arrayHandle, &arrayData));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayWrapDataConstruct(&arrayData, &arrayDataCleanUpFunc, nullptr, &arrayHandle2));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle2, nullptr));
}

TEST(ArrayTests, valid_handle_wrap_data_construct_pinned)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayHandle       arrayHandle2;
    NVCVArrayRequirements req;
    NVCVArrayData         arrayData;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvArrayCalcRequirementsWithTarget(16, NVCV_DATA_TYPE_U8, 0, NVCV_RESOURCE_MEM_HOST_PINNED, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstructWithTarget(&req, nullptr, NVCV_RESOURCE_MEM_HOST_PINNED, &arrayHandle));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayExportData(arrayHandle, &arrayData));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayWrapDataConstruct(&arrayData, &arrayDataCleanUpFunc, nullptr, &arrayHandle2));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle2, nullptr));
}

TEST(ArrayTests, null_basePtr_wrap_data_construct)
{
    NVCVArrayHandle arrayHandle2;
    NVCVArrayData   arrayData;
    arrayData.buffer.strided.basePtr = nullptr;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayWrapDataConstruct(&arrayData, nullptr, nullptr, &arrayHandle2));
}

TEST(ArrayTests, valid_array_inc_ref)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;
    int                   refCount = -1;
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayRefCount(arrayHandle, &refCount));
    EXPECT_EQ(refCount, 1);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayIncRef(arrayHandle, &refCount));
    EXPECT_EQ(refCount, 2);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, &refCount));
    EXPECT_EQ(refCount, 1);
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, smoke_user_pointer)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;
    NVCVUserPointer       userPtr;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetUserPointer(arrayHandle, &userPtr));
    EXPECT_EQ(nullptr, userPtr);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArraySetUserPointer(arrayHandle, reinterpret_cast<NVCVUserPointer>(0x123ULL)));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetUserPointer(arrayHandle, &userPtr));
    EXPECT_EQ(reinterpret_cast<NVCVUserPointer>(0x123ULL), userPtr);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArraySetUserPointer(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetUserPointer(arrayHandle, &userPtr));
    EXPECT_EQ(nullptr, userPtr);
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, invalid_out_get_user_pointer)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayGetUserPointer(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, invalid_out_get_data_type)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayGetDataType(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, valid_get_allocator)
{
    int                   tmp = 1;
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;
    auto                  alloc = reinterpret_cast<NVCVAllocatorHandle>(&tmp);
    EXPECT_NE(alloc, nullptr);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetAllocator(arrayHandle, &alloc));
    EXPECT_EQ(alloc, nullptr);
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, invalid_out_get_allocator)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayGetAllocator(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, invalid_out_export_data)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayExportData(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, invalid_out_get_length)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayGetLength(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, invalid_out_get_capacity)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayGetCapacity(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, invalid_out_get_target)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayGetTarget(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, validResize)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;
    int64_t               length = 0;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayResize(arrayHandle, 8));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetLength(arrayHandle, &length));
    EXPECT_EQ(length, 8);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayTests, invalidResize)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayRequirements req;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayResize(arrayHandle, 17));

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
}

TEST(ArrayWrapTests, validResize)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayHandle       arrayWrapHandle;
    NVCVArrayData         arrayData;
    NVCVArrayRequirements req;
    int64_t               length = 0;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayExportData(arrayHandle, &arrayData));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayWrapDataConstruct(&arrayData, nullptr, nullptr, &arrayWrapHandle));

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayResize(arrayWrapHandle, 8));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetLength(arrayWrapHandle, &length));
    EXPECT_EQ(length, 8);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayWrapHandle, nullptr));
}

TEST(ArrayWrapTests, invalidResize)
{
    NVCVArrayHandle       arrayHandle;
    NVCVArrayHandle       arrayWrapHandle;
    NVCVArrayData         arrayData;
    NVCVArrayRequirements req;

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayExportData(arrayHandle, &arrayData));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayWrapDataConstruct(&arrayData, nullptr, nullptr, &arrayWrapHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvArrayResize(arrayHandle, 17));

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayWrapHandle, nullptr));
}

TEST(ArrayWrapTests, valid_get_allocator)
{
    int                   tmp = 1;
    NVCVArrayHandle       arrayHandle;
    NVCVArrayHandle       arrayWrapHandle;
    NVCVArrayData         arrayData;
    NVCVArrayRequirements req;
    auto                  alloc = reinterpret_cast<NVCVAllocatorHandle>(&tmp);
    EXPECT_NE(alloc, nullptr);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(16, NVCV_DATA_TYPE_U8, 0, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayConstruct(&req, nullptr, &arrayHandle));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayExportData(arrayHandle, &arrayData));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayWrapDataConstruct(&arrayData, nullptr, nullptr, &arrayWrapHandle));

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayGetAllocator(arrayWrapHandle, &alloc));
    EXPECT_EQ(alloc, nullptr);

    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvArrayDecRef(arrayWrapHandle, nullptr));
}

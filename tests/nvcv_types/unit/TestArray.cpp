/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nvcv/src/priv/Array.hpp"
#include "nvcv/src/priv/ArrayWrapData.hpp"

#include <nvcv/Array.hpp>
#include <nvcv/ArrayData.hpp>

void arrayDataCleanUpFunc(void *ctx, const NVCVArrayData *data);

void arrayDataCleanUpFunc(void *ctx, const NVCVArrayData *data) {}

TEST(ArrayTests, rank)
{
    NVCVArrayRequirements req;
    ASSERT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(8, NVCV_DATA_TYPE_U8, 0, &req));

    nvcv::priv::IAllocator &alloc = nvcv::priv::GetAllocator(nullptr);
    nvcv::priv::Array       array(req, alloc, NVCV_RESOURCE_MEM_HOST);

    EXPECT_EQ(array.rank(), 1);
}

TEST(ArrayTests, warp_rank)
{
    NVCVArrayRequirements req;
    NVCVArrayData         data;
    ASSERT_EQ(NVCV_SUCCESS, nvcvArrayCalcRequirements(8, NVCV_DATA_TYPE_U8, 0, &req));

    nvcv::priv::IAllocator &alloc = nvcv::priv::GetAllocator(nullptr);
    nvcv::priv::Array       array(req, alloc, NVCV_RESOURCE_MEM_HOST);
    array.exportData(data);

    nvcv::priv::ArrayWrapData arrayData(data, &arrayDataCleanUpFunc, nullptr);
    EXPECT_EQ(arrayData.rank(), 1);
}

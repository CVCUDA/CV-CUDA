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

#include "Definitions.hpp"

#include <common/ValueTests.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>

#include <list>
#include <random>

#include <nvcv/Fwd.hpp>

namespace t    = ::testing;
namespace test = nvcv::test;

template<typename R>
nvcv::Tensor GetRandomTensor(R &rg, const nvcv::ImageFormat &format)
{
    std::uniform_int_distribution<int32_t> shape_dist(100, 400);
    std::uniform_int_distribution<int32_t> images_num_dist(1, 16);
    return nvcv::Tensor(images_num_dist(rg), {shape_dist(rg), shape_dist(rg)}, format);
}

template<typename It>
void CheckTensorBatchData(const nvcv::TensorBatchData &tbdata, It tensors_begin, It tensors_end, CUstream stream)
{
    auto numTensors = tensors_end - tensors_begin;
    ASSERT_EQ(numTensors, tbdata.numTensors());
    std::vector<NVCVTensorBatchElementStrided> elements(numTensors);
    ASSERT_TRUE(tbdata.cast<nvcv::TensorBatchDataStridedCuda>().hasValue());
    auto buffer = tbdata.cast<nvcv::TensorBatchDataStridedCuda>()->buffer();
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(elements.data(), buffer.tensors, sizeof(NVCVTensorBatchElementStrided) * numTensors,
                              cudaMemcpyDeviceToHost, stream));

    int i = 0;
    for (auto it = tensors_begin; it != tensors_end; ++it)
    {
        nvcv::Tensor &tensor  = *it;
        auto          tdata   = tensor.exportData().cast<nvcv::TensorDataStridedCuda>().value();
        auto         &element = elements[i];
        EXPECT_EQ(tdata.layout(), tbdata.layout());
        EXPECT_EQ(tdata.dtype(), tbdata.dtype());
        EXPECT_EQ(tdata.basePtr(), reinterpret_cast<nvcv::Byte *>(element.data));
        ASSERT_EQ(tdata.rank(), tbdata.rank());
        for (int d = 0; d < tbdata.rank(); ++d)
        {
            EXPECT_EQ(tdata.shape(d), element.shape[d]);
            EXPECT_EQ(tdata.stride(d), element.stride[d]);
        }
        ++i;
    }
}

TEST(TensorBatch, create)
{
    auto                      reqs = nvcv::TensorBatch::CalcRequirements(1);
    std::vector<nvcv::Tensor> tensors;
    tensors.emplace_back(nvcv::Tensor(1, {300, 300}, nvcv::FMT_RGB8));
    {
        nvcv::TensorBatch tb(reqs);
        EXPECT_EQ(tb.layout(), nvcv::TensorLayout(""));
        EXPECT_EQ(tb.dtype(), nvcv::DataType());
        tb.pushBack(tensors[0]);
        ASSERT_EQ(tb.numTensors(), 1);
        ASSERT_EQ(tensors[0].refCount(), 2);
        auto tbdata = tb.exportData(nullptr);
        CheckTensorBatchData(tbdata, tensors.begin(), tensors.end(), nullptr);
    }
    ASSERT_EQ(tensors[0].refCount(), 1);
}

TEST(TensorBatch, ref_counting)
{
    std::mt19937 rg{231};
    nvcv::Tensor tensor = GetRandomTensor(rg, nvcv::FMT_RGB8);
    {
        auto              reqs = nvcv::TensorBatch::CalcRequirements(1);
        nvcv::TensorBatch tb(reqs);
        tb.pushBack(tensor);
        ASSERT_EQ(tb.refCount(), 1);
        ASSERT_EQ(tensor.refCount(), 2);
        int                            numMul = 32;
        std::vector<nvcv::TensorBatch> tbs(numMul, tb);
        ASSERT_EQ(tb.refCount(), numMul + 1);
        ASSERT_EQ(tensor.refCount(), 2);
    }
    ASSERT_EQ(tensor.refCount(), 1);
}

TEST(TensorBatch, properties)
{
    int32_t                   capacity = 32;
    std::vector<nvcv::Tensor> tensors(capacity / 2);
    std::mt19937              rg{321};
    for (int i = 0; i < capacity / 2; ++i)
    {
        tensors[i] = GetRandomTensor(rg, nvcv::FMT_RGB8);
    }
    auto              reqs = nvcv::TensorBatch::CalcRequirements(capacity);
    nvcv::TensorBatch tb(reqs);
    tb.pushBack(tensors.begin(), tensors.end());
    EXPECT_EQ(tb.dtype(), nvcv::TYPE_U8);
    EXPECT_EQ(tb.capacity(), capacity);
    EXPECT_EQ(tb.numTensors(), capacity / 2);
    EXPECT_EQ(tb.layout(), nvcv::TensorLayout("NHWC"));
    EXPECT_EQ(tb.type(), NVCV_TENSOR_BUFFER_STRIDED_CUDA);
}

TEST(TensorBatch, user_pointer)
{
    auto              reqs = nvcv::TensorBatch::CalcRequirements(1);
    nvcv::TensorBatch tb(reqs);
    int               valueA = 0;
    tb.setUserPointer(&valueA);
    EXPECT_EQ(tb.getUserPointer(), &valueA);
    auto tbCopy = tb;
    std::cout << tb.refCount() << std::endl;
    EXPECT_EQ(tbCopy.getUserPointer(), &valueA);
    int valueB = 0;
    tb.setUserPointer(&valueB);
    EXPECT_EQ(tb.getUserPointer(), &valueB);
    EXPECT_EQ(tbCopy.getUserPointer(), &valueB);
}

TEST(TensorBatch, consistency_validation)
{
    std::mt19937 rg{321};
    auto         base_tensor = GetRandomTensor(rg, nvcv::FMT_RGB8);

    auto test_inconsistency = [&](int32_t rank, nvcv::DataType dtype, nvcv::TensorLayout layout)
    {
        auto                 reqs = nvcv::TensorBatch::CalcRequirements(2);
        nvcv::TensorBatch    tb(reqs);
        std::vector<int64_t> shape(rank, 1);
        nvcv::Tensor         tensor(nvcv::TensorShape(shape.data(), rank, layout), dtype);
        tb.pushBack(tensor);
        NVCV_EXPECT_THROW_STATUS(NVCV_ERROR_INVALID_ARGUMENT, tb.pushBack(base_tensor));
    };
    test_inconsistency(4, nvcv::TYPE_U8, nvcv::TensorLayout("FHWC"));
    test_inconsistency(4, nvcv::TYPE_U32, nvcv::TensorLayout("NHWC"));
    test_inconsistency(3, nvcv::TYPE_U8, nvcv::TensorLayout("HWC"));
}

TEST(TensorBatch, push_in_parts)
{
    const int32_t             iters    = 20;
    const int32_t             capacity = iters * (iters + 1) / 2;
    std::vector<nvcv::Tensor> tensors(capacity);
    std::mt19937              rg{123};
    for (int32_t i = 0; i < capacity; ++i)
    {
        tensors[i] = GetRandomTensor(rg, nvcv::FMT_RGB8);
    }
    std::array<CUstream, 2> streams{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));
    {
        auto              reqs = nvcv::TensorBatch::CalcRequirements(capacity);
        nvcv::TensorBatch tb(reqs);
        auto              tensors_begin = tensors.data();
        for (int32_t i = 1; i < 21; ++i)
        {
            tb.pushBack(tensors_begin, tensors_begin + i);
            ASSERT_EQ(tb.numTensors(), tensors_begin + i - tensors.data());
            if (i % 2 == 0)
            {
                auto stream = streams[i / 2 % 2];
                auto tbdata = tb.exportData(stream);
                CheckTensorBatchData(tbdata, tensors.data(), tensors_begin + i, stream);
            }
            tensors_begin += i;
        }
        for (auto &t : tensors)
        {
            ASSERT_EQ(t.refCount(), 2);
        }
    }
    for (auto &t : tensors)
    {
        ASSERT_EQ(t.refCount(), 1);
    }
}

TEST(TensorBatch, push_the_same_tensor)
{
    const int                 numMul = 32;
    std::mt19937              rg{123};
    auto                      tensor = GetRandomTensor(rg, nvcv::FMT_RGB8);
    std::vector<nvcv::Tensor> tensors;
    for (int i = 0; i < numMul; ++i)
    {
        tensors.push_back(tensor);
    }
    ASSERT_EQ(tensor.refCount(), numMul + 1);
    {
        auto              reqs = nvcv::TensorBatch::CalcRequirements(numMul);
        nvcv::TensorBatch tb(reqs);
        tb.pushBack(tensors.begin(), tensors.end());
        ASSERT_EQ(tensor.refCount(), numMul * 2 + 1);
        NVCV_EXPECT_THROW_STATUS(NVCV_ERROR_OVERFLOW, tb.pushBack(tensor));
    }
    ASSERT_EQ(tensor.refCount(), numMul + 1);
}

TEST(TensorBatch, clear)
{
    const int32_t             capacity = 32;
    std::vector<nvcv::Tensor> tensors(capacity);
    std::mt19937              rg{123};
    for (int32_t i = 0; i < capacity; ++i)
    {
        tensors[i] = GetRandomTensor(rg, nvcv::FMT_RGB8);
    }
    auto              reqs = nvcv::TensorBatch::CalcRequirements(capacity);
    nvcv::TensorBatch tb(reqs);
    tb.pushBack(tensors.begin(), tensors.end());
    for (auto &t : tensors)
    {
        EXPECT_EQ(t.refCount(), 2);
    }
    tb.clear();
    for (auto &t : tensors)
    {
        EXPECT_EQ(t.refCount(), 1);
    }
    EXPECT_EQ(tb.layout(), nvcv::TensorLayout(""));
    EXPECT_EQ(tb.dtype(), nvcv::DataType());
}

TEST(TensorBatch, pop_tensors)
{
    const int32_t             capacity = 32;
    std::vector<nvcv::Tensor> tensors(capacity);
    std::mt19937              rg{123};
    for (int32_t i = 0; i < capacity; ++i)
    {
        tensors[i] = GetRandomTensor(rg, nvcv::FMT_RGB8);
    }
    auto              reqs = nvcv::TensorBatch::CalcRequirements(capacity);
    nvcv::TensorBatch tb(reqs);

    tb.pushBack(tensors.data(), tensors.data() + capacity / 2);
    tb.popTensors(capacity / 4); // remove dirty tensors
    // tensor batch should contain the first quarter of the tensors
    ASSERT_EQ(tb.numTensors(), capacity / 4);
    auto data = tb.exportData(nullptr);
    CheckTensorBatchData(data, tensors.data(), tensors.data() + capacity / 4, nullptr);
    for (int i = 0; i < capacity / 4; ++i)
    {
        EXPECT_EQ(tensors[i].refCount(), 2);
    }
    for (int i = capacity / 4; i < capacity / 2; ++i)
    {
        EXPECT_EQ(tensors[i].refCount(), 1);
    }

    tb.pushBack(tensors.data() + capacity / 2, tensors.data() + capacity);
    // tensor batch should contain the first quarter and the last half of the tensors
    EXPECT_EQ(tb.numTensors(), capacity * 3 / 4);
    for (int i = capacity / 2; i < capacity; ++i)
    {
        EXPECT_EQ(tensors[i].refCount(), 2);
    }
    std::vector<nvcv::Tensor> result{};
    result.insert(result.end(), tensors.begin(), tensors.begin() + capacity / 4);
    result.insert(result.end(), tensors.begin() + capacity / 2, tensors.begin() + capacity);
    data = tb.exportData(nullptr);
    CheckTensorBatchData(data, result.begin(), result.end(), nullptr);
    result.clear();

    tb.popTensors(capacity / 4); // remove clean tensors;
    // tensor batch should contain the first and the third quarter of the tensors
    EXPECT_EQ(tb.numTensors(), capacity / 2);
    for (int i = 0; i < capacity / 4; ++i)
    {
        EXPECT_EQ(tensors[i].refCount(), 2);
        EXPECT_EQ(tensors[i + capacity / 4].refCount(), 1);
        EXPECT_EQ(tensors[i + capacity * 2 / 4].refCount(), 2);
        EXPECT_EQ(tensors[i + capacity * 3 / 4].refCount(), 1);
    }
    data = tb.exportData(nullptr);
    result.insert(result.end(), tensors.begin(), tensors.begin() + capacity / 4);
    result.insert(result.end(), tensors.begin() + capacity / 2, tensors.begin() + capacity * 3 / 4);
    CheckTensorBatchData(data, result.begin(), result.end(), nullptr);
    result.clear();

    tb.pushBack(tensors.begin(), tensors.begin() + capacity / 4);
    // tensor batch should contain the first, the third and the first (again) quarter
    EXPECT_EQ(tb.numTensors(), capacity * 3 / 4);
    for (int i = 0; i < capacity / 4; ++i)
    {
        EXPECT_EQ(tensors[i].refCount(), 3);
        EXPECT_EQ(tensors[i + capacity / 4].refCount(), 1);
        EXPECT_EQ(tensors[i + capacity * 2 / 4].refCount(), 2);
        EXPECT_EQ(tensors[i + capacity * 3 / 4].refCount(), 1);
    }
    tb.popTensors(capacity / 2); // remove clean and dirty tensors
    // tensor batch should contain the first quarter of the tensors
    EXPECT_EQ(tb.numTensors(), capacity / 4);
    for (int i = 0; i < capacity / 4; ++i)
    {
        EXPECT_EQ(tensors[i].refCount(), 2);
    }
    for (int i = capacity / 4; i < capacity; ++i)
    {
        EXPECT_EQ(tensors[i].refCount(), 1);
    }
    data = tb.exportData(nullptr);
    result.insert(result.end(), tensors.begin(), tensors.begin() + capacity / 4);
    CheckTensorBatchData(data, result.begin(), result.end(), nullptr);
    result.clear();

    tb.pushBack(tensors[0]);
    EXPECT_EQ(tensors[0].refCount(), 3);
    tb.popTensor(); // pop single tensor
    EXPECT_EQ(tensors[0].refCount(), 2);
    data = tb.exportData(nullptr);
    result.insert(result.end(), tensors.begin(), tensors.begin() + capacity / 4);
    CheckTensorBatchData(data, result.begin(), result.end(), nullptr);

    NVCV_EXPECT_THROW_STATUS(NVCV_ERROR_UNDERFLOW, tb.popTensors(capacity / 4 + 1));
    NVCV_EXPECT_THROW_STATUS(NVCV_ERROR_INVALID_ARGUMENT, tb.popTensors(-1));
}

TEST(TensorBatch, iterator_arithm)
{
    int32_t                   capacity = 4;
    std::vector<nvcv::Tensor> tensors(capacity);
    std::mt19937              rg{321};
    for (int i = 0; i < capacity; ++i)
    {
        tensors[i] = GetRandomTensor(rg, nvcv::FMT_RGB8);
    }
    auto              reqs = nvcv::TensorBatch::CalcRequirements(capacity);
    nvcv::TensorBatch tb(reqs);

    auto it = tb.begin();
    EXPECT_EQ(it, tb.end());

    tb.pushBack(tensors.begin(), tensors.end());
    it = tb.begin();

    EXPECT_EQ(it->handle(), tensors[0].handle());
    EXPECT_EQ((++it)->handle(), tensors[1].handle());
    EXPECT_EQ((it++)->handle(), tensors[1].handle());
    EXPECT_EQ((--it)->handle(), tensors[1].handle());
    EXPECT_EQ((it--)->handle(), tensors[1].handle());
    EXPECT_EQ((it + capacity - 1)->handle(), tensors[capacity - 1].handle());

    EXPECT_EQ((tb.end() - capacity), tb.begin());
    EXPECT_GT(tb.end(), tb.begin());
    EXPECT_GE(it, tb.begin());
    EXPECT_LT(it, it + 2);
    EXPECT_LE(it, it + 1);

    EXPECT_EQ(tb.end() - it, capacity);
}

TEST(TensorBatch, indexing_and_iterating)
{
    int32_t                   capacity = 32;
    std::vector<nvcv::Tensor> tensors(capacity);
    std::mt19937              rg{321};
    for (int i = 0; i < capacity; ++i)
    {
        tensors[i] = GetRandomTensor(rg, nvcv::FMT_RGB8);
    }
    auto              reqs = nvcv::TensorBatch::CalcRequirements(capacity);
    nvcv::TensorBatch tb(reqs);
    tb.pushBack(tensors.begin(), tensors.end());
    for (int i = 0; i < capacity; ++i)
    {
        EXPECT_EQ(tb[i].handle(), tensors[i].handle());
    }

    int i = 0;
    for (auto t : tb)
    {
        EXPECT_EQ(t.handle(), tensors[i++].handle());
    }

    NVCV_EXPECT_THROW_STATUS(NVCV_ERROR_OVERFLOW, tb[capacity]);
    NVCV_EXPECT_THROW_STATUS(NVCV_ERROR_INVALID_ARGUMENT, tb[-1]);
}

TEST(TensorBatch, set_tensor)
{
    int32_t                   capacity = 32;
    std::vector<nvcv::Tensor> tensors(capacity);
    std::mt19937              rg{321};
    for (int i = 0; i < capacity; ++i)
    {
        tensors[i] = GetRandomTensor(rg, nvcv::FMT_RGB8);
    }
    auto              reqs = nvcv::TensorBatch::CalcRequirements(capacity);
    nvcv::TensorBatch tb(reqs);
    tb.pushBack(tensors.begin(), tensors.end());
    auto tensorA = GetRandomTensor(rg, nvcv::FMT_RGB8);
    auto tensorB = GetRandomTensor(rg, nvcv::FMT_RGB8);

    tb.setTensor(0, tensorA); // set at dirty position
    auto data   = tb.exportData(nullptr);
    auto result = tensors;
    result[0]   = tensorA;
    CheckTensorBatchData(data, result.begin(), result.end(), nullptr);
    result.clear();
    EXPECT_EQ(tensors[0].refCount(), 1);
    EXPECT_EQ(tensorA.refCount(), 2);

    tb.setTensor(capacity / 4, tensorA);
    tb.setTensor(capacity / 2, tensorB); // set at clean positions
    data      = tb.exportData(nullptr);
    result    = tensors;
    result[0] = result[capacity / 4] = tensorA;
    result[capacity / 2]             = tensorB;
    CheckTensorBatchData(data, result.begin(), result.end(), nullptr);
    result.clear();
    EXPECT_EQ(tensors[0].refCount(), 1);
    EXPECT_EQ(tensors[capacity / 4].refCount(), 1);
    EXPECT_EQ(tensors[capacity / 2].refCount(), 1);
    EXPECT_EQ(tensorA.refCount(), 3);
    EXPECT_EQ(tensorB.refCount(), 2);

    for (int i = capacity - 10; i < capacity; ++i)
    {
        tensors[i] = GetRandomTensor(rg, nvcv::FMT_RGB8);
    }
    tb.popTensors(10);
    tb.pushBack(tensors.begin() + capacity - 10, tensors.end());
    ASSERT_EQ(tb.numTensors(), capacity);

    tb.setTensor(capacity / 2 + 1, tensorA); // set at clean position
    tb.setTensor(capacity - 2, tensorB);     // set at dirty position
    data      = tb.exportData(nullptr);
    result    = tensors;
    result[0] = result[capacity / 4] = tensorA;
    result[capacity / 2]             = tensorB;
    result[capacity / 2 + 1]         = tensorA;
    result[capacity - 2]             = tensorB;
    CheckTensorBatchData(data, result.begin(), result.end(), nullptr);
    result.clear();
    EXPECT_EQ(tensors[0].refCount(), 1);
    EXPECT_EQ(tensors[capacity / 4].refCount(), 1);
    EXPECT_EQ(tensors[capacity / 2].refCount(), 1);
    EXPECT_EQ(tensors[capacity / 2 + 1].refCount(), 1);
    EXPECT_EQ(tensors[capacity - 2].refCount(), 1);
    EXPECT_EQ(tensorA.refCount(), 4);
    EXPECT_EQ(tensorB.refCount(), 3);
}

TEST(TensorBatch, valid_get_allocator)
{
    int                         tmp = 1;
    NVCVTensorBatchHandle       tensorBatchHandle;
    NVCVTensorBatchRequirements req;
    NVCVAllocatorHandle         alloc = reinterpret_cast<NVCVAllocatorHandle>(&tmp);
    EXPECT_NE(alloc, nullptr);

    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchCalcRequirements(16, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchConstruct(&req, nullptr, &tensorBatchHandle));

    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchGetAllocator(tensorBatchHandle, &alloc));
    EXPECT_EQ(alloc, nullptr);
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchDecRef(tensorBatchHandle, nullptr));
}

TEST(TensorBatch, invalid_out_get_allocator)
{
    NVCVTensorBatchHandle       tensorBatchHandle;
    NVCVTensorBatchRequirements req;

    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchCalcRequirements(16, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchConstruct(&req, nullptr, &tensorBatchHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetAllocator(tensorBatchHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchDecRef(tensorBatchHandle, nullptr));
}

TEST(TensorBatch, calc_req_invalid_arg)
{
    // output is nullptr
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchCalcRequirements(32, nullptr));
}

TEST(TensorBatch, construct_invalid_arg)
{
    NVCVTensorBatchHandle       tensorBatchHandle;
    NVCVTensorBatchRequirements req;
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchCalcRequirements(16, &req));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchConstruct(nullptr, nullptr, &tensorBatchHandle));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchConstruct(&req, nullptr, nullptr));
}

TEST(TensorBatch, push_invalid_arg)
{
    NVCVTensorBatchHandle       tensorBatchHandle;
    NVCVTensorBatchRequirements req;
    NVCVTensorHandle            inTensors;
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchCalcRequirements(16, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchConstruct(&req, nullptr, &tensorBatchHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchPushTensors(tensorBatchHandle, nullptr, 1));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchPushTensors(tensorBatchHandle, &inTensors, 0));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchPushTensors(tensorBatchHandle, &inTensors, -1));

    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchDecRef(tensorBatchHandle, nullptr));
}

TEST(TensorBatch, ref_count_invalid_arg)
{
    NVCVTensorBatchHandle       tensorBatchHandle;
    NVCVTensorBatchRequirements req;
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchCalcRequirements(16, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchConstruct(&req, nullptr, &tensorBatchHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchRefCount(tensorBatchHandle, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchDecRef(tensorBatchHandle, nullptr));
}

TEST(TensorBatch, get_invalid_arg)
{
    NVCVTensorBatchHandle       tensorBatchHandle;
    NVCVTensorBatchRequirements req;
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchCalcRequirements(16, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchConstruct(&req, nullptr, &tensorBatchHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetCapacity(tensorBatchHandle, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetRank(tensorBatchHandle, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetDType(tensorBatchHandle, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetLayout(tensorBatchHandle, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetType(tensorBatchHandle, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetNumTensors(tensorBatchHandle, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetUserPointer(tensorBatchHandle, nullptr));

    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchDecRef(tensorBatchHandle, nullptr));
}

TEST(TensorBatch, export_invalid_arg)
{
    NVCVTensorBatchHandle       tensorBatchHandle;
    NVCVTensorBatchRequirements req;
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchCalcRequirements(16, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchConstruct(&req, nullptr, &tensorBatchHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchExportData(tensorBatchHandle, nullptr, nullptr));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchDecRef(tensorBatchHandle, nullptr));
}

TEST(TensorBatch, getTensors_invalid_arg)
{
    NVCVTensorBatchHandle       tensorBatchHandle;
    NVCVTensorBatchRequirements req;
    NVCVTensorHandle            outTensors;
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchCalcRequirements(16, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchConstruct(&req, nullptr, &tensorBatchHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetTensors(nullptr, 0, &outTensors, 1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetTensors(tensorBatchHandle, -1, &outTensors, 1));
    EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcvTensorBatchGetTensors(tensorBatchHandle, 0, &outTensors, 32));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetTensors(tensorBatchHandle, 0, nullptr, 1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchGetTensors(tensorBatchHandle, 0, &outTensors, -1));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchDecRef(tensorBatchHandle, nullptr));
}

TEST(TensorBatch, setTensors_invalid_arg)
{
    NVCVTensorBatchHandle       tensorBatchHandle;
    NVCVTensorBatchRequirements req;
    NVCVTensorHandle            inTensors;
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchCalcRequirements(16, &req));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchConstruct(&req, nullptr, &tensorBatchHandle));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchSetTensors(nullptr, 0, &inTensors, 1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchSetTensors(tensorBatchHandle, -1, &inTensors, 1));
    EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcvTensorBatchSetTensors(tensorBatchHandle, 0, &inTensors, 32));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchSetTensors(tensorBatchHandle, 0, nullptr, 1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvTensorBatchSetTensors(tensorBatchHandle, 0, &inTensors, -1));
    EXPECT_EQ(NVCV_SUCCESS, nvcvTensorBatchDecRef(tensorBatchHandle, nullptr));
}

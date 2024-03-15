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

#include "DeviceTensorBatchWrap.hpp"

#include <common/HashUtils.hpp>      // for NVCV_INSTANTIATE_TEST_SUITE_P, etc.
#include <common/TypedTests.hpp>     // for NVCV_MIXTYPED_TEST_SUITE_P, etc.
#include <common/ValueTests.hpp>     // for StringLiteral
#include <nvcv/Image.hpp>            // for Image, etc.
#include <nvcv/TensorBatch.hpp>      // for TensorBatch
#include <nvcv/TensorDataAccess.hpp> // for TensorDataAccessStridedImagePlanar, etc.
#include <nvcv/cuda/MathOps.hpp>     // for operator == to allow EXPECT_EQ
#include <nvcv/cuda/TensorBatchWrap.hpp>

#include <limits>
#include <random>
#include <type_traits>

namespace t     = ::testing;
namespace test  = nvcv::test;
namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

static constexpr int kMaxDim = 50;

template<typename T, int NDIM, int INNER_DIM = -1, typename R>
nvcv::Tensor GetRandomTensor(R &rg, nvcv::DataType dtype, cudaStream_t stream)
{
    std::uniform_int_distribution<int32_t> shape_dist(kMaxDim / 2, kMaxDim);
    nvcv::TensorShape::ShapeType           shapeData(NDIM);
    for (auto &d : shapeData)
    {
        d = shape_dist(rg);
    }
    if (INNER_DIM != -1)
    {
        shapeData[NDIM - 1] = INNER_DIM;
    }
    auto t = nvcv::Tensor(nvcv::TensorShape(shapeData, ""), dtype);
    return t;
}

template<typename T, int NDIM, int N = 0>
void VerifyTensorHelper(NVCVByte *data, const int64_t *shape, const int64_t *stride, int64_t startIndex = 0)
{
    if constexpr (N == NDIM)
    {
        auto gold  = cuda::SetAll<T>(startIndex % 255);
        auto value = *reinterpret_cast<T *>(data);
        ASSERT_EQ(value, gold);
    }
    else
    {
        int64_t indexStride = 1;
        for (int i = 1; i + N < NDIM; ++i)
        {
            indexStride *= shape[i];
        }
        for (int i = 0; i < shape[0]; ++i)
        {
            VerifyTensorHelper<T, NDIM, N + 1>(data, shape + 1, stride + 1, startIndex + i * indexStride);
            data += stride[0];
        }
    }
}

template<typename T, int NDIM>
void VerifyTensor(const nvcv::Tensor &tensor, cudaStream_t stream)
{
    auto                  data       = tensor.exportData().cdata();
    auto                  bufferSize = data.shape[0] * data.buffer.strided.strides[0];
    std::vector<NVCVByte> hostBuffer(bufferSize);
    ASSERT_EQ(
        cudaMemcpyAsync(hostBuffer.data(), data.buffer.strided.basePtr, bufferSize, cudaMemcpyDeviceToHost, stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    VerifyTensorHelper<T, NDIM>(hostBuffer.data(), &data.shape[0], &data.buffer.strided.strides[0]);
}

template<typename T, int NDIM, int INNER_DIM, int N, int... Strides>
struct TensorBatchWrapHelper
{
    using type = TensorBatchWrapHelper<T, NDIM, INNER_DIM, N + 1, -1, Strides...>::type;
};

template<typename T, int NDIM, int INNER_DIM>
struct TensorBatchWrapHelper<T, NDIM, INNER_DIM, 0>
{
    using type
        = TensorBatchWrapHelper<T, NDIM, INNER_DIM, 2, (INNER_DIM != -1) ? INNER_DIM * sizeof(T) : -1, sizeof(T)>::type;
};

template<typename T, int INNER_DIM>
struct TensorBatchWrapHelper<T, 1, INNER_DIM, 0>
{
    using type = cuda::TensorBatchWrap<T, sizeof(T)>;
};

template<typename T, int NDIM, int INNER_DIM, int... Strides>
struct TensorBatchWrapHelper<T, NDIM, INNER_DIM, NDIM, Strides...>
{
    using type = cuda::TensorBatchWrap<T, Strides...>;
};

template<typename T, int NDIM, int INNER_DIM>
using TensorBatchWrapHelperT = typename TensorBatchWrapHelper<T, NDIM, INNER_DIM, 0>::type;

#define NVCV_TEST_ROW(NUM_TENSORS, DTYPE, TYPE, NDIM, INNER_DIM, SET_VALUE_METHOD)                                  \
    ttype::Types<ttype::Value<NUM_TENSORS>, ttype::Value<DTYPE>, TYPE, ttype::Value<NDIM>, ttype::Value<INNER_DIM>, \
                 SET_VALUE_METHOD<TensorBatchWrapHelperT<TYPE, NDIM, INNER_DIM>, TYPE>>

NVCV_TYPED_TEST_SUITE(TensorBatchWrapTensorTest,
                      ttype::Types<NVCV_TEST_ROW(16, NVCV_DATA_TYPE_U8, uchar1, 3, 32, SetThroughTensor),
                                   NVCV_TEST_ROW(8, NVCV_DATA_TYPE_4F64, double4, 2, 8, SetThroughTensor),
                                   NVCV_TEST_ROW(16, NVCV_DATA_TYPE_3F32, float3, 4, 8, SetThroughTensor),
                                   NVCV_TEST_ROW(64, NVCV_DATA_TYPE_2U8, uchar2, 1, -1, SetThroughSubscript),
                                   NVCV_TEST_ROW(16, NVCV_DATA_TYPE_3S32, int3, 3, 16, SetThroughSubscript),
                                   NVCV_TEST_ROW(32, NVCV_DATA_TYPE_4U16, ushort4, 2, -1, SetThroughSubscript),
                                   NVCV_TEST_ROW(16, NVCV_DATA_TYPE_4U8, uchar4, 4, 32, SetThroughPtr),
                                   NVCV_TEST_ROW(1, NVCV_DATA_TYPE_F32, float1, 5, 8, SetThroughPtr),
                                   NVCV_TEST_ROW(32, NVCV_DATA_TYPE_4F32, float4, 1, -1, SetThroughPtr)>);

#undef NVCV_TEST_ROW

TYPED_TEST(TensorBatchWrapTensorTest, correct_content)
{
    int            numTensors = ttype::GetValue<TypeParam, 0>;
    nvcv::DataType dtype{ttype::GetValue<TypeParam, 1>};
    using T                 = ttype::GetType<TypeParam, 2>;
    constexpr int NDIM      = ttype::GetValue<TypeParam, 3>;
    constexpr int INNER_DIM = ttype::GetValue<TypeParam, 4>;
    using SET_METHOD        = ttype::GetType<TypeParam, 5>;
    using TensorBatchWrapT  = TensorBatchWrapHelperT<T, NDIM, INNER_DIM>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    nvcv::TensorBatch         tensorBatch(numTensors);
    std::vector<nvcv::Tensor> tensors;

    std::mt19937 rg{231};
    for (int i = 0; i < tensorBatch.capacity(); ++i)
    {
        auto t = GetRandomTensor<T, NDIM, INNER_DIM>(rg, dtype, stream);
        ASSERT_EQ(t.rank(), NDIM);
        tensors.push_back(t);
    }

    tensorBatch.pushBack(tensors.begin(), tensors.end());

    auto tensorBatchData = tensorBatch.exportData(stream).cast<nvcv::TensorBatchDataStridedCuda>();
    ASSERT_TRUE(tensorBatchData.hasValue());

    auto wrap = TensorBatchWrapT(*tensorBatchData);
    SetReference<SET_METHOD>(wrap, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (auto &tensor : tensors)
    {
        VerifyTensor<T, NDIM>(tensor, stream);
    }
}

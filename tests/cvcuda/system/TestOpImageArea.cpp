/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/OpImageArea.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace {

// Independent CPU gold: the area of an image is its spatial pixel count, W * H. It is computed
// directly from the test parameters and never derived from the operator's output.
int64_t GoldArea(int width, int height)
{
    return static_cast<int64_t>(width) * height;
}

nvcv::Tensor CreateImageTensor(const std::string &layout, nvcv::DataType dtype, int batch, int ch, int w, int h)
{
    if (layout == "NHWC")
    {
        return nvcv::Tensor({{batch, h, w, ch}, "NHWC"}, dtype);
    }
    if (layout == "NCHW")
    {
        return nvcv::Tensor({{batch, ch, h, w}, "NCHW"}, dtype);
    }
    if (layout == "HWC")
    {
        return nvcv::Tensor({{h, w, ch}, "HWC"}, dtype);
    }
    return nvcv::Tensor({{ch, h, w}, "CHW"}, dtype);
}

void RunImageArea(const std::string &layout, nvcv::DataType dtype, int batch, int ch, int w, int h, int outBatch,
                  std::vector<int32_t> &result)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor in  = CreateImageTensor(layout, dtype, batch, ch, w, h);
    nvcv::Tensor out = nvcv::Tensor({{outBatch}, "N"}, nvcv::TYPE_S32);

    cvcuda::ImageArea op;
    ASSERT_NO_THROW(op(stream, in, out));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(outData, nullptr);

    result.resize(outBatch);
    for (int n = 0; n < outBatch; ++n)
    {
        const int64_t offset = static_cast<int64_t>(n) * outData->stride(0);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(result.data() + n, outData->basePtr() + offset, sizeof(int32_t),
                                          cudaMemcpyDeviceToHost));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

void CheckImageArea(const std::string &layout, nvcv::DataType dtype, int batch, int ch, int w, int h)
{
    std::vector<int32_t> actual;
    RunImageArea(layout, dtype, batch, ch, w, h, /*outBatch=*/batch, actual);
    ASSERT_EQ(actual.size(), static_cast<size_t>(batch));

    const int64_t expected = GoldArea(w, h);
    for (int n = 0; n < batch; ++n)
    {
        EXPECT_EQ(actual[n], expected) << "layout=" << layout << " sample=" << n << " area=" << w << "x" << h;
    }
}

} // namespace

TEST(OpImageArea, NHWC_Interleaved)
{
    for (int channels = 1; channels <= 4; ++channels)
    {
        CheckImageArea("NHWC", nvcv::TYPE_U8, 5, channels, 23, 17);
    }
}

TEST(OpImageArea, NCHW_Planar)
{
    for (int channels = 2; channels <= 4; ++channels)
    {
        CheckImageArea("NCHW", nvcv::TYPE_U8, 5, channels, 23, 17);
    }
}

TEST(OpImageArea, Unbatched_Interleaved)
{
    CheckImageArea("HWC", nvcv::TYPE_U8, 1, 3, 41, 13);
}

TEST(OpImageArea, Unbatched_Planar)
{
    CheckImageArea("CHW", nvcv::TYPE_U8, 1, 3, 41, 13);
}

TEST(OpImageArea, Float32Input)
{
    CheckImageArea("NHWC", nvcv::TYPE_F32, 3, 3, 64, 32);
}

TEST(OpImageArea, DtypeDoesNotChangeArea)
{
    // Same geometry, four different element types: the reported area must be identical.
    for (auto dtype : {nvcv::TYPE_U8, nvcv::TYPE_S16, nvcv::TYPE_F32, nvcv::TYPE_F64})
    {
        CheckImageArea("NCHW", dtype, 2, 1, 100, 50);
    }
}

TEST(OpImageArea, RejectsWrongOutputDtype)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor in  = CreateImageTensor("NHWC", nvcv::TYPE_U8, 2, 3, 8, 8);
    nvcv::Tensor out = nvcv::Tensor({{2}, "N"}, nvcv::TYPE_F32);

    cvcuda::ImageArea op;
    EXPECT_THROW(op(stream, in, out), nvcv::Exception);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpImageArea, RejectsMismatchedBatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor in  = CreateImageTensor("NHWC", nvcv::TYPE_U8, 2, 3, 8, 8);
    nvcv::Tensor out = nvcv::Tensor({{3}, "N"}, nvcv::TYPE_S32);

    cvcuda::ImageArea op;
    EXPECT_THROW(op(stream, in, out), nvcv::Exception);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpImageArea, RejectsUnsupportedLayout)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor in  = nvcv::Tensor({{2, 8, 8}, "NHW"}, nvcv::TYPE_U8);
    nvcv::Tensor out = nvcv::Tensor({{2}, "N"}, nvcv::TYPE_S32);

    cvcuda::ImageArea op;
    EXPECT_THROW(op(stream, in, out), nvcv::Exception);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

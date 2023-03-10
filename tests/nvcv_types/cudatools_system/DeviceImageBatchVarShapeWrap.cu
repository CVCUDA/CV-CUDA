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

#include "DeviceImageBatchVarShapeWrap.hpp"

#include <gtest/gtest.h>            // for EXPECT_EQ, etc.
#include <nvcv/cuda/MathOps.hpp>    // for operator == to allow EXPECT_EQ
#include <nvcv/cuda/StaticCast.hpp> // for StaticCast, etc.

namespace cuda = nvcv::cuda;

// ----------- To allow testing device-side ImageBatchVarShapeWrap -------------

#define CUDA_EXPECT_EQ(A, B) \
    if (A != B)              \
    {                        \
        assert(false);       \
        return;              \
    }

template<typename PixelType>
__global__ void SetTwos(cuda::ImageBatchVarShapeWrap<PixelType> dst, int numSamples)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (coord.x >= dst.width(coord.z) || coord.y >= dst.height(coord.z) || coord.z >= numSamples)
    {
        return;
    }

    CUDA_EXPECT_EQ(dst.plane(coord.z).width, dst.width(coord.z));
    CUDA_EXPECT_EQ(dst.plane(coord.z).height, dst.height(coord.z));
    CUDA_EXPECT_EQ(dst.plane(coord.z).rowStride, dst.rowStride(coord.z));

    *dst.ptr(coord.z, coord.y, coord.x) = cuda::SetAll<PixelType>(1);

    dst[coord] += cuda::SetAll<PixelType>(1);
}

template<typename ChannelType>
__global__ void SetTwos(cuda::ImageBatchVarShapeWrapNHWC<ChannelType> dst, int numSamples)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (coord.x >= dst.width(coord.z) || coord.y >= dst.height(coord.z) || coord.z >= numSamples)
    {
        return;
    }

    CUDA_EXPECT_EQ(dst.plane(coord.z).width, dst.width(coord.z));
    CUDA_EXPECT_EQ(dst.plane(coord.z).height, dst.height(coord.z));
    CUDA_EXPECT_EQ(dst.plane(coord.z).rowStride, dst.rowStride(coord.z));

    for (int ch = 0; ch < dst.numChannels(); ++ch)
    {
        *dst.ptr(coord.z, coord.y, coord.x, ch) = cuda::SetAll<ChannelType>(1);

        int4 dstCoord{coord.x, coord.y, coord.z, ch};
        dst[dstCoord] += cuda::SetAll<ChannelType>(1);
    }
}

template<class DstWrapper>
void DeviceSetTwos(DstWrapper &dst, int3 maxSize, cudaStream_t &stream)
{
    dim3 block{32, 2, 2};
    dim3 grid{(maxSize.x + block.x - 1) / block.x, (maxSize.y + block.y - 1) / block.y,
              (maxSize.z + block.z - 1) / block.z};

    SetTwos<<<grid, block, 0, stream>>>(dst, maxSize.z);
}

#define NVCV_TEST_INST_SET(PIXEL_TYPE) \
    template void DeviceSetTwos(cuda::ImageBatchVarShapeWrap<PIXEL_TYPE> &, int3, cudaStream_t &)

NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(char1);
NVCV_TEST_INST_SET(int1);
NVCV_TEST_INST_SET(float1);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

#define NVCV_TEST_INST_SET(CHANNEL_TYPE) \
    template void DeviceSetTwos(cuda::ImageBatchVarShapeWrapNHWC<CHANNEL_TYPE> &, int3, cudaStream_t &)

NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(float1);

#undef NVCV_TEST_INST_SET

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Nvtx.hpp"
#include "OpChannelReorder.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

struct TensorByteView
{
    unsigned char *base;
    int64_t        sampleStride;
    int64_t        rowStride;
    int64_t        colStride;
    int64_t        channelStride;
};

inline __device__ int GetOrder(int4 order, int channel)
{
    switch (channel)
    {
    case 0:
        return order.x;
    case 1:
        return order.y;
    case 2:
        return order.z;
    default:
        return order.w;
    }
}

template<typename T>
__global__ void ChannelReorderTensorKernel(TensorByteView src, TensorByteView dst, int64_t totalPixels, int rows,
                                           int cols, int channels, int4 order)
{
    const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step   = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pixel = thread; pixel < totalPixels; pixel += step)
    {
        const int     x = static_cast<int>(pixel % cols);
        const int64_t q = pixel / cols;
        const int     y = static_cast<int>(q % rows);
        const int64_t n = q / rows;

        const int64_t srcPixelOffset = n * src.sampleStride + y * src.rowStride + x * src.colStride;
        const int64_t dstPixelOffset = n * dst.sampleStride + y * dst.rowStride + x * dst.colStride;

#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            if (c >= channels)
            {
                break;
            }
            const int srcChannel = GetOrder(order, c);
            T        *dstValue
                = reinterpret_cast<T *>(dst.base + dstPixelOffset + static_cast<int64_t>(c) * dst.channelStride);
            if (srcChannel < 0)
            {
                *dstValue = T{};
            }
            else
            {
                const T *srcValue = reinterpret_cast<const T *>(src.base + srcPixelOffset
                                                                + static_cast<int64_t>(srcChannel) * src.channelStride);
                *dstValue         = *srcValue;
            }
        }
    }
}

inline int4 PackOrder(const int32_t *order, int32_t orderLength, int channels)
{
    if (order == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Order pointer must not be NULL");
    }
    if (orderLength != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Order length must match the input channel count");
    }

    int values[4]{-1, -1, -1, -1};
    for (int i = 0; i < channels; ++i)
    {
        if (order[i] >= channels)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Non-negative order entries must be smaller than the channel count");
        }
        values[i] = order[i];
    }
    return {values[0], values[1], values[2], values[3]};
}

inline int ElementSize(nvcv::DataType dtype)
{
    if (dtype == nvcv::TYPE_U8)
    {
        return 1;
    }
    if (dtype == nvcv::TYPE_U16 || dtype == nvcv::TYPE_S16)
    {
        return 2;
    }
    if (dtype == nvcv::TYPE_S32 || dtype == nvcv::TYPE_F32)
    {
        return 4;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "ChannelReorder supports U8, U16, S16, S32, and F32 tensors");
}

template<typename T>
inline void LaunchChannelReorderTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                                       const nvcv::TensorDataStridedCuda              &dstData,
                                       const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                       const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, int channels,
                                       int4 order)
{
    const int64_t samples = srcAccess.numSamples();
    const int64_t rows    = srcAccess.numRows();
    const int64_t cols    = srcAccess.numCols();
    if (samples == 0 || rows == 0 || cols == 0)
    {
        return;
    }
    if (samples > std::numeric_limits<int64_t>::max() / rows
        || samples * rows > std::numeric_limits<int64_t>::max() / cols)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Tensor element count overflows int64");
    }
    const int64_t totalPixels  = samples * rows * cols;
    constexpr int threads      = 256;
    const int64_t neededBlocks = (totalPixels + threads - 1) / threads;
    const int     blocks       = static_cast<int>(std::min<int64_t>(neededBlocks, 65535));

    TensorByteView src{reinterpret_cast<unsigned char *>(srcData.basePtr()), srcAccess.sampleStride(),
                       srcAccess.rowStride(), srcAccess.colStride(), srcAccess.chStride()};
    TensorByteView dst{reinterpret_cast<unsigned char *>(dstData.basePtr()), dstAccess.sampleStride(),
                       dstAccess.rowStride(), dstAccess.colStride(), dstAccess.chStride()};
    ChannelReorderTensorKernel<T><<<blocks, threads, 0, stream>>>(src, dst, totalPixels, static_cast<int>(rows),
                                                                  static_cast<int>(cols), channels, order);
    NVCV_CHECK_THROW(cudaGetLastError());
}

} // namespace

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

ChannelReorder::ChannelReorder()
{
    legacy::DataShape maxIn;
    legacy::DataShape maxOut; //maxIn/maxOut not used by op.
    m_legacyOpVarShape = std::make_unique<legacy::ChannelReorderVarShape>(maxIn, maxOut);
}

void ChannelReorder::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                const int32_t *order, int32_t orderLength) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ChannelReorder::operator()[Tensor]");
    auto inData  = in.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (!inData || !outData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must be CUDA-accessible pitch-linear tensors");
    }
    if (inData->layout() != outData->layout()
        || !(inData->layout() == nvcv::TENSOR_HWC || inData->layout() == nvcv::TENSOR_NHWC
             || inData->layout() == nvcv::TENSOR_CHW || inData->layout() == nvcv::TENSOR_NCHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same (N)HWC or (N)CHW layout");
    }
    if (inData->rank() != outData->rank())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output ranks must match");
    }
    for (int i = 0; i < inData->rank(); ++i)
    {
        if (inData->shape(i) != outData->shape(i))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output shapes must match");
        }
    }
    if (inData->dtype() != outData->dtype() || inData->dtype().numChannels() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same scalar data type");
    }

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    if (!inAccess || !outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must support strided planar image access");
    }
    const int channels = inAccess->numChannels();
    if (channels < 1 || channels > 4 || (inAccess->numPlanes() > 1 && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "ChannelReorder supports 1-4 interleaved channels and 1, 3, or 4 planar channels");
    }
    const int4 packedOrder = PackOrder(order, orderLength, channels);
    const int  elementSize = ElementSize(inData->dtype());
    if (elementSize == 1)
    {
        LaunchChannelReorderTensor<uint8_t>(stream, *inData, *outData, *inAccess, *outAccess, channels, packedOrder);
    }
    else if (elementSize == 2)
    {
        LaunchChannelReorderTensor<uint16_t>(stream, *inData, *outData, *inAccess, *outAccess, channels, packedOrder);
    }
    else
    {
        LaunchChannelReorderTensor<uint32_t>(stream, *inData, *outData, *inAccess, *outAccess, channels, packedOrder);
    }
}

void ChannelReorder::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &orders) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ChannelReorder::operator()[ImageBatchVarShape]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto ordersData = orders.exportData<nvcv::TensorDataStridedCuda>();
    if (!ordersData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input channel order tensor must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *outData, *ordersData, stream));
}

} // namespace cvcuda::priv

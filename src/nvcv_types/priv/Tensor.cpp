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

#include "Tensor.hpp"

#include "DataLayout.hpp"
#include "DataType.hpp"
#include "IAllocator.hpp"
#include "Requirements.hpp"
#include "TensorData.hpp"
#include "TensorLayout.hpp"
#include "TensorShape.hpp"

#include <cuda_runtime.h>
#include <util/Assert.h>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cmath>
#include <numeric>

namespace nvcv::priv {

// Tensor implementation -------------------------------------------

NVCVTensorRequirements Tensor::CalcRequirements(int32_t numImages, Size2D imgSize, ImageFormat fmt,
                                                int32_t userBaseAlign, int32_t userRowAlign)
{
    // Check if format is compatible with tensor representation
    if (fmt.memLayout() != NVCV_MEM_LAYOUT_PL)
    {
        throw Exception(NVCV_ERROR_NOT_IMPLEMENTED,
                        "Tensor image batch of block-linear format images is not currently supported.");
    }

    if (fmt.css() != NVCV_CSS_444)
    {
        throw Exception(NVCV_ERROR_NOT_IMPLEMENTED)
            << "Batch image format must not have subsampled planes, but it is: " << fmt;
    }

    if (fmt.numPlanes() != 1 && fmt.numPlanes() != fmt.numChannels())
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format cannot be semi-planar, but it is: " << fmt;
    }

    for (int p = 1; p < fmt.numPlanes(); ++p)
    {
        if (fmt.planePacking(p) != fmt.planePacking(0))
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Format's planes must all have the same packing, but they don't: " << fmt;
        }
    }

    // Calculate the shape based on image parameters
    NVCVTensorLayout layout = GetTensorLayoutFor(fmt, numImages);

    int64_t shapeNCHW[4] = {numImages, fmt.numChannels(), imgSize.h, imgSize.w};

    int64_t shape[NVCV_TENSOR_MAX_RANK];
    PermuteShape(NVCV_TENSOR_NCHW, shapeNCHW, layout, shape);

    // Calculate the element type. It's the data type of the
    // first channel. It assumes that all channels have same packing.
    NVCVPackingParams params = GetPackingParams(fmt.planePacking(0));
    params.swizzle           = NVCV_SWIZZLE_X000;
    std::fill(params.bits + 1, params.bits + sizeof(params.bits) / sizeof(params.bits[0]), 0);
    std::optional<NVCVPacking> chPacking = MakeNVCVPacking(params);
    if (!chPacking)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format can't be represented in a tensor: " << fmt;
    }

    DataType dtype{fmt.dataKind(), *chPacking};

    return CalcRequirements(layout.rank, shape, dtype, layout, userBaseAlign, userRowAlign);
}

NVCVTensorRequirements Tensor::CalcRequirements(int32_t rank, const int64_t *shape, const DataType &dtype,
                                                NVCVTensorLayout layout, int32_t userBaseAlign, int32_t userRowAlign)
{
    NVCVTensorRequirements reqs;

    reqs.layout = layout;
    reqs.dtype  = dtype.value();

    if (layout.rank > 0 && rank != layout.rank)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Number of shape dimensions " << rank << " must be equal to layout dimensions " << layout.rank;
    }

    if (rank <= 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Number of dimensions must be >= 1, not %d", rank);
    }

    std::copy_n(shape, rank, reqs.shape);
    reqs.rank = rank;

    reqs.mem = {};

    int dev;
    NVCV_CHECK_THROW(cudaGetDevice(&dev));

    // Calculate row pitch alignment
    int rowAlign;
    {
        if (userRowAlign == 0)
        {
            // it usually returns 32 bytes
            NVCV_CHECK_THROW(cudaDeviceGetAttribute(&rowAlign, cudaDevAttrTexturePitchAlignment, dev));
            rowAlign = std::lcm(rowAlign, util::RoundUpNextPowerOfTwo(dtype.strideBytes()));
        }
        else
        {
            if (!util::IsPowerOfTwo(userRowAlign))
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                    << "Invalid pitch alignment of " << userRowAlign << ", it must be a power-of-two";
            }
            // must at least satisfy dtype alignment
            rowAlign = std::lcm(userRowAlign, dtype.alignment());
        }
    }

    // Calculate base address alignment
    {
        if (userBaseAlign == 0)
        {
            int addrAlign;
            // it usually returns 512 bytes
            NVCV_CHECK_THROW(cudaDeviceGetAttribute(&addrAlign, cudaDevAttrTextureAlignment, dev));
            reqs.alignBytes = std::lcm(addrAlign, rowAlign);
            reqs.alignBytes = util::RoundUpNextPowerOfTwo(reqs.alignBytes);

            if (reqs.alignBytes > NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                "Alignment requirement of %d is larger than the maximum allowed %ld", reqs.alignBytes,
                                NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE);
            }
        }
        else
        {
            if (!util::IsPowerOfTwo(userBaseAlign))
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                    << "Invalid base address alignment of " << userBaseAlign << ", it must be a power-of-two";
            }

            reqs.alignBytes = std::lcm(rowAlign, userBaseAlign);
        }
    }

    int firstPacked = reqs.layout == NVCV_TENSOR_NHWC ? std::max(0, rank - 2) : rank - 1;

    reqs.strides[rank - 1] = dtype.strideBytes();
    for (int d = rank - 2; d >= 0; --d)
    {
        if (d == firstPacked - 1)
        {
            reqs.strides[d] = util::RoundUpPowerOfTwo(reqs.shape[d + 1] * reqs.strides[d + 1], rowAlign);
        }
        else
        {
            reqs.strides[d] = reqs.strides[d + 1] * reqs.shape[d + 1];
        }
    }

    AddBuffer(reqs.mem.cudaMem, reqs.strides[0] * reqs.shape[0], reqs.alignBytes);

    return reqs;
}

Tensor::Tensor(NVCVTensorRequirements reqs, IAllocator &alloc)
    : m_alloc{alloc}
    , m_reqs{std::move(reqs)}
{
    // Assuming reqs are already validated during its creation

    int64_t bufSize = CalcTotalSizeBytes(m_reqs.mem.cudaMem);
    m_memBuffer     = m_alloc.allocCudaMem(bufSize, m_reqs.alignBytes);
    NVCV_ASSERT(m_memBuffer != nullptr);
}

Tensor::~Tensor()
{
    m_alloc.freeCudaMem(m_memBuffer, CalcTotalSizeBytes(m_reqs.mem.cudaMem), m_reqs.alignBytes);
}

int32_t Tensor::rank() const
{
    return m_reqs.rank;
}

const int64_t *Tensor::shape() const
{
    return m_reqs.shape;
}

const NVCVTensorLayout &Tensor::layout() const
{
    return m_reqs.layout;
}

DataType Tensor::dtype() const
{
    return DataType{m_reqs.dtype};
}

IAllocator &Tensor::alloc() const
{
    return m_alloc;
}

void Tensor::exportData(NVCVTensorData &data) const
{
    data.bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;

    data.dtype  = m_reqs.dtype;
    data.layout = m_reqs.layout;
    data.rank   = m_reqs.rank;

    memcpy(data.shape, m_reqs.shape, sizeof(data.shape));

    NVCVTensorBufferStrided &buf = data.buffer.strided;
    {
        static_assert(sizeof(buf.strides) == sizeof(m_reqs.strides));
        static_assert(
            std::is_same_v<std::decay_t<decltype(buf.strides[0])>, std::decay_t<decltype(m_reqs.strides[0])>>);
        memcpy(buf.strides, m_reqs.strides, sizeof(buf.strides));

        buf.basePtr = reinterpret_cast<NVCVByte *>(m_memBuffer);
    }
}

} // namespace nvcv::priv

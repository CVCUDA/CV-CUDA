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

/**
 * @file OpCLAHE.hpp
 *
 * @brief Defines the private C++ Class for CLAHE operation.
 */

#ifndef CVCUDA_PRIV__CLAHE_HPP
#define CVCUDA_PRIV__CLAHE_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

// Per-device GPU buffers for CLAHE. One instance is created per CUDA device
// via PerDeviceResource, ensuring each GPU has its own LUT allocation.
struct CLAHEDeviceBuffers
{
    unsigned char *luts = nullptr;

    CLAHEDeviceBuffers(int32_t maxBatchSize, int32_t tilesX, int32_t tilesY)
    {
        const size_t lutSize = static_cast<size_t>(maxBatchSize) * static_cast<size_t>(tilesX)
                             * static_cast<size_t>(tilesY) * 256 * sizeof(unsigned char);
        NVCV_CHECK_THROW(cudaMalloc(&luts, lutSize));
    }

    ~CLAHEDeviceBuffers()
    {
        if (luts != nullptr)
        {
            NVCV_CHECK_LOG(cudaFree(luts));
        }
    }

    CLAHEDeviceBuffers(const CLAHEDeviceBuffers &)            = delete;
    CLAHEDeviceBuffers &operator=(const CLAHEDeviceBuffers &) = delete;
    CLAHEDeviceBuffers(CLAHEDeviceBuffers &&)                 = delete;
    CLAHEDeviceBuffers &operator=(CLAHEDeviceBuffers &&)      = delete;
};

class CLAHE final : public IOperator
{
public:
    CLAHE(int32_t maxBatchSize, int32_t tilesX, int32_t tilesY);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float clipLimit) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    float clipLimit) const;

private:
    int32_t                                       m_maxBatchSize;
    int32_t                                       m_tilesX;
    int32_t                                       m_tilesY;
    mutable PerDeviceResource<CLAHEDeviceBuffers> m_deviceBuffers;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__CLAHE_HPP

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

/**
 * @file OpInpaint.hpp
 *
 * @brief Defines the private C++ Class for the inpaint operation.
 */

#ifndef CVCUDA_PRIV_INPAINT_HPP
#define CVCUDA_PRIV_INPAINT_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

namespace cvcuda::priv {

class Inpaint final : public IOperator
{
public:
    explicit Inpaint(int maxBatchSize, nvcv::Size2D maxShape);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &masks, const nvcv::Tensor &out,
                    double inpaintRadius) const;
    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &masks,
                    const nvcv::ImageBatchVarShape &out, double inpaintRadius) const;

private:
    // Persistent device scratch, sized once from (maxBatchSize, maxShape): the 3x3 cross used to
    // dilate the mask into the narrow band, and the fast-marching workspace holding the reduction
    // buffers plus the t/f/band/mask planes.  The cross is uploaded on the first submit, which is
    // what initDilate tracks.
    struct DeviceBuffers
    {
        DeviceBuffers(int maxBatchSize, nvcv::Size2D maxShape);
        ~DeviceBuffers();

        DeviceBuffers(const DeviceBuffers &)            = delete;
        DeviceBuffers &operator=(const DeviceBuffers &) = delete;

        unsigned char *kernel     = nullptr;
        void          *workspace  = nullptr;
        bool           initDilate = false;
    };

    int32_t      m_maxBatchSize;
    nvcv::Size2D m_maxShape;

    // The Tensor and VarShape paths keep separate buffer sets, as the two legacy operators did.
    mutable PerDeviceResource<DeviceBuffers> m_buffers;
    mutable PerDeviceResource<DeviceBuffers> m_buffersVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_INPAINT_HPP

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
 * @file OpBoxBlur.hpp
 *
 * @brief Defines the private C++ Class for the BoxBlur operation.
 */

#ifndef CVCUDA_PRIV__BOX_BLUR_HPP
#define CVCUDA_PRIV__BOX_BLUR_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <cvcuda/Types.h>
#include <nvcv/Tensor.hpp>

#include <cstddef>
#include <vector>

namespace cvcuda::priv {

// One blur region as the kernels see it: the clamped, inclusive bounding box in image
// coordinates plus the sample it belongs to.
struct BoxBlurCommand
{
    int boundingLeft;
    int boundingTop;
    int boundingRight;
    int boundingBottom;
    int batchIndex;
    int kernelSize;
};

class BoxBlur final : public IOperator
{
public:
    explicit BoxBlur();

    ~BoxBlur() override;

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVBlurBoxesI &bboxes) const;

private:
    void ReserveCommandBuffer(size_t count) const;
    void FreeCommandBuffer() const noexcept;
    void UploadCommands(cudaStream_t stream) const;

    // Pinned staging buffer and its device mirror, owned by the operator for its whole lifetime.
    // Preallocated by the constructor and only ever grown, exactly as the legacy operator's cuOSD
    // context was; they are deliberately not per-call or workspace allocations.
    mutable std::vector<BoxBlurCommand> m_commands;
    mutable BoxBlurCommand             *m_hostCommands    = nullptr;
    mutable BoxBlurCommand             *m_deviceCommands  = nullptr;
    mutable size_t                      m_commandCapacity = 0;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__BOX_BLUR_HPP

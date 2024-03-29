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

/**
 * @file OpPillowResize.hpp
 *
 * @brief Defines the private C++ class for the pillow resize operation.
 */

#ifndef CVCUDA_PRIV_PILLOW_RESIZE_HPP
#define CVCUDA_PRIV_PILLOW_RESIZE_HPP

#include "IOperator.hpp"
#include "cvcuda/Workspace.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <memory>

namespace cvcuda::priv {

class PillowResize final : public IOperator
{
public:
    PillowResize();

    WorkspaceRequirements getWorkspaceRequirements(int batchSize, const nvcv::Size2D *in_sizes,
                                                   const nvcv::Size2D *out_sizes, NVCVImageFormat fmt);

    WorkspaceRequirements getWorkspaceRequirements(int batchSize, nvcv::Size2D maxInSize, nvcv::Size2D maxOutSize,
                                                   NVCVImageFormat fmt);

    void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVInterpolationType interpolation) const;

    void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::ImageBatchVarShape &in,
                    const nvcv::ImageBatchVarShape &out, const NVCVInterpolationType interpolation) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::PillowResize>         m_legacyOp;
    std::unique_ptr<nvcv::legacy::cuda_op::PillowResizeVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_PILLOWRESIZE_HPP

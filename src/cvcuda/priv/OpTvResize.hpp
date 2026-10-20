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
 * @file OpTvResize.hpp
 *
 * @brief Defines the private C++ Class for the TvResize operation.
 */

#ifndef CVCUDA_PRIV__TV_RESIZE_HPP
#define CVCUDA_PRIV__TV_RESIZE_HPP

#include "IOperator.hpp"

#include <cvcuda/Types.h>
#include <cvcuda/Workspace.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

namespace cvcuda::priv {

class TvResize final : public IOperator
{
public:
    explicit TvResize() = default;

    WorkspaceRequirements getWorkspaceRequirements(int batchSize, int32_t srcHeight, int32_t srcWidth,
                                                   int32_t dstHeight, int32_t dstWidth, int32_t numChannels,
                                                   NVCVInterpolationType interpolation, bool antialias) const;

    WorkspaceRequirements getWorkspaceRequirements(int batchSize, const int32_t *srcHeights, const int32_t *srcWidths,
                                                   const int32_t *dstHeights, const int32_t *dstWidths,
                                                   int32_t numChannels, NVCVInterpolationType interpolation,
                                                   bool antialias) const;

    void operator()(cudaStream_t stream, const Workspace &workspace, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    NVCVInterpolationType interpolation, bool antialias) const;

    void operator()(cudaStream_t stream, const Workspace &workspace, const nvcv::ImageBatchVarShape &in,
                    const nvcv::ImageBatchVarShape &out, NVCVInterpolationType interpolation, bool antialias) const;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__TV_RESIZE_HPP

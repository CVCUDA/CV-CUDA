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
 * @file OpMorphology.hpp
 *
 * @brief Defines the public C++ Class for the Morphology operation.
 * @defgroup NVCV_CPP_ALGORITHM_MORPHOLOGY Morphology
 * @{
 */

#ifndef CVCUDA_MORPHOLOGY_HPP
#define CVCUDA_MORPHOLOGY_HPP

#include "IOperator.hpp"
#include "OpMorphology.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Morphology final : public IOperator
{
public:
    explicit Morphology();

    ~Morphology();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    nvcv::OptionalTensorConstRef workspace, NVCVMorphologyType morphType, const nvcv::Size2D &maskSize,
                    const int2 &anchor, int32_t iteration, const NVCVBorderType borderMode);

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::OptionalImageBatchVarShapeConstRef workspace, NVCVMorphologyType morphType,
                    const nvcv::Tensor &masks, const nvcv::Tensor &anchors, int32_t iteration,
                    const NVCVBorderType borderMode);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Morphology::Morphology()
{
    nvcv::detail::CheckThrow(cvcudaMorphologyCreate(&m_handle));
    assert(m_handle);
}

inline Morphology::~Morphology()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Morphology::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                   nvcv::OptionalTensorConstRef workspace, NVCVMorphologyType morphType,
                                   const nvcv::Size2D &maskSize, const int2 &anchor, int32_t iteration,
                                   const NVCVBorderType borderMode)
{
    nvcv::detail::CheckThrow(cvcudaMorphologySubmit(m_handle, stream, in.handle(), out.handle(),
                                                    NVCV_OPTIONAL_TO_HANDLE(workspace), morphType, maskSize.w,
                                                    maskSize.h, anchor.x, anchor.y, iteration, borderMode));
}

inline void Morphology::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                   const nvcv::ImageBatchVarShape                &out,
                                   const nvcv::OptionalImageBatchVarShapeConstRef workspace,
                                   NVCVMorphologyType morphType, const nvcv::Tensor &masks, const nvcv::Tensor &anchors,
                                   int32_t iteration, const NVCVBorderType borderMode)
{
    nvcv::detail::CheckThrow(cvcudaMorphologyVarShapeSubmit(m_handle, stream, in.handle(), out.handle(),
                                                            NVCV_OPTIONAL_TO_HANDLE(workspace), morphType,
                                                            masks.handle(), anchors.handle(), iteration, borderMode));
}

inline NVCVOperatorHandle Morphology::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_MORPHOLOGY_HPP

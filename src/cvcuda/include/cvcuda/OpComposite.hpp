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

/**
 * @file OpComposite.hpp
 *
 * @brief Defines the public C++ Class for the Composite operation.
 * @defgroup NVCV_CPP_ALGORITHM_COMPOSITE Composite
 * @{
 */

#ifndef CVCUDA_COMPOSITE_HPP
#define CVCUDA_COMPOSITE_HPP

#include "IOperator.hpp"
#include "OpComposite.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Composite final : public IOperator
{
public:
    explicit Composite();

    ~Composite();

    void operator()(cudaStream_t stream, nvcv::ITensor &foreground, nvcv::ITensor &background, nvcv::ITensor &fgMask,
                    nvcv::ITensor &output);

    void operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &foreground, nvcv::IImageBatchVarShape &background,
                    nvcv::IImageBatchVarShape &fgMask, nvcv::IImageBatchVarShape &output);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Composite::Composite()
{
    nvcv::detail::CheckThrow(cvcudaCompositeCreate(&m_handle));
    assert(m_handle);
}

inline Composite::~Composite()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Composite::operator()(cudaStream_t stream, nvcv::ITensor &foreground, nvcv::ITensor &background,
                                  nvcv::ITensor &fgMask, nvcv::ITensor &output)
{
    nvcv::detail::CheckThrow(cvcudaCompositeSubmit(m_handle, stream, foreground.handle(), background.handle(),
                                                   fgMask.handle(), output.handle()));
}

inline void Composite::operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &foreground,
                                  nvcv::IImageBatchVarShape &background, nvcv::IImageBatchVarShape &fgMask,
                                  nvcv::IImageBatchVarShape &output)
{
    nvcv::detail::CheckThrow(cvcudaCompositeVarShapeSubmit(m_handle, stream, foreground.handle(), background.handle(),
                                                           fgMask.handle(), output.handle()));
}

inline NVCVOperatorHandle Composite::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_COMPOSITE_HPP

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

/**
 * @file OpBndBox.hpp
 *
 * @brief Defines the public C++ Class for the BndBox operation.
 * @defgroup NVCV_CPP_ALGORITHM__BND_BOX BndBox
 * @{
 */

#ifndef CVCUDA__BND_BOX_HPP
#define CVCUDA__BND_BOX_HPP

#include "IOperator.hpp"
#include "OpBndBox.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class BndBox final : public IOperator
{
public:
    explicit BndBox();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVBndBoxesI bboxes) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline BndBox::BndBox()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaBndBoxCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void BndBox::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                               const NVCVBndBoxesI bboxes) const
{
    nvcv::detail::CheckThrow(cvcudaBndBoxSubmit(m_handle.get(), stream, in.handle(), out.handle(), bboxes));
}

inline NVCVOperatorHandle BndBox::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA__BND_BOX_HPP

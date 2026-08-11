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
 * @brief Defines the public C++ Class for CLAHE operation.
 * @defgroup NVCV_CPP_ALGORITHM__CLAHE CLAHE
 * @{
 */

#ifndef CVCUDA__CLAHE_HPP
#define CVCUDA__CLAHE_HPP

#include "IOperator.hpp"
#include "OpCLAHE.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <cassert>

namespace cvcuda {

class CLAHE final : public IOperator
{
public:
    explicit CLAHE(int32_t maxBatchSize, int32_t tilesX = 8, int32_t tilesY = 8);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float clipLimit = 40.f) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    float clipLimit = 40.f) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline CLAHE::CLAHE(int32_t maxBatchSize, int32_t tilesX, int32_t tilesY)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaCLAHECreate(&h, maxBatchSize, tilesX, tilesY));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void CLAHE::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                              float clipLimit) const
{
    nvcv::detail::CheckThrow(cvcudaCLAHESubmit(m_handle.get(), stream, in.handle(), out.handle(), clipLimit));
}

inline void CLAHE::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                              const nvcv::ImageBatchVarShape &out, float clipLimit) const
{
    nvcv::detail::CheckThrow(cvcudaCLAHEVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), clipLimit));
}

inline NVCVOperatorHandle CLAHE::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA__CLAHE_HPP

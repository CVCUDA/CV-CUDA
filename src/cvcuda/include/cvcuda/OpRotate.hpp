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
 * @file OpRotate.hpp
 *
 * @brief Defines the public C++ Class for the rotate operation.
 * @defgroup NVCV_CPP_ALGORITHM_ROTATE Rotate
 * @{
 */

#ifndef CVCUDA_ROTATE_HPP
#define CVCUDA_ROTATE_HPP

#include "IOperator.hpp"
#include "OpRotate.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Rotate final : public IOperator
{
public:
    explicit Rotate(const int maxVarShapeBatchSize);

    ~Rotate();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, const double angleDeg,
                    const double2 shift, const NVCVInterpolationType interpolation);

    void operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::IImageBatchVarShape &out,
                    nvcv::ITensor &angleDeg, nvcv::ITensor &shift, const NVCVInterpolationType interpolation);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Rotate::Rotate(const int maxVarShapeBatchSize)
{
    nvcv::detail::CheckThrow(cvcudaRotateCreate(&m_handle, maxVarShapeBatchSize));
    assert(m_handle);
}

inline Rotate::~Rotate()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Rotate::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, const double angleDeg,
                               const double2 shift, const NVCVInterpolationType interpolation)
{
    nvcv::detail::CheckThrow(
        cvcudaRotateSubmit(m_handle, stream, in.handle(), out.handle(), angleDeg, shift, interpolation));
}

inline void Rotate::operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::IImageBatchVarShape &out,
                               nvcv::ITensor &angleDeg, nvcv::ITensor &shift, const NVCVInterpolationType interpolation)
{
    nvcv::detail::CheckThrow(cvcudaRotateVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), angleDeg.handle(),
                                                        shift.handle(), interpolation));
}

inline NVCVOperatorHandle Rotate::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_ROTATE_HPP

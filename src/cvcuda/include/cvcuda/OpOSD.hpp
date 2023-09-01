/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpOSD.hpp
 *
 * @brief Defines the public C++ Class for the OSD operation.
 * @defgroup NVCV_CPP_ALGORITHM__O_S_D OSD
 * @{
 */

#ifndef CVCUDA__O_S_D_HPP
#define CVCUDA__O_S_D_HPP

#include "IOperator.hpp"
#include "OpOSD.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class OSD final : public IOperator
{
public:
    explicit OSD();

    ~OSD();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const NVCVElements elements);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline OSD::OSD()
{
    nvcv::detail::CheckThrow(cvcudaOSDCreate(&m_handle));
    assert(m_handle);
}

inline OSD::~OSD()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void OSD::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                            const NVCVElements elements)
{
    nvcv::detail::CheckThrow(cvcudaOSDSubmit(m_handle, stream, in.handle(), out.handle(), elements));
}

inline NVCVOperatorHandle OSD::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA__O_S_D_HPP

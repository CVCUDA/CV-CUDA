/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpLabel.hpp
 *
 * @brief Defines the public C++ Class for the Label operation.
 * @defgroup NVCV_CPP_ALGORITHM_LABEL Label
 * @{
 */

#ifndef CVCUDA__LABEL_HPP
#define CVCUDA__LABEL_HPP

#include "IOperator.hpp"
#include "OpLabel.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Label final : public IOperator
{
public:
    explicit Label();

    ~Label();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &bgLabel,
                    const nvcv::Tensor &minThresh, const nvcv::Tensor &maxThresh, const nvcv::Tensor &minSize,
                    const nvcv::Tensor &count, const nvcv::Tensor &stats, const nvcv::Tensor &mask,
                    NVCVConnectivityType connectivity, NVCVLabelType assignLabels, NVCVLabelMaskType maskType) const;

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Label::Label()
{
    nvcv::detail::CheckThrow(cvcudaLabelCreate(&m_handle));
    assert(m_handle);
}

inline Label::~Label()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Label::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                              const nvcv::Tensor &bgLabel, const nvcv::Tensor &minThresh, const nvcv::Tensor &maxThresh,
                              const nvcv::Tensor &minSize, const nvcv::Tensor &count, const nvcv::Tensor &stats,
                              const nvcv::Tensor &mask, NVCVConnectivityType connectivity, NVCVLabelType assignLabels,
                              NVCVLabelMaskType maskType) const
{
    nvcv::detail::CheckThrow(cvcudaLabelSubmit(m_handle, stream, in.handle(), out.handle(), bgLabel.handle(),
                                               minThresh.handle(), maxThresh.handle(), minSize.handle(), count.handle(),
                                               stats.handle(), mask.handle(), connectivity, assignLabels, maskType));
}

inline NVCVOperatorHandle Label::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_LABEL_HPP

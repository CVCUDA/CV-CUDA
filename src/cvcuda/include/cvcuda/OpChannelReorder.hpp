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
 * @file OpChannelReorder.hpp
 *
 * @brief Defines the public C++ Class for the channel reorder operation.
 * @defgroup NVCV_CPP_ALGORITHM_CHANNEL_REORDER Channel Reorder
 * @{
 */

#ifndef CVCUDA_CHANNEL_REORDER_HPP
#define CVCUDA_CHANNEL_REORDER_HPP

#include "IOperator.hpp"
#include "OpChannelReorder.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class ChannelReorder final : public IOperator
{
public:
    explicit ChannelReorder();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const int32_t *order,
                    int32_t orderLength) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &orders) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline ChannelReorder::ChannelReorder()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaChannelReorderCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void ChannelReorder::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                       const int32_t *order, int32_t orderLength) const
{
    nvcv::detail::CheckThrow(
        cvcudaChannelReorderSubmit(m_handle.get(), stream, in.handle(), out.handle(), order, orderLength));
}

inline void ChannelReorder::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                       const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &orders) const
{
    nvcv::detail::CheckThrow(
        cvcudaChannelReorderVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), orders.handle()));
}

inline NVCVOperatorHandle ChannelReorder::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_CHANNEL_REORDER_HPP

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file OpMerge.hpp
 * @brief Public C++ interface for bit-exact channel merging.
 * @defgroup NVCV_CPP_ALGORITHM_MERGE Merge
 * @{
 */

#ifndef CVCUDA_MERGE_HPP
#define CVCUDA_MERGE_HPP

#include "IOperator.hpp"
#include "OpMerge.h"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>

#include <array>

namespace cvcuda {

/** @brief Asynchronous, bit-exact channel merging. See cvcudaMergeSubmit and cvcudaMergeVarShapeSubmit. */
class Merge final : public IOperator
{
public:
    explicit Merge()
    {
        NVCVOperatorHandle handle = nullptr;
        nvcv::detail::CheckThrow(cvcudaMergeCreate(&handle));
        m_handle = detail::OperatorHandle{handle};
    }

    void operator()(cudaStream_t stream, const nvcv::TensorBatch &in, const nvcv::Tensor &out) const
    {
        nvcv::detail::CheckThrow(cvcudaMergeSubmit(m_handle.get(), stream, in.handle(), out.handle()));
    }

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape *in, int32_t numInputs,
                    const nvcv::ImageBatchVarShape &out) const
    {
        if (!in || numInputs < 1 || numInputs > 4)
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Expected one to four band batches");
        std::array<NVCVImageBatchHandle, 4> inputs{};
        for (int32_t channel = 0; channel < numInputs; ++channel) inputs[channel] = in[channel].handle();
        nvcv::detail::CheckThrow(
            cvcudaMergeVarShapeSubmit(m_handle.get(), stream, inputs.data(), numInputs, out.handle()));
    }

    NVCVOperatorHandle handle() const noexcept override
    {
        return m_handle.get();
    }

private:
    detail::OperatorHandle m_handle;
};

} // namespace cvcuda

/** @} */
#endif

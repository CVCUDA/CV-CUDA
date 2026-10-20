/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file OpSplit.hpp
 * @brief Public C++ interface for bit-exact channel splitting.
 * @defgroup NVCV_CPP_ALGORITHM_SPLIT Split
 * @{
 */

#ifndef CVCUDA_SPLIT_HPP
#define CVCUDA_SPLIT_HPP

#include "IOperator.hpp"
#include "OpSplit.h"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>

#include <array>

namespace cvcuda {

/** @brief Asynchronous, bit-exact channel splitting. See cvcudaSplitSubmit and cvcudaSplitVarShapeSubmit. */
class Split final : public IOperator
{
public:
    explicit Split()
    {
        NVCVOperatorHandle handle = nullptr;
        nvcv::detail::CheckThrow(cvcudaSplitCreate(&handle));
        m_handle = detail::OperatorHandle{handle};
    }

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::TensorBatch &out) const
    {
        nvcv::detail::CheckThrow(cvcudaSplitSubmit(m_handle.get(), stream, in.handle(), out.handle()));
    }

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape *out,
                    int32_t numOutputs) const
    {
        if (!out || numOutputs < 1 || numOutputs > 4)
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Expected one to four band batches");
        std::array<NVCVImageBatchHandle, 4> outputs{};
        for (int32_t channel = 0; channel < numOutputs; ++channel) outputs[channel] = out[channel].handle();
        nvcv::detail::CheckThrow(
            cvcudaSplitVarShapeSubmit(m_handle.get(), stream, in.handle(), outputs.data(), numOutputs));
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_PRIV_MERGE_HPP
#define CVCUDA_PRIV_MERGE_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>

namespace cvcuda::priv {

class Merge final : public IOperator
{
public:
    void operator()(cudaStream_t stream, const nvcv::TensorBatch &in, const nvcv::Tensor &out) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape *in, int32_t numInputs,
                    const nvcv::ImageBatchVarShape &out) const;
};

} // namespace cvcuda::priv

#endif

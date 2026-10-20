/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_PRIV_SPLIT_HPP
#define CVCUDA_PRIV_SPLIT_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>

namespace cvcuda::priv {

class Split final : public IOperator
{
public:
    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::TensorBatch &out) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape *out,
                    int32_t numOutputs) const;
};

} // namespace cvcuda::priv

#endif

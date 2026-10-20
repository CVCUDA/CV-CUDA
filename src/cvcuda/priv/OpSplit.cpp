/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OpSplit.hpp"

#include "ChannelCopy.hpp"
#include "Nvtx.hpp"

namespace cvcuda::priv {

void Split::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::TensorBatch &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Split::operator()");
    CopyChannels(stream, in, out, true);
}

void Split::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape *out,
                       int32_t numOutputs) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Split::operator()");
    CopyChannels(stream, in, out, numOutputs, true);
}

} // namespace cvcuda::priv

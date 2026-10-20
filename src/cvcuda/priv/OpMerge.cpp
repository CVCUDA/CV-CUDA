/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OpMerge.hpp"

#include "ChannelCopy.hpp"
#include "Nvtx.hpp"

namespace cvcuda::priv {

void Merge::operator()(cudaStream_t stream, const nvcv::TensorBatch &in, const nvcv::Tensor &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Merge::operator()");
    CopyChannels(stream, out, in, false);
}

void Merge::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape *in, int32_t numInputs,
                       const nvcv::ImageBatchVarShape &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Merge::operator()");
    CopyChannels(stream, out, in, numInputs, false);
}

} // namespace cvcuda::priv

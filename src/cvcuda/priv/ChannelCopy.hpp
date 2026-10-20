/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_PRIV_CHANNEL_COPY_HPP
#define CVCUDA_PRIV_CHANNEL_COPY_HPP

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>

namespace cvcuda::priv {

void CopyChannels(cudaStream_t stream, const nvcv::Tensor &image, const nvcv::TensorBatch &bands, bool split);

void CopyChannels(cudaStream_t stream, const nvcv::ImageBatchVarShape &image, const nvcv::ImageBatchVarShape *bands,
                  int32_t count, bool split);

} // namespace cvcuda::priv

#endif

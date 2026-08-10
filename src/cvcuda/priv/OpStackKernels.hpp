/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_PRIV__STACK_KERNELS_HPP
#define CVCUDA_PRIV__STACK_KERNELS_HPP

#include <cuda_runtime.h>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/TensorBatchData.hpp>
#include <nvcv/TensorData.hpp>

namespace cvcuda::priv {

bool RunStackTensorBatchKernel(cudaStream_t stream, const nvcv::TensorBatchDataStridedCuda &inData,
                               const nvcv::TensorDataStridedCuda &outData);

bool RunStackVarShapeKernel(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                            const nvcv::TensorDataStridedCuda &outData);

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__STACK_KERNELS_HPP

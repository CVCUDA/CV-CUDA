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

#include <cvcuda/cuda_tools/TensorBatchWrap.hpp>
#include <nvcv/TensorBatchData.hpp>

namespace cuda = nvcv::cuda;

template<typename SetValueMethod, typename TensorBatchWrapT>
void SetReference(TensorBatchWrapT wrap, cudaStream_t stream);

template<typename TensorBatchWrapT, typename T>
struct SetThroughTensor
{
    static __device__ void Set(TensorBatchWrapT wrap, int sample, int *coords, T value);
};

template<typename TensorBatchWrapT, typename T>
struct SetThroughSubscript
{
    static __device__ void Set(TensorBatchWrapT wrap, int sample, int *coords, T value);
};

template<typename TensorBatchWrapT, typename T>
struct SetThroughPtr
{
    static __device__ void Set(TensorBatchWrapT wrap, int sample, int *coords, T value);
};

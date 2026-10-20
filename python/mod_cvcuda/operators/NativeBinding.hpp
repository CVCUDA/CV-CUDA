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

#ifndef CVCUDAPY_OPERATORS_NATIVEBINDING_HPP
#define CVCUDAPY_OPERATORS_NATIVEBINDING_HPP

#include "../nvcv/Image.hpp"
#include "../nvcv/ImageBatch.hpp"
#include "../nvcv/Stream.hpp"
#include "../nvcv/Tensor.hpp"
#include "../nvcv/TensorBatch.hpp"

#include <nvcv/Tensor.hpp>

#include <memory>
#include <optional>

namespace cvcudapy::native {

using Image              = nvcvpy::priv::Image;
using ImageBatchVarShape = nvcvpy::priv::ImageBatchVarShape;
using Stream             = nvcvpy::priv::Stream;
using Tensor             = nvcvpy::priv::Tensor;
using TensorBatch        = nvcvpy::priv::TensorBatch;

inline Stream &GetStream(std::optional<Stream *> stream)
{
    return stream && *stream ? **stream : Stream::Current();
}

inline std::shared_ptr<Tensor> CreateTensor(const nvcv::TensorShape &shape, nvcv::DataType dtype, int rowAlign = 0)
{
    nvcv::MemAlignment alignment = rowAlign == 0 ? nvcv::MemAlignment{} : nvcv::MemAlignment{}.rowAddr(rowAlign);
    return Tensor::CreateFromReqs(nvcv::Tensor::CalcRequirements(shape, dtype, alignment));
}

inline std::shared_ptr<Tensor> CreateTensorForImageBatch(int numImages, nvcv::Size2D size, nvcv::ImageFormat format,
                                                         int rowAlign = 0)
{
    return Tensor::CreateForImageBatch(numImages, {size.w, size.h}, format, rowAlign);
}

} // namespace cvcudapy::native

#endif // CVCUDAPY_OPERATORS_NATIVEBINDING_HPP

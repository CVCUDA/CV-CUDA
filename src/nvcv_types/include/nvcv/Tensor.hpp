/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TENSOR_HPP
#define NVCV_TENSOR_HPP

#include "IImage.hpp"
#include "ITensor.hpp"
#include "Size.hpp"
#include "TensorData.hpp"
#include "alloc/IAllocator.hpp"

#include <nvcv/ImageFormat.hpp>

namespace nvcv {

// Tensor tensor definition -------------------------------------
class Tensor : public ITensor
{
public:
    using Requirements = NVCVTensorRequirements;
    static Requirements CalcRequirements(const TensorShape &shape, DataType dtype, const MemAlignment &bufAlign = {});
    static Requirements CalcRequirements(int numImages, Size2D imgSize, ImageFormat fmt,
                                         const MemAlignment &bufAlign = {});

    explicit Tensor(const Requirements &reqs, IAllocator *alloc = nullptr);
    explicit Tensor(const TensorShape &shape, DataType dtype, const MemAlignment &bufAlign = {},
                    IAllocator *alloc = nullptr);
    explicit Tensor(int numImages, Size2D imgSize, ImageFormat fmt, const MemAlignment &bufAlign = {},
                    IAllocator *alloc = nullptr);
    ~Tensor();

    Tensor(const Tensor &) = delete;

private:
    NVCVTensorHandle doGetHandle() const final override;

    NVCVTensorHandle m_handle;
};

// TensorWrapData definition -------------------------------------
using TensorDataCleanupFunc = void(const ITensorData &);

class TensorWrapData : public ITensor
{
public:
    explicit TensorWrapData(const ITensorData &data, std::function<TensorDataCleanupFunc> cleanup = nullptr);
    ~TensorWrapData();

    TensorWrapData(const TensorWrapData &) = delete;

private:
    NVCVTensorHandle doGetHandle() const final override;

    static void doCleanup(void *ctx, const NVCVTensorData *data);

    NVCVTensorHandle m_handle;

    std::function<TensorDataCleanupFunc> m_cleanup;
};

// TensorWrapImage definition -------------------------------------
class TensorWrapImage : public ITensor
{
public:
    explicit TensorWrapImage(const IImage &mg);
    ~TensorWrapImage();

    TensorWrapImage(const TensorWrapImage &) = delete;

private:
    NVCVTensorHandle doGetHandle() const final override;

    NVCVTensorHandle m_handle;
};

// For API backward-compatibility
using TensorWrapHandle = detail::WrapHandle<ITensor>;

} // namespace nvcv

#include "detail/TensorImpl.hpp"

#endif // NVCV_TENSOR_HPP

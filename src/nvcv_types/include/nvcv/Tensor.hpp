/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CoreResource.hpp"
#include "Image.hpp"
#include "ImageFormat.hpp"
#include "Size.hpp"
#include "Tensor.h"
#include "TensorData.hpp"
#include "alloc/Allocator.hpp"
#include "detail/Callback.hpp"

namespace nvcv {

NVCV_IMPL_SHARED_HANDLE(Tensor);

// Tensor tensor definition -------------------------------------
class Tensor : public CoreResource<NVCVTensorHandle, Tensor>
{
public:
    using HandleType   = NVCVTensorHandle;
    using Base         = CoreResource<NVCVTensorHandle, Tensor>;
    using Requirements = NVCVTensorRequirements;

    int          rank() const;
    TensorShape  shape() const;
    DataType     dtype() const;
    TensorLayout layout() const;

    TensorData exportData() const;

    template<typename DerivedTensorData>
    Optional<DerivedTensorData> exportData() const
    {
        return exportData().cast<DerivedTensorData>();
    }

    void  setUserPointer(void *ptr);
    void *userPointer() const;

    static Requirements CalcRequirements(const TensorShape &shape, DataType dtype, const MemAlignment &bufAlign = {});
    static Requirements CalcRequirements(int numImages, Size2D imgSize, ImageFormat fmt,
                                         const MemAlignment &bufAlign = {});

    NVCV_IMPLEMENT_SHARED_RESOURCE(Tensor, Base);

    explicit Tensor(const Requirements &reqs, const Allocator &alloc = nullptr);
    explicit Tensor(const TensorShape &shape, DataType dtype, const MemAlignment &bufAlign = {},
                    const Allocator &alloc = nullptr);
    explicit Tensor(int numImages, Size2D imgSize, ImageFormat fmt, const MemAlignment &bufAlign = {},
                    const Allocator &alloc = nullptr);
};

// TensorWrapData definition -------------------------------------
using TensorDataCleanupFunc = void(const TensorData &);

struct TranslateTensorDataCleanup
{
    template<typename CppCleanup>
    void operator()(CppCleanup &&c, const NVCVTensorData *data) const noexcept
    {
        c(TensorData(*data));
    }
};

using TensorDataCleanupCallback
    = CleanupCallback<TensorDataCleanupFunc, detail::RemovePointer_t<NVCVTensorDataCleanupFunc>,
                      TranslateTensorDataCleanup>;

// TensorWrapImage definition -------------------------------------

inline Tensor TensorWrapData(const TensorData &data, TensorDataCleanupCallback &&cleanup = {});

inline Tensor TensorWrapImage(const Image &img);

using TensorWrapHandle = NonOwningResource<Tensor>;

} // namespace nvcv

#include "detail/TensorImpl.hpp"

#endif // NVCV_TENSOR_HPP

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "Optional.hpp"
#include "Size.hpp"
#include "Tensor.h"
#include "TensorData.hpp"
#include "alloc/Allocator.hpp"
#include "detail/Callback.hpp"

namespace nvcv {

NVCV_IMPL_SHARED_HANDLE(Tensor);

/**
 * @brief Represents a tensor as a core resource in the system.
 *
 * The Tensor class is built upon the CoreResource utility class, which handles the resource management of the tensor.
 * This class provides various interfaces to access and manage tensor properties such as rank, shape, data type, and layout.
 */
class Tensor : public CoreResource<NVCVTensorHandle, Tensor>
{
public:
    using HandleType   = NVCVTensorHandle;
    using Base         = CoreResource<NVCVTensorHandle, Tensor>;
    using Requirements = NVCVTensorRequirements;

    /**
     * @brief Retrieves the rank (number of dimensions) of the tensor.
     *
     * @return Rank of the tensor.
     */
    int rank() const;

    /**
     * @brief Retrieves the shape of the tensor.
     *
     * @return Shape of the tensor.
     */
    TensorShape shape() const;

    /**
     * @brief Retrieves the data type of the tensor elements.
     *
     * @return Data type of the tensor.
     */
    DataType dtype() const;

    /**
     * @brief Retrieves the layout of the tensor.
     *
     * @return Layout of the tensor.
     */
    TensorLayout layout() const;

    /**
     * @brief Exports the data of the tensor.
     *
     * @return TensorData object representing the tensor's data.
     */
    TensorData exportData() const;

    /**
     * @brief Exports the tensor data and casts it to a specified derived data type.
     *
     * @tparam DerivedTensorData The derived tensor data type to cast to.
     * @return An optional object of the derived tensor data type.
     */
    template<typename DerivedTensorData>
    Optional<DerivedTensorData> exportData() const
    {
        return exportData().cast<DerivedTensorData>();
    }

    /**
     * @brief Sets a user-defined pointer associated with the tensor.
     *
     * @param ptr Pointer to set.
     */
    void setUserPointer(void *ptr);

    /**
     * @brief Retrieves the user-defined pointer associated with the tensor.
     *
     * @return User-defined pointer.
     */
    void *userPointer() const;

    /**
     * @brief Creates a view of the tensor with a new shape and layout
     *
     */
    Tensor reshape(const TensorShape &new_shape);

    /**
     * @brief Calculates the requirements for a tensor given its shape and data type.
     *
     * @param shape Shape of the tensor.
     * @param dtype Data type of the tensor elements.
     * @param bufAlign Memory alignment for the tensor.
     * @return Requirements object representing the tensor's requirements.
     */
    static Requirements CalcRequirements(const TensorShape &shape, DataType dtype, const MemAlignment &bufAlign = {});

    /**
     * @brief Calculates the requirements for a tensor representing a set of images.
     *
     * @param numImages Number of images.
     * @param imgSize Dimensions of the images.
     * @param fmt Format of the images.
     * @param bufAlign Memory alignment for the tensor.
     * @return Requirements object representing the tensor's requirements.
     */
    static Requirements CalcRequirements(int numImages, Size2D imgSize, ImageFormat fmt,
                                         const MemAlignment &bufAlign = {});

    NVCV_IMPLEMENT_SHARED_RESOURCE(Tensor, Base);

    /**
     * @brief Constructors
     */
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
/**
 * @brief Wraps tensor data into a tensor object.
 *
 * @param data Tensor data to be wrapped.
 * @param cleanup Cleanup callback to manage the tensor data's lifecycle.
 * @return A tensor object wrapping the given data.
 */
inline Tensor TensorWrapData(const TensorData &data, TensorDataCleanupCallback &&cleanup = {});

/**
 * @brief Wraps an image into a tensor object.
 *
 * @param img Image to be wrapped.
 * @return A tensor object wrapping the given image.
 */
inline Tensor TensorWrapImage(const Image &img);

using TensorWrapHandle = NonOwningResource<Tensor>;

// Tensor const ref optional definition ---------------------------

using OptionalTensorConstRef = nvcv::Optional<std::reference_wrapper<const nvcv::Tensor>>;

#define NVCV_TENSOR_HANDLE_TO_OPTIONAL(X) X ? nvcv::OptionalTensorConstRef(nvcv::TensorWrapHandle{X}) : nvcv::NullOpt
#define NVCV_OPTIONAL_TO_HANDLE(X)        X ? X->get().handle() : nullptr

} // namespace nvcv

#include "detail/TensorImpl.hpp"

#endif // NVCV_TENSOR_HPP

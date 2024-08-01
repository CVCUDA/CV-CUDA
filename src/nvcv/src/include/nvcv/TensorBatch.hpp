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

#ifndef NVCV_TENSORBATCH_HPP
#define NVCV_TENSORBATCH_HPP

#include "CoreResource.hpp"
#include "TensorBatch.h"
#include "TensorBatchData.hpp"
#include "alloc/Allocator.hpp"

#include <nvcv/Tensor.hpp>

#include <vector>

namespace nvcv {

NVCV_IMPL_SHARED_HANDLE(TensorBatch);

/**
 * @brief Handle to a tensor batch object.
 *
 * Tensor batch is a container type that can hold a list of non-uniformly shaped tensors.
 * Rank, data type and layout must be consistent between the tensors.
 */
class TensorBatch : public CoreResource<NVCVTensorBatchHandle, TensorBatch>
{
public:
    using Base         = CoreResource<NVCVTensorBatchHandle, TensorBatch>;
    using Requirements = NVCVTensorBatchRequirements;
    using HandleType   = NVCVTensorBatchHandle;

    static Requirements CalcRequirements(int32_t capacity);

    NVCV_IMPLEMENT_SHARED_RESOURCE(TensorBatch, Base);

    TensorBatch(const Requirements &reqs, const Allocator &alloc = nullptr);

    TensorBatch(int32_t capacity, const Allocator &alloc = nullptr);

    /**
     * @brief Return the maximal number of tensors the tensor batch can hold.
     */
    int32_t capacity() const;

    /**
     * @brief Return the rank of the tensors in the tensor batch or -1 for an empty batch.
     */
    int32_t rank() const;

    /**
     * @brief Return the number of tensors in the tensor batch.
     */
    int32_t numTensors() const;

    /**
     * @brief Return the data type of the tensors in the tensor batch.
     */
    DataType dtype() const;

    /**
     * @brief Return the layout of the tensors in the tensor batch.
     */
    TensorLayout layout() const;

    /**
     * @brief Return the buffer type of the tensors' data.
     */
    NVCVTensorBufferType type() const;

    /**
     * @brief Return the allocator used by the tensor batch.
     */
    Allocator alloc() const;

    /**
     * @brief Append tensors from the given range to the end of the batch.
     *
     * @param begin,end range of the tensors to append.
     */
    template<typename It>
    void pushBack(It begin, It end);

    /**
     * @brief Append the \a tensor to the end of the batch.
     *
     * @param tensor Appended tensor.
     */
    void pushBack(const Tensor &tensor);

    /**
     * @brief Truncate tensors from the end of the batch.
     *
     * @param numTensors Number of tensors to remove.
     */
    void popTensors(int32_t numTensors);

    /**
     * @brief Delete the last tensor from the batch.
     */
    void popTensor();

    /**
     * @brief Generate the tensor batch data descriptor.
     *
     * The necessary copies to GPU are scheduled on the given stream.
     * The struct is valid after the scheduled work is finished.
     *
     * @param stream CUDA stream on which the buffers copy will be scheduled.
     */
    TensorBatchData exportData(CUstream stream);

    void clear();

    /**
     * @brief Associates a user pointer to the tensor batch.
     *
     * @param ptr User pointer
     */
    void setUserPointer(void *ptr);

    /**
     * @brief Get the user pointer that was previously assciated to the tensor batch
     * with the setUserPointer(void*) method. Returns nullptr if no pointer was set.
     */
    void *getUserPointer() const;

    /**
     * @brief Return a handle to a tensor at a given positon.
     *
     * @param idx Index of a tensor to return
     */
    Tensor operator[](int32_t idx) const;

    /**
     * @brief Replace the tensor on position \a index.
     */
    void setTensor(int32_t index, const Tensor &tensor);

    class Iterator;

    Iterator begin() const;

    Iterator end() const;
};

class TensorBatch::Iterator
{
public:
    using value_type        = Tensor;
    using reference         = const Tensor &;
    using pointer           = const Tensor *;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = int32_t;

    reference operator*() const;
    pointer   operator->() const;

    Iterator  operator++(int);
    Iterator &operator++();
    Iterator  operator--(int);
    Iterator &operator--();

    Iterator operator+(difference_type diff) const;
    Iterator operator-(difference_type diff) const;

    difference_type operator-(const Iterator &rhs) const;

    bool operator==(const Iterator &rhs) const;
    bool operator!=(const Iterator &rhs) const;
    bool operator<(const Iterator &rhs) const;
    bool operator>(const Iterator &rhs) const;
    bool operator<=(const Iterator &rhs) const;
    bool operator>=(const Iterator &rhs) const;

    Iterator(Iterator &other)
        : Iterator()
    {
        *this = other;
    }

    Iterator(Iterator &&other)
        : Iterator()
    {
        *this = std::move(other);
    }

    Iterator &operator=(Iterator &other)
    {
        m_tensorBatch   = other.m_tensorBatch;
        m_idx           = other.m_idx;
        m_currentTensor = other.m_currentTensor;
        return *this;
    }

    Iterator &operator=(Iterator &&other)
    {
        m_tensorBatch   = other.m_tensorBatch;
        m_idx           = other.m_idx;
        m_currentTensor = std::move(other.m_currentTensor);
        return *this;
    }

private:
    friend class TensorBatch;

    Iterator() = default;

    Iterator(const TensorBatch *tensorBatch, int32_t idx)
        : m_tensorBatch(tensorBatch)
        , m_idx(idx)
        , m_currentTensor{}
    {
        UpdateCurrentTensor();
    }

    void UpdateCurrentTensor();

    const TensorBatch *m_tensorBatch   = nullptr;
    int32_t            m_idx           = 0;
    mutable Tensor     m_currentTensor = {};
};

using TensorBatchWrapHandle = NonOwningResource<TensorBatch>;

} // namespace nvcv

#include "detail/TensorBatchImpl.hpp"

#endif // NVCV_TENSORBATCH_HPP

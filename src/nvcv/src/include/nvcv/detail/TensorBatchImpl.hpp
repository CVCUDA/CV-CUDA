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

#ifndef NVCV_TENSORBATCH_IMPL_HPP
#define NVCV_TENSORBATCH_IMPL_HPP

namespace nvcv {

// TensorBatch

inline TensorBatch::Requirements TensorBatch::CalcRequirements(int32_t capacity)
{
    TensorBatch::Requirements reqs = {};
    detail::CheckThrow(nvcvTensorBatchCalcRequirements(capacity, &reqs));
    return reqs;
}

inline TensorBatch::TensorBatch(const TensorBatch::Requirements &reqs, const Allocator &alloc)
{
    NVCVTensorBatchHandle handle = nullptr;
    detail::CheckThrow(nvcvTensorBatchConstruct(&reqs, alloc.handle(), &handle));
    reset(std::move(handle));
}

inline TensorBatch::TensorBatch(int32_t capacity, const Allocator &alloc)
{
    auto                  reqs   = TensorBatch::CalcRequirements(capacity);
    NVCVTensorBatchHandle handle = nullptr;
    detail::CheckThrow(nvcvTensorBatchConstruct(&reqs, alloc.handle(), &handle));
    reset(std::move(handle));
}

inline int32_t TensorBatch::capacity() const
{
    int32_t output;
    detail::CheckThrow(nvcvTensorBatchGetCapacity(handle(), &output));
    return output;
}

inline int32_t TensorBatch::rank() const
{
    int32_t output;
    detail::CheckThrow(nvcvTensorBatchGetRank(handle(), &output));
    return output;
}

inline int32_t TensorBatch::numTensors() const
{
    int32_t output;
    detail::CheckThrow(nvcvTensorBatchGetNumTensors(handle(), &output));
    return output;
}

inline DataType TensorBatch::dtype() const
{
    NVCVDataType dataType = {};
    detail::CheckThrow(nvcvTensorBatchGetDType(handle(), &dataType));
    return DataType(dataType);
}

inline TensorLayout TensorBatch::layout() const
{
    NVCVTensorLayout tensorLayout;
    detail::CheckThrow(nvcvTensorBatchGetLayout(handle(), &tensorLayout));
    return TensorLayout(tensorLayout);
}

inline NVCVTensorBufferType TensorBatch::type() const
{
    NVCVTensorBufferType bufferType;
    detail::CheckThrow(nvcvTensorBatchGetType(handle(), &bufferType));
    return bufferType;
}

inline Allocator TensorBatch::alloc() const
{
    NVCVAllocatorHandle halloc;
    detail::CheckThrow(nvcvTensorBatchGetAllocator(handle(), &halloc));
    return Allocator(std::move(halloc));
}

template<typename It>
inline void TensorBatch::pushBack(It begin, It end)
{
    std::vector<NVCVTensorHandle> handles;
    handles.reserve(capacity() - numTensors());
    for (auto it = begin; it != end; ++it)
    {
        handles.push_back(it->handle());
    }
    detail::CheckThrow(nvcvTensorBatchPushTensors(handle(), handles.data(), handles.size()));
}

inline void TensorBatch::pushBack(const Tensor &tensor)
{
    auto hTensor = tensor.handle();
    detail::CheckThrow(nvcvTensorBatchPushTensors(handle(), &hTensor, 1));
}

inline void TensorBatch::popTensors(int32_t numTensors)
{
    detail::CheckThrow(nvcvTensorBatchPopTensors(handle(), numTensors));
}

inline void TensorBatch::popTensor()
{
    detail::CheckThrow(nvcvTensorBatchPopTensors(handle(), 1));
}

inline TensorBatchData TensorBatch::exportData(CUstream stream)
{
    NVCVTensorBatchData output = {};
    detail::CheckThrow(nvcvTensorBatchExportData(handle(), stream, &output));
    return TensorBatchData(output);
}

inline void TensorBatch::clear()
{
    detail::CheckThrow(nvcvTensorBatchClear(handle()));
}

inline void TensorBatch::setUserPointer(void *ptr)
{
    detail::CheckThrow(nvcvTensorBatchSetUserPointer(handle(), ptr));
}

inline void *TensorBatch::getUserPointer() const
{
    void *outPtr = nullptr;
    detail::CheckThrow(nvcvTensorBatchGetUserPointer(handle(), &outPtr));
    return outPtr;
}

inline Tensor TensorBatch::operator[](int32_t idx) const
{
    NVCVTensorHandle hTensor = nullptr;
    detail::CheckThrow(nvcvTensorBatchGetTensors(handle(), idx, &hTensor, 1));
    return Tensor(std::move(hTensor));
}

inline void TensorBatch::setTensor(int32_t idx, const Tensor &tensor)
{
    auto hTensor = tensor.handle();
    detail::CheckThrow(nvcvTensorBatchSetTensors(handle(), idx, &hTensor, 1));
}

inline TensorBatch::Iterator TensorBatch::begin() const
{
    return Iterator(this, 0);
}

inline TensorBatch::Iterator TensorBatch::end() const
{
    return Iterator(this, numTensors());
}

// TensorBatch::Iterator

inline TensorBatch::Iterator::reference TensorBatch::Iterator::operator*() const
{
    return m_currentTensor;
}

inline TensorBatch::Iterator::pointer TensorBatch::Iterator::operator->() const
{
    return &m_currentTensor;
}

inline TensorBatch::Iterator TensorBatch::Iterator::operator++(int)
{
    Iterator output(*this);
    ++(*this);
    return output;
}

inline TensorBatch::Iterator &TensorBatch::Iterator::operator++()
{
    ++m_idx;
    UpdateCurrentTensor();
    return *this;
}

inline TensorBatch::Iterator TensorBatch::Iterator::operator--(int)
{
    Iterator output(*this);
    --(*this);
    return output;
}

inline TensorBatch::Iterator &TensorBatch::Iterator::operator--()
{
    --m_idx;
    UpdateCurrentTensor();
    return *this;
}

inline TensorBatch::Iterator TensorBatch::Iterator::operator+(difference_type diff) const
{
    return Iterator(m_tensorBatch, m_idx + diff);
}

inline TensorBatch::Iterator TensorBatch::Iterator::operator-(difference_type diff) const
{
    return Iterator(m_tensorBatch, m_idx - diff);
}

inline void TensorBatch::Iterator::UpdateCurrentTensor()
{
    if (m_idx < m_tensorBatch->numTensors() && m_idx >= 0)
    {
        m_currentTensor = (*m_tensorBatch)[m_idx];
    }
}

inline TensorBatch::Iterator::difference_type TensorBatch::Iterator::operator-(const Iterator &rhs) const
{
    return m_idx - rhs.m_idx;
}

inline bool TensorBatch::Iterator::operator==(const Iterator &rhs) const
{
    return m_tensorBatch == rhs.m_tensorBatch && m_idx == rhs.m_idx;
}

inline bool TensorBatch::Iterator::operator!=(const Iterator &rhs) const
{
    return !(rhs == *this);
}

inline bool TensorBatch::Iterator::operator<(const Iterator &rhs) const
{
    return std::make_pair(m_tensorBatch, m_idx) < std::make_pair(rhs.m_tensorBatch, rhs.m_idx);
}

inline bool TensorBatch::Iterator::operator>(const Iterator &rhs) const
{
    return std::make_pair(m_tensorBatch, m_idx) > std::make_pair(rhs.m_tensorBatch, rhs.m_idx);
}

inline bool TensorBatch::Iterator::operator<=(const Iterator &rhs) const
{
    return !(rhs < *this);
}

inline bool TensorBatch::Iterator::operator>=(const Iterator &rhs) const
{
    return !(rhs > *this);
}

} // namespace nvcv

#endif // NVCV_TENSORBATCH_IMPL_HPP

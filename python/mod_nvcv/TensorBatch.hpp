/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_PYTHON_PRIV_TENSORBATCH_HPP
#define NVCV_PYTHON_PRIV_TENSORBATCH_HPP

#include "Container.hpp"

#include <nvcv/TensorBatch.hpp>

#include <vector>

namespace nvcvpy::priv {
namespace py = pybind11;

class Tensor;

class TensorBatch : public Container
{
    using TensorList = std::vector<std::shared_ptr<Tensor>>;

public:
    static void Export(py::module &m);

    static std::shared_ptr<TensorBatch> Create(int capacity);
    static std::shared_ptr<TensorBatch> WrapExternalBufferVector(std::vector<py::object>           buffers,
                                                                 std::optional<nvcv::TensorLayout> layout);

    std::shared_ptr<TensorBatch>       shared_from_this();
    std::shared_ptr<const TensorBatch> shared_from_this() const;

    const nvcv::TensorBatch &impl() const;
    nvcv::TensorBatch       &impl();

    int32_t numTensors() const;
    int32_t capacity() const;

    int32_t                           rank() const;
    std::optional<nvcv::DataType>     dtype() const;
    std::optional<nvcv::TensorLayout> layout() const;

    void pushBack(Tensor &tensor);
    void pushBackMany(std::vector<std::shared_ptr<Tensor>> &tensorList);
    void popBack(int tensorCount);
    void clear();

    std::shared_ptr<Tensor> at(int64_t idx) const;
    void                    set_at(int64_t idx, std::shared_ptr<Tensor> tensor);

    TensorList::const_iterator begin() const;
    TensorList::const_iterator end() const;

    class Key final : public IKey
    {
    public:
        Key(int capacity)
            : m_capacity(capacity)
        {
        }

    private:
        int m_capacity;

        virtual size_t doGetHash() const override;
        virtual bool   doIsCompatible(const IKey &that) const override;
    };

    virtual const Key &key() const override
    {
        return m_key;
    }

private:
    TensorBatch(int capacity);
    Key               m_key;
    nvcv::TensorBatch m_impl;
    TensorList        m_list;
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_TENSORBATCH_HPP

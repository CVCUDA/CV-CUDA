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

#ifndef NVCV_PYTHON_PRIV_ARRAY_HPP
#define NVCV_PYTHON_PRIV_ARRAY_HPP

#include "Container.hpp"
#include "Size.hpp"

#include <nvcv/Array.hpp>
#include <nvcv/Shape.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/python/Shape.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

namespace nvcvpy::priv {
namespace py = pybind11;

class ExternalBuffer;

class Array : public Container
{
public:
    static void Export(py::module &m);

    static std::shared_ptr<Array> Create(int64_t length, nvcv::DataType dtype);
    static std::shared_ptr<Array> Create(const Shape &shape, nvcv::DataType dtype);

    static std::shared_ptr<Array> CreateFromReqs(const nvcv::Array::Requirements &reqs);

    static std::shared_ptr<Array> Wrap(ExternalBuffer &buffer);
    static std::shared_ptr<Array> ResizeArray(Array &array, Shape shape);
    static std::shared_ptr<Array> ResizeArray(Array &array, int64_t length);

    std::shared_ptr<Array> Resize(Shape shape);
    std::shared_ptr<Array> Resize(int64_t length);

    std::shared_ptr<Array>       shared_from_this();
    std::shared_ptr<const Array> shared_from_this() const;

    Shape          shape() const;
    nvcv::DataType dtype() const;
    int            rank() const;
    int64_t        length() const;

    nvcv::Array       &impl();
    const nvcv::Array &impl() const;

    class Key final : public IKey
    {
    public:
        explicit Key()
            : m_wrapper(true)
        {
        }

        explicit Key(const nvcv::Array::Requirements &reqs);
        explicit Key(int64_t length, nvcv::DataType dtype);

    private:
        int64_t        m_length;
        nvcv::DataType m_dtype;
        bool           m_wrapper;

        virtual size_t doGetHash() const override;
        virtual bool   doIsCompatible(const IKey &that) const override;
    };

    virtual const Key &key() const override;

    int64_t GetSizeInBytes() const override;

    py::object cuda() const;

private:
    Array(const nvcv::Array::Requirements &reqs);
    Array(const nvcv::ArrayData &data, py::object wrappedObject);
    Array(nvcv::Array &&array);

    int64_t doComputeSizeInBytes(const nvcv::Array::Requirements &reqs);

    nvcv::Array m_impl; // must come before m_key
    Key         m_key;
    int64_t     m_size_inbytes = -1;

    mutable py::object m_cacheExternalObject;

    py::object m_wrappedObject; // null if not wrapping
};

std::ostream &operator<<(std::ostream &out, const Array &array);

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_ARRAY_HPP

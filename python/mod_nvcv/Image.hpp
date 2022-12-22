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

#ifndef NVCV_PYTHON_PRIV_IMAGE_HPP
#define NVCV_PYTHON_PRIV_IMAGE_HPP

#include "Container.hpp"
#include "CudaBuffer.hpp"
#include "ImageFormat.hpp"
#include "Size.hpp"

#include <nvcv/Image.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorLayout.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <optional>
#include <variant>

namespace nvcvpy::priv {
namespace py = pybind11;

class Image final : public Container
{
public:
    static void Export(py::module &m);

    static std::shared_ptr<Image> Zeros(const Size2D &size, nvcv::ImageFormat fmt);
    static std::shared_ptr<Image> Create(const Size2D &size, nvcv::ImageFormat fmt);
    static std::shared_ptr<Image> CreateHost(py::buffer buffer, nvcv::ImageFormat fmt);
    static std::shared_ptr<Image> CreateHostVector(std::vector<py::buffer> buffer, nvcv::ImageFormat fmt);

    static std::shared_ptr<Image> WrapCuda(CudaBuffer &buffer, nvcv::ImageFormat fmt);
    static std::shared_ptr<Image> WrapCudaVector(std::vector<std::shared_ptr<CudaBuffer>> buffer,
                                                 nvcv::ImageFormat                        fmt);

    std::shared_ptr<Image>       shared_from_this();
    std::shared_ptr<const Image> shared_from_this() const;

    Size2D            size() const;
    int32_t           width() const;
    int32_t           height() const;
    nvcv::ImageFormat format() const;

    friend std::ostream &operator<<(std::ostream &out, const Image &img);

    nvcv::IImage &impl()
    {
        return *m_impl;
    }

    const nvcv::IImage &impl() const
    {
        return *m_impl;
    }

    class Key final : public IKey
    {
    public:
        explicit Key()
            : m_wrapper(true)
        {
        }

        explicit Key(Size2D size, nvcv::ImageFormat fmt)
            : m_size(size)
            , m_format(fmt)
            , m_wrapper(false)
        {
        }

    private:
        Size2D            m_size;
        nvcv::ImageFormat m_format;
        bool              m_wrapper;

        virtual size_t doGetHash() const override;
        virtual bool   doIsEqual(const IKey &that) const override;
    };

    virtual const Key &key() const override
    {
        return m_key;
    }

    py::object cpu(std::optional<nvcv::TensorLayout> layout) const;
    py::object cuda(std::optional<nvcv::TensorLayout> layout) const;

private:
    explicit Image(const Size2D &size, nvcv::ImageFormat fmt);
    explicit Image(std::vector<std::shared_ptr<CudaBuffer>> buf, const nvcv::IImageDataStridedCuda &imgData);
    explicit Image(std::vector<py::buffer> buf, const nvcv::IImageDataStridedHost &imgData);

    std::unique_ptr<nvcv::IImage> m_impl; // must come before m_key
    Key                           m_key;

    mutable py::object                        m_cacheCudaObject;
    mutable std::optional<nvcv::TensorLayout> m_cacheCudaObjectLayout;

    py::object m_wrapped;
};

std::ostream &operator<<(std::ostream &out, const Image &img);

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_IMAGE_HPP

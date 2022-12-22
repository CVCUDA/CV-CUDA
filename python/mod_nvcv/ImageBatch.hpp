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

#ifndef NVCV_PYTHON_PRIV_IMAGEBATCH_HPP
#define NVCV_PYTHON_PRIV_IMAGEBATCH_HPP

#include "Container.hpp"
#include "Size.hpp"

#include <nvcv/ImageBatch.hpp>

#include <vector>

namespace nvcvpy::priv {
namespace py = pybind11;

class Image;

class ImageBatchVarShape : public Container
{
    using ImageList = std::vector<std::shared_ptr<Image>>;

public:
    static void Export(py::module &m);

    static std::shared_ptr<ImageBatchVarShape> Create(int capacity);

    std::shared_ptr<ImageBatchVarShape>       shared_from_this();
    std::shared_ptr<const ImageBatchVarShape> shared_from_this() const;

    const nvcv::ImageBatchVarShape &impl() const;
    nvcv::ImageBatchVarShape       &impl();

    // Let's simplify a bit and NOT export the base class ImageBatch,
    // as we currently have only one leaf class (this one).
    py::object uniqueFormat() const;
    int32_t    capacity() const;
    int32_t    numImages() const;
    Size2D     maxSize() const;

    void pushBack(Image &img);
    void pushBackMany(std::vector<std::shared_ptr<Image>> &imgList);
    void popBack(int imgCount);
    void clear();

    ImageList::const_iterator begin() const;
    ImageList::const_iterator end() const;

    class Key final : public IKey
    {
    public:
        explicit Key(int capacity)
            : m_capacity(capacity)
        {
        }

    private:
        int m_capacity;

        virtual size_t doGetHash() const override;
        virtual bool   doIsEqual(const IKey &that) const override;
    };

    virtual const Key &key() const override
    {
        return m_key;
    }

private:
    explicit ImageBatchVarShape(int capacity);
    Key                      m_key;
    ImageList                m_list;
    nvcv::ImageBatchVarShape m_impl;
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_IMAGEBATCH_HPP

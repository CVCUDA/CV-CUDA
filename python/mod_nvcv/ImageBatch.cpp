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

#include "ImageBatch.hpp"

#include "Image.hpp"

#include <common/Assert.hpp>

namespace nvcvpy::priv {

size_t ImageBatchVarShape::Key::doGetHash() const
{
    using util::ComputeHash;
    return ComputeHash(m_capacity);
}

bool ImageBatchVarShape::Key::doIsEqual(const IKey &ithat) const
{
    auto &that = static_cast<const Key &>(ithat);
    return m_capacity == that.m_capacity;
}

std::shared_ptr<ImageBatchVarShape> ImageBatchVarShape::Create(int capacity)
{
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Key{capacity});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<ImageBatchVarShape> batch(new ImageBatchVarShape(capacity));
        Cache::Instance().add(*batch);
        return batch;
    }
    else
    {
        // Get the first one
        auto batch = std::static_pointer_cast<ImageBatchVarShape>(vcont[0]);
        batch->clear(); // make sure it's in pristine state
        return batch;
    }
}

ImageBatchVarShape::ImageBatchVarShape(int capacity)
    : m_key(capacity)
    , m_impl(capacity)
{
    m_list.reserve(capacity);
}

const nvcv::ImageBatchVarShape &ImageBatchVarShape::impl() const
{
    return m_impl;
}

nvcv::ImageBatchVarShape &ImageBatchVarShape::impl()
{
    return m_impl;
}

py::object ImageBatchVarShape::uniqueFormat() const
{
    nvcv::ImageFormat fmt = m_impl.uniqueFormat();
    if (fmt)
    {
        return py::cast(fmt);
    }
    else
    {
        return py::none();
    }
}

Size2D ImageBatchVarShape::maxSize() const
{
    nvcv::Size2D s = m_impl.maxSize();
    return {s.w, s.h};
}

int32_t ImageBatchVarShape::capacity() const
{
    return m_impl.capacity();
}

int32_t ImageBatchVarShape::numImages() const
{
    NVCV_ASSERT(m_impl.numImages() == (int)m_list.size());
    return m_impl.numImages();
}

void ImageBatchVarShape::pushBack(Image &img)
{
    m_impl.pushBack(img.impl());
    m_list.push_back(img.shared_from_this());
}

void ImageBatchVarShape::pushBackMany(std::vector<std::shared_ptr<Image>> &imgList)
{
    // TODO: use an iterator that return the handle when dereferenced, this
    // would avoid creating this vector.
    std::vector<NVCVImageHandle> handles;
    handles.reserve(imgList.size());
    for (auto &img : imgList)
    {
        handles.push_back(img->impl().handle());
        m_list.push_back(img);
    }

    m_impl.pushBack(handles.begin(), handles.end());
}

void ImageBatchVarShape::popBack(int imgCount)
{
    m_impl.popBack(imgCount);
    m_list.erase(m_list.end() - imgCount, m_list.end());
}

void ImageBatchVarShape::clear()
{
    m_impl.clear();
    m_list.clear();
}

auto ImageBatchVarShape::begin() const -> ImageList::const_iterator
{
    return m_list.begin();
}

auto ImageBatchVarShape::end() const -> ImageList::const_iterator
{
    return m_list.end();
}

std::shared_ptr<ImageBatchVarShape> ImageBatchVarShape::shared_from_this()
{
    return std::static_pointer_cast<ImageBatchVarShape>(Container::shared_from_this());
}

std::shared_ptr<const ImageBatchVarShape> ImageBatchVarShape::shared_from_this() const
{
    return std::static_pointer_cast<const ImageBatchVarShape>(Container::shared_from_this());
}

void ImageBatchVarShape::Export(py::module &m)
{
    using namespace py::literals;

    py::class_<ImageBatchVarShape, std::shared_ptr<ImageBatchVarShape>, Container>(m, "ImageBatchVarShape")
        .def(py::init(&ImageBatchVarShape::Create), "capacity"_a)
        .def_property_readonly("uniqueformat", &ImageBatchVarShape::uniqueFormat)
        .def_property_readonly("maxsize", &ImageBatchVarShape::maxSize)
        .def_property_readonly("capacity", &ImageBatchVarShape::capacity)
        .def("__len__", &ImageBatchVarShape::numImages)
        .def("__iter__", [](const ImageBatchVarShape &list) { return py::make_iterator(list); })
        .def("pushback", &ImageBatchVarShape::pushBack)
        .def("pushback", &ImageBatchVarShape::pushBackMany)
        .def("popback", &ImageBatchVarShape::popBack, "count"_a = 1)
        .def("clear", &ImageBatchVarShape::clear);
}

} // namespace nvcvpy::priv

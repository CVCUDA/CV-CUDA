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

#include "ImageBatch.hpp"

#include "CastUtils.hpp"
#include "ExternalBuffer.hpp"
#include "Image.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>

namespace nvcvpy::priv {

size_t ImageBatchVarShape::Key::doGetHash() const
{
    using util::ComputeHash;
    return ComputeHash(m_capacity);
}

bool ImageBatchVarShape::Key::doIsCompatible(const IKey &ithat) const
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
        auto       batch = std::static_pointer_cast<ImageBatchVarShape>(vcont[0]);
        Image::Key key;
        batch->clear(); // make sure it's in pristine state
        return batch;
    }
}

std::shared_ptr<ImageBatchVarShape> ImageBatchVarShape::WrapExternalBufferVector(std::vector<py::object> buffers,
                                                                                 nvcv::ImageFormat       fmt)
{
    std::vector<std::shared_ptr<ExternalBuffer>> buflist;
    buflist.reserve(buffers.size());

    for (py::object &obj : buffers)
    {
        std::shared_ptr<ExternalBuffer> buffer = cast_py_object_as<ExternalBuffer>(obj);
        if (!buffer)
        {
            throw std::runtime_error("Input buffer doesn't provide cuda_array_interface or DLPack interfaces");
        }
        buflist.push_back(buffer);
    }

    std::shared_ptr<ImageBatchVarShape> batch = Create(buffers.size());
    batch->pushBackMany(Image::WrapExternalBufferMany(buflist, fmt));

    return batch;
}

ImageBatchVarShape::ImageBatchVarShape(int capacity)
    : m_key(capacity)
    , m_impl(capacity)
    , m_size_inbytes(doComputeSizeInBytes(nvcv::ImageBatchVarShape::CalcRequirements(capacity)))
{
    m_list.reserve(capacity);
}

int64_t ImageBatchVarShape::doComputeSizeInBytes(const NVCVImageBatchVarShapeRequirements &reqs)
{
    int64_t size_inbytes;
    util::CheckThrow(nvcvMemRequirementsCalcTotalSizeBytes(&(reqs.mem.cudaMem), &size_inbytes));
    return size_inbytes;
}

int64_t ImageBatchVarShape::GetSizeInBytes() const
{
    // m_size_inbytes == -1 indicates failure case and value has not been computed yet
    NVCV_ASSERT(m_size_inbytes != -1
                && "ImageBatchVarShape has m_size_inbytes == -1, ie m_size_inbytes has not been correctly set");
    return m_size_inbytes;
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

void ImageBatchVarShape::pushBackMany(const std::vector<std::shared_ptr<Image>> &imgList)
{
    std::vector<NVCVImageHandle> handlelist;
    handlelist.reserve(imgList.size());
    for (size_t i = 0; i < imgList.size(); ++i)
    {
        if (imgList[i])
        {
            handlelist.push_back(imgList[i]->impl().handle());
        }
        else
        {
            handlelist.push_back(nullptr);
        }
        m_list.push_back(imgList[i]);
    }

    nvcv::detail::CheckThrow(nvcvImageBatchVarShapePushImages(m_impl.handle(), handlelist.data(), handlelist.size()));
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

    py::class_<ImageBatchVarShape, std::shared_ptr<ImageBatchVarShape>, Container>(m, "ImageBatchVarShape",
                                                                                   "Batch of Images.")
        .def(py::init(&ImageBatchVarShape::Create), "capacity"_a,
             "Create a new ImageBatchVarShape object with the specified capacity.")
        .def_property_readonly("uniqueformat", &ImageBatchVarShape::uniqueFormat,
                               "Return True if all the images have the same format, False otherwise.")
        .def_property_readonly("maxsize", &ImageBatchVarShape::maxSize,
                               "Return the maximum size of the ImageBatchVarShape in bytes.")
        .def_property_readonly("capacity", &ImageBatchVarShape::capacity,
                               "Return the capacity of the ImageBatchVarShape in number of images.")
        .def("__len__", &ImageBatchVarShape::numImages, "Return the number of images in the ImageBatchVarShape.")
        .def(
            "__iter__", [](const ImageBatchVarShape &list) { return py::make_iterator(list); },
            "Return an iterator over the images in the ImageBatchVarShape.")
        .def("pushback", &ImageBatchVarShape::pushBack, "Add a new image to the end of the ImageBatchVarShape.")
        .def("pushback", &ImageBatchVarShape::pushBackMany, "Add multiple images to the end of the ImageBatchVarShape.")
        .def("popback", &ImageBatchVarShape::popBack, "count"_a = 1,
             "Remove one or more images from the end of the ImageBatchVarShape.")
        .def("clear", &ImageBatchVarShape::clear, "Remove all images from the ImageBatchVarShape.");

    m.def("as_images", &ImageBatchVarShape::WrapExternalBufferVector, py::arg_v("buffers", std::vector<py::object>{}),
          "format"_a = nvcv::FMT_NONE, py::keep_alive<0, 1>(),
          "Wrap a vector of external buffers as a batch of images, and tie the buffers lifetime to it");
}

} // namespace nvcvpy::priv

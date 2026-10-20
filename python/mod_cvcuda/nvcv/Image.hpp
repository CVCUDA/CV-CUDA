/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ExternalBuffer.hpp"
#include "ImageFormat.hpp"
#include "Size.hpp"

#include <common/Binding.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorLayout.hpp>

#include <array>
#include <memory>
#include <optional>
#include <variant>

namespace nvcvpy::priv {
namespace py = nanobind;

struct BufferAllocation;

class Image final : public Container
{
public:
    static void Export(py::module_ &m);

    static std::shared_ptr<Image> Zeros(const Size2D &size, nvcv::ImageFormat fmt, int rowAlign);
    static std::shared_ptr<Image> Create(const Size2D &size, nvcv::ImageFormat fmt, int rowAlign);
    static std::shared_ptr<Image> CreateHost(py::buffer buffer, nvcv::ImageFormat fmt, int rowAlign);
    static std::shared_ptr<Image> CreateHostVector(const std::vector<py::buffer> &buffer, nvcv::ImageFormat fmt,
                                                   int rowAlign);

    static std::shared_ptr<Image>              WrapExternalBuffer(ExternalBuffer &buffer, nvcv::ImageFormat fmt);
    static std::vector<std::shared_ptr<Image>> WrapExternalBufferMany(
        std::vector<std::shared_ptr<ExternalBuffer>> &buffer, nvcv::ImageFormat fmt);
    static std::shared_ptr<Image> WrapExternalBufferVector(std::vector<py::object> buffer, nvcv::ImageFormat fmt);

    Size2D            size() const;
    int32_t           width() const;
    int32_t           height() const;
    nvcv::ImageFormat format() const;

    int64_t        GetSizeInBytes() const override;
    RetainedMemory retainedMemory() const override;
    bool           allowsCacheReuse(bool pendingRelease) const override;

    friend std::ostream &operator<<(std::ostream &out, const Image &img);

    nvcv::Image &impl()
    {
        return m_impl;
    }

    const nvcv::Image &impl() const
    {
        return m_impl;
    }

    class Key final : public IKey
    {
    public:
        explicit Key()
            : m_isWrapper(true)
        {
        }

        explicit Key(const nvcv::Image::Requirements &reqs);

        bool matchesDescriptor(const Key &requested) const noexcept;
        void setDescriptor(const nvcv::Image::Requirements &reqs) noexcept;

        int32_t alignment() const noexcept
        {
            return m_alignBytes;
        }

    private:
        // Descriptor fields are filtered separately; hash/equality use only
        // format so differently sized images share capacity.
        std::array<int32_t, NVCV_MAX_PLANE_COUNT> m_rowStrides{};
        nvcv::ImageFormat                         m_format;
        Size2D                                    m_size{};
        int32_t                                   m_alignBytes = 0;
        int32_t                                   m_numPlanes  = 0;
        bool                                      m_isWrapper;

        size_t doGetHash() const override;
        bool   doIsCompatible(const IKey &that) const override;
    };

    const Key &key() const override
    {
        return m_key;
    }

    py::object cpu(std::optional<nvcv::TensorLayout> layout) const;
    py::object cuda(std::optional<nvcv::TensorLayout> layout) const;

private:
    void bindSubmissionState();
    explicit Image(const nvcv::Image::Requirements &reqs);
    explicit Image(std::vector<std::shared_ptr<ExternalBuffer>> buf, const nvcv::ImageDataStridedCuda &imgData);

    static int64_t doComputeSizeInBytes(const NVCVImageRequirements &reqs);

    void setWrapData(std::vector<std::shared_ptr<ExternalBuffer>> buf, const nvcv::ImageDataStridedCuda &imgData);

    std::shared_ptr<BufferAllocation> m_buffer;
    nvcv::Image                       m_impl; // must come before m_key
    Key                               m_key;
    int64_t                           m_size_inbytes = -1;

    struct WrapData
    {
        DLDeviceType    devType;
        int             devId;
        py::object      obj;
        ExternalBuffer *singleBuffer = nullptr; // Borrowed from obj.
    };

    // If wrapping external data, it's not nullopt.
    std::optional<WrapData> m_wrapData;
};

std::ostream &operator<<(std::ostream &out, const Image &img);

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_IMAGE_HPP

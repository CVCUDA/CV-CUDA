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

#ifndef NVCV_PYTHON_PRIV_TENSOR_HPP
#define NVCV_PYTHON_PRIV_TENSOR_HPP

#include "Container.hpp"
#include "Size.hpp"

#include <common/Binding.hpp>
#include <nvcv/Shape.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/python/Shape.hpp>

#include <array>

namespace nvcvpy::priv {
namespace py = nanobind;

class ExternalBuffer;
class Image;
struct BufferAllocation;

// Register nvcv::TensorLayout.  Split out of Tensor::Export so it can run
// early — Image::cpu/cuda take a TensorLayout parameter and nanobind needs
// the type registered before Image::Export binds those methods.
void ExportTensorLayout(py::module_ &m);

class Tensor : public Container
{
public:
    static void Export(py::module_ &m);

    static std::shared_ptr<Tensor> CreateForImageBatch(int numImages, const Size2D &size, nvcv::ImageFormat fmt,
                                                       int rowalign);
    static std::shared_ptr<Tensor> Create(Shape shape, nvcv::DataType dtype, std::optional<nvcv::TensorLayout> layout,
                                          int rowalign, Stream *consumer = nullptr);

    static std::shared_ptr<Tensor> CreateFromReqs(const nvcv::Tensor::Requirements &reqs, Stream *consumer = nullptr);

    static std::shared_ptr<Tensor> Wrap(ExternalBuffer &buffer, std::optional<nvcv::TensorLayout> layout);
    static std::shared_ptr<Tensor> WrapImage(Image &img);
    static std::shared_ptr<Tensor> ReshapeTensor(Tensor &tensor, Shape shape, std::optional<nvcv::TensorLayout> layout);

    std::shared_ptr<Tensor> Reshape(Shape shape, std::optional<nvcv::TensorLayout> layout);

    std::optional<nvcv::TensorLayout> layout() const;
    Shape                             shape() const;
    nvcv::DataType                    dtype() const;
    int                               rank() const;

    nvcv::Tensor       &impl();
    const nvcv::Tensor &impl() const;

    class Key final : public IKey
    {
    public:
        explicit Key(const nvcv::Tensor::Requirements &reqs);
        Key(const Key &)            = delete;
        Key &operator=(const Key &) = delete;

        bool matchesDescriptor(const Key &requested) const noexcept;
        void setDescriptor(const nvcv::Tensor::Requirements &reqs) noexcept;

        int32_t alignment() const noexcept
        {
            return m_alignBytes;
        }

    private:
        // Descriptor fields are filtered separately; hash/equality use only
        // layout and dtype so differently shaped allocations share capacity.
        std::array<int64_t, NVCV_TENSOR_MAX_RANK> m_shape;
        std::array<int64_t, NVCV_TENSOR_MAX_RANK> m_strides;
        nvcv::TensorLayout                        m_layout;
        nvcv::DataType                            m_dtype;
        int32_t                                   m_alignBytes = 0;
        int32_t                                   m_rank       = 0;

        size_t doGetHash() const override;
        bool   doIsCompatible(const IKey &that) const override;
    };

    const IKey &key() const override;

    py::object cuda() const;

    int64_t        GetSizeInBytes() const override;
    RetainedMemory retainedMemory() const override;
    bool           allowsCacheReuse(bool pendingRelease) const override;

private:
    class WrapperKey final : public IKey
    {
    public:
        WrapperKey() = default;

        explicit WrapperKey(const IKey &deviceKey)
            : IKey(deviceKey)
        {
        }

    private:
        size_t doGetHash() const override
        {
            return 0;
        }

        bool doIsCompatible(const IKey &) const override
        {
            return true;
        }
    };

    explicit Tensor(const nvcv::Tensor::Requirements &reqs);
    Tensor(const nvcv::TensorData &data, py::object wrappedObject);
    explicit Tensor(Image &img);
    Tensor(nvcv::Tensor &&tensor, const IKey &deviceKey);

    static int64_t doComputeSizeInBytes(const nvcv::Tensor::Requirements &reqs);

    std::shared_ptr<BufferAllocation> m_buffer;
    nvcv::Tensor                      m_impl; // must come before m_key
    std::unique_ptr<Key>              m_key;
    WrapperKey                        m_wrapperKey;
    int64_t                           m_size_inbytes = -1;

    mutable py::object                        m_cacheExternalObject;
    mutable std::optional<nvcv::TensorLayout> m_cacheExternalObjectLayout;

    py::object m_wrappedObject; // null if not wrapping
};

std::ostream &operator<<(std::ostream &out, const Tensor &tensor);

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_TENSOR_HPP

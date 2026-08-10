/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Operators.hpp"
#include "VarShapeUtils.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpInpaint.hpp>
#include <nvcv/TensorLayoutInfo.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

class InpaintError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class PyOpInpaint : public nvcvpy::Container // NOSONAR: operator wrappers share the Python cache hierarchy.
{
public:
    class Key : public nvcvpy::IKey
    {
    public:
        Key(int maxBatchSize, nvcv::Size2D &maxShape)
            : m_maxShape{maxShape}
            , m_maxBatchSize{maxBatchSize}
        {
        }

        size_t payloadSize() const
        {
            return static_cast<size_t>(m_maxShape.w) * m_maxShape.h * m_maxBatchSize;
        }

    private:
        size_t doGetHash() const override
        {
            return ComputeHash(m_maxShape);
        }

        bool doIsCompatible(const nvcvpy::IKey &that_) const override
        {
            const auto &that = static_cast<const Key &>(that_);
            return this->payloadSize() <= that.payloadSize();
        }

        nvcv::Size2D m_maxShape;
        int          m_maxBatchSize;
    };

    PyOpInpaint(int maxBatchSize, nvcv::Size2D &maxShape)
        : m_key(maxBatchSize, maxShape)
        , m_op(maxBatchSize, maxShape)
    {
    }

    template<class... AA>
    void submit(AA &&...args)
    {
        m_op(std::forward<AA>(args)...);
    }

    py::object container() const override
    {
        return py::reinterpret_borrow<py::object>(this->ptr());
    }

    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    static std::shared_ptr<nvcvpy::ICacheItem> fetch(std::vector<std::shared_ptr<nvcvpy::ICacheItem>> &cache)
    {
        assert(!cache.empty());

        // Find the operator with the largest workspace (can handle any smaller request)
        std::shared_ptr<nvcvpy::ICacheItem> retItem        = cache[0];
        size_t                              maxPayloadSize = 0;

        for (const auto &item : cache)
        {
            const auto &key            = static_cast<const Key &>(item.get()->key());
            auto        keyPayloadSize = key.payloadSize();

            if (keyPayloadSize > maxPayloadSize)
            {
                maxPayloadSize = keyPayloadSize;
                retItem        = item;
            }
        }

        // Note: Removed removeAllNotInUseMatching() call to reduce per-call overhead.
        // The cache will naturally evict unused operators when memory pressure occurs.

        return retItem;
    }

private:
    Key             m_key;
    cvcuda::Inpaint m_op;
};

Tensor InpaintInto(Tensor &output, Tensor &input, Tensor &masks, double inpaintRadius, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto info = nvcv::TensorLayoutInfoImage::Create(input.layout());
    if (!info)
    {
        throw InpaintError("Non-supported tensor layout");
    }

    auto         shape     = input.shape();
    auto         batchSize = info->idxSample() >= 0 ? (int)shape[info->idxSample()] : 1;
    auto         h         = (int)shape[info->idxHeight()];
    auto         w         = (int)shape[info->idxWidth()];
    nvcv::Size2D maxShape{w, h};

    auto inpaint = CreateOperator<cvcuda::Inpaint>(batchSize, maxShape);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, masks});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*inpaint});

    guard.run([&inpaint, &pstream, &input, &masks, &output, &inpaintRadius]()
              { inpaint->submit(pstream->cudaHandle(), input, masks, output, inpaintRadius); });

    return output;
}

Tensor Inpaint(Tensor &input, Tensor &masks, double inpaintRadius, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return InpaintInto(output, input, masks, inpaintRadius, pstream);
}

ImageBatchVarShape InpaintVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, ImageBatchVarShape &masks,
                                       double inpaintRadius, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }
    nvcv::Size2D maxShape = input.maxSize();

    // Use simple CreateOperator (like Flip/Gaussian)
    auto inpaint = CreateOperator<cvcuda::Inpaint>(input.numImages(), maxShape);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, masks});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*inpaint});

    guard.run([&inpaint, &pstream, &input, &masks, &output, &inpaintRadius]()
              { inpaint->submit(pstream->cudaHandle(), input, masks, output, inpaintRadius); });

    return output;
}

ImageBatchVarShape InpaintVarShape(ImageBatchVarShape &input, ImageBatchVarShape &masks, double inpaintRadius,
                                   std::optional<Stream> pstream)
{
    auto format = input.uniqueFormat();
    if (!format)
    {
        throw InpaintError("All images in input must have the same format.");
    }

    ImageBatchVarShape output = CreateSameShapeImageBatch(input, format, input.numImages());

    return InpaintVarShapeInto(output, input, masks, inpaintRadius, pstream);
}

} // namespace

void ExportOpInpaint(py::module &m)
{
    using namespace pybind11::literals;

    m.def("inpaint", NvtxTrace("cvcuda.inpaint", &Inpaint), "src"_a, "masks"_a, "inpaintRadius"_a, py::kw_only(),
          "stream"_a = nullptr,
          R"pbdoc(
        Executes the Inpaint operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            masks (cvcuda.Tensor): Mask tensor, 8-bit 1-channel images. Non-zero pixels indicate the area that needs to be inpainted.
            inpaintRadius (float): Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("inpaint_into", NvtxTrace("cvcuda.inpaint_into", &InpaintInto), "dst"_a, "src"_a, "masks"_a,
          "inpaintRadius"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Inpaint operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            masks (cvcuda.Tensor): Mask tensor, 8-bit 1-channel images. Non-zero pixels indicate the area that needs to be inpainted.
            inpaintRadius (float): Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("inpaint", NvtxTrace("cvcuda.inpaint", &InpaintVarShape), "src"_a, "masks"_a, "inpaintRadius"_a,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the Inpaint operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            masks (cvcuda.ImageBatchVarShape): Mask image batch, 8-bit 1-channel images. Non-zero pixels indicate the area that needs to be inpainted.
            inpaintRadius (float): Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("inpaint_into", NvtxTrace("cvcuda.inpaint_into", &InpaintVarShapeInto), "dst"_a, "src"_a, "masks"_a,
          "inpaintRadius"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Inpaint operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            masks (cvcuda.ImageBatchVarShape): Mask image batch, 8-bit 1-channel images. Non-zero pixels indicate the area that needs to be inpainted.
            inpaintRadius (float): Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpInpaint.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {

class PyOpInpaint : public nvcvpy::Container
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
            const Key &that = static_cast<const Key &>(that_);
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
        return *this;
    }

    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    static std::shared_ptr<nvcvpy::ICacheItem> fetch(std::vector<std::shared_ptr<nvcvpy::ICacheItem>> &cache)
    {
        assert(!cache.empty());

        std::shared_ptr<nvcvpy::ICacheItem> retItem        = cache[0];
        size_t                              maxPayloadSize = 0;

        for (const auto &item : cache)
        {
            const Key &key            = static_cast<const Key &>(item.get()->key());
            size_t     keyPayloadSize = key.payloadSize();

            if (keyPayloadSize > maxPayloadSize)
            {
                maxPayloadSize = keyPayloadSize;
                retItem        = item;
            }
        }

        cache.clear();

        nvcvpy::Cache::removeAllNotInUseMatching(retItem.get()->key());

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

    nvcv::TensorShape shape = input.shape();
    nvcv::Size2D      maxShape{(int)shape[2], (int)shape[1]};
    auto              inpaint = CreateOperatorEx<PyOpInpaint>((int)shape[0], maxShape);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, masks});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*inpaint});

    inpaint->submit(pstream->cudaHandle(), input, masks, output, inpaintRadius);

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
    auto         inpaint  = CreateOperatorEx<PyOpInpaint>(input.numImages(), maxShape);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, masks});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*inpaint});

    inpaint->submit(pstream->cudaHandle(), input, masks, output, inpaintRadius);

    return output;
}

ImageBatchVarShape InpaintVarShape(ImageBatchVarShape &input, ImageBatchVarShape &masks, double inpaintRadius,
                                   std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.numImages());

    auto format = input.uniqueFormat();
    if (!format)
    {
        throw std::runtime_error("All images in input must have the same format.");
    }

    for (auto img = input.begin(); img != input.end(); ++img)
    {
        auto newimg = Image::Create(img->size(), format);
        output.pushBack(newimg);
    }

    return InpaintVarShapeInto(output, input, masks, inpaintRadius, pstream);
}

} // namespace

void ExportOpInpaint(py::module &m)
{
    using namespace pybind11::literals;
    py::options options;
    options.disable_function_signatures();

    m.def("inpaint", &Inpaint, "src"_a, "masks"_a, "inpaintRadius"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

	cvcuda.inpaint(src: nvcv.Tensor, masks: Tensor, inpaintRadius: float, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor

        Executes the Inpaint operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Inpaint operator
            for more details and usage examples.

        Args:
            src (nvcv.Tensor): Input tensor containing one or more images.
            masks (nvcv.Tensor): Mask tensor, 8-bit 1-channel images. Non-zero pixels indicate the area that needs to be inpainted.
            inpaintRadius (float): Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("inpaint_into", &InpaintInto, "dst"_a, "src"_a, "masks"_a, "inpaintRadius"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

	cvcuda.inpaint_into(dst: nvcv.Tensor, src: nvcv.Tensor, masks: Tensor, inpaintRadius: float, stream: Optional[nvcv.cuda.Stream] = None)

	Executes the Inpaint operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Inpaint operator
            for more details and usage examples.

        Args:
            dst (nvcv.Tensor): Output tensor to store the result of the operation.
            src (nvcv.Tensor): Input tensor containing one or more images.
            masks (nvcv.Tensor): Mask tensor, 8-bit 1-channel images. Non-zero pixels indicate the area that needs to be inpainted.
            inpaintRadius (float): Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("inpaint", &InpaintVarShape, "src"_a, "masks"_a, "inpaintRadius"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

	cvcuda.inpaint(src: nvcv.ImageBatchVarShape, masks:ImageBatchVarShape, inpaintRadius: float, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.ImageBatchVarShape

	Executes the Inpaint operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Inpaint operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            masks (nvcv.ImageBatchVarShape): Mask image batch, 8-bit 1-channel images. Non-zero pixels indicate the area that needs to be inpainted.
            inpaintRadius (float): Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("inpaint_into", &InpaintVarShapeInto, "dst"_a, "src"_a, "masks"_a, "inpaintRadius"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(


	cvcuda.inpaint_into(dst: nvcv.ImageBatchVarShape, src: nvcv.ImageBatchVarShape, masks:ImageBatchVarShape, inpaintRadius: float, stream: Optional[nvcv.cuda.Stream] = None)

	Executes the Inpaint operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Inpaint operator
            for more details and usage examples.

        Args:
            dst (nvcv.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            masks (nvcv.ImageBatchVarShape): Mask image batch, 8-bit 1-channel images. Non-zero pixels indicate the area that needs to be inpainted.
            inpaintRadius (float): Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpHistogramEq.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {
Tensor HistogramEqInto(Tensor &output, Tensor &input, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }
    nvcv::TensorShape shape = input.shape();
    auto              op    = CreateOperator<cvcuda::HistogramEq>((uint32_t)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    op->submit(pstream->cudaHandle(), input, output);

    return std::move(output);
}

Tensor HistogramEq(Tensor &input, nvcv::DataType dtype, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), dtype);

    return HistogramEqInto(output, input, pstream);
}

ImageBatchVarShape HistogramEqVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                           std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::HistogramEq>((uint32_t)input.numImages());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    op->submit(pstream->cudaHandle(), input, output);

    return output;
}

ImageBatchVarShape HistogramEqVarShape(ImageBatchVarShape &input, std::optional<Stream> pstream)
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

    return HistogramEqVarShapeInto(output, input, pstream);
}

} // namespace

void ExportOpHistogramEq(py::module &m)
{
    using namespace pybind11::literals;
    py::options options;
    options.disable_function_signatures();

    m.def("histogrameq", &HistogramEq, "src"_a, "dtype"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

	cvcuda.histogrameq(src: nvcv.Tensor, dtype: numpy.dtype, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor

        Executes the histogram equalization operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Histogram Eq operator
            for more details and usage examples.

        Args:
            src (nvcv.Tensor): Input image batch containing one or more images.
            dtype (numpy.dtype): The data type of the output.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("histogrameq_into", &HistogramEqInto, "dst"_a, "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

	cvcuda.histogrameq_into(dst: nvcv.Tensor, src: nvcv.Tensor, stream: Optional[nvcv.cuda.Stream] = None)

        Executes the histogram equalization operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Histogram Eq operator
            for more details and usage examples.

        Args:
            dst (nvcv.Tensor): Output image batch containing the result of the operation.
            src (nvcv.Tensor): Input image batch containing one or more images.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("histogrameq", &HistogramEqVarShape, "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

	cvcuda.histogrameq(src: nvcv.ImageBatchVarShape, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.ImageBatchVarShape

	Executes the histogram equalization operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the GaussianNoise operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.nvcv.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("histogrameq_into", &HistogramEqVarShapeInto, "dst"_a, "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

	cvcuda.histogrameq_into(dst: nvcv.ImageBatchVarShape, src: nvcv.ImageBatchVarShape, stream: Optional[nvcv.cuda.Stream] = None)

        Executes the histogram equalization operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the GaussianNoise operator
            for more details and usage examples.

        Args:
            dst (nvcv.ImageBatchVarShape): Output image batch containing the result of the operation.
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

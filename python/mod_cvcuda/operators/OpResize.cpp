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

#include "Operators.hpp"
#include "VarShapeUtils.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpResize.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

class ResizeError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor ResizeInto(Tensor &output, Tensor &input, NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto resize = CreateOperator<cvcuda::Resize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*resize});

    guard.run([&resize, &pstream, &input, &output, &interp]()
              { resize->submit(pstream->cudaHandle(), input, output, interp); });

    return std::move(output);
}

Tensor Resize(Tensor &input, const Shape &out_shape, NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(out_shape, input.dtype(), input.shape().layout());

    return ResizeInto(output, input, interp, pstream);
}

ImageBatchVarShape ResizeVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                      NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto resize = CreateOperator<cvcuda::Resize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*resize});

    guard.run([&resize, &pstream, &input, &output, &interp]()
              { resize->submit(pstream->cudaHandle(), input, output, interp); });

    return output;
}

ImageBatchVarShape ResizeVarShape(ImageBatchVarShape &input, const std::vector<std::tuple<int, int>> &out_size,
                                  NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    if (input.numImages() != (int)out_size.size())
    {
        throw ResizeError("Number of input images must be equal to the number of elements in output size list ");
    }

    ImageBatchVarShape output = CreateSizedImageBatch(input, out_size);

    return ResizeVarShapeInto(output, input, interp, pstream);
}

} // namespace

void ExportOpResize(py::module &m)
{
    using namespace pybind11::literals;

    m.def("resize", NvtxTrace("cvcuda.resize", &Resize), "src"_a, "shape"_a, "interp"_a = NVCV_INTERP_LINEAR,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the Resize operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            shape (Tuple[int]): Shape of output tensor.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("resize_into", NvtxTrace("cvcuda.resize_into", &ResizeInto), "dst"_a, "src"_a,
          "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Resize operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("resize", NvtxTrace("cvcuda.resize", &ResizeVarShape), "src"_a, "sizes"_a, "interp"_a = NVCV_INTERP_LINEAR,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Resize operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            sizes (Tuple vector): Shapes of output images.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("resize_into", NvtxTrace("cvcuda.resize_into", &ResizeVarShapeInto), "dst"_a, "src"_a,
          "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Resize operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

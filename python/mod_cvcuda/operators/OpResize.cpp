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

#include "NativeBinding.hpp"
#include "Operators.hpp"

#include <common/Binding.hpp>
#include <common/PyUtil.hpp>
#include <cvcuda/OpResize.hpp>

#include <optional>
#include <stdexcept>

namespace cvcudapy {

namespace {

using native::Image;
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

class ResizeError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor &ResizeInto(Tensor &output, Tensor &input, NVCVInterpolationType interp, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto resize = CreateOperator<cvcuda::Resize>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*resize});

    guard.run([&resize, &stream, &input, &output, &interp]()
              { resize->submit(stream.handle(), input.impl(), output.impl(), interp); });

    return output;
}

std::shared_ptr<Tensor> Resize(Tensor &input, const Shape &out_shape, NVCVInterpolationType interp,
                               std::optional<Stream *> pstream)
{
    nvcv::TensorShape tensorShape = CreateNVCVTensorShape(out_shape, input.impl().layout());
    auto output = nvcvpy::priv::Tensor::CreateFromReqs(nvcv::Tensor::CalcRequirements(tensorShape, input.dtype()));

    ResizeInto(*output, input, interp, pstream);
    return output;
}

ImageBatchVarShape &ResizeVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                       NVCVInterpolationType interp, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto resize = CreateOperator<cvcuda::Resize>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*resize});

    guard.run([&resize, &stream, &input, &output, &interp]()
              { resize->submit(stream.handle(), input.impl(), output.impl(), interp); });

    return output;
}

std::shared_ptr<ImageBatchVarShape> ResizeVarShape(ImageBatchVarShape                      &input,
                                                   const std::vector<std::tuple<int, int>> &out_size,
                                                   NVCVInterpolationType interp, std::optional<Stream *> pstream)
{
    if (input.numImages() != (int)out_size.size())
    {
        throw ResizeError("Number of input images must be equal to the number of elements in output size list ");
    }

    auto output = ImageBatchVarShape::Create(input.capacity());
    auto image  = input.begin();
    for (const auto &[width, height] : out_size)
    {
        auto resized = Image::Create({width, height}, (*image)->format(), 0);
        output->pushBack(*resized);
        ++image;
    }

    ResizeVarShapeInto(*output, input, interp, pstream);
    return output;
}

} // namespace

void ExportOpResize(py::module_ &m)
{
    using namespace nanobind::literals;

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
          "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
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
          "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
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

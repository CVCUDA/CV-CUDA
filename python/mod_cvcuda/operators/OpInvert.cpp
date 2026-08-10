/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "UnaryElementwiseOp.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpInvert.hpp>

namespace cvcudapy {

namespace {
// Thin Python-facing wrappers; the create/guard/submit body lives in UnaryElementwiseOp.hpp.
// Invert takes no extra parameters, so nothing is forwarded past the stream.
Tensor InvertInto(Tensor &output, Tensor &input, std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::Invert>(output, input, pstream);
}

Tensor Invert(Tensor &input, std::optional<Stream> pstream)
{
    return UnaryElementwiseTensor<cvcuda::Invert>(input, pstream);
}

ImageBatchVarShape InvertVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                      std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::Invert>(output, input, pstream);
}

ImageBatchVarShape InvertVarShape(ImageBatchVarShape &input, std::optional<Stream> pstream)
{
    return UnaryElementwiseVarShape<cvcuda::Invert>(input, pstream);
}

} // namespace

void ExportOpInvert(py::module &m)
{
    using namespace pybind11::literals;

    m.def("invert", NvtxTrace("cvcuda.invert", &Invert), "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Invert operation on the given cuda stream.

        Computes the per-element photometric negative ``out = bound - in``, where ``bound`` is the
        maximum value of the data type (255 for uint8, 65535 for uint16, 1.0 for float32). Mirrors
        torchvision.transforms.v2.functional.invert / OpenCV cv::bitwise_not (unsigned).

        See also:
            Refer to the CV-CUDA C API reference for the Invert operator for more details and usage
            examples.

        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same shape, dtype, and layout as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("invert_into", NvtxTrace("cvcuda.invert_into", &InvertInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Invert operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Invert operator for more details and usage
            examples.

        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("invert", NvtxTrace("cvcuda.invert", &InvertVarShape), "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Invert operation on the given cuda stream.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same formats and sizes as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("invert_into", NvtxTrace("cvcuda.invert_into", &InvertVarShapeInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Invert operation on the given cuda stream.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
}

} // namespace cvcudapy

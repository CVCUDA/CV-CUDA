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
#include <cvcuda/OpSolarize.hpp>

namespace cvcudapy {

namespace {
// Thin Python-facing wrappers; the create/guard/submit body lives in UnaryElementwiseOp.hpp, with
// the threshold forwarded as the trailing submit() parameter.
Tensor SolarizeInto(Tensor &output, Tensor &input, double threshold, std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::Solarize>(output, input, pstream, threshold);
}

Tensor Solarize(Tensor &input, double threshold, std::optional<Stream> pstream)
{
    return UnaryElementwiseTensor<cvcuda::Solarize>(input, pstream, threshold);
}

ImageBatchVarShape SolarizeVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, double threshold,
                                        std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::Solarize>(output, input, pstream, threshold);
}

ImageBatchVarShape SolarizeVarShape(ImageBatchVarShape &input, double threshold, std::optional<Stream> pstream)
{
    return UnaryElementwiseVarShape<cvcuda::Solarize>(input, pstream, threshold);
}

} // namespace

void ExportOpSolarize(py::module &m)
{
    using namespace pybind11::literals;

    m.def("solarize", NvtxTrace("cvcuda.solarize", &Solarize), "src"_a, "threshold"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Solarize operation on the given cuda stream.

        Inverts every pixel at or above ``threshold``: ``out = (in >= threshold) ? (bound - in) : in``,
        where ``bound`` is the data type maximum (255 for uint8, 65535 for uint16, 1.0 for float32).
        Mirrors torchvision.transforms.v2.functional.solarize.

        See also:
            Refer to the CV-CUDA C API reference for the Solarize operator for more details and usage
            examples.

        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            threshold (float): Pixels with value >= threshold are inverted (in the pixel value domain).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same shape, dtype, and layout as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("solarize_into", NvtxTrace("cvcuda.solarize_into", &SolarizeInto), "dst"_a, "src"_a, "threshold"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Solarize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Solarize operator for more details and usage
            examples.

        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            threshold (float): Pixels with value >= threshold are inverted (in the pixel value domain).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("solarize", NvtxTrace("cvcuda.solarize", &SolarizeVarShape), "src"_a, "threshold"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Solarize operation on the given cuda stream.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            threshold (float): Pixels with value >= threshold are inverted (in the pixel value domain).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same formats and sizes as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("solarize_into", NvtxTrace("cvcuda.solarize_into", &SolarizeVarShapeInto), "dst"_a, "src"_a, "threshold"_a,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        Executes the Solarize operation on the given cuda stream.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            threshold (float): Pixels with value >= threshold are inverted (in the pixel value domain).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
}

} // namespace cvcudapy

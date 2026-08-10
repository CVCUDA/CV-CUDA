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
#include <cvcuda/OpAdjustHue.hpp>

namespace cvcudapy {

namespace {
// Thin Python-facing wrappers; the create/guard/submit body lives in UnaryElementwiseOp.hpp, with
// the hue factor forwarded as the trailing submit() parameter.
Tensor AdjustHueInto(Tensor &output, Tensor &input, double hue, std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::AdjustHue>(output, input, pstream, hue);
}

Tensor AdjustHue(Tensor &input, double hue, std::optional<Stream> pstream)
{
    return UnaryElementwiseTensor<cvcuda::AdjustHue>(input, pstream, hue);
}

ImageBatchVarShape AdjustHueVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, double hue,
                                         std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::AdjustHue>(output, input, pstream, hue);
}

ImageBatchVarShape AdjustHueVarShape(ImageBatchVarShape &input, double hue, std::optional<Stream> pstream)
{
    return UnaryElementwiseVarShape<cvcuda::AdjustHue>(input, pstream, hue);
}

} // namespace

void ExportOpAdjustHue(py::module &m)
{
    using namespace pybind11::literals;

    m.def("adjust_hue", NvtxTrace("cvcuda.adjust_hue", &AdjustHue), "src"_a, "hue"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Hue operation on the given cuda stream.

        Rotates the hue of each RGB image in HSV space: the image is converted to HSV, the hue channel
        is shifted by ``hue`` (H normalized to [0, 1)), and converted back to RGB. ``hue`` = 0 leaves
        the image unchanged; +/-0.5 is a full 180-degree hue rotation. Single-channel images are
        returned unchanged. Mirrors torchvision.transforms.v2.functional.adjust_hue.

        See also:
            Refer to the CV-CUDA C API reference for the Adjust Hue operator for more details and
            usage examples.

        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            hue (float): Hue-rotation factor in [-0.5, 0.5], applied to all images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same shape, dtype, and layout as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_hue_into", NvtxTrace("cvcuda.adjust_hue_into", &AdjustHueInto), "dst"_a, "src"_a, "hue"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Hue operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Adjust Hue operator for more details and
            usage examples.

        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            hue (float): Hue-rotation factor in [-0.5, 0.5], applied to all images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_hue", NvtxTrace("cvcuda.adjust_hue", &AdjustHueVarShape), "src"_a, "hue"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Hue operation on the given cuda stream.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            hue (float): Hue-rotation factor in [-0.5, 0.5], applied to all images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same formats and sizes as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_hue_into", NvtxTrace("cvcuda.adjust_hue_into", &AdjustHueVarShapeInto), "dst"_a, "src"_a, "hue"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Hue operation on the given cuda stream.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            hue (float): Hue-rotation factor in [-0.5, 0.5], applied to all images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
}

} // namespace cvcudapy

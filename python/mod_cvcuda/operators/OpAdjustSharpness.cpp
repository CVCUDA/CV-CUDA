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
#include <cvcuda/OpAdjustSharpness.hpp>

namespace cvcudapy {

namespace {
// Thin Python-facing wrappers; the create/guard/submit body lives in UnaryElementwiseOp.hpp, with
// the sharpness factor forwarded as the trailing submit() parameter.
Tensor AdjustSharpnessInto(Tensor &output, Tensor &input, float sharpnessFactor, std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::AdjustSharpness>(output, input, pstream, sharpnessFactor);
}

Tensor AdjustSharpness(Tensor &input, float sharpnessFactor, std::optional<Stream> pstream)
{
    return UnaryElementwiseTensor<cvcuda::AdjustSharpness>(input, pstream, sharpnessFactor);
}

ImageBatchVarShape AdjustSharpnessVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                               float sharpnessFactor, std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::AdjustSharpness>(output, input, pstream, sharpnessFactor);
}

ImageBatchVarShape AdjustSharpnessVarShape(ImageBatchVarShape &input, float sharpnessFactor,
                                           std::optional<Stream> pstream)
{
    return UnaryElementwiseVarShape<cvcuda::AdjustSharpness>(input, pstream, sharpnessFactor);
}

} // namespace

void ExportOpAdjustSharpness(py::module &m)
{
    using namespace pybind11::literals;

    m.def("adjust_sharpness", NvtxTrace("cvcuda.adjust_sharpness", &AdjustSharpness), "src"_a, "sharpness_factor"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Sharpness operation on the given cuda stream.

        Blends each image with a 3x3-smoothed copy of itself over the image interior:
        ``out = sharpness_factor * in + (1 - sharpness_factor) * blur``. ``sharpness_factor`` of 1.0
        leaves the image unchanged, 0.0 yields the fully-smoothed image, and values above 1.0
        sharpen. The 1-pixel border is copied unchanged, and images with height or width below 3 are
        returned unchanged. Mirrors torchvision.transforms.v2.functional.adjust_sharpness.

        See also:
            Refer to the CV-CUDA C API reference for the Adjust Sharpness operator for more details
            and usage examples.

        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            sharpness_factor (float): Non-negative blend weight applied to the original image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same shape, dtype, and layout as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_sharpness_into", NvtxTrace("cvcuda.adjust_sharpness_into", &AdjustSharpnessInto), "dst"_a, "src"_a,
          "sharpness_factor"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Sharpness operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Adjust Sharpness operator for more details
            and usage examples.

        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            sharpness_factor (float): Non-negative blend weight applied to the original image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_sharpness", NvtxTrace("cvcuda.adjust_sharpness", &AdjustSharpnessVarShape), "src"_a,
          "sharpness_factor"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Sharpness operation on the given cuda stream.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            sharpness_factor (float): Non-negative blend weight applied to the original image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same formats and sizes as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_sharpness_into", NvtxTrace("cvcuda.adjust_sharpness_into", &AdjustSharpnessVarShapeInto), "dst"_a,
          "src"_a, "sharpness_factor"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Sharpness operation on the given cuda stream.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            sharpness_factor (float): Non-negative blend weight applied to the original image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
}

} // namespace cvcudapy

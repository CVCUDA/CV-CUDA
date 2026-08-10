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
#include <cvcuda/OpAdjustSaturation.hpp>

namespace cvcudapy {

namespace {
// Thin Python-facing wrappers; the create/guard/submit body lives in UnaryElementwiseOp.hpp, with
// the saturation factor forwarded as the trailing submit() parameter.
Tensor AdjustSaturationInto(Tensor &output, Tensor &input, double saturation, std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::AdjustSaturation>(output, input, pstream, saturation);
}

Tensor AdjustSaturation(Tensor &input, double saturation, std::optional<Stream> pstream)
{
    return UnaryElementwiseTensor<cvcuda::AdjustSaturation>(input, pstream, saturation);
}

ImageBatchVarShape AdjustSaturationVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                                double saturation, std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::AdjustSaturation>(output, input, pstream, saturation);
}

ImageBatchVarShape AdjustSaturationVarShape(ImageBatchVarShape &input, double saturation, std::optional<Stream> pstream)
{
    return UnaryElementwiseVarShape<cvcuda::AdjustSaturation>(input, pstream, saturation);
}

} // namespace

void ExportOpAdjustSaturation(py::module &m)
{
    using namespace pybind11::literals;

    m.def("adjust_saturation", NvtxTrace("cvcuda.adjust_saturation", &AdjustSaturation), "src"_a, "saturation"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Saturation operation on the given cuda stream.

        Blends each RGB image toward its grayscale by a scalar factor:
        ``out = saturation * image + (1 - saturation) * grayscale``, where
        ``grayscale = 0.2989*R + 0.587*G + 0.114*B``. ``saturation`` = 1 leaves the image unchanged,
        0 yields grayscale, and values > 1 over-saturate. Single-channel images are returned
        unchanged. Mirrors torchvision.transforms.v2.functional.adjust_saturation.

        See also:
            Refer to the CV-CUDA C API reference for the Adjust Saturation operator for more details
            and usage examples.

        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            saturation (float): Saturation factor applied to all images (must be >= 0).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same shape, dtype, and layout as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_saturation_into", NvtxTrace("cvcuda.adjust_saturation_into", &AdjustSaturationInto), "dst"_a, "src"_a,
          "saturation"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Saturation operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Adjust Saturation operator for more details
            and usage examples.

        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            saturation (float): Saturation factor applied to all images (must be >= 0).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_saturation", NvtxTrace("cvcuda.adjust_saturation", &AdjustSaturationVarShape), "src"_a,
          "saturation"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Saturation operation on the given cuda stream.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            saturation (float): Saturation factor applied to all images (must be >= 0).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same formats and sizes as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_saturation_into", NvtxTrace("cvcuda.adjust_saturation_into", &AdjustSaturationVarShapeInto), "dst"_a,
          "src"_a, "saturation"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adjust Saturation operation on the given cuda stream.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            saturation (float): Saturation factor applied to all images (must be >= 0).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
}

} // namespace cvcudapy

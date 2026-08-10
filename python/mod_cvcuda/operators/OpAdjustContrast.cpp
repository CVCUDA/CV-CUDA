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
#include <cvcuda/OpAdjustContrast.hpp>

namespace cvcudapy {

namespace {
// Thin Python-facing wrappers; the create/guard/submit body lives in UnaryElementwiseOp.hpp, with
// the contrast factor forwarded as the trailing submit() parameter.
Tensor AdjustContrastInto(Tensor &output, Tensor &input, double contrast_factor, std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::AdjustContrast>(output, input, pstream, contrast_factor);
}

Tensor AdjustContrast(Tensor &input, double contrast_factor, std::optional<Stream> pstream)
{
    return UnaryElementwiseTensor<cvcuda::AdjustContrast>(input, pstream, contrast_factor);
}

ImageBatchVarShape AdjustContrastVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                              double contrast_factor, std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::AdjustContrast>(output, input, pstream, contrast_factor);
}

ImageBatchVarShape AdjustContrastVarShape(ImageBatchVarShape &input, double contrast_factor,
                                          std::optional<Stream> pstream)
{
    return UnaryElementwiseVarShape<cvcuda::AdjustContrast>(input, pstream, contrast_factor);
}

} // namespace

void ExportOpAdjustContrast(py::module &m)
{
    using namespace pybind11::literals;

    m.def("adjust_contrast", NvtxTrace("cvcuda.adjust_contrast", &AdjustContrast), "src"_a, "contrast_factor"_a,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        Executes the AdjustContrast operation on the given cuda stream.

        Blends each image toward its grayscale mean by a scalar factor:
        ``out = clamp(contrast_factor * in + (1 - contrast_factor) * mean, 0, bound)``, where ``mean``
        is the per-image grayscale mean (BT.601 luma ``0.2989 R + 0.587 G + 0.114 B``) and ``bound``
        is 1.0 for float32 and 255 for uint8. Mirrors
        torchvision.transforms.v2.functional.adjust_contrast.

        See also:
            Refer to the CV-CUDA C API reference for the AdjustContrast operator for more details and
            usage examples.

        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images (1 or 3 channels).
            contrast_factor (float): Non-negative contrast multiplier (0 = flat gray, 1 = unchanged).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same shape, dtype, and layout as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_contrast_into", NvtxTrace("cvcuda.adjust_contrast_into", &AdjustContrastInto), "dst"_a, "src"_a,
          "contrast_factor"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the AdjustContrast operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the AdjustContrast operator for more details and
            usage examples.

        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images (1 or 3 channels).
            contrast_factor (float): Non-negative contrast multiplier (0 = flat gray, 1 = unchanged).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_contrast", NvtxTrace("cvcuda.adjust_contrast", &AdjustContrastVarShape), "src"_a, "contrast_factor"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the AdjustContrast operation on the given cuda stream.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one- or three-component images.
                Three-component inputs are interpreted as RGB, with RGB and BGR storage swizzles honored.
            contrast_factor (float): Non-negative contrast multiplier (0 = flat gray, 1 = unchanged).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same formats and sizes as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_contrast_into", NvtxTrace("cvcuda.adjust_contrast_into", &AdjustContrastVarShapeInto), "dst"_a,
          "src"_a, "contrast_factor"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the AdjustContrast operation on the given cuda stream.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one- or three-component images.
                Three-component inputs are interpreted as RGB, with RGB and BGR storage swizzles honored.
            contrast_factor (float): Non-negative contrast multiplier (0 = flat gray, 1 = unchanged).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
}

} // namespace cvcudapy

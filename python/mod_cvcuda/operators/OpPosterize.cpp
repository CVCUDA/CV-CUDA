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
#include <cvcuda/OpPosterize.hpp>

namespace cvcudapy {

namespace {
// Thin Python-facing wrappers; the create/guard/submit body lives in UnaryElementwiseOp.hpp, with
// the bit count forwarded as the trailing submit() parameter.
Tensor PosterizeInto(Tensor &output, Tensor &input, int32_t bits, std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::Posterize>(output, input, pstream, bits);
}

Tensor Posterize(Tensor &input, int32_t bits, std::optional<Stream> pstream)
{
    return UnaryElementwiseTensor<cvcuda::Posterize>(input, pstream, bits);
}

ImageBatchVarShape PosterizeVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, int32_t bits,
                                         std::optional<Stream> pstream)
{
    return UnaryElementwiseInto<cvcuda::Posterize>(output, input, pstream, bits);
}

ImageBatchVarShape PosterizeVarShape(ImageBatchVarShape &input, int32_t bits, std::optional<Stream> pstream)
{
    return UnaryElementwiseVarShape<cvcuda::Posterize>(input, pstream, bits);
}

} // namespace

void ExportOpPosterize(py::module &m)
{
    using namespace pybind11::literals;

    m.def("posterize", NvtxTrace("cvcuda.posterize", &Posterize), "src"_a, "bits"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Posterize operation on the given cuda stream.

        Keeps the ``bits`` most-significant bits of every channel value and zeros the rest:
        ``out = in & ~((1 << (W - bits)) - 1)``, where ``W`` is the data type bit width (8 for uint8,
        16 for uint16). Mirrors torchvision.transforms.v2.functional.posterize / PIL ImageOps.posterize.

        See also:
            Refer to the CV-CUDA C API reference for the Posterize operator for more details and usage
            examples.

        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            bits (int): Number of most-significant bits to keep per channel; must be in [0, W] where W
                is the data type bit width (8 for uint8, 16 for uint16). bits == W leaves the image
                unchanged; bits == 0 zeros the image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same shape, dtype, and layout as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("posterize_into", NvtxTrace("cvcuda.posterize_into", &PosterizeInto), "dst"_a, "src"_a, "bits"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Posterize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Posterize operator for more details and usage
            examples.

        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            bits (int): Number of most-significant bits to keep per channel (see posterize).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("posterize", NvtxTrace("cvcuda.posterize", &PosterizeVarShape), "src"_a, "bits"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Posterize operation on the given cuda stream.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            bits (int): Number of most-significant bits to keep per channel (see posterize).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same formats and sizes as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("posterize_into", NvtxTrace("cvcuda.posterize_into", &PosterizeVarShapeInto), "dst"_a, "src"_a, "bits"_a,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        Executes the Posterize operation on the given cuda stream.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            bits (int): Number of most-significant bits to keep per channel (see posterize).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
}

} // namespace cvcudapy

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

#include "NativeBinding.hpp"
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpAdjustContrast.hpp>
#include <nvcv/RoundMode.h>

namespace cvcudapy {

namespace {
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

template<typename Container>
Container &AdjustContrastIntoImpl(Container &output, Container &input, double contrastFactor, NVCVRoundMode round,
                                  std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto op = CreateOperator<cvcuda::AdjustContrast>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    // The operator owns scratch read by the kernels, so keep and serialize it through completion.
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    guard.run([&op, &stream, &input, &output, contrastFactor, round]()
              { op->submit(stream.handle(), input.impl(), output.impl(), contrastFactor, round); });

    return output;
}

Tensor &AdjustContrastInto(Tensor &output, Tensor &input, double contrast_factor, NVCVRoundMode round,
                           std::optional<Stream *> pstream)
{
    return AdjustContrastIntoImpl(output, input, contrast_factor, round, pstream);
}

std::shared_ptr<Tensor> AdjustContrast(Tensor &input, double contrast_factor, NVCVRoundMode round,
                                       std::optional<Stream *> pstream)
{
    auto output = native::CreateTensor(input.impl().shape(), input.dtype());

    AdjustContrastInto(*output, input, contrast_factor, round, pstream);
    return output;
}

ImageBatchVarShape &AdjustContrastVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                               double contrast_factor, NVCVRoundMode round,
                                               std::optional<Stream *> pstream)
{
    return AdjustContrastIntoImpl(output, input, contrast_factor, round, pstream);
}

std::shared_ptr<ImageBatchVarShape> AdjustContrastVarShape(ImageBatchVarShape &input, double contrast_factor,
                                                           NVCVRoundMode round, std::optional<Stream *> pstream)
{
    auto output = CreateSameShapeImageBatch(input);

    AdjustContrastVarShapeInto(*output, input, contrast_factor, round, pstream);
    return output;
}

} // namespace

void ExportOpAdjustContrast(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("adjust_contrast", NvtxTrace("cvcuda.adjust_contrast", &AdjustContrast), "src"_a, "contrast_factor"_a,
          py::kw_only(), "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr,
          R"pbdoc(

        Executes the AdjustContrast operation on the given cuda stream.

        Blends each image toward its grayscale mean by a scalar factor:
        ``out = clamp(contrast_factor * in + (1 - contrast_factor) * mean, 0, bound)``, where ``mean``
        is the per-image grayscale mean (BT.601 luma ``0.2989 R + 0.587 G + 0.114 B``) and ``bound``
        is 1.0 for float images (float32/float16) and 255 for uint8. Mirrors
        torchvision.transforms.v2.functional.adjust_contrast.

        See also:
            Refer to the CV-CUDA C API reference for the AdjustContrast operator for more details and
            usage examples.

        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images (1 or 3 channels).
            contrast_factor (float): Non-negative contrast multiplier (0 = flat gray, 1 = unchanged).
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST; use cvcuda.Round.TRUNCATE to drop the fractional part, which is
                what torchvision's adjust_contrast does. Has no effect on floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same shape, dtype, and layout as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_contrast_into", NvtxTrace("cvcuda.adjust_contrast_into", &AdjustContrastInto), "dst"_a, "src"_a,
          "contrast_factor"_a, py::kw_only(), "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(

        Executes the AdjustContrast operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the AdjustContrast operator for more details and
            usage examples.

        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images (1 or 3 channels).
            contrast_factor (float): Non-negative contrast multiplier (0 = flat gray, 1 = unchanged).
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST; use cvcuda.Round.TRUNCATE to drop the fractional part, which is
                what torchvision's adjust_contrast does. Has no effect on floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_contrast", NvtxTrace("cvcuda.adjust_contrast", &AdjustContrastVarShape), "src"_a, "contrast_factor"_a,
          py::kw_only(), "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr, R"pbdoc(

        Executes the AdjustContrast operation on the given cuda stream.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one- or three-component images.
                Three-component inputs are interpreted as RGB, with RGB and BGR storage swizzles honored.
            contrast_factor (float): Non-negative contrast multiplier (0 = flat gray, 1 = unchanged).
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST; use cvcuda.Round.TRUNCATE to drop the fractional part, which is
                what torchvision's adjust_contrast does. Has no effect on floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same formats and sizes as src).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("adjust_contrast_into", NvtxTrace("cvcuda.adjust_contrast_into", &AdjustContrastVarShapeInto), "dst"_a,
          "src"_a, "contrast_factor"_a, py::kw_only(), "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr,
          py::return_value_policy::reference,
          R"pbdoc(

        Executes the AdjustContrast operation on the given cuda stream.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one- or three-component images.
                Three-component inputs are interpreted as RGB, with RGB and BGR storage swizzles honored.
            contrast_factor (float): Non-negative contrast multiplier (0 = flat gray, 1 = unchanged).
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST; use cvcuda.Round.TRUNCATE to drop the fractional part, which is
                what torchvision's adjust_contrast does. Has no effect on floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
}

} // namespace cvcudapy

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
#include <common/String.hpp>
#include <cvcuda/OpRotate.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor RotateInto(Tensor &output, Tensor &input, double angleDeg, const std::tuple<double, double> &shift,
                  NVCVInterpolationType interpolation, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto rotate = CreateOperator<cvcuda::Rotate>(0);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*rotate});

    double2 shiftArg{std::get<0>(shift), std::get<1>(shift)};

    guard.run([&rotate, &pstream, &input, &output, &angleDeg, &shiftArg, &interpolation]()
              { rotate->submit(pstream->cudaHandle(), input, output, angleDeg, shiftArg, interpolation); });

    return output;
}

Tensor Rotate(Tensor &input, double angleDeg, const std::tuple<double, double> &shift,
              const NVCVInterpolationType interpolation, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return RotateInto(output, input, angleDeg, shift, interpolation, pstream);
}

ImageBatchVarShape VarShapeRotateInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &angleDeg,
                                      Tensor &shift, NVCVInterpolationType interpolation, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto rotate = CreateOperator<cvcuda::Rotate>(input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, angleDeg, shift});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*rotate});

    guard.run([&rotate, &pstream, &input, &output, &angleDeg, &shift, &interpolation]()
              { rotate->submit(pstream->cudaHandle(), input, output, angleDeg, shift, interpolation); });

    return output;
}

ImageBatchVarShape VarShapeRotate(ImageBatchVarShape &input, Tensor &angleDeg, Tensor &shift,
                                  NVCVInterpolationType interpolation, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = CreateSameShapeImageBatch(input);

    return VarShapeRotateInto(output, input, angleDeg, shift, interpolation, pstream);
}

} // namespace

void ExportOpRotate(py::module &m)
{
    using namespace pybind11::literals;

    m.def("rotate", NvtxTrace("cvcuda.rotate", &Rotate), "src"_a, "angle_deg"_a, "shift"_a, "interpolation"_a,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the Rotate operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            angle_deg (float): Angle used for rotation in degrees.
            shift (Tuple[float, float]): Value of shift in {x, y} directions to move the center at the same coord after rotation.
            interpolation (cvcuda.Interp): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("rotate_into", NvtxTrace("cvcuda.rotate_into", &RotateInto), "dst"_a, "src"_a, "angle_deg"_a, "shift"_a,
          "interpolation"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Rotate operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            angle_deg (float): Angle used for rotation in degrees.
            shift (Tuple[float, float]): Value of shift in {x, y} directions to move the center at the same coord after rotation.
            interpolation (cvcuda.Interp): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("rotate", NvtxTrace("cvcuda.rotate", &VarShapeRotate), "src"_a, "angle_deg"_a, "shift"_a, "interpolation"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Rotate operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            angle_deg (cvcuda.Tensor): Rotation angle in degrees, specified per image.
            shift (cvcuda.Tensor): Shift in {x, y} directions, specified per image.
            interpolation (cvcuda.Interp): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("rotate_into", NvtxTrace("cvcuda.rotate_into", &VarShapeRotateInto), "dst"_a, "src"_a, "angle_deg"_a,
          "shift"_a, "interpolation"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Rotate operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            angle_deg (cvcuda.Tensor): Rotation angle in degrees, specified per image.
            shift (cvcuda.Tensor): Shift in {x, y} directions, specified per image.
            interpolation (cvcuda.Interp): Interpolation type used for transform.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

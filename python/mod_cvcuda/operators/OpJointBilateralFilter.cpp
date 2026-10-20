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
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/Binding.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpJointBilateralFilter.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

namespace cvcudapy {

namespace {
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

Tensor &JointBilateralFilterInto(Tensor &output, Tensor &input, Tensor &inputColor, int diameter, float sigmaColor,
                                 float sigmaSpace, NVCVBorderType borderMode, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto joint_bilateral_filter = CreateOperator<cvcuda::JointBilateralFilter>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, inputColor});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*joint_bilateral_filter});

    guard.run(
        [&joint_bilateral_filter, &stream, &input, &inputColor, &output, &diameter, &sigmaColor, &sigmaSpace,
         &borderMode]()
        {
            joint_bilateral_filter->submit(stream.handle(), input.impl(), inputColor.impl(), output.impl(), diameter,
                                           sigmaColor, sigmaSpace, borderMode);
        });

    return output;
}

std::shared_ptr<Tensor> JointBilateralFilter(Tensor &input, Tensor &inputColor, int diameter, float sigmaColor,
                                             float sigmaSpace, NVCVBorderType borderMode,
                                             std::optional<Stream *> pstream)
{
    auto output = native::CreateTensor(input.impl().shape(), input.dtype());

    JointBilateralFilterInto(*output, input, inputColor, diameter, sigmaColor, sigmaSpace, borderMode, pstream);
    return output;
}

ImageBatchVarShape &VarShapeJointBilateralFilterInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                                     ImageBatchVarShape &inputColor, Tensor &diameter,
                                                     Tensor &sigmaColor, Tensor &sigmaSpace, NVCVBorderType borderMode,
                                                     std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto joint_bilateral_filter = CreateOperator<cvcuda::JointBilateralFilter>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, inputColor, diameter, sigmaColor, sigmaSpace});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*joint_bilateral_filter});

    guard.run(
        [&joint_bilateral_filter, &stream, &input, &inputColor, &output, &diameter, &sigmaColor, &sigmaSpace,
         &borderMode]()
        {
            joint_bilateral_filter->submit(stream.handle(), input.impl(), inputColor.impl(), output.impl(),
                                           diameter.impl(), sigmaColor.impl(), sigmaSpace.impl(), borderMode);
        });

    return output;
}

std::shared_ptr<ImageBatchVarShape> VarShapeJointBilateralFilter(ImageBatchVarShape &input,
                                                                 ImageBatchVarShape &inputColor, Tensor &diameter,
                                                                 Tensor &sigmaColor, Tensor &sigmaSpace,
                                                                 NVCVBorderType          borderMode,
                                                                 std::optional<Stream *> pstream)
{
    auto output = CreateSameShapeImageBatch(input);

    VarShapeJointBilateralFilterInto(*output, input, inputColor, diameter, sigmaColor, sigmaSpace, borderMode, pstream);
    return output;
}

} // namespace

void ExportOpJointBilateralFilter(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("joint_bilateral_filter", NvtxTrace("cvcuda.joint_bilateral_filter", &JointBilateralFilter), "src"_a,
          "srcColor"_a, "diameter"_a, "sigma_color"_a, "sigma_space"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the Joint Bilateral Filter operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            srcColor (cvcuda.Tensor): Input tensor for color distance.
            diameter (int): Bilateral filter diameter.
            sigma_color (float): Gaussian exponent for color difference.
            sigma_space (float): Gaussian exponent for position difference.
            border (cvcuda.Border, optional): Border mode for input tensor.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("joint_bilateral_filter_into", NvtxTrace("cvcuda.joint_bilateral_filter_into", &JointBilateralFilterInto),
          "dst"_a, "src"_a, "srcColor"_a, "diameter"_a, "sigma_color"_a, "sigma_space"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(
        Executes the Joint Bilateral Filter operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            srcColor (cvcuda.Tensor): Input tensor for color distance.
            diameter (int): Bilateral filter diameter.
            sigma_color (float): Gaussian exponent for color difference.
            sigma_space (float): Gaussian exponent for position difference.
            border (cvcuda.Border, optional): Border mode for input tensor.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("joint_bilateral_filter", NvtxTrace("cvcuda.joint_bilateral_filter", &VarShapeJointBilateralFilter), "src"_a,
          "srcColor"_a, "diameter"_a, "sigma_color"_a, "sigma_space"_a, py::kw_only(),
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr,
          R"pbdoc(
        Executes the Joint Bilateral operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            srcColor (cvcuda.ImageBatchVarShape): Input images for color distance.
            diameter (cvcuda.Tensor): Bilateral filter diameter per image.
            sigma_color (cvcuda.Tensor): Gaussian exponent for color difference per image.
            sigma_space (cvcuda.Tensor): Gaussian exponent for position difference per image.
            border (cvcuda.Border, optional): Border mode for input tensor.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("joint_bilateral_filter_into",
          NvtxTrace("cvcuda.joint_bilateral_filter_into", &VarShapeJointBilateralFilterInto), "dst"_a, "src"_a,
          "srcColor"_a, "diameter"_a, "sigma_color"_a, "sigma_space"_a, py::kw_only(),
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Executes the Joint Bilateral operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            diameter (cvcuda.Tensor): Bilateral filter diameter per image.
            sigma_color (cvcuda.Tensor): Gaussian exponent for color difference per image.
            sigma_space (cvcuda.Tensor): Gaussian exponent for position difference per image.
            border (cvcuda.Border, optional): Border mode for input tensor.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

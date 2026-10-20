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
#include <cvcuda/OpBilateralFilter.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

namespace cvcudapy {

namespace {
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

Tensor &BilateralFilterInto(Tensor &output, Tensor &input, int diameter, float sigmaColor, float sigmaSpace,
                            NVCVBorderType borderMode, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto bilateral_filter = CreateOperator<cvcuda::BilateralFilter>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*bilateral_filter});

    guard.run(
        [&bilateral_filter, &stream, &input, &output, &diameter, &sigmaColor, &sigmaSpace, &borderMode]()
        {
            bilateral_filter->submit(stream.handle(), input.impl(), output.impl(), diameter, sigmaColor, sigmaSpace,
                                     borderMode);
        });

    return output;
}

std::shared_ptr<Tensor> BilateralFilter(Tensor &input, int diameter, float sigmaColor, float sigmaSpace,
                                        NVCVBorderType borderMode, std::optional<Stream *> pstream)
{
    auto output = native::CreateTensor(input.impl().shape(), input.dtype());

    BilateralFilterInto(*output, input, diameter, sigmaColor, sigmaSpace, borderMode, pstream);
    return output;
}

ImageBatchVarShape &VarShapeBilateralFilterInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &diameter,
                                                Tensor &sigmaColor, Tensor &sigmaSpace, NVCVBorderType borderMode,
                                                std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto bilateral_filter = CreateOperator<cvcuda::BilateralFilter>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, diameter, sigmaColor, sigmaSpace});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*bilateral_filter});

    guard.run(
        [&bilateral_filter, &stream, &input, &output, &diameter, &sigmaColor, &sigmaSpace, &borderMode]()
        {
            bilateral_filter->submit(stream.handle(), input.impl(), output.impl(), diameter.impl(), sigmaColor.impl(),
                                     sigmaSpace.impl(), borderMode);
        });

    return output;
}

std::shared_ptr<ImageBatchVarShape> VarShapeBilateralFilter(ImageBatchVarShape &input, Tensor &diameter,
                                                            Tensor &sigmaColor, Tensor &sigmaSpace,
                                                            NVCVBorderType borderMode, std::optional<Stream *> pstream)
{
    auto output = CreateSameShapeImageBatch(input);

    VarShapeBilateralFilterInto(*output, input, diameter, sigmaColor, sigmaSpace, borderMode, pstream);
    return output;
}

} // namespace

void ExportOpBilateralFilter(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("bilateral_filter", NvtxTrace("cvcuda.bilateral_filter", &BilateralFilter), "src"_a, "diameter"_a,
          "sigma_color"_a, "sigma_space"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Bilateral Filter operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            diameter (int): Bilateral filter diameter.
            sigma_color (float): Gaussian exponent for color difference.
            sigma_space (float): Gaussian exponent for position difference.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("bilateral_filter_into", NvtxTrace("cvcuda.bilateral_filter_into", &BilateralFilterInto), "dst"_a, "src"_a,
          "diameter"_a, "sigma_color"_a, "sigma_space"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
          py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Executes the Bilateral Filter operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            diameter (int): Bilateral filter diameter.
            sigma_color (float): Gaussian exponent for color difference.
            sigma_space (float): Gaussian exponent for position difference.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("bilateral_filter", NvtxTrace("cvcuda.bilateral_filter", &VarShapeBilateralFilter), "src"_a, "diameter"_a,
          "sigma_color"_a, "sigma_space"_a, py::kw_only(), "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
          "stream"_a = nullptr, R"pbdoc(
        Executes the Bilateral Filter operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input tensor containing one or more images.
            diameter (cvcuda.Tensor): Bilateral filter diameters in each image.
            sigma_color (cvcuda.Tensor): Gaussian exponents for color difference in each image.
            sigma_space (cvcuda.Tensor): Gaussian exponents for position difference in each image.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("bilateral_filter_into", NvtxTrace("cvcuda.bilateral_filter_into", &VarShapeBilateralFilterInto), "dst"_a,
          "src"_a, "diameter"_a, "sigma_color"_a, "sigma_space"_a, py::kw_only(),
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Executes the Bilateral Filter operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            diameter (cvcuda.Tensor): Bilateral filter diameters in each image.
            sigma_color (cvcuda.Tensor): Gaussian exponents for color difference in each image.
            sigma_space (cvcuda.Tensor): Gaussian exponents for position difference in each image.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

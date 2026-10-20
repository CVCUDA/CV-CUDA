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
#include <cvcuda/OpGaussian.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

namespace cvcudapy {

namespace {
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

Tensor &GaussianInto(Tensor &output, Tensor &input, const std::tuple<int, int> &kernel_size,
                     const std::tuple<double, double> &sigma, NVCVBorderType border, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    double2 sigmaArg{std::get<0>(sigma), std::get<1>(sigma)};

    nvcv::Size2D kernelSizeArg{std::get<0>(kernel_size), std::get<1>(kernel_size)};

    auto gaussian = CreateOperator<cvcuda::Gaussian>(kernelSizeArg, 0);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*gaussian});

    guard.run([&gaussian, &stream, &input, &output, &kernelSizeArg, &sigmaArg, &border]()
              { gaussian->submit(stream.handle(), input.impl(), output.impl(), kernelSizeArg, sigmaArg, border); });

    return output;
}

std::shared_ptr<Tensor> Gaussian(Tensor &input, const std::tuple<int, int> &kernel_size,
                                 const std::tuple<double, double> &sigma, NVCVBorderType border,
                                 std::optional<Stream *> pstream)
{
    auto output = native::CreateTensor(input.impl().shape(), input.dtype());

    GaussianInto(*output, input, kernel_size, sigma, border, pstream);
    return output;
}

ImageBatchVarShape &VarShapeGaussianInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                         const std::tuple<int, int> &max_kernel_size, Tensor &ksize, Tensor &sigma,
                                         NVCVBorderType border, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    nvcv::Size2D maxKernelSizeArg{std::get<0>(max_kernel_size), std::get<1>(max_kernel_size)};

    auto gaussian = CreateOperator<cvcuda::Gaussian>(maxKernelSizeArg, input.capacity());

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, ksize, sigma});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*gaussian});

    guard.run([&gaussian, &stream, &input, &output, &ksize, &sigma, &border]()
              { gaussian->submit(stream.handle(), input.impl(), output.impl(), ksize.impl(), sigma.impl(), border); });

    return output;
}

std::shared_ptr<ImageBatchVarShape> VarShapeGaussian(ImageBatchVarShape         &input,
                                                     const std::tuple<int, int> &max_kernel_size, Tensor &ksize,
                                                     Tensor &sigma, NVCVBorderType border,
                                                     std::optional<Stream *> pstream)
{
    auto output = CreateSameShapeImageBatch(input);

    VarShapeGaussianInto(*output, input, max_kernel_size, ksize, sigma, border, pstream);
    return output;
}

} // namespace

void ExportOpGaussian(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("gaussian", NvtxTrace("cvcuda.gaussian", &Gaussian), "src"_a, "kernel_size"_a, "sigma"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Gaussian operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            kernel_size (Tuple[int, int]): Kernel width, height.
            sigma (Tuple[double, double]): Gaussian kernel standard deviation in X,Y directions.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("gaussian_into", NvtxTrace("cvcuda.gaussian_into", &GaussianInto), "dst"_a, "src"_a, "kernel_size"_a,
          "sigma"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(
        Executes the Gaussian operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            kernel_size (Tuple[int, int]): Kernel width, height.
            sigma (Tuple[double, double]): Gaussian kernel standard deviation in X,Y directions.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("gaussian", NvtxTrace("cvcuda.gaussian", &VarShapeGaussian), "src"_a, "max_kernel_size"_a, "kernel_size"_a,
          "sigma"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Gaussian operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            kernel_size (cvcuda.Tensor): Kernel width, height.
            sigma (cvcuda.Tensor): Gaussian kernel standard deviation in X,Y directions.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("gaussian_into", NvtxTrace("cvcuda.gaussian_into", &VarShapeGaussianInto), "dst"_a, "src"_a,
          "max_kernel_size"_a, "kernel_size"_a, "sigma"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
          py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the Gaussian operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            kernel_size (cvcuda.Tensor): Kernel width, height.
            sigma (cvcuda.Tensor): Gaussian kernel standard deviation in X,Y directions.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

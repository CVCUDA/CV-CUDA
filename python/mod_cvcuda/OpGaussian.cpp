/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpGaussian.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor GaussianInto(Tensor &output, Tensor &input, const std::tuple<int, int> &kernel_size,
                    const std::tuple<double, double> &sigma, NVCVBorderType border, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    double2 sigmaArg{std::get<0>(sigma), std::get<1>(sigma)};

    nvcv::Size2D kernelSizeArg{std::get<0>(kernel_size), std::get<1>(kernel_size)};

    auto gaussian = CreateOperator<cvcuda::Gaussian>(kernelSizeArg, 0);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*gaussian});

    gaussian->submit(pstream->cudaHandle(), input, output, kernelSizeArg, sigmaArg, border);

    return output;
}

Tensor Gaussian(Tensor &input, const std::tuple<int, int> &kernel_size, const std::tuple<double, double> &sigma,
                NVCVBorderType border, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return GaussianInto(output, input, kernel_size, sigma, border, pstream);
}

ImageBatchVarShape VarShapeGaussianInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                        const std::tuple<int, int> &max_kernel_size, Tensor &ksize, Tensor &sigma,
                                        NVCVBorderType border, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    nvcv::Size2D maxKernelSizeArg{std::get<0>(max_kernel_size), std::get<1>(max_kernel_size)};

    auto gaussian = CreateOperator<cvcuda::Gaussian>(maxKernelSizeArg, input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, ksize, sigma});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*gaussian});

    gaussian->submit(pstream->cudaHandle(), input, output, ksize, sigma, border);

    return output;
}

ImageBatchVarShape VarShapeGaussian(ImageBatchVarShape &input, const std::tuple<int, int> &max_kernel_size,
                                    Tensor &ksize, Tensor &sigma, NVCVBorderType border, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        output.pushBack(Image::Create(input[i].size(), input[i].format()));
    }

    return VarShapeGaussianInto(output, input, max_kernel_size, ksize, sigma, border, pstream);
}

} // namespace

void ExportOpGaussian(py::module &m)
{
    using namespace pybind11::literals;
    py::options options;
    options.disable_function_signatures();

    m.def("gaussian", &Gaussian, "src"_a, "kernel_size"_a, "sigma"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.gaussian(src: nvcv.Tensor, kernel_size: Tuple[int, int], sigma: Tuple[double, double], border: border_mode: cvcuda.Border, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor
        Executes the Gaussian operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Gaussian operator
            for more details and usage examples.

        Args:
            src (nvcv.Tensor): Input tensor containing one or more images.
            kernel_size (Tuple[int, int]): Kernel width, height.
            sigma (Tuple[double, double]): Gaussian kernel standard deviation in X,Y directions.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("gaussian_into", &GaussianInto, "dst"_a, "src"_a, "kernel_size"_a, "sigma"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.gaussian_into(dst: nvcv.Tensor, src:  Tensor, kernel_size: Tuple[int, int], sigma: Tuple[double, double], border: border_mode: cvcuda.Border, stream: Optional[nvcv.cuda.Stream] = None)
        Executes the Gaussian operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Gaussian operator
            for more details and usage examples.

        Args:
            dst (nvcv.Tensor): Output tensor to store the result of the operation.
            src (nvcv.Tensor): Input tensor containing one or more images.
            kernel_size (Tuple[int, int]): Kernel width, height.
            sigma (Tuple[double, double]): Gaussian kernel standard deviation in X,Y directions.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("gaussian", &VarShapeGaussian, "src"_a, "max_kernel_size"_a, "kernel_size"_a, "sigma"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.gaussian(src: nvcv.ImageBatchVarShape, kernel_size: nvcv.Tensor, sigma: nvcv.Tensor, border: border_mode: cvcuda.Border, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.ImageBatchVarShape

        Executes the Gaussian operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Gaussian operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            kernel_size (nvcv.Tensor): Kernel width, height.
            sigma (nvcv.Tensor): Gaussian kernel standard deviation in X,Y directions.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("gaussian_into", &VarShapeGaussianInto, "dst"_a, "src"_a, "max_kernel_size"_a, "kernel_size"_a, "sigma"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.gaussian_into(dst: nvcv.ImageBatchVarShape, src: nvcv.ImageBatchVarShape, kernel_size: nvcv.Tensor, sigma: nvcv.Tensor, border: border_mode: cvcuda.Border, stream: Optional[nvcv.cuda.Stream] = None)

	Executes the Gaussian operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Gaussian operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            dst (nvcv.ImageBatchVarShape): Output image batch containing the result of the operation.
            kernel_size (nvcv.Tensor): Kernel width, height.
            sigma (nvcv.Tensor): Gaussian kernel standard deviation in X,Y directions.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpJointBilateralFilter.hpp>
#include <cvcuda/Types.h>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor JointBilateralFilterInto(Tensor &output, Tensor &input, Tensor &inputColor, int diameter, float sigmaColor,
                                float sigmaSpace, NVCVBorderType borderMode, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto joint_bilateral_filter = CreateOperator<cvcuda::JointBilateralFilter>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, inputColor});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*joint_bilateral_filter});

    joint_bilateral_filter->submit(pstream->cudaHandle(), input, inputColor, output, diameter, sigmaColor, sigmaSpace,
                                   borderMode);

    return output;
}

Tensor JointBilateralFilter(Tensor &input, Tensor &inputColor, int diameter, float sigmaColor, float sigmaSpace,
                            NVCVBorderType borderMode, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return JointBilateralFilterInto(output, input, inputColor, diameter, sigmaColor, sigmaSpace, borderMode, pstream);
}

ImageBatchVarShape VarShapeJointBilateralFilterInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                                    ImageBatchVarShape &inputColor, Tensor &diameter,
                                                    Tensor &sigmaColor, Tensor &sigmaSpace, NVCVBorderType borderMode,
                                                    std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto joint_bilateral_filter = CreateOperator<cvcuda::JointBilateralFilter>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, inputColor, diameter, sigmaColor, sigmaSpace});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*joint_bilateral_filter});

    joint_bilateral_filter->submit(pstream->cudaHandle(), input, inputColor, output, diameter, sigmaColor, sigmaSpace,
                                   borderMode);

    return output;
}

ImageBatchVarShape VarShapeJointBilateralFilter(ImageBatchVarShape &input, ImageBatchVarShape &inputColor,
                                                Tensor &diameter, Tensor &sigmaColor, Tensor &sigmaSpace,
                                                NVCVBorderType borderMode, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        nvcv::Size2D      size   = input[i].size();
        auto              image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return VarShapeJointBilateralFilterInto(output, input, inputColor, diameter, sigmaColor, sigmaSpace, borderMode,
                                            pstream);
}

} // namespace

void ExportOpJointBilateralFilter(py::module &m)
{
    using namespace pybind11::literals;

    m.def("joint_bilateral_filter", &JointBilateralFilter, "src"_a, "srcColor"_a, "diameter"_a, "sigma_color"_a,
          "sigma_space"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);
    m.def("joint_bilateral_filter_into", &JointBilateralFilterInto, "dst"_a, "src"_a, "srcColor"_a, "diameter"_a,
          "sigma_color"_a, "sigma_space"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(),
          "stream"_a = nullptr);

    m.def("joint_bilateral_filter", &VarShapeJointBilateralFilter, "src"_a, "srcColor"_a, "diameter"_a, "sigma_color"_a,
          "sigma_space"_a, py::kw_only(), "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);
    m.def("joint_bilateral_filter_into", &VarShapeJointBilateralFilterInto, "dst"_a, "src"_a, "srcColor"_a,
          "diameter"_a, "sigma_color"_a, "sigma_space"_a, py::kw_only(),
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);
}

} // namespace cvcudapy

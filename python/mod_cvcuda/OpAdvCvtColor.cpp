/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvtColorUtil.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpAdvCvtColor.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {
Tensor AdvCvtColorInto(Tensor &output, Tensor &input, NVCVColorConversionCode code, NVCVColorSpec spec,
                       std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto          op = CreateOperator<cvcuda::AdvCvtColor>();
    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    op->submit(pstream->cudaHandle(), input, output, code, spec);
    return std::move(output);
}

Tensor AdvCvtColor(Tensor &input, NVCVColorConversionCode code, NVCVColorSpec spec, std::optional<Stream> pstream)
{
    nvcv::ImageFormat outputFormat = GetOutputFormat(input.dtype(), code);
    nvcv::TensorShape outputShape  = GetOutputTensorShape(input.shape(), outputFormat, code);

    Tensor output = Tensor::Create(outputShape, input.dtype());

    return AdvCvtColorInto(output, input, code, spec, pstream);
}

} // namespace

void ExportOpAdvCvtColor(py::module &m)
{
    using namespace pybind11::literals;
    m.def("advcvtcolor", &AdvCvtColor, "src"_a, "code"_a, "spec"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        Executes the Adv Cvt Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Adv Cvt Color operator
            for more details and usage examples.

        Args:
            src (nvcv.Tensor): Input tensor containing one or more images.
            code (cvcuda.ColorConversion): Code describing the desired color conversion.
            spec (cvcuda.ColorSpec): Color specification for the conversion.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output color converted image.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("advcvtcolor_into", &AdvCvtColorInto, "dst"_a, "src"_a, "code"_a, "spec"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Adv Cvt Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Adv Cvt Color operator
            for more details and usage examples.

        Args:
            dst (nvcv.Tensor): Output tensor to store the result of the operation.
            src (nvcv.Tensor): Input tensor containing one or more images.
            code (cvcuda.ColorConversion): Code describing the desired color conversion.
            spec (cvcuda.ColorSpec): Color specification for the conversion.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

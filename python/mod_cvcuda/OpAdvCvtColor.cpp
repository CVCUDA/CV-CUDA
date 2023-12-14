/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*op});
    op->submit(pstream->cudaHandle(), input, output, code, spec);
    return std::move(output);
}

Tensor AdvCvtColor(Tensor &input, NVCVColorConversionCode code, NVCVColorSpec spec, std::optional<Stream> pstream)
{
    if (input.shape().rank() < 3 || input.shape().rank() > 4)
    {
        throw std::runtime_error("Invalid input tensor shape");
    }

    int64_t outputShape[4] = {};
    bool    heightIndex    = input.shape().rank() == 4 ? 1 : 0;
    for (int i = 0; i < input.shape().rank(); i++)
    {
        outputShape[i] = input.shape()[i];
    }

    switch (code)
    {
    case NVCVColorConversionCode::NVCV_COLOR_YUV2RGB_NV12:
    case NVCVColorConversionCode::NVCV_COLOR_YUV2BGR_NV12:
    case NVCVColorConversionCode::NVCV_COLOR_YUV2RGB_NV21:
    case NVCVColorConversionCode::NVCV_COLOR_YUV2BGR_NV21:
    {
        outputShape[heightIndex]     = (2 * outputShape[heightIndex]) / 3; // output height must be 2/3 of input height
        outputShape[heightIndex + 2] = 3;                                  // output channels must be 3
        break;
    }

    case NVCVColorConversionCode::NVCV_COLOR_RGB2YUV_NV12:
    case NVCVColorConversionCode::NVCV_COLOR_BGR2YUV_NV12:
    case NVCVColorConversionCode::NVCV_COLOR_RGB2YUV_NV21:
    case NVCVColorConversionCode::NVCV_COLOR_BGR2YUV_NV21:
    {
        outputShape[heightIndex]
            = (3 * outputShape[heightIndex]) / 2; // output height must be 3/2 of input height for UV plane
        outputShape[heightIndex + 2] = 1;         // output channels must be 1 for NV
        break;
    }
    default:
        break;
    }

    if (input.shape().rank() == 4)
    {
        nvcv::TensorShape yuvCorrectedShape({outputShape[0], outputShape[1], outputShape[2], outputShape[3]}, "NHWC");
        Tensor            output = Tensor::Create(yuvCorrectedShape, input.dtype());
        return AdvCvtColorInto(output, input, code, spec, pstream);
    }
    else
    {
        nvcv::TensorShape yuvCorrectedShape({outputShape[0], outputShape[1], outputShape[2]}, "HWC");
        Tensor            output = Tensor::Create(yuvCorrectedShape, input.dtype());
        return AdvCvtColorInto(output, input, code, spec, pstream);
    }
}

} // namespace

void ExportOpAdvCvtColor(py::module &m)
{
    using namespace pybind11::literals;

    m.def("advcvtcolor", &AdvCvtColor, "src"_a, "code"_a, "spec"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Adv Cvt Color operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Adv Cvt Color operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
            code (NVCVColorConversionCode): Code describing the desired color conversion.
            spec (NVCVColorSpec): Color specification for the conversion.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output color converted image.

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
            dst (Tensor): Output tensor to store the result of the operation.
            src (Tensor): Input tensor containing one or more images.
            code (NVCVColorConversionCode): Code describing the desired color conversion.
            spec (NVCVColorSpec): Color specification for the conversion.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

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

#include "../nvcv/Tensor.hpp"
#include "NativeBinding.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpConvertTo.hpp>
#include <nvcv/RoundMode.h>

#include <optional>

namespace cvcudapy {

namespace {
nvcvpy::priv::Tensor &ConvertToInto(nvcvpy::priv::Tensor &output, nvcvpy::priv::Tensor &input, float scale,
                                    float offset, NVCVRoundMode round, std::optional<nvcvpy::priv::Stream *> pstream)
{
    nvcvpy::priv::Stream &stream = pstream && *pstream ? **pstream : nvcvpy::priv::Stream::Current();

    auto cvt = CreateOperator<cvcuda::ConvertTo>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*cvt});

    guard.run([&cvt, &stream, &input, &output, &scale, &offset, &round]()
              { cvt->submit(stream.handle(), input.impl(), output.impl(), scale, offset, round); });

    return output;
}

std::shared_ptr<nvcvpy::priv::Tensor> ConvertTo(nvcvpy::priv::Tensor &input, nvcv::DataType dtype, float scale,
                                                float offset, NVCVRoundMode round,
                                                std::optional<nvcvpy::priv::Stream *> pstream)
{
    auto stream = native::GetStream(pstream).sharedStream();
    auto output = nvcvpy::priv::Tensor::CreateFromReqs(nvcv::Tensor::CalcRequirements(input.impl().shape(), dtype),
                                                       stream.get());

    ConvertToInto(*output, input, scale, offset, round, stream.get());
    return output;
}

} // namespace

void ExportOpConvertTo(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("convertto", NvtxTrace("cvcuda.convertto", &ConvertTo), "src"_a, "dtype"_a, "scale"_a = 1, "offset"_a = 0,
          py::kw_only(), "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr, R"pbdoc(
        Executes the Convert To operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            scale (float, optional): Scalar for output data.
            offset (float, optional): Offset for the data.
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST (round to nearest); use cvcuda.Round.TRUNCATE to truncate
                toward zero. Has no effect for floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("convertto_into", NvtxTrace("cvcuda.convertto_into", &ConvertToInto), "dst"_a, "src"_a, "scale"_a = 1,
          "offset"_a = 0, py::kw_only(), "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(
        Executes the Convert To operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            scale (float, optional): Scalar for output data.
            offset (float, optional): Offset for the data.
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST (round to nearest); use cvcuda.Round.TRUNCATE to truncate
                toward zero. Has no effect for floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

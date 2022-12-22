/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpConvertTo.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {
Tensor ConvertToInto(Tensor &output, Tensor &input, float scale, float offset, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto cvt = CreateOperator<cvcuda::ConvertTo>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*cvt});

    cvt->submit(pstream->cudaHandle(), input, output, scale, offset);

    return std::move(output);
}

Tensor ConvertTo(Tensor &input, nvcv::DataType dtype, float scale, float offset, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), dtype);

    return ConvertToInto(output, input, scale, offset, pstream);
}

} // namespace

void ExportOpConvertTo(py::module &m)
{
    using namespace pybind11::literals;

    m.def("convertto", &ConvertTo, "src"_a, "dtype"_a, "scale"_a = 1, "offset"_a = 0, py::kw_only(),
          "stream"_a = nullptr);
    m.def("convertto_into", &ConvertToInto, "dst"_a, "src"_a, "scale"_a = 1, "offset"_a = 0, py::kw_only(),
          "stream"_a = nullptr);
}

} // namespace cvcudapy

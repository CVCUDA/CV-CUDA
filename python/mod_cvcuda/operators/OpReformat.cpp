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

#include <common/PyUtil.hpp>
#include <cvcuda/OpReformat.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {
Tensor ReformatInto(Tensor &output, Tensor &input, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto reformat = CreateOperator<cvcuda::Reformat>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*reformat});

    guard.run([&reformat, &pstream, &input, &output]() { reformat->submit(pstream->cudaHandle(), input, output); });

    return std::move(output);
}

Tensor Reformat(Tensor &input, const nvcv::TensorLayout &out_layout, std::optional<Stream> pstream)
{
    nvcv::TensorShape out_shape = Permute(input.shape(), out_layout);

    Tensor output = Tensor::Create(out_shape, input.dtype());

    return ReformatInto(output, input, pstream);
}

} // namespace

void ExportOpReformat(py::module &m)
{
    using namespace pybind11::literals;

    m.def("reformat", NvtxTrace("cvcuda.reformat", &Reformat), "src"_a, "layout"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the Reformat operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            layout (cvcuda.TensorLayout): The tensor layout of the output tensor ((N)CHW/(N)HWC).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("reformat_into", NvtxTrace("cvcuda.reformat_into", &ReformatInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Reformat operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

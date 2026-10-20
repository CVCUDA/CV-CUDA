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
#include <cvcuda/OpReformat.hpp>

#include <optional>

namespace cvcudapy {

namespace {
nvcvpy::priv::Tensor &ReformatInto(nvcvpy::priv::Tensor &output, nvcvpy::priv::Tensor &input,
                                   std::optional<nvcvpy::priv::Stream *> pstream)
{
    nvcvpy::priv::Stream &stream = pstream && *pstream ? **pstream : nvcvpy::priv::Stream::Current();

    auto reformat = CreateOperator<cvcuda::Reformat>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*reformat});

    guard.run([&reformat, &stream, &input, &output]()
              { reformat->submit(stream.handle(), input.impl(), output.impl()); });

    return output;
}

std::shared_ptr<nvcvpy::priv::Tensor> Reformat(nvcvpy::priv::Tensor &input, const nvcv::TensorLayout &out_layout,
                                               std::optional<nvcvpy::priv::Stream *> pstream)
{
    nvcv::TensorShape out_shape = Permute(input.impl().shape(), out_layout);

    auto stream = native::GetStream(pstream).sharedStream();
    auto output
        = nvcvpy::priv::Tensor::CreateFromReqs(nvcv::Tensor::CalcRequirements(out_shape, input.dtype()), stream.get());

    ReformatInto(*output, input, stream.get());
    return output;
}

} // namespace

void ExportOpReformat(py::module_ &m)
{
    using namespace nanobind::literals;

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
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
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

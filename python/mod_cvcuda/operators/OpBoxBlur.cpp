/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpBoxBlur.hpp>
#include <cvcuda/priv/Types.hpp>

#include <memory>

namespace cvcudapy {

template<>
struct OperatorCacheUsesCompletionLease<cvcuda::BoxBlur> : std::true_type
{
};

namespace {
using BlurBoxesImpl = cvcuda::priv::NVCVBlurBoxesImpl;
using native::Stream;
using native::Tensor;

Tensor &BoxBlurInto(Tensor &output, Tensor &input, const std::shared_ptr<BlurBoxesImpl> &bboxes,
                    std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto op = CreateOperator<cvcuda::BoxBlur>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    auto bboxesHandle = static_cast<NVCVBlurBoxesI>(static_cast<void *>(bboxes.get()));
    guard.run([&op, &stream, &input, &output, bboxesHandle]()
              { op->submit(stream.handle(), input.impl(), output.impl(), bboxesHandle); });

    return output;
}

std::shared_ptr<Tensor> BoxBlur(Tensor &input, const std::shared_ptr<BlurBoxesImpl> &bboxes,
                                std::optional<Stream *> pstream)
{
    auto output = native::CreateTensor(input.impl().shape(), input.dtype());
    BoxBlurInto(*output, input, bboxes, pstream);
    return output;
}

} // namespace

void ExportOpBoxBlur(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("boxblur", NvtxTrace("cvcuda.boxblur", &BoxBlur), "src"_a, "bboxes"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the BoxBlur operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            bboxes (cvcuda.BlurBoxesI):  Blur boxes in reference to the input tensor.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("boxblur_into", NvtxTrace("cvcuda.boxblur_into", &BoxBlurInto), "dst"_a, "src"_a, "bboxes"_a, py::kw_only(),
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the BoxBlur operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            bboxes (cvcuda.BlurBoxesI):  Blur boxes in reference to the input tensor.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

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

// NOTE (make-op): if __OPNAME__ is a unary element-wise operator that preserves shape/dtype/layout
// (like Invert / Solarize / Posterize), prefer the shared binding helpers in
// "operators/UnaryElementwiseOp.hpp" over the hand-rolled wrappers below — they remove the
// create/ResourceGuard/submit copy-paste that otherwise trips SonarQube's duplication gate per op:
//     #include "UnaryElementwiseOp.hpp"
//     Tensor __OPNAME__Into(Tensor &o, Tensor &i, std::optional<Stream> s)
//     { return UnaryElementwiseInto<cvcuda::__OPNAME__>(o, i, s /*, extra submit params */); }
//     Tensor __OPNAME__(Tensor &i, std::optional<Stream> s)
//     { return UnaryElementwiseTensor<cvcuda::__OPNAME__>(i, s /*, extra submit params */); }
// Keep the generic wrappers below for operators that change dtype/shape or are not element-wise.
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/Op__OPNAME__.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {
Tensor __OPNAME__Into(Tensor &output, Tensor &input, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::__OPNAME__>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    // TODO if op kernel allocates resources that are accessed by the device change to READWRITE
    // is set to none it is possible for the operator to be destroyed before the kernel is executed.
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    op->submit(pstream->cudaHandle(), input, output);

    return std::move(output);
}

Tensor __OPNAME__(Tensor &input, nvcv::DataType dtype, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), dtype);

    return __OPNAME__Into(output, input, pstream);
}

} // namespace

void ExportOp__OPNAME__(py::module &m)
{
    using namespace pybind11::literals;

    m.def("__OPNAMELOW__", NvtxTrace("cvcuda.__OPNAMELOW__", &__OPNAME__), "src"_a, "dtype"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the __OPNAMESPACE__ operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the __OPNAMESPACE__ operator
            for more details and usage examples.

        Args:
            TBD args
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            TBD

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("__OPNAMELOW___into", NvtxTrace("cvcuda.__OPNAMELOW___into", &__OPNAME__Into), "dst"_a, "src"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the __OPNAMESPACE__ operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the __OPNAMESPACE__ operator
            for more details and usage examples.

        Args:
            TBD args
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            TBD

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

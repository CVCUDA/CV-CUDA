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

#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpBoxBlur.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {
Tensor BoxBlurInto(Tensor &output, Tensor &input, NVCVBlurBoxesI bboxes, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::BoxBlur>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*op});

    op->submit(pstream->cudaHandle(), input, output, bboxes);

    return std::move(output);
}

Tensor BoxBlur(Tensor &input, NVCVBlurBoxesI bboxes, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());
    return BoxBlurInto(output, input, bboxes, pstream);
}

} // namespace

void ExportOpBoxBlur(py::module &m)
{
    using namespace pybind11::literals;

    m.def("boxblur", &BoxBlur, "src"_a, "bboxes"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the BoxBlur operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Convolve 2D operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
            bboxes (NVCVBlurBoxesI):  Blur boxes in reference to the input tensor.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("boxblur_into", &BoxBlurInto, "dst"_a, "src"_a, "bboxes"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the BoxBlur operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Convolve 2D operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output tensor to store the result of the operation.
            src (Tensor): Input tensor containing one or more images.
            bboxes (NVCVBlurBoxesI):  Blur boxes in reference to the input tensor.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

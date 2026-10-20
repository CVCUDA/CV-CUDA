/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpAutoContrast.hpp>

namespace cvcudapy {

namespace {
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

inline std::shared_ptr<Tensor> tensorLike(Tensor &src)
{
    return Tensor::CreateFromReqs(nvcv::Tensor::CalcRequirements(src.impl().shape(), src.dtype()));
}

template<typename Container>
Container &AutoContrastIntoImpl(Container &output, Container &input, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto op = CreateOperator<cvcuda::AutoContrast>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    // The operator allocates a per-(image, channel) min/max scratch buffer accessed by the device,
    // so the operator object must outlive the kernels: lock it READWRITE rather than NONE.
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    guard.run([&op, &stream, &input, &output]() { op->submit(stream.handle(), input.impl(), output.impl()); });

    return output;
}

Tensor &AutoContrastInto(Tensor &output, Tensor &input, std::optional<Stream *> pstream)
{
    return AutoContrastIntoImpl(output, input, pstream);
}

std::shared_ptr<Tensor> AutoContrast(Tensor &input, std::optional<Stream *> pstream)
{
    auto output = tensorLike(input);

    AutoContrastInto(*output, input, pstream);
    return output;
}

ImageBatchVarShape &VarShapeAutoContrastInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                             std::optional<Stream *> pstream)
{
    return AutoContrastIntoImpl(output, input, pstream);
}

std::shared_ptr<ImageBatchVarShape> VarShapeAutoContrast(ImageBatchVarShape &input, std::optional<Stream *> pstream)
{
    auto output = CreateSameShapeImageBatch(input);

    VarShapeAutoContrastInto(*output, input, pstream);
    return output;
}

} // namespace

void ExportOpAutoContrast(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("autocontrast", NvtxTrace("cvcuda.autocontrast", &AutoContrast), "src"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        Executes the Auto Contrast operation on the given cuda stream.

        Maximizes (normalizes) image contrast by remapping each channel independently so its spatial
        minimum maps to 0 and its spatial maximum maps to the data-type maximum (255 for 8-bit, 65535
        for 16-bit, 1.0 for float). A channel that is flat (all pixels equal) is left unchanged. This
        mimics ``torchvision.transforms.v2.functional.autocontrast`` / ``PIL.ImageOps.autocontrast``
        with ``cutoff = 0`` for finite inputs. For floating-point inputs, only finite pixels define
        the channel range; NaN and infinity pixels are copied unchanged.

        See also:
            Refer to the CV-CUDA C API reference for the Auto Contrast operator
            for more details and usage examples.

        Args:
            src (cvcuda.Tensor): Input tensor.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same shape, layout, and dtype as the input).

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("autocontrast_into", NvtxTrace("cvcuda.autocontrast_into", &AutoContrastInto), "dst"_a, "src"_a,
          py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(

        Executes the Auto Contrast operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Auto Contrast operator
            for more details and usage examples.

        Args:
            dst (cvcuda.Tensor): Output tensor (same shape, layout, and dtype as the input).
            src (cvcuda.Tensor): Input tensor.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    // VarShape variants
    m.def("autocontrast", NvtxTrace("cvcuda.autocontrast", &VarShapeAutoContrast), "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Auto Contrast operation on a batch of variable-shaped images.

        For floating-point inputs, only finite pixels define each channel range; NaN and infinity
        pixels are copied unchanged.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("autocontrast_into", NvtxTrace("cvcuda.autocontrast_into", &VarShapeAutoContrastInto), "dst"_a, "src"_a,
          py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(

        Executes the Auto Contrast operation on a batch of variable-shaped images.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch.
            src (cvcuda.ImageBatchVarShape): Input image batch.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

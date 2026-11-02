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
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpImageArea.hpp>

#include <array>
#include <optional>
#include <stdexcept>

namespace cvcudapy {

namespace {
using native::Stream;
using native::Tensor;

Tensor &ImageAreaInto(Tensor &output, Tensor &input, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto op = CreateOperator<cvcuda::ImageArea>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run([&op, &stream, &input, &output]() { op->submit(stream.handle(), input.impl(), output.impl()); });

    return output;
}

std::shared_ptr<Tensor> ImageArea(Tensor &input, std::optional<Stream *> pstream)
{
    ssize_t batch = 0;
    // check for non batched tensors
    if (const nvcv::TensorShape &inputShape = input.impl().shape(); inputShape.rank() == 3)
    {
        batch = 1;
    }
    else if (inputShape.rank() == 4)
    {
        batch = inputShape[0];
    }
    else
    {
        throw std::invalid_argument("Input tensor must be HWC, NHWC, CHW, or NCHW");
    }

    std::array<ssize_t, 1> shape{batch};
    auto output = native::CreateTensor(nvcv::TensorShape(shape.data(), shape.size(), nvcv::TENSOR_N), nvcv::TYPE_S32);

    ImageAreaInto(*output, input, pstream);
    return output;
}

} // namespace

void ExportOpImageArea(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("image_area", NvtxTrace("cvcuda.image_area", &ImageArea), "src"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        Executes the ImageArea operation on the given cuda stream.

        Writes the spatial pixel count (area) of each sample image, ``out[n] = width * height``,
        with the channel dimension excluded. Only the tensor geometry is used; pixel values,
        layout (interleaved or planar) and data type do not change the result. The output is a
        rank-1 tensor with one ``int32`` value per input sample.

        See also:
            Refer to the CV-CUDA C API reference for the ImageArea operator for more details and
            usage examples.

        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images (HWC, NHWC, CHW or NCHW).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: A rank-1 (N) tensor of dtype int32 holding the area of every image.

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");

    m.def("image_area_into", NvtxTrace("cvcuda.image_area_into", &ImageAreaInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(

        Executes the ImageArea operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the ImageArea operator for more details and
            usage examples.

        Args:
            dst (cvcuda.Tensor): Output tensor (rank-1 (N), dtype int32) to store the result.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
}

} // namespace cvcudapy

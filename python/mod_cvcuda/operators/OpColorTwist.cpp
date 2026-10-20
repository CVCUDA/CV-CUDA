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
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpColorTwist.hpp>

namespace cvcudapy {

namespace {
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

inline std::shared_ptr<Tensor> tensorLike(Tensor &src)
{
    return native::CreateTensor(src.impl().shape(), src.dtype());
}

inline std::shared_ptr<ImageBatchVarShape> batchLike(ImageBatchVarShape &src)
{
    return CreateSameShapeImageBatch(src);
}

template<typename Op, typename Src, typename Dst, typename Call>
auto runGuard(Op &op, Src &src, Dst &dst, Tensor &twist, std::optional<Stream *> pstream, Call &&call)
{
    Stream &stream = native::GetStream(pstream);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {src, twist});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run([&call, &stream]() { call(stream); });
}

Tensor &ColorTwistMatrixInto(Tensor &dst, Tensor &src, Tensor &twist, std::optional<Stream *> pstream)
{
    auto op = CreateOperator<cvcuda::ColorTwist>();
    runGuard(op, src, dst, twist, pstream,
             [&op, &src, &dst, &twist](const Stream &stream)
             { op->submit(stream.handle(), src.impl(), dst.impl(), twist.impl()); });
    return dst;
}

std::shared_ptr<Tensor> ColorTwistMatrix(Tensor &src, Tensor &twist, std::optional<Stream *> pstream)
{
    auto dst = tensorLike(src);
    ColorTwistMatrixInto(*dst, src, twist, pstream);
    return dst;
}

ImageBatchVarShape &VarShapeColorTwistMatrixInto(ImageBatchVarShape &dst, ImageBatchVarShape &src, Tensor &twist,
                                                 std::optional<Stream *> pstream)
{
    auto op = CreateOperator<cvcuda::ColorTwist>();
    runGuard(op, src, dst, twist, pstream,
             [&op, &src, &dst, &twist](const Stream &stream)
             { op->submit(stream.handle(), src.impl(), dst.impl(), twist.impl()); });
    return dst;
}

std::shared_ptr<ImageBatchVarShape> VarShapeColorTwistMatrix(ImageBatchVarShape &src, Tensor &twist,
                                                             std::optional<Stream *> pstream)
{
    auto dst = batchLike(src);
    VarShapeColorTwistMatrixInto(*dst, src, twist, pstream);
    return dst;
}

} // namespace

void ExportOpColorTwist(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("color_twist", NvtxTrace("cvcuda.color_twist", &ColorTwistMatrix), "src"_a, "twist"_a, py::kw_only(),
          "stream"_a = nullptr,
          R"pbdoc(
        Transforms an image by applying affine transformation to the channels extent.


        Args:
            src (cvcuda.Tensor): Tensor corresponding to the input image. It must have
                either 3 or 4 channels. In the case of 4 channels, the alpha channel is
                unmodified.
            twist (cvcuda.Tensor): A 2D tensor describing a 3x4 affine transformation matrix.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.
    )pbdoc");
    m.def("color_twist_into", NvtxTrace("cvcuda.color_twist_into", &ColorTwistMatrixInto), "dst"_a, "src"_a, "twist"_a,
          py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Transforms an image by applying affine transformation to the channels extent.


        Args:
            dst (cvcuda.Tensor): Tensor corresponding to the output image. Must match the shape of
                the input image.
            src (cvcuda.Tensor): Tensor corresponding to the input image. It must have
                either 3 or 4 channels. In the case of 4 channels, the alpha channel is
                unmodified.
            twist (cvcuda.Tensor): A 2D tensor describing a 3x4 affine transformation matrix.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    // VarShape variants
    m.def("color_twist", NvtxTrace("cvcuda.color_twist", &VarShapeColorTwistMatrix), "src"_a, "twist"_a, py::kw_only(),
          "stream"_a = nullptr,
          R"pbdoc(
        Transforms a batch of images by applying affine transformation to the channels extent.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch. Each image must have either 3 or 4
                channels. In the case of 4 channels the alpha channel is unmodified.
            twist (cvcuda.Tensor): A 3x4 2D tensor describing an affine transformation matrix or a
                Nx3x4 3D tensor specifying separate transformations for each sample in the input
                image batch.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");
    m.def("color_twist_into", NvtxTrace("cvcuda.color_twist_into", &VarShapeColorTwistMatrixInto), "dst"_a, "src"_a,
          "twist"_a, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Transforms a batch of images by applying affine transformation to the channels extent.

        The twist should be a 2D tensor describing 3x4 affine transformation matrix or a 3D tensor specifying
        separate transformations for each sample in the input image batch.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch. The shapes of the output images
                must match the input image batch.
            src (cvcuda.ImageBatchVarShape): Input image batch. Each image must have either 3 or 4
                channels. In the case of 4 channels the alpha channel is unmodified.
            twist (cvcuda.Tensor): A 3x4 2D tensor describing an affine transformation matrix or an
                Nx3x4 3D tensor specifying separate transformations for each sample in the input
                image batch.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <common/String.hpp>
#include <cvcuda/OpColorTwist.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {

inline Tensor tensorLike(Tensor &src)
{
    const auto &srcShape = src.shape();
    Shape       dstShape = nvcvpy::CreateShape(srcShape);

    return Tensor::Create(dstShape, src.dtype(), src.layout());
}

inline ImageBatchVarShape batchLike(ImageBatchVarShape &src)
{
    ImageBatchVarShape dst = ImageBatchVarShape::Create(src.capacity());
    for (int i = 0; i < src.numImages(); ++i)
    {
        dst.pushBack(Image::Create(src[i].size(), src[i].format()));
    }
    return dst;
}

template<typename Op, typename Src, typename Dst, typename Call>
auto runGuard(Op &op, Src &src, Dst &dst, const Tensor &twist, std::optional<Stream> &pstream, Call &&call)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {src, twist});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    call(*pstream);
}

Tensor ColorTwistMatrixInto(Tensor &dst, Tensor &src, Tensor &twist, std::optional<Stream> pstream)
{
    auto op = CreateOperator<cvcuda::ColorTwist>();
    runGuard(op, src, dst, twist, pstream, [&](Stream &stream) { op->submit(stream.cudaHandle(), src, dst, twist); });
    return dst;
}

Tensor ColorTwistMatrix(Tensor &src, Tensor &twist, std::optional<Stream> pstream)
{
    auto dst = tensorLike(src);
    return ColorTwistMatrixInto(dst, src, twist, pstream);
}

ImageBatchVarShape VarShapeColorTwistMatrixInto(ImageBatchVarShape &dst, ImageBatchVarShape &src, Tensor &twist,
                                                std::optional<Stream> pstream)
{
    auto op = CreateOperator<cvcuda::ColorTwist>();
    runGuard(op, src, dst, twist, pstream, [&](Stream &stream) { op->submit(stream.cudaHandle(), src, dst, twist); });
    return dst;
}

ImageBatchVarShape VarShapeColorTwistMatrix(ImageBatchVarShape &src, Tensor &twist, std::optional<Stream> pstream)
{
    auto dst = batchLike(src);
    return VarShapeColorTwistMatrixInto(dst, src, twist, pstream);
}

} // namespace

void ExportOpColorTwist(py::module &m)
{
    using namespace pybind11::literals;

    m.def("color_twist", &ColorTwistMatrix, "src"_a, "twist"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        cvcuda.color_twist(src: nvcv.Tensor, twist: nvcv.Tensor, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor

        Transforms an image by applying affine transformation to the channels extent.

        See Also:
            Refer to the CV-CUDA C API reference for the ColorTwist operator for more details and
            usage examples.

        Args:
            src (nvcv.Tensor): Tensor corresponding to the input image. It must have
                either 3 or 4 channels. In the case of 4 channels, the alpha channel is
                unmodified.
            twist (nvcv.Tensor): A 2D tensor describing a 3x4 affine transformation matrix.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor.
    )pbdoc");
    m.def("color_twist_into", &ColorTwistMatrixInto, "dst"_a, "src"_a, "twist"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        cvcuda.color_twist_into(dst: nvcv.Tensor, src: nvcv.Tensor, twist: nvcv.Tensor, stream: Optional[nvcv.cuda.Stream] = None)

        Transforms an image by applying affine transformation to the channels extent.

        See Also:
            Refer to the CV-CUDA C API reference for the ColorTwist operator for more details and
            usage examples.

        Args:
            dst (nvcv.Tensor): Tensor corresponding to the output image. Must match the shape of
                the input image.
            src (nvcv.Tensor): Tensor corresponding to the input image. It must have
                either 3 or 4 channels. In the case of 4 channels, the alpha channel is
                unmodified.
            twist (nvcv.Tensor): A 2D tensor describing a 3x4 affine transformation matrix.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None
    )pbdoc");

    // VarShape variants
    m.def("color_twist", &VarShapeColorTwistMatrix, "src"_a, "twist"_a, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        cvcuda.color_twist(src: nvcv.ImageBatchVarShape, twist: nvcv.Tensor, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.ImageBatchVarShape

        Transforms a batch of images by applying affine transformation to the channels extent.

        See Also:
            Refer to the CV-CUDA C API reference for the ColorTwist operator for more details and
            usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch. Each image must have either 3 or 4
                channels. In the case of 4 channels the alpha channel is unmodified.
            twist (nvcv.Tensor): A 3x4 2D tensor describing an affine transformation matrix or a
                Nx3x4 3D tensor specifying separate transformations for each sample in the input
                image batch.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch.

    )pbdoc");
    m.def("color_twist_into", &VarShapeColorTwistMatrixInto, "dst"_a, "src"_a, "twist"_a, py::kw_only(),
          "stream"_a = nullptr,
          R"pbdoc(

        cvcuda.color_twist_into(dst: nvcv.ImageBatchVarShape, src: nvcv.ImageBatchVarShape, twist: nvcv.Tensor, stream: Optional[nvcv.cuda.Stream] = None)

        Transforms a batch of images by applying affine transformation to the channels extent.

        The twist should be a 2D tensor describing 3x4 affine transformation matrix or a 3D tensor specifying
        separate transformations for each sample in the input image batch.

        See Also:
            Refer to the CV-CUDA C API reference for the ColorTwist operator for more details and
            usage examples.

        Args:
            dst (nvcv.ImageBatchVarShape): Output image batch. The shapes of the output images
                must match the input image batch.
            src (nvcv.ImageBatchVarShape): Input image batch. Each image must have either 3 or 4
                channels. In the case of 4 channels the alpha channel is unmodified.
            twist (nvcv.Tensor): A 3x4 2D tensor describing an affine transformation matrix or an
                Nx3x4 3D tensor specifying separate transformations for each sample in the input
                image batch.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None
    )pbdoc");
}

} // namespace cvcudapy

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpComposite.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor CompositeInto(Tensor &output, Tensor &foreground, Tensor &background, Tensor &fgMask,
                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto composite = CreateOperator<cvcuda::Composite>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {foreground, background, fgMask});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*composite});

    composite->submit(pstream->cudaHandle(), foreground, background, fgMask, output);

    return output;
}

Tensor Composite(Tensor &foreground, Tensor &background, Tensor &fgMask, int outChannels, std::optional<Stream> pstream)
{
    Shape out_shape                 = CreateShape(foreground.shape());
    out_shape[out_shape.size() - 1] = outChannels;

    Tensor output = Tensor::Create(out_shape, foreground.dtype(), foreground.layout());

    return CompositeInto(output, foreground, background, fgMask, pstream);
}

ImageBatchVarShape CompositeVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &foreground,
                                         ImageBatchVarShape &background, ImageBatchVarShape &fgMask,
                                         std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto composite = CreateOperator<cvcuda::Composite>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {foreground, background, fgMask});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*composite});

    composite->submit(pstream->cudaHandle(), foreground, background, fgMask, output);

    return output;
}

ImageBatchVarShape CompositeVarShape(ImageBatchVarShape &foreground, ImageBatchVarShape &background,
                                     ImageBatchVarShape &fgMask, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(foreground.numImages());

    nvcv::ImageFormat format = foreground.uniqueFormat();

    for (auto img = foreground.begin(); img != foreground.end(); ++img)
    {
        auto newimg = Image::Create(img->size(), format);
        output.pushBack(newimg);
    }

    return CompositeVarShapeInto(output, foreground, background, fgMask, pstream);
}

} // namespace

void ExportOpComposite(py::module &m)
{
    using namespace pybind11::literals;
    py::options options;
    options.disable_function_signatures();

    m.def("composite", &Composite, "foreground"_a, "background"_a, "fgmask"_a, "outchannels"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        cvcuda.composite(foreground: nvcv.Tensor, background: nvcv.Tensor, fgmask: nvcv.Tensor, outchannels: int, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor

        Executes the Composite operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Composite operator
            for more details and usage examples.

        Args:
            foreground (nvcv.Tensor): Input tensor containing one or more foreground images. Each image is BGR (3-channel) 8-bit.
            background (nvcv.Tensor): Input tensor containing one or more background images. Each image is BGR (3-channel) 8-bit.
            fgmask (nvcv.Tensor): Input foreground mask tensor. Each mask image is grayscale 8-bit
            outchannels (int): Specifies 3 channel for RGB and 4 channel for BGRA.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("composite_into", &CompositeInto, "dst"_a, "foreground"_a, "background"_a, "fgmask"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        cvcuda.composite_into(dst: nvcv.Tensor, foreground: nvcv.Tensor, background: nvcv.Tensor, fgmask: nvcv.Tensor, outchannels: int, stream: Optional[nvcv.cuda.Stream] = None)

	Executes the Composite operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Composite operator
            for more details and usage examples.

        Args:
            dst (nvcv.Tensor): Output tensor to store the result of the operation.
            foreground (nvcv.Tensor): Input tensor containing one or more foreground images. Each image is BGR (3-channel) 8-bit.
            background (nvcv.Tensor): Input tensor containing one or more background images. Each image is BGR (3-channel) 8-bit.
            fgmask (nvcv.Tensor): Input foreground mask tensor. Each mask image is grayscale 8-bit.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("composite", &CompositeVarShape, "foreground"_a, "background"_a, "fgmask"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

	cvcuda.composite(foreground: nvcv.ImageBatchVarShape, background: nvcv.ImageBatchVarShape, fgmask: nvcv.ImageBatchVarShape, outchannels: int, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.ImageBatchVarShape

	Executes the Composite operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Composite operator
            for more details and usage examples.

        Args:
            foreground (nvcv.ImageBatchVarShape): Input tensor containing one or more foreground images. Each image is BGR (3-channel) 8-bit.
            background (nvcv.ImageBatchVarShape): Input tensor containing one or more background images. Each image is BGR (3-channel) 8-bit.
            fgmask (nvcv.ImageBatchVarShape): Input foreground mask image batch. Each mask image is grayscale 8-bit.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("composite_into", &CompositeVarShapeInto, "dst"_a, "foreground"_a, "background"_a, "fgmask"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

	cvcuda.composite_into(dst: nvcv.ImageBatchVarShape, foreground: nvcv.ImageBatchVarShape, background: nvcv.ImageBatchVarShape, fgmask: nvcv.ImageBatchVarShape, outchannels: int, stream: Optional[nvcv.cuda.Stream] = None)

        Executes the Composite operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Composite operator
            for more details and usage examples.

        Args:
            dst (nvcv.ImageBatchVarShape): Output image batch containing the result of the operation.
            foreground (nvcv.ImageBatchVarShape): Input tensor containing one or more foreground images. Each image is BGR (3-channel) 8-bit.
            background (nvcv.ImageBatchVarShape): Input tensor containing one or more background images. Each image is BGR (3-channel) 8-bit.
            fgmask (nvcv.ImageBatchVarShape): Input foreground mask image batch. Each mask image is grayscale 8-bit.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

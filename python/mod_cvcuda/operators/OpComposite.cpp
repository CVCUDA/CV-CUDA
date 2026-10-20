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

#include "../nvcv/ImageBatch.hpp"
#include "../nvcv/Tensor.hpp"
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/Binding.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpComposite.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <stdexcept>

namespace cvcudapy {

namespace {
nvcvpy::priv::Tensor &CompositeInto(nvcvpy::priv::Tensor &output, nvcvpy::priv::Tensor &foreground,
                                    nvcvpy::priv::Tensor &background, nvcvpy::priv::Tensor &fgMask,
                                    std::optional<nvcvpy::priv::Stream *> pstream)
{
    nvcvpy::priv::Stream &stream = pstream && *pstream ? **pstream : nvcvpy::priv::Stream::Current();

    auto composite = CreateOperator<cvcuda::Composite>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {foreground, background, fgMask});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*composite});

    guard.run(
        [&composite, &stream, &foreground, &background, &fgMask, &output]()
        { composite->submit(stream.handle(), foreground.impl(), background.impl(), fgMask.impl(), output.impl()); });

    return output;
}

std::shared_ptr<nvcvpy::priv::Tensor> Composite(nvcvpy::priv::Tensor &foreground, nvcvpy::priv::Tensor &background,
                                                nvcvpy::priv::Tensor &fgMask, int outChannels,
                                                std::optional<nvcvpy::priv::Stream *> pstream)
{
    auto outShape   = foreground.impl().shape().shape();
    auto layout     = foreground.impl().layout();
    int  channelIdx = layout.find('C');
    if (channelIdx < 0)
    {
        throw std::invalid_argument(
            util::ConcatString("Cannot infer Composite output shape for layout=", std::string(layout.m_layout.data)));
    }
    outShape[channelIdx] = outChannels;

    auto output = nvcvpy::priv::Tensor::CreateFromReqs(
        nvcv::Tensor::CalcRequirements(nvcv::TensorShape{std::move(outShape), layout}, foreground.dtype()));

    CompositeInto(*output, foreground, background, fgMask, pstream);
    return output;
}

nvcvpy::priv::ImageBatchVarShape &CompositeVarShapeInto(nvcvpy::priv::ImageBatchVarShape     &output,
                                                        nvcvpy::priv::ImageBatchVarShape     &foreground,
                                                        nvcvpy::priv::ImageBatchVarShape     &background,
                                                        nvcvpy::priv::ImageBatchVarShape     &fgMask,
                                                        std::optional<nvcvpy::priv::Stream *> pstream)
{
    nvcvpy::priv::Stream &stream = pstream && *pstream ? **pstream : nvcvpy::priv::Stream::Current();

    auto composite = CreateOperator<cvcuda::Composite>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {foreground, background, fgMask});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*composite});

    guard.run(
        [&composite, &stream, &foreground, &background, &fgMask, &output]()
        { composite->submit(stream.handle(), foreground.impl(), background.impl(), fgMask.impl(), output.impl()); });

    return output;
}

nvcv::ImageFormat CompositeVarShapeOutputFormat(nvcv::ImageFormat foregroundFormat, int outChannels)
{
    if (foregroundFormat.numChannels() == outChannels)
    {
        return foregroundFormat;
    }

    if (outChannels == 4)
    {
        if (foregroundFormat == nvcv::FMT_RGB8)
        {
            return nvcv::FMT_RGBA8;
        }
        if (foregroundFormat == nvcv::FMT_RGB8p)
        {
            return nvcv::FMT_RGBA8p;
        }
        if (foregroundFormat == nvcv::FMT_BGR8)
        {
            return nvcv::FMT_BGRA8;
        }
    }

    throw std::invalid_argument(
        util::ConcatString("Cannot infer Composite output format for outchannels=", outChannels));
}

std::shared_ptr<nvcvpy::priv::ImageBatchVarShape> CompositeVarShape(nvcvpy::priv::ImageBatchVarShape     &foreground,
                                                                    nvcvpy::priv::ImageBatchVarShape     &background,
                                                                    nvcvpy::priv::ImageBatchVarShape     &fgMask,
                                                                    int                                   outChannels,
                                                                    std::optional<nvcvpy::priv::Stream *> pstream)
{
    nvcv::ImageFormat format = CompositeVarShapeOutputFormat(foreground.impl().uniqueFormat(), outChannels);
    auto              output = CreateSameShapeImageBatch(foreground, format, foreground.capacity());

    CompositeVarShapeInto(*output, foreground, background, fgMask, pstream);
    return output;
}

} // namespace

void ExportOpComposite(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("composite", NvtxTrace("cvcuda.composite", &Composite), "foreground"_a, "background"_a, "fgmask"_a,
          "outchannels"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Composite operation on the given cuda stream.


        Args:
            foreground (cvcuda.Tensor): Input tensor containing one or more foreground images. Each image is BGR (3-channel) 8-bit.
            background (cvcuda.Tensor): Input tensor containing one or more background images. Each image is BGR (3-channel) 8-bit.
            fgmask (cvcuda.Tensor): Input foreground mask tensor. Each mask image is grayscale 8-bit
            outchannels (int): Specifies 3 channels for RGB/BGR and 4 channels for RGBA/BGRA.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("composite_into", NvtxTrace("cvcuda.composite_into", &CompositeInto), "dst"_a, "foreground"_a, "background"_a,
          "fgmask"_a, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the Composite operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            foreground (cvcuda.Tensor): Input tensor containing one or more foreground images. Each image is BGR (3-channel) 8-bit.
            background (cvcuda.Tensor): Input tensor containing one or more background images. Each image is BGR (3-channel) 8-bit.
            fgmask (cvcuda.Tensor): Input foreground mask tensor. Each mask image is grayscale 8-bit.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("composite", NvtxTrace("cvcuda.composite", &CompositeVarShape), "foreground"_a, "background"_a, "fgmask"_a,
          "outchannels"_a = 3, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Composite operation on the given cuda stream.


        Args:
            foreground (cvcuda.ImageBatchVarShape): Input tensor containing one or more foreground images. Each image is BGR (3-channel) 8-bit.
            background (cvcuda.ImageBatchVarShape): Input tensor containing one or more background images. Each image is BGR (3-channel) 8-bit.
            fgmask (cvcuda.ImageBatchVarShape): Input foreground mask image batch. Each mask image is grayscale 8-bit.
            outchannels (int): Specifies 3 channels for RGB/BGR and 4 channels for RGBA/BGRA.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("composite_into", NvtxTrace("cvcuda.composite_into", &CompositeVarShapeInto), "dst"_a, "foreground"_a,
          "background"_a, "fgmask"_a, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the Composite operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            foreground (cvcuda.ImageBatchVarShape): Input tensor containing one or more foreground images. Each image is BGR (3-channel) 8-bit.
            background (cvcuda.ImageBatchVarShape): Input tensor containing one or more background images. Each image is BGR (3-channel) 8-bit.
            fgmask (cvcuda.ImageBatchVarShape): Input foreground mask image batch. Each mask image is grayscale 8-bit.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

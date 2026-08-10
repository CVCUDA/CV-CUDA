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

#include "Operators.hpp"
#include "VarShapeUtils.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpErase.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

class EraseError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor EraseInto(Tensor &output, Tensor &input, Tensor &anchor, Tensor &erasing, Tensor &values, Tensor &imgIdx,
                 bool random, unsigned int seed, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    if (anchor.layout().rank() != 1 || anchor.layout()[0] != 'N')
    {
        throw EraseError("Layout of anchor must be 'N'.");
    }

    nvcv::TensorShape shape = anchor.shape();

    auto erase = CreateOperator<cvcuda::Erase>((int)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, anchor, erasing, values, imgIdx});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*erase});

    guard.run([&erase, &pstream, &input, &output, &anchor, &erasing, &values, &imgIdx, &random, &seed]()
              { erase->submit(pstream->cudaHandle(), input, output, anchor, erasing, values, imgIdx, random, seed); });

    return output;
}

Tensor Erase(Tensor &input, Tensor &anchor, Tensor &erasing, Tensor &values, Tensor &imgIdx, bool random,
             unsigned int seed, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return EraseInto(output, input, anchor, erasing, values, imgIdx, random, seed, pstream);
}

Tensor EraseRegionInto(Tensor &output, Tensor &input, int64_t i, int64_t j, int64_t h, int64_t w, Tensor &values,
                       std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto erase = CreateOperator<cvcuda::Erase>(0);

    ResourceGuard guard(*pstream);
    if (static_cast<const nvcv::Tensor &>(input).handle() == static_cast<const nvcv::Tensor &>(output).handle())
    {
        guard.add(LockMode::LOCK_MODE_READWRITE, {input});
    }
    else
    {
        guard.add(LockMode::LOCK_MODE_READ, {input});
        guard.add(LockMode::LOCK_MODE_WRITE, {output});
    }
    guard.add(LockMode::LOCK_MODE_READ, {values});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*erase});

    guard.run([&erase, &pstream, &input, &output, i, j, h, w, &values]
              { erase->submit(pstream->cudaHandle(), input, output, i, j, h, w, values); });

    return output;
}

Tensor EraseRegion(Tensor &input, int64_t i, int64_t j, int64_t h, int64_t w, Tensor &values,
                   std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());
    return EraseRegionInto(output, input, i, j, h, w, values, pstream);
}

ImageBatchVarShape EraseVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &anchor,
                                     Tensor &erasing, Tensor &values, Tensor &imgIdx, bool random, unsigned int seed,
                                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    if (anchor.layout().rank() != 1 || anchor.layout()[0] != 'N')
    {
        throw EraseError("Layout of anchor must be 'N'.");
    }

    nvcv::TensorShape shape = anchor.shape();

    auto erase = CreateOperator<cvcuda::Erase>((int)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, anchor, erasing, values, imgIdx});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*erase});

    guard.run([&erase, &pstream, &input, &output, &anchor, &erasing, &values, &imgIdx, &random, &seed]()
              { erase->submit(pstream->cudaHandle(), input, output, anchor, erasing, values, imgIdx, random, seed); });

    return output;
}

ImageBatchVarShape EraseVarShape(ImageBatchVarShape &input, Tensor &anchor, Tensor &erasing, Tensor &values,
                                 Tensor &imgIdx, bool random, unsigned int seed, std::optional<Stream> pstream)
{
    auto format = input.uniqueFormat();
    if (!format)
    {
        throw EraseError("All images in input must have the same format.");
    }

    ImageBatchVarShape output = CreateSameShapeImageBatch(input, format, input.numImages());

    return EraseVarShapeInto(output, input, anchor, erasing, values, imgIdx, random, seed, pstream);
}

} // namespace

void ExportOpErase(py::module &m)
{
    using namespace pybind11::literals;

    m.def("erase", NvtxTrace("cvcuda.erase", &EraseRegion), "src"_a, "i"_a, "j"_a, "h"_a, "w"_a, "v"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Erases one rectangular region with torchvision-compatible slice and broadcast semantics.

        The logical operation is ``out = src.clone(); out[..., i:i+h, j:j+w] = v``. For HWC/NHWC
        inputs, ``v`` is still interpreted in logical planar order. The value dtype must match
        ``src`` or be float32.

        Args:
            src (cvcuda.Tensor): Input image tensor.
            i (int): Vertical slice start.
            j (int): Horizontal slice start.
            h (int): Vertical slice extent.
            w (int): Horizontal slice extent.
            v (cvcuda.Tensor): Broadcastable value tensor.
            stream (cvcuda.Stream, optional): CUDA stream on which to submit the operation.

        Returns:
            cvcuda.Tensor: A new erased tensor with the same metadata as ``src``.
    )pbdoc");

    m.def("erase_into", NvtxTrace("cvcuda.erase_into", &EraseRegionInto), "dst"_a, "src"_a, "i"_a, "j"_a, "h"_a, "w"_a,
          "v"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Writes the torchvision-compatible single-region erase result into ``dst``.

        Passing the same tensor as ``src`` and ``dst`` performs the operation in place.

        Args:
            dst (cvcuda.Tensor): Output tensor with the same shape, layout, and dtype as ``src``.
            src (cvcuda.Tensor): Input image tensor.
            i (int): Vertical slice start.
            j (int): Horizontal slice start.
            h (int): Vertical slice extent.
            w (int): Horizontal slice extent.
            v (cvcuda.Tensor): Broadcastable value tensor.
            stream (cvcuda.Stream, optional): CUDA stream on which to submit the operation.

        Returns:
            cvcuda.Tensor: ``dst``.
    )pbdoc");

    m.def("erase", NvtxTrace("cvcuda.erase", &Erase), "src"_a, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a,
          py::kw_only(), "random"_a = false, "seed"_a = 0, "stream"_a = nullptr, R"pbdoc(
        Executes the Erase operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            anchor (cvcuda.Tensor): anchor an array of size num_erasing_area that gives the
                             x coordinate and y coordinate of the top left point in the eraseing areas.
            erasing (cvcuda.Tensor): Eraisng an array of size num_erasing_area that gives the widths of the eraseing areas,
                              the heights of the eraseing areas and integers in range 0-15, each of whose bits
                              indicates whether or not the corresponding channel need to be erased.
            values (cvcuda.Tensor): An array of size num_erasing_area*4 that gives the filling value for each erase area.
            imgIdx (cvcuda.Tensor): An array of size num_erasing_area that maps a erase area idx to img idx in the batch.
            random (int, optional): 8-bit integer value for random op.
            seed (int, optional): seed random seed for random filling erase area.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("erase_into", NvtxTrace("cvcuda.erase_into", &EraseInto), "dst"_a, "src"_a, "anchor"_a, "erasing"_a,
          "values"_a, "imgIdx"_a, py::kw_only(), "random"_a = false, "seed"_a = 0, "stream"_a = nullptr, R"pbdoc(
        Executes the Erase operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            anchor (cvcuda.Tensor): anchor an array of size num_erasing_area that gives the
                             x coordinate and y coordinate of the top left point in the eraseing areas.
            erasing (cvcuda.Tensor): Eraisng an array of size num_erasing_area that gives the widths of the eraseing areas,
                              the heights of the eraseing areas and integers in range 0-15, each of whose bits
                              indicates whether or not the corresponding channel need to be erased.
            values (cvcuda.Tensor): An array of size num_erasing_area*4 that gives the filling value for each erase area.
            imgIdx (cvcuda.Tensor): An array of size num_erasing_area that maps a erase area idx to img idx in the batch.
            random (int, optional): 8-bit integer value for random op.
            seed (int, optional): seed random seed for random filling erase area.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("erase", NvtxTrace("cvcuda.erase", &EraseVarShape), "src"_a, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a,
          py::kw_only(), "random"_a = false, "seed"_a = 0, "stream"_a = nullptr, R"pbdoc(
        Executes the Erase operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            anchor (cvcuda.Tensor): anchor an array of size num_erasing_area that gives the
                             x coordinate and y coordinate of the top left point in the eraseing areas.
            erasing (cvcuda.Tensor): Eraisng an array of size num_erasing_area that gives the widths of the eraseing areas,
                              the heights of the eraseing areas and integers in range 0-15, each of whose bits
                              indicates whether or not the corresponding channel need to be erased.
            values (cvcuda.Tensor): An array of size num_erasing_area*4 that gives the filling value for each erase area.
            imgIdx (cvcuda.Tensor): An array of size num_erasing_area that maps a erase area idx to img idx in the batch.
            random (int, optional): 8-bit integer value for random op.
            seed (int, optional): seed random seed for random filling erase area.

            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("erase_into", NvtxTrace("cvcuda.erase_into", &EraseVarShapeInto), "dst"_a, "src"_a, "anchor"_a, "erasing"_a,
          "values"_a, "imgIdx"_a, py::kw_only(), "random"_a = false, "seed"_a = 0, "stream"_a = nullptr, R"pbdoc(
        Executes the Erase operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            anchor (cvcuda.Tensor): anchor an array of size num_erasing_area that gives the
                             x coordinate and y coordinate of the top left point in the eraseing areas.
            erasing (cvcuda.Tensor): Eraisng an array of size num_erasing_area that gives the widths of the eraseing areas,
                              the heights of the eraseing areas and integers in range 0-15, each of whose bits
                              indicates whether or not the corresponding channel need to be erased.
            values (cvcuda.Tensor): An array of size num_erasing_area*4 that gives the filling value for each erase area.
            imgIdx (cvcuda.Tensor): An array of size num_erasing_area that maps a erase area idx to img idx in the batch.
            random (int, optional): 8-bit integer value for random op.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

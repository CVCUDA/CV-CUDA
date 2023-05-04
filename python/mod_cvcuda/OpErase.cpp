/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpErase.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor EraseInto(Tensor &output, Tensor &input, Tensor &anchor, Tensor &erasing, Tensor &values, Tensor &imgIdx,
                 bool random, unsigned int seed, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    if (anchor.layout().rank() != 1 || anchor.layout()[0] != 'N')
    {
        throw std::runtime_error("Layout of anchor must be 'N'.");
    }

    nvcv::TensorShape shape = anchor.shape();

    auto erase = CreateOperator<cvcuda::Erase>((int)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, anchor, erasing, values, imgIdx});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*erase});

    erase->submit(pstream->cudaHandle(), input, output, anchor, erasing, values, imgIdx, random, seed);

    return output;
}

Tensor Erase(Tensor &input, Tensor &anchor, Tensor &erasing, Tensor &values, Tensor &imgIdx, bool random,
             unsigned int seed, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return EraseInto(output, input, anchor, erasing, values, imgIdx, random, seed, pstream);
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
        throw std::runtime_error("Layout of anchor must be 'N'.");
    }

    nvcv::TensorShape shape = anchor.shape();

    auto erase = CreateOperator<cvcuda::Erase>((int)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, anchor, erasing, values, imgIdx});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*erase});

    erase->submit(pstream->cudaHandle(), input, output, anchor, erasing, values, imgIdx, random, seed);

    return output;
}

ImageBatchVarShape EraseVarShape(ImageBatchVarShape &input, Tensor &anchor, Tensor &erasing, Tensor &values,
                                 Tensor &imgIdx, bool random, unsigned int seed, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.numImages());

    auto format = input.uniqueFormat();
    if (!format)
    {
        throw std::runtime_error("All images in input must have the same format.");
    }

    for (auto img = input.begin(); img != input.end(); ++img)
    {
        auto newimg = Image::Create(img->size(), format);
        output.pushBack(newimg);
    }

    return EraseVarShapeInto(output, input, anchor, erasing, values, imgIdx, random, seed, pstream);
}

} // namespace

void ExportOpErase(py::module &m)
{
    using namespace pybind11::literals;

    m.def("erase", &Erase, "src"_a, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a, py::kw_only(), "random"_a = false,
          "seed"_a = 0, "stream"_a = nullptr, R"pbdoc(

        Executes the Erase operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Erase operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
            anchor (Tensor): anchor an array of size num_erasing_area that gives the
                             x coordinate and y coordinate of the top left point in the eraseing areas.
            erasing (Tensor): Eraisng an array of size num_erasing_area that gives the widths of the eraseing areas,
                              the heights of the eraseing areas and integers in range 0-15, each of whose bits
                              indicates whether or not the corresponding channel need to be erased.
            values (Tensor): An array of size num_erasing_area*4 that gives the filling value for each erase area.
            imgIdx (Tensor): An array of size num_erasing_area that maps a erase area idx to img idx in the batch.
            random (int8 , optional): random an value for random op.
            seed (int ,optional): seed random seed for random filling erase area.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("erase_into", &EraseInto, "dst"_a, "src"_a, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a, py::kw_only(),
          "random"_a = false, "seed"_a = 0, "stream"_a = nullptr, R"pbdoc(

        Executes the Erase operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Erase operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output tensor to store the result of the operation.
            src (Tensor): Input tensor containing one or more images.
            anchor (Tensor): anchor an array of size num_erasing_area that gives the
                             x coordinate and y coordinate of the top left point in the eraseing areas.
            erasing (Tensor): Eraisng an array of size num_erasing_area that gives the widths of the eraseing areas,
                              the heights of the eraseing areas and integers in range 0-15, each of whose bits
                              indicates whether or not the corresponding channel need to be erased.
            values (Tensor): An array of size num_erasing_area*4 that gives the filling value for each erase area.
            imgIdx (Tensor): An array of size num_erasing_area that maps a erase area idx to img idx in the batch.
            random (int8 , optional): random an value for random op.
            seed (int ,optional): seed random seed for random filling erase area.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("erase", &EraseVarShape, "src"_a, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a, py::kw_only(),
          "random"_a = false, "seed"_a = 0, "stream"_a = nullptr, R"pbdoc(

        Executes the Erase operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Erase operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            anchor (Tensor): anchor an array of size num_erasing_area that gives the
                             x coordinate and y coordinate of the top left point in the eraseing areas.
            erasing (Tensor): Eraisng an array of size num_erasing_area that gives the widths of the eraseing areas,
                              the heights of the eraseing areas and integers in range 0-15, each of whose bits
                              indicates whether or not the corresponding channel need to be erased.
            values (Tensor): An array of size num_erasing_area*4 that gives the filling value for each erase area.
            imgIdx (Tensor): An array of size num_erasing_area that maps a erase area idx to img idx in the batch.
            random (int8 , optional): random an value for random op.
            seed (int ,optional): seed random seed for random filling erase area.

            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("erase_into", &EraseVarShapeInto, "dst"_a, "src"_a, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a,
          py::kw_only(), "random"_a = false, "seed"_a = 0, "stream"_a = nullptr, R"pbdoc(

        Executes the Erase operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Erase operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            dst (ImageBatchVarShape): Output image batch containing the result of the operation.
            anchor (Tensor): anchor an array of size num_erasing_area that gives the
                             x coordinate and y coordinate of the top left point in the eraseing areas.
            erasing (Tensor): Eraisng an array of size num_erasing_area that gives the widths of the eraseing areas,
                              the heights of the eraseing areas and integers in range 0-15, each of whose bits
                              indicates whether or not the corresponding channel need to be erased.
            values (Tensor): An array of size num_erasing_area*4 that gives the filling value for each erase area.
            imgIdx (Tensor): An array of size num_erasing_area that maps a erase area idx to img idx in the batch.
            random (int8 , optional): random an value for random op.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

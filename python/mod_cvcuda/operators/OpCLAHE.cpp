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

#include "Operators.hpp"
#include "VarShapeUtils.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpCLAHE.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

class CLAHEError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor CLAHEInto(Tensor &output, Tensor &input, float clipLimit, const std::tuple<int32_t, int32_t> &tileGridSize,
                 std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    // HWC inputs (rank 3) have no N dim, so shape[0] is H — fall back to 1.
    int32_t batchSize = (input.shape().size() == 4) ? (int32_t)input.shape()[0] : 1;
    auto    tx        = std::get<0>(tileGridSize);
    auto    ty        = std::get<1>(tileGridSize);
    auto    op        = CreateOperator<cvcuda::CLAHE>(batchSize, tx, ty);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    guard.run([&op, &pstream, &input, &output, &clipLimit]()
              { op->submit(pstream->cudaHandle(), input, output, clipLimit); });
    return output;
}

Tensor CLAHE(Tensor &input, float clipLimit, const std::tuple<int32_t, int32_t> &tileGridSize,
             std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());
    return CLAHEInto(output, input, clipLimit, tileGridSize, pstream);
}

ImageBatchVarShape CLAHEVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, float clipLimit,
                                     const std::tuple<int32_t, int32_t> &tileGridSize, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto tx = std::get<0>(tileGridSize);
    auto ty = std::get<1>(tileGridSize);
    auto op = CreateOperator<cvcuda::CLAHE>(input.capacity(), tx, ty);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    guard.run([&op, &pstream, &input, &output, &clipLimit]()
              { op->submit(pstream->cudaHandle(), input, output, clipLimit); });
    return output;
}

ImageBatchVarShape CLAHEVarShape(ImageBatchVarShape &input, float clipLimit,
                                 const std::tuple<int32_t, int32_t> &tileGridSize, std::optional<Stream> pstream)
{
    auto format = input.uniqueFormat();
    if (!format)
    {
        throw CLAHEError("All images in input must have the same format.");
    }

    ImageBatchVarShape output = CreateSameShapeImageBatch(input, format, input.numImages());

    return CLAHEVarShapeInto(output, input, clipLimit, tileGridSize, pstream);
}

} // namespace

void ExportOpCLAHE(py::module &m)
{
    using namespace pybind11::literals;

    m.def("clahe", NvtxTrace("cvcuda.clahe", &CLAHE), "src"_a, "clip_limit"_a = 40.0,
          "tile_grid_size"_a = std::tuple<int32_t, int32_t>{8, 8}, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Contrast Limited Adaptive Histogram Equalization (CLAHE) operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            clip_limit (float, optional): The clip limit for the CLAHE operation (default: 40.0).
            tile_grid_size (Tuple[int, int], optional): The tile grid size for the CLAHE operation (default: (8, 8)).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor containing the result of the CLAHE operation.

    )pbdoc");

    m.def("clahe_into", NvtxTrace("cvcuda.clahe_into", &CLAHEInto), "dst"_a, "src"_a, "clip_limit"_a = 40.0,
          "tile_grid_size"_a = std::tuple<int32_t, int32_t>{8, 8}, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Contrast Limited Adaptive Histogram Equalization (CLAHE) operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor containing the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            clip_limit (float, optional): The clip limit for the CLAHE operation (default: 40.0).
            tile_grid_size (Tuple[int, int], optional): The tile grid size for the CLAHE operation (default: (8, 8)).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor ``dst``.

    )pbdoc");

    m.def("clahe", NvtxTrace("cvcuda.clahe", &CLAHEVarShape), "src"_a, "clip_limit"_a = 40.0,
          "tile_grid_size"_a = std::tuple<int32_t, int32_t>{8, 8}, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Contrast Limited Adaptive Histogram Equalization (CLAHE) operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            clip_limit (float, optional): The clip limit for the CLAHE operation (default: 40.0).
            tile_grid_size (Tuple[int, int], optional): The tile grid size for the CLAHE operation (default: (8, 8)).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch containing the result of the CLAHE operation.

    )pbdoc");

    m.def("clahe_into", NvtxTrace("cvcuda.clahe_into", &CLAHEVarShapeInto), "dst"_a, "src"_a, "clip_limit"_a = 40.0,
          "tile_grid_size"_a = std::tuple<int32_t, int32_t>{8, 8}, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Contrast Limited Adaptive Histogram Equalization (CLAHE) operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            clip_limit (float, optional): The clip limit for the CLAHE operation (default: 40.0).
            tile_grid_size (Tuple[int, int], optional): The tile grid size for the CLAHE operation (default: (8, 8)).
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch ``dst``.

    )pbdoc");
}

} // namespace cvcudapy

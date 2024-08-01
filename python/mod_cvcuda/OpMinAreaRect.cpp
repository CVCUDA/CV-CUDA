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
#include <cvcuda/OpMinAreaRect.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {
Tensor MinAreaRectInto(Tensor &output, Tensor &input, Tensor &numPointsInContour, const int totalContours,
                       std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto minAreaRect = CreateOperator<cvcuda::MinAreaRect>(totalContours);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, numPointsInContour});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*minAreaRect});

    minAreaRect->submit(pstream->cudaHandle(), input, output, numPointsInContour, totalContours);

    return std::move(output);
}

Tensor MinAreaRect(Tensor &input, Tensor &numPointsInContour, const int totalContours, std::optional<Stream> pstream)
{
    const auto &srcShape                = input.shape();
    const auto &numPointsInContourShape = numPointsInContour.shape();
    if ((srcShape.rank() - 1) != numPointsInContourShape.rank())
    {
        throw std::runtime_error("Input src rank must 1 greater than numPointsInContourShape tensors rank");
    }
    if (srcShape.shape()[0] != numPointsInContourShape.shape()[1])
    {
        throw std::runtime_error("Input src and numPointsInContourShape must have same batch size");
    }

    Shape dstShape(2);
    dstShape[0] = srcShape.shape()[0];
    dstShape[1] = 8;

    Tensor output = Tensor::Create(dstShape, nvcv::TYPE_F32, numPointsInContour.layout());

    return MinAreaRectInto(output, input, numPointsInContour, totalContours, pstream);
}

} // namespace

void ExportOpMinAreaRect(py::module &m)
{
    using namespace pybind11::literals;

    m.def("minarearect", &MinAreaRect, "src"_a, "numPointsInContour"_a, "totalContours"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the Min Area Rect operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Min Area Rect operator
            for more details and usage examples.

        Args:
            src (nvcv.Tensor): Input tensor containing one or more contours.src[i,j,k] is the set of contours
                where i ranges from 0 to batch-1, j ranges from 0 to max number of points in cotours
                k is the coordinate of each points which is in [0,1]
            numPointsInContour (nvcv.Tensor): Input tensor containing the number of points in each input contours.
            totalContours (int): Number of input contours
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor of rotated bounding boxes.The output will give 4 points' cooridinate(x,y)
            of each contour's minimum rotated bounding boxes

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("minarearect_into", &MinAreaRectInto, "dst"_a, "src"_a, "numPointsInContour"_a, "totalContours"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Min Area Rect operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Min Area Rect operator
            for more details and usage examples.

        Args:
            dst (nvcv.Tensor): Output tensor will give 4 points' cooridinate(x,y)
                of each contour's minimum rotated bounding boxes
            src (nvcv.Tensor): Input tensor containing one or more contours. src[i,j,k] is the set of contours
                where i ranges from 0 to batch-1, j ranges from 0 to max number of points in cotours
                k is the coordinate of each points which is in [0,1]
            numPointsInContour (nvcv.Tensor): Input tensor containing the number of points in each input contours.
            totalContours (int): Number of input contours
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

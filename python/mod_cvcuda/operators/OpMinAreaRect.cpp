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
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpMinAreaRect.hpp>

#include <array>
#include <stdexcept>

namespace cvcudapy {

namespace {
using native::Stream;
using native::Tensor;

class MinAreaRectError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor &MinAreaRectInto(Tensor &output, Tensor &input, Tensor &numPointsInContour, const int totalContours,
                        std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto minAreaRect = CreateOperator<cvcuda::MinAreaRect>(totalContours);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, numPointsInContour});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*minAreaRect});

    guard.run(
        [&minAreaRect, &stream, &input, &output, &numPointsInContour, &totalContours]() {
            minAreaRect->submit(stream.handle(), input.impl(), output.impl(), numPointsInContour.impl(), totalContours);
        });

    return output;
}

std::shared_ptr<Tensor> MinAreaRect(Tensor &input, Tensor &numPointsInContour, const int totalContours,
                                    std::optional<Stream *> pstream)
{
    const nvcv::TensorShape &srcShape                = input.impl().shape();
    const nvcv::TensorShape &numPointsInContourShape = numPointsInContour.impl().shape();
    if (numPointsInContourShape.rank() < 2)
    {
        throw MinAreaRectError("Input numPointsInContour tensor must have rank at least 2");
    }
    if ((srcShape.rank() - 1) != numPointsInContourShape.rank())
    {
        throw MinAreaRectError("Input src rank must 1 greater than numPointsInContourShape tensors rank");
    }
    if (srcShape[0] != numPointsInContourShape[1])
    {
        throw MinAreaRectError("Input src and numPointsInContourShape must have same batch size");
    }

    const std::array<int64_t, 2> dstShape{srcShape[0], 8};

    auto output = native::CreateTensor(
        {dstShape.data(), static_cast<int32_t>(dstShape.size()), numPointsInContourShape.layout()}, nvcv::TYPE_F32);

    MinAreaRectInto(*output, input, numPointsInContour, totalContours, pstream);
    return output;
}

} // namespace

void ExportOpMinAreaRect(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("minarearect", NvtxTrace("cvcuda.minarearect", &MinAreaRect), "src"_a, "numPointsInContour"_a,
          "totalContours"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Min Area Rect operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more contours.src[i,j,k] is the set of contours
                where i ranges from 0 to batch-1, j ranges from 0 to max number of points in cotours
                k is the coordinate of each points which is in [0,1]
            numPointsInContour (cvcuda.Tensor): Input tensor containing the number of points in each input contours.
            totalContours (int): Number of input contours. Must be equal to src.shape[0].
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor of rotated bounding boxes.The output will give 4 points' cooridinate(x,y)
            of each contour's minimum rotated bounding boxes

    )pbdoc");

    m.def("minarearect_into", NvtxTrace("cvcuda.minarearect_into", &MinAreaRectInto), "dst"_a, "src"_a,
          "numPointsInContour"_a, "totalContours"_a, py::kw_only(), "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(
        Executes the Min Area Rect operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor will give 4 points' cooridinate(x,y)
                of each contour's minimum rotated bounding boxes
            src (cvcuda.Tensor): Input tensor containing one or more contours. src[i,j,k] is the set of contours
                where i ranges from 0 to batch-1, j ranges from 0 to max number of points in cotours
                k is the coordinate of each points which is in [0,1]
            numPointsInContour (cvcuda.Tensor): Input tensor containing the number of points in each input contours.
            totalContours (int): Number of input contours. Must be equal to src.shape[0].
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

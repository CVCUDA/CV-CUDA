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
#include <cvcuda/OpFindContours.hpp>
#include <cvcuda/Types.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor FindContoursInto(Tensor &points, Tensor &numPoints, Tensor &input, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    nvcv::Size2D size{static_cast<int>(input.shape()[2]), static_cast<int>(input.shape()[1])};
    auto         findContours = CreateOperator<cvcuda::FindContours>(size, static_cast<int32_t>(input.shape()[0]));

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {points});
    guard.add(LockMode::LOCK_WRITE, {numPoints});
    guard.add(LockMode::LOCK_WRITE, {*findContours});

    findContours->submit(pstream->cudaHandle(), input, points, numPoints);

    return points;
}

Tensor FindContours(Tensor &input, std::optional<Stream> pstream)
{
    auto pointShape = nvcv::TensorShape{
        {input.shape()[0], cvcuda::FindContours::MAX_TOTAL_POINTS, 2},
        nvcv::TENSOR_NHW
    };
    Tensor points = Tensor::Create(pointShape, nvcv::TYPE_S32);

    auto countShape = nvcv::TensorShape{
        {input.shape()[0], cvcuda::FindContours::MAX_NUM_CONTOURS},
        nvcv::TENSOR_NW
    };
    Tensor numPoints = Tensor::Create(countShape, nvcv::TYPE_U32);

    return FindContoursInto(points, numPoints, input, pstream);
}

} // namespace

void ExportOpFindContours(py::module &m)
{
    using namespace pybind11::literals;
    py::options options;
    options.disable_function_signatures();

    m.def("find_contours", &FindContours, "image"_a, "stream"_a = nullptr, R"pbdoc(

        cvcuda.find_contours(src : nvcv.Tensor, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor
        Executes the FindContours operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the FindContours operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("find_contours_into", &FindContoursInto, "points"_a, "num_points"_a, "src"_a, "stream"_a = nullptr, R"pbdoc(

        cvcuda.find_contours_into(points : nvcv.Tensor, num_points : nvcv.Tensor, src : Tensor, stream: Optional[nvcv.cuda.Stream] = None)
        Executes the FindContours operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the FindContours operator
            for more details and usage examples.

        Args:
            points (Tensor): Output tensor to store the coordinates of each contour point.
            num_points (Tensor): Output tensor to store the number of points in a contour.
            src (Tensor): Input tensor containing one or more images.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

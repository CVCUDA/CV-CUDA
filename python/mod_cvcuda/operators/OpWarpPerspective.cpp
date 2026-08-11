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
#include <cvcuda/OpWarpPerspective.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

class WarpPerspectiveError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor WarpPerspectiveInto(Tensor &output, Tensor &input, const pyarray &xform, const int32_t flags,
                           const NVCVBorderType borderMode, const pyarray &borderValue, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    float4 bValue = GetFloat4FromPyArray(borderValue);

    if (size_t xformDims = xform.ndim(); !(xformDims == 2 && xform.shape(0) == 3 && xform.shape(1) == 3))
    {
        throw WarpPerspectiveError(
            util::ConcatString("Details of transformation matrix: nDim == 2, shape == (3, 3) but current is '",
                               xformDims, "', ('", xform.shape(0), "', '", xform.shape(1), "') respectively"));
    }

    NVCVPerspectiveTransform xformOutput;
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            xformOutput[i * 3 + j] = *xform.data(i, j);
        }
    }

    auto warpPerspective = CreateOperator<cvcuda::WarpPerspective>(0);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*warpPerspective});

    guard.run(
        [&warpPerspective, &pstream, &input, &output, &xformOutput, &flags, &borderMode, &bValue]()
        { warpPerspective->submit(pstream->cudaHandle(), input, output, xformOutput, flags, borderMode, bValue); });

    return output;
}

Tensor WarpPerspective(Tensor &input, const pyarray &xform, const int32_t flags, const NVCVBorderType borderMode,
                       const pyarray &borderValue, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return WarpPerspectiveInto(output, input, xform, flags, borderMode, borderValue, pstream);
}

ImageBatchVarShape WarpPerspectiveVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &xform,
                                               const int32_t flags, const NVCVBorderType borderMode,
                                               const pyarray &borderValue, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    size_t bValueSize = borderValue.size();
    if (size_t bValueDims = borderValue.ndim(); bValueSize > 4 || bValueDims != 1)
    {
        throw py::value_error(
            util::ConcatString("Channels of borderValue should <= 4 and dimension should be 2, current is '",
                               bValueSize, "', '", bValueDims, "' respectively"));
    }
    float4 bValue;
    for (int i = 0; i < 4; i++)
    {
        const auto valueIdx               = static_cast<size_t>(i);
        nvcv::cuda::GetElement(bValue, i) = bValueSize > valueIdx ? *borderValue.data(valueIdx) : 0.f;
    }

    auto warpPerspective = CreateOperator<cvcuda::WarpPerspective>(input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, xform});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*warpPerspective});

    guard.run([&warpPerspective, &pstream, &input, &output, &xform, &flags, &borderMode, &bValue]()
              { warpPerspective->submit(pstream->cudaHandle(), input, output, xform, flags, borderMode, bValue); });

    return output;
}

ImageBatchVarShape WarpPerspectiveVarShape(ImageBatchVarShape &input, Tensor &xform, const int32_t flags,
                                           const NVCVBorderType borderMode, const pyarray &borderValue,
                                           std::optional<Stream> pstream)
{
    ImageBatchVarShape output = CreateSameShapeImageBatch(input);

    return WarpPerspectiveVarShapeInto(output, input, xform, flags, borderMode, borderValue, pstream);
}

} // namespace

void ExportOpWarpPerspective(py::module &m)
{
    using namespace pybind11::literals;

    m.def("warp_perspective", NvtxTrace("cvcuda.warp_perspective", &WarpPerspective), "src"_a, "xform"_a, "flags"_a,
          py::kw_only(), "border_mode"_a, "border_value"_a, "stream"_a = nullptr, R"pbdoc(
        Executes the Warp Perspective operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            xform (numpy.ndarray: 3x3 perspective transformation matrix.
            flags (int): Combination of interpolation methods(cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR or cvcuda.Interp.CUBIC)
                and the optional flag cvcuda.Interp.WARP_INVERSE_MAP, that sets trans_matrix as the inverse transformation.
            border_mode (cvcuda.Border): pixel extrapolation method (cvcuda.Border.CONSTANT or
                cvcuda.Border.REPLICATE).
            border_value (numpy.ndarray): Used to specify values for a constant border, should be a size <= 4 and dim of 1,
                where the values specify the border color for each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("warp_perspective_into", NvtxTrace("cvcuda.warp_perspective_into", &WarpPerspectiveInto), "dst"_a, "src"_a,
          "xform"_a, "flags"_a, py::kw_only(), "border_mode"_a, "border_value"_a, "stream"_a = nullptr, R"pbdoc(
        Executes the Warp Perspective operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            xform (numpy.ndarray: 3x3 perspective transformation matrix.
            flags (int): Combination of interpolation methods(cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR or cvcuda.Interp.CUBIC)
                and the optional flag cvcuda.Interp.WARP_INVERSE_MAP, that sets trans_matrix as the inverse transformation.
            border_mode (cvcuda.Border): pixel extrapolation method (cvcuda.Border.CONSTANT or
                cvcuda.Border.REPLICATE).
            border_value (numpy.ndarray): Used to specify values for a constant border, should be a size <= 4 and dim of 1,
                where the values specify the border color for each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("warp_perspective", NvtxTrace("cvcuda.warp_perspective", &WarpPerspectiveVarShape), "src"_a, "xform"_a,
          "flags"_a, py::kw_only(), "border_mode"_a, "border_value"_a, "stream"_a = nullptr, R"pbdoc(
        Executes the Warp Perspective operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            xform (cvcuda.Tensor): 3x3 perspective transformation matrix for each image in the batch.
            flags (int): Combination of interpolation methods(cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR or cvcuda.Interp.CUBIC)
                and the optional flag cvcuda.Interp.WARP_INVERSE_MAP, that sets trans_matrix as the inverse transformation.
            border_mode (cvcuda.Border): pixel extrapolation method (cvcuda.Border.CONSTANT or
                cvcuda.Border.REPLICATE).
            border_value (numpy.ndarray): Used to specify values for a constant border, must be a size <= 4 and dim of 1,
                where the values specify the border color for each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("warp_perspective_into", NvtxTrace("cvcuda.warp_perspective_into", &WarpPerspectiveVarShapeInto), "dst"_a,
          "src"_a, "xform"_a, "flags"_a, py::kw_only(), "border_mode"_a, "border_value"_a, "stream"_a = nullptr,
          R"pbdoc(
        Executes the Warp Perspective operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            xform (cvcuda.Tensor): 3x3 perspective transformation matrix for each image in the batch.
            flags (int): Combination of interpolation methods(cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR or cvcuda.Interp.CUBIC)
                and the optional flag cvcuda.Interp.WARP_INVERSE_MAP, that sets trans_matrix as the inverse transformation.
            border_mode (cvcuda.Border): pixel extrapolation method (cvcuda.Border.CONSTANT or
                cvcuda.Border.REPLICATE).
            border_value (numpy.ndarray): Used to specify values for a constant border, must be a size <= 4 and dim of 1,
                where the values specify the border color for each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

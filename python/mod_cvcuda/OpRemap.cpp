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
#include <cvcuda/OpRemap.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {

// Tensor into -----------------------------------------------------------------

Tensor RemapInto(Tensor &dst, Tensor &src, Tensor &map, NVCVInterpolationType srcInterp,
                 NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType, bool alignCorners,
                 NVCVBorderType borderMode, const pyarray &borderValue, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    float4 bValue = GetFloat4FromPyArray(borderValue);

    auto op = CreateOperator<cvcuda::Remap>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {src, map});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    op->submit(pstream->cudaHandle(), src, dst, map, srcInterp, mapInterp, mapValueType, alignCorners, borderMode,
               bValue);

    return std::move(dst);
}

Tensor Remap(Tensor &src, Tensor &map, NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp,
             NVCVRemapMapValueType mapValueType, bool alignCorners, NVCVBorderType borderMode,
             const pyarray &borderValue, std::optional<Stream> pstream)
{
    const auto &srcShape = src.shape();
    const auto &mapShape = map.shape();

    if (srcShape.rank() != mapShape.rank())
    {
        throw std::runtime_error("Input src and map tensors must have the same rank");
    }

    Shape dstShape = nvcvpy::CreateShape(srcShape);

    if (mapValueType == NVCV_REMAP_ABSOLUTE || mapValueType == NVCV_REMAP_ABSOLUTE_NORMALIZED)
    {
        if (src.layout() == nvcv::TENSOR_HWC)
        {
            dstShape[0] = mapShape[0];
            dstShape[1] = mapShape[1];
        }
        else if (src.layout() == nvcv::TENSOR_NHWC)
        {
            dstShape[1] = mapShape[1];
            dstShape[2] = mapShape[2];
        }
        else
        {
            throw std::runtime_error("Input src tensor must have either HWC or NHWC layout");
        }
    }

    Tensor dst = Tensor::Create(dstShape, src.dtype(), src.layout());

    return RemapInto(dst, src, map, srcInterp, mapInterp, mapValueType, alignCorners, borderMode, borderValue, pstream);
}

// VarShape into ---------------------------------------------------------------

ImageBatchVarShape VarShapeRemapInto(ImageBatchVarShape &dst, ImageBatchVarShape &src, Tensor &map,
                                     NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp,
                                     NVCVRemapMapValueType mapValueType, bool alignCorners, NVCVBorderType borderMode,
                                     const pyarray &borderValue, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    float4 bValue = GetFloat4FromPyArray(borderValue);

    auto op = CreateOperator<cvcuda::Remap>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {src, map});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    op->submit(pstream->cudaHandle(), src, dst, map, srcInterp, mapInterp, mapValueType, alignCorners, borderMode,
               bValue);

    return std::move(dst);
}

ImageBatchVarShape VarShapeRemap(ImageBatchVarShape &src, Tensor &map, NVCVInterpolationType srcInterp,
                                 NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType, bool alignCorners,
                                 NVCVBorderType borderMode, const pyarray &borderValue, std::optional<Stream> pstream)
{
    ImageBatchVarShape dst = ImageBatchVarShape::Create(src.capacity());

    nvcv::Size2D mapSize;

    if (mapValueType == NVCV_REMAP_ABSOLUTE || mapValueType == NVCV_REMAP_ABSOLUTE_NORMALIZED)
    {
        auto mapAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(map.exportData());
        if (!mapAccess)
        {
            throw std::runtime_error("Incompatible map tensor layout");
        }

        mapSize.w = mapAccess->numCols();
        mapSize.h = mapAccess->numRows();
    }

    for (int i = 0; i < src.numImages(); ++i)
    {
        if (mapValueType == NVCV_REMAP_ABSOLUTE || mapValueType == NVCV_REMAP_ABSOLUTE_NORMALIZED)
        {
            dst.pushBack(Image::Create(mapSize, src[i].format()));
        }
        else
        {
            dst.pushBack(Image::Create(src[i].size(), src[i].format()));
        }
    }

    return VarShapeRemapInto(dst, src, map, srcInterp, mapInterp, mapValueType, alignCorners, borderMode, borderValue,
                             pstream);
}

} // namespace

void ExportOpRemap(py::module &m)
{
    using namespace pybind11::literals;

    m.def("remap", &Remap, "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST, "map_interp"_a = NVCV_INTERP_NEAREST,
          "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false, "border"_a = NVCV_BORDER_CONSTANT,
          "border_value"_a = pyarray{}, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.remap(src: nvcv.Tensor, map: nvcv.Tensor, src_interp: cvcuda.Interp = cvcuda.Interp.NEAREST, map_interp: cvcuda.Interp = cvcuda.Interp.NEAREST, map_type: cvcuda.Remap = cvcuda.Remap.ABSOLUTE, align_corners: bool = False, border: cvcuda.Border = cvcuda.Border.CONSTANT, border_value: numpy.ndarray = np.ndarray((0,)), stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor

        Executes the Warp Perspective operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Remap operator for more details and usage
            examples.

        Args:
            src (nvcv.Tensor): Input tensor.
            src_interp (cvcuda.Interp, optional): Interpolation type used when fetching values
                from the source image.
            map_interp (cvcuda.Interp, optional): Interpolation type used when fetching values
                from the map tensor.
            map_type (cvcuda.Remap, optional): This determines how the values inside the map are
                interpreted.  If it is cvcuda.Remap.ABSOLUTE the map values are absolute,
                denormalized positions in the input tensor to fetch values from.  If it is
                cvcuda.Remap.ABSOLUTE_NORMALIZED the map values are absolute, normalized
                positions in [-1, 1] range to fetch values from the input tensor in a resolution
                agnostic way.  If it is cvcuda.Remap.RELATIVE_NORMALIZED the map values are
                relative, normalized offsets to be applied to each output position to fetch values
                from the input tensor, also resolution agnostic.
            align_corners (bool, optional): The remap operation from output to input via the map
                is done in the floating-point domain. If ``True``, they are aligned by the center
                points of their corner pixels. Otherwise, they are aligned by the corner points of
                their corner pixels.
            border (cvcuda.Border, optional): pixel extrapolation method (cvcuda.Border.CONSTANT,
                cvcuda.Border.REPLICATE, cvcuda.Border.REFLECT, cvcuda.Border.REFLECT_101, or
                cvcuda.Border.WRAP).
            border_value (numpy.ndarray, optional): Used to specify values for a constant border,
                should have size <= 4 and dim of 1, where the values specify the border color for
                each color channel.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
    m.def("remap_into", &RemapInto, "dst"_a, "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST,
          "map_interp"_a = NVCV_INTERP_NEAREST, "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false,
          "border"_a = NVCV_BORDER_CONSTANT, "border_value"_a = pyarray{}, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.remap_into(dst: nvcv.Tensor, src: nvcv.Tensor, map: nvcv.Tensor, src_interp: cvcuda.Interp = cvcuda.Interp.NEAREST, map_interp: cvcuda.Interp = cvcuda.Interp.NEAREST, map_type: cvcuda.Remap = cvcuda.Remap.ABSOLUTE, align_corners: bool = False, border: cvcuda.Border = cvcuda.Border.CONSTANT, border_value: numpy.ndarray = np.ndarray((0,)), stream: Optional[nvcv.cuda.Stream] = None)

        Executes the Warp Perspective operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Remap operator for more details and usage
            examples.

        Args:
            dst (nvcv.Tensor): Output tensor.
            src (nvcv.Tensor): Input tensor.
            src_interp (cvcuda.Interp, optional): Interpolation type used when fetching values
                from the source image.
            map_interp (cvcuda.Interp, optional): Interpolation type used when fetching values
                from the map tensor.
            map_type (cvcuda.Remap, optional): This determines how the values inside the map are
                interpreted.  If it is cvcuda.Remap.ABSOLUTE the map values are absolute,
                denormalized positions in the input tensor to fetch values from.  If it is
                cvcuda.Remap.ABSOLUTE_NORMALIZED the map values are absolute, normalized
                positions in [-1, 1] range to fetch values from the input tensor in a resolution
                agnostic way.  If it is cvcuda.Remap.RELATIVE_NORMALIZED the map values are
                relative, normalized offsets to be applied to each output position to fetch values
                from the input tensor, also resolution agnostic.
            align_corners (bool, optional): The remap operation from output to input via the map
                is done in the floating-point domain. If ``True``, they are aligned by the center
                points of their corner pixels. Otherwise, they are aligned by the corner points of
                their corner pixels.
            border (cvcuda.Border, optional): pixel extrapolation method (cvcuda.Border.CONSTANT,
                cvcuda.Border.REPLICATE, cvcuda.Border.REFLECT, cvcuda.Border.REFLECT_101, or
                cvcuda.Border.WRAP).
            border_value (numpy.ndarray, optional): Used to specify values for a constant border,
                should have size <= 4 and dim of 1, where the values specify the border color for
                each color channel.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
    m.def("remap", &VarShapeRemap, "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST,
          "map_interp"_a = NVCV_INTERP_NEAREST, "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false,
          "border"_a = NVCV_BORDER_CONSTANT, "border_value"_a = pyarray{}, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.remap(src: nvcv.ImageBatchVarShape, map: nvcv.Tensor, src_interp: cvcuda.Interp = cvcuda.Interp.NEAREST, map_interp: cvcuda.Interp = cvcuda.Interp.NEAREST, map_type: cvcuda.Remap = cvcuda.Remap.ABSOLUTE, align_corners: bool = False, border: cvcuda.Border = cvcuda.Border.CONSTANT, border_value: numpy.ndarray = np.ndarray((0,)), stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.ImageBatchVarShape

        Executes the Warp Perspective operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Remap operator for more details and usage
            examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch.
            src_interp (cvcuda.Interp, optional): Interpolation type used when fetching values
                from the source image.
            map_interp (cvcuda.Interp, optional): Interpolation type used when fetching values
                from the map tensor.
            map_type (cvcuda.Remap, optional): This determines how the values inside the map are
                interpreted.  If it is cvcuda.Remap.ABSOLUTE the map values are absolute,
                denormalized positions in the input tensor to fetch values from.  If it is
                cvcuda.Remap.ABSOLUTE_NORMALIZED the map values are absolute, normalized
                positions in [-1, 1] range to fetch values from the input tensor in a resolution
                agnostic way.  If it is cvcuda.Remap.RELATIVE_NORMALIZED the map values are
                relative, normalized offsets to be applied to each output position to fetch values
                from the input tensor, also resolution agnostic.
            align_corners (bool, optional): The remap operation from output to input via the map
                is done in the floating-point domain. If ``True``, they are aligned by the center
                points of their corner pixels. Otherwise, they are aligned by the corner points of
                their corner pixels.
            border (cvcuda.Border, optional): pixel extrapolation method (cvcuda.Border.CONSTANT,
                cvcuda.Border.REPLICATE, cvcuda.Border.REFLECT, cvcuda.Border.REFLECT_101, or
                cvcuda.Border.WRAP).
            border_value (numpy.ndarray, optional): Used to specify values for a constant border,
                should have size <= 4 and dim of 1, where the values specify the border color for
                each color channel.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
    m.def("remap_into", &VarShapeRemapInto, "dst"_a, "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST,
          "map_interp"_a = NVCV_INTERP_NEAREST, "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false,
          "border"_a = NVCV_BORDER_CONSTANT, "border_value"_a = pyarray{}, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        cvcuda.remap_into(dst: nvcv.ImageBatchVarShape, src: nvcv.ImageBatchVarShape, map: nvcv.Tensor, src_interp: cvcuda.Interp = cvcuda.Interp.NEAREST, map_interp: cvcuda.Interp = cvcuda.Interp.NEAREST, map_type: cvcuda.Remap = cvcuda.Remap.ABSOLUTE, align_corners: bool = False, border: cvcuda.Border = cvcuda.Border.CONSTANT, border_value: numpy.ndarray = np.ndarray((0,)), stream: Optional[nvcv.cuda.Stream] = None)

        Executes the Warp Perspective operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Remap operator for more details and usage
            examples.

        Args:
            dst (nvcv.ImageBatchVarShape): Output image batch.
            src (nvcv.ImageBatchVarShape): Input image batch.
            src_interp (cvcuda.Interp, optional): Interpolation type used when fetching values
                from the source image.
            map_interp (cvcuda.Interp, optional): Interpolation type used when fetching values
                from the map tensor.
            map_type (cvcuda.Remap, optional): This determines how the values inside the map are
                interpreted.  If it is cvcuda.Remap.ABSOLUTE the map values are absolute,
                denormalized positions in the input tensor to fetch values from.  If it is
                cvcuda.Remap.ABSOLUTE_NORMALIZED the map values are absolute, normalized
                positions in [-1, 1] range to fetch values from the input tensor in a resolution
                agnostic way.  If it is cvcuda.Remap.RELATIVE_NORMALIZED the map values are
                relative, normalized offsets to be applied to each output position to fetch values
                from the input tensor, also resolution agnostic.
            align_corners (bool, optional): The remap operation from output to input via the map
                is done in the floating-point domain. If ``True``, they are aligned by the center
                points of their corner pixels. Otherwise, they are aligned by the corner points of
                their corner pixels.
            border (cvcuda.Border, optional): pixel extrapolation method (cvcuda.Border.CONSTANT,
                cvcuda.Border.REPLICATE, cvcuda.Border.REFLECT, cvcuda.Border.REFLECT_101, or
                cvcuda.Border.WRAP).
            border_value (numpy.ndarray, optional): Used to specify values for a constant border,
                should have size <= 4 and dim of 1, where the values specify the border color for
                each color channel.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA
            operator.
    )pbdoc");
}

} // namespace cvcudapy

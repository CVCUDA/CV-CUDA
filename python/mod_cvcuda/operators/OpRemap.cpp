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
#include <common/String.hpp>
#include <cvcuda/OpRemap.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <stdexcept>

namespace cvcudapy {

namespace {
using native::Image;
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

class RemapError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Tensor into -----------------------------------------------------------------

Tensor &RemapInto(Tensor &dst, Tensor &src, Tensor &map, NVCVInterpolationType srcInterp,
                  NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType, bool alignCorners,
                  NVCVBorderType borderMode, const pyarray &borderValue, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    float4 bValue = GetFloat4FromPyArray(borderValue);

    auto op = CreateOperator<cvcuda::Remap>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {src, map});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run(
        [&op, &stream, &src, &dst, &map, &srcInterp, &mapInterp, &mapValueType, &alignCorners, &borderMode, &bValue]()
        {
            op->submit(stream.handle(), src.impl(), dst.impl(), map.impl(), srcInterp, mapInterp, mapValueType,
                       alignCorners, borderMode, bValue);
        });

    return dst;
}

std::shared_ptr<Tensor> Remap(Tensor &src, Tensor &map, NVCVInterpolationType srcInterp,
                              NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType, bool alignCorners,
                              NVCVBorderType borderMode, const pyarray &borderValue, std::optional<Stream *> pstream)
{
    const nvcv::TensorShape &srcShape = src.impl().shape();
    const nvcv::TensorShape &mapShape = map.impl().shape();

    if (srcShape.rank() != mapShape.rank())
    {
        throw RemapError("Input src and map tensors must have the same rank");
    }

    Shape dstShape = nvcvpy::CreateShape(srcShape);

    if (mapValueType == NVCV_REMAP_ABSOLUTE || mapValueType == NVCV_REMAP_ABSOLUTE_NORMALIZED)
    {
        if (srcShape.layout() == nvcv::TENSOR_HWC)
        {
            dstShape[0] = mapShape[0];
            dstShape[1] = mapShape[1];
        }
        else if (srcShape.layout() == nvcv::TENSOR_NHWC)
        {
            dstShape[1] = mapShape[1];
            dstShape[2] = mapShape[2];
        }
        else if (srcShape.layout() == nvcv::TENSOR_CHW)
        {
            // Planar (CHW): spatial dims follow the channel dimension; map is rank-3 (HWC).
            dstShape[1] = mapShape[0];
            dstShape[2] = mapShape[1];
        }
        else if (srcShape.layout() == nvcv::TENSOR_NCHW)
        {
            // Planar (NCHW): spatial dims are the last two; map is rank-4 (NHWC).
            dstShape[2] = mapShape[1];
            dstShape[3] = mapShape[2];
        }
        else
        {
            throw RemapError("Input src tensor must have HWC, NHWC, CHW, or NCHW layout");
        }
    }

    auto dst = Tensor::Create(dstShape, src.dtype(), srcShape.layout(), 0);

    RemapInto(*dst, src, map, srcInterp, mapInterp, mapValueType, alignCorners, borderMode, borderValue, pstream);
    return dst;
}

// VarShape into ---------------------------------------------------------------

ImageBatchVarShape &VarShapeRemapInto(ImageBatchVarShape &dst, ImageBatchVarShape &src, Tensor &map,
                                      NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp,
                                      NVCVRemapMapValueType mapValueType, bool alignCorners, NVCVBorderType borderMode,
                                      const pyarray &borderValue, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    float4 bValue = GetFloat4FromPyArray(borderValue);

    auto op = CreateOperator<cvcuda::Remap>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {src, map});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run(
        [&op, &stream, &src, &dst, &map, &srcInterp, &mapInterp, &mapValueType, &alignCorners, &borderMode, &bValue]()
        {
            op->submit(stream.handle(), src.impl(), dst.impl(), map.impl(), srcInterp, mapInterp, mapValueType,
                       alignCorners, borderMode, bValue);
        });

    return dst;
}

std::shared_ptr<ImageBatchVarShape> VarShapeRemap(ImageBatchVarShape &src, Tensor &map, NVCVInterpolationType srcInterp,
                                                  NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType,
                                                  bool alignCorners, NVCVBorderType borderMode,
                                                  const pyarray &borderValue, std::optional<Stream *> pstream)
{
    auto dst = ImageBatchVarShape::Create(src.capacity());

    nvcv::Size2D mapSize{};

    if (mapValueType == NVCV_REMAP_ABSOLUTE || mapValueType == NVCV_REMAP_ABSOLUTE_NORMALIZED)
    {
        auto mapAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(map.impl().exportData());
        if (!mapAccess)
        {
            throw RemapError("Incompatible map tensor layout");
        }

        mapSize.w = mapAccess->numCols();
        mapSize.h = mapAccess->numRows();
    }

    for (const std::shared_ptr<Image> &srcImage : src)
    {
        nvcvpy::priv::Size2D outputSize = srcImage->size();
        if (mapValueType == NVCV_REMAP_ABSOLUTE || mapValueType == NVCV_REMAP_ABSOLUTE_NORMALIZED)
        {
            outputSize = {mapSize.w, mapSize.h};
        }
        auto outputImage = Image::Create(outputSize, srcImage->format(), 0);
        dst->pushBack(*outputImage);
    }

    VarShapeRemapInto(*dst, src, map, srcInterp, mapInterp, mapValueType, alignCorners, borderMode, borderValue,
                      pstream);
    return dst;
}

} // namespace

void ExportOpRemap(py::module_ &m)
{
    using namespace nanobind::literals;

    py::object emptyBorderValue = py::module_::import_("numpy").attr("empty")(0, "dtype"_a = py::str("float32"));
    // nanobind's arg::sig replaces the displayed default rather than its type annotation. Let it use this object's
    // repr so both the runtime default and generated signature remain the same empty float32 array.

    m.def("remap", NvtxTrace("cvcuda.remap", &Remap), "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST,
          "map_interp"_a = NVCV_INTERP_NEAREST, "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false,
          "border"_a = NVCV_BORDER_CONSTANT, "border_value"_a.none() = emptyBorderValue, py::kw_only(),
          "stream"_a = nullptr,
          R"pbdoc(
        Executes the Warp Perspective operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor.
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
                cvcuda.Border.REPLICATE, cvcuda.Border.REFLECT, cvcuda.Border.REFLECT_101,
                cvcuda.Border.WRAP, or cvcuda.Border.CONSTANT_COVERAGE).
            border_value (numpy.ndarray, optional): Used to specify values for a constant border,
                should have size <= 4 and dim of 1, where the values specify the border color for
                each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");
    m.def("remap_into", NvtxTrace("cvcuda.remap_into", &RemapInto), "dst"_a, "src"_a, "map"_a,
          "src_interp"_a = NVCV_INTERP_NEAREST, "map_interp"_a = NVCV_INTERP_NEAREST,
          "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false, "border"_a = NVCV_BORDER_CONSTANT,
          "border_value"_a.none() = emptyBorderValue, py::kw_only(), "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(
        Executes the Warp Perspective operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor.
            src (cvcuda.Tensor): Input tensor.
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
                cvcuda.Border.REPLICATE, cvcuda.Border.REFLECT, cvcuda.Border.REFLECT_101,
                cvcuda.Border.WRAP, or cvcuda.Border.CONSTANT_COVERAGE).
            border_value (numpy.ndarray, optional): Used to specify values for a constant border,
                should have size <= 4 and dim of 1, where the values specify the border color for
                each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
    m.def("remap", NvtxTrace("cvcuda.remap", &VarShapeRemap), "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST,
          "map_interp"_a = NVCV_INTERP_NEAREST, "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false,
          "border"_a = NVCV_BORDER_CONSTANT, "border_value"_a.none() = emptyBorderValue, py::kw_only(),
          "stream"_a = nullptr,
          R"pbdoc(
        Executes the Warp Perspective operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch.
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
                cvcuda.Border.WRAP). cvcuda.Border.CONSTANT_COVERAGE is not supported for variable-shape batches.
            border_value (numpy.ndarray, optional): Used to specify values for a constant border,
                should have size <= 4 and dim of 1, where the values specify the border color for
                each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");
    m.def("remap_into", NvtxTrace("cvcuda.remap_into", &VarShapeRemapInto), "dst"_a, "src"_a, "map"_a,
          "src_interp"_a = NVCV_INTERP_NEAREST, "map_interp"_a = NVCV_INTERP_NEAREST,
          "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false, "border"_a = NVCV_BORDER_CONSTANT,
          "border_value"_a.none() = emptyBorderValue, py::kw_only(), "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(
        Executes the Warp Perspective operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch.
            src (cvcuda.ImageBatchVarShape): Input image batch.
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
                cvcuda.Border.WRAP). cvcuda.Border.CONSTANT_COVERAGE is not supported for variable-shape batches.
            border_value (numpy.ndarray, optional): Used to specify values for a constant border,
                should have size <= 4 and dim of 1, where the values specify the border color for
                each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

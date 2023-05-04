/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    guard.add(LockMode::LOCK_READ, {src, map});
    guard.add(LockMode::LOCK_WRITE, {dst});
    guard.add(LockMode::LOCK_NONE, {*op});

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
    guard.add(LockMode::LOCK_READ, {src, map});
    guard.add(LockMode::LOCK_WRITE, {dst});
    guard.add(LockMode::LOCK_NONE, {*op});

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
          "border_value"_a = pyarray{}, py::kw_only(), "stream"_a = nullptr);
    m.def("remap_into", &RemapInto, "dst"_a, "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST,
          "map_interp"_a = NVCV_INTERP_NEAREST, "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false,
          "border"_a = NVCV_BORDER_CONSTANT, "border_value"_a = pyarray{}, py::kw_only(), "stream"_a = nullptr);

    m.def("remap", &VarShapeRemap, "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST,
          "map_interp"_a = NVCV_INTERP_NEAREST, "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false,
          "border"_a = NVCV_BORDER_CONSTANT, "border_value"_a = pyarray{}, py::kw_only(), "stream"_a = nullptr);
    m.def("remap_into", &VarShapeRemapInto, "dst"_a, "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST,
          "map_interp"_a = NVCV_INTERP_NEAREST, "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false,
          "border"_a = NVCV_BORDER_CONSTANT, "border_value"_a = pyarray{}, py::kw_only(), "stream"_a = nullptr);
}

} // namespace cvcudapy

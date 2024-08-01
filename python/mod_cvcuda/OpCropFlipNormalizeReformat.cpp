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
#include <cvcuda/OpCropFlipNormalizeReformat.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <iostream>

namespace cvcudapy {

namespace {

} // namespace

namespace {
Tensor CropFlipNormalizeReformatInto(Tensor &output, ImageBatchVarShape &input, Tensor &cropRect, Tensor &flipCode,
                                     Tensor &base, Tensor &scale, float globalScale, float globalShift, float epsilon,
                                     std::optional<uint32_t> flags, NVCVBorderType borderMode, float borderValue,
                                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::CropFlipNormalizeReformat>();

    if (!flags)
    {
        flags = 0;
    }

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, cropRect, flipCode, base, scale});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    op->submit(pstream->cudaHandle(), input, output, cropRect, borderMode, borderValue, flipCode, base, scale,
               globalScale, globalShift, epsilon, *flags);

    return output;
}

Tensor CropFlipNormalizeReformat(ImageBatchVarShape &input, const Shape &out_shape, nvcv::DataType out_dtype,
                                 nvcv::TensorLayout out_layout, Tensor &cropRect, Tensor &flipCode, Tensor &base,
                                 Tensor &scale, float globalScale, float globalShift, float epsilon,
                                 std::optional<uint32_t> flags, NVCVBorderType borderMode, float borderValue,
                                 std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(out_shape, out_dtype, out_layout);

    return CropFlipNormalizeReformatInto(output, input, cropRect, flipCode, base, scale, globalScale, globalShift,
                                         epsilon, flags, borderMode, borderValue, pstream);
}

} // namespace

void ExportOpCropFlipNormalizeReformat(py::module &m)
{
    using namespace pybind11::literals;

    float defGlobalScale = 1;
    float defGlobalShift = 0;
    float defEpsilon     = 0;

    m.def("crop_flip_normalize_reformat", &CropFlipNormalizeReformat, "src"_a, "out_shape"_a, "out_dtype"_a,
          "out_layout"_a, "rect"_a, "flip_code"_a, "base"_a, "scale"_a, "globalscale"_a = defGlobalScale,
          "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon, "flags"_a = std::nullopt,
          "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the CropFlipNormalizeReformat operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Copy Make Border operator
            for more details and usage examples.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            out_shape (tuple): The shape of the output.
            out_dtype (numpy.dtype): The data type of the output.
            out_layout (nvcv.TensorLayout): The layout of the output.
            rect (nvcv.Tensor): The crop rectangle tensor which has shape of [batch_size, 1, 1, 4] in reference to the input tensor.
                           The crop value of [crop_x, crop_y, crop_width, crop_height] stored in the final dimension of
                           the crop tensor, provided per image.
            flip_code (nvcv.Tensor): A tensor flag to specify how to flip the array; 0 means flipping around the x-axis,
                                1 means flipping around the y-axis, -1 means flipping around both axes,
                                and any other value will result  in no flip, provided per image.
            base (nvcv.Tensor): Tensor providing base values for normalization.
            scale (nvcv.Tensor): Tensor providing scale values for normalization.
            globalscale (float, optional): Additional scale value to be used in addition to scale
            globalshift (float, optional): Additional bias value to be used in addition to base.
            epsilon (float, optional): Epsilon to use when CVCUDA_NORMALIZE_SCALE_IS_STDDEV flag is set as a regularizing term to be
                                       added to variance.
            flags (int, optional): Algorithm flags, use CVCUDA_NORMALIZE_SCALE_IS_STDDEV if scale passed as argument
                                   is standard deviation instead or 0 if it is scaling.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            bvalue (float, optional): Border value to be used for constant border mode cvcuda.Border.CONSTANT.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("crop_flip_normalize_reformat_into", &CropFlipNormalizeReformatInto, "dst"_a, "src"_a, "rect"_a,
          "flip_code"_a, "base"_a, "scale"_a, "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift,
          "epsilon"_a = defEpsilon, "flags"_a = std::nullopt, "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the CropFlipNormalizeReformat operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Copy Make Border operator
            for more details and usage examples.

        Args:
            dst (nvcv.ImageBatchVarShape): Output image batch containing the result of the operation.
            src (nvcv.ImageBatchVarShape): Input image batch containing one or more images.
            rect (nvcv.Tensor): The crop rectangle tensor which has shape of [batch_size, 1, 1, 4] in reference to the input tensor.
                           The crop value of [crop_x, crop_y, crop_width, crop_height] stored in the final dimension of
                           the crop tensor, provided per image.
            flip_code (nvcv.Tensor): A tensor flag to specify how to flip the array; 0 means flipping around the x-axis,
                                1 means flipping around the y-axis, -1 means flipping around both axes,
                                and any other value will result  in no flip, provided per image.
            base (nvcv.Tensor): Tensor providing base values for normalization.
            scale (nvcv.Tensor): Tensor providing scale values for normalization.
            globalscale (float, optional): Additional scale value to be used in addition to scale
            globalshift (float, optional): Additional bias value to be used in addition to base.
            epsilon (float, optional): Epsilon to use when CVCUDA_NORMALIZE_SCALE_IS_STDDEV flag is set as a regularizing term to be
                                       added to variance.
            flags (int, optional): Algorithm flags, use CVCUDA_NORMALIZE_SCALE_IS_STDDEV if scale passed as argument
                                   is standard deviation instead or 0 if it is scaling.
            border (cvcuda.Border, optional): Border mode to be used when accessing elements outside input image.
            bvalue (float, optional): Border value to be used for constant border mode cvcuda.Border.CONSTANT.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

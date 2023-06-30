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
#include <cvcuda/OpRandomResizedCrop.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor RandomResizedCropInto(Tensor &output, Tensor &input, double min_scale, double max_scale, double min_ratio,
                             double max_ratio, NVCVInterpolationType interp, uint32_t seed,
                             std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    int32_t batchSize = static_cast<int32_t>(input.shape()[0]);
    auto    randomResizedCrop
        = CreateOperator<cvcuda::RandomResizedCrop>(min_scale, max_scale, min_ratio, max_ratio, batchSize, seed);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*randomResizedCrop});

    randomResizedCrop->submit(pstream->cudaHandle(), input, output, interp);

    return std::move(output);
}

Tensor RandomResizedCrop(Tensor &input, const Shape &out_shape, double min_scale, double max_scale, double min_ratio,
                         double max_ratio, NVCVInterpolationType interp, uint32_t seed, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(out_shape, input.dtype(), input.shape().layout());

    return RandomResizedCropInto(output, input, min_scale, max_scale, min_ratio, max_ratio, interp, seed, pstream);
}

ImageBatchVarShape RandomResizedCropVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                                 double min_scale, double max_scale, double min_ratio, double max_ratio,
                                                 NVCVInterpolationType interp, uint32_t seed,
                                                 std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto randomResizedCrop
        = CreateOperator<cvcuda::RandomResizedCrop>(min_scale, max_scale, min_ratio, max_ratio, input.capacity(), seed);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*randomResizedCrop});

    randomResizedCrop->submit(pstream->cudaHandle(), input, output, interp);

    return output;
}

ImageBatchVarShape RandomResizedCropVarShape(ImageBatchVarShape                      &input,
                                             const std::vector<std::tuple<int, int>> &out_size, double min_scale,
                                             double max_scale, double min_ratio, double max_ratio,
                                             NVCVInterpolationType interp, uint32_t seed, std::optional<Stream> pstream)
{
    if (input.numImages() != (int)out_size.size())
    {
        throw std::runtime_error("Number of input images must be equal to the number of elements in output size list ");
    }

    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        auto              size   = out_size[i];
        auto              image  = Image::Create({std::get<0>(size), std::get<1>(size)}, format);
        output.pushBack(image);
    }

    return RandomResizedCropVarShapeInto(output, input, min_scale, max_scale, min_ratio, max_ratio, interp, seed,
                                         pstream);
}

} // namespace

void ExportOpRandomResizedCrop(py::module &m)
{
    using namespace pybind11::literals;

    m.def("random_resized_crop", &RandomResizedCrop, "src"_a, "shape"_a, "min_scale"_a = 0.08, "max_scale"_a = 1.0,
          "min_ratio"_a = 0.75, "max_ratio"_a = 1.3333333333333333, "interp"_a = NVCV_INTERP_LINEAR, "seed"_a = 0,
          py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

        Executes the RandomResizedCrop operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the RandomResizedCrop operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
            shape (Tuple): Shape of output tensor.
            min_scale (double, optional): Lower bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            max_scale (double, optional): Upper bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            min_ratio (double, optional): Lower bound for the random aspect ratio of the crop, before resizing.
            max_ratio (double, optional): Upper bound for the random aspect ratio of the crop, before resizing.
            interp (Interp, optional): Interpolation type used for transform.
            seed (int, optional): Random seed, should be unsigned int32.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("random_resized_crop_into", &RandomResizedCropInto, "dst"_a, "src"_a, "min_scale"_a = 0.08,
          "max_scale"_a = 1.0, "min_ratio"_a = 0.75, "max_ratio"_a = 1.3333333333333333,
          "interp"_a = NVCV_INTERP_LINEAR, "seed"_a = 0, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the RandomResizedCrop operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the RandomResizedCrop operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output tensor to store the result of the operation.
            src (Tensor): Input tensor containing one or more images.
            min_scale (double, optional): Lower bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            max_scale (double, optional): Upper bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            min_ratio (double, optional): Lower bound for the random aspect ratio of the crop, before resizing.
            max_ratio (double, optional): Upper bound for the random aspect ratio of the crop, before resizing.
            interp (Interp, optional): Interpolation type used for transform.
            seed (int, optional): Random seed, should be unsigned int32.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("random_resized_crop", &RandomResizedCropVarShape, "src"_a, "sizes"_a, "min_scale"_a = 0.08,
          "max_scale"_a = 1.0, "min_ratio"_a = 0.75, "max_ratio"_a = 1.3333333333333333,
          "interp"_a = NVCV_INTERP_LINEAR, "seed"_a = 0, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the RandomResizedCrop operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the RandomResizedCrop operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            sizes (Tuple vector): Shapes of output images.
            min_scale (double, optional): Lower bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            max_scale (double, optional): Upper bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            min_ratio (double, optional): Lower bound for the random aspect ratio of the crop, before resizing.
            max_ratio (double, optional): Upper bound for the random aspect ratio of the crop, before resizing.
            interp (Interp, optional): Interpolation type used for transform.
            seed (int, optional): Random seed, should be unsigned int32.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("random_resized_crop_into", &RandomResizedCropVarShapeInto, "dst"_a, "src"_a, "min_scale"_a = 0.08,
          "max_scale"_a = 1.0, "min_ratio"_a = 0.75, "max_ratio"_a = 1.3333333333333333,
          "interp"_a = NVCV_INTERP_LINEAR, "seed"_a = 0, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the RandomResizedCrop operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the RandomResizedCrop operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            dst (ImageBatchVarShape): Output image batch containing the result of the operation.
            min_scale (double, optional): Lower bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            max_scale (double, optional): Upper bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            min_ratio (double, optional): Lower bound for the random aspect ratio of the crop, before resizing.
            max_ratio (double, optional): Upper bound for the random aspect ratio of the crop, before resizing.
            interp (Interp, optional): Interpolation type used for transform.
            seed (int, optional): Random seed, should be unsigned int32.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy

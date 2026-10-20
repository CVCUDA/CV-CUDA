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

#include "NativeBinding.hpp"
#include "Operators.hpp"

#include <common/Binding.hpp>
#include <common/PyUtil.hpp>
#include <cvcuda/OpRandomResizedCrop.hpp>

#include <stdexcept>

namespace cvcudapy {

namespace {

using native::Image;
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

class RandomResizedCropError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor &RandomResizedCropInto(Tensor &output, Tensor &input, double min_scale, double max_scale, double min_ratio,
                              double max_ratio, NVCVInterpolationType interp, uint32_t seed,
                              std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    // HWC inputs (rank 3) have no N dim, so shape[0] is H — fall back to 1.
    int32_t batchSize = input.rank() == 4 ? static_cast<int32_t>(input.impl().shape()[0]) : 1;
    auto    randomResizedCrop
        = CreateOperator<cvcuda::RandomResizedCrop>(min_scale, max_scale, min_ratio, max_ratio, batchSize, seed);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*randomResizedCrop});

    guard.run([&randomResizedCrop, &stream, &input, &output, &interp]()
              { randomResizedCrop->submit(stream.handle(), input.impl(), output.impl(), interp); });

    return output;
}

std::shared_ptr<Tensor> RandomResizedCrop(Tensor &input, const Shape &out_shape, double min_scale, double max_scale,
                                          double min_ratio, double max_ratio, NVCVInterpolationType interp,
                                          uint32_t seed, std::optional<Stream *> pstream)
{
    auto output = Tensor::Create(out_shape, input.dtype(), input.layout(), 0);

    RandomResizedCropInto(*output, input, min_scale, max_scale, min_ratio, max_ratio, interp, seed, pstream);
    return output;
}

ImageBatchVarShape &RandomResizedCropVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                                  double min_scale, double max_scale, double min_ratio,
                                                  double max_ratio, NVCVInterpolationType interp, uint32_t seed,
                                                  std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto randomResizedCrop
        = CreateOperator<cvcuda::RandomResizedCrop>(min_scale, max_scale, min_ratio, max_ratio, input.capacity(), seed);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*randomResizedCrop});

    guard.run([&randomResizedCrop, &stream, &input, &output, &interp]()
              { randomResizedCrop->submit(stream.handle(), input.impl(), output.impl(), interp); });

    return output;
}

std::shared_ptr<ImageBatchVarShape> RandomResizedCropVarShape(ImageBatchVarShape                      &input,
                                                              const std::vector<std::tuple<int, int>> &out_size,
                                                              double min_scale, double max_scale, double min_ratio,
                                                              double max_ratio, NVCVInterpolationType interp,
                                                              uint32_t seed, std::optional<Stream *> pstream)
{
    if (input.numImages() != (int)out_size.size())
    {
        throw RandomResizedCropError(
            "Number of input images must be equal to the number of elements in output size list ");
    }

    auto output = ImageBatchVarShape::Create(input.capacity());
    auto image  = input.begin();
    for (const auto &[width, height] : out_size)
    {
        auto resized = Image::Create({width, height}, (*image)->format(), 0);
        output->pushBack(*resized);
        ++image;
    }

    RandomResizedCropVarShapeInto(*output, input, min_scale, max_scale, min_ratio, max_ratio, interp, seed, pstream);
    return output;
}

} // namespace

void ExportOpRandomResizedCrop(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("random_resized_crop", NvtxTrace("cvcuda.random_resized_crop", &RandomResizedCrop), "src"_a, "shape"_a,
          "min_scale"_a = 0.08, "max_scale"_a = 1.0, "min_ratio"_a = 0.75, "max_ratio"_a = 1.3333333333333333,
          "interp"_a = NVCV_INTERP_LINEAR, "seed"_a = 0, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Executes the RandomResizedCrop operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            shape (Tuple): Shape of output tensor.
            min_scale (double, optional): Lower bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            max_scale (double, optional): Upper bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            min_ratio (double, optional): Lower bound for the random aspect ratio of the crop, before resizing.
            max_ratio (double, optional): Upper bound for the random aspect ratio of the crop, before resizing.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            seed (int, optional): Random seed, should be unsigned int32.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("random_resized_crop_into", NvtxTrace("cvcuda.random_resized_crop_into", &RandomResizedCropInto), "dst"_a,
          "src"_a, "min_scale"_a = 0.08, "max_scale"_a = 1.0, "min_ratio"_a = 0.75, "max_ratio"_a = 1.3333333333333333,
          "interp"_a = NVCV_INTERP_LINEAR, "seed"_a = 0, py::kw_only(), "stream"_a = nullptr,
          py::return_value_policy::reference, R"pbdoc(
        Executes the RandomResizedCrop operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            min_scale (double, optional): Lower bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            max_scale (double, optional): Upper bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            min_ratio (double, optional): Lower bound for the random aspect ratio of the crop, before resizing.
            max_ratio (double, optional): Upper bound for the random aspect ratio of the crop, before resizing.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            seed (int, optional): Random seed, should be unsigned int32.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("random_resized_crop", NvtxTrace("cvcuda.random_resized_crop", &RandomResizedCropVarShape), "src"_a,
          "sizes"_a, "min_scale"_a = 0.08, "max_scale"_a = 1.0, "min_ratio"_a = 0.75,
          "max_ratio"_a = 1.3333333333333333, "interp"_a = NVCV_INTERP_LINEAR, "seed"_a = 0, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the RandomResizedCrop operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            sizes (Tuple vector): Shapes of output images.
            min_scale (double, optional): Lower bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            max_scale (double, optional): Upper bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            min_ratio (double, optional): Lower bound for the random aspect ratio of the crop, before resizing.
            max_ratio (double, optional): Upper bound for the random aspect ratio of the crop, before resizing.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            seed (int, optional): Random seed, should be unsigned int32.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("random_resized_crop_into", NvtxTrace("cvcuda.random_resized_crop_into", &RandomResizedCropVarShapeInto),
          "dst"_a, "src"_a, "min_scale"_a = 0.08, "max_scale"_a = 1.0, "min_ratio"_a = 0.75,
          "max_ratio"_a = 1.3333333333333333, "interp"_a = NVCV_INTERP_LINEAR, "seed"_a = 0, py::kw_only(),
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the RandomResizedCrop operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            min_scale (double, optional): Lower bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            max_scale (double, optional): Upper bound for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.
            min_ratio (double, optional): Lower bound for the random aspect ratio of the crop, before resizing.
            max_ratio (double, optional): Upper bound for the random aspect ratio of the crop, before resizing.
            interp (cvcuda.Interp, optional): Interpolation type used for transform.
            seed (int, optional): Random seed, should be unsigned int32.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

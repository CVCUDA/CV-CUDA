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

#include "UnaryElementwiseOp.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpBrightnessContrast.hpp>
#include <nvcv/RoundMode.h>

namespace cvcudapy {

namespace {
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

inline std::shared_ptr<Tensor> tensorLike(Tensor &src)
{
    return Tensor::CreateFromReqs(nvcv::Tensor::CalcRequirements(src.impl().shape(), src.dtype()));
}

inline std::shared_ptr<ImageBatchVarShape> batchLike(ImageBatchVarShape &src)
{
    return CreateSameShapeImageBatch(src);
}

template<typename Op, typename Src, typename Dst, typename Call>
auto runGuard(Op &op, Src &src, Dst &dst, const std::optional<Tensor *> &brightness,
              const std::optional<Tensor *> &contrast, const std::optional<Tensor *> &brightnessShift,
              const std::optional<Tensor *> &contrastCenter, const std::optional<Stream *> &pstream, Call &&call)
{
    Stream &stream = native::GetStream(pstream);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {src});
    for (const auto &arg : {brightness, contrast, brightnessShift, contrastCenter})
    {
        if (arg && *arg)
        {
            guard.add(LockMode::LOCK_MODE_READ, {**arg});
        }
    }
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run(
        [&brightness, &contrast, &brightnessShift, &contrastCenter, &call, &stream]()
        {
            const nvcv::Tensor nullTensor{nullptr};
            auto               asTensor = [&nullTensor](const std::optional<Tensor *> &arg) -> const nvcv::Tensor &
            {
                return arg && *arg ? (*arg)->impl() : nullTensor;
            };
            call(stream, asTensor(brightness), asTensor(contrast), asTensor(brightnessShift), asTensor(contrastCenter));
        });
}

Tensor &BrightnessContrastInto(Tensor &dst, Tensor &src, std::optional<Tensor *> brightness,
                               std::optional<Tensor *> contrast, std::optional<Tensor *> brightnessShift,
                               std::optional<Tensor *> contrastCenter, std::optional<Stream *> pstream)
{
    auto op = CreateOperator<cvcuda::BrightnessContrast>();
    runGuard(op, src, dst, brightness, contrast, brightnessShift, contrastCenter, pstream,
             [&op, &src, &dst](Stream &stream, const nvcv::Tensor &brightnessArg, const nvcv::Tensor &contrastArg,
                               const nvcv::Tensor &brightnessShiftArg, const nvcv::Tensor &contrastCenterArg)
             {
                 op->submit(stream.handle(), src.impl(), dst.impl(), brightnessArg, contrastArg, brightnessShiftArg,
                            contrastCenterArg);
             });
    return dst;
}

std::shared_ptr<Tensor> BrightnessContrast(Tensor &src, std::optional<Tensor *> brightness,
                                           std::optional<Tensor *> contrast, std::optional<Tensor *> brightnessShift,
                                           std::optional<Tensor *> contrastCenter, std::optional<Stream *> pstream)
{
    auto dst = tensorLike(src);
    BrightnessContrastInto(*dst, src, brightness, contrast, brightnessShift, contrastCenter, pstream);
    return dst;
}

ImageBatchVarShape &VarShapeBrightnessContrastInto(ImageBatchVarShape &dst, ImageBatchVarShape &src,
                                                   std::optional<Tensor *> brightness, std::optional<Tensor *> contrast,
                                                   std::optional<Tensor *> brightnessShift,
                                                   std::optional<Tensor *> contrastCenter,
                                                   std::optional<Stream *> pstream)
{
    auto op = CreateOperator<cvcuda::BrightnessContrast>();
    runGuard(op, src, dst, brightness, contrast, brightnessShift, contrastCenter, pstream,
             [&op, &src, &dst](Stream &stream, const nvcv::Tensor &brightnessArg, const nvcv::Tensor &contrastArg,
                               const nvcv::Tensor &brightnessShiftArg, const nvcv::Tensor &contrastCenterArg)
             {
                 op->submit(stream.handle(), src.impl(), dst.impl(), brightnessArg, contrastArg, brightnessShiftArg,
                            contrastCenterArg);
             });
    return dst;
}

std::shared_ptr<ImageBatchVarShape> VarShapeBrightnessContrast(
    ImageBatchVarShape &src, std::optional<Tensor *> brightness, std::optional<Tensor *> contrast,
    std::optional<Tensor *> brightnessShift, std::optional<Tensor *> contrastCenter, std::optional<Stream *> pstream)
{
    auto dst = batchLike(src);
    VarShapeBrightnessContrastInto(*dst, src, brightness, contrast, brightnessShift, contrastCenter, pstream);
    return dst;
}

Tensor &BrightnessContrastScalarInto(Tensor &dst, Tensor &src, double brightness, double contrast,
                                     double brightnessShift, double contrastCenter, bool clamp, NVCVRoundMode round,
                                     std::optional<Stream *> pstream)
{
    return UnaryElementwiseInto<cvcuda::BrightnessContrast>(dst, src, pstream, brightness, contrast, brightnessShift,
                                                            contrastCenter, clamp, round);
}

std::shared_ptr<Tensor> BrightnessContrastScalar(Tensor &src, double brightness, double contrast,
                                                 double brightnessShift, double contrastCenter, bool clamp,
                                                 NVCVRoundMode round, std::optional<Stream *> pstream)
{
    return UnaryElementwiseTensor<cvcuda::BrightnessContrast>(src, pstream, brightness, contrast, brightnessShift,
                                                              contrastCenter, clamp, round);
}

ImageBatchVarShape &VarShapeBrightnessContrastScalarInto(ImageBatchVarShape &dst, ImageBatchVarShape &src,
                                                         double brightness, double contrast, double brightnessShift,
                                                         double contrastCenter, bool clamp, NVCVRoundMode round,
                                                         std::optional<Stream *> pstream)
{
    return UnaryElementwiseInto<cvcuda::BrightnessContrast>(dst, src, pstream, brightness, contrast, brightnessShift,
                                                            contrastCenter, clamp, round);
}

std::shared_ptr<ImageBatchVarShape> VarShapeBrightnessContrastScalar(ImageBatchVarShape &src, double brightness,
                                                                     double contrast, double brightnessShift,
                                                                     double contrastCenter, bool clamp,
                                                                     NVCVRoundMode           round,
                                                                     std::optional<Stream *> pstream)
{
    return UnaryElementwiseVarShape<cvcuda::BrightnessContrast>(src, pstream, brightness, contrast, brightnessShift,
                                                                contrastCenter, clamp, round);
}

} // namespace

void ExportOpBrightnessContrast(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("brightness_contrast", NvtxTrace("cvcuda.brightness_contrast", &BrightnessContrast), "src"_a,
          "brightness"_a = nullptr, "contrast"_a = nullptr, "brightness_shift"_a = nullptr,
          "contrast_center"_a = nullptr, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Adjusts the brightness and contrast of the images according to the formula:
        ``out = brightness_shift + brightness * (contrast_center + contrast * (in - contrast_center))``.


        Args:
            src (cvcuda.Tensor): Input tensor.
            brightness (cvcuda.Tensor, optional): Optional tensor describing brightness multiplier.
                If specified, it must contain only 1 element. If not specified, the neutral ``1.``
                is used.
            contrast (cvcuda.Tensor, optional): Optional tensor describing contrast multiplier.
                If specified, it must contain only 1 element. If not specified, the neutral ``1.``
                is used.
            brightness_shift (cvcuda.Tensor, optional): Optional tensor describing brightness shift.
                If specified, it must contain only 1 element. If not specified, the neutral ``0.``
                is used.
            contrast_center (cvcuda.Tensor, optional): Optional tensor describing contrast center.
                If specified, it must contain only 1 element. If not specified, the middle of the
                assumed input type range is used. For floats it is ``0.5``, for unsigned integer
                types it is ``2 ** (number_of_bits - 1)``, for signed integer types it is
                ``2 ** (number_of_bits - 2)``.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.
    )pbdoc");
    m.def("brightness_contrast_into", NvtxTrace("cvcuda.brightness_contrast_into", &BrightnessContrastInto), "dst"_a,
          "src"_a, "brightness"_a = nullptr, "contrast"_a = nullptr, "brightness_shift"_a = nullptr,
          "contrast_center"_a = nullptr, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Adjusts the brightness and contrast of the images according to the formula:
        ``out = brightness_shift + brightness * (contrast_center + contrast * (in - contrast_center))``.


        Args:
            src (cvcuda.Tensor): Input tensor.
            dst (cvcuda.Tensor): Output tensor containing the result of the operation.
            brightness (cvcuda.Tensor, optional): Optional tensor describing brightness multiplier.
                If specified, it must contain only 1 element. If not specified, the neutral ``1.``
                is used.
            contrast (cvcuda.Tensor, optional): Optional tensor describing contrast multiplier.
                If specified, it must contain only 1 element. If not specified, the neutral ``1.``
                is used.
            brightness_shift (cvcuda.Tensor, optional): Optional tensor describing brightness shift.
                If specified, it must contain only 1 element. If not specified, the neutral ``0.``
                is used.
            contrast_center (cvcuda.Tensor, optional): Optional tensor describing contrast center.
                If specified, it must contain only 1 element. If not specified, the middle of the
                assumed input type range is used. For floats it is ``0.5``, for unsigned integer
                types it is ``2 ** (number_of_bits - 1)``, for signed integer types it is
                ``2 ** (number_of_bits - 2)``.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    // VarShape variants
    m.def("brightness_contrast", NvtxTrace("cvcuda.brightness_contrast", &VarShapeBrightnessContrast), "src"_a,
          "brightness"_a = nullptr, "contrast"_a = nullptr, "brightness_shift"_a = nullptr,
          "contrast_center"_a = nullptr, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Adjusts the brightness and contrast of the images according to the formula:
        ``out = brightness_shift + brightness * (contrast_center + contrast * (in - contrast_center))``.

        The brightness/brightness_shift/contrast/contrast_center tensors' length must match the
        number of samples in the batch.



        Args:
            src (cvcuda.ImageBatchVarShape): Input tensor.
            brightness (cvcuda.Tensor, optional): Optional tensor describing brightness multiplier.
                If specified, it must contain 1 or N elements where N is the number of input
                images. If it contains a single element, the same value is used for all input
                images. If not specified, the neutral ``1.`` is used.
            contrast (cvcuda.Tensor, optional): Optional tensor describing contrast multiplier.
                If specified, it must contain either 1 or N elements where N is the number of
                input images. If it contains a single element, the same value is used for all
                input images. If not specified, the neutral ``1.`` is used.
            brightness_shift (cvcuda.Tensor, optional): Optional tensor describing brightness shift.
                If specified, it must contain either 1 or N elements where N is the number of
                input images. If it contains a single element, the same value is used for all
                input images. If not specified, the neutral ``0.`` is used.
            contrast_center (cvcuda.Tensor, optional): Optional tensor describing contrast center.
                If specified, it must contain either 1 or N elements where N is the number of input
                images. If it contains a single element, the same value is used for all input
                images. If not specified, the middle of the assumed input type range is used. For
                floats it is ``0.5``, for unsigned integer types it is
                ``2 ** (number_of_bits - 1)``, for signed integer types it is
                ``2 ** (number_of_bits - 2)``.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.
    )pbdoc");
    m.def("brightness_contrast_into", NvtxTrace("cvcuda.brightness_contrast_into", &VarShapeBrightnessContrastInto),
          "dst"_a, "src"_a, "brightness"_a = nullptr, "contrast"_a = nullptr, "brightness_shift"_a = nullptr,
          "contrast_center"_a = nullptr, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Adjusts the brightness and contrast of the images according to the formula:
        ``out = brightness_shift + brightness * (contrast_center + contrast * (in - contrast_center))``.

        The brightness/brightness_shift/contrast/contrast_center tensors' length must match the
        number of samples in the batch.



        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            brightness (cvcuda.ImageBatchVarShape, optional): Optional tensor describing brightness multiplier.
                If specified, it must contain 1 or N elements where N is the number of input
                images. If it contains a single element, the same value is used for all input
                images. If not specified, the neutral ``1.`` is used.
            contrast (cvcuda.Tensor, optional): Optional tensor describing contrast multiplier.
                If specified, it must contain either 1 or N elements where N is the number of
                input images. If it contains a single element, the same value is used for all
                input images. If not specified, the neutral ``1.`` is used.
            brightness_shift (cvcuda.Tensor, optional): Optional tensor describing brightness shift.
                If specified, it must contain either 1 or N elements where N is the number of
                input images. If it contains a single element, the same value is used for all
                input images. If not specified, the neutral ``0.`` is used.
            contrast_center (cvcuda.Tensor, optional): Optional tensor describing contrast center.
                If specified, it must contain either 1 or N elements where N is the number of input
                images. If it contains a single element, the same value is used for all input
                images. If not specified, the middle of the assumed input type range is used. For
                floats it is ``0.5``, for unsigned integer types it is
                ``2 ** (number_of_bits - 1)``, for signed integer types it is
                ``2 ** (number_of_bits - 2)``.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");

    // By-value variants
    m.def("brightness_contrast", NvtxTrace("cvcuda.brightness_contrast", &BrightnessContrastScalar), "src"_a,
          "brightness"_a, "contrast"_a, "brightness_shift"_a, "contrast_center"_a, py::kw_only(), "clamp"_a = false,
          "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr,
          R"pbdoc(
        Adjusts brightness and contrast using one set of scalar parameters for every input image.

        Args:
            src (cvcuda.Tensor): Input tensor.
            brightness (float): Brightness multiplier.
            contrast (float): Contrast multiplier.
            brightness_shift (float): Brightness shift.
            contrast_center (float): Contrast center.
            clamp (bool, optional): Clamp to the nominal image range: ``[0, 1]`` for floating-point
                output and ``[0, max]`` for integer output. Defaults to ``False``.
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST; use cvcuda.Round.TRUNCATE to drop the fractional part, which is
                what torchvision's adjust_brightness does. Has no effect on floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.
    )pbdoc");
    m.def("brightness_contrast_into", NvtxTrace("cvcuda.brightness_contrast_into", &BrightnessContrastScalarInto),
          "dst"_a, "src"_a, "brightness"_a, "contrast"_a, "brightness_shift"_a, "contrast_center"_a, py::kw_only(),
          "clamp"_a = false, "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Adjusts brightness and contrast into ``dst`` using scalar parameters.

        Args:
            dst (cvcuda.Tensor): Output tensor.
            src (cvcuda.Tensor): Input tensor.
            brightness (float): Brightness multiplier.
            contrast (float): Contrast multiplier.
            brightness_shift (float): Brightness shift.
            contrast_center (float): Contrast center.
            clamp (bool, optional): Clamp to the nominal image range. Defaults to ``False``.
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST; use cvcuda.Round.TRUNCATE to drop the fractional part, which is
                what torchvision's adjust_brightness does. Has no effect on floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
    m.def("brightness_contrast", NvtxTrace("cvcuda.brightness_contrast", &VarShapeBrightnessContrastScalar), "src"_a,
          "brightness"_a, "contrast"_a, "brightness_shift"_a, "contrast_center"_a, py::kw_only(), "clamp"_a = false,
          "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr, R"pbdoc(
        Adjusts brightness and contrast using one set of scalar parameters for every image in a batch.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch.
            brightness (float): Brightness multiplier.
            contrast (float): Contrast multiplier.
            brightness_shift (float): Brightness shift.
            contrast_center (float): Contrast center.
            clamp (bool, optional): Clamp to the nominal image range. Defaults to ``False``.
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST; use cvcuda.Round.TRUNCATE to drop the fractional part, which is
                what torchvision's adjust_brightness does. Has no effect on floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.
    )pbdoc");
    m.def("brightness_contrast_into",
          NvtxTrace("cvcuda.brightness_contrast_into", &VarShapeBrightnessContrastScalarInto), "dst"_a, "src"_a,
          "brightness"_a, "contrast"_a, "brightness_shift"_a, "contrast_center"_a, py::kw_only(), "clamp"_a = false,
          "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Adjusts brightness and contrast into ``dst`` using scalar parameters.

        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch.
            src (cvcuda.ImageBatchVarShape): Input image batch.
            brightness (float): Brightness multiplier.
            contrast (float): Contrast multiplier.
            brightness_shift (float): Brightness shift.
            contrast_center (float): Contrast center.
            clamp (bool, optional): Clamp to the nominal image range. Defaults to ``False``.
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST; use cvcuda.Round.TRUNCATE to drop the fractional part, which is
                what torchvision's adjust_brightness does. Has no effect on floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy

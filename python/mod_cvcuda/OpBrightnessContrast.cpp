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
#include <cvcuda/OpBrightnessContrast.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {

inline Tensor tensorLike(Tensor &src)
{
    const auto &srcShape = src.shape();
    Shape       dstShape = nvcvpy::CreateShape(srcShape);

    return Tensor::Create(dstShape, src.dtype(), src.layout());
}

inline ImageBatchVarShape batchLike(ImageBatchVarShape &src)
{
    ImageBatchVarShape dst = ImageBatchVarShape::Create(src.capacity());
    for (int i = 0; i < src.numImages(); ++i)
    {
        dst.pushBack(Image::Create(src[i].size(), src[i].format()));
    }
    return dst;
}

template<typename Op, typename Src, typename Dst, typename Call>
auto runGuard(Op &op, Src &src, Dst &dst, std::optional<Tensor> &brightness, std::optional<Tensor> &contrast,
              std::optional<Tensor> &brightnessShift, std::optional<Tensor> &contrastCenter,
              std::optional<Stream> &pstream, Call &&call)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {src});
    for (auto &arg : {brightness, contrast, brightnessShift, contrastCenter})
    {
        if (arg)
        {
            guard.add(LockMode::LOCK_READ, {*arg});
        }
    }
    guard.add(LockMode::LOCK_WRITE, {dst});
    guard.add(LockMode::LOCK_NONE, {*op});

    call(*pstream, brightness ? *brightness : nvcv::Tensor{nullptr}, contrast ? *contrast : nvcv::Tensor{nullptr},
         brightnessShift ? *brightnessShift : nvcv::Tensor{nullptr},
         contrastCenter ? *contrastCenter : nvcv::Tensor{nullptr});
}

Tensor BrightnessContrastInto(Tensor &dst, Tensor &src, std::optional<Tensor> &brightness,
                              std::optional<Tensor> &contrast, std::optional<Tensor> &brightnessShift,
                              std::optional<Tensor> &contrastCenter, std::optional<Stream> pstream)
{
    auto op = CreateOperator<cvcuda::BrightnessContrast>();
    runGuard(op, src, dst, brightness, contrast, brightnessShift, contrastCenter, pstream,
             [&](Stream &stream, const nvcv::Tensor &brightnessArg, const nvcv::Tensor &contrastArg,
                 const nvcv::Tensor &brightnessShiftArg, const nvcv::Tensor &contrastCenterArg) {
                 op->submit(stream.cudaHandle(), src, dst, brightnessArg, contrastArg, brightnessShiftArg,
                            contrastCenterArg);
             });
    return dst;
}

Tensor BrightnessContrast(Tensor &src, std::optional<Tensor> &brightness, std::optional<Tensor> &contrast,
                          std::optional<Tensor> &brightnessShift, std::optional<Tensor> &contrastCenter,
                          std::optional<Stream> pstream)
{
    auto dst = tensorLike(src);
    return BrightnessContrastInto(dst, src, brightness, contrast, brightnessShift, contrastCenter, pstream);
}

ImageBatchVarShape VarShapeBrightnessContrastInto(ImageBatchVarShape &dst, ImageBatchVarShape &src,
                                                  std::optional<Tensor> &brightness, std::optional<Tensor> &contrast,
                                                  std::optional<Tensor> &brightnessShift,
                                                  std::optional<Tensor> &contrastCenter, std::optional<Stream> pstream)
{
    auto op = CreateOperator<cvcuda::BrightnessContrast>();
    runGuard(op, src, dst, brightness, contrast, brightnessShift, contrastCenter, pstream,
             [&](Stream &stream, const nvcv::Tensor &brightnessArg, const nvcv::Tensor &contrastArg,
                 const nvcv::Tensor &brightnessShiftArg, const nvcv::Tensor &contrastCenterArg) {
                 op->submit(stream.cudaHandle(), src, dst, brightnessArg, contrastArg, brightnessShiftArg,
                            contrastCenterArg);
             });
    return dst;
}

ImageBatchVarShape VarShapeBrightnessContrast(ImageBatchVarShape &src, std::optional<Tensor> &brightness,
                                              std::optional<Tensor> &contrast, std::optional<Tensor> &brightnessShift,
                                              std::optional<Tensor> &contrastCenter, std::optional<Stream> pstream)
{
    auto dst = batchLike(src);
    return VarShapeBrightnessContrastInto(dst, src, brightness, contrast, brightnessShift, contrastCenter, pstream);
}

} // namespace

void ExportOpBrightnessContrast(py::module &m)
{
    using namespace pybind11::literals;

    m.def("brightness_contrast", &BrightnessContrast, "src"_a, "brightness"_a = nullptr, "contrast"_a = nullptr,
          "brightness_shift"_a = nullptr, "contrast_center"_a = nullptr, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Adjusts the brightness and contrast of the images according to the formula:
        `out = brightness_shift + brightness * (contrast_center + contrast * (in - contrast_center))`.
    )pbdoc");
    m.def("brightness_contrast_into", &BrightnessContrastInto, "dst"_a, "src"_a, "brightness"_a = nullptr,
          "contrast"_a = nullptr, "brightness_shift"_a = nullptr, "contrast_center"_a = nullptr, py::kw_only(),
          "stream"_a = nullptr);

    // VarShape variants
    m.def("brightness_contrast", &VarShapeBrightnessContrast, "src"_a, "brightness"_a = nullptr, "contrast"_a = nullptr,
          "brightness_shift"_a = nullptr, "contrast_center"_a = nullptr, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(
        Adjusts the brightness and contrast of the images according to the formula:
        `out = brightness_shift + brightness * (contrast_center + contrast * (in - contrast_center))`.

        The brightness/brightness_shift/contrast/contrast_center tensors' length must match the
        number of samples in the batch.
    )pbdoc");
    m.def("brightness_contrast_into", &VarShapeBrightnessContrastInto, "dst"_a, "src"_a, "brightness"_a = nullptr,
          "contrast"_a = nullptr, "brightness_shift"_a = nullptr, "contrast_center"_a = nullptr, py::kw_only(),
          "stream"_a = nullptr);
}

} // namespace cvcudapy

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpMorphology.hpp>
#include <cvcuda/Types.h>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor MorphologyInto(Tensor &output, Tensor &input, NVCVMorphologyType morph_type,
                      const std::tuple<int, int> &maskSize, const std::tuple<int, int> &anchor, int32_t iteration,
                      NVCVBorderType border, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto morphology = CreateOperator<cvcuda::Morphology>(0);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*morphology});

    nvcv::Size2D maskSizeArg{std::get<0>(maskSize), std::get<1>(maskSize)};
    int2         anchorArg;
    anchorArg.x = std::get<0>(anchor);
    anchorArg.y = std::get<1>(anchor);

    morphology->submit(pstream->cudaHandle(), input, output, morph_type, maskSizeArg, anchorArg, iteration, border);

    return output;
}

Tensor Morphology(Tensor &input, NVCVMorphologyType morph_type, const std::tuple<int, int> &maskSize,
                  const std::tuple<int, int> &anchor, int32_t iteration, NVCVBorderType border,
                  std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return MorphologyInto(output, input, morph_type, maskSize, anchor, iteration, border, pstream);
}

ImageBatchVarShape MorphologyVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                          NVCVMorphologyType morph_type, Tensor &masks, Tensor &anchors,
                                          const int32_t iteration, const NVCVBorderType borderMode,
                                          std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto morphology = CreateOperator<cvcuda::Morphology>(input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, masks, anchors});
    guard.add(LockMode::LOCK_WRITE, {output});

    morphology->submit(pstream->cudaHandle(), input, output, morph_type, masks, anchors, iteration, borderMode);

    return output;
}

ImageBatchVarShape MorphologyVarShape(ImageBatchVarShape &input, NVCVMorphologyType morph_type, Tensor &masks,
                                      Tensor &anchors, const int32_t iteration, const NVCVBorderType borderMode,
                                      std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::ImageFormat format = input[i].format();
        nvcv::Size2D      size   = input[i].size();
        auto              image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return MorphologyVarShapeInto(output, input, morph_type, masks, anchors, iteration, borderMode, pstream);
}

} // namespace

void ExportOpMorphology(py::module &m)
{
    using namespace pybind11::literals;

    m.def("morphology", &Morphology, "src"_a, "morphologyType"_a, "maskSize"_a, "anchor"_a, py::kw_only(),
          "iteration"_a = 1, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);

    m.def("morphology_into", &MorphologyInto, "dst"_a, "src"_a, "morphologyType"_a, "maskSize"_a, "anchor"_a,
          py::kw_only(), "iteration"_a = 1, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);

    m.def("morphology", &MorphologyVarShape, "src"_a, "morphologyType"_a, "masks"_a, "anchors"_a, py::kw_only(),
          "iteration"_a = 1, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);

    m.def("morphology_into", &MorphologyVarShapeInto, "dst"_a, "src"_a, "morphologyType"_a, "masks"_a, "anchors"_a,
          py::kw_only(), "iteration"_a = 1, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);
}
} // namespace cvcudapy

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

#include <common/Assert.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpCenterCrop.hpp>
#include <cvcuda/Types.h>
#include <nvcv/TensorLayoutInfo.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor CenterCropInto(Tensor &output, Tensor &input, const std::tuple<int, int> &cropSize,
                      std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto center_crop = CreateOperator<cvcuda::CenterCrop>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*center_crop});

    nvcv::Size2D cropSizeArg{std::get<0>(cropSize), std::get<1>(cropSize)};

    center_crop->submit(pstream->cudaHandle(), input, output, cropSizeArg);

    return output;
}

Tensor CenterCrop(Tensor &input, const std::tuple<int, int> &cropSize, std::optional<Stream> pstream)
{
    auto info = nvcv::TensorLayoutInfoImage::Create(input.layout());
    if (!info)
    {
        throw std::invalid_argument("Non-supported tensor layout");
    }

    int iwidth  = info->idxWidth();
    int iheight = info->idxHeight();

    NVCV_ASSERT(iwidth >= 0 && "All images have width");
    NVCV_ASSERT(iheight >= 0 && "All images have height");

    // Use cropSize (width, height) for output
    nvcv::TensorShape tshape = input.shape();
    Shape             out_shape{&tshape[0], &tshape[0] + tshape.rank()};
    out_shape[iwidth]  = std::get<0>(cropSize);
    out_shape[iheight] = std::get<1>(cropSize);

    Tensor output
        = Tensor::Create(nvcv::TensorShape(out_shape.data(), out_shape.size(), input.layout()), input.dtype());

    return CenterCropInto(output, input, cropSize, pstream);
}

} // namespace

void ExportOpCenterCrop(py::module &m)
{
    using namespace pybind11::literals;

    m.def("center_crop", &CenterCrop, "src"_a, "crop_size"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("center_crop_into", &CenterCropInto, "dst"_a, "src"_a, "crop_size"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace cvcudapy

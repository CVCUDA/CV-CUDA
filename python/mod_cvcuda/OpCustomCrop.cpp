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
#include <cvcuda/OpCustomCrop.hpp>
#include <nvcv/TensorLayoutInfo.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {
Tensor CustomCropInto(Tensor &output, Tensor &input, const NVCVRectI &rcCrop, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto crop = CreateOperator<cvcuda::CustomCrop>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*crop});

    crop->submit(pstream->cudaHandle(), input, output, rcCrop);

    return std::move(output);
}

Tensor CustomCrop(Tensor &input, const NVCVRectI &rcCrop, std::optional<Stream> pstream)
{
    auto info = nvcv::TensorLayoutInfoImage::Create(input.layout());
    if (!info)
    {
        throw std::invalid_argument("Non-supported tensor layout");
    }

    int iwidth  = info->idxWidth();
    int iheight = info->idxHeight();

    NVCV_ASSERT(iwidth >= 0 && "All images have width");

    // If no height, we consider height==1, and this dimension can't be changed
    // in order to keep the output layout the same as input's
    if (iheight < 0 && rcCrop.height != 1)
    {
        throw std::invalid_argument("Non-supported tensor layout");
    }

    // Create the output shape based inputs, changing width/height to match rcCrop's size
    nvcv::Shape          shape = input.shape().shape();
    std::vector<int64_t> out_shape{shape.begin(), shape.end()};
    out_shape[iwidth] = rcCrop.width;
    if (iheight >= 0)
    {
        out_shape[iheight] = rcCrop.height;
    }

    Tensor output
        = Tensor::Create({out_shape.data(), static_cast<int32_t>(out_shape.size()), input.layout()}, input.dtype());

    return CustomCropInto(output, input, rcCrop, pstream);
}

} // namespace

void ExportOpCustomCrop(py::module &m)
{
    using namespace pybind11::literals;

    m.def("customcrop", &CustomCrop, "src"_a, "rect"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("customcrop_into", &CustomCropInto, "dst"_a, "src"_a, "rect"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace cvcudapy

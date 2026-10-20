/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "FacadeCasterTests.hpp"

#include <nvcv/python/Array.hpp>
#include <nvcv/python/DataType.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ImageFormat.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <nvcv/python/TensorBatch.hpp>

namespace py = nanobind;

void ExportFacadeCasterTests(py::module_ &m)
{
    // Each probe body is empty on purpose: what is under test is the facade
    // type caster nanobind runs while converting the argument, so reaching the
    // body at all is the assertion.
    m.def("facade_tensor_caster_probe", [](const nvcvpy::Tensor &) { /* conversion is the probe */ });
    m.def("facade_image_caster_probe", [](const nvcvpy::Image &) { /* conversion is the probe */ });
    m.def("facade_array_caster_probe", [](const nvcvpy::Array &) { /* conversion is the probe */ });
    m.def("facade_tensor_batch_caster_probe", [](const nvcvpy::TensorBatch &) { /* conversion is the probe */ });
    m.def("facade_image_batch_caster_probe", [](const nvcvpy::ImageBatchVarShape &) { /* conversion is the probe */ });
    m.def("facade_stream_caster_probe", [](const nvcvpy::Stream &stream) { (void)stream.cudaHandle(); });

    // These returns intentionally use the public-header casters rather than
    // the binding module's private casters. They verify that the C-API bridge
    // supplies the owned reference nanobind requires from from_cpp().
    m.def("datatype_caster_return_probe", [] { return nvcv::TYPE_U8; });
    m.def("image_format_caster_return_probe", [] { return nvcv::FMT_RGB8; });
}

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

#include "CAPI.hpp"
#include "Cache.hpp"
#include "ColorSpec.hpp"
#include "Container.hpp"
#include "DataType.hpp"
#include "ExternalBuffer.hpp"
#include "Image.hpp"
#include "ImageBatch.hpp"
#include "ImageFormat.hpp"
#include "Rect.hpp"
#include "Resource.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"
#include "TensorBatch.hpp"

#include <nvcv/Version.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(nvcv, m)
{
    m.attr("__version__") = NVCV_VERSION_STRING;

    using namespace nvcvpy::priv;

    // These will be destroyed in the reverse order here
    // Since everything is ref counted the order should not matter
    // but it is safer to ini them in order

    // Core entities
    ExportCAPI(m);
    Resource::Export(m);
    Container::Export(m);
    ExternalBuffer::Export(m);

    // Supporting objects
    ExportColorSpec(m);
    ExportImageFormat(m);
    ExportDataType(m);
    ExportRect(m);

    // Objects
    Tensor::Export(m);
    TensorBatch::Export(m);
    Image::Export(m);
    ImageBatchVarShape::Export(m);

    // Cache and Streams
    Cache::Export(m);
    {
        py::module_ cuda = m.def_submodule("cuda");
        Stream::Export(cuda);
    }
}

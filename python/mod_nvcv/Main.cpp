/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "Definitions.hpp"
#include "ExternalBuffer.hpp"
#include "Image.hpp"
#include "ImageBatch.hpp"
#include "ImageFormat.hpp"
#include "Rect.hpp"
#include "Resource.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"
#include "TensorBatch.hpp"
#include "ThreadScope.hpp"

#include <nvcv/Version.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(nvcv, m)
{
    m.attr("__version__") = NVCV_VERSION_STRING;

    using namespace nvcvpy::priv;

    // Submodule used for additional functionality needed only by tests
    // so that some level of white-box testing is possible.
    //
    // This guarantees a clear separation of public and private APIs.
    // Users are restricted to the public API, allowing us to change the
    // private APIs as needed, without worring in breaking user's code.
    //
    // To retrieve it from inside the Export call, include "Definitions.hpp"
    // and call:
    //    py::module_ internal = m.attr(INTERNAL_SUBMODULE_NAME);
    // Functions and other properties can be then exposed as usual, e.g.
    //    internal.def("foo", &Foo");
    // and accessed in python as you'd expect:
    //    nvcv.internal.foo()
    m.def_submodule(INTERNAL_SUBMODULE_NAME);

    // These will be destroyed in the reverse order here
    // Since everything is ref counted the order should not matter
    // but it is safer to ini them in order

    // Supporting objects
    ExportColorSpec(m);
    ExportImageFormat(m);
    ExportDataType(m);
    ExportRect(m);
    ExportThreadScope(m);

    // Core entities
    ExportCAPI(m);
    Resource::Export(m);
    Cache::Export(m);
    Container::Export(m);
    ExternalBuffer::Export(m);

    // Objects
    Tensor::Export(m);
    TensorBatch::Export(m);
    Image::Export(m);
    ImageBatchVarShape::Export(m);

    // Streams
    {
        py::module_ cuda = m.def_submodule("cuda");
        // cuda submodule also has its submodule to export internal utilities.
        // The code in the export calls below might expect it, as it is unaware that
        // their functionality is not being defined directly under "nvcv" module.
        cuda.def_submodule(INTERNAL_SUBMODULE_NAME);

        Stream::Export(cuda);
    }
}

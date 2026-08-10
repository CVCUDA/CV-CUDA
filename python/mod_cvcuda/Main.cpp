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

// NVCV type exports
#include "nvcv/CAPI.hpp"
#include "nvcv/Cache.hpp"
#include "nvcv/ColorSpec.hpp"
#include "nvcv/Container.hpp"
#include "nvcv/DataType.hpp"
#include "nvcv/Definitions.hpp"
#include "nvcv/ExternalBuffer.hpp"
#include "nvcv/Image.hpp"
#include "nvcv/ImageBatch.hpp"
#include "nvcv/ImageFormat.hpp"
#include "nvcv/Rect.hpp"
#include "nvcv/Resource.hpp"
#include "nvcv/Stream.hpp"
#include "nvcv/Tensor.hpp"
#include "nvcv/TensorBatch.hpp"
#include "nvcv/ThreadScope.hpp"

// CV-CUDA Types exports
#include "AdaptiveThresholdType.hpp"
#include "BorderType.hpp"
#include "ChannelManipType.hpp"
#include "ColorConversionCode.hpp"
#include "ConnectivityType.hpp"
#include "InterpolationType.hpp"
#include "LabelType.hpp"
#include "MorphologyType.hpp"
#include "NormType.hpp"
#include "OsdElement.hpp"
#include "PairwiseMatcherType.hpp"
#include "RemapMapValueType.hpp"
#include "RoundMode.hpp"
#include "SIFTFlagType.hpp"
#include "ThresholdType.hpp"

// CV-CUDA Operators exports
#include "operators/Operators.hpp"

#include <cvcuda/Version.h>
#include <nvcv/python/ResourceGuard.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_cvcuda, m)
{
    m.attr("__name__")    = "cvcuda";
    m.attr("__version__") = CVCUDA_VERSION_STRING;

    {
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
        //    cvcuda.internal.foo()
        m.def_submodule(INTERNAL_SUBMODULE_NAME);

        // Export NVCV core types (Tensor, Image, Format, etc.)
        // Supporting objects
        ExportColorSpec(m);
        ExportImageFormat(m);
        ExportDataType(m);
        ExportRect(m);
        ExportThreadScope(m);
        ExportTensorLayout(m); // Image::cpu/cuda take a TensorLayout — must be registered before Image::Export.

        // Core Entities
        // Registration order matters: (a) base classes must be registered
        // before derived ones (Resource, CacheItem → Container → Tensor/Image/...);
        // (b) when a method signature references another pybind11 type, that
        // type must be registered first, or pybind11 emits the raw C++
        // typename (e.g. `nvcvpy::priv::Stream`) in stubs and reprs.
        ExportCAPI(m);
        Resource::Export(m);  // base of Container (no stream methods yet)
        Cache::Export(m);     // exports CacheItem (base of Container, Stream)
        Container::Export(m); // depends on Resource + CacheItem
        ExternalBuffer::Export(m);

        // Objects
        Image::Export(m); // before Tensor (as_tensor takes Image&)
        Tensor::Export(m);
        TensorBatch::Export(m);
        ImageBatchVarShape::Export(m);
        Stream::Export(m);

        // Deferred method bindings: add methods that reference Stream now
        // that Stream is a known pybind11 type.
        Resource::ExportStreamMethods(m);

        py::module_ test = m.def_submodule("_test");
        test.def("resourceguard_destructor_error",
                 []()
                 {
                     nvcvpy::Stream        stream = nvcvpy::Stream::Current();
                     nvcvpy::ResourceGuard guard(stream);

                     PyErr_SetString(PyExc_RuntimeError, "injected ResourceGuard commit failure");
                 });
        ExportCAPITestHooks(test);
    }

    {
        using namespace cvcudapy;

        // doctag: Non-Operators
        // Operators' auxiliary entities
        ExportAdaptiveThresholdType(m);
        ExportBorderType(m);
        ExportBoxBlur(m);
        ExportChannelManipType(m);
        ExportOSD(m);
        ExportColorConversionCode(m);
        ExportConnectivityType(m);
        ExportInterpolationType(m);
        ExportLabelType(m);
        ExportPairwiseMatcherType(m);
        ExportMorphologyType(m);
        ExportNormType(m);
        ExportRemapMapValueType(m);
        ExportRoundMode(m);
        ExportSIFTFlagType(m);
        ExportThresholdType(m);

        // doctag: Operators
        // CV-CUDA Operators
        ExportOpJpegCompressionDistortion(m);
        ExportOpAdjustHue(m);
        ExportOpAdjustSaturation(m);
        ExportOpAdjustSharpness(m);
        ExportOpAdjustContrast(m);
        ExportOpInvert(m);
        ExportOpSolarize(m);
        ExportOpPosterize(m);
        ExportOpAutoContrast(m);
        ExportOpResizeCropConvertReformat(m);
        ExportOpPairwiseMatcher(m);
        ExportOpLabel(m);
        ExportOpOSD(m);
        ExportOpHistogramEq(m);
        ExportOpAdvCvtColor(m);
        ExportOpSIFT(m);
        ExportOpMinMaxLoc(m);
        ExportOpHistogram(m);
        ExportOpMinAreaRect(m);
        ExportOpBndBox(m);
        ExportOpBoxBlur(m);
        ExportOpBrightnessContrast(m);
        ExportOpColorTwist(m);
        ExportOpHQResize(m);
        ExportOpRemap(m);
        ExportOpCropFlipNormalizeReformat(m);
        ExportOpNonMaximumSuppression(m);
        ExportOpReformat(m);
        ExportOpResize(m);
        ExportOpCustomCrop(m);
        ExportOpNormalize(m);
        ExportOpConvertTo(m);
        ExportOpPadAndStack(m);
        ExportOpCopyMakeBorder(m);
        ExportOpRotate(m);
        ExportOpErase(m);
        ExportOpGaussian(m);
        ExportOpMedianBlur(m);
        ExportOpLaplacian(m);
        ExportOpAverageBlur(m);
        ExportOpConv2D(m);
        ExportOpBilateralFilter(m);
        ExportOpJointBilateralFilter(m);
        ExportOpCenterCrop(m);
        ExportOpWarpAffine(m);
        ExportOpWarpPerspective(m);
        ExportOpChannelReorder(m);
        ExportOpMorphology(m);
        ExportOpFlip(m);
        ExportOpCvtColor(m);
        ExportOpComposite(m);
        ExportOpGammaContrast(m);
        ExportOpPillowResize(m);
        ExportOpThreshold(m);
        ExportOpAdaptiveThreshold(m);
        ExportOpRandomResizedCrop(m);
        ExportOpGaussianNoise(m);
        ExportOpInpaint(m);
        ExportOpStack(m);
        ExportOpFindHomography(m);
        ExportOpCLAHE(m);
    }
}

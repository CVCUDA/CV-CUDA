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

#include "AdaptiveThresholdType.hpp"
#include "BorderType.hpp"
#include "ChannelManipType.hpp"
#include "ColorConversionCode.hpp"
#include "ConnectivityType.hpp"
#include "InterpolationType.hpp"
#include "LabelType.hpp"
#include "MorphologyType.hpp"
#include "NormType.hpp"
#include "Operators.hpp"
#include "OsdElement.hpp"
#include "PairwiseMatcherType.hpp"
#include "RemapMapValueType.hpp"
#include "SIFTFlagType.hpp"
#include "ThresholdType.hpp"

#include <cvcuda/Version.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(cvcuda, m)
{
    m.attr("__version__") = CVCUDA_VERSION_STRING;

    // Import all public names from nvcv
    auto nvcv = py::module::import("nvcv");

    // If has __all__ dict, use it
    if (hasattr(nvcv, "__all__"))
    {
        for (py::handle name : nvcv.attr("__all__"))
        {
            m.add_object(py::str(name).cast<std::string>().c_str(), nvcv.attr(name));
        }
    }
    else
    {
        // Use the "dir" function to get all names
        for (py::handle hname : nvcv.attr("__dir__")())
        {
            auto name = py::str(hname).cast<std::string>();
            // Only import public names (not starting with '_')
            if (!name.empty() && name[0] != '_')
            {
                m.add_object(name.c_str(), nvcv.attr(hname));
            }
        }
    }

    m.add_object("Stream", nvcv.attr("cuda").attr("Stream"));

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
    ExportSIFTFlagType(m);
    ExportThresholdType(m);

    // doctag: Operators
    // CV-CUDA Operators
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
}

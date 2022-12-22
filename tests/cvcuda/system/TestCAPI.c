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

// Include everything to check if C compiler groks them.
#include <cvcuda/OpAverageBlur.h>
#include <cvcuda/OpBilateralFilter.h>
#include <cvcuda/OpCenterCrop.h>
#include <cvcuda/OpChannelReorder.h>
#include <cvcuda/OpComposite.h>
#include <cvcuda/OpConv2D.h>
#include <cvcuda/OpConvertTo.h>
#include <cvcuda/OpCopyMakeBorder.h>
#include <cvcuda/OpCustomCrop.h>
#include <cvcuda/OpCvtColor.h>
#include <cvcuda/OpErase.h>
#include <cvcuda/OpFlip.h>
#include <cvcuda/OpGammaContrast.h>
#include <cvcuda/OpGaussian.h>
#include <cvcuda/OpLaplacian.h>
#include <cvcuda/OpMedianBlur.h>
#include <cvcuda/OpMorphology.h>
#include <cvcuda/OpNormalize.h>
#include <cvcuda/OpPadAndStack.h>
#include <cvcuda/OpPillowResize.h>
#include <cvcuda/OpReformat.h>
#include <cvcuda/OpResize.h>
#include <cvcuda/OpRotate.h>
#include <cvcuda/OpWarpAffine.h>
#include <cvcuda/OpWarpPerspective.h>
#include <cvcuda/Operator.h>
#include <cvcuda/Types.h>
#include <cvcuda/Version.h>

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

// Instantiate structs/enums to check if they are correctly defined (i.e. using typedef)

static void TestCompile_Operator()
{
    NVCVOperatorHandle handle = NULL;
}

static void TestCompile_Warp()
{
    NVCVAffineTransform      xfAffine = {0, 1, 2, 3, 4, 5};
    NVCVPerspectiveTransform xfPersp  = {0, 1, 2, 3, 4, 5, 6, 7, 8};
}

static void TestCompile_Types()
{
    NVCVInterpolationType   interp = NVCV_INTERP_AREA;
    NVCVBorderType          border = NVCV_BORDER_WRAP;
    NVCVMorphologyType      morph  = NVCV_ERODE;
    NVCVColorConversionCode cvt    = NVCV_COLOR_RGBA2YUV_NV12;
}

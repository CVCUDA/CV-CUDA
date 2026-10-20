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

#include "TestCAPIEnum.h"

#include <cvcuda/OpAdaptiveThreshold.h>
#include <cvcuda/OpAdjustContrast.h>
#include <cvcuda/OpAdvCvtColor.h>
#include <cvcuda/OpAverageBlur.h>
#include <cvcuda/OpBilateralFilter.h>
#include <cvcuda/OpBrightnessContrast.h>
#include <cvcuda/OpConv2D.h>
#include <cvcuda/OpConvertTo.h>
#include <cvcuda/OpCopyMakeBorder.h>
#include <cvcuda/OpCropFlipNormalizeReformat.h>
#include <cvcuda/OpCvtColor.h>
#include <cvcuda/OpGammaContrast.h>
#include <cvcuda/OpGaussian.h>
#include <cvcuda/OpHQResize.h>
#include <cvcuda/OpJointBilateralFilter.h>
#include <cvcuda/OpLabel.h>
#include <cvcuda/OpLaplacian.h>
#include <cvcuda/OpMorphology.h>
#include <cvcuda/OpPadAndStack.h>
#include <cvcuda/OpPairwiseMatcher.h>
#include <cvcuda/OpPillowResize.h>
#include <cvcuda/OpRandomResizedCrop.h>
#include <cvcuda/OpRemap.h>
#include <cvcuda/OpResize.h>
#include <cvcuda/OpResizeCropConvertReformat.h>
#include <cvcuda/OpRotate.h>
#include <cvcuda/OpSIFT.h>
#include <cvcuda/OpWarpAffine.h>
#include <cvcuda/OpWarpPerspective.h>
#include <cvcuda/Types.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define TEST_CAPI_MAKE_INVALID_ENUM(Type, Name)                                                               \
    Type     Name;                                                                                            \
    uint32_t Name##Raw = UINT32_MAX;                                                                          \
    _Static_assert(sizeof(Name) == sizeof(Name##Raw), "C API enum must have a 32-bit object representation"); \
    memcpy(&Name, &Name##Raw, sizeof(Name))

NVCVStatus TestCAPIEnumAdjustContrastRoundSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out, double contrastFactor)
{
    TEST_CAPI_MAKE_INVALID_ENUM(NVCVRoundMode, roundMode);
    return cvcudaAdjustContrastRoundSubmit(handle, stream, in, out, contrastFactor, roundMode);
}

NVCVStatus TestCAPIEnumBrightnessContrastScalarRoundSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                           NVCVTensorHandle in, NVCVTensorHandle out, double brightness,
                                                           double contrast, double brightnessShift,
                                                           double contrastCenter, bool clamp)
{
    TEST_CAPI_MAKE_INVALID_ENUM(NVCVRoundMode, roundMode);
    return cvcudaBrightnessContrastScalarRoundSubmit(handle, stream, in, out, brightness, contrast, brightnessShift,
                                                     contrastCenter, clamp, roundMode);
}

NVCVStatus TestCAPIEnumCallWithInvalidValue(TestCAPIEnumCase testCase)
{
    switch (testCase) // NOSONAR: each case calls a different C API with case-specific arguments; no lookup table fits.
    {
    case TEST_CAPI_ENUM_ADAPTIVE_THRESHOLD_METHOD:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVAdaptiveThresholdType, adaptiveMethod);
        return cvcudaAdaptiveThresholdSubmit(NULL, NULL, NULL, NULL, 0, adaptiveMethod, NVCV_THRESH_BINARY, 3, 0);
    }
    case TEST_CAPI_ENUM_ADAPTIVE_THRESHOLD_TYPE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVThresholdType, thresholdType);
        return cvcudaAdaptiveThresholdSubmit(NULL, NULL, NULL, NULL, 0, NVCV_ADAPTIVE_THRESH_MEAN_C, thresholdType, 3,
                                             0);
    }
    case TEST_CAPI_ENUM_ADAPTIVE_THRESHOLD_VARSHAPE_METHOD:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVAdaptiveThresholdType, adaptiveMethod);
        return cvcudaAdaptiveThresholdVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, adaptiveMethod, NVCV_THRESH_BINARY,
                                                     NULL, NULL);
    }
    case TEST_CAPI_ENUM_ADAPTIVE_THRESHOLD_VARSHAPE_TYPE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVThresholdType, thresholdType);
        return cvcudaAdaptiveThresholdVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_ADAPTIVE_THRESH_MEAN_C,
                                                     thresholdType, NULL, NULL);
    }
    case TEST_CAPI_ENUM_ADV_CVT_COLOR_CODE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVColorConversionCode, code);
        return cvcudaAdvCvtColorSubmit(NULL, NULL, NULL, NULL, code, NVCV_COLOR_SPEC_BT601);
    }
    case TEST_CAPI_ENUM_AVERAGE_BLUR_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaAverageBlurSubmit(NULL, NULL, NULL, NULL, 1, 1, 0, 0, borderMode);
    }
    case TEST_CAPI_ENUM_AVERAGE_BLUR_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaAverageBlurVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NULL, borderMode);
    }
    case TEST_CAPI_ENUM_BILATERAL_FILTER_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaBilateralFilterSubmit(NULL, NULL, NULL, NULL, 1, 1, 1, borderMode);
    }
    case TEST_CAPI_ENUM_BILATERAL_FILTER_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaBilateralFilterVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NULL, NULL, borderMode);
    }
    case TEST_CAPI_ENUM_CONV2D_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaConv2DVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NULL, borderMode);
    }
    case TEST_CAPI_ENUM_CONVERT_TO_ROUND:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVRoundMode, roundMode);
        return cvcudaConvertToSubmit(NULL, NULL, NULL, NULL, 1, 0, roundMode);
    }
    case TEST_CAPI_ENUM_COPY_MAKE_BORDER_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaCopyMakeBorderSubmit(NULL, NULL, NULL, NULL, 0, 0, borderMode, (float4){0});
    }
    case TEST_CAPI_ENUM_COPY_MAKE_BORDER_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaCopyMakeBorderVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NULL, borderMode, (float4){0});
    }
    case TEST_CAPI_ENUM_COPY_MAKE_BORDER_VARSHAPE_STACK_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaCopyMakeBorderVarShapeStackSubmit(NULL, NULL, NULL, NULL, NULL, NULL, borderMode, (float4){0});
    }
    case TEST_CAPI_ENUM_CROP_FLIP_NORMALIZE_REFORMAT_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaCropFlipNormalizeReformatSubmit(NULL, NULL, NULL, NULL, NULL, borderMode, 0, NULL, NULL, NULL, 1,
                                                     0, 0, 0);
    }
    case TEST_CAPI_ENUM_CVT_COLOR_CODE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVColorConversionCode, code);
        return cvcudaCvtColorSubmit(NULL, NULL, NULL, NULL, code);
    }
    case TEST_CAPI_ENUM_CVT_COLOR_VARSHAPE_CODE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVColorConversionCode, code);
        return cvcudaCvtColorVarShapeSubmit(NULL, NULL, NULL, NULL, code);
    }
    case TEST_CAPI_ENUM_GAMMA_CONTRAST_ROUND:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVRoundMode, roundMode);
        return cvcudaGammaContrastScalarSubmit(NULL, NULL, NULL, NULL, 1, 1, roundMode);
    }
    case TEST_CAPI_ENUM_GAUSSIAN_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaGaussianSubmit(NULL, NULL, NULL, NULL, 1, 1, 1, 1, borderMode);
    }
    case TEST_CAPI_ENUM_GAUSSIAN_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaGaussianVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NULL, borderMode);
    }
    case TEST_CAPI_ENUM_HQ_RESIZE_TENSOR_REQ_MIN_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaHQResizeTensorGetWorkspaceRequirements(NULL, 0, (HQResizeTensorShapeI){0},
                                                            (HQResizeTensorShapeI){0}, interpolation,
                                                            NVCV_INTERP_LINEAR, false, NULL, NULL);
    }
    case TEST_CAPI_ENUM_HQ_RESIZE_TENSOR_REQ_MAG_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaHQResizeTensorGetWorkspaceRequirements(NULL, 0, (HQResizeTensorShapeI){0},
                                                            (HQResizeTensorShapeI){0}, NVCV_INTERP_LINEAR,
                                                            interpolation, false, NULL, NULL);
    }
    case TEST_CAPI_ENUM_HQ_RESIZE_BATCH_REQ_MIN_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaHQResizeTensorBatchGetWorkspaceRequirements(NULL, 0, (HQResizeTensorShapesI){0},
                                                                 (HQResizeTensorShapesI){0}, interpolation,
                                                                 NVCV_INTERP_LINEAR, false, (HQResizeRoisF){0}, NULL);
    }
    case TEST_CAPI_ENUM_HQ_RESIZE_BATCH_REQ_MAG_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaHQResizeTensorBatchGetWorkspaceRequirements(NULL, 0, (HQResizeTensorShapesI){0},
                                                                 (HQResizeTensorShapesI){0}, NVCV_INTERP_LINEAR,
                                                                 interpolation, false, (HQResizeRoisF){0}, NULL);
    }
    case TEST_CAPI_ENUM_HQ_RESIZE_SUBMIT_MIN_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaHQResizeSubmit(NULL, NULL, NULL, NULL, NULL, interpolation, NVCV_INTERP_LINEAR, false, NULL);
    }
    case TEST_CAPI_ENUM_HQ_RESIZE_SUBMIT_MAG_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaHQResizeSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_INTERP_LINEAR, interpolation, false, NULL);
    }
    case TEST_CAPI_ENUM_HQ_RESIZE_IMAGE_BATCH_MIN_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaHQResizeImageBatchSubmit(NULL, NULL, NULL, NULL, NULL, interpolation, NVCV_INTERP_LINEAR, false,
                                              (HQResizeRoisF){0});
    }
    case TEST_CAPI_ENUM_HQ_RESIZE_IMAGE_BATCH_MAG_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaHQResizeImageBatchSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_INTERP_LINEAR, interpolation, false,
                                              (HQResizeRoisF){0});
    }
    case TEST_CAPI_ENUM_HQ_RESIZE_TENSOR_BATCH_MIN_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaHQResizeTensorBatchSubmit(NULL, NULL, NULL, NULL, NULL, interpolation, NVCV_INTERP_LINEAR, false,
                                               (HQResizeRoisF){0});
    }
    case TEST_CAPI_ENUM_HQ_RESIZE_TENSOR_BATCH_MAG_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaHQResizeTensorBatchSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_INTERP_LINEAR, interpolation, false,
                                               (HQResizeRoisF){0});
    }
    case TEST_CAPI_ENUM_JOINT_BILATERAL_FILTER_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaJointBilateralFilterSubmit(NULL, NULL, NULL, NULL, NULL, 1, 1, 1, borderMode);
    }
    case TEST_CAPI_ENUM_JOINT_BILATERAL_FILTER_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaJointBilateralFilterVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, borderMode);
    }
    case TEST_CAPI_ENUM_SIFT_FLAGS:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVSIFTFlagType, flags);
        return cvcudaSIFTSubmit(NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1, 0.01f, 1.f, 1.f, flags);
    }
    case TEST_CAPI_ENUM_LABEL_CONNECTIVITY:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVConnectivityType, connectivity);
        return cvcudaLabelSubmit(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, connectivity,
                                 NVCV_LABEL_FAST, NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY);
    }
    case TEST_CAPI_ENUM_LABEL_ASSIGN_LABELS:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVLabelType, assignLabels);
        return cvcudaLabelSubmit(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                 NVCV_CONNECTIVITY_4_2D, assignLabels, NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY);
    }
    case TEST_CAPI_ENUM_LABEL_MASK_TYPE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVLabelMaskType, maskType);
        return cvcudaLabelSubmit(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                 NVCV_CONNECTIVITY_4_2D, NVCV_LABEL_FAST, maskType);
    }
    case TEST_CAPI_ENUM_LAPLACIAN_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaLaplacianSubmit(NULL, NULL, NULL, NULL, 1, 1, borderMode);
    }
    case TEST_CAPI_ENUM_LAPLACIAN_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaLaplacianVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NULL, borderMode);
    }
    case TEST_CAPI_ENUM_MORPHOLOGY_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaMorphologySubmit(NULL, NULL, NULL, NULL, NULL, NVCV_ERODE, 1, 1, 0, 0, 1, borderMode);
    }
    case TEST_CAPI_ENUM_MORPHOLOGY_TYPE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVMorphologyType, morphType);
        return cvcudaMorphologySubmit(NULL, NULL, NULL, NULL, NULL, morphType, 1, 1, 0, 0, 1, NVCV_BORDER_CONSTANT);
    }
    case TEST_CAPI_ENUM_MORPHOLOGY_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaMorphologyVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_ERODE, NULL, NULL, 1, borderMode);
    }
    case TEST_CAPI_ENUM_MORPHOLOGY_VARSHAPE_TYPE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVMorphologyType, morphType);
        return cvcudaMorphologyVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, morphType, NULL, NULL, 1,
                                              NVCV_BORDER_CONSTANT);
    }
    case TEST_CAPI_ENUM_PAD_AND_STACK_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaPadAndStackSubmit(NULL, NULL, NULL, NULL, NULL, NULL, borderMode, 0);
    }
    case TEST_CAPI_ENUM_PAIRWISE_MATCHER_ALGORITHM:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVPairwiseMatcherType, algorithm);
        return cvcudaPairwiseMatcherCreate(NULL, algorithm);
    }
    case TEST_CAPI_ENUM_PAIRWISE_MATCHER_NORM:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVNormType, normType);
        return cvcudaPairwiseMatcherSubmit(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, false, 1, normType);
    }
    case TEST_CAPI_ENUM_RESIZE_CROP_CONVERT_REFORMAT_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaResizeCropConvertReformatSubmit(NULL, NULL, NULL, NULL, (NVCVSize2D){1, 1}, interpolation,
                                                     (int2){0, 0}, NVCV_CHANNEL_NO_OP, 1, 0, false);
    }
    case TEST_CAPI_ENUM_RESIZE_CROP_CONVERT_REFORMAT_MANIP:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVChannelManip, manip);
        return cvcudaResizeCropConvertReformatSubmit(NULL, NULL, NULL, NULL, (NVCVSize2D){1, 1}, NVCV_INTERP_LINEAR,
                                                     (int2){0, 0}, manip, 1.f, 0.f, false);
    }
    case TEST_CAPI_ENUM_RESIZE_CROP_CONVERT_REFORMAT_VARSHAPE_MANIP:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVChannelManip, manip);
        return cvcudaResizeCropConvertReformatVarShapeSubmit(NULL, NULL, NULL, NULL, (NVCVSize2D){1, 1},
                                                             NVCV_INTERP_LINEAR, (int2){0, 0}, manip, 1.f, 0.f, false);
    }
    case TEST_CAPI_ENUM_RESIZE_CROP_CONVERT_REFORMAT_VARSHAPE_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaResizeCropConvertReformatVarShapeSubmit(NULL, NULL, NULL, NULL, (NVCVSize2D){1, 1}, interpolation,
                                                             (int2){0, 0}, NVCV_CHANNEL_NO_OP, 1, 0, false);
    }
    case TEST_CAPI_ENUM_ADV_CVT_COLOR_SPEC:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVColorSpec, spec);
        return cvcudaAdvCvtColorSubmit(NULL, NULL, NULL, NULL, NVCV_COLOR_RGB2YUV, spec);
    }
    case TEST_CAPI_ENUM_ELEMENTS_TYPE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVOSDType, type);
        NVCVElements elements   = NULL;
        NVCVBndBoxI  box        = {0};
        const void  *payloads[] = {&box};
        int32_t      counts[]   = {1};
        return nvcvElementsConstruct(&elements, &type, payloads, counts, 1);
    }
    case TEST_CAPI_ENUM_PILLOW_RESIZE_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        NVCVWorkspace workspace = {0};
        return cvcudaPillowResizeSubmit(NULL, NULL, &workspace, NULL, NULL, interpolation);
    }
    case TEST_CAPI_ENUM_PILLOW_RESIZE_VARSHAPE_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        NVCVWorkspace workspace = {0};
        return cvcudaPillowResizeVarShapeSubmit(NULL, NULL, &workspace, NULL, NULL, interpolation);
    }
    case TEST_CAPI_ENUM_RANDOM_RESIZED_CROP_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaRandomResizedCropSubmit(NULL, NULL, NULL, NULL, interpolation);
    }
    case TEST_CAPI_ENUM_RANDOM_RESIZED_CROP_VARSHAPE_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaRandomResizedCropVarShapeSubmit(NULL, NULL, NULL, NULL, interpolation);
    }
    case TEST_CAPI_ENUM_REMAP_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaRemapSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR,
                                 NVCV_REMAP_ABSOLUTE, false, borderMode, (float4){0});
    }
    case TEST_CAPI_ENUM_REMAP_INPUT_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaRemapSubmit(NULL, NULL, NULL, NULL, NULL, interpolation, NVCV_INTERP_LINEAR, NVCV_REMAP_ABSOLUTE,
                                 false, NVCV_BORDER_CONSTANT, (float4){0});
    }
    case TEST_CAPI_ENUM_REMAP_MAP_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaRemapSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_INTERP_LINEAR, interpolation, NVCV_REMAP_ABSOLUTE,
                                 false, NVCV_BORDER_CONSTANT, (float4){0});
    }
    case TEST_CAPI_ENUM_REMAP_MAP_VALUE_TYPE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVRemapMapValueType, mapValueType);
        return cvcudaRemapSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR, mapValueType,
                                 false, NVCV_BORDER_CONSTANT, (float4){0});
    }
    case TEST_CAPI_ENUM_REMAP_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaRemapVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR,
                                         NVCV_REMAP_ABSOLUTE, false, borderMode, (float4){0});
    }
    case TEST_CAPI_ENUM_REMAP_VARSHAPE_INPUT_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaRemapVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, interpolation, NVCV_INTERP_LINEAR,
                                         NVCV_REMAP_ABSOLUTE, false, NVCV_BORDER_CONSTANT, (float4){0});
    }
    case TEST_CAPI_ENUM_REMAP_VARSHAPE_MAP_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaRemapVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_INTERP_LINEAR, interpolation,
                                         NVCV_REMAP_ABSOLUTE, false, NVCV_BORDER_CONSTANT, (float4){0});
    }
    case TEST_CAPI_ENUM_REMAP_VARSHAPE_MAP_VALUE_TYPE:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVRemapMapValueType, mapValueType);
        return cvcudaRemapVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR,
                                         mapValueType, false, NVCV_BORDER_CONSTANT, (float4){0});
    }
    case TEST_CAPI_ENUM_RESIZE_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaResizeSubmit(NULL, NULL, NULL, NULL, interpolation);
    }
    case TEST_CAPI_ENUM_RESIZE_VARSHAPE_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaResizeVarShapeSubmit(NULL, NULL, NULL, NULL, interpolation);
    }
    case TEST_CAPI_ENUM_ROTATE_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaRotateSubmit(NULL, NULL, NULL, NULL, 0, (double2){0}, interpolation);
    }
    case TEST_CAPI_ENUM_ROTATE_VARSHAPE_INTERPOLATION:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVInterpolationType, interpolation);
        return cvcudaRotateVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, NULL, interpolation);
    }
    case TEST_CAPI_ENUM_WARP_AFFINE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaWarpAffineSubmit(NULL, NULL, NULL, NULL, (NVCVAffineTransform){0}, 0, borderMode, (float4){0});
    }
    case TEST_CAPI_ENUM_WARP_AFFINE_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaWarpAffineVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, 0, borderMode, (float4){0});
    }
    case TEST_CAPI_ENUM_WARP_PERSPECTIVE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaWarpPerspectiveSubmit(NULL, NULL, NULL, NULL, (NVCVPerspectiveTransform){0}, 0, borderMode,
                                           (float4){0});
    }
    case TEST_CAPI_ENUM_WARP_PERSPECTIVE_VARSHAPE_BORDER:
    {
        TEST_CAPI_MAKE_INVALID_ENUM(NVCVBorderType, borderMode);
        return cvcudaWarpPerspectiveVarShapeSubmit(NULL, NULL, NULL, NULL, NULL, 0, borderMode, (float4){0});
    }
    case TEST_CAPI_ENUM_COUNT:
        break;
    }

    return NVCV_ERROR_INVALID_ARGUMENT;
}

#undef TEST_CAPI_MAKE_INVALID_ENUM

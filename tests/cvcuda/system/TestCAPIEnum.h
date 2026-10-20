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

#ifndef CVCUDA_TEST_SYSTEM_TEST_CAPI_ENUM_H
#define CVCUDA_TEST_SYSTEM_TEST_CAPI_ENUM_H

#include <cuda_runtime_api.h>
#include <cvcuda/Operator.h>
#include <cvcuda/Types.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define TEST_CAPI_ENUM_CASES(X)                                                                                        \
    X(TEST_CAPI_ENUM_ADAPTIVE_THRESHOLD_METHOD, "AdaptiveThresholdMethod", "Invalid adaptive threshold type:")         \
    X(TEST_CAPI_ENUM_ADAPTIVE_THRESHOLD_TYPE, "AdaptiveThresholdType", "Invalid threshold type:")                      \
    X(TEST_CAPI_ENUM_ADAPTIVE_THRESHOLD_VARSHAPE_METHOD, "AdaptiveThresholdVarShapeMethod",                            \
      "Invalid adaptive threshold type:")                                                                              \
    X(TEST_CAPI_ENUM_ADAPTIVE_THRESHOLD_VARSHAPE_TYPE, "AdaptiveThresholdVarShapeType", "Invalid threshold type:")     \
    X(TEST_CAPI_ENUM_ADV_CVT_COLOR_CODE, "AdvCvtColorCode", "Invalid color conversion code:")                          \
    X(TEST_CAPI_ENUM_ADV_CVT_COLOR_SPEC, "AdvCvtColorSpec", "Invalid color spec:")                                     \
    X(TEST_CAPI_ENUM_AVERAGE_BLUR_BORDER, "AverageBlurBorder", "Invalid border type:")                                 \
    X(TEST_CAPI_ENUM_AVERAGE_BLUR_VARSHAPE_BORDER, "AverageBlurVarShapeBorder", "Invalid border type:")                \
    X(TEST_CAPI_ENUM_BILATERAL_FILTER_BORDER, "BilateralFilterBorder", "Invalid border type:")                         \
    X(TEST_CAPI_ENUM_BILATERAL_FILTER_VARSHAPE_BORDER, "BilateralFilterVarShapeBorder", "Invalid border type:")        \
    X(TEST_CAPI_ENUM_CONV2D_VARSHAPE_BORDER, "Conv2DVarShapeBorder", "Invalid border type:")                           \
    X(TEST_CAPI_ENUM_CONVERT_TO_ROUND, "ConvertToRound", "Invalid round mode:")                                        \
    X(TEST_CAPI_ENUM_COPY_MAKE_BORDER_BORDER, "CopyMakeBorderBorder", "Invalid border type:")                          \
    X(TEST_CAPI_ENUM_COPY_MAKE_BORDER_VARSHAPE_BORDER, "CopyMakeBorderVarShapeBorder", "Invalid border type:")         \
    X(TEST_CAPI_ENUM_COPY_MAKE_BORDER_VARSHAPE_STACK_BORDER, "CopyMakeBorderVarShapeStackBorder",                      \
      "Invalid border type:")                                                                                          \
    X(TEST_CAPI_ENUM_CROP_FLIP_NORMALIZE_REFORMAT_BORDER, "CropFlipNormalizeReformatBorder", "Invalid border type:")   \
    X(TEST_CAPI_ENUM_CVT_COLOR_CODE, "CvtColorCode", "Invalid color conversion code:")                                 \
    X(TEST_CAPI_ENUM_CVT_COLOR_VARSHAPE_CODE, "CvtColorVarShapeCode", "Invalid color conversion code:")                \
    X(TEST_CAPI_ENUM_GAMMA_CONTRAST_ROUND, "GammaContrastRound", "Invalid round mode:")                                \
    X(TEST_CAPI_ENUM_GAUSSIAN_BORDER, "GaussianBorder", "Invalid border type:")                                        \
    X(TEST_CAPI_ENUM_GAUSSIAN_VARSHAPE_BORDER, "GaussianVarShapeBorder", "Invalid border type:")                       \
    X(TEST_CAPI_ENUM_HQ_RESIZE_TENSOR_REQ_MIN_INTERPOLATION, "HQResizeTensorReqMinInterpolation",                      \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_HQ_RESIZE_TENSOR_REQ_MAG_INTERPOLATION, "HQResizeTensorReqMagInterpolation",                      \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_HQ_RESIZE_BATCH_REQ_MIN_INTERPOLATION, "HQResizeBatchReqMinInterpolation",                        \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_HQ_RESIZE_BATCH_REQ_MAG_INTERPOLATION, "HQResizeBatchReqMagInterpolation",                        \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_HQ_RESIZE_SUBMIT_MIN_INTERPOLATION, "HQResizeSubmitMinInterpolation",                             \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_HQ_RESIZE_SUBMIT_MAG_INTERPOLATION, "HQResizeSubmitMagInterpolation",                             \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_HQ_RESIZE_IMAGE_BATCH_MIN_INTERPOLATION, "HQResizeImageBatchMinInterpolation",                    \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_HQ_RESIZE_IMAGE_BATCH_MAG_INTERPOLATION, "HQResizeImageBatchMagInterpolation",                    \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_HQ_RESIZE_TENSOR_BATCH_MIN_INTERPOLATION, "HQResizeTensorBatchMinInterpolation",                  \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_HQ_RESIZE_TENSOR_BATCH_MAG_INTERPOLATION, "HQResizeTensorBatchMagInterpolation",                  \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_JOINT_BILATERAL_FILTER_BORDER, "JointBilateralFilterBorder", "Invalid border type:")              \
    X(TEST_CAPI_ENUM_JOINT_BILATERAL_FILTER_VARSHAPE_BORDER, "JointBilateralFilterVarShapeBorder",                     \
      "Invalid border type:")                                                                                          \
    X(TEST_CAPI_ENUM_LABEL_CONNECTIVITY, "LabelConnectivity", "Invalid connectivity type:")                            \
    X(TEST_CAPI_ENUM_LABEL_ASSIGN_LABELS, "LabelAssignment", "Invalid label type:")                                    \
    X(TEST_CAPI_ENUM_LABEL_MASK_TYPE, "LabelMaskType", "Invalid label mask type:")                                     \
    X(TEST_CAPI_ENUM_LAPLACIAN_BORDER, "LaplacianBorder", "Invalid border type:")                                      \
    X(TEST_CAPI_ENUM_LAPLACIAN_VARSHAPE_BORDER, "LaplacianVarShapeBorder", "Invalid border type:")                     \
    X(TEST_CAPI_ENUM_MORPHOLOGY_BORDER, "MorphologyBorder", "Invalid border type:")                                    \
    X(TEST_CAPI_ENUM_MORPHOLOGY_TYPE, "MorphologyType", "Invalid morphology type:")                                    \
    X(TEST_CAPI_ENUM_MORPHOLOGY_VARSHAPE_BORDER, "MorphologyVarShapeBorder", "Invalid border type:")                   \
    X(TEST_CAPI_ENUM_MORPHOLOGY_VARSHAPE_TYPE, "MorphologyVarShapeType", "Invalid morphology type:")                   \
    X(TEST_CAPI_ENUM_PAD_AND_STACK_BORDER, "PadAndStackBorder", "Invalid border type:")                                \
    X(TEST_CAPI_ENUM_PAIRWISE_MATCHER_ALGORITHM, "PairwiseMatcherAlgorithm", "Invalid pairwise matcher type:")         \
    X(TEST_CAPI_ENUM_PAIRWISE_MATCHER_NORM, "PairwiseMatcherNorm", "Invalid norm type:")                               \
    X(TEST_CAPI_ENUM_PILLOW_RESIZE_INTERPOLATION, "PillowResizeInterpolation", "Invalid interpolation type:")          \
    X(TEST_CAPI_ENUM_PILLOW_RESIZE_VARSHAPE_INTERPOLATION, "PillowResizeVarShapeInterpolation",                        \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_RANDOM_RESIZED_CROP_INTERPOLATION, "RandomResizedCropInterpolation",                              \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_RANDOM_RESIZED_CROP_VARSHAPE_INTERPOLATION, "RandomResizedCropVarShapeInterpolation",             \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_REMAP_BORDER, "RemapBorder", "Invalid border type:")                                              \
    X(TEST_CAPI_ENUM_REMAP_INPUT_INTERPOLATION, "RemapInputInterpolation", "Invalid interpolation type:")              \
    X(TEST_CAPI_ENUM_REMAP_MAP_INTERPOLATION, "RemapMapInterpolation", "Invalid interpolation type:")                  \
    X(TEST_CAPI_ENUM_REMAP_MAP_VALUE_TYPE, "RemapMapValueType", "Invalid remap map value type:")                       \
    X(TEST_CAPI_ENUM_REMAP_VARSHAPE_BORDER, "RemapVarShapeBorder", "Invalid border type:")                             \
    X(TEST_CAPI_ENUM_REMAP_VARSHAPE_INPUT_INTERPOLATION, "RemapVarShapeInputInterpolation",                            \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_REMAP_VARSHAPE_MAP_INTERPOLATION, "RemapVarShapeMapInterpolation", "Invalid interpolation type:") \
    X(TEST_CAPI_ENUM_REMAP_VARSHAPE_MAP_VALUE_TYPE, "RemapVarShapeMapValueType", "Invalid remap map value type:")      \
    X(TEST_CAPI_ENUM_RESIZE_INTERPOLATION, "ResizeInterpolation", "Invalid interpolation type:")                       \
    X(TEST_CAPI_ENUM_RESIZE_VARSHAPE_INTERPOLATION, "ResizeVarShapeInterpolation", "Invalid interpolation type:")      \
    X(TEST_CAPI_ENUM_RESIZE_CROP_CONVERT_REFORMAT_INTERPOLATION, "ResizeCropConvertReformatInterpolation",             \
      "Invalid interpolation type:")                                                                                   \
    X(TEST_CAPI_ENUM_RESIZE_CROP_CONVERT_REFORMAT_MANIP, "ResizeCropConvertReformatManip",                             \
      "Invalid channel manipulation:")                                                                                 \
    X(TEST_CAPI_ENUM_RESIZE_CROP_CONVERT_REFORMAT_VARSHAPE_INTERPOLATION,                                              \
      "ResizeCropConvertReformatVarShapeInterpolation", "Invalid interpolation type:")                                 \
    X(TEST_CAPI_ENUM_RESIZE_CROP_CONVERT_REFORMAT_VARSHAPE_MANIP, "ResizeCropConvertReformatVarShapeManip",            \
      "Invalid channel manipulation:")                                                                                 \
    X(TEST_CAPI_ENUM_ROTATE_INTERPOLATION, "RotateInterpolation", "Invalid interpolation type:")                       \
    X(TEST_CAPI_ENUM_ROTATE_VARSHAPE_INTERPOLATION, "RotateVarShapeInterpolation", "Invalid interpolation type:")      \
    X(TEST_CAPI_ENUM_SIFT_FLAGS, "SIFTFlags", "Invalid SIFT flags:")                                                   \
    X(TEST_CAPI_ENUM_ELEMENTS_TYPE, "ElementsType", "Invalid OSD element type:")                                       \
    X(TEST_CAPI_ENUM_WARP_AFFINE_BORDER, "WarpAffineBorder", "Invalid border type:")                                   \
    X(TEST_CAPI_ENUM_WARP_AFFINE_VARSHAPE_BORDER, "WarpAffineVarShapeBorder", "Invalid border type:")                  \
    X(TEST_CAPI_ENUM_WARP_PERSPECTIVE_BORDER, "WarpPerspectiveBorder", "Invalid border type:")                         \
    X(TEST_CAPI_ENUM_WARP_PERSPECTIVE_VARSHAPE_BORDER, "WarpPerspectiveVarShapeBorder", "Invalid border type:")

typedef enum
{
#define TEST_CAPI_ENUM_DECLARE(Id, Name, Message) Id,
    TEST_CAPI_ENUM_CASES(TEST_CAPI_ENUM_DECLARE)
#undef TEST_CAPI_ENUM_DECLARE
    TEST_CAPI_ENUM_COUNT
} TestCAPIEnumCase;

NVCVStatus TestCAPIEnumCallWithInvalidValue(TestCAPIEnumCase testCase);

NVCVStatus TestCAPIEnumAdjustContrastRoundSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out, double contrastFactor);

NVCVStatus TestCAPIEnumBrightnessContrastScalarRoundSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                           NVCVTensorHandle in, NVCVTensorHandle out, double brightness,
                                                           double contrast, double brightnessShift,
                                                           double contrastCenter, bool clamp);

#ifdef __cplusplus
}
#endif

#endif // CVCUDA_TEST_SYSTEM_TEST_CAPI_ENUM_H

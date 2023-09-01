/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file OpAdvCvtColor.h
 *
 * @brief Defines types and functions to handle the AdvCvtColor operation.
 * @defgroup NVCV_C_ALGORITHM__ADV_CVT_COLOR Adv Cvt Color
 * @{
 */

#ifndef CVCUDA__ADV_CVT_COLOR_H
#define CVCUDA__ADV_CVT_COLOR_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the AdvCvtColor operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdvCvtColorCreate(NVCVOperatorHandle *handle);

/** Executes the AdvCvtColor operation on the given cuda stream. This operation does not
 *  wait for completion. Advanced color conversion from one color space to another from YUV420(NV12/21)YUV to RGB.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 3] kNWHC/KHWC semi planar 420 tensors are allowed
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 3] kNWHC/KHWC semi planar 420 tensors are allowed
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | No
 *       Data Type     | Yes
 *       Number        | Yes
 *       Channels      | Yes (No for semi planar 420 tensors conversion)
 *       Width         | Yes
 *       Height        | Yes (No for semi planar 420 tensors conversion)
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor.
 *
 * @param [out] out output tensor.
 *
 * @param [in] code  color conversion code see \p NVCVColorConversionCode group.
 *                   The following conversion codes are available for this operator:
 *
 *                  Interleaved Y,U,V <-> R,G,B tensors are (n)HWC with C = 3 for YUV/RGB components
 *                       NVCV_COLOR_YUV2BGR
 *                       NVCV_COLOR_YUV2RGB
 *                       NVCV_COLOR_BGR2YUV
 *                       NVCV_COLOR_RGB2YUV
 *
 *                  Semi planar Y,U,V <-> R,G,B tensors are (n)HWC with C = 3 for RGB and C = 1.
 *                  For YUV NV12/21 tensors H = (pixel height) * 3/2, w = (pixel width) and bottom 1/3 of the tensor contains interlaced VU data.
 *                       NVCV_COLOR_YUV2RGB_NV12
 *                       NVCV_COLOR_YUV2BGR_NV12
 *                       NVCV_COLOR_YUV2RGB_NV21
 *                       NVCV_COLOR_YUV2BGR_NV21
 *                       NVCV_COLOR_RGB2YUV_NV12
 *                       NVCV_COLOR_BGR2YUV_NV12
 *                       NVCV_COLOR_RGB2YUV_NV21
 *                       NVCV_COLOR_BGR2YUV_NV21
 *
 * @param [in] spec color conversion spec see \p NVCVColorSpec group.
 *                   The following spec codes are available for this operator:
 *                       NVCV_COLOR_SPEC_BT601 // Color specification for SDTV (Standard Definition TV) - BT.601
 *                       NVCV_COLOR_SPEC_BT709 // Color specification for HDTV (High Definition TV) - BT.709
 *                       NVCV_COLOR_SPEC_BT2020 // Color specification for UHDTV (Ultra High Definition TV) - BT.2020
 *
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdvCvtColorSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out, NVCVColorConversionCode code,
                                                 NVCVColorSpec spec);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__ADV_CVT_COLOR_H */

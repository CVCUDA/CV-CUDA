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
 * @file OpLabel.h
 *
 * @brief Defines types and functions to handle the Label operation.
 * @defgroup NVCV_C_ALGORITHM_LABEL Label
 * @{
 */

#ifndef CVCUDA_LABEL_H
#define CVCUDA_LABEL_H

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

/**
 * Constructs an instance of the Label operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaLabelCreate(NVCVOperatorHandle *handle);

/**
 * Executes the Label operation on the given cuda stream. This operation does not wait for completion.
 *
 * This operation computes the connected-component labeling of one or more input images (in 2D) or volumes (in 3D)
 * inside the input tensor, yielding labels in the output tensor with same rank and shape.  Labels are numbers
 * uniquely assigned to each connected region, for example:
 *
 * Input   0 0 0 0  Output   0 0 0 0
 * image:  1 1 0 1  labels:  4 4 0 7
 *         0 0 0 1           0 0 0 7
 *         0 1 1 1           0 7 7 7
 *
 * In the above example, three distinct regions were identified and labeled as 0, 4 and 7.  Note that the region
 * labeled with 0 remained with the same value as the input, and label numbers 4 and 7 were assigned in
 * non-consecutive ordering.  Some values in the input may be ignored, i.e. not labeled, using the \ref bgLabel
 * tensor to define those values as background, which usually is set to the value zero.  For example:
 *
 * Input   0 0 1 0  Output   0 0 2 3  Zeros in  0 0 2 0
 * image:  0 1 0 1  labels:  0 5 6 7  bgLabel:  0 5 0 7
 *         0 0 1 1           0 0 7 7            0 0 7 7
 *         0 1 1 1           0 7 7 7            0 7 7 7
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [HWC], [NHWC], [DHWC], [NDHWC]
 *       Channels:       [1]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | Yes
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | Yes
 *       32bit Signed   | Yes
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [HWC], [NHWC], [DHWC], [NDHWC]
 *       Channels:       [1]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | No
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | Yes
 *       32bit Signed   | No
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | No
 *       Number        | Yes
 *       Channels      | Yes
 *       Width         | Yes
 *       Height        | Yes
 *       Depth         | Yes
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor.  The expected layout is [HWC] or [NHWC] for 2D labeling or [DHWC] or [NDHWC] for
 *                3D labeling, with either explicit C dimension or missing C with channels embedded in the data type.
 *                The N dimension is the number of samples, i.e. either 2D images with height H and width W or
 *                3D volumes with depth D and height H and width W, inside the tensor.  This operator labels
 *                regions, i.e. connected components, of each input image or volume read from the \ref in tensor.
 *                + Check above limitations table to the input tensor data layout, number of channels and data type.
 *
 * @param [out] out Output tensor.  The expected layout is [HWC] or [NHWC] for 2D labeling or [DHWC] or [NDHWC] for
 *                  3D labeling, with either explicit C dimension or missing C with channels embedded in the data type.
 *                  The N dimension is the number of samples, i.e. either 2D images with height H and width W or
 *                  3D volumes with depth D and height H and width W, inside the tensor.  This operator labels
 *                  regions, i.e. connected components, on the input writing the labels to the \ref out tensor.
 *                  + Check above limitations table to the output tensor data layout, number of channels and data type.
 *
 * @param [in] bgLabel Background label tensor.  The expected layout is [N] or [NC], meaning rank-1 or rank-2
 *                     tensor with first dimension as the number of samples N, matching input and output tensors,
 *                     and a potential last dimension C with number of channels.  If present, this tensor is used
 *                     by the operator to define background values in the input tensor to be ignored during
 *                     labeling.  If not present, all values in the input are labeled.
 *                     + It must have the same number of samples as input and output tensors.
 *                     + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                     + It must have data type the same as the input.
 *                     + It may be NULL to consider all values in the input as valid values to be labeled.
 *
 * @param [in] minThresh Minimum-threshold value tensor.  The expected layout is [N] or [NC], meaning rank-1 or
 *                       rank-2 tensor with first dimension as the number of samples N, matching input and output
 *                       tensors, and a potential last dimension C with number of channels.  If present, this
 *                       tensor is used by the operator as a pre-filter step to define minimum values in the input
 *                       tensor to be thresholded into a binary image, i.e. values below it are set to 0 and above
 *                       or equal it are set to 1.  Labeling is done after this pre-filter step, where \ref
 *                       bgLabel may be applied for instance to ignore zeroes as background.
 *                       + It must have the same number of samples as input and output tensors.
 *                       + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                       + It must have data type the same as the input.
 *                       + It may be NULL to not apply minimum thresholding as a pre-filter.
 *
 * @param [in] maxThresh Maximum-threshold value tensor.  The expected layout is [N] or [NC], meaning rank-1 or
 *                       rank-2 tensor with first dimension as the number of samples N, matching input and output
 *                       tensors, and a potential last dimension C with number of channels.  If present, this
 *                       tensor is used by the operator as a pre-filter step to define maximum values in the input
 *                       tensor to be thresholded into a binary image, i.e. values above it are set to 0 and below
 *                       or equal it are set to 1.  Labeling is done after this pre-filter step, where \ref
 *                       bgLabel may be applied for instance to ignore zeroes as background.
 *                       + It must have the same number of samples as input and output tensors.
 *                       + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                       + It must have data type the same as the input.
 *                       + It may be NULL to not apply maximum thresholding as a pre-filter.
 *
 * @param [in] minSize Minimum-size value tensor.  The expected layout is [N] or [NC], meaning rank-1 or rank-2
 *                     tensor with first dimension as the number of samples N, matching input and output tensors,
 *                     and a potential last dimension C with number of channels.  If present, this tensor is used
 *                     by the operator as a post-filter step to define minimum-size regions in the output tensor to
 *                     keep their labels, i.e. connected-component regions with less than this minimum number of
 *                     elements are set to the background value defined in the \ref bgLabel value.  Labeling is
 *                     done before this post-filter step, also known as island-removal step.
 *                     + It must have the same number of samples as input and output tensors.
 *                     + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                     + It must have U32 data type.
 *                     + It may be NULL to not apply minimum size regions removal as a post-filter.
 *                     + If not NULL, the \ref bgLabel and \ref stats tensors must not be NULL as well.
 *
 * @param [out] count Count of labels tensor.  The expected layout is [N] or [NC], meaning rank-1 or rank-2 tensor
 *                    with first dimension as the number of samples N, matching input and output tensors, and a
 *                    potential last dimension C with number of channels.  If present, this tensor is used by the
 *                    operator to store the number of connected regions, or components, labeled.  The background
 *                    label is ignored and thus not counted.  It counts regions that may be beyond the maximum capacity
 *                    of \ref stats tensor, and regions potentially removed by \ref minSize tensor.
 *                    + It must have the same number of samples as input and output tensors.
 *                    + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                    + It must have U32 data type.
 *                    + It may be NULL to disregard counting the number of different labels found.
 *
 * @param [out] stats Statistics tensor.  The expected layout is [NMA], meaning rank-3 tensor with first dimension
 *                    as the number of samples N, matching input and output tensors, second dimension M as maximum
 *                    number of different labels statistics to be computed, and a third dimension A as the amount
 *                    of statistics to be computed per label (fixed as 6 for 2D or 8 for 3D).  If present, this
 *                    tensor is used by the operator to store information per connected-component label.  The
 *                    background label is ignored and thus its statistics is not computed.
 *                    + It must have the same number of samples as input and output tensors.
 *                    + It must have a number of statistics M per sample N equal to the maximum allowed number of
 *                      label statistics that can be computed by the Label operator per sample image (or volume).
 *                      The actual number of labels found is stored in \ref count (see above).
 *                    + For 2D labeling, it must have in the last dimension A=6 elements to store at: (0) the
 *                      original label number; (1) leftmost position; (2) topmost position; (3) width size; (4)
 *                      height size; (5) count of pixels (i.e. size of the labeled region).  And for 3D labeling,
 *                      it must have in the last dimension A=8 elements to store at: (0) the original label number;
 *                      (1) leftmost position; (2) topmost position; (3) shallowmost position; (4) width size; (5)
 *                      height size; (6) depth size; (7) count of voxels (i.e. size of the labeled region).
 *                    + It must have U32 data type.
 *                    + It may be NULL to disregard computing statistics information on different labels found.
 *                    + It must not be NULL if \ref assignLabel is NVCV_LABEL_SEQUENTIAL, the index of each label
 *                      statistics is used as the new sequential label replacing the original label in the output,
 *                      the sequential labels are up to the maximum capacity M
 *                    + If not NULL, the \ref count tensor must not be NULL as well.
 *
 * @param [in] connectivity Specify connectivity of elements for the operator, see \ref NVCVConnectivityType.
 *                          + It must conform with \ref in and \ref out tensors, i.e. 3D labeling requires [DHWC]
 *                            or [NDHWC] tensor layouts and 2D labeling requires [HWC] or [NHWC], where the C
 *                            channel may be missing as embedded in data type.
 *
 * @param [in] assignLabels Specify how labels are assigned by the operator, see \ref NVCVLabelType.  Use
 *                          NVCV_LABEL_FAST to do fast labeling, i.e. assign non-consecutive label numbers fast.
 *                          Use NCVC_LABEL_SEQUENTIAL to have consecutive label numbers instead.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaLabelSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                           NVCVTensorHandle out, NVCVTensorHandle bgLabel, NVCVTensorHandle minThresh,
                                           NVCVTensorHandle maxThresh, NVCVTensorHandle minSize, NVCVTensorHandle count,
                                           NVCVTensorHandle stats, NVCVConnectivityType connectivity,
                                           NVCVLabelType assignLabels);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_LABEL_H */

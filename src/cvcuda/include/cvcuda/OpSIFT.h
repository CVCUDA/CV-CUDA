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
 * @file OpSIFT.h
 *
 * @brief Defines types and functions to handle the SIFT operation.
 * @defgroup NVCV_C_ALGORITHM_SIFT SIFT
 * @{
 */

#ifndef CVCUDA_SIFT_H
#define CVCUDA_SIFT_H

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

/** Constructs and an instance of the SIFT operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] maxShape Maximum shape of input tensor images as WxHxN, i.e. x=W y=H z=N of int3, where W=width,
 *                      H=height and N=samples, or number of images in tensor.
 *                      + N (z coordinate) must be >= 1 and <= 65535.
 *                      + W and H (x and y) must be >= 2, and they must take into account if the input is to be
 *                        expanded during execution time, \ref NVCVSIFTFlagType.
 *
 * @param [in] maxOctaveLayers Maximum layers per octave to be used by the operator, an octave is a level in the
 *                             Gaussian pyramid containing input scale-space layers in the SIFT algorithm.
 *                             + It must be >= 1 and <= 16.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_INVALID_ARGUMENT An argument is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaSIFTCreate(NVCVOperatorHandle *handle, int3 maxShape, int maxOctaveLayers);

/** Executes the SIFT operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 * @note The SIFT operation does not guarantee deterministic output.  Each output tensor limits the number of
 *       features found by the operator, that is the total number may be greater than this limitation and the order
 *       of features returned might differ in different runs.  Although the order of features found is random
 *       within each image of the input tensor, their relative position across output tensors is consistent.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [HWC, NHWC]
 *       Channels:       [1]
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
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor.  The expected layout is [HWC] or [NHWC], where N is the number of samples,
 *                i.e. images with height H and width W and channels C, inside the tensor.  This operator extracts
 *                features and computes descriptors of each input image in the \ref in tensor.
 *                + Check above limitations table to the input tensor data layout, number of channels and data type.
 *
 * @param [out] featCoords Output tensor with features coordinates.  The expected layout is [NM] or [NMC] meaning a
 *                         rank-2 or rank-3 tensor with first dimension as number of samples N, second dimension M
 *                         as maximum number of features to be found, and a potential last dimension C with number
 *                         of feature coordinates.
 *                         + It must have the same number of samples N as input tensor.
 *                         + It must have a number of elements M per sample N equal to the maximum allowed number
 *                           of features that can be extracted by the SIFT algorithm per sample image.  This number
 *                           M must be the same for all output tensors.  The actual number of features extracted is
 *                           stored in \ref numFeatures (see below).
 *                         + It must have F32 data type with C=4 or 4F32 data type with C=1 (or C not present in
 *                           tensor) to store feature coordinates (x, y) along sample image width W and height H,
 *                           respectively, coordinate (z) of the octave (i.e. the level of the SIFT Gaussian
 *                           pyramid) and coordinate (w) of the layer in the octave (i.e. the scale-space layer in
 *                           the pyramid level) of each extracted feature.
 *
 * @param [out] featMetadata Output tensor with features metadata: orientation angle, score response and size.  The
 *                           expected layout is [NM] or [NMC] meaning a rank-2 or rank-3 tensor with first
 *                           dimension as number of samples N, second dimension M as maximum number of features to
 *                           be found, and a potential last dimension C with number of feature metadata.
 *                           + It must have the same number of samples N as input tensor.
 *                           + It must have a number of elements M per sample N equal to the maximum allowed number
 *                             of features that can be extracted by the SIFT algorithm per sample image.  This
 *                             number M must be the same for all output tensors.  The actual number of features
 *                             extracted is stored in \ref numFeatures (see below).
 *                           + It must have F32 data type with C=3 or 3F32 data type with C=1 (or C not present in
 *                             tensor) to store orientation angle in (x), score response in (y) and feature size in
 *                             (z) of each extracted feature.
 *
 * @param [out] featDescriptors Output tensor with features descriptors.  The expected layout is [NMD] meaning a
 *                              rank-3 tensor with first dimension as number of samples N, second dimension M as
 *                              maximum number of features to be found, and a third dimension D as depth of each
 *                              feature descriptor (SIFT descriptor has a fixed 128-Byte depth).
 *                              + It must have the same number of samples N as input tensor.
 *                              + It must have a number of elements M per sample N equal to the maximum allowed
 *                                number of features that can be extracted by the SIFT algorithm per sample image.
 *                                This number M must be the same for all output tensors.  The actual number of
 *                                features extracted is stored in \ref numFeatures (see below).
 *                              + It must have U8 data type and D=128 to store each 128-Byte feature descriptor.
 *
 * @param [out] numFeatures Output tensor to store the number of features found in the input tensor.  The expected
 *                          layout is [N] or [NC], meaning rank-1 or rank-2 tensor with first dimension as number
 *                          of samples N, and a potential last dimension C with number of channels.  It expresses
 *                          the total number of features found, regardless of the maximum allowed number of
 *                          features M in output tensors (see above).  Since features are found randomly on each
 *                          image in input tensor, they are discarded in a non-deterministic way when the number of
 *                          features found is bigger than M.
 *                          + It must have the same number of samples as input tensor.
 *                          + It must have S32 data type to store number of features found.
 *                          + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *
 * @param [in] numOctaveLayers Number of layers in each octave.  Since the minimum number of layers is 3, the
 *                             actual number is 3 + numOctaveLayers.  One suggestion, given by the original
 *                             algorithm description, is to use numOctaveLayers = 3.  The number of octaves is
 *                             computed from the input image resolution WxH as \f$ log(min(W, H))/log(2) - 2 \f$.
 *                             + It must be positive.
 *                             + It must be at most \ref maxOctaveLayers, that is defined in operator constructor
 *                               \ref cvcudaSIFTCreate.
 *
 * @param [in] contrastThreshold The contrast threshold used to remove features with low contrast.  The larger this
 *                               threshold, the less features are extracted by the operator.  One suggestion, given
 *                               by the original algorithm description, is to use \f$ 0.03 \f$.
 *                               + It must be between 0 and 1.
 *
 * @param [in] edgeThreshold The edge threshold used to remove features that are similar to edges.  The larger this
 *                           threshold, the more features are extracted by the operator. One suggestion, given by
 *                           the original algorithm description, is to use \f$ 10.0 \f$.
 *                           + It must be between 0 and 1.
 *
 * @param [in] initSigma The initial sigma to be applied by the first Gaussian filter done at the first octave.
 *                       This sigma is progressively applied for each scale-space layer within each octave
 *                       (i.e. the level of the SIFT Gaussian pyramid). One suggestion, given by the original
 *                       algorithm description, is to use same sigma equals to 1.6.
 *                       + It must be positive.
 *
 * @param [in] flags Set up additional flags for SIFT operator, see \ref NVCVSIFTFlagType.  It supports one flag to
 *                   control whether to expand input images by a factor of 2, using bilinear interpolation, prior
 *                   to building the SIFT Gaussian scale-space pyramid.  This is to avoid ignoring the highest
 *                   spatial frequencies.  One suggestion, given by the original algorithm description, is to apply
 *                   this expansion and thus use flags = \ref NVCV_SIFT_USE_EXPANDED_INPUT_SIZE.
 *                   + It must be one of \ref NVCVSIFTFlagType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaSIFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                          NVCVTensorHandle featCoords, NVCVTensorHandle featMetadata,
                                          NVCVTensorHandle featDescriptors, NVCVTensorHandle numFeatures,
                                          int numOctaveLayers, float contrastThreshold, float edgeThreshold,
                                          float initSigma, NVCVSIFTFlagType flags);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_SIFT_H */

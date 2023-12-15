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
 * @file OpairwiseMatcher.h
 *
 * @brief Defines types and functions to handle the airwiseMatcher operation.
 * @defgroup NVCV_C_ALGORITHM_PAIRWISE_MATCHER Pairwise Matcher
 * @{
 */

#ifndef CVCUDA_PAIRWISE_MATCHER_H
#define CVCUDA_PAIRWISE_MATCHER_H

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

/** Constructs and an instance of the PairwiseMatcher operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] algoChoice Choice of algorithm to find pair-wise matches.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaPairwiseMatcherCreate(NVCVOperatorHandle *handle, NVCVPairwiseMatcherType algoChoice);

/** Executes the PairwiseMatcher operation on the given CUDA stream. This operation does not wait for completion.
 *
 * This operation computes the pair-wise matcher between two sets of n-dimensional points.  For instance
 * 128-dimensional descriptors as points.  For each point $p1_i$, in the 1st set defined by \ref set1 with size
 * \ref numSet1, the operator finds the best match (minimum distance) from $p1_i$ to a point in the 2nd set $p2_j$,
 * defined by \ref set2 with size \ref numSet2.  If \ref crossCheck is true, $p1_i$ must also be the best match
 * from $p2_j$ considering all possible matches from the 2nd set to the 1st set, to return them as a match.
 *
 * @note This operation does not guarantee deterministic output.  Each output tensor limits the number of matches
 *       found by the operator, that is the total number may be greater than this limitation and the order of
 *       matches returned might differ in different runs.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a CUDA stream.
 *                    + Must be a valid CUDA stream.
 *
 * @param [in] set1 Input 1st set of points tensor.  The first set of points to calculate pair-wise matcher between
 *                  this 1st set and the 2nd set.  The expected layout is [NMD] meaning a rank-3 tensor with first
 *                  dimension as number of samples N, second dimension M as maximum number of points, and a third
 *                  dimension D as depth dimension of each point, e.g. the output of \ref NVCV_C_ALGORITHM_SIFT has
 *                  128-Byte descriptor or D=128 and U8 data type that can be used as a set of points.
 *                  + It must have consistent number of samples N across input and output tensors.
 *                  + The size of the depth dimension D and data type must be consistent across input
 *                    set of points tensors.
 *                  + It must have U8 or U32 or F32 data type.
 *
 * @param [in] set2 Input 2nd set of points tensor.  The second set of points to calculate pair-wise matcher between
 *                  this 2nd set and the 1st set.  The expected layout is [NMD] meaning a rank-3 tensor with first
 *                  dimension as number of samples N, second dimension M as maximum number of points, and a third
 *                  dimension D as depth dimension of each point, e.g. the output of \ref NVCV_C_ALGORITHM_SIFT has
 *                  128-Byte descriptor or D=128 and U8 data type that can be used as a set of points.
 *                  + It must have consistent number of samples N across input and output tensors.
 *                  + The size of the depth dimension D and data type must be consistent across input
 *                    set of points tensors.
 *                  + It must have U8 or U32 or F32 data type.
 *
 * @param [in] numSet1 Input tensor storing the actual number of points in \ref set1 tensor.  The expected layout
 *                     is [N] or [NC], meaning rank-1 or rank-2 tensor with first dimension as number of samples N,
 *                     and a potential last dimension C with number of channels.  It expresses the total number of
 *                     valid points in \ref set1 if less than its maximum capacity M, else uses all M points.
 *                     + It must have consistent number of samples N across input and output tensors.
 *                     + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                     + It must have S32 data type.
 *                     + It may be NULL to use entire set1 maximum capacity M as valid points.
 *
 * @param [in] numSet2 Input tensor storing the actual number of points in \ref set2 tensor.  The expected layout
 *                     is [N] or [NC], meaning rank-1 or rank-2 tensor with first dimension as number of samples N,
 *                     and a potential last dimension C with number of channels.  It expresses the total number of
 *                     valid points in \ref set2 if less than its maximum capacity M, else uses all M points.
 *                     + It must have consistent number of samples N across input and output tensors.
 *                     + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                     + It must have S32 data type.
 *                     + It may be NULL to use entire set2 maximum capacity M as valid points.
 *
 * @param [out] matches Output tensor to store the matches of points between 1st set \ref set1 and 2nd set \ref
 *                      set2.  The expected layout is [NMA], meaning rank-3 tensor with first dimension as the
 *                      number of samples N, same as other tensors, second dimension M as maximum number of
 *                      matches, not necessarily the same as other tensors, and third dimension A as the attributes
 *                      of each match, fixed to 2 attributes: set1 index and set2 index.
 *                      + It must have consistent number of samples N across input and output tensors.
 *                      + It must have a number of matches M per sample N equal to the maximum allowed number of
 *                        matches to be found between \ref set1 and \ref set2.  The actual number
 *                        of matches found is stored in \ref numMatches.
 *                      + It must have size of attributes dimension A equal 2.
 *                      + It must have S32 data type.
 *
 * @param [out] numMatches Output tensor to store the number of matches found by the operator.  The expected layout
 *                         is [N] or [NC], meaning rank-1 or rank-2 tensor with first dimension as number of
 *                         samples N, and a potential last dimension C with number of channels.  It expresses the
 *                         toal number of matches found, regardless of the maximum allowed number of matches M in
 *                         output tensor \ref matches.  Since matches are found randomly, they are discarded in a
 *                         non-deterministic way when the number of matches found is bigger than M.
 *                         + It must have consistent number of samples N across input and output tensors.
 *                         + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                         + It must have S32 data type.
 *                         + It may be NULL if \ref crossCheck is false to disregard storing number of matches.
 *
 * @param [out] distances Output tensor to store distances of matches found by the operator.  The expected layout
 *                        is [NM] or [NMC], meaning rank-2 or rank-3 tensor with first dimension as number of
 *                        samples N, same as other tensors, second dimension M as maximum number of distances, same
 *                        as \ref matches output tensors, and a potential last dimension C with number of channels.
 *                        For each match found in \ref matches, the distance between matched points is stored.
 *                        + It must have consistent number of samples N across input and output tensors.
 *                        + It must have the same dimension M of the \ref matches tensor, meaning the maximum
 *                          allowed number of distances must be equal to the maximum allowed number of matches.
 *                        + It must have one element per sample, i.e. number of channels must be 1 in a [NMC] tensor.
 *                        + It must have F32 data type.
 *                        + It may be NULL to disregard storing distances.
 *
 * @param [in] crossCheck Choice to do cross check.  Use false to search only for matches from 1st set of points in
 *                        \ref set1 to 2nd set of points in \ref set2.  Use true to cross check best matches, a
 *                        best match is only returned if it is the best match (minimum distance) from 1st set to
 *                        2nd set and vice versa.
 *
 * @param [in] matchesPerPoint Number of best matches $k$ per point.  The operator returns the top-$k$ best matches
 *                             from 1st set to 2nd set.
 *                             + It must be between 1 and 64.
 *                             + It has to be 1 if \ref crossCheck is true.
 *
 * @param [in] normType Choice of norm type to normalize distances, used in points difference $|p1 - p2|$.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaPairwiseMatcherSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                     NVCVTensorHandle set1, NVCVTensorHandle set2,
                                                     NVCVTensorHandle numSet1, NVCVTensorHandle numSet2,
                                                     NVCVTensorHandle matches, NVCVTensorHandle numMatches,
                                                     NVCVTensorHandle distances, bool crossCheck, int matchesPerPoint,
                                                     NVCVNormType normType);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_PAIRWISE_MATCHER_H */

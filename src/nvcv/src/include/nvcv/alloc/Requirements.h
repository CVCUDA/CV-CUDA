/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file Requirements.h
 *
 * @brief Defines the public C interface to NVCV resource requirements.
 *
 * Several objects in NVCV require resource allocation. Resource requirements
 * is a way for them to inform how many resources they need. This information
 * can be used by allocators to pre-allocate the resources that will be used.
 */

#ifndef NVCV_REQUIREMENTS_H
#define NVCV_REQUIREMENTS_H

#include "../Status.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define NVCV_MAX_MEM_REQUIREMENTS_LOG2_BLOCK_SIZE (32)

#define NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE (((int64_t)1) << NVCV_MAX_MEM_REQUIREMENTS_LOG2_BLOCK_SIZE)

/** Store memory resource requirements */
typedef struct NVCVMemRequirementsRec
{
    /** Total amount of blocks of given size needed, in bytes.
     * The index is log2(size).
     * Each block will be aligned to 'size'.
     * */
    int64_t numBlocks[NVCV_MAX_MEM_REQUIREMENTS_LOG2_BLOCK_SIZE];
} NVCVMemRequirements;

/** Stores resource requirements. */
typedef struct NVCVRequirementsRec
{
    NVCVMemRequirements cudaMem;       /*< Device memory */
    NVCVMemRequirements hostMem;       /*< Host memory */
    NVCVMemRequirements hostPinnedMem; /*< Host-pinned memory */
} NVCVRequirements;

/** Initializes the resource requirements data to zero.
 *
 * @param [out] req Requirements to be initialized to zero
 *                  + Must not be NULL
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval NVCV_SUCCESS                  Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvRequirementsInit(NVCVRequirements *req);

/** Adds the given requirements together.
 *
 * Used when calculating the total resource requirements needed by
 * several objects that can be used simultaneously.
 *
 * Each resource requirement is added independently.
 *
 * @param [in,out] reqSum Where the summation is stored.
 *                        + Must not be NULL
 *
 * @param [in] req Requirements to be added.
 *                 + Must not be NULL
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval NVCV_SUCCESS                  Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvRequirementsAdd(NVCVRequirements *reqSum, const NVCVRequirements *req);

/** Calculate the total size in bytes of the given memory requirement.
 *
 * @param [in] memReq Memory requirement to be queried.
 *                    + Must not be NULL
 *
 * @param [out] size_t Calculated size in bytes.
 *                 + Must not be NULL
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval NVCV_SUCCESS                  Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMemRequirementsCalcTotalSizeBytes(const NVCVMemRequirements *memReq, int64_t *sizeBytes);

/** Adds the given buffer size to the memory requirements.
 *
 * It'll calculate the correct slot based on the alignment where the buffer request
 * is added. It'll add enough blocks with `bufAlignment` size to satisfy the memory
 * request.
 *
 * @param [in] memReq Memory requirement to be modified.
 *                    + Must not be NULL
 *
 * @param [in] bufSize Size in bytes of the buffer memory request.
 *                     If negative, request will be removed from requirements.
 *                     Underflows are clamped to 0.
 *
 * @param [in] bufAlignment Alignment of the memory buffer, in bytes.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval NVCV_SUCCESS                  Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMemRequirementsAddBuffer(NVCVMemRequirements *memReq, int64_t bufSize, int64_t bufAlignment);

#ifdef __cplusplus
}
#endif

#endif // NVCV_REQUIREMENTS_H

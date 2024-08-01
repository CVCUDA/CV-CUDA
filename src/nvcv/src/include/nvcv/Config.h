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
 * @file Config.h
 *
 * @brief Public C interface to NVCV configuration.
 */

#ifndef NVCV_CONFIG_H
#define NVCV_CONFIG_H

#include "Export.h"
#include "Status.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Set a hard limit on the number of image handles that can be created.
 *
 * The function will preallocate all resources necessary to satisfy creation
 * of a limited number of image handles in the future.
 *
 * @param[in] maxCount Maximum number of image handles that can be created.
 *                     If negative, switches to dynamic allocation, no hard limit is defined.
 *                     + There must be no image handles created and not destroyed.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvConfigSetMaxImageCount(int32_t maxCount);

/**
 * Set a hard limit on the number of image batch handles that can be created.
 *
 * The function will preallocate all resources necessary to satisfy creation
 * of a limited number of image batch handles in the future.
 *
 * @param[in] maxCount Maximum number of image batch handles that can be created.
 *                     If negative, switches to dynamic allocation, no hard limit is defined.
 *                     + There must be no image batch handles created and not destroyed.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvConfigSetMaxImageBatchCount(int32_t maxCount);

/**
 * Set a hard limit on the number of tensor handles that can be created.
 *
 * The function will preallocate all resources necessary to satisfy creation
 * of a limited number of tensor handles in the future.
 *
 * @param[in] maxCount Maximum number of tensor handles that can be created.
 *                     If negative, switches to dynamic allocation, no hard limit is defined.
 *                     + There must be no tensor handles created and not destroyed.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvConfigSetMaxTensorCount(int32_t maxCount);

/**
 * Set a hard limit on the number of array handles that can be created.
 *
 * The function will preallocate all resources necessary to satisfy creation
 * of a limited number of array handles in the future.
 *
 * @param[in] maxCount Maximum number of array handles that can be created.
 *                     If negative, switches to dynamic allocation, no hard limit is defined.
 *                     + There must be no array handles created and not destroyed.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvConfigSetMaxArrayCount(int32_t maxCount);

/**
 * Set a hard limit on the number of allocator handles that can be created.
 *
 * The function will preallocate all resources necessary to satisfy creation
 * of a limited number of allocator handles in the future.
 *
 * @param[in] maxCount Maximum number of allocator handles that can be created.
 *                     If negative, switches to dynamic allocation, no hard limit is defined.
 *                     + There must be no allocator handles created and not destroyed.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvConfigSetMaxAllocatorCount(int32_t maxCount);

#ifdef __cplusplus
}
#endif

#endif // NVCV_CONFIG_H

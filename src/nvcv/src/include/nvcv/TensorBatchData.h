/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TENSORBATCHDATA_H
#define NVCV_TENSORBATCHDATA_H

#include "TensorData.h"
#include "TensorLayout.h"

#include <stdalign.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Describes a single tensor in a batch */
typedef struct NVCVTensorBatchElementStridedRec
{
    alignas(128) NVCVByte *data;
    int64_t shape[NVCV_TENSOR_MAX_RANK];
    int64_t stride[NVCV_TENSOR_MAX_RANK];
} NVCVTensorBatchElementStrided;

/** Describes a batch of tensors */
typedef struct NVCVTensorBatchBufferStridedRec
{
    NVCVTensorBatchElementStrided *tensors;
} NVCVTensorBatchBufferStrided;

typedef union NVCVTensorBatchBufferRec
{
    NVCVTensorBatchBufferStrided strided;
} NVCVTensorBatchBuffer;

typedef struct NVCVTensorBatchDataRec
{
    NVCVDataType     dtype;
    NVCVTensorLayout layout;
    int32_t          rank;
    int32_t          numTensors;

    NVCVTensorBufferType  type;
    NVCVTensorBatchBuffer buffer;
} NVCVTensorBatchData;

#ifdef __cplusplus
}
#endif

#endif // NVCV_TENSORBATCHDATA_H

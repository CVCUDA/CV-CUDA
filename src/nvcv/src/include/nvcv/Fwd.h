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
 * @file Fwd.h
 *
 * @brief Forward declaration of some public C interface entities.
 */

#ifndef NVCV_FWD_H
#define NVCV_FWD_H

#include "alloc/Fwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct NVCVImage       *NVCVImageHandle;
typedef struct NVCVImageBatch  *NVCVImageBatchHandle;
typedef struct NVCVTensor      *NVCVTensorHandle;
typedef struct NVCVTensorBatch *NVCVTensorBatchHandle;
typedef struct NVCVArray       *NVCVArrayHandle;

#ifdef __cplusplus
}
#endif

#endif // NVCV_FWD_H

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Author: Jeff Daily <jeff.daily@amd.com>
 */

#pragma once
#include "CvCudaHipCompat.h"
#include <hip/library_types.h>

// CUDA's library_types.h provides cudaDataType / CUDA_R_*; map to HIP's.
#define cudaDataType hipDataType
#define CUDA_R_16F   HIP_R_16F
#define CUDA_R_32F   HIP_R_32F
#define CUDA_R_64F   HIP_R_64F
#define CUDA_C_32F   HIP_C_32F
#define CUDA_C_64F   HIP_C_64F

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Author: Jeff Daily <jeff.daily@amd.com>
 */

#pragma once
#include "CvCudaHipCompat.h"
#include <hipsolver/hipsolver.h>

// OpFindHomography solves a batched symmetric eigenproblem (syevjBatched). The
// hipSOLVER dense Dn API matches cuSOLVER argument-for-argument, so each call
// site ports by name alone.
#define cusolverStatus_t              hipsolverStatus_t
#define CUSOLVER_STATUS_SUCCESS       HIPSOLVER_STATUS_SUCCESS
#define cusolverDnHandle_t            hipsolverDnHandle_t
#define cusolverEigMode_t             hipsolverEigMode_t
#define CUSOLVER_EIG_MODE_VECTOR      HIPSOLVER_EIG_MODE_VECTOR
#define CUSOLVER_EIG_MODE_NOVECTOR    HIPSOLVER_EIG_MODE_NOVECTOR
#define syevjInfo_t                   hipsolverSyevjInfo_t

#define cusolverDnCreate              hipsolverDnCreate
#define cusolverDnDestroy             hipsolverDnDestroy
#define cusolverDnSetStream           hipsolverDnSetStream
#define cusolverDnCreateSyevjInfo     hipsolverDnCreateSyevjInfo
#define cusolverDnDestroySyevjInfo    hipsolverDnDestroySyevjInfo
#define cusolverDnXsyevjSetTolerance  hipsolverDnXsyevjSetTolerance
#define cusolverDnXsyevjSetMaxSweeps  hipsolverDnXsyevjSetMaxSweeps
#define cusolverDnXsyevjSetSortEig    hipsolverDnXsyevjSetSortEig
#define cusolverDnSsyevjBatched_bufferSize hipsolverDnSsyevjBatched_bufferSize
#define cusolverDnSsyevjBatched       hipsolverDnSsyevjBatched
#define cusolverDnDsyevjBatched_bufferSize hipsolverDnDsyevjBatched_bufferSize
#define cusolverDnDsyevjBatched       hipsolverDnDsyevjBatched

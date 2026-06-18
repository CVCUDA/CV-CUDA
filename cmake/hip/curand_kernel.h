/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Author: Jeff Daily <jeff.daily@amd.com>
 */

#pragma once
#include "CvCudaHipCompat.h"
#include <cstdio> // rocRAND's mtgp32 header calls printf without including it
#include <hiprand/hiprand_kernel.h>

#define curandState   hiprandState
#define curandState_t hiprandState_t
#define curand_init   hiprand_init
#define curand_normal hiprand_normal

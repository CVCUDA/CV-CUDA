/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Author: Jeff Daily <jeff.daily@amd.com>
 */

#pragma once
#include "../CvCudaHipCompat.h"
#include <hipcub/hipcub.hpp>

// CV-CUDA spells the namespace cub::; hipCUB lives in hipcub::. Aliasing the
// namespace keeps every cub::BlockReduce/BlockScan/BlockRadixSort/DeviceReduce
// and cub::BLOCK_* enum call site unchanged.
namespace hipcub
{
}
namespace cub = hipcub;

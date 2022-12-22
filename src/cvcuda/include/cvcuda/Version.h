/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file Version.h
 *
 * Functions and structures for handling NVCV operator library version.
 */

#ifndef CVCUDA_VERSION_H
#define CVCUDA_VERSION_H

#include "detail/Export.h"

#include <cvcuda/VersionDef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Retrieves the library's version number.
 * The number is represented as a integer. It may differ from \ref CVCUDA_VERSION if
 * header doesn't correspond to NVCV operator binary. This can be used by user's program
 * to handle semantic differences between library versions.
 */
CVCUDA_PUBLIC uint32_t cvcudaGetVersion(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif // CVCUDA_VERSION_H

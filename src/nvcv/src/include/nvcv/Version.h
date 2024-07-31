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
 * @file Version.h
 *
 * Functions and structures for handling NVCV library version.
 */

#ifndef NVCV_VERSION_H
#define NVCV_VERSION_H

#include "Export.h"

#include <nvcv/VersionDef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Retrieves the library's version number.
 * The number is represented as a integer. It may differ from \ref NVCV_VERSION if
 * header doesn't correspond to NVCV binary. This can be used by user's program
 * to handle semantic differences between library versions.
 */
NVCV_PUBLIC uint32_t nvcvGetVersion(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif // NVCV_VERSION_H

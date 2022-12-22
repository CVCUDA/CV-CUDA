/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
* @file Export.h
*
* @brief Export : Defines macros used for exporting symbols from DSO
*
*/

#ifndef CVCUDA_EXPORT_H
#define CVCUDA_EXPORT_H

#if defined _WIN32 || defined __CYGWIN__
#    ifdef CVCUDA_EXPORTING
#        define CVCUDA_PUBLIC __declspec(dllexport)
#    elif defined(CVCUDA_STATIC)
#        define CVCUDA_PUBLIC
#    else
#        define CVCUDA_PUBLIC __declspec(dllimport)
#    endif
#else
#    if __GNUC__ >= 4
#        define CVCUDA_PUBLIC __attribute__((visibility("default")))
#    else
#        define CVCUDA_PUBLIC
#    endif
#endif

#endif /* CVCUDA_EXPORT_H */

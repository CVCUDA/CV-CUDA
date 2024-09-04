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
* @file Export.h
*
* @brief Export : Defines macros used for exporting symbols from DSO
*
*/

#ifndef NVCV_EXPORT_H
#define NVCV_EXPORT_H

#if defined _WIN32 || defined __CYGWIN__
#    ifdef NVCV_EXPORTING
#        define NVCV_PUBLIC __declspec(dllexport)
#    elif defined(NVCV_STATIC)
#        define NVCV_PUBLIC
#    else
#        define NVCV_PUBLIC __declspec(dllimport)
#    endif
#else
#    if __GNUC__ >= 4
#        ifdef NVCV_STATIC
#            define NVCV_PUBLIC __attribute__((visibility("hidden")))
#        else
#            define NVCV_PUBLIC __attribute__((visibility("default")))
#        endif
#    else
#        define NVCV_PUBLIC
#    endif
#endif

#endif /* NVCV_EXPORT_H */

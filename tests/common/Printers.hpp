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

#ifndef NVCV_TEST_COMMON_PRINTERS_HPP
#define NVCV_TEST_COMMON_PRINTERS_HPP

#include <cuda_runtime.h>

#include <iostream>

#if NVCV_EXPORTING
#    include <nvcv_types/priv/ColorSpec.hpp>
#    include <nvcv_types/priv/DataLayout.hpp>
#    include <nvcv_types/priv/Status.hpp>
#else
#    include <nvcv/ColorSpec.hpp>
#    include <nvcv/DataLayout.hpp>
#    include <nvcv/Status.hpp>
#endif

#if NVCV_EXPORTING
inline std::ostream &operator<<(std::ostream &out, NVCVStatus status)
{
    return out << nvcv::priv::GetName(status);
}
#endif

std::ostream &operator<<(std::ostream &out, cudaError_t err);

#endif // NVCV_TEST_COMMON_PRINTERS_HPP

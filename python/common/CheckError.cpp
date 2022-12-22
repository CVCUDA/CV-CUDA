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

#include "CheckError.hpp"

#include <iostream>
#include <sstream>

namespace nvcvpy::util {

static std::string ToString(cudaError_t err)
{
    std::ostringstream ss;
    ss << cudaGetErrorName(err) << ": " << cudaGetErrorString(err);
    return ss.str();
}

void CheckThrow(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cudaGetLastError(); // consume the error
        throw std::runtime_error(ToString(err));
    }
}

void CheckLog(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cudaGetLastError(); // consume the error
        std::cerr << ToString(err) << std::endl;
    }
}

} // namespace nvcvpy::util

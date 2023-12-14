/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_UTIL_CUDA_EVENT_H_
#define NVCV_UTIL_CUDA_EVENT_H_

#include "UniqueHandle.hpp"

#include <driver_types.h>

#include <utility>

namespace nvcv::util {

/**
 * @brief A wrapper class for CUDA event handle (cudaEvent_t),
 *
 * The purpose of this class is to provide safe ownership and lifecycle management
 * for CUDA event handles.
 * The event object may be created using the factory functions @ref Create and @ref CreateWithFlags.
 *
 * The object may also assume ownership of a pre-existing handle via constructor or
 * @link UniqueHandle::reset(handle_type) reset @endlink function.
 */
class CudaEvent : public UniqueHandle<cudaEvent_t, CudaEvent>
{
public:
    NVCV_INHERIT_UNIQUE_HANDLE(cudaEvent_t, CudaEvent)
    constexpr CudaEvent() = default;

    /** @brief Creates an event on specified device (or current device, if deviceId < 0) */
    static CudaEvent Create(int deviceId = -1);

    /** @brief Creates an event event with specific flags on the device specified
   *         (or current device, if deviceId < 0)
   */
    static CudaEvent CreateWithFlags(unsigned flags, int deviceId = -1);

    /** @brief Calls cudaEventDestroy on the handle. */
    static void DestroyHandle(cudaEvent_t);
};

} // namespace nvcv::util

#endif // DALI_CORE_CUDA_EVENT_H_

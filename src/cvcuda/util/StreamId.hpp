/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_UTIL_STREAM_ID_HPP
#define NVCV_UTIL_STREAM_ID_HPP

#include <nvcv/detail/CudaFwd.h>

#include <cstdint>

namespace nvcv::util {

/** Retrieves a value that identifies a stream.
 *
 * @warning On older drivers ID aliasing is possible, when a still-running stream is deleted
 *          and a new one is created before the one just deleted completes its work.
 *
 * @param stream    CUDA stream handle (note that CUstram and cudaStream_t are one type)
 * @return int64_t  The ID of the stream within the system.
 */
uint64_t GetCudaStreamIdHint(CUstream stream);

/** Checks whether the stream id hint is unique
 *
 * If the system supports cuStreamGetId, then the value returned by GetCudaStreamIdHint
 * uniquely identifies a stream. This creates some optimization opportunities when managing
 * stream-bound resources.
 */
bool IsCudaStreamIdHintUnambiguous();

} // namespace nvcv::util

#endif // NVCV_UTIL_STREAM_ID_HPP

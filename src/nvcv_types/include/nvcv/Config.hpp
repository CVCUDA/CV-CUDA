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
 * @file Config.hpp
 *
 * @brief Public C++ interface to NVCV configuration.
 */

#ifndef NVCV_CONFIG_HPP
#define NVCV_CONFIG_HPP

#include "Config.h"
#include "detail/CheckError.hpp"

namespace nvcv { namespace cfg {

inline void SetMaxImageCount(int32_t maxCount)
{
    detail::CheckThrow(nvcvConfigSetMaxImageCount(maxCount));
}

inline void SetMaxImageBatchCount(int32_t maxCount)
{
    detail::CheckThrow(nvcvConfigSetMaxImageBatchCount(maxCount));
}

inline void SetMaxTensorCount(int32_t maxCount)
{
    detail::CheckThrow(nvcvConfigSetMaxTensorCount(maxCount));
}

inline void SetMaxAllocatorCount(int32_t maxCount)
{
    detail::CheckThrow(nvcvConfigSetMaxAllocatorCount(maxCount));
}

}} // namespace nvcv::cfg

#endif // NVCV_CONFIG_HPP

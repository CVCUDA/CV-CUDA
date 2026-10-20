/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_PRIV_OP_HQ_RESIZE_CONFIG_HPP
#define CVCUDA_PRIV_OP_HQ_RESIZE_CONFIG_HPP

#include "OpHQResizePolicy.hpp"

#include <nvcv/Exception.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>

namespace cvcuda::priv::hq_resize {

inline constexpr const char *kResizeBlockLanesEnvVar = "CVCUDA_HQ_RESIZE_BLOCK_LANES";
// Internal diagnostic override, not a supported public tuning surface. Any
// explicit value pins the adaptive var-shape layout; the historical value 8
// still permits the older dtype-specific tensor tuning below.
// A lane count scales one block-shape dimension at a time. Bound it by the
// largest dimension so no block-shape multiplication can overflow.
inline constexpr int         kMaximumResizeBlockLanes
    = std::numeric_limits<int>::max() / std::max(kResizeBlockDimX, std::max(kResizeBlockDimY, kResizeBlockDimZ));

struct ResizeBlockLanesConfig
{
    int  lanes;
    bool overridden;
};

constexpr bool AllowsDtypeSpecificTensorBlockLaneTuning(const ResizeBlockLanesConfig &config)
{
    // Explicitly selecting 8 preserves the dtype/device-specific tensor
    // tuning while every other override pins the requested geometry.
    return !config.overridden || config.lanes == kDefaultResizeBlockLanes;
}

inline ResizeBlockLanesConfig ParseResizeBlockLanesConfig(const char *value)
{
    // Preserve the internal override's historical integer-prefix syntax while
    // bounding the result so block-shape multiplication cannot overflow.
    if (value == nullptr)
    {
        return {kDefaultResizeBlockLanes, false};
    }

    errno            = 0;
    char      *end   = nullptr;
    const long lanes = std::strtol(value, &end, 10);
    if (errno == ERANGE || end == value || lanes < 1 || lanes > kMaximumResizeBlockLanes)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must contain an integer prefix in [1, %d]",
                              kResizeBlockLanesEnvVar, kMaximumResizeBlockLanes);
    }
    return {static_cast<int>(lanes), true};
}

inline const ResizeBlockLanesConfig &GetResizeBlockLanesConfig()
{
    static const ResizeBlockLanesConfig config = ParseResizeBlockLanesConfig(std::getenv(kResizeBlockLanesEnvVar));
    return config;
}

} // namespace cvcuda::priv::hq_resize

#endif // CVCUDA_PRIV_OP_HQ_RESIZE_CONFIG_HPP

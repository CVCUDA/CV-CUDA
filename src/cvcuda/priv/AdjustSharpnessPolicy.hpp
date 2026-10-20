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

#ifndef CVCUDA_PRIV_ADJUST_SHARPNESS_POLICY_HPP
#define CVCUDA_PRIV_ADJUST_SHARPNESS_POLICY_HPP

namespace cvcuda::priv {

// Per-pixel is faster for AdjustSharpness on four-channel interleaved pixels wider than a byte.
constexpr bool UseElementPathForShape(bool isPlanar, int numChannels, int baseTypeBytes)
{
    return isPlanar || !(numChannels == 4 && baseTypeBytes >= 2);
}

static_assert(!UseElementPathForShape(false, 4, 2)); // half4 NHWC  -- the +21% case
static_assert(!UseElementPathForShape(false, 4, 4)); // float4 NHWC
static_assert(UseElementPathForShape(true, 4, 4));   // planar is addressed one channel at a time
static_assert(UseElementPathForShape(false, 4, 1));  // uchar4 NHWC -- measured -38%
static_assert(UseElementPathForShape(false, 1, 2));  // uint16 NHWC -- measured -50%

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_ADJUST_SHARPNESS_POLICY_HPP

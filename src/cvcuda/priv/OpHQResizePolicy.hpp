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

#ifndef CVCUDA_PRIV_OP_HQ_RESIZE_POLICY_HPP
#define CVCUDA_PRIV_OP_HQ_RESIZE_POLICY_HPP

namespace cvcuda::priv::hq_resize {

enum class DirectTensorPath
{
    Other,
    LinearExpand2x,
    CubicContract2x,
    CubicExpand2x,
};

struct DirectTensorPathDesc
{
    bool  linear;
    bool  cubic;
    bool  xFirst;
    float originX;
    float originY;
    float scaleX;
    float scaleY;
    int   inWidth;
    int   inHeight;
    int   outWidth;
    int   outHeight;
};

constexpr DirectTensorPath ClassifyDirectTensorPath(const DirectTensorPathDesc &desc)
{
    const bool zeroOrigin = desc.originX == 0.f && desc.originY == 0.f;
    const bool expand2x   = desc.scaleX == 0.5f && desc.scaleY == 0.5f && desc.outWidth == 2 * desc.inWidth
                       && desc.outHeight == 2 * desc.inHeight;
    const bool contract2x = desc.scaleX == 2.f && desc.scaleY == 2.f && desc.inWidth == 2 * desc.outWidth
                         && desc.inHeight == 2 * desc.outHeight;

    if (desc.linear && zeroOrigin && expand2x)
    {
        return DirectTensorPath::LinearExpand2x;
    }
    if (desc.cubic && !desc.xFirst && zeroOrigin && contract2x)
    {
        return DirectTensorPath::CubicContract2x;
    }
    if (desc.cubic && desc.xFirst && zeroOrigin && expand2x)
    {
        return DirectTensorPath::CubicExpand2x;
    }
    return DirectTensorPath::Other;
}

constexpr bool UseDirectTensorPathForSM(int sm, DirectTensorPath path, bool isFloat, int numChannels)
{
    if (sm == 75)
    {
        return path != DirectTensorPath::CubicContract2x || !isFloat || (numChannels != 1 && numChannels != 3);
    }

    if (sm != 100 && sm != 103)
    {
        return true;
    }

    switch (path)
    {
    case DirectTensorPath::Other:
        return true;
    case DirectTensorPath::LinearExpand2x:
        return !(isFloat && numChannels == 3);
    case DirectTensorPath::CubicContract2x:
        return numChannels != 3;
    case DirectTensorPath::CubicExpand2x:
        return sm != 100 || !(isFloat && numChannels == 1);
    }
    return true;
}

constexpr bool ShouldUseDirectTensorPathForSM(int sm, const DirectTensorPathDesc &desc, bool isFloat, int numChannels)
{
    return UseDirectTensorPathForSM(sm, ClassifyDirectTensorPath(desc), isFloat, numChannels);
}

static_assert(!UseDirectTensorPathForSM(100, DirectTensorPath::LinearExpand2x, true, 3));
static_assert(!UseDirectTensorPathForSM(100, DirectTensorPath::CubicContract2x, true, 3));
static_assert(!UseDirectTensorPathForSM(100, DirectTensorPath::CubicContract2x, false, 3));
static_assert(!UseDirectTensorPathForSM(100, DirectTensorPath::CubicExpand2x, true, 1));
static_assert(!UseDirectTensorPathForSM(103, DirectTensorPath::LinearExpand2x, true, 3));
static_assert(!UseDirectTensorPathForSM(103, DirectTensorPath::CubicContract2x, true, 3));
static_assert(!UseDirectTensorPathForSM(103, DirectTensorPath::CubicContract2x, false, 3));
static_assert(UseDirectTensorPathForSM(103, DirectTensorPath::CubicExpand2x, true, 1));
static_assert(!UseDirectTensorPathForSM(75, DirectTensorPath::CubicContract2x, true, 1));
static_assert(!UseDirectTensorPathForSM(75, DirectTensorPath::CubicContract2x, true, 3));
static_assert(UseDirectTensorPathForSM(75, DirectTensorPath::CubicContract2x, false, 1));
static_assert(UseDirectTensorPathForSM(75, DirectTensorPath::CubicContract2x, true, 4));
static_assert(UseDirectTensorPathForSM(75, DirectTensorPath::LinearExpand2x, true, 3));
static_assert(UseDirectTensorPathForSM(80, DirectTensorPath::CubicContract2x, true, 1));

} // namespace cvcuda::priv::hq_resize

#endif // CVCUDA_PRIV_OP_HQ_RESIZE_POLICY_HPP

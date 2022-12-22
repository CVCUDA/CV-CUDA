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

#include "BorderUtils.hpp"

#include <cmath>

namespace nvcv::test {

inline void ReplicateBorderIndex(int &coord, int size)
{
    if (coord < 0)
    {
        coord = 0;
    }
    else
    {
        if (coord >= size)
        {
            coord = size - 1;
        }
    }
}

inline void WrapBorderIndex(int &coord, int size)
{
    coord = coord % size;
    if (coord < 0)
    {
        coord += size;
    }
}

inline void ReflectBorderIndex(int &coord, int size)
{
    // Reflect 1001: starting at size, we slope downards, the value at size - 1 is repeated
    coord = coord % (size * 2);
    if (coord < 0)
    {
        coord += size * 2;
    }
    if (coord >= size)
    {
        coord = size - 1 - (coord - size);
    }
}

inline void Reflect101BorderIndex(int &coord, int size)
{
    coord = coord % (2 * size - 2);
    if (coord < 0)
    {
        coord += 2 * size - 2;
    }
    coord = size - 1 - abs(size - 1 - coord);
}

void ReplicateBorderIndex(int2 &coord, int2 size)
{
    ReplicateBorderIndex(coord.x, size.x);
    ReplicateBorderIndex(coord.y, size.y);
}

void WrapBorderIndex(int2 &coord, int2 size)
{
    WrapBorderIndex(coord.x, size.x);
    WrapBorderIndex(coord.y, size.y);
}

void ReflectBorderIndex(int2 &coord, int2 size)
{
    ReflectBorderIndex(coord.x, size.x);
    ReflectBorderIndex(coord.y, size.y);
}

void Reflect101BorderIndex(int2 &coord, int2 size)
{
    Reflect101BorderIndex(coord.x, size.x);
    Reflect101BorderIndex(coord.y, size.y);
}

bool IsInside(int2 &inCoord, int2 inSize, NVCVBorderType borderMode)
{
    if (inCoord.y >= 0 && inCoord.y < inSize.y && inCoord.x >= 0 && inCoord.x < inSize.x)
    {
        return true;
    }
    else
    {
        if (borderMode == NVCV_BORDER_CONSTANT)
        {
            return false;
        }
        else
        {
            if (borderMode == NVCV_BORDER_REPLICATE)
            {
                test::ReplicateBorderIndex(inCoord, inSize);
            }
            else if (borderMode == NVCV_BORDER_WRAP)
            {
                test::WrapBorderIndex(inCoord, inSize);
            }
            else if (borderMode == NVCV_BORDER_REFLECT)
            {
                test::ReflectBorderIndex(inCoord, inSize);
            }
            else if (borderMode == NVCV_BORDER_REFLECT101)
            {
                test::Reflect101BorderIndex(inCoord, inSize);
            }

            return true;
        }
    }
}

} // namespace nvcv::test

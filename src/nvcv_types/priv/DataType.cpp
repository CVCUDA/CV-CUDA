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

#include "DataType.hpp"

#include "DataLayout.hpp"
#include "Exception.hpp"
#include "ImageFormat.hpp"

#include <util/Assert.h>
#include <util/Math.hpp>

#include <sstream>

namespace nvcv::priv {

int DataType::bpp() const noexcept
{
    return ImageFormat{m_type}.planeBPP(0);
}

int DataType::numChannels() const noexcept
{
    return ImageFormat{m_type}.planeNumChannels(0);
}

int DataType::strideBytes() const noexcept
{
    return (this->bpp() + 7) / 8;
}

std::array<int32_t, 4> DataType::bpc() const noexcept
{
    NVCVPacking packing = this->packing();
    return GetBitsPerComponent(packing);
}

DataType DataType::channelType(int ch) const
{
    if (ch < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Channel must be >= 0");
    }

    if (ch >= 4)
    {
        return DataType{NVCV_DATA_TYPE_NONE};
    }

    std::array<int32_t, 4> bits = this->bpc();
    if (bits[ch] == 0)
    {
        return DataType{NVCV_DATA_TYPE_NONE};
    }

    if (std::optional<NVCVPacking> packing = MakeNVCVPacking(bits[ch]))
    {
        return DataType{this->dataKind(), *packing};
    }
    else
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Channel type cannot be represented");
    }
}

int DataType::alignment() const noexcept
{
    return GetAlignment(packing());
}

std::ostream &operator<<(std::ostream &out, DataType type)
{
    switch (type.value())
    {
#define NVCV_ENUM(E) \
    case E:          \
        return out << #E;
        NVCV_ENUM(NVCV_DATA_TYPE_NONE);

        NVCV_ENUM(NVCV_DATA_TYPE_U8);
        NVCV_ENUM(NVCV_DATA_TYPE_2U8);
        NVCV_ENUM(NVCV_DATA_TYPE_3U8);
        NVCV_ENUM(NVCV_DATA_TYPE_4U8);

        NVCV_ENUM(NVCV_DATA_TYPE_S8);
        NVCV_ENUM(NVCV_DATA_TYPE_2S8);
        NVCV_ENUM(NVCV_DATA_TYPE_3S8);
        NVCV_ENUM(NVCV_DATA_TYPE_4S8);

        NVCV_ENUM(NVCV_DATA_TYPE_U16);
        NVCV_ENUM(NVCV_DATA_TYPE_2U16);
        NVCV_ENUM(NVCV_DATA_TYPE_3U16);
        NVCV_ENUM(NVCV_DATA_TYPE_4U16);

        NVCV_ENUM(NVCV_DATA_TYPE_S16);
        NVCV_ENUM(NVCV_DATA_TYPE_2S16);
        NVCV_ENUM(NVCV_DATA_TYPE_3S16);
        NVCV_ENUM(NVCV_DATA_TYPE_4S16);

        NVCV_ENUM(NVCV_DATA_TYPE_U32);
        NVCV_ENUM(NVCV_DATA_TYPE_2U32);
        NVCV_ENUM(NVCV_DATA_TYPE_3U32);
        NVCV_ENUM(NVCV_DATA_TYPE_4U32);

        NVCV_ENUM(NVCV_DATA_TYPE_S32);
        NVCV_ENUM(NVCV_DATA_TYPE_2S32);
        NVCV_ENUM(NVCV_DATA_TYPE_3S32);
        NVCV_ENUM(NVCV_DATA_TYPE_4S32);

        NVCV_ENUM(NVCV_DATA_TYPE_F32);
        NVCV_ENUM(NVCV_DATA_TYPE_2F32);
        NVCV_ENUM(NVCV_DATA_TYPE_3F32);
        NVCV_ENUM(NVCV_DATA_TYPE_4F32);

        NVCV_ENUM(NVCV_DATA_TYPE_U64);
        NVCV_ENUM(NVCV_DATA_TYPE_2U64);
        NVCV_ENUM(NVCV_DATA_TYPE_3U64);
        NVCV_ENUM(NVCV_DATA_TYPE_4U64);

        NVCV_ENUM(NVCV_DATA_TYPE_S64);
        NVCV_ENUM(NVCV_DATA_TYPE_2S64);
        NVCV_ENUM(NVCV_DATA_TYPE_3S64);
        NVCV_ENUM(NVCV_DATA_TYPE_4S64);

        NVCV_ENUM(NVCV_DATA_TYPE_F64);
        NVCV_ENUM(NVCV_DATA_TYPE_2F64);
        NVCV_ENUM(NVCV_DATA_TYPE_3F64);
        NVCV_ENUM(NVCV_DATA_TYPE_4F64);
#undef NVCV_ENUM
    }

    return out << "NVCVDataType(" << type.dataKind() << "," << type.packing() << ")";
}

} // namespace nvcv::priv

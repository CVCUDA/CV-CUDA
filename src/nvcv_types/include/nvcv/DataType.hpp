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
 * @file DataType.hpp
 *
 * @brief Defines C++ types and functions to handle data types.
 */

#ifndef NVCV_DATA_TYPE_HPP
#define NVCV_DATA_TYPE_HPP

#include "DataLayout.hpp"
#include "DataType.h"

#include <iostream>

namespace nvcv {

/**
 * Byte type, similar to C++17's std::byte
 */
enum class Byte : uint8_t
{
};

/**
 * @brief Defines types and functions to handle data types.
 *
 * @defgroup NVCV_CPP_CORE_DATATYPE Data types
 * @{
 */

class DataType
{
public:
    constexpr DataType();
    explicit constexpr DataType(NVCVDataType type);
    DataType(DataKind dataKind, Packing packing);

    static constexpr DataType ConstCreate(DataKind dataKind, Packing packing);

    constexpr operator NVCVDataType() const;

    Packing                packing() const;
    std::array<int32_t, 4> bitsPerChannel() const;
    DataKind               dataKind() const;
    int32_t                numChannels() const;
    DataType               channelType(int32_t channel) const;
    int32_t                strideBytes() const;
    int32_t                bitsPerPixel() const;
    int32_t                alignment() const;

private:
    NVCVDataType m_type;
};

constexpr DataType::DataType()
    : m_type(NVCV_DATA_TYPE_NONE)
{
}

constexpr DataType::DataType(NVCVDataType type)
    : m_type(type)
{
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** One channel of unsigned 8-bit value. */
constexpr DataType TYPE_U8{NVCV_DATA_TYPE_U8};
/** Two interleaved channels of unsigned 8-bit values. */
constexpr DataType TYPE_2U8{NVCV_DATA_TYPE_2U8};
/** Three interleaved channels of unsigned 8-bit values. */
constexpr DataType TYPE_3U8{NVCV_DATA_TYPE_3U8};
/** Four interleaved channels of unsigned 8-bit values. */
constexpr DataType TYPE_4U8{NVCV_DATA_TYPE_4U8};

/** One channel of signed 8-bit value. */
constexpr DataType TYPE_S8{NVCV_DATA_TYPE_S8};
/** Two interleaved channels of signed 8-bit values. */
constexpr DataType TYPE_2S8{NVCV_DATA_TYPE_2S8};
/** Three interleaved channels of signed 8-bit values. */
constexpr DataType TYPE_3S8{NVCV_DATA_TYPE_3S8};
/** Four interleaved channels of signed 8-bit values. */
constexpr DataType TYPE_4S8{NVCV_DATA_TYPE_4S8};

/** One channel of unsigned 16-bit value. */
constexpr DataType TYPE_U16{NVCV_DATA_TYPE_U16};
/** Two interleaved channels of unsigned 16-bit values. */
constexpr DataType TYPE_2U16{NVCV_DATA_TYPE_2U16};
/** Three interleaved channels of unsigned 16-bit values. */
constexpr DataType TYPE_3U16{NVCV_DATA_TYPE_3U16};
/** Four interleaved channels of unsigned 16-bit values. */
constexpr DataType TYPE_4U16{NVCV_DATA_TYPE_4U16};

/** One channel of signed 16-bit value. */
constexpr DataType TYPE_S16{NVCV_DATA_TYPE_S16};
/** Two interleaved channels of signed 16-bit values. */
constexpr DataType TYPE_2S16{NVCV_DATA_TYPE_2S16};
/** Three interleaved channels of signed 16-bit values. */
constexpr DataType TYPE_3S16{NVCV_DATA_TYPE_3S16};
/** Four interleaved channels of signed 16-bit values. */
constexpr DataType TYPE_4S16{NVCV_DATA_TYPE_4S16};

/** One channel of unsigned 32-bit value. */
constexpr DataType TYPE_U32{NVCV_DATA_TYPE_U32};
/** Two interleaved channels of unsigned 32-bit values. */
constexpr DataType TYPE_2U32{NVCV_DATA_TYPE_2U32};
/** Three interleaved channels of unsigned 32-bit values. */
constexpr DataType TYPE_3U32{NVCV_DATA_TYPE_3U32};
/** Four interleaved channels of unsigned 32-bit values. */
constexpr DataType TYPE_4U32{NVCV_DATA_TYPE_4U32};

/** One channel of signed 32-bit value. */
constexpr DataType TYPE_S32{NVCV_DATA_TYPE_S32};
/** Two interleaved channels of signed 32-bit values. */
constexpr DataType TYPE_2S32{NVCV_DATA_TYPE_2S32};
/** Three interleaved channels of signed 32-bit values. */
constexpr DataType TYPE_3S32{NVCV_DATA_TYPE_3S32};
/** Four interleaved channels of signed 32-bit values. */
constexpr DataType TYPE_4S32{NVCV_DATA_TYPE_4S32};

/** One channel of 32-bit IEEE 754 floating-point value. */
constexpr DataType TYPE_F32{NVCV_DATA_TYPE_F32};
/** Two interleaved channels of 32-bit IEEE 754 floating-point values. */
constexpr DataType TYPE_2F32{NVCV_DATA_TYPE_2F32};
/** Three interleaved channels of 32-bit IEEE 754 floating-point values. */
constexpr DataType TYPE_3F32{NVCV_DATA_TYPE_3F32};
/** Four interleaved channels of 32-bit IEEE 754 floating-point values. */
constexpr DataType TYPE_4F32{NVCV_DATA_TYPE_4F32};

/** One channel of unsigned 64-bit value. */
constexpr DataType TYPE_U64{NVCV_DATA_TYPE_U64};
/** Two interleaved channels of unsigned 64-bit values. */
constexpr DataType TYPE_2U64{NVCV_DATA_TYPE_2U64};
/** Three interleaved channels of unsigned 64-bit values. */
constexpr DataType TYPE_3U64{NVCV_DATA_TYPE_3U64};
/** Four interleaved channels of unsigned 64-bit values. */
constexpr DataType TYPE_4U64{NVCV_DATA_TYPE_4U64};

/** One channel of signed 64-bit value. */
constexpr DataType TYPE_S64{NVCV_DATA_TYPE_S64};
/** Two interleaved channels of signed 64-bit values. */
constexpr DataType TYPE_2S64{NVCV_DATA_TYPE_2S64};
/** Three interleaved channels of signed 64-bit values. */
constexpr DataType TYPE_3S64{NVCV_DATA_TYPE_3S64};
/** Four interleaved channels of signed 64-bit values. */
constexpr DataType TYPE_4S64{NVCV_DATA_TYPE_4S64};

/** One channel of 64-bit IEEE 754 floating-point value. */
constexpr DataType TYPE_F64{NVCV_DATA_TYPE_F64};
/** Two interleaved channels of 64-bit IEEE 754 floating-point values. */
constexpr DataType TYPE_2F64{NVCV_DATA_TYPE_2F64};
/** Three interleaved channels of 64-bit IEEE 754 floating-point values. */
constexpr DataType TYPE_3F64{NVCV_DATA_TYPE_3F64};
/** Four interleaved channels of 64-bit IEEE 754 floating-point values. */
constexpr DataType TYPE_4F64{NVCV_DATA_TYPE_4F64};
#endif

inline DataType::DataType(DataKind dataKind, Packing packing)
{
    detail::CheckThrow(
        nvcvMakeDataType(&m_type, static_cast<NVCVDataKind>(dataKind), static_cast<NVCVPacking>(packing)));
}

constexpr DataType DataType::ConstCreate(DataKind dataKind, Packing packing)
{
    return DataType{NVCV_MAKE_DATA_TYPE(static_cast<NVCVDataKind>(dataKind), static_cast<NVCVPacking>(packing))};
}

constexpr DataType::operator NVCVDataType() const
{
    return m_type;
}

inline Packing DataType::packing() const
{
    NVCVPacking out;
    detail::CheckThrow(nvcvDataTypeGetPacking(m_type, &out));
    return static_cast<Packing>(out);
}

inline int32_t DataType::bitsPerPixel() const
{
    int32_t out;
    detail::CheckThrow(nvcvDataTypeGetBitsPerPixel(m_type, &out));
    return out;
}

inline std::array<int32_t, 4> DataType::bitsPerChannel() const
{
    int32_t bits[4];
    detail::CheckThrow(nvcvDataTypeGetBitsPerChannel(m_type, bits));
    return {bits[0], bits[1], bits[2], bits[3]};
}

inline DataKind DataType::dataKind() const
{
    NVCVDataKind out;
    detail::CheckThrow(nvcvDataTypeGetDataKind(m_type, &out));
    return static_cast<DataKind>(out);
}

inline int32_t DataType::numChannels() const
{
    int32_t out;
    detail::CheckThrow(nvcvDataTypeGetNumChannels(m_type, &out));
    return out;
}

inline DataType DataType::channelType(int32_t channel) const
{
    NVCVDataType out;
    detail::CheckThrow(nvcvDataTypeGetChannelType(m_type, channel, &out));
    return static_cast<DataType>(out);
}

inline int32_t DataType::strideBytes() const
{
    int32_t out;
    detail::CheckThrow(nvcvDataTypeGetStrideBytes(m_type, &out));
    return out;
}

inline int32_t DataType::alignment() const
{
    int32_t out;
    detail::CheckThrow(nvcvDataTypeGetAlignment(m_type, &out));
    return out;
}

inline std::ostream &operator<<(std::ostream &out, DataType type)
{
    return out << nvcvDataTypeGetName(type);
}

/**@}*/

} // namespace nvcv

#endif // NVCV_DATA_TYPE_HPP

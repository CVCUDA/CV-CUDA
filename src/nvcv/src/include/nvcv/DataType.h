/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file DataType.h
 *
 * @brief Defines types and functions to handle data types.
 */

#ifndef NVCV_DATA_TYPE_H
#define NVCV_DATA_TYPE_H

#include "ColorSpec.h"
#include "DataLayout.h"
#include "detail/FormatUtils.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** Definition of a byte type. */
typedef struct
{
    uint8_t value;
} NVCVByte;

/**
 * Pre-defined data types.
 * Data types defines the geometry of value elements, i.e., pixels in a image plane without taking into account what the value represents.
 * For example, a \ref NVCV_IMAGE_FORMAT_NV12 is composed of 2 planes, each one with the following data types:
 * + \ref NVCV_DATA_TYPE_U8 representing pixels as 8-bit unsigned values.
 * + \ref NVCV_DATA_TYPE_2U8 representing pixels as two interleaved 8-bit unsigned values.
 *
 * @defgroup NVCV_C_CORE_DATATYPE Data types
 * @{
 */
typedef uint64_t NVCVDataType;

/* clang-format off */

/** Denotes a special image format that doesn't represent any particular type (like void). */
#define NVCV_DATA_TYPE_NONE ((NVCVDataType)0)

/** One channel of unsigned 8-bit value. */
#define NVCV_DATA_TYPE_U8   NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X8)
/** Two interleaved channels of unsigned 8-bit values. */
#define NVCV_DATA_TYPE_2U8  NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X8_Y8)
/** Three interleaved channels of unsigned 8-bit values. */
#define NVCV_DATA_TYPE_3U8  NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X8_Y8_Z8)
/** Four interleaved channels of unsigned 8-bit values. */
#define NVCV_DATA_TYPE_4U8  NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X8_Y8_Z8_W8)

/** One channel of signed 8-bit value. */
#define NVCV_DATA_TYPE_S8   NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X8)
/** Two interleaved channels of signed 8-bit values. */
#define NVCV_DATA_TYPE_2S8  NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X8_Y8)
/** Three interleaved channels of signed 8-bit values. */
#define NVCV_DATA_TYPE_3S8  NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X8_Y8_Z8)
/** Four interleaved channels of signed 8-bit values. */
#define NVCV_DATA_TYPE_4S8  NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X8_Y8_Z8_W8)

/** One channel of unsigned 16-bit value. */
#define NVCV_DATA_TYPE_U16  NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X16)
/** Two interleaved channels of unsigned 16-bit values. */
#define NVCV_DATA_TYPE_2U16 NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X16_Y16)
/** Three interleaved channels of unsigned 16-bit values. */
#define NVCV_DATA_TYPE_3U16 NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X16_Y16_Z16)
/** Four interleaved channels of unsigned 16-bit values. */
#define NVCV_DATA_TYPE_4U16 NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X16_Y16_Z16_W16)

/** One channel of signed 16-bit value. */
#define NVCV_DATA_TYPE_S16  NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X16)
/** Two interleaved channels of signed 16-bit values. */
#define NVCV_DATA_TYPE_2S16 NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X16_Y16)
/** Three interleaved channels of signed 16-bit values. */
#define NVCV_DATA_TYPE_3S16 NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X16_Y16_Z16)
/** Four interleaved channels of signed 16-bit values. */
#define NVCV_DATA_TYPE_4S16 NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X16_Y16_Z16_W16)

/** One channel of unsigned 32-bit value. */
#define NVCV_DATA_TYPE_U32  NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X32)
/** Two interleaved channels of unsigned 32-bit values. */
#define NVCV_DATA_TYPE_2U32 NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X32_Y32)
/** Three interleaved channels of unsigned 32-bit values. */
#define NVCV_DATA_TYPE_3U32 NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X32_Y32_Z32)
/** Four interleaved channels of unsigned 32-bit values. */
#define NVCV_DATA_TYPE_4U32 NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X32_Y32_Z32_W32)

/** One channel of signed 32-bit value. */
#define NVCV_DATA_TYPE_S32  NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X32)
/** Two interleaved channels of signed 32-bit values. */
#define NVCV_DATA_TYPE_2S32 NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X32_Y32)
/** Three interleaved channels of signed 32-bit values. */
#define NVCV_DATA_TYPE_3S32 NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X32_Y32_Z32)
/** Four interleaved channels of signed 32-bit values. */
#define NVCV_DATA_TYPE_4S32 NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X32_Y32_Z32_W32)

/** One channel of 16-bit IEEE 754 floating-point value. */
#define NVCV_DATA_TYPE_F16  NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X16)
/** Two interleaved channels of 16-bit IEEE 754 floating-point values. */
#define NVCV_DATA_TYPE_2F16 NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X16_Y16)
/** Three interleaved channels of 16-bit IEEE 754 floating-point values. */
#define NVCV_DATA_TYPE_3F16 NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X16_Y16_Z16)
/** Four interleaved channels of 16-bit IEEE 754 floating-point values. */
#define NVCV_DATA_TYPE_4F16 NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X16_Y16_Z16_W16)

/** One channel of 32-bit IEEE 754 floating-point value. */
#define NVCV_DATA_TYPE_F32  NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X32)
/** Two interleaved channels of 32-bit IEEE 754 floating-point values. */
#define NVCV_DATA_TYPE_2F32 NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X32_Y32)
/** Three interleaved channels of 32-bit IEEE 754 floating-point values. */
#define NVCV_DATA_TYPE_3F32 NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X32_Y32_Z32)
/** Four interleaved channels of 32-bit IEEE 754 floating-point values. */
#define NVCV_DATA_TYPE_4F32 NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X32_Y32_Z32_W32)

/** One channel of unsigned 64-bit value. */
#define NVCV_DATA_TYPE_U64  NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X64)
/** Two interleaved channels of unsigned 64-bit values. */
#define NVCV_DATA_TYPE_2U64 NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X64_Y64)
/** Three interleaved channels of unsigned 64-bit values. */
#define NVCV_DATA_TYPE_3U64 NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X64_Y64_Z64)
/** Four interleaved channels of unsigned 64-bit values. */
#define NVCV_DATA_TYPE_4U64 NVCV_DETAIL_MAKE_PIX_TYPE(UNSIGNED, X64_Y64_Z64_W64)

/** One channel of signed 64-bit value. */
#define NVCV_DATA_TYPE_S64  NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X64)
/** Two interleaved channels of signed 64-bit values. */
#define NVCV_DATA_TYPE_2S64 NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X64_Y64)
/** Three interleaved channels of signed 64-bit values. */
#define NVCV_DATA_TYPE_3S64 NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X64_Y64_Z64)
/** Four interleaved channels of signed 64-bit values. */
#define NVCV_DATA_TYPE_4S64 NVCV_DETAIL_MAKE_PIX_TYPE(SIGNED, X64_Y64_Z64_W64)

/** One channel of 64-bit IEEE 754 floating-point value. */
#define NVCV_DATA_TYPE_F64  NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X64)
/** Two interleaved channels of 64-bit IEEE 754 floating-point values. */
#define NVCV_DATA_TYPE_2F64 NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X64_Y64)
/** Three interleaved channels of 64-bit IEEE 754 floating-point values. */
#define NVCV_DATA_TYPE_3F64 NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X64_Y64_Z64)
/** Four interleaved channels of 64-bit IEEE 754 floating-point values. */
#define NVCV_DATA_TYPE_4F64 NVCV_DETAIL_MAKE_PIX_TYPE(FLOAT, X64_Y64_Z64_W64)

/** One channel of 64-bit IEEE 754 complex floating-point value (2 32-bit floats). */
#define NVCV_DATA_TYPE_C64  NVCV_DETAIL_MAKE_PIX_TYPE(COMPLEX, X64)
/** Two interleaved channels of 64-bit IEEE 754 complex floating-point values (4 32-bit floats). */
#define NVCV_DATA_TYPE_2C64 NVCV_DETAIL_MAKE_PIX_TYPE(COMPLEX, X64_Y64)
/** Three interleaved channels of 64-bit IEEE 754 complex floating-point values (6 32-bit floats). */
#define NVCV_DATA_TYPE_3C64 NVCV_DETAIL_MAKE_PIX_TYPE(COMPLEX, X64_Y64_Z64)
/** Four interleaved channels of 64-bit IEEE 754 complex floating-point values (8 32-bit floats). */
#define NVCV_DATA_TYPE_4C64 NVCV_DETAIL_MAKE_PIX_TYPE(COMPLEX, X64_Y64_Z64_W64)

/** One channel of 128-bit IEEE 754 complex floating-point value (2 64-bit floats). */
#define NVCV_DATA_TYPE_C128  NVCV_DETAIL_MAKE_PIX_TYPE(COMPLEX, X128)
/** Two interleaved channels of 128-bit IEEE 754 complex floating-point values (4 64-bit floats). */
#define NVCV_DATA_TYPE_2C128 NVCV_DETAIL_MAKE_PIX_TYPE(COMPLEX, X128_Y128)

/* clang-format on */

/** Creates a user-defined data type constant.
 *
 * Example to create a block-linear format two interleaved 32-bit floating point channels:
 * \code{.c}
 *     NVCVDataType type = NVCV_MAKE_DATA_TYPE(NVCV_DATA_KIND_FLOAT, NVCV_PACKING_X32_Y32);
 * \endcode
 *
 * @param[in] dataKind  \ref NVCVDataKind to be used.
 * @param[in] packing   Format packing used, which also defines the number of channels.
 *
 * @returns The user-defined data type.
 */
#ifdef DOXYGEN_SHOULD_SKIP_THIS
#    define NVCV_MAKE_DATA_TYPE(dataKind, packing)
#else
#    define NVCV_MAKE_DATA_TYPE (NVCVDataType) NVCV_DETAIL_MAKE_DATA_TYPE
#endif

/** Creates a user-defined data type.
 * When the pre-defined data types aren't enough, user-defined formats can be created.
 *
 * @param[out] outPixType The user-defined data type.
 *                        + Cannot be NULL.
 *
 * @param[in] dataKind \ref NVCVDataKind to be used.
 * @param[in] packing Format packing used, which also defines the number of channels.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakeDataType(NVCVDataType *outPixType, NVCVDataKind dataKind, NVCVPacking packing);

/** Get the packing of a data type.
 *
 * @param[in] type Data type to be queried.
 *
 * @param[out] outPacking The format's packing.
 *                 + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvDataTypeGetPacking(NVCVDataType type, NVCVPacking *outPacking);

/** Get the number of bits per pixel of a data type.
 *
 * @param[in] type Data type to be queried.
 *
 * @param[out] outBPP The number of bits per pixel.
 *                    + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvDataTypeGetBitsPerPixel(NVCVDataType type, int32_t *outBPP);

/** Get the number of bits per channel of a data type.
 *
 * @param[in] type Data type to be queried.
 *
 * @param[out] outBits Pointer to an int32_t array with 4 elements where output will be stored.
 *                     + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvDataTypeGetBitsPerChannel(NVCVDataType type, int32_t *outBits);

/** Get the required address alignment for each data element.
 *
 * The returned alignment is guaranteed to be a power-of-two.
 *
 * @param[in] type Data type to be queried.
 *
 * @param[out] outAlignment Pointer to an int32_t where the required alignment is to be stored.
 *                          + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvDataTypeGetAlignment(NVCVDataType type, int32_t *outAlignment);

/** Get the data type of a data type.
 *
 * @param[in] type Data type to be queried.
 *
 * @param[out] outDataKind The data type of the data type.
 *                      + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvDataTypeGetDataKind(NVCVDataType type, NVCVDataKind *outDataKind);

/** Get the number of channels of a data type.
 *
 * @param[in] type Data type to be queried.
 *
 * @param[out] outNumChannels The number of channels of the data type.
 *                            + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvDataTypeGetNumChannels(NVCVDataType type, int32_t *outNumChannels);

/** Returns a string representation of the data type.
 *
 * @param[in] type Data type to be returned.
 *
 * @returns The string representation of the data type.
 *          Returned string is valid until next call of this function from the same calling thread.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvDataTypeGetName(NVCVDataType type);

/** Get the data type for a given channel index.
 *
 * It returns a single-channel data type that corresponds to the given channel
 * of the input data type.
 *
 * For instance: The channel #2 of \ref NVCV_DATA_TYPE_3U8 is \ref NVCV_DATA_TYPE_U8.
 *
 * + The requested channel must have a type whose packing is one of the following:
 *   - \ref NVCV_PACKING_X1
 *   - \ref NVCV_PACKING_X2
 *   - \ref NVCV_PACKING_X4
 *   - \ref NVCV_PACKING_X8
 *   - \ref NVCV_PACKING_X16
 *   - \ref NVCV_PACKING_X24
 *   - \ref NVCV_PACKING_X32
 *   - \ref NVCV_PACKING_X48
 *   - \ref NVCV_PACKING_X64
 *   - \ref NVCV_PACKING_X96
 *   - \ref NVCV_PACKING_X128
 *   - \ref NVCV_PACKING_X192
 *   - \ref NVCV_PACKING_X256
 *
 * @param[in] type Data type to be queried.
 *
 * @param[in] channel Channel whose data type is to be returned.
 *                 + Must be between 0 and the maximum number of channels in \p type.
 *
 * @param[out] outChannelType The data type of the given channel. The memory layout and data type are the same as \p type.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvDataTypeGetChannelType(NVCVDataType type, int32_t channel, NVCVDataType *outChannelTpe);

/** Returns the stride/size in bytes of the pixel in memory.
 *
 * @param[in] type Data type to be queried.
 *
 * @param[out] pixStrideBytes The size in bytes of the pixel
 *                            + Must not be NULL.
 */
NVCV_PUBLIC NVCVStatus nvcvDataTypeGetStrideBytes(NVCVDataType type, int32_t *pixStrideBytes);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* NVCV_DATA_TYPE_H */

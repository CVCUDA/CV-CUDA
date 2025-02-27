/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_DETAIL_FORMATUTILS_H
#define NVCV_DETAIL_FORMATUTILS_H

// Internal implementation of pixel formats/types.
// Not to be used directly.

#include <stdint.h>
#include <stdio.h>

// Utilities ================================

#define NVCV_DETAIL_SET_BITFIELD(value, offset, size) (((uint64_t)(value) & ((1ULL << (size)) - 1)) << (offset))
#define NVCV_DETAIL_GET_BITFIELD(value, offset, size) (((uint64_t)(value) >> (offset)) & ((1ULL << (size)) - 1))

// MAKE_COLOR_SPEC =======================================

#define NVCV_DETAIL_MAKE_COLOR_SPEC(CSpace, Encoding, XferFunc, Range, LocHoriz, LocVert)   \
    (NVCV_DETAIL_SET_BITFIELD((CSpace), 0, 3) | NVCV_DETAIL_SET_BITFIELD(XferFunc, 3, 4)    \
     | NVCV_DETAIL_SET_BITFIELD(Encoding, 7, 3) | NVCV_DETAIL_SET_BITFIELD(LocHoriz, 10, 2) \
     | NVCV_DETAIL_SET_BITFIELD(LocVert, 12, 2) | NVCV_DETAIL_SET_BITFIELD(Range, 14, 1))

#define NVCV_DETAIL_MAKE_CSPC(CSpace, Encoding, XferFunc, Range, LocHoriz, LocVert)                                    \
    NVCV_DETAIL_MAKE_COLOR_SPEC(NVCV_COLOR_##CSpace, NVCV_YCbCr_##Encoding, NVCV_COLOR_##XferFunc, NVCV_COLOR_##Range, \
                                NVCV_CHROMA_##LocHoriz, NVCV_CHROMA_##LocVert)

// DATA TYPE utils =======================================

#define NVCV_DETAIL_ENCODE_BPP(bpp)            \
    ((bpp) <= 8 ? 0                            \
                : ((bpp) <= 32 ? (bpp) / 8 - 1 \
                               : ((bpp) <= 64 ? (bpp) / 16 + 1 : ((bpp) <= 128 ? (bpp) / 32 + 3 : (bpp) / 64 + 5))))

#define NVCV_DETAIL_BPP_NCH(bpp, chcount)                                                                      \
    (NVCV_DETAIL_SET_BITFIELD(NVCV_DETAIL_ENCODE_BPP(bpp), 6, 4) | NVCV_DETAIL_SET_BITFIELD((chcount)-1, 4, 2) \
     | NVCV_DETAIL_SET_BITFIELD((bpp) <= 2 ? (bpp) : ((bpp) == 4 ? 3 : ((bpp) == 8 ? 4 : 0)), 0, 4))

#define NVCV_DETAIL_MAKE_SWIZZLE(x, y, z, w)                                                                   \
    (NVCV_DETAIL_SET_BITFIELD(x, 0, 3) | NVCV_DETAIL_SET_BITFIELD(y, 3, 3) | NVCV_DETAIL_SET_BITFIELD(z, 6, 3) \
     | NVCV_DETAIL_SET_BITFIELD(w, 9, 3))

#define NVCV_DETAIL_MAKE_SWZL(x, y, z, w) \
    NVCV_DETAIL_MAKE_SWIZZLE(NVCV_CHANNEL_##x, NVCV_CHANNEL_##y, NVCV_CHANNEL_##z, NVCV_CHANNEL_##w)

// Image format / data type utils

#define NVCV_DETAIL_ADJUST_BPP_ENCODING(PACK, BPP, PACKLEN) \
    ((PACKLEN) == 0 && (BPP) == 0 && (PACK) == 4 ? (uint64_t)-1 : (BPP))

#define NVCV_DETAIL_ENCODE_PACKING(P, CHLEN, PACKLEN, BPPLEN)                                              \
    (NVCV_DETAIL_SET_BITFIELD(NVCV_DETAIL_ADJUST_BPP_ENCODING(NVCV_DETAIL_GET_BITFIELD(P, 0, 4),           \
                                                              NVCV_DETAIL_GET_BITFIELD(P, 6, 4), PACKLEN), \
                              (PACKLEN) + (CHLEN), BPPLEN)                                                 \
     | NVCV_DETAIL_SET_BITFIELD(NVCV_DETAIL_GET_BITFIELD(P, 4, 2), PACKLEN, CHLEN)                         \
     | NVCV_DETAIL_SET_BITFIELD(NVCV_DETAIL_GET_BITFIELD(P, 0, 4), 0, PACKLEN))

#define NVCV_DETAIL_EXTRACT_PACKING_CHANNELS(Packing) (NVCV_DETAIL_GET_BITFIELD(Packing, 4, 2) + 1)

#define NVCV_DETAIL_SET_EXTRA_CHANNEL_INFO(NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType) \
    (NVCV_DETAIL_SET_BITFIELD(ExtraChannelType, 44, 3) | NVCV_DETAIL_SET_BITFIELD(NumExtraChannel, 47, 3)            \
     | NVCV_DETAIL_SET_BITFIELD(NVCV_DETAIL_ENCODE_BPP(ExtraChannelBPP), 50, 3)                                      \
     | NVCV_DETAIL_SET_BITFIELD(ExtraChannelDataKind, 53, 3))

#define PRINT_MESSAGE(NumExtraChannel)                \
    do                                                \
    {                                                 \
        printf("Num Channels %d\n", NumExtraChannel); \
    }                                                 \
    while (0)

/* clang-format off */
#define NVCV_DETAIL_MAKE_FMTTYPE(ColorModel, ColorSpecOrRawPattern, Subsampling, MemLayout, DataKind, Swizzle, \
                               AlphaType, Packing0, Packing1, Packing2, Packing3, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)                                        \
    (                                                              \
        NVCV_DETAIL_SET_BITFIELD(DataKind, 61, 3) | NVCV_DETAIL_SET_BITFIELD(Swizzle, 0, 6) |           \
        NVCV_DETAIL_SET_BITFIELD(AlphaType, 6, 1) | NVCV_DETAIL_SET_BITFIELD(MemLayout, 12, 3) | \
        ((ColorModel) == NVCV_COLOR_MODEL_YCbCr \
            ? NVCV_DETAIL_SET_BITFIELD(ColorSpecOrRawPattern, 20, 15) | NVCV_DETAIL_SET_BITFIELD(Subsampling, 17, 3) \
            : ((ColorModel) == NVCV_COLOR_MODEL_UNDEFINED \
                ? NVCV_DETAIL_SET_BITFIELD((1U<<19)-1, 16, 19) \
                : (NVCV_DETAIL_SET_BITFIELD(1,16,1) | \
                     ((ColorModel)-2 < 0x7 \
                       ? NVCV_DETAIL_SET_BITFIELD(ColorSpecOrRawPattern, 20, 15) \
                            | NVCV_DETAIL_SET_BITFIELD((ColorModel)-2, 17, 3) \
                       : (NVCV_DETAIL_SET_BITFIELD(0x7, 17, 3) | \
                            ((ColorModel) == NVCV_COLOR_MODEL_RAW \
                              ? NVCV_DETAIL_SET_BITFIELD(ColorSpecOrRawPattern, 21, 6) \
                              : (NVCV_DETAIL_SET_BITFIELD(1, 20, 1) | NVCV_DETAIL_SET_BITFIELD((ColorModel)-(7+2+1), 21, 6)) \
                            ) \
                         ) \
                     ) \
                  ) \
              ) \
        ) | \
        NVCV_DETAIL_SET_BITFIELD(NVCV_DETAIL_ENCODE_PACKING(Packing0, 2, 3, 4), 35, 9) | \
        ((Packing1) == NVCV_PACKING_0 && (NumExtraChannel) > 0 \
        ? NVCV_DETAIL_SET_BITFIELD(ExtraChannelType, 44, 3) | NVCV_DETAIL_SET_BITFIELD(NumExtraChannel, 47, 3) | NVCV_DETAIL_SET_BITFIELD(NVCV_DETAIL_ENCODE_BPP(ExtraChannelBPP), 50, 3) | NVCV_DETAIL_SET_BITFIELD(ExtraChannelDataKind, 53, 3) \
        : NVCV_DETAIL_SET_BITFIELD(1, 7, 1) | NVCV_DETAIL_SET_BITFIELD(NVCV_DETAIL_ENCODE_PACKING(Packing1, 1, 3, 3), 44, 7) | NVCV_DETAIL_SET_BITFIELD(NVCV_DETAIL_ENCODE_PACKING(Packing2, 1, 3, 3), 51, 7) | NVCV_DETAIL_SET_BITFIELD(NVCV_DETAIL_ENCODE_PACKING(Packing3, 0, 0, 3), 58, 3) \
        ) \
    )

/* clang-format on */

#define NVCV_DETAIL_MAKE_FORMAT(ColorModel, ColorSpecOrRawPattern, Subsampling, MemLayout, DataKind, Swizzle,        \
                                AlphaType, Packing0, Packing1, Packing2, Packing3, NumExtraChannel, ExtraChannelBPP, \
                                ExtraChannelDataKind, ExtraChannelType)                                              \
    ((NVCVImageFormat)NVCV_DETAIL_MAKE_FMTTYPE(                                                                      \
        ColorModel, ColorSpecOrRawPattern, Subsampling, MemLayout, DataKind, Swizzle, AlphaType, Packing0, Packing1, \
        Packing2, Packing3, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType))

#define NVCV_DETAIL_MAKE_FMT(ColorModel, ColorSpec, CSS, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3, \
                             NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)            \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_##ColorModel, NVCV_COLOR_SPEC_##ColorSpec, NVCV_CSS_##CSS,           \
                            NVCV_MEM_LAYOUT_##MemLayout, NVCV_DATA_KIND_##DataKind, NVCV_SWIZZLE_##Swizzle,       \
                            NVCV_ALPHA_##AlphaType, NVCV_PACKING_##P0, NVCV_PACKING_##P1, NVCV_PACKING_##P2,      \
                            NVCV_PACKING_##P3, NumExtraChannel, ExtraChannelBPP,                                  \
                            NVCV_DATA_KIND_##ExtraChannelDataKind, NVCV_EXTRA_CHANNEL_##ExtraChannelType)

// MAKE_COLOR ================================================

// Full arg name

// No extra channel

#define NVCV_DETAIL_MAKE_COLOR_FORMAT1(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, P0)           \
    NVCV_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, 0, 0, \
                            0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_COLOR_FORMAT2(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, P0, P1)        \
    NVCV_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, 0, \
                            0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_COLOR_FORMAT3(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2)     \
    NVCV_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, \
                            0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_COLOR_FORMAT4(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3) \
    NVCV_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, \
                            P3, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_COLOR_FORMAT(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, NumPlanes, ...) \
    NVCV_DETAIL_MAKE_COLOR_FORMAT##NumPlanes(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType,          \
                                             __VA_ARGS__)

// Extra Channel

#define NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FORMAT1(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, \
                                                      NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,         \
                                                      ExtraChannelType, P0)                                           \
    NVCV_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, 0, 0,  \
                            0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FORMAT2(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, \
                                                      NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,         \
                                                      ExtraChannelType, P0, P1)                                       \
    NVCV_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, 0, \
                            0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FORMAT3(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType,  \
                                                      NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,          \
                                                      ExtraChannelType, P0, P1, P2)                                    \
    NVCV_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, \
                            0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FORMAT4(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType,  \
                                                      NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,          \
                                                      ExtraChannelType, P0, P1, P2, P3)                                \
    NVCV_DETAIL_MAKE_FORMAT(ColorModel, ColorSpec, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, \
                            P3, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FORMAT(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, \
                                                     NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,         \
                                                     ExtraChannelType, NumPlanes, ...)                               \
    NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FORMAT##NumPlanes(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle,     \
                                                            AlphaType, NumExtraChannel, ExtraChannelBPP,             \
                                                            ExtraChannelDataKind, ExtraChannelType, __VA_ARGS__)

// Abbreviated

// No extra channel

#define NVCV_DETAIL_MAKE_COLOR_FMT1(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, P0)           \
    NVCV_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, 0, 0, 0, 0, 0, \
                         UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_COLOR_FMT2(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, P0, P1)        \
    NVCV_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, 0, 0, 0, 0, \
                         UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_COLOR_FMT3(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P3)     \
    NVCV_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P3, 0, 0, 0, \
                         UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_COLOR_FMT4(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P3, P4)  \
    NVCV_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P3, P4, 0, 0, \
                         UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_COLOR_FMT(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, NumPlanes, ...) \
    NVCV_DETAIL_MAKE_COLOR_FMT##NumPlanes(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, __VA_ARGS__)

// Extra channel

#define NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT1(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, \
                                                   NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,         \
                                                   ExtraChannelType, P0)                                           \
    NVCV_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, 0, 0, 0,        \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT2(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, \
                                                   NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,         \
                                                   ExtraChannelType, P0, P1)                                       \
    NVCV_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, 0, 0,       \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT3(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, \
                                                   NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,         \
                                                   ExtraChannelType, P0, P1, P3)                                   \
    NVCV_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P3, 0,      \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT4(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, \
                                                   NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,         \
                                                   ExtraChannelType, P0, P1, P3, P4)                               \
    NVCV_DETAIL_MAKE_FMT(ColorModel, ColorSpec, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P3, P4,     \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle, AlphaType, \
                                                  NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,         \
                                                  ExtraChannelType, NumPlanes, ...)                               \
    NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT##NumPlanes(ColorModel, ColorSpec, MemLayout, DataKind, Swizzle,     \
                                                         AlphaType, NumExtraChannel, ExtraChannelBPP,             \
                                                         ExtraChannelDataKind, ExtraChannelType, __VA_ARGS__)

// MAKE_DATA_TYPE ========================================

// Full arg name

#define NVCV_DETAIL_MAKE_DATA_TYPE(DataKind, Packing)                                                             \
    ((NVCVDataType)NVCV_DETAIL_MAKE_FMTTYPE(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_NONE, \
                                            NVCV_MEM_LAYOUT_PL, DataKind,                                         \
                                            NVCV_DETAIL_EXTRACT_PACKING_CHANNELS(Packing), NVCV_ALPHA_ASSOCIATED, \
                                            Packing, 0, 0, 0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U))

// Abbreviated

#define NVCV_DETAIL_MAKE_PIX_TYPE(DataKind, Packing) \
    NVCV_DETAIL_MAKE_DATA_TYPE(NVCV_DATA_KIND_##DataKind, NVCV_PACKING_##Packing)

// MAKE_NONCOLOR ==================================

// Full arg name

// No extra Channel

#define NVCV_DETAIL_MAKE_NONCOLOR_FORMAT1(MemLayout, DataKind, Swizzle, AlphaType, P0)                                 \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_NONE, MemLayout, DataKind, \
                            Swizzle, AlphaType, P0, 0, 0, 0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_NONCOLOR_FORMAT2(MemLayout, DataKind, Swizzle, AlphaType, P0, P1)                             \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_NONE, MemLayout, DataKind, \
                            Swizzle, AlphaType, P0, P1, 0, 0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_NONCOLOR_FORMAT3(MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2)                         \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_NONE, MemLayout, DataKind, \
                            Swizzle, AlphaType, P0, P1, P2, 0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_NONCOLOR_FORMAT4(MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3)                     \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_NONE, MemLayout, DataKind, \
                            Swizzle, AlphaType, P0, P1, P2, P3, 0, 0, NVCV_DATA_KIND_UNSPECIFIED,                      \
                            NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_NONCOLOR_FORMAT(MemLayout, DataKind, Swizzle, AlphaType, NumPlanes, ...) \
    NVCV_DETAIL_MAKE_NONCOLOR_FORMAT##NumPlanes(MemLayout, DataKind, Swizzle, AlphaType, __VA_ARGS__)

// Extra Channel

#define NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FORMAT1(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel,     \
                                                         ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0)  \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_NONE, MemLayout, DataKind, \
                            Swizzle, AlphaType, P0, 0, 0, 0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,   \
                            ExtraChannelType)

#define NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FORMAT2(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel,     \
                                                         ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0,  \
                                                         P1)                                                           \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_NONE, MemLayout, DataKind, \
                            Swizzle, AlphaType, P0, P1, 0, 0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,  \
                            ExtraChannelType)

#define NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FORMAT3(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel,     \
                                                         ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0,  \
                                                         P1, P2)                                                       \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_NONE, MemLayout, DataKind, \
                            Swizzle, AlphaType, P0, P1, P2, 0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, \
                            ExtraChannelType)

#define NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FORMAT4(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel,     \
                                                         ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0,  \
                                                         P1, P2, P3)                                                   \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_NONE, MemLayout, DataKind, \
                            Swizzle, AlphaType, P0, P1, P2, P3, NumExtraChannel, ExtraChannelBPP,                      \
                            ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FORMAT(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel,      \
                                                        ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType,       \
                                                        NumPlanes, ...)                                                \
    NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FORMAT##NumPlanes(MemLayout, DataKind, Swizzle, AlphaType,                \
                                                               NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, \
                                                               ExtraChannelType, __VA_ARGS__)

// Abbreviated

// No extra channels

#define NVCV_DETAIL_MAKE_NONCOLOR_FMT1(MemLayout, DataKind, Swizzle, AlphaType, P0)                              \
    NVCV_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, 0, 0, 0, 0, 0, \
                         UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_NONCOLOR_FMT2(MemLayout, DataKind, Swizzle, AlphaType, P0, P1)                           \
    NVCV_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, 0, 0, 0, 0, \
                         UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_NONCOLOR_FMT3(MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2)                        \
    NVCV_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, 0, 0, 0, \
                         UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_NONCOLOR_FMT4(MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3)                     \
    NVCV_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3, 0, 0, \
                         UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_NONCOLOR_FMT(MemLayout, DataKind, Swizzle, AlphaType, NumPlanes, ...) \
    NVCV_DETAIL_MAKE_NONCOLOR_FMT##NumPlanes(MemLayout, DataKind, Swizzle, AlphaType, __VA_ARGS__)

// Extra channels

#define NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FMT1(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel,    \
                                                      ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0) \
    NVCV_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, 0, 0, 0,         \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FMT2(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel,        \
                                                      ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0, P1) \
    NVCV_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, 0, 0,            \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FMT3(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel,        \
                                                      ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0, P1, \
                                                      P2)                                                              \
    NVCV_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, 0,           \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FMT4(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel,        \
                                                      ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0, P1, \
                                                      P2, P3)                                                          \
    NVCV_DETAIL_MAKE_FMT(UNDEFINED, UNDEFINED, NONE, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3,          \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FMT(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel,        \
                                                     ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType,         \
                                                     NumPlanes, ...)                                                  \
    NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FMT##NumPlanes(MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel, \
                                                            ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType,  \
                                                            __VA_ARGS__)

// MAKE_RAW =============================================

// Full arg name

// No extra channel

#define NVCV_DETAIL_MAKE_RAW_FORMAT1(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, P0)                         \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_RAW, RawPattern, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, \
                            P0, 0, 0, 0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_RAW_FORMAT2(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, P0, P1)                     \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_RAW, RawPattern, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, \
                            P0, P1, 0, 0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_RAW_FORMAT3(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2)                 \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_RAW, RawPattern, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, \
                            P0, P1, P2, 0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_RAW_FORMAT4(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3)             \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_RAW, RawPattern, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, \
                            P0, P1, P2, P3, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_RAW_FORMAT(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, NumPlanes, ...) \
    NVCV_DETAIL_MAKE_RAW_FORMAT##NumPlanes(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, __VA_ARGS__)

// Extra channels

#define NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT1(RawPattern, MemLayout, DataKind, Swizzle, AlphaType,              \
                                                    NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,           \
                                                    ExtraChannelType, P0)                                             \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_RAW, RawPattern, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, \
                            P0, 0, 0, 0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT2(RawPattern, MemLayout, DataKind, Swizzle, AlphaType,              \
                                                    NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,           \
                                                    ExtraChannelType, P0, P1)                                         \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_RAW, RawPattern, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, \
                            P0, P1, 0, 0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT3(RawPattern, MemLayout, DataKind, Swizzle, AlphaType,              \
                                                    NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,           \
                                                    ExtraChannelType, P0, P1, P2)                                     \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_RAW, RawPattern, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, \
                            P0, P1, P2, 0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT4(RawPattern, MemLayout, DataKind, Swizzle, AlphaType,              \
                                                    NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,           \
                                                    ExtraChannelType, P0, P1, P2, P3)                                 \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_RAW, RawPattern, NVCV_CSS_NONE, MemLayout, DataKind, Swizzle, AlphaType, \
                            P0, P1, P2, P3, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT(RawPattern, MemLayout, DataKind, Swizzle, AlphaType,           \
                                                   NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,        \
                                                   ExtraChannelType, NumPlanes, ...)                              \
    NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT##NumPlanes(RawPattern, MemLayout, DataKind, Swizzle, AlphaType,    \
                                                          NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, \
                                                          ExtraChannelType, __VA_ARGS__)

// Abbreviated

// No extra channels

#define NVCV_DETAIL_MAKE_RAW_FMT1(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, P0)                      \
    NVCV_DETAIL_MAKE_RAW_FORMAT1(NVCV_RAW_##RawPattern, NVCV_MEM_LAYOUT_##MemLayout, NVCV_DATA_KIND_##DataKind, \
                                 NVCV_SWIZZLE_##Swizzle, NVCV_ALPHA_##AlphaType, NVCV_PACKING_##P0)

#define NVCV_DETAIL_MAKE_RAW_FMT2(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, P0, P1)                  \
    NVCV_DETAIL_MAKE_RAW_FORMAT2(NVCV_RAW_##RawPattern, NVCV_MEM_LAYOUT_##MemLayout, NVCV_DATA_KIND_##DataKind, \
                                 NVCV_SWIZZLE_##Swizzle, NVCV_ALPHA_##AlphaType, NVCV_PACKING_##P0, NVCV_PACKING_##P1)

#define NVCV_DETAIL_MAKE_RAW_FMT3(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2)                     \
    NVCV_DETAIL_MAKE_RAW_FORMAT3(NVCV_RAW_##RawPattern, NVCV_MEM_LAYOUT_##MemLayout, NVCV_DATA_KIND_##DataKind,        \
                                 NVCV_SWIZZLE_##Swizzle, NVCV_ALPHA_##AlphaType, NVCV_PACKING_##P0, NVCV_PACKING_##P1, \
                                 NVCV_PACKING_##P2)

#define NVCV_DETAIL_MAKE_RAW_FMT4(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3)                 \
    NVCV_DETAIL_MAKE_RAW_FORMAT4(NVCV_RAW_##RawPattern, NVCV_MEM_LAYOUT_##MemLayout, NVCV_DATA_KIND_##DataKind,        \
                                 NVCV_SWIZZLE_##Swizzle, NVCV_ALPHA_##AlphaType, NVCV_PACKING_##P0, NVCV_PACKING_##P1, \
                                 NVCV_PACKING_##P2, NVCV_PACKING_##P3)

#define NVCV_DETAIL_MAKE_RAW_FMT(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, NumPlanes, ...) \
    NVCV_DETAIL_MAKE_RAW_FMT##NumPlanes(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, __VA_ARGS__)

// Extra channels

#define NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FMT1(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel, \
                                                 ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0)          \
    NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT1(                                                                       \
        NVCV_RAW_##RawPattern, NVCV_MEM_LAYOUT_##MemLayout, NVCV_DATA_KIND_##DataKind, NVCV_SWIZZLE_##Swizzle,         \
        NVCV_ALPHA_##AlphaType, NVCV_PACKING_##P0, NumExtraChannel, ExtraChannelBPP,                                   \
        NVCV_DATAKIND_##ExtraChannelDataKind, NVCV_EXTRA_CHANNEL_##ExtraChannelType)

#define NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FMT2(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel, \
                                                 ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0, P1)      \
    NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT2(                                                                       \
        NVCV_RAW_##RawPattern, NVCV_MEM_LAYOUT_##MemLayout, NVCV_DATA_KIND_##DataKind, NVCV_SWIZZLE_##Swizzle,         \
        NVCV_ALPHA_##AlphaType, NVCV_PACKING_##P0, NVCV_PACKING_##P1, NumExtraChannel, ExtraChannelBPP,                \
        NVCV_DATAKIND_##ExtraChannelDataKind, NVCV_EXTRA_CHANNEL_##ExtraChannelType)

#define NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FMT3(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel, \
                                                 ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0, P1, P2)  \
    NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT3(                                                                       \
        NVCV_RAW_##RawPattern, NVCV_MEM_LAYOUT_##MemLayout, NVCV_DATA_KIND_##DataKind, NVCV_SWIZZLE_##Swizzle,         \
        NVCV_ALPHA_##AlphaType, NVCV_PACKING_##P0, NVCV_PACKING_##P1, NVCV_PACKING_##P2, NumExtraChannel,              \
        ExtraChannelBPP, NVCV_DATAKIND_##ExtraChannelDataKind, NVCV_EXTRA_CHANNEL_##ExtraChannelType)

#define NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FMT4(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel, \
                                                 ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, P0, P1, P2,  \
                                                 P3)                                                                   \
    NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT4(                                                                       \
        NVCV_RAW_##RawPattern, NVCV_MEM_LAYOUT_##MemLayout, NVCV_DATA_KIND_##DataKind, NVCV_SWIZZLE_##Swizzle,         \
        NVCV_ALPHA_##AlphaType, NVCV_PACKING_##P0, NVCV_PACKING_##P1, NVCV_PACKING_##P2, NVCV_PACKING_##P3,            \
        NumExtraChannel, ExtraChannelBPP, NVCV_DATAKIND_##ExtraChannelDataKind, NVCV_EXTRA_CHANNEL_##ExtraChannelType)

#define NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FMT(RawPattern, MemLayout, DataKind, Swizzle, AlphaType, NumExtraChannel, \
                                                ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType, NumPlanes,   \
                                                ...)                                                                  \
    NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FMT##NumPlanes(RawPattern, MemLayout, DataKind, Swizzle, AlphaType,           \
                                                       NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,        \
                                                       ExtraChannelType, __VA_ARGS__)

// MAKE_YCbCr ===============================================

// Full arg name

// No extra channels

#define NVCV_DETAIL_MAKE_YCbCr_FORMAT1(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0) \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, \
                            AlphaType, P0, 0, 0, 0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_YCbCr_FORMAT2(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1) \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle,     \
                            AlphaType, P0, P1, 0, 0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_YCbCr_FORMAT3(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2) \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle,         \
                            AlphaType, P0, P1, P2, 0, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_YCbCr_FORMAT4(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, \
                                       P3)                                                                            \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle,         \
                            AlphaType, P0, P1, P2, P3, 0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U)

#define NVCV_DETAIL_MAKE_YCbCr_FORMAT(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, Numplanes, \
                                      ...)                                                                          \
    NVCV_DETAIL_MAKE_YCbCr_FORMAT##Numplanes(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType,     \
                                             __VA_ARGS__)

// Extra channels

#define NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FORMAT1(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, \
                                                      AlphaType, NumExtraChannel, ExtraChannelBPP,            \
                                                      ExtraChannelDataKind, ExtraChannelType, P0)             \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, \
                            AlphaType, P0, 0, 0, 0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,   \
                            ExtraChannelType)

#define NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FORMAT2(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, \
                                                      AlphaType, NumExtraChannel, ExtraChannelBPP,            \
                                                      ExtraChannelDataKind, ExtraChannelType, P0, P1)         \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, \
                            AlphaType, P0, P1, 0, 0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,  \
                            ExtraChannelType)

#define NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FORMAT3(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, \
                                                      AlphaType, NumExtraChannel, ExtraChannelBPP,            \
                                                      ExtraChannelDataKind, ExtraChannelType, P0, P1, P2)     \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, \
                            AlphaType, P0, P1, P2, 0, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, \
                            ExtraChannelType)

#define NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FORMAT4(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle,  \
                                                      AlphaType, NumExtraChannel, ExtraChannelBPP,             \
                                                      ExtraChannelDataKind, ExtraChannelType, P0, P1, P2, P3)  \
    NVCV_DETAIL_MAKE_FORMAT(NVCV_COLOR_MODEL_##YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle,  \
                            AlphaType, P0, P1, P2, P3, NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, \
                            ExtraChannelType)

#define NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FORMAT(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle,        \
                                                     AlphaType, NumExtraChannel, ExtraChannelBPP,                   \
                                                     ExtraChannelDataKind, ExtraChannelType, Numplanes, ...)        \
    NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FORMAT##Numplanes(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, \
                                                            AlphaType, NumExtraChannel, ExtraChannelBPP,            \
                                                            ExtraChannelDataKind, ExtraChannelType, __VA_ARGS__)

// Abbreviated

// No extra channels

#define NVCV_DETAIL_MAKE_YCbCr_FMT1(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0)            \
    NVCV_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, 0, 0, 0, 0, 0, \
                         UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_YCbCr_FMT2(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1)         \
    NVCV_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, 0, 0, 0, 0, \
                         UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_YCbCr_FMT3(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2)   \
    NVCV_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, 0, 0, \
                         0, UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_YCbCr_FMT4(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3) \
    NVCV_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3, 0,  \
                         0, UNSPECIFIED, U)

#define NVCV_DETAIL_MAKE_YCbCr_FMT(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, Numplanes, ...) \
    NVCV_DETAIL_MAKE_YCbCr_FMT##NumPlanes(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType,          \
                                          __VA_ARGS__)

// Extra channels

#define NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FMT1(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, \
                                                   NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,            \
                                                   ExtraChannelType, P0)                                              \
    NVCV_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, 0, 0, 0,       \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FMT2(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, \
                                                   NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,            \
                                                   ExtraChannelType, P0, P1)                                          \
    NVCV_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, 0, 0,      \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FMT3(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, \
                                                   NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,            \
                                                   ExtraChannelType, P0, P1, P2)                                      \
    NVCV_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, 0,     \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FMT4(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, \
                                                   NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,            \
                                                   ExtraChannelType, P0, P1, P2, P3)                                  \
    NVCV_DETAIL_MAKE_FMT(YCbCr, ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, P0, P1, P2, P3,    \
                         NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind, ExtraChannelType)

#define NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FMT(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle, AlphaType, \
                                                  NumExtraChannel, ExtraChannelBPP, ExtraChannelDataKind,            \
                                                  ExtraChannelType, Numplanes, ...)                                  \
    NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FMT##NumPlanes(ColorSpec, ChromaSubsamp, MemLayout, DataKind, Swizzle,     \
                                                         AlphaType, NumExtraChannel, ExtraChannelBPP,                \
                                                         ExtraChannelDataKind, ExtraChannelType, __VA_ARGS__)

#endif /* NVCV_DETAIL_FORMATUTILS_H */

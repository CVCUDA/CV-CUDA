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

/**
 * @file ImageFormat.h
 *
 * @brief Defines types and functions to handle image formats.
 */

#ifndef NVCV_FORMAT_IMAGEFORMAT_H
#define NVCV_FORMAT_IMAGEFORMAT_H

#include "ColorSpec.h"
#include "DataLayout.h"
#include "DataType.h"

#include <assert.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Pre-defined image formats.
 * An image format defines how image pixels are interpreted.
 * Each image format is defined by the following components:
 * - \ref NVCVColorModel
 * - \ref NVCVColorSpec
 * - \ref NVCVChromaSubsampling method (when applicable)
 * - \ref NVCVMemLayout
 * - \ref NVCVDataKind
 * - \ref NVCVSwizzle
 * - \ref NVCVAlphaType
 * - Number of planes
 * - Format packing of each plane.
 *
 * These pre-defined formats are guaranteed to work with algorithms that explicitly support them.
 * Image formats can also be user-defined using the nvcvMakeImageFormat family of functions.
 *
 * Using user-defined image formats with algorithms can lead to undefined behavior (segfaults, etc),
 * but usually it works as expected. Result of algorithms using these image formats must be checked
 * for correctness, as it's not guaranteed that they will work.
 *
 * @defgroup NVCV_C_CORE_IMAGETYPE Image Formats
 * @{
 */
typedef uint64_t NVCVImageFormat;

// clang-format off
/** Denotes a special image format that doesn't represent any format. */
#define NVCV_IMAGE_FORMAT_NONE ((NVCVImageFormat)0)

/** Single plane with one 8-bit unsigned integer channel. */
#define NVCV_IMAGE_FORMAT_U8 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, X000, ASSOCIATED, X8)

/** Single plane with one block-linear 8-bit unsigned integer channel. */
#define NVCV_IMAGE_FORMAT_U8_BL NVCV_DETAIL_MAKE_NONCOLOR_FMT1(BL, UNSIGNED, X000, ASSOCIATED, X8)

/** Single plane with one 8-bit signed integer channel. */
#define NVCV_IMAGE_FORMAT_S8 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, X000, ASSOCIATED, X8)

/** Single plane with one 16-bit unsigned integer channel. */
#define NVCV_IMAGE_FORMAT_U16 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, X000, ASSOCIATED, X16)

/** Single plane with one 32-bit unsigned integer channel. */
#define NVCV_IMAGE_FORMAT_U32 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, X000, ASSOCIATED, X32)

/** Single plane with one 32-bit signed integer channel.*/
#define NVCV_IMAGE_FORMAT_S32 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, X000, ASSOCIATED, X32)

/** Single plane with one 16-bit signed integer channel.*/
#define NVCV_IMAGE_FORMAT_S16 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, X000, ASSOCIATED, X16)

/** Single plane with one block-linear 16-bit signed integer channel.*/
#define NVCV_IMAGE_FORMAT_S16_BL NVCV_DETAIL_MAKE_NONCOLOR_FMT1(BL, SIGNED, X000, ASSOCIATED, X16)

/** Single plane with two interleaved 16-bit signed integer channel.*/
#define NVCV_IMAGE_FORMAT_2S16 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, XY00, ASSOCIATED, X16_Y16)

/** Single plane with two interleaved block-linear 16-bit signed integer channel.*/
#define NVCV_IMAGE_FORMAT_2S16_BL NVCV_DETAIL_MAKE_NONCOLOR_FMT1(BL, SIGNED, XY00, ASSOCIATED, X16_Y16)

/** Single plane with one 16-bit floating point channel. */
#define NVCV_IMAGE_FORMAT_F16 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, FLOAT, X000, ASSOCIATED, X16)

/** Single plane with one 32-bit floating point channel. */
#define NVCV_IMAGE_FORMAT_F32 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, FLOAT, X000, ASSOCIATED, X32)

/** Single plane with one 64-bit floating point channel. */
#define NVCV_IMAGE_FORMAT_F64 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, FLOAT, X000, ASSOCIATED, X64)

/** Single plane with two interleaved 16-bit floating point channels. */
#define NVCV_IMAGE_FORMAT_2F16 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, FLOAT, XY00, ASSOCIATED, X16_Y16)

/** Single plane with two interleaved 32-bit floating point channels. */
#define NVCV_IMAGE_FORMAT_2F32 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, FLOAT, XY00, ASSOCIATED, X32_Y32)

/** Single plane with one 64-bit complex floating point channel. */
#define NVCV_IMAGE_FORMAT_C64 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, COMPLEX, X000, ASSOCIATED, X64)

/** Single plane with two interleaved 64-bit complex floating point channels. */
#define NVCV_IMAGE_FORMAT_2C64 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, COMPLEX, XY00, ASSOCIATED, X64_Y64)

/** Single plane with one 128-bit complex floating point channel. */
#define NVCV_IMAGE_FORMAT_C128 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, COMPLEX, X000, ASSOCIATED, X128)

/** Single plane with two interleaved 128-bit complex floating point channels. */
#define NVCV_IMAGE_FORMAT_2C128 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, COMPLEX, XY00, ASSOCIATED, X128_Y128)

/** Single plane with one pitch-linear 8-bit unsigned integer channel with limited-range luma (grayscale) information.
 * Values range from 16 to 235. Below this range is considered black, above is considered white.
 */
#define NVCV_IMAGE_FORMAT_Y8 NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, UNSIGNED, X000, ASSOCIATED, X8)

/** Single plane with one block-linear 8-bit unsigned integer channel with limited-range luma (grayscale) information.
 * Values range from 16 to 235. Below this range is considered black, above is considered white.
 */
#define NVCV_IMAGE_FORMAT_Y8_BL NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, BL, UNSIGNED, X000, ASSOCIATED, X8)

/** Single plane with one pitch-linear 8-bit unsigned integer channel with full-range luma (grayscale) information.
 * Values range from 0 to 255.
 */
#define NVCV_IMAGE_FORMAT_Y8_ER NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, NONE, PL, UNSIGNED, X000, ASSOCIATED, X8)

/** Single plane with one block-linear 8-bit unsigned integer channel with full-range luma (grayscale) information.
 * Values range from 0 to 255.
 */
#define NVCV_IMAGE_FORMAT_Y8_ER_BL NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, NONE, BL, UNSIGNED, X000, ASSOCIATED, X8)

/** Single plane with one pitch-linear 16-bit unsigned integer channel with limited-range luma (grayscale) information.
 * Values range from 4096 to 60160. Below this range is considered black, above is considered white.
 */
#define NVCV_IMAGE_FORMAT_Y16 NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, UNSIGNED, X000, ASSOCIATED, X16)

/** Single plane with one block-linear 16-bit unsigned integer channel with limited-range luma (grayscale) information.
 * Values range from 4096 to 60160. Below this range is considered black, above is considered white.
 */
#define NVCV_IMAGE_FORMAT_Y16_BL NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, BL, UNSIGNED, X000, ASSOCIATED, X16)

/** Single plane with one pitch-linear 16-bit unsigned integer channel with full-range luma (grayscale) information.
 * Values range from 0 to 65535.
 */
#define NVCV_IMAGE_FORMAT_Y16_ER NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, NONE, PL, UNSIGNED, X000, ASSOCIATED, X16)

/** Single plane with one block-linear 16-bit unsigned integer channel with full-range luma (grayscale) information.
 * Values range from 0 to 65535.
 */
#define NVCV_IMAGE_FORMAT_Y16_ER_BL NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, NONE, BL, UNSIGNED, X000, ASSOCIATED, X16)

/** YUV420sp 8-bit pitch-linear format with limited range.
 * Format is composed of two planes:
 * 1. One 8-bit channel with luma (Y'). Values range from 16 to 235.
 * 2. Two interleaved 8-bit channels with chroma (Cb,Cr).
 *    Values range from 16 to 240. Resolution is half of luma plane,
 *    both horizontally and vertically.
 *    For a given pixel, Cb channel has lower memory address than Cr.
 */
#define NVCV_IMAGE_FORMAT_NV12 NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601, 420, PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8)

/** YUV420sp 8-bit block-linear format with limited range.
 * Format is composed of two planes:
 * 1. One 8-bit channel with luma (Y'). Values range from 16 to 235.
 * 2. Two interleaved 8-bit channels with chroma (Cb,Cr).
 *    Values range from 0 to 255. Resolution is half of luma plane,
 *    both horizontally and vertically.
 *    For a given pixel, Cb channel has lower memory address than Cr.
 */
#define NVCV_IMAGE_FORMAT_NV12_BL NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601, 420, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8)

/** YUV420sp 8-bit pitch-linear format with full range.
 * Format is composed of two planes:
 * 1. One 8-bit channel with luma (Y'). Values range from 0 to 255.
 * 2. Two interleaved 8-bit channels with chroma (Cb,Cr).
 *    Values range from 0 to 255. Resolution is half of luma plane,
 *    both horizontally and vertically.
 *    For a given pixel, Cb channel has lower memory address than Cr.
 */
#define NVCV_IMAGE_FORMAT_NV12_ER NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601_ER, 420, PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8)

/** YUV420sp 8-bit block-linear format with full range.
 * Format is composed of two planes:
 * 1. One 8-bit channel with luma (Y'). Values range from 0 to 255.
 * 2. Two interleaved 8-bit channels with chroma (Cb,Cr).
 *    Values range from 0 to 255. Resolution is half of luma plane,
 *    both horizontally and vertically.
 *    For a given pixel, Cb channel has lower memory address than Cr.
 */
#define NVCV_IMAGE_FORMAT_NV12_ER_BL NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601_ER, 420, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8)

/** YUV420sp 8-bit block-linear format with limited range.
 * Format is composed of two planes:
 * 1. One 8-bit channel with luma (Y'). Values range from 16 to 255.
 * 2. Two interleaved 8-bit channels with chroma (Cr,Cb).
 *    Values range from 0 to 255. Resolution is half of luma plane,
 *    both horizontally and vertically.
 *    For a given pixel, Cb channel has lower memory address than Cr.
 */
#define NVCV_IMAGE_FORMAT_NV21 NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601, 420, PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8)

/** YUV420sp 8-bit pitch-linear format with full range.
 * Format is composed of two planes:
 * 1. One 8-bit channel with luma (Y'). Values range from 0 to 255.
 * 2. Two interleaved 8-bit channels with chroma (Cr,Cb).
 *    Values range from 0 to 255. Resolution is half of luma plane,
 *    both horizontally and vertically.
 *    For a given pixel, Cb channel has lower memory address than Cr.
 */
#define NVCV_IMAGE_FORMAT_NV21_ER NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601_ER, 420, PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8)

/** YUV444sp 8-bit pitch-linear format with limited range.
 * Format is composed of two planes:
 * 1. One 8-bit channel with luma (Y'). Values range from 16 to 235.
 * 2. Two interleaved 8-bit channels with chroma (Cb,Cr).
 *    Values range from 16 to 240. It has the same resolution as luma plane.
 *    For a given pixel, Cb channel has lower memory address than Cr.
 */
#define NVCV_IMAGE_FORMAT_NV24 NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601, 444, PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8)

/** YUV444sp 8-bit block-linear format with limited range.
 * Format is composed of two planes:
 * 1. One 8-bit channel with luma (Y'). Values range from 16 to 235.
 * 2. Two interleaved 8-bit channels with chroma (Cb,Cr).
 *    Values range from 0 to 255. It has the same resolution as luma plane.
 *    For a given pixel, Cb channel has lower memory address than Cr.
 */
#define NVCV_IMAGE_FORMAT_NV24_BL NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601, 444, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8)

/** YUV444sp 8-bit pitch-linear format with full range.
 * Format is composed of two planes:
 * 1. One 8-bit channel with luma (Y'). Values range from 0 to 255.
 * 2. Two interleaved 8-bit channels with chroma (Cb,Cr).
 *    Values range from 0 to 255. It has the same resolution as luma plane.
 *    For a given pixel, Cb channel has lower memory address than Cr.
 */
#define NVCV_IMAGE_FORMAT_NV24_ER NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601_ER, 444, PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8)

/** YUV444sp 8-bit block-linear format with full range.
 * Format is composed of two planes:
 * 1. One 8-bit channel with luma (Y'). Values range from 0 to 255.
 * 2. Two interleaved 8-bit channels with chroma (Cb,Cr).
 *    Values range from 0 to 255. It has the same resolution as luma plane.
 *    For a given pixel, Cb channel has lower memory address than Cr.
 */
#define NVCV_IMAGE_FORMAT_NV24_ER_BL NVCV_DETAIL_MAKE_YCbCr_FMT2(BT601_ER, 444, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8)

/** YUV422 8-bit pitch-linear format in one plane with UYVY ordering and limited range. */
#define NVCV_IMAGE_FORMAT_UYVY NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, 422, PL, UNSIGNED, XYZ1, ASSOCIATED, Y8_X8__Z8_X8)

/** YUV422 8-bit block-linear format in one plane with UYVY ordering and limited range. */
#define NVCV_IMAGE_FORMAT_UYVY_BL NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, 422, BL, UNSIGNED, XYZ1, ASSOCIATED, Y8_X8__Z8_X8)

/** YUV422 8-bit pitch-linear format in one plane with UYVY ordering and full range. */
#define NVCV_IMAGE_FORMAT_UYVY_ER NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, 422, PL, UNSIGNED, XYZ1, ASSOCIATED, Y8_X8__Z8_X8)

/** YUV422 8-bit block-linear format in one plane with UYVY ordering and full range. */
#define NVCV_IMAGE_FORMAT_UYVY_ER_BL NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, 422, BL, UNSIGNED, XYZ1, ASSOCIATED, Y8_X8__Z8_X8)

/** YUV422 8-bit pitch-linear format in one plane with VYUY ordering and limited range. */
#define NVCV_IMAGE_FORMAT_VYUY NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, 422, PL, UNSIGNED, XZY1, ASSOCIATED, Y8_X8__Z8_X8)

/** YUV422 8-bit block-linear format in one plane with VYUY ordering and limited range. */
#define NVCV_IMAGE_FORMAT_VYUY_BL NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, 422, BL, UNSIGNED, XZY1, ASSOCIATED, Y8_X8__Z8_X8)

/** YUV422 8-bit pitch-linear format in one plane with VYUY ordering and full range. */
#define NVCV_IMAGE_FORMAT_VYUY_ER NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, 422, PL, UNSIGNED, XZY1, ASSOCIATED, Y8_X8__Z8_X8)

/** YUV422 8-bit block-linear format in one plane with VYUY ordering and full range. */
#define NVCV_IMAGE_FORMAT_VYUY_ER_BL NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, 422, BL, UNSIGNED, XZY1, ASSOCIATED, Y8_X8__Z8_X8)

/** YUV422 8-bit pitch-linear format in one plane with YUYV ordering and limited range.
 * Also known as YUY2 format.
 */
#define NVCV_IMAGE_FORMAT_YUYV NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, 422, PL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8__X8_Z8)

/** YUV422 8-bit block-linear format in one plane with YUYV ordering and limited range.
 * Also known as YUY2 format.
 */
#define NVCV_IMAGE_FORMAT_YUYV_BL NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, 422, BL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8__X8_Z8)

/** YUV422 8-bit pitch-linear format in one plane with YUYV ordering and full range.
 * Also known as YUY2 format.
 */
#define NVCV_IMAGE_FORMAT_YUYV_ER NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, 422, PL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8__X8_Z8)

/** YUV422 8-bit block-linear format in one plane with YUYV ordering and full range.
 * Also known as YUY2 format.
 */
#define NVCV_IMAGE_FORMAT_YUYV_ER_BL NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601_ER, 422, BL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8__X8_Z8)

/** Single plane with interleaved YUV 8-bit channel. */
#define NVCV_IMAGE_FORMAT_YUV8  NVCV_DETAIL_MAKE_YCbCr_FMT1(BT601, NONE, PL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8_Z8)

/** Planar YUV unsigned 8-bit per channel with limited range. */
#define NVCV_IMAGE_FORMAT_YUV8p NVCV_DETAIL_MAKE_YCbCr_FMT3(BT601, NONE, PL, UNSIGNED, XYZ1, ASSOCIATED, X8, X8, X8)

/** Planar YUV unsigned 8-bit per channel with full range. */
#define NVCV_IMAGE_FORMAT_YUV8p_ER NVCV_DETAIL_MAKE_YCbCr_FMT3(BT601_ER, NONE, PL, UNSIGNED, XYZ1, ASSOCIATED, X8, X8, X8)

/** Single plane with interleaved RGB 8-bit channel. */
#define NVCV_IMAGE_FORMAT_RGB8 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8_Z8)

/** Single plane with interleaved BGR 8-bit channel. */
#define NVCV_IMAGE_FORMAT_BGR8 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, ZYX1, ASSOCIATED, X8_Y8_Z8)

/** Single plane with interleaved RGBA 8-bit channel. */
#define NVCV_IMAGE_FORMAT_RGBA8 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8_Z8_W8)

/** Single plane with interleaved BGRA 8-bit channel. */
#define NVCV_IMAGE_FORMAT_BGRA8 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, ZYXW, ASSOCIATED, X8_Y8_Z8_W8)

/** Planar RGB unsigned 8-bit per channel. */
#define NVCV_IMAGE_FORMAT_RGB8p NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8)

/** Planar BGR unsigned 8-bit per channel. */
#define NVCV_IMAGE_FORMAT_BGR8p NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, UNSIGNED, ZYX1, ASSOCIATED, X8, X8, X8)

/** Planar RGBA unsigned 8-bit per channel. */
#define NVCV_IMAGE_FORMAT_RGBA8p NVCV_DETAIL_MAKE_COLOR_FMT4(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X8, X8, X8, X8)

/** Planar BGRA unsigned 8-bit per channel. */
#define NVCV_IMAGE_FORMAT_BGRA8p NVCV_DETAIL_MAKE_COLOR_FMT4(RGB, UNDEFINED, PL, UNSIGNED, ZYXW, ASSOCIATED, X8, X8, X8, X8)

/** Single plane with interleaved RGB float16 channel. */
#define NVCV_IMAGE_FORMAT_RGBf16 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZ1, ASSOCIATED, X16_Y16_Z16)

/** Single plane with interleaved BGR float16 channel. */
#define NVCV_IMAGE_FORMAT_BGRf16 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, ZYX1, ASSOCIATED, X16_Y16_Z16)

/** Single plane with interleaved RGBA float16 channel. */
#define NVCV_IMAGE_FORMAT_RGBAf16 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZW, ASSOCIATED, X16_Y16_Z16_W16)

/** Single plane with interleaved BGRA float16 channel. */
#define NVCV_IMAGE_FORMAT_BGRAf16 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, ZYXW, ASSOCIATED, X16_Y16_Z16_W16)

/** Planar RGB unsigned float16 per channel. */
#define NVCV_IMAGE_FORMAT_RGBf16p NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, FLOAT, XYZ0, ASSOCIATED, X16, X16, X16)

/** Planar BGR unsigned float16 per channel. */
#define NVCV_IMAGE_FORMAT_BGRf16p NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, FLOAT, ZYX1, ASSOCIATED, X16, X16, X16)

/** Planar RGBA unsigned float16 per channel. */
#define NVCV_IMAGE_FORMAT_RGBAf16p NVCV_DETAIL_MAKE_COLOR_FMT4(RGB, UNDEFINED, PL, FLOAT, XYZW, ASSOCIATED, X16, X16, X16, X16)

/** Planar BGRA unsigned float16 per channel. */
#define NVCV_IMAGE_FORMAT_BGRAf16p NVCV_DETAIL_MAKE_COLOR_FMT4(RGB, UNDEFINED, PL, FLOAT, ZYXW, ASSOCIATED, X16, X16, X16, X16)

/** Single plane with interleaved RGB float32 channel. */
#define NVCV_IMAGE_FORMAT_RGBf32 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZ1, ASSOCIATED, X32_Y32_Z32)

/** Single plane with interleaved BGR float32 channel. */
#define NVCV_IMAGE_FORMAT_BGRf32 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, ZYX1, ASSOCIATED, X32_Y32_Z32)

/** Single plane with interleaved RGBA float32 channel. */
#define NVCV_IMAGE_FORMAT_RGBAf32 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZW, ASSOCIATED, X32_Y32_Z32_W32)

/** Single plane with interleaved BGRA float32 channel. */
#define NVCV_IMAGE_FORMAT_BGRAf32 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, ZYXW, ASSOCIATED, X32_Y32_Z32_W32)

/** Planar RGB unsigned float32 per channel. */
#define NVCV_IMAGE_FORMAT_RGBf32p NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, FLOAT, XYZ0, ASSOCIATED, X32, X32, X32)

/** Planar BGR unsigned float32 per channel. */
#define NVCV_IMAGE_FORMAT_BGRf32p NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, FLOAT, ZYX1, ASSOCIATED, X32, X32, X32)

/** Planar RGBA unsigned float32 per channel. */
#define NVCV_IMAGE_FORMAT_RGBAf32p NVCV_DETAIL_MAKE_COLOR_FMT4(RGB, UNDEFINED, PL, FLOAT, XYZW, ASSOCIATED, X32, X32, X32, X32)

/** Planar BGRA unsigned float32 per channel. */
#define NVCV_IMAGE_FORMAT_BGRAf32p NVCV_DETAIL_MAKE_COLOR_FMT4(RGB, UNDEFINED, PL, FLOAT, ZYXW, ASSOCIATED, X32, X32, X32, X32)

/** Single plane with interleaved HSV 8-bit channel. */
#define NVCV_IMAGE_FORMAT_HSV8 NVCV_DETAIL_MAKE_COLOR_FMT1(HSV, UNDEFINED, PL, UNSIGNED, XYZ0, ASSOCIATED, X8_Y8_Z8)

/** Single plane with interleaved CMYK 8-bit channel. */
#define NVCV_IMAGE_FORMAT_CMYK8 NVCV_DETAIL_MAKE_COLOR_FMT1(CMYK, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8_Z8_W8)

/** Single plane with interleaved YCCK 8-bit channel. */
#define NVCV_IMAGE_FORMAT_YCCK8 NVCV_DETAIL_MAKE_COLOR_FMT1(YCCK, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8_Z8_W8)

/** Single plane with interleaved RGBA 8-bit channel with alpha channel is unassociated */
#define NVCV_IMAGE_FORMAT_RGBA8_UNASSOCIATED_ALPHA NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZW, UNASSOCIATED, X8_Y8_Z8_W8)

/** Single plane with interleaved RGB 8-bit channel and 1 extra 8-bit unspecified channel */
#define NVCV_IMAGE_FORMAT_RGB8_1U_U8 NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, 1, 8, UNSIGNED, U, X8_Y8_Z8)

/** Single plane with interleaved RGB 8-bit channel and 7 extra 8-bit unspecified channel */
#define NVCV_IMAGE_FORMAT_RGB8_7U_U8 NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, 7, 8, UNSIGNED, U, X8_Y8_Z8)

/** Single plane with interleaved RGBA 8-bit channels and 3 extra 16-bit unspecified channel */
#define NVCV_IMAGE_FORMAT_RGBA8_3U_U16 NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, 3, 16, UNSIGNED, U, X8_Y8_Z8_W8)

/** Single plane with interleaved RGBA 8-bit channel and 3 extra 32-bit unsigned int 3D position channels */
#define NVCV_IMAGE_FORMAT_RGBA8_3POS3D_U32 NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZW, ASSOCIATED, 3, 32, UNSIGNED, POS3D, X8_Y8_Z8)

/** Single plane with interleaved RGB 8-bit channel and 3 extra 32-bit float depth channels */
#define NVCV_IMAGE_FORMAT_RGB8_3D_F32 NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, 3, 32, FLOAT, D, X8_Y8_Z8)
// clang-format on

/** Creates a user-defined YCbCr color image format constant.
 *
 * Example to create a YUV422R ITU-R BT.709 full-range with SMPTE240M transfer function, block-linear format.
 * If there is no alpha channel, the value of alphaType does not matter.
 * \code{.c}
 * NVCVImageFormat fmt = NVCV_MAKE_YCbCr_IMAGE_FORMAT(NVCV_MAKE_COLOR_SPEC(BT601, SMPTE240M, FULL),
 *                                                  NVCV_CSS_422R, NVCV_BLOCK_LINEAR, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0,
 *                                                  NVCV_ALPHA_ASSOCIATED, 2, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8);
 * \endcode
 * If 4 extra 32-bit floating point Depth type channels are needed in the image then use
 * \ref NVCV_MAKE_YCbCr_IMAGE_EXTRA_CHANNELS_FORMAT macro as follows.
 * Note : extra channels are only supported in non-planar format.
 * Using extra channels with planar format will result in unexpected behavior
 * \code{.c}
 * NVCVImageFormat fmt = NVCV_MAKE_YCbCr_IMAGE_EXTRA_CHANNELS_FORMAT(NVCV_MAKE_COLOR_SPEC(BT601, SMPTE240M, FULL),
 *                                                  NVCV_CSS_444, NVCV_BLOCK_LINEAR, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0,
 *                                                  NVCV_ALPHA_ASSOCIATED, 4, 32, NVCV_DATA_KIND_FLOAT, NVCV_EXTRA_CHANNEL_D,
 *                                                  1, NVCV_PACKING_X8_Y8_Z8)
 * \endcode
 *
 * Fourth plane (packing3) must have at most 64bpp.
 *
 * @param[in] colorModel           \ref NVCVColorModel to be used.
 * @param[in] colorSpec            \ref NVCVColorSpec to be used.
 * @param[in] chromaSubsamp        \ref NVCVChromaSubsampling to be used.
 * @param[in] memLayout            \ref NVCVMemLayout to be used.
 * @param[in] dataKind             \ref NVCVDataKind to be used.
 * @param[in] swizzle              \ref NVCVSwizzle operation to be performed on the channels.
 * @param[in] alphaType            \ref NVCVAlphaType to be used.
 * @param[in] numExtraChannels     Number of extra channels (maximum 7).
 * @param[in] extraChannelsBPP     Bits per pixel of the extra channels.
 * @param[in] extraChannelDataKind \ref NVCVDataKind to be used.
 * @param[in] extraChannelType     \ref NVCVExtraChannel to be used.
 * @param[in] numPlanes            Number of planes this format has.
 * @param[in] packing0,packing1,packing2,packing3 Format packing of each plane.
 *                                                Exactly #numPlanes packings must be passed.
 *
 * @returns The user-defined image format.
 */
#ifdef DOXYGEN_SHOULD_SKIP_THIS
// WAR sphinx is acting up on this
//#    define NVCV_MAKE_YCbCr_IMAGE_FORMAT(colorModel, colorSpec, chromaSubsamp, memLayout, dataKind, swizzle, numPlanes, packing0, packing1, packing2, packing3)
#else
#    define NVCV_MAKE_YCbCr_IMAGE_FORMAT                (NVCVImageFormat) NVCV_DETAIL_MAKE_YCbCr_FORMAT
#    define NVCV_MAKE_YCbCr_IMAGE_EXTRA_CHANNELS_FORMAT (NVCVImageFormat) NVCV_DETAIL_MAKE_YCbCr_EXTRA_CHANNELS_FORMAT
#endif

/** Creates a user-defined color image format constant.
 *
 * Example to create a RGB planar ITU-R BT.709 full-range with SMPTE240M encoding, block-linear format.
 * \code{.c}
 * NVCVImageFormat fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(NVCV_COLOR_MODEL_RGB, NVCV_MAKE_COLOR_SPEC(BT601, SMPTE240M, FULL),
 *                                                  NVCV_MEM_LAYOUT_BL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0,
 *                                                  NVCV_ALPHA_ASSOCIATED, 2, NVCV_PACKING_X8, NVCV_PACKING_X8, NVCV_PACKING_Y8);
 * \endcode
 *
 * Similarly to create a RGB interleaved ITU-R BT.709 full-range with SMPTE240M encoding, block-linear format
 * with 3 extra channels of unsigned 16-bit 3D position.
 * \code{.c}
 * NVCVImageFormat fmt = NVCV_MAKE_COLOR_IMAGE_EXTRA_CHANNELS_FORMAT(NVCV_COLOR_MODEL_RGB, NVCV_MAKE_COLOR_SPEC(BT601, SMPTE240M, FULL),
 *                                                  NVCV_MEM_LAYOUT_BL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_ALPHA_ASSOCIATED,
 *                                                  3, 16, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_POS3D,
 *                                                  1, NVCV_PACKING_X8_Y8_Z8)
 * \endcode
 *
 * If the color model is \ref NVCV_COLOR_MODEL_YCbCr, it's assumed that the chroma subsampling is 4:4:4,
 * i.e, \ref NVCV_CSS_444.
 *
 * @param[in] colorModel           \ref NVCVColorModel to be used.
 * @param[in] colorSpec            \ref NVCVColorSpec to be used.
 * @param[in] memLayout            \ref NVCVMemLayout to be used.
 * @param[in] dataKind             \ref NVCVDataKind to be used.
 * @param[in] swizzle              \ref NVCVSwizzle operation to be performed on the channels.
 * @param[in] alphaType            \ref NVCVAlphaType to be used.
 * @param[in] numExtraChannels     Number of extra channels (maximum 7).
 * @param[in] extraChannelsBPP     Bits per pixel of the extra channels.
 * @param[in] extraChannelDataKind \ref NVCVDataKind to be used.
 * @param[in] extraChannelType     \ref NVCVExtraChannel to be used.
 * @param[in] numPlanes  Number of planes this format has.
 * @param[in] packing0,packing1,packing2,packing3 Format packing of each plane.
 *                                                Exactly #numPlanes packings must be passed.
 *                                                + Fourth plane (packing3), if passed,
 *                                                  must have at most 64bpp.
 * @param[in] numExtraChannels Number of extra channels beyond 4 channels.
 * @param[in] extraChannelDataKind \ref NVCVDataKind to be used. Data kind of extra channels
 *
 * @returns The user-defined image format.
 */
#ifdef DOXYGEN_SHOULD_SKIP_THIS
// WAR sphinx is acting up on this
//#    define NVCV_MAKE_COLOR_IMAGE_FORMAT(colorModel, colorSpec, memLayout, dataKind, swizzle, numPlanes, packing0, packing1, packing2, packing3)
#else
#    define NVCV_MAKE_COLOR_IMAGE_FORMAT                (NVCVImageFormat) NVCV_DETAIL_MAKE_COLOR_FORMAT
#    define NVCV_MAKE_COLOR_IMAGE_EXTRA_CHANNELS_FORMAT (NVCVImageFormat) NVCV_DETAIL_MAKE_COLOR_EXTRA_CHANNELS_FORMAT
#endif

/** Creates a user-defined non-color image format constant.
 *
 * Example to create 3-plane float block-linear image, 1st: 8-bit, 2nd: 16-bit, 3rd: 32-bit
 * \code{.c}
 * NVCVImageFormat fmt = NVCV_MAKE_NONCOLOR_IMAGE_FORMAT(NVCV_MEM_LAYOUT_BL, NVCV_DATA_KIND_UNSIGNED,
 *                                                    3, NVCV_PACKING_X8, NVCV_PACKING_X16, NVCV_PACKING_X32);
 * \endcode
 *
 * Similarly to create a 1-plane float block-linear image with 8-bit unsigned integer data and 2 extra unspecified channels of
 * type 8-bit unsigned integer
 * \code{.c}
 * NVCVImageFormat fmt = NVCV_MAKE_NONCOLOR_IMAGE_EXTRA_CHANNELS_FORMAT(NVCV_MEM_LAYOUT_BL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000,
 *                                                    NVCV_ALPHA_ASSOCIATED, 2, 8, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_U,
 *                                                    1, NVCV_PACKING_X8)
 * \endcode
 *
 * @param[in] memLayout            \ref NVCVMemLayout to be used.
 * @param[in] dataKind             \ref NVCVDataKind to be used.
 * @param[in] swizzle              \ref NVCVSwizzle operation to be performed on the channels.
 * @param[in] alphaType            \ref NVCVAlphaType to be used.
 * @param[in] numExtraChannels     Number of extra channels (maximum 7).
 * @param[in] extraChannelsBPP     Bits per pixel of the extra channels.
 * @param[in] extraChannelDataKind \ref NVCVDataKind to be used.
 * @param[in] extraChannelType     \ref NVCVExtraChannel to be used.
 * @param[in] numPlanes Number of planes this format has.
 * @param[in] packing0,packing1,packing2,packing3 Format packing of each plane.
 *                                                Exactly #numPlanes packings must be passed.
 *                                                + Fourth plane (packing3), if passed,
 *                                                  must have at most 64bpp.
 *
 * @returns The user-defined image format.
 */
#ifdef DOXYGEN_SHOULD_SKIP_THIS
// WAR sphinx is acting up on this
//#    define NVCV_MAKE_NONCOLOR_IMAGE_FORMAT(memLayout, dataKind, swizzle, numPlanes, packing0, packing1, packing2, packing3)
#else
#    define NVCV_MAKE_NONCOLOR_IMAGE_FORMAT (NVCVImageFormat) NVCV_DETAIL_MAKE_NONCOLOR_FORMAT
#    define NVCV_MAKE_NONCOLOR_IMAGE_EXTRA_CHANNELS_FORMAT \
        (NVCVImageFormat) NVCV_DETAIL_MAKE_NONCOLOR_EXTRA_CHANNELS_FORMAT
#endif

/** Creates a user-defined raw (Bayer pattern) image format constant.
 *
 * Example to create a RGGB Bayer pattern format:
 * \code{.c}
 * NVCVImageFormat fmt = NVCV_MAKE_RAW_IMAGE_FORMAT(NVCV_RAW_BAYER_RGGB, NVCV_MEM_LAYOUT_BL,
 *                                                NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_ALPHA_ASSOCIATED,
 *                                                1, NVCV_PACKING_X8);
 * \endcode
 * For the same format with 3 extra unspecified channels of datatype 16-bit signed, the following macro should be used
 * \code{.c}
 * NVCVImageFormat fmt = NVCV_MAKE_RAW_IMAGE_EXTRA_CHANNELS_FORMAT(NVCV_RAW_BAYER_RGGB, NVCV_MEM_LAYOUT_BL,
 *                                                NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_ALPHA_ASSOCIATED,
 *                                                3, 16, NVCV_DATA_KIND_SIGNED, NVCV_EXTRA_CHANNEL_U,
 *                                                1, NVCV_PACKING_X8);
 * \endcode
 *
 * @param[in] rawPattern           \ref NVCVRawPattern to be used.
 * @param[in] memLayout            \ref NVCVMemLayout to be used.
 * @param[in] dataKind             \ref NVCVDataKind to be used.
 * @param[in] swizzle              \ref NVCVSwizzle operation to be performed on the channels.
 * @param[in] alphaType            \ref NVCVAlphaType to be used.
 * @param[in] numExtraChannels     Number of extra channels (maximum 7).
 * @param[in] extraChannelsBPP     Bits per pixel of the extra channels.
 * @param[in] extraChannelDataKind \ref NVCVDataKind to be used.
 * @param[in] extraChannelType     \ref NVCVExtraChannel to be used.
 * @param[in] numPlanes  Number of planes this format has.
 * @param[in] packing    Format packing of image plane.
 *
 * @returns The user-defined image format.
 */
#ifdef DOXYGEN_SHOULD_SKIP_THIS
// WAR sphinx is acting up on this
//#    define NVCV_MAKE_RAW_IMAGE_FORMAT(rawPattern, memLayout, dataKind, numPlanes, swizzle, packing)
#else
#    define NVCV_MAKE_RAW_IMAGE_FORMAT                (NVCVImageFormat) NVCV_DETAIL_MAKE_RAW_FORMAT
#    define NVCV_MAKE_RAW_IMAGE_EXTRA_CHANNELS_FORMAT (NVCVImageFormat) NVCV_DETAIL_MAKE_RAW_EXTRA_CHANNELS_FORMAT
#endif

/** Creates a user-defined YCbCr color image format.
 *
 * When the pre-defined image formats aren't enough, user-defined image formats can be created.
 * @warning It's not guaranteed that algorithms will work correctly with use-defined image formats. It's recommended
 * to check if the results are correct prior deploying the solution in a production environment.
 *
 * Fourth plane (packing3) must have at most 64bpp.
 *
 * @param[out] outFormat The created image format.
 *                       + Cannot be NULL.
 * @param[in] colorSpec \ref NVCVColorSpec to be used.
 * @param[in] chromaSub \ref NVCVChromaSubsampling to be used.
 * @param[in] memLayout \ref NVCVMemLayout to be used.
 * @param[in] dataKind  \ref NVCVDataKind to be used.
 * @param[in] swizzle   \ref NVCVSwizzle operation to be performed on the channels.
 * @param[in] packing0,packing1,packing2,packing3 Format packing of each plane.
 *                                                + When remaining planes aren't needed,
 *                                                  pass \ref NVCV_PACKING_0 for them.
 * @param[in] alphaType \ref NVCVAlphaType to be used. Default: NVCV_ALPHA_ASSOCIATED.
 * @param[in] exChannelInfo \ref NVCVExtraChannelInfo to be used. Default: NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakeYCbCrImageFormat(NVCVImageFormat *outFormat, NVCVColorSpec colorSpec,
                                                NVCVChromaSubsampling chromaSub, NVCVMemLayout memLayout,
                                                NVCVDataKind dataKind, NVCVSwizzle swizzle, NVCVPacking packing0,
                                                NVCVPacking packing1, NVCVPacking packing2, NVCVPacking packing3,
                                                NVCVAlphaType alphaType, const NVCVExtraChannelInfo *exChannelInfo);

/** Creates a user-defined color image format.
 *
 * When the pre-defined image formats aren't enough, user-defined image formats can be created.
 * @warning It's not guaranteed that algorithms will work correctly with use-defined image formats. It's recommended
 * to check if the results are correct prior deploying the solution in a production environment.
 *
 * If the color model is \ref NVCV_COLOR_MODEL_YCbCr, it's assumed that the chroma subsampling is 4:4:4,
 * i.e, \ref NVCV_CSS_444.
 *
 * @param[out] outFormat The created image format.
 *                       + Cannot be NULL.
 * @param[in] colorModel \ref NVCVColorModel to be used.
 * @param[in] colorSpec  \ref NVCVColorSpec to be used.
 * @param[in] memLayout  \ref NVCVMemLayout to be used.
 * @param[in] dataKind   \ref NVCVDataKind to be used.
 * @param[in] swizzle    \ref NVCVSwizzle operation to be performed on the channels.
 * @param[in] packing0,packing1,packing2,packing3 Format packing of each plane.
 *                                                + When remaining planes aren't needed, pass \ref NVCV_PACKING_0 for them.
 *                                                + Fourth plane (packing3) must have at most 64bpp.
 * @param[in] alphaType \ref NVCVAlphaType to be used. Default: NVCV_ALPHA_ASSOCIATED.
 * @param[in] exChannelInfo \ref NVCVExtraChannelInfo to be used. Default: NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakeColorImageFormat(NVCVImageFormat *outFormat, NVCVColorModel colorModel,
                                                NVCVColorSpec colorSpec, NVCVMemLayout memLayout, NVCVDataKind dataKind,
                                                NVCVSwizzle swizzle, NVCVPacking packing0, NVCVPacking packing1,
                                                NVCVPacking packing2, NVCVPacking packing3, NVCVAlphaType alphaType,
                                                const NVCVExtraChannelInfo *exChannelInfo);

/** Creates a user-defined non-color image format.
 *
 * When the pre-defined non-color image formats aren't enough, it is possible to define new ones.
 *
 * @warning It's not guaranteed that algorithms will work correctly with use-defined image formats. It's recommended
 * to check if the results are correct prior deploying the solution in a production environment.
 *
 * @param[out] outFormat The created image format.
 *                       + Cannot be NULL.
 * @param[in] memLayout \ref NVCVMemLayout to be used.
 * @param[in] dataKind  \ref NVCVDataKind to be used.
 * @param[in] swizzle   \ref NVCVSwizzle operation to be performed on the channels.
 * @param[in] packing0,packing1,packing2,packing3 Format packing of each plane.
 *                                                + When remaining planes aren't needed, pass \ref NVCV_PACKING_0 for them.
 *                                                + Fourth plane (packing3) must have at most 64bpp.
 * @param[in] alphaType \ref NVCVAlphaType to be used. Default: NVCV_ALPHA_ASSOCIATED.
 * @param[in] exChannelInfo \ref NVCVExtraChannelInfo to be used. Default: NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakeNonColorImageFormat(NVCVImageFormat *outFormat, NVCVMemLayout memLayout,
                                                   NVCVDataKind dataKind, NVCVSwizzle swizzle, NVCVPacking packing0,
                                                   NVCVPacking packing1, NVCVPacking packing2, NVCVPacking packing3,
                                                   NVCVAlphaType alphaType, const NVCVExtraChannelInfo *exChannelInfo);

/** Creates a user-defined raw image format.
 *
 * When the pre-defined raw image formats aren't enough, it is possible to define new ones.
 * @warning It's not guaranteed that algorithms will work correctly with use-defined image formats. It's recommended
 * to check if the results are correct prior deploying the solution in a production environment.
 *
 * @param[out] outFormat The created image format.
 *                       + Cannot be NULL.
 * @param[in] rawPattern \ref NVCVRawPattern to be used.
 * @param[in] memLayout  \ref NVCVMemLayout to be used.
 * @param[in] dataKind   \ref NVCVDataKind to be used.
 * @param[in] swizzle    \ref NVCVSwizzle operation to be performed on the channels.
 * @param[in] packing0,packing1,packing2,packing3 Format packing of each plane.
 *                                                + When remaining planes aren't needed, pass \ref NVCV_PACKING_0 for them.
 *                                                + Fourth plane (packing3) must have at most 64bpp.
 * @param[in] alphaType \ref NVCVAlphaType to be used. Default: NVCV_ALPHA_ASSOCIATED.
 * @param[in] exChannelInfo \ref NVCVExtraChannelInfo to be used. Default: NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakeRawImageFormat(NVCVImageFormat *outFormat, NVCVRawPattern rawPattern,
                                              NVCVMemLayout memLayout, NVCVDataKind dataKind, NVCVSwizzle swizzle,
                                              NVCVPacking packing0, NVCVPacking packing1, NVCVPacking packing2,
                                              NVCVPacking packing3, NVCVAlphaType alphaType,
                                              const NVCVExtraChannelInfo *exChannelInfo);

/** Creates a image format from a FourCC code.
 *
 * See https://www.fourcc.org for more information about FourCC.
 *
 * @param[out] outFormat The image format corresponding to the FourCC code.
 *                       + Cannot be NULL.
 * @param[in] fourcc FourCC code.
 * @param[in] colorSpec \ref NVCVColorSpec to be used.
 * @param[in] memLayout \ref NVCVMemLayout to be used.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakeImageFormatFromFourCC(NVCVImageFormat *outFormat, uint32_t fourcc,
                                                     NVCVColorSpec colorSpec, NVCVMemLayout memLayout);

/** Returns the FourCC code corresponding to an image format.
 *
 * @param[in] fmt Image format to be queried.
 * @param[out] outFourCC The FourCC code corresponding to the image format.
 *                       + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatToFourCC(NVCVImageFormat fmt, uint32_t *outFourCC);

/** Get the packing for a given plane of an image format.
 *
 * @param[in] fmt   Image format to be queried.
 * @param[in] plane Which plane whose packing must be returned.
 *                  If plane doesn't exist, outPacking will be set to \ref NVCV_PACKING_0.
 *
 * @param[out] outPacking The plane's format packing.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetPlanePacking(NVCVImageFormat fmt, int32_t plane, NVCVPacking *outPacking);

/** Get the plane width of an image with the given image format and width.
 *
 * @param[in] fmt                Image format to be queried.
 * @param[in] imgWidth,imgHeight Size of the image.
 *                               + Must be >= 1.
 * @param[in] plane              Image plane to be queried.
 *                               + Must be >= 0 and < the number of planes in the image format.
 *
 * @param[out] outPlaneWidth,outPlaneHeight The size of the image plane.
 *                                          Only the non-NULL parameter will be returned.
 *                                          + Cannot be both NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetPlaneSize(NVCVImageFormat fmt, int32_t plane, int32_t imgWidth,
                                                   int32_t imgHeight, int32_t *outPlaneWidth, int32_t *outPlaneHeight);

/** Replaces the swizzle and packing of an existing image format.
 *
 * The number of channels represented by the swizzle must be equal to the sum of the number of channels
 * represented by the packings. For instance, XYZ1 -> X8,X8Y8 is a valid combination with 3 channels.
 * XYZW -> X8,X8Y8 isn't as swizzle has 4 channels, and X8,X8Y8 represents in total 3 channels.
 *
 * @param[inout] fmt     Image format to have its packing replaced.
 *                       + Cannot be NULL.
 * @param[in] swizzle The new swizzle.
 * @param[in] packing0,packing1,packing2,packing3 New packing per plane.
 *                                                + If replacing the fourth packing (packing3),
 *                                                  the packing's bits per pixel must be at most 64.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatSetSwizzleAndPacking(NVCVImageFormat *fmt, NVCVSwizzle swizzle,
                                                           NVCVPacking packing0, NVCVPacking packing1,
                                                           NVCVPacking packing2, NVCVPacking packing3);

/** Get the image format's plane bits per pixel count.
 *
 * @param[in] fmt   Image format to be queried.
 * @param[in] plane Which plane is to be queried.
 *
 * @param[out] outBPP The number of bits per pixel the given format plane has.
 *                    If \p plane doesn't exist (even if >= 4), it returns 0.
 *                    + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetPlaneBitsPerPixel(NVCVImageFormat fmt, int32_t plane, int32_t *outBPP);

/** Set the image format's data type.
 *
 * @param[inout] fmt      Image format have its data type replaced.
 *                        + Cannot be NULL.
 * @param[in] dataKind The new data type.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatSetDataKind(NVCVImageFormat *fmt, NVCVDataKind dataKind);

/** Get the image format's data type.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] outDataKind The image format's data type.
 *                         + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetDataKind(NVCVImageFormat fmt, NVCVDataKind *outDataKind);

/** Get the image format's channel swizzle operation.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] outSwizzle The image format's swizzle operation.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetSwizzle(NVCVImageFormat fmt, NVCVSwizzle *outSwizzle);

/** Get the swizzle operation of the given image format's plane.
 *
 * @param[in] fmt   Image format to be queried.
 *                  + Image format should have less packing channels and swizzle channels.
 * @param[in] plane Plane to be queried.
 *                  + Valid values range from 0 (first) to 3 (fourth and last) plane.
 *
 * @param[out] outPlaneSwizzle The channel swizzle operation performed in the given plane.
 *                             + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetPlaneSwizzle(NVCVImageFormat fmt, int32_t plane, NVCVSwizzle *outPlaneSwizzle);

/** Set the image format's memory layout.
 *
 * @param[inout] fmt       Image format have its memory layout replaced.
 *                         + Cannot be NULL.
 * @param[in] memLayout The new memory layout.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatSetMemLayout(NVCVImageFormat *fmt, NVCVMemLayout memLayout);

/** Get the image format's memory layout.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] outMemLayout The image format's memory layout.
 *                          + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetMemLayout(NVCVImageFormat fmt, NVCVMemLayout *outMemLayout);

/** Set the image format's color standard.
 *
 * @param[inout] fmt       Image format have its color spec replaced.
 *                         + Cannot be NULL.
 *                         + Format's color model must represent image coding systems, such as RGB, Y'CrCb, HSV, etc.
 * @param[in] colorSpec The new color standard.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatSetColorSpec(NVCVImageFormat *fmt, NVCVColorSpec colorSpec);

/** Get the image format's color standard.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] outColorSpec The image format's color standard.
 *                          + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetColorSpec(NVCVImageFormat fmt, NVCVColorSpec *outColorSpec);

/** Get the image format's color model.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] outColorModel The image format's color model.
 *                           + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetColorModel(NVCVImageFormat fmt, NVCVColorModel *outColorModel);

/** Set the image format's chroma subsampling type.
 *
 *
 * @param[inout] fmt Image format have its chroma subsampling type replaced.
 *                + It's only applicable if format has YCbCr color model.
 *                + Cannot be NULL.
 * @param[in] css The new chroma subsampling type.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatSetChromaSubsampling(NVCVImageFormat *fmt, NVCVChromaSubsampling css);

/** Get the image format's chroma subsampling type.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] outCSS The image format's chroma subsampling type.
 *                    + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetChromaSubsampling(NVCVImageFormat fmt, NVCVChromaSubsampling *outCSS);

/** Get the number of channels in a plane of an image format.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[in] plane Plane to be queried.
 *                  + Valid values range from 0 (first) to 3 (fourth and last) plane.
 *
 * @param[out] outPlaneNumChannels Number of channels in the given plane.
 *                                 + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetPlaneNumChannels(NVCVImageFormat fmt, int32_t plane,
                                                          int32_t *outPlaneNumChannels);

/** Get the number of planes of an image format.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] outNumPlanes Number of planes defined by the given image format.
 *                          + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetNumPlanes(NVCVImageFormat fmt, int32_t *outNumPlanes);

/** Get the total number of channels of an image format.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] outNumChannels The sum of all channel counts in all planes.
 *                            + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetNumChannels(NVCVImageFormat fmt, int32_t *outNumChannels);

/** Get the image format's bit size for each channel.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] bits Pointer to an int32_t array with 4 elements where output will be stored.
 *                  + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetBitsPerChannel(NVCVImageFormat fmt, int32_t *bits);

/** Get the data type of image format's plane.
 *
 * @param[in] fmt   Image format to be queried.
 * @param[in] plane Plane to be queried.
 *                  + Valid values range from 0 (first) to 3 (fourth and last) plane.
 *                  + Cannot be NULL.
 *
 * @param[out] outPixType The data type of the given plane.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetPlaneDataType(NVCVImageFormat fmt, int32_t plane, NVCVDataType *outPixType);

/** Get the plane format of an image format.
 *
 * @param[in] fmt   Image format to be queried.
 * @param[in] plane Plane to be queried.
 *                  + Valid values range from 0 (first) to 3 (fourth and last) plane.
 *
 * @param[out] outFormat The image format of the given plane.
 *                       + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetPlaneFormat(NVCVImageFormat fmt, int32_t plane, NVCVImageFormat *outFormat);

/** Get the stride of the pixel in the given plane of an image format.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[in] plane Plane to be queried.
 *                  + Valid values range from 0 (first) to 3 (fourth and last) plane.
 *
 * @param[out] outStrideBytes Stride (size) of the pixel in bytes.
 *                            + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetPlanePixelStrideBytes(NVCVImageFormat fmt, int32_t plane,
                                                               int32_t *outStrideBytes);

/** Constructs an image format given the format of each plane.
 *
 * @param[out] outFormat The image format whose planes have the given formats.
 *
 * @param[in] plane0,plane1,plane2,plane3 Image format of each plane.
 *                                        + When plane doesn't exist, pass NVCV_IMAGE_FORMAT_NONE,
 *                                        + All plane formats must have only 1 plane.
 *                                        + First plane must have a valid packing.
 *                                        + Total number of channels must be at most 4.
 *                                        + Color spec, mem layout and data type of all planes must be the same.
 *                                        + Only one kind of chroma subsampling is allowed.
 *                                        + At least one channel is allowed.
 *                                        + All planes after the first invalid one must be invalid.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakeImageFormatFromPlanes(NVCVImageFormat *outFormat, NVCVImageFormat plane0,
                                                     NVCVImageFormat plane1, NVCVImageFormat plane2,
                                                     NVCVImageFormat plane3);

/** Returns a string representation of the image format.
 *
 * @param[in] fmt Image format whose name is to be returned.
 *
 * @returns The string representation of the image format.
 *          Returned string is valid until next call of this function from the same calling thread.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvImageFormatGetName(NVCVImageFormat fmt);

/** Returns the raw color pattern of the image format.
 *
 * @param[in] fmt Image format to be queried.
 *                + Its color model must be \ref NVCV_COLOR_MODEL_RAW .
 *
 * @param[out] outRawPattern The raw pattern of given raw image format.
 *                           + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetRawPattern(NVCVImageFormat fmt, NVCVRawPattern *outRawPattern);

/** Sets the raw color pattern of the image format.
 *
 * @param[inout] fmt        Image format to be updated.
 *                          + Its color model must be \ref NVCV_COLOR_MODEL_RAW.
 *                          + Cannot be NULL.
 * @param[in] rawPattern The new raw pattern.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS              Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatSetRawPattern(NVCVImageFormat *fmt, NVCVRawPattern rawPattern);

/** Returns whether the image formats have the same data layout.
 *
 * Data layout referts to how pixels are laid out in memory. It doesn't take into account
 * the format's color information.
 *
 * The following characteristics are taken into account:
 * - memory layout (block linear, pitch linear, ...)
 * - data type (signed, unsigned, float, ...)
 * - Swizzle (except for 1/0 in 4th channel)
 * - number of planes
 * - packings (X8_Y8, X16, ...)
 * - chroma subsampling (4:4:4, 4:2:0, ...)
 *
 * @param[in] a, b Image formats to be compared.
 * @param[out] outBool != 0 if image formats compares equal with respect to how pixels are laid out in memory,
 *                     0 otherwise.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatHasSameDataLayout(NVCVImageFormat a, NVCVImageFormat b, int8_t *outBool);

/** Get the image format's alpha Channel type.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] outAlphaChannelType The image format's alpha channel type.
 *                         + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetAlphaType(NVCVImageFormat fmt, NVCVAlphaType *outAlphaChannelType);

/** Set the image format's alpha channel type.
 *
 * @param[inout] fmt      Image format have its data type replaced.
 *                        + Cannot be NULL.
 * @param[in] alphaChannelType The new alphaChannel type.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatSetAlphaType(NVCVImageFormat *fmt, NVCVAlphaType alphaChannelType);

/** Get the image format's extra channel information.
 *
 * @param[in] fmt Image format to be queried.
 *
 * @param[out] outExChannelInfo The image format's extra channel information.
 *                         + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatGetExtraChannelInfo(NVCVImageFormat fmt, NVCVExtraChannelInfo *outExChannelInfo);

/** Set the image format's extra channel information.
 *
 * @param[inout] fmt      Image format have its data type replaced.
 *                        + Cannot be NULL.
 * @param[in] exChannelInfo The new extra channel info.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageFormatSetExtraChannelInfo(NVCVImageFormat            *fmt,
                                                          const NVCVExtraChannelInfo *exChannelInfo);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* NVCV_FORMAT_IMAGEFORMAT_H */

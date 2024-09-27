/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TEST_COMMON_CVT_COLOR_UTILS_HPP
#define NVCV_TEST_COMMON_CVT_COLOR_UTILS_HPP

#include <stdint.h>

#include <vector>

// clang-format off


template<typename T>
void changeAlpha(std::vector<T> &dst, const std::vector<T> &src, size_t numPixels, bool srcRGBA, bool dstRGBA);

template<typename T>
void convertRGBtoBGR(std::vector<T> &dst, const std::vector<T> &src, size_t numPixels, bool srcRGBA, bool dstRGBA);

template<typename T>
void convertRGBtoGray(std::vector<T> &dst, const std::vector<T> &src, size_t numPixels, bool rgba, bool bgr);

template<typename T>
void convertGrayToRGB(std::vector<T> &dst, const std::vector<T> &src, size_t numPixels, bool rgba);

template<typename T, bool FullRange>
void convertRGBtoHSV(std::vector<T> &dst, const std::vector<T> &src, size_t numPixels, bool rgba, bool bgr);

template<typename T, bool FullRange>
void convertHSVtoRGB(std::vector<T> &dst, const std::vector<T> &src, size_t numPixels, bool rgba, bool bgr);

template<typename T>
void convertRGBtoYUV_PAL(std::vector<T> &dst, const std::vector<T> &src, size_t numPixels, bool rgba, bool bgr);

template<typename T>
void convertYUVtoRGB_PAL(std::vector<T> &dst, const std::vector<T> &src, size_t numPixels, bool rgba, bool bgr);

template<typename T>
void convertRGBtoYUV_420(std::vector<T> &dst, const std::vector<T> &src, uint wdth, uint hght, uint numImgs,
                         bool rgba, bool bgr, bool yvu);

template<typename T>
void convertYUVtoRGB_420(std::vector<T> &dst, const std::vector<T> &src, uint wdth, uint hght, uint numImgs,
                         bool rgba, bool bgr, bool yvu);

template<typename T>
void convertYUVtoGray_420(std::vector<T> &dst, const std::vector<T> &src, uint wdth, uint hght, uint numImgs);

template<typename T>
void convertRGBtoNV12(std::vector<T> &dst, const std::vector<T> &src, uint wdth, uint hght, uint num,
                      bool rgba, bool bgr, bool yvu);

template<typename T>
void convertNV12toRGB(std::vector<T> &dst, const std::vector<T> &src, uint wdth, uint hght, uint num,
                      bool rgba, bool bgr, bool yvu);

template<typename T, bool LumaFirst>
void convertYUVtoRGB_422(std::vector<T> &dst, const std::vector<T> &src, uint wdth, uint hght, uint numImgs,
                         bool rgba, bool bgr, bool yvu);

template<typename T, bool LumaFirst>
void convertYUVtoGray_422(std::vector<T> &dst, const std::vector<T> &src, size_t numPixels);

#endif // NVCV_TEST_COMMON_CVT_COLOR_UTILS_HPP

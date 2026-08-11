/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TEST_COMMON_CONV_UTILS_HPP
#define NVCV_TEST_COMMON_CONV_UTILS_HPP

#include <common/BorderUtils.hpp>
#include <common/ValueTests.hpp>
#include <cuda_runtime.h> // for long3, etc.
#include <cvcuda/Types.h>
#include <nvcv/BorderType.h>
#include <nvcv/ImageFormat.hpp> // for ImageFormat, etc.
#include <nvcv/Size.hpp>        // for Size2D, etc.

#include <cstdint> // for uint8_t, etc.
#include <vector>  // for std::vector, etc.

namespace nvcv::test {

void Convolve(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
              const long3 &srcStrides, const int3 &shape, const ImageFormat &format, const std::vector<float> &kernel,
              const Size2D &kernelSize, int2 &kernelAnchor, const NVCVBorderType &borderMode,
              const float4 &borderValue);

void Morph(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
           const long3 &srcStrides, const int3 &shape, const ImageFormat &format, const Size2D &kernelSize,
           int2 &kernelAnchor, const NVCVBorderType &borderMode, NVCVMorphologyType type);

std::vector<float> ComputeMeanKernel(nvcv::Size2D kernelSize);

std::vector<float> ComputeGaussianKernel(nvcv::Size2D kernelSize, double2 sigma);

inline auto ConvolutionNegativeParams()
{
    ValueList<nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType> params{
        {  nvcv::FMT_RGB8,  nvcv::FMT_RGB8p, NVCV_BORDER_CONSTANT},
        { nvcv::FMT_RGB8p,  nvcv::FMT_RGB8p, NVCV_BORDER_CONSTANT},
        {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, NVCV_BORDER_CONSTANT},
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(nvcv::FMT_RGB8, nvcv::FMT_RGB8, static_cast<NVCVBorderType>(255));
#endif
    return params;
}

inline auto PlanarConvolutionNegativeParams()
{
    ValueList<nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType> params{
        {  nvcv::FMT_RGB8,  nvcv::FMT_RGB8p, NVCV_BORDER_CONSTANT},
        { nvcv::FMT_RGB8p,   nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT},
        {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, NVCV_BORDER_CONSTANT},
        {  nvcv::FMT_2F32,   nvcv::FMT_2F32, NVCV_BORDER_CONSTANT},
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(nvcv::FMT_RGB8, nvcv::FMT_RGB8, static_cast<NVCVBorderType>(255));
#endif
    return params;
}

inline auto ConvolutionVarShapeNegativeParams()
{
    ValueList<nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType, int, int> params{
        {  nvcv::FMT_RGB8,  nvcv::FMT_RGB8p, NVCV_BORDER_CONSTANT, 3,  3},
        { nvcv::FMT_RGB8p,  nvcv::FMT_RGB8p, NVCV_BORDER_CONSTANT, 3,  3},
        {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, NVCV_BORDER_CONSTANT, 3,  3},
        {  nvcv::FMT_RGB8,   nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, 3, -1},
        {  nvcv::FMT_RGB8,   nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, 5,  3},
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(nvcv::FMT_RGB8, nvcv::FMT_RGB8, static_cast<NVCVBorderType>(255), 3, 3);
#endif
    return params;
}

// Var-shape negative cases for filter ops that DO support planar layout (AverageBlur/Gaussian/
// Laplacian/MedianBlur/...). Same as ConvolutionVarShapeNegativeParams except a planar<->interleaved
// layout MISMATCH (RGB8p in / RGB8 out) is rejected, while planar<->planar is now valid and so is not
// a negative case. Shared so the planar-capable filter ops do not each re-declare this matrix.
inline auto PlanarFilterVarShapeNegativeParams()
{
    ValueList<nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType, int, int> params{
        {  nvcv::FMT_RGB8,  nvcv::FMT_RGB8p, NVCV_BORDER_CONSTANT, 3,  3},
        { nvcv::FMT_RGB8p,   nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, 3,  3},
        {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, NVCV_BORDER_CONSTANT, 3,  3},
        {  nvcv::FMT_RGB8,   nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, 3, -1},
        {  nvcv::FMT_RGB8,   nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, 5,  3},
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(nvcv::FMT_RGB8, nvcv::FMT_RGB8, static_cast<NVCVBorderType>(255), 3, 3);
#endif
    return params;
}

namespace detail {

template<typename T>
inline const T &ValueAt(const std::vector<uint8_t> &vec, long3 pitches, int b, int y, int x)
{
    return *reinterpret_cast<const T *>(&vec[b * pitches.x + y * pitches.y + x * pitches.z]);
}

template<typename T>
inline T &ValueAt(std::vector<uint8_t> &vec, long3 pitches, int b, int y, int x)
{
    return *reinterpret_cast<T *>(&vec[b * pitches.x + y * pitches.y + x * pitches.z]);
}

} // namespace detail

} // namespace nvcv::test

#endif // NVCV_TEST_COMMON_CONV_UTILS_HPP

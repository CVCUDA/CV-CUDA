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

/**
 * @file JpegDistortionMath.hpp
 *
 * @brief Shared host/device math for the JpegCompressionDistortion operator.
 *
 * Ported from NVIDIA DALI (Apache-2.0, Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES):
 * dali/kernels/imgproc/jpeg/dct_8x8_gpu.cuh (fixed-rotation 8-point DCT, itself derived from the
 * NVIDIA CUDA dct8x8 sample), dali/kernels/imgproc/jpeg/jpeg_distortion_gpu_kernel.h (Annex-K
 * quantization tables and libjpeg quality scaling) and
 * dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h (full-range JFIF YCbCr).
 *
 * Every multiply-add is written as an explicit fmaf() in a fixed order, and every remaining
 * operation (add/sub, lone multiply, correctly-rounded reciprocal, roundf, round-to-nearest-even
 * saturating cast) is correctly rounded and identical on host and device. This makes the pipeline
 * immune to compiler contraction (nvcc -fmad, gcc -ffp-contract), so an independent CPU evaluation
 * of the same canonical operation order reproduces the GPU results bit-exactly. This canonical
 * order — not bit-parity with DALI's own (contraction-dependent) GPU output — is the operator's
 * defined semantics.
 */

#ifndef CVCUDA_PRIV_JPEG_DISTORTION_MATH_HPP
#define CVCUDA_PRIV_JPEG_DISTORTION_MATH_HPP

#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/SaturateCast.hpp>

#include <cmath>
#include <cstdint>

namespace cvcuda::priv::jpeg {

// Fixed-rotation DCT constants (sqrt(2)*cos(k*pi/16)) and the 1/sqrt(8) per-pass normalization.
constexpr float kDctA    = 1.387039845322148f;  // sqrt(2) * cos(    pi / 16)
constexpr float kDctB    = 1.306562964876377f;  // sqrt(2) * cos(    pi /  8)
constexpr float kDctC    = 1.175875602419359f;  // sqrt(2) * cos(3 * pi / 16)
constexpr float kDctD    = 0.785694958387102f;  // sqrt(2) * cos(5 * pi / 16)
constexpr float kDctE    = 0.541196100146197f;  // sqrt(2) * cos(3 * pi /  8)
constexpr float kDctF    = 0.275899379282943f;  // sqrt(2) * cos(7 * pi / 16)
constexpr float kDctNorm = 0.3535533905932737f; // 1 / sqrt(8)

// Base quantization tables suggested in Annex K of the JPEG standard (row-major 8x8). The kernels
// index them with runtime thread indices, so under CUDA compilation they must be device-resident
// (__constant__); a plain constexpr host array is not addressable from device code. Host
// compilation sees constexpr arrays. Note: __constant__ definitions have external linkage, so this
// header must stay included by a single CUDA translation unit (OpJpegCompressionDistortion.cu).
#ifdef __CUDACC__
#    define CVCUDA_JPEG_TABLE_STORAGE __constant__ const
#else
#    define CVCUDA_JPEG_TABLE_STORAGE constexpr
#endif

CVCUDA_JPEG_TABLE_STORAGE uint8_t kLumaQuantBase[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,  //
    12, 12, 14, 19, 26,  58,  60,  55,  //
    14, 13, 16, 24, 40,  57,  69,  56,  //
    14, 17, 22, 29, 51,  87,  80,  62,  //
    18, 22, 37, 56, 68,  109, 103, 77,  //
    24, 35, 55, 64, 81,  104, 113, 92,  //
    49, 64, 78, 87, 103, 121, 120, 101, //
    72, 92, 95, 98, 112, 100, 103, 99,  //
};

CVCUDA_JPEG_TABLE_STORAGE uint8_t kChromaQuantBase[64] = {
    17, 18, 24, 47, 99, 99, 99, 99, //
    18, 21, 26, 66, 99, 99, 99, 99, //
    24, 26, 56, 99, 99, 99, 99, 99, //
    47, 66, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
};

// Round-to-nearest-even saturating cast to uint8 (device: cvt.rni.sat.u8.f32).
inline __host__ __device__ uint8_t SatCastU8(float v)
{
    return nvcv::cuda::SaturateCast<uint8_t>(v);
}

// Correctly-rounded reciprocal: __frcp_rn and IEEE round-to-nearest 1/x are the same value.
inline __host__ __device__ float Rcp(float v)
{
#ifdef __CUDA_ARCH__
    return __frcp_rn(v);
#else
    return 1.0f / v;
#endif
}

// Canonical fused dot products: each product-sum is a single-rounded fmaf chain evaluated from the
// first coefficient outward, with the last term a lone (single-rounded) multiply.
inline __host__ __device__ float Dot2(float c0, float v0, float c1, float v1)
{
    return fmaf(c0, v0, c1 * v1);
}

inline __host__ __device__ float Dot3(float c0, float v0, float c1, float v1, float c2, float v2)
{
    return fmaf(c0, v0, Dot2(c1, v1, c2, v2));
}

inline __host__ __device__ float Dot4(float c0, float v0, float c1, float v1, float c2, float v2, float c3, float v3)
{
    return fmaf(c0, v0, Dot3(c1, v1, c2, v2, c3, v3));
}

inline __host__ __device__ float Dot3Bias(float c0, float v0, float c1, float v1, float c2, float v2, float bias)
{
    return fmaf(c0, v0, fmaf(c1, v1, fmaf(c2, v2, bias)));
}

// libjpeg quality scaling (jpeg_quality_scaling pre-divided by 100); quality is clamped to [1, 100].
inline __host__ __device__ float QuantScale(int quality)
{
    const int q = quality < 1 ? 1 : (quality > 100 ? 100 : quality);
    return q < 50 ? 50.0f / static_cast<float>(q) : 2.0f - static_cast<float>(2 * q) / 100.0f;
}

// Scaled table entry, rounded half away from zero and clamped so every quantization step is in
// [1, 255]. DALI builds these tables on the host, where its saturating cast rounds with
// std::roundf; libjpeg's integer mapping agrees. Round-half-even here would diverge at q=75,
// whose exact 0.5 scale puts every odd base entry on a tie.
inline __host__ __device__ float QuantTableEntry(float scale, uint8_t base)
{
    const float entry = roundf(scale * static_cast<float>(base));
    return entry < 1.0f ? 1.0f : (entry > 255.0f ? 255.0f : entry);
}

// Quantization round trip: Q * round(value / Q) with the division as a reciprocal-multiply.
inline __host__ __device__ float Quantize(float value, float q)
{
    return q * roundf(value * Rcp(q));
}

// Forward 8-point DCT over 8 strided elements (rows: Stride=1, columns: Stride=9 with the padded
// shared-memory layout). Two passes (rows then columns) give the orthonormal 2-D DCT.
template<int Stride>
inline __host__ __device__ void FwdDct8(float *data)
{
    float x0 = data[0 * Stride];
    float x1 = data[1 * Stride];
    float x2 = data[2 * Stride];
    float x3 = data[3 * Stride];
    float x4 = data[4 * Stride];
    float x5 = data[5 * Stride];
    float x6 = data[6 * Stride];
    float x7 = data[7 * Stride];

    const float tmp0 = x0 + x7;
    const float tmp1 = x1 + x6;
    const float tmp2 = x2 + x5;
    const float tmp3 = x3 + x4;

    const float tmp4 = x0 - x7;
    const float tmp5 = x6 - x1;
    const float tmp6 = x2 - x5;
    const float tmp7 = x4 - x3;

    const float tmp8  = tmp0 + tmp3;
    const float tmp9  = tmp0 - tmp3;
    const float tmp10 = tmp1 + tmp2;
    const float tmp11 = tmp1 - tmp2;

    x0 = kDctNorm * (tmp8 + tmp10);
    x2 = kDctNorm * Dot2(kDctB, tmp9, kDctE, tmp11);
    x4 = kDctNorm * (tmp8 - tmp10);
    x6 = kDctNorm * Dot2(kDctE, tmp9, -kDctB, tmp11);

    x1 = kDctNorm * Dot4(kDctA, tmp4, -kDctC, tmp5, kDctD, tmp6, -kDctF, tmp7);
    x3 = kDctNorm * Dot4(kDctC, tmp4, kDctF, tmp5, -kDctA, tmp6, kDctD, tmp7);
    x5 = kDctNorm * Dot4(kDctD, tmp4, kDctA, tmp5, kDctF, tmp6, -kDctC, tmp7);
    x7 = kDctNorm * Dot4(kDctF, tmp4, kDctD, tmp5, kDctC, tmp6, kDctA, tmp7);

    data[0 * Stride] = x0;
    data[1 * Stride] = x1;
    data[2 * Stride] = x2;
    data[3 * Stride] = x3;
    data[4 * Stride] = x4;
    data[5 * Stride] = x5;
    data[6 * Stride] = x6;
    data[7 * Stride] = x7;
}

// Inverse 8-point DCT over 8 strided elements (columns first: Stride=9, then rows: Stride=1).
template<int Stride>
inline __host__ __device__ void InvDct8(float *data)
{
    float x0 = data[0 * Stride];
    float x1 = data[1 * Stride];
    float x2 = data[2 * Stride];
    float x3 = data[3 * Stride];
    float x4 = data[4 * Stride];
    float x5 = data[5 * Stride];
    float x6 = data[6 * Stride];
    float x7 = data[7 * Stride];

    const float tmp0 = x0 + x4;
    const float tmp1 = Dot2(kDctB, x2, kDctE, x6);

    const float tmp2 = tmp0 + tmp1;
    const float tmp3 = tmp0 - tmp1;
    const float tmp4 = Dot4(kDctF, x7, kDctA, x1, kDctC, x3, kDctD, x5);
    const float tmp5 = Dot4(kDctA, x7, -kDctF, x1, kDctD, x3, -kDctC, x5);

    const float tmp6 = x0 - x4;
    const float tmp7 = Dot2(kDctE, x2, -kDctB, x6);

    const float tmp8  = tmp6 + tmp7;
    const float tmp9  = tmp6 - tmp7;
    const float tmp10 = Dot4(kDctC, x1, -kDctD, x7, -kDctF, x3, -kDctA, x5);
    const float tmp11 = Dot4(kDctD, x1, kDctC, x7, -kDctA, x3, kDctF, x5);

    x0 = kDctNorm * (tmp2 + tmp4);
    x7 = kDctNorm * (tmp2 - tmp4);
    x4 = kDctNorm * (tmp3 + tmp5);
    x3 = kDctNorm * (tmp3 - tmp5);

    x1 = kDctNorm * (tmp8 + tmp10);
    x5 = kDctNorm * (tmp9 - tmp11);
    x2 = kDctNorm * (tmp9 + tmp11);
    x6 = kDctNorm * (tmp8 - tmp10);

    data[0 * Stride] = x0;
    data[1 * Stride] = x1;
    data[2 * Stride] = x2;
    data[3 * Stride] = x3;
    data[4 * Stride] = x4;
    data[5 * Stride] = x5;
    data[6 * Stride] = x6;
    data[7 * Stride] = x7;
}

// Full-range JFIF color conversions (uint8 in/out, chroma biased by +128).

inline __host__ __device__ uint8_t RgbToY(uchar3 rgb)
{
    return SatCastU8(
        Dot3(0.299f, static_cast<float>(rgb.x), 0.587f, static_cast<float>(rgb.y), 0.114f, static_cast<float>(rgb.z)));
}

inline __host__ __device__ uint8_t RgbToCb(uchar3 rgb)
{
    return SatCastU8(Dot3Bias(-0.16873589f, static_cast<float>(rgb.x), -0.33126411f, static_cast<float>(rgb.y), 0.5f,
                              static_cast<float>(rgb.z), 128.0f));
}

inline __host__ __device__ uint8_t RgbToCr(uchar3 rgb)
{
    return SatCastU8(Dot3Bias(0.5f, static_cast<float>(rgb.x), -0.41868759f, static_cast<float>(rgb.y), -0.08131241f,
                              static_cast<float>(rgb.z), 128.0f));
}

inline __host__ __device__ uchar3 YCbCrToRgb(uint8_t y, uint8_t cb, uint8_t cr)
{
    const float ys = static_cast<float>(y);
    const float tb = static_cast<float>(cb) - 128.0f;
    const float tr = static_cast<float>(cr) - 128.0f;
    uchar3      rgb;
    rgb.x = SatCastU8(fmaf(1.402f, tr, ys));
    rgb.y = SatCastU8(fmaf(-0.714136285f, tr, fmaf(-0.344136285f, tb, ys)));
    rgb.z = SatCastU8(fmaf(1.772f, tb, ys));
    return rgb;
}

// 2x2 box average of RGB pixels: the integer channel sum is exact and the *0.25f scaling is a
// power of two, so the round-to-nearest-even cast is the only rounding step.
inline __host__ __device__ uchar3 Avg4(uchar3 p00, uchar3 p01, uchar3 p10, uchar3 p11)
{
    uchar3 avg;
    avg.x = SatCastU8(static_cast<float>(p00.x + p01.x + p10.x + p11.x) * 0.25f);
    avg.y = SatCastU8(static_cast<float>(p00.y + p01.y + p10.y + p11.y) * 0.25f);
    avg.z = SatCastU8(static_cast<float>(p00.z + p01.z + p10.z + p11.z) * 0.25f);
    return avg;
}

} // namespace cvcuda::priv::jpeg

#endif // CVCUDA_PRIV_JPEG_DISTORTION_MATH_HPP

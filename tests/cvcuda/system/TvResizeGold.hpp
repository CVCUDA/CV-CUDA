/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CVCUDA_TEST_TV_RESIZE_GOLD_HPP
#define CVCUDA_TEST_TV_RESIZE_GOLD_HPP

#include <cuda_fp16.h>
#include <cvcuda/Types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace tvgold {

// Independently derived from SPEC.md's Keys polynomial and normalized surviving taps.
// An opaque call preserves each float32 rounding boundary; noinline alone permits identity-call elimination.
#if defined(__clang__)
[[gnu::noinline, clang::optnone]]
#elif defined(__GNUC__)
[[gnu::noipa]]
#else
#    error "TvResize gold requires a GCC or Clang floating-point optimization barrier"
#endif
inline float
    F32(float x)
{
    return x;
}

inline bool Nearest(NVCVInterpolationType mode)
{
    return mode == NVCV_INTERP_NEAREST || mode == NVCV_INTERP_NEAREST_FLOOR;
}

inline int NearestIndex(int source, int destination, int index, NVCVInterpolationType mode)
{
    float scale   = F32(static_cast<float>(source) / static_cast<float>(destination));
    float pos     = mode == NVCV_INTERP_NEAREST ? F32(static_cast<float>(index) + 0.5f) : float(index);
    auto  sampled = static_cast<int64_t>(std::floor(F32(pos * scale)));
    return static_cast<int>(std::min<int64_t>(source - 1, sampled));
}

inline float Kernel(float distance, bool cubic, bool aa)
{
    float x = std::abs(distance);
    if (!cubic)
        return std::max(0.f, F32(1.f - x));
    float a = aa ? -0.5f : -0.75f;
    // Horner evaluation follows pinned ATen CUDA: q*x*x must not become q*(x*x).
    if (x < 1.f)
    {
        float quadratic = std::fma(a + 2.f, x, -(a + 3.f));
        return std::fma(F32(quadratic * x), x, 1.f);
    }
    if (x < 2.f)
    {
        if (aa)
        {
            float quadratic = std::fma(F32(x - 5.f), x, 8.f);
            return F32(std::fma(quadratic, x, -4.f) * a);
        }
        float quadratic = std::fma(a, x, -5.f * a);
        float linear    = std::fma(quadratic, x, 8.f * a);
        return std::fma(linear, x, -4.f * a);
    }
    return 0.f;
}

struct Tap
{
    int   index;
    float weight;
};

inline std::vector<Tap> Weights(int source, int destination, int output, NVCVInterpolationType mode, bool aa)
{
    float scale    = F32(static_cast<float>(source) / static_cast<float>(destination));
    float center   = F32(F32(F32(static_cast<float>(output) + 0.5f) * scale) - 0.5f);
    bool  cubic    = mode == NVCV_INTERP_CUBIC;
    float dilation = aa ? std::max(1.f, scale) : 1.f;
    if (!aa && !cubic)
        center = std::max(0.f, center);
    auto  anchor   = static_cast<int64_t>(std::floor(center));
    float fraction = F32(center - static_cast<float>(anchor));
    // The exact center is below source; float32 may round it beyond every surviving AA tap.
    if (aa && anchor >= source)
    {
        anchor   = source - 1;
        fraction = 0.f;
    }
    float   radius    = F32((cubic ? 2.f : 1.f) * dilation);
    int64_t first     = aa ? std::max<int64_t>(0, anchor + static_cast<int64_t>(std::ceil(F32(fraction - radius))))
                           : anchor - int(cubic);
    int     nonAaSpan = cubic ? 3 : 1;
    int64_t last = aa ? std::min<int64_t>(source - 1, anchor + static_cast<int64_t>(std::floor(F32(fraction + radius))))
                      : first + nonAaSpan;
    std::vector<Tap> result;
    float            sum = 0.f;
    for (int64_t i = first; i <= last; ++i)
    {
        // Subtract integer tap positions before conversion; adjacent large indices must remain distinct.
        float w = Kernel(F32(F32(static_cast<float>(i - anchor) - fraction) / dilation), cubic, aa);
        result.push_back({static_cast<int>(std::clamp<int64_t>(i, 0, source - 1)), w});
        sum = F32(sum + w);
    }
    if (aa)
        for (auto &tap : result) tap.weight = F32(tap.weight / sum);
    return result;
}

template<class T>
float Float(T x)
{
    if constexpr (std::is_same_v<T, __half>)
        return __half2float(x);
    else
        return static_cast<float>(x);
}

template<class T>
T Store(float x)
{
    if constexpr (std::is_same_v<T, __half>)
        return __float2half_rn(x);
    else if constexpr (std::is_integral_v<T>)
    {
        float base     = std::floor(x);
        float fraction = x - base;
        float rounded  = base + float(fraction > 0.5f || (fraction == 0.5f && std::fmod(base, 2.f) != 0.f));
        return static_cast<T>(
            std::clamp(rounded, float(std::numeric_limits<T>::lowest()), float(std::numeric_limits<T>::max())));
    }
    else
        return x;
}

template<class T>
float WeightedSum(const T *input, size_t stride, const std::vector<Tap> &taps)
{
    float sum = 0.f;
    for (auto tap : taps) sum = F32(sum + F32(Float(input[size_t(tap.index) * stride]) * tap.weight));
    return sum;
}

// Inputs/outputs are compact NHWC; one invocation defines one compared buffer (one image).
template<class T>
std::vector<T> Resize(const std::vector<T> &input, int sh, int sw, int dh, int dw, int channels,
                      NVCVInterpolationType mode, bool aa)
{
    if (sh == dh && sw == dw)
        return input;
    std::vector<T> output(size_t(dh) * dw * channels);
    if (Nearest(mode))
    {
        for (int y = 0; y < dh; ++y)
            for (int x = 0; x < dw; ++x)
            {
                size_t source = (size_t(NearestIndex(sh, dh, y, mode)) * sw + NearestIndex(sw, dw, x, mode)) * channels;
                std::memcpy(output.data() + (size_t(y) * dw + x) * channels, input.data() + source,
                            channels * sizeof(T));
            }
        return output;
    }
    std::vector<std::vector<Tap>> horizontal(dw);
    std::vector<std::vector<Tap>> vertical(dh);
    for (int x = 0; x < dw; ++x) horizontal[x] = Weights(sw, dw, x, mode, aa);
    for (int y = 0; y < dh; ++y) vertical[y] = Weights(sh, dh, y, mode, aa);
    std::vector<float> intermediate(size_t(sh) * dw * channels);
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < dw; ++x)
            for (int c = 0; c < channels; ++c)
            {
                intermediate[(size_t(y) * dw + x) * channels + c]
                    = WeightedSum(input.data() + size_t(y) * sw * channels + c, channels, horizontal[x]);
            }
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x)
            for (int c = 0; c < channels; ++c)
            {
                float sum
                    = WeightedSum(intermediate.data() + size_t(x) * channels + c, size_t(dw) * channels, vertical[y]);
                output[(size_t(y) * dw + x) * channels + c] = Store<T>(sum);
            }
    return output;
}

inline float HalfUlp(float x)
{
    x = std::abs(x);
    return x < 0x1p-14f ? 0x1p-24f : std::ldexp(1.f, int(std::floor(std::log2(x))) - 10);
}

template<class T>
bool WithinTolerance(const std::vector<T> &gold, const std::vector<T> &actual, NVCVInterpolationType mode, bool aa,
                     bool exact = false)
{
    if (gold.size() != actual.size())
        return false;
    if (exact || Nearest(mode))
        return std::memcmp(gold.data(), actual.data(), gold.size() * sizeof(T)) == 0;
    float range = 1.f;
    if constexpr (std::is_same_v<T, float>)
        for (auto x : gold) range = std::max(range, std::abs(x));
    for (size_t i = 0; i < gold.size(); ++i)
    {
        float bound;
        if constexpr (std::is_same_v<T, __half>)
            bound = 4.f * HalfUlp(Float(gold[i])); // Frozen rule: no absolute floor, even at zero.
        else if constexpr (std::is_same_v<T, float>)
            bound = F32((aa ? 5e-6f : 1e-5f) * range);
        else
            bound = 1.f;
        if (!(std::abs(Float(gold[i]) - Float(actual[i])) <= bound))
            return false;
    }
    return true;
}

} // namespace tvgold
#endif

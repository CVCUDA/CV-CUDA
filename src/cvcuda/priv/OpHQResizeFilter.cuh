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
#ifndef CVCUDA_PRIV_HQ_RESIZE_FILTER_CUH
#define CVCUDA_PRIV_HQ_RESIZE_FILTER_CUH

#include <cuda_runtime.h>
#include <cvcuda/Types.h> // for NVCVInterpolationType, etc.
#include <nvcv/Exception.hpp>
#include <util/Assert.h>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>

/* This file implements ResamplingFiltersFactory.
   The class precomputes the coefficients for the supported filter kinds once
   and stores them in the memory of current device.
   Then, the facotory can be used to create ResamplingFilters
   that interpolates the coefficients for a given support size.*/
namespace cvcuda::priv::hq_resize::filter {

/**
 * @brief Internally supported filters.
 *
 * Triangual is short for Linear + antialias (that requiers precomupting an explicit filter's support)
 */
enum class FilterType : uint8_t
{
    Nearest,
    Linear,
    Triangular,
    Gaussian,
    Cubic,
    Lanczos3,
};

/**
 * @brief Internally supported kinds of filters - all the FilterTypes that
 * require coefficients kept in shared memory are mapped to the same kind:
 * `ShmFilter`.
 *
 */
enum class FilterTypeKind : uint8_t
{
    Nearest,
    Linear,
    ShmFilter,
};

inline FilterTypeKind GetFilterTypeKind(FilterType filterType)
{
    FilterTypeKind filterKind;
    switch (filterType)
    {
    case FilterType::Nearest:
        filterKind = FilterTypeKind::Nearest;
        break;
    case FilterType::Linear:
        filterKind = FilterTypeKind::Linear;
        break;
    default:
        filterKind = FilterTypeKind::ShmFilter;
        break;
    }
    return filterKind;
}

struct FilterMode
{
    FilterType filterType;
    bool       antialias;
};

inline FilterMode GetFilterMode(NVCVInterpolationType interpolation, bool antialias)

{
    FilterType filterType;
    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        filterType = FilterType::Nearest;
        break;
    case NVCV_INTERP_LINEAR:
        filterType = FilterType::Linear;
        break;
    case NVCV_INTERP_CUBIC:
        filterType = FilterType::Cubic;
        break;
    case NVCV_INTERP_LANCZOS:
        filterType = FilterType::Lanczos3;
        break;
    case NVCV_INTERP_GAUSSIAN:
        filterType = FilterType::Gaussian;
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_IMPLEMENTED,
                              "The resize operator does not support the selected interpolation method");
    }
    if (antialias && filterType == FilterType::Linear)
    {
        filterType = FilterType::Triangular;
    }
    return {filterType, antialias};
}

inline std::tuple<FilterMode, FilterMode> GetFilterModes(NVCVInterpolationType minInterpolation,
                                                         NVCVInterpolationType magInterpolation, bool antialias)
{
    std::tuple<FilterMode, FilterMode> modes;
    auto &[minFilter, magFilter] = modes;
    minFilter                    = GetFilterMode(minInterpolation, antialias);
    magFilter                    = GetFilterMode(magInterpolation, false);
    return modes;
}

struct ResamplingFilter
{
    float *coeffs;
    int    numCoeffs;
    float  anchor; // support / 2
    float  scale;  // (numCoeffs - 1) / support

    void rescale(float support)
    {
        float old_scale = scale;
        scale           = (numCoeffs - 1) / support;
        anchor          = anchor * old_scale / scale;
    }

    __host__ __device__ int support() const
    {
        return ceilf((numCoeffs - 1) / scale);
    }

    __device__ float operator()(float x) const
    {
        if (!(x > -1)) // negative and NaN arguments
            return 0;
        if (x >= numCoeffs)
            return 0;
        int   x0 = floorf(x);
        int   x1 = x0 + 1;
        float d  = x - x0;
        float f0 = x0 < 0 ? 0.0f : __ldg(coeffs + x0);
        float f1 = x1 >= numCoeffs ? 0.0f : __ldg(coeffs + x1);
        return f0 + d * (f1 - f0);
    }
};

static_assert(std::is_pod_v<ResamplingFilter>);

inline float LanczosWindow(float x, float a)
{
    if (fabsf(x) >= a)
        return 0.0f;
    return nvcv::util::sinc(x) * nvcv::util::sinc(x / a);
}

inline float CubicWindow(float x)
{
    x = fabsf(x);
    if (x >= 2)
        return 0;

    float x2 = x * x;
    float x3 = x2 * x;
    if (x > 1)
        return -0.5f * x3 + 2.5f * x2 - 4.0f * x + 2.0f;
    else
        return 1.5f * x3 - 2.5f * x2 + 1.0f;
}

template<typename Function>
inline void InitFilter(ResamplingFilter &filter, Function F)
{
    for (int i = 0; i < filter.numCoeffs; i++) filter.coeffs[i] = F(i);
}

inline void InitTriangularFilter(ResamplingFilter filter)
{
    filter.coeffs[0] = 0;
    filter.coeffs[1] = 1;
    filter.coeffs[2] = 0;
}

inline void InitGaussianFilter(ResamplingFilter filter)
{
    InitFilter(filter,
               [&](int i)
               {
                   float x = 4 * (i - (filter.numCoeffs - 1) * 0.5f) / (filter.numCoeffs - 1);
                   return expf(-x * x);
               });
}

inline void InitLanczosFilter(ResamplingFilter filter, float a)
{
    InitFilter(filter,
               [&](int i)
               {
                   float x = 2 * a * (i - (filter.numCoeffs - 1) * 0.5f) / (filter.numCoeffs - 1);
                   return LanczosWindow(x, a);
               });
    filter.rescale(6); // rescaling to the minimal allowed support
}

inline void InitCubicFilter(ResamplingFilter filter)
{
    InitFilter(filter,
               [&](int i)
               {
                   float x = 4 * (i - (filter.numCoeffs - 1) * 0.5f) / (filter.numCoeffs - 1);
                   return CubicWindow(x);
               });
    filter.rescale(4); // rescaling to the minimal allowed support
}

class ResamplingFiltersFactory
{
public:
    enum FilterIdx
    {
        Idx_Triangular = 0,
        Idx_Gaussian,
        Idx_Lanczos3,
        Idx_Cubic,
        kNumFilters
    };

    static constexpr int kLanczosResolution = 32;
    static constexpr int kLanczosA          = 3;

    static constexpr int kTriangularSize = 3;
    static constexpr int kGaussianSize   = 65;
    static constexpr int kCubicSize      = 129;
    static constexpr int kLanczosSize    = (2 * kLanczosA * kLanczosResolution + 1);

    static constexpr int kTotalSize = kTriangularSize + kGaussianSize + kCubicSize + kLanczosSize;

    ResamplingFiltersFactory()
        : m_deviceId{[]()
                     {
                         int deviceId;
                         NVCV_CHECK_THROW(cudaGetDevice(&deviceId));
                         return deviceId;
                     }()}

    {
        // Pinned memory is needed for proper synchronization of the synchronous copy
        std::unique_ptr<float, std::function<void(void *)>> filterDataPinned;
        {
            float *ptr = nullptr;
            NVCV_CHECK_THROW(cudaMallocHost(&ptr, kTotalSize * sizeof(float)));
            filterDataPinned = {ptr, [](void *ptr)
                                {
                                    NVCV_CHECK_THROW(cudaFreeHost(ptr));
                                }};
        }
        {
            float *ptr = nullptr;
            NVCV_CHECK_THROW(cudaMalloc(&ptr, kTotalSize * sizeof(float)));
            m_filterDataGpu = {ptr, [](void *ptr)
                               {
                                   NVCV_CHECK_THROW(cudaFree(ptr));
                               }};
        }
        auto addFilter = [&](FilterIdx filterIdx, int size)
        {
            float *base          = filterIdx == 0 ? filterDataPinned.get()
                                                  : m_filters[filterIdx - 1].coeffs + m_filters[filterIdx - 1].numCoeffs;
            m_filters[filterIdx] = {base, size, 1, (size - 1) * 0.5f};
        };
        addFilter(Idx_Triangular, kTriangularSize);
        InitTriangularFilter(m_filters[Idx_Triangular]);
        addFilter(Idx_Gaussian, kGaussianSize);
        InitGaussianFilter(m_filters[Idx_Gaussian]);
        addFilter(Idx_Lanczos3, kLanczosSize);
        InitLanczosFilter(m_filters[Idx_Lanczos3], kLanczosA);
        addFilter(Idx_Cubic, kCubicSize);
        InitCubicFilter(m_filters[Idx_Cubic]);

        // According to cuda-driver-api: For transfers from pinned host memory to device memory,
        // the cudaMemcpy is synchronous with respect to the host.
        NVCV_CHECK_THROW(cudaMemcpy(m_filterDataGpu.get(), filterDataPinned.get(), kTotalSize * sizeof(float),
                                    cudaMemcpyHostToDevice));
        // Set the pointers to the corresponding offsets in m_filterDataGpu
        ptrdiff_t diff = m_filterDataGpu.get() - filterDataPinned.get();
        for (auto &f : m_filters)
        {
            f.coeffs += diff;
        }
    }

    ResamplingFilter CreateCubic(float radius = 2.0f) const noexcept
    {
        validateDeviceId();
        auto flt = m_filters[Idx_Cubic];
        flt.rescale(2.0f * std::max(2.0f, radius));
        return flt;
    }

    ResamplingFilter CreateGaussian(float sigma) const noexcept
    {
        validateDeviceId();
        auto flt = m_filters[Idx_Gaussian];
        flt.rescale(std::max(1.0f, static_cast<float>(4 * M_SQRT2) * sigma));
        return flt;
    }

    ResamplingFilter CreateLanczos3(float radius = 3.0f) const noexcept
    {
        validateDeviceId();
        auto flt = m_filters[Idx_Lanczos3];
        flt.rescale(2.0f * std::max(3.0f, radius));
        return flt;
    }

    ResamplingFilter CreateTriangular(float radius) const noexcept
    {
        validateDeviceId();
        auto flt = m_filters[Idx_Triangular];
        flt.rescale(std::max(1.0f, 2 * radius));
        return flt;
    }

private:
    void validateDeviceId() const
    {
        int deviceId;
        NVCV_CHECK_THROW(cudaGetDevice(&deviceId));
        if (deviceId != m_deviceId)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_DEVICE,
                                  "The HQ resize operator was initialized and called with different current device.");
        }
    }

    int                                                 m_deviceId;
    std::unique_ptr<float, std::function<void(void *)>> m_filterDataGpu;
    ResamplingFilter                                    m_filters[kNumFilters];
};

inline ResamplingFilter GetResamplingFilter(const ResamplingFiltersFactory &filtersFactory,
                                            const FilterMode &filterMode, const float inSize, const float outSize)
{
    bool antialias = filterMode.antialias && (outSize < inSize);
    switch (filterMode.filterType)
    {
    case FilterType::Linear:
    {
        return filtersFactory.CreateTriangular(1);
    }
    break;
    case FilterType::Triangular:
    {
        const float radius = antialias ? inSize / outSize : 1;
        return filtersFactory.CreateTriangular(radius);
    }
    break;
    case FilterType::Gaussian:
    {
        const float radius = antialias ? inSize / outSize : 1;
        return filtersFactory.CreateGaussian(radius * 0.5f / M_SQRT2);
    }
    break;
    case FilterType::Cubic:
    {
        const float radius = antialias ? (2 * inSize / outSize) : 2;
        return filtersFactory.CreateCubic(radius);
    }
    break;
    case FilterType::Lanczos3:
    {
        const float radius = antialias ? (3 * inSize / outSize) : 3;
        return filtersFactory.CreateLanczos3(radius);
    }
    default: // Nearest neighbour
    {
        return {nullptr, 0, 0, 1};
    }
    }
}

} // namespace cvcuda::priv::hq_resize::filter
#endif // CVCUDA_PRIV_HQ_RESIZE_FILTER_CUH

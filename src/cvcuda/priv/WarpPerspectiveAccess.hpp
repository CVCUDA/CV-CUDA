// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CVCUDA_PRIV_WARP_PERSPECTIVE_ACCESS_HPP
#define CVCUDA_PRIV_WARP_PERSPECTIVE_ACCESS_HPP

#include <cvcuda/cuda_tools/InterpolationWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>

#include <cstdint>

namespace cvcuda::priv::perspective {

template<class Pixel, class Byte>
inline bool PixelAligned(const Byte *base, int64_t rowStride, int64_t sampleStride, int rows, int samples)
{
    return reinterpret_cast<uintptr_t>(base) % alignof(Pixel) == 0 && (rows <= 1 || rowStride % alignof(Pixel) == 0)
        && (samples <= 1 || sampleStride % alignof(Pixel) == 0);
}

template<class Element>
using ScalarTensor = nvcv::cuda::TensorWrapT<Element, int32_t, -1, -1, -1>;

template<class Element, NVCVBorderType Border, NVCVInterpolationType Interpolation>
using ScalarInterpolation
    = nvcv::cuda::InterpolationWrap<nvcv::cuda::BorderWrap<ScalarTensor<const Element>, Border, false, true, true>,
                                    Interpolation, true>;

template<class Element>
class ScalarImageBatch
{
public:
    using ValueType = Element;

    __host__ __device__ ScalarImageBatch(const NVCVImageBufferStrided *images, int channel, int channels)
        : m_images(images)
        , m_channel(channel)
        , m_channels(channels)
    {
    }

    __host__ __device__ int width(int sample) const
    {
        return m_images[sample].planes[0].width;
    }

    __host__ __device__ int height(int sample) const
    {
        return m_images[sample].planes[0].height;
    }

    __host__ __device__ ScalarTensor<Element> view(int sample) const
    {
        const auto &plane = m_images[sample].planes[0];
        return ScalarTensor<Element>(plane.basePtr + m_channel * sizeof(Element), 0, plane.rowStride,
                                     m_channels * static_cast<int32_t>(sizeof(Element)));
    }

    __host__ __device__ Element &operator[](int3 coordinate) const
    {
        return view(coordinate.z)[int3{coordinate.x, coordinate.y, 0}];
    }

private:
    const NVCVImageBufferStrided *m_images;
    int                           m_channel;
    int                           m_channels;
};

template<class Element, NVCVBorderType Border, NVCVInterpolationType Interpolation>
class ScalarImageInterpolation
{
    using Sampler = ScalarInterpolation<Element, Border, Interpolation>;

public:
    using ValueType                                           = Element;
    using BorderWrapper                                       = typename Sampler::BorderWrapper;
    static constexpr NVCVInterpolationType kInterpolationType = Interpolation;

    __host__ __device__ ScalarImageInterpolation(const NVCVImageBufferStrided *images, int channel, int channels,
                                                 Element borderValue)
        : m_images(images, channel, channels)
        , m_borderValue(borderValue)
    {
    }

    __host__ __device__ Element operator[](float3 coordinate) const
    {
        const int     sample = static_cast<int>(coordinate.z);
        const Sampler sampler(m_images.view(sample), m_borderValue, 0.f, 0.f, m_images.height(sample),
                              m_images.width(sample));
        return sampler[float3{coordinate.x, coordinate.y, 0.f}];
    }

private:
    ScalarImageBatch<const Element> m_images;
    Element                         m_borderValue;
};

} // namespace cvcuda::priv::perspective

#endif

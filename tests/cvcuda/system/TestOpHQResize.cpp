/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <cvcuda/OpHQResize.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;
using uchar     = unsigned char;

template<typename T>
using uniform_distribution
    = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

namespace baseline {

template<int kSpatialNDim>
struct Roi
{
    using ShapeT = cuda::MakeType<int, kSpatialNDim>;
    ShapeT origin;
    ShapeT shape;
};

template<int kSpatialNDim>
Roi<kSpatialNDim> FullRoi(typename Roi<kSpatialNDim>::ShapeT shape)
{
    Roi<kSpatialNDim> roi;
    roi.origin = decltype(roi.origin){0};
    roi.shape  = shape;
    return roi;
}

template<typename Cb>
void ForAllInRoi(Roi<2> roi, Cb &&cb)
{
    for (int y = roi.origin.y; y < roi.origin.y + roi.shape.y; y++)
    {
        for (int x = roi.origin.x; x < roi.origin.x + roi.shape.x; x++)
        {
            cb(int2{x, y});
        }
    }
}

template<typename Cb>
void ForAllInRoi(Roi<3> roi, Cb &&cb)
{
    for (int z = roi.origin.z; z < roi.origin.z + roi.shape.z; z++)
    {
        for (int y = roi.origin.y; y < roi.origin.y + roi.shape.y; y++)
        {
            for (int x = roi.origin.x; x < roi.origin.x + roi.shape.x; x++)
            {
                cb(int3{x, y, z});
            }
        }
    }
}

template<typename BT, int kSpatialNDim>
struct CpuSample
{
    static_assert(!cuda::IsCompound<BT>);
    using ShapeT   = cuda::MakeType<int, kSpatialNDim>;         // WH or WHD
    using StridesT = cuda::MakeType<int64_t, kSpatialNDim + 1>; // WHN or WHDN

    CpuSample(int64_t size, StridesT strides, int numSamples, ShapeT shape, int numChannels)
        : m_data(size)
        , m_strides{strides}
        , m_numSamples{numSamples}
        , m_shape{shape}
        , m_numChannels{numChannels}
    {
    }

    BT &get(int sampleIdx, const ShapeT idx, int channel)
    {
        return *(reinterpret_cast<BT *>(data() + offset(sampleIdx, idx)) + channel);
    }

    uint8_t *data()
    {
        return m_data.data();
    }

    StridesT strides()
    {
        return m_strides;
    }

    ShapeT shape()
    {
        return m_shape;
    }

    int numSamples() const
    {
        return m_numSamples;
    }

    int numChannels() const
    {
        return m_numChannels;
    }

private:
    int64_t offset(int sampleIdx, int2 idx)
    {
        return sampleIdx * m_strides.z + idx.y * m_strides.y + idx.x * m_strides.x;
    }

    int64_t offset(int sampleIdx, int3 idx)
    {
        return sampleIdx * m_strides.w + idx.z * m_strides.z + idx.y * m_strides.y + idx.x * m_strides.x;
    }

    std::vector<uint8_t> m_data;
    StridesT             m_strides;
    int                  m_numSamples;
    ShapeT               m_shape;
    int                  m_numChannels;
};

template<typename InBT, typename BT>
double CompareTolerance()
{
    if constexpr (std::is_integral_v<BT>)
    {
        return std::is_same_v<BT, uchar> ? 1 : 10;
    }
    else if constexpr (!std::is_integral_v<InBT>)
    {
        return 1e-4;
    }
    else
    {
        return std::is_same_v<BT, uchar> ? 0.1 : 6;
    }
}

inline CpuSample<float, 2> GetIntermediate(int numSamples, int2 shape, int numChannels)
{
    int64_t                    size = sizeof(float) * numSamples * shape.y * shape.x * numChannels;
    cuda::MakeType<int64_t, 3> strides;
    strides.x = sizeof(float) * numChannels;
    strides.y = strides.x * shape.x;
    strides.z = strides.y * shape.y;
    return {size, strides, numSamples, shape, numChannels};
}

inline CpuSample<float, 3> GetIntermediate(int numSamples, int3 shape, int numChannels)
{
    int64_t                    size = sizeof(float) * numSamples * shape.z * shape.y * shape.x * numChannels;
    cuda::MakeType<int64_t, 4> strides;
    strides.x = sizeof(float) * numChannels;
    strides.y = strides.x * shape.x;
    strides.z = strides.y * shape.y;
    strides.w = strides.z * shape.z;
    return {size, strides, numSamples, shape, numChannels};
}

struct FilterTriangular
{
    int size() const
    {
        return 3;
    }

    float operator[](int k) const
    {
        return k == 1 ? 1 : 0;
    }
};

struct FilterCubic
{
    int size() const
    {
        return 129;
    }

    float operator[](int k) const
    {
        float x
            = 4.f * (static_cast<float>(k) - static_cast<float>(size() - 1) * 0.5f) / static_cast<float>(size() - 1);
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
};

struct FilterGaussian
{
    int size() const
    {
        return 65;
    }

    float operator[](int k) const
    {
        float x
            = 4.f * (static_cast<float>(k) - static_cast<float>(size() - 1) * 0.5f) / static_cast<float>(size() - 1);
        return expf(-x * x);
    }
};

struct FilterLanczos
{
    static constexpr int kLanczosA          = 3;
    static constexpr int kLanczosResolution = 32;

    int size() const
    {
        return (2 * kLanczosA * kLanczosResolution + 1);
    }

    float operator[](int k) const
    {
        float x = 2.f * static_cast<float>(kLanczosA) * (static_cast<float>(k) - static_cast<float>(size() - 1) * 0.5f)
                / static_cast<float>(size() - 1);
        if (fabsf(x) >= static_cast<float>(kLanczosA))
            return 0.0f;
        return nvcv::util::sinc(x) * nvcv::util::sinc(x / static_cast<float>(kLanczosA));
    }
};

template<typename FilterType> // NOSONAR: this small wrapper exposes a filter object with a call operator under test.
struct Filter
{
    explicit Filter(float support)
        : m_support{support}
    {
    }

    int support() const
    {
        return static_cast<int>(std::ceil(m_support));
    }

    float scale() const
    {
        return static_cast<float>(m_filter.size() - 1) / m_support;
    }

    float anchor() const
    {
        return m_support / 2;
    }

    float operator()(float x) const
    {
        if (!(x > -1))
            return 0;
        if (x >= static_cast<float>(m_filter.size()))
            return 0;
        auto  x0 = static_cast<int>(std::floor(x));
        int   x1 = x0 + 1;
        float d  = x - static_cast<float>(x0);
        float f0 = x0 < 0 ? 0.0f : m_filter[x0];
        float f1 = x1 >= m_filter.size() ? 0.0f : m_filter[x1];
        return f0 + d * (f1 - f0); // NOSONAR: std::lerp is C++20.
    }

private:
    [[no_unique_address]] FilterType m_filter{};
    float                            m_support;
};

template<typename OutBT, typename InBT, int kSpatialNDim>
void RunNN(int axis, CpuSample<OutBT, kSpatialNDim> &outTensorCpu, CpuSample<InBT, kSpatialNDim> &inTensorCpu,
           Roi<kSpatialNDim> roi)
{
    const int   numSamples  = inTensorCpu.numSamples();
    const int   numChannels = inTensorCpu.numChannels();
    const auto  inShape     = inTensorCpu.shape();
    const auto  outShape    = outTensorCpu.shape();
    const int   inSize      = cuda::GetElement(inShape, axis);
    const int   outSize     = cuda::GetElement(outShape, axis);
    const auto  axisScale   = static_cast<float>(inSize) / static_cast<float>(outSize);
    const float axisOrigin  = 0.5f * axisScale;
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        ForAllInRoi(roi,
                    [axis, axisScale, axisOrigin, inSize, numChannels, sampleIdx, &outTensorCpu,
                     &inTensorCpu](const cuda::MakeType<int, kSpatialNDim> outIdx)
                    {
                        auto inIdx  = outIdx;
                        auto inAxis = static_cast<int>(
                            std::floor(static_cast<float>(cuda::GetElement(outIdx, axis)) * axisScale + axisOrigin));
                        inAxis                        = std::clamp(inAxis, 0, inSize - 1);
                        cuda::GetElement(inIdx, axis) = inAxis;
                        for (int c = 0; c < numChannels; c++)
                        {
                            outTensorCpu.get(sampleIdx, outIdx, c)
                                = cuda::SaturateCast<OutBT>(inTensorCpu.get(sampleIdx, inIdx, c));
                        }
                    });
    }
}

template<typename OutBT, typename InBT, int kSpatialNDim>
void RunLinear(int axis, CpuSample<OutBT, kSpatialNDim> &outTensorCpu, CpuSample<InBT, kSpatialNDim> &inTensorCpu,
               Roi<kSpatialNDim> roi)
{
    const int   numSamples  = inTensorCpu.numSamples();
    const int   numChannels = inTensorCpu.numChannels();
    const auto  inShape     = inTensorCpu.shape();
    const auto  outShape    = outTensorCpu.shape();
    const int   inSize      = cuda::GetElement(inShape, axis);
    const int   outSize     = cuda::GetElement(outShape, axis);
    const auto  axisScale   = static_cast<float>(inSize) / static_cast<float>(outSize);
    const float axisOrigin  = 0.5f * axisScale - 0.5f;
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        ForAllInRoi(roi,
                    [axis, axisScale, axisOrigin, inSize, numChannels, sampleIdx, &outTensorCpu,
                     &inTensorCpu](const cuda::MakeType<int, kSpatialNDim> outIdx)
                    {
                        const auto inAxis0f
                            = static_cast<float>(cuda::GetElement(outIdx, axis)) * axisScale + axisOrigin;
                        auto        inAxis0            = static_cast<int>(std::floor(inAxis0f));
                        int         inAxis1            = inAxis0 + 1;
                        const float q                  = inAxis0f - static_cast<float>(inAxis0);
                        inAxis0                        = std::clamp(inAxis0, 0, inSize - 1);
                        inAxis1                        = std::clamp(inAxis1, 0, inSize - 1);
                        auto inIdx0                    = outIdx;
                        auto inIdx1                    = outIdx;
                        cuda::GetElement(inIdx0, axis) = inAxis0;
                        cuda::GetElement(inIdx1, axis) = inAxis1;
                        for (int c = 0; c < numChannels; c++)
                        {
                            const float a                          = inTensorCpu.get(sampleIdx, inIdx0, c);
                            const float b                          = inTensorCpu.get(sampleIdx, inIdx1, c);
                            const float tmp                        = b - a;
                            outTensorCpu.get(sampleIdx, outIdx, c) = cuda::SaturateCast<OutBT>(std::fmaf(tmp, q, a));
                        }
                    });
    }
}

template<typename OutBT, typename InBT, typename FilterT, int kSpatialNDim>
void RunFilter(int axis, CpuSample<OutBT, kSpatialNDim> &outTensorCpu, CpuSample<InBT, kSpatialNDim> &inTensorCpu,
               const FilterT &filter, Roi<kSpatialNDim> roi)
{
    const int   numSamples    = inTensorCpu.numSamples();
    const int   numChannels   = inTensorCpu.numChannels();
    const auto  inShape       = inTensorCpu.shape();
    const auto  outShape      = outTensorCpu.shape();
    const int   inSize        = cuda::GetElement(inShape, axis);
    const int   outSize       = cuda::GetElement(outShape, axis);
    const int   filterSupport = filter.support();
    const float filterStep    = filter.scale();
    const auto  axisScale     = static_cast<float>(inSize) / static_cast<float>(outSize);
    const float axisOrigin    = 0.5f * axisScale - 0.5f - filter.anchor();

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        ForAllInRoi(roi,
                    [axis, axisScale, axisOrigin, filterStep, filterSupport, inSize, numChannels, sampleIdx, &filter,
                     &outTensorCpu, &inTensorCpu](const cuda::MakeType<int, kSpatialNDim> outIdx)
                    {
                        const auto inAxis0f
                            = static_cast<float>(cuda::GetElement(outIdx, axis)) * axisScale + axisOrigin;
                        auto        inAxis0 = static_cast<int>(std::ceil(inAxis0f));
                        const float fStart  = (static_cast<float>(inAxis0) - inAxis0f) * filterStep;
                        for (int c = 0; c < numChannels; c++)
                        {
                            float tmp  = 0;
                            float norm = 0;
                            for (int k = 0; k < filterSupport; k++)
                            {
                                int inAxis                    = inAxis0 + k;
                                inAxis                        = std::clamp(inAxis, 0, inSize - 1);
                                auto inIdx                    = outIdx;
                                cuda::GetElement(inIdx, axis) = inAxis;
                                const InBT inVal              = inTensorCpu.get(sampleIdx, inIdx, c);
                                float      coeff              = filter(fStart + static_cast<float>(k) * filterStep);
                                tmp                           = std::fmaf(inVal, coeff, tmp);
                                norm += coeff;
                            }
                            outTensorCpu.get(sampleIdx, outIdx, c) = cuda::SaturateCast<OutBT>(tmp / norm);
                        }
                    });
    }
}

template<typename OutBT, typename InBT, int kSpatialNDim>
void RunFilter(int axis, CpuSample<OutBT, kSpatialNDim> &outTensorCpu, CpuSample<InBT, kSpatialNDim> &inTensorCpu,
               const NVCVInterpolationType interpolation, bool antialias, Roi<kSpatialNDim> roi)
{
    const auto inShape  = inTensorCpu.shape();
    const auto outShape = outTensorCpu.shape();
    const auto inSize   = static_cast<float>(cuda::GetElement(inShape, axis));
    const auto outSize  = static_cast<float>(cuda::GetElement(outShape, axis));
    switch (interpolation)
    {
    case NVCV_INTERP_LINEAR:
    {
        float radius  = antialias ? inSize / outSize : 1.f;
        float support = std::max(1.0f, 2.f * radius);
        RunFilter(axis, outTensorCpu, inTensorCpu, Filter<FilterTriangular>{support}, roi);
    }
    break;
    case NVCV_INTERP_CUBIC:
    {
        float radius  = antialias ? (2.f * inSize / outSize) : 2.f;
        float support = std::max(4.0f, 2.f * radius);
        RunFilter(axis, outTensorCpu, inTensorCpu, Filter<FilterCubic>{support}, roi);
    }
    break;
    case NVCV_INTERP_GAUSSIAN:
    {
        float radius  = antialias ? inSize / outSize : 1.f;
        float support = std::max(1.0f, 2.f * radius);
        RunFilter(axis, outTensorCpu, inTensorCpu, Filter<FilterGaussian>{support}, roi);
    }
    break;
    case NVCV_INTERP_LANCZOS:
    {
        float radius  = antialias ? (3.f * inSize / outSize) : 3.f;
        float support = std::max(6.0f, 2.f * radius);
        RunFilter(axis, outTensorCpu, inTensorCpu, Filter<FilterLanczos>{support}, roi);
    }
    break;
    default:
        FAIL() << "Unsupported filter";
    }
}

template<typename OutBT, typename InBT, int kSpatialNDim>
void RunPass(int axis, CpuSample<OutBT, kSpatialNDim> &outTensorCpu, CpuSample<InBT, kSpatialNDim> &inTensorCpu,
             const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation, bool antialias,
             Roi<kSpatialNDim> roi)

{
    const auto inShape       = inTensorCpu.shape();
    const auto outShape      = outTensorCpu.shape();
    const int  inSize        = cuda::GetElement(inShape, axis);
    const int  outSize       = cuda::GetElement(outShape, axis);
    const bool isScalingDown = outSize < inSize;
    antialias &= isScalingDown;
    const auto interpolation = isScalingDown ? minInterpolation : magInterpolation;
    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        RunNN(axis, outTensorCpu, inTensorCpu, roi);
        break;
    case NVCV_INTERP_LINEAR:
    {
        if (antialias)
        {
            RunFilter(axis, outTensorCpu, inTensorCpu, interpolation, antialias, roi);
        }
        else
        {
            RunLinear(axis, outTensorCpu, inTensorCpu, roi);
        }
    }
    break;
    default:
        RunFilter(axis, outTensorCpu, inTensorCpu, interpolation, antialias, roi);
        break;
    }
}

template<typename OutBT, typename InBT>
void Resize(CpuSample<OutBT, 2> &refTensorCpu, CpuSample<InBT, 2> &inTensorCpu,
            const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation, bool antialias,
            std::optional<Roi<2>> inRoiArg = std::nullopt, std::optional<Roi<2>> outRoiArg = std::nullopt)
{
    int        numSamples  = inTensorCpu.numSamples();
    int        numChannels = inTensorCpu.numChannels();
    const int2 inShape     = inTensorCpu.shape();
    const int2 outShape    = refTensorCpu.shape();
    const int2 interShape  = {outShape.x, inShape.y};
    Roi<2>     inRoi       = inRoiArg.value_or(FullRoi<2>(inShape));
    Roi<2>     outRoi      = outRoiArg.value_or(FullRoi<2>(outShape));
    auto       interRoi    = Roi<2>{
                 int2{outRoi.origin.x, inRoi.origin.y},
                 int2{ outRoi.shape.x,  inRoi.shape.y}
    };

    auto intermediateTensor = GetIntermediate(numSamples, interShape, numChannels);
    RunPass(0, intermediateTensor, inTensorCpu, minInterpolation, magInterpolation, antialias, interRoi);
    RunPass(1, refTensorCpu, intermediateTensor, minInterpolation, magInterpolation, antialias, outRoi);
}

template<typename OutBT, typename InBT>
void Resize(CpuSample<OutBT, 3> &refTensorCpu, CpuSample<InBT, 3> &inTensorCpu,
            const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation, bool antialias,
            std::optional<Roi<3>> inRoiArg = std::nullopt, std::optional<Roi<3>> outRoiArg = std::nullopt)
{
    int        numSamples  = inTensorCpu.numSamples();
    int        numChannels = inTensorCpu.numChannels();
    const int3 inShape     = inTensorCpu.shape();
    const int3 outShape    = refTensorCpu.shape();
    const int3 interShape0 = {outShape.x, inShape.y, inShape.z};
    const int3 interShape1 = {outShape.x, outShape.y, inShape.z};
    Roi<3>     inRoi       = inRoiArg.value_or(FullRoi<3>(inShape));
    Roi<3>     outRoi      = outRoiArg.value_or(FullRoi<3>(outShape));
    auto       interRoi0   = Roi<3>{
                int3{outRoi.origin.x, inRoi.origin.y, inRoi.origin.z},
                int3{ outRoi.shape.x,  inRoi.shape.y,  inRoi.shape.z}
    };
    auto interRoi1 = Roi<3>{
        int3{outRoi.origin.x, outRoi.origin.y, inRoi.origin.z},
        int3{ outRoi.shape.x,  outRoi.shape.y,  inRoi.shape.z}
    };

    auto intermediateTensor0 = GetIntermediate(numSamples, interShape0, numChannels);
    RunPass(0, intermediateTensor0, inTensorCpu, minInterpolation, magInterpolation, antialias, interRoi0);
    auto intermediateTensor1 = GetIntermediate(numSamples, interShape1, numChannels);
    RunPass(1, intermediateTensor1, intermediateTensor0, minInterpolation, magInterpolation, antialias, interRoi1);
    RunPass(2, refTensorCpu, intermediateTensor1, minInterpolation, magInterpolation, antialias, outRoi);
}

template<typename BT, int kSpatialNDim, typename Cb>
void CompareElementWise(CpuSample<BT, kSpatialNDim> &tensor, CpuSample<BT, kSpatialNDim> &refTensor,
                        std::optional<Roi<kSpatialNDim>> roi_arg, Cb &&cb)
{
    const int  numSamples  = tensor.numSamples();
    const int  numChannels = tensor.numChannels();
    const auto shape       = tensor.shape();
    const auto roi         = roi_arg.value_or(FullRoi<kSpatialNDim>(shape));
    ASSERT_EQ(numSamples, refTensor.numSamples());
    ASSERT_EQ(numChannels, refTensor.numChannels());
    ASSERT_EQ(shape, refTensor.shape());

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        ForAllInRoi(roi,
                    [numChannels, sampleIdx, &cb](const cuda::MakeType<int, kSpatialNDim> idx)
                    {
                        for (int c = 0; c < numChannels; c++)
                        {
                            cb(sampleIdx, idx, c);
                        }
                    });
    }
}

template<typename InBT, typename BT, int kSpatialNDim>
void Compare(CpuSample<BT, kSpatialNDim> &tensor, CpuSample<BT, kSpatialNDim> &refTensor, bool antialias,
             std::optional<Roi<kSpatialNDim>> roi_arg = std::nullopt)
{
    double  err = 0;
    int64_t vol = 0;
    CompareElementWise(
        tensor, refTensor, roi_arg,
        [&tensor, &refTensor, &err, &vol](int sampleIdx, const cuda::MakeType<int, kSpatialNDim> idx, int c)
        {
            const BT val    = tensor.get(sampleIdx, idx, c);
            const BT refVal = refTensor.get(sampleIdx, idx, c);
            err += abs(val - refVal);
            vol += 1;

            const double tolerance = CompareTolerance<InBT, BT>();
            ASSERT_NEAR(val, refVal, tolerance);
        });
    double mean_err = err / static_cast<double>(vol);
    ASSERT_LE(mean_err, antialias ? 0.1 : 0.4);
}

template<typename BT, int kSpatialNDim>
void CompareExact(CpuSample<BT, kSpatialNDim> &tensor, CpuSample<BT, kSpatialNDim> &refTensor,
                  std::optional<Roi<kSpatialNDim>> roi_arg = std::nullopt)
{
    CompareElementWise(tensor, refTensor, roi_arg,
                       [&tensor, &refTensor](int sampleIdx, const cuda::MakeType<int, kSpatialNDim> idx, int c)
                       {
                           ASSERT_EQ(tensor.get(sampleIdx, idx, c), refTensor.get(sampleIdx, idx, c))
                               << "sampleIdx=" << sampleIdx << ", channel=" << c;
                       });
}
} // namespace baseline

inline void GetMaxShape(HQResizeTensorShapeI &ret, const HQResizeTensorShapeI &other)
{
    ASSERT_EQ(ret.ndim, other.ndim);
    ret.numChannels = std::max(ret.numChannels, other.numChannels);
    for (int d = 0; d < ret.ndim; d++)
    {
        ret.extent[d] = std::max(ret.extent[d], other.extent[d]);
    }
}

inline void GetMaxShape( // NOSONAR: std::span is C++20.
    HQResizeTensorShapeI &ret, const HQResizeTensorShapeI *shapes, int numSamples)
{
    if (numSamples > 0)
    {
        ret = shapes[0];
        for (int i = 1; i < numSamples; i++)
        {
            GetMaxShape(ret, shapes[i]);
        }
    }
}

template<typename BT>
struct TypeAsFormatImpl
{
};

template<>
struct TypeAsFormatImpl<uchar>
{
    static constexpr NVCVDataType value = NVCV_DATA_TYPE_U8;
};

template<>
struct TypeAsFormatImpl<short>
{
    static constexpr NVCVDataType value = NVCV_DATA_TYPE_S16;
};

template<>
struct TypeAsFormatImpl<ushort>
{
    static constexpr NVCVDataType value = NVCV_DATA_TYPE_U16;
};

template<>
struct TypeAsFormatImpl<float>
{
    static constexpr NVCVDataType value = NVCV_DATA_TYPE_F32;
};

template<typename BT>
nvcv::DataType TypeAsFormat()
{
    return nvcv::DataType{TypeAsFormatImpl<BT>::value};
}

template<typename... Extents>
nvcv::Tensor CreateTensorHelper(nvcv::DataType dtype, const char *layoutStr, int numSamples, Extents... extents)
{
    nvcv::TensorLayout layout{layoutStr};
    if (numSamples == 1)
    {
        nvcv::TensorShape shape{{extents...}, layout.last(sizeof...(extents))};
        return nvcv::Tensor{shape, dtype};
    }
    else
    {
        nvcv::TensorShape shape{
            {numSamples, extents...},
            layout
        };
        return nvcv::Tensor{shape, dtype};
    }
}

#define NVCV_SHAPE2D(h, w) (int2{w, h})
#define NVCV_TEST_ROW(NumSamples, InShape, OutShape, NumChannels, InT, OutT, Interpolation)                          \
    ttype::Types<ttype::Value<NumSamples>, ttype::Value<InShape>, ttype::Value<OutShape>, ttype::Value<NumChannels>, \
                 InT, OutT, ttype::Value<Interpolation>>

NVCV_TYPED_TEST_SUITE(
    OpHQResizeTensor2D,
    // [uchar, ushort, short, float] x [same, float] x [1, 2, 3, 4, more channels]
    // the input and output shapes: [x, y] -> [scale_down, scale_up]
    // interpolation methods: [nn, linear, gaussian, cubic, lanczos]
    ttype::Types<
        NVCV_TEST_ROW(1, NVCV_SHAPE2D(769, 211), NVCV_SHAPE2D(40, 40), 1, uchar, uchar, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(2, NVCV_SHAPE2D(1024, 101), NVCV_SHAPE2D(105, 512), 1, uchar, float, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(3, NVCV_SHAPE2D(31, 244), NVCV_SHAPE2D(311, 122), 2, uchar, uchar, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(4, NVCV_SHAPE2D(41, 41), NVCV_SHAPE2D(244, 244), 2, uchar, float, NVCV_INTERP_GAUSSIAN),
        NVCV_TEST_ROW(3, NVCV_SHAPE2D(769, 211), NVCV_SHAPE2D(40, 40), 3, uchar, uchar, NVCV_INTERP_LANCZOS),
        NVCV_TEST_ROW(4, NVCV_SHAPE2D(1024, 101), NVCV_SHAPE2D(105, 512), 3, uchar, float, NVCV_INTERP_LANCZOS),
        NVCV_TEST_ROW(3, NVCV_SHAPE2D(31, 244), NVCV_SHAPE2D(311, 122), 4, uchar, uchar, NVCV_INTERP_GAUSSIAN),
        NVCV_TEST_ROW(4, NVCV_SHAPE2D(41, 41), NVCV_SHAPE2D(244, 244), 4, uchar, float, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(3, NVCV_SHAPE2D(31, 244), NVCV_SHAPE2D(311, 122), 5, uchar, uchar, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(4, NVCV_SHAPE2D(41, 41), NVCV_SHAPE2D(244, 244), 8, uchar, float, NVCV_INTERP_LINEAR),

        NVCV_TEST_ROW(1, NVCV_SHAPE2D(769, 211), NVCV_SHAPE2D(40, 40), 1, ushort, ushort, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(2, NVCV_SHAPE2D(1024, 101), NVCV_SHAPE2D(105, 512), 1, short, float, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(3, NVCV_SHAPE2D(31, 244), NVCV_SHAPE2D(311, 122), 2, short, short, NVCV_INTERP_GAUSSIAN),
        NVCV_TEST_ROW(4, NVCV_SHAPE2D(41, 41), NVCV_SHAPE2D(244, 244), 2, ushort, float, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(3, NVCV_SHAPE2D(769, 211), NVCV_SHAPE2D(40, 40), 3, ushort, ushort, NVCV_INTERP_GAUSSIAN),
        NVCV_TEST_ROW(4, NVCV_SHAPE2D(1024, 101), NVCV_SHAPE2D(105, 512), 3, short, float, NVCV_INTERP_GAUSSIAN),
        NVCV_TEST_ROW(3, NVCV_SHAPE2D(31, 244), NVCV_SHAPE2D(311, 122), 4, ushort, ushort, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(3, NVCV_SHAPE2D(31, 244), NVCV_SHAPE2D(311, 122), 7, ushort, float, NVCV_INTERP_NEAREST),

        NVCV_TEST_ROW(3, NVCV_SHAPE2D(769, 211), NVCV_SHAPE2D(40, 40), 1, float, float, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(4, NVCV_SHAPE2D(1024, 101), NVCV_SHAPE2D(105, 512), 2, float, float, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(3, NVCV_SHAPE2D(31, 244), NVCV_SHAPE2D(311, 122), 3, float, float, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(4, NVCV_SHAPE2D(41, 41), NVCV_SHAPE2D(244, 244), 4, float, float, NVCV_INTERP_GAUSSIAN),
        NVCV_TEST_ROW(3, NVCV_SHAPE2D(769, 211), NVCV_SHAPE2D(40, 40), 7, float, float, NVCV_INTERP_LANCZOS),
        NVCV_TEST_ROW(1, NVCV_SHAPE2D(1 << 14, 1 << 13), NVCV_SHAPE2D(512, 256), 7, float, float, NVCV_INTERP_LINEAR),

        NVCV_TEST_ROW(1, NVCV_SHAPE2D(8192, 8192), NVCV_SHAPE2D(32, 32), 1, uchar, uchar, NVCV_INTERP_LANCZOS)>);

inline long3 Tensor2DStrides(const nvcv::TensorDataStridedCuda &data, int2 shape, int numSamples, int numChannels)
{
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(data);
    EXPECT_TRUE(access);
    if (!access)
    {
        return {};
    }
    EXPECT_EQ(access->numSamples(), numSamples);
    EXPECT_EQ(access->numChannels(), numChannels);
    return {access->colStride(), access->rowStride(),
            access->sampleStride() == 0 ? access->rowStride() * shape.y : access->sampleStride()};
}

template<typename InBT, typename OutBT>
struct Tensor2DTestSamples
{
    long3                         inStrides;
    long3                         outStrides;
    baseline::CpuSample<InBT, 2>  inCpu;
    baseline::CpuSample<OutBT, 2> outCpu;
    baseline::CpuSample<OutBT, 2> refCpu;
};

template<typename InBT, typename OutBT>
Tensor2DTestSamples<InBT, OutBT> MakeTensor2DTestSamples(const nvcv::TensorDataStridedCuda &inData,
                                                         const nvcv::TensorDataStridedCuda &outData, int2 inShape,
                                                         int2 outShape, int numSamples, int numChannels)
{
    const long3 inStrides  = Tensor2DStrides(inData, inShape, numSamples, numChannels);
    const long3 outStrides = Tensor2DStrides(outData, outShape, numSamples, numChannels);

    return {
        inStrides,
        outStrides,
        baseline::CpuSample<InBT, 2>{ inStrides.z * numSamples,  inStrides, numSamples,  inShape, numChannels},
        baseline::CpuSample<OutBT, 2>{outStrides.z * numSamples, outStrides, numSamples, outShape, numChannels},
        baseline::CpuSample<OutBT, 2>{outStrides.z * numSamples, outStrides, numSamples, outShape, numChannels},
    };
}

template<typename TypeParam>
void TestTensor(bool antialias, bool exactOutput = false)
{
    const int  numSamples                     = ttype::GetValue<TypeParam, 0>;
    const int2 inShape                        = ttype::GetValue<TypeParam, 1>;
    const int2 outShape                       = ttype::GetValue<TypeParam, 2>;
    const int  numChannels                    = ttype::GetValue<TypeParam, 3>;
    using InBT                                = ttype::GetType<TypeParam, 4>;
    using OutBT                               = ttype::GetType<TypeParam, 5>;
    const nvcv::DataType        inDtype       = TypeAsFormat<InBT>();
    const nvcv::DataType        outDtype      = TypeAsFormat<OutBT>();
    const NVCVInterpolationType interpolation = ttype::GetValue<TypeParam, 6>;

    nvcv::Tensor inTensor  = CreateTensorHelper(inDtype, "NHWC", numSamples, inShape.y, inShape.x, numChannels);
    nvcv::Tensor outTensor = CreateTensorHelper(outDtype, "NHWC", numSamples, outShape.y, outShape.x, numChannels);

    baseline::Roi<2> inRoi  = baseline::FullRoi<2>(inShape);
    baseline::Roi<2> outRoi = baseline::FullRoi<2>(outShape);
    if (inShape.x * inShape.y > 1 << 23)
    {
        inRoi.shape  = cuda::min(inShape, int2{1 << 12, 1 << 11});
        inRoi.origin = inShape - inRoi.shape;

        double2 scale = cuda::StaticCast<double>(outShape) / cuda::StaticCast<double>(inShape);
        outRoi.shape  = cuda::StaticCast<int>(scale * cuda::StaticCast<double>(inRoi.shape));
        outRoi.origin = cuda::StaticCast<int>(scale * cuda::StaticCast<double>(inRoi.origin));
    }

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inData && outData);
    auto samples = MakeTensor2DTestSamples<InBT, OutBT>(*inData, *outData, inShape, outShape, numSamples, numChannels);

    uniform_distribution<InBT> rand(InBT{0}, std::is_integral_v<InBT> ? cuda::TypeTraits<InBT>::max : InBT{1});
    std::mt19937_64            rng(12345);

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        baseline::ForAllInRoi(inRoi,
                              [sampleIdx, &samples, &rand, &rng](int2 idx)
                              {
                                  for (int c = 0; c < numChannels; c++)
                                  {
                                      samples.inCpu.get(sampleIdx, idx, c) = rand(rng);
                                  }
                              });
    }

    cvcuda::HQResize        op;
    cudaStream_t            stream;
    cvcuda::UniqueWorkspace ws;
    {
        HQResizeTensorShapeI inShapeDesc{
            {inShape.y, inShape.x},
            2,
            numChannels
        };
        HQResizeTensorShapeI outShapeDesc{
            {outShape.y, outShape.x},
            2,
            numChannels
        };
        ASSERT_NO_THROW(ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(
                            numSamples, inShapeDesc, outShapeDesc, interpolation, interpolation, antialias)));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inData->basePtr(), samples.inCpu.data(), samples.inStrides.z * numSamples,
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_NO_THROW(op(stream, ws.get(), inTensor, outTensor, interpolation, interpolation, antialias));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(samples.outCpu.data(), outData->basePtr(), samples.outStrides.z * numSamples,
                                           cudaMemcpyDeviceToHost, stream));
    baseline::Resize(samples.refCpu, samples.inCpu, interpolation, interpolation, antialias, {inRoi}, {outRoi});
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    if (exactOutput)
    {
        baseline::CompareExact(samples.outCpu, samples.refCpu, {outRoi});
    }
    else
    {
        baseline::Compare<InBT>(samples.outCpu, samples.refCpu, antialias, {outRoi});
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TYPED_TEST(OpHQResizeTensor2D, correct_output_no_antialias)
{
    TestTensor<TypeParam>(false);
}

TYPED_TEST(OpHQResizeTensor2D, correct_output_with_antialias)
{
    TestTensor<TypeParam>(true);
}

// =============================================================================
// Planar (NCHW/CHW) parity for the single-tensor path.
//
// HQResize resizes each channel independently, so a planar input is processed
// plane-by-plane and must produce exactly the same pixels as the interleaved
// path. These tests feed identical data through both layouts and require the
// (re-interleaved) planar output to match the interleaved output bit-for-bit,
// for every dtype, channel count, and interpolation mode.
// =============================================================================
namespace planar_parity {

// Copy channel `c` between a tightly-packed interleaved (HWC) buffer and a packed single plane.
// Flattening the (y, x) traversal into a single pixel index keeps callers shallow.
template<typename T>
void ExtractPlane(std::vector<T> &plane, const T *interleaved, int pixels, int channels, int c)
{
    for (int i = 0; i < pixels; ++i) plane[i] = interleaved[i * channels + c];
}

template<typename T>
void StorePlane(T *interleaved, const std::vector<T> &plane, int pixels, int channels, int c)
{
    for (int i = 0; i < pixels; ++i) interleaved[i * channels + c] = plane[i];
}

// Deterministically fill a buffer with well-mixed values so every element differs from its
// neighbors. A bit-exact parity test only needs varied, reproducible input shared by both layouts,
// so a fixed hash is preferable to a PRNG (and avoids a security-hotspot finding on std::mt19937).
template<typename T>
void FillPattern(std::vector<T> &buf, uint64_t seed)
{
    uint64_t i = seed;
    for (T &v : buf)
    {
        uint64_t h = (++i) * 6364136223846793005ULL + 1442695040888963407ULL;
        h ^= h >> 29;
        if constexpr (std::is_integral_v<T>)
            v = static_cast<T>(h);
        else
            v = static_cast<T>((h >> 40) & 0xFFFF) / static_cast<T>(65535);
    }
}

// Upload a per-sample interleaved (HWC) host buffer into an interleaved (NHWC) or planar (NCHW/CHW)
// tensor, deinterleaving into per-channel planes for the planar case.
template<typename T>
void UploadTensor(const nvcv::Tensor &t, bool planar, int numSamples, int2 wh, int channels,
                  const std::vector<T> &interleaved)
{
    auto data = t.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(data);
    auto acc = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
    ASSERT_TRUE(acc);
    const int    W           = wh.x;
    const int    H           = wh.y;
    const size_t sampleElems = static_cast<size_t>(W) * H * channels;
    for (int s = 0; s < numSamples; ++s)
    {
        if (!planar)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(acc->sampleData(s), acc->rowStride(), interleaved.data() + s * sampleElems,
                                   W * channels * sizeof(T), W * channels * sizeof(T), H, cudaMemcpyHostToDevice));
            continue;
        }
        std::vector<T> plane(static_cast<size_t>(W) * H);
        for (int c = 0; c < channels; ++c)
        {
            ExtractPlane<T>(plane, interleaved.data() + s * sampleElems, W * H, channels, c);
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(acc->sampleData(s) + c * acc->chStride(), acc->rowStride(),
                                                plane.data(), W * sizeof(T), W * sizeof(T), H, cudaMemcpyHostToDevice));
        }
    }
}

// Inverse of UploadTensor: download a tensor as a per-sample interleaved (HWC) host buffer.
template<typename T>
std::vector<T> DownloadTensor(const nvcv::Tensor &t, bool planar, int numSamples, int2 wh, int channels)
{
    auto data = t.exportData<nvcv::TensorDataStridedCuda>();
    EXPECT_TRUE(data);
    auto acc = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
    EXPECT_TRUE(acc);
    const int      W           = wh.x;
    const int      H           = wh.y;
    const size_t   sampleElems = static_cast<size_t>(W) * H * channels;
    std::vector<T> interleaved(sampleElems * numSamples);
    for (int s = 0; s < numSamples; ++s)
    {
        if (!planar)
        {
            EXPECT_EQ(cudaSuccess,
                      cudaMemcpy2D(interleaved.data() + s * sampleElems, W * channels * sizeof(T), acc->sampleData(s),
                                   acc->rowStride(), W * channels * sizeof(T), H, cudaMemcpyDeviceToHost));
            continue;
        }
        std::vector<T> plane(static_cast<size_t>(W) * H);
        for (int c = 0; c < channels; ++c)
        {
            EXPECT_EQ(cudaSuccess, cudaMemcpy2D(plane.data(), W * sizeof(T), acc->sampleData(s) + c * acc->chStride(),
                                                acc->rowStride(), W * sizeof(T), H, cudaMemcpyDeviceToHost));
            StorePlane<T>(interleaved.data() + s * sampleElems, plane, W * H, channels, c);
        }
    }
    return interleaved;
}

// Resize identical data in the given layout and return the (re-interleaved) output. The same
// workspace requirements (computed from the public C-channel shape) serve both layouts because the
// single-tensor workspace is volume-only and Volume*C*N == Volume*1*(N*C).
template<typename InT, typename OutT>
std::vector<OutT> Run(bool planar, int numSamples, int2 inWH, int2 outWH, int channels, NVCVInterpolationType interp,
                      bool antialias, const std::vector<InT> &interleavedIn)
{
    const nvcv::DataType inDt  = TypeAsFormat<InT>();
    const nvcv::DataType outDt = TypeAsFormat<OutT>();
    nvcv::Tensor         inT   = planar ? CreateTensorHelper(inDt, "NCHW", numSamples, channels, inWH.y, inWH.x)
                                        : CreateTensorHelper(inDt, "NHWC", numSamples, inWH.y, inWH.x, channels);
    nvcv::Tensor         outT  = planar ? CreateTensorHelper(outDt, "NCHW", numSamples, channels, outWH.y, outWH.x)
                                        : CreateTensorHelper(outDt, "NHWC", numSamples, outWH.y, outWH.x, channels);
    UploadTensor<InT>(inT, planar, numSamples, inWH, channels, interleavedIn);

    cvcuda::HQResize     op;
    HQResizeTensorShapeI inDesc{
        {inWH.y, inWH.x},
        2,
        channels
    };
    HQResizeTensorShapeI outDesc{
        {outWH.y, outWH.x},
        2,
        channels
    };
    cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
        op.getWorkspaceRequirements(numSamples, inDesc, outDesc, interp, interp, antialias));
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(op(stream, ws.get(), inT, outT, interp, interp, antialias));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    return DownloadTensor<OutT>(outT, planar, numSamples, outWH, channels);
}

} // namespace planar_parity

template<typename TypeParam>
void TestTensorPlanarParity()
{
    const int  numSamples              = ttype::GetValue<TypeParam, 0>;
    const int2 inShape                 = ttype::GetValue<TypeParam, 1>;
    const int2 outShape                = ttype::GetValue<TypeParam, 2>;
    const int  channels                = ttype::GetValue<TypeParam, 3>;
    using InBT                         = ttype::GetType<TypeParam, 4>;
    using OutBT                        = ttype::GetType<TypeParam, 5>;
    const NVCVInterpolationType interp = ttype::GetValue<TypeParam, 6>;

    ASSERT_LE(static_cast<int64_t>(inShape.x) * inShape.y, 1 << 22)
        << "planar parity cases must stay small enough to run";

    std::vector<InBT> in(static_cast<size_t>(inShape.x) * inShape.y * channels * numSamples);
    planar_parity::FillPattern<InBT>(in, 777);

    for (bool antialias : {false, true})
    {
        const std::vector<OutBT> interleavedOut
            = planar_parity::Run<InBT, OutBT>(false, numSamples, inShape, outShape, channels, interp, antialias, in);
        const std::vector<OutBT> planarOut
            = planar_parity::Run<InBT, OutBT>(true, numSamples, inShape, outShape, channels, interp, antialias, in);
        ASSERT_EQ(interleavedOut.size(), planarOut.size());
        EXPECT_EQ(interleavedOut, planarOut)
            << "planar != interleaved; channels=" << channels << " interp=" << interp << " antialias=" << antialias;
    }
}

NVCV_TYPED_TEST_SUITE(
    OpHQResizeTensor2DPlanarParity,
    ttype::Types<NVCV_TEST_ROW(2, NVCV_SHAPE2D(53, 97), NVCV_SHAPE2D(29, 31), 1, uchar, uchar, NVCV_INTERP_NEAREST),
                 NVCV_TEST_ROW(2, NVCV_SHAPE2D(57, 43), NVCV_SHAPE2D(88, 119), 2, uchar, float, NVCV_INTERP_LINEAR),
                 NVCV_TEST_ROW(3, NVCV_SHAPE2D(65, 79), NVCV_SHAPE2D(23, 41), 4, ushort, ushort, NVCV_INTERP_GAUSSIAN),
                 NVCV_TEST_ROW(2, NVCV_SHAPE2D(59, 73), NVCV_SHAPE2D(101, 89), 5, short, float, NVCV_INTERP_LANCZOS),
                 NVCV_TEST_ROW(3, NVCV_SHAPE2D(67, 37), NVCV_SHAPE2D(35, 83), 7, float, float, NVCV_INTERP_CUBIC)>);

TYPED_TEST(OpHQResizeTensor2DPlanarParity, planar_matches_interleaved)
{
    TestTensorPlanarParity<TypeParam>();
}

using HQResizeU8CubicExpandC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(74, 82), 1, uchar, uchar, NVCV_INTERP_CUBIC);
using HQResizeU8CubicContractC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(95, 129), NVCV_SHAPE2D(47, 65), 1, uchar, uchar, NVCV_INTERP_CUBIC);
using HQResizeU8LinearContractC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(95, 129), NVCV_SHAPE2D(47, 65), 1, uchar, uchar, NVCV_INTERP_LINEAR);
using HQResizeU8LinearExpandC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(74, 82), 1, uchar, uchar, NVCV_INTERP_LINEAR);
using HQResizeF32LinearContractC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(95, 129), NVCV_SHAPE2D(47, 65), 1, float, float, NVCV_INTERP_LINEAR);
using HQResizeF32LinearExpandC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(74, 82), 1, float, float, NVCV_INTERP_LINEAR);
using HQResizeU8LinearContractC3
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(95, 129), NVCV_SHAPE2D(47, 65), 3, uchar, uchar, NVCV_INTERP_LINEAR);
using HQResizeU8LinearExpandC3
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(74, 82), 3, uchar, uchar, NVCV_INTERP_LINEAR);
using HQResizeF32LinearContractC3
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(95, 129), NVCV_SHAPE2D(47, 65), 3, float, float, NVCV_INTERP_LINEAR);
using HQResizeF32LinearExpandC3
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(74, 82), 3, float, float, NVCV_INTERP_LINEAR);
using HQResizeU8CubicContractC3
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(96, 128), NVCV_SHAPE2D(48, 64), 3, uchar, uchar, NVCV_INTERP_CUBIC);
using HQResizeF32CubicContractC3
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(96, 128), NVCV_SHAPE2D(48, 64), 3, float, float, NVCV_INTERP_CUBIC);
using HQResizeU8CubicExpandC3
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(74, 82), 3, uchar, uchar, NVCV_INTERP_CUBIC);
using HQResizeF32CubicExpandC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(74, 82), 1, float, float, NVCV_INTERP_CUBIC);
using HQResizeF32CubicExpandC3
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(74, 82), 3, float, float, NVCV_INTERP_CUBIC);
using HQResizeF32CubicContractC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(95, 129), NVCV_SHAPE2D(47, 65), 1, float, float, NVCV_INTERP_CUBIC);
using HQResizeU8CubicContract2xC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(96, 128), NVCV_SHAPE2D(48, 64), 1, uchar, uchar, NVCV_INTERP_CUBIC);
using HQResizeF32CubicContract2xC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(96, 128), NVCV_SHAPE2D(48, 64), 1, float, float, NVCV_INTERP_CUBIC);
// Anisotropic (>=2x per axis, unequal factors) cubic magnification: 41->123 is 3x, 37->92 is ~2.49x.
using HQResizeU8CubicMagnifyAnisoC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(92, 123), 1, uchar, uchar, NVCV_INTERP_CUBIC);
using HQResizeU8CubicMagnifyAnisoC3
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(92, 123), 3, uchar, uchar, NVCV_INTERP_CUBIC);
using HQResizeF32CubicMagnifyAnisoC1
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(92, 123), 1, float, float, NVCV_INTERP_CUBIC);
using HQResizeF32CubicMagnifyAnisoC3
    = NVCV_TEST_ROW(2, NVCV_SHAPE2D(37, 41), NVCV_SHAPE2D(92, 123), 3, float, float, NVCV_INTERP_CUBIC);

NVCV_TYPED_TEST_SUITE(
    OpHQResizeTensor2DOptimizedReferencePaths,
    ttype::Types<
        HQResizeU8CubicExpandC1, HQResizeU8CubicContractC1, HQResizeF32CubicExpandC1, HQResizeU8LinearContractC1,
        HQResizeU8LinearExpandC1, HQResizeF32LinearContractC1, HQResizeF32LinearExpandC1, HQResizeU8LinearContractC3,
        HQResizeU8LinearExpandC3, HQResizeF32LinearContractC3, HQResizeF32LinearExpandC3, HQResizeU8CubicContractC3,
        HQResizeF32CubicExpandC3, HQResizeF32CubicContractC3, HQResizeF32CubicContractC1, HQResizeU8CubicExpandC3,
        HQResizeU8CubicContract2xC1, HQResizeF32CubicContract2xC1, HQResizeU8CubicMagnifyAnisoC1,
        HQResizeU8CubicMagnifyAnisoC3, HQResizeF32CubicMagnifyAnisoC1, HQResizeF32CubicMagnifyAnisoC3>);

TYPED_TEST(OpHQResizeTensor2DOptimizedReferencePaths, matches_reference)
{
    TestTensor<TypeParam>(false);
    TestTensor<TypeParam>(true);
}

TEST(OpHQResizeTensor2DOptimizedReferencePaths, direct_linear_u8_matches_reference_bit_exact)
{
    TestTensor<HQResizeU8LinearContractC1>(false, true);
    TestTensor<HQResizeU8LinearExpandC1>(false, true);
}

template<typename BT>
void VerticalFlip(baseline::CpuSample<BT, 2> &outTensorCpu, baseline::CpuSample<BT, 2> &inTensorCpu)
{
    const int  numSamples  = inTensorCpu.numSamples();
    const int  numChannels = inTensorCpu.numChannels();
    const int2 shape       = inTensorCpu.shape();

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        baseline::ForAllInRoi(baseline::FullRoi<2>(shape),
                              [shape, numChannels, sampleIdx, &outTensorCpu, &inTensorCpu](int2 idx)
                              {
                                  int2 srcIdx = idx;
                                  srcIdx.y    = shape.y - 1 - idx.y;
                                  for (int c = 0; c < numChannels; c++)
                                  {
                                      outTensorCpu.get(sampleIdx, idx, c) = inTensorCpu.get(sampleIdx, srcIdx, c);
                                  }
                              });
    }
}

static void TestTensorFlippedRoiReference(NVCVInterpolationType interpolation, bool antialias)
{
    constexpr int numSamples  = 2;
    constexpr int numChannels = 1;
    using InBT                = uchar;
    using OutBT               = uchar;
    const int2 inShape{129, 95};
    const int2 outShape{65, 47};
    const auto inDtype  = TypeAsFormat<InBT>();
    const auto outDtype = TypeAsFormat<OutBT>();

    nvcv::Tensor inTensor  = CreateTensorHelper(inDtype, "NHWC", numSamples, inShape.y, inShape.x, numChannels);
    nvcv::Tensor outTensor = CreateTensorHelper(outDtype, "NHWC", numSamples, outShape.y, outShape.x, numChannels);

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inData && outData);
    auto samples = MakeTensor2DTestSamples<InBT, OutBT>(*inData, *outData, inShape, outShape, numSamples, numChannels);
    baseline::CpuSample<InBT, 2> flippedInputCpu(samples.inStrides.z * numSamples, samples.inStrides, numSamples,
                                                 inShape, numChannels);

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        baseline::ForAllInRoi(baseline::FullRoi<2>(inShape),
                              [sampleIdx, &samples](int2 idx)
                              {
                                  uint64_t h = 0x9e3779b97f4a7c15ULL;
                                  h ^= static_cast<uint64_t>(sampleIdx + 1) * 0xbf58476d1ce4e5b9ULL;
                                  h ^= static_cast<uint64_t>(idx.x + 1) * 0x94d049bb133111ebULL;
                                  h ^= static_cast<uint64_t>(idx.y + 1) * 0xd2b74407b1ce6e93ULL;
                                  h ^= h >> 31;
                                  samples.inCpu.get(sampleIdx, idx, 0) = static_cast<InBT>(h);
                              });
    }

    HQResizeRoiF roi{};
    roi.lo[0] = static_cast<float>(inShape.y);
    roi.hi[0] = 0.f;
    roi.lo[1] = 0.f;
    roi.hi[1] = static_cast<float>(inShape.x);

    cvcuda::HQResize        op;
    cvcuda::UniqueWorkspace ws;
    HQResizeTensorShapeI    inShapeDesc{
        {inShape.y, inShape.x},
        2,
        numChannels
    };
    HQResizeTensorShapeI outShapeDesc{
        {outShape.y, outShape.x},
        2,
        numChannels
    };
    ASSERT_NO_THROW(ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(
                        numSamples, inShapeDesc, outShapeDesc, interpolation, interpolation, antialias, &roi)));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inData->basePtr(), samples.inCpu.data(), samples.inStrides.z * numSamples,
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_NO_THROW(op(stream, ws.get(), inTensor, outTensor, interpolation, interpolation, antialias, &roi));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(samples.outCpu.data(), outData->basePtr(), samples.outStrides.z * numSamples,
                                           cudaMemcpyDeviceToHost, stream));

    VerticalFlip(flippedInputCpu, samples.inCpu);
    baseline::Resize(samples.refCpu, flippedInputCpu, interpolation, interpolation, antialias);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    baseline::Compare<InBT>(samples.outCpu, samples.refCpu, antialias);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpHQResizeTensor2DOptimizedReferencePaths, flipped_roi_contract_matches_reference)
{
    TestTensorFlippedRoiReference(NVCV_INTERP_LINEAR, false);
    TestTensorFlippedRoiReference(NVCV_INTERP_LINEAR, true);
    TestTensorFlippedRoiReference(NVCV_INTERP_CUBIC, false);
    TestTensorFlippedRoiReference(NVCV_INTERP_CUBIC, true);
}

// The fused u8 kernels read the source window as aligned-down 32-bit words; a sample
// whose base pointer is not word-aligned must produce output bit-identical to the
// aligned run without touching bytes below the first row.
static void TestTensorUnalignedBaseReference(NVCVInterpolationType interpolation, const int2 inShape,
                                             const int2 outShape)
{
    constexpr int numChannels = 3;
    using BT                  = uchar;
    const auto dtype          = TypeAsFormat<BT>();

    nvcv::Tensor alignedIn      = CreateTensorHelper(dtype, "HWC", 1, inShape.y, inShape.x, numChannels);
    nvcv::Tensor alignedOut     = CreateTensorHelper(dtype, "HWC", 1, outShape.y, outShape.x, numChannels);
    auto         alignedInData  = alignedIn.exportData<nvcv::TensorDataStridedCuda>();
    auto         alignedOutData = alignedOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(alignedInData && alignedOutData);
    auto samples = MakeTensor2DTestSamples<BT, BT>(*alignedInData, *alignedOutData, inShape, outShape, 1, numChannels);

    baseline::ForAllInRoi(baseline::FullRoi<2>(inShape),
                          [&samples](int2 idx)
                          {
                              for (int c = 0; c < numChannels; c++)
                              {
                                  uint64_t h = 0x9e3779b97f4a7c15ULL;
                                  h ^= static_cast<uint64_t>(idx.x + 1) * 0x94d049bb133111ebULL;
                                  h ^= static_cast<uint64_t>(idx.y + 1) * 0xd2b74407b1ce6e93ULL;
                                  h ^= static_cast<uint64_t>(c + 1) * 0xbf58476d1ce4e5b9ULL;
                                  h ^= h >> 31;
                                  samples.inCpu.get(0, idx, c) = static_cast<BT>(h);
                              }
                          });

    const int64_t colStride    = static_cast<int64_t>(numChannels) * sizeof(BT);
    const int64_t inRowStride  = inShape.x * colStride;
    const int64_t outRowStride = outShape.x * colStride;
    const int64_t inBytes      = inShape.y * inRowStride;
    const int64_t outBytes     = outShape.y * outRowStride;

    NVCVByte *inAlloc  = nullptr;
    NVCVByte *outAlloc = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&inAlloc), inBytes + 4));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&outAlloc), outBytes + 4));

    auto makeUnaligned = [dtype](NVCVByte *alloc, const int2 wh, const int64_t rowStrideArg, const int64_t colStrideArg)
    {
        nvcv::TensorDataStridedCuda::Buffer buf{};
        buf.strides[0] = rowStrideArg;
        buf.strides[1] = colStrideArg;
        buf.strides[2] = sizeof(BT);
        buf.basePtr    = alloc + 1;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{wh.y, wh.x, numChannels}, "HWC"},
            dtype, buf
        });
    };
    nvcv::Tensor unalignedIn  = makeUnaligned(inAlloc, inShape, inRowStride, colStride);
    nvcv::Tensor unalignedOut = makeUnaligned(outAlloc, outShape, outRowStride, colStride);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(alignedInData->basePtr(), samples.inCpu.data(), samples.inStrides.z,
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(inAlloc + 1, inRowStride, samples.inCpu.data(), samples.inStrides.y,
                                             inRowStride, inShape.y, cudaMemcpyHostToDevice, stream));

    cvcuda::HQResize     op;
    HQResizeTensorShapeI inShapeDesc{
        {inShape.y, inShape.x},
        2,
        numChannels
    };
    HQResizeTensorShapeI outShapeDesc{
        {outShape.y, outShape.x},
        2,
        numChannels
    };
    cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
        op.getWorkspaceRequirements(1, inShapeDesc, outShapeDesc, interpolation, interpolation, false));
    ASSERT_NO_THROW(op(stream, ws.get(), alignedIn, alignedOut, interpolation, interpolation, false));
    ASSERT_NO_THROW(op(stream, ws.get(), unalignedIn, unalignedOut, interpolation, interpolation, false));

    const long3                outStrides{colStride, outRowStride, outBytes};
    baseline::CpuSample<BT, 2> unalignedOutCpu(outBytes, outStrides, 1, outShape, numChannels);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(samples.outCpu.data(), alignedOutData->basePtr(), samples.outStrides.z,
                                           cudaMemcpyDeviceToHost, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(unalignedOutCpu.data(), outRowStride, outAlloc + 1, outRowStride,
                                             outRowStride, outShape.y, cudaMemcpyDeviceToHost, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    baseline::Resize(samples.refCpu, samples.inCpu, interpolation, interpolation, false);
    baseline::Compare<BT>(samples.outCpu, samples.refCpu, false);
    baseline::CompareExact(unalignedOutCpu, samples.outCpu);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(inAlloc));
    ASSERT_EQ(cudaSuccess, cudaFree(outAlloc));
}

TEST(OpHQResizeTensor2DOptimizedReferencePaths, unaligned_base_u8_matches_reference)
{
    TestTensorUnalignedBaseReference(NVCV_INTERP_CUBIC, int2{96, 128}, int2{48, 64});
    TestTensorUnalignedBaseReference(NVCV_INTERP_CUBIC, int2{129, 95}, int2{65, 47});
    TestTensorUnalignedBaseReference(NVCV_INTERP_LINEAR, int2{129, 95}, int2{65, 47});
}

NVCV_TYPED_TEST_SUITE(OpHQResizeTensor2DOptimizedPlanarPaths,
                      ttype::Types<HQResizeU8CubicExpandC3, HQResizeF32CubicExpandC3, HQResizeU8CubicContractC3,
                                   HQResizeF32CubicContractC3>);

TYPED_TEST(OpHQResizeTensor2DOptimizedPlanarPaths, planar_matches_interleaved)
{
    TestTensorPlanarParity<TypeParam>();
}

#define NVCV_SHAPE3D(d, h, w) (int3{w, h, d})
NVCV_TYPED_TEST_SUITE(
    OpHQResizeTensor3D,
    ttype::Types<
        NVCV_TEST_ROW(1, NVCV_SHAPE3D(244, 244, 244), NVCV_SHAPE3D(40, 40, 40), 1, uchar, uchar, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(2, NVCV_SHAPE3D(40, 40, 40), NVCV_SHAPE3D(244, 244, 244), 2, uchar, float, NVCV_INTERP_GAUSSIAN),
        NVCV_TEST_ROW(3, NVCV_SHAPE3D(100, 100, 100), NVCV_SHAPE3D(50, 100, 100), 3, ushort, ushort, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(4, NVCV_SHAPE3D(100, 100, 100), NVCV_SHAPE3D(100, 50, 100), 4, ushort, float, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(3, NVCV_SHAPE3D(100, 100, 100), NVCV_SHAPE3D(100, 100, 50), 3, float, float, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(4, NVCV_SHAPE3D(40, 40, 40), NVCV_SHAPE3D(100, 40, 40), 5, uchar, float, NVCV_INTERP_LANCZOS),
        NVCV_TEST_ROW(7, NVCV_SHAPE3D(40, 40, 40), NVCV_SHAPE3D(50, 150, 100), 3, uchar, uchar, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(3, NVCV_SHAPE3D(1 << 10, 1 << 9, 1 << 9), NVCV_SHAPE3D(100, 150, 100), 3, uchar, uchar,
                      NVCV_INTERP_CUBIC)>);

TYPED_TEST(OpHQResizeTensor3D, correct_output_with_antialias)
{
    const int  numSamples                     = ttype::GetValue<TypeParam, 0>;
    const int3 inShape                        = ttype::GetValue<TypeParam, 1>;
    const int3 outShape                       = ttype::GetValue<TypeParam, 2>;
    const int  numChannels                    = ttype::GetValue<TypeParam, 3>;
    using InBT                                = ttype::GetType<TypeParam, 4>;
    using OutBT                               = ttype::GetType<TypeParam, 5>;
    const nvcv::DataType        inDtype       = TypeAsFormat<InBT>();
    const nvcv::DataType        outDtype      = TypeAsFormat<OutBT>();
    const NVCVInterpolationType interpolation = ttype::GetValue<TypeParam, 6>;
    constexpr bool              antialias     = true;

    nvcv::Tensor inTensor
        = CreateTensorHelper(inDtype, "NDHWC", numSamples, inShape.z, inShape.y, inShape.x, numChannels);
    nvcv::Tensor outTensor
        = CreateTensorHelper(outDtype, "NDHWC", numSamples, outShape.z, outShape.y, outShape.x, numChannels);

    baseline::Roi<3> inRoi  = baseline::FullRoi<3>(inShape);
    baseline::Roi<3> outRoi = baseline::FullRoi<3>(outShape);
    if (inShape.x * inShape.y * inShape.z > 1 << 22)
    {
        inRoi.shape  = cuda::min(inShape, int3{1 << 8, 1 << 7, 1 << 7});
        inRoi.origin = inShape - inRoi.shape;

        double3 scale = cuda::StaticCast<double>(outShape) / cuda::StaticCast<double>(inShape);
        outRoi.shape  = cuda::StaticCast<int>(scale * cuda::StaticCast<double>(inRoi.shape));
        outRoi.origin = cuda::StaticCast<int>(scale * cuda::StaticCast<double>(inRoi.origin));
    }

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inData && outData);

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(inAccess && outAccess);
    long4_16a inStrides{inAccess->colStride(), inAccess->rowStride(), inAccess->depthStride(),
                        inAccess->sampleStride() == 0 ? inAccess->depthStride() * inShape.z : inAccess->sampleStride()};
    long4_16a outStrides{
        outAccess->colStride(), outAccess->rowStride(), outAccess->depthStride(),
        outAccess->sampleStride() == 0 ? outAccess->depthStride() * outShape.z : outAccess->sampleStride()};

    ASSERT_EQ(inAccess->numSamples(), numSamples);
    ASSERT_EQ(inAccess->numChannels(), numChannels);
    ASSERT_EQ(outAccess->numChannels(), numChannels);
    ASSERT_EQ(outAccess->numSamples(), numSamples);

    baseline::CpuSample<InBT, 3>  inTensorCpu(inStrides.w * numSamples, inStrides, numSamples, inShape, numChannels);
    baseline::CpuSample<OutBT, 3> outTensorCpu(outStrides.w * numSamples, outStrides, numSamples, outShape,
                                               numChannels);
    baseline::CpuSample<OutBT, 3> refTensorCpu(outStrides.w * numSamples, outStrides, numSamples, outShape,
                                               numChannels);

    uniform_distribution<InBT> rand(InBT{0}, std::is_integral_v<InBT> ? cuda::TypeTraits<InBT>::max : InBT{1});
    std::mt19937_64            rng(12345);

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        baseline::ForAllInRoi(inRoi,
                              [sampleIdx, &inTensorCpu, &rand, &rng](int3 idx)
                              {
                                  for (int c = 0; c < numChannels; c++)
                                  {
                                      inTensorCpu.get(sampleIdx, idx, c) = rand(rng);
                                  }
                              });
    }

    cvcuda::HQResize        op;
    cudaStream_t            stream;
    cvcuda::UniqueWorkspace ws;
    {
        HQResizeTensorShapeI inShapeDesc{
            {inShape.z, inShape.y, inShape.x},
            3,
            numChannels
        };
        HQResizeTensorShapeI outShapeDesc{
            {outShape.z, outShape.y, outShape.x},
            3,
            numChannels
        };
        ASSERT_NO_THROW(ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(
                            numSamples, inShapeDesc, outShapeDesc, interpolation, interpolation, antialias)));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inData->basePtr(), inTensorCpu.data(), inStrides.w * numSamples,
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_NO_THROW(op(stream, ws.get(), inTensor, outTensor, interpolation, interpolation, antialias));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(outTensorCpu.data(), outData->basePtr(), outStrides.w * numSamples,
                                           cudaMemcpyDeviceToHost, stream));
    baseline::Resize(refTensorCpu, inTensorCpu, interpolation, interpolation, antialias, {inRoi}, {outRoi});
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    baseline::Compare<InBT>(outTensorCpu, refTensorCpu, antialias, {outRoi});
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

#define NVCV_TEST_ROW_TB(NumChannels, InT, OutT, Antialias, MinInterpolation, MagInterpolation, LargeSample)    \
    ttype::Types<ttype::Value<NumChannels>, InT, OutT, ttype::Value<Antialias>, ttype::Value<MinInterpolation>, \
                 ttype::Value<MagInterpolation>, ttype::Value<LargeSample>>

NVCV_TYPED_TEST_SUITE(
    OpHQResizeBatch,
    ttype::Types<NVCV_TEST_ROW_TB(1, uchar, float, false, NVCV_INTERP_LANCZOS, NVCV_INTERP_LANCZOS, false),
                 NVCV_TEST_ROW_TB(2, uchar, uchar, true, NVCV_INTERP_LANCZOS, NVCV_INTERP_CUBIC, false),
                 NVCV_TEST_ROW_TB(3, uchar, float, false, NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, false),
                 NVCV_TEST_ROW_TB(4, uchar, uchar, true, NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR, true),
                 NVCV_TEST_ROW_TB(-1, uchar, uchar, false, NVCV_INTERP_CUBIC, NVCV_INTERP_NEAREST, false),
                 NVCV_TEST_ROW_TB(1, ushort, ushort, false, NVCV_INTERP_CUBIC, NVCV_INTERP_CUBIC, false),
                 NVCV_TEST_ROW_TB(2, short, float, false, NVCV_INTERP_LANCZOS, NVCV_INTERP_LINEAR, false),
                 NVCV_TEST_ROW_TB(3, float, float, true, NVCV_INTERP_LINEAR, NVCV_INTERP_GAUSSIAN, false),
                 NVCV_TEST_ROW_TB(-1, float, float, true, NVCV_INTERP_LINEAR, NVCV_INTERP_NEAREST, false),
                 NVCV_TEST_ROW_TB(-1, float, float, true, NVCV_INTERP_LINEAR, NVCV_INTERP_NEAREST, true)>);

TYPED_TEST(OpHQResizeBatch, tensor_batch_2d_correct_output)
{
    const int numChannels                        = ttype::GetValue<TypeParam, 0>;
    using InBT                                   = ttype::GetType<TypeParam, 1>;
    using OutBT                                  = ttype::GetType<TypeParam, 2>;
    const nvcv::DataType        inDtype          = TypeAsFormat<InBT>();
    const nvcv::DataType        outDtype         = TypeAsFormat<OutBT>();
    const bool                  antialias        = ttype::GetValue<TypeParam, 3>;
    const NVCVInterpolationType minInterpolation = ttype::GetValue<TypeParam, 4>;
    const NVCVInterpolationType magInterpolation = ttype::GetValue<TypeParam, 5>;
    const bool                  largeSample      = ttype::GetValue<TypeParam, 6>;

    constexpr int                         numSamples  = 5;
    constexpr std::array<int, numSamples> varChannels = {4, 1, 7, 3, 5};

    std::array<int, 2> inShape1 = {1 << 14, 1 << 13};
    if (sizeof(InBT) == 1)
    {
        inShape1[0] *= 2;
        inShape1[1] *= 2;
    }

    const int sample1Channels = numChannels > 0 ? numChannels : varChannels[0];
    auto      sample1         = largeSample ? HQResizeTensorShapeI{{inShape1[0], inShape1[1]}, 2, sample1Channels}
                                            : HQResizeTensorShapeI{{728, 1024, 0}, 2, sample1Channels};

    std::vector<HQResizeTensorShapeI> inShapes = {
        sample1,
        {{512, 512}, 2, numChannels > 0 ? numChannels : varChannels[1]},
        {{128, 256}, 2, numChannels > 0 ? numChannels : varChannels[2]},
        {{256, 128}, 2, numChannels > 0 ? numChannels : varChannels[3]},
        {  {40, 40}, 2, numChannels > 0 ? numChannels : varChannels[4]}
    };

    std::vector<HQResizeTensorShapeI> outShapes = {
        {{512, 245}, 2, inShapes[0].numChannels},
        { {250, 51}, 2, inShapes[1].numChannels},
        {{243, 128}, 2, inShapes[2].numChannels},
        {{128, 256}, 2, inShapes[3].numChannels},
        {{512, 512}, 2, inShapes[4].numChannels}
    };

    std::vector<baseline::Roi<2>> inRois(numSamples);
    std::vector<baseline::Roi<2>> outRois(numSamples);

    for (int s = 0; s < numSamples; s++)
    {
        int2 inShape{inShapes[s].extent[1], inShapes[s].extent[0]};
        int2 outShape{outShapes[s].extent[1], outShapes[s].extent[0]};

        inRois[s]  = baseline::FullRoi<2>(inShape);
        outRois[s] = baseline::FullRoi<2>(outShape);

        if (inShape.x * inShape.y > 1 << 23)
        {
            inRois[s].shape  = cuda::min(inShape, int2{1 << 12, 1 << 11});
            inRois[s].origin = inShape - inRois[s].shape;

            double2 scale     = cuda::StaticCast<double>(outShape) / cuda::StaticCast<double>(inShape);
            outRois[s].shape  = cuda::StaticCast<int>(scale * cuda::StaticCast<double>(inRois[s].shape));
            outRois[s].origin = cuda::StaticCast<int>(scale * cuda::StaticCast<double>(inRois[s].origin));
        }
    }

    ASSERT_EQ(numSamples, inShapes.size());
    ASSERT_EQ(numSamples, outShapes.size());

    nvcv::TensorBatch inTensors(numSamples);
    nvcv::TensorBatch outTensors(numSamples);
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        if (numChannels == 1)
        {
            inTensors.pushBack(
                CreateTensorHelper(inDtype, "HW", 1, inShapes[sampleIdx].extent[0], inShapes[sampleIdx].extent[1]));
            outTensors.pushBack(
                CreateTensorHelper(outDtype, "HW", 1, outShapes[sampleIdx].extent[0], outShapes[sampleIdx].extent[1]));
        }
        else
        {
            inTensors.pushBack(CreateTensorHelper(inDtype, "HWC", 1, inShapes[sampleIdx].extent[0],
                                                  inShapes[sampleIdx].extent[1], inShapes[sampleIdx].numChannels));
            outTensors.pushBack(CreateTensorHelper(outDtype, "HWC", 1, outShapes[sampleIdx].extent[0],
                                                   outShapes[sampleIdx].extent[1], outShapes[sampleIdx].numChannels));
        }
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    uniform_distribution<InBT> rand(InBT{0}, std::is_integral_v<InBT> ? cuda::TypeTraits<InBT>::max : InBT{1});
    std::mt19937_64            rng(12345);

    std::vector<baseline::CpuSample<InBT, 2>>  inBatchCpu;
    std::vector<baseline::CpuSample<OutBT, 2>> outBatchCpu;
    std::vector<baseline::CpuSample<OutBT, 2>> refBatchCpu;
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        SCOPED_TRACE(sampleIdx);
        auto inData  = inTensors[sampleIdx].exportData<nvcv::TensorDataStridedCuda>();
        auto outData = outTensors[sampleIdx].exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(inData && outData);

        auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
        ASSERT_TRUE(inAccess && outAccess);

        long3 inStrides{inAccess->colStride(), inAccess->rowStride(),
                        inAccess->sampleStride() == 0 ? inAccess->rowStride() * inShapes[sampleIdx].extent[0]
                                                      : inAccess->sampleStride()};
        long3 outStrides{outAccess->colStride(), outAccess->rowStride(),
                         outAccess->sampleStride() == 0 ? outAccess->rowStride() * outShapes[sampleIdx].extent[0]
                                                        : outAccess->sampleStride()};

        ASSERT_EQ(inAccess->numSamples(), 1);
        ASSERT_EQ(outAccess->numSamples(), 1);
        ASSERT_EQ(inAccess->numChannels(), inShapes[sampleIdx].numChannels);
        ASSERT_EQ(outAccess->numChannels(), outShapes[sampleIdx].numChannels);

        int2 inShape{inShapes[sampleIdx].extent[1], inShapes[sampleIdx].extent[0]};
        int2 outShape{outShapes[sampleIdx].extent[1], outShapes[sampleIdx].extent[0]};
        inBatchCpu.push_back(baseline::CpuSample<InBT, 2>{inStrides.z, inStrides, 1, inShape, inAccess->numChannels()});
        outBatchCpu.push_back(
            baseline::CpuSample<OutBT, 2>{outStrides.z, outStrides, 1, outShape, outAccess->numChannels()});
        refBatchCpu.push_back(
            baseline::CpuSample<OutBT, 2>{outStrides.z, outStrides, 1, outShape, outAccess->numChannels()});

        const auto &inRoi       = inRois[sampleIdx];
        auto       &inTensorCpu = inBatchCpu[sampleIdx];
        baseline::ForAllInRoi(inRoi,
                              [sampleIdx, &inShapes, &inTensorCpu, &rand, &rng](int2 idx)
                              {
                                  for (int c = 0; c < inShapes[sampleIdx].numChannels; c++)
                                  {
                                      inTensorCpu.get(0, idx, c) = rand(rng);
                                  }
                              });

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpyAsync(inData->basePtr(), inTensorCpu.data(), inStrides.z, cudaMemcpyHostToDevice, stream));
    }

    cvcuda::HQResize        op;
    cvcuda::UniqueWorkspace ws;

    {
        HQResizeTensorShapesI inShapeDesc{inShapes.data(), numSamples, 2, numChannels};
        HQResizeTensorShapesI outShapeDesc{outShapes.data(), numSamples, 2, numChannels};
        ASSERT_NO_THROW(ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(
                            numSamples, inShapeDesc, outShapeDesc, minInterpolation, magInterpolation, antialias)));
    }
    ASSERT_NO_THROW(op(stream, ws.get(), inTensors, outTensors, minInterpolation, magInterpolation, antialias));

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        SCOPED_TRACE(sampleIdx);
        auto outData = outTensors[sampleIdx].exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(outData);
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(outBatchCpu[sampleIdx].data(), outData->basePtr(),
                                               outBatchCpu[sampleIdx].strides().z, cudaMemcpyDeviceToHost, stream));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        SCOPED_TRACE(sampleIdx);
        baseline::Resize(refBatchCpu[sampleIdx], inBatchCpu[sampleIdx], minInterpolation, magInterpolation, antialias,
                         {inRois[sampleIdx]}, {outRois[sampleIdx]});
        baseline::Compare<InBT>(outBatchCpu[sampleIdx], refBatchCpu[sampleIdx], antialias, {outRois[sampleIdx]});
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TYPED_TEST(OpHQResizeBatch, tensor_batch_3d_correct_output)
{
    const int numChannels                        = ttype::GetValue<TypeParam, 0>;
    using InBT                                   = ttype::GetType<TypeParam, 1>;
    using OutBT                                  = ttype::GetType<TypeParam, 2>;
    const nvcv::DataType        inDtype          = TypeAsFormat<InBT>();
    const nvcv::DataType        outDtype         = TypeAsFormat<OutBT>();
    const bool                  antialias        = ttype::GetValue<TypeParam, 3>;
    const NVCVInterpolationType minInterpolation = ttype::GetValue<TypeParam, 4>;
    const NVCVInterpolationType magInterpolation = ttype::GetValue<TypeParam, 4>;

    constexpr int                         numSamples  = 5;
    constexpr std::array<int, numSamples> varChannels = {6, 2, 3, 4, 1};

    std::vector<HQResizeTensorShapeI> inShapes = {
        {{128, 128, 128}, 3, numChannels > 0 ? numChannels : varChannels[0]},
        {  {512, 40, 40}, 3, numChannels > 0 ? numChannels : varChannels[1]},
        {  {40, 512, 40}, 3, numChannels > 0 ? numChannels : varChannels[2]},
        {  {40, 40, 512}, 3, numChannels > 0 ? numChannels : varChannels[3]},
        {   {40, 40, 40}, 3, numChannels > 0 ? numChannels : varChannels[4]}
    };
    std::vector<HQResizeTensorShapeI> outShapes = {
        {   {45, 64, 50}, 3, inShapes[0].numChannels},
        {   {40, 40, 40}, 3, inShapes[1].numChannels},
        {   {40, 40, 40}, 3, inShapes[2].numChannels},
        {   {40, 40, 40}, 3, inShapes[3].numChannels},
        {{128, 128, 128}, 3, inShapes[4].numChannels}
    };

    ASSERT_EQ(numSamples, inShapes.size());
    ASSERT_EQ(numSamples, outShapes.size());

    nvcv::TensorBatch inTensors(numSamples);
    nvcv::TensorBatch outTensors(numSamples);
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        if (numChannels == 1)
        {
            inTensors.pushBack(CreateTensorHelper(inDtype, "DHW", 1, inShapes[sampleIdx].extent[0],
                                                  inShapes[sampleIdx].extent[1], inShapes[sampleIdx].extent[2]));
            outTensors.pushBack(CreateTensorHelper(outDtype, "DHW", 1, outShapes[sampleIdx].extent[0],
                                                   outShapes[sampleIdx].extent[1], outShapes[sampleIdx].extent[2]));
        }
        else
        {
            inTensors.pushBack(CreateTensorHelper(inDtype, "DHWC", 1, inShapes[sampleIdx].extent[0],
                                                  inShapes[sampleIdx].extent[1], inShapes[sampleIdx].extent[2],
                                                  inShapes[sampleIdx].numChannels));
            outTensors.pushBack(CreateTensorHelper(outDtype, "DHWC", 1, outShapes[sampleIdx].extent[0],
                                                   outShapes[sampleIdx].extent[1], outShapes[sampleIdx].extent[2],
                                                   outShapes[sampleIdx].numChannels));
        }
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    uniform_distribution<InBT> rand(InBT{0}, std::is_integral_v<InBT> ? cuda::TypeTraits<InBT>::max : InBT{1});
    std::mt19937_64            rng(12345);

    std::vector<baseline::CpuSample<InBT, 3>>  inBatchCpu;
    std::vector<baseline::CpuSample<OutBT, 3>> outBatchCpu;
    std::vector<baseline::CpuSample<OutBT, 3>> refBatchCpu;
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        SCOPED_TRACE(sampleIdx);
        auto inData  = inTensors[sampleIdx].exportData<nvcv::TensorDataStridedCuda>();
        auto outData = outTensors[sampleIdx].exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(inData && outData);

        auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
        ASSERT_TRUE(inAccess && outAccess);

        long4_16a inStrides{inAccess->colStride(), inAccess->rowStride(), inAccess->depthStride(),
                            inAccess->sampleStride() == 0 ? inAccess->depthStride() * inShapes[sampleIdx].extent[0]
                                                          : inAccess->sampleStride()};
        long4_16a outStrides{outAccess->colStride(), outAccess->rowStride(), outAccess->depthStride(),
                             outAccess->sampleStride() == 0 ? outAccess->depthStride() * outShapes[sampleIdx].extent[0]
                                                            : outAccess->sampleStride()};

        ASSERT_EQ(inAccess->numSamples(), 1);
        ASSERT_EQ(outAccess->numSamples(), 1);
        ASSERT_EQ(inAccess->numChannels(), inShapes[sampleIdx].numChannels);
        ASSERT_EQ(outAccess->numChannels(), outShapes[sampleIdx].numChannels);

        int3 inShape{inShapes[sampleIdx].extent[2], inShapes[sampleIdx].extent[1], inShapes[sampleIdx].extent[0]};
        int3 outShape{outShapes[sampleIdx].extent[2], outShapes[sampleIdx].extent[1], outShapes[sampleIdx].extent[0]};
        inBatchCpu.push_back(baseline::CpuSample<InBT, 3>{inStrides.w, inStrides, 1, inShape, inAccess->numChannels()});
        outBatchCpu.push_back(
            baseline::CpuSample<OutBT, 3>{outStrides.w, outStrides, 1, outShape, outAccess->numChannels()});
        refBatchCpu.push_back(
            baseline::CpuSample<OutBT, 3>{outStrides.w, outStrides, 1, outShape, outAccess->numChannels()});

        auto &inTensorCpu = inBatchCpu[sampleIdx];
        baseline::ForAllInRoi(baseline::FullRoi<3>(inShape),
                              [&inShapes, &inTensorCpu, &rand, &rng, sampleIdx](const int3 idx)
                              {
                                  for (int c = 0; c < inShapes[sampleIdx].numChannels; c++)
                                  {
                                      inTensorCpu.get(0, idx, c) = rand(rng);
                                  }
                              });
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpyAsync(inData->basePtr(), inTensorCpu.data(), inStrides.w, cudaMemcpyHostToDevice, stream));
    }

    cvcuda::HQResize        op;
    cvcuda::UniqueWorkspace ws;

    {
        HQResizeTensorShapesI inShapeDesc{inShapes.data(), numSamples, 3, numChannels};
        HQResizeTensorShapesI outShapeDesc{outShapes.data(), numSamples, 3, numChannels};
        ASSERT_NO_THROW(ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(
                            numSamples, inShapeDesc, outShapeDesc, minInterpolation, magInterpolation, antialias)));
    }
    ASSERT_NO_THROW(op(stream, ws.get(), inTensors, outTensors, minInterpolation, magInterpolation, antialias));

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        SCOPED_TRACE(sampleIdx);
        auto outData = outTensors[sampleIdx].exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(outData);
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(outBatchCpu[sampleIdx].data(), outData->basePtr(),
                                               outBatchCpu[sampleIdx].strides().w, cudaMemcpyDeviceToHost, stream));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        SCOPED_TRACE(sampleIdx);
        baseline::Resize(refBatchCpu[sampleIdx], inBatchCpu[sampleIdx], minInterpolation, magInterpolation, antialias);
        baseline::Compare<InBT>(outBatchCpu[sampleIdx], refBatchCpu[sampleIdx], antialias);
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

#define NVCV_IMAGE_FORMAT_RGB16U \
    NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, UNSIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_TEST_ROW_IB(NumChannels, InT, InFormat, OutT, OutFormat, Antialias, MinInterpolation, MagInterpolation) \
    ttype::Types<ttype::Value<NumChannels>, InT, ttype::Value<InFormat>, OutT, ttype::Value<OutFormat>,              \
                 ttype::Value<Antialias>, ttype::Value<MinInterpolation>, ttype::Value<MagInterpolation>>

NVCV_TYPED_TEST_SUITE(
    OpHQResizeImageBatch,
    ttype::Types<NVCV_TEST_ROW_IB(1, uchar, NVCV_IMAGE_FORMAT_U8, uchar, NVCV_IMAGE_FORMAT_U8, false,
                                  NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC),
                 NVCV_TEST_ROW_IB(3, uchar3, NVCV_IMAGE_FORMAT_RGB8, float3, NVCV_IMAGE_FORMAT_RGBf32, true,
                                  NVCV_INTERP_LANCZOS, NVCV_INTERP_LINEAR),
                 NVCV_TEST_ROW_IB(3, ushort3, NVCV_IMAGE_FORMAT_RGB16U, float3, NVCV_IMAGE_FORMAT_RGBf32, true,
                                  NVCV_INTERP_LANCZOS, NVCV_INTERP_LINEAR),
                 NVCV_TEST_ROW_IB(4, uchar4, NVCV_IMAGE_FORMAT_RGBA8, uchar4, NVCV_IMAGE_FORMAT_RGBA8, true,
                                  NVCV_INTERP_LINEAR, NVCV_INTERP_GAUSSIAN),
                 NVCV_TEST_ROW_IB(4, float4, NVCV_IMAGE_FORMAT_RGBAf32, float4, NVCV_IMAGE_FORMAT_RGBAf32, false,
                                  NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR)>);

template<typename TypeParam>
void TestImageBatch(int numSamples, std::vector<HQResizeTensorShapeI> &inShapes,
                    std::vector<HQResizeTensorShapeI> &outShapes, cvcuda::UniqueWorkspace &ws,
                    bool allocateWorkspace = true)
{
    const int numChannels = ttype::GetValue<TypeParam, 0>;
    using InT             = ttype::GetType<TypeParam, 1>;
    using InBT            = cuda::BaseType<InT>;
    using OutT            = ttype::GetType<TypeParam, 3>;
    using OutBT           = cuda::BaseType<OutT>;
    const nvcv::ImageFormat     inImgFormat{ttype::GetValue<TypeParam, 2>};
    const nvcv::ImageFormat     outImgFormat{ttype::GetValue<TypeParam, 4>};
    const bool                  antialias        = ttype::GetValue<TypeParam, 5>;
    const NVCVInterpolationType minInterpolation = ttype::GetValue<TypeParam, 6>;
    const NVCVInterpolationType magInterpolation = ttype::GetValue<TypeParam, 7>;

    ASSERT_GE(numChannels, 1);
    ASSERT_LE(numChannels, 4);
    ASSERT_EQ(sizeof(InT), inImgFormat.planePixelStrideBytes(0));
    ASSERT_EQ(sizeof(OutT), outImgFormat.planePixelStrideBytes(0));

    ASSERT_EQ(numSamples, inShapes.size());
    ASSERT_EQ(numSamples, outShapes.size());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    uniform_distribution<InBT> rand(InBT{0}, std::is_integral_v<InBT> ? cuda::TypeTraits<InBT>::max : InBT{1});
    std::mt19937_64            rng(12345);

    std::vector<nvcv::Image>                   imgSrc;
    std::vector<nvcv::Image>                   imgDst;
    std::vector<baseline::CpuSample<InBT, 2>>  inBatchCpu;
    std::vector<baseline::CpuSample<OutBT, 2>> outBatchCpu;
    std::vector<baseline::CpuSample<OutBT, 2>> refBatchCpu;
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        SCOPED_TRACE(sampleIdx);
        nvcv::Size2D inImgShape{inShapes[sampleIdx].extent[1], inShapes[sampleIdx].extent[0]};
        imgSrc.emplace_back(inImgShape, inImgFormat);
        nvcv::Size2D outImgShape{outShapes[sampleIdx].extent[1], outShapes[sampleIdx].extent[0]};
        imgDst.emplace_back(outImgShape, outImgFormat);

        auto inData  = imgSrc[sampleIdx].exportData<nvcv::ImageDataStridedCuda>();
        auto outData = imgDst[sampleIdx].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(inData && outData);

        long3 inStrides{sizeof(InT), inData->plane(0).rowStride, inData->plane(0).rowStride * inData->plane(0).height};
        long3 outStrides{sizeof(OutT), outData->plane(0).rowStride,
                         outData->plane(0).rowStride * outData->plane(0).height};

        inBatchCpu.push_back(baseline::CpuSample<InBT, 2>{
            inStrides.z, inStrides, 1, int2{inImgShape.w, inImgShape.h},
               numChannels
        });
        outBatchCpu.push_back(baseline::CpuSample<OutBT, 2>{
            outStrides.z, outStrides, 1, int2{outImgShape.w, outImgShape.h},
               numChannels
        });
        refBatchCpu.push_back(baseline::CpuSample<OutBT, 2>{
            outStrides.z, outStrides, 1, int2{outImgShape.w, outImgShape.h},
               numChannels
        });

        auto &inTensorCpu = inBatchCpu[sampleIdx];
        baseline::ForAllInRoi(baseline::FullRoi<2>(int2{inImgShape.w, inImgShape.h}),
                              [&inTensorCpu, &rand, &rng](const int2 idx)
                              {
                                  for (int c = 0; c < numChannels; c++)
                                  {
                                      inTensorCpu.get(0, idx, c) = rand(rng);
                                  }
                              });
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inData->plane(0).basePtr, inTensorCpu.data(), inStrides.z,
                                               cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(numSamples);
    nvcv::ImageBatchVarShape batchDst(numSamples);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::HQResize op;
    if (allocateWorkspace)
    {
        HQResizeTensorShapesI inShapeDesc{inShapes.data(), numSamples, 2, numChannels};
        HQResizeTensorShapesI outShapeDesc{outShapes.data(), numSamples, 2, numChannels};
        ASSERT_NO_THROW(ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(
                            numSamples, inShapeDesc, outShapeDesc, minInterpolation, magInterpolation, antialias)));
    }
    ASSERT_NO_THROW(op(stream, ws.get(), batchSrc, batchDst, minInterpolation, magInterpolation, antialias));

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        SCOPED_TRACE(sampleIdx);
        const auto outData = imgDst[sampleIdx].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(outData);
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(outBatchCpu[sampleIdx].data(), outData->plane(0).basePtr,
                                               outBatchCpu[sampleIdx].strides().z, cudaMemcpyDeviceToHost, stream));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        SCOPED_TRACE(sampleIdx);
        baseline::Resize(refBatchCpu[sampleIdx], inBatchCpu[sampleIdx], minInterpolation, magInterpolation, antialias);
        baseline::Compare<InBT>(outBatchCpu[sampleIdx], refBatchCpu[sampleIdx], antialias);
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TYPED_TEST(OpHQResizeImageBatch, varbatch_2d_correct_output)
{
    const int                         numSamples = 4;
    std::vector<HQResizeTensorShapeI> inShapes   = {{{256, 128}}, {{40, 40}}, {{728, 1024}}, {{128, 256}}};
    std::vector<HQResizeTensorShapeI> outShapes  = {{{128, 256}}, {{512, 512}}, {{245, 245}}, {{243, 128}}};
    cvcuda::UniqueWorkspace           ws;
    TestImageBatch<TypeParam>(numSamples, inShapes, outShapes, ws);
}

// =============================================================================
// Planar (RGB8p/RGBA8p/...) var-shape parity: planar output must be bit-identical to interleaved.
// Each plane is processed as an independent single-channel image, so the result matches the
// interleaved (RGB8/...) path for every dtype, channel count, and interpolation mode.
// =============================================================================
namespace planar_parity_vs {

// Upload a per-image interleaved (HWC) host buffer into an interleaved (1-plane) or planar
// (C-plane) image, deinterleaving into per-channel planes for the planar case.
template<typename T>
void UploadImage(nvcv::Image &img, bool planar, int W, int H, int C, const std::vector<T> &interleaved)
{
    auto d = img.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_TRUE(d);
    if (!planar)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(d->plane(0).basePtr, d->plane(0).rowStride, interleaved.data(),
                                            W * C * sizeof(T), W * C * sizeof(T), H, cudaMemcpyHostToDevice));
        return;
    }
    std::vector<T> plane(static_cast<size_t>(W) * H);
    for (int c = 0; c < C; ++c)
    {
        planar_parity::ExtractPlane<T>(plane, interleaved.data(), W * H, C, c);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(d->plane(c).basePtr, d->plane(c).rowStride, plane.data(), W * sizeof(T),
                                            W * sizeof(T), H, cudaMemcpyHostToDevice));
    }
}

// Inverse of UploadImage.
template<typename T>
std::vector<T> DownloadImage(const nvcv::Image &img, bool planar, int W, int H, int C)
{
    auto d = img.exportData<nvcv::ImageDataStridedCuda>();
    EXPECT_TRUE(d);
    std::vector<T> interleaved(static_cast<size_t>(W) * H * C);
    if (!planar)
    {
        EXPECT_EQ(cudaSuccess, cudaMemcpy2D(interleaved.data(), W * C * sizeof(T), d->plane(0).basePtr,
                                            d->plane(0).rowStride, W * C * sizeof(T), H, cudaMemcpyDeviceToHost));
        return interleaved;
    }
    std::vector<T> plane(static_cast<size_t>(W) * H);
    for (int c = 0; c < C; ++c)
    {
        EXPECT_EQ(cudaSuccess, cudaMemcpy2D(plane.data(), W * sizeof(T), d->plane(c).basePtr, d->plane(c).rowStride,
                                            W * sizeof(T), H, cudaMemcpyDeviceToHost));
        planar_parity::StorePlane<T>(interleaved.data(), plane, W * H, C, c);
    }
    return interleaved;
}

// Resize an N-image batch (varying sizes) in the given layout, returning each image's
// re-interleaved output. The planar workspace is sized from expanded (N*C single-channel) shapes;
// the interleaved one from the natural C-channel shapes.
template<typename InT, typename OutT>
std::vector<std::vector<OutT>> Run(bool planar, nvcv::ImageFormat inFmt, nvcv::ImageFormat outFmt, int C,
                                   const std::vector<int2> &inWH, const std::vector<int2> &outWH,
                                   NVCVInterpolationType interp, bool antialias,
                                   const std::vector<std::vector<InT>> &interleavedIn)
{
    const auto               numSamples = static_cast<int>(inWH.size());
    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numSamples; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{inWH[i].x, inWH[i].y}, inFmt);
        imgDst.emplace_back(nvcv::Size2D{outWH[i].x, outWH[i].y}, outFmt);
        UploadImage<InT>(imgSrc[i], planar, inWH[i].x, inWH[i].y, C, interleavedIn[i]);
    }
    nvcv::ImageBatchVarShape batchSrc(numSamples);
    nvcv::ImageBatchVarShape batchDst(numSamples);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Workspace shapes: expanded single-channel (N*C) for planar, natural C-channel for interleaved.
    std::vector<HQResizeTensorShapeI> inShapes;
    std::vector<HQResizeTensorShapeI> outShapes;
    const int                         wsChannels = planar ? 1 : C;
    const int                         reps       = planar ? C : 1;
    for (int i = 0; i < numSamples; ++i)
        for (int r = 0; r < reps; ++r)
        {
            inShapes.push_back(HQResizeTensorShapeI{
                {inWH[i].y, inWH[i].x},
                2,
                wsChannels
            });
            outShapes.push_back(HQResizeTensorShapeI{
                {outWH[i].y, outWH[i].x},
                2,
                wsChannels
            });
        }
    const int wsSamples = numSamples * reps;

    cvcuda::HQResize        op;
    HQResizeTensorShapesI   inDesc{inShapes.data(), wsSamples, 2, wsChannels};
    HQResizeTensorShapesI   outDesc{outShapes.data(), wsSamples, 2, wsChannels};
    cvcuda::UniqueWorkspace ws
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(wsSamples, inDesc, outDesc, interp, interp, antialias));
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(op(stream, ws.get(), batchSrc, batchDst, interp, interp, antialias));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<std::vector<OutT>> out(numSamples);
    for (int i = 0; i < numSamples; ++i) out[i] = DownloadImage<OutT>(imgDst[i], planar, outWH[i].x, outWH[i].y, C);
    return out;
}

template<typename InT, typename OutT>
void Check(nvcv::ImageFormat inFmt, nvcv::ImageFormat inFmtP, nvcv::ImageFormat outFmt, nvcv::ImageFormat outFmtP,
           int C, NVCVInterpolationType interp)
{
    const std::vector<int2> inWH{
        { 64, 48},
        { 31, 97},
        {128, 33}
    };
    const std::vector<int2> outWH{
        { 40,  40},
        {120,  50},
        { 33, 128}
    };
    std::vector<std::vector<InT>> in(inWH.size());
    for (size_t i = 0; i < inWH.size(); ++i)
    {
        in[i].resize(static_cast<size_t>(inWH[i].x) * inWH[i].y * C);
        planar_parity::FillPattern<InT>(in[i], 909 + i);
    }
    for (bool antialias : {false, true})
    {
        auto interleaved = Run<InT, OutT>(false, inFmt, outFmt, C, inWH, outWH, interp, antialias, in);
        auto planar      = Run<InT, OutT>(true, inFmtP, outFmtP, C, inWH, outWH, interp, antialias, in);
        for (size_t i = 0; i < inWH.size(); ++i)
            EXPECT_EQ(interleaved[i], planar[i])
                << "image " << i << " C=" << C << " interp=" << interp << " antialias=" << antialias;
    }
}

} // namespace planar_parity_vs

TEST(OpHQResizeImageBatch, planar_matches_interleaved)
{
    using namespace planar_parity_vs;
    for (auto interp : {NVCV_INTERP_NEAREST, NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_INTERP_LANCZOS})
    {
        Check<uchar, uchar>(nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 3, interp);
        Check<uchar, uchar>(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, 4, interp);
        Check<float, float>(nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p, 3, interp);
    }
}

// =============================================================================
// Planar (NCHW/CHW) tensor-batch parity: a planar tensor batch is expanded into single-channel
// plane views and must produce bit-identical output to the interleaved (HWC) tensor batch.
// =============================================================================
namespace planar_parity_tb {

template<typename InT, typename OutT>
std::vector<std::vector<OutT>> Run(bool planar, int C, const std::vector<int2> &inWH, const std::vector<int2> &outWH,
                                   NVCVInterpolationType interp, bool antialias,
                                   const std::vector<std::vector<InT>> &in)
{
    const auto                N     = static_cast<int>(inWH.size());
    const nvcv::DataType      inDt  = TypeAsFormat<InT>();
    const nvcv::DataType      outDt = TypeAsFormat<OutT>();
    std::vector<nvcv::Tensor> srcTs;
    std::vector<nvcv::Tensor> dstTs;
    for (int i = 0; i < N; ++i)
    {
        srcTs.push_back(planar ? CreateTensorHelper(inDt, "NCHW", 1, C, inWH[i].y, inWH[i].x)
                               : CreateTensorHelper(inDt, "HWC", 1, inWH[i].y, inWH[i].x, C));
        dstTs.push_back(planar ? CreateTensorHelper(outDt, "NCHW", 1, C, outWH[i].y, outWH[i].x)
                               : CreateTensorHelper(outDt, "HWC", 1, outWH[i].y, outWH[i].x, C));
        planar_parity::UploadTensor<InT>(srcTs[i], planar, 1, inWH[i], C, in[i]);
    }
    nvcv::TensorBatch sb(N);
    nvcv::TensorBatch db(N);
    sb.pushBack(srcTs.begin(), srcTs.end());
    db.pushBack(dstTs.begin(), dstTs.end());

    // Planar workspace is sized from expanded (N*C single-channel) shapes; interleaved from C-channel.
    std::vector<HQResizeTensorShapeI> inShapes;
    std::vector<HQResizeTensorShapeI> outShapes;
    const int                         wsCh = planar ? 1 : C;
    const int                         reps = planar ? C : 1;
    for (int i = 0; i < N; ++i)
        for (int r = 0; r < reps; ++r)
        {
            inShapes.push_back(HQResizeTensorShapeI{
                {inWH[i].y, inWH[i].x},
                2,
                wsCh
            });
            outShapes.push_back(HQResizeTensorShapeI{
                {outWH[i].y, outWH[i].x},
                2,
                wsCh
            });
        }
    const int               wsN = N * reps;
    cvcuda::HQResize        op;
    HQResizeTensorShapesI   inDesc{inShapes.data(), wsN, 2, wsCh};
    HQResizeTensorShapesI   outDesc{outShapes.data(), wsN, 2, wsCh};
    cvcuda::UniqueWorkspace ws
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(wsN, inDesc, outDesc, interp, interp, antialias));
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(op(stream, ws.get(), sb, db, interp, interp, antialias));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<std::vector<OutT>> out(N);
    for (int i = 0; i < N; ++i) out[i] = planar_parity::DownloadTensor<OutT>(dstTs[i], planar, 1, outWH[i], C);
    return out;
}

// Build random C-channel data, resize it as both an interleaved (HWC) and a planar (CHW) tensor
// batch, and assert the re-interleaved planar output matches the interleaved output bit-for-bit.
static void Check(int C, const std::vector<int2> &inWH, const std::vector<int2> &outWH, NVCVInterpolationType interp)
{
    std::vector<std::vector<uchar>> in(inWH.size());
    for (size_t i = 0; i < inWH.size(); ++i)
    {
        in[i].resize(static_cast<size_t>(inWH[i].x) * inWH[i].y * C);
        planar_parity::FillPattern<uchar>(in[i], static_cast<uint64_t>(404 + C) * 131 + i);
    }
    for (bool antialias : {false, true})
    {
        auto interleaved = Run<uchar, uchar>(false, C, inWH, outWH, interp, antialias, in);
        auto planar      = Run<uchar, uchar>(true, C, inWH, outWH, interp, antialias, in);
        for (size_t i = 0; i < inWH.size(); ++i)
            EXPECT_EQ(interleaved[i], planar[i])
                << "tensorbatch image " << i << " C=" << C << " interp=" << interp << " antialias=" << antialias;
    }
}

} // namespace planar_parity_tb

TEST(OpHQResizeTensorBatch, planar_matches_interleaved)
{
    using namespace planar_parity_tb;
    const std::vector<int2> inWH{
        {64, 48},
        {31, 97},
        {50, 50}
    };
    const std::vector<int2> outWH{
        { 40, 40},
        { 60, 33},
        {128, 20}
    };
    for (auto interp : {NVCV_INTERP_NEAREST, NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_INTERP_LANCZOS})
        for (int C : {1, 3, 4}) Check(C, inWH, outWH, interp);
}

// =============================================================================
// Uniform-shape batches where one sample resizes through a flipped (lo > hi) ROI:
// the sample shapes match, but the mapping does not, so batch fast paths that share
// sample-0 phase state across samples must fall back, and every sample must match
// the CPU reference computed from its own (flipped) input.
// =============================================================================
namespace batch_roi_mapping {

static void Check(const int2 inWH, const int2 outWH, const NVCVInterpolationType interp)
{
    constexpr int numChannels = 3;
    constexpr int numSamples  = 2;
    using BT                  = float;
    const auto dtype          = TypeAsFormat<BT>();

    nvcv::TensorBatch            inBatch(numSamples);
    nvcv::TensorBatch            outBatch(numSamples);
    std::vector<std::vector<BT>> inData(numSamples);
    for (int s = 0; s < numSamples; s++)
    {
        inData[s].resize(static_cast<size_t>(inWH.x) * inWH.y * numChannels);
        planar_parity::FillPattern<BT>(inData[s], 777 + s);
        nvcv::Tensor inT  = CreateTensorHelper(dtype, "HWC", 1, inWH.y, inWH.x, numChannels);
        nvcv::Tensor outT = CreateTensorHelper(dtype, "HWC", 1, outWH.y, outWH.x, numChannels);
        planar_parity::UploadTensor<BT>(inT, false, 1, inWH, numChannels, inData[s]);
        inBatch.pushBack(inT);
        outBatch.pushBack(outT);
    }

    std::array<HQResizeRoiF, numSamples> roiData{};
    for (int s = 0; s < numSamples; s++)
    {
        roiData[s].lo[0] = s == 0 ? 0.f : static_cast<float>(inWH.y);
        roiData[s].hi[0] = s == 0 ? static_cast<float>(inWH.y) : 0.f;
        roiData[s].hi[1] = static_cast<float>(inWH.x);
    }
    HQResizeRoisF rois{numSamples, 2, roiData.data()};

    std::vector<HQResizeTensorShapeI> inShapes(numSamples, HQResizeTensorShapeI{
                                                               {inWH.y, inWH.x},
                                                               2,
                                                               numChannels
    });
    std::vector<HQResizeTensorShapeI> outShapes(numSamples, HQResizeTensorShapeI{
                                                                {outWH.y, outWH.x},
                                                                2,
                                                                numChannels
    });
    HQResizeTensorShapesI             inDesc{inShapes.data(), numSamples, 2, numChannels};
    HQResizeTensorShapesI             outDesc{outShapes.data(), numSamples, 2, numChannels};

    cvcuda::HQResize        op;
    cvcuda::UniqueWorkspace ws
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(numSamples, inDesc, outDesc, interp, interp, false));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_NO_THROW(op(stream, ws.get(), inBatch, outBatch, interp, interp, false, rois));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    const int64_t colStride    = static_cast<int64_t>(numChannels) * sizeof(BT);
    const int64_t inRowStride  = inWH.x * colStride;
    const int64_t inBytes      = inWH.y * inRowStride;
    const int64_t outRowStride = outWH.x * colStride;
    const int64_t outBytes     = outWH.y * outRowStride;
    for (int s = 0; s < numSamples; s++)
    {
        SCOPED_TRACE(s);
        baseline::CpuSample<BT, 2> inCpu(inBytes, long3{colStride, inRowStride, inBytes}, 1, inWH, numChannels);
        baseline::ForAllInRoi(baseline::FullRoi<2>(inWH),
                              [s, inWH, &inCpu, &inData](int2 idx)
                              {
                                  const int srcY = s == 0 ? idx.y : inWH.y - 1 - idx.y;
                                  for (int c = 0; c < numChannels; c++)
                                  {
                                      inCpu.get(0, idx, c)
                                          = inData[s][(static_cast<size_t>(srcY) * inWH.x + idx.x) * numChannels + c];
                                  }
                              });

        baseline::CpuSample<BT, 2> refCpu(outBytes, long3{colStride, outRowStride, outBytes}, 1, outWH, numChannels);
        baseline::Resize(refCpu, inCpu, interp, interp, false);

        baseline::CpuSample<BT, 2> outCpu(outBytes, long3{colStride, outRowStride, outBytes}, 1, outWH, numChannels);
        const auto                 out = planar_parity::DownloadTensor<BT>(outBatch[s], false, 1, outWH, numChannels);
        std::memcpy(outCpu.data(), out.data(), outBytes);

        baseline::Compare<BT>(outCpu, refCpu, false);
    }
}

} // namespace batch_roi_mapping

TEST(OpHQResizeTensorBatch, flipped_roi_uniform_shapes_matches_reference)
{
    using namespace batch_roi_mapping;
    Check(int2{40, 36}, int2{100, 90}, NVCV_INTERP_CUBIC);
    Check(int2{96, 120}, int2{40, 48}, NVCV_INTERP_CUBIC);
}

TEST(OpHQResizeImageBatch, test_multi_run_single_workspace)
{
    using FirstRun  = typename NVCV_TEST_ROW_IB(1, uchar, NVCV_IMAGE_FORMAT_U8, uchar, NVCV_IMAGE_FORMAT_U8, false,
                                                NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC);
    using SecondRun = typename NVCV_TEST_ROW_IB(3, uchar3, NVCV_IMAGE_FORMAT_RGB8, float3, NVCV_IMAGE_FORMAT_RGBf32,
                                                true, NVCV_INTERP_LANCZOS, NVCV_INTERP_LINEAR);

    const int                         numSamples0 = 1;
    std::vector<HQResizeTensorShapeI> inShapes0   = {
          {{128, 128}, 2, 1}
    };
    std::vector<HQResizeTensorShapeI> outShapes0 = {
        {{40, 50}, 2, 1}
    };

    const int                         numSamples1 = 3;
    std::vector<HQResizeTensorShapeI> inShapes1   = {
          {  {50, 40}, 2, 3},
          {  {64, 64}, 2, 3},
          {{128, 128}, 2, 3}
    };
    std::vector<HQResizeTensorShapeI> outShapes1 = {
        {{128, 128}, 2, 3},
        {{128, 128}, 2, 3},
        {{128, 128}, 2, 3}
    };

    HQResizeTensorShapeI maxShape;
    GetMaxShape(maxShape, inShapes0.data(), numSamples0);
    GetMaxShape(maxShape, outShapes0.data(), numSamples0);
    GetMaxShape(maxShape, inShapes1.data(), numSamples1);
    GetMaxShape(maxShape, outShapes1.data(), numSamples1);

    cvcuda::HQResize        op;
    cvcuda::UniqueWorkspace ws;
    ASSERT_NO_THROW(
        ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(std::max(numSamples0, numSamples1), maxShape)));
    TestImageBatch<FirstRun>(numSamples0, inShapes0, outShapes0, ws, false);
    TestImageBatch<SecondRun>(numSamples1, inShapes1, outShapes1, ws, false);
}

TEST(OpHQResizeNegative, createWithNullHandle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaHQResizeCreate(nullptr));
}

TEST(OpHQResizeNegative, getWorkspaceRequirementsWithNullReqOut)
{
    NVCVOperatorHandle op;
    EXPECT_EQ(NVCV_SUCCESS, cvcudaHQResizeCreate(&op));
    HQResizeTensorShapeI shapeIn;
    HQResizeTensorShapeI shapeOut;
    shapeIn.extent[0]    = 128;
    shapeIn.ndim         = 1;
    shapeIn.numChannels  = 1;
    shapeOut.extent[0]   = 64;
    shapeOut.ndim        = 1;
    shapeOut.numChannels = 1;

    HQResizeRoiF roi;
    roi.lo[0] = 0;
    roi.hi[0] = 128;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaHQResizeTensorGetWorkspaceRequirements(op, 1, shapeIn, shapeOut, NVCV_INTERP_LINEAR,
                                                           NVCV_INTERP_LINEAR, false, &roi, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaHQResizeGetMaxWorkspaceRequirements(op, 1, shapeIn, nullptr));
    EXPECT_NO_THROW(nvcvOperatorDestroy(op));
}

TEST(OpHQResizeNegative, submitWithNullWorkspace)
{
    NVCVOperatorHandle op;
    EXPECT_EQ(NVCV_SUCCESS, cvcudaHQResizeCreate(&op));

    HQResizeRoiF roi;
    roi.lo[0] = 0;
    roi.hi[0] = 128;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaHQResizeSubmit(op, nullptr, nullptr, nullptr, nullptr,
                                                                NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR, false, &roi));
    EXPECT_NO_THROW(nvcvOperatorDestroy(op));
}

TEST(OpHQResizeNegative, submitWithNonFiniteRoi)
{
    NVCVOperatorHandle op;
    ASSERT_EQ(NVCV_SUCCESS, cvcudaHQResizeCreate(&op));

    // NaN lo
    HQResizeRoiF roiNanLo{};
    roiNanLo.lo[0] = std::numeric_limits<float>::quiet_NaN();
    roiNanLo.hi[0] = 128.f;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaHQResizeSubmit(op, nullptr, nullptr, nullptr, nullptr, NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR,
                                   false, &roiNanLo));

    // Inf hi
    HQResizeRoiF roiInfHi{};
    roiInfHi.lo[0] = 0.f;
    roiInfHi.hi[0] = std::numeric_limits<float>::infinity();
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaHQResizeSubmit(op, nullptr, nullptr, nullptr, nullptr, NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR,
                                   false, &roiInfHi));

    EXPECT_NO_THROW(nvcvOperatorDestroy(op));
}

TEST(OpHQResizeNegative, submitWithNonFiniteRoiBatch)
{
    NVCVOperatorHandle op;
    ASSERT_EQ(NVCV_SUCCESS, cvcudaHQResizeCreate(&op));

    std::array<HQResizeRoiF, 1> roiData{};
    roiData[0].lo[0] = std::numeric_limits<float>::quiet_NaN();
    roiData[0].hi[0] = 128.f;

    HQResizeRoisF rois;
    rois.roi  = roiData.data();
    rois.size = 1;
    rois.ndim = 1;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaHQResizeImageBatchSubmit(op, nullptr, nullptr, nullptr, nullptr, NVCV_INTERP_LINEAR,
                                             NVCV_INTERP_LINEAR, false, rois));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaHQResizeTensorBatchSubmit(op, nullptr, nullptr, nullptr, nullptr, NVCV_INTERP_LINEAR,
                                              NVCV_INTERP_LINEAR, false, rois));

    EXPECT_NO_THROW(nvcvOperatorDestroy(op));
}

TEST(OpHQResizeNegative, submitWithNullWorkspaceBatch)
{
    NVCVOperatorHandle op;
    EXPECT_EQ(NVCV_SUCCESS, cvcudaHQResizeCreate(&op));

    std::array<HQResizeRoiF, 1> roi{};
    roi[0].lo[0] = 0;
    roi[0].hi[0] = 128;
    HQResizeRoisF rois;
    rois.roi  = roi.data();
    rois.size = 1;
    rois.ndim = 1;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaHQResizeImageBatchSubmit(op, nullptr, nullptr, nullptr, nullptr, NVCV_INTERP_LINEAR,
                                             NVCV_INTERP_LINEAR, false, rois));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaHQResizeTensorBatchSubmit(op, nullptr, nullptr, nullptr, nullptr, NVCV_INTERP_LINEAR,
                                              NVCV_INTERP_LINEAR, false, rois));
    EXPECT_NO_THROW(nvcvOperatorDestroy(op));
}

TEST(OpHQResizeNegative, getWorkspaceRequirementsWithNullReqOutBatch)
{
    NVCVOperatorHandle op;
    EXPECT_EQ(NVCV_SUCCESS, cvcudaHQResizeCreate(&op));
    std::array<HQResizeTensorShapeI, 1> shapeIn{};
    std::array<HQResizeTensorShapeI, 1> shapeOut{};
    shapeIn[0].extent[0]    = 128;
    shapeIn[0].ndim         = 1;
    shapeIn[0].numChannels  = 1;
    shapeOut[0].extent[0]   = 64;
    shapeOut[0].ndim        = 1;
    shapeOut[0].numChannels = 1;

    HQResizeTensorShapesI shapeInBatch;

    HQResizeTensorShapesI shapeOutBatch;
    shapeInBatch.shape        = shapeIn.data();
    shapeInBatch.size         = 1;
    shapeInBatch.ndim         = 1;
    shapeInBatch.numChannels  = 1;
    shapeOutBatch.shape       = shapeOut.data();
    shapeOutBatch.size        = 1;
    shapeOutBatch.ndim        = 1;
    shapeOutBatch.numChannels = 1;

    std::array<HQResizeRoiF, 1> roi{};
    roi[0].lo[0] = 0;
    roi[0].hi[0] = 128;
    HQResizeRoisF rois;
    rois.roi  = roi.data();
    rois.size = 1;
    rois.ndim = 1;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaHQResizeTensorBatchGetWorkspaceRequirements(op, 1, shapeInBatch, shapeOutBatch, NVCV_INTERP_LINEAR,
                                                                NVCV_INTERP_LINEAR, false, rois, nullptr));
    EXPECT_NO_THROW(nvcvOperatorDestroy(op));
}

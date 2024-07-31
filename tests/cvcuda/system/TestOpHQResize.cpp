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

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <cvcuda/OpHQResize.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/Math.hpp>

#include <cmath>
#include <iostream>
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

template<typename Cb>
void ForAll(int2 shape, Cb &&cb)
{
    for (int y = 0; y < shape.y; y++)
    {
        for (int x = 0; x < shape.x; x++)
        {
            cb(int2{x, y});
        }
    }
}

template<typename Cb>
void ForAll(int3 shape, Cb &&cb)
{
    for (int z = 0; z < shape.z; z++)
        for (int y = 0; y < shape.y; y++)
        {
            for (int x = 0; x < shape.x; x++)
            {
                cb(int3{x, y, z});
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
        return *(reinterpret_cast<BT *>(m_data.data() + offset(sampleIdx, idx)) + channel);
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

    int numSamples()
    {
        return m_numSamples;
    }

    int numChannels()
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
        float x = 4 * (k - (size() - 1) * 0.5f) / (size() - 1);
        x       = fabsf(x);
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
        float x = 4 * (k - (size() - 1) * 0.5f) / (size() - 1);
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
        float x = 2 * kLanczosA * (k - (size() - 1) * 0.5f) / (size() - 1);
        if (fabsf(x) >= kLanczosA)
            return 0.0f;
        return nvcv::util::sinc(x) * nvcv::util::sinc(x / kLanczosA);
    }
};

template<typename FilterType>
struct Filter
{
    Filter(float support)
        : m_filter{}
        , m_support{support}
    {
    }

    float support() const
    {
        return std::ceil(m_support);
    }

    float scale() const
    {
        return (m_filter.size() - 1) / m_support;
    }

    float anchor() const
    {
        return m_support / 2;
    }

    float operator()(float x) const
    {
        if (!(x > -1))
            return 0;
        if (x >= m_filter.size())
            return 0;
        int   x0 = std::floor(x);
        int   x1 = x0 + 1;
        float d  = x - x0;
        float f0 = x0 < 0 ? 0.0f : m_filter[x0];
        float f1 = x1 >= m_filter.size() ? 0.0f : m_filter[x1];
        return f0 + d * (f1 - f0);
    }

private:
    FilterType m_filter;
    float      m_support;
};

template<typename OutBT, typename InBT, int kSpatialNDim>
void RunNN(int axis, CpuSample<OutBT, kSpatialNDim> &outTensorCpu, CpuSample<InBT, kSpatialNDim> &inTensorCpu)
{
    const int   numSamples  = inTensorCpu.numSamples();
    const int   numChannels = inTensorCpu.numChannels();
    const auto  inShape     = inTensorCpu.shape();
    const auto  outShape    = outTensorCpu.shape();
    const int   inSize      = cuda::GetElement(inShape, axis);
    const int   outSize     = cuda::GetElement(outShape, axis);
    const float axisScale   = static_cast<float>(inSize) / outSize;
    const float axisOrigin  = 0.5f * axisScale;
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        ForAll(outShape,
               [&](const cuda::MakeType<int, kSpatialNDim> outIdx)
               {
                   auto inIdx                    = outIdx;
                   int  inAxis                   = std::floor(cuda::GetElement(outIdx, axis) * axisScale + axisOrigin);
                   inAxis                        = inAxis < 0 ? 0 : (inAxis > inSize - 1 ? inSize - 1 : inAxis);
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
void RunLinear(int axis, CpuSample<OutBT, kSpatialNDim> &outTensorCpu, CpuSample<InBT, kSpatialNDim> &inTensorCpu)
{
    const int   numSamples  = inTensorCpu.numSamples();
    const int   numChannels = inTensorCpu.numChannels();
    const auto  inShape     = inTensorCpu.shape();
    const auto  outShape    = outTensorCpu.shape();
    const int   inSize      = cuda::GetElement(inShape, axis);
    const int   outSize     = cuda::GetElement(outShape, axis);
    const float axisScale   = static_cast<float>(inSize) / outSize;
    const float axisOrigin  = 0.5f * axisScale - 0.5f;
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        ForAll(outShape,
               [&](const cuda::MakeType<int, kSpatialNDim> outIdx)
               {
                   const float inAxis0f           = cuda::GetElement(outIdx, axis) * axisScale + axisOrigin;
                   int         inAxis0            = std::floor(inAxis0f);
                   int         inAxis1            = inAxis0 + 1;
                   const float q                  = inAxis0f - inAxis0;
                   inAxis0                        = inAxis0 < 0 ? 0 : (inAxis0 > inSize - 1 ? inSize - 1 : inAxis0);
                   inAxis1                        = inAxis1 < 0 ? 0 : (inAxis1 > inSize - 1 ? inSize - 1 : inAxis1);
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
               const FilterT &filter)
{
    const int   numSamples    = inTensorCpu.numSamples();
    const int   numChannels   = inTensorCpu.numChannels();
    const auto  inShape       = inTensorCpu.shape();
    const auto  outShape      = outTensorCpu.shape();
    const int   inSize        = cuda::GetElement(inShape, axis);
    const int   outSize       = cuda::GetElement(outShape, axis);
    const int   filterSupport = filter.support();
    const float filterStep    = filter.scale();
    const float axisScale     = static_cast<float>(inSize) / outSize;
    const float axisOrigin    = 0.5f * axisScale - 0.5f - filter.anchor();

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        ForAll(outShape,
               [&](const cuda::MakeType<int, kSpatialNDim> outIdx)
               {
                   const float inAxis0f = cuda::GetElement(outIdx, axis) * axisScale + axisOrigin;
                   int         inAxis0  = std::ceil(inAxis0f);
                   const float fStart   = (inAxis0 - inAxis0f) * filterStep;
                   for (int c = 0; c < numChannels; c++)
                   {
                       float tmp  = 0;
                       float norm = 0;
                       for (int k = 0; k < filterSupport; k++)
                       {
                           int inAxis                    = inAxis0 + k;
                           inAxis                        = inAxis < 0 ? 0 : (inAxis > inSize - 1 ? inSize - 1 : inAxis);
                           auto inIdx                    = outIdx;
                           cuda::GetElement(inIdx, axis) = inAxis;
                           const InBT inVal              = inTensorCpu.get(sampleIdx, inIdx, c);
                           float      coeff              = filter(fStart + k * filterStep);
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
               const NVCVInterpolationType interpolation, bool antialias)
{
    const auto  inShape  = inTensorCpu.shape();
    const auto  outShape = outTensorCpu.shape();
    const float inSize   = cuda::GetElement(inShape, axis);
    const float outSize  = cuda::GetElement(outShape, axis);
    switch (interpolation)
    {
    case NVCV_INTERP_LINEAR:
    {
        float radius  = antialias ? inSize / outSize : 1;
        float support = std::max(1.0f, 2 * radius);
        RunFilter(axis, outTensorCpu, inTensorCpu, Filter<FilterTriangular>{support});
    }
    break;
    case NVCV_INTERP_CUBIC:
    {
        float radius  = antialias ? (2 * inSize / outSize) : 2;
        float support = std::max(4.0f, 2 * radius);
        RunFilter(axis, outTensorCpu, inTensorCpu, Filter<FilterCubic>{support});
    }
    break;
    case NVCV_INTERP_GAUSSIAN:
    {
        float radius  = antialias ? inSize / outSize : 1;
        float support = std::max(1.0f, 2 * radius);
        RunFilter(axis, outTensorCpu, inTensorCpu, Filter<FilterGaussian>{support});
    }
    break;
    case NVCV_INTERP_LANCZOS:
    {
        float radius  = antialias ? (3 * inSize / outSize) : 3;
        float support = std::max(6.0f, 2 * radius);
        RunFilter(axis, outTensorCpu, inTensorCpu, Filter<FilterLanczos>{support});
    }
    break;
    default:
        FAIL() << "Unsupported filter";
    }
}

template<typename OutBT, typename InBT, int kSpatialNDim>
void RunPass(int axis, CpuSample<OutBT, kSpatialNDim> &outTensorCpu, CpuSample<InBT, kSpatialNDim> &inTensorCpu,
             const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation, bool antialias)

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
        RunNN(axis, outTensorCpu, inTensorCpu);
        break;
    case NVCV_INTERP_LINEAR:
    {
        if (antialias)
        {
            RunFilter(axis, outTensorCpu, inTensorCpu, interpolation, antialias);
        }
        else
        {
            RunLinear(axis, outTensorCpu, inTensorCpu);
        }
    }
    break;
    default:
        RunFilter(axis, outTensorCpu, inTensorCpu, interpolation, antialias);
        break;
    }
}

template<typename OutBT, typename InBT>
void Resize(CpuSample<OutBT, 2> &refTensorCpu, CpuSample<InBT, 2> &inTensorCpu,
            const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation, bool antialias)
{
    int        numSamples         = inTensorCpu.numSamples();
    int        numChannels        = inTensorCpu.numChannels();
    const int2 inShape            = inTensorCpu.shape();
    const int2 outShape           = refTensorCpu.shape();
    const int2 interShape         = {outShape.x, inShape.y};
    auto       intermediateTensor = GetIntermediate(numSamples, interShape, numChannels);
    RunPass(0, intermediateTensor, inTensorCpu, minInterpolation, magInterpolation, antialias);
    RunPass(1, refTensorCpu, intermediateTensor, minInterpolation, magInterpolation, antialias);
}

template<typename OutBT, typename InBT>
void Resize(CpuSample<OutBT, 3> &refTensorCpu, CpuSample<InBT, 3> &inTensorCpu,
            const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation, bool antialias)
{
    int        numSamples          = inTensorCpu.numSamples();
    int        numChannels         = inTensorCpu.numChannels();
    const int3 inShape             = inTensorCpu.shape();
    const int3 outShape            = refTensorCpu.shape();
    const int3 interShape0         = {outShape.x, inShape.y, inShape.z};
    const int3 interShape1         = {outShape.x, outShape.y, inShape.z};
    auto       intermediateTensor0 = GetIntermediate(numSamples, interShape0, numChannels);
    RunPass(0, intermediateTensor0, inTensorCpu, minInterpolation, magInterpolation, antialias);
    auto intermediateTensor1 = GetIntermediate(numSamples, interShape1, numChannels);
    RunPass(1, intermediateTensor1, intermediateTensor0, minInterpolation, magInterpolation, antialias);
    RunPass(2, refTensorCpu, intermediateTensor1, minInterpolation, magInterpolation, antialias);
}

template<typename InBT, typename BT, int kSpatialNDim>
void Compare(CpuSample<BT, kSpatialNDim> &tensor, CpuSample<BT, kSpatialNDim> &refTensor, bool antialias)
{
    int        numSamples  = tensor.numSamples();
    int        numChannels = tensor.numChannels();
    const auto shape       = tensor.shape();
    ASSERT_EQ(numSamples, refTensor.numSamples());
    ASSERT_EQ(numChannels, refTensor.numChannels());
    ASSERT_EQ(shape, refTensor.shape());
    double  err = 0;
    int64_t vol = 0;
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        ForAll(shape,
               [&](const cuda::MakeType<int, kSpatialNDim> idx)
               {
                   for (int c = 0; c < numChannels; c++)
                   {
                       const BT val    = tensor.get(sampleIdx, idx, c);
                       const BT refVal = refTensor.get(sampleIdx, idx, c);
                       err += abs(val - refVal);
                       vol += 1;

                       if (std::is_integral_v<BT>) // uchar -> uchar, short -> short, ushort -> ushort
                       {
                           ASSERT_NEAR(val, refVal, (std::is_same_v<BT, uchar> ? 1 : 10)); // uchar : short, ushort
                       }
                       else // output type is float
                       {
                           if (!std::is_integral_v<InBT>) // float -> float
                           {
                               ASSERT_NEAR(val, refVal, 1e-4);
                           }
                           else // [uchar, short, ushort] -> float
                           {
                               ASSERT_NEAR(val, refVal, (std::is_same_v<BT, uchar> ? 0.1 : 6));
                           }
                       }
                   }
               });
    }
    double mean_err = err / vol;
    ASSERT_LE(mean_err, antialias ? 0.1 : 0.4);
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

inline void GetMaxShape(HQResizeTensorShapeI &ret, const HQResizeTensorShapeI *shapes, int numSamples)
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
        NVCV_TEST_ROW(1, NVCV_SHAPE2D(1 << 14, 1 << 13), NVCV_SHAPE2D(512, 256), 7, float, float, NVCV_INTERP_LINEAR)>);

template<typename TypeParam>
void TestTensor(bool antialias)
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

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inData && outData);

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(inAccess && outAccess);
    long3 inStrides{inAccess->colStride(), inAccess->rowStride(),
                    inAccess->sampleStride() == 0 ? inAccess->rowStride() * inShape.y : inAccess->sampleStride()};
    long3 outStrides{outAccess->colStride(), outAccess->rowStride(),
                     outAccess->sampleStride() == 0 ? outAccess->rowStride() * outShape.y : outAccess->sampleStride()};

    ASSERT_EQ(inAccess->numSamples(), numSamples);
    ASSERT_EQ(inAccess->numChannels(), numChannels);
    ASSERT_EQ(outAccess->numChannels(), numChannels);
    ASSERT_EQ(outAccess->numSamples(), numSamples);

    baseline::CpuSample<InBT, 2>  inTensorCpu(inStrides.z * numSamples, inStrides, numSamples, inShape, numChannels);
    baseline::CpuSample<OutBT, 2> outTensorCpu(outStrides.z * numSamples, outStrides, numSamples, outShape,
                                               numChannels);
    baseline::CpuSample<OutBT, 2> refTensorCpu(outStrides.z * numSamples, outStrides, numSamples, outShape,
                                               numChannels);

    uniform_distribution<InBT> rand(InBT{0}, std::is_integral_v<InBT> ? cuda::TypeTraits<InBT>::max : InBT{1});
    std::mt19937_64            rng(12345);

    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        for (int y = 0; y < inShape.y; y++)
        {
            for (int x = 0; x < inShape.x; x++)
            {
                for (int c = 0; c < numChannels; c++)
                {
                    inTensorCpu.get(sampleIdx, int2{x, y}, c) = rand(rng);
                }
            }
        }
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
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inData->basePtr(), inTensorCpu.data(), inStrides.z * numSamples,
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_NO_THROW(op(stream, ws.get(), inTensor, outTensor, interpolation, interpolation, antialias));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(outTensorCpu.data(), outData->basePtr(), outStrides.z * numSamples,
                                           cudaMemcpyDeviceToHost, stream));
    baseline::Resize(refTensorCpu, inTensorCpu, interpolation, interpolation, antialias);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    baseline::Compare<InBT>(outTensorCpu, refTensorCpu, antialias);
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
        NVCV_TEST_ROW(3, NVCV_SHAPE3D(1 << 10, 1 << 9, 1 << 9), NVCV_SHAPE3D(50, 150, 100), 3, uchar, uchar,
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

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inData && outData);

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(inAccess && outAccess);
    long4 inStrides{inAccess->colStride(), inAccess->rowStride(), inAccess->depthStride(),
                    inAccess->sampleStride() == 0 ? inAccess->depthStride() * inShape.z : inAccess->sampleStride()};
    long4 outStrides{
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
        for (int z = 0; z < inShape.z; z++)
        {
            for (int y = 0; y < inShape.y; y++)
            {
                for (int x = 0; x < inShape.x; x++)
                {
                    for (int c = 0; c < numChannels; c++)
                    {
                        inTensorCpu.get(sampleIdx, int3{x, y, z}, c) = rand(rng);
                    }
                }
            }
        }
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
    baseline::Resize(refTensorCpu, inTensorCpu, interpolation, interpolation, antialias);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    baseline::Compare<InBT>(outTensorCpu, refTensorCpu, antialias);
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

    constexpr int numSamples              = 5;
    const int     varChannels[numSamples] = {4, 1, 7, 3, 5};

    int inShape1[] = {1 << 14, 1 << 13};
    if (sizeof(InBT) == 1)
    {
        inShape1[0] *= 2;
        inShape1[1] *= 2;
    }

    auto sample1
        = largeSample
            ? HQResizeTensorShapeI({inShape1[0], inShape1[1]}, 2, numChannels > 0 ? numChannels : varChannels[0])
            : HQResizeTensorShapeI({728, 1024, 0}, 2, numChannels > 0 ? numChannels : varChannels[0]);

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

        auto &inTensorCpu = inBatchCpu[sampleIdx];
        for (int y = 0; y < inShape.y; y++)
        {
            for (int x = 0; x < inShape.x; x++)
            {
                for (int c = 0; c < inShapes[sampleIdx].numChannels; c++)
                {
                    inTensorCpu.get(0, int2{x, y}, c) = rand(rng);
                }
            }
        }
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
        baseline::Resize(refBatchCpu[sampleIdx], inBatchCpu[sampleIdx], minInterpolation, magInterpolation, antialias);
        baseline::Compare<InBT>(outBatchCpu[sampleIdx], refBatchCpu[sampleIdx], antialias);
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

    constexpr int numSamples              = 5;
    const int     varChannels[numSamples] = {6, 2, 3, 4, 1};

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

        long4 inStrides{inAccess->colStride(), inAccess->rowStride(), inAccess->depthStride(),
                        inAccess->sampleStride() == 0 ? inAccess->depthStride() * inShapes[sampleIdx].extent[0]
                                                      : inAccess->sampleStride()};
        long4 outStrides{outAccess->colStride(), outAccess->rowStride(), outAccess->depthStride(),
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
        for (int z = 0; z < inShape.z; z++)
        {
            for (int y = 0; y < inShape.y; y++)
            {
                for (int x = 0; x < inShape.x; x++)
                {
                    for (int c = 0; c < inShapes[sampleIdx].numChannels; c++)
                    {
                        inTensorCpu.get(0, int3{x, y, z}, c) = rand(rng);
                    }
                }
            }
        }
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
        for (int y = 0; y < inImgShape.h; y++)
        {
            for (int x = 0; x < inImgShape.w; x++)
            {
                for (int c = 0; c < numChannels; c++)
                {
                    inTensorCpu.get(0, int2{x, y}, c) = rand(rng);
                }
            }
        }
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

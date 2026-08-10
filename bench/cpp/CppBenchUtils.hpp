/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_CPP_BENCH_UTILS_HPP
#define CVCUDA_CPP_BENCH_UTILS_HPP

#include "BenchFillKernels.hpp"
#include "WarmupPolicy.hpp"

#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/BorderType.h>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <new>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace benchutils {

class InvalidBenchmarkDataError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace benchutils

#define CVCUDA_CHECK_DATA(data)                                      \
    if (!data)                                                       \
    {                                                                \
        throw benchutils::InvalidBenchmarkDataError("Invalid data"); \
    }

#define CVCUDA_BENCH_SKIP_ERRORS(state)      \
    catch (const nvcv::Exception &err)       \
    {                                        \
        (state).skip(err.what());            \
    }                                        \
    catch (const std::invalid_argument &err) \
    {                                        \
        (state).skip(err.what());            \
    }                                        \
    catch (const std::out_of_range &err)     \
    {                                        \
        (state).skip(err.what());            \
    }                                        \
    catch (const std::length_error &err)     \
    {                                        \
        (state).skip(err.what());            \
    }                                        \
    catch (const std::runtime_error &err)    \
    {                                        \
        (state).skip(err.what());            \
    }                                        \
    catch (const std::bad_alloc &err)        \
    {                                        \
        (state).skip(err.what());            \
    }

#define CUDA_CHECK_ERROR(RC)                                  \
    {                                                         \
        benchutils::cudaCheckError((RC), __FILE__, __LINE__); \
    }

namespace benchutils {

inline void cudaCheckError(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "\nE In CUDA: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

template<int N, typename RT = nvcv::cuda::MakeType<long, N>>
inline RT GetShape(const std::string &shapeStr, const std::string &delimiter = "x")
{
    std::string str = shapeStr;
    RT          shape;
    for (int i = 0; i < N; ++i)
    {
        size_t pos = str.find(delimiter);

        if ((pos == std::string::npos && i != (N - 1)) || (pos != std::string::npos && i == (N - 1)))
        {
            throw std::invalid_argument("Expecting " + std::to_string(N) + "-rank shape in " + shapeStr
                                        + " (pass shape separated by " + delimiter + ")");
        }

        nvcv::cuda::GetElement(shape, i) = std::stoi(str.substr(0, pos));

        str.erase(0, pos + delimiter.length());
    }

    return shape;
}

inline long3 GetResizeOutputShape(long3 srcShape, const std::string &resizeType)
{
    if (resizeType == "EXPAND")
    {
        return long3{srcShape.x, srcShape.y * 2, srcShape.z * 2};
    }
    if (resizeType == "CONTRACT")
    {
        return long3{srcShape.x, srcShape.y / 2, srcShape.z / 2};
    }

    if (constexpr std::string_view targetPrefix = "TARGET_"; resizeType.rfind(targetPrefix, 0) == 0)
    {
        long2 target = GetShape<2>(resizeType.substr(targetPrefix.size()));
        if (target.x <= 0 || target.y <= 0)
        {
            throw std::invalid_argument("Resize target dimensions must be positive in " + resizeType);
        }
        return long3{srcShape.x, target.x, target.y};
    }

    throw std::invalid_argument("Invalid resizeType = " + resizeType);
}

template<typename T>
inline T GetIntParam(const nvbench::state &state, const std::string &name)
{
    static_assert(std::is_integral_v<T>);

    const std::int64_t value = state.get_int64(name);
    if (value < static_cast<std::int64_t>(std::numeric_limits<T>::lowest())
        || value > static_cast<std::int64_t>(std::numeric_limits<T>::max()))
    {
        throw std::invalid_argument("Benchmark parameter " + name + " is out of range");
    }

    return static_cast<T>(value);
}

// The "inputKind" config axis selects which nvcv container a benchmark builds for
// its batch: a single dense Tensor, or an ImageBatchVarShape. Mirrors the other
// string-axis helpers (GetBorderType, GetInterpolationType): maps the axis string
// to a typed value and throws on anything unexpected (no silent fallback).
enum class InputKind
{
    Tensor,
    VarShape
};

inline InputKind GetInputKind(const std::string &inputKind)
{
    if (inputKind == "Tensor")
    {
        return InputKind::Tensor;
    }
    if (inputKind == "VarShape")
    {
        return InputKind::VarShape;
    }
    throw std::invalid_argument("Unexpected inputKind = " + inputKind);
}

inline std::string GetStringParam(const nvbench::state &state, std::string_view name, std::string_view defaultValue)
{
    return state.get_string_or_default(std::string{name}, std::string{defaultValue});
}

template<typename T>
inline nvcv::DataType GetDataType()
{
#define CVCUDA_BENCH_GET_DATA_TYPE(TYPE, DATA_TYPE) \
    if constexpr (std::is_same_v<T, TYPE>)          \
    {                                               \
        return DATA_TYPE;                           \
    }

    CVCUDA_BENCH_GET_DATA_TYPE(uint8_t, nvcv::TYPE_U8);
    CVCUDA_BENCH_GET_DATA_TYPE(uint16_t, nvcv::TYPE_U16);
    CVCUDA_BENCH_GET_DATA_TYPE(uint32_t, nvcv::TYPE_U32);

    CVCUDA_BENCH_GET_DATA_TYPE(uchar3, nvcv::TYPE_3U8);
    CVCUDA_BENCH_GET_DATA_TYPE(uchar4, nvcv::TYPE_4U8);
    CVCUDA_BENCH_GET_DATA_TYPE(float, nvcv::TYPE_F32);

    CVCUDA_BENCH_GET_DATA_TYPE(float3, nvcv::TYPE_3F32);
    CVCUDA_BENCH_GET_DATA_TYPE(float4, nvcv::TYPE_4F32);

    CVCUDA_BENCH_GET_DATA_TYPE(int, nvcv::TYPE_S32);

    CVCUDA_BENCH_GET_DATA_TYPE(short, nvcv::TYPE_S16);

    CVCUDA_BENCH_GET_DATA_TYPE(ushort3, nvcv::TYPE_3U16);
    CVCUDA_BENCH_GET_DATA_TYPE(ushort4, nvcv::TYPE_4U16);
    CVCUDA_BENCH_GET_DATA_TYPE(short4, nvcv::TYPE_4S16);

#undef CVCUDA_BENCH_GET_DATA_TYPE

    throw std::invalid_argument("Unexpected data type");
}

template<typename T>
inline nvcv::ImageFormat GetFormat()
{
    nvcv::DataType dataType = GetDataType<T>();
    nvcv::Swizzle  swizzle;
    if constexpr (nvcv::cuda::NumElements<T> == 1)
    {
        swizzle = nvcv::Swizzle::S_X000;
    }
    else if constexpr (nvcv::cuda::NumElements<T> == 2)
    {
        swizzle = nvcv::Swizzle::S_XY00;
    }
    else if constexpr (nvcv::cuda::NumElements<T> == 3)
    {
        swizzle = nvcv::Swizzle::S_XYZ0;
    }
    else
    {
        static_assert(nvcv::cuda::NumElements<T> == 4, "Unexpected benchmark image channel count");
        swizzle = nvcv::Swizzle::S_XYZW;
    }
    return nvcv::ImageFormat{nvcv::MemLayout::PL, dataType.dataKind(), swizzle, dataType.packing()};
}

template<typename T>
inline nvcv::ImageFormat GetRGBFormat()
{
    using BT        = typename nvcv::cuda::BaseType<T>;
    constexpr int C = nvcv::cuda::NumElements<T>;

    static_assert(C == 3, "RGB benchmark formats require three channels");
    if constexpr (std::is_same_v<BT, uint8_t>)
    {
        return nvcv::FMT_RGB8;
    }
    else if constexpr (std::is_same_v<BT, float>)
    {
        return nvcv::FMT_RGBf32;
    }
    else
    {
        throw std::invalid_argument("Unsupported RGB benchmark data type");
    }
}

inline NVCVBorderType GetBorderType(const std::string &border)
{
#define CVCUDA_BENCH_GET_BORDER_TYPE(BORDER) \
    if (border == #BORDER)                   \
    {                                        \
        return NVCV_BORDER_##BORDER;         \
    }

    CVCUDA_BENCH_GET_BORDER_TYPE(CONSTANT);
    CVCUDA_BENCH_GET_BORDER_TYPE(REPLICATE);
    CVCUDA_BENCH_GET_BORDER_TYPE(REFLECT);
    CVCUDA_BENCH_GET_BORDER_TYPE(WRAP);
    CVCUDA_BENCH_GET_BORDER_TYPE(REFLECT101);

#undef CVCUDA_BENCH_GET_BORDER_TYPE

    throw std::invalid_argument("Unexpected border type = " + border);
}

inline NVCVNormType GetNormType(const std::string &normType)
{
#define CVCUDA_BENCH_GET_NORM_TYPE(NORM) \
    if (normType == #NORM)               \
    {                                    \
        return NVCV_NORM_##NORM;         \
    }

    CVCUDA_BENCH_GET_NORM_TYPE(HAMMING);
    CVCUDA_BENCH_GET_NORM_TYPE(L1);
    CVCUDA_BENCH_GET_NORM_TYPE(L2);

#undef CVCUDA_BENCH_GET_NORM_TYPE

    throw std::invalid_argument("Unexpected norm type = " + normType);
}

inline NVCVInterpolationType GetInterpolationType(const std::string &interpolation)
{
#define CVCUDA_BENCH_GET_INTERPOLATION_TYPE(INTERP) \
    if (interpolation == #INTERP)                   \
    {                                               \
        return NVCV_INTERP_##INTERP;                \
    }

    CVCUDA_BENCH_GET_INTERPOLATION_TYPE(NEAREST);
    CVCUDA_BENCH_GET_INTERPOLATION_TYPE(LINEAR);
    CVCUDA_BENCH_GET_INTERPOLATION_TYPE(CUBIC);
    CVCUDA_BENCH_GET_INTERPOLATION_TYPE(AREA);
    CVCUDA_BENCH_GET_INTERPOLATION_TYPE(LANCZOS);
    CVCUDA_BENCH_GET_INTERPOLATION_TYPE(GAUSSIAN);
    CVCUDA_BENCH_GET_INTERPOLATION_TYPE(HAMMING);
    CVCUDA_BENCH_GET_INTERPOLATION_TYPE(BOX);

#undef CVCUDA_BENCH_GET_INTERPOLATION_TYPE

    throw std::invalid_argument("Unexpected interpolation type = " + interpolation);
}

template<typename T, class VecType, typename ST, typename RT = std::conditional_t<std::is_const_v<VecType>, const T, T>>
inline RT &ValueAt(VecType &vec, const ST &strides, const ST &coord)
{
    return *reinterpret_cast<RT *>(&vec[nvcv::cuda::dot(coord, strides)]);
}

static std::default_random_engine DefaultGenerator(unsigned long int seed = 0)
{
    static std::default_random_engine defaultRandomGenerator{std::random_device{}()};

    defaultRandomGenerator.seed(seed);

    return defaultRandomGenerator;
}

template<typename VT>
struct LcgGenerator
{
    using BT = nvcv::cuda::BaseType<VT>;
    using RE = std::default_random_engine;
    using UD = std::conditional_t<std::is_floating_point_v<BT>, std::uniform_real_distribution<BT>,
                                  std::uniform_int_distribution<BT>>;

    VT operator()()
    {
        VT ret;
        for (int i = 0; i < nvcv::cuda::NumElements<VT>; ++i)
        {
            nvcv::cuda::GetElement(ret, i) = uniformDistribution(randomGenerator);
        }
        return ret;
    }

    VT operator()(const long4_16a &)
    {
        return operator()();
    }

    UD uniformDistribution;
    RE randomGenerator;
};

template<typename VT, typename R = LcgGenerator<VT>, typename BT = typename R::BT, typename RE = typename R::RE,
         typename UD = typename R::UD>
inline auto LcgValues(BT min = std::is_integral_v<BT> ? nvcv::cuda::TypeTraits<BT>::min : -1,
                      BT max = std::is_integral_v<BT> ? nvcv::cuda::TypeTraits<BT>::max : +1,
                      RE rng = DefaultGenerator())
{
    return R{UD(min, max), rng};
}

/**
 * Per-element checkerboard generator: ((sum of coords) & 1) ? hi : lo.
 *
 * Faster than LcgValues<T>() for filling host buffers prior to cudaMemcpy:
 * a deterministic per-element computation avoids the per-element cost of
 * std::uniform_int_distribution while still producing varied data so the
 * kernel doesn't take a constant-input fast path.
 *
 * Uses (x + y + z + w) & 1 so the pattern is correct regardless of buffer rank
 * (1-D up to 4-D, including the 2-D per-image fills used by FillImageBatch).
 */
template<typename VT>
struct Checkerboard
{
    using BT = nvcv::cuda::BaseType<VT>;

    VT operator()(const long4_16a &coord) const
    {
        const bool on  = ((coord.x + coord.y + coord.z + coord.w) & 1L) != 0;
        const BT   val = on ? hi : lo;
        VT         ret;
        for (int i = 0; i < nvcv::cuda::NumElements<VT>; ++i)
        {
            nvcv::cuda::GetElement(ret, i) = val;
        }
        return ret;
    }

    VT operator()() const
    {
        VT ret;
        for (int i = 0; i < nvcv::cuda::NumElements<VT>; ++i)
        {
            nvcv::cuda::GetElement(ret, i) = hi;
        }
        return ret;
    }

    BT hi;
    BT lo;
};

template<typename VT, typename BT = nvcv::cuda::BaseType<VT>>
inline auto CheckerboardValues(BT hi = std::is_integral_v<BT> ? nvcv::cuda::TypeTraits<BT>::max : static_cast<BT>(1),
                               BT lo = static_cast<BT>(0))
{
    return Checkerboard<VT>{hi, lo};
}

template<typename VT, typename ST, class VG>
inline void FillBufferPlane(std::vector<uint8_t> &vec, const ST &shape, const ST &strides, VG valuesGenerator, long x,
                            long y, long z)
{
    for (long w = 0; w < (nvcv::cuda::NumElements<ST> == 4 ? nvcv::cuda::GetElement(shape, 3) : 1); ++w)
    {
        long4_16a coord16a{x, y, z, w};
        ST        stCoord = nvcv::cuda::DropCast<nvcv::cuda::NumElements<ST>>(
            nvcv::cuda::StaticCast<nvcv::cuda::BaseType<ST>>(coord16a));

        ValueAt<VT>(vec, strides, stCoord) = valuesGenerator(coord16a);
    }
}

template<typename VT, typename ST, class VG>
inline void FillBuffer(std::vector<uint8_t> &vec, const ST &shape, const ST &strides, VG valuesGenerator)
{
    for (long x = 0; x < (nvcv::cuda::NumElements<ST> >= 1 ? nvcv::cuda::GetElement(shape, 0) : 1); ++x)
    {
        for (long y = 0; y < (nvcv::cuda::NumElements<ST> >= 2 ? nvcv::cuda::GetElement(shape, 1) : 1); ++y)
        {
            for (long z = 0; z < (nvcv::cuda::NumElements<ST> >= 3 ? nvcv::cuda::GetElement(shape, 2) : 1); ++z)
            {
                FillBufferPlane<VT>(vec, shape, strides, valuesGenerator, x, y, z);
            }
        }
    }
}

template<typename VT, int RANK, class VG>
inline void FillTensor(const nvcv::Tensor &tensor, VG valuesGenerator)
{
    using longR = nvcv::cuda::MakeType<long, RANK>;

    auto tensorData = tensor.exportData<nvcv::TensorDataStridedCuda>();
    CVCUDA_CHECK_DATA(tensorData);

    longR strides;
    longR shape;

    for (int i = 0; i < RANK; ++i)
    {
        nvcv::cuda::GetElement(strides, i) = tensorData->stride(i);
        nvcv::cuda::GetElement(shape, i)   = tensorData->shape(i);
    }

    long bufSize{nvcv::cuda::GetElement(strides, 0) * nvcv::cuda::GetElement(shape, 0)};

    // GPU-side fast paths for the bench-internal generators. Avoids the
    // host fill + cudaMemcpy round-trip (which on the C++ side was burning
    // 1-6 s per fill on multi-hundred-MB tensors). Each typed kernel writes
    // BT-typed elements directly to device memory and matches the default
    // host-generator distribution (full type range for integers; [-1, +1]
    // for floats), so swapping host -> GPU is a no-op for downstream kernels.
    if constexpr (std::is_same_v<VG, LcgGenerator<VT>>)
    {
        using BT = typename LcgGenerator<VT>::BT;
        // Only take the GPU path when the host generator's range is the
        // default full range — that's what the big-tensor src fills use.
        // Anything narrower (e.g. LcgValues<float>(0.f, 1.f) for the
        // small parameter tensors) falls through to the host implementation
        // so the value distribution stays identical.
        // Range-match check: the GPU LCG kernel produces full type range for
        // ints (matching LcgValues<T>()'s integer default), and [-1, +1]
        // for floats (matching LcgValues<T>()'s float default — note that
        // TypeTraits<float>::min is FLT_MIN, not -1, so we can't use it here).
        const bool defaultRange = std::is_floating_point_v<BT>
                                    ? (valuesGenerator.uniformDistribution.min() == static_cast<BT>(-1)
                                       && valuesGenerator.uniformDistribution.max() == static_cast<BT>(1))
                                    : (valuesGenerator.uniformDistribution.min() == nvcv::cuda::TypeTraits<BT>::min
                                       && valuesGenerator.uniformDistribution.max() == nvcv::cuda::TypeTraits<BT>::max);
        if (defaultRange)
        {
            const size_t n_elements = static_cast<size_t>(bufSize) / sizeof(BT);
            launchRandomFillTyped<BT>(reinterpret_cast<BT *>(tensorData->basePtr()), n_elements,
                                      /*seed*/ 0x9e3779b97f4a7c15ULL, /*stream*/ nullptr);
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
            return;
        }
    }
    if constexpr (std::is_same_v<VG, Checkerboard<VT>>)
    {
        using BT = typename Checkerboard<VT>::BT;
        // Per-element coord-sum pattern matching FillBuffer's iteration over
        // a 4-D shape. Strides come straight from the tensor in BT-element
        // units (bench tensors are tightly packed, inner-most stride == 1).
        const size_t n_elements = static_cast<size_t>(bufSize) / sizeof(BT);
        const size_t s0
            = (RANK >= 1) ? static_cast<size_t>(nvcv::cuda::GetElement(strides, 0)) / sizeof(BT) : n_elements;
        const size_t s1 = (RANK >= 2) ? static_cast<size_t>(nvcv::cuda::GetElement(strides, 1)) / sizeof(BT) : 1;
        const size_t s2 = (RANK >= 3) ? static_cast<size_t>(nvcv::cuda::GetElement(strides, 2)) / sizeof(BT) : 1;
        launchCheckerboardTensorTyped<BT>(reinterpret_cast<BT *>(tensorData->basePtr()), n_elements, s0, s1, s2,
                                          valuesGenerator.hi, valuesGenerator.lo, /*stream*/ nullptr);
        CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        return;
    }

    // Fallback: arbitrary generator (lambdas, etc.) — host fill then memcpy.
    std::vector<uint8_t> tensorVec(bufSize);

    FillBuffer<VT>(tensorVec, shape, strides, valuesGenerator);

    CUDA_CHECK_ERROR(cudaMemcpy(tensorData->basePtr(), tensorVec.data(), bufSize, cudaMemcpyHostToDevice));
}

template<typename VT, class VG>
inline void FillTensor(const nvcv::Tensor &tensor, VG valuesGenerator)
{
    switch (tensor.rank())
    {
#define CVCUDA_BENCH_FILL_TENSOR_CASE(RANK)        \
case RANK:                                         \
    FillTensor<VT, RANK>(tensor, valuesGenerator); \
    break

        CVCUDA_BENCH_FILL_TENSOR_CASE(1);
        CVCUDA_BENCH_FILL_TENSOR_CASE(2);
        CVCUDA_BENCH_FILL_TENSOR_CASE(3);
        CVCUDA_BENCH_FILL_TENSOR_CASE(4);

#undef CVCUDA_BENCH_FILL_TENSOR_CASE
    default:
        throw std::invalid_argument("Tensor has rank not in [1, 4]");
    }
}

template<typename VT, class VG>
inline void FillImageBatch(nvcv::ImageBatchVarShape &imageBatch, long2 size, long2 varSize, VG valuesGenerator,
                           nvcv::ImageFormat format = GetFormat<VT>())
{
    auto randomWidth  = LcgValues<int>(static_cast<int>(size.x - varSize.x), static_cast<int>(size.x));
    auto randomHeight = LcgValues<int>(static_cast<int>(size.y - varSize.y), static_cast<int>(size.y));

    for (int i = 0; i < imageBatch.capacity(); ++i)
    {
        nvcv::Image image(nvcv::Size2D{randomWidth(), randomHeight()}, format);

        auto data = image.exportData<nvcv::ImageDataStridedCuda>();
        CVCUDA_CHECK_DATA(data);

        long2 strides{data->plane(0).rowStride, sizeof(VT)};
        long2 shape{data->plane(0).height, data->plane(0).width};

        long bufSize = strides.x * shape.x;

        // GPU-side fast paths — same conditions as FillTensor. Avoids the
        // per-image host fill + cudaMemcpy2D round-trip.
        bool gpu_path = false;
        if constexpr (std::is_same_v<VG, LcgGenerator<VT>>)
        {
            using BT = typename LcgGenerator<VT>::BT;
            // See FillTensor's matching range check for the float vs. int rationale.
            const bool defaultRange
                = std::is_floating_point_v<BT>
                    ? (valuesGenerator.uniformDistribution.min() == static_cast<BT>(-1)
                       && valuesGenerator.uniformDistribution.max() == static_cast<BT>(1))
                    : (valuesGenerator.uniformDistribution.min() == nvcv::cuda::TypeTraits<BT>::min
                       && valuesGenerator.uniformDistribution.max() == nvcv::cuda::TypeTraits<BT>::max);
            if (defaultRange)
            {
                const size_t n_elements = static_cast<size_t>(bufSize) / sizeof(BT);
                launchRandomFillTyped<BT>(reinterpret_cast<BT *>(data->plane(0).basePtr), n_elements,
                                          /*seed*/ 0x9e3779b97f4a7c15ULL + static_cast<std::uint64_t>(i),
                                          /*stream*/ nullptr);
                gpu_path = true;
            }
        }
        else if constexpr (std::is_same_v<VG, Checkerboard<VT>>)
        {
            using BT                  = typename Checkerboard<VT>::BT;
            const size_t n_elements   = static_cast<size_t>(bufSize) / sizeof(BT);
            const size_t row_pitch_T  = std::max<size_t>(1, static_cast<size_t>(strides.x) / sizeof(BT));
            const size_t pixel_size_T = std::max<size_t>(1, sizeof(VT) / sizeof(BT));
            launchCheckerboardImageTyped<BT>(reinterpret_cast<BT *>(data->plane(0).basePtr), n_elements, row_pitch_T,
                                             pixel_size_T, valuesGenerator.hi, valuesGenerator.lo, /*stream*/ nullptr);
            gpu_path = true;
        }

        if (!gpu_path)
        {
            std::vector<uint8_t> imageBuffer(bufSize);
            FillBuffer<VT>(imageBuffer, shape, strides, valuesGenerator);
            CUDA_CHECK_ERROR(cudaMemcpy2D(data->plane(0).basePtr, strides.x, imageBuffer.data(), strides.x, strides.x,
                                          data->plane(0).height, cudaMemcpyHostToDevice));
        }

        imageBatch.pushBack(image);
    }

    // Make sure all per-image GPU-fill kernels are done before the bench reads them.
    if constexpr (std::is_same_v<VG, LcgGenerator<VT>> || std::is_same_v<VG, Checkerboard<VT>>)
    {
        CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
    }
}

/**
 * Fill an ImageBatchVarShape with images matching the shapes from a source batch.
 *
 * This ensures dst has the same per-sample dimensions as src, which is required
 * by most CV-CUDA operators. Use this instead of FillImageBatch for dst when
 * src/dst shape matching is required.
 *
 * @param imageBatch The destination ImageBatchVarShape to fill (must have sufficient capacity)
 * @param srcBatch The source ImageBatchVarShape to copy shapes from
 * @param valuesGenerator A callable that takes (const long4_16a &) and returns VT
 *
 * Example:
 *   benchutils::FillImageBatch<T>(src, long2{W, H}, long2{0, 0}, LcgValues<T>());
 *   benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
 */
template<typename VT, class VG>
inline void FillImageBatchLike(nvcv::ImageBatchVarShape &imageBatch, const nvcv::ImageBatchVarShape &srcBatch,
                               VG valuesGenerator, std::optional<nvcv::ImageFormat> requestedFormat = std::nullopt)
{
    for (int i = 0; i < srcBatch.numImages(); ++i)
    {
        nvcv::Image srcImage = srcBatch[i];
        auto        srcData  = srcImage.exportData<nvcv::ImageDataStridedCuda>();
        CVCUDA_CHECK_DATA(srcData);

        nvcv::ImageFormat format = requestedFormat.value_or(srcImage.format());
        if (!requestedFormat && (format.numPlanes() != 1 || format.planeDataType(0) != GetDataType<VT>()))
        {
            // Some operators derive single-channel auxiliary batches (for example, Composite masks)
            // from RGB inputs. Preserve semantic metadata only when it describes VT.
            format = GetFormat<VT>();
        }
        nvcv::Image image(nvcv::Size2D{srcData->plane(0).width, srcData->plane(0).height}, format);

        auto data = image.exportData<nvcv::ImageDataStridedCuda>();
        CVCUDA_CHECK_DATA(data);

        long2 strides{data->plane(0).rowStride, sizeof(VT)};
        long2 shape{data->plane(0).height, data->plane(0).width};

        std::vector<uint8_t> imageBuffer(strides.x * shape.x);

        FillBuffer<VT>(imageBuffer, shape, strides, valuesGenerator);

        CUDA_CHECK_ERROR(cudaMemcpy2D(data->plane(0).basePtr, strides.x, imageBuffer.data(), strides.x, strides.x,
                                      data->plane(0).height, cudaMemcpyHostToDevice));

        imageBatch.pushBack(image);
    }
}

/**
 * Map a packed vector type to its planar (NCHW) NVCV image format, in the
 * RGB/RGBA convention: uchar3->RGB8p, uchar4->RGBA8p, float3->RGBf32p,
 * float4->RGBAf32p. Shared by the planar paths of the bench operators so the
 * mapping is not duplicated across bench translation units.
 */
template<typename T>
inline nvcv::ImageFormat GetPlanarFormat()
{
    using BT        = typename nvcv::cuda::BaseType<T>;
    constexpr int C = nvcv::cuda::NumElements<T>;

    if constexpr (std::is_same_v<BT, uint8_t> && C == 3)
    {
        return nvcv::FMT_RGB8p;
    }
    else if constexpr (std::is_same_v<BT, uint8_t> && C == 4)
    {
        return nvcv::FMT_RGBA8p;
    }
    else if constexpr (std::is_same_v<BT, float> && C == 3)
    {
        return nvcv::FMT_RGBf32p;
    }
    else if constexpr (std::is_same_v<BT, float> && C == 4)
    {
        return nvcv::FMT_RGBAf32p;
    }
    else
    {
        throw std::invalid_argument("Unsupported planar benchmark data type");
    }
}

// Fill a var-shape batch with planar images of (up to) the given size. `checker` selects between a
// checkerboard pattern (inputs) and a zero fill (outputs); the latter is useful when the destination
// images differ in size from the sources (e.g. Resize/PillowResize) and cannot be built "like" the
// source batch.
template<typename T>
inline void FillPlanarImageBatch(nvcv::ImageBatchVarShape &imageBatch, long2 size, long2 varSize, bool checker = true)
{
    using BT = typename nvcv::cuda::BaseType<T>;

    nvcv::ImageFormat format       = GetPlanarFormat<T>();
    auto              randomWidth  = LcgValues<int>(static_cast<int>(size.x - varSize.x), static_cast<int>(size.x));
    auto              randomHeight = LcgValues<int>(static_cast<int>(size.y - varSize.y), static_cast<int>(size.y));

    const BT   hi = std::is_integral_v<BT> ? nvcv::cuda::TypeTraits<BT>::max : static_cast<BT>(1);
    const auto lo = static_cast<BT>(0);

    for (int i = 0; i < imageBatch.capacity(); ++i)
    {
        nvcv::Image image(nvcv::Size2D{randomWidth(), randomHeight()}, format);

        auto data = image.exportData<nvcv::ImageDataStridedCuda>();
        CVCUDA_CHECK_DATA(data);

        for (int p = 0; p < format.numPlanes(); ++p)
        {
            if (checker)
            {
                const size_t nElements
                    = static_cast<size_t>(data->plane(p).rowStride) * data->plane(p).height / sizeof(BT);
                const size_t rowPitch = static_cast<size_t>(data->plane(p).rowStride) / sizeof(BT);

                launchCheckerboardImageTyped<BT>(reinterpret_cast<BT *>(data->plane(p).basePtr), nElements, rowPitch, 1,
                                                 hi, lo, /*stream*/ nullptr);
            }
            else
            {
                CUDA_CHECK_ERROR(cudaMemset2D(data->plane(p).basePtr, data->plane(p).rowStride, 0,
                                              data->plane(p).rowStride, data->plane(p).height));
            }
        }

        imageBatch.pushBack(image);
    }

    CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
}

// Fill a planar var-shape batch with the same deterministic LCG stream used by
// FillImageBatch's packed interleaved fast path. Use this for data-dependent
// operators whose Python benchmarks use create_image_batch_varshape(...,
// fill_mode="lcg") for planar images.
template<typename T>
inline void FillPlanarImageBatchLcg(nvcv::ImageBatchVarShape &imageBatch, long2 size, long2 varSize)
{
    using BT = typename nvcv::cuda::BaseType<T>;

    nvcv::ImageFormat format       = GetPlanarFormat<T>();
    auto              randomWidth  = LcgValues<int>(static_cast<int>(size.x - varSize.x), static_cast<int>(size.x));
    auto              randomHeight = LcgValues<int>(static_cast<int>(size.y - varSize.y), static_cast<int>(size.y));

    for (int i = 0; i < imageBatch.capacity(); ++i)
    {
        nvcv::Image image(nvcv::Size2D{randomWidth(), randomHeight()}, format);

        auto data = image.exportData<nvcv::ImageDataStridedCuda>();
        CVCUDA_CHECK_DATA(data);

        for (int p = 0; p < format.numPlanes(); ++p)
        {
            const size_t nElements = static_cast<size_t>(data->plane(p).rowStride) * data->plane(p).height / sizeof(BT);
            launchRandomFillTyped<BT>(reinterpret_cast<BT *>(data->plane(p).basePtr), nElements,
                                      /*seed*/ 0x9e3779b97f4a7c15ULL + static_cast<std::uint64_t>(i),
                                      /*stream*/ nullptr);
        }

        imageBatch.pushBack(image);
    }

    CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
}

/**
 * Fill an ImageBatchVarShape with planar (NCHW) images matching the per-sample
 * sizes of a source batch, zero-initialized. Use for dst when src/dst shapes
 * must match (e.g. Normalize).
 */
template<typename T>
inline void FillPlanarImageBatchLike(nvcv::ImageBatchVarShape &imageBatch, const nvcv::ImageBatchVarShape &srcBatch)
{
    nvcv::ImageFormat format = GetPlanarFormat<T>();

    for (int i = 0; i < srcBatch.numImages(); ++i)
    {
        nvcv::Image srcImage = srcBatch[i];
        auto        srcData  = srcImage.exportData<nvcv::ImageDataStridedCuda>();
        CVCUDA_CHECK_DATA(srcData);

        nvcv::Image image(nvcv::Size2D{srcData->plane(0).width, srcData->plane(0).height}, format);

        auto data = image.exportData<nvcv::ImageDataStridedCuda>();
        CVCUDA_CHECK_DATA(data);

        for (int p = 0; p < format.numPlanes(); ++p)
        {
            CUDA_CHECK_ERROR(cudaMemset2D(data->plane(p).basePtr, data->plane(p).rowStride, 0, data->plane(p).rowStride,
                                          data->plane(p).height));
        }

        imageBatch.pushBack(image);
    }

    CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
}

/**
 * Run warmup iterations to stabilize GPU state before benchmarking.
 *
 * This function runs the operator multiple times before the timed benchmark loop.
 * This helps stabilize GPU frequency/power state and can reduce benchmark noise,
 * especially for operators with small execution times or non-power-of-2 data types.
 *
 * Creates and destroys a CUDA stream internally. If iterations is 0, does nothing.
 *
 * @param iterations Number of warmup iterations (0 = disabled, use BENCH_<OP>_WARMUP_ITERATIONS from config)
 * @param func The operator invocation lambda, taking cudaStream_t as parameter
 *
 * Example:
 *   benchutils::warmup(BENCH_STACK_WARMUP_ITERATIONS, [&op, &src, &dst](cudaStream_t s) {
 *       op(s, src, dst);
 *   });
 */
template<typename Func>
inline void warmup(int iterations, Func &&func)
{
    iterations = resolve_warmup_iterations(iterations);
    if (iterations <= 0)
        return;

    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));
    for (int i = 0; i < iterations; ++i)
    {
        func(stream);
    }
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));
}

/**
 * Execute benchmark with sync tag enabled (matching Python benchmarks' sync=True default).
 *
 * This helper function wraps state.exec() with exec_tag::sync, which:
 * - Disables nvbench's deadlock detection
 * - Uses CPU-based timing instead of GPU events
 * - Indicates that the kernel may perform internal synchronization
 *
 * All CV-CUDA operators may perform internal synchronization (especially in VarShape paths),
 * so this should be used by default for all benchmarks to match Python behavior.
 *
 * @param state The nvbench state object
 * @param func The lambda/function to execute, taking nvbench::launch& as parameter
 *
 * Example:
 *   benchutils::exec_with_sync(state, [&op, &src, &dst](nvbench::launch &launch) {
 *       op(launch.get_stream(), src, dst);
 *   });
 */
template<typename Func>
inline void exec_with_sync(nvbench::state &state, Func &&func)
{
    state.exec(nvbench::exec_tag::sync, std::forward<Func>(func));
}

/**
 * Combined warmup and benchmark execution.
 *
 * Runs warmup iterations then executes the timed benchmark, avoiding lambda duplication.
 * The provided function should take a cudaStream_t parameter.
 *
 * @param state The nvbench state object
 * @param warmup_iterations Number of warmup iterations (0 = disabled)
 * @param func The operator invocation lambda, taking cudaStream_t as parameter
 *
 * Example:
 *   benchutils::warmup_and_exec(state, BENCH_COMPOSITE_WARMUP_ITERATIONS,
 *       [&op, &fg, &bg, &mask, &dst](cudaStream_t s) { op(s, fg, bg, mask, dst); });
 */
template<typename Func>
inline void warmup_and_exec(nvbench::state &state, int warmup_iterations, Func &&func)
{
    warmup(warmup_iterations, func);
    exec_with_sync(state, [&func](nvbench::launch &launch) { func(launch.get_stream()); });
}

} // namespace benchutils

#endif // CVCUDA_CPP_BENCH_UTILS_HPP

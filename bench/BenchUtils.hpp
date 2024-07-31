/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_BENCH_UTILS_HPP
#define CVCUDA_BENCH_UTILS_HPP

#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/BorderType.h>
#include <nvcv/DataType.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CVCUDA_CHECK_DATA(data)                   \
    if (!data)                                    \
    {                                             \
        throw std::runtime_error("Invalid data"); \
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
    return nvcv::ImageFormat{GetDataType<T>()};
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
struct Randomizer
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

    VT operator()(const long4 &)
    {
        return operator()();
    }

    UD uniformDistribution;
    RE randomGenerator;
};

template<typename VT, typename R = Randomizer<VT>, typename BT = typename R::BT, typename RE = typename R::RE,
         typename UD = typename R::UD>
inline auto RandomValues(BT min = std::is_integral_v<BT> ? nvcv::cuda::TypeTraits<BT>::min : -1,
                         BT max = std::is_integral_v<BT> ? nvcv::cuda::TypeTraits<BT>::max : +1,
                         RE rng = DefaultGenerator())
{
    return R{UD(min, max), rng};
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
                for (long w = 0; w < (nvcv::cuda::NumElements<ST> == 4 ? nvcv::cuda::GetElement(shape, 3) : 1); ++w)
                {
                    long4 coord{x, y, z, w};
                    ST    stCoord = nvcv::cuda::DropCast<nvcv::cuda::NumElements<ST>>(coord);

                    ValueAt<VT>(vec, strides, stCoord) = valuesGenerator(coord);
                }
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

    longR strides, shape;

    for (int i = 0; i < RANK; ++i)
    {
        nvcv::cuda::GetElement(strides, i) = tensorData->stride(i);
        nvcv::cuda::GetElement(shape, i)   = tensorData->shape(i);
    }

    long bufSize{nvcv::cuda::GetElement(strides, 0) * nvcv::cuda::GetElement(shape, 0)};

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
inline void FillImageBatch(nvcv::ImageBatchVarShape &imageBatch, long2 size, long2 varSize, VG valuesGenerator)
{
    auto randomWidth  = RandomValues<int>(static_cast<int>(size.x - varSize.x), static_cast<int>(size.x));
    auto randomHeight = RandomValues<int>(static_cast<int>(size.y - varSize.y), static_cast<int>(size.y));

    for (int i = 0; i < imageBatch.capacity(); ++i)
    {
        nvcv::Image image(nvcv::Size2D{randomWidth(), randomHeight()}, GetFormat<VT>());

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

} // namespace benchutils

#endif // CVCUDA_BENCH_UTILS_HPP

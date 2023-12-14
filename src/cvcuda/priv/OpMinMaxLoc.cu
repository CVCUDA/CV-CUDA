/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpMinMaxLoc.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/cuda/Atomics.hpp>
#include <nvcv/cuda/ImageBatchVarShapeWrap.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <nvcv/cuda/TensorWrap.hpp>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cub/cub.cuh>

namespace {

// Utilities for MinMaxLoc operator --------------------------------------------

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

using TensorDataRef         = std::reference_wrapper<nvcv::TensorDataStridedCuda>;
using OptionalTensorData    = nvcv::Optional<nvcv::TensorDataStridedCuda>;
using OptionalTensorDataRef = nvcv::Optional<std::reference_wrapper<nvcv::TensorDataStridedCuda>>;

// Output type used in storing min/max values: U32 for U32/U16/U8 or S32 for S32/S16/S8 or F32/F64 for F32/F64

template<typename T>
using OutputType = std::conditional_t<
    std::is_same_v<T, char1> || std::is_same_v<T, short1> || std::is_same_v<T, int1>, int1,
    std::conditional_t<std::is_same_v<T, uchar1> || std::is_same_v<T, ushort1> || std::is_same_v<T, uint1>, uint1, T>>;

// OutputWrapper1,2 is used to wrap 1 or 2 outputs with minimum or maximum or both values

template<typename T>
struct OutputWrapper1
{
    explicit OutputWrapper1(T &&w0)
        : wrap0(std::forward<T>(w0))
    {
    }

    T wrap0;
};

template<typename T>
struct OutputWrapper2
{
    explicit OutputWrapper2(T &&w0, T &&w1)
        : wrap0(std::forward<T>(w0))
        , wrap1(std::forward<T>(w1))
    {
    }

    T wrap0, wrap1;
};

template<typename T>
auto OutputWrapper(T &&w0)
{
    return OutputWrapper1(std::move(w0));
}

template<typename T>
auto OutputWrapper(T &&w0, T &&w1)
{
    return OutputWrapper2(std::move(w0), std::move(w1));
}

template<int I, class OutWrapper>
__device__ inline auto get(OutWrapper out)
{
    if constexpr (I == 0)
    {
        return out.wrap0;
    }
    else
    {
        static_assert(I == 1);
        return out.wrap1;
    }
}

// OpMin used when finding only minimum value

template<typename T>
struct OpMin
{
    using OutType     = OutputType<T>;
    using BaseOutType = cuda::BaseType<OutType>;

    static constexpr OutType init = {cuda::TypeTraits<OutType>::max};

    template<class OutWrapper>
    __device__ inline static void initFill(OutWrapper out, int z)
    {
        get<0>(out)[z] = init;
    }

    template<typename U>
    __device__ inline static void op(OutType &a, U b)
    {
        a = cuda::min(a, cuda::StaticCast<BaseOutType>(b));
    }

    template<class OutWrapper>
    __device__ inline static void opAtomic(OutWrapper out, int z, OutType b)
    {
        cuda::AtomicMin(get<0>(out)[z].x, b.x);
    }
};

// OpMax used when finding only maximum value

template<typename T>
struct OpMax
{
    using OutType     = OutputType<T>;
    using BaseOutType = cuda::BaseType<OutType>;

    static constexpr OutType init = {cuda::Lowest<OutType>};

    template<class OutWrapper>
    __device__ inline static void initFill(OutWrapper out, int z)
    {
        get<0>(out)[z] = init;
    }

    template<typename U>
    __device__ inline static void op(OutType &a, U b)
    {
        a = cuda::max(a, cuda::StaticCast<BaseOutType>(b));
    }

    template<class OutWrapper>
    __device__ inline static void opAtomic(OutWrapper out, int z, OutType b)
    {
        cuda::AtomicMax(get<0>(out)[z].x, b.x);
    }
};

// OpMinMax used when finding both minimum and maximum values

template<typename T>
struct OpMinMax
{
    using BaseOutType = cuda::BaseType<OutputType<T>>;
    using OutType     = cuda::MakeType<BaseOutType, 2>;

    static constexpr OutType init = {cuda::TypeTraits<OutType>::max, cuda::Lowest<OutType>};

    template<class OutWrapper>
    __device__ inline static void initFill(OutWrapper out, int z)
    {
        get<0>(out)[z] = {init.x};
        get<1>(out)[z] = {init.y};
    }

    template<typename U>
    __device__ inline static void op(OutType &a, U b)
    {
        if constexpr (cuda::NumElements<U> == 2)
        {
            a.x = cuda::min(a.x, static_cast<BaseOutType>(b.x));
            a.y = cuda::max(a.y, static_cast<BaseOutType>(b.y));
        }
        else
        {
            static_assert(cuda::NumElements<U> == 1 && cuda::NumComponents<U> == 1);

            a.x = cuda::min(a.x, static_cast<BaseOutType>(b.x));
            a.y = cuda::max(a.y, static_cast<BaseOutType>(b.x));
        }
    }

    template<class OutWrapper>
    __device__ inline static void opAtomic(OutWrapper out, int z, OutType b)
    {
        cuda::AtomicMin(get<0>(out)[z].x, b.x);
        cuda::AtomicMax(get<1>(out)[z].x, b.y);
    }
};

// Get capacity shape index given the tensor rank for min(max)LocData that can be rank 1 or 2 or 3, meaning
// [M], [NM] or [NMC] tensors with N batches, M capacity and C channels

__host__ inline int GetCapacityIdx(int rank)
{
    return rank == 1 ? 0 : 1;
}

// OpCollect1 used when collecting locations for either minimum or maximum values

template<class LocWrapper, class NumWrapper>
struct OpCollect1
{
    using IdxGlobalType = int;

    __host__ OpCollect1(const LocWrapper &coords, const NumWrapper &coordsSize, int coordsCapacity)
        : m_coords(coords)
        , m_coordsSize(coordsSize)
        , m_coordsCapacity(coordsCapacity)
    {
    }

    template<class T, class OutWrapper>
    __device__ inline static unsigned checkRange(T data, OutWrapper out, int z)
    {
        return static_cast<cuda::BaseType<OutputType<T>>>(data.x) == get<0>(out)[z].x;
    }

    inline __device__ IdxGlobalType atomicIncrementSize(int z, unsigned total) const
    {
        return atomicAdd(&m_coordsSize[z].x, static_cast<int>(total));
    }

    __device__ inline bool isIndexInsideBounds(IdxGlobalType idxGlobal, unsigned total) const
    {
        return total > 0 && idxGlobal < m_coordsCapacity;
    }

    __device__ void writeOutput(unsigned has, int2 coord, int z, IdxGlobalType idxGlobal, unsigned idxLocal)
    {
        if (has)
        {
            int idx = idxGlobal + idxLocal;
            if (idx < m_coordsCapacity)
            {
                *m_coords.ptr(z, idx) = coord;
            }
        }
    }

private:
    LocWrapper m_coords;
    NumWrapper m_coordsSize;
    int        m_coordsCapacity;
};

// OpCollect2 used when collecting locations for both minimum and maximum values

template<class LocWrapper, class NumWrapper>
struct OpCollect2
{
    using IdxGlobalType = int2;

    __host__ OpCollect2(const LocWrapper &minCoords, const NumWrapper &minCoordsSize, int minCoordsCapacity,
                        const LocWrapper &maxCoords, const NumWrapper &maxCoordsSize, int maxCoordsCapacity)
        : m_minCoords(minCoords)
        , m_minCoordsSize(minCoordsSize)
        , m_minCoordsCapacity(minCoordsCapacity)
        , m_maxCoords(maxCoords)
        , m_maxCoordsSize(maxCoordsSize)
        , m_maxCoordsCapacity(maxCoordsCapacity)
    {
    }

    template<class T, class OutWrapper>
    __device__ static unsigned checkRange(T data, OutWrapper out, int z)
    {
        return ((static_cast<cuda::BaseType<OutputType<T>>>(data.x) == get<0>(out)[z].x)
                | ((static_cast<cuda::BaseType<OutputType<T>>>(data.x) == get<1>(out)[z].x) << 16));
    }

    inline __device__ IdxGlobalType atomicIncrementSize(int z, unsigned total) const
    {
        return {atomicAdd(&m_minCoordsSize[z].x, static_cast<int>(total & USHRT_MAX)),
                atomicAdd(&m_maxCoordsSize[z].x, static_cast<int>(total >> 16))};
    }

    __device__ inline bool isIndexInsideBounds(IdxGlobalType idxGlobal, unsigned total) const
    {
        return ((total & USHRT_MAX) > 0 && idxGlobal.x < m_minCoordsCapacity)
            || ((total >> 16) > 0 && idxGlobal.y < m_maxCoordsCapacity);
    }

    __device__ void writeOutput(unsigned has, int2 coord, int z, IdxGlobalType idxGlobal, unsigned idxLocal)
    {
        if (has & 1)
        {
            int idx = idxGlobal.x + idxLocal & ((1 << 16) - 1);
            if (idx < m_minCoordsCapacity)
            {
                *m_minCoords.ptr(z, idx) = coord;
            }
        }

        if (has >> 16)
        {
            int idx = idxGlobal.y + (idxLocal >> 16);
            if (idx < m_maxCoordsCapacity)
            {
                *m_maxCoords.ptr(z, idx) = coord;
            }
        }
    }

private:
    LocWrapper m_minCoords, m_maxCoords;
    NumWrapper m_minCoordsSize, m_maxCoordsSize;
    int        m_minCoordsCapacity, m_maxCoordsCapacity;
};

// CUDA kernels ----------------------------------------------------------------

template<class OP, class OutWrapper>
__global__ void InitMinMax(OutWrapper out)
{
    OP::initFill(out, static_cast<int>(blockIdx.z));
}

template<class OP, int BW, int BH, int TW, int TH, class InWrapper, class OutWrapper>
__global__ __launch_bounds__(BW *BH) void FindMinMax(InWrapper in, int2 size, OutWrapper out)
{
    using T = typename InWrapper::ValueType;

    int z = blockIdx.z;

    if constexpr (!std::is_same_v<InWrapper, cuda::Tensor3DWrap<T>>)
    {
        size = {in.width(z), in.height(z)};
    }

    T dataN[TW];

    int x = (blockIdx.x * blockDim.x + threadIdx.x) * TW;
    int y = blockIdx.y * BH * TH + threadIdx.y;

    auto threadRet = OP::init;

#pragma unroll
    for (int i = 0; i < TH; ++i)
    {
        if (y >= size.y)
        {
            break;
        }

        const T *row = in.ptr(z, y);

        if (x + TW <= size.x)
        {
#pragma unroll
            for (int j = 0; j < TW; ++j)
            {
                dataN[j] = row[x + j]; // TODO: Use AssignAs instead, cf. CVCUDA-641
            }

#pragma unroll
            for (int j = 0; j < TW; ++j)
            {
                OP::op(threadRet, dataN[j]);
            }
        }
        else
        {
            for (int xi = x; xi < size.x; ++xi)
            {
                OP::op(threadRet, row[xi]);
            }
        }

        y += BH;
    }

    using BlockReduce = cub::BlockReduce<typename OP::OutType, BW, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, BH>;

    __shared__ typename BlockReduce::TempStorage cubTempStorage;

    auto opMinMaxReduce = [](auto a, auto b)
    {
        OP::op(a, b);
        return a;
    };

    auto tileRange = BlockReduce(cubTempStorage).Reduce(threadRet, opMinMaxReduce);

    if (threadIdx.y == 0 && threadIdx.x == 0)
    {
        OP::opAtomic(out, z, tileRange);
    }
}

template<int BW, int BH, int TW, int TH, class OP, class InWrapper, class OutWrapper>
__global__ __launch_bounds__(BW *BH) void CollectMinMax(InWrapper in, int2 size, OutWrapper out, OP op)
{
    // We can use an unsigned to represent the position of each min/max coord in the output grid for this block.
    // --> LSW stores min, MSW stores max
    // Since we know there will be at most this many pixels processed per block, and this is less than USHORT_MAX,
    // addition will never overflow, and one simple 32-bit addition can be performed instead of 2 additions.
    static_assert(BW * BH * TW * TH < USHRT_MAX, "Too many pixels processed per block");

    using T = typename InWrapper::ValueType;

    int z = blockIdx.z;

    if constexpr (!std::is_same_v<InWrapper, cuda::Tensor3DWrap<T>>)
    {
        size = {in.width(z), in.height(z)};
    }

    T dataN[TW];

    int x = (blockIdx.x * BW + threadIdx.x) * TW;
    int y = blockIdx.y * BH * TH + threadIdx.y;

    using TIDX = unsigned;

    TIDX has[TW * TH] = {};

#pragma unroll
    for (int i = 0; i < TH; ++i)
    {
        int yi = y + i * BH;

        if (yi >= size.y)
        {
            break;
        }

        const T *row = in.ptr(z, yi);

        if (x + TW <= size.x)
        {
#pragma unroll
            for (int j = 0; j < TW; ++j)
            {
                dataN[j] = row[x + j]; // TODO: Use AssignAs instead, cf. CVCUDA-641
            }

#pragma unroll
            for (int j = 0; j < TW; ++j)
            {
                has[i * TW + j] = op.checkRange(dataN[j], out, z);
            }
        }
        else
        {
#pragma unroll
            for (int j = 0; j < TW; ++j)
            {
                int xj = x + j;
                if (xj >= size.x)
                {
                    break;
                }
                has[i * TW + j] = op.checkRange(row[xj], out, z);
            }
        }
    }

    __shared__ typename OP::IdxGlobalType idxGlobal_;

    // LSW stores min, MSW stores max
    TIDX idxLocal[TW * TH];
    TIDX totalLocal;
    {
        using BlockScan = cub::BlockScan<TIDX, BW, cub::BLOCK_SCAN_RAKING, BH>;
        __shared__ typename BlockScan::TempStorage scan_storage;

        // Can't use ExclusiveSum directly as with cub-1.8.0 as it doesnt' work with composite types.
        BlockScan(scan_storage).ExclusiveSum(has, idxLocal, totalLocal);

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            idxGlobal_ = op.atomicIncrementSize(z, totalLocal);
        }
    }

    __syncthreads();

    typename OP::IdxGlobalType idxGlobal = idxGlobal_;

    if (op.isIndexInsideBounds(idxGlobal, totalLocal))
    {
#pragma unroll
        for (int i = 0; i < TH; ++i)
        {
            if (y + i * BH >= size.y)
            {
                break;
            }

#pragma unroll
            for (int j = 0; j < TW; ++j)
            {
                op.writeOutput(has[i * TW + j], int2{x + j, y + i * BH}, z, idxGlobal, idxLocal[i * TW + j]);
            }
        }
    }
}

// Run functions in layers -----------------------------------------------------

// The 3rd run layer is after template instantiation ---------------------------

template<typename T, class DataStridedCuda>
inline void RunMinMaxLocForType(cudaStream_t stream, const DataStridedCuda &inData, OptionalTensorDataRef minValData,
                                OptionalTensorDataRef minLocData, OptionalTensorDataRef numMinData,
                                OptionalTensorDataRef maxValData, OptionalTensorDataRef maxLocData,
                                OptionalTensorDataRef numMaxData)
{
    constexpr bool IsTensor = std::is_same_v<DataStridedCuda, nvcv::TensorDataStridedCuda>;

    using InputWrapper = std::conditional_t<IsTensor, cuda::Tensor3DWrap<T>, cuda::ImageBatchVarShapeWrap<T>>;

    InputWrapper inWrap;

    int2 inSize;
    int  numSamples;

    if constexpr (IsTensor)
    {
        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);

        inWrap     = cuda::CreateTensorWrapNHW<T>(inData);
        inSize     = cuda::StaticCast<int>(long2{inAccess->numCols(), inAccess->numRows()});
        numSamples = inAccess->numSamples();
    }
    else
    {
        static_assert(std::is_same_v<DataStridedCuda, nvcv::ImageBatchVarShapeDataStridedCuda>);

        inWrap     = InputWrapper(inData);
        inSize     = int2{inData.maxSize().w, inData.maxSize().h};
        numSamples = inData.numImages();
    }

    // CUDA block width x height is BW x BH, where each thread does TW x TH pixels of type T, that is why grid2
    // divides by BW * TW in x direction and BH * TH in y direction, grid1 is used just to initialize min/max

    constexpr int BW = 32;
    constexpr int BH = 8;
    constexpr int TW = (sizeof(T) >= 8 ? 1 : 8 / sizeof(T));
    constexpr int TH = 4;

    dim3 block(BW, BH, 1);
    dim3 grid2(util::DivUp(inSize.x, BW * TW), util::DivUp(inSize.y, BH * TH), numSamples);
    dim3 grid1(1, 1, numSamples);

    // MinMaxLoc algorithm is: (1) initialize minimum and maximum values with their opposite extrema; (2) find
    // actual min/max in input data; (3) collect locations of min/max in input data.  Depending on outputs chosen,
    // the setup uses minimum and maximum data or just minimum or maximum data. The OutputWrapper and OpCollect1,2
    // are used to combined 1 or 2 outputs and use kernels regardless of number of outputs.  Outputs are always
    // tensors, so tensor wraps are used considering their possible ranks of tensors: for the min(max)imum value,
    // min(max)ValData, and the number of min(max)ima, numMin(Max)Data, Tensor1DWrap is used and works for ranks 0
    // or 1 as it only wraps the base pointer; for the min(max)ima locations, min(max)LocData, Tensor2DWrap is used
    // and works for ranks 1 or 2 or 3 as it wraps base pointer and first stride.  The tensors shapes are not
    // passed to kernels directly, only input width and height and the locations capacity via collect operation.

    if (minValData && maxValData)
    {
        cuda::Tensor1DWrap<OutputType<T>> minWrap(minValData->get().basePtr());
        cuda::Tensor1DWrap<OutputType<T>> maxWrap(maxValData->get().basePtr());

        auto outWrap = OutputWrapper(minWrap, maxWrap);

        cuda::Tensor2DWrap<int2> minLocWrap(minLocData->get().basePtr(), (int)minLocData->get().stride(0));
        cuda::Tensor2DWrap<int2> maxLocWrap(maxLocData->get().basePtr(), (int)maxLocData->get().stride(0));
        cuda::Tensor1DWrap<int1> numMinWrap(numMinData->get().basePtr());
        cuda::Tensor1DWrap<int1> numMaxWrap(numMaxData->get().basePtr());

        int minLocCapacity = minLocData->get().shape(GetCapacityIdx(minLocData->get().rank()));
        int maxLocCapacity = maxLocData->get().shape(GetCapacityIdx(maxLocData->get().rank()));

        NVCV_CHECK_LOG(cudaMemsetAsync(numMinData->get().basePtr(), 0, sizeof(int) * numSamples, stream));
        NVCV_CHECK_LOG(cudaMemsetAsync(numMaxData->get().basePtr(), 0, sizeof(int) * numSamples, stream));

        OpCollect2 op(minLocWrap, numMinWrap, minLocCapacity, maxLocWrap, numMaxWrap, maxLocCapacity);

        InitMinMax<OpMinMax<T>><<<grid1, 1, 0, stream>>>(outWrap);

        FindMinMax<OpMinMax<T>, BW, BH, TW, TH><<<grid2, block, 0, stream>>>(inWrap, inSize, outWrap);

        CollectMinMax<BW, BH, TW, TH><<<grid2, block, 0, stream>>>(inWrap, inSize, outWrap, op);
    }
    else if (minValData)
    {
        cuda::Tensor1DWrap<OutputType<T>> minWrap(minValData->get().basePtr());

        auto outWrap = OutputWrapper(minWrap);

        cuda::Tensor2DWrap<int2> minLocWrap(minLocData->get().basePtr(), (int)minLocData->get().stride(0));
        cuda::Tensor1DWrap<int1> numMinWrap(numMinData->get().basePtr());

        int minLocCapacity = minLocData->get().shape(GetCapacityIdx(minLocData->get().rank()));

        NVCV_CHECK_LOG(cudaMemsetAsync(numMinData->get().basePtr(), 0, sizeof(int) * numSamples, stream));

        OpCollect1 op(minLocWrap, numMinWrap, minLocCapacity);

        InitMinMax<OpMin<T>><<<grid1, 1, 0, stream>>>(outWrap);

        FindMinMax<OpMin<T>, BW, BH, TW, TH><<<grid2, block, 0, stream>>>(inWrap, inSize, outWrap);

        CollectMinMax<BW, BH, TW, TH><<<grid2, block, 0, stream>>>(inWrap, inSize, outWrap, op);
    }
    else if (maxValData)
    {
        cuda::Tensor1DWrap<OutputType<T>> maxWrap(maxValData->get().basePtr());

        auto outWrap = OutputWrapper(maxWrap);

        cuda::Tensor2DWrap<int2> maxLocWrap(maxLocData->get().basePtr(), (int)maxLocData->get().stride(0));
        cuda::Tensor1DWrap<int1> numMaxWrap(numMaxData->get().basePtr());

        int maxLocCapacity = maxLocData->get().shape(GetCapacityIdx(maxLocData->get().rank()));

        NVCV_CHECK_LOG(cudaMemsetAsync(numMaxData->get().basePtr(), 0, sizeof(int) * numSamples, stream));

        OpCollect1 op(maxLocWrap, numMaxWrap, maxLocCapacity);

        InitMinMax<OpMax<T>><<<grid1, 1, 0, stream>>>(outWrap);

        FindMinMax<OpMax<T>, BW, BH, TW, TH><<<grid2, block, 0, stream>>>(inWrap, inSize, outWrap);

        CollectMinMax<BW, BH, TW, TH><<<grid2, block, 0, stream>>>(inWrap, inSize, outWrap, op);
    }
}

// The 2nd run layer is after exporting output data ----------------------------

template<class DataStridedCuda>
inline void RunMinMaxLocDataOut(cudaStream_t stream, const DataStridedCuda &inData, nvcv::DataType inDataType,
                                OptionalTensorDataRef minValData, OptionalTensorDataRef minLocData,
                                OptionalTensorDataRef numMinData, OptionalTensorDataRef maxValData,
                                OptionalTensorDataRef maxLocData, OptionalTensorDataRef numMaxData)
{
    switch (inDataType)
    {
#define NVCV_CASE_MINMAXLOC(DT, T)                                                                         \
    case nvcv::TYPE_##DT:                                                                                  \
        RunMinMaxLocForType<T>(stream, inData, minValData, minLocData, numMinData, maxValData, maxLocData, \
                               numMaxData);                                                                \
        break

        NVCV_CASE_MINMAXLOC(U8, uchar1);
        NVCV_CASE_MINMAXLOC(U16, ushort1);
        NVCV_CASE_MINMAXLOC(U32, uint1);
        NVCV_CASE_MINMAXLOC(S8, char1);
        NVCV_CASE_MINMAXLOC(S16, short1);
        NVCV_CASE_MINMAXLOC(S32, int1);
        NVCV_CASE_MINMAXLOC(F32, float1);
        NVCV_CASE_MINMAXLOC(F64, double1);

#undef NVCV_CASE_MINMAXLOC

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type");
    }
}

// This is used in the 1st run layer to checks if input data type matches the min/max value data type

inline bool DataTypeMatches(nvcv::DataType inDataType, nvcv::DataType valDataType)
{
    bool match = false;
    switch (valDataType)
    {
    case nvcv::TYPE_S32:
        match = inDataType == nvcv::TYPE_S32 || inDataType == nvcv::TYPE_S16 || inDataType == nvcv::TYPE_S8;
        break;

    case nvcv::TYPE_U32:
        match = inDataType == nvcv::TYPE_U32 || inDataType == nvcv::TYPE_U16 || inDataType == nvcv::TYPE_U8;
        break;

    case nvcv::TYPE_F32:
    case nvcv::TYPE_F64:
        match = inDataType == valDataType;
        break;

    default:
        break;
    }
    return match;
}

// The 1st run layer is after exporting input data -----------------------------

template<class DataStridedCuda>
inline void RunMinMaxLocDataIn(cudaStream_t stream, const DataStridedCuda &inData, nvcv::DataType inDataType,
                               int inNumSamples, int inNumChannels, const nvcv::Tensor &minVal,
                               const nvcv::Tensor &minLoc, const nvcv::Tensor &numMin, const nvcv::Tensor &maxVal,
                               const nvcv::Tensor &maxLoc, const nvcv::Tensor &numMax)
{
    if (inNumSamples == 0)
    {
        return;
    }
    if (inNumSamples < 0 && inNumSamples > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of samples in the input %d must be in [0, 65535]", inNumSamples);
    }
    if (inNumChannels != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have a single channel, not %d",
                              inNumChannels);
    }

    if ((minVal && (!minLoc || !numMin)) || (minLoc && (!minVal || !numMin)) || (numMin && (!minVal || !minLoc)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output minVal, minLoc and numMin must be provided together");
    }
    if ((maxVal && (!maxLoc || !numMax)) || (maxLoc && (!maxVal || !numMax)) || (numMax && (!maxVal || !maxLoc)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output maxVal, maxLoc and numMax must be provided together");
    }
    if (!minVal && !maxVal)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "At least one output (minVal and/or maxVal) must be chosen");
    }

    OptionalTensorData    minValData, minLocData, numMinData, maxValData, maxLocData, numMaxData;
    OptionalTensorDataRef minValRef, minLocRef, numMinRef, maxValRef, maxLocRef, numMaxRef;

    if (minVal)
    {
        minValData = minVal.exportData<nvcv::TensorDataStridedCuda>();
        if (!minValData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minVal must be cuda-accessible, pitch-linear tensor");
        }
        minLocData = minLoc.exportData<nvcv::TensorDataStridedCuda>();
        if (!minLocData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minLoc must be cuda-accessible, pitch-linear tensor");
        }
        numMinData = numMin.exportData<nvcv::TensorDataStridedCuda>();
        if (!numMinData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMin must be cuda-accessible, pitch-linear tensor");
        }

        minValRef = TensorDataRef(*minValData);
        minLocRef = TensorDataRef(*minLocData);
        numMinRef = TensorDataRef(*numMinData);

        if (!DataTypeMatches(inDataType, minValData->dtype()))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Wrong output minVal data type %s for input tensor data type %s output minVal data "
                                  "type must be S32/U32/F32/F64: for input data type S8/S16/S32 use S32; for "
                                  "U8/U16/U32 use U32; for all other data types use same data type as input tensor",
                                  nvcvDataTypeGetName(minValData->dtype()), nvcvDataTypeGetName(inDataType));
        }
        if (!((minValData->rank() == 0 && inNumSamples == 1)
              || ((minValData->rank() == 1 || minValData->rank() == 2) && inNumSamples == minValData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minVal number of samples must be the same as input tensor");
        }
        if (minValData->rank() == 2 && minValData->shape(1) != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minVal number of channels must be 1, not %ld", minValData->shape(1));
        }
        if (!((minLocData->rank() == 1 && inNumSamples == 1)
              || (minLocData->rank() == 2 && inNumSamples == minLocData->shape(0))
              || (minLocData->rank() == 3 && inNumSamples == minLocData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minLoc number of samples must be the same as input tensor");
        }
        if (!((numMinData->rank() == 0 && inNumSamples == 1)
              || ((numMinData->rank() == 1 || numMinData->rank() == 2) && inNumSamples == numMinData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMin number of samples must be the same as input tensor");
        }
        if (numMinData->rank() == 2 && minValData->shape(1) != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMin number of channels must be 1, not %ld", numMinData->shape(1));
        }
        if (!((minLocData->rank() == 3 && minLocData->dtype() == nvcv::TYPE_S32 && minLocData->shape(2) == 2)
              || (minLocData->rank() == 3 && minLocData->dtype() == nvcv::TYPE_2S32 && minLocData->shape(2) == 1)
              || (minLocData->rank() == 2 && minLocData->dtype() == nvcv::TYPE_2S32)))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minLoc must have rank 2 or 3 and 2xS32 or 2S32 data type, "
                                  "not rank %d and data type %s",
                                  minLocData->rank(), nvcvDataTypeGetName(minLocData->dtype()));
        }
        if (numMinData->dtype() != nvcv::TYPE_S32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output numMin must have S32 data type, not %s",
                                  nvcvDataTypeGetName(numMinData->dtype()));
        }
    }

    if (maxVal)
    {
        maxValData = maxVal.exportData<nvcv::TensorDataStridedCuda>();
        if (!maxValData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxVal must be cuda-accessible, pitch-linear tensor");
        }
        maxLocData = maxLoc.exportData<nvcv::TensorDataStridedCuda>();
        if (!maxLocData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxLoc must be cuda-accessible, pitch-linear tensor");
        }
        numMaxData = numMax.exportData<nvcv::TensorDataStridedCuda>();
        if (!numMaxData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMax must be cuda-accessible, pitch-linear tensor");
        }

        maxValRef = TensorDataRef(*maxValData);
        maxLocRef = TensorDataRef(*maxLocData);
        numMaxRef = TensorDataRef(*numMaxData);

        if (!DataTypeMatches(inDataType, maxValData->dtype()))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Wrong output maxVal data type %s for input tensor data type %s output maxVal data "
                                  "type must be S32/U32/F32/F64: for input data type S8/S16/S32 use S32; for "
                                  "U8/U16/U32 use U32; for all other data types use same data type as input tensor",
                                  nvcvDataTypeGetName(maxValData->dtype()), nvcvDataTypeGetName(inDataType));
        }
        if (!((maxValData->rank() == 0 && inNumSamples == 1)
              || ((maxValData->rank() == 1 || maxValData->rank() == 2) && inNumSamples == maxValData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxVal number of samples must be the same as input tensor");
        }
        if (maxValData->rank() == 2 && maxValData->shape(1) != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxVal number of channels must be 1, not %ld", maxValData->shape(1));
        }
        if (!((maxLocData->rank() == 1 && inNumSamples == 1)
              || (maxLocData->rank() == 2 && inNumSamples == maxLocData->shape(0))
              || (maxLocData->rank() == 3 && inNumSamples == maxLocData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxLoc number of samples must be the same as input tensor");
        }
        if (!((numMaxData->rank() == 0 && inNumSamples == 1)
              || ((numMaxData->rank() == 1 || numMaxData->rank() == 2) && inNumSamples == numMaxData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMax number of samples must be the same as input tensor");
        }
        if (numMaxData->rank() == 2 && maxValData->shape(1) != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMax number of channels must be 1, not %ld", numMaxData->shape(1));
        }
        if (!((maxLocData->rank() == 3 && maxLocData->dtype() == nvcv::TYPE_S32 && maxLocData->shape(2) == 2)
              || (maxLocData->rank() == 3 && maxLocData->dtype() == nvcv::TYPE_2S32 && maxLocData->shape(2) == 1)
              || (maxLocData->rank() == 2 && maxLocData->dtype() == nvcv::TYPE_2S32)))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxLoc must have rank 2 or 3 and 2xS32 or 2S32 data type, "
                                  "not rank %d and data type %s",
                                  maxLocData->rank(), nvcvDataTypeGetName(maxLocData->dtype()));
        }
        if (numMaxData->dtype() != nvcv::TYPE_S32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output numMax must have S32 data type, not %s",
                                  nvcvDataTypeGetName(numMaxData->dtype()));
        }
    }

    RunMinMaxLocDataOut(stream, inData, inDataType, minValRef, minLocRef, numMinRef, maxValRef, maxLocRef, numMaxRef);
}

} // anonymous namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

MinMaxLoc::MinMaxLoc() {}

// Tensor operator -------------------------------------------------------------

void MinMaxLoc::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &minVal,
                           const nvcv::Tensor &minLoc, const nvcv::Tensor &numMin, const nvcv::Tensor &maxVal,
                           const nvcv::Tensor &maxLoc, const nvcv::Tensor &numMax) const
{
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (!inData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    NVCV_ASSERT(inAccess);

    RunMinMaxLocDataIn(stream, *inData, inData->dtype(), inAccess->numSamples(), inAccess->numChannels(), minVal,
                       minLoc, numMin, maxVal, maxLoc, numMax);
}

// VarShape operator -----------------------------------------------------------

void MinMaxLoc::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &minVal,
                           const nvcv::Tensor &minLoc, const nvcv::Tensor &numMin, const nvcv::Tensor &maxVal,
                           const nvcv::Tensor &maxLoc, const nvcv::Tensor &numMax) const
{
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!inData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }

    nvcv::ImageFormat inFormat = inData->uniqueFormat();

    if (inFormat.numPlanes() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image batches must have a single plane, not %d",
                              inFormat.numPlanes());
    }

    RunMinMaxLocDataIn(stream, *inData, inFormat.planeDataType(0), inData->numImages(), inFormat.planeNumChannels(0),
                       minVal, minLoc, numMin, maxVal, maxLoc, numMax);
}

} // namespace cvcuda::priv

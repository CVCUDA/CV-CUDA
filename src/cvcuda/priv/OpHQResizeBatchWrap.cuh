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
#ifndef CVCUDA_PRIV_HQ_RESIZE_BATCH_WRAP_CUH
#define CVCUDA_PRIV_HQ_RESIZE_BATCH_WRAP_CUH

#include "cvcuda/Workspace.hpp"

#include <cuda_runtime.h>
#include <cvcuda/priv/WorkspaceUtil.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/ImageBatchVarShapeWrap.hpp>
#include <nvcv/cuda/TensorBatchWrap.hpp>
#include <nvcv/cuda/TensorWrap.hpp>
#include <util/Assert.h>
#include <util/CheckError.hpp>

// This file contains three kind of helpers
// 1. Helpers to wrap contigious batch with uniform sample stride into TensorWrap
// 2. DynamicBatchWrap class & helpers to wrap dynamically created batch (of intermediate samples)
//    with rigged sample stride
// 3. ImageBatchVarShapeWrap/TensorBatchWrap adapters that handle ROI on top of the usual wrappers

namespace cvcuda::priv::hq_resize::batch_wrapper {

namespace cuda = nvcv::cuda;

template<typename T, int N>
using Vec = typename cuda::MakeType<T, N>;

template<int N>
using VecI = Vec<int, N>;

template<typename T, typename... ExtentParams>
auto ComputeDenseStrides(ExtentParams... extentParams)
{
    constexpr int N = sizeof...(extentParams);
    static_assert(N >= 1);
    static_assert(std::conjunction_v<std::is_same<int, ExtentParams>...>);
    std::array<int, N>     extents = {extentParams...};
    std::array<int64_t, N> strides;
    strides[0] = extents[0] * sizeof(T);
    for (int d = 1; d < N; d++)
    {
        strides[d] = strides[d - 1] * extents[d];
    }
    return strides;
}

template<typename T, typename... Channels>
auto ComputeDenseStrides(VecI<2> shape, Channels... channels)
{
    static_assert(sizeof...(channels) <= 1);
    return ComputeDenseStrides<T>(channels..., shape.x, shape.y);
}

template<typename T, typename... Channels>
auto ComputeDenseStrides(VecI<3> shape, Channels... channels)
{
    static_assert(sizeof...(channels) <= 1);
    return ComputeDenseStrides<T>(channels..., shape.x, shape.y, shape.z);
}

namespace tensor {
template<typename T, typename StrideT, int kNStrides>
auto CreateDenseWrap(cuda::BaseType<T> *base, const std::array<int64_t, kNStrides> strides)
{
    constexpr int N = kNStrides + 1;
    for (auto stride : strides)
    {
        NVCV_ASSERT(stride <= cuda::TypeTraits<StrideT>::max);
    }
    static_assert(2 <= N && N <= 5);
    if constexpr (N == 5)
    {
        return cuda::TensorNDWrap<T, N, StrideT>(base, static_cast<StrideT>(strides[3]),
                                                 static_cast<StrideT>(strides[2]), static_cast<StrideT>(strides[1]),
                                                 static_cast<StrideT>(strides[0]));
    }
    else if constexpr (N == 4)
    {
        return cuda::TensorNDWrap<T, N, StrideT>(base, static_cast<StrideT>(strides[2]),
                                                 static_cast<StrideT>(strides[1]), static_cast<StrideT>(strides[0]));
    }
    else if constexpr (N == 3)
    {
        return cuda::TensorNDWrap<T, N, StrideT>(base, static_cast<StrideT>(strides[1]),
                                                 static_cast<StrideT>(strides[0]));
    }
    else if constexpr (N == 2)
    {
        return cuda::TensorNDWrap<T, N, StrideT>(base, static_cast<StrideT>(strides[0]));
    }
}

template<bool kHasDynamicChannels, typename T, typename StrideT, typename ShapeT>
auto CreateDenseWrap(cuda::BaseType<T> *base, int numChannels, ShapeT shape)
{
    static constexpr int kNStrides = cuda::NumElements<ShapeT> + kHasDynamicChannels;
    if constexpr (kHasDynamicChannels)
    {
        auto strides = ComputeDenseStrides<T>(shape, numChannels);
        return CreateDenseWrap<T, StrideT, kNStrides>(base, strides);
    }
    else if constexpr (!kHasDynamicChannels)
    {
        auto strides = ComputeDenseStrides<T>(shape);
        return CreateDenseWrap<T, StrideT, kNStrides>(base, strides);
    }
}

template<bool kHasDynamicChannels, int kSpatialNDim, typename T, typename StrideT,
         int N = 1 + kSpatialNDim + kHasDynamicChannels>
std::enable_if_t<kSpatialNDim == 2, cuda::TensorNDWrap<T, N, StrideT>> WrapTensor(
    const nvcv::TensorDataAccessStridedImagePlanar &tensorAccess, const ptrdiff_t roiOffset = 0)
{
    NVCV_ASSERT(tensorAccess.sampleStride() <= cuda::TypeTraits<StrideT>::max);
    NVCV_ASSERT(tensorAccess.rowStride() <= cuda::TypeTraits<StrideT>::max);
    NVCV_ASSERT(tensorAccess.colStride() <= cuda::TypeTraits<StrideT>::max);

    if constexpr (kHasDynamicChannels)
    {
        return cuda::TensorNDWrap<T, N, StrideT>(
            tensorAccess.sampleData(0) + roiOffset, static_cast<StrideT>(tensorAccess.sampleStride()),
            static_cast<int>(tensorAccess.rowStride()), static_cast<StrideT>(tensorAccess.colStride()));
    }
    else
    {
        return cuda::TensorNDWrap<T, N, StrideT>(tensorAccess.sampleData(0) + roiOffset,
                                                 static_cast<StrideT>(tensorAccess.sampleStride()),
                                                 static_cast<StrideT>(tensorAccess.rowStride()));
    }
}

template<bool kHasDynamicChannels, int kSpatialNDim, typename T, typename StrideT,
         int N = 1 + kSpatialNDim + kHasDynamicChannels>
std::enable_if_t<kSpatialNDim == 3, cuda::TensorNDWrap<T, N, StrideT>> WrapTensor(
    const nvcv::TensorDataAccessStridedImagePlanar &tensorAccess, const ptrdiff_t roiOffset = 0)
{
    NVCV_ASSERT(tensorAccess.sampleStride() <= cuda::TypeTraits<StrideT>::max);
    NVCV_ASSERT(tensorAccess.depthStride() <= cuda::TypeTraits<StrideT>::max);
    NVCV_ASSERT(tensorAccess.rowStride() <= cuda::TypeTraits<StrideT>::max);
    NVCV_ASSERT(tensorAccess.colStride() <= cuda::TypeTraits<StrideT>::max);

    if constexpr (kHasDynamicChannels)
    {
        return cuda::TensorNDWrap<T, N, StrideT>(
            tensorAccess.sampleData(0) + roiOffset, static_cast<StrideT>(tensorAccess.sampleStride()),
            static_cast<StrideT>(tensorAccess.depthStride()), static_cast<StrideT>(tensorAccess.rowStride()),
            static_cast<StrideT>(tensorAccess.colStride()));
    }
    else
    {
        return cuda::TensorNDWrap<T, N, StrideT>(
            tensorAccess.sampleData(0) + roiOffset, static_cast<StrideT>(tensorAccess.sampleStride()),
            static_cast<StrideT>(tensorAccess.depthStride()), static_cast<StrideT>(tensorAccess.rowStride()));
    }
}

template<bool kHasDynamicChannels, int kSpatialNDim, typename T, typename StrideT,
         int N = 1 + kSpatialNDim + kHasDynamicChannels>
std::enable_if_t<kSpatialNDim == 2, cuda::TensorNDWrap<T, N, StrideT>> WrapTensor(
    const nvcv::TensorDataAccessStridedImagePlanar &tensorAccess, const VecI<2> &roiOffset)
{
    ptrdiff_t offset = tensorAccess.rowStride() * roiOffset.y + tensorAccess.colStride() * roiOffset.x;
    return WrapTensor<kHasDynamicChannels, kSpatialNDim, T, StrideT>(tensorAccess, offset);
}

template<bool kHasDynamicChannels, int kSpatialNDim, typename T, typename StrideT,
         int N = 1 + kSpatialNDim + kHasDynamicChannels>
std::enable_if_t<kSpatialNDim == 3, cuda::TensorNDWrap<T, N, StrideT>> WrapTensor(
    const nvcv::TensorDataAccessStridedImagePlanar &tensorAccess, const VecI<3> &roiOffset)
{
    ptrdiff_t offset = tensorAccess.depthStride() * roiOffset.z + tensorAccess.rowStride() * roiOffset.y
                     + tensorAccess.colStride() * roiOffset.x;
    return WrapTensor<kHasDynamicChannels, kSpatialNDim, T, StrideT>(tensorAccess, offset);
}

template<typename TensorWrap>
auto __device__ GetSampleView(const TensorWrap &batchTensorWrap, const int sampleIdx)
{
    using T                               = typename TensorWrap::ValueType;
    using StrideType                      = typename TensorWrap::StrideType;
    static constexpr int kNumDimensions   = TensorWrap::kNumDimensions;
    static constexpr int kNumSampleDim    = kNumDimensions - 1; // not including sample (N) dim
    static constexpr int kVariableStrides = kNumSampleDim - 1;  // the innermost stride is static - sizeof type
    using TensorWrapT                     = cuda::TensorNDWrap<T, kNumSampleDim, StrideType>;
    static_assert(kVariableStrides == TensorWrapT::kVariableStrides);
    static_assert(kVariableStrides + 1 == TensorWrap::kVariableStrides);
    static_assert(1 <= kVariableStrides && kVariableStrides <= 3);
    auto       *basePtr = batchTensorWrap.ptr(sampleIdx);
    const auto *strides = batchTensorWrap.strides();
    if constexpr (kVariableStrides == 1)
    {
        return TensorWrapT{basePtr, strides[1]};
    }
    else if constexpr (kVariableStrides == 2)
    {
        return TensorWrapT{basePtr, strides[1], strides[2]};
    }
    else if constexpr (kVariableStrides == 3)
    {
        return TensorWrapT{basePtr, strides[1], strides[2], strides[3]};
    }
}

} // namespace tensor

namespace dynamic {

struct TensorAccessDescBase
{
    unsigned char *basePtr;
};

template<typename StrideT>
struct TensorAccessDesc : public TensorAccessDescBase
{
    static constexpr int kMaxNStrides = 3;
    StrideT              strides[kMaxNStrides];
};

template<int kNStrides, typename StrideT>
void SetupTensorAccessStrides(TensorAccessDesc<StrideT> *tensorAccessDesc, const std::array<int64_t, kNStrides> strides)
{
    // we ignore the last stride (sample stride), it's not needed for a single sample
    // as the samples are not assumed to be uniform
    static constexpr int kNSampleStrides = kNStrides - 1;
    static_assert(kNSampleStrides <= TensorAccessDesc<StrideT>::kMaxNStrides);
    for (int d = 0; d < kNSampleStrides; d++)
    {
        NVCV_ASSERT(strides[d] <= cuda::TypeTraits<StrideT>::max);
        tensorAccessDesc->strides[kNSampleStrides - 1 - d] = strides[d];
    }
}

/**
 * @brief Wrapper for batch of dynamically created samples
 *  (here, batch of intermediate samples between resampling passes)
 */
template<typename T, int N, typename StrideT>
struct DynamicBatchWrap
{
    using ValueType                       = T;
    using StrideType                      = StrideT;
    static constexpr int kNumDimensions   = N;
    static constexpr int kNumSampleDim    = kNumDimensions - 1; // not including sample (N) dim
    static constexpr int kVariableStrides = kNumSampleDim - 1;  // the innermost stride is static - sizeof type
    using TensorWrapT                     = cuda::TensorNDWrap<T, kNumSampleDim, StrideT>;
    static_assert(kVariableStrides == TensorWrapT::kVariableStrides);
    static_assert(kVariableStrides >= 1 && kVariableStrides <= TensorAccessDesc<StrideT>::kMaxNStrides);

    DynamicBatchWrap(TensorAccessDesc<StrideT> *samples)
        : m_samples{samples}
    {
    }

    inline __device__ TensorWrapT GetSampleView(const int sampleIdx) const
    {
        static_assert(1 <= kVariableStrides && kVariableStrides <= 3);

        auto                 sample  = m_samples[sampleIdx];
        const unsigned char *basePtr = sample.basePtr;

        if constexpr (kVariableStrides == 1)
        {
            return TensorWrapT{basePtr, sample.strides[0]};
        }
        else if constexpr (kVariableStrides == 2)
        {
            return TensorWrapT{basePtr, sample.strides[0], sample.strides[1]};
        }
        else if constexpr (kVariableStrides == 3)
        {
            return TensorWrapT{basePtr, sample.strides[0], sample.strides[1], sample.strides[2]};
        }
    }

private:
    TensorAccessDesc<StrideT> *m_samples;
};

struct DynamicBatchWrapMeta
{
    TensorAccessDescBase *cpu;
    TensorAccessDescBase *gpu;
};

inline void AddDynamicBatchWrapMeta(WorkspaceEstimator &est, int numSamples)
{
    est.addPinned<TensorAccessDesc<int64_t>>(numSamples);
    est.addCuda<TensorAccessDesc<int64_t>>(numSamples);
}

inline DynamicBatchWrapMeta AllocateDynamicBatchWrapMeta(WorkspaceAllocator &allocator, int numSamples, bool wideStride)
{
    DynamicBatchWrapMeta meta;
    if (wideStride)
    {
        meta.cpu = allocator.getPinned<TensorAccessDesc<int64_t>>(numSamples);
        meta.gpu = allocator.getCuda<TensorAccessDesc<int64_t>>(numSamples);
    }
    else
    {
        meta.cpu = allocator.getPinned<TensorAccessDesc<int32_t>>(numSamples);
        meta.gpu = allocator.getCuda<TensorAccessDesc<int32_t>>(numSamples);
    }
    return meta;
}

template<bool kHasDynamicChannels, typename T, typename StrideT, typename SampleDescT,
         int  N = 1 + SampleDescT::kSpatialNDim + kHasDynamicChannels>
DynamicBatchWrap<T, N, StrideT> CreateDynamicBatchWrap(int pass, cuda::BaseType<T> *intermediate,
                                                       const DynamicBatchWrapMeta tensorBatchMeta,
                                                       const SampleDescT *sampleDescsCpu, int numSamples,
                                                       cudaStream_t stream)
{
    static constexpr int kSpatialNDim = SampleDescT::kSpatialNDim;
    static_assert(N == 1 + kSpatialNDim + kHasDynamicChannels);

    ptrdiff_t sampleOffset = 0;
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
    {
        const SampleDescT &sampleDesc   = sampleDescsCpu[sampleIdx];
        VecI<kSpatialNDim> outputShape  = sampleDesc.shapes[pass + 1];
        auto              *cpuMeta      = reinterpret_cast<TensorAccessDesc<StrideT> *>(tensorBatchMeta.cpu);
        auto              *tensorAccess = &cpuMeta[sampleIdx];
        tensorAccess->basePtr           = reinterpret_cast<unsigned char *>(intermediate) + sampleOffset;
        if constexpr (kHasDynamicChannels)
        {
            constexpr int kNStrides = kSpatialNDim + 1;
            auto          strides   = ComputeDenseStrides<T>(outputShape, sampleDesc.channels);
            SetupTensorAccessStrides<kNStrides>(tensorAccess, strides);
            sampleOffset += strides[kNStrides - 1];
        }
        else if constexpr (!kHasDynamicChannels)
        {
            constexpr int kNStrides = kSpatialNDim;
            auto          strides   = ComputeDenseStrides<T>(outputShape);
            SetupTensorAccessStrides<kNStrides>(tensorAccess, strides);
            sampleOffset += strides[kNStrides - 1];
        }
    }
    NVCV_CHECK_THROW(cudaMemcpyAsync(tensorBatchMeta.gpu, tensorBatchMeta.cpu,
                                     numSamples * sizeof(TensorAccessDesc<StrideT>), cudaMemcpyHostToDevice, stream));

    return {reinterpret_cast<TensorAccessDesc<StrideT> *>(tensorBatchMeta.gpu)};
}
} // namespace dynamic

template<typename T, typename StrideT>
struct ImageBatchVarShapeWrapAdapter
{
    using ValueType                       = T;
    using StrideType                      = StrideT;
    static constexpr int kNumDimensions   = 3; // NHW
    static constexpr int kNumSampleDim    = 2; // HW
    static constexpr int kVariableStrides = 1; // the innermost stride is static - sizeof type
    using TensorWrapT                     = cuda::TensorNDWrap<T, kNumSampleDim, StrideT>;
    static_assert(kVariableStrides == TensorWrapT::kVariableStrides);

    ImageBatchVarShapeWrapAdapter(const nvcv::ImageBatchVarShapeDataStridedCuda &batchData)
        : m_batch{cuda::ImageBatchVarShapeWrap<T>{batchData}}
    {
    }

    inline __device__ TensorWrapT GetSampleView(const int sampleIdx, const VecI<2> roi) const
    {
        return TensorWrapT{m_batch.ptr(sampleIdx, 0, roi.y, roi.x), m_batch.rowStride(sampleIdx)};
    }

    inline __device__ TensorWrapT GetSampleView(const int sampleIdx) const
    {
        return TensorWrapT{m_batch.ptr(sampleIdx, 0, 0, 0), m_batch.rowStride(sampleIdx)};
    }

private:
    cuda::ImageBatchVarShapeWrap<T> m_batch;
};

template<typename T, int N, typename StrideT>
struct TensorBatchWrapAdapter
{
    using ValueType                       = T;
    using StrideType                      = StrideT;
    static constexpr int kNumDimensions   = N;
    static constexpr int kNumSampleDim    = kNumDimensions - 1; // not including sample (N) dim
    static constexpr int kVariableStrides = kNumSampleDim - 1;
    using TensorWrapT                     = cuda::TensorNDWrap<T, kNumSampleDim, StrideT>;
    using TensorBatchWrapT                = cuda::TensorBatchNDWrap<T, kNumSampleDim, StrideT>;
    static_assert(kVariableStrides == TensorWrapT::kVariableStrides);
    static_assert(kVariableStrides == TensorBatchWrapT::kVariableStrides);

    TensorBatchWrapAdapter(const nvcv::TensorBatchDataStridedCuda &batchData)
        : m_batch{TensorBatchWrapT{batchData}}
    {
    }

    inline __device__ TensorWrapT GetSampleView(const int sampleIdx, const VecI<2> roi) const
    {
        return TensorWrapT{m_batch.ptr(sampleIdx, roi.y, roi.x), m_batch.strides(sampleIdx)};
    }

    inline __device__ TensorWrapT GetSampleView(const int sampleIdx, const VecI<3> roi) const
    {
        return TensorWrapT{m_batch.ptr(sampleIdx, roi.z, roi.y, roi.x), m_batch.strides(sampleIdx)};
    }

    inline __device__ TensorWrapT GetSampleView(const int sampleIdx) const
    {
        return m_batch.tensor(sampleIdx);
    }

private:
    TensorBatchWrapT m_batch;
};
} // namespace cvcuda::priv::hq_resize::batch_wrapper
#endif // CVCUDA_PRIV_HQ_RESIZE_BATCH_WRAP_CUH

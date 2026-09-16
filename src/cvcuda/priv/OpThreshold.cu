/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
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

#include "Nvtx.hpp"
#include "OpThreshold.hpp"
#include "PlanarTensorView.hpp"

#include <cuda_fp16.h>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

using uchar  = unsigned char;
using ushort = unsigned short;

// Number of intensity bins the automatic (OTSU/TRIANGLE) modes accumulate, which also fixes the
// block size of the histogram and threshold-selection kernels: one lane owns one bin.
constexpr int kHistogramBins = 256;

// The CUDA grid-z dimension carries one plane per (sample, channel) pair on the planar paths.
constexpr int64_t kMaxGridZ = 65535;

// Mirrors the nvcv::legacy::cuda_op::DataType codes, which classify a dtype by data kind plus
// bits-per-channel rather than by exact type: the packed multi-channel types (TYPE_3U8, TYPE_4U8,
// ...) collapse onto the same code as their scalar counterpart, and the operator admitted them.
enum class DataTypeCode : int
{
    kU8  = 0,
    kS8  = 1,
    kU16 = 2,
    kS16 = 3,
    kS32 = 4,
    kF32 = 5,
    kF64 = 6,
    kF16 = 7,
};

// Mirrors nvcv::legacy::cuda_op::DataFormat, whose numeric values appear in the rejection messages.
enum class DataFormatCode : int
{
    kNCHW = 0,
    kNHWC = 1,
    kCHW  = 2,
    kHWC  = 3,
};

inline bool IsPlanar(DataFormatCode format)
{
    return format == DataFormatCode::kNCHW || format == DataFormatCode::kCHW;
}

// Reproduces nvcv::legacy::helpers::GetLegacyDataType(bpc, kind) exactly, including the rejection of
// every (kind, width) pair the legacy enum has no name for -- 32/64-bit unsigned, 64-bit signed,
// 8-bit float, complex and unspecified kinds all threw here before the operator's own dtype list was
// ever consulted.
inline DataTypeCode ClassifyDataType(int32_t bpc, nvcv::DataKind kind)
{
    switch (kind)
    {
    case nvcv::DataKind::FLOAT:
        if (bpc == 64)
            return DataTypeCode::kF64;
        if (bpc == 32)
            return DataTypeCode::kF32;
        if (bpc == 16)
            return DataTypeCode::kF16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc);

    case nvcv::DataKind::SIGNED:
        if (bpc == 8)
            return DataTypeCode::kS8;
        if (bpc == 16)
            return DataTypeCode::kS16;
        if (bpc == 32)
            return DataTypeCode::kS32;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc);

    case nvcv::DataKind::UNSIGNED:
        if (bpc == 8)
            return DataTypeCode::kU8;
        if (bpc == 16)
            return DataTypeCode::kU16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ", bpc);

    case nvcv::DataKind::COMPLEX:
    case nvcv::DataKind::UNSPECIFIED:
        break;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Only floating-point, signed integer and unsigned integer data kinds are supported ");
}

inline DataTypeCode ClassifyDataType(const nvcv::DataType &dtype)
{
    // NVCV_PACKING_X32_Y24b8 and friends are real, so non-uniform channel widths must be rejected
    // before the dtype can be classified at all.
    auto      bpc         = dtype.bitsPerChannel();
    const int numChannels = dtype.numChannels();
    for (int i = 1; i < numChannels; ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    return ClassifyDataType(bpc[0], dtype.dataKind());
}

inline DataTypeCode ClassifyDataType(const nvcv::ImageFormat &fmt)
{
    // numPlanes()/planeDataType() cross the nvcv shared-library boundary, so they cannot be hoisted
    // by the compiler.
    const int            numPlanes = fmt.numPlanes();
    const nvcv::DataType plane0    = fmt.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (fmt.planeDataType(i) != plane0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    return ClassifyDataType(plane0);
}

// The exact set the operator admits: (unsigned, 8), (unsigned, 16), (signed, 16), (float, 16),
// (float, 32) and (float, 64). OTSU and TRIANGLE narrow this further to 8-bit unsigned below.
inline bool IsSupportedDataType(DataTypeCode code)
{
    return code == DataTypeCode::kU8 || code == DataTypeCode::kS16 || code == DataTypeCode::kU16
        || code == DataTypeCode::kF16 || code == DataTypeCode::kF32 || code == DataTypeCode::kF64;
}

// Calls cb with a value-initialized instance of the element type that code names, so the caller's
// generic lambda is instantiated for that type. The kF64 arm is the default because IsSupportedDataType
// runs first and admits nothing else -- the legacy function-pointer tables this replaces reserved
// entries for kS8 and kS32 that the same gate made unreachable.
template<typename Cb>
inline void DispatchByDataType(DataTypeCode code, const Cb &cb)
{
    switch (code)
    {
    case DataTypeCode::kU8:
        return cb(uchar{});
    case DataTypeCode::kU16:
        return cb(ushort{});
    case DataTypeCode::kS16:
        return cb(short{});
    case DataTypeCode::kF16:
        return cb(__half{});
    case DataTypeCode::kF32:
        return cb(float{});
    default: // DataTypeCode::kF64
        return cb(double{});
    }
}

inline DataFormatCode ClassifyDataFormat(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW)
    {
        return DataFormatCode::kNCHW;
    }
    if (layout == nvcv::TENSOR_CHW)
    {
        return DataFormatCode::kCHW;
    }
    if (layout == nvcv::TENSOR_NHWC)
    {
        return DataFormatCode::kNHWC;
    }
    if (layout == nvcv::TENSOR_HWC)
    {
        return DataFormatCode::kHWC;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
}

inline DataFormatCode ClassifyDataFormat(const nvcv::ImageBatchVarShapeDataStridedCuda &imgBatch)
{
    nvcv::ImageFormat fmt = imgBatch.uniqueFormat();
    if (!fmt)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images must have the same format");
    }

    const int            numPlanes = fmt.numPlanes();
    const nvcv::DataType plane0    = fmt.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (fmt.planeDataType(i) != plane0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    if (numPlanes >= 2)
    {
        if (numPlanes != fmt.numChannels())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar images must have one channel per plane");
        }
        return imgBatch.numImages() >= 2 ? DataFormatCode::kNCHW : DataFormatCode::kCHW;
    }

    return imgBatch.numImages() >= 2 ? DataFormatCode::kNHWC : DataFormatCode::kHWC;
}

// Validates the thresh / maxval parameter tensors, which must both be rank-1 F64 arrays of size N.
inline void ValidateParamTensor(const nvcv::TensorDataStridedCuda &param, const char *name)
{
    DataTypeCode code = ClassifyDataType(param.dtype());
    if (code != DataTypeCode::kF64)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s DataType %d", name,
                              static_cast<int>(code));
    }
    int rank = param.layout().rank();
    if (rank != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s Dim %d", name, rank);
    }
}

// The two threshold-type rejections, in the order both overloads apply them. The "Threhold" spelling
// of the second message is the legacy text and is asserted by the frozen negative tests.
inline void ValidateThresholdType(uint32_t type, uint32_t automaticThresh, uint32_t maskedType)
{
    if (automaticThresh == ((uint32_t)NVCV_THRESH_OTSU | (uint32_t)NVCV_THRESH_TRIANGLE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Threshold Type %u", type);
    }

    if (!util::IsPowerOfTwo(maskedType))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Threhold Type %u", maskedType);
    }
}

// The preconditions the histogram pass needs, shared by the OTSU and TRIANGLE arms of both overloads:
// the accumulation kernel reads bytes and owns one 256-bin histogram per image.
inline void ValidateAutomaticModeInput(DataTypeCode inCode, int numChannels)
{
    if (inCode != DataTypeCode::kU8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Data Type %d", static_cast<int>(inCode));
    }
    if (numChannels != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Only support 1 channel");
    }
}

// A packed load/store that falls back to element-wise access when the pointer is not aligned for the
// vector type: var-shape rows start at arbitrary offsets, so the wide path cannot be assumed.
template<typename P, typename T>
__device__ __forceinline__ P LoadPacked(const T *ptr)
{
    static_assert(sizeof(P) % sizeof(T) == 0);

    if (reinterpret_cast<std::uintptr_t>(ptr) % alignof(P) == 0)
        return *reinterpret_cast<const P *>(ptr);

    P  value;
    T *elements = reinterpret_cast<T *>(&value);
#pragma unroll
    for (int i = 0; i < sizeof(P) / sizeof(T); ++i) elements[i] = ptr[i];
    return value;
}

template<typename P, typename T>
__device__ __forceinline__ void StorePacked(T *ptr, P value)
{
    static_assert(sizeof(P) % sizeof(T) == 0);

    if (reinterpret_cast<std::uintptr_t>(ptr) % alignof(P) == 0)
    {
        *reinterpret_cast<P *>(ptr) = value;
        return;
    }

    const T *elements = reinterpret_cast<const T *>(&value);
#pragma unroll
    for (int i = 0; i < sizeof(P) / sizeof(T); ++i) ptr[i] = elements[i];
}

// Integer thresholding: the threshold and maxval arrive as doubles and are folded to the element
// type with the same round/floor/saturate sequence and the same out-of-range short circuits the
// legacy kernels used.
template<typename T>
__device__ __forceinline__ T ThresholdOverflowValue(T inval, double th, double maxv, NVCVThresholdType type)
{
    T   maxType = cuda::TypeTraits<T>::max;
    T   minType = cuda::TypeTraits<T>::min;
    int imaxval = round(maxv);
    T   maxval  = cuda::SaturateCast<T>(imaxval);
    int ithresh = floor(th);

    switch (type)
    {
    case NVCV_THRESH_BINARY:
        if (ithresh >= minType && ithresh <= maxType)
        {
            T thresh = (T)ithresh;
            return inval > thresh ? maxval : 0;
        }
        return ithresh < minType ? maxval : 0;
    case NVCV_THRESH_BINARY_INV:
        if (ithresh >= minType && ithresh <= maxType)
        {
            T thresh = (T)ithresh;
            return inval > thresh ? 0 : maxval;
        }
        return ithresh < minType ? 0 : maxval;
    case NVCV_THRESH_TRUNC:
        if (ithresh >= minType && ithresh <= maxType)
        {
            T thresh = (T)ithresh;
            return inval > thresh ? thresh : inval;
        }
        return ithresh < minType ? minType : inval;
    case NVCV_THRESH_TOZERO:
        if (ithresh >= minType && ithresh <= maxType)
        {
            T thresh = (T)ithresh;
            return inval > thresh ? inval : 0;
        }
        return ithresh < minType ? inval : 0;
    default: // NVCV_THRESH_TOZERO_INV
        if (ithresh >= minType && ithresh <= maxType)
        {
            T thresh = (T)ithresh;
            return inval > thresh ? 0 : inval;
        }
        return ithresh < minType ? 0 : inval;
    }
}

// Floating-point thresholding: no range folding, the parameters cast straight to the element type.
template<typename T>
__device__ __forceinline__ T ThresholdGenericValue(T inval, double th, double maxv, NVCVThresholdType type)
{
    // (T)0 keeps both conditional operands a single type: for __half the mixed __half/int
    // conditional is ambiguous, as each operand converts to the other's type.
    T thresh = (T)th;
    T maxval = (T)maxv;

    switch (type)
    {
    case NVCV_THRESH_BINARY:
        return inval > thresh ? maxval : (T)0;
    case NVCV_THRESH_BINARY_INV:
        return inval > thresh ? (T)0 : maxval;
    case NVCV_THRESH_TRUNC:
        return inval > thresh ? thresh : inval;
    case NVCV_THRESH_TOZERO:
        return inval > thresh ? inval : (T)0;
    default: // NVCV_THRESH_TOZERO_INV
        return inval > thresh ? (T)0 : inval;
    }
}

// __half thresholds through the Generic (float-family) kernels and value helper:
// std::is_floating_point does not classify it, and the overflow variants need the integer
// TypeTraits limits that do not exist for the non-literal __half.
template<typename T>
inline constexpr bool UseGenericThreshold = cuda::detail::IsFloatingPointV<T>;

// Per-value threshold, shared by the planar tensor and var-shape kernels. The legacy kernels took the
// dtype code as a kernel argument and tested it per element, but the dispatch switches below pick T
// from that same code, so the test is exactly UseGenericThreshold<T> and resolves at compile time.
template<typename T>
__device__ __forceinline__ T ThresholdValueDispatch(T inval, double th, double maxv, NVCVThresholdType type)
{
    if constexpr (UseGenericThreshold<T>)
    {
        return ThresholdGenericValue(inval, th, maxv, type);
    }
    else
    {
        return ThresholdOverflowValue(inval, th, maxv, type);
    }
}

// TRIANGLE threshold selection, shared by the tensor and var-shape paths: one block of 256 lanes per
// image reduces the accumulated histogram. Every reduction below is a fixed 128/64/32/16/8/4/2/1
// tree over the 256 bins with ties broken toward the lower bin index, which is what pins the chosen
// level: reordering it would shift the result by one level on histograms with equal candidates.
__global__ void triangle_cal(int *histogram, cuda::Tensor1DWrap<double, int32_t> thresh)
{
    int                     localid = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ int          hist[256];
    __shared__ volatile int reduce[256];
    hist[localid] = histogram[blockIdx.z * 256 + localid];

    int left_bound = INT_MAX, right_bound = -1;

    // find the left_bound of the histogram (the leftmost non-zero number).
    // Reduce to find the smallest localid
    if (hist[localid] > 0)
    {
        left_bound  = localid;
        right_bound = localid;
    }

    reduce[localid] = left_bound;
    __syncthreads();

    if (localid < 128)
        reduce[localid] = min(reduce[localid], reduce[localid + 128]);
    __syncthreads();
    if (localid < 64)
        reduce[localid] = min(reduce[localid], reduce[localid + 64]);
    __syncthreads();
    if (localid < 32)
    {
        reduce[localid] = min(reduce[localid], reduce[localid + 32]);
        reduce[localid] = min(reduce[localid], reduce[localid + 16]);
        reduce[localid] = min(reduce[localid], reduce[localid + 8]);
        reduce[localid] = min(reduce[localid], reduce[localid + 4]);
        reduce[localid] = min(reduce[localid], reduce[localid + 2]);
        reduce[localid] = min(reduce[localid], reduce[localid + 1]);
    }
    __syncthreads();

    left_bound = reduce[0];
    if (left_bound > 0)
        left_bound--;

    // find the right_bound of the histogram (the rightmost non-zero number).
    // Reduce to find the largest localid
    __syncthreads();
    reduce[localid] = right_bound;
    __syncthreads();

    if (localid < 128)
        reduce[localid] = max(reduce[localid], reduce[localid + 128]);
    __syncthreads();
    if (localid < 64)
        reduce[localid] = max(reduce[localid], reduce[localid + 64]);
    __syncthreads();
    if (localid < 32)
    {
        reduce[localid] = max(reduce[localid], reduce[localid + 32]);
        reduce[localid] = max(reduce[localid], reduce[localid + 16]);
        reduce[localid] = max(reduce[localid], reduce[localid + 8]);
        reduce[localid] = max(reduce[localid], reduce[localid + 4]);
        reduce[localid] = max(reduce[localid], reduce[localid + 2]);
        reduce[localid] = max(reduce[localid], reduce[localid + 1]);
    }
    __syncthreads();
    right_bound = reduce[0];

    if (right_bound < 255)
        right_bound++;
    __syncthreads();

    // find the coordinate with the maximum value in the histogram
    // reduce to find the maximum value and record the coordinate
    __shared__ uchar idx[256];
    idx[localid]    = (uchar)localid;
    reduce[localid] = hist[localid];
    __syncthreads();

    if (localid < 128 && reduce[localid + 128] >= reduce[localid])
    {
        if (reduce[localid + 128] == reduce[localid])
            idx[localid] = min(idx[localid], idx[localid + 128]);
        else
            idx[localid] = idx[localid + 128];
        reduce[localid] = reduce[localid + 128];
    }
    __syncthreads();
    if (localid < 64 && reduce[localid + 64] >= reduce[localid])
    {
        if (reduce[localid + 64] == reduce[localid])
            idx[localid] = min(idx[localid], idx[localid + 64]);
        else
            idx[localid] = idx[localid + 64];
        reduce[localid] = reduce[localid + 64];
    }
    __syncthreads();

    if (localid < 32)
    {
        if (reduce[localid + 32] >= reduce[localid])
        {
            if (reduce[localid + 32] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 32]);
            else
                idx[localid] = idx[localid + 32];
            reduce[localid] = reduce[localid + 32];
        }
        if (reduce[localid + 16] >= reduce[localid])
        {
            if (reduce[localid + 16] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 16]);
            else
                idx[localid] = idx[localid + 16];
            reduce[localid] = reduce[localid + 16];
        }
        if (reduce[localid + 8] >= reduce[localid])
        {
            if (reduce[localid + 8] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 8]);
            else
                idx[localid] = idx[localid + 8];
            reduce[localid] = reduce[localid + 8];
        }
        if (reduce[localid + 4] >= reduce[localid])
        {
            if (reduce[localid + 4] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 4]);
            else
                idx[localid] = idx[localid + 4];
            reduce[localid] = reduce[localid + 4];
        }
        if (reduce[localid + 2] >= reduce[localid])
        {
            if (reduce[localid + 2] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 2]);
            else
                idx[localid] = idx[localid + 2];
            reduce[localid] = reduce[localid + 2];
        }
        if (reduce[localid + 1] >= reduce[localid])
        {
            if (reduce[localid + 1] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 1]);
            else
                idx[localid] = idx[localid + 1];
            reduce[localid] = reduce[localid + 1];
        }
    }
    __syncthreads();
    int max = reduce[0], maxid = idx[0];

    // determine if the histogram needs to be flipped
    bool isfliped = false;
    if (maxid - left_bound < right_bound - maxid)
    {
        isfliped = true;
        int temp = hist[255 - localid];
        __syncthreads();
        hist[localid] = temp;
        left_bound    = 255 - right_bound;
        maxid         = 255 - maxid;
    }

    // from left_bound to the coordinate with the maximum value in the histogram(maxid),
    // calculate the distance : 'max_value * i + (left_bound - maxid) * histogram[i]'
    int val = -1;
    if (localid > left_bound && localid <= maxid)
        val = max * localid + (left_bound - maxid) * hist[localid];

    // find the coordinate with the largest distance
    // reduce to find the largest distance and record the coordinate
    __syncthreads();
    reduce[localid] = val;
    idx[localid]    = localid;
    __syncthreads();

    if (localid < 128 && reduce[localid + 128] >= reduce[localid])
    {
        if (reduce[localid + 128] == reduce[localid])
            idx[localid] = min(idx[localid], idx[localid + 128]);
        else
            idx[localid] = idx[localid + 128];
        reduce[localid] = reduce[localid + 128];
    }
    __syncthreads();
    if (localid < 64 && reduce[localid + 64] >= reduce[localid])
    {
        if (reduce[localid + 64] == reduce[localid])
            idx[localid] = min(idx[localid], idx[localid + 64]);
        else
            idx[localid] = idx[localid + 64];
        reduce[localid] = reduce[localid + 64];
    }
    __syncthreads();

    if (localid < 32)
    {
        if (reduce[localid + 32] >= reduce[localid])
        {
            if (reduce[localid + 32] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 32]);
            else
                idx[localid] = idx[localid + 32];
            reduce[localid] = reduce[localid + 32];
        }
        if (reduce[localid + 16] >= reduce[localid])
        {
            if (reduce[localid + 16] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 16]);
            else
                idx[localid] = idx[localid + 16];
            reduce[localid] = reduce[localid + 16];
        }
        if (reduce[localid + 8] >= reduce[localid])
        {
            if (reduce[localid + 8] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 8]);
            else
                idx[localid] = idx[localid + 8];
            reduce[localid] = reduce[localid + 8];
        }
        if (reduce[localid + 4] >= reduce[localid])
        {
            if (reduce[localid + 4] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 4]);
            else
                idx[localid] = idx[localid + 4];
            reduce[localid] = reduce[localid + 4];
        }
        if (reduce[localid + 2] >= reduce[localid])
        {
            if (reduce[localid + 2] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 2]);
            else
                idx[localid] = idx[localid + 2];
            reduce[localid] = reduce[localid + 2];
        }
        if (reduce[localid + 1] >= reduce[localid])
        {
            if (reduce[localid + 1] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 1]);
            else
                idx[localid] = idx[localid + 1];
            reduce[localid] = reduce[localid + 1];
        }
    }
    __syncthreads();

    // write to gpu memory
    if (localid == 0)
    {
        double res = (double)idx[0] - 1;
        if (isfliped)
            res = 255 - res;
        thresh[(int)blockIdx.z] = res;
    }
}

// OTSU threshold selection, shared by the tensor and var-shape paths. The between-class variance is
// built from two shuffle-based prefix scans over the 256 bins and the winning bin comes out of a
// fixed 128/64/32/16/8/4/2/1 reduction tree with ties broken toward the lower bin index; both orders
// are load bearing, since a different summation or tie rule can move the chosen level by one.
// Only the source of the normalization size differs between the two callers, so it is a parameter.
__device__ __forceinline__ void OtsuCalBody(int *histogram, cuda::Tensor1DWrap<double, int32_t> thresh, int size)
{
    int            localid = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ int hist[256];
    hist[localid] = histogram[blockIdx.z * 256 + localid];
    __syncthreads();

    __shared__ volatile double reduce[256];
    double                     mu, scale = 1. / size;

    // reduce to calculate the sum of 'i * histogram[i]' (mu)
    reduce[localid] = localid * (double)hist[localid];
    __syncthreads();

    if (localid < 128)
        reduce[localid] = reduce[localid] + reduce[localid + 128];
    __syncthreads();
    if (localid < 64)
        reduce[localid] = reduce[localid] + reduce[localid + 64];
    __syncthreads();
    if (localid < 32)
    {
        reduce[localid] = reduce[localid] + reduce[localid + 32];
        reduce[localid] = reduce[localid] + reduce[localid + 16];
        reduce[localid] = reduce[localid] + reduce[localid + 8];
        reduce[localid] = reduce[localid] + reduce[localid + 4];
        reduce[localid] = reduce[localid] + reduce[localid + 2];
        reduce[localid] = reduce[localid] + reduce[localid + 1];
    }
    __syncthreads();

    mu = reduce[0] * scale;
    __syncthreads();

    // reduce to calculate the prefix sum of histogram[i] (q1)
    // the prefix sum of histogram[i] = histogram[0] + histogram[1] + ... + histogram[i-1] + histogram[i]
    double q1   = hist[localid] * scale;
    int    lane = localid % 32, warp = localid / 32;
    // sum of q1 in warp
    double temp = q1;
    temp += __shfl_xor_sync(0xffffffff, temp, 1);
    temp += __shfl_xor_sync(0xffffffff, temp, 2);
    temp += __shfl_xor_sync(0xffffffff, temp, 4);
    temp += __shfl_xor_sync(0xffffffff, temp, 8);
    temp += __shfl_xor_sync(0xffffffff, temp, 16);
    if (lane == 0)
        reduce[warp] = temp;
    __syncthreads();
    // prefix scan of the sum
    if (warp == 0)
    {
        temp = reduce[lane];
        reduce[lane + 1] += temp;
        temp = reduce[lane];
        reduce[lane + 2] += temp;
        temp = reduce[lane];
        reduce[lane + 4] += temp;
        temp             = reduce[lane];
        reduce[lane]     = 0;
        reduce[lane + 1] = temp;
    }
    __syncthreads();
    // prefix scan in warp
    temp = __shfl_up_sync(0xffffffff, q1, 1);
    if (lane >= 1)
        q1 += temp;
    temp = __shfl_up_sync(0xffffffff, q1, 2);
    if (lane >= 2)
        q1 += temp;
    temp = __shfl_up_sync(0xffffffff, q1, 4);
    if (lane >= 4)
        q1 += temp;
    temp = __shfl_up_sync(0xffffffff, q1, 8);
    if (lane >= 8)
        q1 += temp;
    temp = __shfl_up_sync(0xffffffff, q1, 16);
    if (lane >= 16)
        q1 += temp;
    q1 += reduce[warp];
    double q2 = 1 - q1;
    __syncthreads();

    // reduce to calculate the prefix sum of i * histogram[i] (one)
    // the prefix sum of i * histogram[i] = 0*histogram[0] + 1*histogram[1] + ... + (i-1)*histogram[i-1] + i*histogram[i]
    double one = localid * hist[localid] * scale;
    // sum of q1 in warp
    temp = one;
    temp += __shfl_xor_sync(0xffffffff, temp, 1);
    temp += __shfl_xor_sync(0xffffffff, temp, 2);
    temp += __shfl_xor_sync(0xffffffff, temp, 4);
    temp += __shfl_xor_sync(0xffffffff, temp, 8);
    temp += __shfl_xor_sync(0xffffffff, temp, 16);
    if (lane == 0)
        reduce[warp] = temp;
    __syncthreads();
    // prefix scan of the sum
    if (warp == 0)
    {
        temp = reduce[lane];
        reduce[lane + 1] += temp;
        temp = reduce[lane];
        reduce[lane + 2] += temp;
        temp = reduce[lane];
        reduce[lane + 4] += temp;
        temp             = reduce[lane];
        reduce[lane]     = 0;
        reduce[lane + 1] = temp;
    }
    __syncthreads();
    // prefix scan in warp
    temp = __shfl_up_sync(0xffffffff, one, 1);
    if (lane >= 1)
        one += temp;
    temp = __shfl_up_sync(0xffffffff, one, 2);
    if (lane >= 2)
        one += temp;
    temp = __shfl_up_sync(0xffffffff, one, 4);
    if (lane >= 4)
        one += temp;
    temp = __shfl_up_sync(0xffffffff, one, 8);
    if (lane >= 8)
        one += temp;
    temp = __shfl_up_sync(0xffffffff, one, 16);
    if (lane >= 16)
        one += temp;
    one += reduce[warp];
    __syncthreads(); // if change reduce later

    // calulate sigma
    double mu1 = one / q1, mu2 = (mu - q1 * mu1) / q2;
    double sigma;
    if (min(q1, q2) < FLT_EPSILON || max(q1, q2) > 1. - FLT_EPSILON)
        sigma = -1;
    else
        sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);

    // find the coordinate with the largest sigma
    // reduce to find the largest sigma and record the cooridinate
    reduce[localid] = sigma;
    __shared__ uchar idx[256];
    idx[localid] = localid;
    __syncthreads();

    if (localid < 128 && reduce[localid + 128] >= reduce[localid])
    {
        if (reduce[localid + 128] == reduce[localid])
            idx[localid] = min(idx[localid], idx[localid + 128]);
        else
            idx[localid] = idx[localid + 128];
        reduce[localid] = reduce[localid + 128];
    }
    __syncthreads();
    if (localid < 64 && reduce[localid + 64] >= reduce[localid])
    {
        if (reduce[localid + 64] == reduce[localid])
            idx[localid] = min(idx[localid], idx[localid + 64]);
        else
            idx[localid] = idx[localid + 64];
        reduce[localid] = reduce[localid + 64];
    }
    __syncthreads();

    if (localid < 32)
    {
        if (reduce[localid + 32] >= reduce[localid])
        {
            if (reduce[localid + 32] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 32]);
            else
                idx[localid] = idx[localid + 32];
            reduce[localid] = reduce[localid + 32];
        }
        if (reduce[localid + 16] >= reduce[localid])
        {
            if (reduce[localid + 16] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 16]);
            else
                idx[localid] = idx[localid + 16];
            reduce[localid] = reduce[localid + 16];
        }
        if (reduce[localid + 8] >= reduce[localid])
        {
            if (reduce[localid + 8] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 8]);
            else
                idx[localid] = idx[localid + 8];
            reduce[localid] = reduce[localid + 8];
        }
        if (reduce[localid + 4] >= reduce[localid])
        {
            if (reduce[localid + 4] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 4]);
            else
                idx[localid] = idx[localid + 4];
            reduce[localid] = reduce[localid + 4];
        }
        if (reduce[localid + 2] >= reduce[localid])
        {
            if (reduce[localid + 2] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 2]);
            else
                idx[localid] = idx[localid + 2];
            reduce[localid] = reduce[localid + 2];
        }
        if (reduce[localid + 1] >= reduce[localid])
        {
            if (reduce[localid + 1] == reduce[localid])
                idx[localid] = min(idx[localid], idx[localid + 1]);
            else
                idx[localid] = idx[localid + 1];
            reduce[localid] = reduce[localid + 1];
        }
    }
    __syncthreads();

    // write to gpu memory
    if (localid == 0)
        thresh[(int)blockIdx.z] = (double)idx[0];
}

// ---------------------------------------------------------------------------------------------
// Tensor path
// ---------------------------------------------------------------------------------------------
namespace tensor_impl {

// The packed kernels below deliberately re-inline the five threshold modes instead of calling
// ThresholdOverflowValue / ThresholdGenericValue: they hoist the loop-invariant floor/round/saturate
// fold and the threshold-range test above the element loop, and skip the source load entirely when
// the threshold is out of range. The per-value helpers are for the scalar planar kernels, where one
// thread owns one element and there is nothing to hoist.

// Every tensor kernel below addresses through int32 TensorWraps, so the largest byte offset either
// tensor can produce has to fit in an int32.
inline void ValidateTensorFitsInt32(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                    const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    auto outMaxStride = outAccess.sampleStride() * outAccess.numSamples();
    auto inMaxStride  = inAccess.sampleStride() * inAccess.numSamples();
    if (std::max(outMaxStride, inMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.", cuda::TypeTraits<int32_t>::max);
    }
}

template<typename P, typename SrcWrap, typename DstWrap>
__global__ void Binary_overflow(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh,
                                cuda::Tensor1DWrap<double, int32_t> _maxval, int height, int width, int channel)
{
    static_assert(std::is_same_v<SrcWrap, DstWrap>);
    using T = typename SrcWrap::ValueType;

    int cn       = cuda::NumElements<P>;
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid * cn >= height * width * channel)
        return;

    int h     = globalid / (width * channel / cn);
    int wid   = globalid % (width * channel / cn);
    int w     = (wid * cn) / channel;
    int c     = (wid * cn) % channel;
    int batch = blockIdx.z;

    T      MAX     = cuda::TypeTraits<T>::max;
    T      MIN     = cuda::TypeTraits<T>::min;
    double maxv    = _maxval[batch];
    double th      = _thresh[batch];
    int    imaxval = round(maxv);
    T      maxval  = cuda::SaturateCast<T>(imaxval);
    int    ithresh = floor(th);
    P      out;
    if (ithresh >= MIN && ithresh <= MAX)
    {
        T  thresh = (T)ithresh;
        P  in     = *((P *)src.ptr(batch, h, w, c));
        T *inval  = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]                   = inval[i] > thresh ? maxval : 0;
            cuda::GetElement<P>(out, i) = outval[i];
        }
        *((P *)dst.ptr(batch, h, w, c)) = out;
        return;
    }
    if (ithresh < MIN)
    {
        out                             = cuda::SetAll<P>(maxval);
        *((P *)dst.ptr(batch, h, w, c)) = out;
        return;
    }

    out                             = cuda::SetAll<P>(0);
    *((P *)dst.ptr(batch, h, w, c)) = out;
}

template<typename P, typename SrcWrap, typename DstWrap>
__global__ void Binary_Generic(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh,
                               cuda::Tensor1DWrap<double, int32_t> _maxval, int height, int width, int channel)
{
    static_assert(std::is_same_v<SrcWrap, DstWrap>);
    using T      = typename SrcWrap::ValueType;
    int cn       = cuda::NumElements<P>;
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid * cn >= height * width * channel)
        return;
    int h     = globalid / (width * channel / cn);
    int wid   = globalid % (width * channel / cn);
    int w     = (wid * cn) / channel;
    int c     = (wid * cn) % channel;
    int batch = blockIdx.z;

    T maxval = (T)_maxval[batch];
    T thresh = (T)_thresh[batch];

    P  out;
    P  in    = *((P *)src.ptr(batch, h, w, c));
    T *inval = reinterpret_cast<T *>((void *)&in);
    T  outval[4];
#pragma unroll
    for (int i = 0; i < cn; i++)
    {
        // (T)0 keeps both conditional operands a single type; the mixed __half/int conditional
        // is ambiguous (same in the other Generic kernels below).
        outval[i]                   = inval[i] > thresh ? maxval : (T)0;
        cuda::GetElement<P>(out, i) = outval[i];
    }
    *((P *)dst.ptr(batch, h, w, c)) = out;
}

template<typename P, typename SrcWrap, typename DstWrap>
__global__ void BinaryInv_overflow(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh,
                                   cuda::Tensor1DWrap<double, int32_t> _maxval, int height, int width, int channel)
{
    static_assert(std::is_same_v<SrcWrap, DstWrap>);
    using T = typename SrcWrap::ValueType;

    int cn       = cuda::NumElements<P>;
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid * cn >= height * width * channel)
        return;
    int h     = globalid / (width * channel / cn);
    int wid   = globalid % (width * channel / cn);
    int w     = (wid * cn) / channel;
    int c     = (wid * cn) % channel;
    int batch = blockIdx.z;

    T      MAX     = cuda::TypeTraits<T>::max;
    T      MIN     = cuda::TypeTraits<T>::min;
    double maxv    = _maxval[batch];
    double th      = _thresh[batch];
    int    imaxval = round(maxv);
    T      maxval  = cuda::SaturateCast<T>(imaxval);
    int    ithresh = floor(th);
    P      out;
    if (ithresh >= MIN && ithresh <= MAX)
    {
        T  thresh = (T)ithresh;
        P  in     = *((P *)src.ptr(batch, h, w, c));
        T *inval  = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]                   = inval[i] > thresh ? 0 : maxval;
            cuda::GetElement<P>(out, i) = outval[i];
        }
        *((P *)dst.ptr(batch, h, w, c)) = out;
        return;
    }
    if (ithresh < MIN)
    {
        out                             = cuda::SetAll<P>(0);
        *((P *)dst.ptr(batch, h, w, c)) = out;
        return;
    }

    out                             = cuda::SetAll<P>(maxval);
    *((P *)dst.ptr(batch, h, w, c)) = out;
}

template<typename P, typename SrcWrap, typename DstWrap>
__global__ void BinaryInv_Generic(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh,
                                  cuda::Tensor1DWrap<double, int32_t> _maxval, int height, int width, int channel)
{
    static_assert(std::is_same_v<SrcWrap, DstWrap>);
    using T = typename SrcWrap::ValueType;

    int cn       = cuda::NumElements<P>;
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid * cn >= height * width * channel)
        return;
    int h     = globalid / (width * channel / cn);
    int wid   = globalid % (width * channel / cn);
    int w     = (wid * cn) / channel;
    int c     = (wid * cn) % channel;
    int batch = blockIdx.z;

    T maxval = (T)_maxval[batch];
    T thresh = (T)_thresh[batch];

    P  out;
    P  in    = *((P *)src.ptr(batch, h, w, c));
    T *inval = reinterpret_cast<T *>((void *)&in);
    T  outval[4];
#pragma unroll
    for (int i = 0; i < cn; i++)
    {
        outval[i]                   = inval[i] > thresh ? (T)0 : maxval;
        cuda::GetElement<P>(out, i) = outval[i];
    }
    *((P *)dst.ptr(batch, h, w, c)) = out;
}

template<typename P, typename SrcWrap, typename DstWrap>
__global__ void Trunc_overflow(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh, int height,
                               int width, int channel)
{
    static_assert(std::is_same_v<SrcWrap, DstWrap>);
    using T = typename SrcWrap::ValueType;

    int cn       = cuda::NumElements<P>;
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid * cn >= height * width * channel)
        return;
    int h     = globalid / (width * channel / cn);
    int wid   = globalid % (width * channel / cn);
    int w     = (wid * cn) / channel;
    int c     = (wid * cn) % channel;
    int batch = blockIdx.z;

    T      MAX     = cuda::TypeTraits<T>::max;
    T      MIN     = cuda::TypeTraits<T>::min;
    double th      = _thresh[batch];
    int    ithresh = floor(th);
    P      out;
    if (ithresh >= MIN && ithresh <= MAX)
    {
        T  thresh = (T)ithresh;
        P  in     = *((P *)src.ptr(batch, h, w, c));
        T *inval  = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]                   = inval[i] > thresh ? thresh : inval[i];
            cuda::GetElement<P>(out, i) = outval[i];
        }
        *((P *)dst.ptr(batch, h, w, c)) = out;
        return;
    }
    if (ithresh < MIN)
    {
        out                             = cuda::SetAll<P>(MIN);
        *((P *)dst.ptr(batch, h, w, c)) = out;
        return;
    }

    *((P *)dst.ptr(batch, h, w, c)) = *((P *)src.ptr(batch, h, w, c));
}

template<typename P, typename SrcWrap, typename DstWrap>
__global__ void Trunc_Generic(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh, int height,
                              int width, int channel)
{
    static_assert(std::is_same_v<SrcWrap, DstWrap>);
    using T = typename SrcWrap::ValueType;

    int cn       = cuda::NumElements<P>;
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid * cn >= height * width * channel)
        return;
    int h     = globalid / (width * channel / cn);
    int wid   = globalid % (width * channel / cn);
    int w     = (wid * cn) / channel;
    int c     = (wid * cn) % channel;
    int batch = blockIdx.z;

    T  thresh = (T)_thresh[batch];
    P  out;
    P  in    = *((P *)src.ptr(batch, h, w, c));
    T *inval = reinterpret_cast<T *>((void *)&in);
    T  outval[4];
#pragma unroll
    for (int i = 0; i < cn; i++)
    {
        outval[i]                   = inval[i] > thresh ? thresh : inval[i];
        cuda::GetElement<P>(out, i) = outval[i];
    }
    *((P *)dst.ptr(batch, h, w, c)) = out;
}

template<typename P, typename SrcWrap, typename DstWrap>
__global__ void Tozero_overflow(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh, int height,
                                int width, int channel)
{
    static_assert(std::is_same_v<SrcWrap, DstWrap>);
    using T = typename SrcWrap::ValueType;

    int cn       = cuda::NumElements<P>;
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid * cn >= height * width * channel)
        return;
    int h     = globalid / (width * channel / cn);
    int wid   = globalid % (width * channel / cn);
    int w     = (wid * cn) / channel;
    int c     = (wid * cn) % channel;
    int batch = blockIdx.z;

    T      MAX     = cuda::TypeTraits<T>::max;
    T      MIN     = cuda::TypeTraits<T>::min;
    double th      = _thresh[batch];
    int    ithresh = floor(th);
    P      out;
    if (ithresh >= MIN && ithresh <= MAX)
    {
        T  thresh = (T)ithresh;
        P  in     = *((P *)src.ptr(batch, h, w, c));
        T *inval  = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]                   = inval[i] > thresh ? inval[i] : 0;
            cuda::GetElement<P>(out, i) = outval[i];
        }
        *((P *)dst.ptr(batch, h, w, c)) = out;
        return;
    }
    if (ithresh < MIN)
    {
        *((P *)dst.ptr(batch, h, w, c)) = *((P *)src.ptr(batch, h, w, c));
        return;
    }

    out                             = cuda::SetAll<P>(0);
    *((P *)dst.ptr(batch, h, w, c)) = out;
}

template<typename P, typename SrcWrap, typename DstWrap>
__global__ void Tozero_Generic(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh, int height,
                               int width, int channel)
{
    static_assert(std::is_same_v<SrcWrap, DstWrap>);
    using T = typename SrcWrap::ValueType;

    int cn       = cuda::NumElements<P>;
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid * cn >= height * width * channel)
        return;
    int h     = globalid / (width * channel / cn);
    int wid   = globalid % (width * channel / cn);
    int w     = (wid * cn) / channel;
    int c     = (wid * cn) % channel;
    int batch = blockIdx.z;

    T thresh = (T)_thresh[batch];

    P  out;
    P  in    = *((P *)src.ptr(batch, h, w, c));
    T *inval = reinterpret_cast<T *>((void *)&in);
    T  outval[4];
#pragma unroll
    for (int i = 0; i < cn; i++)
    {
        outval[i]                   = inval[i] > thresh ? inval[i] : (T)0;
        cuda::GetElement<P>(out, i) = outval[i];
    }
    *((P *)dst.ptr(batch, h, w, c)) = out;
}

template<typename P, typename SrcWrap, typename DstWrap>
__global__ void TozeroInv_overflow(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh, int height,
                                   int width, int channel)
{
    static_assert(std::is_same_v<SrcWrap, DstWrap>);
    using T = typename SrcWrap::ValueType;

    int cn       = cuda::NumElements<P>;
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid * cn >= height * width * channel)
        return;
    int h     = globalid / (width * channel / cn);
    int wid   = globalid % (width * channel / cn);
    int w     = (wid * cn) / channel;
    int c     = (wid * cn) % channel;
    int batch = blockIdx.z;

    T      MAX     = cuda::TypeTraits<T>::max;
    T      MIN     = cuda::TypeTraits<T>::min;
    double th      = _thresh[batch];
    int    ithresh = floor(th);
    P      out;
    if (ithresh >= MIN && ithresh <= MAX)
    {
        T  thresh = (T)ithresh;
        P  in     = *((P *)src.ptr(batch, h, w, c));
        T *inval  = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]                   = inval[i] > thresh ? 0 : inval[i];
            cuda::GetElement<P>(out, i) = outval[i];
        }
        *((P *)dst.ptr(batch, h, w, c)) = out;
        return;
    }
    if (ithresh < MIN)
    {
        out                             = cuda::SetAll<P>(0);
        *((P *)dst.ptr(batch, h, w, c)) = out;
        return;
    }

    *((P *)dst.ptr(batch, h, w, c)) = *((P *)src.ptr(batch, h, w, c));
}

template<typename P, typename SrcWrap, typename DstWrap>
__global__ void TozeroInv_Generic(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh, int height,
                                  int width, int channel)
{
    static_assert(std::is_same_v<SrcWrap, DstWrap>);
    using T = typename SrcWrap::ValueType;

    int cn       = cuda::NumElements<P>;
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid * cn >= height * width * channel)
        return;
    int h     = globalid / (width * channel / cn);
    int wid   = globalid % (width * channel / cn);
    int w     = (wid * cn) / channel;
    int c     = (wid * cn) % channel;
    int batch = blockIdx.z;

    T thresh = (T)_thresh[batch];

    P  out;
    P  in    = *((P *)src.ptr(batch, h, w, c));
    T *inval = reinterpret_cast<T *>((void *)&in);
    T  outval[4];
#pragma unroll
    for (int i = 0; i < cn; i++)
    {
        outval[i]                   = inval[i] > thresh ? (T)0 : inval[i];
        cuda::GetElement<P>(out, i) = outval[i];
    }
    *((P *)dst.ptr(batch, h, w, c)) = out;
}

__global__ void hist_kernel(cuda::Tensor3DWrap<uchar, int32_t> img, int *histogram, int rows, int cols)
{
    __shared__ int hist[256];
    int            localid = threadIdx.x;
    hist[localid]          = 0;
    __syncthreads();

    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int threadCol = ceil((float)cols / 16);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * 16;
    int batch     = blockIdx.z;

    if (h < rows)
    {
        uchar *ptr = img.ptr(batch, h, w);
        // Use a 16-byte vectorized load when the pointer is aligned and 16 bytes remain.
        // When the tensor stride does not guarantee 16-byte alignment, or fewer than
        // 16 columns remain, fall back to per-element loads to avoid out-of-bounds
        // reads and misaligned-access crashes.
        if (w + 16 <= cols && (reinterpret_cast<std::uintptr_t>(ptr) % 16) == 0)
        {
            int4   src   = *((int4 *)ptr);
            uchar *inval = reinterpret_cast<uchar *>((void *)&src);
            for (int i = 0; i < 16; i++) atomicAdd(&hist[inval[i]], 1);
        }
        else
        {
            int end = min(w + 16, cols);
            for (int i = w; i < end; i++) atomicAdd(&hist[*img.ptr(batch, h, i)], 1);
        }
    }
    __syncthreads();

    int val = hist[localid];
    if (val > 0)
        atomicAdd(&histogram[blockIdx.z * 256 + localid], val);
}

__global__ void otsu_cal(int *histogram, cuda::Tensor1DWrap<double, int32_t> thresh, int size)
{
    OtsuCalBody(histogram, thresh, size);
}

template<typename T, typename SrcWrap, typename DstWrap>
__global__ void ThresholdPlanarTensor(SrcWrap src, DstWrap dst, cuda::Tensor1DWrap<double, int32_t> _thresh,
                                      cuda::Tensor1DWrap<double, int32_t> _maxval, int rows, int cols, int channels,
                                      NVCVThresholdType type)
{
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalid >= rows * cols)
        return;

    int batch = blockIdx.z / channels;
    int plane = blockIdx.z % channels;
    int y     = globalid / cols;
    int x     = globalid % cols;

    T inval                      = *src.ptr(batch, plane, y, x);
    T out                        = ThresholdValueDispatch(inval, _thresh[batch], _maxval[batch], type);
    *dst.ptr(batch, plane, y, x) = out;
}

template<typename T, int N>
void thresholdDispatch(const nvcv::TensorDataStridedCuda &input, const nvcv::TensorDataStridedCuda &output,
                       const nvcv::TensorDataStridedCuda &_thresh, const nvcv::TensorDataStridedCuda &_maxval,
                       int batch, int rows, int cols, int channel, NVCVThresholdType type, cudaStream_t stream)
{
    using vectype = cuda::MakeType<T, N>;

    int                                 size = rows * cols * channel;
    cuda::Tensor1DWrap<double, int32_t> thresh(_thresh);
    cuda::Tensor1DWrap<double, int32_t> maxval(_maxval);

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(input);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(output);

    ValidateTensorFitsInt32(*inAccess, *outAccess);

    if constexpr (N > 1)
    {
        std::uintptr_t packAlignmentBits
            = reinterpret_cast<std::uintptr_t>(input.basePtr()) | reinterpret_cast<std::uintptr_t>(output.basePtr())
            | static_cast<std::uintptr_t>(inAccess->rowStride()) | static_cast<std::uintptr_t>(outAccess->rowStride())
            | static_cast<std::uintptr_t>(inAccess->sampleStride())
            | static_cast<std::uintptr_t>(outAccess->sampleStride());

        if ((packAlignmentBits & (alignof(vectype) - 1)) != 0)
        {
            thresholdDispatch<T, 1>(input, output, _thresh, _maxval, batch, rows, cols, channel, type, stream);
            return;
        }
    }

    auto src_ptr = cuda::CreateTensorWrapNHWC<T, int32_t>(input);
    auto dst_ptr = cuda::CreateTensorWrapNHWC<T, int32_t>(output);
    dim3 block(256);
    dim3 grid(util::DivUp(size, static_cast<int>(block.x) * N), 1, batch);

    constexpr bool useGeneric = UseGenericThreshold<T>;

    switch (type)
    {
    case NVCV_THRESH_BINARY:
        if constexpr (useGeneric)
            Binary_Generic<vectype><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, rows, cols, channel);
        else
            Binary_overflow<vectype><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, rows, cols, channel);
        break;
    case NVCV_THRESH_BINARY_INV:
        if constexpr (useGeneric)
            BinaryInv_Generic<vectype>
                <<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, rows, cols, channel);
        else
            BinaryInv_overflow<vectype>
                <<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, rows, cols, channel);
        break;
    case NVCV_THRESH_TRUNC:
        if constexpr (useGeneric)
            Trunc_Generic<vectype><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, rows, cols, channel);
        else
            Trunc_overflow<vectype><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, rows, cols, channel);
        break;
    case NVCV_THRESH_TOZERO:
        if constexpr (useGeneric)
            Tozero_Generic<vectype><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, rows, cols, channel);
        else
            Tozero_overflow<vectype><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, rows, cols, channel);
        break;
    default: //NVCV_THRESH_TOZERO_INV
        if constexpr (useGeneric)
            TozeroInv_Generic<vectype><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, rows, cols, channel);
        else
            TozeroInv_overflow<vectype><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, rows, cols, channel);
        break;
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

// Element batching: rows whose element count is a multiple of four (two for 8-byte types) get one
// packed access per thread. This only establishes that the count divides evenly; thresholdDispatch
// re-checks the actual pointer and stride alignment and falls back to N == 1 when it does not hold.
template<typename T>
void thresholdScale(const nvcv::TensorDataStridedCuda &input, const nvcv::TensorDataStridedCuda &output,
                    const nvcv::TensorDataStridedCuda &threshold, const nvcv::TensorDataStridedCuda &maxval, int batch,
                    int rows, int cols, int channel, NVCVThresholdType type, cudaStream_t stream)
{
    int stride = cols * channel;

    if (stride % 4 == 0)
    {
        if constexpr (std::is_same_v<T, double>)
            thresholdDispatch<T, 2>(input, output, threshold, maxval, batch, rows, cols, channel, type, stream);
        else
            thresholdDispatch<T, 4>(input, output, threshold, maxval, batch, rows, cols, channel, type, stream);
    }
    else if (stride % 2 == 0)
        thresholdDispatch<T, 2>(input, output, threshold, maxval, batch, rows, cols, channel, type, stream);
    else
        thresholdDispatch<T, 1>(input, output, threshold, maxval, batch, rows, cols, channel, type, stream);
}

// Planar input keeps a dedicated NCHW kernel instead of reusing PlanarSingleChannelViews the way the
// automatic modes do below: thresh and maxval are indexed per sample, so flattening the N*C planes
// into N*C samples would make plane (n, c) read thresh[n * C + c] instead of thresh[n]. The automatic
// modes escape that only because they additionally require a single channel.
template<typename T>
void thresholdScalePlanar(const nvcv::TensorDataStridedCuda &input, const nvcv::TensorDataStridedCuda &output,
                          const nvcv::TensorDataStridedCuda &threshold, const nvcv::TensorDataStridedCuda &maxval,
                          int batch, int rows, int cols, int channels, NVCVThresholdType type, cudaStream_t stream)
{
    cuda::Tensor1DWrap<double, int32_t> thresh(threshold);
    cuda::Tensor1DWrap<double, int32_t> maxv(maxval);

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(input);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(output);

    ValidateTensorFitsInt32(*inAccess, *outAccess);

    const int64_t planarBatch = static_cast<int64_t>(batch) * channels;
    if (planarBatch > kMaxGridZ)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar Threshold requires batch * channels <= 65535 (CUDA grid-z limit)");
    }

    auto src_ptr = cuda::CreateTensorWrapNCHW<T, int32_t>(input);
    auto dst_ptr = cuda::CreateTensorWrapNCHW<T, int32_t>(output);

    dim3 block(256);
    dim3 grid(util::DivUp(rows * cols, static_cast<int>(block.x)), 1, batch * channels);
    ThresholdPlanarTensor<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxv, rows, cols, channels, type);

    NVCV_CHECK_THROW(cudaGetLastError());
}

// Clears the per-image histograms and accumulates one pass over the input. Both automatic modes
// consume the same histogram; only the selection kernel launched afterwards differs.
inline void AccumulateHistogram(const nvcv::TensorDataStridedCuda &inData, int *histogram, int rows, int cols,
                                int batch, cudaStream_t stream)
{
    NVCV_CHECK_THROW(cudaMemsetAsync(histogram, 0, sizeof(int) * kHistogramBins * batch, stream));

    auto wrap = cuda::CreateTensorWrapNHW<uchar, int32_t>(inData);

    dim3 block(256);
    int  td = util::DivUp(cols, 16) * rows;
    dim3 grid(util::DivUp(td, 256), 1, batch);
    hist_kernel<<<grid, block, 0, stream>>>(wrap, histogram, rows, cols);
}

inline void getThreshVal_Triangle(const nvcv::TensorDataStridedCuda &inData,
                                  const nvcv::TensorDataStridedCuda &threshold, int *histogram, int rows, int cols,
                                  int batch, cudaStream_t stream)
{
    AccumulateHistogram(inData, histogram, rows, cols, batch, stream);

    cuda::Tensor1DWrap<double, int32_t> thresh(threshold);
    dim3                                block2(256);
    dim3                                grid2(1, 1, batch);
    triangle_cal<<<grid2, block2, 0, stream>>>(histogram, thresh);
}

inline void getThreshVal_Otsu(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &threshold,
                              int *histogram, int rows, int cols, int batch, cudaStream_t stream)
{
    AccumulateHistogram(inData, histogram, rows, cols, batch, stream);

    cuda::Tensor1DWrap<double, int32_t> thresh(threshold);
    dim3                                block2(256);
    dim3                                grid2(1, 1, batch);
    otsu_cal<<<grid2, block2, 0, stream>>>(histogram, thresh, rows * cols);
}

inline void RunThresholdScale(DataTypeCode code, const nvcv::TensorDataStridedCuda &input,
                              const nvcv::TensorDataStridedCuda &output, const nvcv::TensorDataStridedCuda &threshold,
                              const nvcv::TensorDataStridedCuda &maxval, int batch, int rows, int cols, int channel,
                              NVCVThresholdType type, cudaStream_t stream)
{
    DispatchByDataType(code,
                       [&](auto elem) {
                           thresholdScale<decltype(elem)>(input, output, threshold, maxval, batch, rows, cols, channel,
                                                          type, stream);
                       });
}

inline void RunThresholdScalePlanar(DataTypeCode code, const nvcv::TensorDataStridedCuda &input,
                                    const nvcv::TensorDataStridedCuda &output,
                                    const nvcv::TensorDataStridedCuda &threshold,
                                    const nvcv::TensorDataStridedCuda &maxval, int batch, int rows, int cols,
                                    int channels, NVCVThresholdType type, cudaStream_t stream)
{
    DispatchByDataType(code,
                       [&](auto elem) {
                           thresholdScalePlanar<decltype(elem)>(input, output, threshold, maxval, batch, rows, cols,
                                                                channels, type, stream);
                       });
}

} // namespace tensor_impl

// ---------------------------------------------------------------------------------------------
// VarShape path
// ---------------------------------------------------------------------------------------------
namespace varshape_impl {

constexpr int kU8BinaryElementsPerThread = sizeof(uint4) / sizeof(uchar);

__device__ __forceinline__ uint4 SetAllU8Pack(uchar value)
{
    unsigned int word = 0x01010101u * value;
    uint4        out;
    out.x = word;
    out.y = word;
    out.z = word;
    out.w = word;
    return out;
}

__device__ __forceinline__ uint4 BinaryThresholdU8Pack(uint4 in, uchar thresh, uchar maxval)
{
    uint4  out;
    uchar *inval  = reinterpret_cast<uchar *>(&in);
    uchar *outval = reinterpret_cast<uchar *>(&out);

#pragma unroll
    for (int i = 0; i < kU8BinaryElementsPerThread; i++) outval[i] = inval[i] > thresh ? maxval : 0;

    return out;
}

template<typename T, typename P = cuda::MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Binary_overflow(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                cuda::Tensor1DWrap<double, int32_t> _thresh,
                                cuda::Tensor1DWrap<double, int32_t> _maxval, int channel)
{
    int cn        = cuda::NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T      MAX     = cuda::TypeTraits<T>::max;
    T      MIN     = cuda::TypeTraits<T>::min;
    double maxv    = _maxval[batch];
    double th      = _thresh[batch];
    int    imaxval = round(maxv);
    T      maxval  = cuda::SaturateCast<T>(imaxval);
    int    ithresh = floor(th);

    int loop = width * channel - w;
    int c    = w % channel;
    w        = w / channel;
    if (loop >= cn)
    {
        P out;
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T  thresh = (T)ithresh;
            P  in     = LoadPacked<P>(src.ptr(batch, h, w, c));
            T *inval  = reinterpret_cast<T *>((void *)&in);
            T  outval[4];
#pragma unroll
            for (int i = 0; i < cn; i++)
            {
                outval[i]                   = inval[i] > thresh ? maxval : 0;
                cuda::GetElement<P>(out, i) = outval[i];
            }
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }
        if (ithresh < MIN)
        {
            out = cuda::SetAll<P>(maxval);
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }

        out = cuda::SetAll<P>(0);
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T thresh = (T)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
            {
                T inval                        = *(src.ptr(batch, h, w, c) + i);
                *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? maxval : 0;
            }
            return;
        }
        if (ithresh < MIN)
        {
#pragma unroll
            for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = maxval;
            return;
        }

#pragma unroll
        for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = 0;
    }
}

// Sixteen-byte element batching for the 8-bit binary case: four times the elements per thread of the
// generic packed path, with an unaligned fallback because var-shape rows are not guaranteed aligned.
__global__ void Binary_overflow_u8_nix16(cuda::ImageBatchVarShapeWrapNHWC<uchar> src,
                                         cuda::ImageBatchVarShapeWrapNHWC<uchar> dst,
                                         cuda::Tensor1DWrap<double, int32_t>     _thresh,
                                         cuda::Tensor1DWrap<double, int32_t> _maxval, int channel)
{
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch    = blockIdx.z;
    int width    = src.width(batch);
    int height   = src.height(batch);
    int rowElems = width * channel;
    if (rowElems == 0)
        return;

    int threadCol = (rowElems + kU8BinaryElementsPerThread - 1) / kU8BinaryElementsPerThread;
    int h         = globalid / threadCol;
    int elem      = (globalid % threadCol) * kU8BinaryElementsPerThread;
    if (h >= height || elem >= rowElems)
        return;

    int    imaxval = round(_maxval[batch]);
    uchar  maxval  = cuda::SaturateCast<uchar>(imaxval);
    int    ithresh = floor(_thresh[batch]);
    int    loop    = rowElems - elem;
    int    c       = elem % channel;
    int    w       = elem / channel;
    uchar *srcRow  = src.ptr(batch, h, w, c);
    uchar *dstRow  = dst.ptr(batch, h, w, c);
    bool   aligned
        = ((reinterpret_cast<std::uintptr_t>(srcRow) | reinterpret_cast<std::uintptr_t>(dstRow)) & (alignof(uint4) - 1))
       == 0;

    if (loop >= kU8BinaryElementsPerThread)
    {
        uint4 out;
        if (ithresh >= cuda::TypeTraits<uchar>::min && ithresh <= cuda::TypeTraits<uchar>::max)
        {
            uchar thresh = (uchar)ithresh;
            if (aligned)
            {
                uint4 in                             = *(reinterpret_cast<uint4 *>(srcRow));
                *(reinterpret_cast<uint4 *>(dstRow)) = BinaryThresholdU8Pack(in, thresh, maxval);
            }
            else
            {
#pragma unroll
                for (int i = 0; i < kU8BinaryElementsPerThread; i++)
                {
                    uchar inval = srcRow[i];
                    dstRow[i]   = inval > thresh ? maxval : 0;
                }
            }
            return;
        }

        uchar value;
        if (ithresh < cuda::TypeTraits<uchar>::min)
        {
            out   = SetAllU8Pack(maxval);
            value = maxval;
        }
        else
        {
            out   = SetAllU8Pack(0);
            value = 0;
        }

        if (aligned)
            *(reinterpret_cast<uint4 *>(dstRow)) = out;
        else
        {
#pragma unroll
            for (int i = 0; i < kU8BinaryElementsPerThread; i++) dstRow[i] = value;
        }
    }
    else
    {
        if (ithresh >= cuda::TypeTraits<uchar>::min && ithresh <= cuda::TypeTraits<uchar>::max)
        {
            uchar thresh = (uchar)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
            {
                uchar inval = srcRow[i];
                dstRow[i]   = inval > thresh ? maxval : 0;
            }
            return;
        }

        uchar out = ithresh < cuda::TypeTraits<uchar>::min ? maxval : 0;
#pragma unroll
        for (int i = 0; i < loop; i++) dstRow[i] = out;
    }
}

template<typename T, typename P = cuda::MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Binary_Generic(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                               cuda::Tensor1DWrap<double, int32_t> _thresh, cuda::Tensor1DWrap<double, int32_t> _maxval,
                               int channel)
{
    int cn        = cuda::NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T   maxval = (T)_maxval[batch];
    T   thresh = (T)_thresh[batch];
    int loop   = width * channel - w;
    int c      = w % channel;
    w          = w / channel;

    if (loop >= cn)
    {
        P  out;
        P  in    = LoadPacked<P>(src.ptr(batch, h, w, c));
        T *inval = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            // (T)0 keeps both conditional operands a single type; the mixed __half/int
            // conditional is ambiguous (same in the other Generic kernels below).
            outval[i]                   = inval[i] > thresh ? maxval : (T)0;
            cuda::GetElement<P>(out, i) = outval[i];
        }
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < loop; i++)
        {
            T inval                        = *(src.ptr(batch, h, w, c) + i);
            *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? maxval : (T)0;
        }
    }
}

template<typename T, typename P = cuda::MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void BinaryInv_overflow(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                   cuda::Tensor1DWrap<double, int32_t> _thresh,
                                   cuda::Tensor1DWrap<double, int32_t> _maxval, int channel)
{
    int cn        = cuda::NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T      MAX     = cuda::TypeTraits<T>::max;
    T      MIN     = cuda::TypeTraits<T>::min;
    double maxv    = _maxval[batch];
    double th      = _thresh[batch];
    int    imaxval = round(maxv);
    T      maxval  = cuda::SaturateCast<T>(imaxval);
    int    ithresh = floor(th);

    int loop = width * channel - w;
    int c    = w % channel;
    w        = w / channel;
    if (loop >= cn)
    {
        P out;
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T  thresh = (T)ithresh;
            P  in     = LoadPacked<P>(src.ptr(batch, h, w, c));
            T *inval  = reinterpret_cast<T *>((void *)&in);
            T  outval[4];
#pragma unroll
            for (int i = 0; i < cn; i++)
            {
                outval[i]                   = inval[i] > thresh ? 0 : maxval;
                cuda::GetElement<P>(out, i) = outval[i];
            }
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }
        if (ithresh < MIN)
        {
            out = cuda::SetAll<P>(0);
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }

        out = cuda::SetAll<P>(maxval);
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T thresh = (T)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
                *(dst.ptr(batch, h, w, c) + i) = *(src.ptr(batch, h, w, c) + i) > thresh ? 0 : maxval;
            return;
        }
        if (ithresh < MIN)
        {
#pragma unroll
            for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = 0;
            return;
        }

#pragma unroll
        for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = maxval;
    }
}

template<typename T, typename P = cuda::MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void BinaryInv_Generic(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                  cuda::Tensor1DWrap<double, int32_t> _thresh,
                                  cuda::Tensor1DWrap<double, int32_t> _maxval, int channel)
{
    int cn        = cuda::NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T   maxval = (T)_maxval[batch];
    T   thresh = (T)_thresh[batch];
    int loop   = width * channel - w;
    int c      = w % channel;
    w          = w / channel;

    if (loop >= cn)
    {
        P  out;
        P  in    = LoadPacked<P>(src.ptr(batch, h, w, c));
        T *inval = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]                   = inval[i] > thresh ? (T)0 : maxval;
            cuda::GetElement<P>(out, i) = outval[i];
        }
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < loop; i++)
        {
            T inval                        = *(src.ptr(batch, h, w, c) + i);
            *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? (T)0 : maxval;
        }
    }
}

template<typename T, typename P = cuda::MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Trunc_overflow(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                               cuda::Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = cuda::NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T      MAX     = cuda::TypeTraits<T>::max;
    T      MIN     = cuda::TypeTraits<T>::min;
    double th      = _thresh[batch];
    int    ithresh = floor(th);

    int loop = width * channel - w;
    int c    = w % channel;
    w        = w / channel;
    if (loop >= cn)
    {
        P out;
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T  thresh = (T)ithresh;
            P  in     = LoadPacked<P>(src.ptr(batch, h, w, c));
            T *inval  = reinterpret_cast<T *>((void *)&in);
            T  outval[4];
#pragma unroll
            for (int i = 0; i < cn; i++)
            {
                outval[i]                   = inval[i] > thresh ? thresh : inval[i];
                cuda::GetElement<P>(out, i) = outval[i];
            }
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }
        if (ithresh < MIN)
        {
            out = cuda::SetAll<P>(MIN);
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }

        StorePacked(dst.ptr(batch, h, w, c), LoadPacked<P>(src.ptr(batch, h, w, c)));
    }
    else
    {
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T thresh = (T)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
            {
                T inval                        = *(src.ptr(batch, h, w, c) + i);
                *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? thresh : inval;
            }
            return;
        }
        if (ithresh < MIN)
        {
#pragma unroll
            for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = MIN;
            return;
        }

#pragma unroll
        for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = *(src.ptr(batch, h, w, c) + i);
    }
}

template<typename T, typename P = cuda::MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Trunc_Generic(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                              cuda::Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = cuda::NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T   thresh = (T)_thresh[batch];
    int loop   = width * channel - w;
    int c      = w % channel;
    w          = w / channel;

    if (loop >= cn)
    {
        P  out;
        P  in    = LoadPacked<P>(src.ptr(batch, h, w, c));
        T *inval = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]                   = inval[i] > thresh ? thresh : inval[i];
            cuda::GetElement<P>(out, i) = outval[i];
        }
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < loop; i++)
        {
            T inval                        = *(src.ptr(batch, h, w, c) + i);
            *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? thresh : inval;
        }
    }
}

template<typename T, typename P = cuda::MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Tozero_overflow(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                cuda::Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = cuda::NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T      MAX     = cuda::TypeTraits<T>::max;
    T      MIN     = cuda::TypeTraits<T>::min;
    double th      = _thresh[batch];
    int    ithresh = floor(th);

    int loop = width * channel - w;
    int c    = w % channel;
    w        = w / channel;
    if (loop >= cn)
    {
        P out;
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T  thresh = (T)ithresh;
            P  in     = LoadPacked<P>(src.ptr(batch, h, w, c));
            T *inval  = reinterpret_cast<T *>((void *)&in);
            T  outval[4];
#pragma unroll
            for (int i = 0; i < cn; i++)
            {
                outval[i]                   = inval[i] > thresh ? inval[i] : 0;
                cuda::GetElement<P>(out, i) = outval[i];
            }
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }
        if (ithresh < MIN)
        {
            StorePacked(dst.ptr(batch, h, w, c), LoadPacked<P>(src.ptr(batch, h, w, c)));
            return;
        }

        out = cuda::SetAll<P>(0);
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T thresh = (T)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
            {
                T inval                        = *(src.ptr(batch, h, w, c) + i);
                *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? inval : 0;
            }
            return;
        }
        if (ithresh < MIN)
        {
#pragma unroll
            for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = *(src.ptr(batch, h, w, c) + i);
            return;
        }

#pragma unroll
        for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = 0;
    }
}

template<typename T, typename P = cuda::MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void Tozero_Generic(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                               cuda::Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = cuda::NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T   thresh = (T)_thresh[batch];
    int loop   = width * channel - w;
    int c      = w % channel;
    w          = w / channel;

    if (loop >= cn)
    {
        P  out;
        P  in    = LoadPacked<P>(src.ptr(batch, h, w, c));
        T *inval = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]                   = inval[i] > thresh ? inval[i] : (T)0;
            cuda::GetElement<P>(out, i) = outval[i];
        }
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < loop; i++)
        {
            T inval                        = *(src.ptr(batch, h, w, c) + i);
            *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? inval : (T)0;
        }
    }
}

template<typename T, typename P = cuda::MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void TozeroInv_overflow(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                   cuda::Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = cuda::NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T      MAX     = cuda::TypeTraits<T>::max;
    T      MIN     = cuda::TypeTraits<T>::min;
    double th      = _thresh[batch];
    int    ithresh = floor(th);

    int loop = width * channel - w;
    int c    = w % channel;
    w        = w / channel;
    if (loop >= cn)
    {
        P out;
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T  thresh = (T)ithresh;
            P  in     = LoadPacked<P>(src.ptr(batch, h, w, c));
            T *inval  = reinterpret_cast<T *>((void *)&in);
            T  outval[4];
#pragma unroll
            for (int i = 0; i < cn; i++)
            {
                outval[i]                   = inval[i] > thresh ? 0 : inval[i];
                cuda::GetElement<P>(out, i) = outval[i];
            }
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }
        if (ithresh < MIN)
        {
            out = cuda::SetAll<P>(0);
            StorePacked(dst.ptr(batch, h, w, c), out);
            return;
        }

        StorePacked(dst.ptr(batch, h, w, c), LoadPacked<P>(src.ptr(batch, h, w, c)));
    }
    else
    {
        if (ithresh >= MIN && ithresh <= MAX)
        {
            T thresh = (T)ithresh;
#pragma unroll
            for (int i = 0; i < loop; i++)
            {
                T inval                        = *(src.ptr(batch, h, w, c) + i);
                *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? 0 : inval;
            }
            return;
        }
        if (ithresh < MIN)
        {
#pragma unroll
            for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = 0;
            return;
        }

#pragma unroll
        for (int i = 0; i < loop; i++) *(dst.ptr(batch, h, w, c) + i) = *(src.ptr(batch, h, w, c) + i);
    }
}

template<typename T, typename P = cuda::MakeType<T, sizeof(T) == 8 ? 2 : 4>>
__global__ void TozeroInv_Generic(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                  cuda::Tensor1DWrap<double, int32_t> _thresh, int channel)
{
    int cn        = cuda::NumElements<P>;
    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int width     = src.width(batch);
    int height    = src.height(batch);
    int threadCol = ceil((float)width * channel / cn);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * cn;
    if (h >= height || w >= width * channel)
        return;

    T   thresh = (T)_thresh[batch];
    int loop   = width * channel - w;
    int c      = w % channel;
    w          = w / channel;

    if (loop >= cn)
    {
        P  out;
        P  in    = LoadPacked<P>(src.ptr(batch, h, w, c));
        T *inval = reinterpret_cast<T *>((void *)&in);
        T  outval[4];
#pragma unroll
        for (int i = 0; i < cn; i++)
        {
            outval[i]                   = inval[i] > thresh ? (T)0 : inval[i];
            cuda::GetElement<P>(out, i) = outval[i];
        }
        StorePacked(dst.ptr(batch, h, w, c), out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < loop; i++)
        {
            T inval                        = *(src.ptr(batch, h, w, c) + i);
            *(dst.ptr(batch, h, w, c) + i) = inval > thresh ? (T)0 : inval;
        }
    }
}

template<typename T>
__global__ void ThresholdPlanar(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst,
                                cuda::Tensor1DWrap<double, int32_t> _thresh,
                                cuda::Tensor1DWrap<double, int32_t> _maxval, int channels, NVCVThresholdType type)
{
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch    = blockIdx.z / channels;
    int plane    = blockIdx.z % channels;
    int width    = src.width(batch);
    int height   = src.height(batch);

    if (globalid >= width * height)
        return;

    int y = globalid / width;
    int x = globalid % width;

    T inval                      = *src.ptr(batch, plane, y, x);
    T out                        = ThresholdValueDispatch(inval, _thresh[batch], _maxval[batch], type);
    *dst.ptr(batch, plane, y, x) = out;
}

// Planar u8 BINARY specialization of ThresholdPlanar: one thread owns 16 consecutive elements of a
// row instead of one, so a full pack is a single 16-byte load/store. Rows are contiguous within a
// plane, and the thresh/maxval folding is loop-invariant, so only the tail of a row (fewer than 16
// remaining elements) falls back to element-wise access.
__global__ void BinaryThresholdPlanarU8(cuda::ImageBatchVarShapeWrap<uchar> src,
                                        cuda::ImageBatchVarShapeWrap<uchar> dst,
                                        cuda::Tensor1DWrap<double, int32_t> _thresh,
                                        cuda::Tensor1DWrap<double, int32_t> _maxval, int channels)
{
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch    = blockIdx.z / channels;
    int plane    = blockIdx.z % channels;
    int width    = src.width(batch);
    int height   = src.height(batch);
    int rowPacks = (width + kU8BinaryElementsPerThread - 1) / kU8BinaryElementsPerThread;
    int y        = globalid / rowPacks;
    int x        = (globalid % rowPacks) * kU8BinaryElementsPerThread;

    if (y >= height || x >= width)
        return;

    int    ithresh = floor(_thresh[batch]);
    uchar  maxval  = cuda::SaturateCast<uchar>(round(_maxval[batch]));
    uchar *srcPtr  = src.ptr(batch, plane, y, x);
    uchar *dstPtr  = dst.ptr(batch, plane, y, x);

    if (x + kU8BinaryElementsPerThread <= width)
    {
        uint4 out;
        if (ithresh < cuda::TypeTraits<uchar>::min)
            out = SetAllU8Pack(maxval);
        else if (ithresh > cuda::TypeTraits<uchar>::max)
            out = SetAllU8Pack(0);
        else
            out = BinaryThresholdU8Pack(LoadPacked<uint4>(srcPtr), static_cast<uchar>(ithresh), maxval);
        StorePacked(dstPtr, out);
    }
    else
    {
        int count = width - x;
        if (ithresh < cuda::TypeTraits<uchar>::min)
        {
#pragma unroll
            for (int i = 0; i < kU8BinaryElementsPerThread; ++i)
                if (i < count)
                    dstPtr[i] = maxval;
        }
        else if (ithresh > cuda::TypeTraits<uchar>::max)
        {
#pragma unroll
            for (int i = 0; i < kU8BinaryElementsPerThread; ++i)
                if (i < count)
                    dstPtr[i] = 0;
        }
        else
        {
            uchar thresh = static_cast<uchar>(ithresh);
#pragma unroll
            for (int i = 0; i < kU8BinaryElementsPerThread; ++i)
                if (i < count)
                    dstPtr[i] = srcPtr[i] > thresh ? maxval : 0;
        }
    }
}

__global__ void hist_kernel(cuda::ImageBatchVarShapeWrapNHWC<uchar> img, int *histogram)
{
    __shared__ int hist[256];
    int            localid = threadIdx.x;
    hist[localid]          = 0;
    __syncthreads();

    int globalid  = blockIdx.x * blockDim.x + threadIdx.x;
    int batch     = blockIdx.z;
    int cols      = img.width(batch);
    int rows      = img.height(batch);
    int threadCol = ceil((float)cols / 16);
    int h         = globalid / threadCol;
    int w         = (globalid % threadCol) * 16;

    if (h < rows)
    {
        if (w + 16 > cols)
        {
            for (int i = w; i < cols; i++) atomicAdd(&hist[*img.ptr(batch, h, i)], 1);
        }
        else
        {
            int4   src   = LoadPacked<int4>(img.ptr(batch, h, w));
            uchar *inval = reinterpret_cast<uchar *>((void *)&src);
            for (int i = 0; i < 16; i++) atomicAdd(&hist[inval[i]], 1);
        }
    }
    __syncthreads();

    int val = hist[localid];
    if (val > 0)
        atomicAdd(&histogram[blockIdx.z * 256 + localid], val);
}

// The var-shape twin of tensor_impl::otsu_cal: the normalization size comes from the per-image
// dimensions rather than a host-side constant.
__global__ void otsu_cal_varshape(int *histogram, cuda::Tensor1DWrap<double, int32_t> thresh,
                                  cuda::ImageBatchVarShapeWrapNHWC<uchar> img)
{
    OtsuCalBody(histogram, thresh, img.width((int)blockIdx.z) * img.height((int)blockIdx.z));
}

template<typename T>
void thresholdDispatch(const nvcv::ImageBatchVarShapeDataStridedCuda &input,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &output,
                       const nvcv::TensorDataStridedCuda &_thresh, const nvcv::TensorDataStridedCuda &_maxval,
                       NVCVThresholdType type, cudaStream_t stream)
{
    cuda::Tensor1DWrap<double, int32_t> thresh(_thresh);
    cuda::Tensor1DWrap<double, int32_t> maxval(_maxval);

    nvcv::Size2D maxsize = input.maxSize();
    int          batch   = input.numImages();
    int          channel = input.uniqueFormat().numChannels();

    cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(input, channel);
    cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(output, channel);

    dim3 block(256);
    int  N  = sizeof(T) == 8 ? 2 : 4;
    int  td = util::DivUp(maxsize.w * channel, N) * maxsize.h;
    dim3 grid(util::DivUp(td, 256), 1, batch);

    constexpr bool useGeneric = UseGenericThreshold<T>;

    switch (type)
    {
    case NVCV_THRESH_BINARY:
        if constexpr (useGeneric)
            Binary_Generic<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channel);
        else if constexpr (std::is_same_v<T, uchar>)
        {
            int  tdU8 = util::DivUp(maxsize.w * channel, kU8BinaryElementsPerThread) * maxsize.h;
            dim3 gridU8(util::DivUp(tdU8, 256), 1, batch);
            Binary_overflow_u8_nix16<<<gridU8, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channel);
        }
        else
            Binary_overflow<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channel);
        break;
    case NVCV_THRESH_BINARY_INV:
        if constexpr (useGeneric)
            BinaryInv_Generic<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channel);
        else
            BinaryInv_overflow<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channel);
        break;
    case NVCV_THRESH_TRUNC:
        if constexpr (useGeneric)
            Trunc_Generic<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        else
            Trunc_overflow<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        break;
    case NVCV_THRESH_TOZERO:
        if constexpr (useGeneric)
            Tozero_Generic<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        else
            Tozero_overflow<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        break;
    default: //NVCV_THRESH_TOZERO_INV
        if constexpr (useGeneric)
            TozeroInv_Generic<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        else
            TozeroInv_overflow<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, channel);
        break;
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void thresholdDispatchPlanar(const nvcv::ImageBatchVarShapeDataStridedCuda &input,
                             const nvcv::ImageBatchVarShapeDataStridedCuda &output,
                             const nvcv::TensorDataStridedCuda &_thresh, const nvcv::TensorDataStridedCuda &_maxval,
                             int channels, NVCVThresholdType type, cudaStream_t stream)
{
    cuda::Tensor1DWrap<double, int32_t> thresh(_thresh);
    cuda::Tensor1DWrap<double, int32_t> maxval(_maxval);

    nvcv::Size2D maxsize = input.maxSize();
    int          batch   = input.numImages();

    cuda::ImageBatchVarShapeWrap<T> src_ptr(input);
    cuda::ImageBatchVarShapeWrap<T> dst_ptr(output);

    dim3 block(256);
    if constexpr (std::is_same_v<T, uchar>)
    {
        if (type == NVCV_THRESH_BINARY)
        {
            int  tdU8 = util::DivUp(maxsize.w, kU8BinaryElementsPerThread) * maxsize.h;
            dim3 gridU8(util::DivUp(tdU8, static_cast<int>(block.x)), 1, batch * channels);
            BinaryThresholdPlanarU8<<<gridU8, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channels);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }
    }

    int  td = maxsize.w * maxsize.h;
    dim3 grid(util::DivUp(td, static_cast<int>(block.x)), 1, batch * channels);
    ThresholdPlanar<T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, thresh, maxval, channels, type);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// The var-shape twin of tensor_impl::AccumulateHistogram; returns the wrap because otsu_cal_varshape
// needs it again to recover each image's size.
inline cuda::ImageBatchVarShapeWrapNHWC<uchar> AccumulateHistogram(
    const nvcv::ImageBatchVarShapeDataStridedCuda &inData, int *histogram, cudaStream_t stream)
{
    int batch = inData.numImages();
    NVCV_CHECK_THROW(cudaMemsetAsync(histogram, 0, sizeof(int) * kHistogramBins * batch, stream));

    cuda::ImageBatchVarShapeWrapNHWC<uchar> wrap(inData, inData.uniqueFormat().numChannels());
    nvcv::Size2D                            maxsize = inData.maxSize();

    dim3 block(256);
    int  td = util::DivUp(maxsize.w, 16) * maxsize.h;
    dim3 grid(util::DivUp(td, 256), 1, batch);
    hist_kernel<<<grid, block, 0, stream>>>(wrap, histogram);

    return wrap;
}

inline void getThreshVal_Triangle(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                  const nvcv::TensorDataStridedCuda &threshold, int *histogram, cudaStream_t stream)
{
    AccumulateHistogram(inData, histogram, stream);

    cuda::Tensor1DWrap<double, int32_t> thresh(threshold);
    dim3                                block2(256);
    dim3                                grid2(1, 1, inData.numImages());
    triangle_cal<<<grid2, block2, 0, stream>>>(histogram, thresh);
}

inline void getThreshVal_Otsu(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                              const nvcv::TensorDataStridedCuda &threshold, int *histogram, cudaStream_t stream)
{
    cuda::ImageBatchVarShapeWrapNHWC<uchar> wrap = AccumulateHistogram(inData, histogram, stream);

    cuda::Tensor1DWrap<double, int32_t> thresh(threshold);
    dim3                                block2(256);
    dim3                                grid2(1, 1, inData.numImages());
    otsu_cal_varshape<<<grid2, block2, 0, stream>>>(histogram, thresh, wrap);
}

inline void RunThresholdDispatch(DataTypeCode code, const nvcv::ImageBatchVarShapeDataStridedCuda &input,
                                 const nvcv::ImageBatchVarShapeDataStridedCuda &output,
                                 const nvcv::TensorDataStridedCuda             &threshold,
                                 const nvcv::TensorDataStridedCuda &maxval, NVCVThresholdType type, cudaStream_t stream)
{
    DispatchByDataType(
        code, [&](auto elem) { thresholdDispatch<decltype(elem)>(input, output, threshold, maxval, type, stream); });
}

inline void RunThresholdDispatchPlanar(DataTypeCode code, const nvcv::ImageBatchVarShapeDataStridedCuda &input,
                                       const nvcv::ImageBatchVarShapeDataStridedCuda &output,
                                       const nvcv::TensorDataStridedCuda             &threshold,
                                       const nvcv::TensorDataStridedCuda &maxval, int channels, NVCVThresholdType type,
                                       cudaStream_t stream)
{
    DispatchByDataType(
        code, [&](auto elem)
        { thresholdDispatchPlanar<decltype(elem)>(input, output, threshold, maxval, channels, type, stream); });
}

} // namespace varshape_impl

} // namespace

namespace cvcuda::priv {

Threshold::Histogram::Histogram(uint32_t automaticThresh, int maxBatchSize)
{
    if (automaticThresh != 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(&data, sizeof(int) * kHistogramBins * maxBatchSize));
        allocated = true;
    }
}

Threshold::Histogram::~Histogram()
{
    if (allocated)
    {
        NVCV_CHECK_LOG(cudaFree(data));
    }
}

Threshold::Threshold(uint32_t type, int maxBatchSize)
    : m_type(type)
    , m_automaticThresh(type & ~(uint32_t)NVCV_THRESH_MASK)
    , m_maskedType(type & (uint32_t)NVCV_THRESH_MASK)
    , m_maxBatchSize(maxBatchSize)
    // Histogram::data is a raw device allocation bound to whichever device was current when it was
    // made, so PerDeviceResource gives each CUDA device its own.
    , m_histogram([automatic = m_automaticThresh, maxBatchSize](int)
                  { return std::make_unique<Histogram>(automatic, maxBatchSize); })
    , m_histogramVarShape([automatic = m_automaticThresh, maxBatchSize](int)
                          { return std::make_unique<Histogram>(automatic, maxBatchSize); })
{
    if (maxBatchSize < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBatchSize must be >= 0");
    }
}

void Threshold::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                           const nvcv::Tensor &thresh, const nvcv::Tensor &maxval) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Threshold::operator()[Tensor]");
    // Check order is frozen: dtype, then format. The var-shape overload below deliberately does the
    // opposite, because the two legacy infer() implementations did, and the negative tests assert
    // which error an input that is invalid in both ways reports.
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto threshData = thresh.exportData<nvcv::TensorDataStridedCuda>();
    if (threshData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "thresh must be cuda-accessible, pitch-linear tensor");
    }

    auto maxvalData = maxval.exportData<nvcv::TensorDataStridedCuda>();
    if (maxvalData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "maxval must be cuda-accessible, pitch-linear tensor");
    }

    Histogram &histogram = m_histogram.get();

    DataTypeCode inCode = ClassifyDataType(inData->dtype());
    if (!IsSupportedDataType(inCode))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Data Type %d", static_cast<int>(inCode));
    }

    DataTypeCode outCode = ClassifyDataType(outData->dtype());
    if (inCode != outCode)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DataType of input and output must be equal, but got %d and %d", static_cast<int>(inCode),
                              static_cast<int>(outCode));
    }

    const DataFormatCode inputFormat  = ClassifyDataFormat(inData->layout());
    const DataFormatCode outputFormat = ClassifyDataFormat(outData->layout());

    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", static_cast<int>(inputFormat),
                              static_cast<int>(outputFormat));
    }
    const bool isPlanar = IsPlanar(inputFormat);

    ValidateParamTensor(*threshData, "thresh");
    ValidateParamTensor(*maxvalData, "maxval");

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    NVCV_ASSERT(inAccess);

    const int channels = inAccess->numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    ValidateThresholdType(m_type, m_automaticThresh, m_maskedType);

    const int batch = inAccess->numSamples();
    if (m_automaticThresh != 0 && batch > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }

    // The histogram accumulation and threshold selection are single-channel kernels, so a planar
    // automatic request runs over flattened (N*C, H, W, 1) views of the two tensors.
    std::optional<std::pair<nvcv::TensorDataStridedCuda, nvcv::TensorDataStridedCuda>> planarViews;
    const nvcv::TensorDataStridedCuda                                                 *workInData   = &(*inData);
    const nvcv::TensorDataStridedCuda                                                 *workOutData  = &(*outData);
    int                                                                                workBatch    = batch;
    int                                                                                workChannels = channels;
    if (isPlanar && m_automaticThresh != 0)
    {
        planarViews = PlanarSingleChannelViews(*inData, *outData);
        NVCV_ASSERT(planarViews);
        workInData   = &planarViews->first;
        workOutData  = &planarViews->second;
        workBatch    = batch * channels;
        workChannels = 1;
    }

    // Tested as the explicit disjunction rather than m_automaticThresh != 0: m_automaticThresh is
    // type & ~NVCV_THRESH_MASK, so a bit above the two automatic flags leaves it non-zero while
    // selecting neither mode, and that input runs no histogram pass and none of these checks.
    const bool isOtsu     = m_automaticThresh == (uint32_t)NVCV_THRESH_OTSU;
    const bool isTriangle = m_automaticThresh == (uint32_t)NVCV_THRESH_TRIANGLE;
    if (isOtsu || isTriangle)
    {
        ValidateAutomaticModeInput(inCode, channels);
        if (inAccess->sampleStride() * batch > cuda::TypeTraits<int32_t>::max)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }
        if (isOtsu)
        {
            tensor_impl::getThreshVal_Otsu(*workInData, *threshData, histogram.data, inAccess->numRows(),
                                           inAccess->numCols(), workBatch, stream);
        }
        else
        {
            tensor_impl::getThreshVal_Triangle(*workInData, *threshData, histogram.data, inAccess->numRows(),
                                               inAccess->numCols(), workBatch, stream);
        }
    }

    const NVCVThresholdType thresholdType = NVCVThresholdType(m_maskedType);
    if (isPlanar && m_automaticThresh == 0)
    {
        tensor_impl::RunThresholdScalePlanar(inCode, *inData, *outData, *threshData, *maxvalData, batch,
                                             inAccess->numRows(), inAccess->numCols(), channels, thresholdType, stream);
        return;
    }

    tensor_impl::RunThresholdScale(inCode, *workInData, *workOutData, *threshData, *maxvalData, workBatch,
                                   inAccess->numRows(), inAccess->numCols(), workChannels, thresholdType, stream);
}

void Threshold::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                           const nvcv::Tensor &thresh, const nvcv::Tensor &maxval) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Threshold::operator()[ImageBatchVarShape]");
    // Check order is frozen: format, then dtype -- the mirror image of the tensor overload. See there.
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    auto threshData = thresh.exportData<nvcv::TensorDataStridedCuda>();
    if (threshData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "thresh must be cuda-accessible, pitch-linear tensor");
    }

    auto maxvalData = maxval.exportData<nvcv::TensorDataStridedCuda>();
    if (maxvalData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "maxval must be cuda-accessible, pitch-linear tensor");
    }

    Histogram &histogram = m_histogramVarShape.get();

    const DataFormatCode inputFormat  = ClassifyDataFormat(*inData);
    const DataFormatCode outputFormat = ClassifyDataFormat(*outData);
    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", static_cast<int>(inputFormat),
                              static_cast<int>(outputFormat));
    }
    const bool isPlanar = IsPlanar(inputFormat);

    DataTypeCode inCode = ClassifyDataType(inData->uniqueFormat());
    if (!IsSupportedDataType(inCode))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Data Type %d", static_cast<int>(inCode));
    }

    DataTypeCode outCode = ClassifyDataType(outData->uniqueFormat());
    if (inCode != outCode)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DataType of input and output must be equal, but got %d and %d", static_cast<int>(inCode),
                              static_cast<int>(outCode));
    }

    const int channels = inData->uniqueFormat().numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    ValidateParamTensor(*threshData, "thresh");
    ValidateParamTensor(*maxvalData, "maxval");

    ValidateThresholdType(m_type, m_automaticThresh, m_maskedType);

    if (m_automaticThresh != 0 && inData->numImages() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }
    if (isPlanar && static_cast<int64_t>(inData->numImages()) * channels > kMaxGridZ)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar Threshold requires numImages * channels <= 65535 (CUDA grid-z limit)");
    }

    // See the tensor overload: the disjunction, not m_automaticThresh != 0, is what legacy selected on.
    const bool isOtsu     = m_automaticThresh == (uint32_t)NVCV_THRESH_OTSU;
    const bool isTriangle = m_automaticThresh == (uint32_t)NVCV_THRESH_TRIANGLE;
    if (isOtsu || isTriangle)
    {
        ValidateAutomaticModeInput(inCode, channels);
        if (isOtsu)
        {
            varshape_impl::getThreshVal_Otsu(*inData, *threshData, histogram.data, stream);
        }
        else
        {
            varshape_impl::getThreshVal_Triangle(*inData, *threshData, histogram.data, stream);
        }
    }

    const NVCVThresholdType thresholdType = NVCVThresholdType(m_maskedType);
    if (isPlanar)
    {
        varshape_impl::RunThresholdDispatchPlanar(inCode, *inData, *outData, *threshData, *maxvalData, channels,
                                                  thresholdType, stream);
    }
    else
    {
        varshape_impl::RunThresholdDispatch(inCode, *inData, *outData, *threshData, *maxvalData, thresholdType, stream);
    }
}

} // namespace cvcuda::priv

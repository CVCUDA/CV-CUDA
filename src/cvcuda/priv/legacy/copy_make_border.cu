/* Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#define BLOCK 32

namespace nvcv::legacy::cuda_op {

namespace {

static bool IsPlanar(DataFormat format)
{
    return format == kNCHW || format == kCHW;
}

static nvcv::TensorDataStridedCuda PlanarChannelView(const nvcv::TensorDataStridedCuda              &data,
                                                     const nvcv::TensorDataAccessStridedImagePlanar &access, int plane)
{
    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(data.basePtr()) + plane * access.chStride();
    buf.strides[0] = access.sampleStride();
    buf.strides[1] = access.rowStride();
    buf.strides[2] = access.colStride();
    buf.strides[3] = access.colStride();

    return nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{access.numSamples(), access.numRows(), access.numCols(), 1}, "NHWC"},
        data.dtype(), buf
    };
}

static float4 PlanarBorderValue(const float4 &borderValue, int plane)
{
    const float value = plane == 0 ? borderValue.x
                      : plane == 1 ? borderValue.y
                      : plane == 2 ? borderValue.z
                                   : borderValue.w;
    return float4{value, value, value, value};
}

} // namespace

template<class SrcWrapper, class DstWrapper>
__global__ void copyMakeBorderKernel(SrcWrapper src, DstWrapper dst, int2 dstSize, int left, int top)
{
    int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);
    int3 srcCoord = {dstCoord.x - left, dstCoord.y - top, dstCoord.z};

    if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        dst[dstCoord] = src[srcCoord];
    }
}

// Vector pack type for T: uint3 (12B) for 3-element T, else uint4 (16B). NIX = elements/thread.
template<typename T>
using CMB_DPT = std::conditional_t<cuda::NumElements<T> == 3, uint3, uint4>;
template<typename T>
constexpr int CMB_NIX = sizeof(CMB_DPT<T>) / sizeof(T);
template<typename T>
constexpr uintptr_t CMB_MSK = (sizeof(CMB_DPT<T>) == sizeof(uint3) ? sizeof(uint) : sizeof(CMB_DPT<T>)) - 1;

// copy_make_border is a pure copy: the only per-pixel cost is the BorderWrap's bounds test + index
// clamp, which dominates (compute-bound) even though the interior is just a straight copy. This kernel
// processes CMB_NIX consecutive output x-elements per thread and takes a fast path for the INTERIOR
// (the [left,left+srcW) x [top,top+srcH) region, i.e. the bulk): a single aligned vector load+store
// straight from the raw source, skipping the BorderWrap entirely. Threads touching the border (or an
// unaligned/partial span) fall back to the per-element BorderWrap path. Bit-exact (a copy is exact;
// border pixels go through the same BorderWrap as before).
template<class SrcWrapper, class SrcRawWrapper, class DstWrapper, typename T>
__global__ void copyMakeBorderKernelVec(SrcWrapper src, SrcRawWrapper srcRaw, DstWrapper dst, int2 dstSize,
                                        int2 srcSize, int left, int top)
{
    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    const int z    = blockIdx.z;
    if (dstY >= dstSize.y)
        return;
    const int dstX0 = (blockIdx.x * blockDim.x + threadIdx.x) * CMB_NIX<T>;
    if (dstX0 >= dstSize.x)
        return;

    const int  srcY         = dstY - top;
    const int  srcX0        = dstX0 - left;
    const bool full         = (dstX0 + CMB_NIX<T> - 1 < dstSize.x);
    const bool interiorRow  = (srcY >= 0 && srcY < srcSize.y);
    const bool interiorSpan = (srcX0 >= 0 && srcX0 + CMB_NIX<T> - 1 < srcSize.x);

    if (full && interiorRow && interiorSpan)
    {
        const T *sp = &srcRaw[int3{srcX0, srcY, z}];
        T       *dp = &dst[int3{dstX0, dstY, z}];
        if ((reinterpret_cast<uintptr_t>(sp) & CMB_MSK<T>) == 0 && (reinterpret_cast<uintptr_t>(dp) & CMB_MSK<T>) == 0)
            *reinterpret_cast<CMB_DPT<T> *>(dp) = *reinterpret_cast<const CMB_DPT<T> *>(sp);
        else
#pragma unroll
            for (int i = 0; i < CMB_NIX<T>; ++i) dp[i] = sp[i];
    }
    else
    {
#pragma unroll
        for (int i = 0; i < CMB_NIX<T>; ++i)
        {
            const int dstX = dstX0 + i;
            if (dstX < dstSize.x)
                dst[int3{dstX, dstY, z}] = src[int3{dstX - left, srcY, z}];
        }
    }
}

template<typename T, NVCVBorderType B>
struct copyMakeBorderDispatcher
{
    static ErrorCode call(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                          const T &borderValue, const int left, const int top, cudaStream_t stream)
    {
        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
        NVCV_ASSERT(inAccess);

        int2 dstSize{outAccess->numCols(), outAccess->numRows()};

        dim3 blockSize(BLOCK, BLOCK / 4, 1);

        int64_t srcMaxStride = inAccess->sampleStride() * inAccess->numSamples();
        int64_t dstMaxStride = outAccess->sampleStride() * outAccess->numSamples();

        if (std::max(srcMaxStride, dstMaxStride) <= cuda::TypeTraits<int32_t>::max)
        {
            // BorderWrap for the border pixels; a raw TensorWrap for the interior vector fast path.
            auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, borderValue);
            auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

            // The uint3 (12B, 3-element interleaved) vector pack and the REPLICATE border both regress
            // on A100 (and give no win on H100), so keep the scalar copy for those; the vectorized
            // interior fast path is used only where it measurably helps (1-/4-element interleaved and
            // planar single-channel, which dispatch through the single-channel path).
            if constexpr (cuda::NumElements<T> == 3 || B == NVCV_BORDER_REPLICATE)
            {
                dim3 gridSize(divUp(dstSize.x, blockSize.x), divUp(dstSize.y, blockSize.y), outAccess->numSamples());
                copyMakeBorderKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, left, top);
            }
            else
            {
                int2 srcSize{inAccess->numCols(), inAccess->numRows()};
                dim3 gridSize(divUp(dstSize.x, blockSize.x * CMB_NIX<T>), divUp(dstSize.y, blockSize.y),
                              outAccess->numSamples());
                auto srcRaw = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);

                copyMakeBorderKernelVec<decltype(src), decltype(srcRaw), decltype(dst), T>
                    <<<gridSize, blockSize, 0, stream>>>(src, srcRaw, dst, dstSize, srcSize, left, top);
            }
        }
        else
        {
            LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
            return ErrorCode::INVALID_PARAMETER;
        }

        checkKernelErrors();
        return ErrorCode::SUCCESS;
    }
};

template<typename T> // uchar3 float3 uchar float
ErrorCode copyMakeBorder(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, const int top,
                         const int left, const NVCVBorderType border_type, const float4 &borderValue,
                         cudaStream_t stream)
{
    const T bvalue = cuda::DropCast<cuda::NumElements<T>>(cuda::StaticCast<cuda::BaseType<T>>(borderValue));

    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                const T &borderValue, const int left, const int top, cudaStream_t stream);

    static const func_t funcs[]
        = {copyMakeBorderDispatcher<T, NVCV_BORDER_CONSTANT>::call,
           copyMakeBorderDispatcher<T, NVCV_BORDER_REPLICATE>::call,
           copyMakeBorderDispatcher<T, NVCV_BORDER_REFLECT>::call, copyMakeBorderDispatcher<T, NVCV_BORDER_WRAP>::call,
           copyMakeBorderDispatcher<T, NVCV_BORDER_REFLECT101>::call};

    return funcs[border_type](inData, outData, bvalue, left, top, stream);
}

ErrorCode CopyMakeBorder::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                const int top, const int left, const NVCVBorderType border_type,
                                const float4 &borderValue, cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(input_format == kNHWC || input_format == kHWC || input_format == kNCHW || input_format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << input_format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = IsPlanar(input_format);

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << inData.dtype() << ") and output (" << outData.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType data_type = helpers::GetLegacyDataType(inData.dtype());

    const int channels = inAccess->numChannels();

    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!isPlanar && channels == 2 && data_type != kCV_8U)
    {
        LOG_ERROR("Invalid channel number " << channels << " for DataType " << data_type
                                            << ", 2 channels only supported for 8bit Unsigned");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(border_type == NVCVBorderType::NVCV_BORDER_CONSTANT || border_type == NVCVBorderType::NVCV_BORDER_REPLICATE
          || border_type == NVCVBorderType::NVCV_BORDER_REFLECT || border_type == NVCVBorderType::NVCV_BORDER_REFLECT101
          || border_type == NVCVBorderType::NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderType " << border_type);
        return ErrorCode::INVALID_PARAMETER;
    }

    const int rows     = inAccess->numRows();
    const int cols     = inAccess->numCols();
    const int out_rows = outAccess->numRows();
    const int out_cols = outAccess->numCols();

    if (!(top >= 0 && out_rows >= top + rows && left >= 0 && out_cols >= left + cols))
    {
        LOG_ERROR("Invalid border " << top << " " << out_rows - top - rows << " " << left << " "
                                    << out_cols - left - cols
                                    << ", top >= 0 && bottom >= 0 && left >= 0 && right >= 0, in resolution: " << rows
                                    << "x" << cols << ", out resolution: " << out_rows << "x" << out_cols);
        return ErrorCode::INVALID_PARAMETER;
    }

    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                const int top, const int left, const NVCVBorderType border_type,
                                const float4 &borderValue, cudaStream_t stream);

    // clang-format off
    static const func_t funcs[6][4] = {
        {copyMakeBorder<uchar1>      , copyMakeBorder<uchar2>       , copyMakeBorder<uchar3>      , copyMakeBorder<uchar4>      },
        {0 /*copyMakeBorder<char1>*/, 0 /*copyMakeBorder<char2>*/ , 0 /*copyMakeBorder<char3>*/, 0 /*copyMakeBorder<char4>*/},
        {copyMakeBorder<ushort1>     , 0 /*copyMakeBorder<ushort2>*/, copyMakeBorder<ushort3>     , copyMakeBorder<ushort4>     },
        {copyMakeBorder<short1>      , 0 /*copyMakeBorder<short2>*/, copyMakeBorder<short3>      , copyMakeBorder<short4>      },
        {0 /*copyMakeBorder<int, 1>*/  , 0 /*copyMakeBorder<int, 2>*/   , 0 /*copyMakeBorder<int, 3>*/  , 0 /*copyMakeBorder<int, 4>*/  },
        {copyMakeBorder<float1>      , 0 /*copyMakeBorder<float2>*/, copyMakeBorder<float3>      , copyMakeBorder<float4>      }
    };
    // clang-format on

    const func_t func = funcs[data_type][channels - 1];

    if (isPlanar)
    {
        const func_t planarFunc = funcs[data_type][0];
        NVCV_ASSERT(planarFunc != 0);

        for (int c = 0; c < channels; ++c)
        {
            auto      planeIn  = PlanarChannelView(inData, *inAccess, c);
            auto      planeOut = PlanarChannelView(outData, *outAccess, c);
            ErrorCode ec
                = planarFunc(planeIn, planeOut, top, left, border_type, PlanarBorderValue(borderValue, c), stream);
            if (ec != ErrorCode::SUCCESS)
            {
                return ec;
            }
        }
        return ErrorCode::SUCCESS;
    }

    NVCV_ASSERT(func != 0);

    return func(inData, outData, top, left, border_type, borderValue, stream);
}

} // namespace nvcv::legacy::cuda_op

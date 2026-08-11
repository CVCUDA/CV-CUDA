/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "warp_cubic.cuh"

#define BLOCK 32
#define PI    3.1415926535897932384626433832795

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

__global__ void compute_warpAffine(const int numImages, const cuda::Tensor1DWrap<double> angleDeg,
                                   const cuda::Tensor2DWrap<double> shift, float *d_aCoeffs)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    // Trig in FP64 for accuracy (one-shot, 1 thread per image; perf-irrelevant).
    // Stored as FP32 — the per-pixel rotate kernel reads these in FP32 to avoid
    // the 1/64-rate FP64 path on consumer GPUs.
    float *aCoeffs = (float *)((char *)d_aCoeffs + (sizeof(float) * 6) * index);

    double angle  = angleDeg[index];
    double xShift = *shift.ptr(index, 0);
    double yShift = *shift.ptr(index, 1);

    aCoeffs[0] = static_cast<float>(cos(angle * PI / 180));
    aCoeffs[1] = static_cast<float>(sin(angle * PI / 180));
    aCoeffs[2] = static_cast<float>(xShift);
    aCoeffs[3] = static_cast<float>(-sin(angle * PI / 180));
    aCoeffs[4] = static_cast<float>(cos(angle * PI / 180));
    aCoeffs[5] = static_cast<float>(yShift);
}

// Number of output rows each thread emits (Y-tiling factor), chosen per interpolation type and element
// size, matching the tensor kernel rationale: small-element NEAREST/LINEAR benefit from a wide Y-tile
// (amortized per-thread invariants + overlapped per-row gathers); CUBIC's 4x4 neighborhood and wide
// (>2-byte component) elements regress when tiled, so they stay at one row per thread (the original
// mapping). Y-tiling keeps the X dimension one thread per column so warp stores stay coalesced.
template<typename T, NVCVInterpolationType I>
constexpr int RotateRowsPerThread = (I == NVCV_INTERP_CUBIC || sizeof(cuda::BaseType<T>) > 2) ? 1 : 8;

// Bit-exact: src_x/src_y use the same FP multiply form and operand order as the original scalar kernel
// (the column term is the same left operand of the same add). With NIY == 1 the body collapses to the
// original one-pixel-per-thread mapping.
template<int NIY, class SrcWrapper, class DstWrapper>
__global__ void rotate(SrcWrapper src, DstWrapper dst, const float *d_aCoeffs_)
{
    const int dst_x     = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y0    = (blockDim.y * blockIdx.y + threadIdx.y) * NIY;
    const int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;

    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    if (dst_x >= dstWidth)
        return;

    const float *d_aCoeffs   = (const float *)((char *)d_aCoeffs_ + (sizeof(float) * 6) * batch_idx);
    const float  c0          = d_aCoeffs[0];
    const float  c1          = d_aCoeffs[1];
    const float  c3          = d_aCoeffs[3];
    const float  c4          = d_aCoeffs[4];
    const float  c5          = d_aCoeffs[5];
    const float  dst_x_shift = static_cast<float>(dst_x) - d_aCoeffs[2];
    const float  src_x_col   = dst_x_shift * c0;
    const float  src_y_col   = dst_x_shift * (-c3);

    const int width  = src.borderWrap().imageBatchWrap().width(batch_idx);
    const int height = src.borderWrap().imageBatchWrap().height(batch_idx);

#pragma unroll
    for (int i = 0; i < NIY; ++i)
    {
        const int dst_y = dst_y0 + i;
        if (dst_y >= dstHeight)
            break;

        const float dst_y_shift = static_cast<float>(dst_y) - c5;
        const float src_x       = src_x_col + dst_y_shift * (-c1);
        const float src_y       = src_y_col + dst_y_shift * c4;

        if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
        {
            const int3 dstCoord{dst_x, dst_y, batch_idx};

            using SrcValueT = std::remove_cv_t<typename SrcWrapper::ValueType>;
            constexpr bool kFastCubic
                = SrcWrapper::kInterpolationType == NVCV_INTERP_CUBIC && kCubicFastSampler<SrcValueT>;
            if constexpr (kFastCubic)
            {
                dst[dstCoord] = CubicSampleVarShape<SrcWrapper, true>(src, batch_idx, float2{src_x, src_y});
            }
            else
            {
                const float3 srcCoord{src_x, src_y, static_cast<float>(batch_idx)};
                dst[dstCoord] = src[srcCoord];
            }
        }
    }
}

template<typename T, NVCVInterpolationType I>
void rotate(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out, float *d_aCoeffs,
            cudaStream_t stream)
{
    Size2D outMaxSize = out.maxSize();

    constexpr int kNIY = RotateRowsPerThread<T, I>;

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    // Each thread emits kNIY output rows, so the grid covers ceil(maxH / (blockY*kNIY)).
    dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y * kNIY), in.numImages());

    cuda::InterpolationVarShapeWrap<const T, NVCV_BORDER_REPLICATE, I> src(in);
    cuda::ImageBatchVarShapeWrap<T>                                    dst(out);

    rotate<kNIY><<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs);
    checkKernelErrors();
}

template<typename T> // uchar3 float3 uchar1 float3
void rotate(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out, float *d_aCoeffs,
            const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        rotate<T, NVCV_INTERP_NEAREST>(in, out, d_aCoeffs, stream);
        break;

    case NVCV_INTERP_LINEAR:
        rotate<T, NVCV_INTERP_LINEAR>(in, out, d_aCoeffs, stream);
        break;

    case NVCV_INTERP_CUBIC:
        rotate<T, NVCV_INTERP_CUBIC>(in, out, d_aCoeffs, stream);
        break;

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation type");
    }
}

// Planar (NCHW/CHW) rotate. Rotate maps each output pixel to a source pixel with the same affine
// coefficients regardless of the channel, so the (src_x, src_y) mapping is computed once per output
// pixel and reused across every channel plane (grid.z runs over images, the kernel loops the planes).
// This avoids the per-plane redundant coordinate math of one thread per (image, plane). Each plane is
// sampled through the same InterpolationVarShapeWrap the interleaved kernel uses -- its operator[]
// takes a 4D {x, y, plane, sample} coordinate -- so each plane's result is bit-exact with the
// interleaved single-channel rotate and the interleaved codegen is left untouched.
template<int NIY, class SrcWrapper, class DstWrapper>
__global__ void rotate_planar(SrcWrapper src, DstWrapper dst, const float *d_aCoeffs_, int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y0    = (blockIdx.y * blockDim.y + threadIdx.y) * NIY;
    const int batch_idx = get_batch_idx();

    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    if (dst_x >= dstWidth)
        return;

    const float *d_aCoeffs   = (const float *)((char *)d_aCoeffs_ + (sizeof(float) * 6) * batch_idx);
    const float  c0          = d_aCoeffs[0];
    const float  c1          = d_aCoeffs[1];
    const float  c3          = d_aCoeffs[3];
    const float  c4          = d_aCoeffs[4];
    const float  c5          = d_aCoeffs[5];
    const float  dst_x_shift = static_cast<float>(dst_x) - d_aCoeffs[2];
    const float  src_x_col   = dst_x_shift * c0;
    const float  src_y_col   = dst_x_shift * (-c3);

    const int width  = src.borderWrap().imageBatchWrap().width(batch_idx);
    const int height = src.borderWrap().imageBatchWrap().height(batch_idx);

#pragma unroll
    for (int i = 0; i < NIY; ++i)
    {
        const int dst_y = dst_y0 + i;
        if (dst_y >= dstHeight)
            break;

        const float dst_y_shift = static_cast<float>(dst_y) - c5;
        const float src_x       = src_x_col + dst_y_shift * (-c1);
        const float src_y       = src_y_col + dst_y_shift * c4;

        if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
        {
            for (int plane = 0; plane < channels; ++plane)
            {
                const float4 srcCoord{src_x, src_y, static_cast<float>(plane), static_cast<float>(batch_idx)};
                *dst.ptr(batch_idx, plane, dst_y, dst_x) = src[srcCoord];
            }
        }
    }
}

// Fused planar CUBIC rotate: shares the coordinate transform, cubic weights, and border
// resolution across all NP channel planes; the generic kernel above re-resolves them per plane
// through the interpolation wrap.
template<int NP, class SrcWrapper, class DstWrapper>
__global__ void rotate_planar_fused(SrcWrapper src, DstWrapper dst, const float *d_aCoeffs_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    const float *d_aCoeffs   = (const float *)((char *)d_aCoeffs_ + (sizeof(float) * 6) * batch_idx);
    const float  dst_x_shift = static_cast<float>(dst_x) - d_aCoeffs[2];
    const float  dst_y_shift = static_cast<float>(dst_y) - d_aCoeffs[5];
    const float2 coord{dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]),
                       dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]};

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    if (coord.x > -0.5f && coord.x < static_cast<float>(width) && coord.y > -0.5f
        && coord.y < static_cast<float>(height))
    {
        using BT = std::remove_cv_t<typename SrcWrapper::ValueType>;
        CubicWarpPlanes<BT, NVCV_BORDER_REPLICATE, NP, int32_t>(
            [&](int p, int yy) { return src.ptr(batch_idx, p, yy, 0); },
            [&](int p, BT v) { *dst.ptr(batch_idx, p, dst_y, dst_x) = v; }, float4{0.f, 0.f, 0.f, 0.f},
            int2{width, height}, coord);
    }
}

template<typename T, NVCVInterpolationType I>
void rotate_planar(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                   float *d_aCoeffs, const int channels, cudaStream_t stream)
{
    Size2D outMaxSize = out.maxSize();

    constexpr int kNIY = RotateRowsPerThread<T, I>;

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    // Each thread emits kNIY output rows, so the grid covers ceil(maxH / (blockY*kNIY)).
    dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y * kNIY), in.numImages());

    cuda::ImageBatchVarShapeWrap<T> dst(out);

    if constexpr (I == NVCV_INTERP_CUBIC)
    {
        cuda::ImageBatchVarShapeWrap<const T> src(in);

        switch (channels)
        {
        case 1:
            rotate_planar_fused<1><<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs);
            checkKernelErrors();
            return;
        case 3:
            rotate_planar_fused<3><<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs);
            checkKernelErrors();
            return;
        case 4:
            rotate_planar_fused<4><<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs);
            checkKernelErrors();
            return;
        default:
            break;
        }
    }

    cuda::InterpolationVarShapeWrap<const T, NVCV_BORDER_REPLICATE, I> src(in);

    rotate_planar<kNIY><<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs, channels);
    checkKernelErrors();
}

template<typename T>
void rotate_planar(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                   float *d_aCoeffs, const int channels, const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        rotate_planar<T, NVCV_INTERP_NEAREST>(in, out, d_aCoeffs, channels, stream);
        break;

    case NVCV_INTERP_LINEAR:
        rotate_planar<T, NVCV_INTERP_LINEAR>(in, out, d_aCoeffs, channels, stream);
        break;

    case NVCV_INTERP_CUBIC:
        rotate_planar<T, NVCV_INTERP_CUBIC>(in, out, d_aCoeffs, channels, stream);
        break;

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation type");
    }
}

RotateVarShape::RotateVarShape(const int maxBatchSize)
    : CudaBaseOp()
    , d_aCoeffs(nullptr)
    , m_maxBatchSize(maxBatchSize)
{
    if (m_maxBatchSize > 0)
    {
        size_t      bufferSize = sizeof(float) * 6 * m_maxBatchSize;
        cudaError_t err        = cudaMalloc(&d_aCoeffs, bufferSize);
        if (err != cudaSuccess)
        {
            LOG_ERROR("CUDA memory allocation error of size: " << bufferSize);
            throw LegacyCudaAllocationError("CUDA memory allocation error!");
        }
    }
}

RotateVarShape::~RotateVarShape()
{
    if (d_aCoeffs != nullptr)
    {
        cudaError_t err = cudaFree(d_aCoeffs);
        if (err != cudaSuccess)
        {
            LOG_ERROR("CUDA memory free error, possible memory leak!");
        }
    }
    d_aCoeffs = nullptr;
}

ErrorCode RotateVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &angleDeg,
                                const TensorDataStridedCuda &shift, const NVCVInterpolationType interpolation,
                                cudaStream_t stream)
{
    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (m_maxBatchSize <= 0)
    {
        LOG_ERROR("Operator rotate var shape is not initialized properly, maxVarShapeBatchSize: " << m_maxBatchSize);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (m_maxBatchSize < inData.numImages())
    {
        LOG_ERROR("Invalid number of images, it should not exceed " << m_maxBatchSize);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    int channels = inData.uniqueFormat().numChannels();

    // Planar 2-channel layout is rejected: there is no defined 2-plane planar format, and it matches
    // the Resize and Normalize operators.
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // The planar path launches grid.z over images and loops the channel planes inside the kernel, so
    // numImages must fit CUDA's 65535 grid-z limit. Compute in 64-bit to avoid overflow.
    if (isPlanar && static_cast<int64_t>(outData.numImages()) > 65535)
    {
        LOG_ERROR("Planar rotate requires numImages <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType data_type          = helpers::GetLegacyDataType(inData.uniqueFormat());
    DataType angleDec_data_type = helpers::GetLegacyDataType(angleDeg.dtype());
    DataType shift_data_type    = helpers::GetLegacyDataType(shift.dtype());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (angleDec_data_type != kCV_64F)
    {
        LOG_ERROR("Invalid angleDeg DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (shift_data_type != kCV_64F)
    {
        LOG_ERROR("Invalid shift DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(interpolation == NVCV_INTERP_LINEAR || interpolation == NVCV_INTERP_NEAREST
          || interpolation == NVCV_INTERP_CUBIC))
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }

    cuda::Tensor1DWrap<double> angleDecPtr(angleDeg);
    cuda::Tensor2DWrap<double> shiftPtr(shift);

    compute_warpAffine<<<1, inData.numImages(), 0, stream>>>(inData.numImages(), angleDecPtr, shiftPtr, d_aCoeffs);
    checkKernelErrors();

    if (isPlanar)
    {
        // Planar dispatch indexes by dtype only: each channel is rotated as a separate single-channel
        // plane, so one scalar specialization per dtype covers any channel count.
        typedef void (*planar_func_t)(
            const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out, float *d_aCoeffs,
            const int channels, const NVCVInterpolationType interpolation, cudaStream_t stream);

        static const planar_func_t planar_funcs[6] = {
            rotate_planar<uchar>, 0 /*schar*/, rotate_planar<ushort>,
            rotate_planar<short>, 0 /*int*/,   rotate_planar<float>,
        };

        const planar_func_t planar_func = planar_funcs[data_type];
        NVCV_ASSERT(planar_func != 0);

        planar_func(inData, outData, d_aCoeffs, channels, interpolation, stream);
        return SUCCESS;
    }

    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                           float *d_aCoeffs, const NVCVInterpolationType interpolation, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      rotate<uchar>,  0 /*rotate<uchar2>*/,      rotate<uchar3>,      rotate<uchar4>},
        {0 /*rotate<schar>*/,   0 /*rotate<char2>*/, 0 /*rotate<char3>*/, 0 /*rotate<char4>*/},
        {     rotate<ushort>, 0 /*rotate<ushort2>*/,     rotate<ushort3>,     rotate<ushort4>},
        {      rotate<short>,  0 /*rotate<short2>*/,      rotate<short3>,      rotate<short4>},
        {  0 /*rotate<int>*/,    0 /*rotate<int2>*/,  0 /*rotate<int3>*/,  0 /*rotate<int4>*/},
        {      rotate<float>,  0 /*rotate<float2>*/,      rotate<float3>,      rotate<float4>}
    };

    const func_t func = funcs[data_type][channels - 1];

    func(inData, outData, d_aCoeffs, interpolation, stream);
    assert(func != 0);
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

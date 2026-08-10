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

namespace nvcv::legacy::cuda_op {

__global__ void compute_warpAffine(const double angle, const double xShift, const double yShift, float *aCoeffs)
{
    // Trig in FP64 for accuracy (one-shot, 1 thread; perf-irrelevant).
    // Stored as FP32 — the per-pixel rotate kernel reads these in FP32 to avoid
    // the 1/64-rate FP64 path on consumer GPUs.
    aCoeffs[0] = static_cast<float>(cos(angle * PI / 180));
    aCoeffs[1] = static_cast<float>(sin(angle * PI / 180));
    aCoeffs[2] = static_cast<float>(xShift);
    aCoeffs[3] = static_cast<float>(-sin(angle * PI / 180));
    aCoeffs[4] = static_cast<float>(cos(angle * PI / 180));
    aCoeffs[5] = static_cast<float>(yShift);
}

// Number of output rows each thread emits (Y-tiling factor), chosen per interpolation type and element
// size. The rotate kernel is latency/issue-bound on consumer Ampere (one interpolated gather per pixel
// through L1/TEX, low SOL). For small-element NEAREST/LINEAR the per-pixel working set is tiny, so a wide
// Y-tile amortizes the per-thread invariants (six affine coefficients, the column term, the source-bounds
// load) and overlaps several independent per-row gathers to hide latency -- up to ~2x on uint8 NEAREST,
// ~1.5x on uint8 LINEAR. Two cases must stay at one row per thread (the original one-pixel-per-thread
// mapping) to avoid regressing: CUBIC touches a 4x4 neighborhood per pixel and tiling it thrashes L1
// (measured up to ~5x slower on float4); and wide (>2-byte component) elements such as float move enough
// bytes per pixel that tiling also costs more than it saves (measured ~1.25x slower on float NEAREST).
// Y-tiling (rather than X-tiling) keeps the X dimension one thread per column so warp stores stay
// coalesced.
template<typename T, NVCVInterpolationType I>
constexpr int RotateRowsPerThread = (I == NVCV_INTERP_CUBIC || sizeof(cuda::BaseType<T>) > 2) ? 1 : 8;

// Bit-exactness: src_x/src_y are recomputed per pixel with the same FP multiply form and operand order as
// the original scalar kernel (the column term is the same left operand of the same add), not accumulated
// incrementally, so the rounding is identical to one-pixel-per-thread. With NIY == 1 the body collapses
// to exactly the original mapping.
template<int NIY, class SrcWrapper, class DstWrapper>
__global__ void rotate(SrcWrapper src, DstWrapper dst, int2 dstSize, const float *d_aCoeffs)
{
    const int dst_x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y0 = (blockIdx.y * blockDim.y + threadIdx.y) * NIY;
    const int dst_z  = blockIdx.z * blockDim.z + threadIdx.z;

    if (dst_x >= dstSize.x)
    {
        return;
    }

    const float c0 = d_aCoeffs[0];
    const float c1 = d_aCoeffs[1];
    const float c2 = d_aCoeffs[2];
    const float c3 = d_aCoeffs[3];
    const float c4 = d_aCoeffs[4];
    const float c5 = d_aCoeffs[5];

    const float dst_x_shift = static_cast<float>(dst_x) - c2;
    const float src_x_col   = dst_x_shift * c0;
    const float src_y_col   = dst_x_shift * (-c3);

    const long2 srcSize{src.borderWrap().tensorShape()[1], src.borderWrap().tensorShape()[0]};

#pragma unroll
    for (int i = 0; i < NIY; ++i)
    {
        const int dst_y = dst_y0 + i;
        if (dst_y >= dstSize.y)
        {
            break;
        }

        const float  dst_y_shift = static_cast<float>(dst_y) - c5;
        const float3 srcCoord{src_x_col + dst_y_shift * (-c1), src_y_col + dst_y_shift * c4, static_cast<float>(dst_z)};

        if (srcCoord.x > -0.5 && srcCoord.x < srcSize.x && srcCoord.y > -0.5 && srcCoord.y < srcSize.y)
        {
            const int3 dstCoord{dst_x, dst_y, dst_z};

            using SrcValueT = std::remove_cv_t<typename SrcWrapper::ValueType>;
            constexpr bool kFastCubic
                = SrcWrapper::kInterpolationType == NVCV_INTERP_CUBIC && kCubicFastSampler<SrcValueT>;
            if constexpr (kFastCubic)
            {
                dst[dstCoord] = CubicSampleTensor<SrcWrapper, true>(
                    src, dst_z, float2{srcCoord.x, srcCoord.y},
                    int2{static_cast<int>(srcSize.x), static_cast<int>(srcSize.y)});
            }
            else
            {
                dst[dstCoord] = src[srcCoord];
            }
        }
    }
}

// Fused planar CUBIC rotate: one thread maps its output pixel once and gathers all NP channel
// planes, sharing the coordinate transform, cubic weights, and border resolution that the
// flattened per-plane launch recomputes NP times. Wraps address the flattened (N*C, H, W) view;
// plane p of sample z sits at flattened sample z*NP + p.
template<int NP, class SrcWrapper, class DstWrapper>
__global__ void rotate_planar_fused(SrcWrapper src, DstWrapper dst, int2 dstSize, int2 srcSize, const float *d_aCoeffs)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int dst_z = blockIdx.z;

    if (dst_x >= dstSize.x || dst_y >= dstSize.y)
    {
        return;
    }

    const float  dst_x_shift = static_cast<float>(dst_x) - d_aCoeffs[2];
    const float  dst_y_shift = static_cast<float>(dst_y) - d_aCoeffs[5];
    const float2 coord{dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]),
                       dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]};

    if (coord.x > -0.5f && coord.x < static_cast<float>(srcSize.x) && coord.y > -0.5f
        && coord.y < static_cast<float>(srcSize.y))
    {
        using BT = std::remove_cv_t<typename SrcWrapper::ValueType>;
        CubicWarpPlanes<BT, NVCV_BORDER_REPLICATE, NP, int32_t>(
            [&](int p, int yy) { return src.ptr(dst_z * NP + p, yy); },
            [&](int p, BT v) { *(dst.ptr(dst_z * NP + p, dst_y) + dst_x) = v; }, float4{0.f, 0.f, 0.f, 0.f}, srcSize,
            coord);
    }
}

template<typename T, NVCVInterpolationType I>
ErrorCode rotate(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, float *d_aCoeffs,
                 const double angleDeg, const double2 shift, cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
    const int  batchSize{static_cast<int>(outAccess->numSamples())};

    compute_warpAffine<<<1, 1, 0, stream>>>(angleDeg, shift.x, shift.y, d_aCoeffs);
    checkKernelErrors();

    constexpr int kNIY = RotateRowsPerThread<T, I>;

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    // Each thread emits kNIY output rows, so the grid covers ceil(dstSize.y / (blockY*kNIY)).
    dim3 gridSize(divUp(dstSize.x, blockSize.x), divUp(dstSize.y, blockSize.y * kNIY), batchSize);

    int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateInterpolationWrapNHW<const T, NVCV_BORDER_REPLICATE, I, int32_t>(inData);
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

        rotate<kNIY><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, d_aCoeffs);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
    return ErrorCode::SUCCESS;
}

template<typename T> // uchar3 float3 uchar1 float3
ErrorCode rotate(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, float *d_aCoeffs,
                 const double angleDeg, const double2 shift, const NVCVInterpolationType interpolation,
                 cudaStream_t stream)
{
    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        return rotate<T, NVCV_INTERP_NEAREST>(inData, outData, d_aCoeffs, angleDeg, shift, stream);

    case NVCV_INTERP_LINEAR:
        return rotate<T, NVCV_INTERP_LINEAR>(inData, outData, d_aCoeffs, angleDeg, shift, stream);

    case NVCV_INTERP_CUBIC:
        return rotate<T, NVCV_INTERP_CUBIC>(inData, outData, d_aCoeffs, angleDeg, shift, stream);

    default:
        LOG_ERROR("Invalid rotate interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }
}

// Build a single-channel (N*C, H, W, 1) NHWC view of a packed planar (NCHW/CHW) tensor. Rotate maps
// each output pixel to a source pixel with the same per-sample affine coefficients regardless of the
// channel, so plane (n, c) is just a single-channel image; viewing the N*C planes as flat samples lets
// the existing interleaved single-channel rotate kernel handle planar data unchanged (bit-exact with
// the equivalent NHWC single-channel rotate). Plane (n, c) is placed at sample index n*C + c, byte
// offset (n*C + c)*chStride, which matches the real layout only when the channel planes are tightly
// packed across samples (sampleStride == C*chStride); the caller guards that. A single sample
// (N == 1) always satisfies it. The scalar angle/shift apply to every plane, so the single set of
// d_aCoeffs computed by compute_warpAffine drives all N*C flattened samples.
static nvcv::TensorDataStridedCuda PlanarAsSingleChannelView(const nvcv::TensorDataStridedCuda              &data,
                                                             const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    const int64_t numSamples  = access.numSamples();
    const int64_t numChannels = access.numChannels();

    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(data.basePtr());
    buf.strides[0] = access.chStride();  // N*C flattened planes
    buf.strides[1] = access.rowStride(); // H
    buf.strides[2] = access.colStride(); // W
    buf.strides[3] = access.colStride(); // C == 1
    return nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{numSamples * numChannels, access.numRows(), access.numCols(), 1}, "NHWC"},
        data.dtype(),
        buf
    };
}

template<typename BT>
ErrorCode rotatePlanarFusedCubic(const TensorDataStridedCuda &inView, const TensorDataStridedCuda &outView,
                                 int numPlanes, int numSamples, float *d_aCoeffs, const double angleDeg,
                                 const double2 shift, cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outView);
    NVCV_ASSERT(outAccess);
    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inView);
    NVCV_ASSERT(inAccess);

    const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
    const int2 srcSize{inAccess->numCols(), inAccess->numRows()};

    const int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    const int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    compute_warpAffine<<<1, 1, 0, stream>>>(angleDeg, shift.x, shift.y, d_aCoeffs);
    checkKernelErrors();

    auto src = cuda::CreateTensorWrapNHW<const BT, int32_t>(inView);
    auto dst = cuda::CreateTensorWrapNHW<BT, int32_t>(outView);

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(dstSize.x, blockSize.x), divUp(dstSize.y, blockSize.y), numSamples);

    switch (numPlanes)
    {
    case 1:
        rotate_planar_fused<1><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, srcSize, d_aCoeffs);
        break;
    case 3:
        rotate_planar_fused<3><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, srcSize, d_aCoeffs);
        break;
    case 4:
        rotate_planar_fused<4><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, srcSize, d_aCoeffs);
        break;
    default:
        LOG_ERROR("Invalid planar channel number " << numPlanes);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    checkKernelErrors();
    return ErrorCode::SUCCESS;
}

static ErrorCode rotatePlanarFusedCubicCaller(const TensorDataStridedCuda &inView, const TensorDataStridedCuda &outView,
                                              int dataType, int numPlanes, int numSamples, float *d_aCoeffs,
                                              const double angleDeg, const double2 shift, cudaStream_t stream)
{
    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inView, const TensorDataStridedCuda &outView,
                                int numPlanes, int numSamples, float *d_aCoeffs, const double angleDeg,
                                const double2 shift, cudaStream_t stream);

    static const func_t funcs[6] = {rotatePlanarFusedCubic<uchar>, 0, rotatePlanarFusedCubic<ushort>,
                                    rotatePlanarFusedCubic<short>, 0, rotatePlanarFusedCubic<float>};

    const func_t func = funcs[dataType];
    NVCV_ASSERT(func != 0);
    return func(inView, outView, numPlanes, numSamples, d_aCoeffs, angleDeg, shift, stream);
}

Rotate::Rotate(DataShape max_input_shape, DataShape max_output_shape)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , d_aCoeffs(nullptr)
{
    size_t      bufferSize = calBufferSize(max_input_shape, max_output_shape, DataType::kCV_8U /*not in use*/);
    cudaError_t err        = cudaMalloc(&d_aCoeffs, bufferSize);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA memory allocation error of size: " << bufferSize);
        throw LegacyCudaAllocationError("CUDA memory allocation error!");
    }
}

Rotate::~Rotate()
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

size_t Rotate::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 6 * sizeof(float);
}

ErrorCode Rotate::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                        const double angleDeg, const double2 shift, const NVCVInterpolationType interpolation,
                        cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

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

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    DataType  data_type   = helpers::GetLegacyDataType(inData.dtype());
    DataShape input_shape = helpers::GetLegacyDataShape(inAccess->infoShape());

    int channels = input_shape.C;

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(interpolation == NVCV_INTERP_LINEAR || interpolation == NVCV_INTERP_NEAREST
          || interpolation == NVCV_INTERP_CUBIC))
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }

    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                float *d_aCoeffs, const double angleDeg, const double2 shift,
                                const NVCVInterpolationType interpolation, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      rotate<uchar>,  0 /*rotate<uchar2>*/,      rotate<uchar3>,      rotate<uchar4>},
        {0 /*rotate<schar>*/,   0 /*rotate<char2>*/, 0 /*rotate<char3>*/, 0 /*rotate<char4>*/},
        {     rotate<ushort>, 0 /*rotate<ushort2>*/,     rotate<ushort3>,     rotate<ushort4>},
        {      rotate<short>,  0 /*rotate<short2>*/,      rotate<short3>,      rotate<short4>},
        {  0 /*rotate<int>*/,    0 /*rotate<int2>*/,  0 /*rotate<int3>*/,  0 /*rotate<int4>*/},
        {      rotate<float>,  0 /*rotate<float2>*/,      rotate<float3>,      rotate<float4>}
    };

    if (isPlanar)
    {
        // View each of the N*C channel planes as a single-channel sample and reuse the interleaved
        // single-channel rotate kernel (funcs column 0). Channels are independent in rotate, so this
        // is bit-exact with rotating the equivalent NHWC single-channel data. The flattened plane count
        // becomes the kernel's grid.z, capped at CUDA's 65535 limit; compute it in 64-bit to avoid
        // overflow. The flattened view is only valid when the planes are tightly packed across samples.
        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        const int64_t numSamples = inAccess->numSamples();
        if (numSamples > 1
            && (inAccess->sampleStride() != static_cast<int64_t>(channels) * inAccess->chStride()
                || outAccess->sampleStride() != static_cast<int64_t>(channels) * outAccess->chStride()))
        {
            LOG_ERROR("Planar rotate of a batched tensor requires tightly packed channel planes");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        if (numSamples * channels > 65535)
        {
            LOG_ERROR("Planar rotate requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        auto srcView = PlanarAsSingleChannelView(inData, *inAccess);
        auto dstView = PlanarAsSingleChannelView(outData, *outAccess);

        if (interpolation == NVCV_INTERP_CUBIC)
        {
            return rotatePlanarFusedCubicCaller(srcView, dstView, data_type, channels, static_cast<int>(numSamples),
                                                d_aCoeffs, angleDeg, shift, stream);
        }

        return funcs[data_type][0](srcView, dstView, d_aCoeffs, angleDeg, shift, interpolation, stream);
    }

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    return func(inData, outData, d_aCoeffs, angleDeg, shift, interpolation, stream);
}

} // namespace nvcv::legacy::cuda_op

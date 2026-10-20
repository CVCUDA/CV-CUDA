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
#include "OpCustomCrop.hpp"
#include "PlanarTensorView.hpp"
#include "TensorAccessValidation.hpp"

#include "ElementStreamCommon.cuh"

#include <cuda_runtime_api.h>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;
namespace es   = cvcuda::priv::elem_stream;

namespace {

constexpr int kBitsPerByte = 8;

// Independent 16-byte transfers issued per thread.
constexpr int kCropCarriersPerThread = 4;

// Tensors whose extent exceeds this type's range are rejected before the launch; widening it here
// without also relaxing that check, or vice versa, silently decouples the two.
using StrideType = int32_t;

template<int NRY, class SrcWrapper, class DstWrapper>
__global__ void CustomCropKernel(const SrcWrapper src, DstWrapper dst, int startX, int startY, int width, int height)
{
    const int x        = blockIdx.x * blockDim.x + threadIdx.x;
    const int y        = (blockIdx.y * blockDim.y + threadIdx.y) * NRY;
    const int batchIdx = blockIdx.z;
    if (x >= width)
        return;

#pragma unroll
    for (int i = 0; i < NRY; ++i)
    {
        if (y + i < height)
        {
            *dst.ptr(batchIdx, y + i, x) = *src.ptr(batchIdx, y + i + startY, x + startX);
        }
    }
}

template<typename T>
inline void LaunchCustomCrop(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                             const nvcv::TensorDataStridedCuda &outData, const NVCVRectI &roi, int numSamples)
{
    // Carried over verbatim from the legacy kernel. Output is identical for any launch geometry, so
    // normalising these to one shape looks like a free cleanup; it is a performance change and
    // needs a benchmark.
    constexpr int blockWidth    = sizeof(T) <= 3 ? 32 : 16;
    constexpr int rowsPerThread = sizeof(T) == 3 ? 2 : 1;
    constexpr int blockHeight   = 256 / blockWidth;

    const dim3 block(blockWidth, blockHeight);
    const dim3 grid(util::DivUp(roi.width, blockWidth), util::DivUp(roi.height, blockHeight * rowsPerThread),
                    numSamples);

    auto src = cuda::CreateTensorWrapNHW<const T, StrideType>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, StrideType>(outData);

    CustomCropKernel<rowsPerThread><<<grid, block, 0, stream>>>(src, dst, roi.x, roi.y, roi.width, roi.height);

    NVCV_CHECK_THROW(cudaGetLastError());
}

// BaseT is a width and alignment carrier, not a semantic type. MakeType<double, 4> resolves to the
// 16-byte-aligned double4 variant, which the eight-byte four-channel element needs; a hand-written
// double4 would be 8-byte aligned.
template<typename BaseT>
inline void RunChannelSwitch(int channels, cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                             const nvcv::TensorDataStridedCuda &outData, const NVCVRectI &roi, int numSamples)
{
    switch (channels)
    {
    case 1:
        LaunchCustomCrop<cuda::MakeType<BaseT, 1>>(stream, inData, outData, roi, numSamples);
        break;
    case 2:
        LaunchCustomCrop<cuda::MakeType<BaseT, 2>>(stream, inData, outData, roi, numSamples);
        break;
    case 3:
        LaunchCustomCrop<cuda::MakeType<BaseT, 3>>(stream, inData, outData, roi, numSamples);
        break;
    case 4:
        LaunchCustomCrop<cuda::MakeType<BaseT, 4>>(stream, inData, outData, roi, numSamples);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number ch = %d", channels);
    }
}

// Deliberately not an identity test against particular data types: a packed 2U8/3U8/4U8 qualifies
// through its per-channel width, and the channel count is validated separately. Tightening this to
// compare against named types would silently drop the packed forms. The returned width also selects
// the element type in the dispatch below, so the gate and that dispatch cannot drift apart.
inline int SupportedBytesPerChannel(const nvcv::DataType &dtype)
{
    const auto bitsPerChannel = dtype.bitsPerChannel();

    // A non-uniform packing such as X32_Y24b8 must not qualify through its first channel alone.
    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bitsPerChannel[i] != bitsPerChannel[0])
        {
            return 0;
        }
    }

    // Cropping moves bytes and never reads a value, so the data kind only has to name a real
    // numeric type -- the width is what selects the copy. Every 8-, 16-, 32- and 64-bit channel is
    // handled by the four widths dispatched below.
    bool supported = false;
    switch (dtype.dataKind())
    {
    case nvcv::DataKind::UNSIGNED:
    case nvcv::DataKind::SIGNED:
    case nvcv::DataKind::FLOAT:
        supported
            = bitsPerChannel[0] == 8 || bitsPerChannel[0] == 16 || bitsPerChannel[0] == 32 || bitsPerChannel[0] == 64;
        break;
    default:
        break;
    }

    // An 8-bit float would land on the 1-byte copy, which is still a correct move, but no such
    // type reaches here today; leaving the width test kind-independent keeps the rule one line.
    return supported ? bitsPerChannel[0] / kBitsPerByte : 0;
}

// The gate is deliberately narrow: benchmarking of the equivalent CenterCrop path found
// cudaMemcpy3DAsync losing to the kernel on the wider interleaved rows, so every interleaved
// non-scalar case stays on the kernel. Planar input is flattened to scalar planes before it reaches
// here, so it is covered without a planar branch. Called after the full validation chain on
// purpose: the copy needs no 32-bit strides, so a tensor too large for the kernel is still eligible
// for it.
inline bool TryCopyDenseByteNHWC(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                 const NVCVRectI &roi, cudaStream_t stream)
{
    if (inData.rank() != 4 || outData.rank() != 4 || inData.layout() != nvcv::TENSOR_NHWC
        || outData.layout() != nvcv::TENSOR_NHWC || inData.dtype() != outData.dtype())
    {
        return false;
    }

    // One byte per channel and one channel per pixel, so every extent below is both a pixel count
    // and a byte count.
    if (inData.dtype().bitsPerChannel()[0] != kBitsPerByte || inData.shape(3) != 1 || outData.shape(3) != 1)
    {
        return false;
    }

    const int64_t inBatch = inData.shape(0);
    const int64_t inRows  = inData.shape(1);
    const int64_t inCols  = inData.shape(2);
    const int64_t outRows = outData.shape(1);
    const int64_t outCols = outData.shape(2);

    if (inBatch != outData.shape(0) || inData.stride(3) != 1 || outData.stride(3) != 1 || inData.stride(2) != 1
        || outData.stride(2) != 1 || inData.stride(1) < inCols || outData.stride(1) < outCols
        || inData.stride(0) != inData.stride(1) * inRows || outData.stride(0) != outData.stride(1) * outRows)
    {
        return false;
    }

    cudaMemcpy3DParms params{};
    params.srcPtr = make_cudaPitchedPtr(inData.basePtr(), static_cast<std::size_t>(inData.stride(1)),
                                        static_cast<std::size_t>(inCols), static_cast<std::size_t>(inRows));
    params.dstPtr = make_cudaPitchedPtr(outData.basePtr(), static_cast<std::size_t>(outData.stride(1)),
                                        static_cast<std::size_t>(outCols), static_cast<std::size_t>(outRows));
    params.srcPos = make_cudaPos(static_cast<std::size_t>(roi.x), static_cast<std::size_t>(roi.y), 0);
    params.dstPos = make_cudaPos(0, 0, 0);
    params.extent = make_cudaExtent(static_cast<std::size_t>(roi.width), static_cast<std::size_t>(roi.height),
                                    static_cast<std::size_t>(inBatch));
    params.kind   = cudaMemcpyDeviceToDevice;

    NVCV_CHECK_THROW(cudaMemcpy3DAsync(&params, stream));
    return true;
}

// A crop moves bytes and computes nothing, so nothing about it depends on the dtype or the channel
// count: it is the same byte rectangle either way. The per-pixel kernel is capped at the pixel's own
// footprint -- one byte per access for a uchar1 -- while the shifted carrier stream moves 16 bytes
// per thread whatever the pixel is. What a crop adds over a plain copy is that its source is offset
// by roi.x * pixelBytes, an arbitrary byte phase, so source and destination cannot both be 16-byte
// aligned; the stream keeps the alignment on the store side and repairs the source in registers.
inline bool TryCropCarrierStream(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                                 const nvcv::TensorDataStridedCuda              &outData,
                                 const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                 const nvcv::TensorDataAccessStridedImagePlanar &outAccess, const NVCVRectI &roi,
                                 int pixelBytes)
{
    // Whole-source F32 RGBA copies are faster on the existing 16-byte per-pixel kernel.
    if (pixelBytes == 16 && inData.dtype().channelType(0) == nvcv::TYPE_F32 && roi.x == 0 && roi.y == 0
        && roi.width == inAccess.numCols() && roi.height == inAccess.numRows())
    {
        return false;
    }

    // Columns must be packed for a crop row to be a byte run.
    if (inAccess.colStride() != pixelBytes || outAccess.colStride() != pixelBytes)
    {
        return false;
    }

    // Tried before the dense-copy gate and before the per-pixel kernel's extent check, so it holds
    // itself to the same extent limit: a tensor too large for that kernel must still report it
    // rather than quietly succeeding here.
    if (std::max(inAccess.sampleStride() * inAccess.numSamples(), outAccess.sampleStride() * outAccess.numSamples())
        > cuda::TypeTraits<StrideType>::max)
    {
        return false;
    }

    es::Plan plan;
    if (!es::MakeRunPlan(static_cast<int64_t>(roi.width) * pixelBytes, roi.height, inAccess.numSamples(),
                         inAccess.rowStride(), outAccess.rowStride(), inAccess.sampleStride(), outAccess.sampleStride(),
                         true, plan))
    {
        return false;
    }

    const auto          *srcLo = reinterpret_cast<const unsigned char *>(inData.basePtr());
    const unsigned char *srcHi = es::SourceEnd(srcLo, inAccess);
    const unsigned char *src   = srcLo + inAccess.rowStride() * roi.y + static_cast<int64_t>(roi.x) * pixelBytes;

    auto               *dst = reinterpret_cast<unsigned char *>(outData.basePtr());
    // A shifted 24-byte pixel run with a partial final carrier is faster on the pixel kernel.
    constexpr uintptr_t carrierMask = es::kCarrierBytes - 1;
    const uintptr_t     sourcePhase = (reinterpret_cast<uintptr_t>(src) ^ reinterpret_cast<uintptr_t>(dst))
                                | plan.srcRegionStride | plan.srcSampleStride;
    if (pixelBytes == 24 && (sourcePhase & carrierMask) != 0
        && ((reinterpret_cast<uintptr_t>(dst) + plan.runBytes) & carrierMask) != 0)
    {
        return false;
    }

    return es::TryRunShifted<kCropCarriersPerThread, es::ShiftCarrier<es::IdentityRegions>>(stream, src, dst, srcLo,
                                                                                            srcHi, plan, {});
}

inline void RunCustomCrop(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                          const nvcv::TensorDataStridedCuda &outData, const NVCVRectI &roi)
{
    if (roi.width <= 0 || roi.height <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "ROI width and height must be positive");
    }

    const nvcv::TensorLayout &inLayout  = inData.layout();
    const nvcv::TensorLayout &outLayout = outData.layout();
    if (!(inLayout == nvcv::TENSOR_NHWC || inLayout == nvcv::TENSOR_HWC)
        || !(outLayout == nvcv::TENSOR_NHWC || outLayout == nvcv::TENSOR_HWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat both Input and Output must be kHWC or kNHWC");
    }

    if (inData.dtype() != outData.dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and Output formats must be same input format");
    }

    const int bytesPerChannel = SupportedBytesPerChannel(inData.dtype());
    if (bytesPerChannel == 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input DataFormat");
    }

    const int batch    = inAccess->numSamples();
    const int channels = inAccess->numChannels();
    const int rows     = inAccess->numRows();
    const int cols     = inAccess->numCols();

    if (channels > 4 || channels < 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number ch = %d", channels);
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output DataFormat");
    }

    if (batch != outAccess->numSamples() || channels != outAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same sample and channel counts");
    }

    if (roi.height > outAccess->numRows() || roi.width > outAccess->numCols())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "ROI larger than dst buffer");
    }

    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width - 1 >= cols || roi.y + roi.height - 1 >= rows)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Roi range x %d y %d width %d height %d",
                              roi.x, roi.y, roi.width, roi.height);
    }

    if (TryCropCarrierStream(stream, inData, outData, *inAccess, *outAccess, roi, bytesPerChannel * channels))
    {
        return;
    }

    if (TryCopyDenseByteNHWC(inData, outData, roi, stream))
    {
        return;
    }

    const int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    const int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) > cuda::TypeTraits<StrideType>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<StrideType>::max);
    }

    // The gate above admits only these four widths, so the default arm is unreachable unless that
    // gate is ever widened.
    switch (bytesPerChannel)
    {
    case 1:
        RunChannelSwitch<unsigned char>(channels, stream, inData, outData, roi, batch);
        break;
    case 2:
        RunChannelSwitch<unsigned short>(channels, stream, inData, outData, roi, batch);
        break;
    case 4:
        RunChannelSwitch<int>(channels, stream, inData, outData, roi, batch);
        break;
    case 8:
        RunChannelSwitch<double>(channels, stream, inData, outData, roi, batch);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }
}

} // namespace

namespace cvcuda::priv {

void CustomCrop::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                            const NVCVRectI &cropRect) const
{
    CVCUDA_NVTX_RANGE("cvcuda::CustomCrop::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    // Cropping copies each channel plane independently and identically, so a planar (NCHW/CHW) image
    // is just N*C single-channel planes: flatten them into the sample dimension and reuse the
    // interleaved single-channel copy path unchanged, producing bit-exact planar output.
    if (auto planarViews = PlanarSingleChannelViews(*inData, *outData))
    {
        RunCustomCrop(stream, planarViews->first, planarViews->second, cropRect);
        return;
    }

    RunCustomCrop(stream, *inData, *outData, cropRect);
}

} // namespace cvcuda::priv

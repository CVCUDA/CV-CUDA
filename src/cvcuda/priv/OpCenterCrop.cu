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
#include "OpCenterCrop.hpp"
#include "PlanarTensorView.hpp"

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

namespace {

constexpr int kBitsPerByte = 8;

// The legacy kernel used BLOCK=32 with BLOCK/4 rows; keep the shape so the launch geometry, and
// therefore the access pattern, is unchanged.
constexpr dim3 kBlock{32, 8, 1};

// int32_t strides are sufficient only because the caller rejects tensors whose extent exceeds the
// int32 range; widening that guard without widening this breaks the address arithmetic.
using StrideType = int32_t;

template<class SrcWrapper, class DstWrapper>
__global__ void CenterCropKernel(SrcWrapper src, DstWrapper dst, int leftIndices, int topIndices, int cropRows,
                                 int cropColumns)
{
    const int dstX     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dstY     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batchIdx = blockIdx.z;

    if ((dstX < cropColumns) && (dstY < cropRows))
    {
        *dst.ptr(batchIdx, dstY, dstX) = *src.ptr(batchIdx, dstY + topIndices, dstX + leftIndices);
    }
}

// Deliberately not same_shape::DispatchChannels: that one rejects 2 channels and this operator must
// accept them (the legacy table had a uchar2/ushort2/int2/double2 column).
template<typename BaseT, typename Cb>
void DispatchElemChannels(int channels, const Cb &cb)
{
    switch (channels)
    {
    case 1:
        cb(cuda::MakeType<BaseT, 1>{});
        break;
    case 2:
        cb(cuda::MakeType<BaseT, 2>{});
        break;
    case 3:
        cb(cuda::MakeType<BaseT, 3>{});
        break;
    case 4:
        cb(cuda::MakeType<BaseT, 4>{});
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number ch = %d", channels);
    }
}

// The base type is a width carrier only, so these four families must keep matching the legacy 5x4
// table's rows. In particular the 8-byte 4-channel entry must stay cuda::MakeType<double, 4>, which
// is the 16-byte-aligned double4_16a; plain double4 is a different type under CUDA 13.
template<typename Cb>
void DispatchPixelType(int bytesPerChannel, int channels, const Cb &cb)
{
    switch (bytesPerChannel)
    {
    case 1:
        DispatchElemChannels<unsigned char>(channels, cb);
        break;
    case 2:
        DispatchElemChannels<unsigned short>(channels, cb);
        break;
    case 4:
        DispatchElemChannels<int>(channels, cb);
        break;
    case 8:
        DispatchElemChannels<double>(channels, cb);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }
}

inline void RunCenterCrop(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                          const nvcv::TensorDataStridedCuda &outData, int cropRows, int cropColumns)
{
    // Legacy classified both layouts before comparing dtypes, so a doubly-invalid input reported the
    // layout error. Keep the layout check first to preserve which error a caller sees.
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

    // Reproduce the legacy admitted set exactly: 8U/8S/16U/16S/16F/32S/32F/64F. Gating on "not
    // unsigned-32 and a whole number of bytes" would be wider than the legacy path, which reached
    // this list through GetLegacyDataType and therefore also rejected every (kind, width) pair its
    // enum had no name for -- 64-bit unsigned and signed among them.
    const nvcv::DataType dtype = inData.dtype();
    const auto           bpc   = dtype.bitsPerChannel();
    for (int c = 1; c < dtype.numChannels(); ++c)
    {
        // GetLegacyDataType rejected a mixed-width packing before classifying it; NVCV_PACKING_X32_Y24b8
        // is a real one, so this is reachable rather than defensive.
        if (bpc[c] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }
    const nvcv::DataKind kind      = dtype.dataKind();
    const int            bitsPerCh = bpc[0];
    const bool           admitted  = (kind == nvcv::DataKind::UNSIGNED && (bitsPerCh == 8 || bitsPerCh == 16))
                       || (kind == nvcv::DataKind::SIGNED && (bitsPerCh == 8 || bitsPerCh == 16 || bitsPerCh == 32))
                       || (kind == nvcv::DataKind::FLOAT && (bitsPerCh == 16 || bitsPerCh == 32 || bitsPerCh == 64));
    if (!admitted)
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
    const int columns  = inAccess->numCols();

    // Legacy checked the channel count before creating the output accessor; an input with both a bad
    // channel count and an unusable output layout must keep reporting the channel error.
    if (channels > 4 || channels < 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number ch = %d", channels);
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output DataFormat");
    }

    if ((batch != outAccess->numSamples()) || (cropRows > outAccess->numRows()) || (cropColumns > outAccess->numCols()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output shape for the requested crop");
    }

    const int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    const int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    // ERROR_INVALID_ARGUMENT, not the ERROR_OVERFLOW that sibling operators use for this same guard:
    // legacy returned INVALID_PARAMETER here, which mapped to ERROR_INVALID_ARGUMENT. Normalizing
    // this to match the siblings would change the status callers observe.
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<StrideType>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<StrideType>::max);
    }

    const int topIndices  = (rows - cropRows) / 2;
    const int leftIndices = (columns - cropColumns) / 2;

    // Deliberate divergence from legacy, which computed this as ceil((float)a/b). Past 2^24 that
    // conversion loses the low bits, so the final partial block is never launched and 1..32 columns
    // (or 1..8 rows) of the output stay unwritten; DivUp is exact. Reachable, not theoretical: the
    // smallest divergent extent is 16777217, and a 1 x 16777217 x 1 U16 image clears the int32
    // stride guard above at ~32 MB and is not byte-wide, so it misses the memcpy path below.
    const dim3 grid(util::DivUp(cropColumns, static_cast<int>(kBlock.x)),
                    util::DivUp(cropRows, static_cast<int>(kBlock.y)), batch);

    DispatchPixelType(bitsPerCh / kBitsPerByte, channels,
                      [&](auto pixel)
                      {
                          using T  = decltype(pixel);
                          auto src = cuda::CreateTensorWrapNHW<const T, StrideType>(inData);
                          auto dst = cuda::CreateTensorWrapNHW<T, StrideType>(outData);

                          CenterCropKernel<<<grid, kBlock, 0, stream>>>(src, dst, leftIndices, topIndices, cropRows,
                                                                        cropColumns);

                          NVCV_CHECK_THROW(cudaGetLastError());
                      });
}

inline bool TryCopyDenseByteNHWCCenterCrop(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                                           const nvcv::TensorDataStridedCuda &outData, const nvcv::Size2D &cropSize)
{
    if (inData.rank() != 4 || outData.rank() != 4 || inData.layout() != nvcv::TENSOR_NHWC
        || outData.layout() != nvcv::TENSOR_NHWC || inData.dtype() != outData.dtype())
    {
        return false;
    }

    // Measured: cudaMemcpy3DAsync is slower than the kernel for every interleaved non-scalar case,
    // so this gate is a benchmarked choice rather than an implementation limit -- widening it
    // regresses the wider rows. Native-planar U8 is flattened to scalar planes before reaching here.
    if (inData.dtype().bitsPerChannel()[0] != kBitsPerByte || inData.shape(3) != 1 || outData.shape(3) != 1)
    {
        return false;
    }

    const int64_t inBatch = inData.shape(0);
    const int64_t inRows  = inData.shape(1);
    const int64_t inCols  = inData.shape(2);

    const int64_t outBatch = outData.shape(0);
    const int64_t outRows  = outData.shape(1);
    const int64_t outCols  = outData.shape(2);

    if (cropSize.h <= 0 || cropSize.w <= 0 || inBatch != outBatch || outRows != cropSize.h || outCols != cropSize.w
        || cropSize.h > inRows || cropSize.w > inCols)
    {
        return false;
    }

    // Gated above to one byte-wide channel per pixel, so a pixel is one byte and the literal 1s
    // below are the element, column and channel strides; the extents are likewise counts of bytes.
    if (inData.stride(3) != 1 || outData.stride(3) != 1 || inData.stride(2) != 1 || outData.stride(2) != 1
        || inData.stride(1) < inCols || outData.stride(1) < outCols || inData.stride(0) != inData.stride(1) * inRows
        || outData.stride(0) != outData.stride(1) * outRows)
    {
        return false;
    }

    const int64_t top  = (inRows - cropSize.h) / 2;
    const int64_t left = (inCols - cropSize.w) / 2;

    cudaMemcpy3DParms params{};
    params.srcPtr = make_cudaPitchedPtr(inData.basePtr(), static_cast<std::size_t>(inData.stride(1)),
                                        static_cast<std::size_t>(inCols), static_cast<std::size_t>(inRows));
    params.dstPtr = make_cudaPitchedPtr(outData.basePtr(), static_cast<std::size_t>(outData.stride(1)),
                                        static_cast<std::size_t>(outCols), static_cast<std::size_t>(outRows));
    params.srcPos = make_cudaPos(static_cast<std::size_t>(left), static_cast<std::size_t>(top), 0);
    params.dstPos = make_cudaPos(0, 0, 0);
    params.extent = make_cudaExtent(static_cast<std::size_t>(cropSize.w), static_cast<std::size_t>(cropSize.h),
                                    static_cast<std::size_t>(inBatch));
    params.kind   = cudaMemcpyDeviceToDevice;

    NVCV_CHECK_THROW(cudaMemcpy3DAsync(&params, stream));
    return true;
}

} // namespace

namespace cvcuda::priv {

void CenterCrop::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                            const nvcv::Size2D &cropSize) const
{
    CVCUDA_NVTX_RANGE("cvcuda::CenterCrop::operator()[Tensor]");
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

    if (TryCopyDenseByteNHWCCenterCrop(stream, *inData, *outData, cropSize))
    {
        return;
    }

    // Center-cropping copies each channel plane independently and identically, so a planar (NCHW/CHW)
    // image is just N*C single-channel planes: flatten them into the sample dimension and reuse the
    // dense-copy path or the interleaved single-channel kernel unchanged, producing bit-exact
    // planar output.
    if (auto planarViews = PlanarSingleChannelViews(*inData, *outData))
    {
        if (TryCopyDenseByteNHWCCenterCrop(stream, planarViews->first, planarViews->second, cropSize))
        {
            return;
        }

        RunCenterCrop(stream, planarViews->first, planarViews->second, cropSize.h, cropSize.w);
        return;
    }

    RunCenterCrop(stream, *inData, *outData, cropSize.h, cropSize.w);
}

} // namespace cvcuda::priv

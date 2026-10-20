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

#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpComposite.hpp"
#include "TensorAccessValidation.hpp"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataLayout.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <array>
#include <cstdint>
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// Composite only ever addresses 8-bit samples, so the whole operator is written against uint8_t.
using SampleType = uint8_t;

// 32-bit pitches keep the interleaved Tensor path's offset arithmetic in 32-bit registers; an
// int64_t wrap measured 0.6% slower than the legacy kernel on the basic tier.
using InterleavedStride = int32_t;

// All four pitches stay run-time, so the planar Tensor path also addresses tensors whose column
// pitch is padded rather than the packed sizeof(SampleType) a constant stride would fold in.
using PlanarTensorWrap = cuda::TensorWrap<SampleType, -1, -1, -1, -1>;

// Each planar Tensor thread handles this many columns, blockDim.x apart. The kernel's thread ->
// column mapping and RunCompositePlanarTensor's grid width must agree on it.
constexpr int kPlanarColsPerThread = 2;

// ClassifyDataFormat admits exactly four image layouts and rejects everything else, so no separate
// "is this an image layout" guard is needed downstream.
using ImageLayout = cvcuda::priv::legacy_format::DataFormatCode;
using cvcuda::priv::legacy_format::ClassifyDataFormat;
using cvcuda::priv::legacy_format::IsPlanar;

// Classifying (not comparing against TYPE_U8) admits packed 2U8/3U8/4U8 and keeps the shared
// classifier's rejections firing first.
template<class T>
inline bool IsU8DataType(const T &typeOrFormat)
{
    return cvcuda::priv::legacy_format::ClassifyDataType(typeOrFormat)
        == cvcuda::priv::legacy_format::DataTypeCode::kU8;
}

inline PlanarTensorWrap MakePlanarTensorWrap(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    return PlanarTensorWrap{access.sampleData(0), access.sampleStride(), access.chStride(), access.rowStride(),
                            access.colStride()};
}

__device__ __forceinline__ int AlphaLerp(uint8_t c0, uint8_t c1, uint8_t alpha)
{
    int value = (int)c0 * 255 + ((int)c1 - (int)c0) * (int)alpha + 128;
    // This is exact rounded division by 255 over the full uint8 input range.
    return (value + (value >> 8)) >> 8;
}

// cols/rows are read only by the Tensor instantiation; the VarShape one bounds each sample by its
// own extent.
template<bool VarShape, class SrcWrapper, class MaskWrapper, class DstWrapper>
__global__ void CompositeKernel(SrcWrapper fg, SrcWrapper bg, MaskWrapper fgMask, DstWrapper dst, int cols, int rows)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int batch_idx = blockIdx.z;

    if constexpr (VarShape)
    {
        if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
            return;
    }
    else
    {
        if (dst_x >= cols || dst_y >= rows)
            return;
    }

    using T = std::remove_const_t<typename SrcWrapper::ValueType>;
    using U = std::remove_const_t<typename MaskWrapper::ValueType>;
    using D = std::remove_const_t<typename DstWrapper::ValueType>;

    // Compile-time channels let each instantiation fold the loop and avoid wrapper metadata loads.
    constexpr int dst_ch = cuda::NumElements<D>;
    constexpr int src_ch = cuda::NumElements<T>;

    U mask_val = *fgMask.ptr(batch_idx, dst_y, dst_x);
    T bg_val   = *bg.ptr(batch_idx, dst_y, dst_x);
    T fg_val   = *fg.ptr(batch_idx, dst_y, dst_x);
    D out;

    for (int i = 0; i < src_ch; i++)
    {
        uint8_t c0               = cuda::GetElement(bg_val, i);
        uint8_t c1               = cuda::GetElement(fg_val, i);
        cuda::GetElement(out, i) = AlphaLerp(c0, c1, mask_val);
    }
    if (src_ch == 3 && dst_ch == 4)
        cuda::GetElement(out, 3) = 255;
    *dst.ptr(batch_idx, dst_y, dst_x) = out;
}

// Deliberately not merged into CompositeKernel: this path keeps the legacy planar schedule.
template<int dcn>
__global__ void CompositePlanarTensorKernel(PlanarTensorWrap fg, PlanarTensorWrap bg, PlanarTensorWrap fgMask,
                                            PlanarTensorWrap dst, int cols, int rows)
{
    int dst_x = blockIdx.x * (blockDim.x * kPlanarColsPerThread) + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= cols || dst_y >= rows)
        return;

    const int batch_idx = blockIdx.z;
    // The columns must be unrolled: issuing both pixels' loads together is what gives this kernel
    // its memory-level parallelism, and the legacy build unrolled it on its own.
#pragma unroll
    for (int dx = 0; dx < kPlanarColsPerThread; ++dx)
    {
        const int x = dst_x + dx * blockDim.x;
        if (x < cols)
        {
            uint8_t mask_val = __ldg(fgMask.ptr(batch_idx, 0, dst_y, x));

#pragma unroll
            for (int c = 0; c < 3; ++c)
            {
                const uint8_t c0                 = __ldg(bg.ptr(batch_idx, c, dst_y, x));
                const uint8_t c1                 = __ldg(fg.ptr(batch_idx, c, dst_y, x));
                *dst.ptr(batch_idx, c, dst_y, x) = AlphaLerp(c0, c1, mask_val);
            }
            if constexpr (dcn == 4)
            {
                *dst.ptr(batch_idx, 3, dst_y, x) = 255;
            }
        }
    }
}

template<int dcn>
__global__ void CompositePlanarVarShapeKernel(const cuda::ImageBatchVarShapeWrap<const SampleType> fg,
                                              const cuda::ImageBatchVarShapeWrap<const SampleType> bg,
                                              const cuda::ImageBatchVarShapeWrap<const SampleType> fgMask,
                                              cuda::ImageBatchVarShapeWrap<SampleType>             dst)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    SampleType mask_val = *fgMask.ptr(batch_idx, 0, dst_y, dst_x);

#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        const SampleType c0                  = *bg.ptr(batch_idx, c, dst_y, dst_x);
        const SampleType c1                  = *fg.ptr(batch_idx, c, dst_y, dst_x);
        *dst.ptr(batch_idx, c, dst_y, dst_x) = AlphaLerp(c0, c1, mask_val);
    }
    if constexpr (dcn == 4)
    {
        *dst.ptr(batch_idx, 3, dst_y, dst_x) = 255;
    }
}

template<int dcn>
inline void RunCompositeTensor(const nvcv::TensorDataStridedCuda &foreground,
                               const nvcv::TensorDataStridedCuda &background, const nvcv::TensorDataStridedCuda &fgMask,
                               const nvcv::TensorDataStridedCuda &outData, int batchSize, int outWidth, int outHeight,
                               cudaStream_t stream)
{
    using SrcType = cuda::MakeType<SampleType, 3>;
    using DstType = cuda::MakeType<SampleType, dcn>;

    const dim3 blockSize(32, 8, 1);
    const dim3 gridSize(util::DivUp(outWidth, blockSize.x), util::DivUp(outHeight, blockSize.y), batchSize);

    auto fg   = cuda::CreateTensorWrapNHW<const SrcType, InterleavedStride>(foreground);
    auto bg   = cuda::CreateTensorWrapNHW<const SrcType, InterleavedStride>(background);
    auto mask = cuda::CreateTensorWrapNHW<const SampleType, InterleavedStride>(fgMask);
    auto dst  = cuda::CreateTensorWrapNHW<DstType, InterleavedStride>(outData);

    CompositeKernel<false><<<gridSize, blockSize, 0, stream>>>(fg, bg, mask, dst, outWidth, outHeight);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<int dcn>
inline void RunCompositePlanarTensor(const nvcv::TensorDataAccessStridedImagePlanar &foreground,
                                     const nvcv::TensorDataAccessStridedImagePlanar &background,
                                     const nvcv::TensorDataAccessStridedImagePlanar &fgMask,
                                     const nvcv::TensorDataAccessStridedImagePlanar &outData, int batchSize,
                                     int outWidth, int outHeight, cudaStream_t stream)
{
    const dim3 blockSize(64, 4, 1);
    const dim3 gridSize(util::DivUp(outWidth, blockSize.x * kPlanarColsPerThread), util::DivUp(outHeight, blockSize.y),
                        batchSize);

    CompositePlanarTensorKernel<dcn><<<gridSize, blockSize, 0, stream>>>(
        MakePlanarTensorWrap(foreground), MakePlanarTensorWrap(background), MakePlanarTensorWrap(fgMask),
        MakePlanarTensorWrap(outData), outWidth, outHeight);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<int dcn>
inline void RunCompositeVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &foreground,
                                 const nvcv::ImageBatchVarShapeDataStridedCuda &background,
                                 const nvcv::ImageBatchVarShapeDataStridedCuda &fgMask,
                                 const nvcv::ImageBatchVarShapeDataStridedCuda &outData, cudaStream_t stream)
{
    using SrcType = cuda::MakeType<SampleType, 3>;
    using DstType = cuda::MakeType<SampleType, dcn>;

    cuda::ImageBatchVarShapeWrap<const SrcType>    fgWrap(foreground);
    cuda::ImageBatchVarShapeWrap<const SrcType>    bgWrap(background);
    cuda::ImageBatchVarShapeWrap<const SampleType> maskWrap(fgMask);
    cuda::ImageBatchVarShapeWrap<DstType>          dstWrap(outData);

    const int          batchSize  = outData.numImages();
    const nvcv::Size2D outMaxSize = outData.maxSize();

    const dim3 blockSize(dcn == 3 ? 32 : 16, dcn == 3 ? 8 : 16, 1);
    const dim3 gridSize(util::DivUp(outMaxSize.w, blockSize.x), util::DivUp(outMaxSize.h, blockSize.y), batchSize);

    CompositeKernel<true>
        <<<gridSize, blockSize, 0, stream>>>(fgWrap, bgWrap, maskWrap, dstWrap, outMaxSize.w, outMaxSize.h);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<int dcn>
inline void RunCompositePlanarVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &foreground,
                                       const nvcv::ImageBatchVarShapeDataStridedCuda &background,
                                       const nvcv::ImageBatchVarShapeDataStridedCuda &fgMask,
                                       const nvcv::ImageBatchVarShapeDataStridedCuda &outData, cudaStream_t stream)
{
    cuda::ImageBatchVarShapeWrap<const SampleType> fgWrap(foreground);
    cuda::ImageBatchVarShapeWrap<const SampleType> bgWrap(background);
    cuda::ImageBatchVarShapeWrap<const SampleType> maskWrap(fgMask);
    cuda::ImageBatchVarShapeWrap<SampleType>       dstWrap(outData);

    const int          batchSize  = outData.numImages();
    const nvcv::Size2D outMaxSize = outData.maxSize();

    const dim3 blockSize(32, 8, 1);
    const dim3 gridSize(util::DivUp(outMaxSize.w, blockSize.x), util::DivUp(outMaxSize.h, blockSize.y), batchSize);

    CompositePlanarVarShapeKernel<dcn><<<gridSize, blockSize, 0, stream>>>(fgWrap, bgWrap, maskWrap, dstWrap);
    NVCV_CHECK_THROW(cudaGetLastError());
}

} // namespace

namespace cvcuda::priv {

void Composite::operator()(cudaStream_t stream, const nvcv::Tensor &foreground, const nvcv::Tensor &background,
                           const nvcv::Tensor &fgMask, const nvcv::Tensor &output) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Composite::operator()[Tensor]");
    auto foregroundData = RequireCudaTensor(foreground, "Input foreground");
    auto backgroundData = RequireCudaTensor(background, "Input background");
    auto fgMaskData     = RequireCudaTensor(fgMask, "Input fgMask");
    auto outData        = RequireCudaTensor(output, "Output");

    const ImageLayout backgroundLayout = ClassifyDataFormat(backgroundData->layout());
    const ImageLayout foregroundLayout = ClassifyDataFormat(foregroundData->layout());
    // The mask's layout is otherwise unconstrained, but it still has to be an image layout: called
    // for its throw.
    ClassifyDataFormat(fgMaskData->layout());
    const ImageLayout outputLayout = ClassifyDataFormat(outData->layout());

    if (!((foregroundLayout == backgroundLayout) && (foregroundLayout == outputLayout)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between foreground, background, foreground mask and output");
    }

    auto foregroundAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*foregroundData);
    NVCV_ASSERT(foregroundAccess);

    auto backgroundAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*backgroundData);
    NVCV_ASSERT(backgroundAccess);

    auto fgMaskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*fgMaskData);
    NVCV_ASSERT(fgMaskAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    NVCV_ASSERT(outAccess);

    // Eager, and before the shape check below: the legacy operator classified all four dtypes at
    // this point, so a tensor that is both wrong-dtype and wrong-shape reports the dtype error.
    const bool foregroundIsU8 = IsU8DataType(foregroundData->dtype());
    const bool backgroundIsU8 = IsU8DataType(backgroundData->dtype());
    const bool fgMaskIsU8     = IsU8DataType(fgMaskData->dtype());
    const bool outputIsU8     = IsU8DataType(outData->dtype());

    if (!((foregroundAccess->numSamples() == backgroundAccess->numSamples())
          && (foregroundAccess->numSamples() == fgMaskAccess->numSamples())
          && (foregroundAccess->numSamples() == outAccess->numSamples())
          && (foregroundAccess->numRows() == backgroundAccess->numRows())
          && (foregroundAccess->numRows() == fgMaskAccess->numRows())
          && (foregroundAccess->numRows() == outAccess->numRows())
          && (foregroundAccess->numCols() == backgroundAccess->numCols())
          && (foregroundAccess->numCols() == fgMaskAccess->numCols())
          && (foregroundAccess->numCols() == outAccess->numCols())))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input/output shape");
    }

    const int foregroundChannels = foregroundAccess->numChannels();
    const int backgroundChannels = backgroundAccess->numChannels();
    const int fgMaskChannels     = fgMaskAccess->numChannels();
    const int outputChannels     = outAccess->numChannels();

    if (!((foregroundChannels == 3) && (backgroundChannels == 3) && (fgMaskChannels == 1)
          && (outputChannels == 3 || outputChannels == 4)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", outputChannels);
    }

    if (!(foregroundIsU8 && backgroundIsU8 && fgMaskIsU8 && outputIsU8))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }

    const int batchSize = static_cast<int>(foregroundAccess->numSamples());
    const int outWidth  = outAccess->numCols();
    const int outHeight = outAccess->numRows();

    if (IsPlanar(foregroundLayout))
    {
        if (foregroundAccess->numSamples() > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar Composite requires number of images <= 65535");
        }

        if (outputChannels == 3)
        {
            RunCompositePlanarTensor<3>(*foregroundAccess, *backgroundAccess, *fgMaskAccess, *outAccess, batchSize,
                                        outWidth, outHeight, stream);
        }
        else
        {
            RunCompositePlanarTensor<4>(*foregroundAccess, *backgroundAccess, *fgMaskAccess, *outAccess, batchSize,
                                        outWidth, outHeight, stream);
        }
        return;
    }

    // See OpComposite_Negative/interleaved_tensor_batch_exceeds_cuda_grid_limit
    if (batchSize > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Composite requires number of samples <= 65535");
    }

    if (outputChannels == 3)
    {
        RunCompositeTensor<3>(*foregroundData, *backgroundData, *fgMaskData, *outData, batchSize, outWidth, outHeight,
                              stream);
    }
    else
    {
        RunCompositeTensor<4>(*foregroundData, *backgroundData, *fgMaskData, *outData, batchSize, outWidth, outHeight,
                              stream);
    }
}

void Composite::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &foreground,
                           const nvcv::ImageBatchVarShape &background, const nvcv::ImageBatchVarShape &fgMask,
                           const nvcv::ImageBatchVarShape &output) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Composite::operator()[ImageBatchVarShape]");
    const int numImages = foreground.numImages();
    if (!((numImages == background.numImages()) && (numImages == fgMask.numImages())
          && (numImages == output.numImages())))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input foreground, background, fgMask and output must have same number of images");
    }

    for (int i = 0; i < numImages; ++i)
    {
        const nvcv::Size2D foregroundSize = foreground[i].size();
        const nvcv::Size2D backgroundSize = background[i].size();
        const nvcv::Size2D fgMaskSize     = fgMask[i].size();
        const nvcv::Size2D outputSize     = output[i].size();

        if (!((foregroundSize.w == backgroundSize.w) && (foregroundSize.w == fgMaskSize.w)
              && (foregroundSize.w == outputSize.w) && (foregroundSize.h == backgroundSize.h)
              && (foregroundSize.h == fgMaskSize.h) && (foregroundSize.h == outputSize.h)))
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Input foreground, background, fgMask and output images must have same width and height");
        }
    }

    auto foregroundData = RequireCudaVarShape(foreground, stream, "Input foreground");
    auto backgroundData = RequireCudaVarShape(background, stream, "Input background");
    auto fgMaskData     = RequireCudaVarShape(fgMask, stream, "Input fgMask");
    auto outData        = RequireCudaVarShape(output, stream, "Output");

    if (!(foregroundData->uniqueFormat() && backgroundData->uniqueFormat() && fgMaskData->uniqueFormat()
          && outData->uniqueFormat()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in the input batch must all have the same format");
    }

    const ImageLayout backgroundLayout = ClassifyDataFormat(*backgroundData);
    const ImageLayout foregroundLayout = ClassifyDataFormat(*foregroundData);
    // As on the tensor path, called for its throw.
    ClassifyDataFormat(*fgMaskData);
    const ImageLayout outputLayout = ClassifyDataFormat(*outData);

    if (!((foregroundLayout == backgroundLayout) && (foregroundLayout == outputLayout)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between foreground, background, foreground mask and output");
    }

    const bool foregroundIsU8 = IsU8DataType(foregroundData->uniqueFormat());
    const bool backgroundIsU8 = IsU8DataType(backgroundData->uniqueFormat());
    const bool fgMaskIsU8     = IsU8DataType(fgMaskData->uniqueFormat());
    const bool outputIsU8     = IsU8DataType(outData->uniqueFormat());

    if (!(foregroundIsU8 && backgroundIsU8 && fgMaskIsU8 && outputIsU8))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }

    const int foregroundChannels = foregroundData->uniqueFormat().numChannels();
    const int backgroundChannels = backgroundData->uniqueFormat().numChannels();
    const int fgMaskChannels     = fgMaskData->uniqueFormat().numChannels();
    const int outputChannels     = outData->uniqueFormat().numChannels();

    if (!((foregroundChannels == 3) && (backgroundChannels == 3) && (fgMaskChannels == 1)
          && (outputChannels == 3 || outputChannels == 4)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", outputChannels);
    }

    // The batch sizes were already matched above; the legacy operator re-checked them here against
    // the exported batches, and that check is kept in the same position.
    if (!((foregroundData->numImages() == backgroundData->numImages())
          && (foregroundData->numImages() == fgMaskData->numImages())
          && (foregroundData->numImages() == outData->numImages())))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input/output batch size");
    }

    if (IsPlanar(foregroundLayout))
    {
        if (outData->numImages() > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar Composite requires number of images <= 65535");
        }

        if (outputChannels == 3)
        {
            RunCompositePlanarVarShape<3>(*foregroundData, *backgroundData, *fgMaskData, *outData, stream);
        }
        else
        {
            RunCompositePlanarVarShape<4>(*foregroundData, *backgroundData, *fgMaskData, *outData, stream);
        }
        return;
    }

    // See OpComposite_Negative/interleaved_varshape_batch_exceeds_cuda_grid_limit
    if (outData->numImages() > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Composite requires number of images <= 65535");
    }

    if (outputChannels == 3)
    {
        RunCompositeVarShape<3>(*foregroundData, *backgroundData, *fgMaskData, *outData, stream);
    }
    else
    {
        RunCompositeVarShape<4>(*foregroundData, *backgroundData, *fgMaskData, *outData, stream);
    }
}

} // namespace cvcuda::priv

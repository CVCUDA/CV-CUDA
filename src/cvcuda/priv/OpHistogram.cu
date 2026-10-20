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

#include "CudaDeviceUtils.hpp"
#include "Nvtx.hpp"
#include "OpHistogram.hpp"
#include "OpReformat.hpp"
#include "TensorAccessValidation.hpp"

#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>

#include <array>
#include <limits>

namespace cuda = nvcv::cuda;

namespace {

using uchar = unsigned char;

// The histogram is defined over the whole 8-bit intensity range, which also sets the block size:
// one lane owns one bin of the block-local histogram held in dynamic shared memory.
constexpr int kHistogramBins = 256;

constexpr bool UseOnePixelHistogramKernel(int sm, int64_t numPixels)
{
    // The sixteen-pixel schedule regresses Turing because it trades fewer global
    // histogram merges for uncoalesced loads and serialized shared-memory atomics.
    // The restored kernel uses a signed 32-bit flattened index; larger inputs stay on the 2-D schedule.
    return sm == 75 && numPixels <= std::numeric_limits<int>::max();
}

static_assert(UseOnePixelHistogramKernel(75, std::numeric_limits<int>::max()));
static_assert(!UseOnePixelHistogramKernel(75, static_cast<int64_t>(std::numeric_limits<int>::max()) + 1));
static_assert(!UseOnePixelHistogramKernel(80, 1));
static_assert(!UseOnePixelHistogramKernel(90, 1));

template<class SrcWrapper, class DstWrapper>
__global__ void CalcHistOnePixelKernel(const SrcWrapper src, DstWrapper histogram, int numPixels, int width)
{
    extern __shared__ int shist[]; //size 256 * sizeof(int)

    int  batch_idx = blockIdx.z;                            //Z this will be the batch index.
    int  tid       = blockIdx.x * blockDim.x + threadIdx.x; // this is the thread index in the block
    int  x         = tid % width;                           // this is the x index of the pixel assigned to this tid
    int  y         = tid / width;                           // this is the y index in the pixel assigned to this tid
    int3 coord{x, y, batch_idx};

    // histogram index only used to sub the output histogram
    int sub_tid = threadIdx.x; //histogram index

    shist[sub_tid] = 0; //initialize the histogram for this bin in this block

    __syncthreads();

    if (tid < numPixels)
    {
        atomicAdd(&shist[src[coord]], 1);
    }
    __syncthreads(); // wait for all of the threads in this block to finish

    int hist_val = shist[sub_tid]; // get the bin value for this thread

    // this is the output histogram must be init to and atomicly added to.
    if (hist_val > 0)
    {
        atomicAdd(histogram.ptr(batch_idx, sub_tid), hist_val);
    }
}

template<class SrcWrapper, class MaskWrapper, class DstWrapper>
__global__ void CalcHistOnePixelKernel(const SrcWrapper src, DstWrapper histogram, MaskWrapper mask, int numPixels,
                                       int width)
{
    extern __shared__ int shist[];                                           //size 256 * sizeof(int)
    int                   batch_idx = blockIdx.z;                            //Z this will be the batch index.
    int                   tid       = blockIdx.x * blockDim.x + threadIdx.x; // this is the thread index in the block
    int                   x         = tid % width; // this is the x index in the pixel assigned to this tid
    int                   y         = tid / width; // this is the y index in the pixel assigned to this tid
    int3                  coord{x, y, batch_idx};

    // histogram index only used to sub the output histogram
    int sub_tid = threadIdx.x; //histogram index

    shist[sub_tid] = 0; //initialize the histogram for this bin in this block

    __syncthreads();

    if (tid < numPixels)
    {
        if (mask[coord])
            atomicAdd(&shist[src[coord]], 1);
    }
    __syncthreads(); // wait for all of the threads in this block to finish

    int hist_val = shist[sub_tid]; // get the bin value for this thread

    // this is the output histogram must be init to and atomicly added to.
    if (hist_val > 0)
    {
        atomicAdd(histogram.ptr(batch_idx, sub_tid), hist_val);
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void CalcHistKernel(const SrcWrapper src, DstWrapper histogram, int height, int width)
{
    extern __shared__ int shist[]; //size 256 * sizeof(int)

    int batch_idx = blockIdx.z; //Z this will be the batch index.
    int x         = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    int y         = blockIdx.y * blockDim.y + threadIdx.y;

    // histogram index only used to sub the output histogram
    int sub_tid = threadIdx.y * blockDim.x + threadIdx.x; //histogram index

    shist[sub_tid] = 0; //initialize the histogram for this bin in this block

    __syncthreads();

    if (y < height)
    {
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (x + i < width)
            {
                atomicAdd(&shist[src[int3{x + i, y, batch_idx}]], 1);
            }
        }
    }
    __syncthreads(); // wait for all of the threads in this block to finish

    int hist_val = shist[sub_tid]; // get the bin value for this thread

    // this is the output histogram must be init to and atomicly added to.
    if (hist_val > 0)
    {
        atomicAdd(histogram.ptr(batch_idx, sub_tid), hist_val);
    }
}

template<class SrcWrapper, class MaskWrapper, class DstWrapper>
__global__ void CalcHistKernel(const SrcWrapper src, DstWrapper histogram, MaskWrapper mask, int height, int width)
{
    extern __shared__ int shist[];                //size 256 * sizeof(int)
    int                   batch_idx = blockIdx.z; //Z this will be the batch index.
    int                   x         = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    int                   y         = blockIdx.y * blockDim.y + threadIdx.y;

    // histogram index only used to sub the output histogram
    int sub_tid = threadIdx.y * blockDim.x + threadIdx.x; //histogram index

    shist[sub_tid] = 0; //initialize the histogram for this bin in this block

    __syncthreads();

    if (y < height)
    {
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (x + i < width)
            {
                int3 coord{x + i, y, batch_idx};
                if (mask[coord])
                    atomicAdd(&shist[src[coord]], 1);
            }
        }
    }
    __syncthreads(); // wait for all of the threads in this block to finish

    int hist_val = shist[sub_tid]; // get the bin value for this thread

    // this is the output histogram must be init to and atomicly added to.
    if (hist_val > 0)
    {
        atomicAdd(histogram.ptr(batch_idx, sub_tid), hist_val);
    }
}

} // namespace

namespace cvcuda::priv {

HistogramPlanarBridgeWorkspace::HistogramPlanarBridgeWorkspace()
{
    NVCV_CHECK_THROW(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
}

HistogramPlanarBridgeWorkspace::~HistogramPlanarBridgeWorkspace()
{
    if (ready != nullptr)
    {
        if (busy)
        {
            NVCV_CHECK_LOG(cudaEventSynchronize(ready));
        }
        NVCV_CHECK_LOG(cudaEventDestroy(ready));
    }
}

void HistogramPlanarBridgeWorkspace::ensure(const nvcv::TensorShape &inShape, nvcv::DataType inDtype,
                                            const nvcv::TensorShape *maskShape, nvcv::DataType maskDtype,
                                            cudaStream_t stream)
{
    const bool reallocateIn = !interleavedIn || interleavedIn.shape() != inShape || interleavedIn.dtype() != inDtype;
    const bool reallocateMask
        = maskShape != nullptr
       && (!interleavedMask || interleavedMask.shape() != *maskShape || interleavedMask.dtype() != maskDtype);

    if (busy)
    {
        if (reallocateIn || reallocateMask)
        {
            NVCV_CHECK_THROW(cudaEventSynchronize(ready));
            busy = false;
        }
        else
        {
            NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, ready));
        }
    }

    if (reallocateIn)
    {
        interleavedIn = nvcv::Tensor(inShape, inDtype);
    }
    if (reallocateMask)
    {
        interleavedMask = nvcv::Tensor(*maskShape, maskDtype);
    }
}

void HistogramPlanarBridgeWorkspace::record(cudaStream_t stream)
{
    NVCV_CHECK_THROW(cudaEventRecord(ready, stream));
    busy = true;
}

namespace {

inline bool IsHistogramPlanarLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

inline bool IsHistogramInterleavedLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC;
}

// The legacy gate compared GetLegacyDataType(dtype) against a DataType code, and that classifier
// keys on bits-per-channel plus data kind rather than on the exact nvcv type. Its kCV_8U code
// therefore also covered the packed TYPE_2U8/TYPE_3U8/TYPE_4U8 types, and kCV_32S the packed
// signed 32-bit ones. Nothing upstream narrows the dtype to a scalar type first -- the channel
// gates below read the shape's C axis, not the type -- so width plus kind is the equivalent test
// and an exact `dtype() == TYPE_U8` comparison would not be.
inline bool HasElementFormat(nvcv::DataType dtype, nvcv::DataKind kind, int32_t bitsPerChannel)
{
    auto bpc = dtype.bitsPerChannel();

    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        // Legacy rejected mixed-depth types outright, with the same status.
        if (bpc[i] != bpc[0])
        {
            return false;
        }
    }

    return dtype.dataKind() == kind && bpc[0] == bitsPerChannel;
}

inline void RunHistogram(const nvcv::TensorDataStridedCuda &inData, nvcv::OptionalTensorConstRef mask,
                         const nvcv::TensorDataStridedCuda &histogram, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be an image-like tensor");
    }
    auto histoAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(histogram);
    if (!histoAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Histogram must be an image-like tensor");
    }

    if (!IsHistogramInterleavedLayout(inData.layout()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid input DataFormat for calculating histogram");
    }

    if (!IsHistogramInterleavedLayout(histogram.layout()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid histogram DataFormat for calculating histogram");
    }

    if (!HasElementFormat(inData.dtype(), nvcv::DataKind::UNSIGNED, 8))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType for calculating histogram");
    }

    if (!HasElementFormat(histogram.dtype(), nvcv::DataKind::SIGNED, 32))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid histogram DataType for calculating histogram");
    }

    int batch    = static_cast<int>(inAccess->numSamples());
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int cols     = inAccess->numCols();

    if (batch != histoAccess->numRows())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Histogram tensor does not contain enough rows for an input batch tensor of N = %d",
                              batch);
    }

    if (histoAccess->numCols() < kHistogramBins)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Histogram tensor must have at least %d columns, got %d", kHistogramBins,
                              histoAccess->numCols());
    }

    if (histoAccess->numChannels() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid histogram channel number %d",
                              histoAccess->numChannels());
    }

    if (channels != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    // clear the histogram since we will just add to it only 1 sample in thi HW tensor by definition
    NVCV_CHECK_THROW(cudaMemset2DAsync(histoAccess->sampleData(0), histoAccess->rowStride(), 0,
                                       histoAccess->numCols() * histoAccess->colStride(), histoAccess->numRows(),
                                       stream));

    auto src   = cuda::CreateTensorWrapNHW<uchar>(inData);
    auto histo = cuda::Tensor2DWrap<int>(histogram);

    const int64_t num_pixels = static_cast<int64_t>(rows) * cols;

    int sm = 0;
    NVCV_CHECK_THROW(GetCurrentDeviceSM(sm));
    const bool use_one_pixel_kernel = UseOnePixelHistogramKernel(sm, num_pixels);

    constexpr int one_pixel_block_size = kHistogramBins;
    int           one_pixel_num_pixels = 0;
    dim3          one_pixel_grid_size;
    if (use_one_pixel_kernel)
    {
        one_pixel_num_pixels = static_cast<int>(num_pixels);
        one_pixel_grid_size.x
            = one_pixel_num_pixels / one_pixel_block_size + (one_pixel_num_pixels % one_pixel_block_size != 0);
        one_pixel_grid_size.z = batch;
    }

    dim3 block_size(32, 8);

    // Amortize each block-local histogram across sixteen horizontal pixels per lane.
    dim3 grid_size((cols + block_size.x * 16 - 1) / (block_size.x * 16), (rows + block_size.y - 1) / block_size.y,
                   batch);
    int  smem_size = kHistogramBins * sizeof(int);

    if (!mask)
    {
        if (use_one_pixel_kernel)
        {
            CalcHistOnePixelKernel<<<one_pixel_grid_size, one_pixel_block_size, smem_size, stream>>>(
                src, histo, one_pixel_num_pixels, cols);
        }
        else
        {
            CalcHistKernel<<<grid_size, block_size, smem_size, stream>>>(src, histo, rows, cols);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        // below retruns optional
        auto maskTensorData = mask->get().exportData<nvcv::TensorDataStridedCuda>();
        if (!maskTensorData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Mask must be a cuda-accessible, pitch-linear tensor");
        }
        // just check if the mask is formatted correcty.
        auto inMask = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskTensorData);
        if (!inMask)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask must be an image-like tensor");
        }

        if (!IsHistogramInterleavedLayout(maskTensorData->layout()))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid mask DataFormat for calculating histogram");
        }

        if (!HasElementFormat(maskTensorData->dtype(), nvcv::DataKind::UNSIGNED, 8))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid mask DataType for calculating histogram");
        }

        if (static_cast<int>(inMask->numSamples()) != batch || inMask->numChannels() != channels
            || inMask->numRows() != rows || inMask->numCols() != cols)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Mask tensor does not match input tensor shape");
        }

        auto maskAccess = cuda::CreateTensorWrapNHW<uchar>(*maskTensorData);

        if (use_one_pixel_kernel)
        {
            CalcHistOnePixelKernel<<<one_pixel_grid_size, one_pixel_block_size, smem_size, stream>>>(
                src, histo, maskAccess, one_pixel_num_pixels, cols);
        }
        else
        {
            CalcHistKernel<<<grid_size, block_size, smem_size, stream>>>(src, histo, maskAccess, rows, cols);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

inline nvcv::TensorShape MakeHistogramLegacyInputShape(const nvcv::TensorDataStridedCuda              &data,
                                                       const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    std::array<nvcv::TensorShape::DimType, NVCV_TENSOR_MAX_RANK> shape{};
    const char                                                  *layout = nullptr;
    int32_t                                                      rank   = 0;

    const bool hasBatch = data.layout() == nvcv::TENSOR_NCHW;
    if (hasBatch)
    {
        shape[rank] = access.numSamples();
        ++rank;
    }

    if (hasBatch || data.layout() == nvcv::TENSOR_CHW)
    {
        shape[rank] = access.numRows();
        ++rank;
        shape[rank] = access.numCols();
        ++rank;
        shape[rank] = access.numChannels();
        ++rank;
        layout = hasBatch ? "NHWC" : "HWC";
        return nvcv::TensorShape(shape.data(), rank, layout);
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Input/mask must have NCHW or CHW layout for planar Histogram adaptation");
}

inline void InferPlanarHistogramNoMask(cudaStream_t stream, const nvcv::TensorDataStridedCuda &interleavedInData,
                                       const nvcv::TensorDataStridedCuda &outHistogram)
{
    RunHistogram(interleavedInData, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, outHistogram, stream);
}

inline void InferPlanarHistogramWithMask(Reformat &reformatOp, HistogramPlanarBridgeWorkspace &workspace,
                                         cudaStream_t stream, const nvcv::Tensor &mask,
                                         const nvcv::TensorDataStridedCuda &inData,
                                         const nvcv::TensorDataStridedCuda &interleavedInData,
                                         const nvcv::TensorDataStridedCuda &outHistogram)
{
    auto maskData = RequireCudaTensor(mask, "Mask");
    if (!IsHistogramPlanarLayout(maskData->layout()) || maskData->layout() != inData.layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Mask layout must match planar Histogram input layout");
    }

    if (auto maskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskData); !maskAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask must be an image-like planar tensor");
    }

    reformatOp(stream, mask, workspace.interleavedMask);

    RunHistogram(interleavedInData, nvcv::OptionalTensorConstRef{workspace.interleavedMask}, outHistogram, stream);
}

inline void InferPlanarHistogram(Reformat &reformatOp, HistogramPlanarBridgeWorkspace &workspace, cudaStream_t stream,
                                 const nvcv::Tensor &in, nvcv::OptionalTensorConstRef mask,
                                 const nvcv::TensorDataStridedCuda &inData,
                                 const nvcv::TensorDataStridedCuda &outHistogram)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be an image-like planar tensor");
    }

    nvcv::TensorShape interleavedInShape = MakeHistogramLegacyInputShape(inData, *inAccess);

    nvcv::TensorShape maskShape;
    nvcv::DataType    maskDtype = nvcv::TYPE_U8;
    if (mask)
    {
        auto maskData = mask->get().exportData<nvcv::TensorDataStridedCuda>();
        if (maskData == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Mask must be cuda-accessible, pitch-linear tensor");
        }
        auto maskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskData);
        if (!maskAccess)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask must be an image-like planar tensor");
        }
        maskShape = MakeHistogramLegacyInputShape(*maskData, *maskAccess);
        maskDtype = maskData->dtype();
    }

    std::lock_guard lock(workspace.mutex);
    workspace.ensure(interleavedInShape, inData.dtype(), mask ? &maskShape : nullptr, maskDtype, stream);

    reformatOp(stream, in, workspace.interleavedIn);

    auto interleavedInData = workspace.interleavedIn.exportData<nvcv::TensorDataStridedCuda>();
    if (interleavedInData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Internal Histogram input tensor must be cuda-accessible");
    }

    if (!mask)
    {
        InferPlanarHistogramNoMask(stream, *interleavedInData, outHistogram);
        workspace.record(stream);
        return;
    }

    InferPlanarHistogramWithMask(reformatOp, workspace, stream, mask->get(), inData, *interleavedInData, outHistogram);
    workspace.record(stream);
}

inline void ValidateInterleavedMask(nvcv::OptionalTensorConstRef mask, nvcv::TensorLayout inputLayout)
{
    if (!mask)
    {
        return;
    }

    auto maskData = mask->get().exportData<nvcv::TensorDataStridedCuda>();
    if (maskData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Mask must be cuda-accessible, pitch-linear tensor");
    }
    if (!IsHistogramInterleavedLayout(maskData->layout()) || maskData->layout() != inputLayout)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask layout must match Histogram input layout");
    }
}

} // namespace

std::unique_ptr<HistogramPlanarBridgeWorkspace> Histogram::CreatePlanarBridgeWorkspace(int)
{
    return std::make_unique<HistogramPlanarBridgeWorkspace>();
}

Histogram::Histogram()
{
    m_reformatOp = std::make_unique<Reformat>();
}

Histogram::~Histogram() = default;

void Histogram::operator()(cudaStream_t stream, const nvcv::Tensor &in, nvcv::OptionalTensorConstRef mask,
                           const nvcv::Tensor &histogram) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Histogram::operator()[Tensor]");
    auto inData       = RequireCudaTensor(in, "Input");
    auto outHistogram = RequireCudaTensor(histogram, "Output");

    if (const bool inPlanar = IsHistogramPlanarLayout(inData->layout()); inPlanar)
    {
        InferPlanarHistogram(*m_reformatOp, m_planarWorkspace.get(), stream, in, mask, *inData, *outHistogram);
        return;
    }

    if (!IsHistogramInterleavedLayout(inData->layout()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input layout must be NHWC/HWC or single-channel NCHW/CHW");
    }

    ValidateInterleavedMask(mask, inData->layout());

    RunHistogram(*inData, mask, *outHistogram, stream);
}

} // namespace cvcuda::priv

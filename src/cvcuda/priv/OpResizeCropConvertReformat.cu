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

#include "OpResizeCropConvertReformat.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/cuda/DropCast.hpp>
#include <nvcv/cuda/InterpolationVarShapeWrap.hpp>
#include <nvcv/cuda/InterpolationWrap.hpp>
#include <nvcv/cuda/MathOps.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <util/Assert.h>
#include <util/Math.hpp>

#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace cuda_op = nvcv::legacy::cuda_op;
namespace helpers = nvcv::legacy::helpers;

namespace {

//******************** NN = Nearest Neighbor (TensorWrap src)

template<class SrcWrapper, class DstT, typename SrcT = typename SrcWrapper::ValueType>
__global__ void resizeCrop_NN(SrcWrapper src, DstT *dst, const int src_w, const int src_h, const int dst_w,
                              const int dst_h, const float scale_x, const float scale_y, const int crop_x,
                              const int crop_y, const size_t incrN, const size_t incrH, const size_t incrW,
                              const size_t incrC, const uchar4 mapC)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((dst_x < dst_w) && (dst_y < dst_h))
    { // Generic copy pixel to pixel.
        const int sample = blockIdx.z;

        dst += sample * incrN + dst_y * incrH + dst_x * incrW;

        const int sx = cuda::min(cuda::round<cuda::RoundMode::DOWN, int>((dst_x + crop_x) * scale_x), src_w - 1);
        const int sy = cuda::min(cuda::round<cuda::RoundMode::DOWN, int>((dst_y + crop_y) * scale_y), src_h - 1);

        SrcT v = *src.ptr(sample, sy, sx);

        // Channel manipulation, convert type, and reformat.
        dst[mapC.x * incrC] = (DstT)v.x;
        dst[mapC.y * incrC] = (DstT)v.y;
        dst[mapC.z * incrC] = (DstT)v.z;
    }
} // resizeCrop_NN

//******************** Bilinear (TensorWrap src)

template<class SrcWrapper, class DstT, typename SrcT = typename SrcWrapper::ValueType>
__global__ void resizeCrop_bilinear(SrcWrapper src, DstT *dst, const int src_w, const int src_h, const int dst_w,
                                    const int dst_h, const float scale_x, const float scale_y, const int crop_x,
                                    const int crop_y, const size_t incrN, const size_t incrH, const size_t incrW,
                                    const size_t incrC, const uchar4 mapC)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h)
    {
        const int sample = blockIdx.z;

        // Float space for weighted addition.
        // Compute y coordinate.
        float fy = (float)((dst_y + crop_y + 0.5f) * scale_y - 0.5f);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
        fy -= sy;
        sy = cuda::max(0, cuda::min(sy, src_h - 2));

        // Row pointers.
        const SrcT *aPtr = src.ptr(sample, sy, 0);     // Start of upper row.
        const SrcT *bPtr = src.ptr(sample, sy + 1, 0); // Start of lower row.

        dst += sample * incrN + dst_y * incrH + dst_x * incrW;

        { // Compute source data position and weight for [x0] components.
            float fx = (float)((dst_x + crop_x + 0.5f) * scale_x - 0.5f);
            int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
            fx -= sx;

            fx *= ((sx >= 0) && (sx < src_w - 1));
            sx = cuda::max(0, cuda::min(sx, src_w - 2));

            SrcT v = cuda::SaturateCast<SrcT>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                              + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
            // Channel manipulation, convert type, and reformat.
            dst[mapC.x * incrC] = (DstT)v.x;
            dst[mapC.y * incrC] = (DstT)v.y;
            dst[mapC.z * incrC] = (DstT)v.z;
        }
    }
} // resizeCrop_bilinear

//******************** NN = Nearest Neighbor (ImageBatchVarShape src)

template<class SrcWrapper, class DstT, typename SrcT = typename SrcWrapper::ValueType>
__global__ void resizeCrop_NN_varShape(SrcWrapper src, DstT *dst, const int dst_w, const int dst_h,
                                       const float resize_w, const float resize_h, const int crop_x, const int crop_y,
                                       const size_t incrN, const size_t incrH, const size_t incrW, const size_t incrC,
                                       const uchar4 mapC)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h)
    { // Generic copy pixel to pixel.
        const int sample = blockIdx.z;
        const int src_w  = src.width(sample);
        const int src_h  = src.height(sample);

        const float scale_x = static_cast<float>(src_w) / resize_w;
        const float scale_y = static_cast<float>(src_h) / resize_h;

        dst += sample * incrN + dst_y * incrH + dst_x * incrW;

        const int sx = cuda::min(cuda::round<cuda::RoundMode::DOWN, int>((dst_x + crop_x) * scale_x), src_w - 1);
        const int sy = cuda::min(cuda::round<cuda::RoundMode::DOWN, int>((dst_y + crop_y) * scale_y), src_h - 1);

        SrcT v = *src.ptr(sample, sy, sx);

        // Channel manipulation, convert type, and reformat.
        dst[mapC.x * incrC] = (DstT)v.x;
        dst[mapC.y * incrC] = (DstT)v.y;
        dst[mapC.z * incrC] = (DstT)v.z;
    }
} // resizeCrop_NN_varShape

//******************** Bilinear (ImageBatchVarShape src)

template<class SrcWrapper, class DstT, typename SrcT = typename SrcWrapper::ValueType>
__global__ void resizeCrop_bilinear_varShape(SrcWrapper src, DstT *dst, const int dst_w, const int dst_h,
                                             const float resize_w, const float resize_h, const int crop_x,
                                             const int crop_y, const size_t incrN, const size_t incrH,
                                             const size_t incrW, const size_t incrC, const uchar4 mapC)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((dst_x < dst_w) && (dst_y < dst_h))
    {
        const int sample = blockIdx.z;
        const int src_w  = src.width(sample);
        const int src_h  = src.height(sample);

        // Float space for weighted addition.
        float scale_x = static_cast<float>(src_w) / resize_w;
        float scale_y = static_cast<float>(src_h) / resize_h;

        // Compute y coordinate.
        float fy = (float)((dst_y + crop_y + 0.5f) * scale_y - 0.5f);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
        fy -= sy;
        sy = cuda::max(0, cuda::min(sy, src_h - 2));

        // Row pointers.
        const SrcT *aPtr = src.ptr(sample, sy, 0);     // Start of upper row.
        const SrcT *bPtr = src.ptr(sample, sy + 1, 0); // Start of lower row.

        dst += sample * incrN + dst_y * incrH + dst_x * incrW;

        { // Cimpute source data position and weight for [x0] components.
            float fx = (float)((dst_x + crop_x + 0.5f) * scale_x - 0.5f);
            int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
            fx -= sx;

            fx *= ((sx >= 0) && (sx < src_w - 1));
            sx = cuda::max(0, cuda::min(sx, src_w - 2));

            SrcT v = cuda::SaturateCast<SrcT>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                              + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
            // Channel manipulation, convert type, and reformat.
            dst[mapC.x * incrC] = (DstT)v.x;
            dst[mapC.y * incrC] = (DstT)v.y;
            dst[mapC.z * incrC] = (DstT)v.z;
        }
    }
} // resizeCrop_bilinear_varShape

#define MAP(m, i, v) ((uint8_t *)&(m))[i] = (v)

inline uchar4 remapChannels(const NVCVChannelManip manip, int channels)
{
    uchar4 map = make_uchar4(0, 1, 2, 3);

    if (manip == NVCV_CHANNEL_REVERSE)
    {
        for (int c = 0; c < channels; ++c) MAP(map, c, channels - c - 1);
    }
    return map;
}

#undef MAP

template<typename SrcT, typename DstT>
void resizeCropConvertReformat(const nvcv::TensorDataStridedCuda &srcData, const nvcv::TensorDataStridedCuda &dstData,
                               const NVCVSize2D resizeDim, NVCVInterpolationType interpolation, const int2 cropPos,
                               const NVCVChannelManip manip, cudaStream_t stream)

{
    using SrcBaseT = cuda::BaseType<SrcT>;
    using DstBaseT = cuda::BaseType<DstT>;

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    NVCV_ASSERT(srcAccess);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    NVCV_ASSERT(dstAccess);

    const int samples  = srcAccess->numSamples();
    const int channels = srcAccess->numChannels();
    const int src_w    = srcAccess->numCols();
    const int src_h    = srcAccess->numRows();
    const int dst_w    = dstAccess->numCols();
    const int dst_h    = dstAccess->numRows();

    NVCV_ASSERT(samples == dstAccess->numSamples());
    NVCV_ASSERT(channels == dstAccess->numChannels());

    float scale_x = (float)src_w / resizeDim.w;
    float scale_y = (float)src_h / resizeDim.h;

    const int planes = dstAccess->numPlanes();

    const uchar4 remap = remapChannels(manip, channels);

    const size_t incrC = (planes > 1 ? dstAccess->planeStride() / sizeof(DstBaseT) : 1);
    const size_t incrW = channels / planes; // 1 if planar; channels if not.
    const size_t incrH = dstAccess->rowStride() / sizeof(DstBaseT);
    const size_t incrN = dstAccess->rowStride() * dst_h * dstAccess->numPlanes() / sizeof(DstBaseT);

    const int THREADS_PER_BLOCK = 256; //256?  64?
    const int BLOCK_WIDTH       = 16;  //as in 32x4 or 32x8.  16x8 and 16x16 are also viable

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(util::DivUp(dst_w, blockSize.x), util::DivUp(dst_h, blockSize.y), samples);

    auto src = cuda::CreateTensorWrapNHW<const SrcT>(srcData);

    DstBaseT *dst = reinterpret_cast<DstBaseT *>(dstData.basePtr());

    //Note: resize is fundamentally a gather memory operation, with a little bit of compute
    //      our goals are to (a) maximize throughput, and (b) minimize occupancy for the same performance

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        resizeCrop_NN<<<gridSize, blockSize, 0, stream>>>(src, dst, src_w, src_h, dst_w, dst_h, scale_x, scale_y,
                                                          cropPos.x, cropPos.y, incrN, incrH, incrW, incrC, remap);
        break;

    case NVCV_INTERP_LINEAR:
        resizeCrop_bilinear<<<gridSize, blockSize, 0, stream>>>(src, dst, src_w, src_h, dst_w, dst_h, scale_x, scale_y,
                                                                cropPos.x, cropPos.y, incrN, incrH, incrW, incrC,
                                                                remap);
        break;

    case NVCV_INTERP_CUBIC:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Interpolation not implemented: NVCV_INTERP_CUBIC");
        break;

    case NVCV_INTERP_AREA:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Interpolation not implemented: NVCV_INTERP_AREA");
        break;

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation");
        break;
    } //switch
} //resize

template<typename SrcT, typename DstT>
void resizeCropConvertReformat(const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                               const nvcv::TensorDataStridedCuda &dstData, const NVCVSize2D resizeDim,
                               const NVCVInterpolationType interpolation, const int2 cropPos,
                               const NVCVChannelManip manip, cudaStream_t stream)
{
    using SrcBaseT = cuda::BaseType<SrcT>;
    using DstBaseT = cuda::BaseType<DstT>;

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    NVCV_ASSERT(dstAccess);

    const nvcv::ImageFormat srcFrmt = srcData.uniqueFormat();
    NVCV_ASSERT(srcFrmt);

    const int samples  = srcData.numImages();
    const int channels = srcFrmt.numChannels();
    const int dst_w    = dstAccess->numCols();
    const int dst_h    = dstAccess->numRows();

    NVCV_ASSERT(samples == dstAccess->numSamples());
    NVCV_ASSERT(channels == dstAccess->numChannels());

    const int    planes = dstAccess->numPlanes();
    const uchar4 remap  = remapChannels(manip, channels);

    const size_t incrC = (planes > 1 ? dstAccess->planeStride() / sizeof(DstBaseT) : 1);
    const size_t incrW = channels / planes; // 1 if planar; channels if not.
    const size_t incrH = dstAccess->rowStride() / sizeof(DstBaseT);
    const size_t incrN = dstAccess->rowStride() * dst_h * dstAccess->numPlanes() / sizeof(DstBaseT);

    const int THREADS_PER_BLOCK = 256; //Performance degrades above 256 and below 16 (GMEM speed limited)
    const int BLOCK_WIDTH       = 8;   //as in 32x4 or 32x8 or 8x32.

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(util::DivUp(dst_w, blockSize.x), util::DivUp(dst_h, blockSize.y), samples);

    cuda::ImageBatchVarShapeWrap<const SrcT> src(srcData);

    DstBaseT *dst = reinterpret_cast<DstBaseT *>(dstData.basePtr());

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        resizeCrop_NN_varShape<<<gridSize, blockSize, 0, stream>>>(
            src, dst, dst_w, dst_h, resizeDim.w, resizeDim.h, cropPos.x, cropPos.y, incrN, incrH, incrW, incrC, remap);
        break;

    case NVCV_INTERP_LINEAR:
        resizeCrop_bilinear_varShape<<<gridSize, blockSize, 0, stream>>>(
            src, dst, dst_w, dst_h, resizeDim.w, resizeDim.h, cropPos.x, cropPos.y, incrN, incrH, incrW, incrC, remap);
        break;

    case NVCV_INTERP_CUBIC:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Interpolation not implemented: NVCV_INTERP_CUBIC");
        break;

    case NVCV_INTERP_AREA:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Interpolation not implemented: NVCV_INTERP_AREA");
        break;

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation");
        break;

    } //switch interpolation
}

} // anonymous namespace

// clang-format off
namespace cvcuda::priv {
ResizeCropConvertReformat::ResizeCropConvertReformat() { }

// clang-format on

void ResizeCropConvertReformat::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                           const NVCVSize2D resizeDim, const NVCVInterpolationType interpolation,
                                           const int2 cropPos, const NVCVChannelManip manip) const
{
    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be a cuda-accessible, pitch-linear tensor");
    }

    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be a cuda-accessible, pitch-linear tensor");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);

    const int samples  = srcAccess->numSamples();
    const int channels = srcAccess->numChannels();

    const int dst_w = dstAccess->numCols();
    const int dst_h = dstAccess->numRows();

    if (samples != dstAccess->numSamples())
    {
        std::string msg = "Input and output must have the same batch size (i.e., same number of images): Provided "
                        + std::to_string(samples) + " input and " + std::to_string(dstAccess->numSamples())
                        + " output images / samples";
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", msg.c_str());
    }

    if (channels != dstAccess->numChannels())
    {
        std::string msg = "Input and output must have same number of channels: Provided " + std::to_string(channels)
                        + " input and " + std::to_string(dstAccess->numChannels()) + " output channels";
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    if (channels != 3)
    {
        std::string msg = "Only three-channel input is currently supported: Provided " + std::to_string(channels)
                        + " input channels";
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    cuda_op::DataType srcType = helpers::GetLegacyDataType((*srcData).dtype());
    cuda_op::DataType dstType = helpers::GetLegacyDataType((*dstData).dtype());

    if (srcType != cuda_op::kCV_8U)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "Input must be of data type uchar.");
    }

    if (dstType != cuda_op::kCV_8U && dstType != cuda_op::kCV_32F)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "Output must be of data type uchar or float.");
    }

    nvcv::TensorLayout srcLayout = srcData->layout();
    nvcv::TensorLayout dstLayout = dstData->layout();

    if (!(srcLayout == NVCV_TENSOR_NHWC || srcLayout == NVCV_TENSOR_HWC))
    {
        const char *layout = nvcvTensorLayoutGetName(&srcLayout.m_layout);
        std::string msg    = "Input tensor must have 'NHWC' or 'HWC' layout: Layout provided " + std::string(layout);
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    if (!(dstLayout == NVCV_TENSOR_NHWC || dstLayout == NVCV_TENSOR_HWC || dstLayout == NVCV_TENSOR_NCHW
          || dstLayout == NVCV_TENSOR_CHW))
    {
        const char *layout = nvcvTensorLayoutGetName(&dstLayout.m_layout);
        std::string msg
            = "Output tensor must have 'NHWC', 'NCHW', 'HWC', or 'CHW' layout: Layout provided " + std::string(layout);
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    if (cropPos.x < 0 || cropPos.y < 0 || cropPos.x + dst_w > resizeDim.w || cropPos.y + dst_h > resizeDim.h)
    {
        std::string msg = "Invalid crop region: crop region(x, y, w, h) = (" + std::to_string(cropPos.x) + ", "
                        + std::to_string(cropPos.y) + ", " + std::to_string(dst_w) + ", " + std::to_string(dst_h)
                        + ") extends beyond bounds of resized tensor";
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", msg.c_str());
    }

    if (srcType == cuda_op::kCV_8U)
    {
        if (dstType == cuda_op::kCV_8U)
        {
            resizeCropConvertReformat<uchar3, uint8_t>(*srcData, *dstData, resizeDim, interpolation, cropPos, manip,
                                                       stream);
        }
        else if (dstType == cuda_op::kCV_32F)
        {
            resizeCropConvertReformat<uchar3, float>(*srcData, *dstData, resizeDim, interpolation, cropPos, manip,
                                                     stream);
        }
    }
}

void ResizeCropConvertReformat::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                           const nvcv::Tensor &dst, const NVCVSize2D resizeDim,
                                           const NVCVInterpolationType interpolation, const int2 cropPos,
                                           const NVCVChannelManip manip) const
{
    auto srcData = src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input data must be a cuda-accessible, varshape image batch");
    }

    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be a cuda-accessible, pitch-linear tensor");
    }

    const nvcv::ImageFormat srcFrmt = src.uniqueFormat();

    if (!srcFrmt)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "All input images in a batch must have the same format (including number of channels)");
    }

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);

    const int samples  = srcData->numImages();
    const int channels = srcFrmt.numChannels();

    const int dst_w = dstAccess->numCols();
    const int dst_h = dstAccess->numRows();

    if (samples != dstAccess->numSamples())
    {
        std::string msg = "Input and output must have the same batch size (i.e., same number of images): Provided "
                        + std::to_string(samples) + " input and " + std::to_string(dstAccess->numChannels())
                        + " output channels";
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", msg.c_str());
    }

    if (channels != dstAccess->numChannels())
    {
        std::string msg = "Input and output must have same number of channels: Provided " + std::to_string(channels)
                        + " input and " + std::to_string(dstAccess->numChannels()) + " output channels";
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    if (channels != 3)
    {
        std::string msg = "Only three-channel input is currently supported: Provided " + std::to_string(channels)
                        + " input channels";
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    cuda_op::DataType srcType = helpers::GetLegacyDataType(srcFrmt);
    cuda_op::DataType dstType = helpers::GetLegacyDataType((*dstData).dtype());

    if (srcType != cuda_op::kCV_8U)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "Input must be of data type uchar.");
    }

    if (dstType != cuda_op::kCV_8U && dstType != cuda_op::kCV_32F)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "Output must be of data type uchar or float.");
    }

    nvcv::TensorLayout dstLayout = dstData->layout();

    if (srcFrmt.numPlanes() > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "Input must be non-planar (i.e., interleaved).");
    }

    if (!(dstLayout == NVCV_TENSOR_NHWC || dstLayout == NVCV_TENSOR_HWC || dstLayout == NVCV_TENSOR_NCHW
          || dstLayout == NVCV_TENSOR_CHW))
    {
        const char *layout = nvcvTensorLayoutGetName(&dstLayout.m_layout);
        std::string msg
            = "Output tensor must have 'NHWC', 'NCHW', 'HWC', or 'CHW' layout: Layout provided " + std::string(layout);
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    if (cropPos.x < 0 || cropPos.y < 0 || cropPos.x + dst_w > resizeDim.w || cropPos.y + dst_h > resizeDim.h)
    {
        std::string msg = "Invalid crop region: crop region(x, y, w, h) = (" + std::to_string(cropPos.x) + ", "
                        + std::to_string(cropPos.y) + ", " + std::to_string(dst_w) + ", " + std::to_string(dst_h)
                        + ") extends beyond bounds of resized tensor";
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", msg.c_str());
    }

    if (srcType == cuda_op::kCV_8U)
    {
        if (dstType == cuda_op::kCV_8U)
        {
            resizeCropConvertReformat<uchar3, uint8_t>(*srcData, *dstData, resizeDim, interpolation, cropPos, manip,
                                                       stream);
        }
        else if (dstType == cuda_op::kCV_32F)
        {
            resizeCropConvertReformat<uchar3, float>(*srcData, *dstData, resizeDim, interpolation, cropPos, manip,
                                                     stream);
        }
    }
}
} // namespace cvcuda::priv

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpCropFlipNormalizeReformat.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/cuda/BorderVarShapeWrap.hpp>
#include <nvcv/cuda/ImageBatchVarShapeWrap.hpp>
#include <nvcv/cuda/MathOps.hpp>
#include <nvcv/cuda/SaturateCast.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <nvcv/cuda/TensorWrap.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <util/CheckError.hpp>

#include <iostream>
#include <sstream>

namespace cuda = nvcv::cuda;

template<class TensorWrapper>
__device__ float get_base_value(TensorWrapper data, int ch, int numChannels)
{
    const int ch_idx = numChannels == 1 ? 0 : ch;
    return *data.ptr(0, 0, 0, ch_idx);
}

template<class TensorWrapper>
__device__ float get_scale_value(TensorWrapper data, int ch, int numChannels, float epsilon, bool scaleIsStdDev)
{
    const int ch_idx = numChannels == 1 ? 0 : ch;
    float     s      = *data.ptr(0, 0, 0, ch_idx);
    if (scaleIsStdDev)
    {
        float x   = s * s + epsilon;
        float mul = 1.0f / sqrt(x);
        return mul;
    }
    else
    {
        return s;
    }
}

template<class T1, NVCVBorderType B, class T2, class TensorWrapper>
__device__ void transfer_data(cuda::BorderVarShapeWrap<const T1, B> srcWrap, cuda::Tensor4DWrap<T2> dstWrap,
                              int2 src_idx, int2 dst_idx, int batchidx, int ch, TensorWrapper baseWrap,
                              TensorWrapper scaleWrap, float global_scale, float global_shift, float epsilon,
                              uint32_t flags, int base_channels, int scale_channels, bool dst_planar)
{
    if (dst_planar)
    {
        for (int c = 0; c < ch; c++)
        {
            float base  = get_base_value(baseWrap, c, base_channels);
            float scale = get_scale_value(scaleWrap, c, scale_channels, epsilon, flags);
            dstWrap[(int4){dst_idx.x, dst_idx.y, c, batchidx}] = cuda::SaturateCast<T2>(
                (srcWrap[(int4){src_idx.x, src_idx.y, c, batchidx}] - base) * scale * global_scale + global_shift);
        }
    }
    else
    {
        for (int c = 0; c < ch; c++)
        {
            float base  = get_base_value(baseWrap, c, base_channels);
            float scale = get_scale_value(scaleWrap, c, scale_channels, epsilon, flags);
            dstWrap[(int4){c, dst_idx.x, dst_idx.y, batchidx}] = cuda::SaturateCast<T2>(
                (srcWrap[(int4){src_idx.x, src_idx.y, c, batchidx}] - base) * scale * global_scale + global_shift);
        }
    }
}

template<class T1, NVCVBorderType B, class T2, class TensorWrapper>
__device__ void transfer_data(cuda::BorderVarShapeWrapNHWC<const T1, B> srcWrap, cuda::Tensor4DWrap<T2> dstWrap,
                              int2 src_idx, int2 dst_idx, int batchidx, int ch, TensorWrapper baseWrap,
                              TensorWrapper scaleWrap, float global_scale, float global_shift, float epsilon,
                              uint32_t flags, int base_channels, int scale_channels, bool dst_planar)
{
    if (dst_planar)
    {
        for (int c = 0; c < ch; c++)
        {
            float base  = get_base_value(baseWrap, c, base_channels);
            float scale = get_scale_value(scaleWrap, c, scale_channels, epsilon, flags);
            dstWrap[(int4){dst_idx.x, dst_idx.y, c, batchidx}] = cuda::SaturateCast<T2>(
                (srcWrap[(int4){src_idx.x, src_idx.y, batchidx, c}] - base) * scale * global_scale + global_shift);
        }
    }
    else
    {
        for (int c = 0; c < ch; c++)
        {
            float base  = get_base_value(baseWrap, c, base_channels);
            float scale = get_scale_value(scaleWrap, c, scale_channels, epsilon, flags);
            dstWrap[(int4){c, dst_idx.x, dst_idx.y, batchidx}] = cuda::SaturateCast<T2>(
                (srcWrap[(int4){src_idx.x, src_idx.y, batchidx, c}] - base) * scale * global_scale + global_shift);
        }
    }
}

template<class T, class T2>
__device__ void set_data(cuda::Tensor4DWrap<T> dstWrap, int2 dst_idx, int batchidx, int ch, T2 val, bool dst_planar)
{
    if (dst_planar)
    {
        for (int c = 0; c < ch; c++)
        {
            dstWrap[(int4){dst_idx.x, dst_idx.y, c, batchidx}] = cuda::StaticCast<T>(val);
        }
    }
    else
    {
        for (int c = 0; c < ch; c++)
        {
            dstWrap[(int4){c, dst_idx.x, dst_idx.y, batchidx}] = cuda::StaticCast<T>(val);
        }
    }
}

template<class SrcWrapper, class DstWrapper, class TensorWrapper, class FlipWrapper, class CropRectWrap>
__global__ void slice_flip_normalize(SrcWrapper srcWrap, DstWrapper dstWrap, FlipWrapper flipCodeWrap,
                                     TensorWrapper baseWrap, TensorWrapper scaleWrap, CropRectWrap cropRect,
                                     float global_scale, float global_shift, float epsilon, uint32_t flags,
                                     int input_channels, int base_ch, int scale_ch, int3 out_size, bool dst_planar)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    int4 *crop_ptr    = reinterpret_cast<int4 *>(cropRect.ptr(dstCoord.z, 0, 0, 0));
    int   crop_val[4] = {(*crop_ptr).x, (*crop_ptr).y, (*crop_ptr).z,
                         (*crop_ptr).w}; // crop_x, crop_y, crop_w, crop_h stored in (x,y,z,w)

    if (dstCoord.x >= out_size.x || dstCoord.y >= out_size.y)
    {
        return;
    }
    // out of crop width and height
    if (dstCoord.x >= crop_val[2] || dstCoord.y >= crop_val[3])
    {
        set_data(dstWrap, (int2){dstCoord.x, dstCoord.y}, blockIdx.z, input_channels, 0, dst_planar);
        return;
    }

    int  flip_code = flipCodeWrap[dstCoord.z];
    int2 srcCoord;

    if (flip_code == 1)
    { // horizental
        srcCoord.x = crop_val[2] - 1 - dstCoord.x + crop_val[0];
        srcCoord.y = dstCoord.y + crop_val[1];
    }
    else if (flip_code == 0)
    { // vertical
        srcCoord.x = dstCoord.x + crop_val[0];
        srcCoord.y = crop_val[3] - 1 - dstCoord.y + crop_val[1];
    }
    else if (flip_code == -1)
    { // horizental + vertical
        srcCoord.x = crop_val[2] - 1 - dstCoord.x + crop_val[0];
        srcCoord.y = crop_val[3] - 1 - dstCoord.y + crop_val[1];
    }
    else
    { // no flip
        srcCoord.x = dstCoord.x + crop_val[0];
        srcCoord.y = dstCoord.y + crop_val[1];
    }

    transfer_data(srcWrap, dstWrap, srcCoord, {dstCoord.x, dstCoord.y}, dstCoord.z, input_channels, baseWrap, scaleWrap,
                  global_scale, global_shift, epsilon, flags, base_ch, scale_ch, dst_planar);
}

template<class T_Src, class T_Dst, NVCVBorderType B>
void RunCropFlipNormalizeReformat(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                                  const nvcv::TensorDataStridedCuda &dstData,
                                  const nvcv::TensorDataStridedCuda &flipCodeData,
                                  const nvcv::TensorDataStridedCuda &baseData,
                                  const nvcv::TensorDataStridedCuda &scaleData, const float borderValue,
                                  const nvcv::TensorDataStridedCuda &cropRect, float global_scale, float shift,
                                  float epsilon, uint32_t flags, int channel)
{
    int  num_channels = srcData.uniqueFormat().numChannels();
    auto outAccess    = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    NVCV_ASSERT(outAccess);
    const int3 out_size = {outAccess->numCols(), outAccess->numRows(), num_channels};

    nvcv::TensorLayout dst_layout = dstData.layout();
    if (!(dst_layout == nvcv::TENSOR_NHWC || dst_layout == nvcv::TENSOR_NCHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid tensor layout in output");
    }

    bool src_planar = srcData.uniqueFormat().numPlanes() > 1;
    bool dst_planar = dst_layout == nvcv::TENSOR_NCHW;

    nvcv::Size2D maxSize   = {outAccess->numCols(), outAccess->numRows()};
    int32_t      batchSize = srcData.numImages();
    dim3         block(32, 32, 1);
    dim3 grid(std::ceil(maxSize.w / static_cast<float>(block.x)), std::ceil(maxSize.h / static_cast<float>(block.y)),
              batchSize);

    auto baseAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(baseData);
    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(scaleData);

    int scale_channels = scaleAccess->numChannels();
    int base_channels  = baseAccess->numChannels();

    auto                    cropRectWrap = cuda::CreateTensorWrapNHWC<int>(cropRect);
    cuda::Tensor1DWrap<int> flipCodeWrap(flipCodeData);
    auto                    baseWrap  = cuda::CreateTensorWrapNHWC<float>(baseData);
    auto                    scaleWrap = cuda::CreateTensorWrapNHWC<float>(scaleData);

    if (src_planar && dst_planar)
    {
        cuda::ImageBatchVarShapeWrap<const T_Src> srcWrap(srcData); // planar
        cuda::BorderVarShapeWrap<const T_Src, B>  srcBorderWrap(srcWrap, static_cast<T_Src>(borderValue));
        cuda::Tensor4DWrap<T_Dst>                 dstWrap(dstData); // planar
        slice_flip_normalize<<<grid, block, 0, stream>>>(srcBorderWrap, dstWrap, flipCodeWrap, baseWrap, scaleWrap,
                                                         cropRectWrap, global_scale, shift, epsilon, flags, channel,
                                                         base_channels, scale_channels, out_size, dst_planar);
    }
    else if (src_planar)
    {
        cuda::ImageBatchVarShapeWrap<const T_Src> srcWrap(srcData); // planar
        cuda::BorderVarShapeWrap<const T_Src, B>  srcBorderWrap(srcWrap, static_cast<T_Src>(borderValue));
        auto                                      dstWrap = cuda::CreateTensorWrapNHWC<T_Dst>(dstData); // interleaved
        slice_flip_normalize<<<grid, block, 0, stream>>>(srcBorderWrap, dstWrap, flipCodeWrap, baseWrap, scaleWrap,
                                                         cropRectWrap, global_scale, shift, epsilon, flags, channel,
                                                         base_channels, scale_channels, out_size, dst_planar);
    }
    else if (dst_planar)
    {
        cuda::ImageBatchVarShapeWrapNHWC<const T_Src> srcWrap(srcData, channel); // interleaved
        cuda::BorderVarShapeWrapNHWC<const T_Src, B>  srcBorderWrap(srcWrap, static_cast<T_Src>(borderValue));
        cuda::Tensor4DWrap<T_Dst>                     dstWrap(dstData); // planar
        slice_flip_normalize<<<grid, block, 0, stream>>>(srcBorderWrap, dstWrap, flipCodeWrap, baseWrap, scaleWrap,
                                                         cropRectWrap, global_scale, shift, epsilon, flags, channel,
                                                         base_channels, scale_channels, out_size, dst_planar);
    }
    else
    {
        cuda::ImageBatchVarShapeWrapNHWC<const T_Src> srcWrap(srcData, channel); // interleaved
        cuda::BorderVarShapeWrapNHWC<const T_Src, B>  srcBorderWrap(srcWrap, static_cast<T_Src>(borderValue));
        auto dstWrap = cuda::CreateTensorWrapNHWC<T_Dst>(dstData); // interleaved
        slice_flip_normalize<<<grid, block, 0, stream>>>(srcBorderWrap, dstWrap, flipCodeWrap, baseWrap, scaleWrap,
                                                         cropRectWrap, global_scale, shift, epsilon, flags, channel,
                                                         base_channels, scale_channels, out_size, dst_planar);
    }
}

template<class T_Src, class T_Dst>
void RunCropFlipNormalizeReformat(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                                  const nvcv::TensorDataStridedCuda &dstData,
                                  const nvcv::TensorDataStridedCuda &flipCodeData,
                                  const nvcv::TensorDataStridedCuda &baseData,
                                  const nvcv::TensorDataStridedCuda &scaleData, const NVCVBorderType borderMode,
                                  const float borderValue, const nvcv::TensorDataStridedCuda &cropRect,
                                  float global_scale, float shift, float epsilon, uint32_t flags, int channel)
{
    typedef void (*func_t)(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                           const nvcv::TensorDataStridedCuda &dstData, const nvcv::TensorDataStridedCuda &flipCodeData,
                           const nvcv::TensorDataStridedCuda &baseData, const nvcv::TensorDataStridedCuda &scaleData,
                           const float borderValue, const nvcv::TensorDataStridedCuda &cropRect, float global_scale,
                           float shift, float epsilon, uint32_t flags, int channel);

    static const func_t funcs[5] = {RunCropFlipNormalizeReformat<T_Src, T_Dst, NVCV_BORDER_CONSTANT>,
                                    RunCropFlipNormalizeReformat<T_Src, T_Dst, NVCV_BORDER_REPLICATE>,
                                    RunCropFlipNormalizeReformat<T_Src, T_Dst, NVCV_BORDER_REFLECT>,
                                    RunCropFlipNormalizeReformat<T_Src, T_Dst, NVCV_BORDER_WRAP>,
                                    RunCropFlipNormalizeReformat<T_Src, T_Dst, NVCV_BORDER_REFLECT101>};

    const func_t func = funcs[(int)borderMode];
    if (func == 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported border mode");
    }
    func(stream, srcData, dstData, flipCodeData, baseData, scaleData, borderValue, cropRect, global_scale, shift,
         epsilon, flags, channel);
}

template<class T_Src>
void RunCropFlipNormalizeReformat(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                                  const nvcv::TensorDataStridedCuda &dstData,
                                  const nvcv::TensorDataStridedCuda &flipCodeData,
                                  const nvcv::TensorDataStridedCuda &baseData,
                                  const nvcv::TensorDataStridedCuda &scaleData, const NVCVBorderType borderMode,
                                  const float borderValue, const nvcv::TensorDataStridedCuda &cropRect,
                                  float global_scale, float shift, float epsilon, uint32_t flags, int channel)
{
    typedef void (*func_t)(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                           const nvcv::TensorDataStridedCuda &dstData, const nvcv::TensorDataStridedCuda &flipCodeData,
                           const nvcv::TensorDataStridedCuda &baseData, const nvcv::TensorDataStridedCuda &scaleData,
                           const NVCVBorderType borderMode, const float borderValue,
                           const nvcv::TensorDataStridedCuda &cropRect, float global_scale, float shift, float epsilon,
                           uint32_t flags, int channel);

    static const func_t funcs[3][4] = {
        {RunCropFlipNormalizeReformat<T_Src, unsigned char>, RunCropFlipNormalizeReformat<T_Src,                    unsigned short>,      0,
         RunCropFlipNormalizeReformat<T_Src, unsigned int>},
        {RunCropFlipNormalizeReformat<T_Src,          char>, RunCropFlipNormalizeReformat<T_Src,                             short>,      0,
         RunCropFlipNormalizeReformat<T_Src, int>},
        {                                 0,              0,                                  0, RunCropFlipNormalizeReformat<T_Src, float> }
    };
    nvcv::DataType datatype = dstData.dtype();

    nvcv::DataKind data_kind = datatype.dataKind();
    auto           bpc       = datatype.bitsPerChannel();
    if (data_kind == nvcv::DataKind::COMPLEX)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output DataKind must be unsigned, signed or float");
    }

    const func_t func = funcs[(int)data_kind][(bpc[0] >> 3) - 1];
    if (func == 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported output data type");
    }

    func(stream, srcData, dstData, flipCodeData, baseData, scaleData, borderMode, borderValue, cropRect, global_scale,
         shift, epsilon, flags, channel);
}

namespace cvcuda::priv {

CropFlipNormalizeReformat::CropFlipNormalizeReformat() {}

void CropFlipNormalizeReformat::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                           const nvcv::Tensor &out, const nvcv::Tensor &cropRect,
                                           const NVCVBorderType borderMode, const float borderValue,
                                           const nvcv::Tensor &flipCode, const nvcv::Tensor &base,
                                           const nvcv::Tensor &scale, float global_scale, float shift, float epsilon,
                                           uint32_t flags) const
{
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
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

    auto flipCodeData = flipCode.exportData<nvcv::TensorDataStridedCuda>();
    if (flipCodeData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input flipCode must be cuda-accessible, pitch-linear tensor");
    }

    auto baseData = base.exportData<nvcv::TensorDataStridedCuda>();
    if (baseData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input base must be cuda-accessible, pitch-linear tensor");
    }

    auto scaleData = scale.exportData<nvcv::TensorDataStridedCuda>();
    if (scaleData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input scale must be cuda-accessible, pitch-linear tensor");
    }

    auto cropRectData = cropRect.exportData<nvcv::TensorDataStridedCuda>();
    if (cropRectData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input cropRect must be cuda-accessible, pitch-linear tensor");
    }

    nvcv::ImageFormat fmt       = inData->uniqueFormat();
    int32_t           channels  = fmt.numChannels();
    int32_t           planes    = fmt.numPlanes();
    nvcv::DataKind    data_kind = fmt.dataKind();
    auto              bpc       = fmt.bitsPerChannel();

    auto baseAccess     = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
    auto scaleAccess    = nvcv::TensorDataAccessStridedImagePlanar::Create(*scaleData);
    auto cropRectAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*cropRectData);

    int scale_channels = scaleAccess->numChannels();
    int base_channels  = baseAccess->numChannels();

    NVCV_ASSERT(baseAccess && scaleAccess && cropRectAccess);

    if (!(base_channels == 1 || (base_channels >= channels)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Channel of base tensor must be 1 or >= input/output tensor channels");
    }

    if (!(scale_channels == 1 || (scale_channels >= channels)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Channel of scale tensor must be 1 or >= input/output tensor channels");
    }

    if (data_kind == nvcv::DataKind::COMPLEX)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input DataKind must be unsigned, signed or float");
    }
    typedef void (*func_t)(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                           const nvcv::TensorDataStridedCuda &dstData, const nvcv::TensorDataStridedCuda &flipCodeData,
                           const nvcv::TensorDataStridedCuda &baseData, const nvcv::TensorDataStridedCuda &scaleData,
                           const NVCVBorderType borderMode, const float borderValue,
                           const nvcv::TensorDataStridedCuda &cropRect, float global_scale, float shift, float epsilon,
                           uint32_t flags, int channel);

    static const func_t funcs[3][4] = {
        {RunCropFlipNormalizeReformat<unsigned char>, RunCropFlipNormalizeReformat<unsigned short>, 0,
         RunCropFlipNormalizeReformat<unsigned int>                                                                                       },
        {         RunCropFlipNormalizeReformat<char>,          RunCropFlipNormalizeReformat<short>, 0,   RunCropFlipNormalizeReformat<int>},
        {                                          0,                                            0, 0, RunCropFlipNormalizeReformat<float>}
    };

    const func_t func = funcs[(int)data_kind][(bpc[0] >> 3) - 1];
    if (func == 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input data type");
    }

    func(stream, *inData, *outData, *flipCodeData, *baseData, *scaleData, borderMode, borderValue, *cropRectData,
         global_scale, shift, epsilon, flags, channels);
}

} // namespace cvcuda::priv

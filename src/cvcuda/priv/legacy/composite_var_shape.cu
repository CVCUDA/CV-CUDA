/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

using namespace nvcv;
using namespace nvcv::legacy::helpers;
using namespace nvcv::legacy::cuda_op;

#define Inv_255              0.00392156862f // 1.f/255.f
#define AlphaLerp(c0, c1, a) int(((int)c1 - (int)c0) * (int)a * Inv_255 + c0 + 0.5f)

template<typename T, typename U, typename D>
__global__ void composite_kernel(const Ptr2dVarShapeNHWC<T> fg, const Ptr2dVarShapeNHWC<T> bg,
                                 const Ptr2dVarShapeNHWC<U> fgMask, Ptr2dVarShapeNHWC<D> dst)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    int dst_ch = dst.nch;
    int src_ch = fg.nch;

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

template<typename T, int scn, int dcn> // uchar
void composite(const nvcv::IImageBatchVarShapeDataStridedCuda &foregroundData,
               const nvcv::IImageBatchVarShapeDataStridedCuda &backgroundData,
               const nvcv::IImageBatchVarShapeDataStridedCuda &fgMaskData,
               const nvcv::IImageBatchVarShapeDataStridedCuda &outData, cudaStream_t stream)
{
    typedef typename cuda::MakeType<T, scn> src_type;
    typedef typename cuda::MakeType<T, dcn> dst_type;

    Ptr2dVarShapeNHWC<src_type> fg_ptr(foregroundData);
    Ptr2dVarShapeNHWC<src_type> bg_ptr(backgroundData);
    Ptr2dVarShapeNHWC<T>        fgMask_ptr(fgMaskData);
    Ptr2dVarShapeNHWC<dst_type> dst_ptr(outData);

    const int batch_size = dst_ptr.batches;
    Size2D    outMaxSize = outData.maxSize();

    dim3 blockSize(16, 16, 1);
    dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y), batch_size);

    composite_kernel<<<gridSize, blockSize, 0, stream>>>(fg_ptr, bg_ptr, fgMask_ptr, dst_ptr);
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nvcv::legacy::cuda_op {

ErrorCode CompositeVarShape::infer(const IImageBatchVarShapeDataStridedCuda &foreground,
                                   const IImageBatchVarShapeDataStridedCuda &background,
                                   const IImageBatchVarShapeDataStridedCuda &fgMask,
                                   const IImageBatchVarShapeDataStridedCuda &outData, cudaStream_t stream)
{
    DataFormat background_format = helpers::GetLegacyDataFormat(background);
    DataFormat foreground_format = helpers::GetLegacyDataFormat(foreground);
    DataFormat fgMask_format     = helpers::GetLegacyDataFormat(fgMask);
    DataFormat output_format     = helpers::GetLegacyDataFormat(outData);

    if (!((foreground_format == background_format) && (foreground_format == fgMask_format)
          && (foreground_format == output_format)))
    {
        LOG_ERROR("Invalid DataFormat between foreground ("
                  << foreground_format << "), background (" << background_format << "), foreground mask ("
                  << fgMask_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(foreground.uniqueFormat() && background.uniqueFormat() && fgMask.uniqueFormat() && outData.uniqueFormat()))
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = foreground_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType foreground_data_type = helpers::GetLegacyDataType(foreground.uniqueFormat());
    DataType background_data_type = helpers::GetLegacyDataType(background.uniqueFormat());
    DataType fgMask_data_type     = helpers::GetLegacyDataType(fgMask.uniqueFormat());
    DataType output_data_type     = helpers::GetLegacyDataType(outData.uniqueFormat());

    if (!((foreground_data_type == kCV_8U) && (background_data_type == kCV_8U) && (fgMask_data_type == kCV_8U)
          && (output_data_type == kCV_8U)))
    {
        LOG_ERROR("Invalid DataType " << foreground_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int foreground_channels = foreground.uniqueFormat().numChannels();
    const int background_channels = background.uniqueFormat().numChannels();
    const int fgMask_channels     = fgMask.uniqueFormat().numChannels();
    const int output_channels     = outData.uniqueFormat().numChannels();
    const int channels            = output_channels;

    if (!((foreground_channels == 3) && (background_channels == 3) && (fgMask_channels == 1)
          && (output_channels == 3 || output_channels == 4)))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    typedef void (*func_t)(const nvcv::IImageBatchVarShapeDataStridedCuda &foregroundData,
                           const nvcv::IImageBatchVarShapeDataStridedCuda &backgroundData,
                           const nvcv::IImageBatchVarShapeDataStridedCuda &fgMaskData,
                           const nvcv::IImageBatchVarShapeDataStridedCuda &outData, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        { 0 /*composite<uchar,1,1>*/,  0 /*composite<uchar,2,2>*/,             composite<uchar,3, 3>, composite<uchar, 3, 4>},
        { 0 /*composite<schar,1,1>*/,  0 /*composite<schar,2,2>*/,  0 /*composite<schar,3,3>*/,
         0 /*composite<schar,3,4>*/   },
        {0 /*composite<ushort,1,1>*/, 0 /*composite<ushort,2,2>*/, 0 /*composite<ushort,3,3>*/,
         0 /*composite<ushort,3,4>*/   },
        { 0 /*composite<short,1,1>*/,  0 /*composite<short,2,2>*/,  0 /*composite<short,3,3>*/,
         0 /*composite<short,3,4>*/   },
        {   0 /*composite<int,1,1>*/,    0 /*composite<int,2,2>*/,    0 /*composite<int,3,3>*/, 0 /*composite<int,3,4>*/   },
        { 0 /*composite<float,1,1>*/,  0 /*composite<float,2,2>*/,  0 /*composite<float,3,3>*/,
         0 /*composite<float,3,4>*/   },
    };

    const func_t func = funcs[foreground_data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    func(foreground, background, fgMask, outData, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

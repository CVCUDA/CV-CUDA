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
__global__ void composite_kernel(const Ptr2dNHWC<T> fg, const Ptr2dNHWC<T> bg, const Ptr2dNHWC<U> fgMask,
                                 Ptr2dNHWC<D> dst)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst.cols || dst_y >= dst.rows)
        return;

    const int batch_idx = get_batch_idx();

    int dst_ch = dst.ch;
    int src_ch = fg.ch;

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
void composite(const nvcv::TensorDataAccessStridedImagePlanar &foregroundData,
               const nvcv::TensorDataAccessStridedImagePlanar &backgroundData,
               const nvcv::TensorDataAccessStridedImagePlanar &fgMaskData,
               const nvcv::TensorDataAccessStridedImagePlanar &outData, cudaStream_t stream)
{
    const int batch_size = foregroundData.numSamples();
    const int out_width  = outData.numCols();
    const int out_height = outData.numRows();

    dim3 blockSize(16, 16, 1);
    dim3 gridSize(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), batch_size);

    typedef typename cuda::MakeType<T, scn> src_type;
    typedef typename cuda::MakeType<T, dcn> dst_type;

    Ptr2dNHWC<src_type> fg_ptr(foregroundData);
    Ptr2dNHWC<src_type> bg_ptr(backgroundData);
    Ptr2dNHWC<T>        fgMask_ptr(fgMaskData);
    Ptr2dNHWC<dst_type> dst_ptr(outData);

    composite_kernel<<<gridSize, blockSize, 0, stream>>>(fg_ptr, bg_ptr, fgMask_ptr, dst_ptr);
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nvcv::legacy::cuda_op {

ErrorCode Composite::infer(const ITensorDataStridedCuda &foreground, const ITensorDataStridedCuda &background,
                           const ITensorDataStridedCuda &fgMask, const ITensorDataStridedCuda &outData,
                           cudaStream_t stream)
{
    DataFormat background_format = GetLegacyDataFormat(background.layout());
    DataFormat foreground_format = GetLegacyDataFormat(foreground.layout());
    DataFormat fgMask_format     = GetLegacyDataFormat(fgMask.layout());
    DataFormat output_format     = GetLegacyDataFormat(outData.layout());

    if (!((foreground_format == background_format) && (foreground_format == fgMask_format)
          && (foreground_format == output_format)))
    {
        LOG_ERROR("Invalid DataFormat between foreground ("
                  << foreground_format << "), background (" << background_format << "), foreground mask ("
                  << fgMask_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = foreground_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto foregroundAccess = TensorDataAccessStridedImagePlanar::Create(foreground);
    NVCV_ASSERT(foregroundAccess);

    auto backgroundAccess = TensorDataAccessStridedImagePlanar::Create(background);
    NVCV_ASSERT(backgroundAccess);

    auto fgMaskAccess = TensorDataAccessStridedImagePlanar::Create(fgMask);
    NVCV_ASSERT(fgMaskAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    DataType foreground_data_type = GetLegacyDataType(foreground.dtype());
    DataType background_data_type = GetLegacyDataType(background.dtype());
    DataType fgMask_data_type     = GetLegacyDataType(fgMask.dtype());
    DataType output_data_type     = GetLegacyDataType(outData.dtype());

    DataShape foreground_shape = GetLegacyDataShape(foregroundAccess->infoShape());
    DataShape background_shape = GetLegacyDataShape(backgroundAccess->infoShape());
    DataShape fgMask_shape     = GetLegacyDataShape(fgMaskAccess->infoShape());
    DataShape output_shape     = GetLegacyDataShape(outAccess->infoShape());

    int foreground_channels = foreground_shape.C;
    int background_channels = background_shape.C;
    int fgMask_channels     = fgMask_shape.C;
    int output_channels     = output_shape.C;
    int channels            = output_channels;

    if (!((foreground_channels == 3) && (background_channels == 3) && (fgMask_channels == 1)
          && (output_channels == 3 || output_channels == 4)))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!((foreground_data_type == kCV_8U) && (background_data_type == kCV_8U) && (fgMask_data_type == kCV_8U)
          && (output_data_type == kCV_8U)))
    {
        LOG_ERROR("Invalid DataType " << foreground_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    typedef void (*func_t)(const nvcv::TensorDataAccessStridedImagePlanar &foregroundData,
                           const nvcv::TensorDataAccessStridedImagePlanar &backgroundData,
                           const nvcv::TensorDataAccessStridedImagePlanar &fgMaskData,
                           const nvcv::TensorDataAccessStridedImagePlanar &outData, cudaStream_t stream);

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

    func(*foregroundAccess, *backgroundAccess, *fgMaskAccess, *outAccess, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

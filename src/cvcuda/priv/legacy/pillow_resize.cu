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

#include <nvcv/Rect.h>

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

#define BLOCK           32
#define SHARE_MEM_LIMIT 4096
#define work_type       float

namespace nvcv::legacy::cuda_op {

static constexpr float        bilinear_filter_support = 1.;
static constexpr unsigned int precision_bits          = 32 - 8 - 2;

class BilinearFilter
{
public:
    __host__ __device__ BilinearFilter()
        : _support(bilinear_filter_support){};

    __host__ __device__ work_type filter(work_type x)
    {
        if (x < 0.0)
        {
            x = -x;
        }
        if (x < 1.0)
        {
            return 1.0 - x;
        }
        return 0.0;
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

template<class Filter>
__global__ void _precomputeCoeffs(int in_size, int in0, work_type scale, work_type filterscale, work_type support,
                                  int out_size, int k_size, Filter filterp, int *bounds_out, work_type *kk_out,
                                  bool normalize_coeff, bool use_share_mem)
{
    const int  xx       = blockIdx.x * blockDim.x + threadIdx.x;
    const int  local_id = threadIdx.x;
    const int  x_offset = blockIdx.x * blockDim.x;
    work_type *kk       = kk_out + x_offset * k_size;
    if (use_share_mem)
    {
        extern __shared__ __align__(sizeof(work_type)) unsigned char smem_raw[];
        kk = reinterpret_cast<work_type *>(smem_raw);
    }

    if (xx < out_size)
    {
        int       x      = 0;
        int       xmin   = 0;
        int       xmax   = 0;
        work_type center = 0;
        work_type ww     = 0;
        work_type ss     = 0;

        const work_type half_pixel = 0.5;

        center = in0 + (xx + half_pixel) * scale;
        ww     = 0.0;
        ss     = 1.0 / filterscale;
        // Round the value.
        xmin = static_cast<int>(center - support + half_pixel);
        if (xmin < 0)
        {
            xmin = 0;
        }
        // Round the value.
        xmax = static_cast<int>(center + support + half_pixel);
        if (xmax > in_size)
        {
            xmax = in_size;
        }
        xmax -= xmin;
        work_type *k = &kk[local_id * k_size];
        for (x = 0; x < xmax; ++x)
        {
            work_type w = filterp.filter((x + xmin - center + half_pixel) * ss);
            k[x]        = w;
            ww += w;
        }
        for (x = 0; x < xmax; ++x)
        {
            if (std::fabs(ww) > 1e-5)
            {
                k[x] /= ww;
            }
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < k_size; ++x)
        {
            k[x] = .0f;
        }
        if (normalize_coeff)
        {
            for (int i = 0; i < k_size; i++)
            {
                work_type val = k[i];
                if (val < 0)
                {
                    k[i] = static_cast<int>(-half_pixel + val * (1U << precision_bits));
                }
                else
                {
                    k[i] = static_cast<int>(half_pixel + val * (1U << precision_bits));
                }
            }
        }

        bounds_out[xx * 2]     = xmin;
        bounds_out[xx * 2 + 1] = xmax;
    }
    if (use_share_mem)
    {
        __syncthreads();
        for (int i = local_id; i < (out_size - x_offset) * k_size && i < blockDim.x * k_size; i += blockDim.x)
        {
            kk_out[x_offset * k_size + i] = kk[i];
        }
    }
}

template<class T, class Filter>
__global__ void horizontal_pass(const cuda_op::Ptr2dNHWC<T> src, cuda_op::Ptr2dNHWC<T> dst, NVCVRectI roi,
                                Filter &filterp, int h_ksize, int v_ksize, int *h_bounds, work_type *h_kk,
                                int *v_bounds, work_type *v_kk, work_type init_buffer, bool round_up,
                                bool use_share_mem)
{
    const int  dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int  dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int  local_x    = threadIdx.x;
    const int  x_offset   = blockIdx.x * blockDim.x;
    const int  batch_idx  = get_batch_idx();
    int        out_height = dst.rows, out_width = dst.cols;
    work_type *h_k_tmp = h_kk + x_offset * h_ksize;

    if (use_share_mem)
    {
        const int         local_tid = threadIdx.x + blockDim.x * threadIdx.y;
        extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_h[];
        h_k_tmp = reinterpret_cast<work_type *>(kk_smem_h);

        for (int i = local_tid; i < blockDim.x * h_ksize && i < (out_width - x_offset) * h_ksize;
             i += blockDim.x * blockDim.y)
        {
            h_k_tmp[i] = h_kk[x_offset * h_ksize + i];
        }

        __syncthreads();
    }

    if (dst_x < out_width && dst_y < out_height)
    {
        int xmin = h_bounds[dst_x * 2];
        int xmax = h_bounds[dst_x * 2 + 1];

        work_type *h_k = &h_k_tmp[local_x * h_ksize];

        for (int c = 0; c < src.ch; ++c)
        {
            work_type h_ss = 0.0;
            for (int x = 0; x < xmax; ++x)
            {
                h_ss = h_ss + *src.ptr(batch_idx, dst_y, x + xmin, c) * h_k[x];
            }

            if (round_up)
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(std::round(h_ss));
            else
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(h_ss);
        }
    }
}

template<class T, class Filter>
__global__ void vertical_pass(const cuda_op::Ptr2dNHWC<T> src, cuda_op::Ptr2dNHWC<T> dst, NVCVRectI roi,
                              Filter &filterp, int h_ksize, int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds,
                              work_type *v_kk, work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int  dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int  dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int  local_y    = threadIdx.y;
    const int  y_offset   = blockIdx.y * blockDim.y;
    const int  batch_idx  = get_batch_idx();
    int        out_height = dst.rows, out_width = dst.cols;
    work_type *v_k_tmp = v_kk + y_offset * v_ksize;

    if (use_share_mem)
    {
        const int         local_tid = threadIdx.x + blockDim.x * threadIdx.y;
        extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_v[];
        v_k_tmp = reinterpret_cast<work_type *>(kk_smem_v);

        for (int i = local_tid; i < blockDim.y * v_ksize && i < (out_height - y_offset) * v_ksize;
             i += blockDim.x * blockDim.y)
        {
            v_k_tmp[i] = v_kk[y_offset * v_ksize + i];
        }

        __syncthreads();
    }

    if (dst_x < out_width && dst_y < out_height)
    {
        int ymin = v_bounds[dst_y * 2];
        int ymax = v_bounds[dst_y * 2 + 1];

        work_type *v_k = &v_k_tmp[local_y * v_ksize];

        for (int c = 0; c < src.ch; ++c)
        {
            work_type ss = init_buffer;
            for (int y = 0; y < ymax; ++y)
            {
                ss = ss + *src.ptr(batch_idx, y + ymin, dst_x, c) * v_k[y];
            }

            if (round_up)
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(std::round(ss));
            else
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(ss);
        }
    }
}

template<typename Filter, typename elem_type>
void pillow_resize_v2(const TensorDataAccessStridedImagePlanar &inData,
                      const TensorDataAccessStridedImagePlanar &outData, void *gpu_workspace, bool normalize_coeff,
                      work_type init_buffer, bool round_up, cudaStream_t stream)
{
    cuda_op::DataShape   input_shape = GetLegacyDataShape(inData.infoShape());
    Ptr2dNHWC<elem_type> src_ptr(inData);
    Ptr2dNHWC<elem_type> dst_ptr(outData);
    NVCVRectI            roi = {0, 0, src_ptr.cols, src_ptr.rows};
    Filter               filterp;
    work_type            h_scale = 0, v_scale = 0;
    work_type            h_filterscale = 0, v_filterscale = 0;
    h_filterscale = h_scale = static_cast<work_type>(roi.width) / dst_ptr.cols;
    v_filterscale = v_scale = static_cast<work_type>(roi.height) / dst_ptr.rows;

    int out_width  = dst_ptr.cols;
    int out_height = dst_ptr.rows;

    if (h_filterscale < 1.0)
    {
        h_filterscale = 1.0;
    }
    if (v_filterscale < 1.0)
    {
        v_filterscale = 1.0;
    }

    // Determine support size (length of resampling filter).
    work_type h_support = filterp.support() * h_filterscale;
    work_type v_support = filterp.support() * v_filterscale;

    // Maximum number of coeffs.
    int h_k_size = static_cast<int>(ceil(h_support)) * 2 + 1;
    int v_k_size = static_cast<int>(ceil(v_support)) * 2 + 1;

    work_type *h_kk     = (work_type *)((char *)gpu_workspace);
    work_type *v_kk     = (work_type *)((char *)h_kk + dst_ptr.cols * h_k_size * sizeof(work_type));
    int       *h_bounds = (int *)((char *)v_kk + dst_ptr.rows * v_k_size * sizeof(work_type));
    int       *v_bounds = (int *)((char *)h_bounds + dst_ptr.cols * 2 * sizeof(int));
    elem_type *d_h_data = (elem_type *)((char *)v_bounds + dst_ptr.rows * 2 * sizeof(int));

    Ptr2dNHWC<elem_type> h_ptr(input_shape.N, input_shape.H, out_width, input_shape.C, (elem_type *)d_h_data);

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSizeH(divUp(out_width, blockSize.x), divUp(input_shape.H, blockSize.y), input_shape.N);
    dim3 gridSizeV(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), input_shape.N);

    dim3 coef_block(BLOCK * 2, 1, 1);
    dim3 h_coef_grid(divUp(dst_ptr.cols, coef_block.x), 1, 1);
    dim3 v_coef_grid(divUp(dst_ptr.rows, coef_block.x), 1, 1);

    size_t h_sm_size = coef_block.x * (h_k_size * sizeof(work_type));
    size_t v_sm_size = coef_block.x * (v_k_size * sizeof(work_type));

    size_t hv_sm_size1     = h_k_size * sizeof(work_type) * blockSize.x;
    size_t hv_sm_size2     = v_k_size * sizeof(work_type) * blockSize.y;
    bool   h_use_share_mem = h_sm_size <= SHARE_MEM_LIMIT;
    if (!h_use_share_mem)
    {
        h_sm_size = 0;
    }
    bool v_use_share_mem = v_sm_size <= SHARE_MEM_LIMIT;
    if (!v_use_share_mem)
    {
        v_sm_size = 0;
    }
    bool hv_use_share_mem = (hv_sm_size1 <= SHARE_MEM_LIMIT) && (hv_sm_size2 <= SHARE_MEM_LIMIT);
    if (!hv_use_share_mem)
    {
        hv_sm_size1 = 0;
        hv_sm_size2 = 0;
    }
    // compute horizental coef
    _precomputeCoeffs<Filter><<<h_coef_grid, coef_block, h_sm_size, stream>>>(
        src_ptr.cols, roi.x, h_scale, h_filterscale, h_support, dst_ptr.cols, h_k_size, filterp, h_bounds, h_kk,
        normalize_coeff, h_use_share_mem);

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    // compute vertical coef
    _precomputeCoeffs<Filter><<<v_coef_grid, coef_block, v_sm_size, stream>>>(
        src_ptr.rows, roi.y, v_scale, v_filterscale, v_support, dst_ptr.rows, v_k_size, filterp, v_bounds, v_kk,
        normalize_coeff, v_use_share_mem);

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    horizontal_pass<elem_type, Filter>
        <<<gridSizeH, blockSize, hv_sm_size1, stream>>>(src_ptr, h_ptr, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                        h_kk, v_bounds, v_kk, init_buffer, round_up, hv_use_share_mem);
    checkKernelErrors();
    vertical_pass<elem_type, Filter>
        <<<gridSizeV, blockSize, hv_sm_size2, stream>>>(h_ptr, dst_ptr, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                        h_kk, v_bounds, v_kk, init_buffer, round_up, hv_use_share_mem);

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename Filter>
void pillow_resize_filter(const TensorDataAccessStridedImagePlanar &inData,
                          const TensorDataAccessStridedImagePlanar &outData, void *gpu_workspace,
                          NVCVInterpolationType interpolation, cudaStream_t stream)
{
    cuda_op::DataType data_type = GetLegacyDataType(inData.dtype());
    switch (data_type)
    {
    case kCV_8U:
        pillow_resize_v2<Filter, unsigned char>(inData, outData, gpu_workspace, false, 0., false, stream);
        break;
    case kCV_8S:
        pillow_resize_v2<Filter, signed char>(inData, outData, gpu_workspace, false, 0., true, stream);
        break;
    case kCV_16U:
        pillow_resize_v2<Filter, std::uint16_t>(inData, outData, gpu_workspace, false, 0., false, stream);
        break;
    case kCV_16S:
        pillow_resize_v2<Filter, std::int16_t>(inData, outData, gpu_workspace, false, 0., true, stream);
        break;
    case kCV_32S:
        pillow_resize_v2<Filter, int>(inData, outData, gpu_workspace, false, 0., true, stream);
        break;
    case kCV_32F:
        pillow_resize_v2<Filter, float>(inData, outData, gpu_workspace, false, 0., false, stream);
        break;
    default:
        break;
    }
}

PillowResize::PillowResize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
    : CudaBaseOp(max_input_shape, max_output_shape)
{
    int    max_support = 1; //3
    size_t size
        = std::ceil(
              max_output_shape.H
                  * (((1.0 * max_input_shape.H / max_output_shape.H + 1) * max_support * 2 + 1) * sizeof(work_type)
                     + 2 * sizeof(int))
              + max_output_shape.W
                    * (((1.0 * max_input_shape.W / max_output_shape.W + 1) * max_support * 2 + 1) * sizeof(work_type)
                       + 2 * sizeof(int)))
        + max_input_shape.N * max_input_shape.C * max_input_shape.H * max_output_shape.W * DataSize(max_data_type);
    NVCV_CHECK_LOG(cudaMalloc(&gpu_workspace, size));
}

PillowResize::~PillowResize()
{
    NVCV_CHECK_LOG(cudaFree(gpu_workspace));
}

size_t PillowResize::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    int    max_support = 1; //3
    size_t size
        = std::ceil(
              max_output_shape.H
                  * (((1.0 * max_input_shape.H / max_output_shape.H + 1) * max_support * 2 + 1) * sizeof(work_type)
                     + 2 * sizeof(int))
              + max_output_shape.W
                    * (((1.0 * max_input_shape.W / max_output_shape.W + 1) * max_support * 2 + 1) * sizeof(work_type)
                       + 2 * sizeof(int)))
        + max_input_shape.N * max_input_shape.C * max_input_shape.H * max_output_shape.W * DataSize(max_data_type);
    return size;
}

ErrorCode PillowResize::infer(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData,
                              const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    DataFormat format        = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    int channels = input_shape.C;

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(interpolation == NVCV_INTERP_LINEAR))
    {
        LOG_ERROR("Unsupported interpolation method " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }

    switch (interpolation)
    {
    case NVCV_INTERP_LINEAR:
        pillow_resize_filter<BilinearFilter>(*inAccess, *outAccess, gpu_workspace, interpolation, stream);
        break;
    default:
        break;
    }
    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

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

#include <nvcv/ImageBatch.hpp>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

#define BLOCK           32
#define SHARE_MEM_LIMIT 4096
#define work_type       float

namespace nvcv::legacy::cuda_op {

static constexpr float        bilinear_filter_support_var_shape = 1.;
static constexpr unsigned int precision_bits_var_shape          = 32 - 8 - 2;

namespace {

class BilinearFilterVarShape
{
public:
    __host__ __device__ BilinearFilterVarShape()
        : _support(bilinear_filter_support_var_shape){};

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
__global__ void _precomputeCoeffsVarShape(int *in_size_batch, int *in0_batch, work_type *scale_batch,
                                          work_type *filterscale_batch, work_type *support_batch, int *out_size_batch,
                                          int *k_size_batch, Filter filterp, int *bounds_out_batch,
                                          int *bound_out_offset, work_type *kk_out_batch, int *kk_out_offset,
                                          bool normalize_coeff, bool use_share_mem)
{
    const int xx       = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_id = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    const int  batch_idx   = get_batch_idx();
    int        in_size     = in_size_batch[batch_idx];
    int        in0         = in0_batch[batch_idx];
    work_type  scale       = scale_batch[batch_idx];
    work_type  filterscale = filterscale_batch[batch_idx];
    work_type  support     = support_batch[batch_idx];
    int        out_size    = out_size_batch[batch_idx];
    int        k_size      = k_size_batch[batch_idx];
    int       *bounds_out  = bounds_out_batch + bound_out_offset[batch_idx];
    work_type *kk_out      = kk_out_batch + kk_out_offset[batch_idx];

    work_type *kk = kk_out + x_offset * k_size;
    if (use_share_mem)
    {
        extern __shared__ __align__(sizeof(work_type)) unsigned char smem_raw[];
        kk = reinterpret_cast<work_type *>(smem_raw);
    }

    if (xx < out_size)
    {
        int             x          = 0;
        int             xmin       = 0;
        int             xmax       = 0;
        work_type       center     = 0;
        work_type       ww         = 0;
        work_type       ss         = 0;
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
                    k[i] = static_cast<int>(-half_pixel + val * (1U << precision_bits_var_shape));
                }
                else
                {
                    k[i] = static_cast<int>(half_pixel + val * (1U << precision_bits_var_shape));
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

template<class T1, class T2, class Filter>
__global__ void horizontal_pass_var_shape(const Ptr2dVarShapeNHWC<T1> src, Ptr2dNHWC<T2> dst, Filter &filterp,
                                          int *h_ksize_batch, int *v_ksize_batch, int *h_bounds_batch,
                                          int *h_bounds_offset, work_type *h_kk_batch, int *h_kk_offset,
                                          int *v_bounds_batch, int *v_bounds_offset, work_type *v_kk_batch,
                                          int *v_kk_offset, work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x  = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    const int  batch_idx = get_batch_idx();
    int        h_ksize   = h_ksize_batch[batch_idx];
    int       *h_bounds  = h_bounds_batch + h_bounds_offset[batch_idx];
    work_type *h_kk      = h_kk_batch + h_kk_offset[batch_idx];

    int out_height = dst.at_rows(batch_idx), out_width = dst.at_cols(batch_idx);

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
        // int        offset_src = dst_y * src.at_cols(batch_idx) * src.nch + xmin * src.nch;
        // int        offset_dst = dst_y * dst.at_cols(batch_idx) * dst.nch + dst_x * dst.nch;
        for (int c = 0; c < src.nch; ++c)
        {
            work_type h_ss = 0.0;
            for (int x = 0; x < xmax; ++x)
            {
                // offset = offset_src + x * src.nch + c = (dst_y * src.at_cols(batch_idx) + xmin  + x) * src.nch + c
                h_ss = h_ss
                     + *src.ptr(batch_idx, dst_y + (xmin + x) / src.at_cols(batch_idx),
                                (xmin + x) % src.at_cols(batch_idx), c)
                           * h_k[x];
            }
            if (round_up)
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<cuda::BaseType<T2>>(std::round(h_ss));
            else
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<cuda::BaseType<T2>>(h_ss);
        }
    }
}

template<class T1, class T2, class Filter>
__global__ void vertical_pass_var_shape(const Ptr2dNHWC<T1> src, Ptr2dVarShapeNHWC<T2> dst, Filter &filterp,
                                        int *h_ksize_batch, int *v_ksize_batch, int *h_bounds_batch,
                                        int *h_bounds_offset, work_type *h_kk_batch, int *h_kk_offset,
                                        int *v_bounds_batch, int *v_bounds_offset, work_type *v_kk_batch,
                                        int *v_kk_offset, work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_y  = threadIdx.y;
    const int y_offset = blockIdx.y * blockDim.y;

    const int  batch_idx = get_batch_idx();
    int        v_ksize   = v_ksize_batch[batch_idx];
    int       *v_bounds  = v_bounds_batch + v_bounds_offset[batch_idx];
    work_type *v_kk      = v_kk_batch + v_kk_offset[batch_idx];

    int out_height = dst.at_rows(batch_idx), out_width = dst.at_cols(batch_idx);

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
        // int        offset_src     = ymin * src.at_cols(batch_idx) * src.nch + dst_x * src.nch;
        // int        col_offset_src = src.at_cols(batch_idx) * src.nch;
        // int        offset_dst     = dst_y * dst.at_cols(batch_idx) * dst.nch + dst_x * dst.nch;
        for (int c = 0; c < src.ch; ++c)
        {
            work_type ss = init_buffer;
            for (int y = 0; y < ymax; ++y)
            {
                // offset =  offset_src + y * col_offset_src + c = ((y + ymin)* src.at_cols(batch_idx) + dst_x) * src.nch + c
                ss = ss
                   + *src.ptr(batch_idx, y + ymin + (dst_x / src.at_cols(batch_idx)), dst_x % src.at_cols(batch_idx), c)
                         * v_k[y];
            }

            if (round_up)
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<cuda::BaseType<T2>>(std::round(ss));
            else
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<cuda::BaseType<T2>>(ss);
        }
    }
}

template<typename Filter, typename elem_type>
void pillow_resize_var_shape(const IImageBatchVarShape &inDataBase, const IImageBatchVarShape &outDataBase,
                             void *gpu_workspace, void *cpu_workspace, bool normalize_coeff, work_type init_buffer,
                             bool round_up, cudaStream_t stream)
{
    auto *inDataPtr = dynamic_cast<const IImageBatchVarShapeDataStridedCuda *>(inDataBase.exportData(stream));
    if (inDataPtr == nullptr)
    {
        throw std::runtime_error("Something wrong happend during conversion of type...!!!");
    }

    auto *outDataPtr = dynamic_cast<const IImageBatchVarShapeDataStridedCuda *>(outDataBase.exportData(stream));
    if (outDataPtr == nullptr)
    {
        throw std::runtime_error("Something wrong happend during conversion of type...!!!");
    }

    const IImageBatchVarShapeDataStridedCuda &inData  = *inDataPtr;
    const IImageBatchVarShapeDataStridedCuda &outData = *outDataPtr;

    int channels = inData.uniqueFormat().numChannels();
    int batch    = inData.numImages();

    Filter filterp;

    Size2D outMaxSize = outData.maxSize();
    Size2D inMaxSize  = inData.maxSize();

    int max_height = outMaxSize.h, max_width = outMaxSize.w;
    int max_input_height = inMaxSize.h;

    const void **inputs              = (const void **)cpu_workspace;
    void       **outputs             = (void **)((char *)inputs + sizeof(void *) * batch);
    void       **hori                = (void **)((char *)outputs + sizeof(void *) * batch);
    int         *rows                = (int *)((char *)hori + sizeof(void *) * batch);
    int         *cols                = (int *)((char *)rows + sizeof(int) * batch);
    int         *out_rows            = (int *)((char *)cols + sizeof(int) * batch);
    int         *out_cols            = (int *)((char *)out_rows + sizeof(int) * batch);
    int         *roi_x               = (int *)((char *)out_cols + sizeof(int) * batch);
    int         *roi_y               = (int *)((char *)roi_x + sizeof(int) * batch);
    work_type   *h_scale_batch       = (work_type *)((char *)roi_y + sizeof(int) * batch);
    work_type   *v_scale_batch       = (work_type *)((char *)h_scale_batch + sizeof(work_type) * batch);
    work_type   *h_filterscale_batch = (work_type *)((char *)v_scale_batch + sizeof(work_type) * batch);
    work_type   *v_filterscale_batch = (work_type *)((char *)h_filterscale_batch + sizeof(work_type) * batch);
    work_type   *h_support_batch     = (work_type *)((char *)v_filterscale_batch + sizeof(work_type) * batch);
    work_type   *v_support_batch     = (work_type *)((char *)h_support_batch + sizeof(work_type) * batch);
    int         *h_k_size_batch      = (int *)((char *)v_support_batch + sizeof(work_type) * batch);
    int         *v_k_size_batch      = (int *)((char *)h_k_size_batch + sizeof(int) * batch);
    int         *h_bounds_offset     = (int *)((char *)v_k_size_batch + sizeof(int) * batch);
    int         *v_bounds_offset     = (int *)((char *)h_bounds_offset + sizeof(int) * batch);
    int         *h_kk_offset         = (int *)((char *)v_bounds_offset + sizeof(int) * batch);
    int         *v_kk_offset         = (int *)((char *)h_kk_offset + sizeof(int) * batch);

    int h_kk_total = 0, v_kk_total = 0;
    int max_h_k_size = 0, max_v_k_size = 0;
    int h_bounds_total = 0, v_bounds_total = 0;

    for (int i = 0; i < batch; i++)
    {
        rows[i]     = inDataBase[i].size().h;
        cols[i]     = inDataBase[i].size().w;
        out_rows[i] = outDataBase[i].size().h;
        out_cols[i] = outDataBase[i].size().w;

        roi_x[i] = 0;
        roi_y[i] = 0;

        work_type h_scale = 0, v_scale = 0;
        work_type h_filterscale = 0, v_filterscale = 0;
        h_filterscale = h_scale = static_cast<work_type>(inDataBase[i].size().w) / out_cols[i];
        v_filterscale = v_scale = static_cast<work_type>(inDataBase[i].size().h) / out_rows[i];
        if (h_filterscale < 1.0)
        {
            h_filterscale = 1.0;
        }
        if (v_filterscale < 1.0)
        {
            v_filterscale = 1.0;
        }
        h_scale_batch[i]       = h_scale;
        v_scale_batch[i]       = v_scale;
        h_filterscale_batch[i] = h_filterscale;
        v_filterscale_batch[i] = v_filterscale;

        // Determine support size (length of resampling filter).
        work_type h_support = filterp.support() * h_filterscale;
        work_type v_support = filterp.support() * v_filterscale;
        // Maximum number of coeffs.
        int       h_k_size = static_cast<int>(ceil(h_support)) * 2 + 1;
        int       v_k_size = static_cast<int>(ceil(v_support)) * 2 + 1;
        h_support_batch[i] = h_support;
        v_support_batch[i] = v_support;
        h_k_size_batch[i]  = h_k_size;
        v_k_size_batch[i]  = v_k_size;
        h_kk_offset[i]     = h_kk_total;
        v_kk_offset[i]     = v_kk_total;
        h_kk_total += out_cols[i] * h_k_size;
        v_kk_total += out_rows[i] * v_k_size;
        h_bounds_offset[i] = h_bounds_total;
        v_bounds_offset[i] = v_bounds_total;
        h_bounds_total += out_cols[i] * 2;
        v_bounds_total += out_rows[i] * 2;

        if (h_k_size > max_h_k_size)
            max_h_k_size = h_k_size;
        if (v_k_size > max_v_k_size)
            max_v_k_size = v_k_size;
    }

    const void **inputs_gpu              = (const void **)gpu_workspace;
    void       **outputs_gpu             = (void **)((char *)inputs_gpu + sizeof(void *) * batch);
    void       **hori_gpu                = (void **)((char *)outputs_gpu + sizeof(void *) * batch);
    int         *rows_gpu                = (int *)((char *)hori_gpu + sizeof(void *) * batch);
    int         *cols_gpu                = (int *)((char *)rows_gpu + sizeof(int) * batch);
    int         *out_rows_gpu            = (int *)((char *)cols_gpu + sizeof(int) * batch);
    int         *out_cols_gpu            = (int *)((char *)out_rows_gpu + sizeof(int) * batch);
    int         *roi_x_gpu               = (int *)((char *)out_cols_gpu + sizeof(int) * batch);
    int         *roi_y_gpu               = (int *)((char *)roi_x_gpu + sizeof(int) * batch);
    work_type   *h_scale_batch_gpu       = (work_type *)((char *)roi_y_gpu + sizeof(int) * batch);
    work_type   *v_scale_batch_gpu       = (work_type *)((char *)h_scale_batch_gpu + sizeof(work_type) * batch);
    work_type   *h_filterscale_batch_gpu = (work_type *)((char *)v_scale_batch_gpu + sizeof(work_type) * batch);
    work_type   *v_filterscale_batch_gpu = (work_type *)((char *)h_filterscale_batch_gpu + sizeof(work_type) * batch);
    work_type   *h_support_batch_gpu     = (work_type *)((char *)v_filterscale_batch_gpu + sizeof(work_type) * batch);
    work_type   *v_support_batch_gpu     = (work_type *)((char *)h_support_batch_gpu + sizeof(work_type) * batch);
    int         *h_k_size_batch_gpu      = (int *)((char *)v_support_batch_gpu + sizeof(work_type) * batch);
    int         *v_k_size_batch_gpu      = (int *)((char *)h_k_size_batch_gpu + sizeof(int) * batch);
    int         *h_bounds_offset_gpu     = (int *)((char *)v_k_size_batch_gpu + sizeof(int) * batch);
    int         *v_bounds_offset_gpu     = (int *)((char *)h_bounds_offset_gpu + sizeof(int) * batch);
    int         *h_kk_offset_gpu         = (int *)((char *)v_bounds_offset_gpu + sizeof(int) * batch);
    int         *v_kk_offset_gpu         = (int *)((char *)h_kk_offset_gpu + sizeof(int) * batch);

    work_type *h_kk_batch_gpu     = (work_type *)((char *)v_kk_offset_gpu + sizeof(int) * batch);
    work_type *v_kk_batch_gpu     = (work_type *)((char *)h_kk_batch_gpu + sizeof(work_type) * h_kk_total);
    int       *h_bounds_batch_gpu = (int *)((char *)v_kk_batch_gpu + sizeof(work_type) * v_kk_total);
    int       *v_bounds_batch_gpu = (int *)((char *)h_bounds_batch_gpu + sizeof(int) * h_bounds_total);

    int current_buffer_size = (sizeof(void *) * 3 + sizeof(int) * 12 + sizeof(work_type) * 6) * batch
                            + sizeof(work_type) * (h_kk_total + v_kk_total)
                            + sizeof(int) * (h_bounds_total + v_bounds_total);

    // buffer for storing results from horizontal pass
    void *hori_gpu_data = (void *)((char *)gpu_workspace + current_buffer_size);

    checkCudaErrors(cudaMemcpyAsync((void *)gpu_workspace, (void *)cpu_workspace, current_buffer_size,
                                    cudaMemcpyHostToDevice, stream));

    Ptr2dVarShapeNHWC<elem_type> src_ptr(inData);
    Ptr2dVarShapeNHWC<elem_type> dst_ptr(outData);
    Ptr2dNHWC<work_type>         ptr_h_out(batch, max_input_height, max_width, channels, (work_type *)hori_gpu_data);

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSizeH(divUp(max_width, blockSize.x), divUp(max_input_height, blockSize.y), batch);
    dim3 gridSizeV(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), batch);

    dim3 coef_block(BLOCK * 2, 1, 1);
    dim3 h_coef_grid(divUp(max_width, coef_block.x), 1, batch);
    dim3 v_coef_grid(divUp(max_height, coef_block.x), 1, batch);

    size_t h_sm_size       = coef_block.x * (max_h_k_size * sizeof(work_type));
    size_t v_sm_size       = coef_block.x * (max_v_k_size * sizeof(work_type));
    size_t hv_sm_size1     = max_h_k_size * sizeof(work_type) * blockSize.x;
    size_t hv_sm_size2     = max_v_k_size * sizeof(work_type) * blockSize.y;
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

    // compute horizontal coef
    _precomputeCoeffsVarShape<Filter><<<h_coef_grid, coef_block, h_sm_size, stream>>>(
        cols_gpu, roi_x_gpu, h_scale_batch_gpu, h_filterscale_batch_gpu, h_support_batch_gpu, out_cols_gpu,
        h_k_size_batch_gpu, filterp, h_bounds_batch_gpu, h_bounds_offset_gpu, h_kk_batch_gpu, h_kk_offset_gpu,
        normalize_coeff, h_use_share_mem);

    checkKernelErrors();
    // checkCudaErrors(cudaStreamSynchronize(stream));
    // compute vertical coef
    _precomputeCoeffsVarShape<Filter><<<v_coef_grid, coef_block, v_sm_size, stream>>>(
        rows_gpu, roi_y_gpu, v_scale_batch_gpu, v_filterscale_batch_gpu, v_support_batch_gpu, out_rows_gpu,
        v_k_size_batch_gpu, filterp, v_bounds_batch_gpu, v_bounds_offset_gpu, v_kk_batch_gpu, v_kk_offset_gpu,
        normalize_coeff, v_use_share_mem);
    checkKernelErrors();
    // checkCudaErrors(cudaStreamSynchronize(stream));
    horizontal_pass_var_shape<elem_type, work_type, Filter><<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
        src_ptr, ptr_h_out, filterp, h_k_size_batch_gpu, v_k_size_batch_gpu, h_bounds_batch_gpu, h_bounds_offset_gpu,
        h_kk_batch_gpu, h_kk_offset_gpu, v_bounds_batch_gpu, v_bounds_offset_gpu, v_kk_batch_gpu, v_kk_offset_gpu,
        init_buffer, round_up, hv_use_share_mem);
    checkKernelErrors();
    // checkCudaErrors(cudaStreamSynchronize(stream));
    vertical_pass_var_shape<work_type, elem_type, Filter><<<gridSizeV, blockSize, hv_sm_size2, stream>>>(
        ptr_h_out, dst_ptr, filterp, h_k_size_batch_gpu, v_k_size_batch_gpu, h_bounds_batch_gpu, h_bounds_offset_gpu,
        h_kk_batch_gpu, h_kk_offset_gpu, v_bounds_batch_gpu, v_bounds_offset_gpu, v_kk_batch_gpu, v_kk_offset_gpu,
        init_buffer, round_up, hv_use_share_mem);

    checkKernelErrors();
}

} // namespace

template<typename Filter>
void pillow_resize_filter_var_shape(const IImageBatchVarShape &inData, const IImageBatchVarShape &outData,
                                    void *gpu_workspace, void *cpu_workspace, NVCVInterpolationType interpolation,
                                    cudaStream_t stream)
{
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());
    switch (data_type)
    {
    case kCV_8U:
        pillow_resize_var_shape<Filter, unsigned char>(inData, outData, gpu_workspace, cpu_workspace, false, 0., false,
                                                       stream);
        break;
    case kCV_8S:
        pillow_resize_var_shape<Filter, signed char>(inData, outData, gpu_workspace, cpu_workspace, false, 0., true,
                                                     stream);
        break;
    case kCV_16U:
        pillow_resize_var_shape<Filter, std::uint16_t>(inData, outData, gpu_workspace, cpu_workspace, false, 0., false,
                                                       stream);
        break;
    case kCV_16S:
        pillow_resize_var_shape<Filter, std::int16_t>(inData, outData, gpu_workspace, cpu_workspace, false, 0., true,
                                                      stream);
        break;
    case kCV_32S:
        pillow_resize_var_shape<Filter, int>(inData, outData, gpu_workspace, cpu_workspace, false, 0., true, stream);
        break;
    case kCV_32F:
        pillow_resize_var_shape<Filter, float>(inData, outData, gpu_workspace, cpu_workspace, false, 0., false, stream);
        break;
    case kCV_64F:
    default:
        break;
    }
}

PillowResizeVarShape::PillowResizeVarShape(DataShape max_input_shape, DataShape max_output_shape,
                                           DataType max_data_type)
    : CudaBaseOp(max_input_shape, max_output_shape)
{
    int    max_support = 1; //3
    size_t size        = std::ceil(
               max_output_shape.H
                   * (((1.0 * max_input_shape.H / max_output_shape.H + 1) * max_support * 2 + 1) * sizeof(work_type)
               + 2 * sizeof(int))
               + max_output_shape.W
                     * (((1.0 * max_input_shape.W / max_output_shape.W + 1) * max_support * 2 + 1) * sizeof(work_type)
                 + 2 * sizeof(int)));
    size_t buffer_size = (sizeof(void *) * 3 + sizeof(int) * 12 + sizeof(work_type) * 6 + size) * max_input_shape.N;
    buffer_size += max_input_shape.N * max_input_shape.C * max_input_shape.H * max_output_shape.W * sizeof(float);

    NVCV_CHECK_LOG(cudaMalloc(&gpu_workspace, buffer_size));

    cpu_workspace = malloc(buffer_size);
    if (!cpu_workspace)
    {
        LOG_ERROR("Memory allocation error of size: " << buffer_size);
        throw std::runtime_error("Memory allocation error!");
    }
}

PillowResizeVarShape::~PillowResizeVarShape()
{
    NVCV_CHECK_LOG(cudaFree(gpu_workspace));
    free(cpu_workspace);
}

size_t PillowResizeVarShape::calBufferSize(DataShape max_input_shape, DataShape max_output_shape,
                                           DataType max_data_type)
{
    int    max_support = 1; //3
    size_t size        = std::ceil(
               max_output_shape.H
                   * (((1.0 * max_input_shape.H / max_output_shape.H + 1) * max_support * 2 + 1) * sizeof(work_type)
               + 2 * sizeof(int))
               + max_output_shape.W
                     * (((1.0 * max_input_shape.W / max_output_shape.W + 1) * max_support * 2 + 1) * sizeof(work_type)
                 + 2 * sizeof(int)));
    size_t buffer_size = (sizeof(void *) * 3 + sizeof(int) * 12 + sizeof(work_type) * 6 + size) * max_input_shape.N;
    buffer_size += max_input_shape.N * max_input_shape.C * max_input_shape.H * max_output_shape.W * sizeof(float);

    return buffer_size;
}

ErrorCode PillowResizeVarShape::infer(const nvcv::IImageBatchVarShape &inDataBase,
                                      const nvcv::IImageBatchVarShape &outDataBase,
                                      const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    if (!inDataBase.uniqueFormat() || !outDataBase.uniqueFormat())
    {
        LOG_ERROR("Images in input and outut batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inDataBase.uniqueFormat() != outDataBase.uniqueFormat())
    {
        LOG_ERROR("Invalid DataFormat between input and output");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = GetLegacyDataFormat(inDataBase);

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = inDataBase.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType data_type = helpers::GetLegacyDataType(inDataBase.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
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
        pillow_resize_filter_var_shape<BilinearFilterVarShape>(inDataBase, outDataBase, gpu_workspace, cpu_workspace,
                                                               interpolation, stream);
        break;
    default:
        break;
    }
    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cub/cub.cuh"
#include "inpaint_utils.cuh"
#include "reduce_kernel_utils.cuh"

using namespace nvcv::legacy::helpers;

using namespace nvcv::legacy::cuda_op;

using namespace nvcv::cuda;

#define KNOWN  0 //known outside narrow band
#define BAND   1 //narrow band (known)
#define INSIDE 2 //unknown
#define CHANGE 3 //servise

#define BLOCK            32
#define BLOCK_S          16
#define REDUCE_GRID_SIZE 64

template<typename T>
__global__ void copy_mask_data(Tensor4DWrap<T> src, Ptr2dNHWC<T> dst, int row_offset, int col_offset, int value,
                               int2 size)
{
    int src_x = blockIdx.x * blockDim.x + threadIdx.x;
    int src_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (src_x >= size.x || src_y >= size.y)
        return;
    const int batch_idx = get_batch_idx();

    for (int c = 0; c < dst.ch; ++c)
    {
        int cond = *src.ptr(batch_idx, src_y, src_x, c);
        if (cond)
        {
            *dst.ptr(batch_idx, src_y + row_offset, src_x + col_offset, c) = value;
        }
    }
}

template<typename T>
__device__ void inpaint(Ptr2dNHWC<unsigned char> f, Ptr2dNHWC<float> t, Tensor4DWrap<T> out, int i, int j, int range,
                        int ch)
{
    const int batch_idx = get_batch_idx();

    for (int color = 0; color < ch; color++)
    {
        float2 gradI, gradT, r;
        float  Ia = 0, Jx = 0, Jy = 0, s = 1.0e-20f, w, dst, lev, dir, sat;

        if (*f.ptr(batch_idx, i, j + 1) != INSIDE)
        {
            if (*f.ptr(batch_idx, i, j - 1) != INSIDE)
            {
                gradT.x = (float)(*t.ptr(batch_idx, i, j + 1) - *t.ptr(batch_idx, i, j - 1)) * 0.5f;
            }
            else
            {
                gradT.x = (float)(*t.ptr(batch_idx, i, j + 1) - *t.ptr(batch_idx, i, j));
            }
        }
        else
        {
            if (*f.ptr(batch_idx, i, j - 1) != INSIDE)
            {
                gradT.x = (float)(*t.ptr(batch_idx, i, j) - *t.ptr(batch_idx, i, j - 1));
            }
            else
            {
                gradT.x = 0;
            }
        }
        if (*f.ptr(batch_idx, i + 1, j) != INSIDE)
        {
            if (*f.ptr(batch_idx, i - 1, j) != INSIDE)
            {
                gradT.y = (float)(*t.ptr(batch_idx, i + 1, j) - *t.ptr(batch_idx, i - 1, j)) * 0.5f;
            }
            else
            {
                gradT.y = (float)(*t.ptr(batch_idx, i + 1, j) - *t.ptr(batch_idx, i, j));
            }
        }
        else
        {
            if (*f.ptr(batch_idx, i - 1, j) != INSIDE)
            {
                gradT.y = (float)(*t.ptr(batch_idx, i, j) - *t.ptr(batch_idx, i - 1, j));
            }
            else
            {
                gradT.y = 0;
            }
        }
        for (int k = i - range; k <= i + range; k++)
        {
            int km = k - 1 + (k == 1), kp = k - 1 - (k == t.rows - 2);
            for (int l = j - range; l <= j + range; l++)
            {
                int lm = l - 1 + (l == 1), lp = l - 1 - (l == t.cols - 2);
                if (k > 0 && l > 0 && k < t.rows - 1 && l < t.cols - 1)
                {
                    if ((*f.ptr(batch_idx, k, l) != INSIDE) && ((l - j) * (l - j) + (k - i) * (k - i) <= range * range)
                        && ((i != k) || (j != l))) // r != 0
                    {
                        r.y = (float)(i - k);
                        r.x = (float)(j - l);

                        dst = (float)(1. / (VectorLength(r) * sqrt((double)VectorLength(r))));
                        lev = (float)(1. / (1 + fabs(*t.ptr(batch_idx, k, l) - *t.ptr(batch_idx, i, j))));

                        dir = VectorScalMult(r, gradT);
                        if (fabs(dir) <= 0.01)
                            dir = 0.000001f;
                        w = (float)fabs(dst * lev * dir);

                        if (*f.ptr(batch_idx, k, l + 1) != INSIDE)
                        {
                            if (*f.ptr(batch_idx, k, l - 1) != INSIDE)
                            {
                                gradI.x = (float)((*out.ptr(batch_idx, km, lp + 1, color)
                                                   - *out.ptr(batch_idx, km, lm - 1, color)))
                                        * 2.0f;
                            }
                            else
                            {
                                gradI.x = (float)((*out.ptr(batch_idx, km, lp + 1, color)
                                                   - *out.ptr(batch_idx, km, lm, color)));
                            }
                        }
                        else
                        {
                            if (*f.ptr(batch_idx, k, l - 1) != INSIDE)
                            {
                                gradI.x = (float)((*out.ptr(batch_idx, km, lp, color)
                                                   - *out.ptr(batch_idx, km, lm - 1, color)));
                            }
                            else
                            {
                                gradI.x = 0;
                            }
                        }
                        if (*f.ptr(batch_idx, k + 1, l) != INSIDE)
                        {
                            if (*f.ptr(batch_idx, k - 1, l) != INSIDE)
                            {
                                gradI.y = (float)((*out.ptr(batch_idx, kp + 1, lm, color)
                                                   - *out.ptr(batch_idx, km - 1, lm, color)))
                                        * 2.0f;
                            }
                            else
                            {
                                gradI.y = (float)((*out.ptr(batch_idx, kp + 1, lm, color)
                                                   - *out.ptr(batch_idx, km, lm, color)));
                            }
                        }
                        else
                        {
                            if (*f.ptr(batch_idx, k - 1, l) != INSIDE)
                            {
                                gradI.y = (float)((*out.ptr(batch_idx, kp, lm, color)
                                                   - *out.ptr(batch_idx, km - 1, lm, color)));
                            }
                            else
                            {
                                gradI.y = 0;
                            }
                        }
                        //  float Iaorg = Ia, Jxorg = Jx, Jyorg = Jy, sorg = s;
                        Ia += (float)w * (float)(*out.ptr(batch_idx, km, lm, color));
                        Jx -= (float)w * (float)(gradI.x * r.x);
                        Jy -= (float)w * (float)(gradI.y * r.y);
                        s += w;
                    }
                }
            }
        }
        sat = (float)((Ia / s + (Jx + Jy) / (sqrt(Jx * Jx + Jy * Jy) + 1.0e-20f) + 0.5f));
        {
            *out.ptr(batch_idx, i - 1, j - 1, color) = SaturateCast<uchar>(sat); // nan
        }
    }
}

template<typename T>
__global__ void TeleaInpaintFMM(Ptr2dNHWC<unsigned char> f, Ptr2dNHWC<float> t, Tensor4DWrap<T> out, int range,
                                Ptr2dNHWC<unsigned char> band, int ch)
{
    int       i = 0, j = 0;
    float     dist;
    const int ii        = blockIdx.x * blockDim.x + threadIdx.x; // row
    const int jj        = blockIdx.y * blockDim.y + threadIdx.y; // col
    const int batch_idx = get_batch_idx();
    if (ii >= f.rows || jj >= f.cols)
        return;
    // method1 all the thread do the computation toghther for iteration times until convergence
    if (*band.ptr(batch_idx, ii, jj) != 0)
    {
        *f.ptr(batch_idx, ii, jj) = KNOWN;
        for (int q = 0; q < 4; q++)
        {
            if (q == 0)
            {
                i = ii - 1;
                j = jj;
            }
            else if (q == 1)
            {
                i = ii;
                j = jj - 1;
            }
            else if (q == 2)
            {
                i = ii + 1;
                j = jj;
            }
            else if (q == 3)
            {
                i = ii;
                j = jj + 1;
            }
            if ((i <= 0) || (j <= 0) || (i >= t.rows - 1) || (j >= t.cols - 1))
                continue; // fix bug of incorrect border

            if (*f.ptr(batch_idx, i, j) == INSIDE)
            {
                dist = min4(FastMarching_solve(i - 1, j, i, j - 1, f, t), FastMarching_solve(i + 1, j, i, j - 1, f, t),
                            FastMarching_solve(i - 1, j, i, j + 1, f, t), FastMarching_solve(i + 1, j, i, j + 1, f, t));
                *t.ptr(batch_idx, i, j) = dist;

                inpaint(f, t, out, i, j, range, ch);

                *f.ptr(batch_idx, i, j)    = BAND;
                *band.ptr(batch_idx, i, j) = 1; // non-zero
            }
        }
        *band.ptr(batch_idx, ii, jj) = 0;
    }
}

template<typename Ptr2D, typename T, typename D, typename BrdRd>
__global__ void dilate(const BrdRd src, Ptr2D dst, const unsigned char *__restrict__ kernel, const int kWidth,
                       const int kHeight, const int anchorX, const int anchorY, T maxmin)
{
    using work_type = nvcv::cuda::ConvertBaseTypeTo<T, D>;
    work_type res   = nvcv::cuda::SetAll<T>(maxmin);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.cols || y >= dst.rows)
        return;

    int kInd = 0;

    for (int i = 0; i < kHeight; ++i)
    {
        for (int j = 0; j < kWidth; ++j)
        {
            if (kernel[kInd++])
            {
                res = max(res, src(batch_idx, y - anchorY + i, x - anchorX + j));
            }
        }
    }

    *dst.ptr(batch_idx, y, x) = SaturateCast<D>(res);
}

template<typename T>
void MorphFilter2DCaller(Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, const unsigned char *kernel, int kWidth, int kHeight,
                         int anchorX, int anchorY, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y), dst.batches);

    T               val = std::numeric_limits<T>::min();
    BrdReplicate<T> brd(dst.rows, dst.cols, nvcv::cuda::SetAll<T>(val)); // set all value as val
    BorderReader<Ptr2dNHWC<T>, BrdReplicate<T>> brdSrc(src, brd);

    dilate<Ptr2dNHWC<T>, T, T, BorderReader<Ptr2dNHWC<T>, BrdReplicate<T>>>
        <<<grid, block, 0, stream>>>(brdSrc, dst, kernel, kWidth, kHeight, anchorX, anchorY, val);
    checkKernelErrors();
}

template<typename T>
__global__ void sub_kernel(Ptr2dNHWC<T> src1, Ptr2dNHWC<T> src2, Ptr2dNHWC<T> dst)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (x >= dst.cols || y >= dst.rows)
        return;

    for (int c = 0; c < dst.ch; c++)
    {
        *dst.ptr(batch_idx, y, x, c) = *src1.ptr(batch_idx, y, x, c) - *src2.ptr(batch_idx, y, x, c);
    }
}

template<typename T>
void sub_helper(Ptr2dNHWC<T> src1, Ptr2dNHWC<T> src2, Ptr2dNHWC<T> dst, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y), dst.batches);
    sub_kernel<T><<<grid, block, 0, stream>>>(src1, src2, dst);
    checkKernelErrors();
}

template<typename T>
__global__ void set_border_hori(Ptr2dNHWC<T> src, T value)
{
    const int i         = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = get_batch_idx();
    if (i >= src.cols)
        return;

    for (int c = 0; c < src.ch; c++)
    {
        *src.ptr(batch_idx, 0, i, c)            = value;
        *src.ptr(batch_idx, src.rows - 1, i, c) = value;
    }
}

template<typename T>
__global__ void set_border_vert(Ptr2dNHWC<T> src, T value)
{
    const int i         = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = get_batch_idx();
    if (i >= src.rows)
        return;

    for (int c = 0; c < src.ch; c++)
    {
        *src.ptr(batch_idx, i, 0, c)            = value;
        *src.ptr(batch_idx, i, src.cols - 1, c) = value;
    }
}

template<typename T>
void set_border_helper(Ptr2dNHWC<T> src, T value, cudaStream_t stream)
{
    dim3 block(16, 1);
    dim3 grid(divUp(max(src.cols, src.rows), block.x), 1, src.batches);
    set_border_hori<T><<<grid, block, 0, stream>>>(src, value);
    checkKernelErrors();
    set_border_vert<T><<<grid, block, 0, stream>>>(src, value);
    checkKernelErrors();
}

template<typename T, typename T2>
__global__ void set_value_kernel(Ptr2dNHWC<T> src, Ptr2dNHWC<T2> mask, T value)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (x >= src.cols || y >= src.rows)
        return;

    if (*mask.ptr(batch_idx, y, x))
    {
        for (int c = 0; c < src.ch; c++)
        {
            *src.ptr(batch_idx, y, x, c) = value;
        }
    }
}

template<typename T, typename T2>
void set_value(Ptr2dNHWC<T> src, Ptr2dNHWC<T2> mask, T value, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y), src.batches);
    set_value_kernel<<<grid, block, 0, stream>>>(src, mask, value);
    checkKernelErrors();
}

__global__ void deviceReduceImage(Ptr2dNHWC<unsigned char> src_ptr, int *out)
{
    int batch_idx        = get_batch_idx();
    int block_idx        = get_bid();
    int grid_size        = get_grid_size();
    int height           = src_ptr.rows;
    int width            = src_ptr.cols;
    int tid              = threadIdx.x;
    int out_batch_offset = batch_idx * grid_size + block_idx; // xmin, ymin, xmax, ymax intervleaved
    int sum              = 0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < height * width; i += blockDim.x * gridDim.x)
    {
        int x = i % width;
        int y = i / width;
        sum += *src_ptr.ptr(batch_idx, y, x);
    }

    sum = blockReduceSum(sum);
    if (tid == 0)
    {
        out[out_batch_offset] = sum;
    }
}

inline int finish_flag_reduce(Ptr2dNHWC<unsigned char> src_ptr, int *d_out, int *d_out2, cudaStream_t stream)
{
    int  block_size = 256; // maximum threads per block
    int  g_size     = REDUCE_GRID_SIZE;
    dim3 grid_size(g_size, 1, src_ptr.batches); // 32 resident blocks per sm

    deviceReduceImage<<<grid_size, block_size, 0, stream>>>(src_ptr, d_out);
    checkKernelErrors();

    int  block_size2 = g_size;
    dim3 grid_size2(1, 1, 1);
    int  points_set_num = g_size * src_ptr.batches;
    deviceReducePoints<<<grid_size2, block_size2, 0, stream>>>(d_out, d_out2, points_set_num);
    checkKernelErrors();
    int rst = 0;
    checkCudaErrors(cudaMemcpyAsync((void *)&rst, (void *)d_out2, sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    return rst;
}

template<typename T>
void inpaint_helper(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &mask,
                    const nvcv::TensorDataStridedCuda &outData, void *workspace, unsigned char *kernel_ptr, int range,
                    bool &init_flag, int batch, int height, int width, int channel, int maxBatchSize,
                    cudaStream_t stream)
{
    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(width + 2, blockSize.x), divUp(height + 2, blockSize.y), batch);

    auto dst = CreateTensorWrapNHWC<T>(outData);
    // data type for mask is 8UC1
    auto org_mask = CreateTensorWrapNHWC<unsigned char>(mask);

    // create t and f pointer
    int ecols = width + 2;
    int erows = height + 2;

    int   *block_reduce_buffer1 = (int *)workspace;
    int   *block_reduce_buffer2 = (int *)((char *)block_reduce_buffer1 + sizeof(int) * REDUCE_GRID_SIZE * maxBatchSize);
    float *t_ptr                = (float *)((char *)block_reduce_buffer2 + sizeof(int) * 1);
    unsigned char *f_ptr        = (unsigned char *)((char *)t_ptr + sizeof(float) * batch * erows * ecols * 1);
    unsigned char *band_ptr     = (unsigned char *)((char *)f_ptr + sizeof(unsigned char) * batch * erows * ecols * 1);
    unsigned char *inpaint_mask_ptr
        = (unsigned char *)((char *)band_ptr + sizeof(unsigned char) * batch * erows * ecols * 1);

    Ptr2dNHWC<unsigned char> f(batch, erows, ecols, 1, (unsigned char *)f_ptr);
    Ptr2dNHWC<float>         t(batch, erows, ecols, 1, (float *)t_ptr);
    Ptr2dNHWC<unsigned char> band(batch, erows, ecols, 1, (unsigned char *)band_ptr);
    Ptr2dNHWC<unsigned char> inpaint_mask(batch, erows, ecols, 1, (unsigned char *)inpaint_mask_ptr);

    // copy input to output
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    for (uint32_t i = 0; i < inAccess->numSamples(); ++i)
    {
        void *inSampData  = inAccess->sampleData(i);
        void *outSampData = outAccess->sampleData(i);

        checkCudaErrors(cudaMemcpy2DAsync(outSampData, outAccess->rowStride(), inSampData, inAccess->rowStride(),
                                          inAccess->numCols() * inAccess->colStride(), inAccess->numRows(),
                                          cudaMemcpyDeviceToDevice, stream));
    }

    // step1 init mask

    // set inpaint mask to KNOWN
    checkCudaErrors(cudaMemsetAsync(inpaint_mask_ptr, KNOWN, sizeof(unsigned char) * batch * erows * ecols * 1,
                                    stream)); // cvSet(mask,cvScalar(KNOWN,0,0,0));
    // copy !=0 value to mask
    int2 size = {width, height};
    copy_mask_data<unsigned char><<<gridSize, blockSize, 0, stream>>>(
        org_mask, inpaint_mask, 1, 1, INSIDE, size); // COPY_MASK_BORDER1_C1(inpaint_mask,mask,uchar);
    checkKernelErrors();

    // set border to 0
    set_border_helper<unsigned char>(inpaint_mask, KNOWN, stream); // SET_BORDER1_C1(mask,uchar,0); KNOWN = 0
    // dump_img(erows, ecols, 1, inpaint_mask_ptr, kCV_8U, "./inpaint_mask_set_border.png");
    // step2 init t and f

    checkCudaErrors(cudaMemsetAsync(f_ptr, KNOWN, sizeof(unsigned char) * batch * erows * ecols * 1,
                                    stream)); // cvSet(f,cvScalar(KNOWN,0,0,0));
    checkCudaErrors(cudaMemsetAsync(t_ptr, 1.0e6f, sizeof(float) * batch * erows * ecols * 1,
                                    stream)); // cvSet(t,cvScalar(1.0e6f,0,0,0));

    // step3 init band

    if (!init_flag)
    {
        init_dilate_kernel(kernel_ptr, stream);
        init_flag = true;
    }

    // src, dst, ele, iteration
    MorphFilter2DCaller(inpaint_mask, band, kernel_ptr, 3, 3, 1, 1,
                        stream); // cvDilate(mask,b   and,el_cross,1);   // image with narrow band

    // step 4 init Heap

    sub_helper(band, inpaint_mask, band, stream); // cvSub(band,mask,band,NULL);

    set_border_helper<unsigned char>(band, 0, stream); // SET_BORDER1_C1(band,uchar,0);
    // dump_img(erows, ecols, 1, band_ptr, kCV_8U, "./band_set_border.png");
    // !!!! here we use band to inpaint the image

    // src, value, mask
    set_value<unsigned char, unsigned char>(f, band, BAND, stream);           // cvSet(f,cvScalar(BAND,0,0,0),band);
    set_value<unsigned char, unsigned char>(f, inpaint_mask, INSIDE, stream); // cvSet(f,cvScalar(INSIDE,0,0,0),mask);
    set_value<float, unsigned char>(t, band, 0, stream);                      // cvSet(t,cvScalar(0,0,0,0),band);

    // step5 FMM

    int  iteration = 20;
    dim3 block(BLOCK_S, BLOCK_S);
    dim3 grid(divUp(f.rows, block.x), divUp(f.cols, block.y), f.batches);
    int  flag = 1;

    while (flag)
    {
        for (int i = 0; i < iteration; i++)
        {
            TeleaInpaintFMM<T><<<grid, block, 0, stream>>>(
                inpaint_mask, t, dst, range, band, channel); // icvTeleaInpaintFMM<uchar>(mask,t,output_img,range,Heap);
        }
        flag = finish_flag_reduce(band, block_reduce_buffer1, block_reduce_buffer2, stream);
    }

    checkKernelErrors();
}

namespace nvcv::legacy::cuda_op {

Inpaint::Inpaint(DataShape max_input_shape, DataShape max_output_shape, int maxBatchSize, Size2D maxShape)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_init_dilate(false)
    , m_maxBatchSize(maxBatchSize)
    , m_kernel_ptr(nullptr)
    , m_workspace(nullptr)
{
    cudaError_t err = cudaMalloc(&m_kernel_ptr, sizeof(unsigned char) * maxBatchSize * 3 * 3);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA memory allocation error of size: " << sizeof(uchar) * maxBatchSize * 3 * 3);
        throw std::runtime_error("CUDA memory allocation error!");
    }

    int    erows      = (maxShape.h + 2);
    int    ecols      = (maxShape.w + 2);
    size_t buffersize = sizeof(int) * (REDUCE_GRID_SIZE * maxBatchSize + 1)
                      + maxBatchSize * erows * ecols * 1 * (sizeof(float) + sizeof(unsigned char) * 3);
    err = cudaMalloc(&m_workspace, buffersize);
    if (err != cudaSuccess)
    {
        cudaFree(m_kernel_ptr);
        LOG_ERROR("CUDA memory allocation error of size: " << buffersize);
        throw std::runtime_error("CUDA memory allocation error!");
    }
}

Inpaint::~Inpaint()
{
    cudaError_t err0 = cudaFree(m_kernel_ptr);
    cudaError_t err1 = cudaFree(m_workspace);
    if (err0 != cudaSuccess || err1 != cudaSuccess)
    {
        LOG_ERROR("CUDA memory free error, possible memory leak!");
    }
}

ErrorCode Inpaint::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &masks,
                         const TensorDataStridedCuda &outData, double inpaintRadius, cudaStream_t stream)
{
    DataFormat in_format    = GetLegacyDataFormat(inData.layout());
    DataType   in_data_type = GetLegacyDataType(inData.dtype());
    if (!(in_format == kNHWC || in_format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << in_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(in_data_type == kCV_8U || in_data_type == kCV_32S || in_data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << in_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);
    const int in_channels = inAccess->numChannels();

    if (in_channels > 4)
    {
        LOG_ERROR("Invalid channel number " << in_channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataFormat out_format    = GetLegacyDataFormat(outData.layout());
    DataType   out_data_type = GetLegacyDataType(outData.dtype());
    if (!(out_format == kNHWC || out_format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << out_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (in_data_type != out_data_type)
    {
        LOG_ERROR("Invalid DataType " << out_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);
    const int out_channels = outAccess->numChannels();

    if (out_channels != in_channels)
    {
        LOG_ERROR("Invalid channel number " << out_channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType mask_data_type = GetLegacyDataType(masks.dtype());
    if (mask_data_type != kCV_8U)
    {
        LOG_ERROR("Invalid mask DataType " << mask_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    auto maskAccess = TensorDataAccessStridedImagePlanar::Create(masks);
    NVCV_ASSERT(maskAccess);
    const int mask_channels = maskAccess->numChannels();
    if (mask_channels != 1)
    {
        LOG_ERROR("Invalid mask channel number " << mask_channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    typedef void (*inpaint_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &mask,
                              const TensorDataStridedCuda &outData, void *workspace, unsigned char *kernel_ptr,
                              int range, bool &init_flag, int batch, int height, int width, int channel,
                              int maxBatchSize, cudaStream_t stream);

    static const inpaint_t funcs[6] = {
        inpaint_helper<unsigned char>, inpaint_helper<char>, 0, 0, inpaint_helper<int>, inpaint_helper<float>,

    };
    int range = (int)std::round(inpaintRadius);
    range     = std::max(range, 1);
    range     = std::min(range, 100);
    funcs[in_data_type](inData, masks, outData, m_workspace, m_kernel_ptr, range, m_init_dilate, inAccess->numSamples(),
                        inAccess->numRows(), inAccess->numCols(), in_channels, m_maxBatchSize, stream);
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

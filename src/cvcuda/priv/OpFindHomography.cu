/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpFindHomography.hpp"

#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>
#include <driver_types.h>
#include <float.h>
#include <nvcv/ArrayData.hpp>
#include <nvcv/ArrayDataAccess.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <iostream>

#define BLOCK_SIZE 128
#define PIPELINES  8

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

typedef cuda::math::Vector<float, 8>     vector8;
typedef cuda::math::Vector<float, 9>     vector9;
typedef cuda::math::Vector<int, 8>       intvector8;
typedef cuda::math::Vector<float, 32>    vector32;
typedef cuda::math::Matrix<float, 8, 8>  matrix8x8;
typedef cuda::math::Matrix<float, 8, 32> matrix8x32;
typedef cuda::math::Matrix<double, 8, 8> dmatrix8x8;
typedef cuda::math::Vector<double, 8>    dvector8;

namespace {

#define is_aligned(POINTER, BYTE_COUNT, msg)                                                \
    do                                                                                      \
    {                                                                                       \
        if (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) != 0)                       \
        {                                                                                   \
            std::cerr << msg << " at line " << __LINE__ << " in " << __FILE__ << std::endl; \
            return;                                                                         \
        }                                                                                   \
    }                                                                                       \
    while (0)

#define CUDA_CHECK_ERROR(err, msg)                                                                                  \
    do                                                                                                              \
    {                                                                                                               \
        cudaError_t _err = (err);                                                                                   \
        if (_err != cudaSuccess)                                                                                    \
        {                                                                                                           \
            std::cerr << "(" << cudaGetErrorString(_err) << ") at line " << __LINE__ << " in " << __FILE__ << " : " \
                      << msg << std::endl;                                                                          \
            return;                                                                                                 \
        }                                                                                                           \
    }                                                                                                               \
    while (0)

#define CUBLAS_CHECK_ERROR(err, msg)                                                                                \
    do                                                                                                              \
    {                                                                                                               \
        cublasStatus_t _err = (err);                                                                                \
        if (_err != CUBLAS_STATUS_SUCCESS)                                                                          \
        {                                                                                                           \
            std::cerr << "CUBLAS error (" << _err << ") at line " << __LINE__ << " in " << __FILE__ << " : " << msg \
                      << std::endl;                                                                                 \
            return;                                                                                                 \
        }                                                                                                           \
    }                                                                                                               \
    while (0)

#define CUSOLVER_CHECK_ERROR(err, msg)                                                                                \
    do                                                                                                                \
    {                                                                                                                 \
        cusolverStatus_t _err = (err);                                                                                \
        if (_err != CUSOLVER_STATUS_SUCCESS)                                                                          \
        {                                                                                                             \
            std::cerr << "CUSOLVER error (" << _err << ") at line " << __LINE__ << " in " << __FILE__ << " : " << msg \
                      << std::endl;                                                                                   \
            return;                                                                                                   \
        }                                                                                                             \
    }                                                                                                                 \
    while (0)

#ifdef DEBUG
template<typename T>
__global__ void printKernel(T *data, int numPoints, int batchIdx)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < numPoints)
        printf("Batch = %d, i = %d, val = %.9g,\n", batchIdx, i, (double)data[i]);
}

__global__ void printKernelfloat2(float2 *data, int numPoints, int batchIdx)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < numPoints)
        printf("Batch = %d, i = %d, val = %.9g,%.9g\n", batchIdx, i, (double)data[i].x, (double)data[i].y);
}
#endif

#ifdef DEBUG_MODEL_KERNEL
template<typename T>
__global__ void printMatrix(T *data, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.9g, ", (double)data[i * N + j]);
        }
        printf("\n");
    }
}

template<typename T>
__global__ void printMatrixCols(T *data, int M, int N)
{
    for (int j = 0; j < N; j++)
    {
        printf("ROw %d\n", j);
        for (int i = 0; i < M; i++)
        {
            printf("A[%d + %d * lda] = %g;\n", i, j, (double)data[i * N + j]);
        }
        printf("\n");
    }
}

template<typename T, int N>
__device__ void printMatrixDevice(cuda::math::Matrix<T, N> &A)
{
    for (int i = 0; i < N; i++)
    {
        printf("[");
        for (int j = 0; j < N; j++)
        {
            printf("%.9g, ", A[i][j]);
        }
        printf("],\n");
    }
}

template<typename T, int N>
__device__ void printMatrixDeviceParallel(cuda::math::Matrix<T, N> &A)
{
    __threadfence();
    if (threadIdx.x < N)
    {
        for (int i = 0; i < N; i++)
        {
            printf("A[%d][%d] = %g\n", i, threadIdx.x, (double)A[i][threadIdx.x]);
        }
    }
}

template<typename T>
__device__ void printMatrixDeviceRaw(T *A, int M, int N, int batch)
{
    printf("Batch = %d\n", batch);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%g, ", (double)A[i * N + j]);
        }
        printf("\n");
    }
}

template<typename T, int N>
__device__ void printVectorDevice(cuda::math::Vector<T, N> &x, int batch)
{
    printf("Batch = %d\n", batch);
    for (int i = 0; i < N; i++) printf("%.9g, ", x[i]);
}

template<typename T, int N>
__device__ void printVectorDeviceParallel(cuda::math::Vector<T, N> &x, int batch)
{
    if (threadIdx.x < N)
        printf("x[%d] = %g\n", threadIdx.x, (double)x[threadIdx.x]);
}

template<typename T>
__device__ void printVectorDeviceRaw(T *x, int N, int batch)
{
    printf("Batch = %d\n", batch);
    for (int i = 0; i < N; i++) printf("%g, ", (double)x[i]);
}
#endif

__device__ void calculate_residual_and_jacobian_device(float2 *src, float2 *dst, vector8 &h, int numPoints, float *Jptr,
                                                       float *errptr);
__device__ void calculate_residual_norm(float *r, float *r_norm_2, vector32 &warpSums, int numPoints);

__device__ void calculate_Jtx_matvec(float *A, float *B, float *result, matrix8x32 &warpSums, int row, int numPoints);

__device__ void calculate_JtJ(float *Jt, matrix8x8 &A, matrix8x32 &warpSums, float *reductionBuffer, int numPoints);

__device__ void calculate_Jtr(float *Jt, float *r, vector8 &v, matrix8x32 &warpSums, float *reductionBuffer,
                              int numPoints);

__device__ void fetch_diagonal(matrix8x8 &A, vector8 &D, int tid);

__device__ void copy_A_to_Ap_App(matrix8x8 &A, matrix8x8 &Ap, matrix8x8 &App);

__device__ void scale_diagonal8(vector8 &D, matrix8x8 &Ap, float lambda);

__device__ void compute_qr8x8(matrix8x8 &sA, matrix8x8 &sQ);

__device__ bool backsolve_inplace(matrix8x8 &A, vector8 &d);

__device__ bool solve8x8(matrix8x8 &A, matrix8x8 &Q, vector8 &v, vector8 &d, int tid);

__device__ bool invert8x8(matrix8x8 &A, matrix8x8 &Q, matrix8x8 &invA, int tid);

__device__ void subtract8(vector8 &x, vector8 &d, vector8 &xd, int tid);

__device__ void max_diag_val8(matrix8x8 &A, float *maxval);

__device__ void max8(vector8 &v, float *maxval);

__device__ static float atomicMax(float *address, float val);

__device__ void calculate_temp_d(matrix8x8 &A, vector8 &x, vector8 &y, vector8 &z, float alpha, float beta, int tid);

__device__ int compute_model_estimate(float2 cM, float2 cm, float2 sM, float2 sm, float *W, float *V, vector8 &x,
                                      cuda::Tensor3DWrap<float> model, int batch, int numPoints);

__device__ void calculate_residual_and_jacobian_device(float2 *src, float2 *dst, vector8 &h, int numPoints, float *Jptr,
                                                       float *errptr)
{
    int idx = threadIdx.x;

    for (int tid = idx; tid < numPoints; tid += blockDim.x)
    {
        float2 M_i = src[tid];
        float2 m_i = dst[tid];
        float  Mx = M_i.x, My = M_i.y;
        float  mx = m_i.x, my = m_i.y;

        float ww = h[6] * Mx + h[7] * My + 1.;
        ww       = fabs(ww) > FLT_EPSILON ? 1. / ww : 0;
        float xi = (h[0] * Mx + h[1] * My + h[2]) * ww;
        float yi = (h[3] * Mx + h[4] * My + h[5]) * ww;

        errptr[tid * 2]     = xi - mx;
        errptr[tid * 2 + 1] = yi - my;

        if (Jptr)
        {
            // Column major format
            Jptr[tid * 2 + numPoints * 0 + 0]  = Mx * ww;
            Jptr[tid * 2 + numPoints * 0 + 1]  = 0;
            Jptr[tid * 2 + numPoints * 2 + 0]  = My * ww;
            Jptr[tid * 2 + numPoints * 2 + 1]  = 0;
            Jptr[tid * 2 + numPoints * 4 + 0]  = ww;
            Jptr[tid * 2 + numPoints * 4 + 1]  = 0;
            Jptr[tid * 2 + numPoints * 6 + 0]  = 0;
            Jptr[tid * 2 + numPoints * 6 + 1]  = Mx * ww;
            Jptr[tid * 2 + numPoints * 8 + 0]  = 0;
            Jptr[tid * 2 + numPoints * 8 + 1]  = My * ww;
            Jptr[tid * 2 + numPoints * 10 + 0] = 0;
            Jptr[tid * 2 + numPoints * 10 + 1] = ww;
            Jptr[tid * 2 + numPoints * 12 + 0] = -Mx * ww * xi;
            Jptr[tid * 2 + numPoints * 12 + 1] = -Mx * ww * yi;
            Jptr[tid * 2 + numPoints * 14 + 0] = -My * ww * xi;
            Jptr[tid * 2 + numPoints * 14 + 1] = -My * ww * yi;
        }
    }
}

__device__ inline float myfabs(float val)
{
    return fabsf(val);
}

inline __device__ float2 myfabs2(float2 val)
{
    float2 ret;
    ret.x = fabsf(val.x);
    ret.y = fabsf(val.y);
    return ret;
}

__device__ inline int getNumPoints(cuda::Tensor2DWrap<float2> src, int numPoints, int batch)
{
    return numPoints;
}

struct MeanOp
{
    __device__ float2 eval(float2 val, int numPoints, int batch)
    {
        return val / numPoints;
    }
};

struct SquareOp
{
    __device__ float eval(float val, int batch)
    {
        return val * val;
    }
};

class AbsShiftOp
{
private:
    float2 *_data;

public:
    // Constructor that takes a float* pointer as a parameter
    __host__ AbsShiftOp(float2 *data)
        : _data(data){};

    // Method to update the float value pointed to by the pointer
    __device__ float2 eval(float2 newVal, int numPoints, int batch)
    {
        _data += batch;
        return myfabs2(newVal - _data[0]);
    }
};

class LtLOp
{
private:
    float2 *cm, *cM, *sm, *sM;

public:
    __host__ LtLOp(float2 *srcMean, float2 *dstMean, float2 *srcShiftSum, float2 *dstShiftSum)
    {
        cM = srcMean;
        sM = srcShiftSum;
        cm = dstMean;
        sm = dstShiftSum;
    }

    __device__ float eval(float2 *src, float2 *dst, int batch, int numPoints, int tid, int j, int k)
    {
        cm += batch;
        cM += batch;
        sm += batch;
        sM += batch;
        float X     = (src[tid].x - cM[0].x) * (numPoints / sM[0].x);
        float Y     = (src[tid].y - cM[0].y) * (numPoints / sM[0].y);
        float x     = (dst[tid].x - cm[0].x) * (numPoints / sm[0].x);
        float y     = (dst[tid].y - cm[0].y) * (numPoints / sm[0].y);
        float Lx[9] = {X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x};
        float Ly[9] = {0, 0, 0, X, Y, 1, -y * X, -y * Y, -y};
        return Lx[j] * Lx[k] + Ly[j] * Ly[k];
    }
};

template<class Func>
__device__ void reducef(float *data, cuda::math::Vector<float, 32> &warpSums, float *result, Func op, int numPoints,
                        int batch)
{
    int      tid    = threadIdx.x;
    int      idx    = threadIdx.x + blockIdx.x * blockDim.x;
    float    val    = 0.0f;
    unsigned mask   = 0xFFFFFFFFU;
    int      lane   = threadIdx.x % warpSize;
    int      warpID = threadIdx.x / warpSize;
    while (idx < numPoints)
    {
        val += op.eval(data[idx], batch);
        idx += gridDim.x * blockDim.x;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) val += __shfl_down_sync(mask, val, offset);
    if (lane == 0)
        warpSums[warpID] = val;

    __syncthreads();

    if (warpID == 0)
    {
        val = (tid < blockDim.x / warpSize) ? warpSums[lane] : 0.0f;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) val += __shfl_down_sync(mask, val, offset);

        if (tid == 0)
            atomicAdd(result, val);
    }
}

template<class Func>
__device__ void reducef2(float2 *data, cuda::math::Vector<float2, 32> &warpSums, float2 *result, Func op, int numPoints,
                         int batch)
{
    int      tid    = threadIdx.x;
    int      idx    = threadIdx.x + blockIdx.x * blockDim.x;
    float2   val    = {0.0f, 0.0f};
    unsigned mask   = 0xFFFFFFFFU;
    int      lane   = threadIdx.x % warpSize;
    int      warpID = threadIdx.x / warpSize;
    while (idx < numPoints)
    {
        val += op.eval(data[idx], numPoints, batch);
        idx += gridDim.x * blockDim.x;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val.x += __shfl_down_sync(mask, val.x, offset);
        val.y += __shfl_down_sync(mask, val.y, offset);
    }
    if (lane == 0)
        warpSums[warpID] = val;

    __syncthreads();

    if (warpID == 0)
    {
        val = (tid < blockDim.x / warpSize) ? warpSums[lane] : float2{0.0f, 0.0f};

        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val.x += __shfl_down_sync(mask, val.x, offset);
            val.y += __shfl_down_sync(mask, val.y, offset);
        }

        if (tid == 0)
        {
            atomicAdd(&result[0].x, val.x);
            atomicAdd(&result[0].y, val.y);
        }
    }
}

template<class Func>
__device__ void reduceLtL(float2 *src, float2 *dst, cuda::math::Vector<float, 32> &warpSums, float *result, Func op,
                          int numPoints, int batch, int j, int k)
{
    int   tid = threadIdx.x;
    int   idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.0f;
    ;
    unsigned mask   = 0xFFFFFFFFU;
    int      lane   = threadIdx.x % warpSize;
    int      warpID = threadIdx.x / warpSize;
    while (idx < numPoints)
    {
        // j < 9 and k < 9 are indices of the LtL matrix
        val += op.eval(src, dst, batch, numPoints, idx, j, k);
        idx += gridDim.x * blockDim.x;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) val += __shfl_down_sync(mask, val, offset);
    if (lane == 0)
        warpSums[warpID] = val;
    __syncthreads();

    if (warpID == 0)
    {
        val = (tid < blockDim.x / warpSize) ? warpSums[lane] : 0.0f;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) val += __shfl_down_sync(mask, val, offset);

        if (tid == 0)
            atomicAdd(result, val);
    }
}

__device__ void calculate_residual_norm(float *r, float *r_norm_2, vector32 &warpSums, int numPoints)
{
    SquareOp square_op;
    reducef<SquareOp>(r, warpSums, r_norm_2, square_op, numPoints, 0);
    __syncthreads();
}

__device__ void calculate_Jtx_matvec(float *A, float *B, float *result, matrix8x32 &warpSums, int row, int numPoints)
{
    // NOTE : Jt has to be of dimension (8 x innerDim) where innerDim = numPoints x 2
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x < 8)
    {
        for (int i = 0; i < 8; i++)
        {
            warpSums[i][threadIdx.x] = 0;
        }
    }
    __syncthreads();

    float    val[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    unsigned mask   = 0xFFFFFFFFU;
    int      lane   = threadIdx.x % warpSize;
    int      warpID = threadIdx.x / warpSize;
    while (idx < numPoints)
    {
        float src_data_val = A[row * numPoints + idx];
#pragma unroll
        for (int r = row; r < 8; r++) val[r] += src_data_val * B[r * numPoints + idx];
        idx += gridDim.x * blockDim.x;
    }

    for (int r = row; r < 8; r++)
    {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) val[r] += __shfl_down_sync(mask, val[r], offset);
        if (lane == 0)
            warpSums[r][warpID] = val[r];
    }
    __syncthreads();

    if (warpID == 0)
    {
#pragma unroll
        for (int r = row; r < 8; r++)
        {
            val[r] = (tid < blockDim.x / warpSize) ? warpSums[r][lane] : 0;

            for (int offset = warpSize / 2; offset > 0; offset >>= 1) val[r] += __shfl_down_sync(mask, val[r], offset);

            if (tid == 0)
                atomicAdd(&result[r], val[r]);
        }
    }
}

__device__ void calculate_JtJ(float *Jt, matrix8x8 &A, matrix8x32 &warpSums, float *reductionBuffer, int numPoints)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int row = 0; row < 8; row++)
    {
        if (tid < 8)
            reductionBuffer[tid] = 0;
        calculate_Jtx_matvec(Jt, Jt, reductionBuffer, warpSums, row, numPoints);
        __syncthreads();
        if (tid < 8)
        {
            A[row][tid] = reductionBuffer[tid];
        }
    }
    __syncwarp();

    for (int row = 1; row < 8; row++)
    {
        if (tid < row)
            A[row][tid] = A[tid][row];
    }
    __syncwarp();
}

__device__ void calculate_Jtr(float *Jt, float *r, vector8 &v, matrix8x32 &warpSums, float *reductionBuffer,
                              int numPoints)
{
    if (threadIdx.x < 8)
        reductionBuffer[threadIdx.x] = 0.0f;
    calculate_Jtx_matvec(r, Jt, reductionBuffer, warpSums, 0, numPoints);
    __syncthreads();
    if (threadIdx.x < 8)
        v[threadIdx.x] = reductionBuffer[threadIdx.x];
    __syncwarp();
}

__device__ void fetch_diagonal(matrix8x8 &A, vector8 &D, int tid)
{
    if (tid < 8)
        D[tid] = A[tid][tid];
    __syncwarp();
}

__device__ void copy_A_to_Ap_App(matrix8x8 &A, matrix8x8 &Ap, matrix8x8 &App)
{
    if (threadIdx.x < 8)
    {
        for (int i = 0; i < 8; i++)
        {
            Ap[i][threadIdx.x]  = A[i][threadIdx.x];
            App[i][threadIdx.x] = A[i][threadIdx.x];
        }
    }
    __syncwarp();
}

__device__ void scale_diagonal8(vector8 &D, matrix8x8 &Ap, float lambda)
{
    if (threadIdx.x < 8)
        Ap[threadIdx.x][threadIdx.x] += lambda * D[threadIdx.x];
    __syncwarp();
}

__device__ void compute_qr8x8(matrix8x8 &sA, matrix8x8 &sQ)
{
    int       tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int N   = 8;
    if (tid < N)
    {
        for (int i = 0; i < N; i++)
        {
            sQ[i][tid] = 0;
            if (i == tid)
                sQ[i][tid] = 1;
        }
    }
    __syncwarp();

    float  s[2];
    double temp[2];
    for (int j = 0; j < N; j++)
    {
        int pivot_row = j;
        for (int i = j + 1; i < N; i++)
        {
            if (tid < N)
            {
                double theta   = atan(-(double)sA[i][j] / (double)sA[pivot_row][j]);
                double ctheta  = cos(theta);
                double stheta  = sin(theta);
                float  sthetaf = (float)stheta;
                float  cthetaf = (float)ctheta;

                temp[0]            = ctheta * sA[pivot_row][tid] - stheta * sA[i][tid];
                temp[1]            = stheta * sA[pivot_row][tid] + ctheta * sA[i][tid];
                sA[pivot_row][tid] = temp[0];
                sA[i][tid]         = temp[1];

                s[0]               = cthetaf * sQ[pivot_row][tid] - sthetaf * sQ[i][tid];
                s[1]               = sthetaf * sQ[pivot_row][tid] + cthetaf * sQ[i][tid];
                sQ[pivot_row][tid] = s[0];
                sQ[i][tid]         = s[1];
            }
            __syncwarp();
        }
    }
    __syncwarp();
}

__device__ bool backsolve_inplace(matrix8x8 &A, vector8 &d)
{
    const int N = 8;
    for (int j = N - 1; j >= 0; j--)
    {
        if (A[j][j] < FLT_EPSILON)
            return false;
        d[j] /= A[j][j];
        for (int i = j - 1; i >= 0; i--)
        {
            d[i] = d[i] - A[i][j] * d[j];
        }
    }
    return true;
}

__device__ bool solve8x8(matrix8x8 &A, matrix8x8 &Q, vector8 &v, vector8 &d, int tid)
{
    // Do Q^T * d
    if (tid < 8)
    {
        d[tid] = 0;
        for (int i = 0; i < 8; i++) d[tid] += Q[tid][i] * v[i];
    }

    __syncwarp();

    if (tid == 0)
    {
        if (!backsolve_inplace(A, d))
            return false;
    }

    __syncwarp();

    return true;
}

__device__ bool invert8x8(matrix8x8 &A, matrix8x8 &Q, matrix8x8 &invA, int tid)
{
    if (tid < 8)
    {
        vector8 d = Q.col(tid);
        if (!backsolve_inplace(A, d))
            return false;
        invA.set_col(tid, d);
    }
    __syncwarp();
    return true;
}

__device__ void subtract8(vector8 &x, vector8 &d, vector8 &xd, int tid)
{
    if (tid < 8)
        xd[tid] = x[tid] - d[tid];
    __syncwarp();
}

__device__ inline void dot8(vector8 &x, vector8 &y, float *r)
{
    *r = x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3] + x[4] * y[4] + x[5] * y[5] + x[6] * y[6] + x[7] * y[7];
}

__device__ void max_diag_val8(matrix8x8 &A, float *maxval)
{
    *maxval = A[0][0];
    *maxval = fmaxf(A[1][1], *maxval);
    *maxval = fmaxf(A[2][2], *maxval);
    *maxval = fmaxf(A[3][3], *maxval);
    *maxval = fmaxf(A[4][4], *maxval);
    *maxval = fmaxf(A[5][5], *maxval);
    *maxval = fmaxf(A[6][6], *maxval);
    *maxval = fmaxf(A[7][7], *maxval);
}

__device__ void max8(vector8 &v, float *maxval)
{
    *maxval = fabsf(v[0]);
    *maxval = fmaxf(fabsf(v[1]), *maxval);
    *maxval = fmaxf(fabsf(v[2]), *maxval);
    *maxval = fmaxf(fabsf(v[3]), *maxval);
    *maxval = fmaxf(fabsf(v[4]), *maxval);
    *maxval = fmaxf(fabsf(v[5]), *maxval);
    *maxval = fmaxf(fabsf(v[6]), *maxval);
    *maxval = fmaxf(fabsf(v[7]), *maxval);
}

__device__ static float atomicMax(float *address, float val)
{
    int *address_as_i = (int *)address;
    int  old          = *address_as_i, assumed;
    do
    {
        assumed = old;
        old     = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    }
    while (assumed != old);
    return __int_as_float(old);
}

template<float (*Func)(float)>
__device__ void max(float *data, vector32 &warpSums, float *result, int numPoints)
{
    int      tid    = threadIdx.x;
    int      idx    = threadIdx.x + blockIdx.x * blockDim.x;
    float    val    = 0.0f;
    unsigned mask   = 0xFFFFFFFFU;
    int      lane   = threadIdx.x % warpSize;
    int      warpID = threadIdx.x / warpSize;
    while (idx < numPoints)
    {
        val = fmaxf(val, Func(data[idx]));
        idx += gridDim.x * blockDim.x;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    if (lane == 0)
        warpSums[warpID] = val;
    __syncthreads();

    if (warpID == 0)
    {
        val = (tid < blockDim.x / warpSize) ? warpSums[lane] : 0;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) val = fmaxf(val, __shfl_down_sync(mask, val, offset));

        if (tid == 0)
            atomicMax(result, val);
    }
}

__device__ void calculate_temp_d(matrix8x8 &A, vector8 &x, vector8 &y, vector8 &z, float alpha, float beta, int tid)
{
    if (tid < 8)
    {
        z[tid] = beta * y[tid];
#pragma unroll
        for (int i = 0; i < 8; i++) z[tid] += alpha * A[tid][i] * x[i];
    }
    __syncwarp();
}

__device__ int compute_model_estimate(float2 cM, float2 cm, float2 sM, float2 sm, float *W, float *V, vector8 &x,
                                      cuda::Tensor3DWrap<float> model, int batch, int numPoints)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (sm.x < FLT_EPSILON || sm.y < FLT_EPSILON || sM.x < FLT_EPSILON || sM.y < FLT_EPSILON)
    {
        if (tid < 8)
            x[tid] = 0;
        __syncwarp();
        return 1;
    }

    // compute model estimate
    float2 _sm{numPoints / sm.x, numPoints / sm.y};
    float2 _sM{numPoints / sM.x, numPoints / sM.y};

    int   minIdx = 0;
    float minEig = fabs(W[0]);

    for (int i = 1; i < 9; i++)
    {
        if (fabs(W[i]) < minEig)
        {
            minIdx = i;
            minEig = fabs(W[i]);
        }
    }

    float *H0 = V + 9 * minIdx;

#ifdef DEBUG_MODEL_ESTIMATE
    if (tid == 0)
    {
        for (int i = 0; i < 9; i++) printf("H0[%d] = %.9g\n", i, H0[i]);
    }
#endif

    cuda::math::Matrix<float, 3, 3> tH0;
    cuda::math::Matrix<float, 3, 3> tHtemp1;
    cuda::math::Matrix<float, 3, 3> tHtemp2;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            tH0[i][j] = H0[i * 3 + j];
        }
    }

    // load inv_Hnorm
    tHtemp2[0][0] = 1.0f / _sm.x;
    tHtemp2[0][1] = 0.0f;
    tHtemp2[0][2] = cm.x;
    tHtemp2[1][0] = 0.0f;
    tHtemp2[1][1] = 1.0f / _sm.y;
    tHtemp2[1][2] = cm.y;
    tHtemp2[2][0] = 0.0f;
    tHtemp2[2][1] = 0.0f;
    tHtemp2[2][2] = 1.0f;
    tHtemp1       = tHtemp2 * tH0;

#ifdef DEBUG_MODEL_ESTIMATE
    if (tid == 0)
    {
        printf("\n========================_Htemp=========================\n");
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                printf("_Htemp[%d][%d] = %.9g,", i, j, tHtemp1[i][j]);
            }
            printf("\n");
        }
    }
#endif

    // load Hnorm2
    tHtemp2[0][0] = _sM.x;
    tHtemp2[0][1] = 0.0f;
    tHtemp2[0][2] = -cM.x * _sM.x;
    tHtemp2[1][0] = 0.0f;
    tHtemp2[1][1] = _sM.y;
    tHtemp2[1][2] = -cM.y * _sM.y;
    tHtemp2[2][0] = 0.0f;
    tHtemp2[2][1] = 0.0f;
    tHtemp2[2][2] = 1.0f;
    tH0           = tHtemp1 * tHtemp2;

#ifdef DEBUG_MODEL_ESTIMATE
    if (tid == 0)
    {
        printf("\n===============_H0====================\n");
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                printf("_H0[%d][%d] = %.9g,", i, j, tH0[i][j]);
            }
            printf("\n");
        }
    }
#endif

#pragma unroll
    for (int i = 0; i < 3; i++)
#pragma unroll
        for (int j = 0; j < 3; j++) tH0[i][j] = tH0[i][j] / tH0[2][2];

#ifdef DEBUG_MODEL_ESTIMATE
    if (tid == 0)
    {
        printf("\n===============_H0====================\n");
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                printf("_H0[%d][%d] = %.9g,", i, j, tH0[i][j]);
            }
            printf("\n");
        }
    }
#endif

    if (tid == 0)
    {
        x[0] = tH0[0][0];
        x[1] = tH0[0][1];
        x[2] = tH0[0][2];
        x[3] = tH0[1][0];
        x[4] = tH0[1][1];
        x[5] = tH0[1][2];
        x[6] = tH0[2][0];
        x[7] = tH0[2][1];
    }
    __syncwarp();
    __syncthreads();
    return 0;
}

template<class SrcDstWrapper, class ModelWrapper>
__global__ void computeModel(SrcDstWrapper src, SrcDstWrapper dst, float2 *srcMean, float2 *dstMean,
                             float2 *srcShiftSum, float2 *dstShiftSum, float *V_batch, float *W_batch, float *r_batch,
                             float *J_batch, float *calc_buffer_batch, ModelWrapper model, int maxNumPoints,
                             int batchSize)
{
    int tid   = threadIdx.x + blockIdx.x * blockDim.x;
    int batch = blockIdx.y;

    if (batch < batchSize)
    {
        int     numPoints   = getNumPoints(src, maxNumPoints, batch);
        float2 *srcPtr      = src.ptr(batch);
        float2 *dstPtr      = dst.ptr(batch);
        float2  cM          = srcMean[batch];
        float2  sM          = srcShiftSum[batch];
        float2  cm          = dstMean[batch];
        float2  sm          = dstShiftSum[batch];
        float  *W           = W_batch + 9 * batch;
        float  *V           = V_batch + 81 * batch;
        float  *r           = r_batch + 2 * numPoints * batch;
        float  *J           = J_batch + 2 * numPoints * 8 * batch;
        float  *calc_buffer = calc_buffer_batch + numPoints * batch;
        float  *modelPtr    = model.ptr(batch);
        bool    status      = true;

        __shared__ matrix8x32 shared_mem;
        __shared__ vector8    v;
        __shared__ vector8    d;
        __shared__ vector8    D;
        __shared__ vector8    xd;
        __shared__ vector8    x;
        __shared__ vector8    temp_d;
        __shared__ matrix8x8  A;
        __shared__ matrix8x8  Ap;
        __shared__ matrix8x8  App;
        __shared__ matrix8x8  Q;

        int ret = compute_model_estimate(cM, cm, sM, sm, W, V, x, model, batch, numPoints);
        if (!(ret || numPoints == 4))
        {
#ifdef DEBUG_MODEL_KERNEL
            if (tid == 0 && blockIdx.y == 0)
            {
                printf("Model estimated Matrix\n");
                printVectorDevice(x, blockIdx.y);
                printf("\n");
            }
#endif

            // Begin refinement
            calculate_residual_and_jacobian_device(srcPtr, dstPtr, x, numPoints, J, r);

            calculate_residual_norm(r, calc_buffer, shared_mem[0], numPoints * 2);
            float S = calc_buffer[0];

#ifdef DEBUG_MODEL_KERNEL
            if (tid == 0)
            {
                printf("\n\n============Residual================\n");
                printVectorDeviceRaw(r, 2 * numPoints, blockIdx.y);
                printf("\n\n============Jacobian================\n");
                printMatrixDeviceRaw(J, 8, 2 * numPoints, blockIdx.y);
                printf("\n\n============Residual L2 norm==================\n");
                printf("S = %f\n", S);
            }
#endif

            int nfJ = 2;

            if (tid < 8)
                calc_buffer[tid] = 0;
            calculate_JtJ(J, A, shared_mem, calc_buffer, numPoints * 2);

#ifdef DEBUG_MODEL_KERNEL
            if (tid == 0)
            {
                printf("\n================ J^T * J = A ================\n");
                printMatrixDevice(A);
                printf("\n\n");
            }
#endif

            if (tid < 8)
                calc_buffer[tid] = 0;
            calculate_Jtr(J, r, v, shared_mem, calc_buffer, numPoints * 2);

            // only blockIdx.x == 0 needs to do this right now.
            fetch_diagonal(A, D, tid);

#ifdef DEBUG_MODEL_KERNEL
            if (tid == 0)
            {
                printf("\n=============== J^T * r = v ===================\n");
                printVectorDevice(v, blockIdx.y);
                printf("\n================ D ========================\n");
                printVectorDevice(D, blockIdx.y);
                printf("\n");
            }
#endif

            const float Rlo = 0.25, Rhi = 0.75;
            float       lambda = 1, lc = 0.75;
            int         iter = 0, maxIters = 10;
            float       epsx = 1.19209290e-7f;
            float       epsf = 1.19209290e-7f;
            bool        status;

            while (true)
            {
#ifdef DEBUG_MODEL_KERNEL
                if (tid == 0)
                {
                    printf("\n========================================\n");
                    printf("================== ITER = %d =============\n", iter);
                    printf("==========================================\n");
                    printf("\n=============== A before copying ===================\n");
                    printMatrixDevice(A);
                }

#endif
                copy_A_to_Ap_App(A, Ap, App);

#ifdef DEBUG_MODEL_KERNEL
                if (tid == 0)
                {
                    printf("\n=============== Ap before scaling of diagonal ===================\n");
                    printMatrixDevice(Ap);
                }
#endif
                // blockIdx.x == 0
                scale_diagonal8(D, Ap, lambda);

#ifdef DEBUG_MODEL_KERNEL
                if (tid == 0)
                {
                    printf("\n================ D ========================\n");
                    printVectorDevice(D, blockIdx.y);
                    printf("\n=============== Ap after scaling of diagonal ===================\n");
                    printMatrixDevice(Ap);
                }
#endif

                compute_qr8x8(Ap, Q);
                status = solve8x8(Ap, Q, v, d, tid);
                if (!status)
                    break;

                subtract8(x, d, xd, tid);

#ifdef DEBUG_MODEL_KERNEL
                if (tid == 0)
                {
                    printf("\n=============== d ====================\n");
                    printVectorDevice(d, blockIdx.y);
                    printf("\n=============== xd ===================\n");
                    printVectorDevice(xd, blockIdx.y);
                }
#endif

                // calculate residual but not Jacobian
                __syncthreads();
                calculate_residual_and_jacobian_device(srcPtr, dstPtr, xd, numPoints, nullptr, r);

                nfJ++;

                float Sd;
                if (tid < 8)
                    calc_buffer[tid] = 0;
                calculate_residual_norm(r, calc_buffer, shared_mem[0], numPoints * 2);
                Sd = calc_buffer[0];

                calculate_temp_d(A, d, v, temp_d, -1.0f, 2.0f, tid);

                float dS;
                __syncthreads();
                dot8(d, temp_d, &dS);

                float R = (S - Sd) / (fabsf(dS) > FLT_EPSILON ? dS : 1);

#ifdef DEBUG_MODEL_KERNEL
                if (tid == 0)
                {
                    printf("\n=============== r ====================\n");
                    printVectorDeviceRaw(r, 2 * numPoints, blockIdx.y);
                    printf("\n============== || r || ==================\n");
                    printf("||r||^2 = %f\n", Sd);
                    printf("\ndS = %f\n", dS);
                    printf("\nR = %f\n", R);
                }
#endif

                if (R > Rhi)
                {
                    lambda *= 0.5;
                    if (lambda < lc)
                        lambda = 0;
                }
                else if (R < Rlo)
                {
                    float t;
                    dot8(d, v, &t);

                    float nu = (Sd - S) / (fabsf(t) > FLT_EPSILON ? t : 1.0f) + 2.0f;
                    nu       = fminf(fmaxf(nu, 2.0f), 10.0f);

                    if (lambda == 0)
                    {
                        compute_qr8x8(App, Q);
                        status = invert8x8(App, Q, Ap, tid);
                        if (!status)
                            break;

                        float maxval;
                        max_diag_val8(Ap, &maxval);

                        lambda = lc = 1. / maxval;
                        nu *= 0.5;
                    }
                    lambda *= nu;
                }

#ifdef DEBUG_MODEL_KERNEL
                if (tid == 0)
                {
                    printf("\nlambda = %f\n", lambda);
                }
#endif

                if (Sd < S)
                {
                    nfJ++;
                    S = Sd;

#ifdef DEBUG_MODEL_KERNEL
                    if (tid == 0)
                    {
                        printf("\n================== Before swapping =======================\n");
                        printf("\n =================== X =======================\n");
                        printVectorDevice(x, blockIdx.y);
                        printf("\n =================== Xd =======================\n");
                        printVectorDevice(xd, blockIdx.y);
                    }
#endif

                    if (tid < 8)
                        cuda::math::detail::swap(x[tid], xd[tid]);
                    __syncwarp();
                    __syncthreads();

#ifdef DEBUG_MODEL_KERNEL
                    if (tid == 0)
                    {
                        printf("\n================== After swapping =======================\n");
                        printf("\n =================== X =======================\n");
                        printVectorDevice(x, blockIdx.y);
                        printf("\n =================== Xd =======================\n");
                        printVectorDevice(xd, blockIdx.y);
                    }
#endif

                    calculate_residual_and_jacobian_device(srcPtr, dstPtr, x, numPoints, J, r);
                    calculate_JtJ(J, A, shared_mem, calc_buffer, numPoints * 2);
                    calculate_Jtr(J, r, v, shared_mem, calc_buffer, numPoints * 2);

#ifdef DEBUG_MODEL_KERNEL
                    if (tid == 0)
                    {
                        printf("\n =================== J =======================\n");
                        printMatrixDeviceRaw(J, 8, 2 * numPoints, blockIdx.y);
                        printf("\n\n =================== r =======================\n");
                        printVectorDeviceRaw(r, 2 * numPoints, blockIdx.y);
                        printf("\n\n==================== A ========================\n");
                        printMatrixDevice(A);
                        printf("\n\n===================== v ========================\n");
                        printVectorDevice(v, blockIdx.y);
                        printf("\n");
                    }
#endif
                }

                iter++;

                if (tid == 0)
                    calc_buffer[tid] = 0;
                max<myfabs>(r, shared_mem[0], calc_buffer, numPoints * 2);
                __syncthreads();
                float maxResidualValue = calc_buffer[0];
                float maxDvecValue;
                max8(d, &maxDvecValue);

                bool proceed = maxDvecValue >= epsx && maxResidualValue >= epsf && iter < maxIters;
                if (!proceed)
                    break;
            }
        }

        // Copy back the estimate to output buffer
        if (tid == 0)
        {
            if (status)
            {
                *(model.ptr(batch, 0, 0)) = x[0];
                *(model.ptr(batch, 0, 1)) = x[1];
                *(model.ptr(batch, 0, 2)) = x[2];
                *(model.ptr(batch, 1, 0)) = x[3];
                *(model.ptr(batch, 1, 1)) = x[4];
                *(model.ptr(batch, 1, 2)) = x[5];
                *(model.ptr(batch, 2, 0)) = x[6];
                *(model.ptr(batch, 2, 1)) = x[7];
                *(model.ptr(batch, 2, 2)) = 1;
            }
            else
            {
                *(model.ptr(batch, 0, 0)) = 0;
                *(model.ptr(batch, 0, 1)) = 0;
                *(model.ptr(batch, 0, 2)) = 0;
                *(model.ptr(batch, 1, 0)) = 0;
                *(model.ptr(batch, 1, 1)) = 0;
                *(model.ptr(batch, 1, 2)) = 0;
                *(model.ptr(batch, 2, 0)) = 0;
                *(model.ptr(batch, 2, 1)) = 0;
                *(model.ptr(batch, 2, 2)) = 0;
            }
        }
    }
}

template<class SrcDstWrapper, class Func>
__global__ void compute_src_dst_mean(SrcDstWrapper src, SrcDstWrapper dst, float2 *srcMean, float2 *dstMean,
                                     Func src_op, Func dst_op, int maxNumPoints, int batchSize)
{
    int        batch = blockIdx.y;
    __shared__ cuda::math::Vector<float2, 32> warpSums;
    if (batch < batchSize)
    {
        int     numPoints    = getNumPoints(src, maxNumPoints, batch);
        float2 *srcMeanBatch = srcMean + batch;
        float2 *dstMeanBatch = dstMean + batch;
        float2 *srcPtr       = src.ptr(batch);
        float2 *dstPtr       = dst.ptr(batch);
        reducef2<Func>(srcPtr, warpSums, srcMeanBatch, src_op, numPoints, batch);
        __syncthreads();
        reducef2<Func>(dstPtr, warpSums, dstMeanBatch, dst_op, numPoints, batch);
    }
}

template<class SrcDstWrapper, class Func>
__global__ void compute_LtL(SrcDstWrapper src, SrcDstWrapper dst, float *LtL, Func ltl_op, int maxNumPoints,
                            int batchSize)
{
    int        batch = blockIdx.z;
    int        j     = blockIdx.y / 9; // LtL row index
    int        k     = blockIdx.y % 9; // LtL col index
    __shared__ cuda::math::Vector<float, 32> warpSums;
    if (batch < batchSize)
    {
        int     numPoints = getNumPoints(src, maxNumPoints, batch);
        float  *LtLBatch  = LtL + 81 * batch;
        float2 *srcPtr    = src.ptr(batch);
        float2 *dstPtr    = dst.ptr(batch);
        reduceLtL<Func>(srcPtr, dstPtr, warpSums, &LtLBatch[j * 9 + k], ltl_op, numPoints, batch, j, k);
    }
}

/* numPoints should be maxNumPoints in the case of varshape. */
template<typename SrcDstWrapper, class ModelType>
void FindHomographyWrapper(SrcDstWrapper srcWrap, SrcDstWrapper dstWrap, ModelType &models,
                           const BufferOffsets *bufferOffset, const cuSolver *cusolverData, int numPoints,
                           cudaStream_t stream)
{
    dim3                      block(256, 1, 1);
    cuda::Tensor3DWrap<float> modelWrap = cuda::CreateTensorWrapNHW<float>(models);
    int                       batchSize = models.shape(0);

    float2            *srcMean        = bufferOffset->srcMean;
    float2            *dstMean        = bufferOffset->dstMean;
    float2            *srcShiftSum    = bufferOffset->srcShiftSum;
    float2            *dstShiftSum    = bufferOffset->dstShiftSum;
    float             *J              = bufferOffset->J;
    float             *r              = bufferOffset->r;
    float             *LtL            = bufferOffset->LtL;
    float             *W              = bufferOffset->W;
    float             *calc_buffer    = bufferOffset->calc_buffer;
    float             *cusolverBuffer = cusolverData->cusolverBuffer;
    int               *cusolverInfo   = cusolverData->cusolverInfo;
    int                lwork          = cusolverData->lwork;
    cusolverDnHandle_t cusolverH      = cusolverData->cusolverH;
    syevjInfo_t        syevj_params   = cusolverData->syevj_params;

    cudaMemsetAsync(reinterpret_cast<void *>(srcMean), 0, batchSize * sizeof(float2), stream);
    cudaMemsetAsync(reinterpret_cast<void *>(dstMean), 0, batchSize * sizeof(float2), stream);
    cudaMemsetAsync(reinterpret_cast<void *>(srcShiftSum), 0, batchSize * sizeof(float2), stream);
    cudaMemsetAsync(reinterpret_cast<void *>(dstShiftSum), 0, batchSize * sizeof(float2), stream);
    cudaMemsetAsync(reinterpret_cast<void *>(J), 0, 2 * numPoints * 8 * batchSize * sizeof(float), stream);
    cudaMemsetAsync(reinterpret_cast<void *>(r), 0, 2 * numPoints * batchSize * sizeof(float), stream);
    cudaMemsetAsync(reinterpret_cast<void *>(LtL), 0, 81 * batchSize * sizeof(float), stream);
    cudaMemsetAsync(reinterpret_cast<void *>(W), 0, 9 * batchSize * sizeof(float), stream);
    cudaMemsetAsync(reinterpret_cast<void *>(calc_buffer), 0, numPoints * batchSize * sizeof(float), stream);
    cudaMemsetAsync(reinterpret_cast<void *>(cusolverBuffer), 0, lwork * sizeof(float), stream);
    cudaMemsetAsync(reinterpret_cast<void *>(cusolverInfo), 0, batchSize * sizeof(int), stream);

    dim3 grid((numPoints + block.x - 1) / block.x, batchSize, 1);

    MeanOp meanop;
    compute_src_dst_mean<<<grid, block, 0, stream>>>(srcWrap, dstWrap, srcMean, dstMean, meanop, meanop, numPoints,
                                                     batchSize);
#ifdef DEBUG
    int check_batch = 0;
    printKernelfloat2<<<1, 1, 0, stream>>>(srcMean + check_batch, 1, 0);
    printKernelfloat2<<<1, 1, 0, stream>>>(dstMean + check_batch, 1, 0);
#endif

    AbsShiftOp src_abs_shift_op(srcMean);
    AbsShiftOp dst_abs_shift_op(dstMean);
    compute_src_dst_mean<<<grid, block, 0, stream>>>(srcWrap, dstWrap, srcShiftSum, dstShiftSum, src_abs_shift_op,
                                                     dst_abs_shift_op, numPoints, batchSize);
#ifdef DEBUG
    printKernelfloat2<<<1, 1, 0, stream>>>(srcShiftSum + check_batch, 1, 0);
    printKernelfloat2<<<1, 1, 0, stream>>>(dstShiftSum + check_batch, 1, 0);
#endif

    grid.y = 81;
    grid.z = batchSize;
    LtLOp ltl_op(srcMean, dstMean, srcShiftSum, dstShiftSum);
    compute_LtL<<<grid, block, 0, stream>>>(srcWrap, dstWrap, LtL, ltl_op, numPoints, batchSize);
#ifdef DEBUG
    for (int b = 0; b < batchSize; b++)
    {
        std::cout << "==================== Batch " << b << " =======================" << std::endl;
        printMatrix<<<1, 1, 0, stream>>>(LtL + 81 * b, 9, 9);
        cudaStreamSynchronize(stream);
    }
#endif

    // compute Eigen values
    CUSOLVER_CHECK_ERROR(cusolverDnSetStream(cusolverH, stream), "Failed to set cuda stream in cusolver");
    CUSOLVER_CHECK_ERROR(cusolverDnSsyevjBatched(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 9, LtL, 9,
                                                 W, cusolverBuffer, lwork, cusolverInfo, syevj_params, batchSize),
                         "Failed to calculate eigen values using syevj");
#ifdef DEBUG
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream), "synchronization failed after eigen value solver");
    std::vector<int> info(batchSize);
    cudaMemcpyAsync((void *)info.data(), (void *)cusolverInfo, batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream), "synchronization failed after copying back cusolverInfo");

    for (int b = 0; b < batchSize; b++)
    {
        if (info[b] == 0)
        {
            std::cout << "cusolver converged for matrix " << b << std::endl;
            printKernel<<<1, 9, 0, stream>>>(W + 9 * b, 9, 0);
            printf("\n");
        }
        else if (info[b] < 0)
        {
            std::cout << info[b] << "th parameter is wrong for image " << b << std::endl;
        }
        else
        {
            std::cout << "cusolver did not converge for image " << b << std::endl;
        }
        CUDA_CHECK_ERROR(cudaStreamSynchronize(stream), "failed to synchronize");
    }
#endif

    block.x = 256;
    grid.x  = 1;
    grid.y  = batchSize;
    grid.z  = 1;
    computeModel<<<grid, block, 0, stream>>>(srcWrap, dstWrap, srcMean, dstMean, srcShiftSum, dstShiftSum, LtL, W, r, J,
                                             calc_buffer, modelWrap, numPoints, batchSize);
}

inline void RunFindHomography(const nvcv::TensorDataStridedCuda &src, const nvcv::TensorDataStridedCuda &dst,
                              const nvcv::TensorDataStridedCuda &models, const BufferOffsets *bufferOffset,
                              const cuSolver *cusolverData, cudaStream_t stream)
{
    // validation of input data
    if ((src.rank() != 2 && src.rank() != 3) || (dst.rank() != 2 && dst.rank() != 3))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "source and destination points must have rank 2 or 3");
    }

    if (!(src.shape(0) == dst.shape(0) && src.shape(0) == models.shape(0)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "source, destination and model must have same batch size");
    }

    if (src.shape(1) != dst.shape(1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "source and destination array length must be same length to return a valid model");
    }

    if (src.shape(1) < 4 || dst.shape(1) < 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "source and destination array length must be >=4 to return a valid model");
    }

    if (!(models.rank() == 3 && models.shape(1) == 3 && models.shape(2) == 3 && models.dtype() == nvcv::TYPE_F32))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "model tensor must be 2D with shape 3x3 and data type F32");
    }

    if (!((src.rank() == 2 && src.dtype() == nvcv::TYPE_2F32)
          || (src.rank() == 3 && src.dtype() == nvcv::TYPE_F32 && src.shape(2) == 2)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "source tensor must have data type 2F32 or F32 with last shape 2");
    }
    if (!((dst.rank() == 2 && dst.dtype() == nvcv::TYPE_2F32)
          || (dst.rank() == 3 && dst.dtype() == nvcv::TYPE_F32 && dst.shape(2) == 2)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "destination tensor must have data type 2F32 or F32 with last shape 2");
    }
    if (!(src.stride(1) == sizeof(float2) && dst.stride(1) == sizeof(float2)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "source and destination tensors must have last dimensions packed");
    }

    using SrcDstWrapper = cuda::Tensor2DWrap<float2>;
    SrcDstWrapper srcWrap(src);
    SrcDstWrapper dstWrap(dst);
    int           numPoints = src.shape(1);
    FindHomographyWrapper(srcWrap, dstWrap, models, bufferOffset, cusolverData, numPoints, stream);
}

} // namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

FindHomography::FindHomography(int batchSize, int maxNumPoints)
{
    cudaMalloc(reinterpret_cast<void **>(&(bufferOffset.srcMean)), sizeof(float2) * batchSize);
    cudaMalloc(reinterpret_cast<void **>(&(bufferOffset.dstMean)), sizeof(float2) * batchSize);
    cudaMalloc(reinterpret_cast<void **>(&(bufferOffset.srcShiftSum)), sizeof(float2) * batchSize);
    cudaMalloc(reinterpret_cast<void **>(&(bufferOffset.dstShiftSum)), sizeof(float2) * batchSize);
    cudaMalloc(reinterpret_cast<void **>(&(bufferOffset.LtL)), 81 * sizeof(float) * batchSize);
    cudaMalloc(reinterpret_cast<void **>(&(bufferOffset.W)), 9 * sizeof(float) * batchSize);
    cudaMalloc(reinterpret_cast<void **>(&(bufferOffset.r)), 2 * maxNumPoints * sizeof(float) * batchSize);
    cudaMalloc(reinterpret_cast<void **>(&(bufferOffset.J)), 2 * maxNumPoints * 8 * sizeof(float) * batchSize);
    cudaMalloc(reinterpret_cast<void **>(&(bufferOffset.calc_buffer)), maxNumPoints * sizeof(float) * batchSize);
    CUSOLVER_CHECK_ERROR(cusolverDnCreate(&(cusolverData.cusolverH)), "Failed to create cusolver handle");
    CUSOLVER_CHECK_ERROR(cusolverDnCreateSyevjInfo(&(cusolverData.syevj_params)), "Failed to create syevj params");
    CUSOLVER_CHECK_ERROR(cusolverDnXsyevjSetTolerance(cusolverData.syevj_params, 1e-7),
                         "Failed to set tolerance for syevj");
    CUSOLVER_CHECK_ERROR(cusolverDnXsyevjSetMaxSweeps(cusolverData.syevj_params, 15),
                         "Failed to set max sweeps for syevj");
    CUSOLVER_CHECK_ERROR(cusolverDnXsyevjSetSortEig(cusolverData.syevj_params, 1),
                         "Failed to set sorting of eigen values in syevj");
    CUSOLVER_CHECK_ERROR(
        cusolverDnSsyevjBatched_bufferSize(cusolverData.cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 9,
                                           NULL, 9, NULL, &(cusolverData.lwork), cusolverData.syevj_params, batchSize),
        "Failed to calculate buffer size for syevj");
    cudaMalloc(reinterpret_cast<void **>(&(cusolverData.cusolverBuffer)), cusolverData.lwork * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&(cusolverData.cusolverInfo)), batchSize * sizeof(int));
}

FindHomography::~FindHomography()
{
    cudaFree(bufferOffset.srcMean);
    cudaFree(bufferOffset.dstMean);
    cudaFree(bufferOffset.srcShiftSum);
    cudaFree(bufferOffset.dstShiftSum);
    cudaFree(bufferOffset.LtL);
    cudaFree(bufferOffset.W);
    cudaFree(bufferOffset.r);
    cudaFree(bufferOffset.J);
    cudaFree(bufferOffset.calc_buffer);
    cusolverDnDestroySyevjInfo(cusolverData.syevj_params);
    cusolverDnDestroy(cusolverData.cusolverH);
    cudaFree(cusolverData.cusolverBuffer);
    cudaFree(cusolverData.cusolverInfo);
}

// Operator --------------------------------------------------------------------

// Tensor input variant
void FindHomography::operator()(cudaStream_t stream, const nvcv::Tensor &srcPoints, const nvcv::Tensor &dstPoints,
                                const nvcv::Tensor &models) const
{
    auto srcData = srcPoints.exportData<nvcv::TensorDataStridedCuda>();
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto dstData = dstPoints.exportData<nvcv::TensorDataStridedCuda>();
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto modelData = models.exportData<nvcv::TensorDataStridedCuda>();
    if (!modelData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    RunFindHomography(*srcData, *dstData, *modelData, &bufferOffset, &cusolverData, stream);
}

void FindHomography::operator()(cudaStream_t stream, const nvcv::TensorBatch &srcPoints,
                                const nvcv::TensorBatch &dstPoints, const nvcv::TensorBatch &models) const
{
    if (!(srcPoints.numTensors() == dstPoints.numTensors() && srcPoints.numTensors() == models.numTensors()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "source, destination and model tensors must have same batch size");
    }

    for (int b = 0; b < srcPoints.numTensors(); b++)
    {
        auto srcData = srcPoints[b].exportData<nvcv::TensorDataStridedCuda>();
        if (!srcData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input src points must be cuda-accessible, pitch-linear tensor");
        }

        auto dstData = dstPoints[b].exportData<nvcv::TensorDataStridedCuda>();
        if (!dstData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input dst points must be cuda-accessible, pitch-linear tensor");
        }

        auto modelData = models[b].exportData<nvcv::TensorDataStridedCuda>();
        if (!modelData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "model must be cuda-accessible, pitch-linear tensor");
        }

        RunFindHomography(*srcData, *dstData, *modelData, &bufferOffset, &cusolverData, stream);
    }
}

} // namespace cvcuda::priv

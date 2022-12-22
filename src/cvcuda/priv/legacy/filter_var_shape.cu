/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

#include "../Assert.h"
#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

template<typename D, typename BrdRd>
__global__ void filter2D(const BrdRd src, Ptr2dVarShapeNHWC<D> dst, Ptr2dVarShapeNHWC<float> kernel,
                         cuda::Tensor1DWrap<int2> kernelAnchor)
{
    using work_type = cuda::ConvertBaseTypeTo<float, D>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.at_cols(batch_idx) || y >= dst.at_rows(batch_idx))
        return;

    int2 anchor = *kernelAnchor.ptr(batch_idx);

    int2 kernelSize{kernel.at_cols(batch_idx), kernel.at_rows(batch_idx)};

    if (anchor.x < 0)
        anchor.x = kernelSize.x / 2;

    if (anchor.y < 0)
        anchor.y = kernelSize.y / 2;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        for (int j = 0; j < kernelSize.x; ++j)
        {
            res = res + (src(batch_idx, y - anchor.y + i, x - anchor.x + j)) * (*kernel.ptr(batch_idx, i, j));
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<cuda::BaseType<D>>(res);
}

template<typename D, template<typename> class Brd>
void Filter2DCaller(const IImageBatchVarShapeDataStridedCuda &inData, const IImageBatchVarShapeDataStridedCuda &outData,
                    const IImageBatchVarShapeDataStridedCuda &kernelData,
                    const ITensorDataStridedCuda &kernelAnchorData, float borderValue, cudaStream_t stream)
{
    Ptr2dVarShapeNHWC<D> src(inData);
    Ptr2dVarShapeNHWC<D> dst(outData);

    Ptr2dVarShapeNHWC<float> kernel(kernelData);
    cuda::Tensor1DWrap<int2> kernelAnchor(kernelAnchorData);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

    Brd<work_type>                                     brd(0, 0, cuda::SetAll<work_type>(borderValue));
    BorderReader<Ptr2dVarShapeNHWC<D>, Brd<work_type>> brdSrc(src, brd);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    filter2D<D, BorderReader<Ptr2dVarShapeNHWC<D>, Brd<work_type>>>
        <<<grid, block, 0, stream>>>(brdSrc, dst, kernel, kernelAnchor);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void Filter2D(const IImageBatchVarShapeDataStridedCuda &inData, const IImageBatchVarShapeDataStridedCuda &outData,
              const IImageBatchVarShapeDataStridedCuda &kernelData, const ITensorDataStridedCuda &kernelAnchorData,
              NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(const IImageBatchVarShapeDataStridedCuda &inData,
                           const IImageBatchVarShapeDataStridedCuda &outData,
                           const IImageBatchVarShapeDataStridedCuda &kernelData,
                           const ITensorDataStridedCuda &kernelAnchorData, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {Filter2DCaller<D, BrdConstant>, Filter2DCaller<D, BrdReplicate>, Filter2DCaller<D, BrdReflect>,
           Filter2DCaller<D, BrdWrap>, Filter2DCaller<D, BrdReflect101>};

    funcs[borderMode](inData, outData, kernelData, kernelAnchorData, borderValue, stream);
}

// Conv2DVarShape --------------------------------------------------------------

ErrorCode Conv2DVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                const IImageBatchVarShapeDataStridedCuda &outData,
                                const IImageBatchVarShapeDataStridedCuda &kernelData,
                                const ITensorDataStridedCuda &kernelAnchorData, NVCVBorderType borderMode,
                                cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    typedef void (*filter2D_t)(
        const IImageBatchVarShapeDataStridedCuda &inData, const IImageBatchVarShapeDataStridedCuda &outData,
        const IImageBatchVarShapeDataStridedCuda &kernelData, const ITensorDataStridedCuda &kernelAnchorData,
        NVCVBorderType borderMode, float borderValue, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { Filter2D<uchar>, 0,  Filter2D<uchar3>,  Filter2D<uchar4>},
        {               0, 0,                 0,                 0},
        {Filter2D<ushort>, 0, Filter2D<ushort3>, Filter2D<ushort4>},
        { Filter2D<short>, 0,  Filter2D<short3>,  Filter2D<short4>},
        {   Filter2D<int>, 0,    Filter2D<int3>,    Filter2D<int4>},
        { Filter2D<float>, 0,  Filter2D<float3>,  Filter2D<float4>},
    };

    const filter2D_t func = funcs[data_type][channels - 1];

    NVCV_ASSERT(func != 0);

    func(inData, outData, kernelData, kernelAnchorData, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

// LaplacianVarShape -----------------------------------------------------------

// @brief Laplacian 3x3 kernels for ksize == 1 and ksize == 3

// clang-format off

constexpr __device__ cuda::math::Vector<float, 9> kLaplacianKernel1{
    {0.0f,  1.0f, 0.0f,
     1.0f, -4.0f, 1.0f,
     0.0f,  1.0f, 0.0f}
};
constexpr __device__ cuda::math::Vector<float, 9> kLaplacianKernel3{
    {2.0f,  0.0f, 2.0f,
     0.0f, -8.0f, 0.0f,
     2.0f,  0.0f, 2.0f}
};

// clang-format on

// Laplacian kernels are either one or the other (above)
template<typename D, typename BrdRd>
__global__ void laplacianFilter2D(const BrdRd src, Ptr2dVarShapeNHWC<D> dst, cuda::Tensor1DWrap<int> ksize,
                                  cuda::Tensor1DWrap<float> scale)
{
    using work_type = cuda::ConvertBaseTypeTo<float, D>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.at_cols(batch_idx) || y >= dst.at_rows(batch_idx))
        return;

    constexpr int2 kernelSize = int2{3, 3};
    constexpr int2 anchor     = int2{1, 1};

    const int ksizeVal = *ksize.ptr(batch_idx);

    NVCV_CUDA_ASSERT(ksizeVal == 1 || ksizeVal == 3, "E Wrong ksize = %d, expected: 1 or 3", ksizeVal);
    cuda::math::Vector<float, 9> kernel = ksizeVal == 1 ? kLaplacianKernel1 : kLaplacianKernel3;

    kernel *= *scale.ptr(batch_idx);

    int kidx = 0;

#pragma unroll
    for (int i = 0; i < kernelSize.y; ++i)
    {
#pragma unroll
        for (int j = 0; j < kernelSize.x; ++j)
        {
            res = res + (src(batch_idx, y - anchor.y + i, x - anchor.x + j)) * kernel[kidx++];
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<cuda::BaseType<D>>(res);
}

template<typename D, template<typename> class Brd>
void LaplacianFilter2DCaller(const IImageBatchVarShapeDataStridedCuda &inData,
                             const IImageBatchVarShapeDataStridedCuda &outData, const ITensorDataStridedCuda &ksize,
                             const ITensorDataStridedCuda &scale, float borderValue, cudaStream_t stream)
{
    Ptr2dVarShapeNHWC<D> src(inData);
    Ptr2dVarShapeNHWC<D> dst(outData);

    cuda::Tensor1DWrap<int>   kernelApertureSize(ksize);
    cuda::Tensor1DWrap<float> kernelScale(scale);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

    Brd<work_type>                                     brd(0, 0, cuda::SetAll<work_type>(borderValue));
    BorderReader<Ptr2dVarShapeNHWC<D>, Brd<work_type>> brdSrc(src, brd);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    laplacianFilter2D<D, BorderReader<Ptr2dVarShapeNHWC<D>, Brd<work_type>>>
        <<<grid, block, 0, stream>>>(brdSrc, dst, kernelApertureSize, kernelScale);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void LaplacianFilter2D(const IImageBatchVarShapeDataStridedCuda &inData,
                       const IImageBatchVarShapeDataStridedCuda &outData, const ITensorDataStridedCuda &ksize,
                       const ITensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
                       cudaStream_t stream)
{
    typedef void (*func_t)(const IImageBatchVarShapeDataStridedCuda &inData,
                           const IImageBatchVarShapeDataStridedCuda &outData, const ITensorDataStridedCuda &ksize,
                           const ITensorDataStridedCuda &scale, float borderValue, cudaStream_t stream);

    static const func_t funcs[] = {LaplacianFilter2DCaller<D, BrdConstant>, LaplacianFilter2DCaller<D, BrdReplicate>,
                                   LaplacianFilter2DCaller<D, BrdReflect>, LaplacianFilter2DCaller<D, BrdWrap>,
                                   LaplacianFilter2DCaller<D, BrdReflect101>};

    funcs[borderMode](inData, outData, ksize, scale, borderValue, stream);
}

ErrorCode LaplacianVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                   const IImageBatchVarShapeDataStridedCuda &outData,
                                   const ITensorDataStridedCuda &ksize, const ITensorDataStridedCuda &scale,
                                   NVCVBorderType borderMode, cudaStream_t stream)
{
    DataFormat input_format  = GetLegacyDataFormat(inData);
    DataFormat output_format = GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    typedef void (*filter2D_t)(const IImageBatchVarShapeDataStridedCuda &inData,
                               const IImageBatchVarShapeDataStridedCuda &outData, const ITensorDataStridedCuda &ksize,
                               const ITensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
                               cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { LaplacianFilter2D<uchar>, 0,  LaplacianFilter2D<uchar3>,  LaplacianFilter2D<uchar4>},
        {                        0, 0,                          0,                          0},
        {LaplacianFilter2D<ushort>, 0, LaplacianFilter2D<ushort3>, LaplacianFilter2D<ushort4>},
        {                        0, 0,                          0,                          0},
        {                        0, 0,                          0,                          0},
        { LaplacianFilter2D<float>, 0,  LaplacianFilter2D<float3>,  LaplacianFilter2D<float4>},
    };

    const filter2D_t func = funcs[data_type][channels - 1];

    NVCV_ASSERT(func != 0);

    func(inData, outData, ksize, scale, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

// GaussianVarShape ------------------------------------------------------------

__global__ void CalculateGaussianKernel(cuda::Tensor3DWrap<float> kernel, int dataKernelSize, Size2D maxKernelSize,
                                        cuda::Tensor1DWrap<int2> kernelSizeArr, cuda::Tensor1DWrap<double2> sigmaArr)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    int2 kernelSize = *kernelSizeArr.ptr(coord.z);

    if (coord.x >= kernelSize.x || coord.y >= kernelSize.y)
    {
        return;
    }

    double2 sigma = *sigmaArr.ptr(coord.z);

    if (sigma.y <= 0)
        sigma.y = sigma.x;

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    int2 half{kernelSize.x / 2, kernelSize.y / 2};

    sigma.x = cuda::max(sigma.x, 0.0);
    sigma.y = cuda::max(sigma.y, 0.0);

    float sx = 2.f * sigma.x * sigma.x;
    float sy = 2.f * sigma.y * sigma.y;
    float s  = 2.f * sigma.x * sigma.y * M_PI;

    float sum = 0.f;

    for (int y = -half.y; y <= half.y; ++y)
    {
        for (int x = -half.x; x <= half.x; ++x)
        {
            sum += cuda::exp(-((x * x) / sx + (y * y) / sy)) / s;
        }
    }

    int x = coord.x - half.x;
    int y = coord.y - half.y;

    kernel[coord] = cuda::exp(-((x * x) / sx + (y * y) / sy)) / (s * sum);
}

template<typename D, typename BrdRd>
__global__ void gaussianFilter2D(const BrdRd src, Ptr2dVarShapeNHWC<D> dst, cuda::Tensor3DWrap<float> kernel,
                                 cuda::Tensor1DWrap<int2> kernelSizeArr)
{
    using work_type = cuda::ConvertBaseTypeTo<float, D>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.at_cols(batch_idx) || y >= dst.at_rows(batch_idx))
        return;

    int2 kernelSize = *kernelSizeArr.ptr(batch_idx);

    int2 anchor{kernelSize.x / 2, kernelSize.y / 2};

    for (int i = 0; i < kernelSize.y; ++i)
    {
        for (int j = 0; j < kernelSize.x; ++j)
        {
            res = res + (src(batch_idx, y - anchor.y + i, x - anchor.x + j)) * (*kernel.ptr(batch_idx, i, j));
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<cuda::BaseType<D>>(res);
}

template<typename D, template<typename> class Brd>
void GaussianFilter2DCaller(const IImageBatchVarShapeDataStridedCuda &inData,
                            const IImageBatchVarShapeDataStridedCuda &outData,
                            const cuda::Tensor3DWrap<float>          &kernelTensor,
                            const cuda::Tensor1DWrap<int2> &kernelSizeTensor, float borderValue, cudaStream_t stream)
{
    Ptr2dVarShapeNHWC<D> src(inData);
    Ptr2dVarShapeNHWC<D> dst(outData);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

    Brd<work_type>                                     brd(0, 0, cuda::SetAll<work_type>(borderValue));
    BorderReader<Ptr2dVarShapeNHWC<D>, Brd<work_type>> brdSrc(src, brd);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    gaussianFilter2D<D, BorderReader<Ptr2dVarShapeNHWC<D>, Brd<work_type>>>
        <<<grid, block, 0, stream>>>(brdSrc, dst, kernelTensor, kernelSizeTensor);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void GaussianFilter2D(const IImageBatchVarShapeDataStridedCuda &inData,
                      const IImageBatchVarShapeDataStridedCuda &outData, const cuda::Tensor3DWrap<float> &kernelTensor,
                      const cuda::Tensor1DWrap<int2> &kernelSizeTensor, NVCVBorderType borderMode, float borderValue,
                      cudaStream_t stream)
{
    typedef void (*func_t)(const IImageBatchVarShapeDataStridedCuda &inData,
                           const IImageBatchVarShapeDataStridedCuda &outData,
                           const cuda::Tensor3DWrap<float>          &kernelTensor,
                           const cuda::Tensor1DWrap<int2> &kernelSizeTensor, float borderValue, cudaStream_t stream);

    static const func_t funcs[] = {GaussianFilter2DCaller<D, BrdConstant>, GaussianFilter2DCaller<D, BrdReplicate>,
                                   GaussianFilter2DCaller<D, BrdReflect>, GaussianFilter2DCaller<D, BrdWrap>,
                                   GaussianFilter2DCaller<D, BrdReflect101>};

    funcs[borderMode](inData, outData, kernelTensor, kernelSizeTensor, borderValue, stream);
}

GaussianVarShape::GaussianVarShape(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize,
                                   int maxBatchSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_maxKernelSize(maxKernelSize)
    , m_maxBatchSize(maxBatchSize)
{
    if (maxBatchSize > 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(&m_kernel, maxKernelSize.w * maxKernelSize.h * maxBatchSize * sizeof(float)));
    }
}

GaussianVarShape::~GaussianVarShape()
{
    NVCV_CHECK_LOG(cudaFree(m_kernel));
}

size_t GaussianVarShape::calBufferSize(Size2D maxKernelSize, int maxBatchSize)
{
    return maxKernelSize.w * maxKernelSize.h * maxBatchSize * sizeof(float);
}

ErrorCode GaussianVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                  const IImageBatchVarShapeDataStridedCuda &outData,
                                  const ITensorDataStridedCuda &kernelSize, const ITensorDataStridedCuda &sigma,
                                  NVCVBorderType borderMode, cudaStream_t stream)
{
    if (m_maxBatchSize <= 0 || inData.numImages() > m_maxBatchSize)
    {
        LOG_ERROR("Invalid maximum batch size");
        return ErrorCode::INVALID_PARAMETER;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    int dataKernelSize = (data_type == kCV_8U ? 3 : 4);

    dim3 block(32, 4);
    dim3 grid(divUp(m_maxKernelSize.w, block.x), divUp(m_maxKernelSize.h, block.y), outData.numImages());

    cuda::Tensor1DWrap<int2>    kernelSizeTensor(kernelSize);
    cuda::Tensor1DWrap<double2> sigmaTensor(sigma);

    int kernelPitch2 = static_cast<int>(m_maxKernelSize.w * sizeof(float));
    int kernelPitch1 = m_maxKernelSize.h * kernelPitch2;

    cuda::Tensor3DWrap<float> kernelTensor(m_kernel, kernelPitch1, kernelPitch2);

    CalculateGaussianKernel<<<grid, block, 0, stream>>>(kernelTensor, dataKernelSize, m_maxKernelSize, kernelSizeTensor,
                                                        sigmaTensor);

    checkKernelErrors();

    typedef void (*filter2D_t)(
        const IImageBatchVarShapeDataStridedCuda &inData, const IImageBatchVarShapeDataStridedCuda &outData,
        const cuda::Tensor3DWrap<float> &kernelTensor, const cuda::Tensor1DWrap<int2> &kernelSizeTensor,
        NVCVBorderType borderMode, float borderValue, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { GaussianFilter2D<uchar>, 0,  GaussianFilter2D<uchar3>,  GaussianFilter2D<uchar4>},
        {                       0, 0,                         0,                         0},
        {GaussianFilter2D<ushort>, 0, GaussianFilter2D<ushort3>, GaussianFilter2D<ushort4>},
        { GaussianFilter2D<short>, 0,  GaussianFilter2D<short3>,  GaussianFilter2D<short4>},
        {   GaussianFilter2D<int>, 0,    GaussianFilter2D<int3>,    GaussianFilter2D<int4>},
        { GaussianFilter2D<float>, 0,  GaussianFilter2D<float3>,  GaussianFilter2D<float4>},
    };

    const filter2D_t func = funcs[data_type][channels - 1];

    NVCV_ASSERT(func != 0);

    func(inData, outData, kernelTensor, kernelSizeTensor, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

// AverageBlurVarShape ---------------------------------------------------------

__global__ void compute_average_blur_kernel(cuda::Tensor3DWrap<float> kernel, cuda::Tensor1DWrap<int2> kernelSizeArr,
                                            cuda::Tensor1DWrap<int2> kernelAnchorArr)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    int2 kernelSize = *kernelSizeArr.ptr(coord.z);

    if (coord.x >= kernelSize.x || coord.y >= kernelSize.y)
    {
        return;
    }

    bool kernelAnchorUpdated = false;
    int2 kernelAnchor        = *kernelAnchorArr.ptr(coord.z);

    if (kernelAnchor.x < 0)
    {
        kernelAnchor.x      = kernelSize.x / 2;
        kernelAnchorUpdated = true;
    }

    if (kernelAnchor.y < 0)
    {
        kernelAnchor.y      = kernelSize.y / 2;
        kernelAnchorUpdated = true;
    }

    if (kernelAnchorUpdated)
    {
        *kernelAnchorArr.ptr(coord.z) = kernelAnchor;
    }

    kernel[coord] = 1.f / (kernelSize.x * kernelSize.y);
}

template<typename D, typename BrdRd>
__global__ void avgBlurFilter2D(const BrdRd src, Ptr2dVarShapeNHWC<D> dst, cuda::Tensor3DWrap<float> kernel,
                                cuda::Tensor1DWrap<int2> kernelSizeArr, cuda::Tensor1DWrap<int2> kernelAnchorArr)
{
    using work_type = cuda::ConvertBaseTypeTo<float, D>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.at_cols(batch_idx) || y >= dst.at_rows(batch_idx))
        return;

    int2 kernelSize = *kernelSizeArr.ptr(batch_idx);

    int2 anchor = *kernelAnchorArr.ptr(batch_idx);

    for (int i = 0; i < kernelSize.y; ++i)
    {
        for (int j = 0; j < kernelSize.x; ++j)
        {
            res = res + (src(batch_idx, y - anchor.y + i, x - anchor.x + j)) * (*kernel.ptr(batch_idx, i, j));
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<cuda::BaseType<D>>(res);
}

template<typename D, template<typename> class Brd>
void AverageBlurFilter2DCaller(const IImageBatchVarShapeDataStridedCuda &inData,
                               const IImageBatchVarShapeDataStridedCuda &outData,
                               const cuda::Tensor3DWrap<float>          &kernelTensor,
                               const cuda::Tensor1DWrap<int2>           &kernelSizeTensor,
                               const cuda::Tensor1DWrap<int2> &kernelAnchorTensor, float borderValue,
                               cudaStream_t stream)
{
    Ptr2dVarShapeNHWC<D> src(inData);
    Ptr2dVarShapeNHWC<D> dst(outData);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

    Brd<work_type>                                     brd(0, 0, cuda::SetAll<work_type>(borderValue));
    BorderReader<Ptr2dVarShapeNHWC<D>, Brd<work_type>> brdSrc(src, brd);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    avgBlurFilter2D<D, BorderReader<Ptr2dVarShapeNHWC<D>, Brd<work_type>>>
        <<<grid, block, 0, stream>>>(brdSrc, dst, kernelTensor, kernelSizeTensor, kernelAnchorTensor);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void AverageBlurFilter2D(const IImageBatchVarShapeDataStridedCuda &inData,
                         const IImageBatchVarShapeDataStridedCuda &outData,
                         const cuda::Tensor3DWrap<float>          &kernelTensor,
                         const cuda::Tensor1DWrap<int2>           &kernelSizeTensor,
                         const cuda::Tensor1DWrap<int2> &kernelAnchorTensor, NVCVBorderType borderMode,
                         float borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(
        const IImageBatchVarShapeDataStridedCuda &inData, const IImageBatchVarShapeDataStridedCuda &outData,
        const cuda::Tensor3DWrap<float> &kernelTensor, const cuda::Tensor1DWrap<int2> &kernelSizeTensor,
        const cuda::Tensor1DWrap<int2> &kernelAnchorTensor, float borderValue, cudaStream_t stream);

    static const func_t funcs[] = {AverageBlurFilter2DCaller<D, BrdConstant>,
                                   AverageBlurFilter2DCaller<D, BrdReplicate>, AverageBlurFilter2DCaller<D, BrdReflect>,
                                   AverageBlurFilter2DCaller<D, BrdWrap>, AverageBlurFilter2DCaller<D, BrdReflect101>};

    funcs[borderMode](inData, outData, kernelTensor, kernelSizeTensor, kernelAnchorTensor, borderValue, stream);
}

AverageBlurVarShape::AverageBlurVarShape(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize,
                                         int maxBatchSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_maxKernelSize(maxKernelSize)
    , m_maxBatchSize(maxBatchSize)
{
    if (maxBatchSize > 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(&m_kernel, maxKernelSize.w * maxKernelSize.h * maxBatchSize * sizeof(float)));
    }
}

AverageBlurVarShape::~AverageBlurVarShape()
{
    NVCV_CHECK_LOG(cudaFree(m_kernel));
}

size_t AverageBlurVarShape::calBufferSize(Size2D maxKernelSize, int maxBatchSize)
{
    return maxKernelSize.w * maxKernelSize.h * maxBatchSize * sizeof(float);
}

ErrorCode AverageBlurVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                     const IImageBatchVarShapeDataStridedCuda &outData,
                                     const ITensorDataStridedCuda             &kernelSize,
                                     const ITensorDataStridedCuda &kernelAnchor, NVCVBorderType borderMode,
                                     cudaStream_t stream)
{
    if (m_maxBatchSize <= 0 || inData.numImages() > m_maxBatchSize)
    {
        LOG_ERROR("Invalid maximum batch size");
        return ErrorCode::INVALID_PARAMETER;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    dim3 block(32, 4);
    dim3 grid(divUp(m_maxKernelSize.w, block.x), divUp(m_maxKernelSize.h, block.y), outData.numImages());

    cuda::Tensor1DWrap<int2> kernelSizeTensor(kernelSize);
    cuda::Tensor1DWrap<int2> kernelAnchorTensor(kernelAnchor);

    int kernelPitch2 = static_cast<int>(m_maxKernelSize.w * sizeof(float));
    int kernelPitch1 = m_maxKernelSize.h * kernelPitch2;

    cuda::Tensor3DWrap<float> kernelTensor(m_kernel, kernelPitch1, kernelPitch2);

    compute_average_blur_kernel<<<grid, block, 0, stream>>>(kernelTensor, kernelSizeTensor, kernelAnchorTensor);

    checkKernelErrors();

    typedef void (*filter2D_t)(
        const IImageBatchVarShapeDataStridedCuda &inData, const IImageBatchVarShapeDataStridedCuda &outData,
        const cuda::Tensor3DWrap<float> &kernelTensor, const cuda::Tensor1DWrap<int2> &kernelSizeTensor,
        const cuda::Tensor1DWrap<int2> &kernelAnchorTensor, NVCVBorderType borderMode, float borderValue,
        cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { AverageBlurFilter2D<uchar>, 0,  AverageBlurFilter2D<uchar3>,  AverageBlurFilter2D<uchar4>},
        {                          0, 0,                            0,                            0},
        {AverageBlurFilter2D<ushort>, 0, AverageBlurFilter2D<ushort3>, AverageBlurFilter2D<ushort4>},
        { AverageBlurFilter2D<short>, 0,  AverageBlurFilter2D<short3>,  AverageBlurFilter2D<short4>},
        {   AverageBlurFilter2D<int>, 0,    AverageBlurFilter2D<int3>,    AverageBlurFilter2D<int4>},
        { AverageBlurFilter2D<float>, 0,  AverageBlurFilter2D<float3>,  AverageBlurFilter2D<float4>},
    };

    const filter2D_t func = funcs[data_type][channels - 1];

    NVCV_ASSERT(func != 0);

    func(inData, outData, kernelTensor, kernelSizeTensor, kernelAnchorTensor, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op

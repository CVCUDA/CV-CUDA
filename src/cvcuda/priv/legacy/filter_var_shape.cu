/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "filter_utils.cuh"

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

template<class SrcWrapper, class DstWrapper>
__global__ void filter2D(const SrcWrapper src, DstWrapper dst, cuda::ImageBatchVarShapeWrap<float> kernel,
                         cuda::Tensor1DWrap<int2> kernelAnchor)
{
    using work_type = cuda::ConvertBaseTypeTo<float, typename DstWrapper::ValueType>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    int2 anchor = kernelAnchor[batch_idx];

    int2 kernelSize{kernel.width(batch_idx), kernel.height(batch_idx)};

    if (anchor.x < 0)
        anchor.x = kernelSize.x / 2;

    if (anchor.y < 0)
        anchor.y = kernelSize.y / 2;

    int3 srcCoord{0, 0, batch_idx};

    for (int i = 0; i < kernelSize.y; ++i)
    {
        srcCoord.y = y - anchor.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            srcCoord.x = x - anchor.x + j;

            res = res + src[srcCoord] * (*kernel.ptr(batch_idx, i, j));
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
}

template<typename D, NVCVBorderType B>
void Filter2DCaller(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const ImageBatchVarShapeDataStridedCuda &kernelData, const TensorDataStridedCuda &kernelAnchorData,
                    float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);
    cuda::ImageBatchVarShapeWrap<float>  kernel(kernelData);
    cuda::Tensor1DWrap<int2>             kernelAnchor(kernelAnchorData);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    filter2D<<<grid, block, 0, stream>>>(src, dst, kernel, kernelAnchor);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void Filter2D(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
              const ImageBatchVarShapeDataStridedCuda &kernelData, const TensorDataStridedCuda &kernelAnchorData,
              NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData,
                           const ImageBatchVarShapeDataStridedCuda &kernelData,
                           const TensorDataStridedCuda &kernelAnchorData, float borderValue, cudaStream_t stream);

    static const func_t funcs[] = {Filter2DCaller<D, NVCV_BORDER_CONSTANT>, Filter2DCaller<D, NVCV_BORDER_REPLICATE>,
                                   Filter2DCaller<D, NVCV_BORDER_REFLECT>, Filter2DCaller<D, NVCV_BORDER_WRAP>,
                                   Filter2DCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, kernelData, kernelAnchorData, borderValue, stream);
}

// Conv2DVarShape --------------------------------------------------------------

ErrorCode Conv2DVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                const ImageBatchVarShapeDataStridedCuda &outData,
                                const ImageBatchVarShapeDataStridedCuda &kernelData,
                                const TensorDataStridedCuda &kernelAnchorData, NVCVBorderType borderMode,
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
        const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
        const ImageBatchVarShapeDataStridedCuda &kernelData, const TensorDataStridedCuda &kernelAnchorData,
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

__device__ cuda::math::Vector<float, 9> kLaplacianKernel1{
    {0.0f,  1.0f, 0.0f,
     1.0f, -4.0f, 1.0f,
     0.0f,  1.0f, 0.0f}
};
__device__ cuda::math::Vector<float, 9> kLaplacianKernel3{
    {2.0f,  0.0f, 2.0f,
     0.0f, -8.0f, 0.0f,
     2.0f,  0.0f, 2.0f}
};

// clang-format on

// Laplacian kernels are either one or the other (above)
template<class SrcWrapper, class DstWrapper>
__global__ void laplacianFilter2D(const SrcWrapper src, DstWrapper dst, cuda::Tensor1DWrap<int> ksize,
                                  cuda::Tensor1DWrap<float> scale)
{
    using work_type = cuda::ConvertBaseTypeTo<float, typename DstWrapper::ValueType>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    constexpr int2 kernelSize = int2{3, 3};
    constexpr int2 anchor     = int2{1, 1};

    const int ksizeVal = ksize[batch_idx];

    NVCV_CUDA_ASSERT(ksizeVal == 1 || ksizeVal == 3, "E Wrong ksize = %d, expected: 1 or 3", ksizeVal);
    cuda::math::Vector<float, 9> kernel = ksizeVal == 1 ? kLaplacianKernel1 : kLaplacianKernel3;

    kernel *= scale[batch_idx];

    int  kidx = 0;
    int3 srcCoord{0, 0, batch_idx};

#pragma unroll
    for (int i = 0; i < kernelSize.y; ++i)
    {
        srcCoord.y = y - anchor.y + i;

#pragma unroll
        for (int j = 0; j < kernelSize.x; ++j)
        {
            srcCoord.x = x - anchor.x + j;

            res = res + src[srcCoord] * kernel[kidx++];
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
}

template<typename D, NVCVBorderType B>
void LaplacianFilter2DCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                             const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                             const TensorDataStridedCuda &scale, float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);
    cuda::Tensor1DWrap<int>              kernelApertureSize(ksize);
    cuda::Tensor1DWrap<float>            kernelScale(scale);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    laplacianFilter2D<<<grid, block, 0, stream>>>(src, dst, kernelApertureSize, kernelScale);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void LaplacianFilter2D(const ImageBatchVarShapeDataStridedCuda &inData,
                       const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                       const TensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
                       cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                           const TensorDataStridedCuda &scale, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {LaplacianFilter2DCaller<D, NVCV_BORDER_CONSTANT>, LaplacianFilter2DCaller<D, NVCV_BORDER_REPLICATE>,
           LaplacianFilter2DCaller<D, NVCV_BORDER_REFLECT>, LaplacianFilter2DCaller<D, NVCV_BORDER_WRAP>,
           LaplacianFilter2DCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, ksize, scale, borderValue, stream);
}

ErrorCode LaplacianVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                   const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                                   const TensorDataStridedCuda &scale, NVCVBorderType borderMode, cudaStream_t stream)
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

    typedef void (*filter2D_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                               const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                               const TensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
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

template<class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2D(const SrcWrapper src, DstWrapper dst, cuda::Tensor3DWrap<float> kernel,
                                 cuda::Tensor1DWrap<int2> kernelSizeArr)
{
    using work_type = cuda::ConvertBaseTypeTo<float, typename DstWrapper::ValueType>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    int2 kernelSize = kernelSizeArr[batch_idx];

    int2 anchor{kernelSize.x / 2, kernelSize.y / 2};

    int3 srcCoord{0, 0, batch_idx};

    for (int i = 0; i < kernelSize.y; ++i)
    {
        srcCoord.y = y - anchor.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            srcCoord.x = x - anchor.x + j;

            res = res + src[srcCoord] * (*kernel.ptr(batch_idx, i, j));
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
}

template<typename D, NVCVBorderType B>
void GaussianFilter2DCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                            const ImageBatchVarShapeDataStridedCuda &outData,
                            const cuda::Tensor3DWrap<float>         &kernelTensor,
                            const cuda::Tensor1DWrap<int2> &kernelSizeTensor, float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    gaussianFilter2D<<<grid, block, 0, stream>>>(src, dst, kernelTensor, kernelSizeTensor);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void GaussianFilter2D(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                      const cuda::Tensor3DWrap<float> &kernelTensor, const cuda::Tensor1DWrap<int2> &kernelSizeTensor,
                      NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData,
                           const cuda::Tensor3DWrap<float>         &kernelTensor,
                           const cuda::Tensor1DWrap<int2> &kernelSizeTensor, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {GaussianFilter2DCaller<D, NVCV_BORDER_CONSTANT>, GaussianFilter2DCaller<D, NVCV_BORDER_REPLICATE>,
           GaussianFilter2DCaller<D, NVCV_BORDER_REFLECT>, GaussianFilter2DCaller<D, NVCV_BORDER_WRAP>,
           GaussianFilter2DCaller<D, NVCV_BORDER_REFLECT101>};

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

ErrorCode GaussianVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                  const ImageBatchVarShapeDataStridedCuda &outData,
                                  const TensorDataStridedCuda &kernelSize, const TensorDataStridedCuda &sigma,
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

    computeGaussianKernelVarShape<<<grid, block, 0, stream>>>(kernelTensor, dataKernelSize, m_maxKernelSize,
                                                              kernelSizeTensor, sigmaTensor);

    checkKernelErrors();

    typedef void (*filter2D_t)(
        const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
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

template<class SrcWrapper, class DstWrapper>
__global__ void avgBlurFilter2D(const SrcWrapper src, DstWrapper dst, cuda::Tensor3DWrap<float> kernel,
                                cuda::Tensor1DWrap<int2> kernelSizeArr, cuda::Tensor1DWrap<int2> kernelAnchorArr)
{
    using work_type = cuda::ConvertBaseTypeTo<float, typename DstWrapper::ValueType>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    int2 kernelSize = kernelSizeArr[batch_idx];

    int2 anchor = kernelAnchorArr[batch_idx];

    int3 srcCoord{0, 0, batch_idx};

    for (int i = 0; i < kernelSize.y; ++i)
    {
        srcCoord.y = y - anchor.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            srcCoord.x = x - anchor.x + j;

            res = res + src[srcCoord] * (*kernel.ptr(batch_idx, i, j));
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
}

template<typename D, NVCVBorderType B>
void AverageBlurFilter2DCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                               const ImageBatchVarShapeDataStridedCuda &outData,
                               const cuda::Tensor3DWrap<float>         &kernelTensor,
                               const cuda::Tensor1DWrap<int2>          &kernelSizeTensor,
                               const cuda::Tensor1DWrap<int2> &kernelAnchorTensor, float borderValue,
                               cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    avgBlurFilter2D<<<grid, block, 0, stream>>>(src, dst, kernelTensor, kernelSizeTensor, kernelAnchorTensor);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void AverageBlurFilter2D(const ImageBatchVarShapeDataStridedCuda &inData,
                         const ImageBatchVarShapeDataStridedCuda &outData,
                         const cuda::Tensor3DWrap<float>         &kernelTensor,
                         const cuda::Tensor1DWrap<int2>          &kernelSizeTensor,
                         const cuda::Tensor1DWrap<int2> &kernelAnchorTensor, NVCVBorderType borderMode,
                         float borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(
        const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
        const cuda::Tensor3DWrap<float> &kernelTensor, const cuda::Tensor1DWrap<int2> &kernelSizeTensor,
        const cuda::Tensor1DWrap<int2> &kernelAnchorTensor, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {AverageBlurFilter2DCaller<D, NVCV_BORDER_CONSTANT>, AverageBlurFilter2DCaller<D, NVCV_BORDER_REPLICATE>,
           AverageBlurFilter2DCaller<D, NVCV_BORDER_REFLECT>, AverageBlurFilter2DCaller<D, NVCV_BORDER_WRAP>,
           AverageBlurFilter2DCaller<D, NVCV_BORDER_REFLECT101>};

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

ErrorCode AverageBlurVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                     const ImageBatchVarShapeDataStridedCuda &outData,
                                     const TensorDataStridedCuda &kernelSize, const TensorDataStridedCuda &kernelAnchor,
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

    dim3 block(32, 4);
    dim3 grid(divUp(m_maxKernelSize.w, block.x), divUp(m_maxKernelSize.h, block.y), outData.numImages());

    cuda::Tensor1DWrap<int2> kernelSizeTensor(kernelSize);
    cuda::Tensor1DWrap<int2> kernelAnchorTensor(kernelAnchor);

    int kernelPitch2 = static_cast<int>(m_maxKernelSize.w * sizeof(float));
    int kernelPitch1 = m_maxKernelSize.h * kernelPitch2;

    cuda::Tensor3DWrap<float> kernelTensor(m_kernel, kernelPitch1, kernelPitch2);

    computeMeanKernelVarShape<<<grid, block, 0, stream>>>(kernelTensor, kernelSizeTensor, kernelAnchorTensor);

    checkKernelErrors();

    typedef void (*filter2D_t)(
        const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
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

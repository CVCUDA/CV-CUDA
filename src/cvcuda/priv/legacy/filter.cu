/* Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"
#include "filter_utils.cuh"

#include <cvcuda/cuda_tools/TypeTraits.hpp>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

template<class SrcWrapper, class DstWrapper, class KernelWrapper>
__global__ void filter2D(SrcWrapper src, DstWrapper dst, Size2D dstSize, KernelWrapper kernel, Size2D kernelSize,
                         int2 kernelAnchor)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dstSize.w || y >= dstSize.h)
        return;

    int  kInd = 0;
    int3 coord{x, y, batch_idx};

    for (int i = 0; i < kernelSize.h; ++i)
    {
        coord.y = y - kernelAnchor.y + i;

        for (int j = 0; j < kernelSize.w; ++j)
        {
            coord.x = x - kernelAnchor.x + j;

            res = res + src[coord] * kernel[kInd++];
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
}

template<typename T, NVCVBorderType B, class KernelWrapper>
ErrorCode Filter2DCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                         KernelWrapper kernel, Size2D kernelSize, int2 kernelAnchor, float borderValue,
                         cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(16, 16);
    dim3 grid(divUp(dstSize.w, block.x), divUp(dstSize.h, block.y), outAccess->numSamples());

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue));
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);
        filter2D<<<grid, block, 0, stream>>>(src, dst, dstSize, kernel, kernelSize, kernelAnchor);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
    return ErrorCode::SUCCESS;
}

template<typename T, class KernelWrapper>
ErrorCode Filter2D(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, KernelWrapper kernel,
                   Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode, float borderValue,
                   cudaStream_t stream)
{
    switch (borderMode)
    {
#define NVCV_FILTER_CASE(BORDERTYPE) \
    case BORDERTYPE:                 \
        return Filter2DCaller<T, BORDERTYPE>(inData, outData, kernel, kernelSize, kernelAnchor, borderValue, stream);

        NVCV_FILTER_CASE(NVCV_BORDER_CONSTANT);
        NVCV_FILTER_CASE(NVCV_BORDER_REPLICATE);
        NVCV_FILTER_CASE(NVCV_BORDER_REFLECT);
        NVCV_FILTER_CASE(NVCV_BORDER_WRAP);
        NVCV_FILTER_CASE(NVCV_BORDER_REFLECT101);

#undef NVCV_FILTER_CASE
    default:
        break;
    }
    return ErrorCode::SUCCESS;
}

// Laplacian -------------------------------------------------------------------

// @brief Laplacian 3x3 kernels for ksize == 1 and ksize == 3

// clang-format off
constexpr Size2D kLaplacianKernelSize{3, 3};

constexpr cuda::math::Vector<float, 9> kLaplacianKernel1{
    {0.0f,  1.0f, 0.0f,
     1.0f, -4.0f, 1.0f,
     0.0f,  1.0f, 0.0f}
};
constexpr cuda::math::Vector<float, 9> kLaplacianKernel3{
    {2.0f,  0.0f, 2.0f,
     0.0f, -8.0f, 0.0f,
     2.0f,  0.0f, 2.0f}
};

// clang-format on

ErrorCode Laplacian::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, int ksize,
                           float scale, NVCVBorderType borderMode, cudaStream_t stream)
{
    if (!(ksize == 1 || ksize == 3))
    {
        LOG_ERROR("Invalid ksize " << ksize);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << inData.dtype() << ") and output (" << outData.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid input DataFormat " << format << ", the valid DataFormats are: \"NHWC\", \"HWC\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = input_shape.C;

    int2 kernelAnchor{-1, -1};
    normalizeAnchor(kernelAnchor, kLaplacianKernelSize);
    float borderValue = .0f;

    typedef ErrorCode (*filter2D_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                    cuda::math::Vector<float, 9> kernel, Size2D kernelSize, int2 kernelAnchor,
                                    NVCVBorderType borderMode, float borderValue, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { Filter2D<uchar>, 0,  Filter2D<uchar3>,  Filter2D<uchar4>},
        {               0, 0,                 0,                 0},
        {Filter2D<ushort>, 0, Filter2D<ushort3>, Filter2D<ushort4>},
        {               0, 0,                 0,                 0},
        {               0, 0,                 0,                 0},
        { Filter2D<float>, 0,  Filter2D<float3>,  Filter2D<float4>},
    };

    cuda::math::Vector<float, 9> kernel;

    if (ksize == 1)
    {
        kernel = kLaplacianKernel1;
    }
    else if (ksize == 3)
    {
        kernel = kLaplacianKernel3;
    }

    if (scale != 1)
    {
        kernel *= scale;
    }

    return funcs[data_type][channels - 1](inData, outData, kernel, kLaplacianKernelSize, kernelAnchor, borderMode,
                                          borderValue, stream);
}

// Gaussian --------------------------------------------------------------------

Gaussian::Gaussian(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_maxKernelSize(maxKernelSize)
{
    NVCV_CHECK_THROW(cudaMalloc(&m_kernel, maxKernelSize.w * maxKernelSize.h * sizeof(float)));
}

Gaussian::~Gaussian()
{
    NVCV_CHECK_LOG(cudaFree(m_kernel));
}

ErrorCode Gaussian::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, Size2D kernelSize,
                          double2 sigma, NVCVBorderType borderMode, cudaStream_t stream)
{
    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << inData.dtype() << ") and output (" << outData.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid input DataFormat " << format << ", the valid DataFormats are: \"NHWC\", \"HWC\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (sigma.y <= 0)
        sigma.y = sigma.x;

    // automatic detection of kernel size from sigma
    if (kernelSize.w <= 0 && sigma.x > 0)
        kernelSize.w = nvcv::cuda::round<int>(sigma.x * (data_type == kCV_8U ? 3 : 4) * 2 + 1) | 1;
    if (kernelSize.h <= 0 && sigma.y > 0)
        kernelSize.h = nvcv::cuda::round<int>(sigma.y * (data_type == kCV_8U ? 3 : 4) * 2 + 1) | 1;

    if (!(kernelSize.w > 0 && kernelSize.w % 2 == 1 && kernelSize.w <= m_maxKernelSize.w && kernelSize.h > 0
          && kernelSize.h % 2 == 1 && kernelSize.h <= m_maxKernelSize.h))
    {
        LOG_ERROR("Invalid kernel size = " << kernelSize.w << " " << kernelSize.h);
        return ErrorCode::INVALID_PARAMETER;
    }

    sigma.x = std::max(sigma.x, 0.0);
    sigma.y = std::max(sigma.y, 0.0);

    if (m_curSigma != sigma || m_curKernelSize != kernelSize)
    {
        dim3 block(32, 4);
        dim3 grid(divUp(kernelSize.w, block.x), divUp(kernelSize.h, block.y));

        computeGaussianKernel<<<grid, block, 0, stream>>>(m_kernel, kernelSize, sigma);

        checkKernelErrors();

        m_curKernelSize = kernelSize;
        m_curSigma      = sigma;
    }

    const int channels = input_shape.C;

    int2 kernelAnchor{-1, -1};
    normalizeAnchor(kernelAnchor, kernelSize);
    float borderValue = .0f;

    typedef ErrorCode (*filter2D_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                    float *kernel, Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode,
                                    float borderValue, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { Filter2D<uchar>, 0,  Filter2D<uchar3>,  Filter2D<uchar4>},
        {               0, 0,                 0,                 0},
        {Filter2D<ushort>, 0, Filter2D<ushort3>, Filter2D<ushort4>},
        { Filter2D<short>, 0,  Filter2D<short3>,  Filter2D<short4>},
        {   Filter2D<int>, 0,    Filter2D<int3>,    Filter2D<int4>},
        { Filter2D<float>, 0,  Filter2D<float3>,  Filter2D<float4>},
    };

    return funcs[data_type][channels - 1](inData, outData, m_kernel, kernelSize, kernelAnchor, borderMode, borderValue,
                                          stream);
}

// Average Blur ----------------------------------------------------------------

AverageBlur::AverageBlur(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_maxKernelSize(maxKernelSize)
{
    NVCV_CHECK_THROW(cudaMalloc(&m_kernel, maxKernelSize.w * maxKernelSize.h * sizeof(float)));
}

AverageBlur::~AverageBlur()
{
    NVCV_CHECK_LOG(cudaFree(m_kernel));
}

ErrorCode AverageBlur::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                             Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode, cudaStream_t stream)
{
    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << inData.dtype() << ") and output (" << outData.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid input DataFormat " << format << ", the valid DataFormats are: \"NHWC\", \"HWC\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(kernelSize.w > 0 && kernelSize.w % 2 == 1 && kernelSize.w <= m_maxKernelSize.w && kernelSize.h > 0
          && kernelSize.h % 2 == 1 && kernelSize.h <= m_maxKernelSize.h))
    {
        LOG_ERROR("Invalid ksize " << kernelSize.w << " " << kernelSize.h);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!(kernelAnchor.x == -1 || (kernelAnchor.x >= 0 && kernelAnchor.x < kernelSize.w)))
    {
        LOG_ERROR("Invalid kernelAnchor.x " << kernelAnchor.x);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!(kernelAnchor.y == -1 || (kernelAnchor.y >= 0 && kernelAnchor.y < kernelSize.h)))
    {
        LOG_ERROR("Invalid kernelAnchor.y " << kernelAnchor.y);
        return ErrorCode::INVALID_PARAMETER;
    }

    const int channels = input_shape.C;

    normalizeAnchor(kernelAnchor, kernelSize);
    float borderValue = .0f;

    typedef ErrorCode (*filter2D_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                    float *kernel, Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode,
                                    float borderValue, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { Filter2D<uchar>, 0,  Filter2D<uchar3>,  Filter2D<uchar4>},
        {               0, 0,                 0,                 0},
        {Filter2D<ushort>, 0, Filter2D<ushort3>, Filter2D<ushort4>},
        { Filter2D<short>, 0,  Filter2D<short3>,  Filter2D<short4>},
        {   Filter2D<int>, 0,    Filter2D<int3>,    Filter2D<int4>},
        { Filter2D<float>, 0,  Filter2D<float3>,  Filter2D<float4>},
    };

    if (m_curKernelSize != kernelSize)
    {
        int k_size = kernelSize.h * kernelSize.w;

        computeMeanKernel<<<1, k_size, 0, stream>>>(m_kernel, k_size);

        checkKernelErrors();

        m_curKernelSize = kernelSize;
    }

    return funcs[data_type][channels - 1](inData, outData, m_kernel, kernelSize, kernelAnchor, borderMode, borderValue,
                                          stream);
}

} // namespace nvcv::legacy::cuda_op

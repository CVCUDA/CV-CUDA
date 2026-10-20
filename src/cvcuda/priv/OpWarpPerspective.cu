/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

#include "Nvtx.hpp"
#include "OpWarpPerspective.hpp"
#include "TensorAccessValidation.hpp"
#include "WarpPerspectiveAccess.hpp"

#include "WarpCommon.cuh"

#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cstdint>
#include <memory>

namespace cuda = nvcv::cuda;
namespace warp = cvcuda::priv::warp;

namespace {

template<class Pixel>
bool PixelAligned(const nvcv::TensorDataStridedCuda &data)
{
    const auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(data);
    return cvcuda::priv::perspective::PixelAligned<Pixel>(data.basePtr(), access->rowStride(), access->sampleStride(),
                                                          access->numRows(), access->numSamples());
}

template<class Pixel>
bool PixelAligned(const nvcv::ImageBatchVarShape &images)
{
    for (int sample = 0; sample < images.numImages(); ++sample)
    {
        const auto  data  = images[sample].exportData<nvcv::ImageDataStridedCuda>();
        const auto &plane = data->plane(0);
        if (!cvcuda::priv::perspective::PixelAligned<Pixel>(plane.basePtr, plane.rowStride, 0, plane.height, 1))
            return false;
    }
    return true;
}

static void WarpScalarTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                             warp::PerspectiveTransform transform, const warp::WarpParams &params,
                             NVCVBorderType borderMode, const float4 &borderValue, cudaStream_t stream)
{
    namespace access  = cvcuda::priv::perspective;
    const auto input  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    const auto output = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    const int2 inputSize{input->numCols(), input->numRows()};
    const int2 outputSize{output->numCols(), output->numRows()};
    const int  batches = output->numSamples();
    warp::CheckTensorFitsInt32(*input, *output, batches);
    warp::DispatchVec1Type(
        params.elem,
        [&](auto tag)
        {
            using Element = typename decltype(tag)::type;
            warp::DispatchInterpBorder<true>(
                params.interpolation, borderMode,
                [&](auto interpolation, auto border)
                {
                    constexpr auto interpolationMode = decltype(interpolation)::value;
                    constexpr auto borderType        = decltype(border)::value;
                    constexpr int  columns           = warp::TensorWarpXUnroll<interpolationMode>();
                    constexpr int  rows              = warp::kTensorRowsPerThread<warp::PerspectiveTransform, Element,
                                                                    interpolationMode, borderType>;
                    const auto [block, grid]
                        = warp::WarpLaunchConfig(outputSize.x, outputSize.y, batches, columns, rows);
                    for (int channel = 0; channel < params.channels; ++channel)
                    {
                        const access::ScalarTensor<const Element> source(
                            inData.basePtr() + channel * input->chStride(), static_cast<int32_t>(input->sampleStride()),
                            static_cast<int32_t>(input->rowStride()), static_cast<int32_t>(input->colStride()));
                        const access::ScalarTensor<Element> destination(
                            outData.basePtr() + channel * output->chStride(),
                            static_cast<int32_t>(output->sampleStride()), static_cast<int32_t>(output->rowStride()),
                            static_cast<int32_t>(output->colStride()));
                        const auto channelBorder
                            = warp::WrapBorderValue<Element, borderType>(warp::PlanarBorderValue(borderValue, channel));
                        const access::ScalarInterpolation<Element, borderType, interpolationMode> interpolated(
                            source, channelBorder, 0.f, 0.f, inputSize.y, inputSize.x);
                        warp::tensor::warp<false, columns, rows, warp::PerspectiveTransform>
                            <<<grid, block, 0, stream>>>(interpolated, destination, outputSize, inputSize, transform);
                        NVCV_CHECK_THROW(cudaGetLastError());
                    }
                });
        });
}

static void WarpScalarVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                               const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                               cuda::Tensor2DWrap<float, int32_t> transform, const warp::WarpParams &params,
                               NVCVBorderType borderMode, const float4 &borderValue, cudaStream_t stream)
{
    namespace access = cvcuda::priv::perspective;
    warp::DispatchVec1Type(
        params.elem,
        [&](auto tag)
        {
            using Element = typename decltype(tag)::type;
            warp::DispatchInterpBorder(
                params.interpolation, borderMode,
                [&](auto interpolation, auto border)
                {
                    constexpr auto interpolationMode = decltype(interpolation)::value;
                    constexpr auto borderType        = decltype(border)::value;
                    constexpr bool tiled             = interpolationMode == NVCV_INTERP_LINEAR;
                    constexpr int  columns           = tiled ? 2 : 1;
                    constexpr int  rows
                        = tiled ? warp::kVarShapeRowsPerThread<warp::PerspectiveTransform, interpolationMode> : 1;
                    const auto size = outData.maxSize();
                    const auto [block, grid]
                        = warp::WarpLaunchConfig(size.w, size.h, outData.numImages(), columns, rows);
                    for (int channel = 0; channel < params.channels; ++channel)
                    {
                        const auto channelBorder
                            = warp::WrapBorderValue<Element, borderType>(warp::PlanarBorderValue(borderValue, channel));
                        const access::ScalarImageInterpolation<Element, borderType, interpolationMode> source(
                            inData.imageList(), channel, params.channels, channelBorder);
                        const access::ScalarImageBatch<Element> destination(outData.imageList(), channel,
                                                                            params.channels);
                        if constexpr (tiled)
                        {
                            warp::varshape::warp_x2<columns, rows, warp::PerspectiveTransform>
                                <<<grid, block, warp::kNumCoeffs * sizeof(float), stream>>>(source, destination,
                                                                                            transform);
                        }
                        else
                        {
                            warp::varshape::warp<warp::PerspectiveTransform>
                                <<<grid, block, warp::kNumCoeffs * sizeof(float), stream>>>(source, destination,
                                                                                            transform);
                        }
                        NVCV_CHECK_THROW(cudaGetLastError());
                    }
                });
        });
}

// One thread per image inverts that image's 3x3 perspective matrix into the operator's coefficient
// buffer, which is strided by 9 floats per image.
__global__ void inverseMatWarpPerspective(const int numImages, const cuda::Tensor2DWrap<float, int32_t> in,
                                          cuda::Tensor2DWrap<float, int32_t> out)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    cuda::math::Matrix<float, 3, 3> transMatrix;

    transMatrix.load(in.ptr(index));

    cuda::math::inv_inplace(transMatrix);

    transMatrix.store(out.ptr(index));
}

} // namespace

namespace cvcuda::priv {

WarpPerspective::WarpPerspective(const int32_t maxVarShapeBatchSize)
    : m_maxVarShapeBatchSize(maxVarShapeBatchSize)
    , m_transformMatrix([maxVarShapeBatchSize](int)
                        { return std::make_unique<warp::TransformMatrix>(maxVarShapeBatchSize); })
{
}

void WarpPerspective::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                 const NVCVPerspectiveTransform transMatrix, const int32_t flags,
                                 const NVCVBorderType borderMode, const float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::WarpPerspective::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    const warp::WarpParams params = warp::ValidateTensor(*inData, *outData, flags, borderMode);

    warp::PerspectiveTransform transform(transMatrix);

    if (!(flags & NVCV_WARP_INVERSE_MAP))
    {
        cuda::math::Matrix<float, 3, 3> tempMatrixForInverse;

        tempMatrixForInverse.load(transMatrix);

        cuda::math::inv_inplace(tempMatrixForInverse);

        tempMatrixForInverse.store(transform.xform);
    }

    bool aligned = true;
    if (!warp::IsPlanar(params.layout))
    {
        warp::DispatchPixelType(params.elem, params.channels,
                                [&](auto tag)
                                {
                                    using Pixel = typename decltype(tag)::type;
                                    aligned     = PixelAligned<Pixel>(*inData) && PixelAligned<Pixel>(*outData);
                                });
    }
    if (aligned)
    {
        warp::RunWarpTensor<warp::PerspectiveTransform>(*inData, *outData, transform, params, borderMode, borderValue,
                                                        stream);
    }
    else
    {
        WarpScalarTensor(*inData, *outData, transform, params, borderMode, borderValue, stream);
    }
}

void WarpPerspective::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                 const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &transMatrix,
                                 const int32_t flags, const NVCVBorderType borderMode, const float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::WarpPerspective::operator()[ImageBatchVarShape]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    auto transMatrixData = RequireCudaTensor(transMatrix, "transformation matrix");

    if (m_maxVarShapeBatchSize <= 0)
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Operator warp perspective var shape is not initialized properly, maxVarShapeBatchSize: %d",
            m_maxVarShapeBatchSize);
    }

    if (m_maxVarShapeBatchSize < inData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid number of images, it should not exceed %d",
                              m_maxVarShapeBatchSize);
    }

    RequireUniqueFormat(*inData, "input");

    const warp::WarpParams params = warp::ValidateVarShape(*inData, *outData, flags, borderMode);

    // Both branches below read all nine coefficients for every image out of transMatrix, so reject
    // a matrix that cannot supply them before wrapping it.
    if (transMatrixData->dtype() != nvcv::TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Transformation matrix must be F32");
    }
    if (transMatrixData->rank() != 2 || transMatrixData->shape(0) < inData->numImages()
        || transMatrixData->shape(1) < warp::kNumCoeffs)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Transformation matrix must be a 2D tensor of at least %d x %d, one row per image",
                              inData->numImages(), warp::kNumCoeffs);
    }

    float *d_matrix = m_transformMatrix.get().matrix;

    // Wrap the matrix in 2D wrappers with proper pitch
    cuda::Tensor2DWrap<float, int32_t> transMatrixInput(*transMatrixData);
    cuda::Tensor2DWrap<float, int32_t> transMatrixOutput(d_matrix, static_cast<int>(sizeof(float) * warp::kNumCoeffs));

    // Check if inverse op is needed
    const bool performInverse = flags & NVCV_WARP_INVERSE_MAP;

    if (!performInverse)
    {
        // Cap at 256 threads per block so the launch remains valid when
        // numImages exceeds the per-block thread limit (typically 1024).
        constexpr int kThreadsPerBlock = 256;
        const int     numBlocks        = (inData->numImages() + kThreadsPerBlock - 1) / kThreadsPerBlock;
        inverseMatWarpPerspective<<<numBlocks, kThreadsPerBlock, 0, stream>>>(inData->numImages(), transMatrixInput,
                                                                              transMatrixOutput);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        NVCV_CHECK_THROW(cudaMemcpy2DAsync(d_matrix, sizeof(float) * warp::kNumCoeffs, transMatrixInput.ptr(0, 0),
                                           transMatrixInput.strides()[0], sizeof(float) * warp::kNumCoeffs,
                                           inData->numImages(), cudaMemcpyDeviceToDevice, stream));
    }

    bool aligned = true;
    if (!warp::IsPlanar(params.layout))
    {
        warp::DispatchPixelType(params.elem, params.channels,
                                [&](auto tag)
                                {
                                    using Pixel = typename decltype(tag)::type;
                                    if constexpr (alignof(Pixel) > alignof(cuda::BaseType<Pixel>))
                                        aligned = PixelAligned<Pixel>(in) && PixelAligned<Pixel>(out);
                                });
    }
    if (aligned)
    {
        warp::RunWarpVarShape<warp::PerspectiveTransform>(*inData, *outData, transMatrixOutput, params, borderMode,
                                                          borderValue, stream);
    }
    else
    {
        WarpScalarVarShape(*inData, *outData, transMatrixOutput, params, borderMode, borderValue, stream);
    }
}

} // namespace cvcuda::priv

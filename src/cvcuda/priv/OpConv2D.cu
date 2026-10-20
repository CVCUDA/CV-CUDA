/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Nvtx.hpp"
#include "OpConv2D.hpp"
#include "TensorAccessValidation.hpp"

#include "FilterCommon.cuh"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <type_traits>

namespace cvcuda::priv {

namespace {

namespace cuda   = nvcv::cuda;
namespace util   = nvcv::util;
namespace filter = cvcuda::priv::filter;

constexpr int kBlockWidth  = 16;
constexpr int kBlockHeight = 16;

// Output columns per thread, gated on the value type from full-surface benchmarks: x8 wins for byte
// scalar/uchar4, while float3/float4 need x2 to avoid register pressure from the wider vector work
// type.
template<typename D>
constexpr int kOutputs = std::is_same_v<D, filter::uchar> || std::is_same_v<D, uchar4>
                           ? 8
                           : (std::is_same_v<D, float3> || std::is_same_v<D, float4> ? 2 : 4);

// Planar dispatch is per data type only, so the byte case is the only one that gets the wider tile.
template<typename D>
constexpr int kPlanarOutputs = std::is_same_v<D, filter::uchar> ? 8 : 4;

template<typename D, NVCVBorderType B>
void Conv2DInterleaved(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &kernelData,
                       const nvcv::TensorDataStridedCuda &kernelAnchorData, float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));

    auto dst = filter::MakeVarShapeDst(cuda::ImageBatchVarShapeWrap<D>(outData));

    filter::ImageBatchCoefs coefs{cuda::ImageBatchVarShapeWrap<float>(kernelData),
                                  cuda::Tensor1DWrap<int2, int32_t>(kernelAnchorData)};

    constexpr int outputs = kOutputs<D>;

    dim3 block(kBlockWidth, kBlockHeight);
    dim3 grid(util::DivUp(inData.maxSize().w, kBlockWidth), util::DivUp(inData.maxSize().h, kBlockHeight),
              outData.numImages());
    dim3 tiledGrid(util::DivUp(inData.maxSize().w, kBlockWidth * outputs),
                   util::DivUp(inData.maxSize().h, kBlockHeight), outData.numImages());

    const int2   maxKernelSize{kernelData.maxSize().w, kernelData.maxSize().h};
    const size_t sharedBytes = filter::TiledSharedBytes(outputs, block, maxKernelSize, sizeof(D));

    auto untiled = [&]
    {
        filter::Filter2D<<<grid, block, 0, stream>>>(src, dst, coefs);
    };

    const bool launched
        = filter::TryCommonKernelSizes(maxKernelSize,
                                       [&](auto ksize)
                                       {
                                           constexpr int kSize = decltype(ksize)::value;
                                           if (sharedBytes <= filter::kMaxSharedBytes)
                                           {
                                               filter::Filter2DTiledFixed<outputs, kSize, kSize, false,
                                                                          /*LoadCoefsBeforeBounds=*/true>
                                                   <<<tiledGrid, block, sharedBytes, stream>>>(src, dst, coefs);
                                           }
                                           else
                                           {
                                               untiled();
                                           }
                                       });

    if (!launched)
    {
        if (sharedBytes <= filter::kMaxSharedBytes)
        {
            filter::Filter2DTiled<outputs><<<tiledGrid, block, sharedBytes, stream>>>(src, dst, coefs);
        }
        else
        {
            untiled();
        }
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename D, NVCVBorderType B>
void Conv2DPlanar(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                  const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                  const nvcv::ImageBatchVarShapeDataStridedCuda &kernelData,
                  const nvcv::TensorDataStridedCuda &kernelAnchorData, int channels, float borderValue,
                  cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));

    auto dst = filter::MakeVarShapePlanarDst(cuda::ImageBatchVarShapeWrap<D>(outData), channels);

    filter::ImageBatchCoefs coefs{cuda::ImageBatchVarShapeWrap<float>(kernelData),
                                  cuda::Tensor1DWrap<int2, int32_t>(kernelAnchorData)};

    constexpr int outputs = kPlanarOutputs<D>;

    dim3          block(kBlockWidth, kBlockHeight);
    const int64_t tiledGridZ = static_cast<int64_t>(outData.numImages()) * channels;

    dim3 fallbackGrid(util::DivUp(outData.maxSize().w, kBlockWidth), util::DivUp(outData.maxSize().h, kBlockHeight),
                      outData.numImages());

    auto untiled = [&]
    {
        filter::Filter2DPlanar<<<fallbackGrid, block, 0, stream>>>(src, dst, coefs, channels);
    };

    // Tiling gives every (image, plane) its own block, so it needs gridDim.z = numImages * channels;
    // past the cap the untiled kernel walks the planes inside one thread instead.
    if (tiledGridZ > filter::kMaxGridZ)
    {
        untiled();
        NVCV_CHECK_THROW(cudaGetLastError());
        return;
    }

    dim3 tiledGrid(util::DivUp(outData.maxSize().w, kBlockWidth * outputs),
                   util::DivUp(outData.maxSize().h, kBlockHeight), static_cast<unsigned int>(tiledGridZ));

    const int2   maxKernelSize{kernelData.maxSize().w, kernelData.maxSize().h};
    const size_t sharedBytes = filter::TiledSharedBytes(outputs, block, maxKernelSize, sizeof(D));

    const bool launched
        = filter::TryCommonKernelSizes(maxKernelSize,
                                       [&](auto ksize)
                                       {
                                           constexpr int kSize = decltype(ksize)::value;
                                           if (sharedBytes <= filter::kMaxSharedBytes)
                                           {
                                               filter::Filter2DTiledFixed<outputs, kSize, kSize, false,
                                                                          /*LoadCoefsBeforeBounds=*/true>
                                                   <<<tiledGrid, block, sharedBytes, stream>>>(src, dst, coefs);
                                           }
                                           else
                                           {
                                               untiled();
                                           }
                                       });

    if (!launched)
    {
        if (sharedBytes <= filter::kMaxSharedBytes)
        {
            filter::Filter2DTiled<outputs><<<tiledGrid, block, sharedBytes, stream>>>(src, dst, coefs);
        }
        else
        {
            untiled();
        }
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

} // namespace

void Conv2D::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                        const nvcv::ImageBatchVarShape &kernel, const nvcv::Tensor &kernelAnchor,
                        NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Conv2D::operator()[ImageBatchVarShape]");
    auto inData           = RequireCudaVarShape(in, stream, "Input");
    auto outData          = RequireCudaVarShape(out, stream, "Output");
    auto kernelData       = RequireCudaVarShape(kernel, stream, "Kernel");
    auto kernelAnchorData = RequireCudaTensor(kernelAnchor, "Kernel anchor");

    RequireUniqueFormat(*inData, "input");
    RequireUniqueFormat(*outData, "output");

    // The dispatch derives the element type and channel count from the input alone and then wraps
    // the output and the kernel batch with them, and indexes the anchors up to numImages.
    if (inData->numImages() != outData->numImages() || inData->numImages() != kernelData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input, output and kernel must have the same number of images, got %d, %d and %d",
                              inData->numImages(), outData->numImages(), kernelData->numImages());
    }
    RequireSameFormat(*inData, *outData);
    if (!kernelData->uniqueFormat() || kernelData->uniqueFormat().planeDataType(0) != nvcv::TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Kernel images must all have the same format and be F32");
    }

    cvcuda::priv::RequireSameVarShapeImageSizes(in, out);

    cvcuda::priv::RequirePerImageParam(*kernelAnchorData, nvcv::TYPE_2S32, inData->numImages(), "Kernel anchor");

    const filter::BatchLayout inLayout  = filter::ClassifyBatchLayout(*inData);
    const filter::BatchLayout outLayout = filter::ClassifyBatchLayout(*outData);
    filter::RequireSameBatchLayout(inLayout, outLayout);

    filter::ValidateBorderMode(borderMode);

    const filter::ElemType elemType = filter::ClassifyFilterElemType(inData->uniqueFormat());

    const int channels = inData->uniqueFormat().numChannels();
    filter::ValidateChannelCount(channels);

    constexpr float borderValue = 0.f;

    if (inLayout.planar)
    {
        filter::RequirePlanarGridZ(outData->numImages());

        // Planar images carry one channel per plane, so the value type is always the scalar.
        filter::RunBaseTypeBorderSwitch(elemType, borderMode,
                                        [&](auto base, auto border)
                                        {
                                            Conv2DPlanar<decltype(base), decltype(border)::value>(
                                                *inData, *outData, *kernelData, *kernelAnchorData, channels,
                                                borderValue, stream);
                                        });
        return;
    }

    filter::RunTypeBorderSwitch(elemType, channels, borderMode,
                                [&](auto value, auto border)
                                {
                                    Conv2DInterleaved<decltype(value), decltype(border)::value>(
                                        *inData, *outData, *kernelData, *kernelAnchorData, borderValue, stream);
                                });
}

} // namespace cvcuda::priv

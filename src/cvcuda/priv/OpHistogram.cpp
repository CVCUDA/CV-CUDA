/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpHistogram.hpp"

#include "Nvtx.hpp"
#include "OpReformat.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <cuda_runtime.h>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <array>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

HistogramPlanarBridgeWorkspace::HistogramPlanarBridgeWorkspace()
{
    NVCV_CHECK_THROW(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
}

HistogramPlanarBridgeWorkspace::~HistogramPlanarBridgeWorkspace()
{
    if (ready != nullptr)
    {
        if (busy)
        {
            NVCV_CHECK_LOG(cudaEventSynchronize(ready));
        }
        NVCV_CHECK_LOG(cudaEventDestroy(ready));
    }
}

void HistogramPlanarBridgeWorkspace::ensure(const nvcv::TensorShape &inShape, nvcv::DataType inDtype,
                                            const nvcv::TensorShape *maskShape, nvcv::DataType maskDtype,
                                            cudaStream_t stream)
{
    const bool reallocateIn = !interleavedIn || interleavedIn.shape() != inShape || interleavedIn.dtype() != inDtype;
    const bool reallocateMask
        = maskShape != nullptr
       && (!interleavedMask || interleavedMask.shape() != *maskShape || interleavedMask.dtype() != maskDtype);

    if (busy)
    {
        if (reallocateIn || reallocateMask)
        {
            NVCV_CHECK_THROW(cudaEventSynchronize(ready));
            busy = false;
        }
        else
        {
            NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, ready));
        }
    }

    if (reallocateIn)
    {
        interleavedIn = nvcv::Tensor(inShape, inDtype);
    }
    if (reallocateMask)
    {
        interleavedMask = nvcv::Tensor(*maskShape, maskDtype);
    }
}

void HistogramPlanarBridgeWorkspace::record(cudaStream_t stream)
{
    NVCV_CHECK_THROW(cudaEventRecord(ready, stream));
    busy = true;
}

namespace {

bool IsHistogramPlanarLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

bool IsHistogramInterleavedLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC;
}

nvcv::TensorShape MakeHistogramLegacyInputShape(const nvcv::TensorDataStridedCuda              &data,
                                                const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    std::array<nvcv::TensorShape::DimType, NVCV_TENSOR_MAX_RANK> shape{};
    const char                                                  *layout = nullptr;
    int32_t                                                      rank   = 0;

    const bool hasBatch = data.layout() == nvcv::TENSOR_NCHW;
    if (hasBatch)
    {
        shape[rank] = access.numSamples();
        ++rank;
    }

    if (hasBatch || data.layout() == nvcv::TENSOR_CHW)
    {
        shape[rank] = access.numRows();
        ++rank;
        shape[rank] = access.numCols();
        ++rank;
        shape[rank] = access.numChannels();
        ++rank;
        layout = hasBatch ? "NHWC" : "HWC";
        return nvcv::TensorShape(shape.data(), rank, layout);
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Input/mask must have NCHW or CHW layout for planar Histogram adaptation");
}

void InferPlanarHistogramNoMask(legacy::Histogram &legacyOp, cudaStream_t stream,
                                const nvcv::TensorDataStridedCuda &interleavedInData,
                                const nvcv::TensorDataStridedCuda &outHistogram)
{
    NVCV_CHECK_THROW(
        legacyOp.infer(interleavedInData, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, outHistogram, stream));
}

void InferPlanarHistogramWithMask(legacy::Histogram &legacyOp, Reformat &reformatOp,
                                  HistogramPlanarBridgeWorkspace &workspace, cudaStream_t stream,
                                  const nvcv::Tensor &mask, const nvcv::TensorDataStridedCuda &inData,
                                  const nvcv::TensorDataStridedCuda &interleavedInData,
                                  const nvcv::TensorDataStridedCuda &outHistogram)
{
    auto maskData = mask.exportData<nvcv::TensorDataStridedCuda>();
    if (maskData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Mask must be cuda-accessible, pitch-linear tensor");
    }
    if (!IsHistogramPlanarLayout(maskData->layout()) || maskData->layout() != inData.layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Mask layout must match planar Histogram input layout");
    }

    if (auto maskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskData); !maskAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask must be an image-like planar tensor");
    }

    reformatOp(stream, mask, workspace.interleavedMask);

    NVCV_CHECK_THROW(legacyOp.infer(interleavedInData, nvcv::OptionalTensorConstRef{workspace.interleavedMask},
                                    outHistogram, stream));
}

void InferPlanarHistogram(legacy::Histogram &legacyOp, Reformat &reformatOp, HistogramPlanarBridgeWorkspace &workspace,
                          cudaStream_t stream, const nvcv::Tensor &in, nvcv::OptionalTensorConstRef mask,
                          const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outHistogram)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be an image-like planar tensor");
    }

    nvcv::TensorShape interleavedInShape = MakeHistogramLegacyInputShape(inData, *inAccess);

    nvcv::TensorShape maskShape;
    nvcv::DataType    maskDtype = nvcv::TYPE_U8;
    if (mask)
    {
        auto maskData = mask->get().exportData<nvcv::TensorDataStridedCuda>();
        if (maskData == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Mask must be cuda-accessible, pitch-linear tensor");
        }
        auto maskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskData);
        if (!maskAccess)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask must be an image-like planar tensor");
        }
        maskShape = MakeHistogramLegacyInputShape(*maskData, *maskAccess);
        maskDtype = maskData->dtype();
    }

    std::lock_guard lock(workspace.mutex);
    workspace.ensure(interleavedInShape, inData.dtype(), mask ? &maskShape : nullptr, maskDtype, stream);

    reformatOp(stream, in, workspace.interleavedIn);

    auto interleavedInData = workspace.interleavedIn.exportData<nvcv::TensorDataStridedCuda>();
    if (interleavedInData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Internal Histogram input tensor must be cuda-accessible");
    }

    if (!mask)
    {
        InferPlanarHistogramNoMask(legacyOp, stream, *interleavedInData, outHistogram);
        workspace.record(stream);
        return;
    }

    InferPlanarHistogramWithMask(legacyOp, reformatOp, workspace, stream, mask->get(), inData, *interleavedInData,
                                 outHistogram);
    workspace.record(stream);
}

void ValidateInterleavedMask(nvcv::OptionalTensorConstRef mask, nvcv::TensorLayout inputLayout)
{
    if (!mask)
    {
        return;
    }

    auto maskData = mask->get().exportData<nvcv::TensorDataStridedCuda>();
    if (maskData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Mask must be cuda-accessible, pitch-linear tensor");
    }
    if (!IsHistogramInterleavedLayout(maskData->layout()) || maskData->layout() != inputLayout)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask layout must match Histogram input layout");
    }
}

} // namespace

std::unique_ptr<HistogramPlanarBridgeWorkspace> Histogram::CreatePlanarBridgeWorkspace(int)
{
    return std::make_unique<HistogramPlanarBridgeWorkspace>();
}

Histogram::Histogram()
{
    m_legacyOp   = std::make_unique<legacy::Histogram>();
    m_reformatOp = std::make_unique<Reformat>();
}

Histogram::~Histogram() = default;

void Histogram::operator()(cudaStream_t stream, const nvcv::Tensor &in, nvcv::OptionalTensorConstRef mask,
                           const nvcv::Tensor &histogram) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Histogram::operator()[Tensor]");
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outHistogram = histogram.exportData<nvcv::TensorDataStridedCuda>();
    if (outHistogram == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    if (const bool inPlanar = IsHistogramPlanarLayout(inData->layout()); inPlanar)
    {
        InferPlanarHistogram(*m_legacyOp, *m_reformatOp, m_planarWorkspace.get(), stream, in, mask, *inData,
                             *outHistogram);
        return;
    }

    if (!IsHistogramInterleavedLayout(inData->layout()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input layout must be NHWC/HWC or single-channel NCHW/CHW");
    }

    ValidateInterleavedMask(mask, inData->layout());

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, mask, *outHistogram, stream));
}

} // namespace cvcuda::priv

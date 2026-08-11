/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpMorphology.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>

#include <utility>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

namespace {

std::pair<NVCVMorphologyType, NVCVMorphologyType> SplitMorphologyPasses(NVCVMorphologyType morphType)
{
    if (morphType == NVCVMorphologyType::NVCV_OPEN)
    {
        return {NVCVMorphologyType::NVCV_ERODE, NVCVMorphologyType::NVCV_DILATE};
    }

    return {NVCVMorphologyType::NVCV_DILATE, NVCVMorphologyType::NVCV_ERODE};
}

void RunTensorDilateErode(legacy::Morphology &legacyOp, cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                          const nvcv::TensorDataStridedCuda &outData, nvcv::OptionalTensorConstRef workspace,
                          NVCVMorphologyType morphType, nvcv::Size2D maskSize, int2 anchor, int32_t iteration,
                          NVCVBorderType borderMode)
{
    if (workspace == nullptr && iteration > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Workspace must be provided for iterations > 1");
    }

    if (workspace == nullptr)
    {
        NVCV_CHECK_THROW(
            legacyOp.infer(inData, outData, morphType, maskSize, anchor, iteration == 0, borderMode, stream));
        return;
    }

    auto workspaceData = workspace->get().exportData<nvcv::TensorDataStridedCuda>();
    NVCV_ASSERT(workspaceData);

    const nvcv::TensorDataStridedCuda *iterIn  = &inData;
    const nvcv::TensorDataStridedCuda *iterOut = (iteration % 2 == 1) ? &outData : &(*workspaceData);
    NVCV_CHECK_THROW(legacyOp.infer(*iterIn, *iterOut, morphType, maskSize, anchor, false, borderMode, stream));

    std::swap(iterIn, iterOut);
    iterOut = (iteration % 2 == 0) ? &outData : &(*workspaceData);

    for (int i = 1; i < iteration; ++i)
    {
        NVCV_CHECK_THROW(legacyOp.infer(*iterIn, *iterOut, morphType, maskSize, anchor, false, borderMode, stream));
        std::swap(iterIn, iterOut);
    }
}

void RunTensorOpenClose(legacy::Morphology &legacyOp, cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                        const nvcv::TensorDataStridedCuda &outData, nvcv::OptionalTensorConstRef workspace,
                        NVCVMorphologyType morphType, nvcv::Size2D maskSize, int2 anchor, int32_t iteration,
                        NVCVBorderType borderMode)
{
    if (workspace == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Workspace must be provided for NVCV_CLOSE or NVCV_OPEN");
    }

    auto [first, second] = SplitMorphologyPasses(morphType);

    auto workspaceData = workspace->get().exportData<nvcv::TensorDataStridedCuda>();
    NVCV_ASSERT(workspaceData);

    NVCV_CHECK_THROW(
        legacyOp.infer(inData, *workspaceData, first, maskSize, anchor, iteration == 0, borderMode, stream));
    NVCV_CHECK_THROW(
        legacyOp.infer(*workspaceData, outData, second, maskSize, anchor, iteration == 0, borderMode, stream));

    for (int i = 1; i < iteration; ++i)
    {
        NVCV_CHECK_THROW(legacyOp.infer(outData, *workspaceData, first, maskSize, anchor, false, borderMode, stream));
        NVCV_CHECK_THROW(legacyOp.infer(*workspaceData, outData, second, maskSize, anchor, false, borderMode, stream));
    }
}

void RunVarShapeDilateErode(legacy::MorphologyVarShape &legacyOp, cudaStream_t stream,
                            const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                            nvcv::OptionalImageBatchVarShapeConstRef workspace, NVCVMorphologyType morphType,
                            const nvcv::TensorDataStridedCuda &masksData,
                            const nvcv::TensorDataStridedCuda &anchorsData, int32_t iteration,
                            NVCVBorderType borderMode)
{
    if (workspace == nullptr && iteration > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Workspace must be provided for iterations > 1");
    }

    if (workspace == nullptr)
    {
        NVCV_CHECK_THROW(legacyOp.infer(in, out, morphType, masksData, anchorsData, iteration == 0, borderMode,
                                        iteration > 1, stream));
        return;
    }

    NVCV_ASSERT(workspace);

    const nvcv::ImageBatchVarShape *iterIn  = &in;
    const nvcv::ImageBatchVarShape *iterOut = (iteration % 2 == 1) ? &out : &workspace->get();

    NVCV_CHECK_THROW(legacyOp.infer(*iterIn, *iterOut, morphType, masksData, anchorsData, iteration == 0, borderMode,
                                    iteration > 1, stream));

    std::swap(iterIn, iterOut);
    iterOut = (iteration % 2 == 0) ? &out : &workspace->get();

    for (int i = 1; i < iteration; ++i)
    {
        NVCV_CHECK_THROW(legacyOp.infer(*iterIn, *iterOut, morphType, masksData, anchorsData, iteration == 0,
                                        borderMode, iteration > 1, stream));
        std::swap(iterIn, iterOut);
    }
}

void RunVarShapeOpenClose(legacy::MorphologyVarShape &legacyOp, cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                          const nvcv::ImageBatchVarShape &out, nvcv::OptionalImageBatchVarShapeConstRef workspace,
                          NVCVMorphologyType morphType, const nvcv::TensorDataStridedCuda &masksData,
                          const nvcv::TensorDataStridedCuda &anchorsData, int32_t iteration, NVCVBorderType borderMode)
{
    if (workspace == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Workspace must be provided for NVCV_CLOSE");
    }

    auto [first, second] = SplitMorphologyPasses(morphType);

    NVCV_CHECK_THROW(
        legacyOp.infer(in, *workspace, first, masksData, anchorsData, iteration == 0, borderMode, false, stream));
    NVCV_CHECK_THROW(
        legacyOp.infer(*workspace, out, second, masksData, anchorsData, iteration == 0, borderMode, false, stream));

    for (int i = 1; i < iteration; ++i)
    {
        NVCV_CHECK_THROW(
            legacyOp.infer(out, *workspace, first, masksData, anchorsData, iteration == 0, borderMode, false, stream));
        NVCV_CHECK_THROW(
            legacyOp.infer(*workspace, out, second, masksData, anchorsData, iteration == 0, borderMode, false, stream));
    }
}

} // namespace

Morphology::Morphology()
{
    m_legacyOp         = std::make_unique<legacy::Morphology>();
    m_legacyOpVarShape = std::make_unique<legacy::MorphologyVarShape>();
}

void Morphology::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                            nvcv::OptionalTensorConstRef workspace, NVCVMorphologyType morph_type,
                            nvcv::Size2D mask_size, int2 anchor, int32_t iteration,
                            const NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Morphology::operator()[Tensor]");
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }
    if (iteration < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Iteration must be >= 0");
    }

    switch (morph_type)
    {
    case NVCVMorphologyType::NVCV_DILATE:
    case NVCVMorphologyType::NVCV_ERODE:
        RunTensorDilateErode(*m_legacyOp, stream, *inData, *outData, workspace, morph_type, mask_size, anchor,
                             iteration, borderMode);
        break;
    case NVCVMorphologyType::NVCV_OPEN:
    case NVCVMorphologyType::NVCV_CLOSE:
        RunTensorOpenClose(*m_legacyOp, stream, *inData, *outData, workspace, morph_type, mask_size, anchor, iteration,
                           borderMode);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Wrong morph_type");
    }
}

void Morphology::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                            const nvcv::ImageBatchVarShape &out, nvcv::OptionalImageBatchVarShapeConstRef workspace,
                            NVCVMorphologyType morph_type, const nvcv::Tensor &masks, const nvcv::Tensor &anchors,
                            int32_t iteration, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Morphology::operator()[ImageBatchVarShape]");
    auto masksData = masks.exportData<nvcv::TensorDataStridedCuda>();
    if (masksData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "masksData must be a tensor");
    }

    auto anchorsData = anchors.exportData<nvcv::TensorDataStridedCuda>();
    if (anchorsData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "anchors must be a tensor");
    }
    if (iteration < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Iteration must be >= 0");
    }

    switch (morph_type)
    {
    case NVCVMorphologyType::NVCV_DILATE:
    case NVCVMorphologyType::NVCV_ERODE:
        RunVarShapeDilateErode(*m_legacyOpVarShape, stream, in, out, workspace, morph_type, *masksData, *anchorsData,
                               iteration, borderMode);
        break;

    case NVCVMorphologyType::NVCV_CLOSE:
    case NVCVMorphologyType::NVCV_OPEN:
        RunVarShapeOpenClose(*m_legacyOpVarShape, stream, in, out, workspace, morph_type, *masksData, *anchorsData,
                             iteration, borderMode);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Wrong morph_type");
    }
}

} // namespace cvcuda::priv

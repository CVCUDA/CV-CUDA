/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

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
    {
        if (workspace == nullptr && iteration > 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Workspace must be provided for iterations > 1");
        }

        if (workspace == nullptr)
        {
            NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, morph_type, mask_size, anchor,
                                               iteration == 0 ? true : false, borderMode, stream));
        }
        else
        {
            // With a workspace, we can do in-place operation depending on iteration parity
            // we want to avoid copying data back and forth so we pick workspace or output tensor
            // as the output of the first iteration, then alternate between them in such a way that
            // the output will be in the output tensor after the last iteration.
            auto workspaceData = workspace->get().exportData<nvcv::TensorDataStridedCuda>();
            NVCV_ASSERT(workspaceData);

            // pick for parity of iteration
            nvcv::TensorDataStridedCuda *in  = &(*inData);
            nvcv::TensorDataStridedCuda *out = (iteration % 2 == 1) ? &(*outData) : &(*workspaceData);
            NVCV_CHECK_THROW(m_legacyOp->infer(*in, *out, morph_type, mask_size, anchor, false, borderMode, stream));

            std::swap(in, out);
            out = (iteration % 2 == 0) ? &(*outData) : &(*workspaceData);

            for (int i = 1; i < iteration; ++i)
            {
                NVCV_CHECK_THROW(
                    m_legacyOp->infer(*in, *out, morph_type, mask_size, anchor, false, borderMode, stream));
                std::swap(in, out);
            }
        }
    }
    break;
    case NVCVMorphologyType::NVCV_OPEN:
    case NVCVMorphologyType::NVCV_CLOSE:
    {
        if (workspace == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Workspace must be provided for NVCV_CLOSE or NVCV_OPEN");
        }

        // For open/close operations we must have a workspace, as it will be the ouput of the first operation.
        // We then alternate between the workspace and the output tensor as the output of the first operation.
        NVCVMorphologyType first  = (morph_type == NVCVMorphologyType::NVCV_OPEN ? NVCVMorphologyType::NVCV_ERODE
                                                                                 : NVCVMorphologyType::NVCV_DILATE);
        NVCVMorphologyType second = (morph_type == NVCVMorphologyType::NVCV_OPEN ? NVCVMorphologyType::NVCV_DILATE
                                                                                 : NVCVMorphologyType::NVCV_ERODE);

        auto workspaceData = workspace->get().exportData<nvcv::TensorDataStridedCuda>();
        NVCV_ASSERT(workspaceData);

        NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *workspaceData, first, mask_size, anchor,
                                           iteration == 0 ? true : false, borderMode, stream));
        NVCV_CHECK_THROW(m_legacyOp->infer(*workspaceData, *outData, second, mask_size, anchor,
                                           iteration == 0 ? true : false, borderMode, stream));
        for (int i = 1; i < iteration; ++i)
        {
            NVCV_CHECK_THROW(
                m_legacyOp->infer(*outData, *workspaceData, first, mask_size, anchor, false, borderMode, stream));
            NVCV_CHECK_THROW(
                m_legacyOp->infer(*workspaceData, *outData, second, mask_size, anchor, false, borderMode, stream));
        }
        break;
    }
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Wrong morph_type");
        break;
    }
}

void Morphology::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                            const nvcv::ImageBatchVarShape &out, nvcv::OptionalImageBatchVarShapeConstRef workspace,
                            NVCVMorphologyType morph_type, const nvcv::Tensor &masks, const nvcv::Tensor &anchors,
                            int32_t iteration, NVCVBorderType borderMode) const
{
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
    {
        if (workspace == nullptr && iteration > 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Workspace must be provided for iterations > 1");
        }
        if (workspace == nullptr)
        {
            NVCV_CHECK_THROW(m_legacyOpVarShape->infer(in, out, morph_type, *masksData, *anchorsData,
                                                       iteration == 0 ? true : false, borderMode, stream));
        }
        else
        {
            NVCV_ASSERT(workspace);
            // With a workspace, we can do in-place operation depending on iteration parity
            // we want to avoid copying data back and forth so we pick workspace or output tensor
            // as the output of the first iteration, then alternate between them in such a way that
            // the output will be in the output tensor after the last iteration.
            const nvcv::ImageBatchVarShape *pIn  = &in;
            const nvcv::ImageBatchVarShape *pOut = (iteration % 2 == 1) ? &out : &workspace->get();

            NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*pIn, *pOut, morph_type, *masksData, *anchorsData,
                                                       iteration == 0 ? true : false, borderMode, stream));
            std::swap(pIn, pOut);
            pOut = (iteration % 2 == 0) ? &out : &workspace->get();
            for (int i = 1; i < iteration; ++i)
            {
                NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*pIn, *pOut, morph_type, *masksData, *anchorsData,
                                                           iteration == 0 ? true : false, borderMode, stream));
                std::swap(pIn, pOut);
            }
        }
    }
    break;

    case NVCVMorphologyType::NVCV_CLOSE:
    case NVCVMorphologyType::NVCV_OPEN:
    {
        if (workspace == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Workspace must be provided for NVCV_CLOSE");
        }
        NVCVMorphologyType first  = (morph_type == NVCVMorphologyType::NVCV_OPEN ? NVCVMorphologyType::NVCV_ERODE
                                                                                 : NVCVMorphologyType::NVCV_DILATE);
        NVCVMorphologyType second = (morph_type == NVCVMorphologyType::NVCV_OPEN ? NVCVMorphologyType::NVCV_DILATE
                                                                                 : NVCVMorphologyType::NVCV_ERODE);

        // For open/close operations we must have a workspace, as it will be the ouput of the first operation.
        // We then alternate between the workspace and the output tensor as the output of the first operation.
        NVCV_CHECK_THROW(m_legacyOpVarShape->infer(in, *workspace, first, *masksData, *anchorsData,
                                                   iteration == 0 ? true : false, borderMode, stream));
        NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*workspace, out, second, *masksData, *anchorsData,
                                                   iteration == 0 ? true : false, borderMode, stream));
        for (int i = 1; i < iteration; ++i)
        {
            NVCV_CHECK_THROW(m_legacyOpVarShape->infer(out, *workspace, first, *masksData, *anchorsData,
                                                       iteration == 0 ? true : false, borderMode, stream));
            NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*workspace, out, second, *masksData, *anchorsData,
                                                       iteration == 0 ? true : false, borderMode, stream));
        }
        break;
    }
    default:
        break;
    }
}

} // namespace cvcuda::priv

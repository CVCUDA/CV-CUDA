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

#include "OpBndBox.hpp"

#include "Nvtx.hpp"
#include "OpReformat.hpp"
#include "Types.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <cuda_runtime.h>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cstddef>
#include <mutex>

namespace cvcuda::priv {

namespace legacy  = nvcv::legacy::cuda_op;
namespace helpers = nvcv::legacy::helpers;

BndBoxPlanarBridgeWorkspace::BndBoxPlanarBridgeWorkspace()
{
    NVCV_CHECK_THROW(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
}

BndBoxPlanarBridgeWorkspace::~BndBoxPlanarBridgeWorkspace()
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

void BndBoxPlanarBridgeWorkspace::ensure(const nvcv::TensorShape &shape, nvcv::DataType dtype, cudaStream_t stream)
{
    const bool reallocate = !interleaved || interleaved.shape() != shape || interleaved.dtype() != dtype;

    if (busy)
    {
        if (reallocate)
        {
            NVCV_CHECK_THROW(cudaEventSynchronize(ready));
        }
        else
        {
            NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, ready));
        }
    }

    if (reallocate)
    {
        interleaved = nvcv::Tensor(shape, dtype);
    }
}

void BndBoxPlanarBridgeWorkspace::record(cudaStream_t stream)
{
    NVCV_CHECK_THROW(cudaEventRecord(ready, stream));
    busy = true;
}

namespace {

bool IsPacked8BitImageBatch(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    return access.chStride() == 1 && access.colStride() == access.numChannels();
}

bool CanCopyThenDrawInplace(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                            const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                            const nvcv::TensorDataAccessStridedImagePlanar &outAccess, NVCVBndBoxesI bboxes)
{
    if (bboxes == nullptr || inData.basePtr() == outData.basePtr())
    {
        return false;
    }

    {
        const auto inputFormat  = helpers::GetLegacyDataFormat(inData.layout());
        const auto outputFormat = helpers::GetLegacyDataFormat(outData.layout());
        if (!(inputFormat == legacy::kNHWC || inputFormat == legacy::kHWC)
            || !(outputFormat == legacy::kNHWC || outputFormat == legacy::kHWC))
        {
            return false;
        }
    }

    if (inData.dtype() != outData.dtype() || helpers::GetLegacyDataType(inData.dtype()) != legacy::kCV_8U)
    {
        return false;
    }

    if (inAccess.numSamples() != outAccess.numSamples() || inAccess.numRows() != outAccess.numRows()
        || inAccess.numCols() != outAccess.numCols() || inAccess.numChannels() != outAccess.numChannels())
    {
        return false;
    }

    int channels = inAccess.numChannels();
    // The legacy renderer updates 2x2 packed U8 RGB(A) quads; other channel counts, odd extents,
    // or non-packed pixels must keep the general out-of-place path.
    if (channels < 3 || channels > 4 || (inAccess.numRows() & 1) != 0 || (inAccess.numCols() & 1) != 0
        || !IsPacked8BitImageBatch(inAccess) || !IsPacked8BitImageBatch(outAccess))
    {
        return false;
    }

    if (const auto rowBytes = static_cast<int64_t>(inAccess.numCols() * channels);
        rowBytes <= 0 || inAccess.rowStride() < rowBytes || outAccess.rowStride() < rowBytes)
    {
        return false;
    }

    if (inAccess.numSamples() > 1
        && (inAccess.sampleStride() < inAccess.rowStride() * inAccess.numRows()
            || outAccess.sampleStride() < outAccess.rowStride() * outAccess.numRows()))
    {
        return false;
    }

    auto bboxesImpl = reinterpret_cast<NVCVBndBoxesImpl *>(bboxes);
    if (bboxesImpl->batch() != inAccess.numSamples())
    {
        return false;
    }

    int64_t totalBoxes = 0;
    for (int32_t i = 0; i < bboxesImpl->batch(); ++i)
    {
        totalBoxes += bboxesImpl->numBoxesAt(i);
    }

    return totalBoxes > 0;
}

void CopyTensorImageBatch(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                          const nvcv::TensorDataAccessStridedImagePlanar &outAccess, cudaStream_t stream)
{
    const auto rowBytes = static_cast<std::size_t>(inAccess.numCols() * inAccess.numChannels());

    if (inAccess.sampleStride() == inAccess.rowStride() * inAccess.numRows()
        && outAccess.sampleStride() == outAccess.rowStride() * outAccess.numRows())
    {
        NVCV_CHECK_THROW(cudaMemcpy2DAsync(
            outAccess.sampleData(0), static_cast<std::size_t>(outAccess.rowStride()), inAccess.sampleData(0),
            static_cast<std::size_t>(inAccess.rowStride()), rowBytes,
            static_cast<std::size_t>(inAccess.numRows() * inAccess.numSamples()), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    for (int32_t i = 0; i < inAccess.numSamples(); ++i)
    {
        NVCV_CHECK_THROW(cudaMemcpy2DAsync(outAccess.sampleData(i), static_cast<std::size_t>(outAccess.rowStride()),
                                           inAccess.sampleData(i), static_cast<std::size_t>(inAccess.rowStride()),
                                           rowBytes, static_cast<std::size_t>(inAccess.numRows()),
                                           cudaMemcpyDeviceToDevice, stream));
    }
}

bool IsPlanar(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

bool IsInterleaved(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC;
}

nvcv::TensorShape MakeInterleavedTensorShape(const nvcv::TensorDataStridedCuda              &data,
                                             const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    if (data.layout() == nvcv::TENSOR_NCHW)
    {
        return nvcv::TensorShape{
            {access.numSamples(), access.numRows(), access.numCols(), access.numChannels()},
            "NHWC"
        };
    }
    if (data.layout() == nvcv::TENSOR_CHW)
    {
        return nvcv::TensorShape{
            {access.numRows(), access.numCols(), access.numChannels()},
            "HWC"
        };
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Input/output must have NCHW or CHW layout for planar BndBox adaptation");
}

} // namespace

std::unique_ptr<BndBoxPlanarBridgeWorkspace> BndBox::CreatePlanarBridgeWorkspace(int)
{
    return std::make_unique<BndBoxPlanarBridgeWorkspace>();
}

BndBox::BndBox()
{
    legacy::DataShape maxIn;
    legacy::DataShape maxOut; //maxIn/maxOut not used by op.
    m_legacyOp   = std::make_unique<legacy::OSD>(maxIn, maxOut);
    m_reformatOp = std::make_unique<Reformat>();
}

BndBox::~BndBox() = default;

void BndBox::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                        const NVCVBndBoxesI &bboxes) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BndBox::operator()[Tensor]");
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

    const bool inPlanar = IsPlanar(inData->layout());
    if (const bool outPlanar = IsPlanar(outData->layout()); inPlanar || outPlanar)
    {
        if (!inPlanar || !outPlanar || inData->layout() != outData->layout())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output layouts must both be NCHW or both be CHW for planar BndBox");
        }

        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
        if (!inAccess)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be an image-like planar tensor");
        }

        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
        if (!outAccess)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be an image-like planar tensor");
        }

        if (inData->dtype() != outData->dtype() || inAccess->numSamples() != outAccess->numSamples()
            || inAccess->numRows() != outAccess->numRows() || inAccess->numCols() != outAccess->numCols()
            || inAccess->numChannels() != outAccess->numChannels())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output planar BndBox tensors must have matching shape and data type");
        }

        nvcv::TensorShape interleavedShape = MakeInterleavedTensorShape(*inData, *inAccess);
        auto             &workspace        = m_planarWorkspace.get();
        std::lock_guard   lock(workspace.mutex);

        workspace.ensure(interleavedShape, inData->dtype(), stream);

        (*m_reformatOp)(stream, in, workspace.interleaved);

        auto interleavedData = workspace.interleaved.exportData<nvcv::TensorDataStridedCuda>();
        if (interleavedData == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Internal BndBox reformat tensors must be cuda-accessible");
        }

        NVCV_CHECK_THROW(m_legacyOp->inferBox(*interleavedData, *interleavedData, bboxes, stream));

        (*m_reformatOp)(stream, workspace.interleaved, out);
        workspace.record(stream);
        return;
    }

    if (!IsInterleaved(inData->layout()) || !IsInterleaved(outData->layout()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output layouts must be NHWC/HWC or matching NCHW/CHW");
    }

    {
        auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
        if (inAccess && outAccess && CanCopyThenDrawInplace(*inData, *outData, *inAccess, *outAccess, bboxes))
        {
            CopyTensorImageBatch(*inAccess, *outAccess, stream);
            NVCV_CHECK_THROW(m_legacyOp->inferBox(*outData, *outData, bboxes, stream));
            return;
        }
    }

    NVCV_CHECK_THROW(m_legacyOp->inferBox(*inData, *outData, bboxes, stream));
}

} // namespace cvcuda::priv

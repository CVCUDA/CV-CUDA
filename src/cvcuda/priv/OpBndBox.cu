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

#include "BndBoxPolicy.hpp"
#include "CudaDeviceUtils.hpp"
#include "Nvtx.hpp"
#include "OpBndBox.hpp"
#include "OpReformat.hpp"
#include "OsdRenderer.hpp"
#include "TensorAccessValidation.hpp"
#include "Types.hpp"

#include <cuda_runtime.h>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cstddef>
#include <mutex>

namespace cvcuda::priv {

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

inline bool UseFlattenedBndBoxCopyOnCurrentDevice()
{
    int sm = 0;
    NVCV_CHECK_THROW(GetCurrentDeviceSM(sm));
    return UseFlattenedBndBoxCopyForDevice(sm);
}

inline bool IsPacked8BitImageBatch(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    return access.chStride() == 1 && access.colStride() == access.numChannels();
}

inline bool CanCopyThenDrawInplace(const nvcv::TensorDataStridedCuda              &inData,
                                   const nvcv::TensorDataStridedCuda              &outData,
                                   const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                   const nvcv::TensorDataAccessStridedImagePlanar &outAccess, NVCVBndBoxesI bboxes)
{
    if (bboxes == nullptr || inData.basePtr() == outData.basePtr())
    {
        return false;
    }

    if (!osd::IsInterleavedLayout(inData.layout()) || !osd::IsInterleavedLayout(outData.layout()))
    {
        return false;
    }

    if (inData.dtype() != outData.dtype() || !osd::IsLegacyU8(inData.dtype()))
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

inline void CopyTensorImageBatch(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                 const nvcv::TensorDataAccessStridedImagePlanar &outAccess, cudaStream_t stream)
{
    const auto rowBytes = static_cast<std::size_t>(inAccess.numCols() * inAccess.numChannels());

    if (inAccess.sampleStride() == inAccess.rowStride() * inAccess.numRows()
        && outAccess.sampleStride() == outAccess.rowStride() * outAccess.numRows())
    {
        if (inAccess.rowStride() == static_cast<int64_t>(rowBytes)
            && outAccess.rowStride() == static_cast<int64_t>(rowBytes) && UseFlattenedBndBoxCopyOnCurrentDevice())
        {
            const auto storageBytes = static_cast<std::size_t>(inAccess.sampleStride()) * inAccess.numSamples();
            NVCV_CHECK_THROW(cudaMemcpyAsync(outAccess.sampleData(0), inAccess.sampleData(0), storageBytes,
                                             cudaMemcpyDeviceToDevice, stream));
            return;
        }

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

// Replaces the legacy OSD operator's inferBox(). Every rejection below was an ErrorCode funnelled
// through NVCV_CHECK_THROW; all four codes mapped to NVCV_ERROR_INVALID_ARGUMENT, so the
// observable status of each is unchanged and the order is preserved -- ValidateInterleavedU8Pair
// is the prefix shared with the OSD element path, but the checks after it differ in order: the
// element path validated the channel count before the input/output shapes.
inline void RunBndBox(osd::cuOSDContext_t context, const nvcv::TensorDataStridedCuda &inData,
                      const nvcv::TensorDataStridedCuda &outData, NVCVBndBoxesI bboxes, cudaStream_t stream)
{
    auto inAccess = osd::ValidateInterleavedU8Pair(inData, outData);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be an image-like tensor");
    }

    int batch    = inAccess.numSamples();
    int channels = inAccess.numChannels();
    int rows     = inAccess.numRows();
    int cols     = inAccess.numCols();

    if (outAccess->numRows() != rows || outAccess->numCols() != cols || outAccess->numSamples() != batch
        || outAccess->numChannels() != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input/output shape");
    }

    if (channels > 4 || channels < 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number ch = %d", channels);
    }

    if (bboxes == nullptr) // tested by OpBndBox_Negative.rejects_null_bboxes
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Bboxes must not be null");
    }

    auto *bboxesImpl = reinterpret_cast<NVCVBndBoxesImpl *>(bboxes);
    if (bboxesImpl->batch() != batch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bboxes batch = %d", bboxesImpl->batch());
    }

    osd::DrawBndBoxes(context, cols, rows, bboxesImpl);
    osd::Render(context, inData, outData, batch, channels, rows, cols, stream);
}

} // namespace

std::unique_ptr<BndBoxPlanarBridgeWorkspace> BndBox::CreatePlanarBridgeWorkspace(int)
{
    return std::make_unique<BndBoxPlanarBridgeWorkspace>();
}

BndBox::BndBox()
{
    // One rasterizer context per operator instance: the command staging buffers it caches are what
    // make repeated draws cheap.
    m_context    = std::make_unique<osd::cuOSDContext>();
    m_reformatOp = std::make_unique<Reformat>();
}

BndBox::~BndBox() = default;

void BndBox::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                        const NVCVBndBoxesI &bboxes) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BndBox::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    const bool inPlanar = osd::IsPlanarLayout(inData->layout());
    if (const bool outPlanar = osd::IsPlanarLayout(outData->layout()); inPlanar || outPlanar)
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

        nvcv::TensorShape interleavedShape = osd::InterleavedShapeFor(inData->layout(), *inAccess, "BndBox");
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

        RunBndBox(m_context.get(), *interleavedData, *interleavedData, bboxes, stream);

        (*m_reformatOp)(stream, workspace.interleaved, out);
        workspace.record(stream);
        return;
    }

    if (!osd::IsInterleavedLayout(inData->layout()) || !osd::IsInterleavedLayout(outData->layout()))
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
            RunBndBox(m_context.get(), *outData, *outData, bboxes, stream);
            return;
        }
    }

    RunBndBox(m_context.get(), *inData, *outData, bboxes, stream);
}

} // namespace cvcuda::priv

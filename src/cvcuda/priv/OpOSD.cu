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

#include "Nvtx.hpp"
#include "OpOSD.hpp"
#include "OpReformat.hpp"
#include "OsdRenderer.hpp"
#include "TensorAccessValidation.hpp"

#include <cuda_runtime.h>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <array>
#include <mutex>

namespace cvcuda::priv {

namespace {

inline nvcv::TensorShape InterleavedShapeForOSD(const nvcv::TensorDataStridedCuda &data, const char *tensorName)
{
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(data);
    if (!access)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must be an image-like planar tensor",
                              tensorName);
    }

    return osd::InterleavedShapeFor(data.layout(), *access, "OSD");
}

// Replaces the legacy OSD operator's infer(). Every rejection below was an ErrorCode funnelled
// through NVCV_CHECK_THROW; all four codes mapped to NVCV_ERROR_INVALID_ARGUMENT, so the
// observable status of each is unchanged and the order is preserved -- ValidateInterleavedU8Pair
// is the prefix shared with the BndBox path, and the checks after it are ordered as the legacy
// element path ordered them.
inline void RunOSD(osd::cuOSDContext_t context, const nvcv::TensorDataStridedCuda &inData,
                   const nvcv::TensorDataStridedCuda &outData, const NVCVElements &elements, cudaStream_t stream)
{
    auto inAccess = osd::ValidateInterleavedU8Pair(inData, outData);

    int batch    = inAccess.numSamples();
    int channels = inAccess.numChannels();
    int rows     = inAccess.numRows();
    int cols     = inAccess.numCols();

    if (channels > 4 || channels < 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number ch = %d", channels);
    }

    if (elements == nullptr) // tested by OpOSD_Negative.rejects_null_elements
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Elements must not be null");
    }

    auto *elementsImpl = reinterpret_cast<NVCVElementsImpl *>(elements);
    if (elementsImpl->batch() != batch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid elements batch = %d",
                              elementsImpl->batch());
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be an image-like tensor");
    }

    // Legacy re-derived the output's legacy DataType here and compared it to the input's; the
    // data-type equality check above already settles that, so only the shape comparison remains.
    if (outAccess->numRows() != rows || outAccess->numCols() != cols || outAccess->numSamples() != batch
        || outAccess->numChannels() != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input/output shape");
    }

    osd::DrawElements(context, cols, rows, elementsImpl);
    osd::Render(context, inData, outData, batch, channels, rows, cols, stream);
}

} // namespace

OSDPlanarBridgeWorkspace::OSDPlanarBridgeWorkspace()
{
    NVCV_CHECK_THROW(cudaEventCreateWithFlags(&m_ready, cudaEventDisableTiming));
}

OSDPlanarBridgeWorkspace::~OSDPlanarBridgeWorkspace()
{
    if (m_ready == nullptr)
    {
        return;
    }
    if (m_pending)
    {
        NVCV_CHECK_LOG(cudaEventSynchronize(m_ready));
    }
    NVCV_CHECK_LOG(cudaEventDestroy(m_ready));
}

std::unique_lock<std::mutex> OSDPlanarBridgeWorkspace::lock()
{
    return std::unique_lock<std::mutex>{m_mutex};
}

OSDPlanarBridgeTensors OSDPlanarBridgeWorkspace::prepare(cudaStream_t stream, const nvcv::Tensor &input,
                                                         const nvcv::TensorDataStridedCuda &inputData,
                                                         const nvcv::TensorDataStridedCuda &outputData,
                                                         Reformat                          &reformat)
{
    const std::array<nvcv::TensorShape, kBridgeTensorCount> targetShapes{
        InterleavedShapeForOSD(inputData, "Input"),
        InterleavedShapeForOSD(outputData, "Output"),
    };

    synchronizeIfNeeded(stream, targetShapes, inputData.dtype());
    for (int which : {kInputIndex, kOutputIndex})
    {
        resizeBuffer(which, targetShapes[which], inputData.dtype());
    }

    reformat(stream, input, m_interleaved[kInputIndex]);

    auto inputBridge  = m_interleaved[kInputIndex].exportData<nvcv::TensorDataStridedCuda>();
    auto outputBridge = m_interleaved[kOutputIndex].exportData<nvcv::TensorDataStridedCuda>();
    if (!inputBridge || !outputBridge)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Internal OSD reformat tensors must be cuda-accessible");
    }

    return {*inputBridge, *outputBridge};
}

void OSDPlanarBridgeWorkspace::finish(cudaStream_t stream, const nvcv::Tensor &output, Reformat &reformat)
{
    reformat(stream, m_interleaved[kOutputIndex], output);
    markPending(stream);
}

void OSDPlanarBridgeWorkspace::markPending(cudaStream_t stream)
{
    NVCV_CHECK_THROW(cudaEventRecord(m_ready, stream));
    m_pending = true;
}

void OSDPlanarBridgeWorkspace::synchronizeIfNeeded(
    cudaStream_t stream, const std::array<nvcv::TensorShape, kBridgeTensorCount> &targetShapes, nvcv::DataType dtype)
{
    if (!m_pending)
    {
        return;
    }

    bool shapeOrTypeChanged = false;
    for (int which : {kInputIndex, kOutputIndex})
    {
        const nvcv::Tensor &tensor = m_interleaved[which];
        shapeOrTypeChanged |= !tensor || tensor.shape() != targetShapes[which] || tensor.dtype() != dtype;
    }

    if (shapeOrTypeChanged)
    {
        NVCV_CHECK_THROW(cudaEventSynchronize(m_ready));
        m_pending = false;
    }
    else
    {
        NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, m_ready));
    }
}

void OSDPlanarBridgeWorkspace::resizeBuffer(int which, const nvcv::TensorShape &shape, nvcv::DataType dtype)
{
    nvcv::Tensor &tensor = m_interleaved[which];
    if (!tensor || tensor.shape() != shape || tensor.dtype() != dtype)
    {
        tensor = nvcv::Tensor(shape, dtype);
    }
}

std::unique_ptr<OSDPlanarBridgeWorkspace> OSD::CreatePlanarBridgeWorkspace(int)
{
    return std::make_unique<OSDPlanarBridgeWorkspace>();
}

OSD::OSD()
{
    // One rasterizer context per operator instance: the glyph bitmap and the command staging
    // buffers it caches are what make repeated draws cheap.
    m_context    = std::make_unique<osd::cuOSDContext>();
    m_reformatOp = std::make_unique<Reformat>();
}

OSD::~OSD() = default;

void OSD::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                     const NVCVElements &elements) const
{
    CVCUDA_NVTX_RANGE("cvcuda::OSD::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    const bool inPlanar = osd::IsPlanarLayout(inData->layout());

    if (const bool outPlanar = osd::IsPlanarLayout(outData->layout()); inPlanar || outPlanar)
    {
        if (!inPlanar || !outPlanar || inData->layout() != outData->layout())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output layouts must both be NCHW or both be CHW for planar OSD");
        }

        auto                  &workspace = m_planarWorkspace.get();
        auto                   lock      = workspace.lock();
        OSDPlanarBridgeTensors bridged   = workspace.prepare(stream, in, *inData, *outData, *m_reformatOp);

        try
        {
            RunOSD(m_context.get(), bridged.input, bridged.output, elements, stream);
            workspace.finish(stream, out, *m_reformatOp);
        }
        catch (...)
        {
            workspace.markPending(stream);
            throw;
        }
        return;
    }

    if (!osd::IsInterleavedLayout(inData->layout()) || !osd::IsInterleavedLayout(outData->layout())
        || inData->layout() != outData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must both be interleaved and have identical layouts");
    }

    RunOSD(m_context.get(), *inData, *outData, elements, stream);
}

} // namespace cvcuda::priv

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

#include "OpOSD.hpp"

#include "Nvtx.hpp"
#include "OpReformat.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <cuda_runtime.h>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <array>
#include <mutex>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

namespace {

bool IsOSDPlanarLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

bool IsOSDInterleavedLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC;
}

nvcv::TensorShape InterleavedShapeForOSD(const nvcv::TensorDataStridedCuda &data, const char *tensorName)
{
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(data);
    if (!access)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must be an image-like planar tensor",
                              tensorName);
    }

    if (data.layout() == nvcv::TENSOR_NCHW)
    {
        return nvcv::TensorShape{
            {access->numSamples(), access->numRows(), access->numCols(), access->numChannels()},
            "NHWC"
        };
    }
    if (data.layout() == nvcv::TENSOR_CHW)
    {
        return nvcv::TensorShape{
            {access->numRows(), access->numCols(), access->numChannels()},
            "HWC"
        };
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Input/output must have NCHW or CHW layout for planar OSD adaptation");
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
    legacy::DataShape maxIn;
    legacy::DataShape maxOut; //maxIn/maxOut not used by op.
    m_legacyOp   = std::make_unique<legacy::OSD>(maxIn, maxOut);
    m_reformatOp = std::make_unique<Reformat>();
}

OSD::~OSD() = default;

void OSD::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                     const NVCVElements &elements) const
{
    CVCUDA_NVTX_RANGE("cvcuda::OSD::operator()[Tensor]");
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

    const bool inPlanar = IsOSDPlanarLayout(inData->layout());

    if (const bool outPlanar = IsOSDPlanarLayout(outData->layout()); inPlanar || outPlanar)
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
            NVCV_CHECK_THROW(m_legacyOp->infer(bridged.input, bridged.output, elements, stream));
            workspace.finish(stream, out, *m_reformatOp);
        }
        catch (...)
        {
            workspace.markPending(stream);
            throw;
        }
        return;
    }

    if (!IsOSDInterleavedLayout(inData->layout()) || !IsOSDInterleavedLayout(outData->layout())
        || inData->layout() != outData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must both be interleaved and have identical layouts");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, elements, stream));
}

} // namespace cvcuda::priv

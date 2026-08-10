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

#include "OpStack.hpp"

#include "Nvtx.hpp"
#include "OpStackKernels.hpp"
#include "nvcv/TensorDataAccess.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace {

bool CanRunTensorBatchKernel(cudaStream_t stream, const nvcv::TensorBatch &in, const nvcv::Tensor &out)
{
    const nvcv::TensorLayout inLayout      = in.layout();
    const bool               layoutMatches = (out.layout() == nvcv::TENSOR_NCHW && inLayout == nvcv::TENSOR_CHW)
                            || (out.layout() == nvcv::TENSOR_NHWC && inLayout == nvcv::TENSOR_HWC);
    return stream != nullptr && in.rank() == 3 && in.dtype() == out.dtype() && layoutMatches;
}

void ValidateStackTensor(const nvcv::Tensor &in, uint32_t copyIndex, uint32_t outN, uint32_t outH, uint32_t outW,
                         uint32_t outC)
{
    if (copyIndex >= outN)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output tensor is not large enough to hold all input tensors");
    }

    if (in.rank() != 3 && in.rank() != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensors must be 3D (CHW, HWC) or 4D (NCHW, NHWC)");
    }

    if (nvcv::TensorLayout inLayout = in.layout(); !(inLayout == nvcv::TENSOR_CHW || inLayout == nvcv::TENSOR_HWC
                                                     || inLayout == nvcv::TENSOR_NCHW || inLayout == nvcv::TENSOR_NHWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor layout must be CHW, HWC, NCHW, or NHWC");
    }

    const uint32_t isN = in.rank() == 4 ? 1 : 0;
    if (outH != in.shape()[0 + isN] || outW != in.shape()[1 + isN] || outC != in.shape()[2 + isN])
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensors must have the same H, W, and C as output Tensor");
    }
}

} // namespace

void Stack::operator()(cudaStream_t stream, const nvcv::TensorBatch &in, const nvcv::Tensor &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Stack::operator()[TensorBatch]");
    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    // read out data N, H, W and C
    if (out.rank() != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be NCHW or NHWC tensor");
    }

    if (nvcv::TensorLayout outLayout = out.layout();
        !(outLayout == nvcv::TENSOR_NCHW || outLayout == nvcv::TENSOR_NHWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output tensor layout must be NCHW or NHWC");
    }

    auto outN = static_cast<uint32_t>(out.shape()[0]);
    // this works for both NCHW and NHWC since we are just checking if H,W,C are the same
    auto outH = static_cast<uint32_t>(out.shape()[1]);
    auto outW = static_cast<uint32_t>(out.shape()[2]);
    auto outC = static_cast<uint32_t>(out.shape()[3]);

    // A rank-3 batch maps one input tensor to one output sample, so the device descriptor index is the output index.
    // Rank-4 or conversion-like inputs need the existing copy path to preserve their sample and layout semantics.
    const bool useKernel = CanRunTensorBatchKernel(stream, in, out);
    uint32_t   copyIndex = 0;
    for (auto it = in.begin(); it != in.end(); ++it)
    {
        ValidateStackTensor(*it, copyIndex, outN, outH, outW, outC);

        auto inData = it->exportData<nvcv::TensorDataStridedCuda>();
        if (inData == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input must be cuda-accessible, pitch-linear tensor");
        }

        if (useKernel)
        {
            auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
            NVCV_ASSERT(inAccess);
            copyIndex += inAccess->numSamples();
        }
        else
        {
            copyIndex = copyTensorToNTensor(*outData, *inData, copyIndex, stream);
        }
    }

    if (useKernel)
    {
        if (auto inBatchData = in.exportData(stream).cast<nvcv::TensorBatchDataStridedCuda>();
            inBatchData && RunStackTensorBatchKernel(stream, *inBatchData, *outData))
        {
            return;
        }

        copyIndex = 0;
        for (auto it = in.begin(); it != in.end(); ++it)
        {
            auto inData = it->exportData<nvcv::TensorDataStridedCuda>();
            NVCV_ASSERT(inData);
            copyIndex = copyTensorToNTensor(*outData, *inData, copyIndex, stream);
        }
    }
}

// copies all samples from indata to out data, returns the next index in out data.
int Stack::copyTensorToNTensor(const nvcv::TensorDataStridedCuda &outData, const nvcv::TensorDataStridedCuda &inData,
                               uint32_t outIndex, cudaStream_t stream) const
{
    auto in = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(in);
    auto out = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(out);

    for (uint32_t i = 0; i < in->numSamples(); ++i)
    {
        nvcv::Byte *inSampData  = in->sampleData(i);
        nvcv::Byte *outSampData = out->sampleData(outIndex);
        for (int32_t p = 0; p < in->numPlanes(); ++p)
        {
            NVCV_CHECK_LOG(cudaMemcpy2DAsync(
                out->planeData(p, outSampData), out->rowStride(), in->planeData(p, inSampData), in->rowStride(),
                in->numCols() * in->colStride(), in->numRows(), cudaMemcpyDeviceToDevice, stream));
        }
        outIndex++;
    }
    return outIndex;
}

void Stack::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Stack::operator()[ImageBatchVarShape]");
    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    // read out data N, H, W and C
    if (out.rank() != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be NCHW or NHWC tensor");
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be a valid image-like tensor (NHWC or NCHW)");
    }

    auto     outN = static_cast<uint32_t>(outAccess->numSamples());
    uint32_t outH = outAccess->numRows();
    uint32_t outW = outAccess->numCols();
    uint32_t outC = outAccess->numChannels();

    int32_t numImages = in.numImages();

    if (static_cast<uint32_t>(numImages) > outN)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output tensor is not large enough to hold all input images");
    }

    bool     useKernel = stream != nullptr;
    uint32_t copyIndex = 0;
    for (int32_t i = 0; i < numImages; ++i)
    {
        // Get each image from the batch using operator[]
        nvcv::Image img = in[i];

        auto imgData = img.exportData<nvcv::ImageDataStridedCuda>();
        if (imgData == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input image must be cuda-accessible, pitch-linear");
        }

        const NVCVImageBufferStrided &imgBuffer = imgData->cdata().buffer.strided;

        // Check dimensions match (using plane 0 as reference)
        if (static_cast<uint32_t>(imgBuffer.planes[0].height) != outH
            || static_cast<uint32_t>(imgBuffer.planes[0].width) != outW)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input images must have the same H, W as output Tensor");
        }

        // Check number of planes matches channels expectation
        if (static_cast<uint32_t>(imgBuffer.numPlanes) != outC && imgBuffer.numPlanes != 1)
        {
            // For interleaved format (1 plane), we can't directly check channel count here
            // For planar format, numPlanes should match outC
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input images must have the same number of channels as output Tensor");
        }

        useKernel &= imgBuffer.numPlanes == outAccess->numPlanes();
        for (int32_t planeIndex = 0; planeIndex < imgBuffer.numPlanes; ++planeIndex)
        {
            const NVCVImagePlaneStrided &plane = imgBuffer.planes[planeIndex];
            // Subsampled planes cannot share the rectangular launch used by the fast path.
            useKernel &= static_cast<uint32_t>(plane.height) == outH && static_cast<uint32_t>(plane.width) == outW;
        }

        ++copyIndex;
    }

    if (useKernel)
    {
        auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
        if (inData && RunStackVarShapeKernel(stream, *inData, *outData))
        {
            return;
        }
    }

    copyIndex = 0;
    for (int32_t i = 0; i < numImages; ++i)
    {
        auto imgData = in[i].exportData<nvcv::ImageDataStridedCuda>();
        NVCV_ASSERT(imgData);
        copyIndex = copyImageToNTensor(imgData->cdata().buffer.strided, *outData, copyIndex, stream);
    }
}

int Stack::copyImageToNTensor(const NVCVImageBufferStrided &inData, const nvcv::TensorDataStridedCuda &outData,
                              uint32_t outIndex, cudaStream_t stream) const
{
    auto out = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(out);

    nvcv::Byte *outSampData = out->sampleData(outIndex);

    for (int32_t p = 0; p < inData.numPlanes; ++p)
    {
        const NVCVImagePlaneStrided &plane = inData.planes[p];
        NVCV_CHECK_LOG(cudaMemcpy2DAsync(out->planeData(p, outSampData), out->rowStride(), plane.basePtr,
                                         plane.rowStride, plane.width * out->colStride(), plane.height,
                                         cudaMemcpyDeviceToDevice, stream));
    }

    return outIndex + 1;
}

} // namespace cvcuda::priv

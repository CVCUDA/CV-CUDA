/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvcv/TensorDataAccess.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

void Stack::operator()(cudaStream_t stream, const nvcv::TensorBatch &in, const nvcv::Tensor &out) const
{
    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    // read out data N, H, W and C
    if (out.rank() != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be NCHW orNHWC tensor");
    }

    uint32_t outN = out.shape()[0];
    // this works for both NCHW and NHWC since we are just checking if H,W,C are the same
    uint32_t outH = out.shape()[1];
    uint32_t outW = out.shape()[2];
    uint32_t outC = out.shape()[3];

    uint32_t copyIndex = 0;
    for (auto it = in.begin(); it != in.end(); ++it)
    {
        // check if output is large enough since we could have a combo of N and non N tensors on input.
        if (copyIndex >= outN)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output tensor is not large enough to hold all input tensors");
        }

        //check if data layout and shape is are equal.
        uint32_t isN = (it->rank() == 4) ? 1 : 0;
        if (outH != it->shape()[0 + isN] || outW != it->shape()[1 + isN] || outC != it->shape()[2 + isN])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input tensors must have the same H, W, and C as output Tensor");
        }

        auto inData = it->exportData<nvcv::TensorDataStridedCuda>();
        if (inData == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output must be cuda-accessible, pitch-linear tensor");
        }

        copyIndex = copyTensorToNTensor(*outData, *inData, copyIndex, stream);
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

} // namespace cvcuda::priv

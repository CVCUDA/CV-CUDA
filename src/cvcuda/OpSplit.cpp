/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "priv/OpSplit.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>

#include <vector>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 19, NVCVStatus, cvcudaSplitCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [handle]
        {
            if (!handle)
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Operator handle destination is null");
            *handle = priv::CreateOperatorHandle<priv::Split>();
        });
}

CVCUDA_DEFINE_API(0, 19, NVCVStatus, cvcudaSplitSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorBatchHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaSplitSubmit");
    return nvcv::ProtectCall(
        [handle, stream, in, out]
        {
            if (!in || !out)
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor and band batch must not be null");
            nvcv::TensorWrapHandle      input(in);
            nvcv::TensorBatchWrapHandle output(out);
            priv::ToDynamicRef<priv::Split>(handle)(stream, input.resource(), output.resource());
        });
}

CVCUDA_DEFINE_API(0, 19, NVCVStatus, cvcudaSplitVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in,
                   const NVCVImageBatchHandle *out, int32_t numOutputs))
{
    CVCUDA_NVTX_RANGE("cvcudaSplitVarShapeSubmit");
    return nvcv::ProtectCall(
        [handle, stream, in, out, numOutputs]
        {
            if (!in || !out || numOutputs < 1 || numOutputs > 4)
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Image batch and one to four band batches must not be null");
            nvcv::ImageBatchVarShapeWrapHandle    input(in);
            std::vector<nvcv::ImageBatchVarShape> outputs;
            outputs.reserve(numOutputs);
            for (int32_t channel = 0; channel < numOutputs; ++channel)
            {
                if (!out[channel])
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Band batch must not be null");
                nvcv::ImageBatchVarShapeWrapHandle output(out[channel]);
                outputs.push_back(output.resource());
            }
            priv::ToDynamicRef<priv::Split>(handle)(stream, input.resource(), outputs.data(), numOutputs);
        });
}

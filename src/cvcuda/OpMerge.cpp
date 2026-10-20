/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "priv/OpMerge.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>

#include <vector>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 19, NVCVStatus, cvcudaMergeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [handle]
        {
            if (!handle)
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Operator handle destination is null");
            *handle = priv::CreateOperatorHandle<priv::Merge>();
        });
}

CVCUDA_DEFINE_API(0, 19, NVCVStatus, cvcudaMergeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorBatchHandle in, NVCVTensorHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaMergeSubmit");
    return nvcv::ProtectCall(
        [handle, stream, in, out]
        {
            if (!in || !out)
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Band batch and tensor must not be null");
            nvcv::TensorBatchWrapHandle input(in);
            nvcv::TensorWrapHandle      output(out);
            priv::ToDynamicRef<priv::Merge>(handle)(stream, input.resource(), output.resource());
        });
}

CVCUDA_DEFINE_API(0, 19, NVCVStatus, cvcudaMergeVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVImageBatchHandle *in, int32_t numInputs,
                   NVCVImageBatchHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaMergeVarShapeSubmit");
    return nvcv::ProtectCall(
        [handle, stream, in, numInputs, out]
        {
            if (!in || !out || numInputs < 1 || numInputs > 4)
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "One to four band batches and image batch must not be null");
            std::vector<nvcv::ImageBatchVarShape> inputs;
            inputs.reserve(numInputs);
            for (int32_t channel = 0; channel < numInputs; ++channel)
            {
                if (!in[channel])
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Band batch must not be null");
                nvcv::ImageBatchVarShapeWrapHandle input(in[channel]);
                inputs.push_back(input.resource());
            }
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            priv::ToDynamicRef<priv::Merge>(handle)(stream, inputs.data(), numInputs, output.resource());
        });
}

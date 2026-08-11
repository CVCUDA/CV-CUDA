/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/OpCLAHE.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <cuda_runtime.h>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 8, NVCVStatus, cvcudaCLAHECreate,
                  (NVCVOperatorHandle * handle, int32_t maxBatchSize, int32_t tilesX, int32_t tilesY))
{
    return nvcv::ProtectCall(
        [&handle, &maxBatchSize, &tilesX, &tilesY]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::CLAHE>(maxBatchSize, tilesX, tilesY);
        });
}

CVCUDA_DEFINE_API(0, 8, NVCVStatus, cvcudaCLAHESubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   float clipLimit))
{
    CVCUDA_NVTX_RANGE("cvcudaCLAHESubmit");
    return nvcv::ProtectCall(
        [&handle, &stream, &in, &out, &clipLimit]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::CLAHE>(handle)(stream, input.resource(), output.resource(), clipLimit);
        });
}

CVCUDA_DEFINE_API(0, 8, NVCVStatus, cvcudaCLAHEVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   float clipLimit))
{
    CVCUDA_NVTX_RANGE("cvcudaCLAHEVarShapeSubmit");
    return nvcv::ProtectCall(
        [&handle, &stream, &in, &out, &clipLimit]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            priv::ToDynamicRef<priv::CLAHE>(handle)(stream, input.resource(), output.resource(), clipLimit);
        });
}

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

#include "priv/OpFindHomography.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaFindHomographyCreate,
                  (NVCVOperatorHandle * handle, int batchSize, int numPoints))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::FindHomography(batchSize, numPoints));
        });
}

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaFindHomographySubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle srcPts, NVCVTensorHandle dstPts,
                   NVCVTensorHandle models))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle _srcPts(srcPts), _dstPts(dstPts), _models(models);
            priv::ToDynamicRef<priv::FindHomography>(handle)(stream, _srcPts, _dstPts, _models);
        });
}

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaFindHomographyVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorBatchHandle srcPts,
                   NVCVTensorBatchHandle dstPts, NVCVTensorBatchHandle models))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorBatchWrapHandle _srcPts(srcPts), _dstPts(dstPts);
            nvcv::TensorBatchWrapHandle _models(models);
            priv::ToDynamicRef<priv::FindHomography>(handle)(stream, _srcPts, _dstPts, _models);
        });
}

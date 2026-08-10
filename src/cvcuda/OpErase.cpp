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

#include "priv/OpErase.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

template<typename InWrap, typename OutWrap, typename InHandle, typename OutHandle>
void SubmitErase(NVCVOperatorHandle handle, cudaStream_t stream, InHandle in, OutHandle out, NVCVTensorHandle anchor,
                 NVCVTensorHandle erasing, NVCVTensorHandle values, NVCVTensorHandle imgIdx, int8_t random,
                 uint32_t seed)
{
    InWrap                 input(in);
    OutWrap                output(out);
    nvcv::TensorWrapHandle anchorwrap(anchor);
    nvcv::TensorWrapHandle erasingwrap(erasing);
    nvcv::TensorWrapHandle valueswrap(values);
    nvcv::TensorWrapHandle imgIdxwrap(imgIdx);
    priv::ToDynamicRef<priv::Erase>(handle)(stream, input.resource(), output.resource(), anchorwrap.resource(),
                                            erasingwrap.resource(), valueswrap.resource(), imgIdxwrap.resource(),
                                            random, seed);
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaEraseCreate, (NVCVOperatorHandle * handle, int32_t max_num_erasing_area))
{
    return nvcv::ProtectCall(
        [&handle, &max_num_erasing_area]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::Erase>(max_num_erasing_area);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaEraseSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle anchor, NVCVTensorHandle erasing, NVCVTensorHandle values, NVCVTensorHandle imgIdx,
                   int8_t random, uint32_t seed))
{
    CVCUDA_NVTX_RANGE("cvcudaEraseSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &anchor, &erasing, &values, &imgIdx, &handle, &stream, &random, &seed]
        {
            SubmitErase<nvcv::TensorWrapHandle, nvcv::TensorWrapHandle>(handle, stream, in, out, anchor, erasing,
                                                                        values, imgIdx, random, seed);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaEraseVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle anchor, NVCVTensorHandle erasing, NVCVTensorHandle values, NVCVTensorHandle imgIdx,
                   int8_t random, uint32_t seed))
{
    CVCUDA_NVTX_RANGE("cvcudaEraseVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &anchor, &erasing, &values, &imgIdx, &handle, &stream, &random, &seed]
        {
            SubmitErase<nvcv::ImageBatchVarShapeWrapHandle, nvcv::ImageBatchVarShapeWrapHandle>(
                handle, stream, in, out, anchor, erasing, values, imgIdx, random, seed);
        });
}

CVCUDA_DEFINE_API(0, 17, NVCVStatus, cvcudaEraseRegionSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out, int64_t i,
                   int64_t j, int64_t h, int64_t w, NVCVTensorHandle values))
{
    CVCUDA_NVTX_RANGE("cvcudaEraseRegionSubmit");
    return nvcv::ProtectCall(
        [handle, stream, in, out, i, j, h, w, values]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            nvcv::TensorWrapHandle value(values);
            priv::ToDynamicRef<priv::Erase>(handle)(stream, input.resource(), output.resource(), i, j, h, w,
                                                    value.resource());
        });
}
